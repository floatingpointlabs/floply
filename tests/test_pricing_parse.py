"""Parsing the Price List API response into a single defensible number.

The fixtures deliberately include the SKUs the filters are meant to exclude,
so these tests still pass if a filter is dropped — proving the parser is a
second line of defence rather than the only one.
"""

import json

import boto3
import pytest
from botocore.stub import Stubber

from src.cost_modelling.pricing.aws_client import (
    GB_PER_TB,
    extract_on_demand_price,
    fetch_instance_price,
    fetch_storage_prices,
)


def _product(price: str, unit: str = "Hrs", begin: str = "0", **attributes) -> dict:
    """Build a Price List product with one on-demand price dimension."""
    return {
        "product": {"attributes": {"instanceType": "p5.48xlarge", **attributes}},
        "terms": {
            "OnDemand": {
                "SKU.JRTCKXETXF": {
                    "priceDimensions": {
                        "SKU.JRTCKXETXF.6YS6EN2CT7": {
                            "unit": unit,
                            "beginRange": begin,
                            "pricePerUnit": {"USD": price},
                        }
                    }
                }
            }
        },
    }


@pytest.fixture
def pricing_client():
    return boto3.client(
        "pricing", region_name="us-east-1",
        aws_access_key_id="test", aws_secret_access_key="test",
    )


def _stub_products(stub, products: list[dict]):
    stub.add_response(
        "get_products",
        {"PriceList": [json.dumps(p) for p in products], "FormatVersion": "aws_v1"},
        None,
    )


# -- Single-price extraction ------------------------------------------------

def test_extracts_the_hourly_price():
    assert extract_on_demand_price(_product("55.04")) == pytest.approx(55.04)


def test_ignores_dimensions_with_the_wrong_unit():
    assert extract_on_demand_price(_product("0.023", unit="GB-Mo"), unit="Hrs") is None


def test_ignores_higher_tiers():
    """S3 Standard is tiered at 50TB/500TB; only the base rate is the headline."""
    assert extract_on_demand_price(_product("0.022", unit="GB-Mo", begin="51200"),
                                   unit="GB-Mo") is None


def test_zero_priced_dimensions_are_skipped():
    """Free-tier and $0.00 placeholder dimensions are not real prices."""
    assert extract_on_demand_price(_product("0.0000000000")) is None


def test_malformed_price_does_not_raise():
    broken = _product("55.04")
    dimension = next(iter(next(iter(broken["terms"]["OnDemand"].values()))["priceDimensions"].values()))
    dimension["pricePerUnit"] = {}
    assert extract_on_demand_price(broken) is None


def test_missing_terms_does_not_raise():
    assert extract_on_demand_price({"product": {}}) is None


# -- Instance pricing -------------------------------------------------------

def test_single_matching_sku_yields_no_warning(pricing_client):
    with Stubber(pricing_client) as stub:
        _stub_products(stub, [_product("55.04")])
        price, warning = fetch_instance_price(pricing_client, "p5.48xlarge", "us-east-1")
    assert price == pytest.approx(55.04)
    assert warning is None


def test_duplicate_skus_take_the_minimum_and_warn(pricing_client):
    """A filter regression surfacing Windows/dedicated/capacity-block SKUs must
    be reported, not silently resolved by taking whichever came first."""
    with Stubber(pricing_client) as stub:
        _stub_products(stub, [
            _product("124.15", operatingSystem="Windows"),
            _product("55.04", operatingSystem="Linux"),
            _product("88.00", tenancy="Dedicated"),
        ])
        price, warning = fetch_instance_price(pricing_client, "p5.48xlarge", "us-east-1")

    assert price == pytest.approx(55.04)
    assert warning is not None
    assert "3 SKUs" in warning
    assert "p5.48xlarge/us-east-1" in warning


def test_no_matching_sku_returns_none(pricing_client):
    """An instance not sold in a region is not an error."""
    with Stubber(pricing_client) as stub:
        _stub_products(stub, [])
        price, warning = fetch_instance_price(pricing_client, "p5.48xlarge", "eu-west-1")
    assert price is None
    assert warning is None


def test_price_list_entries_are_json_strings(pricing_client):
    """The API returns each product as a JSON string, not an object."""
    with Stubber(pricing_client) as stub:
        stub.add_response(
            "get_products",
            {"PriceList": [json.dumps(_product("26.87"))], "FormatVersion": "aws_v1"},
            None,
        )
        price, _ = fetch_instance_price(pricing_client, "p4d.24xlarge", "us-east-1")
    assert price == pytest.approx(26.87)


# -- Storage pricing --------------------------------------------------------

def _storage_product(price: str, usagetype: str) -> dict:
    product = _product(price, unit="GB-Mo")
    product["product"]["attributes"]["usagetype"] = usagetype
    return product


def test_storage_price_converts_per_gb_to_per_tb(pricing_client):
    with Stubber(pricing_client) as stub:
        for _ in range(5):        # one call per storage class
            _stub_products(stub, [_storage_product("0.023", "TimedStorage-ByteHrs")])
        storage, _ = fetch_storage_prices(pricing_client, "us-east-1")

    assert storage["standard"] == pytest.approx(0.023 * GB_PER_TB)
    assert storage["standard"] == pytest.approx(23.0)


def test_usagetype_suffix_matching_tolerates_region_prefixes(pricing_client):
    """Non-us-east-1 regions prefix usagetype, e.g. EUC1-TimedStorage-ByteHrs."""
    with Stubber(pricing_client) as stub:
        for _ in range(5):
            _stub_products(stub, [_storage_product("0.0245", "EUC1-TimedStorage-ByteHrs")])
        storage, _ = fetch_storage_prices(pricing_client, "eu-central-1")

    assert storage["standard"] == pytest.approx(24.5)


def test_wrong_usagetype_is_not_matched(pricing_client):
    """Intelligent-Tiering exposes several tier SKUs under one volumeType."""
    with Stubber(pricing_client) as stub:
        for _ in range(5):
            _stub_products(stub, [
                _storage_product("0.0125", "TimedStorage-INT-IA-ByteHrs"),   # infrequent tier
            ])
        storage, warnings = fetch_storage_prices(pricing_client, "us-east-1")

    assert "intelligent_tiering" not in storage
    assert any("intelligent_tiering" in w for w in warnings)


def test_a_renamed_volume_type_warns_without_failing_the_region(pricing_client):
    """AWS has renamed these strings before; one miss must not lose the region."""
    with Stubber(pricing_client) as stub:
        _stub_products(stub, [_storage_product("0.023", "TimedStorage-ByteHrs")])
        for _ in range(4):
            _stub_products(stub, [])
        storage, warnings = fetch_storage_prices(pricing_client, "us-east-1")

    assert storage == {"standard": pytest.approx(23.0)}
    assert len(warnings) == 4
