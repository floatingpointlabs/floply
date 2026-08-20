"""Pin the exact filters sent to the Price List API.

This is the highest-risk surface in the live-pricing change. Each filter
excludes a class of SKU that would otherwise be silently returned at a
different price — capacity blocks, Windows bundles, dedicated tenancy,
capacity reservations. A dropped filter produces plausible-looking numbers
that are simply wrong, which no downstream assertion would catch.

Stubber validates outgoing parameters against botocore's service model, so
these also catch malformed filter shapes.
"""

import boto3
import pytest
from botocore.stub import Stubber

from src.cost_modelling.pricing import aws_client
from src.cost_modelling.pricing.aws_client import (
    ec2_price_filters,
    s3_price_filters,
)
from src.cost_modelling.pricing.settings import PRICING_API_REGIONS, build_settings


@pytest.fixture
def pricing_client():
    return boto3.client(
        "pricing", region_name="us-east-1",
        aws_access_key_id="test", aws_secret_access_key="test",
    )


def _fields(filters: list[dict[str, str]]) -> dict[str, str]:
    return {f["Field"]: f["Value"] for f in filters}


# -- EC2 --------------------------------------------------------------------

def test_ec2_filters_pin_a_single_sku():
    fields = _fields(ec2_price_filters("p5.48xlarge", "us-west-2"))
    assert fields == {
        "instanceType": "p5.48xlarge",
        "regionCode": "us-west-2",
        "operatingSystem": "Linux",
        "preInstalledSw": "NA",
        "tenancy": "Shared",
        "capacitystatus": "Used",
        "licenseModel": "No License required",
        "marketoption": "OnDemand",
    }


def test_every_ec2_filter_is_a_term_match():
    assert {f["Type"] for f in ec2_price_filters("p5.48xlarge", "us-east-1")} == {"TERM_MATCH"}


def test_capacitystatus_is_spelled_lowercase():
    """The literal attribute name is 'capacitystatus', not 'capacityStatus'."""
    assert "capacitystatus" in _fields(ec2_price_filters("p4d.24xlarge", "us-east-1"))


def test_marketoption_excludes_capacity_blocks():
    """p5/p5e sell Capacity Blocks priced per block, not per on-demand hour."""
    assert _fields(ec2_price_filters("p5.48xlarge", "us-east-1"))["marketoption"] == "OnDemand"


def test_region_is_sent_as_code_not_display_name():
    """regionCode avoids depending on 'US East (N. Virginia)' style strings."""
    fields = _fields(ec2_price_filters("p5.48xlarge", "eu-central-1"))
    assert fields["regionCode"] == "eu-central-1"
    assert "location" not in fields


def test_ec2_filters_are_accepted_by_the_service_model(pricing_client):
    filters = ec2_price_filters("p5.48xlarge", "us-east-1")
    with Stubber(pricing_client) as stub:
        stub.add_response(
            "get_products",
            {"PriceList": [], "FormatVersion": "aws_v1"},
            {"ServiceCode": "AmazonEC2", "Filters": filters},
        )
        pricing_client.get_products(ServiceCode="AmazonEC2", Filters=filters)
        stub.assert_no_pending_responses()


# -- S3 ---------------------------------------------------------------------

def test_s3_filters_scope_to_storage_products():
    fields = _fields(s3_price_filters("us-east-1", "Standard"))
    assert fields == {
        "regionCode": "us-east-1",
        "productFamily": "Storage",
        "volumeType": "Standard",
    }


def test_every_storage_class_has_a_volume_type_and_usagetype_suffix():
    """volumeType alone is ambiguous for Intelligent-Tiering."""
    for storage_class, (volume_type, suffix) in aws_client.S3_STORAGE_CLASSES.items():
        assert volume_type, storage_class
        assert suffix.startswith("TimedStorage"), storage_class


def test_storage_classes_cover_what_the_app_offers():
    assert set(aws_client.S3_STORAGE_CLASSES) == {
        "standard", "intelligent_tiering", "standard_ia", "one_zone_ia", "glacier",
    }


# -- Endpoint configuration -------------------------------------------------

def test_pricing_api_region_must_be_a_real_endpoint(monkeypatch):
    """The Price List API is served from only three regions."""
    monkeypatch.setenv("FLOPLY_PRICING_API_REGION", "eu-west-1")
    with pytest.raises(ValueError, match="not\n?\\s*served|must be one of"):
        build_settings()


@pytest.mark.parametrize("region", PRICING_API_REGIONS)
def test_valid_pricing_endpoints_are_accepted(monkeypatch, region):
    monkeypatch.setenv("FLOPLY_PRICING_API_REGION", region)
    assert build_settings().pricing_api_region == region


def test_default_region_must_be_in_the_configured_region_list(monkeypatch):
    monkeypatch.setenv("FLOPLY_AWS_REGIONS", "us-east-1,us-west-2")
    monkeypatch.setenv("FLOPLY_DEFAULT_REGION", "ap-south-1")
    with pytest.raises(ValueError, match="not in"):
        build_settings()
