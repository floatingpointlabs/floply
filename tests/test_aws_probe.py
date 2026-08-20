"""The startup credential probe and whole-region fetch orchestration.

Fetching is lazy, so without the probe a misconfigured IAM policy is discovered
by a user several form sections deep. The probe must therefore never raise —
its whole job is to turn every failure into a readable message.
"""

import json

import boto3
import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.stub import Stubber

from src.cost_modelling.pricing import aws_client
from src.cost_modelling.pricing.aws_client import fetch_region_pricing, probe_credentials
from src.cost_modelling.pricing.settings import build_settings


@pytest.fixture
def pricing_client():
    return boto3.client(
        "pricing", region_name="us-east-1",
        aws_access_key_id="test", aws_secret_access_key="test",
    )


@pytest.fixture
def use_client(monkeypatch):
    def _use(client):
        monkeypatch.setattr(aws_client, "_client", lambda *a, **k: client)
    return _use


# -- Probe ------------------------------------------------------------------

def test_probe_succeeds_against_a_reachable_api(pricing_client, use_client):
    use_client(pricing_client)
    with Stubber(pricing_client) as stub:
        stub.add_response("get_products", {"PriceList": [], "FormatVersion": "aws_v1"}, None)
        result = probe_credentials(build_settings())

    assert result.ok
    assert "us-east-1" in result.detail


def test_probe_reports_missing_credentials(use_client, monkeypatch):
    def explode(*_a, **_k):
        raise NoCredentialsError()
    monkeypatch.setattr(aws_client, "_client", explode)

    result = probe_credentials(build_settings())
    assert not result.ok
    assert "credentials" in result.detail.lower()


def test_probe_names_the_missing_iam_action(pricing_client, use_client):
    """A denied policy should be one copy-paste away from being fixed."""
    use_client(pricing_client)
    with Stubber(pricing_client) as stub:
        stub.add_client_error("get_products", service_error_code="AccessDeniedException")
        result = probe_credentials(build_settings())

    assert not result.ok
    assert result.missing_action == "pricing:GetProducts"


def test_probe_never_raises_on_an_unexpected_error(use_client, monkeypatch):
    def explode(*_a, **_k):
        raise RuntimeError("DNS exploded")
    monkeypatch.setattr(aws_client, "_client", explode)

    result = probe_credentials(build_settings())
    assert not result.ok
    assert "DNS exploded" in result.detail


def test_probe_reports_other_client_errors(pricing_client, use_client):
    use_client(pricing_client)
    with Stubber(pricing_client) as stub:
        stub.add_client_error("get_products", service_error_code="ThrottlingException")
        result = probe_credentials(build_settings())

    assert not result.ok
    assert "Throttling" in result.detail


# -- Region orchestration ---------------------------------------------------

def _hourly(price: str) -> dict:
    return {
        "product": {"attributes": {}},
        "terms": {"OnDemand": {"s": {"priceDimensions": {"d": {
            "unit": "Hrs", "beginRange": "0", "pricePerUnit": {"USD": price}}}}}},
    }


def _storage(price: str, usagetype: str) -> dict:
    return {
        "product": {"attributes": {"usagetype": usagetype}},
        "terms": {"OnDemand": {"s": {"priceDimensions": {"d": {
            "unit": "GB-Mo", "beginRange": "0", "pricePerUnit": {"USD": price}}}}}},
    }


def _respond(stub, products):
    stub.add_response(
        "get_products",
        {"PriceList": [json.dumps(p) for p in products], "FormatVersion": "aws_v1"},
        None,
    )


def test_fetch_region_pricing_collects_instances_and_storage(pricing_client, use_client):
    use_client(pricing_client)
    with Stubber(pricing_client) as stub:
        _respond(stub, [_hourly("55.04")])                 # p5
        _respond(stub, [_hourly("26.87")])                 # p4d
        _respond(stub, [_storage("0.023", "TimedStorage-ByteHrs")])
        for _ in range(4):                                 # remaining storage classes
            _respond(stub, [])
        result = fetch_region_pricing(
            "us-east-1", ["p5.48xlarge", "p4d.24xlarge"], build_settings())

    assert result.region == "us-east-1"
    assert result.prices == {
        "p5.48xlarge": pytest.approx(55.04),
        "p4d.24xlarge": pytest.approx(26.87),
    }
    assert result.storage["standard"] == pytest.approx(23.0)
    assert result.offered == ("p4d.24xlarge", "p5.48xlarge")


def test_instance_missing_from_a_region_is_omitted_not_zeroed(pricing_client, use_client):
    """A missing price must never become $0/hr, which would read as free compute."""
    use_client(pricing_client)
    with Stubber(pricing_client) as stub:
        _respond(stub, [_hourly("55.04")])
        _respond(stub, [])                                 # p4d not sold here
        for _ in range(5):
            _respond(stub, [])
        result = fetch_region_pricing(
            "eu-west-1", ["p5.48xlarge", "p4d.24xlarge"], build_settings())

    assert "p4d.24xlarge" not in result.prices
    assert result.offered == ("p5.48xlarge",)


def test_duplicate_sku_warnings_propagate_to_the_region(pricing_client, use_client):
    use_client(pricing_client)
    with Stubber(pricing_client) as stub:
        _respond(stub, [_hourly("124.15"), _hourly("55.04")])
        for _ in range(5):
            _respond(stub, [])
        result = fetch_region_pricing("us-east-1", ["p5.48xlarge"], build_settings())

    assert result.prices["p5.48xlarge"] == pytest.approx(55.04)
    assert any("2 SKUs" in w for w in result.warnings)
