"""The resolution chain: live -> cache -> error.

There is deliberately no bundled price fallback, so "no data" is a real state
the app must handle honestly. These tests force each layer to fail and assert
the app degrades in the intended order — and that the final state is a clean
error rather than a traceback, a zero, or an empty calculator.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.cost_modelling.pricing import aws_client, catalog as catalog_module
from src.cost_modelling.pricing.catalog import (
    PricingUnavailableError,
    resolve_catalog,
)
from src.cost_modelling.pricing.settings import build_settings
from src.cost_modelling.pricing.store import InMemoryStore
from tests.conftest import FIXTURE_HARDWARE, FIXTURE_PRICES, FIXTURE_STORAGE, primed_store

pytestmark = pytest.mark.no_primed_cache


@pytest.fixture
def settings():
    return build_settings()


@pytest.fixture
def offline(monkeypatch):
    """Every AWS call fails, as if the network were unreachable."""
    def boom(*_a, **_k):
        raise ConnectionError("network unreachable")

    for name in ("fetch_instance_hardware", "fetch_offered_instance_types",
                 "fetch_region_pricing"):
        monkeypatch.setattr(aws_client, name, boom)


@pytest.fixture
def online(monkeypatch):
    """AWS answers with the fixture data."""
    from src.cost_modelling.pricing.models import RegionPricing, SpecsSnapshot

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        aws_client, "fetch_instance_hardware",
        lambda *_a, **_k: SpecsSnapshot(fetched_at=now, instances=dict(FIXTURE_HARDWARE)))
    monkeypatch.setattr(aws_client, "fetch_offered_instance_types", lambda *_a, **_k: set())
    monkeypatch.setattr(
        aws_client, "fetch_region_pricing",
        lambda region, types, *a, **k: RegionPricing(
            region=region, fetched_at=now,
            prices=dict(FIXTURE_PRICES), storage=dict(FIXTURE_STORAGE)))


# -- Layer 1: live ----------------------------------------------------------

def test_cold_cache_fetches_live(settings, online):
    catalog = resolve_catalog("us-east-1", settings, InMemoryStore())

    assert catalog.provenance.source == "live"
    assert catalog.provenance.is_usable
    assert "p5.48xlarge" in catalog.instances


def test_live_fetch_populates_the_cache(settings, online):
    store = InMemoryStore()
    resolve_catalog("us-east-1", settings, store)

    from src.cost_modelling.pricing.cache import load_region, load_specs
    assert load_specs(store) is not None
    assert load_region(store, "us-east-1") is not None


# -- Layer 2: cache ---------------------------------------------------------

def test_warm_cache_serves_without_touching_aws(settings, monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("a fresh cache must not call AWS")

    for name in ("fetch_instance_hardware", "fetch_region_pricing"):
        monkeypatch.setattr(aws_client, name, explode)

    catalog = resolve_catalog("us-east-1", settings, primed_store())

    assert catalog.provenance.source == "cache"
    assert catalog.spec_for("p5.48xlarge")["hourly_cost"] == pytest.approx(66.64)


def test_stale_cache_is_still_served(settings, offline):
    """TTL means 'try to refresh', never 'delete'. An outage must not blank the app."""
    store = primed_store()
    _age_cache(store, "us-east-1", days=45)

    catalog = resolve_catalog("us-east-1", settings, store)

    assert catalog.provenance.is_usable, "stale data beats no data"
    assert catalog.provenance.stale
    assert catalog.spec_for("p5.48xlarge")["hourly_cost"] == pytest.approx(66.64)


def test_stale_cache_says_how_old_it_is(settings, offline):
    store = primed_store()
    _age_cache(store, "us-east-1", days=45)

    provenance = resolve_catalog("us-east-1", settings, store).provenance

    assert provenance.age_days >= 44
    assert any("45 days old" in w for w in provenance.warnings)
    assert "refresh failed" not in provenance.summary() or provenance.last_error


# -- Layer 3: nothing -------------------------------------------------------

def test_cold_cache_and_no_network_is_unavailable(settings, offline):
    catalog = resolve_catalog("us-east-1", settings, InMemoryStore())

    assert catalog.provenance.source == "unavailable"
    assert not catalog.provenance.is_usable
    assert catalog.instances == {}


def test_unavailable_catalog_raises_rather_than_returning_zero(offline, monkeypatch):
    """A $0/hr instance would read as free compute — the worst failure mode."""
    from src.cost_modelling import gpu_specs

    monkeypatch.setattr(catalog_module, "build_store", lambda _s: InMemoryStore())
    catalog_module.invalidate()

    with pytest.raises(PricingUnavailableError) as exc:
        gpu_specs.get_gpu_instance("p5.48xlarge", "us-east-1")

    assert "us-east-1" in str(exc.value)


def test_storage_cost_raises_rather_than_returning_zero(offline, monkeypatch):
    from src.cost_modelling import gpu_specs

    monkeypatch.setattr(catalog_module, "build_store", lambda _s: InMemoryStore())
    catalog_module.invalidate()

    with pytest.raises(PricingUnavailableError):
        gpu_specs.get_storage_cost("standard", "us-east-1")


def test_unavailable_error_explains_itself(settings, offline):
    catalog = resolve_catalog("us-east-1", settings, InMemoryStore())
    assert catalog.provenance.last_error
    assert "unavailable" in catalog.provenance.summary().lower()


def test_listing_instances_is_safe_when_unavailable(settings, offline):
    """Listing must degrade to empty, not raise — the sidebar renders regardless."""
    catalog = resolve_catalog("us-east-1", settings, InMemoryStore())
    assert catalog.list_instances() == []
    assert catalog.list_storage_classes() == []


def test_network_disabled_never_calls_aws(settings, monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("allow_network=False must not call AWS")
    monkeypatch.setattr(aws_client, "fetch_instance_hardware", explode)

    catalog = resolve_catalog(
        "us-east-1", settings, InMemoryStore(), allow_network=False)
    assert catalog.provenance.source == "unavailable"


# -- Region isolation -------------------------------------------------------

def test_an_unfetched_region_does_not_borrow_another_regions_prices(settings, offline):
    """Cross-region leakage would silently misprice an entire estimate."""
    store = primed_store(region="us-east-1")

    usable = resolve_catalog("us-east-1", settings, store)
    other = resolve_catalog("eu-west-1", settings, store)

    assert usable.provenance.is_usable
    assert not other.provenance.is_usable


def test_regions_are_memoised_independently(settings, online):
    catalog_module.invalidate()
    first = catalog_module.get_catalog("us-east-1", settings=settings, store=primed_store())
    second = catalog_module.get_catalog("us-east-1", settings=settings, store=primed_store())
    assert first is second

    catalog_module.invalidate("us-east-1")
    third = catalog_module.get_catalog("us-east-1", settings=settings, store=primed_store())
    assert third is not first


# -- Helpers ----------------------------------------------------------------

def _age_cache(store: InMemoryStore, region: str, days: int) -> None:
    from src.cost_modelling.pricing.cache import region_key

    key = region_key(region)
    payload = store.read(key)
    payload["fetched_at"] = (
        datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    store.write(key, payload)
