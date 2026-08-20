"""Cache keys, serialization, staleness, and the acceptance gate.

The central rule: staleness is not expiry. A stale entry is still served, so an
AWS outage costs freshness rather than availability. Only a schema mismatch or
corruption discards data.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.cost_modelling.pricing import cache
from src.cost_modelling.pricing.cache import (
    CACHE_SCHEMA_VERSION,
    SPECS_KEY,
    is_stale,
    load_region,
    load_specs,
    mark_stale_warning,
    payload_to_region,
    region_key,
    save_region,
    save_specs,
)
from src.cost_modelling.pricing.models import InstanceHardware, RegionPricing, SpecsSnapshot
from src.cost_modelling.pricing.settings import build_settings
from src.cost_modelling.pricing.store import InMemoryStore


def ago(**kwargs) -> datetime:
    return datetime.now(timezone.utc) - timedelta(**kwargs)


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def settings():
    return build_settings()


def a_region(region="us-east-1", fetched_at=None, prices=None, storage=None) -> RegionPricing:
    prices = prices if prices is not None else {
        "p5.48xlarge": 55.04, "p4d.24xlarge": 26.87, "p3.16xlarge": 24.48}
    return RegionPricing(
        region=region,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        prices=prices,
        storage=storage if storage is not None else {"standard": 23.0},
        offered=tuple(sorted(prices)),
    )


def a_specs(fetched_at=None) -> SpecsSnapshot:
    return SpecsSnapshot(
        fetched_at=fetched_at or datetime.now(timezone.utc),
        instances={"p5.48xlarge": InstanceHardware(
            "p5.48xlarge", "H100", 8, 80, 640, 192, 2048, 3200)},
    )


# -- Keys -------------------------------------------------------------------

def test_keys_carry_the_schema_version():
    """Old and new containers must be able to share a volume."""
    assert SPECS_KEY == f"specs_v{CACHE_SCHEMA_VERSION}"
    assert region_key("us-east-1") == f"catalog_v{CACHE_SCHEMA_VERSION}_us-east-1"


def test_regions_get_separate_keys():
    assert region_key("us-east-1") != region_key("eu-west-1")


# -- Staleness --------------------------------------------------------------

@pytest.mark.parametrize("days, expected", [(0, False), (29, False), (31, True)])
def test_ttl_boundary(days, expected):
    assert is_stale(ago(days=days), ttl_days=30) is expected


def test_missing_timestamp_is_stale():
    assert is_stale(None, ttl_days=30) is True


def test_future_timestamp_is_stale():
    """Clock skew must not pin an entry as permanently fresh."""
    future = datetime.now(timezone.utc) + timedelta(days=2)
    assert is_stale(future, ttl_days=30) is True


def test_specs_use_their_own_longer_ttl(settings):
    """Hardware barely changes; re-paginating 800 instance types monthly is waste."""
    assert settings.specs_ttl_days > settings.pricing_ttl_days
    sixty_days = ago(days=60)
    assert is_stale(sixty_days, settings.pricing_ttl_days) is True
    assert is_stale(sixty_days, settings.specs_ttl_days) is False


# -- Round trip -------------------------------------------------------------

def test_region_round_trip(store, settings):
    original = a_region()
    save_region(store, original, settings)
    loaded = load_region(store, "us-east-1")

    assert loaded is not None
    assert loaded.region == "us-east-1"
    assert loaded.prices == original.prices
    assert loaded.storage == original.storage
    assert loaded.fetched_at == original.fetched_at


def test_specs_round_trip(store):
    save_specs(store, a_specs())
    loaded = load_specs(store)

    assert loaded is not None
    hardware = loaded.instances["p5.48xlarge"]
    assert hardware.gpu == "H100"
    assert hardware.gpu_count == 8
    assert hardware.memory_per_gpu == 80


def test_missing_entry_loads_as_none(store):
    assert load_region(store, "ap-south-1") is None
    assert load_specs(store) is None


# -- Schema versioning ------------------------------------------------------

def test_schema_mismatch_is_ignored(store, settings):
    save_region(store, a_region(), settings)
    payload = store.read(region_key("us-east-1"))
    payload["schema_version"] = 99
    store.write(region_key("us-east-1"), payload)

    assert load_region(store, "us-east-1") is None


def test_missing_schema_version_is_ignored():
    assert payload_to_region({"region": "us-east-1", "prices": {}}) is None


def test_unparseable_timestamp_is_ignored():
    assert payload_to_region({
        "schema_version": CACHE_SCHEMA_VERSION,
        "region": "us-east-1",
        "fetched_at": "not-a-date",
    }) is None


def test_non_numeric_prices_are_dropped_not_fatal():
    loaded = payload_to_region({
        "schema_version": CACHE_SCHEMA_VERSION,
        "region": "us-east-1",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "prices": {"p5.48xlarge": 55.04, "broken": "not-a-number"},
    })
    assert loaded is not None
    assert loaded.prices == {"p5.48xlarge": 55.04}


# -- Acceptance gate --------------------------------------------------------

def test_thin_fetch_does_not_overwrite_a_good_entry(store, settings):
    """A partial outage must not degrade the only availability layer."""
    good = a_region()
    save_region(store, good, settings)

    thin = a_region(prices={"p5.48xlarge": 55.04})
    written, reason = save_region(store, thin, settings, previous=good)

    assert written is False
    assert reason is not None and "at least" in reason
    assert load_region(store, "us-east-1").prices == good.prices


def test_thin_fetch_is_accepted_when_nothing_is_cached(store, settings):
    """Something beats nothing on a cold cache."""
    thin = a_region(prices={"p5.48xlarge": 55.04})
    written, reason = save_region(store, thin, settings, previous=None)

    assert written is True
    assert reason is None
    assert load_region(store, "us-east-1").prices == {"p5.48xlarge": 55.04}


def test_full_fetch_replaces_the_previous_entry(store, settings):
    previous = a_region(prices={"p5.48xlarge": 60.0, "p4d.24xlarge": 30.0, "p3.16xlarge": 25.0})
    save_region(store, previous, settings)
    save_region(store, a_region(), settings, previous=previous)

    assert load_region(store, "us-east-1").prices["p5.48xlarge"] == pytest.approx(55.04)


def test_empty_specs_snapshot_is_refused(store):
    assert save_specs(store, SpecsSnapshot(fetched_at=datetime.now(timezone.utc))) is False
    assert load_specs(store) is None


def test_write_is_skipped_when_the_lock_is_held(settings, monkeypatch, store):
    from contextlib import contextmanager

    @contextmanager
    def busy(_key):
        yield False

    monkeypatch.setattr(store, "lock", busy)
    written, reason = save_region(store, a_region(), settings)
    assert written is False
    assert "lock" in reason


# -- Stale annotation -------------------------------------------------------

def test_stale_entry_gains_a_warning_naming_its_age():
    stale = a_region(fetched_at=ago(days=41))
    annotated = mark_stale_warning(stale, ttl_days=30)

    assert annotated.prices == stale.prices, "data must still be served"
    assert any("41 days old" in w for w in annotated.warnings)


def test_fresh_entry_is_left_alone():
    fresh = a_region(fetched_at=ago(days=2))
    assert mark_stale_warning(fresh, ttl_days=30).warnings == fresh.warnings
