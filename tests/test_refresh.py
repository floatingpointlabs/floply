"""Fetch -> validate -> persist orchestration.

Covers the degradation paths: regions are independent, a failure serves cache
rather than raising, and instances without curated FLOPs are quarantined rather
than priced with a guess.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.cost_modelling.pricing import aws_client, cache, refresh
from src.cost_modelling.pricing.models import (
    InstanceHardware,
    RegionPricing,
    SpecsSnapshot,
)
from src.cost_modelling.pricing.refresh import (
    partition_by_curation,
    refresh_all,
    refresh_region,
    refresh_specs,
)
from src.cost_modelling.pricing.settings import build_settings
from src.cost_modelling.pricing.store import InMemoryStore


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def settings():
    return build_settings()


def hardware(instance_type="p5.48xlarge", gpu="H100", count=8, per_gpu=80):
    return InstanceHardware(
        instance_type, gpu, count, per_gpu, per_gpu * count, 192, 2048, 3200)


def specs(*instances, fetched_at=None) -> SpecsSnapshot:
    return SpecsSnapshot(
        fetched_at=fetched_at or datetime.now(timezone.utc),
        instances={h.instance_type: h for h in instances},
    )


def pricing(region="us-east-1", prices=None, fetched_at=None) -> RegionPricing:
    prices = prices if prices is not None else {
        "p5.48xlarge": 55.04, "p4d.24xlarge": 26.87, "p3.16xlarge": 24.48}
    return RegionPricing(
        region=region,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        prices=prices,
        storage={"standard": 23.0},
        offered=tuple(sorted(prices)),
    )


# -- Quarantine -------------------------------------------------------------

def test_curated_gpus_are_usable():
    usable, quarantined = partition_by_curation(
        specs(hardware("p5.48xlarge", "H100"), hardware("p4d.24xlarge", "A100", per_gpu=40)))
    assert usable == ["p4d.24xlarge", "p5.48xlarge"]
    assert quarantined == {}


def test_uncurated_gpu_is_quarantined_with_an_actionable_message():
    usable, quarantined = partition_by_curation(
        specs(hardware("p5.48xlarge", "H100"), hardware("p9.future", "Z900")))

    assert usable == ["p5.48xlarge"]
    assert "Z900" in quarantined["p9.future"]
    assert "gpu_hardware.yaml" in quarantined["p9.future"]


def test_inconsistent_hardware_is_quarantined():
    bad = InstanceHardware("p5.48xlarge", "H100", 8, 80, 99999, 192, 2048, 3200)
    usable, quarantined = partition_by_curation(specs(bad))
    assert usable == []
    assert "total_gpu_memory" in quarantined["p5.48xlarge"]


def test_seeded_future_gpus_are_not_quarantined():
    """H200/B200 were seeded so p5e/p6 discovery needs no code change."""
    usable, quarantined = partition_by_curation(
        specs(hardware("p5e.48xlarge", "H200"), hardware("p6.48xlarge", "B200")))
    assert len(usable) == 2
    assert quarantined == {}


# -- Specs refresh ----------------------------------------------------------

def test_fresh_specs_are_not_refetched(store, settings, monkeypatch):
    cache.save_specs(store, specs(hardware()))

    def explode(*_a, **_k):
        raise AssertionError("should not have called AWS")
    monkeypatch.setattr(aws_client, "fetch_instance_hardware", explode)

    snapshot, warnings = refresh_specs(store, settings)
    assert snapshot is not None and warnings == []


def test_stale_specs_are_refetched(store, settings, monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(days=200)
    cache.save_specs(store, specs(hardware("p4d.24xlarge", "A100", per_gpu=40), fetched_at=old))
    monkeypatch.setattr(
        aws_client, "fetch_instance_hardware", lambda *_a, **_k: specs(hardware()))

    snapshot, _ = refresh_specs(store, settings)
    assert set(snapshot.instances) == {"p5.48xlarge"}


def test_specs_fetch_failure_serves_cache(store, settings, monkeypatch):
    cache.save_specs(store, specs(hardware()))

    def boom(*_a, **_k):
        raise ConnectionError("network down")
    monkeypatch.setattr(aws_client, "fetch_instance_hardware", boom)

    snapshot, warnings = refresh_specs(store, settings, force=True)
    assert snapshot is not None, "must degrade to cache, not fail"
    assert any("cached specs" in w for w in warnings)


def test_specs_fetch_failure_with_no_cache_returns_none(store, settings, monkeypatch):
    def boom(*_a, **_k):
        raise ConnectionError("network down")
    monkeypatch.setattr(aws_client, "fetch_instance_hardware", boom)

    snapshot, warnings = refresh_specs(store, settings)
    assert snapshot is None
    assert warnings


def test_empty_aws_response_does_not_wipe_cached_specs(store, settings, monkeypatch):
    cache.save_specs(store, specs(hardware()))
    monkeypatch.setattr(
        aws_client, "fetch_instance_hardware",
        lambda *_a, **_k: SpecsSnapshot(fetched_at=datetime.now(timezone.utc)))

    snapshot, warnings = refresh_specs(store, settings, force=True)
    assert snapshot.instances, "an empty fetch must not empty the catalog"
    assert warnings


# -- Region refresh ---------------------------------------------------------

def test_fresh_region_is_not_refetched(store, settings, monkeypatch):
    cache.save_region(store, pricing(), settings)

    def explode(*_a, **_k):
        raise AssertionError("should not have called AWS")
    monkeypatch.setattr(aws_client, "fetch_region_pricing", explode)
    monkeypatch.setattr(aws_client, "fetch_offered_instance_types", explode)

    result, warnings = refresh_region(store, "us-east-1", ["p5.48xlarge"], settings)
    assert result is not None and warnings == []


def test_region_fetch_failure_serves_stale_cache_with_a_warning(store, settings, monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(days=41)
    cache.save_region(store, pricing(fetched_at=old), settings)

    def boom(*_a, **_k):
        raise ConnectionError("network down")
    monkeypatch.setattr(aws_client, "fetch_offered_instance_types", boom)

    result, warnings = refresh_region(store, "us-east-1", ["p5.48xlarge"], settings)
    assert result is not None, "stale data beats no data"
    assert result.prices["p5.48xlarge"] == pytest.approx(55.04)
    assert any("could not be refreshed" in w for w in warnings)
    assert any("41 days old" in w for w in result.warnings)


def test_region_fetch_failure_with_no_cache_returns_none(store, settings, monkeypatch):
    def boom(*_a, **_k):
        raise ConnectionError("network down")
    monkeypatch.setattr(aws_client, "fetch_offered_instance_types", boom)

    result, warnings = refresh_region(store, "us-east-1", ["p5.48xlarge"], settings)
    assert result is None
    assert warnings


def test_only_instances_offered_in_the_region_are_priced(store, settings, monkeypatch):
    asked: dict[str, list[str]] = {}

    monkeypatch.setattr(
        aws_client, "fetch_offered_instance_types", lambda *_a, **_k: {"p5.48xlarge"})

    def record(region, instance_types, *_a, **_k):
        asked["types"] = instance_types
        return pricing(region, {"p5.48xlarge": 55.04})
    monkeypatch.setattr(aws_client, "fetch_region_pricing", record)

    refresh_region(store, "eu-west-1", ["p5.48xlarge", "p4d.24xlarge"], settings)
    assert asked["types"] == ["p5.48xlarge"]


# -- Whole refresh ----------------------------------------------------------

def test_refresh_all_walks_every_region(store, settings, monkeypatch):
    monkeypatch.setattr(
        aws_client, "fetch_instance_hardware", lambda *_a, **_k: specs(hardware()))
    monkeypatch.setattr(aws_client, "fetch_offered_instance_types", lambda *_a, **_k: set())
    monkeypatch.setattr(
        aws_client, "fetch_region_pricing",
        lambda region, types, *a, **k: pricing(region))

    result = refresh_all(regions=("us-east-1", "eu-west-1"), settings=settings, store=store)

    assert result.ok
    assert set(result.regions) == {"us-east-1", "eu-west-1"}


def test_one_failing_region_does_not_abort_the_others(store, settings, monkeypatch):
    monkeypatch.setattr(
        aws_client, "fetch_instance_hardware", lambda *_a, **_k: specs(hardware()))
    monkeypatch.setattr(aws_client, "fetch_offered_instance_types", lambda *_a, **_k: set())

    def flaky(region, *_a, **_k):
        if region == "eu-west-1":
            raise ConnectionError("region down")
        return pricing(region)
    monkeypatch.setattr(aws_client, "fetch_region_pricing", flaky)

    result = refresh_all(regions=("us-east-1", "eu-west-1"), settings=settings, store=store)

    assert "us-east-1" in result.regions
    assert "eu-west-1" not in result.regions
    assert not result.ok
    assert any("eu-west-1" in w for w in result.warnings)


def test_refresh_all_reports_quarantined_instances(store, settings, monkeypatch):
    monkeypatch.setattr(
        aws_client, "fetch_instance_hardware",
        lambda *_a, **_k: specs(hardware(), hardware("p9.future", "Z900")))
    monkeypatch.setattr(aws_client, "fetch_offered_instance_types", lambda *_a, **_k: set())
    monkeypatch.setattr(
        aws_client, "fetch_region_pricing", lambda region, *a, **k: pricing(region))

    result = refresh_all(regions=("us-east-1",), settings=settings, store=store)
    assert "p9.future" in result.quarantined


def test_all_instances_quarantined_is_a_failure_not_an_empty_success(store, settings, monkeypatch):
    monkeypatch.setattr(
        aws_client, "fetch_instance_hardware", lambda *_a, **_k: specs(hardware("p9.future", "Z900")))

    result = refresh_all(regions=("us-east-1",), settings=settings, store=store)
    assert not result.ok
    assert any("quarantined" in w for w in result.warnings)


def test_refresh_budget_stops_the_walk_and_says_so(store, monkeypatch):
    """A truncated walk must be reported, not look like a clean partial result."""
    import src.cost_modelling.pricing.refresh as refresh_module
    from dataclasses import replace as replace_dataclass

    settings = replace_dataclass(build_settings(), max_refresh_seconds=30)

    monkeypatch.setattr(
        aws_client, "fetch_instance_hardware", lambda *_a, **_k: specs(hardware()))
    monkeypatch.setattr(aws_client, "fetch_offered_instance_types", lambda *_a, **_k: set())
    monkeypatch.setattr(
        aws_client, "fetch_region_pricing", lambda region, *a, **k: pricing(region))

    # Clock jumps past the budget after the first region is priced.
    ticks = iter([0.0, 1.0, 999.0, 999.0])
    monkeypatch.setattr(refresh_module.time, "monotonic", lambda: next(ticks))

    result = refresh_all(regions=("us-east-1", "eu-west-1"), settings=settings, store=store)

    assert set(result.regions) == {"us-east-1"}
    assert not result.ok
    assert any("budget" in w and "eu-west-1" in w for w in result.warnings)
