"""Lazy fetching and background refresh.

Two properties this file exists to protect:

- Nothing touches AWS until a price is genuinely needed. Import, page load, and
  the About tab must all be free.
- A stale cache is served immediately and refreshed off the render path, so a
  30-day-old entry never makes a page wait.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.cost_modelling.pricing import aws_client, background, catalog as catalog_module
from src.cost_modelling.pricing.cache import region_key
from src.cost_modelling.pricing.catalog import resolve_catalog
from src.cost_modelling.pricing.settings import build_settings
from src.cost_modelling.pricing.store import InMemoryStore
from tests.conftest import FIXTURE_HARDWARE, FIXTURE_PRICES, FIXTURE_STORAGE, primed_store

pytestmark = pytest.mark.no_primed_cache


@pytest.fixture(autouse=True)
def _reset_background():
    background.reset()
    yield
    # Join before clearing: a thread still in flight would otherwise record its
    # result into the next test's state.
    _join_refresh_threads()
    background.reset()


@pytest.fixture
def settings():
    return build_settings()


@pytest.fixture
def counting_aws(monkeypatch):
    """Record every AWS call so 'lazy' can be asserted, not assumed."""
    from src.cost_modelling.pricing.models import RegionPricing, SpecsSnapshot

    calls: list[str] = []
    now = datetime.now(timezone.utc)

    def specs(*_a, **_k):
        calls.append("specs")
        return SpecsSnapshot(fetched_at=now, instances=dict(FIXTURE_HARDWARE))

    def prices(region, _types, *_a, **_k):
        calls.append(f"prices:{region}")
        return RegionPricing(region=region, fetched_at=now,
                             prices=dict(FIXTURE_PRICES), storage=dict(FIXTURE_STORAGE))

    monkeypatch.setattr(aws_client, "fetch_instance_hardware", specs)
    monkeypatch.setattr(aws_client, "fetch_offered_instance_types", lambda *_a, **_k: set())
    monkeypatch.setattr(aws_client, "fetch_region_pricing", prices)
    return calls


def _age(store, region: str, days: int) -> None:
    key = region_key(region)
    payload = store.read(key)
    payload["fetched_at"] = (
        datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    store.write(key, payload)


# -- Laziness ---------------------------------------------------------------

def test_importing_the_app_makes_no_aws_calls(counting_aws):
    import importlib

    import src.app.app as app_module
    importlib.reload(app_module)

    assert counting_aws == [], "import time must be free of network I/O"


def test_a_fresh_cache_makes_no_aws_calls(settings, counting_aws):
    resolve_catalog("us-east-1", settings, primed_store())
    assert counting_aws == []


def test_only_the_requested_region_is_fetched(settings, counting_aws):
    resolve_catalog("us-east-1", settings, InMemoryStore())
    assert counting_aws == ["specs", "prices:us-east-1"]
    assert not any("eu-west-1" in call for call in counting_aws)


def test_a_second_region_fetches_only_that_region(settings, counting_aws):
    store = primed_store(region="us-east-1")

    resolve_catalog("us-east-1", settings, store)
    assert counting_aws == []

    resolve_catalog("eu-west-1", settings, store)
    assert "prices:eu-west-1" in counting_aws
    assert "prices:us-east-1" not in counting_aws


def test_memoised_catalog_is_not_re_resolved(settings, counting_aws):
    catalog_module.invalidate()
    store = InMemoryStore()

    catalog_module.get_catalog("us-east-1", settings=settings, store=store)
    calls_after_first = len(counting_aws)
    catalog_module.get_catalog("us-east-1", settings=settings, store=store)

    assert len(counting_aws) == calls_after_first


# -- Stale: serve now, refresh behind ---------------------------------------

def test_stale_cache_is_served_without_blocking(settings, counting_aws, monkeypatch):
    """The render path must not wait; the refresh goes to a thread."""
    scheduled: list[str] = []
    monkeypatch.setattr(
        background, "schedule_refresh",
        lambda region, *a, **k: scheduled.append(region) or True)

    store = primed_store()
    _age(store, "us-east-1", days=45)

    catalog = resolve_catalog("us-east-1", settings, store)

    assert catalog.provenance.stale
    assert catalog.spec_for("p5.48xlarge")["hourly_cost"] == pytest.approx(66.64)
    assert scheduled == ["us-east-1"], "a stale read should schedule a refresh"
    assert counting_aws == [], "and must not fetch inline"


def test_background_refresh_updates_the_cache(settings, counting_aws):
    store = primed_store()
    _age(store, "us-east-1", days=45)

    thread_started = background.schedule_refresh("us-east-1", settings, store)
    assert thread_started

    _join_refresh_threads()

    assert "prices:us-east-1" in counting_aws
    refreshed = resolve_catalog("us-east-1", settings, store)
    assert not refreshed.provenance.stale


def test_only_one_refresh_runs_per_region(settings, monkeypatch):
    """Concurrent Streamlit reruns must not each start their own fetch."""
    started: list[str] = []
    monkeypatch.setattr(
        background, "_run", lambda region, *a: started.append(region))

    assert background.schedule_refresh("us-east-1", settings, InMemoryStore())
    # _run is stubbed so the region stays in-flight; a second call is refused.
    assert not background.schedule_refresh("us-east-1", settings, InMemoryStore())


# -- Backoff ----------------------------------------------------------------

def test_failure_suppresses_immediate_retries(settings, monkeypatch):
    """One outage must not become a request storm across reruns."""
    def boom(*_a, **_k):
        raise ConnectionError("network unreachable")
    monkeypatch.setattr(aws_client, "fetch_instance_hardware", boom)

    assert background.schedule_refresh("us-east-1", settings, InMemoryStore())
    _join_refresh_threads()

    assert background.last_error("us-east-1") is not None
    assert not background.schedule_refresh("us-east-1", settings, InMemoryStore()), \
        "a recent failure should hold off the next attempt"


def test_backoff_expires(settings, monkeypatch, counting_aws):
    """With the backoff window elapsed, the next attempt is allowed through."""
    from dataclasses import replace

    no_backoff = replace(settings, retry_backoff_s=0)

    def boom(*_a, **_k):
        raise ConnectionError("network unreachable")
    monkeypatch.setattr(aws_client, "fetch_instance_hardware", boom)

    background.schedule_refresh("us-east-1", no_backoff, InMemoryStore())
    _join_refresh_threads()
    assert background.last_error("us-east-1") is not None

    # Still patched to fail — we are asserting the gate opens, not that it works.
    assert background.schedule_refresh("us-east-1", no_backoff, InMemoryStore())
    _join_refresh_threads()


def test_success_clears_the_error(settings, counting_aws):
    store = primed_store()
    _age(store, "us-east-1", days=45)

    background.schedule_refresh("us-east-1", settings, store)
    _join_refresh_threads()

    assert background.last_error("us-east-1") is None


def _join_refresh_threads(timeout: float = 10.0) -> None:
    import threading

    for thread in threading.enumerate():
        if thread.name.startswith("floply-refresh-"):
            thread.join(timeout)
