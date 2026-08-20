"""Shared fixtures.

With no bundled price fallback, every test that touches costs needs a primed
cache. These fixtures stand in for AWS so the suite runs offline and without
credentials.
"""

from datetime import datetime, timezone

import pytest

from src.cost_modelling.pricing import cache, catalog as catalog_module
from src.cost_modelling.pricing.models import InstanceHardware, RegionPricing, SpecsSnapshot
from src.cost_modelling.pricing.settings import build_settings
from src.cost_modelling.pricing.store import InMemoryStore

# The five instances Floply shipped with, at their February 2026 prices. Used
# as a stable stand-in for AWS, not as a fallback the app can reach.
FIXTURE_HARDWARE = {
    "p4d.24xlarge":  InstanceHardware("p4d.24xlarge", "A100", 8, 40, 320, 96, 1152, 400),
    "p4de.24xlarge": InstanceHardware("p4de.24xlarge", "A100", 8, 80, 640, 96, 1152, 400),
    "p5.48xlarge":   InstanceHardware("p5.48xlarge", "H100", 8, 80, 640, 192, 2048, 3200),
    "p3.16xlarge":   InstanceHardware("p3.16xlarge", "V100", 8, 16, 128, 64, 488, 25),
    "p3dn.24xlarge": InstanceHardware("p3dn.24xlarge", "V100", 8, 32, 256, 96, 768, 100),
}

FIXTURE_PRICES = {
    "p4d.24xlarge": 26.87,
    "p4de.24xlarge": 32.77,
    "p5.48xlarge": 66.64,
    "p3.16xlarge": 24.48,
    "p3dn.24xlarge": 31.22,
}

FIXTURE_STORAGE = {
    "standard": 23.0,
    "intelligent_tiering": 23.0,
    "standard_ia": 13.8,
    "one_zone_ia": 11.0,
    "glacier": 4.0,
}


def primed_store(region: str = "us-east-1", prices=None, storage=None) -> InMemoryStore:
    """A store already holding fresh specs and prices for one region."""
    store = InMemoryStore()
    now = datetime.now(timezone.utc)

    cache.save_specs(store, SpecsSnapshot(fetched_at=now, instances=dict(FIXTURE_HARDWARE)))
    prices = FIXTURE_PRICES if prices is None else prices
    cache.save_region(
        store,
        RegionPricing(
            region=region,
            fetched_at=now,
            prices=dict(prices),
            storage=dict(FIXTURE_STORAGE if storage is None else storage),
            offered=tuple(sorted(prices)),
        ),
        build_settings(),
    )
    return store


@pytest.fixture
def store() -> InMemoryStore:
    return primed_store()


@pytest.fixture(autouse=True)
def _isolated_catalog(monkeypatch, request):
    """Serve every catalog lookup from a primed in-memory store.

    Autouse so no test can accidentally reach AWS, and so the memoised catalog
    never leaks between tests.
    """
    catalog_module.invalidate()

    if "no_primed_cache" in request.keywords:
        yield
        catalog_module.invalidate()
        return

    shared = primed_store()
    monkeypatch.setattr(catalog_module, "build_store", lambda _settings: shared)
    yield shared
    catalog_module.invalidate()
