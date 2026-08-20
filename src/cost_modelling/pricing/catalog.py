"""Resolution: cache -> join with curated GPU facts -> usable instance catalog.

The resolution chain is live fetch -> disk cache -> error. There is deliberately
no bundled price fallback, so this module's job is to make the *absence* of data
explicit rather than papering over it with a guess.

Fetching is lazy: nothing here touches AWS until a price is actually asked for.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping

from src.cost_modelling.gpu_hardware import hardware_fields_for
from src.cost_modelling.pricing import cache, refresh
from src.cost_modelling.pricing.models import RegionPricing, SpecsSnapshot
from src.cost_modelling.pricing.settings import PricingSettings, get_settings
from src.cost_modelling.pricing.store import CacheStore, build_store

log = logging.getLogger(__name__)

Source = Literal["live", "cache", "unavailable"]


class PricingUnavailableError(RuntimeError):
    """No pricing data at all: nothing cached and AWS could not be reached."""

    def __init__(self, region: str, detail: str) -> None:
        self.region = region
        self.detail = detail
        super().__init__(f"No pricing data available for {region}. {detail}")


@dataclass(frozen=True)
class Provenance:
    """Where the numbers came from, and how much to trust them."""

    source: Source
    region: str
    fetched_at: datetime | None = None
    stale: bool = False
    age_days: int | None = None
    warnings: tuple[str, ...] = ()
    quarantined: Mapping[str, str] = field(default_factory=dict)
    last_error: str | None = None
    store_description: str = ""

    @property
    def is_usable(self) -> bool:
        return self.source != "unavailable"

    def summary(self) -> str:
        """One-line description for the sidebar chip."""
        if self.source == "unavailable":
            return f"Pricing unavailable · {self.region}"
        when = "just now" if not self.age_days else f"{self.age_days}d ago"
        label = "Live" if self.source == "live" else "Cached"
        suffix = " · refresh failed" if self.stale and self.last_error else ""
        return f"{label} · {self.region} · fetched {when}{suffix}"


@dataclass(frozen=True)
class Catalog:
    """Instances and storage prices for one region, ready for costing."""

    region: str
    instances: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    storage: Mapping[str, float] = field(default_factory=dict)
    provenance: Provenance = field(
        default_factory=lambda: Provenance(source="unavailable", region=""))

    def list_instances(self) -> list[str]:
        """Usable instance types, most capable first.

        Ordering is by raw throughput rather than price so that the default
        selection does not move when AWS changes prices, and so the first
        option is a current-generation part rather than the cheapest legacy one.
        """
        return sorted(
            self.instances,
            key=lambda name: (
                -self.instances[name]["peak_flops_fp16"],
                -self.instances[name]["gpu_count"],
                self.instances[name]["hourly_cost"],
                name,
            ),
        )

    def list_storage_classes(self) -> list[str]:
        return sorted(self.storage)

    def spec_for(self, instance_type: str) -> dict[str, Any]:
        if instance_type not in self.instances:
            available = ", ".join(self.list_instances()) or "none"
            raise ValueError(
                f"Unknown instance type: {instance_type} in {self.region}. "
                f"Available: {available}"
            )
        return self.instances[instance_type]

    def storage_cost(self, storage_class: str) -> float:
        if storage_class not in self.storage:
            available = ", ".join(self.list_storage_classes()) or "none"
            raise ValueError(
                f"Unknown storage class: {storage_class} in {self.region}. "
                f"Available: {available}"
            )
        return self.storage[storage_class]


def build_catalog(
    region: str,
    specs: SpecsSnapshot,
    pricing: RegionPricing,
    provenance: Provenance,
) -> Catalog:
    """Join fetched hardware and prices with curated GPU throughput.

    An instance is included only when it has all three: a spec, a price in this
    region, and a curated GPU entry. Anything missing one is left out rather
    than filled in with a default.
    """
    instances: dict[str, dict[str, Any]] = {}
    quarantined = dict(provenance.quarantined)

    for instance_type, hardware in specs.instances.items():
        price = pricing.prices.get(instance_type)
        if price is None:
            continue                                   # not sold in this region

        curated = hardware_fields_for(hardware.gpu, instance_type)
        if curated is None:
            quarantined.setdefault(instance_type, (
                f'no curated FLOPs for GPU "{hardware.gpu}". '
                f"Add it to src/data/gpu_hardware.yaml."
            ))
            continue

        instances[instance_type] = {
            **hardware.as_spec(),
            **curated,
            "hourly_cost": price,
            "region": region,
        }

    from dataclasses import replace
    return Catalog(
        region=region,
        instances=instances,
        storage=dict(pricing.storage),
        provenance=replace(provenance, quarantined=quarantined),
    )


def resolve_catalog(
    region: str,
    settings: PricingSettings | None = None,
    store: CacheStore | None = None,
    allow_network: bool = True,
) -> Catalog:
    """Build the catalog for a region, fetching only if the cache cannot serve.

    Cold cache blocks on one specs fetch plus one region (~8 calls). A stale
    cache is served immediately and flagged; refreshing it is the caller's job
    so that page rendering never waits on the network.
    """
    settings = settings or get_settings()
    store = store or build_store(settings)

    cached_specs = cache.load_specs(store)
    cached_pricing = cache.load_region(store, region)

    specs_cold = cached_specs is None
    pricing_cold = cached_pricing is None
    warnings: list[str] = []
    last_error: str | None = None
    source: Source = "cache"

    if (specs_cold or pricing_cold) and allow_network:
        # Cold: nothing to serve, so this one is worth blocking on.
        specs_result, specs_warnings = refresh.refresh_specs(store, settings)
        warnings.extend(specs_warnings)
        if specs_result is not None:
            cached_specs = specs_result
            usable, quarantined_now = refresh.partition_by_curation(specs_result)
            pricing_result, pricing_warnings = refresh.refresh_region(
                store, region, usable, settings)
            warnings.extend(pricing_warnings)
            if pricing_result is not None:
                cached_pricing = pricing_result
                source = "live"
        if cached_specs is None or cached_pricing is None:
            last_error = "; ".join(warnings) or "AWS could not be reached."

    if cached_specs is None or cached_pricing is None:
        detail = last_error or (
            "No cached pricing and live fetch is disabled."
            if not allow_network else "AWS could not be reached."
        )
        return Catalog(
            region=region,
            provenance=Provenance(
                source="unavailable",
                region=region,
                warnings=tuple(warnings),
                last_error=detail,
                store_description=store.description,
            ),
        )

    stale = cache.is_stale(cached_pricing.fetched_at, settings.pricing_ttl_days)
    if stale:
        cached_pricing = cache.mark_stale_warning(cached_pricing, settings.pricing_ttl_days)
        if allow_network and source != "live":
            # Serve the stale numbers now; refresh off the render path so the
            # page never waits. The next rerun picks up the result.
            from src.cost_modelling.pricing import background
            background.schedule_refresh(region, settings, store)
            last_error = last_error or background.last_error(region)

    _, quarantined = refresh.partition_by_curation(cached_specs)

    provenance = Provenance(
        source=source,
        region=region,
        fetched_at=cached_pricing.fetched_at,
        stale=stale,
        age_days=max(cache.age(cached_pricing.fetched_at).days, 0),
        warnings=tuple(warnings) + cached_pricing.warnings,
        quarantined=quarantined,
        last_error=last_error,
        store_description=store.description,
    )
    return build_catalog(region, cached_specs, cached_pricing, provenance)


# -- Process-wide memo ------------------------------------------------------
# Streamlit runs one ScriptRunner thread per session sharing module globals, so
# without this every concurrent rerun would resolve (and potentially fetch) its
# own copy.

_lock = threading.RLock()
_catalogs: dict[str, Catalog] = {}


def get_catalog(region: str, **kwargs) -> Catalog:
    """Memoised catalog for a region. Only the first caller pays the fetch."""
    with _lock:
        cached = _catalogs.get(region)
        if cached is not None:
            return cached

    resolved = resolve_catalog(region, **kwargs)

    with _lock:
        # Another thread may have resolved the same region meanwhile; either
        # result is equally valid, so keep whichever landed first.
        return _catalogs.setdefault(region, resolved)


def invalidate(region: str | None = None) -> None:
    """Drop memoised catalogs so the next call re-reads the cache."""
    with _lock:
        if region is None:
            _catalogs.clear()
        else:
            _catalogs.pop(region, None)
