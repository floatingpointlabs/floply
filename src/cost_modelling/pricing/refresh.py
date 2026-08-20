"""Fetch -> validate -> persist orchestration.

Sits between the raw AWS client and the catalog. Everything here is
Streamlit-free and safe to call from a background thread.
"""

import logging
import time
from dataclasses import dataclass, field

from src.cost_modelling.gpu_hardware import resolve_gpu
from src.cost_modelling.pricing import aws_client, cache
from src.cost_modelling.pricing.models import (
    ProbeResult,
    RegionPricing,
    SpecsSnapshot,
)
from src.cost_modelling.pricing.sanity import check_hardware, validate_region_pricing
from src.cost_modelling.pricing.settings import PricingSettings, get_settings
from src.cost_modelling.pricing.store import CacheStore, build_store

log = logging.getLogger(__name__)


@dataclass
class RefreshResult:
    ok: bool = True
    specs: SpecsSnapshot | None = None
    regions: dict[str, RegionPricing] = field(default_factory=dict)
    quarantined: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    store_description: str = ""


def probe(settings: PricingSettings | None = None) -> ProbeResult:
    return aws_client.probe_credentials(settings or get_settings())


def partition_by_curation(snapshot: SpecsSnapshot) -> tuple[list[str], dict[str, str]]:
    """Split discovered instances into usable and quarantined.

    An instance whose GPU model has no curated throughput must not be priced or
    shown — a guessed FLOPs figure produces a confidently wrong cost.
    """
    usable: list[str] = []
    quarantined: dict[str, str] = {}

    for name, hardware in snapshot.instances.items():
        defect = check_hardware(hardware)
        if defect:
            quarantined[name] = defect
            continue
        if resolve_gpu(hardware.gpu) is None:
            quarantined[name] = (
                f'no curated FLOPs for GPU "{hardware.gpu}". '
                f"Add it to src/data/gpu_hardware.yaml."
            )
            continue
        usable.append(name)

    return sorted(usable), quarantined


def refresh_specs(
    store: CacheStore,
    settings: PricingSettings,
    force: bool = False,
) -> tuple[SpecsSnapshot | None, list[str]]:
    """Refresh hardware specs if stale, otherwise return the cached snapshot."""
    cached = cache.load_specs(store)
    if not force and cached and not cache.is_stale(cached.fetched_at, settings.specs_ttl_days):
        return cached, []

    try:
        fetched = aws_client.fetch_instance_hardware(settings)
    except Exception as exc:                       # noqa: BLE001 - degrade, never crash
        log.warning("Specs refresh failed: %s", exc)
        message = f"Hardware specs could not be refreshed ({type(exc).__name__}: {exc})."
        if cached:
            return cached, [message + " Using cached specs."]
        return None, [message]

    if not fetched.instances:
        message = "AWS returned no NVIDIA GPU instance types; keeping previous specs."
        return (cached, [message]) if cached else (None, [message])

    cache.save_specs(store, fetched)
    return fetched, list(fetched.warnings)


def refresh_region(
    store: CacheStore,
    region: str,
    instance_types: list[str],
    settings: PricingSettings,
    force: bool = False,
) -> tuple[RegionPricing | None, list[str]]:
    """Refresh one region's prices if stale, otherwise return the cached entry."""
    cached = cache.load_region(store, region)
    if not force and cached and not cache.is_stale(cached.fetched_at, settings.pricing_ttl_days):
        return cached, []

    try:
        offered = aws_client.fetch_offered_instance_types(region, settings)
        wanted = [t for t in instance_types if t in offered] if offered else instance_types
        fetched = aws_client.fetch_region_pricing(region, wanted, settings)
    except Exception as exc:                       # noqa: BLE001 - degrade, never crash
        log.warning("Pricing refresh failed for %s: %s", region, exc)
        message = f"{region}: prices could not be refreshed ({type(exc).__name__}: {exc})."
        if cached:
            return cache.mark_stale_warning(cached, settings.pricing_ttl_days), [message]
        return None, [message]

    validated = validate_region_pricing(fetched, cached, settings)
    written, reason = cache.save_region(store, validated, settings, previous=cached)

    warnings = list(validated.warnings)
    if not written and reason:
        warnings.append(reason)
        if cached:
            # The fetch was rejected; keep serving what we already trusted.
            return cache.mark_stale_warning(cached, settings.pricing_ttl_days), warnings
    return validated, warnings


def refresh_all(
    regions: tuple[str, ...] | None = None,
    force: bool = False,
    settings: PricingSettings | None = None,
    store: CacheStore | None = None,
) -> RefreshResult:
    """Refresh specs and every configured region.

    Regions are independent: one failing does not abort the others, and each is
    gated separately so a partial outage cannot poison good cache entries.
    """
    settings = settings or get_settings()
    store = store or build_store(settings)
    regions = regions or settings.regions

    result = RefreshResult(store_description=store.description)
    started = time.monotonic()

    specs, warnings = refresh_specs(store, settings, force=force)
    result.specs = specs
    result.warnings.extend(warnings)

    if specs is None:
        result.ok = False
        result.warnings.append("No hardware specs available; cannot price instances.")
        return result

    usable, quarantined = partition_by_curation(specs)
    result.quarantined = quarantined

    if not usable:
        result.ok = False
        result.warnings.append(
            "Every discovered instance was quarantined; none have curated FLOPs.")
        return result

    for region in regions:
        if time.monotonic() - started > settings.max_refresh_seconds:
            result.ok = False
            result.warnings.append(
                f"Refresh budget of {settings.max_refresh_seconds}s exhausted; "
                f"stopped before {region}.")
            break

        pricing, region_warnings = refresh_region(
            store, region, usable, settings, force=force)
        result.warnings.extend(region_warnings)
        if pricing is None:
            result.ok = False
            continue
        result.regions[region] = pricing

    if not result.regions:
        result.ok = False

    return result
