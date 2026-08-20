"""Cache keys, serialization, and staleness rules.

Split by access pattern rather than one blob: hardware specs are near-static
and region-agnostic, prices change per region. That split is what makes a cold
start cost one region's worth of calls instead of all six.

Staleness is not expiry. A stale entry is still served — with a warning — so
that an AWS outage degrades the numbers' freshness rather than taking the app
down. Only a schema mismatch or corruption discards data.
"""

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from src.cost_modelling.pricing.models import (
    InstanceHardware,
    RegionPricing,
    SpecsSnapshot,
)
from src.cost_modelling.pricing.sanity import is_acceptable_snapshot
from src.cost_modelling.pricing.settings import PricingSettings
from src.cost_modelling.pricing.store import CacheStore

log = logging.getLogger(__name__)

# Bumping this invalidates every cached entry. The version lives in the key so
# old and new containers can share a volume without fighting.
CACHE_SCHEMA_VERSION = 1

SPECS_KEY = f"specs_v{CACHE_SCHEMA_VERSION}"


def region_key(region: str) -> str:
    return f"catalog_v{CACHE_SCHEMA_VERSION}_{region}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _parse_iso(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def age(fetched_at: datetime) -> timedelta:
    return _now() - fetched_at


def is_stale(fetched_at: datetime | None, ttl_days: int) -> bool:
    """Whether an entry is past its refresh threshold.

    A timestamp in the future counts as stale: clock skew must not pin a cache
    as permanently fresh.
    """
    if fetched_at is None:
        return True
    delta = age(fetched_at)
    return delta > timedelta(days=ttl_days) or delta < timedelta(0)


# -- Specs ------------------------------------------------------------------

def specs_to_payload(snapshot: SpecsSnapshot) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fetched_at": _iso(snapshot.fetched_at),
        "warnings": list(snapshot.warnings),
        "instances": {
            name: {
                "gpu": hw.gpu,
                "gpu_count": hw.gpu_count,
                "memory_per_gpu": hw.memory_per_gpu,
                "total_gpu_memory": hw.total_gpu_memory,
                "vcpus": hw.vcpus,
                "system_memory": hw.system_memory,
                "network_bandwidth": hw.network_bandwidth,
                "current_generation": hw.current_generation,
            }
            for name, hw in snapshot.instances.items()
        },
    }


def payload_to_specs(payload: dict[str, Any]) -> SpecsSnapshot | None:
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        log.info("Ignoring specs cache with schema_version %r",
                 payload.get("schema_version"))
        return None

    fetched_at = _parse_iso(payload.get("fetched_at"))
    if fetched_at is None:
        return None

    instances: dict[str, InstanceHardware] = {}
    for name, raw in (payload.get("instances") or {}).items():
        try:
            instances[name] = InstanceHardware(instance_type=name, **raw)
        except TypeError:
            log.warning("Skipping malformed cached spec for %s", name)

    return SpecsSnapshot(
        fetched_at=fetched_at,
        instances=instances,
        warnings=tuple(payload.get("warnings") or ()),
    )


def load_specs(store: CacheStore) -> SpecsSnapshot | None:
    payload = store.read(SPECS_KEY)
    return payload_to_specs(payload) if payload else None


def save_specs(store: CacheStore, snapshot: SpecsSnapshot) -> bool:
    if not snapshot.instances:
        log.warning("Refusing to cache an empty specs snapshot")
        return False
    with store.lock(SPECS_KEY) as acquired:
        if not acquired:
            log.info("Another process is writing specs; skipping")
            return False
        return store.write(SPECS_KEY, specs_to_payload(snapshot))


# -- Region pricing ---------------------------------------------------------

def region_to_payload(pricing: RegionPricing) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "region": pricing.region,
        "fetched_at": _iso(pricing.fetched_at),
        "prices": dict(pricing.prices),
        "storage": dict(pricing.storage),
        "offered": list(pricing.offered),
        "warnings": list(pricing.warnings),
    }


def payload_to_region(payload: dict[str, Any]) -> RegionPricing | None:
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        log.info("Ignoring pricing cache with schema_version %r",
                 payload.get("schema_version"))
        return None

    fetched_at = _parse_iso(payload.get("fetched_at"))
    region = payload.get("region")
    if fetched_at is None or not isinstance(region, str):
        return None

    def _floats(raw: Any) -> dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in raw.items():
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                log.warning("Dropping non-numeric cached price %s=%r", key, value)
        return out

    return RegionPricing(
        region=region,
        fetched_at=fetched_at,
        prices=_floats(payload.get("prices")),
        storage=_floats(payload.get("storage")),
        offered=tuple(payload.get("offered") or ()),
        warnings=tuple(payload.get("warnings") or ()),
    )


def load_region(store: CacheStore, region: str) -> RegionPricing | None:
    payload = store.read(region_key(region))
    return payload_to_region(payload) if payload else None


def save_region(
    store: CacheStore,
    pricing: RegionPricing,
    settings: PricingSettings,
    previous: RegionPricing | None = None,
) -> tuple[bool, str | None]:
    """Persist a region, subject to the acceptance gate.

    Returns (written, reason_if_rejected). A partial fetch is never allowed to
    replace a good entry — with no bundled fallback, the cache is the only
    thing keeping the app usable when AWS is unreachable.
    """
    acceptable, reason = is_acceptable_snapshot(pricing, settings)
    if not acceptable:
        if previous is None:
            # Nothing cached yet, so a thin entry still beats nothing.
            log.info("%s Accepting it anyway: no previous entry exists.", reason)
        else:
            log.warning(reason)
            return False, reason

    with store.lock(region_key(pricing.region)) as acquired:
        if not acquired:
            return False, f"{pricing.region}: another process holds the cache lock."
        written = store.write(region_key(pricing.region), region_to_payload(pricing))
    return written, None if written else f"{pricing.region}: cache write failed."


def mark_stale_warning(pricing: RegionPricing, ttl_days: int) -> RegionPricing:
    """Append a staleness note so the UI can surface it verbatim."""
    if not is_stale(pricing.fetched_at, ttl_days):
        return pricing
    days = max(age(pricing.fetched_at).days, 0)
    return replace(
        pricing,
        warnings=pricing.warnings + (
            f"{pricing.region}: prices are {days} days old (refresh threshold is "
            f"{ttl_days} days) and could not be refreshed.",
        ),
    )
