"""Validation for fetched pricing and hardware.

Two tiers, because there is no bundled baseline to compare against:

- Absolute bounds, always applied. Catches a value that cannot be right
  regardless of history.
- Relative drift, only when a previous cached value exists. Catches a filter
  regression that returns a plausible number for the *wrong product* — a
  Windows or capacity-block SKU is usually a multiple of the on-demand rate,
  not an implausible absolute figure.

A rejected value must never reach a cost calculation and must never be dropped
silently.
"""

import math
from dataclasses import replace

from src.cost_modelling.pricing.models import InstanceHardware, RegionPricing
from src.cost_modelling.pricing.settings import PricingSettings

# (minimum, maximum) inclusive. Wide enough to admit any real AWS GPU
# instance, narrow enough to catch a parse landing on the wrong field.
HOURLY_COST_BOUNDS = (0.05, 1000.0)
STORAGE_COST_BOUNDS = (0.1, 500.0)
GPU_COUNT_BOUNDS = (1, 64)
GPU_MEMORY_BOUNDS = (1, 1024)
PEAK_FLOPS_BOUNDS = (1e12, 1e16)


def _out_of_bounds(value: float, bounds: tuple[float, float]) -> bool:
    low, high = bounds
    return not math.isfinite(value) or value < low or value > high


def check_hourly_cost(instance_type: str, region: str, price: float) -> str | None:
    """Reason the price is implausible, or None if acceptable."""
    if _out_of_bounds(price, HOURLY_COST_BOUNDS):
        low, high = HOURLY_COST_BOUNDS
        return (
            f"{instance_type}/{region}: ${price:,.2f}/hr is outside the plausible "
            f"range ${low}–${high}/hr."
        )
    return None


def check_storage_cost(storage_class: str, region: str, price: float) -> str | None:
    if _out_of_bounds(price, STORAGE_COST_BOUNDS):
        low, high = STORAGE_COST_BOUNDS
        return (
            f"{storage_class}/{region}: ${price:,.2f}/TB/mo is outside the "
            f"plausible range ${low}–${high}."
        )
    return None


def check_drift(
    label: str,
    new_value: float,
    previous_value: float | None,
    max_drift: float,
) -> str | None:
    """Reason the change from the cached value is suspicious, or None.

    The band is deliberately loose. AWS's p5 reduction was ~0.56x of list and
    must pass; a filter regression picking up a Windows SKU is typically 2x+.
    """
    if previous_value is None or previous_value <= 0 or max_drift <= 1:
        return None

    ratio = new_value / previous_value
    if ratio > max_drift or ratio < 1 / max_drift:
        direction = "jumped" if ratio > 1 else "dropped"
        return (
            f"{label}: price {direction} {ratio:.1f}x versus the cached value "
            f"(${previous_value:,.2f} -> ${new_value:,.2f}); keeping the cached "
            f"price. Check the on-demand filters."
        )
    return None


def check_hardware(hardware: InstanceHardware) -> str | None:
    """Reason the hardware specs are internally inconsistent, or None."""
    name = hardware.instance_type

    if _out_of_bounds(hardware.gpu_count, GPU_COUNT_BOUNDS):
        return f"{name}: implausible gpu_count {hardware.gpu_count}."
    if _out_of_bounds(hardware.memory_per_gpu, GPU_MEMORY_BOUNDS):
        return f"{name}: implausible memory_per_gpu {hardware.memory_per_gpu} GB."
    if hardware.vcpus < 1:
        return f"{name}: implausible vcpus {hardware.vcpus}."
    if not hardware.gpu:
        return f"{name}: missing GPU model name."

    expected = hardware.gpu_count * hardware.memory_per_gpu
    if abs(hardware.total_gpu_memory - expected) > hardware.gpu_count:
        return (
            f"{name}: total_gpu_memory {hardware.total_gpu_memory} GB does not "
            f"match {hardware.gpu_count} x {hardware.memory_per_gpu} GB."
        )
    return None


def validate_region_pricing(
    fetched: RegionPricing,
    previous: RegionPricing | None,
    settings: PricingSettings,
) -> RegionPricing:
    """Drop implausible prices, falling back to the cached value where possible.

    Returns a new RegionPricing with rejections removed and every rejection
    appended to ``warnings``.
    """
    previous_prices = previous.prices if previous else {}
    previous_storage = previous.storage if previous else {}

    prices: dict[str, float] = {}
    storage: dict[str, float] = {}
    warnings = list(fetched.warnings)

    for instance_type, price in fetched.prices.items():
        cached = previous_prices.get(instance_type)
        reason = check_hourly_cost(instance_type, fetched.region, price)
        if reason is None:
            reason = check_drift(
                f"{instance_type}/{fetched.region}", price, cached, settings.max_drift)
        if reason is None:
            prices[instance_type] = price
            continue

        warnings.append(reason)
        if cached is not None:
            prices[instance_type] = cached      # keep the last trusted value

    for storage_class, price in fetched.storage.items():
        cached = previous_storage.get(storage_class)
        reason = check_storage_cost(storage_class, fetched.region, price)
        if reason is None:
            reason = check_drift(
                f"{storage_class}/{fetched.region}", price, cached, settings.max_drift)
        if reason is None:
            storage[storage_class] = price
            continue

        warnings.append(reason)
        if cached is not None:
            storage[storage_class] = cached

    return replace(
        fetched,
        prices=prices,
        storage=storage,
        offered=tuple(sorted(prices)),
        warnings=tuple(warnings),
    )


def is_acceptable_snapshot(
    pricing: RegionPricing,
    settings: PricingSettings,
) -> tuple[bool, str | None]:
    """Whether a fetched region is complete enough to replace the cached one.

    A half-successful fetch overwriting a good cache is worse than no fetch,
    since the cache is the only availability layer.
    """
    if len(pricing.prices) < settings.min_instances:
        return False, (
            f"{pricing.region}: only {len(pricing.prices)} instance price(s) "
            f"fetched, need at least {settings.min_instances}; keeping the "
            f"previous cache entry."
        )
    return True, None
