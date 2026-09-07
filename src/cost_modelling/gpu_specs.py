"""GPU instance specifications and pricing.

Prices and hardware specs come from AWS via the cache in
``src.cost_modelling.pricing``; throughput and MFU come from the curated
``src/data/gpu_hardware.yaml``. There is no bundled price fallback — when
neither AWS nor the cache can supply a region, callers get
:class:`PricingUnavailableError` rather than a plausible-looking guess.

Streamlit-free by design: the UI passes ``region`` explicitly, and ``None``
means the configured default.
"""

from typing import Any

from src.cost_modelling.gpu_hardware import UnsupportedPrecisionError
from src.cost_modelling.pricing.catalog import (
    Catalog,
    PricingUnavailableError,
    Provenance,
    get_catalog,
    invalidate,
)
from src.cost_modelling.pricing.settings import get_settings

_PRECISION_ORDER = ["fp4", "int8", "fp8", "bf16", "fp16", "tf32", "fp32"]

# Populated from src/providers/aws.yaml (storage section)
S3_STORAGE_PRICING: dict[str, float] = load_storage_pricing()


def _region(region: str | None) -> str:
    return region or get_settings().default_region


def catalog_for(region: str | None = None) -> Catalog:
    """The resolved catalog for a region. Fetches on a cold cache."""
    return get_catalog(_region(region))


def _require(region: str | None) -> Catalog:
    catalog = catalog_for(region)
    if not catalog.provenance.is_usable:
        raise PricingUnavailableError(
            catalog.region, catalog.provenance.last_error or "No cached pricing.")
    return catalog


def get_provenance(region: str | None = None) -> Provenance:
    """Where this region's numbers came from. Never raises."""
    return catalog_for(region).provenance


def invalidate_catalog(region: str | None = None) -> None:
    """Drop the memoised catalog so the next call re-reads the cache."""
    invalidate(region)


def list_regions() -> tuple[str, ...]:
    """Regions this deployment is configured to offer."""
    return get_settings().regions


def get_gpu_instance(instance_type: str, region: str | None = None) -> dict[str, Any]:
    """Full spec for an instance in a region, including its hourly cost."""
    return _require(region).spec_for(instance_type)


def get_storage_cost(storage_class: str = "standard", region: str | None = None) -> float:
    """S3 storage cost per TB per month. Region-dependent."""
    return _require(region).storage_cost(storage_class)


def list_available_instances(region: str | None = None) -> list[str]:
    """Usable instance types, cheapest first. Empty when pricing is unavailable."""
    return catalog_for(region).list_instances()


def list_storage_classes(region: str | None = None) -> list[str]:
    """Storage classes with a known price in this region."""
    return catalog_for(region).list_storage_classes()


def format_instance_label(spec: dict[str, Any]) -> str:
    """Selectbox label. Built from data, so it can never drift from the price."""
    return f"{spec['display_name']} — ${spec['hourly_cost']:,.2f}/hr"


def supported_precisions(instance_spec: dict[str, Any]) -> list[str]:
    """Numeric formats this instance's GPU has hardware support for.

    Callers should offer only these — an unsupported format used to silently
    return the fp16 baseline, overstating throughput (and so understating cost)
    by up to 2x.
    """
    multipliers = instance_spec.get("precision_multipliers", {})
    return [p for p in _PRECISION_ORDER if p in multipliers]


def peak_flops_for_precision(instance_spec: dict[str, Any], mixed_precision: str) -> float:
    """Effective peak FLOPs/s for the given instance and numeric precision.

    Multipliers come from src/data/gpu_hardware.yaml, keyed by GPU model, so a
    new GPU generation needs no code change here.

    Raises:
        UnsupportedPrecisionError: if the GPU has no hardware support for the
            format (fp4 on Ampere or Hopper, bf16 on Volta, and so on).
    """
    multipliers = instance_spec.get("precision_multipliers", {})
    if mixed_precision not in multipliers:
        supported = ", ".join(supported_precisions(instance_spec))
        raise UnsupportedPrecisionError(
            f"{instance_spec.get('gpu', 'This GPU')} has no hardware support for "
            f"{mixed_precision}. Supported: {supported}"
        )

    multiplier = multipliers[mixed_precision]
    if multiplier is None:
        return instance_spec["peak_flops_fp32"]
    return instance_spec["peak_flops_fp16"] * multiplier
