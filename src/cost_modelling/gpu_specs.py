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


# Which GPUs have hardware acceleration for each low-precision format.
#
# FP4 is Blackwell-only (B100/B200/GB200), so no GPU currently in data/providers
# supports it. The set is kept rather than deleted so adding a Blackwell instance is a
# one-line change here.
#
# FP8 arrived with Hopper (Transformer Engine); Ampere and Volta have no FP8 path.
# INT8 tensor cores arrived with Turing/Ampere; Volta only has the slower DP4A path.
FP4_GPUS: set[str] = set()
FP8_GPUS = {"H100"}
INT8_TENSOR_GPUS = {"A100", "H100"}


def peak_flops_for_precision(instance_spec: dict, mixed_precision: str) -> float:
    """Return effective peak FLOPs/s for the given instance and numeric precision.

    Sources:
    - A100 fp16/bf16: 312 TFLOPS (NVIDIA datasheet, without sparsity)
    - A100 tf32: 156 TFLOPS (0.5× fp16)
    - A100 fp32: 19.5 TFLOPS (NVIDIA datasheet)
    - A100 int8: 624 TOPS (2× fp16 tensor cores)
    - H100 fp16/bf16: 989 TFLOPS; fp8: ~1979 TFLOPS (2× fp16)
    - V100 fp16: 125 TFLOPS; int8: limited support (~0.9× fp16)

    A precision the GPU cannot accelerate falls back to its FP16 rate — the honest
    reading being "you would run this in FP16 instead". Inventing a speedup for absent
    hardware understates cost, which is the dangerous direction for a budget estimate.
    """
    fp16 = instance_spec["peak_flops_fp16"]
    fp32 = instance_spec["peak_flops_fp32"]
    gpu  = instance_spec["gpu"]
    return {
        "fp4":  fp16 * (2.0 if gpu in FP4_GPUS else 1.0),
        "int8": fp16 * (2.0 if gpu in INT8_TENSOR_GPUS else 0.9),
        "fp8":  fp16 * (2.0 if gpu in FP8_GPUS else 1.0),
        "bf16": fp16,
        "fp16": fp16,
        "tf32": fp16 * 0.5,
        "fp32": fp32,
    }.get(mixed_precision, fp16)


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
