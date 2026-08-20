"""Loader for curated GPU hardware facts from src/data/gpu_hardware.yaml.

These are the values no AWS API publishes — NVIDIA datasheet throughput and
empirical MFU. Everything else about an instance (price, vCPUs, memory, GPU
model, GPU count) comes from AWS.

Adding support for a new GPU generation is a YAML edit, not a code change.
A GPU with no entry here is quarantined by the caller rather than guessed at.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

GPU_HARDWARE_PATH = Path(__file__).parent.parent / "data" / "gpu_hardware.yaml"

SCHEMA_VERSION = 1


class UnsupportedPrecisionError(ValueError):
    """The requested numeric format has no hardware support on this GPU."""


@dataclass(frozen=True)
class GpuHardware:
    """Curated datasheet facts for one GPU model."""

    name: str
    vendor: str
    peak_flops_fp16: float
    peak_flops_fp32: float
    typical_mfu: float
    precision_multipliers: dict[str, float | None] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    description: str = ""
    source: str = ""

    @property
    def supported_precisions(self) -> list[str]:
        """Precisions this die has hardware support for, in config.py order."""
        return [p for p in _PRECISION_ORDER if p in self.precision_multipliers]

    def peak_flops(self, precision: str) -> float:
        """Effective peak FLOPs/s for a numeric format.

        Raises:
            UnsupportedPrecisionError: if the die has no support for it. An
                absent multiplier means unsupported, never 1.0 — silently
                falling back to the fp16 baseline is what previously made
                fp4 on an A100 report double the real throughput.
        """
        if precision not in self.precision_multipliers:
            supported = ", ".join(self.supported_precisions)
            raise UnsupportedPrecisionError(
                f"{self.name} has no hardware support for {precision}. "
                f"Supported: {supported}"
            )
        multiplier = self.precision_multipliers[precision]
        if multiplier is None:
            return self.peak_flops_fp32
        return self.peak_flops_fp16 * multiplier


# Display order for precision lists, fastest/smallest -> most precise.
_PRECISION_ORDER = ["fp4", "int8", "fp8", "bf16", "fp16", "tf32", "fp32"]

_FLOAT_FIELDS = ("peak_flops_fp16", "peak_flops_fp32", "typical_mfu")


def _coerce(spec: dict[str, Any], gpu_name: str) -> GpuHardware:
    """Build a GpuHardware, casting YAML scientific notation to float.

    PyYAML parses ``312.0e12`` as a string, so numeric fields need explicit
    casting.
    """
    values = {f: float(spec[f]) for f in _FLOAT_FIELDS}
    multipliers: dict[str, float | None] = {
        precision: (None if raw is None else float(raw))
        for precision, raw in (spec.get("precision_multipliers") or {}).items()
    }
    return GpuHardware(
        name=gpu_name,
        vendor=spec.get("vendor", "NVIDIA"),
        precision_multipliers=multipliers,
        aliases=tuple(spec.get("aliases", ())),
        description=spec.get("description", ""),
        source=spec.get("source", ""),
        **values,
    )


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, GpuHardware], dict[str, dict[str, Any]]]:
    with open(GPU_HARDWARE_PATH) as f:
        data = yaml.safe_load(f)

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{GPU_HARDWARE_PATH.name} has schema_version {version}, "
            f"expected {SCHEMA_VERSION}"
        )

    gpus = {name: _coerce(spec, name) for name, spec in (data.get("gpus") or {}).items()}
    overrides = data.get("instance_overrides") or {}
    return gpus, overrides


def load_gpu_hardware() -> dict[str, GpuHardware]:
    """All curated GPU entries, keyed by canonical GPU model name."""
    return _load()[0]


def load_instance_overrides() -> dict[str, dict[str, Any]]:
    """Per-instance overrides that win over the GPU-model entry."""
    return _load()[1]


def resolve_gpu(gpu_name: str) -> GpuHardware | None:
    """Look up a GPU by canonical name or alias. None when uncurated."""
    gpus = load_gpu_hardware()
    if gpu_name in gpus:
        return gpus[gpu_name]
    for hardware in gpus.values():
        if gpu_name in hardware.aliases:
            return hardware
    return None


def hardware_fields_for(gpu_name: str, instance_type: str = "") -> dict[str, Any] | None:
    """Curated fields to merge into an instance spec, or None if uncurated.

    Applies ``instance_overrides`` on top of the GPU-model entry.
    """
    hardware = resolve_gpu(gpu_name)
    if hardware is None:
        return None

    fields: dict[str, Any] = {
        "peak_flops_fp16": hardware.peak_flops_fp16,
        "peak_flops_fp32": hardware.peak_flops_fp32,
        "typical_mfu": hardware.typical_mfu,
        "precision_multipliers": dict(hardware.precision_multipliers),
        "gpu_description": hardware.description,
    }
    fields.update(load_instance_overrides().get(instance_type, {}))
    return fields


def clear_cache() -> None:
    """Drop the memoised YAML parse. For tests."""
    _load.cache_clear()
