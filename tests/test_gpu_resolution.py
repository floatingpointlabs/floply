"""GPU hardware curation: lookup, aliases, overrides, and quarantine.

The quarantine path is what keeps auto-discovery safe — an instance whose GPU
has no curated throughput must disappear from the catalog rather than appear
with a guessed number.
"""

import pytest
import yaml

from src.cost_modelling import gpu_hardware
from src.cost_modelling.gpu_hardware import (
    GpuHardware,
    UnsupportedPrecisionError,
    hardware_fields_for,
    load_gpu_hardware,
    resolve_gpu,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    gpu_hardware.clear_cache()
    yield
    gpu_hardware.clear_cache()


# -- Lookup -----------------------------------------------------------------

def test_every_current_gpu_is_curated():
    assert {"V100", "A100", "H100"} <= set(load_gpu_hardware())


def test_resolve_by_canonical_name():
    assert resolve_gpu("H100").peak_flops_fp16 == pytest.approx(989e12)


def test_resolve_by_alias():
    """DescribeInstanceTypes reports names like 'A100-SXM4-80GB'."""
    assert resolve_gpu("A100-SXM4-80GB") is resolve_gpu("A100")


def test_uncurated_gpu_resolves_to_none():
    assert resolve_gpu("RTX-9090-Ti") is None


def test_scientific_notation_is_cast_to_float():
    """PyYAML parses 312.0e12 as a string; the loader must coerce it."""
    hardware = resolve_gpu("A100")
    assert isinstance(hardware.peak_flops_fp16, float)
    assert hardware.peak_flops_fp16 == 312e12


# -- Quarantine -------------------------------------------------------------

def test_uncurated_gpu_yields_no_hardware_fields():
    assert hardware_fields_for("RTX-9090-Ti", "g99.xlarge") is None


def test_quarantine_excludes_instance_from_the_built_catalog():
    """An instance with an unknown GPU is dropped, not priced with a guess."""
    from datetime import datetime, timezone

    from src.cost_modelling.pricing.catalog import Provenance, build_catalog
    from src.cost_modelling.pricing.models import (
        InstanceHardware, RegionPricing, SpecsSnapshot)

    now = datetime.now(timezone.utc)
    specs = SpecsSnapshot(fetched_at=now, instances={
        "p5.48xlarge": InstanceHardware("p5.48xlarge", "H100", 8, 80, 640, 192, 2048, 3200),
        "p9.future":   InstanceHardware("p9.future", "Z900", 8, 80, 640, 192, 2048, 3200),
    })
    pricing = RegionPricing(
        region="us-east-1", fetched_at=now,
        prices={"p5.48xlarge": 66.64, "p9.future": 99.0})

    catalog = build_catalog(
        "us-east-1", specs, pricing,
        Provenance(source="live", region="us-east-1"))

    assert catalog.list_instances() == ["p5.48xlarge"]
    assert "Z900" in catalog.provenance.quarantined["p9.future"]
    assert "gpu_hardware.yaml" in catalog.provenance.quarantined["p9.future"]


def test_unknown_instance_raises_listing_what_is_available():
    from src.cost_modelling.gpu_specs import get_gpu_instance

    with pytest.raises(ValueError, match="Unknown instance type"):
        get_gpu_instance("p9.future")


# -- Instance overrides -----------------------------------------------------

def test_instance_override_beats_the_gpu_model_entry(monkeypatch):
    """AWS reports 'H100' for both SXM and PCIe parts, which differ in throughput."""
    monkeypatch.setattr(
        gpu_hardware, "load_instance_overrides",
        lambda: {"p5-pcie.48xlarge": {"typical_mfu": 0.31, "peak_flops_fp16": 756e12}},
    )
    fields = hardware_fields_for("H100", "p5-pcie.48xlarge")
    assert fields["typical_mfu"] == 0.31
    assert fields["peak_flops_fp16"] == 756e12

    untouched = hardware_fields_for("H100", "p5.48xlarge")
    assert untouched["typical_mfu"] == 0.55
    assert untouched["peak_flops_fp16"] == 989e12


# -- Precision support ------------------------------------------------------

def _gpu(**multipliers) -> GpuHardware:
    return GpuHardware(
        name="Test", vendor="NVIDIA", peak_flops_fp16=100.0,
        peak_flops_fp32=10.0, typical_mfu=0.5, precision_multipliers=multipliers,
    )


def test_multiplier_scales_fp16():
    assert _gpu(fp8=2.0).peak_flops("fp8") == 200.0


def test_null_multiplier_means_use_the_fp32_spec():
    assert _gpu(fp32=None).peak_flops("fp32") == 10.0


def test_absent_multiplier_means_unsupported_not_one():
    with pytest.raises(UnsupportedPrecisionError):
        _gpu(fp16=1.0).peak_flops("fp4")


def test_error_names_the_supported_formats():
    with pytest.raises(UnsupportedPrecisionError, match="fp16, fp32"):
        _gpu(fp16=1.0, fp32=None).peak_flops("fp4")


def test_supported_precisions_uses_display_order():
    gpu = _gpu(fp32=None, fp16=1.0, int8=2.0, bf16=1.0)
    assert gpu.supported_precisions == ["int8", "bf16", "fp16", "fp32"]


# -- Curated data integrity -------------------------------------------------

def test_curated_entries_are_internally_consistent():
    for name, hardware in load_gpu_hardware().items():
        assert hardware.peak_flops_fp16 > 0, name
        assert hardware.peak_flops_fp32 > 0, name
        assert 0 < hardware.typical_mfu <= 1.0, name
        assert hardware.precision_multipliers, name
        assert hardware.source, f"{name} must cite a datasheet"
        assert "fp16" in hardware.precision_multipliers, name
        assert hardware.precision_multipliers.get("fp32", "missing") is None, name


def test_aliases_are_unique_across_gpus():
    seen: dict[str, str] = {}
    for name, hardware in load_gpu_hardware().items():
        for alias in hardware.aliases:
            assert alias not in seen, f"{alias} claimed by both {seen.get(alias)} and {name}"
            seen[alias] = name


def test_only_blackwell_and_later_advertise_fp4():
    """The bug this file exists to prevent: fp4 on pre-Blackwell silicon."""
    for name, hardware in load_gpu_hardware().items():
        if "fp4" in hardware.precision_multipliers:
            assert name in {"B200"}, f"{name} should not advertise fp4"


def test_schema_version_mismatch_is_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "gpu_hardware.yaml"
    bad.write_text(yaml.safe_dump({"schema_version": 99, "gpus": {}}))
    monkeypatch.setattr(gpu_hardware, "GPU_HARDWARE_PATH", bad)
    gpu_hardware.clear_cache()
    with pytest.raises(ValueError, match="schema_version"):
        load_gpu_hardware()
