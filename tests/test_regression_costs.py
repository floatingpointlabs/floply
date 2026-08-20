"""Golden-value regression tests for the cost engine.

Written before the live-pricing migration as a safety net: the arithmetic in
``calculator.py`` must not change while the data source underneath it moves from
a bundled YAML to the AWS Price List API.

Values here were captured from the pre-migration code against the bundled
``aws.yaml`` snapshot. Tests that depend on a *price* are parameterised on the
spec so they keep working once specs come from the catalog rather than YAML.
"""

import pytest

from src.cost_modelling.calculator import (
    calculate_checkpoint_storage_tb,
    calculate_lora_trainable_params,
    calculate_project_compute_cost,
    calculate_storage_cost,
    calculate_training_flops,
    estimate_compute_cost,
    estimate_gpu_memory_gb,
    solve_for_parameter_count,
    solve_for_training_tokens,
)
from src.cost_modelling.gpu_hardware import UnsupportedPrecisionError
from src.cost_modelling.gpu_specs import (
    get_gpu_instance,
    get_storage_cost,
    peak_flops_for_precision,
    supported_precisions,
)
from src.cost_modelling.model_loader import load_models

H100_PEAK_FP16 = 989e12


# -- FLOPs ------------------------------------------------------------------

def test_training_flops_is_6nd():
    assert calculate_training_flops(7_000_000_000, 1_000_000_000_000) == 4.2e22


def test_gradient_checkpointing_is_8nd():
    base = calculate_training_flops(1_000_000_000, 1_000_000_000)
    ckpt = calculate_training_flops(1_000_000_000, 1_000_000_000, gradient_checkpointing=True)
    assert ckpt / base == pytest.approx(8 / 6)


@pytest.mark.parametrize(
    "architecture, multiplier",
    [("transformer", 6.0), ("cnn", 4.0), ("rnn", 8.0), ("vit", 6.0), ("diffusion", 6.5)],
)
def test_architecture_multipliers(architecture, multiplier):
    flops = calculate_training_flops(1_000_000_000, 1_000_000_000, architecture)
    assert flops == multiplier * 1e18


def test_unknown_architecture_defaults_to_transformer():
    assert calculate_training_flops(1e9, 1e9, "mamba") == calculate_training_flops(1e9, 1e9)


def test_epochs_multiply_tokens():
    single = calculate_training_flops(1e9, 1e9, epochs=1.0)
    triple = calculate_training_flops(1e9, 1e9, epochs=3.0)
    assert triple == 3 * single


# -- Compute cost -----------------------------------------------------------

def test_compute_cost_golden():
    flops = calculate_training_flops(7_000_000_000, 1_000_000_000_000)
    hours, gpu_hours, cost, days = estimate_compute_cost(
        flops, H100_PEAK_FP16, mfu=0.30, total_gpus=8, num_instances=1, hourly_cost=66.64
    )
    assert hours == pytest.approx(4915.178069879788)
    assert gpu_hours == pytest.approx(39321.42455903831)
    assert cost == pytest.approx(327547.4665767891)
    assert days == pytest.approx(204.79908624499117)


def test_compute_cost_scales_linearly_with_instances():
    _, _, one, _ = estimate_compute_cost(1e21, H100_PEAK_FP16, 0.3, 8, 1, 66.64)
    _, _, four, _ = estimate_compute_cost(1e21, H100_PEAK_FP16, 0.3, 32, 4, 66.64)
    assert four == pytest.approx(one)  # 4x GPUs, 4x rate, 1/4 the time


# -- Solvers (must invert the cost formula exactly) -------------------------

def test_solve_for_parameter_count_golden():
    assert solve_for_parameter_count(
        100_000.0, 1_000_000_000_000, H100_PEAK_FP16, 0.3, 8, 1, 66.64
    ) == 2_137_094_837


def test_solve_for_training_tokens_golden():
    assert solve_for_training_tokens(
        100_000.0, 7_000_000_000, H100_PEAK_FP16, 0.3, 8, 1, 66.64
    ) == 305_299_262_562


def test_solve_for_parameter_count_round_trips():
    budget, tokens = 250_000.0, 500_000_000_000
    params = solve_for_parameter_count(budget, tokens, H100_PEAK_FP16, 0.4, 16, 2, 66.64)
    flops = calculate_training_flops(params, tokens)
    _, _, cost, _ = estimate_compute_cost(flops, H100_PEAK_FP16, 0.4, 16, 2, 66.64)
    assert cost == pytest.approx(budget, rel=1e-6)


@pytest.mark.parametrize("bad_kwargs", [
    {"training_tokens": 0},
    {"hourly_cost": 0.0},
    {"num_instances": 0},
])
def test_solvers_guard_against_division_by_zero(bad_kwargs):
    kwargs = dict(
        compute_budget_usd=1000.0, training_tokens=1_000_000, peak_flops_per_gpu=H100_PEAK_FP16,
        mfu=0.3, total_gpus=8, num_instances=1, hourly_cost=66.64,
    )
    kwargs.update(bad_kwargs)
    assert solve_for_parameter_count(**kwargs) == 0


# -- Memory -----------------------------------------------------------------

def test_full_finetune_memory_golden():
    weights, activations, per_gpu = estimate_gpu_memory_gb(
        effective_params=7_000_000_000, trainable_params=7_000_000_000, ft_method=None,
        total_gpus=8, rl_multiplier=1, d_model=4096, num_layers=32,
        seq_len=2048, batch_size=8, gradient_checkpointing=False,
    )
    assert weights == pytest.approx(14.0)
    assert activations == pytest.approx(2.147483648)
    assert per_gpu == pytest.approx(16.147483648)


def test_lora_memory_golden():
    weights, activations, per_gpu = estimate_gpu_memory_gb(
        effective_params=8_030_000_000, trainable_params=20_971_520, ft_method="LoRA",
        total_gpus=8, rl_multiplier=1, d_model=4096, num_layers=32,
        seq_len=2048, batch_size=8, gradient_checkpointing=False,
    )
    assert weights == pytest.approx(2.04944304)
    assert per_gpu == pytest.approx(4.196926688)


def test_qlora_base_is_quarter_of_lora_base():
    lora, _, _ = estimate_gpu_memory_gb(1e9, 0, "LoRA", 1, 1, 0, 0, 0, 1, False)
    qlora, _, _ = estimate_gpu_memory_gb(1e9, 0, "QLoRA", 1, 1, 0, 0, 0, 1, False)
    assert qlora == pytest.approx(lora / 4)


def test_checkpointing_holds_one_layer_of_activations():
    _, without, _ = estimate_gpu_memory_gb(
        1e9, 1e9, None, 1, 1, 4096, 32, 2048, 8, gradient_checkpointing=False)
    _, with_ckpt, _ = estimate_gpu_memory_gb(
        1e9, 1e9, None, 1, 1, 4096, 32, 2048, 8, gradient_checkpointing=True)
    assert without == pytest.approx(with_ckpt * 32)


# -- LoRA parameter counts --------------------------------------------------

def test_lora_params_use_gqa_dimensions():
    """LLaMA 3 8B has 8 KV heads vs 32 query heads, so k_proj/v_proj are narrower."""
    llama = {m.slug: m for m in load_models()}["llama3_8b"]
    assert calculate_lora_trainable_params(
        "LoRA", llama.parameter_count, ["q_proj", "k_proj", "v_proj", "o_proj"],
        4096, 32, 16, llama.architecture,
    ) == 13_631_488


def test_lora_params_uniform_approximation_for_custom_models():
    assert calculate_lora_trainable_params(
        "LoRA", 7_000_000_000, ["q_proj", "v_proj"], 4096, 32, 16, {},
    ) == 8_388_608


def test_full_finetune_trains_every_parameter():
    assert calculate_lora_trainable_params("Full Fine-Tuning", 7e9, [], 0, 0, 0, {}) == 7e9


@pytest.mark.parametrize("missing", ["modules", "d_model", "layers", "rank"])
def test_lora_params_none_when_inputs_missing(missing):
    args = dict(target_modules=["q_proj"], d_model=4096, num_layers=32, lora_rank=16)
    args[{"modules": "target_modules", "d_model": "d_model",
          "layers": "num_layers", "rank": "lora_rank"}[missing]] = [] if missing == "modules" else 0
    assert calculate_lora_trainable_params("LoRA", 7e9, architecture={}, **args) is None


# -- Storage and project cost ----------------------------------------------

def test_storage_cost_golden():
    assert calculate_storage_cost(10.0, 3.0, "standard") == pytest.approx(690.0)


def test_checkpoint_storage_golden():
    assert calculate_checkpoint_storage_tb(7_000_000_000, 5, 1, 0, 0) == pytest.approx(0.49)


def test_hp_trials_and_ablations_cost_a_fraction_of_a_full_run():
    assert calculate_project_compute_cost(
        single_run_cost=1000.0, num_training_runs=2,
        num_hp_trials=4, hp_fraction=0.3,
        num_ablations=2, ablation_fraction=0.5,
    ) == pytest.approx(4200.0)


# -- Catalog-backed values --------------------------------------------------
# These read through the data layer being migrated. They must survive the move
# from bundled YAML to the AWS catalog unchanged.

def test_h100_instance_spec():
    spec = get_gpu_instance("p5.48xlarge")
    assert spec["gpu"] == "H100"
    assert spec["gpu_count"] == 8
    assert spec["memory_per_gpu"] == 80
    assert spec["peak_flops_fp16"] == pytest.approx(989e12)


def test_unknown_instance_raises_with_available_listed():
    with pytest.raises(ValueError, match="Unknown instance type"):
        get_gpu_instance("does-not-exist")


def test_unknown_storage_class_raises():
    with pytest.raises(ValueError, match="Unknown storage class"):
        get_storage_cost("does-not-exist")


def test_s3_standard_price():
    assert get_storage_cost("standard") == pytest.approx(23.0)


# -- Precision multipliers --------------------------------------------------

def test_bf16_is_the_fp16_baseline():
    spec = get_gpu_instance("p5.48xlarge")
    assert peak_flops_for_precision(spec, "bf16") == pytest.approx(989e12)


def test_fp32_uses_the_fp32_spec():
    spec = get_gpu_instance("p5.48xlarge")
    assert peak_flops_for_precision(spec, "fp32") == pytest.approx(67e12)


def test_tf32_is_half_fp16():
    spec = get_gpu_instance("p4d.24xlarge")
    assert peak_flops_for_precision(spec, "tf32") == pytest.approx(156e12)


def test_h100_fp8_doubles_fp16():
    spec = get_gpu_instance("p5.48xlarge")
    assert peak_flops_for_precision(spec, "fp8") == pytest.approx(1978e12)


def test_a100_rejects_fp8():
    """Ampere has no fp8 tensor cores. Previously reported 1x fp16, claiming a
    capability the die does not have."""
    spec = get_gpu_instance("p4d.24xlarge")
    with pytest.raises(UnsupportedPrecisionError, match="fp8"):
        peak_flops_for_precision(spec, "fp8")


@pytest.mark.parametrize("instance", ["p4d.24xlarge", "p5.48xlarge"])
def test_fp4_is_rejected_on_ampere_and_hopper(instance):
    """fp4 is Blackwell-only. Reporting 2x fp16 here understated cost by 2x."""
    spec = get_gpu_instance(instance)
    with pytest.raises(UnsupportedPrecisionError, match="fp4"):
        peak_flops_for_precision(spec, "fp4")


@pytest.mark.parametrize("precision", ["bf16", "tf32", "fp8", "fp4"])
def test_volta_rejects_post_volta_formats(precision):
    """V100 predates bf16, tf32, fp8 and fp4 alike."""
    spec = get_gpu_instance("p3.16xlarge")
    with pytest.raises(UnsupportedPrecisionError):
        peak_flops_for_precision(spec, precision)


def test_unknown_precision_does_not_silently_return_fp16():
    spec = get_gpu_instance("p5.48xlarge")
    with pytest.raises(UnsupportedPrecisionError):
        peak_flops_for_precision(spec, "not-a-real-precision")


def test_supported_precisions_reflects_the_die():
    assert supported_precisions(get_gpu_instance("p5.48xlarge")) == [
        "int8", "fp8", "bf16", "fp16", "tf32", "fp32"]
    assert supported_precisions(get_gpu_instance("p4d.24xlarge")) == [
        "int8", "bf16", "fp16", "tf32", "fp32"]
    assert supported_precisions(get_gpu_instance("p3.16xlarge")) == [
        "int8", "fp16", "fp32"]


def test_curated_flops_survive_the_join():
    """peak_flops moved to gpu_hardware.yaml but must still reach the spec dict."""
    assert get_gpu_instance("p4d.24xlarge")["peak_flops_fp16"] == pytest.approx(312e12)
    assert get_gpu_instance("p3dn.24xlarge")["typical_mfu"] == pytest.approx(0.40)
