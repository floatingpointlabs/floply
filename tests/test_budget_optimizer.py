"""Regression check for the extracted Budget Optimizer solve.

Values were captured from the Streamlit implementation before the extraction and verified
byte-identical across 13 rendered scenarios, so they pin the behaviour the TypeScript port
has to reproduce.

Run: python tests/test_budget_optimizer.py   (or under pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cost_modelling.budget_optimizer import (  # noqa: E402
    OptimizerInputs,
    Selection,
    auto_configure_hardware,
    derive_schedule_defaults,
    logspace,
    resolve_selection,
    solve_budget_optimum,
)
from src.cost_modelling.model_loader import load_models  # noqa: E402

REL_TOL = 1e-9

# name, kwargs, (n_opt, d_opt, total_project_cost, selected_tokens, selected_params, instances)
CASES = [
    (
        "pretrain_text_10k",
        dict(
            compute_budget=10_000.0, modality="Text (LLM)", training_type="Pre-Training",
            epochs=1, num_hp_trials=2, hp_fraction_pct=25,
            storage_duration_months=1, storage_class="standard",
        ),
        (3611193123.708093, 72223862474.16187, 9999.99999801808, 70794578438, 3684100128, 1),
    ),
    (
        "pretrain_diffusion_1m",
        dict(
            compute_budget=1_000_000.0, modality="Diffusion (Image Gen)",
            training_type="Pre-Training", epochs=1, num_hp_trials=10, hp_fraction_pct=10,
            storage_duration_months=6, storage_class="standard_ia",
        ),
        (28567566714.2911, 571351334285.822, 999999.9999893594, 562341325190, 29025285228, 5),
    ),
    (
        "lora_llama8b_50k",
        dict(
            compute_budget=50_000.0, modality="Text (LLM)", training_type="Fine-Tuning",
            ft_method="LoRA", ft_base_model="LLaMA 3 8B", lora_rank=16,
            epochs=5, num_hp_trials=5, hp_fraction_pct=15,
            storage_duration_months=3, storage_class="standard",
        ),
        (13631488.0, 681574400.0, 1222.6024195281632, 707945784, 13631488, 1),
    ),
    (
        # MoE: FLOPs must use 37B active params while checkpoints keep 671B total.
        "sft_moe_deepseek_10m",
        dict(
            compute_budget=10_000_000.0, modality="Text (LLM)", training_type="Fine-Tuning",
            ft_method="Full Fine-Tuning (SFT)", ft_base_model="DeepSeek V3",
            epochs=3, num_hp_trials=20, hp_fraction_pct=5,
            storage_duration_months=12, storage_class="standard_ia",
        ),
        (671000000000.0, 1757798837370.4834, 9999999.99999726, 1778279410038, 671000000000, 53),
    ),
]


def _close(got: float, want: float) -> bool:
    if want == 0:
        return got == 0
    return abs(got - want) / abs(want) <= REL_TOL


def test_solve_matches_streamlit_baseline() -> None:
    models = {m.name: m for m in load_models()}
    hardware = auto_configure_hardware()

    for name, kwargs, expected in CASES:
        kwargs = dict(kwargs)
        if isinstance(kwargs.get("ft_base_model"), str):
            kwargs["ft_base_model"] = models[kwargs["ft_base_model"]]

        inputs = OptimizerInputs(hardware=hardware, **kwargs)
        optimum = solve_budget_optimum(inputs)
        sel = resolve_selection(inputs, optimum)

        got = (
            optimum.n_opt, optimum.d_opt, optimum.total_project_cost,
            sel.selected_tokens, sel.selected_params, sel.recommended_instances,
        )
        for i, (g, w) in enumerate(zip(got, expected)):
            assert _close(g, w), f"{name}[{i}]: got {g!r}, want {w!r}"

        # The solve must never exceed the budget it was given.
        assert optimum.total_project_cost <= inputs.compute_budget * (1 + 1e-6), name


def test_moe_uses_active_params_for_flops() -> None:
    """DeepSeek V3 is 671B total / 37B active — compute must not be costed at the total."""
    models = {m.name: m for m in load_models()}
    deepseek = models["DeepSeek V3"]
    assert deepseek.effective_parameter_count == 37_000_000_000
    assert deepseek.parameter_count == 671_000_000_000

    inputs = OptimizerInputs(
        compute_budget=1_000_000.0, modality="Text (LLM)", training_type="Fine-Tuning",
        ft_method="Full Fine-Tuning (SFT)", ft_base_model=deepseek,
    )
    assert inputs.base_params_active == 37_000_000_000
    assert inputs.base_params_total == 671_000_000_000

    sel = resolve_selection(inputs, solve_budget_optimum(inputs))
    assert sel.flops_params == 37_000_000_000, "FLOPs must use active params"
    assert sel.selected_params == 671_000_000_000, "checkpoints must use total params"


def test_selection_overrides_beat_derived_defaults() -> None:
    """An explicit override wins; absence falls back to the derived default."""
    inputs = OptimizerInputs(
        compute_budget=100_000.0, modality="Text (LLM)", training_type="Pre-Training",
    )
    optimum = solve_budget_optimum(inputs)

    default = resolve_selection(inputs, optimum, Selection())
    overridden = resolve_selection(inputs, optimum, Selection(log_d=11.35))

    assert default.selected_tokens == int(10 ** default.d_opt_log_default)
    assert overridden.selected_tokens == int(10 ** 11.35)
    assert overridden.selected_tokens != default.selected_tokens

    assert resolve_selection(inputs, optimum, Selection(num_instances=37)).num_instances == 37
    assert default.num_instances == default.recommended_instances


def test_derive_schedule_defaults_brackets() -> None:
    base = dict(modality="Text (LLM)", training_type="Pre-Training", ft_method=None, quick_n=1e9)
    assert derive_schedule_defaults(compute_budget=1_000, **base)["hp_trials"] == 0
    assert derive_schedule_defaults(compute_budget=10_000, **base)["hp_trials"] == 2
    assert derive_schedule_defaults(compute_budget=100_000, **base)["hp_trials"] == 5
    assert derive_schedule_defaults(compute_budget=1_000_000, **base)["hp_trials"] == 10
    assert derive_schedule_defaults(compute_budget=10_000_000, **base)["hp_trials"] == 20

    # Storage-heavy modalities halve the retention window (integer floor division).
    heavy = derive_schedule_defaults(
        compute_budget=1_000_000, modality="Diffusion (Image Gen)",
        training_type="Pre-Training", ft_method=None, quick_n=1e9,
    )
    assert heavy["storage_months"] == 6
    assert heavy["storage_class"] == "standard_ia"


def test_lora_with_no_target_modules_does_not_crash() -> None:
    """Clearing every LoRA target module is one click in the UI.

    The pre-extraction code took the pre-training branch here, leaving d_eff/d_budget
    unbound, and then the LoRA display block read them — an UnboundLocalError.
    """
    models = {m.name: m for m in load_models()}
    inputs = OptimizerInputs(
        compute_budget=50_000.0, modality="Text (LLM)", training_type="Fine-Tuning",
        ft_method="LoRA", ft_base_model=models["LLaMA 3 8B"],
        lora_rank=16, target_modules=[],
    )
    optimum = solve_budget_optimum(inputs)
    assert optimum.n_adapter == 0
    assert optimum.d_eff == 0.0 and optimum.d_budget == 0.0
    resolve_selection(inputs, optimum)   # must not raise


def test_logspace_matches_numpy_semantics() -> None:
    got = logspace(1.0, 3.0, 3)
    assert [round(v, 9) for v in got] == [10.0, 100.0, 1000.0]
    assert len(logspace(6.0, 12.0, 300)) == 300


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
    print("\nall checks passed")
