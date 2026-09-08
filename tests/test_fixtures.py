"""Replay the golden fixtures against the Python engine.

This is the Python twin of the test the TypeScript port will run. If it fails, either the
engine changed behaviour (regenerate deliberately with scripts/gen_fixtures.py and review
the diff) or the fixtures were hand-edited.

Run: python tests/test_fixtures.py   (or under pytest)
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cost_modelling import budget_optimizer as bo  # noqa: E402
from src.cost_modelling import calculator as calc  # noqa: E402
from src.cost_modelling import dataset as ds  # noqa: E402
from src.cost_modelling import formatting as fmt  # noqa: E402
from src.cost_modelling import scaling_laws as sl  # noqa: E402
from src.cost_modelling.gpu_specs import get_gpu_instance, peak_flops_for_precision  # noqa: E402
from src.cost_modelling.model_loader import load_models  # noqa: E402

FIXTURES = ROOT / "fixtures"
REL_TOL = 1e-12
# Cases flagged as exceeding 2^53 are compared loosely: Python keeps exact integers there,
# JS float64 cannot, so demanding exactness would fail the port on unreachable inputs.
UNSAFE_TOL = 1e-9

TUPLE_FIELDS = {
    "estimate_compute_cost": ("wall_clock_hours", "gpu_hours", "compute_cost", "wall_clock_days"),
    "estimate_gpu_memory_gb": ("weights_gb", "activation_gb", "memory_per_gpu_gb"),
}

DISPATCH = {
    "_get_flops_multiplier": calc._get_flops_multiplier,
    "calculate_training_flops": calc.calculate_training_flops,
    "estimate_compute_cost": calc.estimate_compute_cost,
    "solve_for_parameter_count": calc.solve_for_parameter_count,
    "solve_for_training_tokens": calc.solve_for_training_tokens,
    "estimate_gpu_memory_gb": calc.estimate_gpu_memory_gb,
    "calculate_lora_trainable_params": calc.calculate_lora_trainable_params,
    "calculate_checkpoint_storage_tb": calc.calculate_checkpoint_storage_tb,
    "calculate_project_compute_cost": calc.calculate_project_compute_cost,
    "calculate_storage_cost": calc.calculate_storage_cost,
    "tokens_per_text_sample": ds.tokens_per_text_sample,
    "bytes_per_text_sample": ds.bytes_per_text_sample,
    "tokens_per_image_sample": ds.tokens_per_image_sample,
    "bytes_per_image_sample": ds.bytes_per_image_sample,
    "tokens_per_audio_sample": ds.tokens_per_audio_sample,
    "bytes_per_audio_sample": ds.bytes_per_audio_sample,
    "tokens_per_video_sample": ds.tokens_per_video_sample,
    "bytes_per_video_sample": ds.bytes_per_video_sample,
    "fmt_tokens": fmt.fmt_tokens,
    "fmt_samples": fmt.fmt_samples,
    "format_wall_clock_time": fmt.format_wall_clock_time,
    "format_gpu_hours": fmt.format_gpu_hours,
    "display_value": fmt.display_value,
    "assess_training_config": sl.assess_training_config,
    "assess_lora_ratio": sl.assess_lora_ratio,
    "assess_chinchilla_ratio": sl.assess_chinchilla_ratio,
    "derive_schedule_defaults": bo.derive_schedule_defaults,
    "logspace": bo.logspace,
    "budget_curve_n": bo.budget_curve_n,
}


def approx(got, want, tol) -> bool:
    """Structural comparison with a relative tolerance on floats."""
    if isinstance(want, bool) or isinstance(got, bool):
        return got == want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        if want == 0:
            return abs(got) <= tol
        return abs(got - want) / abs(want) <= tol
    if isinstance(want, dict) and isinstance(got, dict):
        return want.keys() == got.keys() and all(approx(got[k], want[k], tol) for k in want)
    if isinstance(want, list) and isinstance(got, list):
        return len(want) == len(got) and all(approx(g, w, tol) for g, w in zip(got, want))
    return got == want


def call(name, args):
    """Invoke the engine for a fixture group, translating indirect args."""
    if name == "peak_flops_for_precision":
        return peak_flops_for_precision(
            get_gpu_instance(args["instance_type"]), args["mixed_precision"]
        )
    result = DISPATCH[name](**args)
    if name in TUPLE_FIELDS:
        return dict(zip(TUPLE_FIELDS[name], result))
    if hasattr(result, "level"):   # Assessment
        return {"level": result.level, "message": result.message}
    return result


def test_engine_fixtures() -> None:
    data = json.loads((FIXTURES / "engine.json").read_text())
    models = {m.slug: m for m in load_models()}
    checked = 0

    for group, cases in data.items():
        if group == "_meta":
            continue
        for i, case in enumerate(cases):
            where = f"{group}[{i}]"
            args = case["args"]

            if group == "training_budget":
                from dataclasses import asdict as _asdict

                from src.cost_modelling.training_budget import (
                    TrainingBudgetInputs,
                    estimate_training_budget,
                )
                got = _asdict(estimate_training_budget(TrainingBudgetInputs(**args)))
                assert approx(got, case["expect"], REL_TOL), where
                checked += 1
                continue

            if group == "tiers":
                from src.app import config as app_config
                # Colour is presentation, not behaviour — see gen_fixtures.py.
                got = [
                    {k: v for k, v in tier.items() if k != "color"}
                    for tier in getattr(app_config, args["table"])
                ]
                assert got == case["expect"], where
                checked += 1
                continue

            if group == "model_definitions":
                m = models[args["slug"]]
                got = {
                    "name": m.name, "family": m.family,
                    "parameter_count": m.parameter_count,
                    "effective_parameter_count": m.effective_parameter_count,
                    "display_name": m.display_name,
                }
                assert got == case["expect"], where
                checked += 1
                continue

            if "raises" in case:
                try:
                    call(group, args)
                except Exception as exc:  # noqa: BLE001
                    assert type(exc).__name__ == case["raises"], (
                        f"{where}: raised {type(exc).__name__}, fixture says {case['raises']}"
                    )
                    checked += 1
                    continue
                raise AssertionError(f"{where}: expected {case['raises']}, nothing raised")

            tol = UNSAFE_TOL if case.get("exceeds_float64_safe_int") else REL_TOL
            got = call(group, args)
            assert approx(got, case["expect"], tol), (
                f"{where}: got {got!r}, want {case['expect']!r}"
            )
            checked += 1

    assert checked == sum(len(v) for k, v in data.items() if k != "_meta")
    print(f"    engine.json: {checked} cases replayed")


def test_budget_optimizer_fixtures() -> None:
    data = json.loads((FIXTURES / "budget_optimizer.json").read_text())
    models = {m.slug: m for m in load_models()}
    hw = bo.auto_configure_hardware()
    checked = 0

    for case in data["solve"]:
        args = dict(case["args"])
        if isinstance(args.get("ft_base_model"), str):
            args["ft_base_model"] = models[args["ft_base_model"]]

        inputs = bo.OptimizerInputs(hardware=hw, **args)
        optimum = bo.solve_budget_optimum(inputs)
        sel = bo.resolve_selection(inputs, optimum, bo.Selection(**case["selection"]))

        tol = UNSAFE_TOL if case.get("exceeds_float64_safe_int") else REL_TOL
        got = {"optimum": asdict(optimum), "selection": asdict(sel)}
        assert approx(got, case["expect"], tol), case["name"]
        checked += 1

    for group in ("derive_schedule_defaults", "logspace", "budget_curve_n"):
        for i, case in enumerate(data[group]):
            if "raises" in case:
                continue
            got = call(group, case["args"])
            assert approx(got, case["expect"], REL_TOL), f"{group}[{i}]"
            checked += 1

    print(f"    budget_optimizer.json: {checked} cases replayed")


def test_fixtures_cover_the_known_hazards() -> None:
    """The fixtures exist to catch specific porting mistakes. Assert those cases are present."""
    engine = json.loads((FIXTURES / "engine.json").read_text())

    # Floor division on image patches: (336 // 32) ** 2 == 100, not 110.25
    img = {(c["args"]["resolution"], c["args"]["patch_size"]): c["expect"]
           for c in engine["tokens_per_image_sample"]}
    assert img[(336, 32)] == 100
    assert img[(255, 16)] == 225

    # d_model % num_heads != 0, so `//` and `/` disagree — no real model pins this.
    uneven = [c for c in engine["calculate_lora_trainable_params"]
              if c["args"]["architecture"].get("num_heads") == 30]
    assert uneven, "missing the floor-division tripwire case"
    assert uneven[0]["expect"] == 5_029_888

    # MoE active-vs-total split
    deepseek = [c for c in engine["model_definitions"] if c["args"]["slug"] == "deepseek_v3"][0]
    assert deepseek["expect"]["parameter_count"] == 671_000_000_000
    assert deepseek["expect"]["effective_parameter_count"] == 37_000_000_000

    # Divergence markers: Python raises where JS would produce Infinity.
    assert any(c.get("raises") == "ZeroDivisionError" for c in engine["estimate_compute_cost"])
    assert any(c.get("raises") == "ValueError" for c in engine["calculate_storage_cost"])

    # Every None-return path of the LoRA calculator.
    assert sum(1 for c in engine["calculate_lora_trainable_params"]
               if c.get("expect") is None) == 4

    # Exhaustive precision lookup: 5 instances x 7 precisions, plus an unknown.
    assert len(engine["peak_flops_for_precision"]) == 36

    # FP4 is Blackwell-only, so no GPU in data/providers accelerates it. Granting a 2x
    # speedup here halves the estimated cost — pin fp4 to the fp16 rate on every
    # instance so the old behaviour cannot creep back in.
    by_key = {
        (c["args"]["instance_type"], c["args"]["mixed_precision"]): c["expect"]
        for c in engine["peak_flops_for_precision"]
        if "expect" in c
    }
    instances = {inst for inst, _ in by_key}
    for inst in instances:
        assert by_key[(inst, "fp4")] == by_key[(inst, "fp16")], (
            f"{inst}: fp4 must fall back to the fp16 rate"
        )

    # FP8 is Hopper-only and must still be twice the fp16 rate there, and only there.
    assert by_key[("p5.48xlarge", "fp8")] == 2 * by_key[("p5.48xlarge", "fp16")]
    assert by_key[("p4d.24xlarge", "fp8")] == by_key[("p4d.24xlarge", "fp16")]


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
    print("\nall fixture checks passed")
