"""Generate golden fixtures pinning the Python engine's behaviour.

These fixtures are the contract for the TypeScript port: the TS implementation must
reproduce every value in them. This generator is throwaway — it is deleted along with
the Python engine once the port lands, but `fixtures/*.json` stay.

Run:  python scripts/gen_fixtures.py
Writes: fixtures/engine.json, fixtures/budget_optimizer.json

Conventions
-----------
* Args are always named keywords, never positional, so a fixture diff is readable.
* Tuple returns are recorded as named objects — the callers currently do
  ``_, _, cost, days = ...`` and positional destructuring in TS would preserve exactly
  the fragility those throwaway underscores create.
* A case that raises records ``{"raises": "<ExceptionName>"}`` instead of ``expect``.
  The TS port may legitimately differ here (JS yields Infinity where Python raises);
  those are flagged so the divergence is a decision rather than a discovery.
* Budgets are capped at $1e12. Above ~2^53 tokens Python's arbitrary-precision ints and
  JS float64 genuinely diverge; one documented case sits above the line to make that
  visible rather than surprising.
"""

import json
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cost_modelling import budget_optimizer as bo  # noqa: E402
from src.cost_modelling import calculator as calc  # noqa: E402
from src.cost_modelling import dataset as ds  # noqa: E402
from src.cost_modelling import formatting as fmt  # noqa: E402
from src.cost_modelling import scaling_laws as sl  # noqa: E402
from src.cost_modelling.gpu_specs import (  # noqa: E402
    get_gpu_instance,
    list_available_instances,
    peak_flops_for_precision,
)
from src.cost_modelling.model_loader import load_models  # noqa: E402

ARCHES = ["transformer", "cnn", "rnn", "vit", "diffusion", "definitely_not_an_arch"]
PRECISIONS = ["fp4", "int8", "fp8", "bf16", "fp16", "tf32", "fp32"]
STORAGE_CLASSES = ["standard", "intelligent_tiering", "standard_ia", "one_zone_ia", "glacier"]
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


def record(fn, **kwargs):
    """Call fn(**kwargs) and record either its result or the exception type."""
    try:
        value = fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - recording behaviour, including failures
        return {"args": kwargs, "raises": type(exc).__name__}
    if is_dataclass(value):
        value = asdict(value)
    return {"args": kwargs, "expect": value}


def named(fn, fields, **kwargs):
    """Record a tuple-returning function with its fields named."""
    try:
        value = fn(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"args": kwargs, "raises": type(exc).__name__}
    return {"args": kwargs, "expect": dict(zip(fields, value))}


def build_engine() -> dict:
    out: dict = {}

    # ── _get_flops_multiplier — exhaustive (6 arches × checkpointing) ────────
    out["_get_flops_multiplier"] = [
        record(calc._get_flops_multiplier, architecture=a, gradient_checkpointing=g)
        for a in ARCHES for g in (False, True)
    ]

    # ── calculate_training_flops ─────────────────────────────────────────────
    out["calculate_training_flops"] = [
        record(
            calc.calculate_training_flops,
            parameter_count=n, training_tokens=d, architecture=a,
            epochs=e, gradient_checkpointing=g,
        )
        for n, d in [(7_000_000_000, 140_000_000_000), (1, 1), (0, 1_000), (1_000, 0)]
        for a in ("transformer", "diffusion")
        for e in (1.0, 3.0)
        for g in (False, True)
    ]

    # ── estimate_compute_cost (4-tuple) ──────────────────────────────────────
    fields = ("wall_clock_hours", "gpu_hours", "compute_cost", "wall_clock_days")
    out["estimate_compute_cost"] = [
        named(
            calc.estimate_compute_cost, fields,
            total_flops=f, peak_flops_per_gpu=9.89e14, mfu=m,
            total_gpus=gpus, num_instances=inst, hourly_cost=66.64,
        )
        for f in (5.88e21, 1.0)
        for m in (0.05, 0.55)
        for gpus, inst in [(8, 1), (1024, 128)]
    ]
    # Divergence markers: Python raises, JS would yield Infinity.
    out["estimate_compute_cost"] += [
        named(calc.estimate_compute_cost, fields, total_flops=1e21,
              peak_flops_per_gpu=9.89e14, mfu=0.0, total_gpus=8,
              num_instances=1, hourly_cost=66.64),
        named(calc.estimate_compute_cost, fields, total_flops=1e21,
              peak_flops_per_gpu=9.89e14, mfu=0.5, total_gpus=0,
              num_instances=1, hourly_cost=66.64),
    ]

    # ── solvers — all three zero-guards, truncation, epochs<1 ────────────────
    solver_common = dict(
        peak_flops_per_gpu=9.89e14, mfu=0.55, total_gpus=8,
        num_instances=1, hourly_cost=66.64, architecture="transformer",
        gradient_checkpointing=False,
    )
    out["solve_for_parameter_count"] = [
        record(calc.solve_for_parameter_count, compute_budget_usd=b,
               training_tokens=d, epochs=e, **solver_common)
        for b, d, e in [
            (10_000.0, 140_000_000_000, 1.0),
            (1_000_000.0, 1_000_000_000_000, 1.0),
            (1_000_000_000_000.0, 1_000_000_000_000, 1.0),   # $1e12 cap
            (0.01, 1_000_000_000_000, 1.0),                  # tiny budget → 0
            (10_000.0, 0, 1.0),                              # zero-guard: tokens
            (10_000.0, 140_000_000_000, 0.0),                # zero-guard: epochs→tokens
            (10_000.0, 140_000_000_000, 3.0),
            (10_000.0, 140_000_000_000, 0.5),
        ]
    ] + [
        record(calc.solve_for_parameter_count, compute_budget_usd=10_000.0,
               training_tokens=1_000_000_000, epochs=1.0,
               **{**solver_common, "hourly_cost": 0.0}),       # zero-guard: hourly_cost
        record(calc.solve_for_parameter_count, compute_budget_usd=10_000.0,
               training_tokens=1_000_000_000, epochs=1.0,
               **{**solver_common, "num_instances": 0}),       # zero-guard: instances
    ]
    out["solve_for_training_tokens"] = [
        record(calc.solve_for_training_tokens, compute_budget_usd=b,
               parameter_count=n, epochs=e, **solver_common)
        for b, n, e in [
            (10_000.0, 7_000_000_000, 1.0),
            (1_000_000.0, 70_000_000_000, 1.0),
            (1_000_000_000_000.0, 7_000_000_000, 1.0),
            (0.01, 7_000_000_000, 1.0),
            (10_000.0, 0, 1.0),                              # zero-guard: params
            (10_000.0, 7_000_000_000, 3.0),
            (10_000.0, 7_000_000_000, 0.5),                  # exercises max(epochs, 1.0)
            (10_000.0, 7_000_000_000, 0.0),
        ]
    ] + [
        record(calc.solve_for_training_tokens, compute_budget_usd=10_000.0,
               parameter_count=7_000_000_000, epochs=1.0,
               **{**solver_common, "hourly_cost": 0.0}),
        record(calc.solve_for_training_tokens, compute_budget_usd=10_000.0,
               parameter_count=7_000_000_000, epochs=1.0,
               **{**solver_common, "num_instances": 0}),
    ]

    # ── estimate_gpu_memory_gb (3-tuple) ─────────────────────────────────────
    mem_fields = ("weights_gb", "activation_gb", "memory_per_gpu_gb")
    out["estimate_gpu_memory_gb"] = [
        named(
            calc.estimate_gpu_memory_gb, mem_fields,
            effective_params=7_000_000_000, trainable_params=trainable,
            ft_method=method, total_gpus=8, rl_multiplier=rl,
            d_model=d_model, num_layers=32, seq_len=4096,
            batch_size=8, gradient_checkpointing=g,
        )
        for method, trainable in [
            ("QLoRA", 13_631_488), ("LoRA", 13_631_488),
            ("Full Fine-Tuning", 7_000_000_000), (None, 7_000_000_000),
        ]
        for d_model, g in [(4096, False), (4096, True), (0, False)]
        for rl in (1, 2, 4)
    ]

    # ── calculate_lora_trainable_params — every branch, incl. GQA ────────────
    llama70b = {"num_layers": 80, "d_model": 8192, "num_heads": 64,
                "num_kv_heads": 8, "ffn_intermediate": 28672}
    mha = {"num_layers": 32, "d_model": 4096, "num_heads": 32,
           "num_kv_heads": 32, "ffn_intermediate": 11008}
    no_heads = {"num_layers": 32, "d_model": 4096, "num_heads": 0}
    # 4096 // 30 == 136, but 4096 / 30 == 136.533… — the floor-division tripwire.
    uneven_heads = {"num_layers": 32, "d_model": 4096, "num_heads": 30,
                    "num_kv_heads": 6, "ffn_intermediate": 11008}
    lora_cases = [
        # Full FT short-circuits and returns base_params
        dict(ft_method="Full Fine-Tuning", base_params=7_000_000_000,
             target_modules=[], d_model=0, num_layers=0, lora_rank=0, architecture={}),
        # Each None-return path, one at a time
        dict(ft_method="LoRA", base_params=7e9, target_modules=[],
             d_model=4096, num_layers=32, lora_rank=16, architecture=mha),
        dict(ft_method="LoRA", base_params=7e9, target_modules=["q_proj"],
             d_model=0, num_layers=32, lora_rank=16, architecture={}),
        dict(ft_method="LoRA", base_params=7e9, target_modules=["q_proj"],
             d_model=4096, num_layers=0, lora_rank=16, architecture=mha),
        dict(ft_method="LoRA", base_params=7e9, target_modules=["q_proj"],
             d_model=4096, num_layers=32, lora_rank=0, architecture=mha),
        # GQA: num_kv_heads < num_heads, exercising the k/v_proj kv_dim path
        dict(ft_method="LoRA", base_params=70e9, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
             d_model=8192, num_layers=80, lora_rank=16, architecture=llama70b),
        # Synthetic: d_model % num_heads != 0, so `//` and `/` genuinely disagree.
        # Every real model divides evenly, so nothing realistic pins this — but a TS
        # port using `/` instead of Math.floor produces a different number here.
        dict(ft_method="LoRA", base_params=7e9, target_modules=["k_proj", "v_proj"],
             d_model=4096, num_layers=32, lora_rank=16, architecture=uneven_heads),
        # Uniform fallback for custom models (architecture={})
        dict(ft_method="LoRA", base_params=7e9, target_modules=["q_proj", "k_proj"],
             d_model=4096, num_layers=32, lora_rank=16, architecture={}),
        # num_heads=0 → _head_dim falls back to _d
        dict(ft_method="QLoRA", base_params=7e9, target_modules=["k_proj"],
             d_model=4096, num_layers=32, lora_rank=8, architecture=no_heads),
        # Unknown module name → (_d, _d)
        dict(ft_method="LoRA", base_params=7e9, target_modules=["mystery_proj"],
             d_model=4096, num_layers=32, lora_rank=16, architecture=mha),
        # All six modules; ffn_intermediate default (4 × d_model) when absent
        dict(ft_method="LoRA", base_params=7e9, target_modules=LORA_MODULES,
             d_model=4096, num_layers=32, lora_rank=64, architecture=mha),
        dict(ft_method="LoRA", base_params=7e9, target_modules=["up_proj", "down_proj"],
             d_model=4096, num_layers=32, lora_rank=16,
             architecture={"d_model": 4096, "num_heads": 32}),
    ]
    out["calculate_lora_trainable_params"] = [
        record(calc.calculate_lora_trainable_params, **c) for c in lora_cases
    ]

    # ── storage / project cost ───────────────────────────────────────────────
    out["calculate_checkpoint_storage_tb"] = [
        record(calc.calculate_checkpoint_storage_tb, checkpoint_params=p,
               num_checkpoints=c, num_training_runs=r, num_hp_trials=h, num_ablations=a)
        for p, c, r, h, a in [
            (7_000_000_000, 5, 1, 0, 0), (7_000_000_000, 5, 3, 10, 2),
            (13_631_488, 5, 1, 5, 0), (0, 5, 1, 0, 0),
        ]
    ]
    out["calculate_project_compute_cost"] = [
        record(calc.calculate_project_compute_cost, single_run_cost=s,
               num_training_runs=r, num_hp_trials=h, hp_fraction=hf,
               num_ablations=a, ablation_fraction=af)
        for s, r, h, hf, a, af in [
            (10_000.0, 1, 0, 0.1, 0, 0.5), (10_000.0, 3, 10, 0.1, 2, 0.5),
            (0.0, 1, 1, 0.1, 1, 0.5), (1.0, 1, 20, 0.05, 0, 0.5),
        ]
    ]
    out["calculate_storage_cost"] = [
        record(calc.calculate_storage_cost, dataset_size_tb=10.0,
               storage_duration_months=3.0, storage_class=sc)
        for sc in STORAGE_CLASSES + ["not_a_storage_class"]
    ] + [
        record(calc.calculate_storage_cost, dataset_size_tb=0.0,
               storage_duration_months=1.0, storage_class="standard"),
    ]

    # ── peak_flops_for_precision — exhaustive 5 instances × 7 precisions ─────
    out["peak_flops_for_precision"] = [
        {
            "args": {"instance_type": inst, "mixed_precision": p},
            "expect": peak_flops_for_precision(get_gpu_instance(inst), p),
        }
        for inst in list_available_instances() for p in PRECISIONS
    ] + [
        {
            "args": {"instance_type": "p5.48xlarge", "mixed_precision": "not_a_precision"},
            "expect": peak_flops_for_precision(get_gpu_instance("p5.48xlarge"), "not_a_precision"),
        }
    ]

    # ── dataset.py — the floor-division and int() truncation hazards ─────────
    out["tokens_per_text_sample"] = [
        record(ds.tokens_per_text_sample, avg_seq_len_words=w) for w in (0, 1, 100, 500, 999)
    ]
    out["bytes_per_text_sample"] = [
        record(ds.bytes_per_text_sample, tokens=t) for t in (0, 1, 650, 1_000_000)
    ]
    out["tokens_per_image_sample"] = [
        # 336 // 32 = 10 → 100, NOT 110.25
        record(ds.tokens_per_image_sample, resolution=r, patch_size=p)
        for r, p in [(224, 16), (336, 32), (224, 14), (512, 16), (255, 16), (1, 16)]
    ]
    out["bytes_per_image_sample"] = [
        record(ds.bytes_per_image_sample, resolution=r) for r in (224, 336, 512, 1, 3)
    ]
    out["tokens_per_audio_sample"] = [
        record(ds.tokens_per_audio_sample, clip_duration=d, base_rate=r, num_codebooks=c)
        for d, r, c in [(10.0, 75, 1), (10.5, 75, 4), (0.5, 50, 1), (3.33, 50, 8), (0.0, 75, 1)]
    ]
    out["bytes_per_audio_sample"] = [
        record(ds.bytes_per_audio_sample, clip_duration=d) for d in (10.0, 0.5, 3.33, 0.0)
    ]
    out["tokens_per_video_sample"] = [
        record(ds.tokens_per_video_sample, duration=d, fps=f, resolution=r, patch_size=p)
        for d, f, r, p in [(10.0, 30, 224, 16), (2.5, 24, 336, 32), (0.5, 30, 224, 16), (0.0, 30, 224, 16)]
    ]
    out["bytes_per_video_sample"] = [
        record(ds.bytes_per_video_sample, duration=d, fps=f, resolution=r)
        for d, f, r in [(10.0, 30, 224), (2.5, 24, 336), (0.04, 30, 512)]
    ]

    # ── formatters — boundaries at each 1e3 step ─────────────────────────────
    token_vals = [0, 1, 999, 1_000, 1_001, 999_999, 1_000_000, 1_500_000,
                  999_999_999, 1_000_000_000, 7_000_000_000,
                  999_999_999_999, 1_000_000_000_000, 671_000_000_000]
    out["fmt_tokens"] = [record(fmt.fmt_tokens, n=n) for n in token_vals]
    out["fmt_samples"] = [record(fmt.fmt_samples, n=n) for n in token_vals]
    out["format_wall_clock_time"] = [
        record(fmt.format_wall_clock_time, wall_clock_days=d)
        for d in (0.0, 0.0005, 1 / 24 - 1e-9, 1 / 24, 0.5, 0.999, 1.0, 5.176, 365.0)
    ]
    out["format_gpu_hours"] = [
        record(fmt.format_gpu_hours, gpu_hours=g)
        for g in (0.0, 1.0, 999.9, 1_000.0, 999_999.0, 1_000_000.0, 2_500_000.0)
    ]
    # JSON cannot distinguish Python's 1000.0 from 1000, but display_value formats them
    # differently ("1.0K" vs "1,000"). Record the source type so the TS port can too.
    out["display_value"] = [
        {**record(fmt.display_value, val=v), "arg_type": type(v).__name__}
        for v in [0.5, 3.7, 999.0, 1_000.0, 1.5e6, 2.5e9, 1e12, 5e12,
                  True, False, 0, 42, 999, 1_000, 1_500_000, "standard", "p5.48xlarge"]
    ]

    # ── scaling-law assessments — every threshold boundary ───────────────────
    ratios = [0.05, 0.09, 0.1, 0.5, 0.99, 1.0, 1.5, 5.0, 9.99, 10.0, 20.0,
              30.0, 30.1, 100.0, 200.0, 200.1, 999.0, 1000.0, 1000.1, 5000.0]
    n = 1_000_000_000
    out["assess_training_config"] = [
        record(sl.assess_training_config, total_tokens=int(r * n), ft_method=m,
               base_params=b, pre_params=p, adapter_params=a)
        for r in ratios
        for m, b, p, a in [
            (None, 0, n, 0),
            ("Full Fine-Tuning", n, 0, 0),
            ("LoRA", 8_000_000_000, 0, n),
            ("QLoRA", 8_000_000_000, 0, n),
        ]
    ] + [
        record(sl.assess_training_config, total_tokens=t, ft_method=m,
               base_params=b, pre_params=p, adapter_params=a)
        for t, m, b, p, a in [
            (0, None, 0, 1_000, 0), (-5, None, 0, 1_000, 0),
            (100, None, 0, 0, 0), (100, "Full Fine-Tuning", 0, 0, 0),
            (100, "LoRA", 0, 0, 0), (100, "LoRA", 1_000, 0, 0),
        ]
    ]
    out["assess_lora_ratio"] = [
        record(sl.assess_lora_ratio, ratio=r)
        for r in [0.5, 5.0, 9.99, 10.0, 25.0, 50.0, 99.9, 100.0, 100.1, 500.0, 5000.0]
    ]
    out["assess_chinchilla_ratio"] = [
        record(sl.assess_chinchilla_ratio, ratio=r,
               optimal_tokens=20_000_000_000, fix_hint=h)
        for r in [0.5, 0.99, 1.0, 5.0, 9.99, 10.0, 25.0, 30.0, 30.1, 150.0, 200.0, 200.1, 1e4]
        for h in ["", "Move the slider toward the star."]
    ]

    # ── training-budget pipeline — pins the wiring, not the maths ────────────
    # Verified against the running Streamlit Tab 3 (10/10 rendered metrics) before
    # these were generated.
    from src.cost_modelling.training_budget import (
        TrainingBudgetInputs,
        estimate_training_budget,
    )

    tb_cases = [
        # The scenario checked against the live app.
        dict(modality="Text", dataset_size=100_000_000, tokens_per_sample=520,
             bytes_per_sample=2080, storage_months=3, architecture="Transformer",
             parameter_count=7_000_000_000, d_model=4096, num_layers=32, seq_len=520,
             epochs=1, instance_type="p4d.24xlarge", num_instances=1,
             mixed_precision="bf16", mfu=0.30, batch_size=1024),
        # Full fine-tuning off a base model.
        dict(modality="Text", dataset_size=1_000_000, tokens_per_sample=650,
             bytes_per_sample=2600, storage_months=6, architecture="Transformer",
             base_params=7_000_000_000, base_flops_params=7_000_000_000,
             ft_method="Full Fine-Tuning", trainable_params=7_000_000_000,
             d_model=4096, num_layers=32, seq_len=650, epochs=3,
             instance_type="p5.48xlarge", num_instances=2, mixed_precision="fp8",
             mfu=0.45, batch_size=64, num_training_runs=2, num_hp_trials=4,
             storage_class="standard_ia"),
        # LoRA — checkpoints hold adapters only, not the base model.
        dict(modality="Text", dataset_size=500_000, tokens_per_sample=520,
             bytes_per_sample=2080, storage_months=1, architecture="Transformer",
             base_params=70_000_000_000, base_flops_params=70_000_000_000,
             ft_method="LoRA", trainable_params=13_631_488, d_model=8192,
             num_layers=80, seq_len=520, epochs=5, instance_type="p5.48xlarge",
             num_instances=4, mixed_precision="bf16", mfu=0.4, batch_size=16,
             num_hp_trials=8, num_ablations=2),
        # MoE base: FLOPs on active params, checkpoints and VRAM on the total.
        dict(modality="Text", dataset_size=2_000_000, tokens_per_sample=520,
             bytes_per_sample=2080, storage_months=12, architecture="Transformer",
             base_params=671_000_000_000, base_flops_params=37_000_000_000,
             ft_method="QLoRA", trainable_params=27_262_976, d_model=7168,
             num_layers=61, seq_len=520, epochs=2, instance_type="p5.48xlarge",
             num_instances=8, mixed_precision="bf16", mfu=0.5, batch_size=8,
             storage_class="glacier"),
        # Vision, RL memory multiplier, and a V100 cluster.
        dict(modality="Image", dataset_size=10_000_000, tokens_per_sample=196,
             bytes_per_sample=15052, storage_months=3, architecture="ViT",
             parameter_count=1_000_000_000, d_model=1024, num_layers=24, seq_len=196,
             rl_multiplier=4, epochs=1, instance_type="p3.16xlarge", num_instances=16,
             mixed_precision="fp16", mfu=0.25, batch_size=256,
             num_training_runs=3, num_ablations=5, ablation_fraction=0.25),
        # Degenerate: no data at all.
        dict(modality="Text", dataset_size=0, tokens_per_sample=0, bytes_per_sample=0,
             parameter_count=1_000_000_000),
    ]
    out["training_budget"] = [
        {"args": c, "expect": asdict(estimate_training_budget(TrainingBudgetInputs(**c)))}
        for c in tb_cases
    ]

    # ── scaling-law tier tables — pure data the UI renders as the tier chart ──
    from src.app.config import FULL_FT_TIERS, LORA_TIERS, PRE_TRAINING_TIERS
    # `color` is deliberately excluded: the thresholds and their semantic mapping are
    # engine behaviour and must not drift, but the hex is presentation and has to be free
    # to change with branding. Pinning it here would make a re-theme fail the maths tests.
    out["tiers"] = [
        {
            "args": {"table": name},
            "expect": [{k: v for k, v in tier.items() if k != "color"} for tier in table],
        }
        for name, table in [
            ("PRE_TRAINING_TIERS", PRE_TRAINING_TIERS),
            ("FULL_FT_TIERS", FULL_FT_TIERS),
            ("LORA_TIERS", LORA_TIERS),
        ]
    ]

    # ── model definitions, incl. the MoE active/total split ──────────────────
    out["model_definitions"] = [
        {
            "args": {"slug": m.slug},
            "expect": {
                "name": m.name, "family": m.family,
                "parameter_count": m.parameter_count,
                "effective_parameter_count": m.effective_parameter_count,
                "display_name": m.display_name,
            },
        }
        for m in load_models()
    ]
    return out


def build_budget_optimizer() -> dict:
    """Scenario-level fixtures for the two-stage solve."""
    models = {m.name: m for m in load_models()}
    hw = bo.auto_configure_hardware()
    cases = []

    setups = [
        ("pretrain_text_10k", dict(compute_budget=10_000.0, modality="Text (LLM)",
                                   training_type="Pre-Training")),
        ("pretrain_text_1e12_cap", dict(compute_budget=1e12, modality="Text (LLM)",
                                        training_type="Pre-Training")),
        ("pretrain_vision_500k", dict(compute_budget=500_000.0, modality="Vision (ViT / CLIP)",
                                      training_type="Pre-Training", num_hp_trials=10)),
        ("pretrain_diffusion_1m", dict(compute_budget=1_000_000.0,
                                       modality="Diffusion (Image Gen)",
                                       training_type="Pre-Training")),
        ("pretrain_audio_250k", dict(compute_budget=250_000.0, modality="Audio",
                                     training_type="Pre-Training", epochs=4,
                                     num_hp_trials=7, hp_fraction_pct=20,
                                     storage_duration_months=9, storage_class="glacier")),
        ("pretrain_multimodal_2m", dict(compute_budget=2_000_000.0,
                                        modality="Multimodal (VLM)",
                                        training_type="Pre-Training")),
        ("lora_llama8b_5k", dict(compute_budget=5_000.0, modality="Text (LLM)",
                                 training_type="Fine-Tuning", ft_method="LoRA",
                                 ft_base_model="LLaMA 3 8B", lora_rank=16, epochs=5)),
        ("qlora_llama70b_50k", dict(compute_budget=50_000.0, modality="Text (LLM)",
                                    training_type="Fine-Tuning", ft_method="QLoRA",
                                    ft_base_model="LLaMA 3 70B", lora_rank=128, epochs=5)),
        ("sft_mistral7b_100k", dict(compute_budget=100_000.0, modality="Text (LLM)",
                                    training_type="Fine-Tuning",
                                    ft_method="Full Fine-Tuning (SFT)",
                                    ft_base_model="Mistral 7B", epochs=3)),
        ("lora_moe_deepseek_1m", dict(compute_budget=1_000_000.0, modality="Text (LLM)",
                                      training_type="Fine-Tuning", ft_method="LoRA",
                                      ft_base_model="DeepSeek V3", lora_rank=32, epochs=5)),
        ("sft_moe_deepseek_10m", dict(compute_budget=10_000_000.0, modality="Text (LLM)",
                                      training_type="Fine-Tuning",
                                      ft_method="Full Fine-Tuning (SFT)",
                                      ft_base_model="DeepSeek V3", epochs=3)),
        ("lora_no_target_modules", dict(compute_budget=50_000.0, modality="Text (LLM)",
                                        training_type="Fine-Tuning", ft_method="LoRA",
                                        ft_base_model="LLaMA 3 8B", lora_rank=16,
                                        target_modules=[])),
        ("lora_gqa_qwen72b", dict(compute_budget=2_000_000.0, modality="Multimodal (VLM)",
                                  training_type="Fine-Tuning", ft_method="QLoRA",
                                  ft_base_model="Qwen 2.5 72B", lora_rank=8,
                                  target_modules=["q_proj", "up_proj", "down_proj"])),
        ("ft_custom_no_base", dict(compute_budget=50_000.0, modality="Text (LLM)",
                                   training_type="Fine-Tuning", ft_method="LoRA")),
        ("tiny_budget", dict(compute_budget=1.0, modality="Text (LLM)",
                             training_type="Pre-Training")),
    ]

    selections = [
        ("default", {}),
        ("dataset_slider", {"log_d": 11.35}),
        ("model_slider", {"explore_dir": bo.EXPLORE_MODEL_TO_TOKENS, "log_n": 9.75}),
        ("rank_slider", {"explore_dir": bo.EXPLORE_LORA_RANK, "rank": 64}),
        ("instances_override", {"num_instances": 37}),
    ]

    for setup_name, setup in setups:
        setup = dict(setup)
        if isinstance(setup.get("ft_base_model"), str):
            setup["ft_base_model"] = models[setup["ft_base_model"]]

        inputs = bo.OptimizerInputs(hardware=hw, **setup)
        optimum = bo.solve_budget_optimum(inputs)

        for sel_name, sel_kwargs in selections:
            sel = bo.resolve_selection(inputs, optimum, bo.Selection(**sel_kwargs))
            args = {k: (v.slug if hasattr(v, "slug") else v) for k, v in setup.items()}
            cases.append({
                "name": f"{setup_name}__{sel_name}",
                "args": args,
                "selection": sel_kwargs,
                "expect": {"optimum": asdict(optimum), "selection": asdict(sel)},
            })

    schedule = [
        record(bo.derive_schedule_defaults, compute_budget=b, modality=m,
               training_type=t, ft_method=f, quick_n=q)
        for b in (1_000.0, 5_000.0, 10_000.0, 50_000.0, 100_000.0,
                  500_000.0, 1_000_000.0, 5_000_000.0, 10_000_000.0)
        for m in ("Text (LLM)", "Diffusion (Image Gen)")
        for t, f in [("Pre-Training", None), ("Fine-Tuning", "LoRA"),
                     ("Fine-Tuning", "Full Fine-Tuning (SFT)")]
        for q in (1e8, 6e8, 5e9, 50e9)
    ]

    return {
        "solve": cases,
        "derive_schedule_defaults": schedule,
        "logspace": [
            record(bo.logspace, start=s, stop=e, num=n)
            for s, e, n in [(1.0, 3.0, 3), (6.0, 12.0, 5), (9.0, 13.0, 2), (1.0, 2.0, 1), (1.0, 2.0, 0)]
        ],
        "budget_curve_n": [
            {
                "args": {"d_tokens": [1e9, 1e11, 1e13], "budget": 1e5,
                         "peak_flops_per_gpu": 9.89e14, "mfu": 0.55,
                         "gpus_per_instance": 8, "hourly_cost": 66.64,
                         "multiplier": 6.0, "epochs": 1},
                "expect": bo.budget_curve_n(
                    d_tokens=[1e9, 1e11, 1e13], budget=1e5,
                    peak_flops_per_gpu=9.89e14, mfu=0.55, gpus_per_instance=8,
                    hourly_cost=66.64, multiplier=6.0, epochs=1,
                ),
            }
        ],
    }


MAX_SAFE_INT = 2 ** 53


def _has_unsafe_int(value) -> bool:
    """True if any integer in the payload exceeds JS's exact-integer range."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) > MAX_SAFE_INT
    if isinstance(value, dict):
        return any(_has_unsafe_int(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_unsafe_int(v) for v in value)
    return False


def flag_float64_limits(payload: dict) -> int:
    """Mark cases whose expectation exceeds 2^53.

    Python ints are arbitrary-precision; JS numbers are float64 and lose exactness above
    2^53. These cases are only reachable at absurd budgets, but the TS suite must compare
    them with a relative tolerance rather than exact equality.
    """
    flagged = 0
    for group in payload.values():
        if not isinstance(group, list):
            continue
        for case in group:
            if isinstance(case, dict) and _has_unsafe_int(case.get("expect")):
                case["exceeds_float64_safe_int"] = True
                flagged += 1
    return flagged


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    meta = {
        "generated_from": git_sha(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "note": "Golden fixtures for the TypeScript port. Regenerate with scripts/gen_fixtures.py.",
    }
    out_dir = ROOT / "fixtures"
    out_dir.mkdir(exist_ok=True)

    for filename, payload in [
        ("engine.json", build_engine()),
        ("budget_optimizer.json", build_budget_optimizer()),
    ]:
        flagged = flag_float64_limits(payload)
        data = {"_meta": meta, **payload}
        path = out_dir / filename
        path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
        count = sum(len(v) for k, v in payload.items() if isinstance(v, list))
        print(
            f"  {path.relative_to(ROOT)}: {len(payload)} groups, {count} cases"
            f" ({flagged} flagged above 2^53)"
        )


if __name__ == "__main__":
    main()
