"""Interactive budget optimizer: find the optimal model + dataset size within a total budget."""

import math

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.app.config import CHART_COLORS, CHINCHILLA_OPTIMAL_RATIO, LORA_OPTIMAL_RATIO
from src.app.helpers import (
    _cached_models,
    _fmt_tokens,
    _format_wall_clock_time,
    _UNIT_MULTIPLIERS,
    _render_chinchilla_assessment,
)
from src.cost_modelling.calculator import (
    solve_for_parameter_count,
    solve_for_training_tokens,
    calculate_training_flops,
    estimate_compute_cost,
    calculate_storage_cost,
    calculate_checkpoint_storage_tb,
    calculate_lora_trainable_params,
    BYTES_PER_PARAM_CHECKPOINT,
)
from src.app.components.region_selector import current_region
from src.cost_modelling.gpu_specs import (
    PricingUnavailableError,
    get_gpu_instance,
    get_provenance,
    get_storage_cost,
    list_available_instances,
    list_storage_classes,
    peak_flops_for_precision,
    supported_precisions,
)


def _catalog_stamp(region: str) -> str:
    """Cache-busting token for @st.cache_data functions that read prices.

    Changes whenever the underlying pricing does, so cached derivations are
    recomputed after a refresh instead of outliving the data they came from.
    """
    provenance = get_provenance(region)
    return provenance.fetched_at.isoformat() if provenance.fetched_at else "unavailable"


_ARCH_MULTIPLIERS = {
    "Transformer": 6.0,
    "CNN": 4.0,
    "RNN": 8.0,
    "ViT": 6.0,
    "Diffusion": 6.5,
}

# bytes_per_token: average object-storage bytes per training token/sample
MODALITY_DEFAULTS = {
    "Text (LLM)": {
        "arch": "Transformer",
        "bytes_per_token": 4,
        "tokens_per_sample": 512,    # ~512 tokens per web document / article
        "sample_noun": "documents",
    },
    "Vision (ViT / CLIP)": {
        "arch": "ViT",
        "bytes_per_token": 1536,
        "tokens_per_sample": 196,    # 14×14 patches for 224×224 at 16×16 patch size
        "sample_noun": "images",
    },
    "Audio": {
        "arch": "Transformer",
        "bytes_per_token": 8,
        "tokens_per_sample": 750,    # 10-second clip at 75 tok/s (EnCodec / Whisper rate)
        "sample_noun": "clips",
    },
    "Multimodal (VLM)": {
        "arch": "Transformer",
        "bytes_per_token": 512,
        "tokens_per_sample": 512,    # mixed text + image patches per caption/pair
        "sample_noun": "samples",
    },
    "Diffusion (Image Gen)": {
        "arch": "Diffusion",
        "bytes_per_token": 2048,
        "tokens_per_sample": 1024,   # 32×32 latent patches for a 512×512 image (8× VAE)
        "sample_noun": "images",
    },
}

# Checkpoints saved per full training run (used for storage estimation)
_CHECKPOINTS_PER_RUN = 5

# Target wall-clock days for auto-scaling num_instances
_TARGET_WALL_CLOCK_DAYS = 60

# Hard cap on recommended instances — represents a realistic large-scale cluster
# (~8,192 H100 GPUs).  Wall-clock is reported honestly for whatever this delivers.
_MAX_INSTANCES = 1_024


@st.cache_data
def _auto_configure_hardware(region: str, catalog_stamp: str) -> dict:
    """Pick the most cost-efficient available instance and return its config.

    Cost efficiency = effective_TFLOPS_per_$ = peak_flops × gpus × typical_mfu / hourly_cost.
    H100 (p5.48xlarge) typically wins: ~65 TFLOPS/$ vs A100 ~42 TFLOPS/$.

    Args:
        region: AWS region. Prices differ per region, so the winner can differ too.
        catalog_stamp: Provenance timestamp. Both arguments exist purely to key
            the cache — without them this returned the first region's answer
            forever, even after a price refresh.
    """
    best_instance = None
    best_efficiency = -1.0

    for instance_type in list_available_instances(region):
        spec = get_gpu_instance(instance_type, region)
        flops_per_dollar = (
            spec["peak_flops_fp16"] * spec["gpu_count"] * spec["typical_mfu"]
            / spec["hourly_cost"]
        )
        if flops_per_dollar > best_efficiency:
            best_efficiency = flops_per_dollar
            best_instance = instance_type

    if best_instance is None:
        raise PricingUnavailableError(region, "No priced instances in this region.")

    spec = get_gpu_instance(best_instance, region)
    # bf16 is the modern training default, but Volta-era GPUs predate it.
    precisions = supported_precisions(spec)
    precision = "bf16" if "bf16" in precisions else "fp16"
    peak_flops = peak_flops_for_precision(spec, precision)
    return {
        "instance_type": best_instance,
        "mixed_precision": precision,
        "instance_spec": spec,
        "peak_flops_per_gpu": peak_flops,
        "mfu": spec["typical_mfu"],
        "gpus_per_instance": spec["gpu_count"],
        "hourly_cost": spec["hourly_cost"],
        "gradient_checkpointing": False,
    }


def _quick_n_estimate(compute_budget: float, arch_multiplier: float, hw: dict) -> float:
    """Estimate model parameter count from budget alone (1-epoch Chinchilla, auto hardware).

    Used only for deriving sensible Training Schedule defaults — no storage correction needed.
    Approximate outputs at H100 pricing:
      $1K → ~140M params | $10K → ~450M | $100K → ~1.4B | $1M → ~14B | $10M → ~45B
    """
    quick_cost_per_tp = (
        arch_multiplier
        / (hw["peak_flops_per_gpu"] * hw["mfu"] * hw["gpus_per_instance"] * 3600)
    ) * hw["hourly_cost"]
    if quick_cost_per_tp <= 0:
        return 0.0
    return math.sqrt(compute_budget / (CHINCHILLA_OPTIMAL_RATIO * quick_cost_per_tp))


def _derive_schedule_defaults(
    compute_budget: float,
    modality: str,
    training_type: str,
    ft_method: str | None,
    quick_n: float,
) -> dict:
    """Derive recommended Training Schedule & Storage defaults from project-setup inputs.

    Returns a dict with keys: epochs, hp_trials, hp_fraction_pct, storage_months, storage_class.
    """
    is_lora = ft_method in ("LoRA", "QLoRA")

    # Epochs — driven by training type
    # Pre-training is 1-epoch optimal under Chinchilla; FT benefits from more passes.
    if training_type == "Pre-Training":
        epochs = 1
    elif is_lora:
        epochs = 5
    else:
        epochs = 3

    # HP Tuning Trials — scaled with budget (more budget = can afford more HP search)
    if compute_budget < 5_000:
        hp_trials = 0
    elif compute_budget < 50_000:
        hp_trials = 2
    elif compute_budget < 500_000:
        hp_trials = 5
    elif compute_budget < 5_000_000:
        hp_trials = 10
    else:
        hp_trials = 20

    # HP Trial Cost % — larger models need shorter (cheaper) HP trials
    if quick_n >= 30e9:
        hp_fraction_pct = 5
    elif quick_n >= 3e9:
        hp_fraction_pct = 10
    elif quick_n >= 500e6:
        hp_fraction_pct = 15
    else:
        hp_fraction_pct = 25

    # Storage Duration — bigger budget = more valuable artifacts = keep longer
    # Vision/Diffusion have expensive storage so halve the duration
    bytes_per_token = MODALITY_DEFAULTS[modality]["bytes_per_token"]
    storage_heavy = bytes_per_token > 100  # Vision, Multimodal, Diffusion

    if compute_budget < 10_000:
        base_months = 1
    elif compute_budget < 100_000:
        base_months = 3
    elif compute_budget < 1_000_000:
        base_months = 6
    else:
        base_months = 12

    storage_months = max(1, base_months // 2 if storage_heavy else base_months)

    # Storage Class — longer retention → cheaper tier
    storage_class = "standard_ia" if storage_months > 3 else "standard"

    return {
        "epochs": epochs,
        "hp_trials": hp_trials,
        "hp_fraction_pct": hp_fraction_pct,
        "storage_months": storage_months,
        "storage_class": storage_class,
    }


def render_budget_optimizer_page():
    st.markdown(
        "<p style='text-align:center;color:#888;'>Find the optimal model dimensions within your budget!</p>",
        unsafe_allow_html=True,
    )

    # ── 1. Combined project-setup expander ──────────────────────────────────
    ft_method: str | None = None
    lora_rank: int = 16
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

    with st.expander("Project Setup", expanded=True):
        col_budget, col_modality, col_task = st.columns(3)

        with col_budget:
            bv_col, bu_col = st.columns([3, 1])
            with bv_col:
                budget_value = st.number_input(
                    "Total Budget (USD)",
                    min_value=0.001,
                    max_value=9_999.0,
                    value=None,
                    step=0.1,
                    format="%.3f",
                    help="Total spend cap: GPU compute + HP tuning runs + storage.",
                    key="ms_budget_val",
                    placeholder="e.g. 10.0",
                )
            with bu_col:
                budget_unit = st.selectbox(
                    "Scale",
                    options=["K", "M", "B", "T"],
                    index=0,
                    key="ms_budget_unit",
                )
            if budget_value is not None:
                _raw = budget_value * _UNIT_MULTIPLIERS[budget_unit]
                _fmt = (
                    f"${_raw / 1e9:.2f}B" if _raw >= 1e9
                    else f"${_raw / 1e6:.2f}M" if _raw >= 1e6
                    else f"${_raw / 1e3:.2f}K" if _raw >= 1e3
                    else f"${_raw:,.2f}"
                )
                st.caption(f"Total: **{_fmt}**")

        with col_modality:
            modality_keys = list(MODALITY_DEFAULTS.keys())
            modality = st.selectbox(
                "Modality",
                options=modality_keys,
                index=0,
                key="ms_modality",
                help="Determines default architecture and storage bytes per training token.",
            )
            modality_cfg = MODALITY_DEFAULTS[modality]

        with col_task:
            training_type = st.selectbox(
                "Training Type",
                options=["Pre-Training", "Fine-Tuning"],
                index=0,
                help="Pre-Training trains from scratch; Fine-Tuning adapts an existing model.",
                key="ms_training_type",
            )

        if training_type == "Fine-Tuning":
            st.divider()
            ft_method_col, ft_model_col = st.columns(2)

            with ft_method_col:
                ft_method = st.selectbox(
                    "Fine-Tuning Method",
                    options=["Full Fine-Tuning (SFT)", "LoRA", "QLoRA"],
                    index=0,
                    key="ms_ft_method",
                )

            with ft_model_col:
                _known_models = _cached_models()
                _model_names  = [m.name for m in _known_models] + ["Custom"]
                ft_base_model_name = st.selectbox(
                    "Base Model",
                    options=_model_names,
                    index=None,
                    placeholder="Select a base model…",
                    help="Open-source model to fine-tune. Architecture is auto-filled for GQA-aware adapter param calculation.",
                    key="ms_ft_base_model",
                )

            # Base model info row
            if ft_base_model_name and ft_base_model_name != "Custom":
                _sel_model = next(m for m in _known_models if m.name == ft_base_model_name)
                _param_str = _fmt_tokens(_sel_model.parameter_count)
                _arch      = _sel_model.architecture
                st.caption(
                    f"**{_sel_model.name}** — {_param_str} params · "
                    f"{_arch.get('num_layers', '?')} layers · "
                    f"d_model {_arch.get('d_model', '?')} · "
                    f"{_arch.get('num_heads', '?')}/{_arch.get('num_kv_heads', '?')} heads (Q/KV)"
                )
                if _sel_model.notes:
                    st.caption(_sel_model.notes.strip())
            elif ft_base_model_name == "Custom":
                st.caption("Custom model: parameters estimated using uniform d_model approximation.")

            if ft_method in ("LoRA", "QLoRA"):
                lora_col1, lora_col2, lora_col3 = st.columns(3)
                with lora_col1:
                    lora_rank = st.slider(
                        "LoRA Rank (r)", min_value=1, max_value=256, value=16, step=1,
                        help="Rank of the low-rank adapter matrices. Higher = more expressive, more parameters.",
                        key="ms_lora_rank",
                    )
                with lora_col2:
                    target_modules = st.multiselect(
                        "Target Modules",
                        options=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Weight matrices that receive LoRA adapters.",
                        key="ms_lora_modules",
                    )
                with lora_col3:
                    # Live adapter param preview when a base model is selected
                    if ft_base_model_name and ft_base_model_name != "Custom" and target_modules:
                        _sel_model  = next(m for m in _known_models if m.name == ft_base_model_name)
                        _n_adapter  = calculate_lora_trainable_params(
                            ft_method=ft_method,
                            base_params=_sel_model.parameter_count,
                            target_modules=target_modules,
                            d_model=_sel_model.architecture.get("d_model", 0),
                            num_layers=_sel_model.architecture.get("num_layers", 0),
                            lora_rank=lora_rank,
                            architecture=_sel_model.architecture,
                        ) or 0
                        _pct = (_n_adapter / _sel_model.parameter_count * 100) if _sel_model.parameter_count else 0
                        st.metric(
                            "Adapter Params",
                            _fmt_tokens(_n_adapter),
                            delta=f"{_pct:.2f}% of base",
                            delta_color="off",
                            help=f"{len(target_modules)} module(s) × rank {lora_rank} "
                                 f"across {_sel_model.architecture.get('num_layers', '?')} layers.",
                        )

    if budget_value is None:
        st.caption("Enter a budget above to continue.")
        return

    compute_budget = float(budget_value * _UNIT_MULTIPLIERS[budget_unit])
    bytes_per_token: int = modality_cfg["bytes_per_token"]
    architecture: str = modality_cfg["arch"]
    is_lora = ft_method in ("LoRA", "QLoRA")

    # ── Fine-tuning base model resolution ────────────────────────────────────
    # Read selections from session state (they were set inside the expander above).
    ft_base_model_name = st.session_state.get("ms_ft_base_model")
    lora_rank          = st.session_state.get("ms_lora_rank", 16)
    target_modules     = st.session_state.get("ms_lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

    _known_models   = _cached_models()
    is_ft_with_base = (
        training_type == "Fine-Tuning"
        and ft_base_model_name
        and ft_base_model_name != "Custom"
    )
    ft_base_model  = None
    ft_base_params = 0
    ft_arch        = {}
    n_adapter      = 0   # trainable / displayed parameter count for fine-tuning

    if is_ft_with_base:
        ft_base_model = next((m for m in _known_models if m.name == ft_base_model_name), None)
        if ft_base_model:
            ft_base_params = ft_base_model.parameter_count
            ft_arch        = ft_base_model.architecture
            if is_lora:
                n_adapter = calculate_lora_trainable_params(
                    ft_method=ft_method,
                    base_params=ft_base_params,
                    target_modules=target_modules or [],
                    d_model=ft_arch.get("d_model", 0),
                    num_layers=ft_arch.get("num_layers", 0),
                    lora_rank=lora_rank,
                    architecture=ft_arch,
                ) or 0
            else:
                n_adapter = ft_base_params   # SFT: all params are trainable

    # ── 2. Auto-configure hardware (pure computation, needed for schedule defaults) ──
    region = current_region()
    hw = _auto_configure_hardware(region, _catalog_stamp(region))
    peak_flops_per_gpu = hw["peak_flops_per_gpu"]
    mfu = hw["mfu"]
    gpus_per_instance = hw["gpus_per_instance"]
    hourly_cost = hw["hourly_cost"]
    gradient_checkpointing = hw["gradient_checkpointing"]
    instance_spec = hw["instance_spec"]

    multiplier = _ARCH_MULTIPLIERS.get(architecture, 6.0)

    budget_display = (
        f"${compute_budget / 1e9:.1f}B" if compute_budget >= 1e9
        else f"${compute_budget / 1e6:.1f}M" if compute_budget >= 1e6
        else f"${compute_budget / 1e3:.1f}K" if compute_budget >= 1e3
        else f"${compute_budget:,.0f}"
    )

    # ── 3. Cascade defaults into Training Schedule & Storage ─────────────────
    # Write derived defaults into session state whenever any Project Setup field changes.
    # Widgets read session state (key= argument), so they'll reflect the new defaults
    # automatically. If the user manually overrides a widget, the override sticks until
    # the next Project Setup change.
    quick_n = _quick_n_estimate(compute_budget, multiplier, hw)
    # Region belongs in the signature: changing it changes prices, so the
    # derived schedule defaults must be recomputed alongside them.
    setup_sig = (
        compute_budget, modality, training_type, ft_method or "",
        ft_base_model_name or "", lora_rank, region,
    )

    if st.session_state.get("ms_setup_sig") != setup_sig:
        st.session_state["ms_setup_sig"] = setup_sig
        defaults = _derive_schedule_defaults(
            compute_budget=compute_budget,
            modality=modality,
            training_type=training_type,
            ft_method=ft_method,
            quick_n=quick_n,
        )
        st.session_state["ms_epochs"]          = defaults["epochs"]
        st.session_state["ms_hp_trials"]       = defaults["hp_trials"]
        st.session_state["ms_hp_fraction_pct"] = defaults["hp_fraction_pct"]
        st.session_state["ms_storage_months"]  = defaults["storage_months"]
        st.session_state["ms_storage_class"]   = defaults["storage_class"]
        # Reset trade-off sliders and instance override so they default to the new optimal point.
        st.session_state.pop("ms_dataset_slider", None)
        st.session_state.pop("ms_model_slider", None)
        st.session_state.pop("ms_rank_slider", None)
        st.session_state.pop("ms_num_instances", None)

    # ── 4. Bind schedule variables from session state & run full solve ───────
    # The cascade above wrote defaults into session state before any widgets render.
    # Reading here gives the same values the widgets will display, so the solve and
    # recommendation are always in sync with what the user sees in the expander below.
    # When the user changes a widget, Streamlit reruns, session state updates, and
    # both the solve and recommendation refresh automatically.
    epochs                  = st.session_state.get("ms_epochs", 1)
    num_hp_trials           = st.session_state.get("ms_hp_trials", 0)
    hp_fraction_pct         = st.session_state.get("ms_hp_fraction_pct", 10)
    storage_duration_months = st.session_state.get("ms_storage_months", 3)
    storage_class           = st.session_state.get("ms_storage_class", "standard")

    hp_fraction        = hp_fraction_pct / 100.0
    project_multiplier = 1.0 + num_hp_trials * hp_fraction
    cost_per_tb_month  = get_storage_cost(storage_class, region)
    storage_cost_per_token = (
        (bytes_per_token / 1e12) * cost_per_tb_month * storage_duration_months
    )

    optimal_ratio = LORA_OPTIMAL_RATIO if is_lora else CHINCHILLA_OPTIMAL_RATIO

    # cost per (parameter × token) — num_instances cancels in cost formula
    cost_per_token_param = (
        epochs * multiplier / (peak_flops_per_gpu * mfu * gpus_per_instance * 3600)
    ) * hourly_cost

    # ── 5. Solve for optimal (n_opt, d_opt) ──────────────────────────────────
    ckpt_cost_per_param = (
        BYTES_PER_PARAM_CHECKPOINT / 1e12
        * cost_per_tb_month
        * storage_duration_months
        * (_CHECKPOINTS_PER_RUN + num_hp_trials)
    )

    if is_ft_with_base and n_adapter > 0:
        # Fine-tuning with a known base model: N is fixed by the model + rank.
        # For LoRA/QLoRA the forward/backward pass runs through the full base model,
        # so compute cost uses ft_base_params; adapter N is used for checkpoints only.
        #
        # Two competing bounds on D:
        #
        # 1) LoRA efficiency bound — empirical optimal ratio (D/N_adapter ≈ 50):
        #      D_eff = LORA_OPTIMAL_RATIO × n_adapter
        #    This scales with rank and is the primary bound at large budgets.
        #
        # 2) Budget bound — how much data the full budget can afford:
        #      D_budget = (B − ckpt_cost) / (cost_per_token_param × N_base + storage_per_token)
        #    This is the hard ceiling; at small budgets it overrides the efficiency bound.
        #
        # D_opt = min(D_eff, D_budget)
        n_flops_base  = ft_base_params   # used for compute FLOPs
        n_opt         = float(n_adapter) # displayed / checkpoint param count
        ckpt_total    = ckpt_cost_per_param * n_adapter
        denom         = (
            project_multiplier * cost_per_token_param * n_flops_base
            + storage_cost_per_token
        )
        d_budget = max(compute_budget - ckpt_total, 0.0) / denom if denom > 0 else 0.0
        d_eff    = float(LORA_OPTIMAL_RATIO * n_adapter) if is_lora else d_budget
        d_opt    = min(d_eff, d_budget)
    else:
        # Pre-training (or FT without a selected base model): quadratic solve.
        # Total budget B = compute_total + dataset_storage + checkpoint_storage
        # Under Chinchilla D = optimal_ratio × N:  a·N² + b·N = B
        #   a = optimal_ratio × project_multiplier × cost_per_token_param
        #   b = optimal_ratio × storage_cost_per_token + ckpt_cost_per_param
        # N* = (−b + √(b² + 4aB)) / (2a)
        n_flops_base = 0   # not used in pre-training path
        a_coeff = optimal_ratio * project_multiplier * cost_per_token_param
        b_coeff = optimal_ratio * storage_cost_per_token + ckpt_cost_per_param
        if a_coeff > 0:
            discriminant = b_coeff ** 2 + 4 * a_coeff * compute_budget
            n_opt = (-b_coeff + math.sqrt(max(discriminant, 0.0))) / (2 * a_coeff)
            d_opt = optimal_ratio * n_opt
        else:
            n_opt, d_opt = 0.0, 0.0

    # ── Cost breakdown at optimal point ──────────────────────────────────────
    # Use ft_base_params for FLOPs when fine-tuning (forward pass through full model);
    # use n_opt (adapter params or pre-train N) for checkpoint storage.
    _opt_flops_n = max(int(ft_base_params if (is_ft_with_base and n_flops_base) else n_opt), 1)
    opt_flops = calculate_training_flops(
        parameter_count=_opt_flops_n,
        training_tokens=max(int(d_opt), 1),
        architecture=architecture.lower(),
        epochs=epochs,
        gradient_checkpointing=gradient_checkpointing,
    )
    _, _, opt_compute_cost_per_run, _ = estimate_compute_cost(
        total_flops=opt_flops,
        peak_flops_per_gpu=peak_flops_per_gpu,
        mfu=mfu,
        total_gpus=gpus_per_instance,
        num_instances=1,
        hourly_cost=hourly_cost,
    )
    opt_compute_cost_total = opt_compute_cost_per_run * project_multiplier
    opt_dataset_storage_cost = calculate_storage_cost(
        dataset_size_tb=max(int(d_opt), 1) * bytes_per_token / 1e12,
        storage_duration_months=storage_duration_months,
        storage_class=storage_class,
    )
    opt_ckpt_storage_cost = calculate_storage_cost(
        dataset_size_tb=calculate_checkpoint_storage_tb(
            checkpoint_params=max(int(n_opt), 1),
            num_checkpoints=_CHECKPOINTS_PER_RUN,
            num_training_runs=1,
            num_hp_trials=num_hp_trials,
            num_ablations=0,
        ),
        storage_duration_months=storage_duration_months,
        storage_class=storage_class,
    )
    opt_storage_cost   = opt_dataset_storage_cost + opt_ckpt_storage_cost
    total_project_cost = opt_compute_cost_total + opt_storage_cost

    # recommended_instances is computed later, after the slider selection is resolved,
    # so it stays in sync with the FLOP count actually being displayed.

    # ── 6. Pre-compute slider selection from session state ───────────────────
    # Reading slider values from session state *before* the column layout means
    # the left panel can display the selected (not just optimal) values on every
    # rerun.  The slider widgets in the right column write back to the same keys,
    # so dragging them reruns the page and refreshes the left column automatically.
    tokens_per_sample = modality_cfg.get("tokens_per_sample", 512)
    sample_noun       = modality_cfg.get("sample_noun", "samples")

    slider_compute_budget   = max(compute_budget - opt_storage_cost, compute_budget * 0.5)
    slider_effective_budget = slider_compute_budget / project_multiplier

    # Dynamic slider bounds — scale to the optimal point.
    # For LoRA, d_opt can be well below 1B tokens (e.g. 50 × a few-million adapter params),
    # so we anchor the minimum ~2 log-decades below d_opt instead of hardcoding 1B.
    if is_ft_with_base and is_lora and d_opt > 0:
        d_slider_min = max(6.0, math.floor(math.log10(max(d_opt, 1e6)) / 0.05) * 0.05 - 2.0)
    else:
        d_slider_min = 9.0   # pre-training / SFT: 1B token minimum

    d_log_opt    = math.log10(max(d_opt, 10 ** d_slider_min))
    d_slider_max = max(d_slider_min + 4.0, math.ceil(d_log_opt / 0.05) * 0.05 + 0.05)

    n_log_opt    = math.log10(max(n_opt, 1e7))
    n_slider_min = 7.0
    n_slider_max = max(11.0, math.ceil(n_log_opt / 0.05) * 0.05 + 0.05)

    # For fine-tuning, the explore directions are adapted:
    #   LoRA/QLoRA: "Dataset size" (D varies, adapter N fixed) | "LoRA Rank" (rank varies → N + D both change)
    #   SFT: only "Dataset size" (N fixed = ft_base_params)
    #   Pre-training: standard "Token count → model size" | "Model size → token count"
    _ft_explore_opts = (
        ["Dataset size → token/param ratio", "LoRA Rank → dataset size"]
        if (is_ft_with_base and is_lora)
        else ["Dataset size → token/param ratio"]
        if is_ft_with_base
        else ["Token count → model size", "Model size → token count"]
    )
    _explore_default = st.session_state.get("ms_explore_dir", _ft_explore_opts[0])
    # Guard: if the stored direction isn't valid for the current mode, reset it.
    if _explore_default not in _ft_explore_opts:
        _explore_default = _ft_explore_opts[0]
    explore_dir = _explore_default

    # ── Slider pre-computation (read from session state so left col reflects selection) ──

    if is_ft_with_base and is_lora:
        # LoRA mode: N is determined by base model + rank.  Both slider directions are useful.
        if explore_dir == "LoRA Rank → dataset size":
            # Rank slider: user picks rank, we solve for optimal D.
            rank_default = st.session_state.get("ms_rank_slider", lora_rank)
            sel_rank     = int(rank_default)
            sel_n_adapter = calculate_lora_trainable_params(
                ft_method=ft_method,
                base_params=ft_base_params,
                target_modules=target_modules or [],
                d_model=ft_arch.get("d_model", 0),
                num_layers=ft_arch.get("num_layers", 0),
                lora_rank=sel_rank,
                architecture=ft_arch,
            ) or 1
            # Linear solve for D with this hypothetical rank
            _ckpt_n   = sel_n_adapter
            _denom    = (
                project_multiplier * cost_per_token_param * ft_base_params
                + storage_cost_per_token
            )
            _eff_bgt  = max(compute_budget - ckpt_cost_per_param * _ckpt_n, 0.0)
            selected_tokens = max(int(_eff_bgt / _denom), 1) if _denom > 0 else 1
            selected_params = sel_n_adapter
        else:
            # Dataset slider: D varies, adapter N stays fixed at Project Setup rank.
            d_opt_log_raw     = math.log10(max(d_opt, 10 ** d_slider_min))
            d_opt_log_default = round(max(d_slider_min, min(d_slider_max, d_opt_log_raw)) / 0.05) * 0.05
            log_d             = st.session_state.get("ms_dataset_slider", d_opt_log_default)
            selected_tokens   = int(10 ** log_d)
            selected_params   = max(n_adapter, 1)
        # FLOPs: always use full base model (frozen forward/backward pass)
        flops_params = ft_base_params
        n_axis_title = "Adapter params"
        n_hover      = "Adapter N"

    elif is_ft_with_base:
        # SFT: all params are trained.  Only dataset exploration makes sense.
        d_opt_log_raw     = math.log10(max(d_opt, 10 ** d_slider_min))
        d_opt_log_default = round(max(d_slider_min, min(d_slider_max, d_opt_log_raw)) / 0.05) * 0.05
        log_d             = st.session_state.get("ms_dataset_slider", d_opt_log_default)
        selected_tokens   = int(10 ** log_d)
        selected_params   = ft_base_params
        flops_params      = ft_base_params
        n_axis_title      = "Model size (params)"
        n_hover           = "N"

    else:
        # Pre-training: free solve in both directions.
        if explore_dir == "Token count → model size":
            d_opt_log_raw     = math.log10(max(d_opt, 10 ** d_slider_min))
            d_opt_log_default = round(max(d_slider_min, min(d_slider_max, d_opt_log_raw)) / 0.05) * 0.05
            log_d             = st.session_state.get("ms_dataset_slider", d_opt_log_default)
            selected_tokens   = int(10 ** log_d)
            selected_params   = solve_for_parameter_count(
                compute_budget_usd=slider_effective_budget,
                training_tokens=selected_tokens,
                peak_flops_per_gpu=peak_flops_per_gpu,
                mfu=mfu,
                total_gpus=gpus_per_instance,
                num_instances=1,
                hourly_cost=hourly_cost,
                architecture=architecture.lower(),
                epochs=epochs,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            n_opt_log_raw     = math.log10(max(n_opt, 1e7))
            n_opt_log_default = round(max(n_slider_min, min(n_slider_max, n_opt_log_raw)) / 0.05) * 0.05
            log_n             = st.session_state.get("ms_model_slider", n_opt_log_default)
            selected_params   = int(10 ** log_n)
            selected_tokens   = solve_for_training_tokens(
                compute_budget_usd=slider_effective_budget,
                parameter_count=selected_params,
                peak_flops_per_gpu=peak_flops_per_gpu,
                mfu=mfu,
                total_gpus=gpus_per_instance,
                num_instances=1,
                hourly_cost=hourly_cost,
                architecture=architecture.lower(),
                epochs=epochs,
                gradient_checkpointing=gradient_checkpointing,
            )
        flops_params = selected_params
        n_axis_title = "Max model size (params)"
        n_hover      = "Max N"

    total_flops_sel = calculate_training_flops(
        parameter_count=max(flops_params, 1),
        training_tokens=max(selected_tokens, 1),
        architecture=architecture.lower(),
        epochs=epochs,
        gradient_checkpointing=gradient_checkpointing,
    )
    _, _, compute_cost_sel, wall_clock_days_sel = estimate_compute_cost(
        total_flops=total_flops_sel,
        peak_flops_per_gpu=peak_flops_per_gpu,
        mfu=mfu,
        total_gpus=gpus_per_instance,
        num_instances=1,
        hourly_cost=hourly_cost,
    )

    # Recommend num_instances to hit ≤ _TARGET_WALL_CLOCK_DAYS for the *selected* point.
    # Keeping this here (after the slider solve) ensures the hardware card and the
    # wall-clock metric in the left column are always derived from the same FLOP count.
    wall_clock_hours_1inst = total_flops_sel / (peak_flops_per_gpu * mfu * gpus_per_instance * 3600)
    recommended_instances  = min(
        _MAX_INSTANCES,
        max(1, math.ceil(wall_clock_hours_1inst / (_TARGET_WALL_CLOCK_DAYS * 24))),
    )
    # User can override the cluster size; default resets whenever setup changes.
    num_instances = st.session_state.get("ms_num_instances", recommended_instances)

    sel_dataset_storage = calculate_storage_cost(
        dataset_size_tb=max(selected_tokens, 1) * bytes_per_token / 1e12,
        storage_duration_months=storage_duration_months,
        storage_class=storage_class,
    )
    sel_ckpt_storage = calculate_storage_cost(
        dataset_size_tb=calculate_checkpoint_storage_tb(
            checkpoint_params=max(selected_params, 1),
            num_checkpoints=_CHECKPOINTS_PER_RUN,
            num_training_runs=1,
            num_hp_trials=num_hp_trials,
            num_ablations=0,
        ),
        storage_duration_months=storage_duration_months,
        storage_class=storage_class,
    )
    sel_compute_cost_total = compute_cost_sel * project_multiplier
    sel_total_cost         = sel_compute_cost_total + sel_dataset_storage + sel_ckpt_storage
    sel_wall_clock_days = wall_clock_hours_1inst / (num_instances * 24)
    sel_samples         = selected_tokens / max(tokens_per_sample, 1)
    current_ratio       = selected_tokens / max(selected_params, 1)
    budget_used_pct     = min(sel_total_cost / max(compute_budget, 1e-9), 1.0)

    # ── Two-column layout ─────────────────────────────────────────────────────
    col_dims, col_explorer = st.columns(2)

    # ── Left: model dimensions driven by the slider ──────────────────────────
    with col_dims:
        st.subheader("Optimal Model Dimensions")
        if is_ft_with_base and ft_base_model:
            st.caption(
                f"**Base:** {ft_base_model_name} ({_fmt_tokens(ft_base_params)} params)"
            )

        dm1, dm2 = st.columns(2)
        with dm1:
            # Label and help text differ for LoRA vs SFT vs pre-training
            if is_ft_with_base and is_lora:
                _n_label  = "Adapter Params"
                _n_help   = (
                    f"Trainable params from rank-{lora_rank} adapters on "
                    f"{', '.join(target_modules or [])}. "
                    f"Base model has {_fmt_tokens(ft_base_params)} frozen params."
                )
            elif is_ft_with_base:
                _n_label = "Model Params (all)"
                _n_help  = f"All {_fmt_tokens(ft_base_params)} params updated during SFT."
            else:
                _n_label = "Model Size"
                _n_help  = f"Chinchilla-optimal N = {_fmt_tokens(int(n_opt))}. Drag the slider to deviate."

            st.metric(_n_label, _fmt_tokens(int(selected_params)), help=_n_help)
            if is_ft_with_base and is_lora:
                _d_bound = "efficiency-bound" if d_opt < d_budget else "budget-bound"
                _d_help  = (
                    f"LoRA efficiency optimal: **{_fmt_tokens(int(d_eff))}** "
                    f"({LORA_OPTIMAL_RATIO}× adapter params). "
                    f"Budget ceiling: **{_fmt_tokens(int(d_budget))}**. "
                    f"Showing the {_d_bound} value. "
                    f"Increase rank to raise the efficiency bound."
                )
            elif is_ft_with_base:
                _d_help = f"Budget-optimal D = {_fmt_tokens(int(d_opt))} for this base model."
            else:
                _d_help = (
                    f"At ~{tokens_per_sample:,} tokens/{sample_noun[:-1]}, "
                    f"that's ≈ {_fmt_tokens(int(sel_samples))} {sample_noun}. "
                    f"Chinchilla-optimal D = {_fmt_tokens(int(d_opt))}."
                )
            st.metric(
                "Required Dataset",
                _fmt_tokens(int(selected_tokens)),
                delta=f"≈ {_fmt_tokens(int(sel_samples))} {sample_noun}",
                delta_color="off",
                help=_d_help,
            )
            st.metric(
                "Wall-clock Time",
                _format_wall_clock_time(sel_wall_clock_days),
                help=f"With {num_instances} instance(s) × {gpus_per_instance} GPUs. "
                     f"Adjust cluster size in Training Schedule & Storage.",
            )
            st.metric(
                "HP Trials",
                str(num_hp_trials),
                help=f"Each trial costs {hp_fraction:.0%} of a full run.",
            )
        with dm2:
            st.metric(
                "Compute (all runs)",
                f"${sel_compute_cost_total:,.0f}",
                help=f"Per run × {project_multiplier:.2f}× multiplier."
                + (f" Compute uses {_fmt_tokens(ft_base_params)}-param base model FLOPs." if is_ft_with_base and is_lora else ""),
            )
            st.metric(
                "Dataset Storage",
                f"${sel_dataset_storage:,.2f}",
                help=(
                    f"{_fmt_tokens(int(selected_tokens))} tokens × {bytes_per_token} B/tok "
                    f"= {int(selected_tokens) * bytes_per_token / 1e12:.4f} TB "
                    f"× ${cost_per_tb_month:.2f}/TB/mo × {storage_duration_months} mo."
                ),
            )
            st.metric(
                "Checkpoint Storage",
                f"${sel_ckpt_storage:,.2f}",
                help=f"{_CHECKPOINTS_PER_RUN} checkpoints/run + {num_hp_trials} HP trial checkpoints."
                + (" Adapters only — base weights aren't saved." if is_ft_with_base and is_lora else ""),
            )
            over_budget = sel_total_cost > compute_budget * 1.001
            st.metric(
                "Total Project Cost",
                f"${sel_total_cost:,.0f}",
                delta=f"{'over' if over_budget else 'within'} budget",
                delta_color="inverse" if over_budget else "off",
                help="Compute + dataset storage + checkpoint storage.",
            )

    # ── Right: trade-off explorer controls ───────────────────────────────────
    with col_explorer:
        st.subheader("Trade-off Explorer")
        if is_ft_with_base and is_lora:
            st.caption(
                "Explore how dataset size or LoRA rank shifts the allocation within your budget."
            )
        elif is_ft_with_base:
            st.caption("Drag to explore how much data you can afford for this model.")

        # Direction radio — options depend on training mode
        st.radio(
            "Explore by",
            options=_ft_explore_opts,
            index=_ft_explore_opts.index(_explore_default),
            horizontal=True,
            key="ms_explore_dir",
        )

        # Slider widget — differs by explore direction and training mode
        if explore_dir == "LoRA Rank → dataset size":
            st.slider(
                "LoRA Rank (r)",
                min_value=1, max_value=256,
                value=lora_rank, step=1,
                help="Vary rank to see how adapter capacity changes the affordable dataset size.",
                key="ms_rank_slider",
            )
            st.caption(
                f"Adapter params at this rank: **{_fmt_tokens(selected_params)}** "
                f"({selected_params / ft_base_params * 100:.2f}% of {_fmt_tokens(ft_base_params)})"
                if ft_base_params else ""
            )
        elif explore_dir == "Model size → token count":
            st.slider(
                "Model size (params)",
                min_value=n_slider_min, max_value=n_slider_max,
                value=n_opt_log_default, step=0.05,
                format="10^%.2f",
                help="Drag to fix model size; dataset size updates to stay within budget.",
                key="ms_model_slider",
            )
        else:
            # Dataset slider — all modes that explore by token count
            st.slider(
                "Dataset size (tokens)",
                min_value=d_slider_min, max_value=d_slider_max,
                value=d_opt_log_default, step=0.05,
                format="10^%.2f",
                help=(
                    "Drag to explore dataset sizes. "
                    + ("Compute uses full base model FLOPs; adapter N stays fixed." if is_ft_with_base and is_lora
                       else "Model size updates to stay within budget." if not is_ft_with_base
                       else "All base-model params are trained.")
                ),
                key="ms_dataset_slider",
            )

        # Chinchilla / LoRA assessment banner
        if is_ft_with_base and is_lora:
            if current_ratio < 10:
                st.warning(
                    f"**Below LoRA sweet spot.** {current_ratio:.1f} tok/adapter_param "
                    f"(target: 10–100×). Use more data or reduce rank."
                )
            elif current_ratio <= 100:
                st.success(
                    f"**Within LoRA sweet spot.** {current_ratio:.1f} tok/adapter_param — good range."
                )
            else:
                st.info(
                    f"**Above LoRA sweet spot.** {current_ratio:.1f} tok/adapter_param. "
                    f"Consider increasing rank or reducing dataset."
                )
        elif not is_ft_with_base:
            _render_chinchilla_assessment(
                ratio=current_ratio,
                optimal_tokens=int(selected_params * CHINCHILLA_OPTIMAL_RATIO),
                fix_hint="Move the slider toward the star to return to Chinchilla-optimal.",
            )



    # ── 7. Chart (full-width, paired with the Trade-off Explorer above) ──────
    d_range = np.logspace(d_slider_min, d_slider_max, 300)
    n_budget = _budget_curve_n(
        d_tokens=d_range,
        budget=slider_effective_budget,
        peak_flops_per_gpu=peak_flops_per_gpu,
        mfu=mfu,
        gpus_per_instance=gpus_per_instance,
        hourly_cost=hourly_cost,
        multiplier=multiplier,
        epochs=epochs,
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=d_range.tolist(),
        y=n_budget.tolist(),
        name="Budget curve (max N)",
        mode="lines",
        line=dict(color=CHART_COLORS["compute"], width=2),
        hovertemplate=f"D=%{{x:.2e}} tokens<br>{n_hover}=%{{y:.2e}}<extra>Budget curve</extra>",
    ))

    n_optimal_line = d_range / optimal_ratio
    opt_line_label = (
        f"Chinchilla optimal (N = D / {CHINCHILLA_OPTIMAL_RATIO})"
        if not is_lora
        else f"LoRA sweet spot (D / {LORA_OPTIMAL_RATIO})"
    )
    fig.add_trace(go.Scatter(
        x=d_range.tolist(),
        y=n_optimal_line.tolist(),
        name=opt_line_label,
        mode="lines",
        line=dict(color=CHART_COLORS["success"], width=2, dash="dash"),
        hovertemplate=f"D=%{{x:.2e}}<br>Optimal N=%{{y:.2e}}<extra>{opt_line_label}</extra>",
    ))

    # For LoRA with a base model: show the adapter parameter count as a
    # horizontal reference so the user can see exactly where their adapter sits
    # relative to the budget curve and sweet-spot line.
    if is_ft_with_base and is_lora and n_adapter > 0:
        fig.add_trace(go.Scatter(
            x=d_range.tolist(),
            y=[n_adapter] * len(d_range),
            name=f"Adapter params (rank {lora_rank})",
            mode="lines",
            line=dict(color=CHART_COLORS.get("warning", "#f5a623"), width=1, dash="dot"),
            hovertemplate=(
                f"Adapter: {_fmt_tokens(n_adapter)} params<br>"
                f"Base: {_fmt_tokens(ft_base_params)}<extra>Adapter line</extra>"
            ),
        ))

    if d_opt >= 10 ** d_slider_min:
        _star_n = n_adapter if (is_ft_with_base and is_lora) else n_opt
        fig.add_trace(go.Scatter(
            x=[d_opt], y=[_star_n],
            name="Budget-optimal point" if is_ft_with_base else "Optimal sweet spot",
            mode="markers",
            marker=dict(color=CHART_COLORS["warning"], size=14, symbol="star"),
            hovertemplate=(
                f"Budget-optimal<br>D={_fmt_tokens(int(d_opt))}<br>"
                f"{'Adapter' if is_ft_with_base and is_lora else 'N'}={_fmt_tokens(int(_star_n))}"
                f"<extra></extra>"
            ),
        ))

    if selected_tokens >= 10 ** d_slider_min and selected_params > 0:
        fig.add_trace(go.Scatter(
            x=[selected_tokens], y=[selected_params],
            name="Your selection",
            mode="markers",
            marker=dict(color="white", size=12, symbol="circle",
                        line=dict(color=CHART_COLORS["compute"], width=2)),
            hovertemplate=(
                f"Your point<br>D={_fmt_tokens(selected_tokens)}"
                f"<br>N={_fmt_tokens(selected_params)}<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(
            type="log",
            title="Dataset size (tokens)",
            gridcolor="rgba(255,255,255,0.08)",
            range=[d_slider_min, d_slider_max],
        ),
        yaxis=dict(
            type="log",
            title=n_axis_title,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=10, l=10, r=10),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 8. Training Schedule & Storage expander ───────────────────────────────
    # Widgets read from (and write back to) session state via key=.
    # Any change triggers a Streamlit rerun, updating the solve and recommendation above.
    with st.expander("Training Schedule & Storage", expanded=False):
        _class_label = "Standard-IA" if storage_class == "standard_ia" else "Standard"
        st.caption(
            f"Auto-configured from **{budget_display}** budget, **{modality}**, "
            f"**{training_type}**: "
            f"{epochs} epoch(s) · {num_hp_trials} HP trial(s) · {hp_fraction_pct}% HP cost · "
            f"{storage_duration_months} month(s) storage · {_class_label}. "
            f"Override any value below."
        )

        sched_col1, sched_col2, sched_col3, sched_col4 = st.columns(4)
        with sched_col1:
            st.number_input(
                "Epochs",
                min_value=1, max_value=1000, step=1,
                help="Passes through the training dataset. More epochs multiply compute cost.",
                key="ms_epochs",
            )
            st.number_input(
                "HP Tuning Trials",
                min_value=0, max_value=500, step=1,
                help="Hyperparameter search runs before the primary run. "
                     "Each trial is costed against the total budget.",
                key="ms_hp_trials",
            )
        with sched_col2:
            st.slider(
                "HP Trial Cost (% of full run)",
                min_value=1, max_value=100, step=1,
                disabled=(num_hp_trials == 0),
                help="Fraction of a full training run's compute each HP trial costs.",
                key="ms_hp_fraction_pct",
            )
            st.metric(
                "Project Compute Multiplier",
                f"{project_multiplier:.2f}×",
                help=f"1 primary run + {num_hp_trials} HP trial(s) × {hp_fraction:.0%} each.",
            )
        with sched_col3:
            st.number_input(
                "Storage Duration (months)",
                min_value=1, max_value=60, step=1,
                help="How long to retain dataset + checkpoints in object storage.",
                key="ms_storage_months",
            )
            # Options and prices both come from live pricing; the previous
            # hardcoded "$23/TB/mo" labels were a fourth copy of this data.
            st.selectbox(
                "Storage Class",
                options=list_storage_classes(region),
                help="S3 storage tier.",
                format_func=lambda name: f"{name} (${get_storage_cost(name, region):,.2f}/TB/mo)",
                key="ms_storage_class",
            )
        with sched_col4:
            st.number_input(
                "GPU Instances",
                min_value=1, max_value=_MAX_INSTANCES, step=1,
                value=recommended_instances,
                help=(
                    f"Number of {instance_spec['display_name']} instances to run in parallel. "
                    f"Auto-recommendation: **{recommended_instances}** (targets ≤ {_TARGET_WALL_CLOCK_DAYS} day wall-clock). "
                    f"Increasing this reduces training time; decreasing it lowers parallelism costs."
                ),
                key="ms_num_instances",
            )
            st.metric(
                "Total GPU Count",
                f"{num_instances * gpus_per_instance:,}",
                help=f"{num_instances} instance(s) × {gpus_per_instance} GPUs each.",
            )

    # ── 9. Hardware info card ─────────────────────────────────────────────────
    _rec_note = (
        f" (auto: {recommended_instances})" if num_instances != recommended_instances else ""
    )
    st.info(
        f"**Auto-configured hardware:** {instance_spec['display_name']}  "
        f"| bf16 | MFU {mfu:.0%} | ${hourly_cost:.2f}/hr per instance | "
        f"**Cluster:** {num_instances}{_rec_note} instance(s) × {gpus_per_instance} GPUs "
        f"= {num_instances * gpus_per_instance} GPUs total "
        f"→ ~{sel_wall_clock_days:.1f} day wall-clock",
        icon="⚙️",
    )



def _budget_curve_n(
    d_tokens: np.ndarray,
    budget: float,
    peak_flops_per_gpu: float,
    mfu: float,
    gpus_per_instance: int,
    hourly_cost: float,
    multiplier: float,
    epochs: int,
) -> np.ndarray:
    """Vectorised: N_max = budget / (cost_per_token_param × D).

    num_instances cancels in the cost formula, so we pass gpus_per_instance directly.
    """
    cost_per_token_param = (
        epochs * multiplier / (peak_flops_per_gpu * mfu * gpus_per_instance * 3600)
    ) * hourly_cost
    return budget / (cost_per_token_param * d_tokens)
