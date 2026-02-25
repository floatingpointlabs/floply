"""Interactive budget explorer: find the largest model you can afford to train."""

import math

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.app.config import ARCHITECTURE_OPTIONS, CHART_COLORS, CHINCHILLA_OPTIMAL_RATIO
from src.app.helpers import (
    _fmt_tokens,
    _format_wall_clock_time,
    _UNIT_MULTIPLIERS,
    _render_chinchilla_assessment,
)
from src.cost_modelling.calculator import (
    solve_for_parameter_count,
    calculate_training_flops,
    estimate_compute_cost,
)

# FLOPs multipliers per architecture (must mirror calculator._get_flops_multiplier)
_ARCH_MULTIPLIERS = {
    "Transformer": 6.0,
    "CNN": 4.0,
    "RNN": 8.0,
    "ViT": 6.0,
    "Diffusion": 6.5,
}


def render_model_size_page():
    st.markdown(
        "<p style='text-align:center;color:#888;'>Given a compute budget and hardware — explore the model-size / dataset-size trade-off.</p>",
        unsafe_allow_html=True,
    )

    # ── Section 0: Compute Budget ────────────────────────────────────────────
    with st.expander("Compute Budget", expanded=True):
        bv_col, bu_col = st.columns([3, 1])
        with bv_col:
            budget_value = st.number_input(
                "Amount (USD)",
                min_value=0.001,
                max_value=9_999.0,
                value=None,
                step=0.1,
                format="%.3f",
                help=(
                    "Total GPU compute budget. Enter a number then choose a scale. "
                    "Storage costs are not included."
                ),
                key="ms_budget_val",
                placeholder="e.g. 10.0",
            )
        with bu_col:
            budget_unit = st.selectbox(
                "Scale",
                options=["K", "M", "B", "T"],
                index=1,
                key="ms_budget_unit",
            )

    if budget_value is None:
        st.caption("Enter a compute budget above to continue.")
        return

    compute_budget = float(budget_value * _UNIT_MULTIPLIERS[budget_unit])
    st.caption(f"Budget: **${compute_budget:,.0f}**")

    # ── Section 1: Training Task ─────────────────────────────────────────────
    ft_method: str | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

    with st.expander("Training Task", expanded=True):
        training_type = st.selectbox(
            "Training Type",
            options=["Pre-Training", "Fine-Tuning"],
            index=None,
            help="Pre-Training trains from scratch; Fine-Tuning adapts an existing model.",
            key="ms_training_type",
        )

        if training_type == "Fine-Tuning":
            ft_method = st.selectbox(
                "Fine-Tuning Method",
                options=["Full Fine-Tuning (SFT)", "LoRA", "QLoRA"],
                index=0,
                help=(
                    "SFT updates all parameters. "
                    "LoRA trains small low-rank adapter matrices. "
                    "QLoRA quantises the frozen base to 4-bit to cut memory."
                ),
                key="ms_ft_method",
            )

            if ft_method in ("LoRA", "QLoRA"):
                lora_col1, lora_col2 = st.columns(2)
                with lora_col1:
                    lora_rank = st.slider(
                        "LoRA Rank (r)",
                        min_value=1,
                        max_value=128,
                        value=16,
                        step=1,
                        help="Rank of the low-rank adapter matrices. Higher = more expressive, more parameters.",
                        key="ms_lora_rank",
                    )
                    lora_alpha = st.slider(
                        "LoRA Alpha (α)",
                        min_value=1,
                        max_value=256,
                        value=32,
                        step=1,
                        help="Scaling factor applied to adapter outputs. Effective scale = α / r.",
                        key="ms_lora_alpha",
                    )
                with lora_col2:
                    target_modules = st.multiselect(
                        "Target Modules",
                        options=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Which weight matrices receive LoRA adapters.",
                        key="ms_lora_modules",
                    )
                    if lora_rank and target_modules:
                        st.metric(
                            "Effective Scale (α / r)",
                            f"{lora_alpha / lora_rank:.2f}×",
                            help="Values > 1 amplify adapter contributions; 1× is neutral.",
                        )
                        st.caption(
                            f"{len(target_modules)} module(s) × rank {lora_rank} "
                            f"— adapters added to every layer of the base model."
                        )

    if training_type is None:
        st.caption("Select a training type to continue.")
        return

    is_lora = ft_method in ("LoRA", "QLoRA")

    # ── Section 2: Training Schedule ─────────────────────────────────────────
    with st.expander("Training Schedule", expanded=True):
        sched_col1, sched_col2 = st.columns(2)
        with sched_col1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=1000,
                value=1,
                step=1,
                help=(
                    "Passes through the training dataset per run. "
                    "More epochs multiply compute cost, shrinking the optimal model size."
                ),
                key="ms_epochs",
            )
            num_hp_trials = st.number_input(
                "HP Tuning Trials",
                min_value=0,
                max_value=500,
                value=0,
                step=1,
                help=(
                    "Hyperparameter search runs before the primary training run. "
                    "Each trial eats into the total budget, leaving less for the main run."
                ),
                key="ms_hp_trials",
            )
        with sched_col2:
            hp_fraction_pct = st.slider(
                "HP Trial Cost (% of full run)",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                disabled=(num_hp_trials == 0),
                help="Fraction of a full training run's compute each HP trial costs.",
                key="ms_hp_fraction_pct",
            )
            hp_fraction = hp_fraction_pct / 100.0
            project_multiplier = 1.0 + num_hp_trials * hp_fraction
            effective_budget = compute_budget / project_multiplier

            st.metric(
                "Project Compute Multiplier",
                f"{project_multiplier:.2f}×",
                help=(
                    f"1 primary run + {num_hp_trials} HP trial(s) × {hp_fraction:.0%} each. "
                    "Higher = less budget per primary run → smaller optimal model."
                ),
            )
            if project_multiplier > 1.0:
                st.metric(
                    "Budget per Primary Run",
                    f"${effective_budget:,.0f}",
                    delta=f"-${compute_budget - effective_budget:,.0f} reserved for HP tuning",
                    delta_color="off",
                    help=f"${compute_budget:,.0f} ÷ {project_multiplier:.2f}× = ${effective_budget:,.0f}.",
                )

    # ── Section 3: Hardware & Architecture ───────────────────────────────────
    with st.expander("Hardware & Architecture", expanded=True):
        arch_col, _ = st.columns([1, 2])
        with arch_col:
            architecture = st.selectbox(
                "Architecture",
                options=ARCHITECTURE_OPTIONS,
                index=0,
                help="Architecture type determines the FLOPs multiplier (Transformer: 6×, CNN: 4×, RNN: 8×, ViT: 6×, Diffusion: 6.5×).",
                key="ms_architecture",
            )

        hw_col1, hw_col2 = st.columns(2)
        hw_config = _render_inline_hardware(hw_col1, hw_col2, epochs=epochs)

    peak_flops_per_gpu = hw_config["peak_flops_per_gpu"]
    mfu = hw_config["mfu"]
    total_gpus = hw_config["total_gpus"]
    num_instances = hw_config["num_instances"]
    hourly_cost = hw_config["hourly_cost"]
    gradient_checkpointing = hw_config["gradient_checkpointing"]

    multiplier = _ARCH_MULTIPLIERS.get(architecture, 6.0)
    if gradient_checkpointing:
        multiplier += 2.0

    # ── Analytic optimal allocation ───────────────────────────────────────────
    # For pre-training / SFT: Chinchilla-optimal D = 20 × N  (Hoffmann et al. 2022)
    # For LoRA / QLoRA:       empirical sweet spot  D ≈ 50 × adapter_params
    #
    # From the cost formula:  budget = cost_per_NxD × N × D = cost_per_NxD × ratio × N²
    # → N_opt = sqrt(budget / (cost_per_NxD × ratio))
    # → D_opt = ratio × N_opt
    LORA_OPTIMAL_RATIO = 50
    optimal_ratio = LORA_OPTIMAL_RATIO if is_lora else CHINCHILLA_OPTIMAL_RATIO

    cost_per_token_param = (
        epochs / (peak_flops_per_gpu * mfu * total_gpus / multiplier * 3600)
    ) * num_instances * hourly_cost

    if cost_per_token_param > 0:
        n_opt = math.sqrt(effective_budget / (cost_per_token_param * optimal_ratio))
        d_opt = optimal_ratio * n_opt
    else:
        n_opt, d_opt = 0.0, 0.0

    opt_flops = calculate_training_flops(
        parameter_count=max(int(n_opt), 1),
        training_tokens=max(int(d_opt), 1),
        architecture=architecture.lower(),
        epochs=epochs,
        gradient_checkpointing=gradient_checkpointing,
    )
    _, _, opt_cost, opt_days = estimate_compute_cost(
        total_flops=opt_flops,
        peak_flops_per_gpu=peak_flops_per_gpu,
        mfu=mfu,
        total_gpus=total_gpus,
        num_instances=num_instances,
        hourly_cost=hourly_cost,
    )
    opt_budget_pct = min(opt_cost / max(effective_budget, 1e-9), 1.0)

    # ── Section 4: Recommendation ────────────────────────────────────────────
    st.divider()
    budget_context = (
        f" (${effective_budget:,.0f} / run after {num_hp_trials} HP trial(s))"
        if project_multiplier > 1.0
        else f" (${effective_budget:,.0f})"
    )
    if is_lora:
        st.subheader("Optimal Adapter Allocation")
        st.caption(
            f"At **{optimal_ratio}× tok/adapter_param** (empirical LoRA sweet spot), "
            f"your per-run budget{budget_context} is best spent training "
            f"**{_fmt_tokens(int(n_opt))} adapter params** on **{_fmt_tokens(int(d_opt))} tokens**."
        )
    else:
        st.subheader("Chinchilla-Optimal Recommendation")
        st.caption(
            f"At **{optimal_ratio}× tok/param** (Hoffmann et al. 2022), your per-run budget"
            f"{budget_context} is most efficiently spent on a "
            f"**{_fmt_tokens(int(n_opt))}** model trained on **{_fmt_tokens(int(d_opt))} tokens**."
        )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric(
            "Optimal Model Size" if not is_lora else "Optimal Adapter Params",
            _fmt_tokens(int(n_opt)),
            help=f"N at the {optimal_ratio}× tok/param optimal ratio for a ${effective_budget:,.0f} per-run budget.",
        )
    with r2:
        st.metric(
            "Required Dataset",
            _fmt_tokens(int(d_opt)),
            help=f"D = {optimal_ratio} × N — tokens needed to hit the optimal ratio.",
        )
    with r3:
        st.metric(
            "Wall-clock Time",
            _format_wall_clock_time(opt_days),
            help="Training time for one primary run at this configuration.",
        )
    with r4:
        total_project_cost = opt_cost * project_multiplier
        st.metric(
            "Total Project Cost",
            f"${total_project_cost:,.0f}",
            help=(
                f"1 primary run (${opt_cost:,.0f}) + "
                f"{num_hp_trials} HP trial(s) × {hp_fraction:.0%} = ${total_project_cost:,.0f} "
                f"of ${compute_budget:,.0f} total budget."
            ),
        )

    st.progress(
        min(total_project_cost / max(compute_budget, 1e-9), 1.0),
        text=(
            f"**${total_project_cost:,.0f}** total project cost of **${compute_budget:,.0f}** budget "
            f"({total_project_cost / compute_budget:.1%})"
        ),
    )

    # ── Section 4: Trade-off Explorer ────────────────────────────────────────
    st.divider()
    if is_lora:
        st.subheader("Adapter Budget Explorer")
        st.caption(
            "Drag to explore how adapter size changes as you trade dataset size against compute. "
            "The star marks the optimal point above."
        )
        n_axis_title = "Max adapter params"
        n_hover = "Max adapter N"
    else:
        st.subheader("Trade-off Explorer")
        st.caption(
            "Drag to see how model size changes across the compute–data trade-off. "
            "The star marks the Chinchilla-optimal point. Moving left = larger model, less data; "
            "right = smaller model, more data."
        )
        n_axis_title = "Max model size (params)"
        n_hover = "Max N"

    # Default to the optimal point; clamp + snap to slider step
    d_opt_log_raw = math.log10(max(d_opt, 1e9)) if d_opt >= 1e9 else 9.0
    d_opt_log = round(max(9.0, min(13.0, d_opt_log_raw)) / 0.05) * 0.05

    log_d = st.slider(
        "Dataset size (tokens)",
        min_value=9.0,
        max_value=13.0,
        value=d_opt_log,
        step=0.05,
        format="10^%.2f",
        help="Drag to explore different dataset sizes. The model size and cost update instantly.",
        key="ms_dataset_slider",
    )
    selected_tokens = int(10 ** log_d)

    solved_n = solve_for_parameter_count(
        compute_budget_usd=effective_budget,
        training_tokens=selected_tokens,
        peak_flops_per_gpu=peak_flops_per_gpu,
        mfu=mfu,
        total_gpus=total_gpus,
        num_instances=num_instances,
        hourly_cost=hourly_cost,
        architecture=architecture.lower(),
        epochs=epochs,
        gradient_checkpointing=gradient_checkpointing,
    )

    total_flops = calculate_training_flops(
        parameter_count=max(solved_n, 1),
        training_tokens=selected_tokens,
        architecture=architecture.lower(),
        epochs=epochs,
        gradient_checkpointing=gradient_checkpointing,
    )
    _, _, compute_cost, wall_clock_days = estimate_compute_cost(
        total_flops=total_flops,
        peak_flops_per_gpu=peak_flops_per_gpu,
        mfu=mfu,
        total_gpus=total_gpus,
        num_instances=num_instances,
        hourly_cost=hourly_cost,
    )

    current_ratio = selected_tokens / max(solved_n, 1)
    budget_used_pct = min(compute_cost / max(effective_budget, 1e-9), 1.0)

    # Deviation from optimal
    if not is_lora:
        _render_chinchilla_assessment(
            ratio=current_ratio,
            optimal_tokens=int(solved_n * CHINCHILLA_OPTIMAL_RATIO),
            fix_hint="Move the slider toward the star to return to Chinchilla-optimal.",
        )
    else:
        if current_ratio < 10:
            st.warning(
                f"**Below LoRA sweet spot.** {current_ratio:.1f} tok/adapter_param "
                f"(target: 10–100×). Move the slider right for more data."
            )
        elif current_ratio <= 100:
            st.success(
                f"**Within LoRA sweet spot.** {current_ratio:.1f} tok/adapter_param — good range."
            )
        else:
            st.info(
                f"**Above LoRA sweet spot.** {current_ratio:.1f} tok/adapter_param. "
                f"Consider increasing LoRA rank to give adapters more capacity."
            )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        label = "Adapter Params" if is_lora else "Model Size"
        st.metric(
            label,
            _fmt_tokens(solved_n),
            help=f"{'Adapter params' if is_lora else 'Largest model'} affordable at this dataset size.",
        )
    with m2:
        st.metric(
            "Dataset Size",
            _fmt_tokens(selected_tokens),
            help="Tokens selected via the slider.",
        )
    with m3:
        st.metric(
            "Wall-clock Time",
            _format_wall_clock_time(wall_clock_days),
            help="Training time for one run at this configuration.",
        )
    with m4:
        tok_per_param_label = "Tokens / Adapter Param" if is_lora else "Tokens / Param"
        ratio_target = f"{optimal_ratio}× target" if is_lora else f"optimal: {CHINCHILLA_OPTIMAL_RATIO}×"
        st.metric(
            tok_per_param_label,
            f"{current_ratio:.1f}×",
            help=f"{current_ratio:.1f}× ({ratio_target}).",
        )

    st.progress(
        budget_used_pct,
        text=(
            f"Per-run budget: **${compute_cost:,.0f}** of **${effective_budget:,.0f}** ({budget_used_pct:.1%})"
            + (
                f" — total project: **${compute_cost * project_multiplier:,.0f}**"
                if project_multiplier > 1.0 else ""
            )
        ),
    )

    # Chart
    d_range = np.logspace(9, 13, 300)
    n_budget = _budget_curve_n(
        d_tokens=d_range,
        budget=effective_budget,
        peak_flops_per_gpu=peak_flops_per_gpu,
        mfu=mfu,
        total_gpus=total_gpus,
        num_instances=num_instances,
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

    # Optimal ratio reference line
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

    # Optimal sweet spot marker
    if 1e9 <= d_opt <= 1e13:
        fig.add_trace(go.Scatter(
            x=[d_opt],
            y=[n_opt],
            name="Optimal sweet spot",
            mode="markers",
            marker=dict(color=CHART_COLORS["warning"], size=14, symbol="star"),
            hovertemplate=(
                f"Optimal<br>D={_fmt_tokens(int(d_opt))}<br>"
                f"N={_fmt_tokens(int(n_opt))}<extra></extra>"
            ),
        ))

    fig.add_trace(go.Scatter(
        x=[selected_tokens],
        y=[solved_n],
        name="Your selection",
        mode="markers",
        marker=dict(color="white", size=12, symbol="circle",
                    line=dict(color=CHART_COLORS["compute"], width=2)),
        hovertemplate=f"Your point<br>D={_fmt_tokens(selected_tokens)}<br>N={_fmt_tokens(solved_n)}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(
            type="log",
            title="Dataset size (tokens)",
            gridcolor="rgba(255,255,255,0.08)",
            tickvals=[1e9, 1e10, 1e11, 1e12, 1e13],
            ticktext=["1B", "10B", "100B", "1T", "10T"],
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


def _budget_curve_n(
    d_tokens: np.ndarray,
    budget: float,
    peak_flops_per_gpu: float,
    mfu: float,
    total_gpus: int,
    num_instances: int,
    hourly_cost: float,
    multiplier: float,
    epochs: int,
) -> np.ndarray:
    """Vectorised: N_max = budget_per_epoch / (multiplier × D)."""
    throughput = peak_flops_per_gpu * mfu * total_gpus / multiplier
    cost_per_token_param = (epochs / (throughput * 3600)) * num_instances * hourly_cost
    return budget / (cost_per_token_param * d_tokens)


def _render_inline_hardware(col1, col2, epochs: int) -> dict:
    """Render compact hardware inputs into two provided columns.

    ``epochs`` is collected upstream in the Training Schedule expander and
    forwarded here so it can be included in the returned config dict.
    """
    from src.app.config import INSTANCE_DISPLAY_NAMES, MIXED_PRECISION_OPTIONS
    from src.cost_modelling.gpu_specs import list_available_instances, get_gpu_instance, peak_flops_for_precision

    def _key(name: str) -> str:
        return f"ms_hw_{name}"

    with col1:
        st.caption("Compute cluster")
        available_instances = list_available_instances()
        display_opts = [INSTANCE_DISPLAY_NAMES.get(i, i) for i in available_instances]
        instance_display = st.selectbox(
            "Instance Type",
            options=display_opts,
            index=0,
            key=_key("instance"),
        )
        instance_type = available_instances[display_opts.index(instance_display)]
        num_instances = st.number_input(
            "Number of Instances",
            min_value=1,
            max_value=512,
            value=1,
            step=1,
            help="Each instance has 8 GPUs.",
            key=_key("num_instances"),
        )
        mixed_precision = st.selectbox(
            "Mixed Precision",
            options=MIXED_PRECISION_OPTIONS,
            index=3,
            key=_key("precision"),
        )

    instance_spec = get_gpu_instance(instance_type)
    total_gpus = instance_spec["gpu_count"] * num_instances
    peak_flops_per_gpu = peak_flops_for_precision(instance_spec, mixed_precision)
    hourly_cost = instance_spec["hourly_cost"]

    with col2:
        st.caption("Performance tuning")
        mfu_pct = st.slider(
            "MFU",
            min_value=5,
            max_value=100,
            value=30,
            step=5,
            format="%d%%",
            help="Model FLOPs Utilization.",
            key=_key("mfu"),
        )
        mfu = mfu_pct / 100.0
        st.caption(f"Typical MFU for {instance_spec['gpu']}: {instance_spec['typical_mfu']:.0%}")
        gradient_checkpointing = st.checkbox(
            "Gradient Checkpointing",
            value=False,
            help="Recomputes activations during the backward pass, increasing FLOPs by ~33% but reducing peak memory.",
            key=_key("grad_ckpt"),
        )

    return {
        "epochs": epochs,
        "gradient_checkpointing": gradient_checkpointing,
        "peak_flops_per_gpu": peak_flops_per_gpu,
        "mfu": mfu,
        "total_gpus": total_gpus,
        "num_instances": num_instances,
        "hourly_cost": hourly_cost,
    }
