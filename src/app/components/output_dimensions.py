from typing import Dict, Any

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.app.config import COMMON_MODEL_SIZES, CHART_COLORS
from src.app.helpers import (
    _display,
    _fmt_tokens,
    _format_wall_clock_time,
    _format_gpu_hours,
    _render_chinchilla_assessment,
)


def render_estimate_summary_numbers(training_config: Dict[str, Any]) -> None:
    """Render the combined time + cost summary metric strip for the Estimate tab.

    Two rows: time-focused metrics on top, cost breakdown below.
    """
    st.divider()
    st.subheader("Summary")

    # Row 1 — time
    t1, t2, t3 = st.columns(3)
    with t1:
        st.metric(
            label="Wall-clock Time (1 run)",
            value=_format_wall_clock_time(training_config.get("Wall-clock Days", 0)),
            help="Real elapsed time for a single training run on the selected cluster.",
        )
    with t2:
        st.metric(
            label="GPU-hours (1 run)",
            value=f"{_format_gpu_hours(training_config.get('GPU-hours', 0))} GPU-hrs",
            help="Total GPU-hours consumed across all devices for one training run.",
        )
    with t3:
        st.metric(
            label="Total FLOPs",
            value=f"{training_config.get('Total FLOPs', 0):.2e}",
            help="Total floating-point operations for the full training run.",
        )

    # Row 2 — cost
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Compute Cost (1 run)",
            value=f"${training_config.get('Compute Cost (USD)', 0):,.2f}",
            help="GPU cost for a single training run.",
        )
    with c2:
        st.metric(
            label="Dataset Storage Cost",
            value=f"${training_config['Dataset Storage Cost (USD)']:,.2f}",
            help="S3 cost to store the training dataset.",
        )
    with c3:
        st.metric(
            label="Checkpoint Storage Cost",
            value=f"${training_config['Checkpoint Storage Cost (USD)']:,.2f}",
            help="S3 cost to store all saved model checkpoints.",
        )
    with c4:
        st.metric(
            label="Total Project Cost",
            value=f"${training_config['Total Project Cost (USD)']:,.2f}",
            help="Compute + dataset storage + checkpoint storage across all runs.",
        )


def render_cost_summary_charts(training_config: Dict[str, Any]) -> None:
    """Render cost distribution donut and compute-by-run-type stacked bar."""
    total_compute = training_config["Total Compute Cost (USD)"]
    dataset_store = training_config["Dataset Storage Cost (USD)"]
    ckpt_store    = training_config["Checkpoint Storage Cost (USD)"]
    total_project = training_config["Total Project Cost (USD)"]

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.subheader("Cost Distribution")
        donut_labels = ["Compute", "Dataset Storage", "Checkpoint Storage"]
        donut_values = [total_compute, dataset_store, ckpt_store]
        donut_colors = [
            CHART_COLORS["compute"],
            CHART_COLORS["storage"],
            CHART_COLORS["checkpoint"],
        ]

        fig_donut = go.Figure(go.Pie(
            labels=donut_labels,
            values=donut_values,
            hole=0.55,
            marker=dict(colors=donut_colors, line=dict(color="#0e1117", width=2)),
            textinfo="label+percent",
            hovertemplate="%{label}<br>$%{value:,.2f}<br>%{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"${total_project:,.0f}",
            x=0.5, y=0.52, showarrow=False,
            font=dict(size=20, color="white", family="sans-serif"),
        )
        fig_donut.add_annotation(
            text="total",
            x=0.5, y=0.42, showarrow=False,
            font=dict(size=12, color="#aaaaaa", family="sans-serif"),
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        st.subheader("Compute Cost by Run Type")
        single_run_cost = training_config.get("Compute Cost (USD)", 0)
        num_full  = training_config.get("Full Training Runs", 1)
        num_hp    = training_config.get("HP Tuning Trials", 0)
        hp_frac   = training_config.get("HP Run Fraction", 0)
        num_abl   = training_config.get("Ablation Studies", 0)
        abl_frac  = training_config.get("Ablation Run Fraction", 0)

        full_cost = single_run_cost * num_full
        hp_cost   = single_run_cost * hp_frac * num_hp
        abl_cost  = single_run_cost * abl_frac * num_abl

        bar_cats   = ["Full Runs", "HP Trials", "Ablations"]
        bar_values = [full_cost, hp_cost, abl_cost]
        bar_colors = [
            CHART_COLORS["compute"],
            CHART_COLORS["warning"],
            CHART_COLORS["success"],
        ]

        fig_bar = go.Figure()
        for cat, val, col in zip(bar_cats, bar_values, bar_colors):
            fig_bar.add_trace(go.Bar(
                name=cat,
                x=["Compute"],
                y=[val],
                marker_color=col,
                text=f"${val:,.0f}",
                textposition="inside",
                hovertemplate=f"{cat}<br>${val:,.2f}<extra></extra>",
            ))
        fig_bar.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(
                title="USD",
                gridcolor="rgba(255,255,255,0.08)",
                zeroline=False,
                tickprefix="$",
            ),
            margin=dict(t=30, b=10, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def render_final_display(training_config: Dict[str, Any]) -> None:
    """Render the full run-configuration summary table."""
    st.subheader("Run Configuration")

    summary_rows = [
        # Dataset
        ("Dataset", "Modality", training_config.get("Modality", "-")),
        ("Dataset", "Dataset Size (samples)", training_config.get("Total tokens", 0) // max(training_config.get("Tokens per sample", 1), 1)),
        ("Dataset", "Tokens per Sample", training_config.get("Tokens per sample", "-")),
        ("Dataset", "Total Tokens", training_config.get("Total tokens", "-")),
        ("Dataset", "Estimated Size", training_config.get("Dataset Size (TB)", "-")),
        ("Dataset", "Storage Duration", f"{training_config.get('Storage Duration (months)', '-')} months"),
        # Model
        ("Model", "Training Type", training_config.get("Training Type", "-")),
        ("Model", "Model Type / Base", training_config.get("Model Type", training_config.get("Base Model", "-"))),
        ("Model", "Parameter Count", training_config.get("Parameter Count", training_config.get("Base Model Params", "-"))),
        ("Model", "Fine-Tuning Method", training_config.get("Fine-Tuning Method", "N/A")),
        ("Model", "Trainable Parameters", training_config.get("Trainable Parameters", training_config.get("Parameter Count", "-"))),
        # Training
        ("Training", "Epochs", training_config.get("Epochs", "-")),
        ("Training", "Batch Size", training_config.get("Batch Size", "-")),
        ("Training", "Effective Batch Size", training_config.get("Effective Batch Size", "-")),
        ("Training", "Gradient Checkpointing", training_config.get("Gradient Checkpointing", "-")),
        ("Training", "Mixed Precision", training_config.get("Mixed Precision", "-")),
        # Compute
        ("Compute", "Instance Type", training_config.get("Instance Type", "-")),
        ("Compute", "Num Instances", training_config.get("Num Instances", "-")),
        ("Compute", "Total GPUs", training_config.get("Total GPUs", "-")),
        ("Compute", "MFU", f"{training_config.get('MFU', 0):.0%}"),
        ("Compute", "GPU-hours (1 run)", training_config.get("GPU-hours", "-")),
        ("Compute", "Wall-clock (1 run)", f"{training_config.get('Wall-clock Days', 0):.2f} days"),
        ("Compute", "Total FLOPs", training_config.get("Total FLOPs", "-")),
        ("Compute", "Est. Memory / GPU", f"{training_config.get('Est. Memory per GPU (GB)', 0):.1f} GB"),
        # Experiment
        ("Experiment", "Full Training Runs", training_config.get("Full Training Runs", "-")),
        ("Experiment", "HP Tuning Trials", training_config.get("HP Tuning Trials", "-")),
        ("Experiment", "HP Run Fraction", f"{training_config.get('HP Run Fraction', 0):.0%}"),
        ("Experiment", "Ablation Studies", training_config.get("Ablation Studies", "-")),
        ("Experiment", "Ablation Run Fraction", f"{training_config.get('Ablation Run Fraction', 0):.0%}"),
        ("Experiment", "Checkpoints per Run", training_config.get("Num Checkpoints", "-")),
        ("Experiment", "Checkpoint Storage", f"{training_config.get('Checkpoint Storage (TB)', 0):.4f} TB"),
        ("Experiment", "S3 Storage Class", training_config.get("S3 Storage Class", "-")),
        ("Experiment", "Dataset Storage Cost", f"${training_config.get('Dataset Storage Cost (USD)', 0):,.2f}"),
        ("Experiment", "Total Experiment Runs", training_config.get("Total Experiment Runs", "-")),
        ("Experiment", "Checkpoint Storage Cost", f"${training_config.get('Checkpoint Storage Cost (USD)', 0):,.2f}"),
        ("Experiment", "Total Compute Cost", f"${training_config.get('Total Compute Cost (USD)', 0):,.2f}"),
        ("Experiment", "Total Project Cost", f"${training_config.get('Total Project Cost (USD)', 0):,.2f}"),
    ]

    df = pd.DataFrame(summary_rows, columns=["Category", "Parameter", "Value"])
    df["Value"] = df["Value"].apply(lambda v: _display(v) if not isinstance(v, str) else v)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="medium"),
        },
    )



def render_model_size_results(training_config: Dict[str, Any]) -> None:
    """Render results for the Max Model Size solver page."""
    parameter_count = training_config.get("Solved Parameter Count", 0)
    total_tokens = training_config.get("Total tokens", 0)
    wall_clock_days = training_config.get("Solved Wall-clock Days", 0)
    gpu_hours = training_config.get("Solved GPU-hours", 0)
    total_flops = training_config.get("Solved Total FLOPs", 0)
    budget = training_config.get("Compute Budget (USD)", 0)

    st.divider()
    st.subheader("Max Model Size")

    if parameter_count <= 0:
        st.error("Could not solve — check that budget, tokens, and hardware are all set.")
        return

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Max Model Size",
            value=_fmt_tokens(parameter_count),
            help=f"Largest model trainable on {_fmt_tokens(total_tokens)} tokens within your ${budget:,.0f} budget.",
        )
    with m2:
        st.metric(
            label="Wall-clock Time",
            value=_format_wall_clock_time(wall_clock_days),
            help="Estimated wall-clock time for a single training run at this model size.",
        )
    with m3:
        st.metric(
            label="GPU-hours",
            value=_format_gpu_hours(gpu_hours),
            help="GPU-hours consumed for one run at this model size.",
        )
    with m4:
        st.metric(
            label="Total FLOPs",
            value=f"{total_flops:.2e}",
            help="Total FLOPs for training this model on the full dataset.",
        )

    if parameter_count > 0:
        ratio = total_tokens / parameter_count
        optimal_tokens = parameter_count * 20
        _render_chinchilla_assessment(
            ratio=ratio,
            optimal_tokens=optimal_tokens,
            fix_hint="You could use a smaller model or increase your dataset.",
        )

    _render_nearest_model_sizes(parameter_count)



def _render_nearest_model_sizes(parameter_count: int) -> None:
    """Show a reference table of where the solved parameter count falls among well-known model sizes."""
    st.divider()
    st.caption("**Reference model sizes**")

    rows = []
    solved_inserted = False
    for name, size in COMMON_MODEL_SIZES.items():
        if not solved_inserted and parameter_count < size:
            rows.append({
                "Model Size": f"► {_fmt_tokens(parameter_count)} (your budget)",
                "Parameters": f"{parameter_count:,}",
                "vs. Solved": "← solved",
            })
            solved_inserted = True
        rows.append({
            "Model Size": name,
            "Parameters": f"{size:,}",
            "vs. Solved": f"{parameter_count / size:.2f}×" if size > 0 else "-",
        })

    if not solved_inserted:
        rows.append({
            "Model Size": f"► {_fmt_tokens(parameter_count)} (your budget)",
            "Parameters": f"{parameter_count:,}",
            "vs. Solved": "← solved",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
