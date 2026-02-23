from typing import Dict, Any

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.app.helpers import _display


def render_cost_summary_numbers(training_config: Dict[str, Any]) -> None:
    """Render the top metric strip: key cost figures and wall-clock time."""
    st.divider()
    st.subheader("Cost Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            label="Total Project Cost",
            value=f"${training_config['Total Project Cost (USD)']:,.2f}",
            help="Compute + dataset storage + checkpoint storage.",
        )
    with m2:
        st.metric(
            label="Total Compute Cost",
            value=f"${training_config['Total Compute Cost (USD)']:,.2f}",
            help="GPU cost across all training runs, HP trials, and ablations.",
        )
    with m3:
        st.metric(
            label="Dataset Storage Cost",
            value=f"${training_config['Dataset Storage Cost (USD)']:,.2f}",
            help="S3 cost to store the training dataset.",
        )
    with m4:
        st.metric(
            label="Checkpoint Storage Cost",
            value=f"${training_config['Checkpoint Storage Cost (USD)']:,.2f}",
            help="S3 cost to store all saved model checkpoints.",
        )
    with m5:
        wall_clock_days = training_config.get("Wall-clock Days", 0)
        wall_clock_hours = wall_clock_days * 24
        if wall_clock_days >= 1:
            wc_label = f"{wall_clock_days:.2f} days"
        elif wall_clock_days >= 1 / 24:
            wc_label = f"{wall_clock_hours:.2f} hrs"
        else:
            wc_label = f"{wall_clock_hours * 60:.1f} min"
        st.metric(label="Wall-clock Time (1 run)", value=wc_label)


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
        donut_colors = ["#4C78A8", "#72B7B2", "#F58518"]

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
        bar_colors = ["#4C78A8", "#E45756", "#54A24B"]

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
