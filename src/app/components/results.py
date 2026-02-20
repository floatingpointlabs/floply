"""Results display components for the Streamlit app."""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.cost_modelling.calculator import CostBreakdown
from src.cost_modelling.utils import format_currency, format_number, format_duration


def render_cost_summary(cost_breakdown: CostBreakdown):
    """Render cost summary with key metrics.
    
    Args:
        cost_breakdown: Calculated cost breakdown
    """
    st.subheader("Cost Summary")
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Cost",
            value=format_currency(cost_breakdown.total_cost),
            help="Total estimated cost for this training run"
        )
    
    with col2:
        st.metric(
            label="Duration",
            value=format_duration(cost_breakdown.wall_clock_hours),
            help="Wall-clock time for training completion"
        )
    
    with col3:
        st.metric(
            label="GPU Hours",
            value=format_number(cost_breakdown.gpu_hours),
            help="Total GPU-hours consumed"
        )
    
    with col4:
        st.metric(
            label="MFU",
            value=f"{cost_breakdown.mfu * 100:.1f}%",
            help="Model FLOPs Utilization - efficiency of GPU usage"
        )


def render_cost_breakdown_table(cost_breakdown: CostBreakdown):
    """Render detailed cost breakdown table.
    
    Args:
        cost_breakdown: Calculated cost breakdown
    """
    st.subheader("Detailed Breakdown")
    
    # Create breakdown data
    breakdown_data = {
        "Metric": [
            "Model Size",
            "Training Data",
            "Total FLOPs",
            "Instance Type",
            "Number of Instances",
            "Total GPUs",
            "GPU Type",
            "Wall-Clock Time",
            "GPU-Hours",
            "Compute Cost",
            "Storage Cost",
            "Total Cost",
        ],
        "Value": [
            f"{cost_breakdown.model_params / 1e9:.2f}B parameters" if cost_breakdown.model_params >= 1e9 else f"{cost_breakdown.model_params / 1e6:.0f}M parameters",
            format_number(cost_breakdown.training_tokens, " tokens"),
            f"{cost_breakdown.total_flops:.2e} FLOPs",
            cost_breakdown.instance_type,
            str(cost_breakdown.num_instances),
            str(cost_breakdown.total_gpus),
            cost_breakdown.gpu_type,
            f"{cost_breakdown.wall_clock_days:.1f} days ({cost_breakdown.wall_clock_hours:.1f} hours)",
            f"{cost_breakdown.gpu_hours:,.0f} hours",
            format_currency(cost_breakdown.compute_cost),
            format_currency(cost_breakdown.storage_cost),
            format_currency(cost_breakdown.total_cost),
        ]
    }
    
    df = pd.DataFrame(breakdown_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_efficiency_metrics(cost_breakdown: CostBreakdown):
    """Render efficiency metrics.
    
    Args:
        cost_breakdown: Calculated cost breakdown
    """
    st.subheader("Efficiency Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Cost per TFLOP",
            value=f"${cost_breakdown.cost_per_tflop:.4f}",
            help="Cost efficiency per trillion FLOPs"
        )
    
    with col2:
        st.metric(
            label="Cost per Million Parameters",
            value=f"${cost_breakdown.cost_per_million_params:.4f}",
            help="Training cost per million model parameters"
        )
    
    with col3:
        st.metric(
            label="Cost per Billion Tokens",
            value=f"${cost_breakdown.cost_per_billion_tokens:.2f}",
            help="Training cost per billion tokens"
        )


def render_project_cost_summary(project_cost: Dict[str, Any]):
    """Render project-level cost summary.
    
    Args:
        project_cost: Project cost breakdown dictionary
    """
    st.subheader("Project Cost Summary")
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Project Cost",
            value=format_currency(project_cost["total_cost"]),
        )
    
    with col2:
        st.metric(
            label="Total Compute Cost",
            value=format_currency(project_cost["compute_cost"]),
        )
    
    with col3:
        st.metric(
            label="Total GPU-Hours",
            value=format_number(project_cost["total_gpu_hours"]),
        )
    
    # Detailed breakdown
    st.subheader("Cost Components")
    
    components_data = {
        "Component": [
            "Base Training Runs",
            "Hyperparameter Tuning",
            "Ablation Studies",
            "Development/Debug",
            "Data Storage",
        ],
        "Cost": [
            format_currency(project_cost["base_training_cost"]),
            format_currency(project_cost["hyperparameter_cost"]),
            format_currency(project_cost["ablation_cost"]),
            format_currency(project_cost["dev_cost"]),
            format_currency(project_cost["storage_cost"]),
        ],
        "Details": [
            f"{project_cost['num_training_runs']} full training runs",
            f"{project_cost['num_hyperparameter_trials']} trials (30% of full training each)",
            f"{project_cost['num_ablations']} studies (50% of full training each)",
            "Development instance hours",
            f"{project_cost['dataset_size_tb']} TB for {project_cost['storage_months']} months",
        ]
    }
    
    df = pd.DataFrame(components_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_comparison_table(scenarios: list):
    """Render scenario comparison table.
    
    Args:
        scenarios: List of scenario dictionaries
    """
    st.subheader("Scenario Comparison")
    
    # Create comparison dataframe
    comparison_data = {
        "Scenario": [s["name"] for s in scenarios],
        "Model Size": [s["model_size"] for s in scenarios],
        "Training Data": [s["training_tokens"] for s in scenarios],
        "Instance": [s["instance"] for s in scenarios],
        "Duration": [s["duration_days"] + " days" for s in scenarios],
        "GPU Hours": [s["gpu_hours"] for s in scenarios],
        "Total Cost": [s["total_cost"] for s in scenarios],
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Style the dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
    
    return df


def render_scenario_cards(scenarios: list):
    """Render scenario cards with details.
    
    Args:
        scenarios: List of scenario dictionaries
    """
    for scenario in scenarios:
        with st.expander(f"📋 {scenario['name']}", expanded=False):
            st.markdown(f"**Goal:** {scenario['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Size", scenario['model_size'])
                st.metric("Training Data", scenario['training_tokens'])
                st.metric("Instance", scenario['instance'])
            
            with col2:
                st.metric("Duration", scenario['duration_days'] + " days")
                st.metric("GPU Hours", scenario['gpu_hours'])
                st.metric("Total Cost", scenario['total_cost'])
            
            st.divider()


def render_export_buttons(df: pd.DataFrame, filename: str = "cost_estimate"):
    """Render export buttons for data download.
    
    Args:
        df: DataFrame to export
        filename: Base filename for download
    """
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"{filename}.csv",
            mime="text/csv",
        )
    
    with col2:
        # Excel export (requires openpyxl)
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Cost Estimate')
            
            st.download_button(
                label="Download as Excel",
                data=buffer.getvalue(),
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            st.info("Install openpyxl to enable Excel export")
