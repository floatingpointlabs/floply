"""Chart components for the Streamlit app."""

import plotly.graph_objects as go
from typing import List, Dict, Any
from src.cost_modelling.calculator import CostBreakdown

# Use a template that works well in both light and dark modes
PLOTLY_TEMPLATE = "plotly"  # or "plotly_dark" for dark mode
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
}


def create_cost_breakdown_pie_chart(cost_breakdown: CostBreakdown) -> go.Figure:
    """Create pie chart showing cost breakdown.
    
    Args:
        cost_breakdown: Cost breakdown data
        
    Returns:
        Plotly figure
    """
    # For now, compute cost is dominant; in future can add storage, data transfer
    labels = ["Compute Cost"]
    values = [cost_breakdown.compute_cost]
    
    if cost_breakdown.storage_cost > 0:
        labels.append("Storage Cost")
        values.append(cost_breakdown.storage_cost)
    
    if cost_breakdown.data_transfer_cost > 0:
        labels.append("Data Transfer Cost")
        values.append(cost_breakdown.data_transfer_cost)
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(colors=['#FF4B4B', '#4B8BFF', '#FFB84B'])
    )])
    
    fig.update_layout(
        title="Cost Breakdown by Component",
        height=400,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_project_cost_breakdown_chart(project_cost: Dict[str, Any]) -> go.Figure:
    """Create bar chart showing project cost breakdown.
    
    Args:
        project_cost: Project cost dictionary
        
    Returns:
        Plotly figure
    """
    components = [
        "Base Training",
        "Hyperparameter\nTuning",
        "Ablation\nStudies",
        "Development",
        "Storage"
    ]
    
    costs = [
        project_cost["base_training_cost"],
        project_cost["hyperparameter_cost"],
        project_cost["ablation_cost"],
        project_cost["dev_cost"],
        project_cost["storage_cost"],
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=components,
            y=costs,
            marker_color=['#FF4B4B', '#FF8B4B', '#FFBB4B', '#4B8BFF', '#4BBBFF'],
            text=[f"${c:,.0f}" for c in costs],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Project Cost Breakdown",
        xaxis_title="Component",
        yaxis_title="Cost (USD)",
        height=400,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_scenario_comparison_chart(scenarios: List[Dict]) -> go.Figure:
    """Create bar chart comparing scenario costs.
    
    Args:
        scenarios: List of scenario dictionaries
        
    Returns:
        Plotly figure
    """
    names = [s["name"] for s in scenarios]
    costs = [s["cost_numeric"] for s in scenarios]
    
    # Color code by cost level
    colors = []
    for cost in costs:
        if cost < 1000:
            colors.append('#44AA44')  # Green for low cost
        elif cost < 10000:
            colors.append('#FFBB44')  # Yellow for medium
        else:
            colors.append('#FF4444')  # Red for high
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=costs,
            marker_color=colors,
            text=[f"${c:,.0f}" for c in costs],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Scenario Cost Comparison",
        xaxis_title="Scenario",
        yaxis_title="Cost (USD)",
        height=500,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_duration_comparison_chart(scenarios: List[Dict]) -> go.Figure:
    """Create bar chart comparing scenario durations.
    
    Args:
        scenarios: List of scenario dictionaries
        
    Returns:
        Plotly figure
    """
    names = [s["name"] for s in scenarios]
    durations = [float(s["duration_days"]) for s in scenarios]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=durations,
            marker_color='#4B8BFF',
            text=[f"{d:.1f}d" for d in durations],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Training Duration Comparison",
        xaxis_title="Scenario",
        yaxis_title="Duration (days)",
        height=400,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_cost_vs_model_size_chart(scenarios: List[Dict]) -> go.Figure:
    """Create scatter plot of cost vs model size.
    
    Args:
        scenarios: List of scenario dictionaries
        
    Returns:
        Plotly figure
    """
    # Extract numeric model sizes
    model_sizes = []
    costs = []
    names = []
    
    for s in scenarios:
        size_str = s["model_size"]
        if "B" in size_str:
            size = float(size_str.replace("B", "")) * 1e9
        elif "M" in size_str:
            size = float(size_str.replace("M", "")) * 1e6
        else:
            continue
        
        model_sizes.append(size / 1e9)  # Convert to billions for display
        costs.append(s["cost_numeric"])
        names.append(s["name"])
    
    fig = go.Figure(data=[
        go.Scatter(
            x=model_sizes,
            y=costs,
            mode='markers+text',
            marker=dict(size=12, color='#FF4B4B'),
            text=names,
            textposition="top center",
        )
    ])
    
    fig.update_layout(
        title="Cost vs Model Size",
        xaxis_title="Model Size (Billions of Parameters)",
        yaxis_title="Cost (USD)",
        height=400,
        xaxis_type="log",
        yaxis_type="log",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_gpu_utilization_timeline(cost_breakdown: CostBreakdown) -> go.Figure:
    """Create timeline showing GPU utilization.
    
    Args:
        cost_breakdown: Cost breakdown data
        
    Returns:
        Plotly figure
    """
    # Create a simple timeline visualization
    hours = cost_breakdown.wall_clock_hours
    num_gpus = cost_breakdown.total_gpus
    
    # Create hourly breakdown
    timeline_hours = min(int(hours) + 1, 100)  # Limit to 100 points
    step = hours / timeline_hours
    
    time_points = [i * step for i in range(timeline_hours)]
    utilization = [cost_breakdown.mfu * 100] * timeline_hours
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=utilization,
        mode='lines',
        name='MFU',
        line=dict(color='#FF4B4B', width=2),
        fill='tozeroy',
    ))
    
    fig.update_layout(
        title=f"GPU Utilization Timeline ({num_gpus} GPUs)",
        xaxis_title="Training Time (hours)",
        yaxis_title="Model FLOPs Utilization (%)",
        height=300,
        yaxis_range=[0, 100],
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_sensitivity_analysis_chart(
    base_cost: float,
    parameter_name: str,
    parameter_values: List[float],
    costs: List[float]
) -> go.Figure:
    """Create sensitivity analysis chart.
    
    Args:
        base_cost: Baseline cost
        parameter_name: Name of parameter being varied
        parameter_values: List of parameter values
        costs: List of corresponding costs
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=parameter_values,
        y=costs,
        mode='lines+markers',
        line=dict(color='#FF4B4B', width=2),
        marker=dict(size=8),
    ))
    
    # Add baseline reference line
    fig.add_hline(
        y=base_cost,
        line_dash="dash",
        line_color="gray",
        annotation_text="Baseline",
    )
    
    fig.update_layout(
        title=f"Sensitivity Analysis: {parameter_name}",
        xaxis_title=parameter_name,
        yaxis_title="Cost (USD)",
        height=400,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_cost_efficiency_comparison(scenarios: List[Dict]) -> go.Figure:
    """Create chart comparing cost efficiency across scenarios.
    
    Args:
        scenarios: List of scenario dictionaries
        
    Returns:
        Plotly figure
    """
    names = [s["name"] for s in scenarios]
    costs = [s["cost_numeric"] for s in scenarios]
    
    # Calculate parameters (extract from model_size)
    params = []
    for s in scenarios:
        size_str = s["model_size"]
        if "B" in size_str:
            params.append(float(size_str.replace("B", "")))
        elif "M" in size_str:
            params.append(float(size_str.replace("M", "")) / 1000)
        else:
            params.append(1.0)
    
    # Cost per billion parameters
    efficiency = [c / p if p > 0 else 0 for c, p in zip(costs, params)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=efficiency,
            marker_color='#4B8BFF',
            text=[f"${e:,.0f}/B" for e in efficiency],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Cost Efficiency ($ per Billion Parameters)",
        xaxis_title="Scenario",
        yaxis_title="Cost per Billion Parameters (USD)",
        height=400,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig
