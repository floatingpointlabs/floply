"""Components package for Streamlit app."""

from src.app.components.input_forms import (
    render_model_config_form,
    render_training_config_form,
    render_project_config_form,
)
from src.app.components.results import (
    render_cost_summary,
    render_cost_breakdown_table,
    render_efficiency_metrics,
    render_project_cost_summary,
    render_comparison_table,
    render_scenario_cards,
    render_export_buttons,
)
from src.app.components.charts import (
    create_cost_breakdown_pie_chart,
    create_project_cost_breakdown_chart,
    create_scenario_comparison_chart,
    create_duration_comparison_chart,
    create_cost_vs_model_size_chart,
    create_gpu_utilization_timeline,
    create_sensitivity_analysis_chart,
    create_cost_efficiency_comparison,
)

__all__ = [
    "render_model_config_form",
    "render_training_config_form",
    "render_project_config_form",
    "render_cost_summary",
    "render_cost_breakdown_table",
    "render_efficiency_metrics",
    "render_project_cost_summary",
    "render_comparison_table",
    "render_scenario_cards",
    "render_export_buttons",
    "create_cost_breakdown_pie_chart",
    "create_project_cost_breakdown_chart",
    "create_scenario_comparison_chart",
    "create_duration_comparison_chart",
    "create_cost_vs_model_size_chart",
    "create_gpu_utilization_timeline",
    "create_sensitivity_analysis_chart",
    "create_cost_efficiency_comparison",
]
