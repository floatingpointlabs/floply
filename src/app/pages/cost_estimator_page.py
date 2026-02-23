import streamlit as st
from src.app.components.dataset_dimensions import render_dataset_dimensions
from src.app.components.training_dimensions import render_training_dimensions
from src.app.components.compute_dimensions import render_compute_dimensions
from src.app.components.eval_dimensions import render_eval_dimensions
from src.app.components.output_dimensions import (
    render_cost_summary_numbers,
    render_cost_summary_charts,
    render_final_display,
)
from src.app.helpers import _chinchilla_banner


TRAINING_FLAG = "Dataset Size (TB)"
EVAL_FLAG = "Compute Cost (USD)"
COST_FLAG = "Total Project Cost (USD)"

def render_model_training_cost_estimator_page():
    training_config = {}

    # Dataset Dimensions
    training_config = render_dataset_dimensions(training_config)

    if TRAINING_FLAG in training_config:
        # Training Configuration
        training_config, can_compute = render_training_dimensions(training_config)

    try:
        if can_compute:
            # Dataset size <-> Parameter Count calculation
            _chinchilla_banner(training_config)

            # Compute Configuration
            training_config = render_compute_dimensions(training_config)

    except UnboundLocalError as e:
        st.write("Select a modality & enter a dataset size to continue.")


    if EVAL_FLAG in training_config:
        # Evaluation Configuration
        training_config = render_eval_dimensions(training_config)

    if COST_FLAG in training_config:
        # Cost Summary
        render_cost_summary_numbers(training_config)
        st.divider()
        render_cost_summary_charts(training_config)
        st.divider()
        render_final_display(training_config)