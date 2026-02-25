"""Shared rendering logic for pages that follow the standard training-config flow.

The Estimate page collects inputs (dataset → training → compute → eval) and then calls
a caller-supplied summary output function. This module centralises that flow so page
files remain thin wrappers.
"""

from typing import Callable

import streamlit as st

from src.app.components.dataset_dimensions import render_dataset_dimensions
from src.app.components.training_dimensions import render_training_dimensions
from src.app.components.compute_dimensions import render_compute_dimensions
from src.app.components.eval_dimensions import render_eval_dimensions
from src.app.components.output_dimensions import render_cost_summary_charts, render_final_display
from src.app.helpers import _chinchilla_banner

# Keys used to gate each successive section of the form
TRAINING_FLAG = "Dataset Size (TB)"
EVAL_FLAG = "Compute Cost (USD)"
COST_FLAG = "Total Project Cost (USD)"


def _render_standard_training_page(output_fn: Callable, subtitle: str = "") -> None:
    """Render the full training-config flow with a caller-supplied summary section.

    Args:
        output_fn: Function that renders the primary summary metrics strip.
                   Receives ``training_config`` as its only argument.
                   Example: ``render_estimate_summary_numbers``.
        subtitle:  Optional descriptive subtitle shown below the page heading.
    """
    if subtitle:
        st.markdown(
            f"<p style='text-align:center;color:#888;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )

    training_config: dict = {}

    training_config = render_dataset_dimensions(training_config)

    if TRAINING_FLAG in training_config:
        training_config, can_compute = render_training_dimensions(training_config)

    try:
        if can_compute:
            _chinchilla_banner(training_config)
            training_config = render_compute_dimensions(training_config)
    except UnboundLocalError:
        st.write("Select a modality & enter a dataset size to continue.")

    if EVAL_FLAG in training_config:
        training_config = render_eval_dimensions(training_config)

    if COST_FLAG in training_config:
        output_fn(training_config)
        st.divider()
        render_cost_summary_charts(training_config)
        st.divider()
        render_final_display(training_config)
