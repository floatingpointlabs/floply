"""Streamlit rendering helpers.

The formatting and scaling-law logic that used to live here is now pure and lives in
src/cost_modelling/{formatting,scaling_laws}.py. What remains is the thin Streamlit
layer: widgets, and turning an Assessment into the matching st.* banner.

The underscore-prefixed names are kept as aliases so the existing page/component
imports keep working unchanged.
"""

import streamlit as st

from src.cost_modelling.formatting import (
    UNIT_MULTIPLIERS as _UNIT_MULTIPLIERS,
    display_value as _display,
    fmt_samples as _fmt_samples,
    fmt_tokens as _fmt_tokens,
    format_gpu_hours as _format_gpu_hours,
    format_wall_clock_time as _format_wall_clock_time,
)
from src.cost_modelling.model_loader import load_models
from src.cost_modelling.scaling_laws import (
    Assessment,
    assess_chinchilla_ratio,
    assess_training_config,
)

__all__ = [
    "_cached_models",
    "_chinchilla_banner",
    "_display",
    "_fmt_samples",
    "_fmt_tokens",
    "_format_gpu_hours",
    "_format_wall_clock_time",
    "_render_assessment",
    "_render_chinchilla_assessment",
    "_scaled_number_input",
    "_UNIT_MULTIPLIERS",
]


@st.cache_data
def _cached_models():
    return load_models()


def _render_assessment(assessment: Assessment | None) -> None:
    """Render an Assessment via the matching Streamlit banner, or nothing if None."""
    if assessment is None:
        return
    getattr(st, assessment.level)(assessment.message)


def _chinchilla_banner(training_config: dict) -> None:
    """Render a scaling-law health banner above the Compute Config."""
    _render_assessment(
        assess_training_config(
            total_tokens=training_config.get("Total tokens", 0),
            ft_method=training_config.get("Fine-Tuning Method"),   # None for pre-training
            base_params=training_config.get("Base Model Params", 0),
            pre_params=training_config.get("Parameter Count", 0),
            adapter_params=training_config.get("Trainable Parameters", 0),
        )
    )


def _render_chinchilla_assessment(
    ratio: float, optimal_tokens: int, fix_hint: str = ""
) -> None:
    """Render a pre-training Chinchilla scaling-law assessment banner."""
    _render_assessment(assess_chinchilla_ratio(ratio, optimal_tokens, fix_hint))


def _scaled_number_input(
    label: str,
    units: list[str],
    default_value: float,
    default_unit: str,
    min_value: float = 0.001,
    max_value: float = 9_999.0,
    help: str = "",
    key: str | None = None,
) -> int | None:
    """Render a float input + unit selectbox that returns a plain integer.

    Example: value=7.0, unit="B"  →  7,000,000,000
    """
    val_col, unit_col = st.columns([3, 1])
    with val_col:
        value = st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=0.1,
            format="%.3f",
            help=help,
            key=f"{key}_val" if key else None,
        )
    with unit_col:
        unit = st.selectbox(
            " ",
            options=units,
            index=units.index(default_unit),
            key=f"{key}_unit" if key else None,
        )
    if value is None:
        return None
    result = int(value * _UNIT_MULTIPLIERS[unit])
    st.caption(f"{result:,}")
    return result
