"""Streamlit app configuration."""

import streamlit as st

# Scaling-law ratios are owned by the engine; re-exported here so UI modules can keep
# importing them from config without a second definition drifting out of sync.
from src.cost_modelling.budget_optimizer import (
    CHINCHILLA_OPTIMAL_RATIO,
    LORA_OPTIMAL_RATIO,
)

# Page configuration
PAGE_TITLE = "Floply"
LAYOUT = "wide"

# Chart colour palette (Plotly hex strings)
CHART_COLORS = {
    "compute":    "#4C78A8",
    "storage":    "#72B7B2",
    "checkpoint": "#F58518",
    "warning":    "#E45756",
    "success":    "#54A24B",
}

# Model architecture options (used across training config and solver pages)
ARCHITECTURE_OPTIONS = ["Transformer", "CNN", "RNN", "ViT", "Diffusion"]

# Mixed-precision options are no longer a global list — they come from
# gpu_specs.supported_precisions(), which reflects what the selected GPU
# actually implements. See src/data/gpu_hardware.yaml.

# Common model sizes (parameters) — used for reference tables and quick-select UIs
COMMON_MODEL_SIZES = {
    "100M":  100_000_000,
    "350M":  350_000_000,
    "1B":    1_000_000_000,
    "3B":    3_000_000_000,
    "7B":    7_000_000_000,
    "13B":   13_000_000_000,
    "30B":   30_000_000_000,
    "70B":   70_000_000_000,
    "175B":  175_000_000_000,
    "540B":  540_000_000_000,
}

# Scaling-law tier definitions used by the Minimum Data Calculator.
# Each tier describes a token-per-param (or tok/adapter_param) range:
#   ratio_min / ratio_max  — the zone boundaries
#   ratio_min_chart        — log-scale chart lower bound (must be > 0)
#   color                  — Plotly bar colour (key into CHART_COLORS)
#   label                  — human range string
#   meaning                — one-line explanation
PRE_TRAINING_TIERS = [
    {
        "tier": "Hard floor",
        "ratio_min": 0, "ratio_max": 1, "ratio_min_chart": 0.1,
        "color": CHART_COLORS["warning"],
        "label": "< 1 tok/param",
        "meaning": "Fewer tokens than parameters — training will likely diverge or severely underfit.",
    },
    {
        "tier": "Practical minimum",
        "ratio_min": 1, "ratio_max": 10, "ratio_min_chart": 1,
        "color": CHART_COLORS["checkpoint"],
        "label": "1–10 tok/param",
        "meaning": "Usable but significantly undertrained — expect poor generalisation and high loss.",
    },
    {
        "tier": "Compute-optimal",
        "ratio_min": 10, "ratio_max": 30, "ratio_min_chart": 10,
        "color": CHART_COLORS["success"],
        "label": "10–30 tok/param",
        "meaning": "Chinchilla-optimal zone (Hoffmann et al. 2022) — best loss per FLOP.",
    },
    {
        "tier": "Inference-optimal",
        "ratio_min": 30, "ratio_max": 200, "ratio_min_chart": 30,
        "color": CHART_COLORS["storage"],
        "label": "30–200 tok/param",
        "meaning": "Training smaller models longer reduces inference cost — the LLaMA / Mistral strategy.",
    },
]

FULL_FT_TIERS = [
    {
        "tier": "Too small",
        "ratio_min": 0, "ratio_max": 0.5, "ratio_min_chart": 0.05,
        "color": CHART_COLORS["warning"],
        "label": "< 0.5 tok/param",
        "meaning": "Likely insufficient for meaningful task adaptation — model may not converge on the target task.",
    },
    {
        "tier": "Minimum viable",
        "ratio_min": 0.5, "ratio_max": 1, "ratio_min_chart": 0.5,
        "color": CHART_COLORS["checkpoint"],
        "label": "0.5–1 tok/param",
        "meaning": "Lower bound for stable full-parameter fine-tuning — expect some instability.",
    },
    {
        "tier": "Sweet spot",
        "ratio_min": 1, "ratio_max": 5, "ratio_min_chart": 1,
        "color": CHART_COLORS["success"],
        "label": "1–5 tok/param",
        "meaning": "Standard supervised fine-tuning range for large models — stable and effective.",
    },
    {
        "tier": "Forgetting risk",
        "ratio_min": 5, "ratio_max": 20, "ratio_min_chart": 5,
        "color": CHART_COLORS["storage"],
        "label": "5–20 tok/param",
        "meaning": "Generous dataset — effective, but monitor for catastrophic forgetting of base model capabilities.",
    },
]

LORA_TIERS = [
    {
        "tier": "Too small",
        "ratio_min": 0, "ratio_max": 10, "ratio_min_chart": 1,
        "color": CHART_COLORS["warning"],
        "label": "< 10 tok/adapter_param",
        "meaning": "Adapter likely won't converge reliably — insufficient signal for low-rank matrices.",
    },
    {
        "tier": "Minimum viable",
        "ratio_min": 10, "ratio_max": 50, "ratio_min_chart": 10,
        "color": CHART_COLORS["checkpoint"],
        "label": "10–50 tok/adapter_param",
        "meaning": "Lower bound of the convergence zone — reliable but not yet in the ideal range.",
    },
    {
        "tier": "Sweet spot",
        "ratio_min": 50, "ratio_max": 100, "ratio_min_chart": 50,
        "color": CHART_COLORS["success"],
        "label": "50–100 tok/adapter_param",
        "meaning": "Practical sweet spot for task adaptation — strong convergence with low forgetting risk.",
    },
    {
        "tier": "Consider full FT",
        "ratio_min": 100, "ratio_max": 1000, "ratio_min_chart": 100,
        "color": CHART_COLORS["storage"],
        "label": "100–1000 tok/adapter_param",
        "meaning": "Large dataset relative to adapter size — consider increasing LoRA rank or switching to full fine-tuning.",
    },
]


_TAB_CSS = """
<style>
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    gap: 16px;
}
.stTabs [data-baseweb="tab"] {
    white-space: normal;
    text-align: center;
}
</style>
"""



def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🧮",
        layout=LAYOUT,
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "Floply — ML Training Cost Estimator",
        },
    )
