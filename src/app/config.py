"""Streamlit app configuration."""

import streamlit as st

# Page configuration
PAGE_TITLE = "Floply — ML Training Cost Estimator"
PAGE_ICON = "💰"
LAYOUT = "wide"

# App theme colors
PRIMARY_COLOR = "#FF4B4B"
BACKGROUND_COLOR = "#FFFFFF"
SECONDARY_BACKGROUND_COLOR = "#F0F2F6"

# Instance type display names
INSTANCE_DISPLAY_NAMES = {
    "p4d.24xlarge": "p4d.24xlarge (8x A100 40GB) - $26.87/hr",
    "p4de.24xlarge": "p4de.24xlarge (8x A100 80GB) - $32.77/hr",
    "p5.48xlarge": "p5.48xlarge (8x H100 80GB) - $66.64/hr",
    "p3.16xlarge": "p3.16xlarge (8x V100 16GB) - $24.48/hr",
    "p3dn.24xlarge": "p3dn.24xlarge (8x V100 32GB) - $31.22/hr",
}

# Common model sizes (parameters)
COMMON_MODEL_SIZES = {
    "100M": 100_000_000,
    "350M": 350_000_000,
    "1B": 1_000_000_000,
    "3B": 3_000_000_000,
    "7B": 7_000_000_000,
    "13B": 13_000_000_000,
    "30B": 30_000_000_000,
    "70B": 70_000_000_000,
}

# Common dataset sizes (tokens)
COMMON_DATASET_SIZES = {
    "1B tokens": 1_000_000_000,
    "10B tokens": 10_000_000_000,
    "50B tokens": 50_000_000_000,
    "100B tokens": 100_000_000_000,
    "200B tokens": 200_000_000_000,
    "500B tokens": 500_000_000_000,
    "1T tokens": 1_000_000_000_000,
    "2T tokens": 2_000_000_000_000,
}

# Help text
HELP_TEXT = {
    "parameter_count": "Number of trainable parameters in the model",
    "training_tokens": (
        "Number of tokens in the dataset (per epoch). Total tokens seen = this × Epochs. "
        "For long-context models (>8K tokens), actual FLOPs may be 10–30% higher due to "
        "attention scaling — adjust MFU downward to compensate."
    ),
    "batch_size": "Total batch size across all GPUs",
    "mfu": "Model FLOPs Utilization - percentage of theoretical peak performance achieved",
    "gradient_accumulation": "Number of forward/backward passes before updating weights",
    "num_instances": "Number of GPU instances to use (scales training speed)",
    "gradient_checkpointing": "Recomputes activations to save memory, increasing FLOPs by ~33%",
}


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded",
    )


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        /* Remove metric background to use default styling */
        [data-testid="stMetricValue"] {
            color: inherit;
        }
        [data-testid="stMetricLabel"] {
            color: inherit;
        }
        .cost-high {
            color: #ff4444;
            font-weight: bold;
        }
        .cost-medium {
            color: #ff8800;
            font-weight: bold;
        }
        .cost-low {
            color: #44aa44;
            font-weight: bold;
        }
        h1 {
            padding-bottom: 1rem;
        }
        h2 {
            padding-top: 1rem;
        }
        .scenario-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #ff4b4b;
        }
        </style>
    """, unsafe_allow_html=True)
