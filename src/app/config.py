"""Streamlit app configuration."""

import streamlit as st

# Page configuration
PAGE_TITLE = "Floply"
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
        layout=LAYOUT,
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "Floply — ML Training Cost Estimator",
        },
    )

def apply_custom_css():
    """Apply custom CSS dark-theme styling."""
    st.markdown("""
        <style>
        /* ── Design tokens ───────────────────────────────────────────────── */
        :root {
            --bg-base:        #0b0f1a;
            --bg-surface:     #131929;
            --bg-surface-2:   #1a2235;
            --bg-surface-3:   #1e2840;
            --accent:         #5b8dee;
            --accent-dim:     rgba(91,141,238,.15);
            --accent-border:  rgba(91,141,238,.35);
            --text-primary:   #e8ecf4;
            --text-muted:     #7a8499;
            --text-caption:   #5a6278;
            --border:         rgba(255,255,255,.07);
            --border-hover:   rgba(255,255,255,.13);
            --radius-sm:      6px;
            --radius-md:      10px;
            --radius-lg:      14px;
            --shadow:         0 2px 12px rgba(0,0,0,.45);
            --shadow-lg:      0 4px 24px rgba(0,0,0,.6);
        }

        /* ── Page & layout ───────────────────────────────────────────────── */
        [data-testid="stAppViewContainer"] {
            background: var(--bg-base);
        }
        [data-testid="stMain"] {
            background: var(--bg-base);
        }
        .main .block-container {
            padding-top: 1.75rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        /* ── Sidebar ─────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: var(--bg-surface);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] .stRadio label {
            color: var(--text-primary) !important;
            font-size: 0.9rem;
            padding: 0.35rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
            color: var(--text-muted) !important;
            font-size: 0.75rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        /* ── Typography ──────────────────────────────────────────────────── */
        h1 {
            font-size: 2rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            color: var(--text-primary) !important;
            padding-bottom: 0.5rem;
        }
        h2 {
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
        }
        h3 {
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
        }
        p, li {
            color: var(--text-primary);
            line-height: 1.65;
        }
        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--text-caption) !important;
            font-size: 0.78rem !important;
        }
        code {
            background: var(--bg-surface-3) !important;
            color: #a8c7fa !important;
            border-radius: var(--radius-sm);
            padding: 0.15em 0.4em;
            font-size: 0.85em;
        }
        pre code {
            padding: 0 !important;
            background: transparent !important;
        }
        pre {
            background: var(--bg-surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 1rem 1.25rem !important;
        }

        /* ── Expanders ───────────────────────────────────────────────────── */
        [data-testid="stExpander"] {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            margin-bottom: 0.6rem;
            box-shadow: var(--shadow);
            overflow: hidden;
        }
        [data-testid="stExpander"]:hover {
            border-color: var(--border-hover) !important;
        }
        [data-testid="stExpanderToggleIcon"] {
            color: var(--text-muted) !important;
        }
        [data-testid="stExpanderDetails"] {
            background: var(--bg-surface) !important;
            padding: 0.25rem 0.25rem 0.5rem !important;
        }
        details summary {
            padding: 0.75rem 1rem !important;
            color: var(--text-primary) !important;
            font-weight: 500;
            font-size: 0.95rem;
        }
        details summary:hover {
            background: var(--bg-surface-2) !important;
        }

        /* ── Metrics ─────────────────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 0.9rem 1.1rem !important;
            box-shadow: var(--shadow);
            transition: border-color .2s;
        }
        [data-testid="stMetric"]:hover {
            border-color: var(--accent-border) !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.02em;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            font-weight: 500 !important;
            color: var(--text-muted) !important;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.78rem !important;
            color: var(--text-muted) !important;
        }
        [data-testid="stMetricDelta"] svg {
            display: none;
        }

        /* ── Form inputs ─────────────────────────────────────────────────── */
        [data-testid="stWidgetLabel"] {
            color: var(--text-primary) !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            margin-bottom: 0.2rem;
        }
        .stSelectbox > div > div,
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {
            background: var(--bg-surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text-primary) !important;
            font-size: 0.9rem;
        }
        .stSelectbox > div > div:hover,
        .stNumberInput > div > div > input:hover,
        .stTextInput > div > div > input:hover {
            border-color: var(--border-hover) !important;
        }
        .stSelectbox > div > div:focus-within,
        .stNumberInput > div > div > input:focus,
        .stTextInput > div > div > input:focus {
            border-color: var(--accent-border) !important;
            box-shadow: 0 0 0 2px var(--accent-dim) !important;
        }
        /* Slider */
        [data-testid="stSlider"] > div > div > div > div {
            background: var(--accent) !important;
        }
        [data-testid="stSlider"] .stSlider > div > div > div:first-child {
            background: var(--bg-surface-3) !important;
        }

        /* ── Checkboxes ──────────────────────────────────────────────────── */
        .stCheckbox label {
            color: var(--text-primary) !important;
            font-size: 0.88rem !important;
        }

        /* ── Multiselect ─────────────────────────────────────────────────── */
        .stMultiSelect > div > div {
            background: var(--bg-surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
        }
        [data-testid="stMultiSelectTag"] {
            background: var(--accent-dim) !important;
            border: 1px solid var(--accent-border) !important;
            color: var(--accent) !important;
            border-radius: 4px !important;
            font-size: 0.78rem !important;
        }

        /* ── Buttons ─────────────────────────────────────────────────────── */
        .stButton > button {
            background: var(--bg-surface-2) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            font-weight: 500;
            font-size: 0.88rem;
            padding: 0.4rem 1rem;
            transition: all .2s;
        }
        .stButton > button:hover {
            border-color: var(--accent-border) !important;
            background: var(--accent-dim) !important;
            color: var(--accent) !important;
        }

        /* ── DataFrames / tables ─────────────────────────────────────────── */
        [data-testid="stDataFrame"] > div {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            overflow: hidden;
        }

        /* ── Alert banners ───────────────────────────────────────────────── */
        [data-testid="stAlert"] {
            border-radius: var(--radius-md) !important;
            border-width: 1px !important;
            border-style: solid !important;
            font-size: 0.88rem !important;
            padding: 0.8rem 1rem !important;
        }
        /* success */
        [data-testid="stAlert"][kind="success"],
        .stSuccess {
            background: rgba(40,167,69,.12) !important;
            border-color: rgba(40,167,69,.3) !important;
            color: #6fd98a !important;
        }
        /* warning */
        [data-testid="stAlert"][kind="warning"],
        .stWarning {
            background: rgba(255,152,0,.12) !important;
            border-color: rgba(255,152,0,.3) !important;
            color: #ffb84d !important;
        }
        /* error */
        [data-testid="stAlert"][kind="error"],
        .stError {
            background: rgba(220,53,69,.12) !important;
            border-color: rgba(220,53,69,.3) !important;
            color: #ff7a85 !important;
        }
        /* info */
        [data-testid="stAlert"][kind="info"],
        .stInfo {
            background: rgba(91,141,238,.12) !important;
            border-color: rgba(91,141,238,.3) !important;
            color: #93b8ff !important;
        }

        /* ── Dividers ────────────────────────────────────────────────────── */
        hr {
            border-color: var(--border) !important;
            margin: 1.25rem 0 !important;
        }

        /* ── Subheaders ──────────────────────────────────────────────────── */
        [data-testid="stSubheader"] {
            color: var(--text-primary) !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
            padding-top: 0.5rem;
        }

        /* ── Plotly chart containers ─────────────────────────────────────── */
        [data-testid="stPlotlyChart"] {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 0.5rem !important;
            box-shadow: var(--shadow);
        }
        </style>
    """, unsafe_allow_html=True)
