"""Streamlit app configuration."""

import streamlit as st

# Page configuration
PAGE_TITLE = "Floply"
LAYOUT = "wide"

# Chinchilla scaling-law optimal ratio (Hoffmann et al. 2022)
# Compute-optimal pre-training requires ~20 tokens per parameter.
CHINCHILLA_OPTIMAL_RATIO = 20

# LoRA efficiency optimal token-to-adapter-param ratio.
# Empirical sweet spot: ~50 tokens per trainable adapter parameter.
LORA_OPTIMAL_RATIO = 50

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

# Mixed-precision training format options (ordered fastest/smallest → most precise)
MIXED_PRECISION_OPTIONS = ["fp4", "int8", "fp8", "bf16", "fp16", "tf32", "fp32"]

# Instance type display names
INSTANCE_DISPLAY_NAMES = {
    "p4d.24xlarge": "p4d.24xlarge (8x A100 40GB) - $26.87/hr",
    "p4de.24xlarge": "p4de.24xlarge (8x A100 80GB) - $32.77/hr",
    "p5.48xlarge": "p5.48xlarge (8x H100 80GB) - $66.64/hr",
    "p3.16xlarge": "p3.16xlarge (8x V100 16GB) - $24.48/hr",
    "p3dn.24xlarge": "p3dn.24xlarge (8x V100 32GB) - $31.22/hr",
}

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
