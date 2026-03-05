"""Minimum dataset size reference tool based on Chinchilla scaling laws."""

from typing import Optional

import streamlit as st
import plotly.graph_objects as go

from src.app.config import (
    CHINCHILLA_OPTIMAL_RATIO,
    PRE_TRAINING_TIERS,
    FULL_FT_TIERS,
    LORA_TIERS,
)
from src.app.helpers import (
    _fmt_tokens,
    _fmt_samples,
    _scaled_number_input,
    _cached_models,
)
from src.cost_modelling.calculator import calculate_lora_trainable_params
from src.cost_modelling.dataset import (
    tokens_per_text_sample,
    tokens_per_image_sample,
    tokens_per_audio_sample,
    tokens_per_video_sample,
    ENCODEC_TOKENS_PER_SECOND,
    SOUNDSTREAM_TOKENS_PER_SECOND,
    WHISPER_TOKENS_PER_SECOND,
)


def render_min_dataset_size_page() -> None:
    st.markdown(
        "<p style='text-align:center;color:#888;'>How much data do I need?</p>",
        unsafe_allow_html=True,
    )

    # ── Model Configuration ───────────────────────────────────────────────────
    with st.expander("Model Configuration", expanded=True):
        training_type = st.selectbox(
            "Training Type",
            options=["Pre-Training", "Fine-Tuning"],
            index=None,
            help="The training paradigm determines which scaling-law thresholds apply.",
            key="mds_training_type",
        )

        if training_type is None:
            st.caption("Select a training type to continue.")
            return

        # ── Pre-Training ──────────────────────────────────────────────────────
        pre_param_count: Optional[int] = None

        if training_type == "Pre-Training":
            pre_param_count = _scaled_number_input(
                label="Parameter Count",
                units=["M", "B"],
                default_value=7.0,
                default_unit="B",
                help="Total parameter count of the model being trained from scratch.",
                key="mds_pre_params",
            )

        # ── Fine-Tuning ───────────────────────────────────────────────────────
        ft_param_count: Optional[int] = None
        ft_d_model: Optional[int] = None
        ft_num_layers: Optional[int] = None
        ft_architecture: dict = {}
        ft_method: Optional[str] = None
        trainable_params: Optional[int] = None

        if training_type == "Fine-Tuning":
            ft_col1, ft_col2 = st.columns(2)
            with ft_col1:
                known_models = _cached_models()
                preset_options = [m.display_name for m in known_models] + ["Custom"]
                base_model_name = st.selectbox(
                    "Base Model",
                    options=preset_options,
                    index=None,
                    help="Select a known model to auto-fill architecture params, or choose Custom.",
                    key="mds_base_model",
                )

            if base_model_name is None:
                st.caption("Select a base model to continue.")
                return

            if base_model_name != "Custom":
                selected_model = next(
                    m for m in known_models if m.display_name == base_model_name
                )
                ft_param_count = selected_model.parameter_count
                ft_d_model = selected_model.architecture.get("d_model", 4096)
                ft_num_layers = selected_model.architecture.get("num_layers", 32)
                ft_architecture = selected_model.architecture
                with ft_col1:
                    st.caption(selected_model.notes)
            else:
                with ft_col1:
                    ft_param_count = _scaled_number_input(
                        "Parameter Count",
                        units=["M", "B"],
                        default_value=7.0,
                        default_unit="B",
                        key="mds_ft_params",
                    )
                    ft_d_model = st.number_input(
                        "Hidden Dimension (d_model)",
                        min_value=64,
                        max_value=65536,
                        value=4096,
                        step=64,
                        key="mds_ft_d_model",
                    )
                    ft_num_layers = st.number_input(
                        "Number of Layers",
                        min_value=1,
                        max_value=512,
                        value=32,
                        step=1,
                        key="mds_ft_num_layers",
                    )

            if not ft_param_count:
                return

            ft_method = st.selectbox(
                "Fine-Tuning Method",
                options=["Full Fine-Tuning", "LoRA", "QLoRA"],
                index=0,
                help="Full: all parameters updated. LoRA/QLoRA: only small low-rank adapter matrices are trained.",
                key="mds_ft_method",
            )

            if ft_method in ("LoRA", "QLoRA"):
                lora_col1, lora_col2 = st.columns(2)
                with lora_col1:
                    lora_rank = st.slider(
                        "LoRA Rank (r)",
                        min_value=1,
                        max_value=128,
                        value=16,
                        step=1,
                        help="Higher rank = more expressive adapters but more parameters.",
                        key="mds_lora_rank",
                    )
                    target_modules = st.multiselect(
                        "Target Modules",
                        options=[
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "up_proj",
                            "down_proj",
                        ],
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Which weight matrices to attach LoRA adapters to.",
                        key="mds_target_modules",
                    )

                trainable_params = calculate_lora_trainable_params(
                    ft_method=ft_method,
                    base_params=ft_param_count,
                    target_modules=target_modules,
                    d_model=ft_d_model or 0,
                    num_layers=ft_num_layers or 0,
                    lora_rank=lora_rank,
                    architecture=ft_architecture,
                )

                if trainable_params:
                    pct_trainable = trainable_params / ft_param_count * 100
                    with lora_col2:
                        st.metric(
                            "Trainable Parameters",
                            _fmt_tokens(trainable_params),
                            help="Number of adapter parameters updated during training.",
                        )
                        st.metric(
                            "% of Total Params",
                            f"{pct_trainable:.2f}%",
                            help="Fraction of base model parameters that are trainable.",
                        )
                    if not ft_architecture:
                        st.caption(
                            "⚠ Approximate LoRA param count — select a preset model for GQA-aware calculation."
                        )
                else:
                    st.warning(
                        "Configure LoRA settings above to calculate trainable parameters."
                    )
                    return
            else:
                trainable_params = ft_param_count

    # ── Guards ────────────────────────────────────────────────────────────────
    if training_type == "Pre-Training" and not pre_param_count:
        return
    if training_type == "Fine-Tuning" and not ft_param_count:
        return

    # ── Data Requirements ─────────────────────────────────────────────────────
    st.divider()

    if training_type == "Pre-Training":
        tiers, effective_count = PRE_TRAINING_TIERS, pre_param_count
        _render_requirements_section(
            "Pre-Training Data Requirements",
            f"Chinchilla scaling laws (Hoffmann et al. 2022) for a **{_fmt_tokens(pre_param_count)}-parameter** model.",
            tiers,
            effective_count,
        )
    else:  # Fine-Tuning
        if ft_method == "Full Fine-Tuning":
            tiers, effective_count = FULL_FT_TIERS, ft_param_count
            _render_requirements_section(
                "Full Fine-Tuning Data Requirements",
                f"Practical thresholds for a **{_fmt_tokens(ft_param_count)}-parameter** base model.",
                tiers,
                effective_count,
            )
        else:
            tiers, effective_count = LORA_TIERS, trainable_params
            _render_requirements_section(
                f"{ft_method} Data Requirements",
                (
                    f"Thresholds for **{ft_method}** on a **{_fmt_tokens(ft_param_count)}-parameter** base model "
                    f"with **{_fmt_tokens(trainable_params)} trainable adapter parameters** "
                    f"({trainable_params / ft_param_count * 100:.2f}% of base)."
                ),
                tiers,
                effective_count,
                lora_note=True,
            )

    st.divider()
    _render_reverse_calculator(tiers, effective_count)


# ── Output helpers ────────────────────────────────────────────────────────────


def _render_requirements_section(
    title: str,
    caption: str,
    tiers: list,
    effective_count: int,
    lora_note: bool = False,
) -> None:
    st.subheader(title)
    st.caption(caption)
    _render_tier_chart(tiers, effective_count)
    if lora_note:
        st.caption(
            "Note: thresholds are relative to **adapter parameters**, not the full base model. "
            f"The frozen base provides strong priors — far less data is needed than Chinchilla's "
            f"{CHINCHILLA_OPTIMAL_RATIO} tok/param pre-training recommendation."
        )


def _render_tier_chart(tiers: list, effective_count: int) -> None:
    """Horizontal range bar chart, ordered top-to-bottom: smallest tier first."""
    fig = go.Figure()

    for tier in reversed(tiers):
        x_start = effective_count * tier["ratio_min_chart"]
        x_end = effective_count * tier["ratio_max"]
        width = x_end - x_start
        label = f"{_fmt_tokens(int(x_start))}–{_fmt_tokens(int(x_end))}"

        fig.add_trace(
            go.Bar(
                x=[width],
                y=[tier["tier"]],
                base=[x_start],
                orientation="h",
                marker_color=tier["color"],
                name=tier["tier"],
                text=label,
                textposition="inside",
                hovertemplate=(
                    f"<b>{tier['tier']}</b><br>"
                    f"{label} tokens<br>"
                    f"{tier['label']}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(
            type="log",
            title="Tokens (log scale)",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="",
            gridcolor="rgba(255,255,255,0.08)",
            categoryorder="array",
            categoryarray=[t["tier"] for t in tiers],
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=230,
    )
    st.plotly_chart(fig, width="stretch")


def _render_reverse_calculator(tiers: list, effective_count: int) -> None:
    """Translate tier token requirements into real example counts for a chosen modality."""
    with st.expander(
        "Reverse Dataset Calculator — How much data do I need?", expanded=True
    ):
        modality = st.selectbox(
            "Modality",
            options=["Text", "Image", "Audio", "Video"],
            index=0,
            key="mds_rc_modality",
        )

        rc_col1, rc_col2 = st.columns(2)

        if modality == "Text":
            with rc_col1:
                avg_words = st.number_input(
                    "Avg sequence length (words)",
                    min_value=1,
                    max_value=100_000,
                    value=400,
                    step=50,
                    help="Average number of words per training example. 1 word ≈ 1.3 BPE tokens.",
                    key="mds_rc_avg_words",
                )
            tokens_per_sample = tokens_per_text_sample(avg_words)
            sample_unit = "text examples"
            with rc_col2:
                st.metric("Tokens per example", _fmt_tokens(tokens_per_sample))

        elif modality == "Image":
            with rc_col1:
                resolution = st.selectbox(
                    "Image resolution (px)",
                    options=[224, 336, 512, 1024],
                    index=0,
                    help="Square image side length in pixels.",
                    key="mds_rc_resolution",
                )
                patch_size = st.selectbox(
                    "Patch size (px)",
                    options=[14, 16, 32],
                    index=1,
                    help="ViT patch size. Smaller patches = more tokens per image.",
                    key="mds_rc_patch_size",
                )
            tokens_per_sample = tokens_per_image_sample(resolution, patch_size)
            sample_unit = "images"
            with rc_col2:
                st.metric("Tokens per image", _fmt_tokens(tokens_per_sample))

        elif modality == "Audio":
            with rc_col1:
                clip_duration = st.number_input(
                    "Avg clip duration (seconds)",
                    min_value=0.1,
                    max_value=3600.0,
                    value=30.0,
                    step=5.0,
                    help="Average audio clip length.",
                    key="mds_rc_clip_duration",
                )
                audio_tokenizer = st.selectbox(
                    "Tokenizer style",
                    options=[
                        "Whisper (50 tok/sec)",
                        "EnCodec 24kHz (75 tok/sec × codebooks)",
                        "SoundStream (50 tok/sec × codebooks)",
                    ],
                    key="mds_rc_audio_tokenizer",
                )
                if "EnCodec" in audio_tokenizer or "SoundStream" in audio_tokenizer:
                    num_codebooks = st.number_input(
                        "Codebooks",
                        min_value=1,
                        max_value=16,
                        value=8,
                        step=1,
                        key="mds_rc_codebooks",
                    )
                    base_rate = (
                        ENCODEC_TOKENS_PER_SECOND
                        if "EnCodec" in audio_tokenizer
                        else SOUNDSTREAM_TOKENS_PER_SECOND
                    )
                    tokens_per_sample = tokens_per_audio_sample(
                        clip_duration, base_rate, num_codebooks
                    )
                else:
                    tokens_per_sample = tokens_per_audio_sample(
                        clip_duration, WHISPER_TOKENS_PER_SECOND
                    )
            sample_unit = "audio clips"
            with rc_col2:
                st.metric("Tokens per clip", _fmt_tokens(tokens_per_sample))

        else:  # Video
            with rc_col1:
                vid_duration = st.number_input(
                    "Avg clip duration (seconds)",
                    min_value=0.1,
                    max_value=3600.0,
                    value=10.0,
                    step=1.0,
                    key="mds_rc_vid_duration",
                )
                sampled_fps = st.number_input(
                    "Sampled FPS",
                    min_value=1,
                    max_value=60,
                    value=1,
                    step=1,
                    help="Frames sampled per second (not source FPS). Lower = fewer tokens.",
                    key="mds_rc_fps",
                )
                vid_resolution = st.selectbox(
                    "Frame resolution (px)",
                    options=[224, 336, 512],
                    index=0,
                    key="mds_rc_vid_resolution",
                )
                vid_patch_size = st.selectbox(
                    "Patch size (px)",
                    options=[14, 16, 32],
                    index=1,
                    key="mds_rc_vid_patch_size",
                )
            tokens_per_sample = tokens_per_video_sample(
                vid_duration, sampled_fps, vid_resolution, vid_patch_size
            )
            sample_unit = "video clips"
            with rc_col2:
                st.metric("Tokens per clip", _fmt_tokens(tokens_per_sample))

        if tokens_per_sample <= 0:
            return

        st.divider()

        st.markdown(f"**Total {sample_unit.capitalize()} Samples needed by tier**")
        cols = st.columns(len(tiers))
        for col, tier in zip(cols, tiers):
            tok_min = int(effective_count * tier["ratio_min_chart"])
            tok_max = int(effective_count * tier["ratio_max"])
            samp_min = max(1, tok_min // tokens_per_sample)
            samp_max = max(1, tok_max // tokens_per_sample)
            range_label = (
                f"< {_fmt_samples(samp_max)}"
                if tier["ratio_min"] == 0
                else f"{_fmt_samples(samp_min)}–{_fmt_samples(samp_max)}"
            )
            col.metric(
                label=tier["tier"],
                value=range_label,
                help=(
                    f"{tier['label']}: "
                    f"{_fmt_tokens(tok_min)}–{_fmt_tokens(tok_max)} tokens ÷ "
                    f"{_fmt_tokens(tokens_per_sample)} tok/{sample_unit[:-1]}"
                ),
            )
