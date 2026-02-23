from typing import Dict, Any

import streamlit as st

from src.app.helpers import _fmt_tokens

MIN_VALUE = 1
MAX_VALUE = 1000000000000
STEP = 10

def render_dataset_dimensions(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Render the dataset dimensions form.
    Args:
        training_config: The training configuration dictionary.
    Returns:
        The training configuration dictionary with the dataset dimensions added.
    """
    with st.expander("Dataset Dimensions", expanded=True):

        ds_col1, ds_col2, ds_col3 = st.columns(3)
        with ds_col1:
            data_type = st.selectbox("Select Modality", options=["Text", "Image", "Audio", "Video"], index=None)

        with ds_col2:
            dataset_size = st.number_input("Enter Dataset Size", value=None, min_value=MIN_VALUE, max_value=MAX_VALUE, step=STEP)
            if dataset_size:
                st.caption(f"{int(dataset_size):,} samples")

        with ds_col3:
            if data_type == "Text":
                avg_seq_len_words = st.number_input(
                    "Avg sequence length (words)",
                    min_value=1,
                    max_value=100000,
                    value=400,
                    step=50,
                    help="Average number of words per training example. 1 word ≈ 1.3 BPE tokens.",
                )
                tokens_per_sample = int(avg_seq_len_words * 1.3)
                bytes_per_sample = tokens_per_sample * 4  # ~4 bytes/token for raw UTF-8 text
            elif data_type == "Image":
                resolution = st.selectbox(
                    "Image resolution (px)",
                    options=[224, 336, 512, 1024],
                    index=0,
                    help="Square image side length in pixels.",
                )
                patch_size = st.selectbox(
                    "Patch size (px)",
                    options=[14, 16, 32],
                    index=1,
                    help="ViT patch size. Smaller patches = more tokens per image.",
                )
                tokens_per_sample = (resolution // patch_size) ** 2
                bytes_per_sample = (resolution * resolution * 3) // 10  # JPEG ~10:1
            elif data_type == "Audio":
                clip_duration = st.number_input(
                    "Avg clip duration (seconds)",
                    min_value=0.1,
                    max_value=3600.0,
                    value=30.0,
                    step=5.0,
                    help="Average audio clip length.",
                )
                audio_tokenizer = st.selectbox(
                    "Tokenizer Style",
                    options=[
                        "Whisper (50 tok/sec)",
                        "EnCodec 24kHz (75 tok/sec x codebooks)",
                        "SoundStream (50 tok/sec x codebooks)",
                    ],
                    help="Whisper produces mel-spectrogram tokens. EnCodec/SoundStream use residual vector quantization codebooks.",
                )
                if "EnCodec" in audio_tokenizer or "SoundStream" in audio_tokenizer:
                    num_codebooks = st.number_input(
                        "Codebooks",
                        min_value=1,
                        max_value=16,
                        value=8,
                        step=1,
                        help="Number of RVQ codebook levels.",
                    )
                    base_rate = 75 if "EnCodec" in audio_tokenizer else 50
                    tokens_per_sample = int(clip_duration * base_rate * num_codebooks)
                else:
                    tokens_per_sample = int(clip_duration * 50)  # Whisper
                bytes_per_sample = int(clip_duration * 32_000)  # 16 kHz mono 16-bit
            elif data_type == "Video":
                vid_duration = st.number_input(
                    "Avg clip duration (seconds)",
                    min_value=0.1,
                    max_value=3600.0,
                    value=10.0,
                    step=1.0,
                )
                sampled_fps = st.number_input(
                    "Sampled FPS",
                    min_value=1,
                    max_value=60,
                    value=1,
                    step=1,
                    help="Frames sampled per second (not source FPS). Lower = fewer tokens.",
                )
                vid_resolution = st.selectbox(
                    "Frame resolution (px)",
                    options=[224, 336, 512],
                    index=0,
                )
                vid_patch_size = st.selectbox(
                    "Patch size (px)",
                    options=[14, 16, 32],
                    index=1,
                )
                tokens_per_frame = (vid_resolution // vid_patch_size) ** 2
                num_frames = int(vid_duration * sampled_fps)
                tokens_per_sample = tokens_per_frame * num_frames
                bytes_per_frame = (vid_resolution * vid_resolution * 3) // 10  # JPEG ~10:1 per extracted frame
                bytes_per_sample = bytes_per_frame * num_frames

        if data_type and dataset_size:
            total_tokens = int(tokens_per_sample * dataset_size)
            dataset_size_tb = (bytes_per_sample * dataset_size) / 1e12

            st.divider()
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(
                    label="Tokens per sample",
                    value=_fmt_tokens(tokens_per_sample),
                )
            with res_col2:
                st.metric(
                    label="Total tokens",
                    value=_fmt_tokens(total_tokens),
                )
            with res_col3:
                if dataset_size_tb >= 1:
                    tb_label = f"{dataset_size_tb:.2f} TB"
                elif dataset_size_tb >= 0.001:
                    tb_label = f"{dataset_size_tb * 1000:.2f} GB"
                else:
                    tb_label = f"{dataset_size_tb * 1e6:.2f} MB"
                st.metric(
                    label="Dataset Size (est.)",
                    value=tb_label,
                    help="Estimated raw storage size based on modality and sample count.",
                )

            storage_months = st.slider(
                "Storage Duration (months)",
                min_value=1,
                max_value=36,
                value=3,
                step=1,
                help="How long the dataset will be stored in S3.",
            )

            training_config["Modality"] = data_type
            training_config["Tokens per sample"] = tokens_per_sample
            training_config["Total tokens"] = total_tokens
            training_config["Dataset Size (TB)"] = round(dataset_size_tb, 6)
            training_config["Storage Duration (months)"] = storage_months
    return training_config