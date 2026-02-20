"""Input form components for the Streamlit app."""

import streamlit as st
from src.cost_modelling.calculator import ModelConfig, TrainingConfig
from src.cost_modelling.gpu_specs import list_available_instances
from src.cost_modelling.model_loader import load_models
from src.app.config import (
    INSTANCE_DISPLAY_NAMES,
    COMMON_MODEL_SIZES,
    COMMON_DATASET_SIZES,
    HELP_TEXT,
)


@st.cache_data
def _cached_models():
    return load_models()

# Chinchilla-based token thresholds (tokens per parameter)
# Source: Hoffmann et al. 2022 (https://arxiv.org/abs/2203.15556)
TOKENS_PER_PARAM_MINIMUM = 10    # Below this: model likely undertrained, poor generalisation
TOKENS_PER_PARAM_CHINCHILLA = 20  # Compute-optimal (best loss per training FLOP)
TOKENS_PER_PARAM_INFERENCE = 100  # Inference-optimal (LLaMA-style over-training)


def render_model_config_form() -> ModelConfig:
    """Render model configuration input form.

    Users can either select a known model from src/models/*.yaml (auto-fills
    all parameters) or configure a custom model manually.
    """
    st.subheader("Model Configuration")

    known_models = _cached_models()
    preset_options = ["Custom"] + [m.display_name for m in known_models]

    col1, col2 = st.columns(2)

    with col1:
        preset = st.selectbox(
            "Model Preset",
            options=preset_options,
            index=0,
            help=(
                "Select a known architecture to auto-fill parameters, "
                "or choose Custom to enter values manually. "
                "New models can be added by contributing a YAML file to src/models/."
            ),
        )

    selected_definition = None
    if preset != "Custom":
        selected_definition = next(
            m for m in known_models if m.display_name == preset
        )

    with col1:
        if selected_definition:
            parameter_count = selected_definition.parameter_count
            st.info(f"{parameter_count:,} parameters")
            if selected_definition.notes:
                st.caption(selected_definition.notes)
        else:
            parameter_count = st.number_input(
                "Parameter Count",
                min_value=1_000_000,
                max_value=1_000_000_000_000,
                value=1_000_000_000,
                step=100_000_000,
                format="%d",
                help=HELP_TEXT["parameter_count"],
            )

    with col2:
        if selected_definition:
            architecture = selected_definition.family
            st.info(f"Architecture: {architecture}")
        else:
            architecture = st.selectbox(
                "Architecture Type",
                options=["transformer", "cnn", "rnn", "vit", "diffusion"],
                help="Architecture family (affects FLOPs multiplier)",
            )

    return ModelConfig(
        parameter_count=parameter_count,
        architecture=architecture,
        model_definition=selected_definition,
    )


def render_training_config_form(model_config: ModelConfig) -> TrainingConfig:
    """Render training configuration input form.
    
    Args:
        model_config: Model configuration for context
        
    Returns:
        TrainingConfig object with user inputs
    """
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Dataset size selection — default to 50B tokens
        dataset_options = ["Custom"] + list(COMMON_DATASET_SIZES.keys())
        dataset_option = st.selectbox(
            "Dataset Size (per epoch)",
            options=dataset_options,
            index=dataset_options.index("50B tokens"),
            help=HELP_TEXT["training_tokens"]
        )
        
        if dataset_option == "Custom":
            training_tokens = st.number_input(
                "Training Tokens",
                min_value=1_000_000,
                max_value=10_000_000_000_000,
                value=50_000_000_000,
                step=1_000_000_000,
                format="%d",
                help=HELP_TEXT["training_tokens"]
            )
        else:
            training_tokens = COMMON_DATASET_SIZES[dataset_option]
            st.info(f"Selected: {training_tokens:,} tokens")
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=16384,
            value=1024,
            step=128,
            help=HELP_TEXT["batch_size"]
        )
        
        gradient_accumulation_steps = st.number_input(
            "Gradient Accumulation Steps",
            min_value=1,
            max_value=128,
            value=1,
            step=1,
            help=HELP_TEXT["gradient_accumulation"]
        )
    
    with col2:
        # Instance selection
        available_instances = list_available_instances()
        instance_options = [INSTANCE_DISPLAY_NAMES.get(inst, inst) for inst in available_instances]
        instance_display = st.selectbox(
            "Instance Type",
            options=instance_options,
            index=0,
        )
        # Extract actual instance type from display name
        instance_type = [k for k, v in INSTANCE_DISPLAY_NAMES.items() if v == instance_display][0]
        
        num_instances = st.number_input(
            "Number of Instances",
            min_value=1,
            max_value=128,
            value=1,
            step=1,
            help=HELP_TEXT["num_instances"]
        )
        
        mixed_precision = st.selectbox(
            "Mixed Precision",
            options=["fp16", "bf16", "fp32"],
            index=1,
        )
        
        mfu_override = st.slider(
            "Model FLOPs Utilization (MFU)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help=HELP_TEXT["mfu"]
        )
        
        epochs = st.number_input(
            "Epochs",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.5,
            format="%.2f",
            help=(
                "Number of passes through the dataset. "
                "Total tokens seen = Dataset Size × Epochs. "
                "LLM pre-training is typically 1 epoch; fine-tuning is often 3–10."
            )
        )
        
        gradient_checkpointing = st.checkbox(
            "Gradient Checkpointing",
            value=False,
            help=(
                "Recomputes activations during the backward pass to save GPU memory. "
                "Increases FLOPs by ~33% (6ND → 8ND). "
                "Typically required for large models that don't fit in GPU memory."
            )
        )
    
    # ── Chinchilla / scaling-law sanity check ────────────────────────────────
    total_tokens = training_tokens * epochs
    tokens_per_param = total_tokens / model_config.parameter_count

    if tokens_per_param < TOKENS_PER_PARAM_MINIMUM:
        st.error(
            f"⚠️ **Severely undertrained** — {tokens_per_param:.1f} tokens/param "
            f"(minimum recommended: {TOKENS_PER_PARAM_MINIMUM}×). "
            "The model is unlikely to generalise. Increase dataset size or epochs."
        )
    elif tokens_per_param < TOKENS_PER_PARAM_CHINCHILLA:
        st.warning(
            f"⚠️ **Below Chinchilla-optimal** — {tokens_per_param:.1f} tokens/param "
            f"(Chinchilla optimal: {TOKENS_PER_PARAM_CHINCHILLA}×, "
            f"source: Hoffmann et al. 2022). "
            "Consider more data for better compute efficiency."
        )
    elif tokens_per_param < TOKENS_PER_PARAM_INFERENCE:
        st.success(
            f"✅ **Chinchilla-optimal range** — {tokens_per_param:.1f} tokens/param "
            f"({TOKENS_PER_PARAM_CHINCHILLA}–{TOKENS_PER_PARAM_INFERENCE}×). "
            "Good compute efficiency."
        )
    else:
        st.success(
            f"✅ **Inference-optimal (over-trained)** — {tokens_per_param:.1f} tokens/param "
            f"(>{TOKENS_PER_PARAM_INFERENCE}×). "
            "Smaller model trained longer — efficient for deployment."
        )

    return TrainingConfig(
        training_tokens=training_tokens,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        mixed_precision=mixed_precision,
        instance_type=instance_type,
        num_instances=num_instances,
        mfu_override=mfu_override,
        gradient_checkpointing=gradient_checkpointing,
    )


def render_project_config_form():
    """Render project-level configuration form.
    
    Returns:
        Dictionary with project parameters
    """
    st.subheader("Project Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_training_runs = st.number_input(
            "Number of Full Training Runs",
            min_value=1,
            max_value=100,
            value=1,
            help="Number of complete training runs from scratch"
        )
        
        num_hyperparameter_trials = st.number_input(
            "Hyperparameter Tuning Trials",
            min_value=0,
            max_value=1000,
            value=0,
            help="Number of hyperparameter search trials (typically shorter runs)"
        )
        
        num_ablations = st.number_input(
            "Ablation Studies",
            min_value=0,
            max_value=100,
            value=0,
            help="Number of ablation experiments"
        )
    
    with col2:
        dataset_size_tb = st.number_input(
            "Dataset Size (TB)",
            min_value=0.0,
            max_value=10000.0,
            value=1.0,
            step=0.1,
            help="Size of training dataset for storage cost calculation"
        )
        
        storage_months = st.number_input(
            "Storage Duration (months)",
            min_value=1,
            max_value=120,
            value=3,
            help="How long to store the dataset"
        )
        
        dev_instance_hours = st.number_input(
            "Dev/Debug Instance Hours",
            min_value=0,
            max_value=10000,
            value=0,
            help="Additional instance hours for development and debugging"
        )
    
    return {
        "num_training_runs": num_training_runs,
        "num_hyperparameter_trials": num_hyperparameter_trials,
        "num_ablations": num_ablations,
        "dataset_size_tb": dataset_size_tb,
        "storage_months": storage_months,
        "dev_instance_hours": dev_instance_hours,
    }
