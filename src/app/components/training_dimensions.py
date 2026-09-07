import streamlit as st
from typing import Dict, Any, Tuple
from src.app.config import ARCHITECTURE_OPTIONS
from src.app.helpers import _cached_models, _fmt_tokens, _scaled_number_input
from src.cost_modelling.calculator import calculate_lora_trainable_params

def render_training_dimensions(training_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Render the compute dimensions form.
    Args:
        training_config: The training configuration dictionary.
    Returns:
        The training configuration dictionary with the compute dimensions added.
    """
    training_type = st.selectbox("Training Type", options=["Pre-Training", "Fine-Tuning", "Pre-training + Fine-Tuning"], index=None)
    training_config["Training Type"] = training_type
    
    if training_type:
        match training_type:
            case "Pre-Training":
                training_config = render_pre_training_dimensions(training_config)

            case "Fine-Tuning":
                training_config = render_fine_tuning_dimensions(training_config, pre_training=False)

            case "Pre-training + Fine-Tuning":
                training_config = render_pre_training_dimensions(training_config)
                if "Parameter Count" in training_config:
                    training_config = render_fine_tuning_dimensions(training_config, pre_training=True)
                else:
                    st.warning("Enter a parameter count for pre-training to continue.")
                    return training_config, False

    can_compute = (
        "Total tokens" in training_config
        and (
            "Parameter Count" in training_config       
            or "Base Model Params" in training_config 
        )
    )
    return training_config, can_compute


def render_pre_training_dimensions(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Render the pre-training dimensions form.
    Args:
        training_config: The training configuration dictionary.
    Returns:
        The training configuration dictionary with the pre-training dimensions added.
    """
    with st.expander("Pre-Training Configuration", expanded=True):
        pre_col1, pre_col2 = st.columns(2)
        with pre_col1:
            model_type = st.selectbox("Model Type", options=ARCHITECTURE_OPTIONS, index=0)
        with pre_col2:
            parameter_count = _scaled_number_input(
                "Parameter Count",
                units=["M", "B", "T"],
                default_value=7.0,
                default_unit="B",
                help="Total number of model parameters.",
                key="pre_parameter_count",
            )
        pre_col3, pre_col4 = st.columns(2)
        with pre_col3:
            pre_d_model = st.number_input(
                "Hidden Dimension (d_model)",
                min_value=64, max_value=65536, value=4096, step=64,
                help="Used for activation memory estimation. Leave at default if unknown.",
            )
        with pre_col4:
            pre_num_layers = st.number_input(
                "Number of Layers",
                min_value=1, max_value=512, value=32, step=1,
                help="Used for activation memory estimation. Leave at default if unknown.",
            )
        if model_type and parameter_count:
            with pre_col1:
                st.metric(label="Model Type", value=model_type)
            with pre_col2:
                st.metric(label="Parameter Count", value=_fmt_tokens(parameter_count))
            training_config["Model Type"] = model_type
            training_config["Parameter Count"] = parameter_count
            training_config["d_model"] = pre_d_model
            training_config["num_layers"] = pre_num_layers
    return training_config

def render_fine_tuning_dimensions(training_config: Dict[str, Any], pre_training: bool = False) -> Dict[str, Any]:
    """Render the fine-tuning dimensions form.
    Args:
        training_config: The training configuration dictionary.
    Returns:
        The training configuration dictionary with the fine-tuning dimensions added.
    """
    fine_tuning_type = st.selectbox("Fine-Tuning Type", options=["Supervised", "RL"], index=None)
    training_config["Fine-Tuning Type"] = fine_tuning_type
    if fine_tuning_type:
        with st.expander("Fine-Tuning Configuration", expanded=True):
            ft_col1, ft_col2 = st.columns(2)

            # Bound unconditionally: the `pre_training or base_model_name` check below
            # reads it on the pre-training path, where the selectbox never runs.
            base_model_name = None

            if pre_training:
                selected_model = None
                base_params = training_config["Parameter Count"]
                d_model = training_config["d_model"]
                num_layers = training_config["num_layers"]
                with ft_col1:
                    st.info("Pre-training parameters will be used for fine-tuning.")
            else:
                known_models = _cached_models()
                preset_options = [m.display_name for m in known_models] + ["Custom"]

                with ft_col1:
                    base_model_name = st.selectbox(
                        "Base Model",
                        options=preset_options,
                        index=None,
                        help="Select a known model to auto-fill architecture params, or choose Custom.",
                    )

                selected_model = None
                if base_model_name and base_model_name != "Custom":
                    selected_model = next(
                        m for m in known_models if m.display_name == base_model_name
                    )
                    base_params = selected_model.parameter_count
                    d_model = selected_model.architecture.get("d_model")
                    num_layers = selected_model.architecture.get("num_layers")
                    with ft_col1:
                        st.caption(selected_model.notes)
                elif base_model_name == "Custom":
                    with ft_col1:
                        base_params = _scaled_number_input(
                            "Parameter Count",
                            units=["M", "B", "T"],
                            default_value=7.0,
                            default_unit="B",
                            key="ft_parameter_count",
                        )
                        d_model = st.number_input(
                            "Hidden Dimension (d_model)",
                            min_value=64,
                            max_value=65536,
                            value=4096,
                            step=64,
                        )
                        num_layers = st.number_input(
                            "Number of Layers",
                            min_value=1,
                            max_value=256,
                            value=32,
                            step=1,
                        )

            if pre_training or base_model_name:
                training_config["Base Model Params"] = base_params
                # MoE models run each token through a subset of experts, so compute
                # scales with active params while storage/VRAM scale with the total.
                training_config["Base Model FLOPs Params"] = (
                    selected_model.effective_parameter_count
                    if selected_model
                    else base_params
                )
                training_config["d_model"] = d_model
                training_config["num_layers"] = num_layers
                with ft_col2:
                    ft_method = st.selectbox(
                        "Fine-Tuning Method",
                        options=["Full Fine-Tuning", "LoRA", "QLoRA"],
                        index=0,
                        help=(
                            "Full: all parameters updated. "
                            "LoRA/QLoRA: only small low-rank adapter matrices are trained."
                        ),
                    )

                training_config["Fine-Tuning Method"] = ft_method
                lora_rank = None
                target_modules = None
                if ft_method in ("LoRA", "QLoRA"):
                    with ft_col2:
                        lora_rank = st.slider(
                            "LoRA Rank (r)",
                            min_value=1,
                            max_value=128,
                            value=16,
                            step=1,
                            help="Higher rank = more expressive adapters but more parameters.",
                        )
                        target_modules = st.multiselect(
                            "Target Modules",
                            options=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
                            default=["q_proj", "k_proj", "v_proj", "o_proj"],
                            help="Which weight matrices to attach LoRA adapters to.",
                        )
                        training_config["LoRA Rank"] = lora_rank
                        training_config["Target Modules"] = target_modules
                        match ft_method:
                            case "LoRA":
                                training_config["memory_footprint"] = "fp32"
                            case "QLoRA":
                                training_config["memory_footprint"] = "bf16"

                rl_algorithm = None
                if fine_tuning_type == "RL":
                    rl_algorithm = st.selectbox(
                        "RL Algorithm",
                        options=["DPO", "PPO"],
                        index=0,
                        help=(
                            "DPO: policy + frozen reference (2× memory). "
                            "PPO: actor + reference + reward model + critic (4× memory)."
                        ),
                    )
                    training_config["RL Algorithm"] = rl_algorithm

                # --- Trainable parameter calculation ---
                arch = selected_model.architecture if selected_model is not None else {}
                trainable_params = calculate_lora_trainable_params(
                    ft_method=ft_method,
                    base_params=base_params,
                    target_modules=target_modules or [],
                    d_model=d_model or 0,
                    num_layers=num_layers or 0,
                    lora_rank=lora_rank or 0,
                    architecture=arch,
                )
                if ft_method in ("LoRA", "QLoRA") and trainable_params is not None and not arch:
                    st.caption("⚠ Approximate LoRA param count — select a preset for GQA-aware calculation.")

                if trainable_params is not None:
                    pct_of_total = (trainable_params / base_params) * 100

                    st.divider()
                    m_col1, m_col2, *m_extra = st.columns(3 if rl_algorithm else 2)
                    with m_col1:
                        st.metric(
                            label="Trainable Parameters",
                            value=_fmt_tokens(trainable_params),
                            help="Number of parameters updated during training.",
                        )
                        training_config["Trainable Parameters"] = trainable_params
                    with m_col2:
                        st.metric(
                            label="% of Total Params",
                            value=f"{pct_of_total:.2f}%",
                            help="Fraction of base model parameters that are trainable.",
                        )
                        training_config["Fraction of Total Params"] = f"{pct_of_total:.2f}%"
                    if rl_algorithm and m_extra:
                        memory_multiplier = 4 if rl_algorithm == "PPO" else 2
                        with m_extra[0]:
                            st.metric(
                                label="Memory Multiplier",
                                value=f"{memory_multiplier}×",
                                help=(
                                    f"{rl_algorithm} requires {memory_multiplier} models in memory simultaneously."
                                ),
                            )
                        training_config["Memory Multiplier"] = memory_multiplier
    return training_config