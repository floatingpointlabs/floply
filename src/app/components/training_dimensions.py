import streamlit as st
from typing import Dict, Any, Tuple
from src.app.helpers import _cached_models, _fmt_tokens

def render_training_dimensions(training_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Render the compute dimensions form.
    Args:
        training_config: The training configuration dictionary.
    Returns:
        The training configuration dictionary with the compute dimensions added.
    """
    training_type = st.selectbox("Training Type", options=["Pre-Training", "Fine-Tuning"], index=None)
    training_config["Training Type"] = training_type
    
    if training_type:
        st.write(f"Training Type: {training_type}")
        if training_type == "Pre-Training":
            with st.expander("Pre-Training Configuration", expanded=True):
                # model_config = render_model_config_form()
                pre_col1, pre_col2 = st.columns(2)
                with pre_col1:
                    model_type = st.selectbox("Model Type", options=["Transformer", "CNN", "RNN", "ViT", "Diffusion"], index=0)
                with pre_col2:
                    parameter_count = st.number_input(
                        "Parameter Count",
                        value=None,
                        min_value=1_000_000,
                        max_value=10_000_000_000_000,
                        step=1_000_000_000,
                        format="%d",
                        placeholder="e.g. 7000000000",
                        help="Total number of model parameters.",
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
                    
        elif training_type == "Fine-Tuning":
            fine_tuning_type = st.selectbox("Fine-Tuning Type", options=["Supervised", "RL"], index=None)
            training_config["Fine-Tuning Type"] = fine_tuning_type
            if fine_tuning_type:
                with st.expander("Fine-Tuning Configuration", expanded=True):
                    known_models = _cached_models()
                    preset_options = [m.display_name for m in known_models] + ["Custom"]

                    ft_col1, ft_col2 = st.columns(2)
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
                            base_params = st.number_input(
                                "Parameter Count",
                                min_value=1_000_000,
                                max_value=1_000_000_000_000,
                                value=7_000_000_000,
                                step=1_000_000_000,
                                format="%d",
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

                    if base_model_name:
                        training_config["Base Model Params"] = base_params
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
                        if ft_method == "Full Fine-Tuning":
                            trainable_params = base_params
                        else:
                            if target_modules and d_model and num_layers and lora_rank:
                                if selected_model is not None:
                                    # Per-module dims using actual architecture (accounts for GQA)
                                    arch       = selected_model.architecture
                                    _d         = arch.get("d_model", d_model)
                                    _nh        = arch.get("num_heads", 0)
                                    _nkv       = arch.get("num_kv_heads", _nh)
                                    _head_dim  = _d // _nh if _nh else _d
                                    _kv_dim    = _nkv * _head_dim
                                    _ffn       = arch.get("ffn_intermediate", _d * 4)
                                    _module_dims = {
                                        "q_proj":   (_d, _d),
                                        "k_proj":   (_d, _kv_dim),
                                        "v_proj":   (_d, _kv_dim),
                                        "o_proj":   (_d, _d),
                                        "up_proj":  (_d, _ffn),
                                        "down_proj": (_ffn, _d),
                                    }
                                    trainable_params = sum(
                                        lora_rank * (in_d + out_d)
                                        for mod in target_modules
                                        for in_d, out_d in [_module_dims.get(mod, (_d, _d))]
                                    ) * num_layers
                                else:
                                    # Custom model: approximate formula (uniform d_model)
                                    trainable_params = len(target_modules) * 2 * lora_rank * d_model * num_layers
                                    st.caption("⚠ Approximate LoRA param count — select a preset for GQA-aware calculation.")
                            else:
                                trainable_params = None

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



    # Pre-Training needs a parameter count; Fine-Tuning needs a base model selected
    # (signalled by "Base Model Params" being stored in training_config)
    can_compute = (
        "Total tokens" in training_config
        and (
            "Parameter Count" in training_config       # pre-training path
            or "Base Model Params" in training_config  # fine-tuning path
        )
    )
    return training_config, can_compute