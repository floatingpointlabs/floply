
from typing import Dict, Any
import streamlit as st
from src.app.config import INSTANCE_DISPLAY_NAMES
from src.cost_modelling.calculator import calculate_training_flops, estimate_compute_cost, estimate_gpu_memory_gb
from src.cost_modelling.gpu_specs import list_available_instances, get_gpu_instance, peak_flops_for_precision


def render_compute_dimensions(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Render the compute dimensions form.
    Args:
        training_config: The training configuration dictionary.
    Returns:
        The training configuration dictionary with the compute dimensions added.
    """
    with st.expander("Compute Config", expanded=True):
        cc_col1, cc_col2, cc_col3 = st.columns(3)

        with cc_col1:
            st.caption("Training loop")
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=1000,
                value=1,
                step=1,
                help="Passes through the dataset. Total tokens = dataset tokens × epochs.",
            )
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=16384,
                value=1024,
                step=128,
                help="Total batch size across all GPUs.",
            )
            gradient_accumulation_steps = st.number_input(
                "Gradient Accumulation Steps",
                min_value=1,
                max_value=128,
                value=1,
                step=1,
                help="Forward/backward passes before a weight update. Effective batch = batch × steps.",
            )
            gradient_checkpointing = st.checkbox(
                "Gradient Checkpointing",
                value=False,
                help="Recomputes activations during backward pass to save memory. Increases FLOPs by ~33% (6ND → 8ND).",
            )
            effective_batch = batch_size * gradient_accumulation_steps
            st.caption(f"Effective batch size: {effective_batch:,}")

        with cc_col2:
            st.caption("Compute cluster")
            available_instances = list_available_instances()
            instance_display_options = [
                INSTANCE_DISPLAY_NAMES.get(i, i) for i in available_instances
            ]
            instance_display = st.selectbox(
                "Instance Type",
                options=instance_display_options,
                index=0,
            )
            instance_type = available_instances[instance_display_options.index(instance_display)]

            num_instances = st.number_input(
                "Number of Instances",
                min_value=1,
                max_value=512,
                value=1,
                step=1,
                help="Each instance has 8 GPUs. Total GPUs = instances × 8.",
            )
            mixed_precision = st.selectbox(
                "Mixed Precision",
                options=["fp4", "int8", "fp8", "bf16", "fp16", "tf32", "fp32"],
                index=3,
                help="Numeric format for weights and activations. bf16 is standard for modern LLM training.",
            )

        with cc_col3:
            st.caption("Performance tuning")
            instance_spec = get_gpu_instance(instance_type)
            default_mfu = instance_spec["typical_mfu"]
            mfu_override = st.slider(
                "Model FLOPs Utilization (MFU)",
                min_value=5,
                max_value=100,
                value=30,
                step=5,
                format="%d%%",
                help="Percentage of theoretical peak GPU throughput actually achieved. 30% is a conservative real-world default.",
            )
            mfu_override = mfu_override / 100.0
            st.caption(f"Instance default MFU for {instance_spec['gpu']}: {default_mfu:.0%}")

        # Resolve effective params and architecture from whichever training path was taken
        if "Parameter Count" in training_config:
            effective_params = int(training_config["Parameter Count"])
            architecture = training_config.get("Model Type", "Transformer").lower()
        else:
            effective_params = int(training_config["Base Model Params"])
            architecture = "transformer"

        total_gpus = instance_spec["gpu_count"] * num_instances
        total_flops = calculate_training_flops(
            parameter_count=effective_params,
            training_tokens=training_config["Total tokens"],
            architecture=architecture,
            epochs=epochs,
            gradient_checkpointing=gradient_checkpointing,
        )
        wall_clock_hours, gpu_hours, compute_cost, wall_clock_days = estimate_compute_cost(
            total_flops=total_flops,
            peak_flops_per_gpu=peak_flops_for_precision(instance_spec, mixed_precision),
            mfu=mfu_override,
            total_gpus=total_gpus,
            num_instances=num_instances,
            hourly_cost=instance_spec["hourly_cost"],
        )

        training_config["Epochs"] = epochs
        training_config["Batch Size"] = batch_size
        training_config["Gradient Accumulation Steps"] = gradient_accumulation_steps
        training_config["Effective Batch Size"] = effective_batch
        training_config["Gradient Checkpointing"] = gradient_checkpointing
        training_config["Instance Type"] = instance_type
        training_config["Num Instances"] = num_instances
        training_config["Total GPUs"] = total_gpus
        training_config["Mixed Precision"] = mixed_precision
        training_config["MFU"] = mfu_override
        training_config["Total FLOPs"] = total_flops
        training_config["GPU-hours"] = gpu_hours
        training_config["Wall-clock Days"] = wall_clock_days
        training_config["Compute Cost (USD)"] = compute_cost

        weights_gb, activation_gb, memory_per_gpu_gb = estimate_gpu_memory_gb(
            effective_params=effective_params,
            trainable_params=training_config.get("Trainable Parameters", effective_params),
            ft_method=training_config.get("Fine-Tuning Method"),
            total_gpus=total_gpus,
            rl_multiplier=training_config.get("Memory Multiplier", 1),
            d_model=training_config.get("d_model") or 0,
            num_layers=training_config.get("num_layers") or 0,
            seq_len=training_config.get("Tokens per sample") or 0,
            batch_size=batch_size,
            gradient_checkpointing=gradient_checkpointing,
        )

        vram_per_gpu = instance_spec["memory_per_gpu"]
        headroom_gb = vram_per_gpu - memory_per_gpu_gb

        training_config["Est. Memory per GPU (GB)"] = round(memory_per_gpu_gb, 2)
        training_config["VRAM per GPU (GB)"] = vram_per_gpu
        training_config["Memory Fits"] = headroom_gb >= 0

        if headroom_gb < 0:
            st.warning(
                f"Estimated GPU memory ({memory_per_gpu_gb:.1f} GB/GPU) exceeds "
                f"{instance_spec['gpu']} VRAM ({vram_per_gpu} GB/GPU) by "
                f"{abs(headroom_gb):.1f} GB. Consider more instances, gradient checkpointing, or QLoRA."
            )
        else:
            act_note = f" (weights: {weights_gb:.1f} GB + activations: {activation_gb:.1f} GB)" if activation_gb > 0 else ""
            st.success(
                f"GPU memory fits: {memory_per_gpu_gb:.1f} GB / {vram_per_gpu} GB per GPU{act_note} "
                f"({headroom_gb:.1f} GB headroom)."
            )

        st.divider()
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            st.metric(
                label="Total FLOPs",
                value=f"{total_flops:.2e}",
                help="Estimated floating-point operations for the full training run.",
            )
        with r_col2:
            if gpu_hours >= 1_000_000:
                gpu_hours_label = f"{gpu_hours / 1_000_000:.2f}M"
            elif gpu_hours >= 1_000:
                gpu_hours_label = f"{gpu_hours / 1_000:.1f}K"
            else:
                gpu_hours_label = f"{gpu_hours:.1f}"
            st.metric(
                label="GPU-hours",
                value=gpu_hours_label,
                help="Total GPU-hours consumed across all devices.",
            )
        with r_col3:
            if wall_clock_days >= 1:
                time_label = f"{wall_clock_days:.2f} days"
            elif wall_clock_days >= 1 / 24:
                time_label = f"{wall_clock_hours:.2f} hrs"
            else:
                time_label = f"{wall_clock_hours * 60:.1f} min"
            st.metric(
                label="Wall-clock Time",
                value=time_label,
            )
        with r_col4:
            mem_delta = f"{headroom_gb:+.1f} GB headroom"
            if activation_gb > 0:
                mem_delta += f" | wts {weights_gb:.1f} + act {activation_gb:.1f} GB"
            st.metric(
                label="GPU Memory / GPU",
                value=f"{memory_per_gpu_gb:.1f} GB",
                delta=mem_delta,
                delta_color="normal" if headroom_gb >= 0 else "inverse",
            )
    return training_config
