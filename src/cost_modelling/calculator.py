"""Core cost calculation engine for ML training estimation.

This module implements the FLOPs-based cost estimation model for training
machine learning models on GPU instances.
"""

from src.cost_modelling.gpu_specs import get_storage_cost


def calculate_training_flops(
    parameter_count: int,
    training_tokens: int,
    architecture: str = "transformer",
    epochs: float = 1.0,
    gradient_checkpointing: bool = False,
) -> float:
    """Calculate total FLOPs required for training.
    
    Uses the standard formula: C = 6 × N × D  (Kaplan et al. 2020,
    "Scaling Laws for Neural Language Models", https://arxiv.org/abs/2001.08361)

    where:
    - N = number of parameters
    - D = total tokens seen = training_tokens × epochs
    - 6 = forward pass (2N) + backward pass (4N) per token

    With gradient checkpointing enabled, the forward pass is recomputed during
    the backward pass, increasing the multiplier from 6 to ~8:
    - C = 8 × N × D  (see Bahdanau, "The FLOPs Calculus of Language Model Training")

    Note on embedding parameters: strictly, N should count only non-embedding
    parameters (Kaplan et al.), as embedding lookups are memory reads with
    negligible FLOPs. For typical models the embedding table is < 5% of total
    parameters so the effect on the estimate is minor.
    
    Args:
        parameter_count: Total number of model parameters
        training_tokens: Number of tokens in the dataset (per epoch)
        architecture: Model architecture type
        epochs: Number of passes through the dataset (default 1.0)
        gradient_checkpointing: Whether activation checkpointing is enabled
        
    Returns:
        Total FLOPs required
    """
    # Architecture-specific base multipliers (forward + backward, no checkpointing)
    # Transformer (6×) is sourced from Kaplan et al. 2020 / Chinchilla 2022.
    # CNN and RNN multipliers are rough approximations; results may vary by architecture.
    base_multipliers = {
        "transformer": 6.0,
        "cnn": 4.0,
        "rnn": 8.0,
        "vit": 6.0,       # Vision Transformer — same derivation as decoder transformer
        "diffusion": 6.5,  # Slightly higher due to noise-prediction overhead
    }

    multiplier = base_multipliers.get(architecture.lower(), 6.0)

    # Gradient checkpointing recomputes activations in the backward pass,
    # adding an extra ~forward pass worth of FLOPs: 2N → multiplier increases by 2
    if gradient_checkpointing:
        multiplier += 2.0

    total_tokens = training_tokens * epochs
    total_flops = multiplier * parameter_count * total_tokens
    
    return total_flops


def estimate_compute_cost(
    total_flops: float,
    peak_flops_per_gpu: float,
    mfu: float,
    total_gpus: int,
    num_instances: int,
    hourly_cost: float,
) -> tuple[float, float, float, float]:
    """Estimate wall-clock time, GPU-hours, and compute cost for a training run.

    Args:
        total_flops: Total FLOPs required for the run
        peak_flops_per_gpu: Effective peak FLOPs/s per GPU for the chosen precision
        mfu: Model FLOPs Utilization (0.0–1.0)
        total_gpus: Total number of GPUs across all instances
        num_instances: Number of instances (for cost calculation)
        hourly_cost: Per-instance hourly cost in USD

    Returns:
        Tuple of (wall_clock_hours, gpu_hours, compute_cost, wall_clock_days)
    """
    effective_cluster_flops = peak_flops_per_gpu * mfu * total_gpus
    wall_clock_seconds = total_flops / effective_cluster_flops
    wall_clock_hours = wall_clock_seconds / 3600
    gpu_hours = wall_clock_hours * total_gpus
    compute_cost = wall_clock_hours * hourly_cost * num_instances
    wall_clock_days = wall_clock_hours / 24
    return wall_clock_hours, gpu_hours, compute_cost, wall_clock_days


# Memory bytes per parameter by training regime:
# QLoRA: 4-bit quantized base (0.5 bytes) + fp16/fp32 adapter optimizer states
# LoRA:  fp16 frozen base (2 bytes) + fp16/fp32 adapter optimizer states
# Full / Pre-training: fp16 weights + fp32 Adam states (m+v) + fp32 master copy ≈ 16 bytes
BYTES_PER_PARAM_QLORA_BASE = 0.5
BYTES_PER_PARAM_LORA_BASE  = 2
BYTES_PER_PARAM_TRAINABLE  = 16   # fp16 weights + fp32 Adam states + gradient buffer
BYTES_PER_PARAM_FULL_FT    = 16   # same as TRAINABLE for full fine-tuning / pre-training

# Activation memory per token per layer (bf16 transformer):
# 4 tensors (QKV projections, attention scores, MLP intermediate, residual) × 2 bytes (bf16)
ACTIVATION_BYTES_PER_TOKEN_PER_LAYER = 4 * 2


def estimate_gpu_memory_gb(
    effective_params: int,
    trainable_params: int,
    ft_method: str | None,
    total_gpus: int,
    rl_multiplier: int,
    d_model: int,
    num_layers: int,
    seq_len: int,
    batch_size: int,
    gradient_checkpointing: bool,
) -> tuple[float, float, float]:
    """Estimate GPU memory requirements for a training run.

    Args:
        effective_params: Total (or active, for MoE) model parameter count
        trainable_params: Parameters that receive gradient updates
        ft_method: "QLoRA", "LoRA", "Full Fine-Tuning", or None (pre-training)
        total_gpus: Total GPUs (memory is distributed across them)
        rl_multiplier: Extra model copies for RL (2 for DPO, 4 for PPO)
        d_model: Hidden dimension (0 to skip activation estimate)
        num_layers: Number of transformer layers (0 to skip activation estimate)
        seq_len: Sequence length in tokens (0 to skip activation estimate)
        batch_size: Global batch size
        gradient_checkpointing: Whether activation recomputation is enabled

    Returns:
        Tuple of (weights_gb, activation_gb, memory_per_gpu_gb)
    """
    if ft_method == "QLoRA":
        model_memory_bytes = (
            effective_params * BYTES_PER_PARAM_QLORA_BASE
            + trainable_params * BYTES_PER_PARAM_TRAINABLE
        )
    elif ft_method == "LoRA":
        model_memory_bytes = (
            effective_params * BYTES_PER_PARAM_LORA_BASE
            + trainable_params * BYTES_PER_PARAM_TRAINABLE
        )
    else:
        model_memory_bytes = effective_params * BYTES_PER_PARAM_FULL_FT

    model_memory_bytes *= rl_multiplier
    weights_gb = model_memory_bytes / total_gpus / 1e9

    if d_model and num_layers and seq_len:
        if gradient_checkpointing:
            # With checkpointing, only one layer of activations is live at a time
            activation_bytes = (
                batch_size * seq_len * d_model * ACTIVATION_BYTES_PER_TOKEN_PER_LAYER
            )
        else:
            activation_bytes = (
                batch_size * seq_len * d_model * num_layers
                * ACTIVATION_BYTES_PER_TOKEN_PER_LAYER
            )
    else:
        activation_bytes = 0

    activation_gb = activation_bytes / total_gpus / 1e9
    memory_per_gpu_gb = weights_gb + activation_gb
    return weights_gb, activation_gb, memory_per_gpu_gb


def calculate_lora_trainable_params(
    ft_method: str,
    base_params: int,
    target_modules: list[str],
    d_model: int,
    num_layers: int,
    lora_rank: int,
    architecture: dict,
) -> int | None:
    """Calculate the number of trainable parameters for a fine-tuning run.

    For Full Fine-Tuning returns base_params.

    For LoRA / QLoRA with a known architecture dict (non-empty), uses per-module
    dimensions to account for GQA and non-square projection matrices.

    For custom models (architecture={}), falls back to the uniform approximation:
        2 × rank × d_model × num_modules × num_layers

    Returns None when required inputs (target_modules, d_model, num_layers,
    lora_rank) are missing or zero.
    """
    if ft_method == "Full Fine-Tuning":
        return base_params

    if not (target_modules and d_model and num_layers and lora_rank):
        return None

    if architecture:
        _d        = architecture.get("d_model", d_model)
        _nh       = architecture.get("num_heads", 0)
        _nkv      = architecture.get("num_kv_heads", _nh)
        _head_dim = _d // _nh if _nh else _d
        _kv_dim   = _nkv * _head_dim
        _ffn      = architecture.get("ffn_intermediate", _d * 4)
        module_dims = {
            "q_proj":    (_d, _d),
            "k_proj":    (_d, _kv_dim),
            "v_proj":    (_d, _kv_dim),
            "o_proj":    (_d, _d),
            "up_proj":   (_d, _ffn),
            "down_proj": (_ffn, _d),
        }
        return sum(
            lora_rank * (in_d + out_d)
            for mod in target_modules
            for in_d, out_d in [module_dims.get(mod, (_d, _d))]
        ) * num_layers
    else:
        # Uniform d_model approximation for custom models
        return len(target_modules) * 2 * lora_rank * d_model * num_layers


# Bytes per parameter in a mixed-precision checkpoint:
# fp16 weights = 2 bytes
# fp32 Adam first moment (m) = 4 bytes
# fp32 Adam second moment (v) = 4 bytes
# fp32 master weight copy = 4 bytes
# Total = 14 bytes/param
BYTES_PER_PARAM_CHECKPOINT = 14


def calculate_checkpoint_storage_tb(
    checkpoint_params: int,
    num_checkpoints: int,
    num_training_runs: int,
    num_hp_trials: int,
    num_ablations: int,
) -> float:
    """Calculate total S3 storage required for model checkpoints in TB.

    Full training runs save num_checkpoints each; HP trials and ablations
    save one final checkpoint each.

    Args:
        checkpoint_params: Parameters saved per checkpoint (full model or LoRA adapters)
        num_checkpoints: Checkpoints saved per full training run
        num_training_runs: Number of full training runs
        num_hp_trials: Number of hyperparameter tuning trials
        num_ablations: Number of ablation studies

    Returns:
        Total checkpoint storage in terabytes
    """
    total_bytes = (
        checkpoint_params * BYTES_PER_PARAM_CHECKPOINT * num_checkpoints * num_training_runs
        + checkpoint_params * BYTES_PER_PARAM_CHECKPOINT * num_hp_trials
        + checkpoint_params * BYTES_PER_PARAM_CHECKPOINT * num_ablations
    )
    return total_bytes / 1e12


def calculate_project_compute_cost(
    single_run_cost: float,
    num_training_runs: int,
    num_hp_trials: int,
    hp_fraction: float,
    num_ablations: int,
    ablation_fraction: float,
) -> float:
    """Calculate total GPU compute cost across all training runs, HP trials, and ablations.

    Args:
        single_run_cost: Compute cost of one full training run in USD
        num_training_runs: Number of full training runs
        num_hp_trials: Number of hyperparameter tuning trials
        hp_fraction: Fraction of a full run's cost per HP trial
        num_ablations: Number of ablation studies
        ablation_fraction: Fraction of a full run's cost per ablation

    Returns:
        Total compute cost in USD
    """
    return (
        single_run_cost * num_training_runs
        + single_run_cost * hp_fraction * num_hp_trials
        + single_run_cost * ablation_fraction * num_ablations
    )


def calculate_storage_cost(
    dataset_size_tb: float,
    storage_duration_months: float = 1.0,
    storage_class: str = "standard"
) -> float:
    """Calculate S3 storage cost for training data.
    
    Args:
        dataset_size_tb: Dataset size in terabytes
        storage_duration_months: How long to store the data
        storage_class: S3 storage class
        
    Returns:
        Total storage cost in USD
    """
    cost_per_tb_month = get_storage_cost(storage_class)
    return dataset_size_tb * cost_per_tb_month * storage_duration_months
