"""Core cost calculation engine for ML training estimation.

This module implements the FLOPs-based cost estimation model for training
machine learning models on AWS GPU instances.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from src.cost_modelling.gpu_specs import get_gpu_instance, get_storage_cost



class ModelConfig(BaseModel):
    """Configuration for the ML model being trained."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    parameter_count: int = Field(
        ...,
        description="Total number of model parameters",
        gt=0,
    )
    architecture: str = Field(
        default="transformer",
        description="Model architecture family (transformer, cnn, rnn, vit, diffusion)",
    )
    # Optional: populated when user selects a known model from src/models/*.yaml.
    # When set, effective_parameter_count will use active params for MoE models.
    model_definition: Optional[Any] = Field(
        default=None,
        description="Rich architecture metadata loaded from a YAML model definition",
        exclude=True,  # not serialised — runtime only
    )

    @property
    def effective_parameter_count(self) -> int:
        """Active parameters used for FLOPs calculation.
        
        For standard models this equals parameter_count.
        For MoE models (e.g. DeepSeek V3) this returns the active parameter
        count per forward pass, as defined in the YAML model definition.
        """
        if self.model_definition is not None:
            return self.model_definition.effective_parameter_count
        return self.parameter_count

    def to_human_readable(self) -> str:
        if self.parameter_count >= 1e9:
            return f"{self.parameter_count / 1e9:.1f}B"
        elif self.parameter_count >= 1e6:
            return f"{self.parameter_count / 1e6:.0f}M"
        return f"{self.parameter_count:,}"


class TrainingConfig(BaseModel):
    """Configuration for the training run."""
    
    model_config = ConfigDict(frozen=False)
    
    training_tokens: int = Field(
        ...,
        description="Number of training tokens/samples",
        gt=0
    )
    batch_size: int = Field(
        default=1024,
        description="Total batch size across all GPUs",
        gt=0
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of gradient accumulation steps",
        ge=1
    )
    epochs: float = Field(
        default=1.0,
        description="Number of passes through the training dataset. Total tokens seen = training_tokens × epochs.",
        gt=0
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description=(
            "Whether gradient (activation) checkpointing is enabled. "
            "Recomputes activations during the backward pass to save memory, "
            "increasing FLOPs from ~6ND to ~8ND (~33% more compute). "
            "Typically required for models that don't fit in GPU memory without it."
        )
    )
    mixed_precision: str = Field(
        default="fp16",
        description="Mixed precision mode (fp16, bf16, fp32)"
    )
    instance_type: str = Field(
        default="p4d.24xlarge",
        description="AWS EC2 instance type"
    )
    num_instances: int = Field(
        default=1,
        description="Number of GPU instances",
        ge=1
    )
    mfu_override: Optional[float] = Field(
        default=None,
        description="Override Model FLOPs Utilization (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    @property
    def total_gpus(self) -> int:
        """Calculate total number of GPUs."""
        instance = get_gpu_instance(self.instance_type)
        return instance["gpu_count"] * self.num_instances
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


class CostBreakdown(BaseModel):
    """Detailed cost breakdown for a training run."""
    
    model_config = ConfigDict(frozen=False)
    
    # Compute metrics
    total_flops: float = Field(description="Total FLOPs required")
    gpu_hours: float = Field(description="Total GPU-hours needed")
    wall_clock_hours: float = Field(description="Wall-clock time in hours")
    wall_clock_days: float = Field(description="Wall-clock time in days")
    
    # Cost components
    compute_cost: float = Field(description="GPU compute cost in USD")
    storage_cost: float = Field(default=0.0, description="Data storage cost in USD")
    data_transfer_cost: float = Field(default=0.0, description="Data transfer cost in USD")
    total_cost: float = Field(description="Total cost in USD")
    
    # Instance details
    instance_type: str = Field(description="AWS instance type used")
    num_instances: int = Field(description="Number of instances")
    total_gpus: int = Field(description="Total number of GPUs")
    gpu_type: str = Field(description="GPU model name")
    
    # Efficiency metrics
    mfu: float = Field(description="Model FLOPs Utilization (0.0-1.0)")
    cost_per_tflop: float = Field(description="Cost per TFLOP in USD")
    cost_per_million_params: float = Field(description="Cost per million parameters")
    cost_per_billion_tokens: float = Field(description="Cost per billion tokens")
    
    # Training details
    model_params: int = Field(description="Number of model parameters")
    training_tokens: int = Field(description="Number of training tokens")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a summary dictionary for display."""
        return {
            "Total Cost": f"${self.total_cost:,.2f}",
            "Compute Cost": f"${self.compute_cost:,.2f}",
            "Duration": f"{self.wall_clock_days:.1f} days ({self.wall_clock_hours:.1f} hours)",
            "GPU Hours": f"{self.gpu_hours:,.0f}",
            "Instance": f"{self.num_instances}x {self.instance_type}",
            "Total GPUs": self.total_gpus,
            "GPU Type": self.gpu_type,
            "Model Size": f"{self.model_params / 1e9:.2f}B parameters" if self.model_params >= 1e9 else f"{self.model_params / 1e6:.0f}M parameters",
            "Training Data": f"{self.training_tokens / 1e9:.1f}B tokens" if self.training_tokens >= 1e9 else f"{self.training_tokens / 1e6:.0f}M tokens",
            "MFU": f"{self.mfu * 100:.1f}%",
            "Total FLOPs": f"{self.total_flops:.2e}",
        }


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


def estimate_gpu_hours(
    total_flops: float,
    instance_type: str,
    num_instances: int = 1,
    mfu_override: Optional[float] = None
) -> tuple[float, float]:
    """Estimate GPU-hours and wall-clock hours for training.
    
    Args:
        total_flops: Total FLOPs required
        instance_type: AWS instance type
        num_instances: Number of instances
        mfu_override: Override for Model FLOPs Utilization
        
    Returns:
        Tuple of (gpu_hours, wall_clock_hours)
    """
    instance = get_gpu_instance(instance_type)
    
    # Get MFU (Model FLOPs Utilization)
    mfu = mfu_override if mfu_override is not None else instance["typical_mfu"]
    
    # Calculate effective throughput per GPU
    peak_flops_per_gpu = instance["peak_flops_fp16"]
    effective_flops_per_gpu = peak_flops_per_gpu * mfu
    
    # Calculate total effective throughput
    total_gpus = instance["gpu_count"] * num_instances
    total_effective_flops = effective_flops_per_gpu * total_gpus
    
    # Calculate time
    wall_clock_hours = total_flops / total_effective_flops / 3600  # Convert seconds to hours
    gpu_hours = wall_clock_hours * total_gpus
    
    return gpu_hours, wall_clock_hours


def calculate_single_run_cost(
    model_config: ModelConfig,
    training_config: TrainingConfig
) -> CostBreakdown:
    """Calculate the cost of a single training run.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        
    Returns:
        Detailed cost breakdown
    """
    # Use effective_parameter_count so MoE models (e.g. DeepSeek V3) use
    # active params per forward pass rather than total parameter count.
    total_flops = calculate_training_flops(
        parameter_count=model_config.effective_parameter_count,
        training_tokens=training_config.training_tokens,
        architecture=model_config.architecture,
        epochs=training_config.epochs,
        gradient_checkpointing=training_config.gradient_checkpointing,
    )
    
    # Estimate GPU hours
    gpu_hours, wall_clock_hours = estimate_gpu_hours(
        total_flops=total_flops,
        instance_type=training_config.instance_type,
        num_instances=training_config.num_instances,
        mfu_override=training_config.mfu_override
    )
    
    # Get instance details
    instance = get_gpu_instance(training_config.instance_type)
    
    # Calculate compute cost
    compute_cost = wall_clock_hours * instance["hourly_cost"] * training_config.num_instances
    
    # Calculate derived metrics
    total_gpus = training_config.total_gpus
    mfu = training_config.mfu_override if training_config.mfu_override is not None else instance["typical_mfu"]
    
    cost_per_tflop = compute_cost / (total_flops / 1e12) if total_flops > 0 else 0
    cost_per_million_params = compute_cost / (model_config.parameter_count / 1e6) if model_config.parameter_count > 0 else 0
    cost_per_billion_tokens = compute_cost / (training_config.training_tokens / 1e9) if training_config.training_tokens > 0 else 0
    
    total_tokens_seen = int(training_config.training_tokens * training_config.epochs)

    return CostBreakdown(
        total_flops=total_flops,
        gpu_hours=gpu_hours,
        wall_clock_hours=wall_clock_hours,
        wall_clock_days=wall_clock_hours / 24,
        compute_cost=compute_cost,
        storage_cost=0.0,
        data_transfer_cost=0.0,
        total_cost=compute_cost,
        instance_type=training_config.instance_type,
        num_instances=training_config.num_instances,
        total_gpus=total_gpus,
        gpu_type=instance["gpu"],
        mfu=mfu,
        cost_per_tflop=cost_per_tflop,
        cost_per_million_params=cost_per_million_params,
        cost_per_billion_tokens=cost_per_billion_tokens,
        model_params=model_config.parameter_count,
        training_tokens=total_tokens_seen,
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


def calculate_project_cost(
    single_run_cost: CostBreakdown,
    num_training_runs: int = 1,
    num_hyperparameter_trials: int = 0,
    num_ablations: int = 0,
    dataset_size_tb: float = 0.0,
    storage_months: float = 3.0,
    dev_instance_hours: float = 0.0,
    dev_instance_type: str = "p4d.24xlarge"
) -> Dict[str, Any]:
    """Calculate total project cost including multiple runs and overhead.
    
    Args:
        single_run_cost: Cost breakdown for a single training run
        num_training_runs: Number of full training runs
        num_hyperparameter_trials: Number of hyperparameter tuning trials
        num_ablations: Number of ablation studies
        dataset_size_tb: Dataset size in TB
        storage_months: How long to store data
        dev_instance_hours: Development/debugging instance hours
        dev_instance_type: Instance type for dev work
        
    Returns:
        Dictionary with project-level cost breakdown
    """
    # Base training costs
    base_training_cost = single_run_cost.compute_cost * num_training_runs
    
    # Hyperparameter tuning (typically shorter runs, assume 30% of full training)
    hyperparameter_cost = single_run_cost.compute_cost * 0.3 * num_hyperparameter_trials
    
    # Ablation studies (assume 50% of full training on average)
    ablation_cost = single_run_cost.compute_cost * 0.5 * num_ablations
    
    # Storage costs
    storage_cost = calculate_storage_cost(dataset_size_tb, storage_months)
    
    # Development/debugging costs
    dev_instance = get_gpu_instance(dev_instance_type)
    dev_cost = dev_instance_hours * dev_instance["hourly_cost"]
    
    # Total costs
    total_compute_cost = base_training_cost + hyperparameter_cost + ablation_cost + dev_cost
    total_cost = total_compute_cost + storage_cost
    
    return {
        "total_cost": total_cost,
        "compute_cost": total_compute_cost,
        "base_training_cost": base_training_cost,
        "hyperparameter_cost": hyperparameter_cost,
        "ablation_cost": ablation_cost,
        "dev_cost": dev_cost,
        "storage_cost": storage_cost,
        "num_training_runs": num_training_runs,
        "num_hyperparameter_trials": num_hyperparameter_trials,
        "num_ablations": num_ablations,
        "dataset_size_tb": dataset_size_tb,
        "storage_months": storage_months,
        "total_gpu_hours": (
            single_run_cost.gpu_hours * num_training_runs +
            single_run_cost.gpu_hours * 0.3 * num_hyperparameter_trials +
            single_run_cost.gpu_hours * 0.5 * num_ablations
        ),
    }
