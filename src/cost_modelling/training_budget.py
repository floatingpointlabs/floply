"""Create-A-Training-Budget pipeline — pure functions, no Streamlit.

Extracted from the mutable ``training_config`` dict that was threaded through
src/app/components/{dataset,training,compute,eval,output}_dimensions.py. Those
components gathered inputs and computed costs in the same pass, so the wiring — which
value reaches which calculator argument — was only expressed as UI code.

The maths itself is already covered by the engine fixtures; what this pins is the wiring.
"""

from dataclasses import dataclass

from src.cost_modelling.calculator import (
    calculate_checkpoint_storage_tb,
    calculate_project_compute_cost,
    calculate_storage_cost,
    calculate_training_flops,
    estimate_compute_cost,
    estimate_gpu_memory_gb,
)
from src.cost_modelling.gpu_specs import get_gpu_instance, peak_flops_for_precision


@dataclass
class TrainingBudgetInputs:
    """Everything the four form sections collect, flattened."""

    modality: str = "Text"
    dataset_size: int = 100_000_000        # samples
    tokens_per_sample: int = 520
    bytes_per_sample: int = 2080
    storage_months: int = 3

    architecture: str = "Transformer"
    # Pre-training parameter count, or 0 when fine-tuning a base model.
    parameter_count: int = 0
    # Base model totals — checkpoints and VRAM scale with these.
    base_params: int = 0
    # Base model active params — FLOPs scale with these (differs for MoE).
    base_flops_params: int = 0
    ft_method: str | None = None           # None, "Full Fine-Tuning", "LoRA", "QLoRA"
    trainable_params: int = 0
    d_model: int = 0
    num_layers: int = 0
    seq_len: int = 0
    rl_multiplier: int = 1                 # extra model copies for RL (DPO 2, PPO 4)

    epochs: int = 1
    gradient_checkpointing: bool = False
    instance_type: str = "p5.48xlarge"
    num_instances: int = 1
    mixed_precision: str = "bf16"
    mfu: float = 0.30
    batch_size: int = 8

    num_training_runs: int = 1
    num_checkpoints: int = 5
    num_hp_trials: int = 0
    hp_fraction: float = 0.30
    num_ablations: int = 0
    ablation_fraction: float = 0.50
    storage_class: str = "standard"

    @property
    def total_tokens(self) -> int:
        return int(self.tokens_per_sample * self.dataset_size)

    @property
    def dataset_size_tb(self) -> float:
        return (self.bytes_per_sample * self.dataset_size) / 1e12

    @property
    def is_adapter(self) -> bool:
        return self.ft_method in ("LoRA", "QLoRA")

    @property
    def flops_params(self) -> int:
        """Params that drive FLOPs.

        Pre-training uses its own count. Fine-tuning runs the forward/backward pass
        through the whole base model, and for MoE that means active params, not total.
        """
        if self.parameter_count:
            return int(self.parameter_count)
        return int(self.base_flops_params or self.base_params)

    @property
    def memory_params(self) -> int:
        """Params that occupy VRAM — always the total, even for MoE."""
        if self.parameter_count:
            return int(self.parameter_count)
        return int(self.base_params)

    @property
    def checkpoint_params(self) -> int:
        """Params written per checkpoint: adapters only for LoRA/QLoRA."""
        if self.is_adapter:
            return int(self.trainable_params)
        return int(self.parameter_count or self.base_params)


@dataclass
class TrainingBudget:
    """Everything the summary strip, charts and table render."""

    total_tokens: int
    dataset_size_tb: float
    total_gpus: int
    peak_flops_per_gpu: float
    hourly_cost: float
    total_flops: float
    wall_clock_hours: float
    wall_clock_days: float
    gpu_hours: float
    compute_cost: float                    # one run
    weights_gb: float
    activation_gb: float
    memory_per_gpu_gb: float
    vram_per_gpu: float
    memory_fits: bool
    checkpoint_storage_tb: float
    total_compute_cost: float              # all runs, trials and ablations
    dataset_storage_cost: float
    checkpoint_storage_cost: float
    total_project_cost: float
    total_experiment_runs: int


def estimate_training_budget(inputs: TrainingBudgetInputs) -> TrainingBudget:
    """Cost and time for a whole training project."""
    spec = get_gpu_instance(inputs.instance_type)
    total_gpus = spec["gpu_count"] * inputs.num_instances
    peak_flops_per_gpu = peak_flops_for_precision(spec, inputs.mixed_precision)

    total_flops = calculate_training_flops(
        parameter_count=inputs.flops_params,
        training_tokens=inputs.total_tokens,
        architecture=inputs.architecture.lower(),
        epochs=inputs.epochs,
        gradient_checkpointing=inputs.gradient_checkpointing,
    )
    wall_clock_hours, gpu_hours, compute_cost, wall_clock_days = estimate_compute_cost(
        total_flops=total_flops,
        peak_flops_per_gpu=peak_flops_per_gpu,
        mfu=inputs.mfu,
        total_gpus=total_gpus,
        num_instances=inputs.num_instances,
        hourly_cost=spec["hourly_cost"],
    )

    weights_gb, activation_gb, memory_per_gpu_gb = estimate_gpu_memory_gb(
        effective_params=inputs.memory_params,
        trainable_params=inputs.trainable_params or inputs.memory_params,
        ft_method=inputs.ft_method,
        total_gpus=total_gpus,
        rl_multiplier=inputs.rl_multiplier,
        d_model=inputs.d_model,
        num_layers=inputs.num_layers,
        seq_len=inputs.seq_len,
        batch_size=inputs.batch_size,
        gradient_checkpointing=inputs.gradient_checkpointing,
    )

    checkpoint_storage_tb = calculate_checkpoint_storage_tb(
        checkpoint_params=inputs.checkpoint_params,
        num_checkpoints=inputs.num_checkpoints,
        num_training_runs=inputs.num_training_runs,
        num_hp_trials=inputs.num_hp_trials,
        num_ablations=inputs.num_ablations,
    )
    total_compute_cost = calculate_project_compute_cost(
        single_run_cost=compute_cost,
        num_training_runs=inputs.num_training_runs,
        num_hp_trials=inputs.num_hp_trials,
        hp_fraction=inputs.hp_fraction,
        num_ablations=inputs.num_ablations,
        ablation_fraction=inputs.ablation_fraction,
    )
    dataset_storage_cost = calculate_storage_cost(
        inputs.dataset_size_tb, inputs.storage_months, inputs.storage_class
    )
    checkpoint_storage_cost = calculate_storage_cost(
        checkpoint_storage_tb, inputs.storage_months, inputs.storage_class
    )

    vram_per_gpu = spec["memory_per_gpu"]
    return TrainingBudget(
        total_tokens=inputs.total_tokens,
        dataset_size_tb=inputs.dataset_size_tb,
        total_gpus=total_gpus,
        peak_flops_per_gpu=peak_flops_per_gpu,
        hourly_cost=spec["hourly_cost"],
        total_flops=total_flops,
        wall_clock_hours=wall_clock_hours,
        wall_clock_days=wall_clock_days,
        gpu_hours=gpu_hours,
        compute_cost=compute_cost,
        weights_gb=weights_gb,
        activation_gb=activation_gb,
        memory_per_gpu_gb=memory_per_gpu_gb,
        vram_per_gpu=vram_per_gpu,
        memory_fits=(vram_per_gpu - memory_per_gpu_gb) >= 0,
        checkpoint_storage_tb=checkpoint_storage_tb,
        total_compute_cost=total_compute_cost,
        dataset_storage_cost=dataset_storage_cost,
        checkpoint_storage_cost=checkpoint_storage_cost,
        total_project_cost=(
            total_compute_cost + dataset_storage_cost + checkpoint_storage_cost
        ),
        total_experiment_runs=(
            inputs.num_training_runs + inputs.num_hp_trials + inputs.num_ablations
        ),
    )
