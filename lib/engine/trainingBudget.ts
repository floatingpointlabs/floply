import {
  calculateCheckpointStorageTb,
  calculateProjectComputeCost,
  calculateStorageCost,
  calculateTrainingFlops,
  estimateComputeCost,
  estimateGpuMemoryGb
} from "./calculator";
import { getGpuInstance, peakFlopsForPrecision } from "./gpuSpecs";
import type { FineTuningMethod } from "./types";

export interface TrainingBudgetInputs {
  modality?: string;
  dataset_size?: number; // samples
  tokens_per_sample?: number;
  bytes_per_sample?: number;
  storage_months?: number;

  architecture?: string;
  /** Pre-training parameter count, or 0 when fine-tuning a base model. */
  parameter_count?: number;
  /** Base model total — checkpoints and VRAM scale with this. */
  base_params?: number;
  /** Base model active params — FLOPs scale with this (differs for MoE). */
  base_flops_params?: number;
  ft_method?: FineTuningMethod;
  trainable_params?: number;
  d_model?: number;
  num_layers?: number;
  seq_len?: number;
  /** Extra model copies for RL (DPO 2, PPO 4). */
  rl_multiplier?: number;

  epochs?: number;
  gradient_checkpointing?: boolean;
  instance_type?: string;
  num_instances?: number;
  mixed_precision?: string;
  mfu?: number;
  batch_size?: number;

  num_training_runs?: number;
  num_checkpoints?: number;
  num_hp_trials?: number;
  hp_fraction?: number;
  num_ablations?: number;
  ablation_fraction?: number;
  storage_class?: string;
}

export interface TrainingBudget {
  total_tokens: number;
  dataset_size_tb: number;
  total_gpus: number;
  peak_flops_per_gpu: number;
  hourly_cost: number;
  total_flops: number;
  wall_clock_hours: number;
  wall_clock_days: number;
  gpu_hours: number;
  /** One run. */
  compute_cost: number;
  weights_gb: number;
  activation_gb: number;
  memory_per_gpu_gb: number;
  vram_per_gpu: number;
  memory_fits: boolean;
  checkpoint_storage_tb: number;
  /** All runs, trials and ablations. */
  total_compute_cost: number;
  dataset_storage_cost: number;
  checkpoint_storage_cost: number;
  total_project_cost: number;
  total_experiment_runs: number;
}

const DEFAULTS = {
  modality: "Text",
  dataset_size: 100_000_000,
  tokens_per_sample: 520,
  bytes_per_sample: 2080,
  storage_months: 3,
  architecture: "Transformer",
  parameter_count: 0,
  base_params: 0,
  base_flops_params: 0,
  ft_method: null as FineTuningMethod,
  trainable_params: 0,
  d_model: 0,
  num_layers: 0,
  seq_len: 0,
  rl_multiplier: 1,
  epochs: 1,
  gradient_checkpointing: false,
  instance_type: "p5.48xlarge",
  num_instances: 1,
  mixed_precision: "bf16",
  mfu: 0.3,
  batch_size: 8,
  num_training_runs: 1,
  num_checkpoints: 5,
  num_hp_trials: 0,
  hp_fraction: 0.3,
  num_ablations: 0,
  ablation_fraction: 0.5,
  storage_class: "standard"
};

const totalTokens = (i: Required<TrainingBudgetInputs>): number =>
  Math.trunc(i.tokens_per_sample * i.dataset_size);

const datasetSizeTb = (i: Required<TrainingBudgetInputs>): number =>
  (i.bytes_per_sample * i.dataset_size) / 1e12;

const isAdapter = (ftMethod: FineTuningMethod): boolean =>
  ftMethod === "LoRA" || ftMethod === "QLoRA";

export function estimateTrainingBudget(raw: TrainingBudgetInputs): TrainingBudget {
  const i = { ...DEFAULTS, ...raw } as Required<TrainingBudgetInputs>;

  // Pre-training uses its own count. Fine-tuning runs the forward/backward pass through
  // the whole base model, and for MoE that means active params, not total.
  const flopsParams = i.parameter_count
    ? Math.trunc(i.parameter_count)
    : Math.trunc(i.base_flops_params || i.base_params);
  // Params that occupy VRAM — always the total, even for MoE.
  const memoryParams = Math.trunc(i.parameter_count || i.base_params);
  // Params written per checkpoint: adapters only for LoRA/QLoRA.
  const checkpointParams = isAdapter(i.ft_method) ? Math.trunc(i.trainable_params) : memoryParams;

  const spec = getGpuInstance(i.instance_type);
  const totalGpus = spec.gpu_count * i.num_instances;
  const peakFlopsPerGpu = peakFlopsForPrecision(spec, i.mixed_precision);
  const tokens = totalTokens(i);
  const sizeTb = datasetSizeTb(i);

  const totalFlops = calculateTrainingFlops(
    flopsParams,
    tokens,
    i.architecture.toLowerCase(),
    i.epochs,
    i.gradient_checkpointing
  );
  const run = estimateComputeCost(
    totalFlops,
    peakFlopsPerGpu,
    i.mfu,
    totalGpus,
    i.num_instances,
    spec.hourly_cost
  );

  const memory = estimateGpuMemoryGb(
    memoryParams,
    i.trainable_params || memoryParams,
    i.ft_method,
    totalGpus,
    i.rl_multiplier,
    i.d_model,
    i.num_layers,
    i.seq_len,
    i.batch_size,
    i.gradient_checkpointing
  );

  const checkpointStorageTb = calculateCheckpointStorageTb(
    checkpointParams,
    i.num_checkpoints,
    i.num_training_runs,
    i.num_hp_trials,
    i.num_ablations
  );
  const totalComputeCost = calculateProjectComputeCost(
    run.compute_cost,
    i.num_training_runs,
    i.num_hp_trials,
    i.hp_fraction,
    i.num_ablations,
    i.ablation_fraction
  );
  const datasetStorageCost = calculateStorageCost(sizeTb, i.storage_months, i.storage_class);
  const checkpointStorageCost = calculateStorageCost(
    checkpointStorageTb,
    i.storage_months,
    i.storage_class
  );

  return {
    total_tokens: tokens,
    dataset_size_tb: sizeTb,
    total_gpus: totalGpus,
    peak_flops_per_gpu: peakFlopsPerGpu,
    hourly_cost: spec.hourly_cost,
    total_flops: totalFlops,
    wall_clock_hours: run.wall_clock_hours,
    wall_clock_days: run.wall_clock_days,
    gpu_hours: run.gpu_hours,
    compute_cost: run.compute_cost,
    weights_gb: memory.weights_gb,
    activation_gb: memory.activation_gb,
    memory_per_gpu_gb: memory.memory_per_gpu_gb,
    vram_per_gpu: spec.memory_per_gpu,
    memory_fits: spec.memory_per_gpu - memory.memory_per_gpu_gb >= 0,
    checkpoint_storage_tb: checkpointStorageTb,
    total_compute_cost: totalComputeCost,
    dataset_storage_cost: datasetStorageCost,
    checkpoint_storage_cost: checkpointStorageCost,
    total_project_cost: totalComputeCost + datasetStorageCost + checkpointStorageCost,
    total_experiment_runs: i.num_training_runs + i.num_hp_trials + i.num_ablations
  };
}
