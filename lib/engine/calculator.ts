import { getStorageCost } from "./gpuSpecs";
import type { ComputeCost, FineTuningMethod, GpuMemory, ModelArchitecture } from "./types";

/**
 * Memory bytes per parameter by training regime:
 *   QLoRA: 4-bit quantized base (0.5 bytes) + fp16/fp32 adapter optimizer states
 *   LoRA:  fp16 frozen base (2 bytes) + fp16/fp32 adapter optimizer states
 *   Full / Pre-training: fp16 weights + fp32 Adam states (m+v) + fp32 master copy
 */
export const BYTES_PER_PARAM_QLORA_BASE = 0.5;
export const BYTES_PER_PARAM_LORA_BASE = 2;
export const BYTES_PER_PARAM_TRAINABLE = 16;
export const BYTES_PER_PARAM_FULL_FT = 16;

/** 4 tensors (QKV projections, attention scores, MLP intermediate, residual) x 2 bytes (bf16). */
export const ACTIVATION_BYTES_PER_TOKEN_PER_LAYER = 4 * 2;

/**
 * Bytes per parameter in a mixed-precision checkpoint:
 * fp16 weights (2) + fp32 Adam m (4) + fp32 Adam v (4) + fp32 master copy (4) = 14.
 */
export const BYTES_PER_PARAM_CHECKPOINT = 14;

const BASE_FLOPS_MULTIPLIERS: Record<string, number> = {
  transformer: 6.0,
  cnn: 4.0,
  rnn: 8.0,
  vit: 6.0,
  diffusion: 6.5
};

export function getFlopsMultiplier(architecture: string, gradientCheckpointing: boolean): number {
  let multiplier = BASE_FLOPS_MULTIPLIERS[architecture.toLowerCase()] ?? 6.0;
  if (gradientCheckpointing) multiplier += 2.0;
  return multiplier;
}

export function calculateTrainingFlops(
  parameterCount: number,
  trainingTokens: number,
  architecture = "transformer",
  epochs = 1.0,
  gradientCheckpointing = false
): number {
  const multiplier = getFlopsMultiplier(architecture, gradientCheckpointing);
  return multiplier * parameterCount * trainingTokens * epochs;
}

export function estimateComputeCost(
  totalFlops: number,
  peakFlopsPerGpu: number,
  mfu: number,
  totalGpus: number,
  numInstances: number,
  hourlyCost: number
): ComputeCost {
  const effectiveClusterFlops = peakFlopsPerGpu * mfu * totalGpus;
  // Python raises ZeroDivisionError here; returning zeroes keeps the UI from rendering
  // "$Infinity" if component state reaches a transient the widget minimums don't allow.
  if (effectiveClusterFlops === 0) {
    return { wall_clock_hours: 0, gpu_hours: 0, compute_cost: 0, wall_clock_days: 0 };
  }
  const wallClockSeconds = totalFlops / effectiveClusterFlops;
  const wallClockHours = wallClockSeconds / 3600;
  return {
    wall_clock_hours: wallClockHours,
    gpu_hours: wallClockHours * totalGpus,
    compute_cost: wallClockHours * hourlyCost * numInstances,
    wall_clock_days: wallClockHours / 24
  };
}

const safeCount = (value: number): number =>
  Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : 0;

export function solveForParameterCount(
  computeBudgetUsd: number,
  trainingTokens: number,
  peakFlopsPerGpu: number,
  mfu: number,
  totalGpus: number,
  numInstances: number,
  hourlyCost: number,
  architecture = "transformer",
  epochs = 1.0,
  gradientCheckpointing = false
): number {
  const multiplier = getFlopsMultiplier(architecture, gradientCheckpointing);
  const totalTokens = trainingTokens * epochs;
  const effectiveClusterFlops = peakFlopsPerGpu * mfu * totalGpus;

  if (totalTokens <= 0 || hourlyCost <= 0 || numInstances <= 0) return 0;

  const parameterCount =
    (computeBudgetUsd * effectiveClusterFlops * 3600) /
    (multiplier * totalTokens * hourlyCost * numInstances);
  return safeCount(parameterCount);
}

export function solveForTrainingTokens(
  computeBudgetUsd: number,
  parameterCount: number,
  peakFlopsPerGpu: number,
  mfu: number,
  totalGpus: number,
  numInstances: number,
  hourlyCost: number,
  architecture = "transformer",
  epochs = 1.0,
  gradientCheckpointing = false
): number {
  const multiplier = getFlopsMultiplier(architecture, gradientCheckpointing);
  const effectiveClusterFlops = peakFlopsPerGpu * mfu * totalGpus;

  if (parameterCount <= 0 || hourlyCost <= 0 || numInstances <= 0) return 0;

  const totalTokens =
    (computeBudgetUsd * effectiveClusterFlops * 3600) /
    (multiplier * parameterCount * hourlyCost * numInstances);
  return safeCount(totalTokens / Math.max(epochs, 1.0));
}

export function estimateGpuMemoryGb(
  effectiveParams: number,
  trainableParams: number,
  ftMethod: FineTuningMethod,
  totalGpus: number,
  rlMultiplier: number,
  dModel: number,
  numLayers: number,
  seqLen: number,
  batchSize: number,
  gradientCheckpointing: boolean
): GpuMemory {
  let modelMemoryBytes: number;
  if (ftMethod === "QLoRA") {
    modelMemoryBytes =
      effectiveParams * BYTES_PER_PARAM_QLORA_BASE + trainableParams * BYTES_PER_PARAM_TRAINABLE;
  } else if (ftMethod === "LoRA") {
    modelMemoryBytes =
      effectiveParams * BYTES_PER_PARAM_LORA_BASE + trainableParams * BYTES_PER_PARAM_TRAINABLE;
  } else {
    modelMemoryBytes = effectiveParams * BYTES_PER_PARAM_FULL_FT;
  }

  modelMemoryBytes *= rlMultiplier;
  if (totalGpus === 0) {
    return { weights_gb: 0, activation_gb: 0, memory_per_gpu_gb: 0 };
  }
  const weightsGb = modelMemoryBytes / totalGpus / 1e9;

  let activationBytes = 0;
  if (dModel && numLayers && seqLen) {
    activationBytes = gradientCheckpointing
      ? // With checkpointing only one layer of activations is live at a time
        batchSize * seqLen * dModel * ACTIVATION_BYTES_PER_TOKEN_PER_LAYER
      : batchSize * seqLen * dModel * numLayers * ACTIVATION_BYTES_PER_TOKEN_PER_LAYER;
  }

  const activationGb = activationBytes / totalGpus / 1e9;
  return {
    weights_gb: weightsGb,
    activation_gb: activationGb,
    memory_per_gpu_gb: weightsGb + activationGb
  };
}

export function calculateLoraTrainableParams(
  ftMethod: string,
  baseParams: number,
  targetModules: string[],
  dModel: number,
  numLayers: number,
  loraRank: number,
  architecture: ModelArchitecture
): number | null {
  if (ftMethod === "Full Fine-Tuning") return baseParams;

  if (!(targetModules.length && dModel && numLayers && loraRank)) return null;

  if (architecture && Object.keys(architecture).length > 0) {
    const d = architecture.d_model ?? dModel;
    const numHeads = architecture.num_heads ?? 0;
    const numKvHeads = architecture.num_kv_heads ?? numHeads;
    // Floor division, matching Python's `//`. Using `/` silently changes k/v_proj sizing.
    const headDim = numHeads ? Math.floor(d / numHeads) : d;
    const kvDim = numKvHeads * headDim;
    const ffn = architecture.ffn_intermediate ?? d * 4;
    const moduleDims: Record<string, [number, number]> = {
      q_proj: [d, d],
      k_proj: [d, kvDim],
      v_proj: [d, kvDim],
      o_proj: [d, d],
      up_proj: [d, ffn],
      down_proj: [ffn, d]
    };
    const total = targetModules.reduce((sum, mod) => {
      const [inD, outD] = moduleDims[mod] ?? [d, d];
      return sum + loraRank * (inD + outD);
    }, 0);
    return total * numLayers;
  }

  // Uniform d_model approximation for custom models
  return targetModules.length * 2 * loraRank * dModel * numLayers;
}

export function calculateCheckpointStorageTb(
  checkpointParams: number,
  numCheckpoints: number,
  numTrainingRuns: number,
  numHpTrials: number,
  numAblations: number
): number {
  const totalBytes =
    checkpointParams * BYTES_PER_PARAM_CHECKPOINT * numCheckpoints * numTrainingRuns +
    checkpointParams * BYTES_PER_PARAM_CHECKPOINT * numHpTrials +
    checkpointParams * BYTES_PER_PARAM_CHECKPOINT * numAblations;
  return totalBytes / 1e12;
}

export function calculateProjectComputeCost(
  singleRunCost: number,
  numTrainingRuns: number,
  numHpTrials: number,
  hpFraction: number,
  numAblations: number,
  ablationFraction: number
): number {
  return (
    singleRunCost * numTrainingRuns +
    singleRunCost * hpFraction * numHpTrials +
    singleRunCost * ablationFraction * numAblations
  );
}

/** S3 storage cost for training data. Throws on an unknown storage class. */
export function calculateStorageCost(
  datasetSizeTb: number,
  storageDurationMonths = 1.0,
  storageClass = "standard"
): number {
  const costPerTbMonth = getStorageCost(storageClass);
  return datasetSizeTb * costPerTbMonth * storageDurationMonths;
}
