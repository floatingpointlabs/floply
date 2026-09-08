import {
  AWS_GPU_INSTANCES,
  INSTANCE_ORDER,
  MODELS,
  S3_STORAGE_PRICING,
} from "../data/generated";
import type { InstanceSpec, ModelDefinition } from "./types";

export { AWS_GPU_INSTANCES, INSTANCE_ORDER, MODELS, S3_STORAGE_PRICING };

export function getGpuInstance(instanceType: string): InstanceSpec {
  const spec = AWS_GPU_INSTANCES[instanceType];
  if (!spec) {
    throw new Error(
      `Unknown instance type "${instanceType}". Available: ${INSTANCE_ORDER.join(", ")}`,
    );
  }
  return spec;
}

/** Get the per-TB-month cost for an S3 storage class. */
export function getStorageCost(storageClass = "standard"): number {
  const cost = S3_STORAGE_PRICING[storageClass];
  if (cost === undefined) {
    const available = Object.keys(S3_STORAGE_PRICING).join(", ");
    throw new Error(`Unknown storage class "${storageClass}". Available: ${available}`);
  }
  return cost;
}

/** Instance types in YAML order — the default selection is the first entry. */
export function listAvailableInstances(): string[] {
  return [...INSTANCE_ORDER];
}

/**
 * Which GPUs have hardware acceleration for each low-precision format.
 *
 * FP4 is Blackwell-only (B100/B200/GB200), so no GPU currently in data/providers
 * supports it. The set is kept rather than deleted so adding a Blackwell instance is a
 * one-line change here.
 *
 * FP8 arrived with Hopper (Transformer Engine); Ampere and Volta have no FP8 path.
 * INT8 tensor cores arrived with Turing/Ampere; Volta only has the slower DP4A path.
 */
export const FP4_GPUS = new Set<string>();
export const FP8_GPUS = new Set(["H100"]);
export const INT8_TENSOR_GPUS = new Set(["A100", "H100"]);

/**
 * Effective peak FLOPs/s for an instance at a given numeric precision.
 *
 * Sources: A100 fp16/bf16 312 TFLOPS, tf32 156, fp32 19.5, int8 624 TOPS;
 * H100 fp16/bf16 989, fp8 ~1979; V100 fp16 125, int8 limited (~0.9x fp16).
 *
 * A precision the GPU cannot accelerate falls back to its FP16 rate — the honest
 * reading being "you would run this in FP16 instead". Inventing a speedup for absent
 * hardware understates cost, which is the dangerous direction for a budget estimate.
 * An unrecognised precision also falls back to fp16.
 */
export function peakFlopsForPrecision(
  instanceSpec: InstanceSpec,
  mixedPrecision: string,
): number {
  const fp16 = instanceSpec.peak_flops_fp16;
  const fp32 = instanceSpec.peak_flops_fp32;
  const gpu = instanceSpec.gpu;
  const table: Record<string, number> = {
    fp4: fp16 * (FP4_GPUS.has(gpu) ? 2.0 : 1.0),
    int8: fp16 * (INT8_TENSOR_GPUS.has(gpu) ? 2.0 : 0.9),
    fp8: fp16 * (FP8_GPUS.has(gpu) ? 2.0 : 1.0),
    bf16: fp16,
    fp16: fp16,
    tf32: fp16 * 0.5,
    fp32: fp32,
  };
  return table[mixedPrecision] ?? fp16;
}

/**
 * Active params per token — what FLOPs scale with. Differs from `model.parameter_count`
 * for MoE models, where checkpoints and VRAM still need the total.
 */
export function effectiveParameterCount(model: ModelDefinition): number {
  const active = model.architecture?.moe?.active_parameter_count;
  return active !== undefined ? Math.trunc(active) : model.parameter_count;
}

export function displayName(model: ModelDefinition): string {
  const params = model.parameter_count;
  if (params >= 1e12) return `${model.name} (${(params / 1e12).toFixed(1)}T params)`;
  if (params >= 1e9) return `${model.name} (${(params / 1e9).toFixed(1)}B params)`;
  return `${model.name} (${(params / 1e6).toFixed(0)}M params)`;
}
