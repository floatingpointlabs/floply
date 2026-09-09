import { GPU_ALIASES, GPU_HARDWARE, INSTANCE_OVERRIDES, MODELS } from "../data/generated";
import type { Catalog, GpuHardware, InstanceSpec, ModelDefinition } from "./types";

export { GPU_HARDWARE, MODELS };

const PRECISION_ORDER = ["fp4", "int8", "fp8", "bf16", "fp16", "tf32", "fp32"];

export class UnsupportedPrecisionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedPrecisionError";
  }
}

/**
 * Curated facts for a GPU as AWS names it, or null when uncurated.
 *
 * `GpuInfo.Gpus[].Name` is not a die identifier, so the alias table maps the strings AWS
 * actually returns ("A100-SXM4-80GB") onto one canonical entry. A GPU with no entry is
 * quarantined by the caller rather than guessed at — a made-up FLOPs figure produces a
 * confidently wrong cost.
 */
export function resolveGpu(gpuName: string): GpuHardware | null {
  const canonical = GPU_ALIASES[gpuName];
  return canonical ? GPU_HARDWARE[canonical] : null;
}

export function curatedFieldsFor(gpuName: string, instanceType = ""): Partial<InstanceSpec> | null {
  const hardware = resolveGpu(gpuName);
  if (!hardware) return null;
  return {
    peak_flops_fp16: hardware.peak_flops_fp16,
    peak_flops_fp32: hardware.peak_flops_fp32,
    typical_mfu: hardware.typical_mfu,
    precision_multipliers: { ...hardware.precision_multipliers },
    ...INSTANCE_OVERRIDES[instanceType]
  };
}

export function getGpuInstance(catalog: Catalog, instanceType: string): InstanceSpec {
  const spec = catalog.instances[instanceType];
  if (!spec) {
    const available = catalog.order.join(", ") || "none";
    throw new Error(
      `Unknown instance type "${instanceType}" in ${catalog.region}. Available: ${available}`
    );
  }
  return spec;
}

/** Get the per-TB-month cost for an S3 storage class. Region-dependent. */
export function getStorageCost(catalog: Catalog, storageClass = "standard"): number {
  const cost = catalog.storage[storageClass];
  if (cost === undefined) {
    const available = Object.keys(catalog.storage).join(", ") || "none";
    throw new Error(
      `Unknown storage class "${storageClass}" in ${catalog.region}. Available: ${available}`
    );
  }
  return cost;
}

/** Instance types, most capable first — the default selection is the first entry. */
export function listAvailableInstances(catalog: Catalog): string[] {
  return [...catalog.order];
}

export function listStorageClasses(catalog: Catalog): string[] {
  return Object.keys(catalog.storage).sort();
}

export function supportedPrecisions(instanceSpec: InstanceSpec): string[] {
  const multipliers = instanceSpec.precision_multipliers ?? {};
  return PRECISION_ORDER.filter((p) => p in multipliers);
}

/**
 * @throws {UnsupportedPrecisionError} if the die has no hardware support for the format
 *   (fp4 on Ampere or Hopper, bf16 on Volta, and so on).
 */
export function peakFlopsForPrecision(instanceSpec: InstanceSpec, mixedPrecision: string): number {
  const multipliers = instanceSpec.precision_multipliers ?? {};
  if (!(mixedPrecision in multipliers)) {
    throw new UnsupportedPrecisionError(
      `${instanceSpec.gpu || "This GPU"} has no hardware support for ${mixedPrecision}. ` +
        `Supported: ${supportedPrecisions(instanceSpec).join(", ")}`
    );
  }
  const multiplier = multipliers[mixedPrecision];
  return multiplier === null
    ? instanceSpec.peak_flops_fp32
    : instanceSpec.peak_flops_fp16 * multiplier;
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
