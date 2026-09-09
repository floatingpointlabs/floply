export interface MoeSpec {
  num_experts?: number;
  active_experts?: number;
  /** Params active per token. Used for FLOPs instead of parameter_count. */
  active_parameter_count?: number;
}

export interface ModelArchitecture {
  num_layers?: number;
  d_model?: number;
  num_heads?: number;
  num_kv_heads?: number;
  ffn_intermediate?: number;
  ffn_type?: string;
  vocab_size?: number;
  moe?: MoeSpec;
}

export interface ModelDefinition {
  name: string;
  slug: string;
  family: string;
  /** Total params across all experts. Checkpoints and VRAM scale with this. */
  parameter_count: number;
  architecture: ModelArchitecture;
  source: string;
  notes: string;
}

/**
 * Multiplier on `peak_flops_fp16` per numeric format.
 *
 * `null` means "use peak_flops_fp32 instead". An ABSENT key means the die has no hardware
 * support for that format — never 1.0. Falling back to the fp16 baseline is what used to
 * make fp4 on an A100 report double the real throughput, understating cost by 2x.
 */
export type PrecisionMultipliers = Record<string, number | null>;

export interface GpuHardware {
  vendor: string;
  peak_flops_fp16: number;
  peak_flops_fp32: number;
  typical_mfu: number;
  precision_multipliers: PrecisionMultipliers;
  description: string;
  source: string;
}

export interface InstanceSpec {
  display_name: string;
  gpu: string;
  gpu_count: number;
  peak_flops_fp16: number;
  peak_flops_fp32: number;
  precision_multipliers: PrecisionMultipliers;
  memory_per_gpu: number;
  total_gpu_memory: number;
  vcpus: number;
  system_memory: number;
  network_bandwidth: number;
  hourly_cost: number;
  typical_mfu: number;
  description: string;
}

export interface Provenance {
  source: "live" | "cache" | "unavailable";
  region: string;
  fetched_at: string | null;
  stale: boolean;
  age_days: number | null;
  warnings: string[];
  /** Instance types AWS offers that we refuse to price, and why. */
  quarantined: Record<string, string>;
  last_error: string | null;
  cache_location: string;
}

/**
 * Instances and storage prices for one region, ready for costing.
 *
 * Passed explicitly rather than held in a module global: pages are server-rendered, so a
 * module-level catalog would leak one visitor's region into another's request.
 */
export interface Catalog {
  region: string;
  instances: Record<string, InstanceSpec>;
  /** Instance types, most capable first. The default selection is `order[0]`. */
  order: string[];
  /** Storage class -> USD per TB per month. */
  storage: Record<string, number>;
  provenance: Provenance;
}

export interface ComputeCost {
  wall_clock_hours: number;
  gpu_hours: number;
  compute_cost: number;
  wall_clock_days: number;
}

export interface GpuMemory {
  weights_gb: number;
  activation_gb: number;
  memory_per_gpu_gb: number;
}

export type FineTuningMethod = "QLoRA" | "LoRA" | "Full Fine-Tuning" | null;
