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

export interface InstanceSpec {
  display_name: string;
  gpu: string;
  gpu_count: number;
  peak_flops_fp16: number;
  peak_flops_fp32: number;
  memory_per_gpu: number;
  total_gpu_memory: number;
  vcpus: number;
  system_memory: number;
  network_bandwidth: number;
  hourly_cost: number;
  typical_mfu: number;
  description: string;
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
