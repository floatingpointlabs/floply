import { curatedFieldsFor, resolveGpu } from "./gpuSpecs";
import type { Catalog, InstanceSpec, Provenance } from "./types";

/** Hardware specs for one GPU instance type, as ec2:DescribeInstanceTypes reports them. */
export interface InstanceHardware {
  instance_type: string;
  gpu: string;
  gpu_count: number;
  memory_per_gpu: number;
  total_gpu_memory: number;
  vcpus: number;
  system_memory: number;
  network_bandwidth: number;
  current_generation: boolean;
}

export interface SpecsSnapshot {
  fetched_at: string;
  instances: Record<string, InstanceHardware>;
  warnings: string[];
}

export interface RegionPricing {
  region: string;
  fetched_at: string;
  /** instance type -> USD/hour */
  prices: Record<string, number>;
  /** storage class -> USD/TB/month */
  storage: Record<string, number>;
  warnings: string[];
}

const uncuratedReason = (gpu: string) =>
  `no curated FLOPs for GPU "${gpu}". Add it to data/gpu_hardware.yaml.`;

export function checkHardware(hardware: InstanceHardware): string | null {
  const name = hardware.instance_type;
  if (!Number.isFinite(hardware.gpu_count) || hardware.gpu_count < 1 || hardware.gpu_count > 64) {
    return `${name}: implausible gpu_count ${hardware.gpu_count}.`;
  }
  if (
    !Number.isFinite(hardware.memory_per_gpu) ||
    hardware.memory_per_gpu < 1 ||
    hardware.memory_per_gpu > 1024
  ) {
    return `${name}: implausible memory_per_gpu ${hardware.memory_per_gpu} GB.`;
  }
  if (hardware.vcpus < 1) return `${name}: implausible vcpus ${hardware.vcpus}.`;
  if (!hardware.gpu) return `${name}: missing GPU model name.`;

  // Tolerance of one GB per GPU absorbs the integer division on MiB totals.
  const expected = hardware.gpu_count * hardware.memory_per_gpu;
  if (Math.abs(hardware.total_gpu_memory - expected) > hardware.gpu_count) {
    return (
      `${name}: total_gpu_memory ${hardware.total_gpu_memory} GB does not match ` +
      `${hardware.gpu_count} x ${hardware.memory_per_gpu} GB.`
    );
  }
  return null;
}

export function partitionByCuration(snapshot: SpecsSnapshot): [string[], Record<string, string>] {
  const usable: string[] = [];
  const quarantined: Record<string, string> = {};

  for (const [name, hardware] of Object.entries(snapshot.instances)) {
    const defect = checkHardware(hardware);
    if (defect) quarantined[name] = defect;
    else if (!resolveGpu(hardware.gpu)) quarantined[name] = uncuratedReason(hardware.gpu);
    else usable.push(name);
  }

  return [usable.sort(), quarantined];
}

function orderInstances(instances: Record<string, InstanceSpec>): string[] {
  return Object.keys(instances).sort(
    (a, b) =>
      instances[b].peak_flops_fp16 - instances[a].peak_flops_fp16 ||
      instances[b].gpu_count - instances[a].gpu_count ||
      instances[a].hourly_cost - instances[b].hourly_cost ||
      a.localeCompare(b)
  );
}

export function buildCatalog(
  region: string,
  specs: SpecsSnapshot,
  pricing: RegionPricing,
  provenance: Provenance
): Catalog {
  const instances: Record<string, InstanceSpec> = {};
  const quarantined = { ...provenance.quarantined };

  for (const [instanceType, hardware] of Object.entries(specs.instances)) {
    const price = pricing.prices[instanceType];
    if (price === undefined) continue;

    const curated = curatedFieldsFor(hardware.gpu, instanceType);
    if (!curated) {
      quarantined[instanceType] ??= uncuratedReason(hardware.gpu);
      continue;
    }

    instances[instanceType] = {
      gpu: hardware.gpu,
      gpu_count: hardware.gpu_count,
      memory_per_gpu: hardware.memory_per_gpu,
      total_gpu_memory: hardware.total_gpu_memory,
      vcpus: hardware.vcpus,
      system_memory: hardware.system_memory,
      network_bandwidth: hardware.network_bandwidth,
      display_name: `${instanceType} (${hardware.gpu_count}x ${hardware.gpu} ${hardware.memory_per_gpu}GB)`,
      description: `NVIDIA ${hardware.gpu} ${hardware.memory_per_gpu}GB (${hardware.gpu_count} GPUs)`,
      hourly_cost: price,
      ...curated
    } as InstanceSpec;
  }

  return {
    region,
    instances,
    order: orderInstances(instances),
    storage: { ...pricing.storage },
    provenance: { ...provenance, quarantined }
  };
}
