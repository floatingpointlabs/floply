/**
 * Prices move; the engine's arithmetic must not. These are the us-east-1 on-demand rates
 * that were checked into data/providers/aws.yaml before pricing went live, kept purely as a
 * test baseline — live pricing never reads them; FLOPLY_PRICING_FIXTURE is what serves them
 * (see lib/server/pricing.ts).
 */
import { buildCatalog, type RegionPricing, type SpecsSnapshot } from "./catalog";
import type { Catalog } from "./types";

const FETCHED_AT = "2026-02-13T00:00:00.000Z";

const hardware = (
  instance_type: string,
  gpu: string,
  memory_per_gpu: number,
  vcpus: number,
  system_memory: number,
  network_bandwidth: number
) => ({
  instance_type,
  gpu,
  gpu_count: 8,
  memory_per_gpu,
  total_gpu_memory: memory_per_gpu * 8,
  vcpus,
  system_memory,
  network_bandwidth,
  current_generation: true
});

export const FIXTURE_SPECS: SpecsSnapshot = {
  fetched_at: FETCHED_AT,
  instances: {
    "p4d.24xlarge": hardware("p4d.24xlarge", "A100", 40, 96, 1152, 400),
    "p4de.24xlarge": hardware("p4de.24xlarge", "A100", 80, 96, 1152, 400),
    "p5.48xlarge": hardware("p5.48xlarge", "H100", 80, 192, 2048, 3200),
    "p3.16xlarge": hardware("p3.16xlarge", "V100", 16, 64, 488, 25),
    "p3dn.24xlarge": hardware("p3dn.24xlarge", "V100", 32, 96, 768, 100)
  },
  warnings: []
};

export const FIXTURE_PRICING: RegionPricing = {
  region: "us-east-1",
  fetched_at: FETCHED_AT,
  prices: {
    "p4d.24xlarge": 26.87,
    "p4de.24xlarge": 32.77,
    "p5.48xlarge": 66.64,
    "p3.16xlarge": 24.48,
    "p3dn.24xlarge": 31.22
  },
  storage: {
    standard: 23.0,
    intelligent_tiering: 23.0,
    standard_ia: 13.8,
    one_zone_ia: 11.0,
    glacier: 4.0
  },
  warnings: []
};

export function fixtureCatalog(): Catalog {
  return buildCatalog("us-east-1", FIXTURE_SPECS, FIXTURE_PRICING, {
    source: "cache",
    region: "us-east-1",
    fetched_at: FETCHED_AT,
    stale: false,
    age_days: 0,
    warnings: ["Fixture pricing — not live AWS rates."],
    quarantined: {},
    last_error: null,
    cache_location: "fixture"
  });
}
