import { describe, expect, it } from "vitest";

import { buildCatalog, checkHardware, partitionByCuration } from "../engine/catalog";
import type { RegionPricing, SpecsSnapshot } from "../engine/catalog";
import { peakFlopsForPrecision, UnsupportedPrecisionError } from "../engine/gpuSpecs";
import {
  buildSettings,
  checkDrift,
  ec2PriceFilters,
  extractOnDemandPrice,
  isStale,
  networkBandwidthGbps,
  parseInstanceType,
  validateRegionPricing
} from "./pricing";

const settings = buildSettings();

const PROVENANCE = {
  source: "live" as const,
  region: "us-east-1",
  fetched_at: "2026-01-01T00:00:00.000Z",
  stale: false,
  age_days: 0,
  warnings: [],
  quarantined: {},
  last_error: null,
  cache_location: "test"
};

const hardware = (over: Record<string, any> = {}) => ({
  instance_type: "p5.48xlarge",
  gpu: "H100",
  gpu_count: 8,
  memory_per_gpu: 80,
  total_gpu_memory: 640,
  vcpus: 192,
  system_memory: 2048,
  network_bandwidth: 3200,
  current_generation: true,
  ...over
});

describe("parseInstanceType", () => {
  it("converts an NVIDIA GPU instance, in GiB", () => {
    const parsed = parseInstanceType({
      InstanceType: "p5.48xlarge",
      GpuInfo: {
        Gpus: [
          { Name: "H100", Manufacturer: "NVIDIA", Count: 8, MemoryInfo: { SizeInMiB: 81920 } }
        ],
        TotalGpuMemoryInMiB: 655360
      },
      VCpuInfo: { DefaultVCpus: 192 },
      MemoryInfo: { SizeInMiB: 2097152 },
      NetworkInfo: { NetworkCards: [{ PeakBandwidthInGbps: 3200 }] }
    } as any);

    expect(parsed).toMatchObject({ gpu: "H100", gpu_count: 8, memory_per_gpu: 80, vcpus: 192 });
    expect(parsed?.total_gpu_memory).toBe(640);
  });

  it("skips non-NVIDIA accelerators and instances with no GPU", () => {
    expect(parseInstanceType({ InstanceType: "m5.large" } as any)).toBeNull();
    expect(
      parseInstanceType({
        InstanceType: "g4ad.xlarge",
        GpuInfo: { Gpus: [{ Name: "Radeon Pro V520", Manufacturer: "AMD", Count: 1 }] }
      } as any)
    ).toBeNull();
  });
});

describe("networkBandwidthGbps", () => {
  it("prefers the structured per-card total over the free-text field", () => {
    expect(
      networkBandwidthGbps({
        NetworkCards: [{ PeakBandwidthInGbps: 400 }, { PeakBandwidthInGbps: 400 }],
        NetworkPerformance: "Up to 25 Gigabit"
      })
    ).toBe(800);
  });

  it("falls back to parsing the free text", () => {
    expect(networkBandwidthGbps({ NetworkPerformance: "Up to 25 Gigabit" })).toBe(25);
    expect(networkBandwidthGbps({ NetworkPerformance: "High" })).toBe(0);
  });
});

describe("extractOnDemandPrice", () => {
  const product = (dimensions: Record<string, any>) => ({
    terms: { OnDemand: { term: { priceDimensions: dimensions } } }
  });

  it("takes the base tier and ignores later storage tiers", () => {
    const price = extractOnDemandPrice(
      product({
        a: { unit: "GB-Mo", beginRange: "0", pricePerUnit: { USD: "0.023" } },
        b: { unit: "GB-Mo", beginRange: "51200", pricePerUnit: { USD: "0.022" } }
      }),
      "GB-Mo"
    );
    expect(price).toBe(0.023);
  });

  it("ignores a free or unparseable dimension", () => {
    expect(
      extractOnDemandPrice(product({ a: { unit: "Hrs", pricePerUnit: { USD: "0.00" } } }))
    ).toBeNull();
    expect(extractOnDemandPrice(product({ a: { unit: "Hrs", pricePerUnit: {} } }))).toBeNull();
    expect(extractOnDemandPrice({})).toBeNull();
  });
});

describe("ec2PriceFilters", () => {
  it("keeps every filter that excludes a wrong-product SKU", () => {
    const fields = Object.fromEntries(
      ec2PriceFilters("p5.48xlarge", "eu-west-1").map((f) => [f.Field, f.Value])
    );
    expect(fields).toMatchObject({
      instanceType: "p5.48xlarge",
      regionCode: "eu-west-1",
      operatingSystem: "Linux",
      capacitystatus: "Used",
      marketoption: "OnDemand",
      tenancy: "Shared",
      preInstalledSw: "NA"
    });
  });
});

describe("checkHardware", () => {
  it("accepts consistent specs", () => {
    expect(checkHardware(hardware())).toBeNull();
  });

  it("rejects a total that does not match count x per-GPU memory", () => {
    expect(checkHardware(hardware({ total_gpu_memory: 320 }))).toMatch(/does not match/);
  });

  it("rejects implausible counts and missing GPU names", () => {
    expect(checkHardware(hardware({ gpu_count: 0 }))).toMatch(/gpu_count/);
    expect(checkHardware(hardware({ gpu: "" }))).toMatch(/missing GPU model/);
    expect(checkHardware(hardware({ vcpus: 0 }))).toMatch(/vcpus/);
  });
});

describe("validateRegionPricing", () => {
  const fetched = (prices: Record<string, number>): RegionPricing => ({
    region: "us-east-1",
    fetched_at: PROVENANCE.fetched_at,
    prices,
    storage: {},
    warnings: []
  });

  it("drops a price outside the absolute bounds", () => {
    const result = validateRegionPricing(fetched({ "p5.48xlarge": 0.001 }), null, settings);
    expect(result.prices).toEqual({});
    expect(result.warnings[0]).toMatch(/outside the plausible range/);
  });

  it("keeps the cached price when the new one has drifted implausibly", () => {
    const previous = { ...fetched({ "p5.48xlarge": 66.64 }) };
    const result = validateRegionPricing(fetched({ "p5.48xlarge": 700 }), previous, settings);
    expect(result.prices["p5.48xlarge"]).toBe(66.64);
    expect(result.warnings[0]).toMatch(/jumped/);
  });

  it("passes a real AWS price cut", () => {
    const previous = { ...fetched({ "p5.48xlarge": 98.32 }) };
    const result = validateRegionPricing(fetched({ "p5.48xlarge": 55.04 }), previous, settings);
    expect(result.prices["p5.48xlarge"]).toBe(55.04);
    expect(result.warnings).toEqual([]);
  });
});

describe("checkDrift", () => {
  it("is inert without a previous value", () => {
    expect(checkDrift("x", 100, undefined, 4)).toBeNull();
    expect(checkDrift("x", 100, 0, 4)).toBeNull();
  });
});

describe("isStale", () => {
  it("treats a missing or future timestamp as stale", () => {
    expect(isStale(null, 30)).toBe(true);
    expect(isStale(new Date(Date.now() + 86_400_000).toISOString(), 30)).toBe(true);
  });

  it("respects the TTL", () => {
    expect(isStale(new Date(Date.now() - 86_400_000).toISOString(), 30)).toBe(false);
    expect(isStale(new Date(Date.now() - 40 * 86_400_000).toISOString(), 30)).toBe(true);
  });
});

describe("catalog join", () => {
  const specs: SpecsSnapshot = {
    fetched_at: PROVENANCE.fetched_at,
    instances: {
      "p5.48xlarge": hardware(),
      "p4d.24xlarge": hardware({
        instance_type: "p4d.24xlarge",
        gpu: "A100",
        memory_per_gpu: 40,
        total_gpu_memory: 320
      }),
      "p9.48xlarge": hardware({ instance_type: "p9.48xlarge", gpu: "Z900" })
    },
    warnings: []
  };

  const pricing: RegionPricing = {
    region: "us-east-1",
    fetched_at: PROVENANCE.fetched_at,
    prices: { "p5.48xlarge": 66.64, "p9.48xlarge": 99.0 },
    storage: { standard: 23.0 },
    warnings: []
  };

  it("quarantines an uncurated GPU instead of guessing its FLOPs", () => {
    const [usable, quarantined] = partitionByCuration(specs);
    expect(usable).toEqual(["p4d.24xlarge", "p5.48xlarge"]);
    expect(quarantined["p9.48xlarge"]).toMatch(/no curated FLOPs/);
  });

  it("includes only instances with a spec, a price, and curated throughput", () => {
    const catalog = buildCatalog("us-east-1", specs, pricing, PROVENANCE);
    expect(Object.keys(catalog.instances)).toEqual(["p5.48xlarge"]);
    expect(catalog.provenance.quarantined["p9.48xlarge"]).toMatch(/no curated FLOPs/);
  });

  it("carries the curated throughput and precision support onto the priced instance", () => {
    const spec = buildCatalog("us-east-1", specs, pricing, PROVENANCE).instances["p5.48xlarge"];
    expect(spec.hourly_cost).toBe(66.64);
    expect(spec.peak_flops_fp16).toBe(989.0e12);
    expect(peakFlopsForPrecision(spec, "fp8")).toBe(989.0e12 * 2);
    expect(peakFlopsForPrecision(spec, "fp32")).toBe(67.0e12);
    expect(() => peakFlopsForPrecision(spec, "fp4")).toThrow(UnsupportedPrecisionError);
  });

  it("orders by throughput, not price", () => {
    const both: RegionPricing = {
      ...pricing,
      prices: { "p5.48xlarge": 66.64, "p4d.24xlarge": 26.87 }
    };
    const catalog = buildCatalog("us-east-1", specs, both, PROVENANCE);
    expect(catalog.order).toEqual(["p5.48xlarge", "p4d.24xlarge"]);
  });
});
