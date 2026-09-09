import { readFileSync, writeFileSync, mkdirSync, renameSync, unlinkSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  EC2Client,
  paginateDescribeInstanceTypeOfferings,
  paginateDescribeInstanceTypes,
  type InstanceTypeInfo
} from "@aws-sdk/client-ec2";
import { PricingClient, paginateGetProducts } from "@aws-sdk/client-pricing";
// Relative, not $lib: the rest of lib/ imports this way, and it keeps the module loadable
// from vitest without wiring SvelteKit's aliases into the test config.
import { buildCatalog, partitionByCuration } from "../engine/catalog";
import { fixtureCatalog } from "../engine/fixtureCatalog";
import type { InstanceHardware, RegionPricing, SpecsSnapshot } from "../engine/catalog";
import type { Catalog, Provenance } from "../engine/types";

// -- Settings ---------------------------------------------------------------

/** The Price List Query API is served only from these three regions. This is the API
 *  *endpoint*, unrelated to the region whose prices are being requested. */
export const PRICING_API_REGIONS = ["us-east-1", "eu-central-1", "ap-south-1"];

const DEFAULT_REGIONS = [
  "us-east-1",
  "us-west-2",
  "eu-west-1",
  "eu-central-1",
  "ap-northeast-1",
  "ap-south-1"
];

const str = (name: string, fallback: string) => (process.env[name] ?? fallback).trim();
const int = (name: string, fallback: number) =>
  Number.isFinite(Number(process.env[name])) && process.env[name]
    ? Number(process.env[name])
    : fallback;
const bool = (name: string, fallback: boolean) =>
  process.env[name] === undefined
    ? fallback
    : ["1", "true", "yes", "on"].includes(String(process.env[name]).trim().toLowerCase());
const list = (name: string, fallback: string[]) => {
  const parsed = (process.env[name] ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  return parsed.length ? parsed : fallback;
};

export interface PricingSettings {
  pricingTtlDays: number;
  specsTtlDays: number;
  cacheDir: string;
  defaultRegion: string;
  regions: string[];
  pricingApiRegion: string;
  specsRegion: string;
  timeoutMs: number;
  connectTimeoutMs: number;
  maxAttempts: number;
  retryBackoffMs: number;
  /** Reject a fetched price this many times off the cached one. */
  maxDrift: number;
  minInstances: number;
  familyAllowlist: string[];
  discoverMinGpus: number;
}

export function buildSettings(): PricingSettings {
  const regions = list("FLOPLY_AWS_REGIONS", DEFAULT_REGIONS);
  const pricingApiRegion = str("FLOPLY_PRICING_API_REGION", "us-east-1");
  const defaultRegion = str("FLOPLY_DEFAULT_REGION", regions[0]);

  if (!PRICING_API_REGIONS.includes(pricingApiRegion)) {
    throw new Error(
      `FLOPLY_PRICING_API_REGION must be one of ${PRICING_API_REGIONS.join(", ")} — ` +
        `the Price List API is not served from ${pricingApiRegion}.`
    );
  }
  if (!regions.includes(defaultRegion)) {
    throw new Error(
      `FLOPLY_DEFAULT_REGION "${defaultRegion}" is not in FLOPLY_AWS_REGIONS (${regions.join(", ")}).`
    );
  }

  return {
    pricingTtlDays: int("FLOPLY_PRICING_TTL_DAYS", 30),
    specsTtlDays: int("FLOPLY_SPECS_TTL_DAYS", 90),
    cacheDir: str("FLOPLY_CACHE_DIR", join(process.cwd(), ".cache", "floply")),
    defaultRegion,
    regions,
    pricingApiRegion,
    specsRegion: str("FLOPLY_SPECS_REGION", "us-east-1"),
    timeoutMs: int("FLOPLY_PRICING_TIMEOUT_S", 10) * 1000,
    connectTimeoutMs: 3000,
    maxAttempts: 3,
    retryBackoffMs: int("FLOPLY_PRICING_RETRY_BACKOFF_S", 900) * 1000,
    maxDrift: int("FLOPLY_PRICING_MAX_DRIFT", 4),
    minInstances: int("FLOPLY_PRICING_MIN_INSTANCES", 3),
    familyAllowlist: list("FLOPLY_INSTANCE_FAMILY_ALLOWLIST", ["p"]),
    discoverMinGpus: int("FLOPLY_DISCOVER_MIN_GPUS", 4)
  };
}

let cachedSettings: PricingSettings | null = null;
export function getSettings(): PricingSettings {
  return (cachedSettings ??= buildSettings());
}

// -- AWS clients ------------------------------------------------------------

const clientConfig = (region: string, s: PricingSettings) => ({
  region,
  maxAttempts: s.maxAttempts,
  requestHandler: {
    connectionTimeout: s.connectTimeoutMs,
    requestTimeout: s.timeoutMs
  }
});

const MIB_PER_GIB = 1024;

/**
 * Best available bandwidth figure, in Gbps.
 *
 * `NetworkPerformance` is free text ("400 Gigabit", "Up to 25 Gigabit"), so the structured
 * per-card values are preferred when present.
 */
export function networkBandwidthGbps(networkInfo: Record<string, any> = {}): number {
  const cards = networkInfo.NetworkCards ?? [];
  for (const key of ["PeakBandwidthInGbps", "BaselineBandwidthInGbps"]) {
    const total = cards.reduce((sum: number, card: any) => sum + (card[key] ?? 0), 0);
    if (total) return Math.round(total);
  }
  const match = /(\d+(?:\.\d+)?)\s*Gigabit/.exec(networkInfo.NetworkPerformance ?? "");
  return match ? Math.trunc(Number(match[1])) : 0;
}

/**
 * Convert one DescribeInstanceTypes entry, or null if it has no NVIDIA GPU.
 *
 * Excludes Inferentia and Trainium (which report InferenceAcceleratorInfo, not GpuInfo),
 * plus AMD (g4ad) and Habana (dl1) accelerators.
 */
export function parseInstanceType(raw: InstanceTypeInfo): InstanceHardware | null {
  const gpuInfo: any = raw.GpuInfo;
  if (!gpuInfo) return null;

  const gpus = gpuInfo.Gpus ?? [];
  if (!gpus.length || gpus[0].Manufacturer !== "NVIDIA") return null;

  const gpuCount = gpus.reduce((sum: number, gpu: any) => sum + (gpu.Count ?? 0), 0);
  if (gpuCount <= 0) return null;

  const perGpuMib = gpus[0].MemoryInfo?.SizeInMiB ?? 0;
  const totalMib = gpuInfo.TotalGpuMemoryInMiB || perGpuMib * gpuCount;

  return {
    instance_type: raw.InstanceType as string,
    gpu: gpus[0].Name,
    gpu_count: gpuCount,
    memory_per_gpu: Math.trunc(perGpuMib / MIB_PER_GIB),
    total_gpu_memory: Math.trunc(totalMib / MIB_PER_GIB),
    vcpus: raw.VCpuInfo?.DefaultVCpus ?? 0,
    system_memory: Math.trunc((raw.MemoryInfo?.SizeInMiB ?? 0) / MIB_PER_GIB),
    network_bandwidth: networkBandwidthGbps(raw.NetworkInfo as any),
    current_generation: raw.CurrentGeneration ?? true
  };
}

const isEligibleFamily = (instanceType: string, s: PricingSettings) => {
  const family = instanceType.split(".", 1)[0];
  return s.familyAllowlist.some((prefix) => family.startsWith(prefix));
};

/**
 * Enumerate NVIDIA GPU instance types and their hardware specs.
 *
 * DescribeInstanceTypes has no gpu-info filter, so all ~800 types are paginated and
 * filtered client-side.
 */
export async function fetchInstanceHardware(s: PricingSettings): Promise<SpecsSnapshot> {
  const ec2 = new EC2Client(clientConfig(s.specsRegion, s));
  const instances: Record<string, InstanceHardware> = {};
  const skippedFamily: string[] = [];
  const skippedSmall: string[] = [];

  try {
    for await (const page of paginateDescribeInstanceTypes({ client: ec2 }, {})) {
      for (const raw of page.InstanceTypes ?? []) {
        const hardware = parseInstanceType(raw);
        if (!hardware) continue;
        if (!isEligibleFamily(hardware.instance_type, s)) {
          skippedFamily.push(hardware.instance_type);
          continue;
        }
        if (hardware.gpu_count < s.discoverMinGpus) {
          skippedSmall.push(hardware.instance_type);
          continue;
        }
        instances[hardware.instance_type] = hardware;
      }
    }
  } finally {
    ec2.destroy();
  }

  const warnings: string[] = [];
  if (skippedFamily.length) {
    warnings.push(
      `${skippedFamily.length} GPU instance types outside the ` +
        `${s.familyAllowlist.join("/")} family allowlist were skipped.`
    );
  }
  if (skippedSmall.length) {
    warnings.push(
      `${skippedSmall.length} GPU instance types with fewer than ${s.discoverMinGpus} GPUs were skipped.`
    );
  }

  return { fetched_at: new Date().toISOString(), instances, warnings };
}

export async function fetchOfferedInstanceTypes(
  region: string,
  s: PricingSettings
): Promise<Set<string>> {
  const ec2 = new EC2Client(clientConfig(region, s));
  const offered = new Set<string>();
  try {
    for await (const page of paginateDescribeInstanceTypeOfferings(
      { client: ec2 },
      { LocationType: "region", Filters: [{ Name: "location", Values: [region] }] }
    )) {
      for (const offering of page.InstanceTypeOfferings ?? []) {
        if (offering.InstanceType) offered.add(offering.InstanceType);
      }
    }
  } finally {
    ec2.destroy();
  }
  return offered;
}

/**
 * Filters that reduce EC2 on-demand pricing to a single SKU. Every entry is load-bearing:
 *
 * - `capacitystatus=Used` (lowercase s — that is the literal attribute name) excludes the
 *   UnusedCapacityReservation / AllocatedCapacityReservation SKUs, separate products.
 * - `marketoption=OnDemand` excludes Capacity Block SKUs, which exist for p5/p5e and are
 *   priced per block rather than per hour. Dropping this silently returns block rates.
 * - `preInstalledSw=NA` excludes SQL Server bundles.
 * - `licenseModel` excludes BYOL variants.
 * - `tenancy=Shared` excludes Dedicated and Host.
 * - `regionCode` avoids the brittle human-readable `location` names.
 */
export function ec2PriceFilters(instanceType: string, region: string) {
  return [
    { Type: "TERM_MATCH" as const, Field: "instanceType", Value: instanceType },
    { Type: "TERM_MATCH" as const, Field: "regionCode", Value: region },
    { Type: "TERM_MATCH" as const, Field: "operatingSystem", Value: "Linux" },
    { Type: "TERM_MATCH" as const, Field: "preInstalledSw", Value: "NA" },
    { Type: "TERM_MATCH" as const, Field: "tenancy", Value: "Shared" },
    { Type: "TERM_MATCH" as const, Field: "capacitystatus", Value: "Used" },
    { Type: "TERM_MATCH" as const, Field: "licenseModel", Value: "No License required" },
    { Type: "TERM_MATCH" as const, Field: "marketoption", Value: "OnDemand" }
  ];
}

/**
 * S3 storage class -> [pricing volumeType, usagetype suffix].
 *
 * volumeType alone is not selective enough: Intelligent-Tiering exposes one SKU per access
 * tier. The usagetype suffix disambiguates, and is matched with endsWith() because
 * non-us-east-1 regions prefix it (EUC1-TimedStorage-ByteHrs).
 */
export const S3_STORAGE_CLASSES: Record<string, [string, string]> = {
  standard: ["Standard", "TimedStorage-ByteHrs"],
  intelligent_tiering: ["Intelligent-Tiering", "TimedStorage-INT-FA-ByteHrs"],
  standard_ia: ["Standard - Infrequent Access", "TimedStorage-SIA-ByteHrs"],
  one_zone_ia: ["One Zone - Infrequent Access", "TimedStorage-ZIA-ByteHrs"],
  glacier: ["Amazon Glacier", "TimedStorage-GlacierByteHrs"]
};

/** AWS bills storage per GB-month; the app works in TB-month using decimal GB, which is
 *  what makes S3 Standard's $0.023/GB read as 23.0. */
const GB_PER_TB = 1000;

/**
 * Pull the base-tier on-demand price from a Price List product.
 *
 * Only the `beginRange == "0"` dimension is used: S3 storage is tiered (50 TB / 500 TB
 * breakpoints) and the later tiers are not the headline rate.
 */
export function extractOnDemandPrice(product: any, unit = "Hrs"): number | null {
  for (const term of Object.values<any>(product?.terms?.OnDemand ?? {})) {
    for (const dimension of Object.values<any>(term?.priceDimensions ?? {})) {
      if (dimension.unit !== unit) continue;
      if (dimension.beginRange !== undefined && dimension.beginRange !== "0") continue;
      const price = Number(dimension.pricePerUnit?.USD);
      if (Number.isFinite(price) && price > 0) return price;
    }
  }
  return null;
}

async function* iterProducts(client: PricingClient, ServiceCode: string, Filters: any[]) {
  for await (const page of paginateGetProducts({ client }, { ServiceCode, Filters })) {
    for (const raw of page.PriceList ?? []) {
      yield typeof raw === "string" ? JSON.parse(raw) : raw;
    }
  }
}

async function fetchInstancePrice(
  client: PricingClient,
  instanceType: string,
  region: string
): Promise<[number | null, string | null]> {
  const prices: number[] = [];
  for await (const product of iterProducts(
    client,
    "AmazonEC2",
    ec2PriceFilters(instanceType, region)
  )) {
    const price = extractOnDemandPrice(product, "Hrs");
    if (price !== null) prices.push(price);
  }

  if (!prices.length) return [null, null];
  const min = Math.min(...prices);
  if (prices.length > 1) {
    return [
      min,
      `${instanceType}/${region}: ${prices.length} SKUs matched the on-demand filters, ` +
        `took the minimum ($${min.toFixed(2)}/hr).`
    ];
  }
  return [min, null];
}

/** S3 $/TB/month by storage class for one region. */
async function fetchStoragePrices(
  client: PricingClient,
  region: string
): Promise<[Record<string, number>, string[]]> {
  const storage: Record<string, number> = {};
  const warnings: string[] = [];

  for (const [storageClass, [volumeType, usageSuffix]] of Object.entries(S3_STORAGE_CLASSES)) {
    const candidates: number[] = [];
    const filters = [
      { Type: "TERM_MATCH" as const, Field: "regionCode", Value: region },
      { Type: "TERM_MATCH" as const, Field: "productFamily", Value: "Storage" },
      { Type: "TERM_MATCH" as const, Field: "volumeType", Value: volumeType }
    ];
    for await (const product of iterProducts(client, "AmazonS3", filters)) {
      if (!String(product?.product?.attributes?.usagetype ?? "").endsWith(usageSuffix)) continue;
      const price = extractOnDemandPrice(product, "GB-Mo");
      if (price !== null) candidates.push(price);
    }
    if (!candidates.length) {
      // AWS has renamed these volumeType strings before; a miss for one class must not
      // fail the whole region.
      warnings.push(
        `${storageClass}/${region}: no SKU matched volumeType "${volumeType}" with ` +
          `usagetype ending "${usageSuffix}".`
      );
      continue;
    }
    storage[storageClass] = Math.min(...candidates) * GB_PER_TB;
  }

  return [storage, warnings];
}

export async function fetchRegionPricing(
  region: string,
  instanceTypes: string[],
  s: PricingSettings
): Promise<RegionPricing> {
  const client = new PricingClient(clientConfig(s.pricingApiRegion, s));
  const prices: Record<string, number> = {};
  const warnings: string[] = [];

  try {
    for (const instanceType of instanceTypes) {
      const [price, warning] = await fetchInstancePrice(client, instanceType, region);
      if (warning) warnings.push(warning);
      if (price !== null) prices[instanceType] = price;
    }
    const [storage, storageWarnings] = await fetchStoragePrices(client, region);
    warnings.push(...storageWarnings);
    return { region, fetched_at: new Date().toISOString(), prices, storage, warnings };
  } finally {
    client.destroy();
  }
}

// -- Sanity -----------------------------------------------------------------

const HOURLY_COST_BOUNDS: [number, number] = [0.05, 1000.0];
const STORAGE_COST_BOUNDS: [number, number] = [0.1, 500.0];

const outOfBounds = (value: number, [low, high]: [number, number]) =>
  !Number.isFinite(value) || value < low || value > high;

/**
 * Reason the change from the cached value is suspicious, or null.
 *
 * The band is deliberately loose. AWS's p5 reduction was ~0.56x of list and must pass; a
 * filter regression picking up a Windows SKU is typically 2x+.
 */
export function checkDrift(
  label: string,
  newValue: number,
  previousValue: number | undefined,
  maxDrift: number
): string | null {
  if (previousValue === undefined || previousValue <= 0 || maxDrift <= 1) return null;
  const ratio = newValue / previousValue;
  if (ratio > maxDrift || ratio < 1 / maxDrift) {
    const direction = ratio > 1 ? "jumped" : "dropped";
    return (
      `${label}: price ${direction} ${ratio.toFixed(1)}x versus the cached value ` +
      `($${previousValue.toFixed(2)} -> $${newValue.toFixed(2)}); keeping the cached price. ` +
      `Check the on-demand filters.`
    );
  }
  return null;
}

export function validateRegionPricing(
  fetched: RegionPricing,
  previous: RegionPricing | null,
  s: PricingSettings
): RegionPricing {
  const prices: Record<string, number> = {};
  const storage: Record<string, number> = {};
  const warnings = [...fetched.warnings];

  const sift = (
    entries: Record<string, number>,
    previousEntries: Record<string, number>,
    bounds: [number, number],
    unit: string,
    out: Record<string, number>
  ) => {
    for (const [key, value] of Object.entries(entries)) {
      const cached = previousEntries[key];
      const label = `${key}/${fetched.region}`;
      let reason = outOfBounds(value, bounds)
        ? `${label}: $${value.toFixed(2)}${unit} is outside the plausible range ` +
          `$${bounds[0]}–$${bounds[1]}${unit}.`
        : checkDrift(label, value, cached, s.maxDrift);

      if (reason === null) {
        out[key] = value;
        continue;
      }
      warnings.push(reason);
      if (cached !== undefined) out[key] = cached;
    }
  };

  sift(fetched.prices, previous?.prices ?? {}, HOURLY_COST_BOUNDS, "/hr", prices);
  sift(fetched.storage, previous?.storage ?? {}, STORAGE_COST_BOUNDS, "/TB/mo", storage);

  return { ...fetched, prices, storage, warnings };
}

// -- Disk cache -------------------------------------------------------------

/** Bumping this invalidates every cached entry. The version lives in the key so old and
 *  new containers can share a volume without fighting. */
const CACHE_SCHEMA_VERSION = 1;
const SPECS_KEY = `specs_v${CACHE_SCHEMA_VERSION}`;
const regionKey = (region: string) => `catalog_v${CACHE_SCHEMA_VERSION}_${region}`;

const DAY_MS = 86_400_000;

export const ageDays = (fetchedAt: string): number =>
  Math.max(Math.floor((Date.now() - Date.parse(fetchedAt)) / DAY_MS), 0);

/** A timestamp in the future counts as stale: clock skew must not pin a cache as
 *  permanently fresh. */
export function isStale(fetchedAt: string | null, ttlDays: number): boolean {
  if (!fetchedAt) return true;
  const delta = Date.now() - Date.parse(fetchedAt);
  return !Number.isFinite(delta) || delta > ttlDays * DAY_MS || delta < 0;
}

/** Where the cache actually landed, for the provenance panel. Null once writes have failed
 *  — a read-only filesystem degrades to "fetch works, nothing persists". */
let cacheDirectory: string | null | undefined;

function resolveCacheDir(s: PricingSettings): string | null {
  if (cacheDirectory !== undefined) return cacheDirectory;
  for (const dir of [
    s.cacheDir,
    process.env.XDG_CACHE_HOME ? join(process.env.XDG_CACHE_HOME, "floply") : null,
    join(homedir(), ".cache", "floply")
  ]) {
    if (!dir) continue;
    try {
      mkdirSync(dir, { recursive: true });
      const probe = join(dir, ".write-test");
      writeFileSync(probe, "");
      unlinkSync(probe);
      return (cacheDirectory = dir);
    } catch {
      continue;
    }
  }
  console.warn("[pricing] No writable cache directory; pricing will not persist across restarts.");
  return (cacheDirectory = null);
}

function cacheRead<T>(key: string, s: PricingSettings): T | null {
  const dir = resolveCacheDir(s);
  if (!dir) return null;
  try {
    const payload = JSON.parse(readFileSync(join(dir, `${key}.json`), "utf8"));
    return payload?.schema_version === CACHE_SCHEMA_VERSION ? (payload as T) : null;
  } catch {
    return null;
  }
}

function cacheWrite(key: string, payload: object, s: PricingSettings): boolean {
  const dir = resolveCacheDir(s);
  if (!dir) return false;
  const target = join(dir, `${key}.json`);
  const tmp = `${target}.${process.pid}.tmp`;
  try {
    // Written via a temp file so a reader never observes a half-written payload and a
    // failed write never destroys a good previous value.
    writeFileSync(
      tmp,
      JSON.stringify({ ...payload, schema_version: CACHE_SCHEMA_VERSION }, null, 2)
    );
    renameSync(tmp, target);
    return true;
  } catch (error) {
    console.warn(`[pricing] Could not write cache entry ${target}: ${error}`);
    try {
      unlinkSync(tmp);
    } catch {
      /* nothing to clean up */
    }
    return false;
  }
}

// -- Resolution -------------------------------------------------------------

const unavailable = (region: string, detail: string, warnings: string[]): Catalog => ({
  region,
  instances: {},
  order: [],
  storage: {},
  provenance: {
    source: "unavailable",
    region,
    fetched_at: null,
    stale: false,
    age_days: null,
    warnings,
    quarantined: {},
    last_error: detail,
    cache_location: cacheDirectory ?? "not persisted"
  }
});

async function refreshSpecs(
  s: PricingSettings,
  force = false
): Promise<[SpecsSnapshot | null, string[]]> {
  const cached = cacheRead<SpecsSnapshot>(SPECS_KEY, s);
  if (!force && cached && !isStale(cached.fetched_at, s.specsTtlDays)) return [cached, []];

  try {
    const fetched = await fetchInstanceHardware(s);
    if (!Object.keys(fetched.instances).length) {
      const message = "AWS returned no NVIDIA GPU instance types; keeping previous specs.";
      return [cached, [message]];
    }
    cacheWrite(SPECS_KEY, fetched, s);
    return [fetched, fetched.warnings];
  } catch (error) {
    const message = `Hardware specs could not be refreshed (${error}).`;
    return cached ? [cached, [`${message} Using cached specs.`]] : [null, [message]];
  }
}

/**
 * `refreshed` says whether new prices actually landed — not merely whether something is
 * returnable. On failure this still returns the cached entry so the caller can keep
 * serving it, so the two must be reported separately: conflating them lets the background
 * backoff read a failed refresh as a success and retry a dead endpoint on every request.
 */
interface RegionRefresh {
  pricing: RegionPricing | null;
  warnings: string[];
  refreshed: boolean;
}

async function refreshRegion(
  region: string,
  instanceTypes: string[],
  s: PricingSettings,
  force = false
): Promise<RegionRefresh> {
  const cached = cacheRead<RegionPricing>(regionKey(region), s);
  if (!force && cached && !isStale(cached.fetched_at, s.pricingTtlDays)) {
    return { pricing: cached, warnings: [], refreshed: false };
  }

  let validated: RegionPricing;
  try {
    const offered = await fetchOfferedInstanceTypes(region, s);
    const wanted = offered.size ? instanceTypes.filter((t) => offered.has(t)) : instanceTypes;
    validated = validateRegionPricing(await fetchRegionPricing(region, wanted, s), cached, s);
  } catch (error) {
    return {
      pricing: cached,
      warnings: [`${region}: prices could not be refreshed (${error}).`],
      refreshed: false
    };
  }

  if (Object.keys(validated.prices).length < s.minInstances && cached) {
    return {
      pricing: cached,
      warnings: [
        ...validated.warnings,
        `${region}: only ${Object.keys(validated.prices).length} instance price(s) fetched, ` +
          `need at least ${s.minInstances}; keeping the previous cache entry.`
      ],
      refreshed: false
    };
  }

  cacheWrite(regionKey(region), validated, s);
  return { pricing: validated, warnings: validated.warnings, refreshed: true };
}

// -- Background refresh with failure backoff --------------------------------

const inFlight = new Set<string>();
const lastAttempt = new Map<string, { at: number; error: string | null }>();

function scheduleRefresh(region: string, usable: string[], s: PricingSettings): void {
  const attempt = lastAttempt.get(region);
  if (inFlight.has(region)) return;
  if (attempt?.error && Date.now() - attempt.at < s.retryBackoffMs) return;

  inFlight.add(region);
  void (async () => {
    try {
      const { warnings, refreshed } = await refreshRegion(region, usable, s, true);
      lastAttempt.set(region, {
        at: Date.now(),
        error: refreshed ? null : warnings.join("; ") || "pricing unavailable"
      });
      // Only on a real refresh: dropping the memo after a failure would just make the next
      // request re-resolve and re-schedule, which is the storm the backoff exists to stop.
      if (refreshed) catalogMemo.delete(region);
    } catch (error) {
      lastAttempt.set(region, { at: Date.now(), error: String(error) });
    } finally {
      inFlight.delete(region);
    }
  })();
}

export async function resolveCatalog(region: string, s = getSettings()): Promise<Catalog> {
  if (bool("FLOPLY_PRICING_FIXTURE", false)) {
    // Opt-in only, and never a silent fallback: e2e asserts exact dollar figures, and a
    // live rate would make those flap. A misconfigured deployment must show "unavailable"
    // rather than quietly serving numbers that look real and are not.
    console.warn("[pricing] FLOPLY_PRICING_FIXTURE is set — serving frozen fixture prices.");
    return fixtureCatalog();
  }

  let specs = cacheRead<SpecsSnapshot>(SPECS_KEY, s);
  let pricing = cacheRead<RegionPricing>(regionKey(region), s);
  const warnings: string[] = [];
  let source: Provenance["source"] = "cache";

  if (!specs || !pricing) {
    const [fetchedSpecs, specsWarnings] = await refreshSpecs(s);
    warnings.push(...specsWarnings);
    if (fetchedSpecs) {
      specs = fetchedSpecs;
      const result = await refreshRegion(region, partitionByCuration(fetchedSpecs)[0], s);
      warnings.push(...result.warnings);
      if (result.pricing) {
        pricing = result.pricing;
        if (result.refreshed) source = "live";
      }
    }
  }

  if (!specs || !pricing) {
    return unavailable(region, warnings.join("; ") || "AWS could not be reached.", warnings);
  }

  const [usable, quarantined] = partitionByCuration(specs);
  const stale = isStale(pricing.fetched_at, s.pricingTtlDays);

  if (stale && source !== "live") {
    warnings.push(
      `${region}: prices are ${ageDays(pricing.fetched_at)} days old (refresh threshold is ` +
        `${s.pricingTtlDays} days).`
    );
    scheduleRefresh(region, usable, s);
  }

  return buildCatalog(region, specs, pricing, {
    source,
    region,
    fetched_at: pricing.fetched_at,
    stale,
    age_days: ageDays(pricing.fetched_at),
    warnings: [...warnings, ...pricing.warnings],
    quarantined,
    last_error: lastAttempt.get(region)?.error ?? null,
    cache_location: cacheDirectory ?? "not persisted"
  });
}

const catalogMemo = new Map<string, { at: number; catalog: Promise<Catalog> }>();

/** How long a resolved catalog is reused before re-resolving. Not the price TTL — this only
 *  decides how often the disk cache is re-read, which is what notices a price going stale.
 *  Without it a long-lived process would serve its first resolve forever. */
const MEMO_TTL_MS = 10 * 60 * 1000;

export function getCatalog(region: string, s = getSettings()): Promise<Catalog> {
  const existing = catalogMemo.get(region);
  if (existing && Date.now() - existing.at < MEMO_TTL_MS) return existing.catalog;

  const resolved = resolveCatalog(region, s).catch((error) => {
    // A rejected promise must not stay memoised, or one blip pins the region forever.
    catalogMemo.delete(region);
    return unavailable(region, String(error), [String(error)]);
  });
  catalogMemo.set(region, { at: Date.now(), catalog: resolved });
  return resolved;
}

export function listRegions(s = getSettings()): string[] {
  return [...s.regions];
}

export const AWS_REGION_LABELS: Record<string, string> = {
  "us-east-1": "N. Virginia",
  "us-east-2": "Ohio",
  "us-west-1": "N. California",
  "us-west-2": "Oregon",
  "ca-central-1": "Central Canada",
  "eu-west-1": "Ireland",
  "eu-west-2": "London",
  "eu-west-3": "Paris",
  "eu-central-1": "Frankfurt",
  "eu-north-1": "Stockholm",
  "ap-south-1": "Mumbai",
  "ap-southeast-1": "Singapore",
  "ap-southeast-2": "Sydney",
  "ap-northeast-1": "Tokyo",
  "ap-northeast-2": "Seoul",
  "sa-east-1": "São Paulo"
};
