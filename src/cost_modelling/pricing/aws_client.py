"""AWS Price List and EC2 clients.

Everything here is read-only. The required IAM policy is:

    pricing:GetProducts
    ec2:DescribeInstanceTypes
    ec2:DescribeInstanceTypeOfferings

None of those support resource-level constraints, so ``"Resource": "*"`` is
both required and correct.

boto3 is imported lazily inside each function so the cost-modelling library
stays importable without it.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterator

from src.cost_modelling.pricing.models import (
    InstanceHardware,
    ProbeResult,
    RegionPricing,
    SpecsSnapshot,
)
from src.cost_modelling.pricing.settings import PricingSettings, get_settings

log = logging.getLogger(__name__)

EC2_SERVICE_CODE = "AmazonEC2"
S3_SERVICE_CODE = "AmazonS3"

# S3 storage class -> (pricing volumeType, usagetype suffix).
#
# volumeType alone is not selective enough: Intelligent-Tiering exposes one SKU
# per access tier. The usagetype suffix disambiguates, and is matched with
# endswith() because non-us-east-1 regions prefix it (EUC1-TimedStorage-ByteHrs).
S3_STORAGE_CLASSES: dict[str, tuple[str, str]] = {
    "standard":            ("Standard",                     "TimedStorage-ByteHrs"),
    "intelligent_tiering": ("Intelligent-Tiering",           "TimedStorage-INT-FA-ByteHrs"),
    "standard_ia":         ("Standard - Infrequent Access",  "TimedStorage-SIA-ByteHrs"),
    "one_zone_ia":         ("One Zone - Infrequent Access",  "TimedStorage-ZIA-ByteHrs"),
    "glacier":             ("Amazon Glacier",                "TimedStorage-GlacierByteHrs"),
}

# AWS bills storage per GB-month; the app works in TB-month using decimal GB,
# which is what makes S3 Standard's $0.023/GB read as 23.0.
GB_PER_TB = 1000

_GIB_PER_MIB = 1024


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _client(service: str, region: str, settings: PricingSettings):
    """Build a boto3 client with hard timeouts and bounded retries."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        service,
        region_name=region,
        config=Config(
            connect_timeout=settings.connect_timeout_s,
            read_timeout=settings.timeout_s,
            retries={"max_attempts": settings.max_attempts, "mode": "standard"},
            user_agent_extra="floply",
        ),
    )


# -- EC2 hardware specs -----------------------------------------------------

def _network_bandwidth_gbps(network_info: dict[str, Any]) -> int:
    """Best available bandwidth figure, in Gbps.

    ``NetworkPerformance`` is free text ("400 Gigabit", "Up to 25 Gigabit"), so
    the structured per-card values are preferred when present.
    """
    cards = network_info.get("NetworkCards") or []
    for key in ("PeakBandwidthInGbps", "BaselineBandwidthInGbps"):
        total = sum(card.get(key) or 0 for card in cards)
        if total:
            return int(round(total))

    match = re.search(r"(\d+(?:\.\d+)?)\s*Gigabit", network_info.get("NetworkPerformance", ""))
    return int(float(match.group(1))) if match else 0


def parse_instance_type(raw: dict[str, Any]) -> InstanceHardware | None:
    """Convert one DescribeInstanceTypes entry, or None if it has no NVIDIA GPU.

    Excludes Inferentia and Trainium (which report InferenceAcceleratorInfo,
    not GpuInfo), plus AMD (g4ad) and Habana (dl1) accelerators.
    """
    gpu_info = raw.get("GpuInfo")
    if not gpu_info:
        return None

    gpus = gpu_info.get("Gpus") or []
    if not gpus or gpus[0].get("Manufacturer") != "NVIDIA":
        return None

    first = gpus[0]
    gpu_count = sum(gpu.get("Count", 0) for gpu in gpus)
    if gpu_count <= 0:
        return None

    per_gpu_mib = (first.get("MemoryInfo") or {}).get("SizeInMiB", 0)
    total_mib = gpu_info.get("TotalGpuMemoryInMiB") or per_gpu_mib * gpu_count

    return InstanceHardware(
        instance_type=raw["InstanceType"],
        gpu=first["Name"],
        gpu_count=gpu_count,
        memory_per_gpu=per_gpu_mib // _GIB_PER_MIB,
        total_gpu_memory=total_mib // _GIB_PER_MIB,
        vcpus=(raw.get("VCpuInfo") or {}).get("DefaultVCpus", 0),
        system_memory=(raw.get("MemoryInfo") or {}).get("SizeInMiB", 0) // _GIB_PER_MIB,
        network_bandwidth=_network_bandwidth_gbps(raw.get("NetworkInfo") or {}),
        current_generation=raw.get("CurrentGeneration", True),
    )


def fetch_instance_hardware(settings: PricingSettings | None = None) -> SpecsSnapshot:
    """Enumerate NVIDIA GPU instance types and their hardware specs.

    DescribeInstanceTypes has no gpu-info filter, so all ~800 types are
    paginated and filtered client-side.
    """
    settings = settings or get_settings()
    ec2 = _client("ec2", settings.specs_region, settings)

    instances: dict[str, InstanceHardware] = {}
    skipped_small: list[str] = []
    skipped_family: list[str] = []

    for page in ec2.get_paginator("describe_instance_types").paginate():
        for raw in page["InstanceTypes"]:
            hardware = parse_instance_type(raw)
            if hardware is None:
                continue
            if settings.discover_instances:
                if not settings.is_eligible_family(hardware.instance_type):
                    skipped_family.append(hardware.instance_type)
                    continue
                if hardware.gpu_count < settings.discover_min_gpus:
                    skipped_small.append(hardware.instance_type)
                    continue
            instances[hardware.instance_type] = hardware

    warnings: list[str] = []
    if skipped_family:
        warnings.append(
            f"{len(skipped_family)} GPU instance types outside the "
            f"{'/'.join(settings.family_allowlist)} family allowlist were skipped."
        )
    if skipped_small:
        warnings.append(
            f"{len(skipped_small)} GPU instance types with fewer than "
            f"{settings.discover_min_gpus} GPUs were skipped."
        )

    return SpecsSnapshot(fetched_at=_now(), instances=instances, warnings=tuple(warnings))


def fetch_offered_instance_types(
    region: str,
    settings: PricingSettings | None = None,
) -> set[str]:
    """Instance types actually offered in a region.

    More reliable than inferring availability from a pricing miss, which
    conflates "not sold here" with "our filters are wrong".
    """
    settings = settings or get_settings()
    ec2 = _client("ec2", region, settings)

    offered: set[str] = set()
    paginator = ec2.get_paginator("describe_instance_type_offerings")
    for page in paginator.paginate(
        LocationType="region",
        Filters=[{"Name": "location", "Values": [region]}],
    ):
        for offering in page["InstanceTypeOfferings"]:
            offered.add(offering["InstanceType"])
    return offered


# -- Pricing ----------------------------------------------------------------

def ec2_price_filters(instance_type: str, region: str) -> list[dict[str, str]]:
    """Filters that reduce EC2 on-demand pricing to a single SKU.

    Every entry is load-bearing:

    - ``capacitystatus=Used`` (lowercase s — that is the literal attribute
      name) excludes the UnusedCapacityReservation / AllocatedCapacityReservation
      SKUs, which are separate products.
    - ``marketoption=OnDemand`` excludes Capacity Block SKUs, which exist for
      p5/p5e and are priced per block rather than per hour. Dropping this
      silently returns capacity-block rates.
    - ``preInstalledSw=NA`` excludes SQL Server bundles.
    - ``licenseModel`` excludes BYOL variants.
    - ``tenancy=Shared`` excludes Dedicated and Host.
    - ``regionCode`` avoids the brittle human-readable ``location`` names.
    """
    return [
        {"Type": "TERM_MATCH", "Field": "instanceType",    "Value": instance_type},
        {"Type": "TERM_MATCH", "Field": "regionCode",      "Value": region},
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
        {"Type": "TERM_MATCH", "Field": "preInstalledSw",  "Value": "NA"},
        {"Type": "TERM_MATCH", "Field": "tenancy",         "Value": "Shared"},
        {"Type": "TERM_MATCH", "Field": "capacitystatus",  "Value": "Used"},
        {"Type": "TERM_MATCH", "Field": "licenseModel",    "Value": "No License required"},
        {"Type": "TERM_MATCH", "Field": "marketoption",    "Value": "OnDemand"},
    ]


def s3_price_filters(region: str, volume_type: str) -> list[dict[str, str]]:
    return [
        {"Type": "TERM_MATCH", "Field": "regionCode",    "Value": region},
        {"Type": "TERM_MATCH", "Field": "productFamily", "Value": "Storage"},
        {"Type": "TERM_MATCH", "Field": "volumeType",    "Value": volume_type},
    ]


def extract_on_demand_price(product: dict[str, Any], unit: str = "Hrs") -> float | None:
    """Pull the base-tier on-demand price from a Price List product.

    Only the ``beginRange == "0"`` dimension is used: S3 storage is tiered
    (50 TB / 500 TB breakpoints) and the later tiers are not the headline rate.
    """
    for term in (product.get("terms", {}).get("OnDemand") or {}).values():
        for dimension in (term.get("priceDimensions") or {}).values():
            if dimension.get("unit") != unit:
                continue
            if dimension.get("beginRange") not in (None, "0"):
                continue
            try:
                price = float(dimension["pricePerUnit"]["USD"])
            except (KeyError, TypeError, ValueError):
                continue
            if price > 0:
                return price
    return None


def _iter_products(pricing, service_code: str, filters: list[dict[str, str]]) -> Iterator[dict]:
    """Yield parsed products; the API returns each as a JSON string."""
    for page in pricing.get_paginator("get_products").paginate(
        ServiceCode=service_code, Filters=filters
    ):
        for raw in page["PriceList"]:
            yield json.loads(raw) if isinstance(raw, str) else raw


def fetch_instance_price(
    pricing,
    instance_type: str,
    region: str,
) -> tuple[float | None, str | None]:
    """On-demand $/hour for one instance in one region.

    Returns (price, warning). Multiple surviving SKUs indicate a filter
    regression, so the minimum is taken and reported rather than silently
    picking the first or averaging.
    """
    prices = [
        price
        for product in _iter_products(pricing, EC2_SERVICE_CODE,
                                      ec2_price_filters(instance_type, region))
        if (price := extract_on_demand_price(product, unit="Hrs")) is not None
    ]

    if not prices:
        return None, None
    if len(prices) > 1:
        return min(prices), (
            f"{instance_type}/{region}: {len(prices)} SKUs matched the on-demand "
            f"filters, took the minimum (${min(prices):.2f}/hr)."
        )
    return prices[0], None


def fetch_storage_prices(pricing, region: str) -> tuple[dict[str, float], list[str]]:
    """S3 $/TB/month by storage class for one region."""
    storage: dict[str, float] = {}
    warnings: list[str] = []

    for storage_class, (volume_type, usage_suffix) in S3_STORAGE_CLASSES.items():
        candidates = [
            price
            for product in _iter_products(pricing, S3_SERVICE_CODE,
                                          s3_price_filters(region, volume_type))
            if str(product.get("product", {})
                   .get("attributes", {})
                   .get("usagetype", "")).endswith(usage_suffix)
            and (price := extract_on_demand_price(product, unit="GB-Mo")) is not None
        ]
        if not candidates:
            # AWS has renamed these volumeType strings before; a miss for one
            # class must not fail the whole region.
            warnings.append(
                f"{storage_class}/{region}: no SKU matched volumeType "
                f"{volume_type!r} with usagetype ending {usage_suffix!r}."
            )
            continue
        storage[storage_class] = min(candidates) * GB_PER_TB

    return storage, warnings


def fetch_region_pricing(
    region: str,
    instance_types: list[str],
    settings: PricingSettings | None = None,
) -> RegionPricing:
    """Fetch on-demand instance prices and S3 storage costs for one region."""
    settings = settings or get_settings()
    pricing = _client("pricing", settings.pricing_api_region, settings)

    prices: dict[str, float] = {}
    warnings: list[str] = []

    for instance_type in instance_types:
        price, warning = fetch_instance_price(pricing, instance_type, region)
        if warning:
            warnings.append(warning)
        if price is not None:
            prices[instance_type] = price

    storage, storage_warnings = fetch_storage_prices(pricing, region)
    warnings.extend(storage_warnings)

    return RegionPricing(
        region=region,
        fetched_at=_now(),
        prices=prices,
        storage=storage,
        offered=tuple(sorted(prices)),
        warnings=tuple(warnings),
    )


# -- Startup probe ----------------------------------------------------------

def probe_credentials(settings: PricingSettings | None = None) -> ProbeResult:
    """One cheap call to surface credential and IAM problems at deploy time.

    Without this, a misconfigured policy is discovered by a user several form
    sections deep, since fetching is otherwise fully lazy.
    """
    settings = settings or get_settings()
    try:
        from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
    except ImportError:
        return ProbeResult(False, "boto3 is not installed.")

    try:
        pricing = _client("pricing", settings.pricing_api_region, settings)
        pricing.get_products(ServiceCode=EC2_SERVICE_CODE, Filters=[], MaxResults=1)
    except (NoCredentialsError, PartialCredentialsError):
        return ProbeResult(False, "No AWS credentials found.")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation"}:
            return ProbeResult(
                False,
                f"Credentials lack pricing:GetProducts ({code}).",
                missing_action="pricing:GetProducts",
            )
        return ProbeResult(False, f"AWS returned {code or 'an error'}.")
    except Exception as exc:                      # noqa: BLE001 - probe must never raise
        return ProbeResult(False, f"{type(exc).__name__}: {exc}")

    return ProbeResult(True, f"Pricing API reachable via {settings.pricing_api_region}.")
