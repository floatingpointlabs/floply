"""Configuration for the AWS pricing layer.

Every environment variable the pricing layer reads is declared here, once, so
defaults are testable and discoverable rather than scattered through the code.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# src/cost_modelling/pricing/settings.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

# The Price List Query API is only served from these three regions. This is the
# API *endpoint*, unrelated to the region whose prices are being requested.
PRICING_API_REGIONS = ("us-east-1", "eu-central-1", "ap-south-1")

DEFAULT_REGIONS = (
    "us-east-1", "us-west-2", "eu-west-1",
    "eu-central-1", "ap-northeast-1", "ap-south-1",
)


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    parsed = tuple(item.strip() for item in raw.split(",") if item.strip())
    return parsed or default


@dataclass(frozen=True)
class PricingSettings:
    """Resolved pricing configuration. Build via :func:`get_settings`."""

    # -- Freshness ----------------------------------------------------------
    pricing_ttl_days: int = 30
    specs_ttl_days: int = 90          # hardware specs barely change

    # -- Cache --------------------------------------------------------------
    cache_backend: str = "file"
    cache_dir: Path = REPO_ROOT / ".cache" / "floply"

    # -- Regions ------------------------------------------------------------
    default_region: str = "us-east-1"
    regions: tuple[str, ...] = DEFAULT_REGIONS
    pricing_api_region: str = "us-east-1"
    specs_region: str = "us-east-1"

    # -- Network ------------------------------------------------------------
    timeout_s: int = 10
    connect_timeout_s: int = 3
    max_attempts: int = 3
    max_refresh_seconds: int = 120
    blocking: bool = False
    retry_backoff_s: int = 900
    startup_probe: bool = True

    # -- Guards -------------------------------------------------------------
    max_drift: float = 4.0
    min_instances: int = 3

    # -- Discovery ----------------------------------------------------------
    discover_instances: bool = True
    family_allowlist: tuple[str, ...] = ("p",)
    discover_min_gpus: int = 4

    def __post_init__(self) -> None:
        if self.pricing_api_region not in PRICING_API_REGIONS:
            raise ValueError(
                f"FLOPLY_PRICING_API_REGION must be one of "
                f"{', '.join(PRICING_API_REGIONS)} — the Price List API is not "
                f"served from {self.pricing_api_region}."
            )
        if self.default_region not in self.regions:
            raise ValueError(
                f"FLOPLY_DEFAULT_REGION {self.default_region!r} is not in "
                f"FLOPLY_AWS_REGIONS ({', '.join(self.regions)})."
            )

    def is_eligible_family(self, instance_type: str) -> bool:
        """Whether a discovered instance type is in an offered family.

        Unconstrained, DescribeInstanceTypes yields 60+ NVIDIA types including
        single-GPU g4dn/g5 sizes that make no sense in a large-model training
        estimator.
        """
        family = instance_type.split(".", 1)[0]
        return any(family.startswith(prefix) for prefix in self.family_allowlist)


def build_settings() -> PricingSettings:
    """Read settings from the environment, applying defaults."""
    regions = _list("FLOPLY_AWS_REGIONS", DEFAULT_REGIONS)
    return PricingSettings(
        pricing_ttl_days=_int("FLOPLY_PRICING_TTL_DAYS", 30),
        specs_ttl_days=_int("FLOPLY_SPECS_TTL_DAYS", 90),
        cache_backend=_str("FLOPLY_CACHE_BACKEND", "file"),
        cache_dir=Path(_str("FLOPLY_CACHE_DIR", str(REPO_ROOT / ".cache" / "floply"))),
        default_region=_str("FLOPLY_DEFAULT_REGION", regions[0]),
        regions=regions,
        pricing_api_region=_str("FLOPLY_PRICING_API_REGION", "us-east-1"),
        specs_region=_str("FLOPLY_SPECS_REGION", "us-east-1"),
        timeout_s=_int("FLOPLY_PRICING_TIMEOUT_S", 10),
        max_refresh_seconds=_int("FLOPLY_PRICING_MAX_SECONDS", 120),
        blocking=_bool("FLOPLY_PRICING_BLOCKING", False),
        retry_backoff_s=_int("FLOPLY_PRICING_RETRY_BACKOFF_S", 900),
        startup_probe=_bool("FLOPLY_STARTUP_PROBE", True),
        max_drift=_float("FLOPLY_PRICING_MAX_DRIFT", 4.0),
        min_instances=_int("FLOPLY_PRICING_MIN_INSTANCES", 3),
        discover_instances=_bool("FLOPLY_DISCOVER_INSTANCES", True),
        family_allowlist=_list("FLOPLY_INSTANCE_FAMILY_ALLOWLIST", ("p",)),
        discover_min_gpus=_int("FLOPLY_DISCOVER_MIN_GPUS", 4),
    )


@lru_cache(maxsize=1)
def get_settings() -> PricingSettings:
    """Memoised settings for the process lifetime."""
    return build_settings()


def clear_cache() -> None:
    """Drop the memoised settings. For tests that manipulate the environment."""
    get_settings.cache_clear()
