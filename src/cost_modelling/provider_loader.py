"""Loader for compute provider definitions from src/providers/*.yaml.

Each YAML file defines a cloud provider's GPU instances and pricing.
Adding a new provider (e.g. CoreWeave, Lambda Labs) requires only dropping
a new .yaml file into src/providers/ — no Python changes needed.
"""

from pathlib import Path
from typing import Any

import yaml

PROVIDERS_DIR = Path(__file__).parent.parent / "data" / "providers"


_FLOAT_FIELDS = {"peak_flops_fp16", "peak_flops_fp32", "hourly_cost", "typical_mfu"}
_INT_FIELDS = {
    "gpu_count",
    "memory_per_gpu",
    "total_gpu_memory",
    "vcpus",
    "system_memory",
    "network_bandwidth",
}


def load_instances() -> dict[str, dict[str, Any]]:
    """Load and merge all instance definitions from src/providers/*.yaml.

    YAML scientific notation (e.g. ``312.0e12``) is parsed as a string by
    PyYAML, so numeric fields are explicitly cast to the correct Python type.
    """
    if not PROVIDERS_DIR.exists():
        return {}

    instances: dict[str, dict[str, Any]] = {}
    for yaml_file in sorted(PROVIDERS_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        for name, spec in data.get("instances", {}).items():
            coerced = dict(spec)
            for field in _FLOAT_FIELDS:
                if field in coerced:
                    coerced[field] = float(coerced[field])
            for field in _INT_FIELDS:
                if field in coerced:
                    coerced[field] = int(coerced[field])
            instances[name] = coerced

    return instances


def load_storage_pricing() -> dict[str, float]:
    """Load S3-equivalent storage pricing from the first provider that defines it."""
    for yaml_file in sorted(PROVIDERS_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        storage = data.get("storage", {})
        if storage:
            return {k: v["cost_per_tb_month"] for k, v in storage.items()}
    return {}
