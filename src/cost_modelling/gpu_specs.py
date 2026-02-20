"""GPU instance specifications and pricing.

Instance data is loaded from src/providers/*.yaml at import time.
To add a new provider or instance, add or edit a YAML file in src/providers/ —
no changes to this file are needed.
"""

from typing import Any

from src.cost_modelling.provider_loader import load_instances, load_storage_pricing

# Populated from src/providers/*.yaml
AWS_GPU_INSTANCES: dict[str, dict[str, Any]] = load_instances()

# Populated from src/providers/aws.yaml (storage section)
S3_STORAGE_PRICING: dict[str, float] = load_storage_pricing()

# Data transfer pricing (per TB) — kept inline as it rarely changes
DATA_TRANSFER_PRICING = {
    "data_in": 0.0,
    "data_out_first_10tb": 90.0,
    "data_out_next_40tb": 85.0,
    "data_out_next_100tb": 70.0,
}


def get_gpu_instance(instance_type: str) -> dict[str, Any]:
    """Get GPU instance specifications by instance type name."""
    if instance_type not in AWS_GPU_INSTANCES:
        available = ", ".join(AWS_GPU_INSTANCES.keys())
        raise ValueError(
            f"Unknown instance type: {instance_type}. Available: {available}"
        )
    return AWS_GPU_INSTANCES[instance_type]


def get_storage_cost(storage_class: str = "standard") -> float:
    """Get S3 storage cost per TB per month."""
    if storage_class not in S3_STORAGE_PRICING:
        available = ", ".join(S3_STORAGE_PRICING.keys())
        raise ValueError(
            f"Unknown storage class: {storage_class}. Available: {available}"
        )
    return S3_STORAGE_PRICING[storage_class]


def list_available_instances() -> list[str]:
    return list(AWS_GPU_INSTANCES.keys())


def get_cheapest_instance(min_memory_per_gpu: int = 0) -> str:
    valid = [
        (name, spec)
        for name, spec in AWS_GPU_INSTANCES.items()
        if spec["memory_per_gpu"] >= min_memory_per_gpu
    ]
    if not valid:
        raise ValueError(f"No instances with at least {min_memory_per_gpu}GB per GPU")
    return min(valid, key=lambda x: x[1]["hourly_cost"])[0]


def get_most_efficient_instance() -> str:
    """Return the instance with the highest effective FLOPs per dollar."""
    return max(
        AWS_GPU_INSTANCES.keys(),
        key=lambda x: (
            AWS_GPU_INSTANCES[x]["peak_flops_fp16"]
            * AWS_GPU_INSTANCES[x]["gpu_count"]
            * AWS_GPU_INSTANCES[x]["typical_mfu"]
        ) / AWS_GPU_INSTANCES[x]["hourly_cost"],
    )
