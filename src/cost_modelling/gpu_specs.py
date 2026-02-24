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


def peak_flops_for_precision(instance_spec: dict, mixed_precision: str) -> float:
    """Return effective peak FLOPs/s for the given instance and numeric precision.

    Sources:
    - A100 fp16/bf16: 312 TFLOPS (NVIDIA datasheet, without sparsity)
    - A100 tf32: 156 TFLOPS (0.5× fp16)
    - A100 fp32: 19.5 TFLOPS (NVIDIA datasheet)
    - A100 int8: 624 TOPS (2× fp16 tensor cores)
    - H100 fp16/bf16: 989 TFLOPS; fp8: ~1979 TFLOPS (2× fp16)
    - V100 fp16: 125 TFLOPS; int8: limited support (~0.9× fp16)
    """
    fp16 = instance_spec["peak_flops_fp16"]
    fp32 = instance_spec["peak_flops_fp32"]
    gpu  = instance_spec["gpu"]
    return {
        "fp4":  fp16 * (2.0 if gpu in ("A100", "H100") else 1.0),
        "int8": fp16 * (2.0 if gpu in ("A100", "H100") else 0.9),
        "fp8":  fp16 * (2.0 if gpu == "H100" else 1.0),
        "bf16": fp16,
        "fp16": fp16,
        "tf32": fp16 * 0.5,
        "fp32": fp32,
    }.get(mixed_precision, fp16)

