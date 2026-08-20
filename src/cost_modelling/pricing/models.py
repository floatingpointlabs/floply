"""Data structures for fetched AWS pricing and hardware specs.

Deliberately plain: these mirror what AWS returns, before any join with the
curated GPU facts in src/data/gpu_hardware.yaml.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class InstanceHardware:
    """Hardware specs for one GPU instance type, from ec2:DescribeInstanceTypes."""

    instance_type: str
    gpu: str
    gpu_count: int
    memory_per_gpu: int          # GB
    total_gpu_memory: int        # GB
    vcpus: int
    system_memory: int           # GB
    network_bandwidth: int       # Gbps
    current_generation: bool = True

    def as_spec(self) -> dict[str, Any]:
        """The instance-spec dict shape the rest of the app consumes."""
        return {
            "gpu": self.gpu,
            "gpu_count": self.gpu_count,
            "memory_per_gpu": self.memory_per_gpu,
            "total_gpu_memory": self.total_gpu_memory,
            "vcpus": self.vcpus,
            "system_memory": self.system_memory,
            "network_bandwidth": self.network_bandwidth,
            "current_generation": self.current_generation,
            "display_name": self.display_name,
            "description": self.description,
        }

    @property
    def display_name(self) -> str:
        return (
            f"{self.instance_type} "
            f"({self.gpu_count}x {self.gpu} {self.memory_per_gpu}GB)"
        )

    @property
    def description(self) -> str:
        return f"NVIDIA {self.gpu} {self.memory_per_gpu}GB ({self.gpu_count} GPUs)"


@dataclass(frozen=True)
class SpecsSnapshot:
    """All discovered GPU instance hardware, region-agnostic."""

    fetched_at: datetime
    instances: dict[str, InstanceHardware] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegionPricing:
    """On-demand prices and storage costs for one region."""

    region: str
    fetched_at: datetime
    prices: dict[str, float] = field(default_factory=dict)        # instance -> $/hr
    storage: dict[str, float] = field(default_factory=dict)       # class -> $/TB/mo
    offered: tuple[str, ...] = ()                                 # types available here
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of the cheap startup credential/permission check."""

    ok: bool
    detail: str
    missing_action: str | None = None
