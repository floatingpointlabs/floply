"""Loader for model architecture definitions from src/models/*.yaml.

Each YAML file defines a known model architecture. Adding a new model to Floply
requires only dropping a new .yaml file into src/models/ — no Python changes needed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"


@dataclass
class ModelDefinition:
    """A known model architecture loaded from a YAML file."""

    name: str
    slug: str
    family: str
    parameter_count: int
    architecture: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    notes: str = ""

    @property
    def display_name(self) -> str:
        params = self.parameter_count
        if params >= 1e12:
            return f"{self.name} ({params / 1e12:.1f}T params)"
        if params >= 1e9:
            return f"{self.name} ({params / 1e9:.1f}B params)"
        return f"{self.name} ({params / 1e6:.0f}M params)"

    @property
    def effective_parameter_count(self) -> int:
        """For MoE models, return active parameters rather than total."""
        moe = self.architecture.get("moe")
        if moe and "active_parameter_count" in moe:
            return int(moe["active_parameter_count"])
        return self.parameter_count


def load_models() -> list[ModelDefinition]:
    """Load all model definitions from src/models/*.yaml, sorted by parameter count."""
    if not MODELS_DIR.exists():
        return []

    models = []
    for yaml_file in sorted(MODELS_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        models.append(
            ModelDefinition(
                name=data["name"],
                slug=data["slug"],
                family=data["family"],
                parameter_count=int(data["parameter_count"]),
                architecture=data.get("architecture", {}),
                source=data.get("source", ""),
                notes=str(data.get("notes", "")),
            )
        )

    return sorted(models, key=lambda m: m.parameter_count)
