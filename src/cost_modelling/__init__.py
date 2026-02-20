"""Floply — ML Training Cost Estimation Engine.

Core calculation and data-loading modules. Usable as a standalone Python
library independent of the Streamlit UI.
"""

from src.cost_modelling.calculator import (
    ModelConfig,
    TrainingConfig,
    CostBreakdown,
    calculate_training_flops,
    estimate_gpu_hours,
    calculate_single_run_cost,
    calculate_project_cost,
    calculate_storage_cost,
)
from src.cost_modelling.gpu_specs import AWS_GPU_INSTANCES, get_gpu_instance
from src.cost_modelling.model_loader import ModelDefinition, load_models, get_model
from src.cost_modelling.provider_loader import load_instances

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "CostBreakdown",
    "calculate_training_flops",
    "estimate_gpu_hours",
    "calculate_single_run_cost",
    "calculate_project_cost",
    "calculate_storage_cost",
    "AWS_GPU_INSTANCES",
    "get_gpu_instance",
    "ModelDefinition",
    "load_models",
    "get_model",
    "load_instances",
]
