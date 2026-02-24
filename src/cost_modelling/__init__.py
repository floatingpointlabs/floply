"""Floply — ML Training Cost Estimation Engine.

Core calculation and data-loading modules. Usable as a standalone Python
library independent of the Streamlit UI.
"""

from src.cost_modelling.calculator import (
    calculate_training_flops,
    estimate_compute_cost,
    estimate_gpu_memory_gb,
    calculate_lora_trainable_params,
    calculate_checkpoint_storage_tb,
    calculate_project_compute_cost,
    calculate_storage_cost,
    BYTES_PER_PARAM_CHECKPOINT,
    BYTES_PER_PARAM_QLORA_BASE,
    BYTES_PER_PARAM_LORA_BASE,
    BYTES_PER_PARAM_TRAINABLE,
    BYTES_PER_PARAM_FULL_FT,
    ACTIVATION_BYTES_PER_TOKEN_PER_LAYER,
)
from src.cost_modelling.gpu_specs import (
    AWS_GPU_INSTANCES,
    get_gpu_instance,
    get_storage_cost,
    list_available_instances,
    peak_flops_for_precision,
)
from src.cost_modelling.model_loader import ModelDefinition, load_models
from src.cost_modelling.provider_loader import load_instances
from src.cost_modelling.dataset import (
    tokens_per_text_sample,
    bytes_per_text_sample,
    tokens_per_image_sample,
    bytes_per_image_sample,
    tokens_per_audio_sample,
    bytes_per_audio_sample,
    tokens_per_video_sample,
    bytes_per_video_sample,
    WORDS_TO_BPE_TOKENS,
    TEXT_BYTES_PER_TOKEN,
    JPEG_COMPRESSION_RATIO,
    AUDIO_BYTES_PER_SECOND,
    ENCODEC_TOKENS_PER_SECOND,
    SOUNDSTREAM_TOKENS_PER_SECOND,
    WHISPER_TOKENS_PER_SECOND,
)

__all__ = [
    # calculator
    "calculate_training_flops",
    "estimate_compute_cost",
    "estimate_gpu_memory_gb",
    "calculate_lora_trainable_params",
    "calculate_checkpoint_storage_tb",
    "calculate_project_compute_cost",
    "calculate_storage_cost",
    "BYTES_PER_PARAM_CHECKPOINT",
    "BYTES_PER_PARAM_QLORA_BASE",
    "BYTES_PER_PARAM_LORA_BASE",
    "BYTES_PER_PARAM_TRAINABLE",
    "BYTES_PER_PARAM_FULL_FT",
    "ACTIVATION_BYTES_PER_TOKEN_PER_LAYER",
    # gpu_specs
    "AWS_GPU_INSTANCES",
    "get_gpu_instance",
    "get_storage_cost",
    "list_available_instances",
    "peak_flops_for_precision",
    # model_loader
    "ModelDefinition",
    "load_models",
    # provider_loader
    "load_instances",
    # dataset
    "tokens_per_text_sample",
    "bytes_per_text_sample",
    "tokens_per_image_sample",
    "bytes_per_image_sample",
    "tokens_per_audio_sample",
    "bytes_per_audio_sample",
    "tokens_per_video_sample",
    "bytes_per_video_sample",
    "WORDS_TO_BPE_TOKENS",
    "TEXT_BYTES_PER_TOKEN",
    "JPEG_COMPRESSION_RATIO",
    "AUDIO_BYTES_PER_SECOND",
    "ENCODEC_TOKENS_PER_SECOND",
    "SOUNDSTREAM_TOKENS_PER_SECOND",
    "WHISPER_TOKENS_PER_SECOND",
]
