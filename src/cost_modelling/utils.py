"""Utility functions for the cost model."""

from typing import Dict


def format_currency(amount: float) -> str:
    """Format amount as USD currency.
    
    Args:
        amount: Dollar amount
        
    Returns:
        Formatted string (e.g., '$1,234.56')
    """
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    else:
        return f"${amount:.2f}"


def format_number(num: float, unit: str = "") -> str:
    """Format large numbers in human-readable format.
    
    Args:
        num: Number to format
        unit: Optional unit suffix
        
    Returns:
        Formatted string
    """
    if num >= 1e12:
        return f"{num / 1e12:.2f}T{unit}"
    elif num >= 1e9:
        return f"{num / 1e9:.2f}B{unit}"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M{unit}"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K{unit}"
    else:
        return f"{num:.0f}{unit}"


def format_duration(hours: float) -> str:
    """Format duration in human-readable format.
    
    Args:
        hours: Duration in hours
        
    Returns:
        Formatted string
    """
    if hours >= 24 * 7:
        weeks = hours / (24 * 7)
        return f"{weeks:.1f} weeks"
    elif hours >= 24:
        days = hours / 24
        return f"{days:.1f} days"
    else:
        return f"{hours:.1f} hours"


def estimate_memory_requirements(
    parameter_count: int,
    batch_size: int,
    sequence_length: int,
    num_gpus: int,
    precision: str = "fp16"
) -> Dict[str, float]:
    """Estimate GPU memory requirements for training.
    
    Args:
        parameter_count: Number of model parameters
        batch_size: Batch size per GPU
        sequence_length: Sequence length
        num_gpus: Number of GPUs
        precision: Training precision (fp16, fp32)
        
    Returns:
        Dictionary with memory estimates in GB
        
    Note:
        This is a rough estimate. Actual memory usage depends on:
        - Model architecture
        - Optimizer states (Adam uses ~2x model memory)
        - Gradient checkpointing
        - Activation memory
    """
    # Bytes per parameter
    bytes_per_param = 2 if precision == "fp16" else 4
    
    # Model parameters memory
    model_memory_gb = (parameter_count * bytes_per_param) / (1024**3)
    
    # Optimizer states (Adam: 2x model parameters for momentum + variance)
    optimizer_memory_gb = model_memory_gb * 2
    
    # Gradients (same size as model)
    gradient_memory_gb = model_memory_gb
    
    # Activations (rough estimate based on batch size and sequence length)
    # Assume ~12 bytes per token per layer for transformer
    num_layers = estimate_num_layers(parameter_count)
    activation_memory_gb = (
        batch_size * sequence_length * num_layers * 12 / (1024**3)
    )
    
    # Total per GPU
    total_memory_gb = (
        model_memory_gb + optimizer_memory_gb + 
        gradient_memory_gb + activation_memory_gb
    )
    
    # Distribute across GPUs (model parallelism assumed if needed)
    per_gpu_memory_gb = total_memory_gb / num_gpus if num_gpus > 1 else total_memory_gb
    
    return {
        "model_memory_gb": model_memory_gb,
        "optimizer_memory_gb": optimizer_memory_gb,
        "gradient_memory_gb": gradient_memory_gb,
        "activation_memory_gb": activation_memory_gb,
        "total_memory_gb": total_memory_gb,
        "per_gpu_memory_gb": per_gpu_memory_gb,
    }


def estimate_num_layers(parameter_count: int) -> int:
    """Estimate number of layers based on parameter count.
    
    Args:
        parameter_count: Number of model parameters
        
    Returns:
        Estimated number of layers
    """
    # Rough heuristic based on common transformer architectures
    if parameter_count >= 70e9:
        return 80
    elif parameter_count >= 13e9:
        return 40
    elif parameter_count >= 7e9:
        return 32
    elif parameter_count >= 1e9:
        return 24
    elif parameter_count >= 350e6:
        return 12
    else:
        return 6


def estimate_optimal_batch_size(
    parameter_count: int,
    gpu_memory_gb: int,
    sequence_length: int = 2048
) -> int:
    """Estimate optimal batch size per GPU.
    
    Args:
        parameter_count: Number of model parameters
        gpu_memory_gb: Available GPU memory in GB
        sequence_length: Sequence length
        
    Returns:
        Estimated batch size per GPU
    """
    # Very rough heuristic
    # Assume 70% of memory can be used for activations
    available_for_activations = gpu_memory_gb * 0.7
    
    # Rough estimate: each sample takes ~parameter_count / 1e9 GB
    gb_per_sample = (parameter_count / 1e9) * 0.1 * (sequence_length / 2048)
    
    batch_size = int(available_for_activations / gb_per_sample)
    
    # Clamp to reasonable values
    return max(1, min(batch_size, 128))


def calculate_tokens_per_dataset_size(
    dataset_size_gb: float,
    avg_token_length_bytes: int = 2
) -> int:
    """Calculate approximate number of tokens for a dataset size.
    
    Args:
        dataset_size_gb: Dataset size in GB
        avg_token_length_bytes: Average bytes per token
        
    Returns:
        Estimated number of tokens
    """
    total_bytes = dataset_size_gb * (1024**3)
    return int(total_bytes / avg_token_length_bytes)


def calculate_dataset_size_for_tokens(
    num_tokens: int,
    avg_token_length_bytes: int = 2
) -> float:
    """Calculate dataset size in GB for a number of tokens.
    
    Args:
        num_tokens: Number of tokens
        avg_token_length_bytes: Average bytes per token
        
    Returns:
        Dataset size in GB
    """
    total_bytes = num_tokens * avg_token_length_bytes
    return total_bytes / (1024**3)
