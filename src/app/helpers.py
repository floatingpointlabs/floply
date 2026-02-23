import streamlit as st
from typing import Any
from src.cost_modelling.model_loader import load_models


@st.cache_data
def _cached_models():
    return load_models()


def _chinchilla_banner(training_config: dict) -> None:
    """Render a scaling-law health banner above the Compute Config.

    Pre-training uses Chinchilla (Hoffmann et al. 2022): D* ≈ 20 x N.
    Full fine-tuning uses a softer 1-5 tok/param rule (base model already converged).
    LoRA / QLoRA use trainable-adapter params with task-specific data thresholds
    (adapter literature: 100k-10M tokens is typically sufficient).

    Thresholds vary by training method — see inline comments.
    """
    total_tokens = training_config.get("Total tokens", 0)
    ft_method    = training_config.get("Fine-Tuning Method")          # None for pre-training
    base_params  = training_config.get("Base Model Params", 0)
    pre_params   = training_config.get("Parameter Count", 0)
    adapter_params = training_config.get("Trainable Parameters", 0)

    if total_tokens <= 0:
        return

    # ── Pre-Training ──────────────────────────────────────────────────────────
    if ft_method is None:
        num_params = int(pre_params)
        if num_params <= 0:
            return
        ratio = total_tokens / num_params
        optimal_tokens = num_params * 20

        if ratio < 1:
            st.error(
                f"**Dataset critically undersized.** "
                f"{ratio:.2f} tok/param — fewer tokens than parameters. "
                f"Chinchilla-optimal requires **{_fmt_tokens(optimal_tokens)} tokens** "
                f"(20x N). Training will likely diverge or severely underfit."
            )
        elif ratio < 10:
            st.warning(
                f"**Undertrained (below Chinchilla-optimal).** "
                f"{ratio:.1f} tok/param. Need ≥ 20 tok/param for compute efficiency — "
                f"at least **{_fmt_tokens(optimal_tokens)} tokens**. "
                f"Consider more data or a smaller model."
            )
        elif ratio <= 30:
            st.success(
                f"**Chinchilla-optimal.** "
                f"{ratio:.1f} tok/param — within the 10-30x compute-optimal zone "
                f"(Hoffmann et al. 2022)."
            )
        elif ratio <= 200:
            st.info(
                f"**Inference-optimal (over-trained vs Chinchilla).** "
                f"{ratio:.1f} tok/param. Training smaller models longer reduces "
                f"inference cost — the LLaMA / Mistral strategy. Fine if you expect "
                f"high inference volume."
            )
        else:
            st.warning(
                f"**Heavily over-trained relative to model size.** "
                f"{ratio:.0f} tok/param (> 200x). Diminishing returns; "
                f"consider scaling the model up."
            )

    # ── Full Fine-Tuning ──────────────────────────────────────────────────────
    elif ft_method == "Full Fine-Tuning":
        num_params = int(base_params)
        if num_params <= 0:
            return
        ratio = total_tokens / num_params
        # Full FT starts from a converged base; 1–5 tok/param is a practical sweet spot.
        # Going much above 10 tok/param risks catastrophic forgetting.
        if ratio < 0.1:
            st.error(
                f"**Dataset too small for full fine-tuning.** "
                f"{ratio:.3f} tok/param. You need at least ~0.5–1 tok/param "
                f"(**{_fmt_tokens(int(num_params * 0.5))} tokens**) to see meaningful adaptation."
            )
        elif ratio < 1:
            st.warning(
                f"**Likely undertrained for full fine-tuning.** "
                f"{ratio:.2f} tok/param. Aim for 1–5 tok/param "
                f"(**{_fmt_tokens(int(num_params))}–{_fmt_tokens(int(num_params * 5))} tokens**) "
                f"for stable full-parameter adaptation."
            )
        elif ratio <= 5:
            st.success(
                f"**Good range for full fine-tuning.** "
                f"{ratio:.1f} tok/param — standard for supervised fine-tuning of large models."
            )
        elif ratio <= 20:
            st.info(
                f"**Generous dataset for full fine-tuning.** "
                f"{ratio:.1f} tok/param. Effective, but watch for catastrophic forgetting "
                f"of the base model's general capabilities at high token counts."
            )
        else:
            st.warning(
                f"**Very large dataset for full fine-tuning ({ratio:.0f} tok/param).** "
                f"Above ~20 tok/param you risk catastrophic forgetting. "
                f"Consider using LoRA, which handles large datasets without degrading base capabilities."
            )

    # ── LoRA / QLoRA ─────────────────────────────────────────────────────────
    else:
        num_adapter_params = int(adapter_params)
        if num_adapter_params <= 0 or base_params <= 0:
            return
        ratio_adapter  = total_tokens / num_adapter_params
        ratio_base     = total_tokens / int(base_params)
        pct_trainable  = num_adapter_params / int(base_params) * 100
        method_label   = ft_method  # "LoRA" or "QLoRA"

        # LoRA data thresholds are relative to adapter params, not base params.
        # ~10–100 tok/adapter_param is the practical sweet spot for task adaptation.
        # The base model's knowledge means you need far less data than Chinchilla.
        if ratio_adapter < 10:
            st.warning(
                f"**Possibly too little data for {method_label}.** "
                f"{ratio_adapter:.1f} tok/adapter_param "
                f"({ratio_base:.2f} tok/base_param, {pct_trainable:.2f}% trainable). "
                f"Aim for ≥ 10 tok/adapter_param "
                f"(**{_fmt_tokens(int(num_adapter_params * 10))} tokens**) for reliable convergence."
            )
        elif ratio_adapter <= 100:
            st.success(
                f"**Good range for {method_label}.** "
                f"{ratio_adapter:.0f} tok/adapter_param "
                f"({ratio_base:.2f} tok/base_param, {pct_trainable:.2f}% trainable). "
                f"The frozen base model provides strong priors — far less data is needed "
                f"than Chinchilla's 20 tok/param pre-training recommendation."
            )
        elif ratio_adapter <= 1000:
            st.info(
                f"**Large dataset relative to {method_label} adapter size.** "
                f"{ratio_adapter:.0f} tok/adapter_param. This is fine — LoRA adapters "
                f"don't suffer catastrophic forgetting, so training longer generally helps. "
                f"Consider increasing LoRA rank (r) to give the adapter more capacity."
            )
        else:
            st.info(
                f"**Very large dataset for {method_label} adapter size "
                f"({ratio_adapter:.0f} tok/adapter_param).** "
                f"At this scale the adapter may be the bottleneck — consider full fine-tuning "
                f"or significantly increasing LoRA rank (r)."
            )


def _fmt_tokens(n: int) -> str:
    """Format a large integer as a human-readable token/parameter count."""
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _peak_flops_for_precision(instance_spec: dict, mixed_precision: str) -> float:
    """Return effective peak FLOPs/s for the given precision on the given instance.

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


def _display(val: Any) -> str:
    """Format a value for human-readable display in the summary table."""
    if isinstance(val, float):
        if val >= 1e12:
            return f"{val:.2e}"
        if val >= 1e9:
            return f"{val / 1e9:.2f}B"
        if val >= 1e6:
            return f"{val / 1e6:.2f}M"
        if val >= 1e3:
            return f"{val / 1e3:.1f}K"
        return f"{val:.4g}"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, int) and val >= 1_000:
        return f"{val:,}"
    return str(val)