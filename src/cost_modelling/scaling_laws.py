"""Scaling-law health assessments — pure, no Streamlit.

Extracted from src/app/helpers.py, where the same threshold logic was fused with
``st.error`` / ``st.warning`` / ``st.success`` / ``st.info`` calls. Each function now
returns an :class:`Assessment` and the caller decides how to render it; ``level`` maps
one-to-one onto the Streamlit callable it replaced.

Thresholds by training method:
  Pre-training     — Chinchilla (Hoffmann et al. 2022), D* ≈ 20 × N.
  Full fine-tuning — softer 1–5 tok/param rule (the base model has already converged).
  LoRA / QLoRA     — measured against trainable adapter params, not base params.
"""

from dataclasses import dataclass

from src.cost_modelling.constants import CHINCHILLA_OPTIMAL_RATIO
from src.cost_modelling.formatting import fmt_tokens

LEVELS = ("error", "warning", "success", "info")


@dataclass(frozen=True)
class Assessment:
    """A rendered-agnostic banner: ``level`` selects the styling, ``message`` is markdown."""

    level: str
    message: str


def assess_pre_training(total_tokens: int, num_params: int) -> Assessment | None:
    """Chinchilla assessment for pre-training from a raw token/parameter count."""
    if total_tokens <= 0 or num_params <= 0:
        return None

    ratio = total_tokens / num_params
    optimal_tokens = num_params * CHINCHILLA_OPTIMAL_RATIO

    if ratio < 1:
        return Assessment("error", (
            f"**Dataset critically undersized.** "
            f"{ratio:.2f} tok/param — fewer tokens than parameters. "
            f"Chinchilla-optimal requires **{fmt_tokens(optimal_tokens)} tokens** "
            f"({CHINCHILLA_OPTIMAL_RATIO}x N). Training will likely diverge or severely underfit."
        ))
    if ratio < 10:
        return Assessment("warning", (
            f"**Undertrained (below Chinchilla-optimal).** "
            f"{ratio:.1f} tok/param. Need ≥ {CHINCHILLA_OPTIMAL_RATIO} tok/param for compute efficiency — "
            f"at least **{fmt_tokens(optimal_tokens)} tokens**. "
            f"Consider more data or a smaller model."
        ))
    if ratio <= 30:
        return Assessment("success", (
            f"**Chinchilla-optimal.** "
            f"{ratio:.1f} tok/param — within the 10-30x compute-optimal zone "
            f"(Hoffmann et al. 2022)."
        ))
    if ratio <= 200:
        return Assessment("info", (
            f"**Inference-optimal (over-trained vs Chinchilla).** "
            f"{ratio:.1f} tok/param. Training smaller models longer reduces "
            f"inference cost — the LLaMA / Mistral strategy. Fine if you expect "
            f"high inference volume."
        ))
    return Assessment("warning", (
        f"**Heavily over-trained relative to model size.** "
        f"{ratio:.0f} tok/param (> 200x). Diminishing returns; "
        f"consider scaling the model up."
    ))


def assess_full_fine_tuning(total_tokens: int, num_params: int) -> Assessment | None:
    """Full fine-tuning starts from a converged base; 1–5 tok/param is the sweet spot.

    Much above 10 tok/param risks catastrophic forgetting.
    """
    if total_tokens <= 0 or num_params <= 0:
        return None

    ratio = total_tokens / num_params

    if ratio < 0.1:
        return Assessment("error", (
            f"**Dataset too small for full fine-tuning.** "
            f"{ratio:.3f} tok/param. You need at least ~0.5–1 tok/param "
            f"(**{fmt_tokens(int(num_params * 0.5))} tokens**) to see meaningful adaptation."
        ))
    if ratio < 1:
        return Assessment("warning", (
            f"**Likely undertrained for full fine-tuning.** "
            f"{ratio:.2f} tok/param. Aim for 1–5 tok/param "
            f"(**{fmt_tokens(int(num_params))}–{fmt_tokens(int(num_params * 5))} tokens**) "
            f"for stable full-parameter adaptation."
        ))
    if ratio <= 5:
        return Assessment("success", (
            f"**Good range for full fine-tuning.** "
            f"{ratio:.1f} tok/param — standard for supervised fine-tuning of large models."
        ))
    if ratio <= 20:
        return Assessment("info", (
            f"**Generous dataset for full fine-tuning.** "
            f"{ratio:.1f} tok/param. Effective, but watch for catastrophic forgetting "
            f"of the base model's general capabilities at high token counts."
        ))
    return Assessment("warning", (
        f"**Very large dataset for full fine-tuning ({ratio:.0f} tok/param).** "
        f"Above ~20 tok/param you risk catastrophic forgetting. "
        f"Consider using LoRA, which handles large datasets without degrading base capabilities."
    ))


def assess_adapter(
    total_tokens: int,
    num_adapter_params: int,
    base_params: int,
    method_label: str,
) -> Assessment | None:
    """LoRA/QLoRA thresholds, measured against adapter params rather than base params.

    ~10–100 tok/adapter_param is the practical sweet spot for task adaptation; the frozen
    base model's knowledge means far less data is needed than Chinchilla would suggest.
    """
    if total_tokens <= 0 or num_adapter_params <= 0 or base_params <= 0:
        return None

    ratio_adapter = total_tokens / num_adapter_params
    ratio_base = total_tokens / int(base_params)
    pct_trainable = num_adapter_params / int(base_params) * 100

    if ratio_adapter < 10:
        return Assessment("warning", (
            f"**Possibly too little data for {method_label}.** "
            f"{ratio_adapter:.1f} tok/adapter_param "
            f"({ratio_base:.2f} tok/base_param, {pct_trainable:.2f}% trainable). "
            f"Aim for ≥ 10 tok/adapter_param "
            f"(**{fmt_tokens(int(num_adapter_params * 10))} tokens**) for reliable convergence."
        ))
    if ratio_adapter <= 100:
        return Assessment("success", (
            f"**Good range for {method_label}.** "
            f"{ratio_adapter:.0f} tok/adapter_param "
            f"({ratio_base:.2f} tok/base_param, {pct_trainable:.2f}% trainable). "
            f"The frozen base model provides strong priors — far less data is needed "
            f"than Chinchilla's 20 tok/param pre-training recommendation."
        ))
    if ratio_adapter <= 1000:
        return Assessment("info", (
            f"**Large dataset relative to {method_label} adapter size.** "
            f"{ratio_adapter:.0f} tok/adapter_param. This is fine — LoRA adapters "
            f"don't suffer catastrophic forgetting, so training longer generally helps. "
            f"Consider increasing LoRA rank (r) to give the adapter more capacity."
        ))
    return Assessment("info", (
        f"**Very large dataset for {method_label} adapter size "
        f"({ratio_adapter:.0f} tok/adapter_param).** "
        f"At this scale the adapter may be the bottleneck — consider full fine-tuning "
        f"or significantly increasing LoRA rank (r)."
    ))


def assess_training_config(
    total_tokens: int,
    ft_method: str | None,
    base_params: int = 0,
    pre_params: int = 0,
    adapter_params: int = 0,
) -> Assessment | None:
    """Dispatch to the right assessment for the configured training method.

    ``ft_method`` of None means pre-training. Returns None when there isn't enough
    information to say anything useful.
    """
    if total_tokens <= 0:
        return None
    if ft_method is None:
        return assess_pre_training(total_tokens, int(pre_params))
    if ft_method == "Full Fine-Tuning":
        return assess_full_fine_tuning(total_tokens, int(base_params))
    return assess_adapter(total_tokens, int(adapter_params), base_params, ft_method)


def assess_lora_ratio(ratio: float) -> Assessment:
    """LoRA trade-off banner for the budget optimizer.

    Measured against adapter params, where 10-100 tok/adapter_param is the useful band.
    """
    if ratio < 10:
        return Assessment("warning", (
            f"**Below LoRA sweet spot.** {ratio:.1f} tok/adapter_param "
            f"(target: 10–100×). Use more data or reduce rank."
        ))
    if ratio <= 100:
        return Assessment("success", (
            f"**Within LoRA sweet spot.** {ratio:.1f} tok/adapter_param — good range."
        ))
    return Assessment("info", (
        f"**Above LoRA sweet spot.** {ratio:.1f} tok/adapter_param. "
        f"Consider increasing rank or reducing dataset."
    ))


def assess_chinchilla_ratio(
    ratio: float, optimal_tokens: int, fix_hint: str = ""
) -> Assessment:
    """Chinchilla assessment for callers that already know the N/D ratio.

    Args:
        ratio: Tokens-per-parameter ratio (total_tokens_seen / N).
        optimal_tokens: Chinchilla-optimal token count (N × CHINCHILLA_OPTIMAL_RATIO).
        fix_hint: Context-specific suggestion appended to warning/error messages.
    """
    _suffix = f" {fix_hint}" if fix_hint else ""

    if ratio < 1:
        return Assessment("error", (
            f"**Extremely data-sparse.** {ratio:.2f} tok/param — fewer tokens than parameters."
            + _suffix
        ))
    if ratio < 10:
        return Assessment("warning", (
            f"**Undertrained vs. Chinchilla-optimal.** {ratio:.1f} tok/param. "
            f"For compute-efficient training aim for ≥ {CHINCHILLA_OPTIMAL_RATIO} tok/param "
            f"(**{fmt_tokens(optimal_tokens)} tokens**)."
            + _suffix
        ))
    if ratio <= 30:
        return Assessment("success", (
            f"**Chinchilla-optimal.** {ratio:.1f} tok/param — within the 10–30× compute-optimal zone "
            f"This is a well-balanced configuration."
        ))
    if ratio <= 200:
        return Assessment("info", (
            f"**Inference-optimal (over-trained vs. Chinchilla).** {ratio:.1f} tok/param. "
            f"Training smaller models longer reduces inference cost — the LLaMA / Mistral strategy."
        ))
    return Assessment("warning", (
        f"**Heavily over-trained relative to model size.** {ratio:.0f} tok/param (> 200×). "
        f"Diminishing returns; consider scaling the model up."
        + _suffix
    ))
