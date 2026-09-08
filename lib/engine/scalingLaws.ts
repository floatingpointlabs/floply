/**
 * Scaling-law health assessments. Port of src/cost_modelling/scaling_laws.py.
 *
 * Every message here must match Python's f-string output byte for byte — the fixtures
 * compare full strings, and `:.2f` maps to toFixed(2). Note the non-ASCII characters
 * (≥, —, ×) are load-bearing for that comparison.
 */

import { CHINCHILLA_OPTIMAL_RATIO } from "./constants";
import { fmtTokens, toFixedHalfEven } from "./formatting";

export type AssessmentLevel = "error" | "warning" | "success" | "info";

export interface Assessment {
  level: AssessmentLevel;
  message: string;
}

const at = (level: AssessmentLevel, message: string): Assessment => ({ level, message });

export function assessPreTraining(totalTokens: number, numParams: number): Assessment | null {
  if (totalTokens <= 0 || numParams <= 0) return null;

  const ratio = totalTokens / numParams;
  const optimalTokens = numParams * CHINCHILLA_OPTIMAL_RATIO;

  if (ratio < 1) {
    return at(
      "error",
      `**Dataset critically undersized.** ` +
        `${toFixedHalfEven(ratio, 2)} tok/param — fewer tokens than parameters. ` +
        `Chinchilla-optimal requires **${fmtTokens(optimalTokens)} tokens** ` +
        `(${CHINCHILLA_OPTIMAL_RATIO}x N). Training will likely diverge or severely underfit.`
    );
  }
  if (ratio < 10) {
    return at(
      "warning",
      `**Undertrained (below Chinchilla-optimal).** ` +
        `${toFixedHalfEven(ratio, 1)} tok/param. Need ≥ ${CHINCHILLA_OPTIMAL_RATIO} tok/param for compute efficiency — ` +
        `at least **${fmtTokens(optimalTokens)} tokens**. ` +
        `Consider more data or a smaller model.`
    );
  }
  if (ratio <= 30) {
    return at(
      "success",
      `**Chinchilla-optimal.** ` +
        `${toFixedHalfEven(ratio, 1)} tok/param — within the 10-30x compute-optimal zone ` +
        `(Hoffmann et al. 2022).`
    );
  }
  if (ratio <= 200) {
    return at(
      "info",
      `**Inference-optimal (over-trained vs Chinchilla).** ` +
        `${toFixedHalfEven(ratio, 1)} tok/param. Training smaller models longer reduces ` +
        `inference cost — the LLaMA / Mistral strategy. Fine if you expect ` +
        `high inference volume.`
    );
  }
  return at(
    "warning",
    `**Heavily over-trained relative to model size.** ` +
      `${toFixedHalfEven(ratio, 0)} tok/param (> 200x). Diminishing returns; ` +
      `consider scaling the model up.`
  );
}

/**
 * Full fine-tuning starts from a converged base; 1–5 tok/param is the sweet spot.
 * Much above 10 tok/param risks catastrophic forgetting.
 */
export function assessFullFineTuning(totalTokens: number, numParams: number): Assessment | null {
  if (totalTokens <= 0 || numParams <= 0) return null;

  const ratio = totalTokens / numParams;

  if (ratio < 0.1) {
    return at(
      "error",
      `**Dataset too small for full fine-tuning.** ` +
        `${toFixedHalfEven(ratio, 3)} tok/param. You need at least ~0.5–1 tok/param ` +
        `(**${fmtTokens(Math.trunc(numParams * 0.5))} tokens**) to see meaningful adaptation.`
    );
  }
  if (ratio < 1) {
    return at(
      "warning",
      `**Likely undertrained for full fine-tuning.** ` +
        `${toFixedHalfEven(ratio, 2)} tok/param. Aim for 1–5 tok/param ` +
        `(**${fmtTokens(Math.trunc(numParams))}–${fmtTokens(Math.trunc(numParams * 5))} tokens**) ` +
        `for stable full-parameter adaptation.`
    );
  }
  if (ratio <= 5) {
    return at(
      "success",
      `**Good range for full fine-tuning.** ` +
        `${toFixedHalfEven(ratio, 1)} tok/param — standard for supervised fine-tuning of large models.`
    );
  }
  if (ratio <= 20) {
    return at(
      "info",
      `**Generous dataset for full fine-tuning.** ` +
        `${toFixedHalfEven(ratio, 1)} tok/param. Effective, but watch for catastrophic forgetting ` +
        `of the base model's general capabilities at high token counts.`
    );
  }
  return at(
    "warning",
    `**Very large dataset for full fine-tuning (${toFixedHalfEven(ratio, 0)} tok/param).** ` +
      `Above ~20 tok/param you risk catastrophic forgetting. ` +
      `Consider using LoRA, which handles large datasets without degrading base capabilities.`
  );
}

/**
 * LoRA/QLoRA thresholds, measured against adapter params rather than base params.
 * ~10–100 tok/adapter_param is the practical sweet spot for task adaptation.
 */
export function assessAdapter(
  totalTokens: number,
  numAdapterParams: number,
  baseParams: number,
  methodLabel: string
): Assessment | null {
  if (totalTokens <= 0 || numAdapterParams <= 0 || baseParams <= 0) return null;

  const ratioAdapter = totalTokens / numAdapterParams;
  const ratioBase = totalTokens / Math.trunc(baseParams);
  const pctTrainable = (numAdapterParams / Math.trunc(baseParams)) * 100;

  if (ratioAdapter < 10) {
    return at(
      "warning",
      `**Possibly too little data for ${methodLabel}.** ` +
        `${toFixedHalfEven(ratioAdapter, 1)} tok/adapter_param ` +
        `(${toFixedHalfEven(ratioBase, 2)} tok/base_param, ${toFixedHalfEven(pctTrainable, 2)}% trainable). ` +
        `Aim for ≥ 10 tok/adapter_param ` +
        `(**${fmtTokens(Math.trunc(numAdapterParams * 10))} tokens**) for reliable convergence.`
    );
  }
  if (ratioAdapter <= 100) {
    return at(
      "success",
      `**Good range for ${methodLabel}.** ` +
        `${toFixedHalfEven(ratioAdapter, 0)} tok/adapter_param ` +
        `(${toFixedHalfEven(ratioBase, 2)} tok/base_param, ${toFixedHalfEven(pctTrainable, 2)}% trainable). ` +
        `The frozen base model provides strong priors — far less data is needed ` +
        `than Chinchilla's 20 tok/param pre-training recommendation.`
    );
  }
  if (ratioAdapter <= 1000) {
    return at(
      "info",
      `**Large dataset relative to ${methodLabel} adapter size.** ` +
        `${toFixedHalfEven(ratioAdapter, 0)} tok/adapter_param. This is fine — LoRA adapters ` +
        `don't suffer catastrophic forgetting, so training longer generally helps. ` +
        `Consider increasing LoRA rank (r) to give the adapter more capacity.`
    );
  }
  return at(
    "info",
    `**Very large dataset for ${methodLabel} adapter size ` +
      `(${toFixedHalfEven(ratioAdapter, 0)} tok/adapter_param).** ` +
      `At this scale the adapter may be the bottleneck — consider full fine-tuning ` +
      `or significantly increasing LoRA rank (r).`
  );
}

/** Dispatch to the right assessment. A null ftMethod means pre-training. */
export function assessTrainingConfig(
  totalTokens: number,
  ftMethod: string | null,
  baseParams = 0,
  preParams = 0,
  adapterParams = 0
): Assessment | null {
  if (totalTokens <= 0) return null;
  if (ftMethod === null) return assessPreTraining(totalTokens, Math.trunc(preParams));
  if (ftMethod === "Full Fine-Tuning") {
    return assessFullFineTuning(totalTokens, Math.trunc(baseParams));
  }
  return assessAdapter(totalTokens, Math.trunc(adapterParams), baseParams, ftMethod);
}

/**
 * LoRA trade-off banner for the budget optimizer.
 * Measured against adapter params, where 10–100 tok/adapter_param is the useful band.
 */
export function assessLoraRatio(ratio: number): Assessment {
  if (ratio < 10) {
    return at(
      "warning",
      `**Below LoRA sweet spot.** ${toFixedHalfEven(ratio, 1)} tok/adapter_param ` +
        `(target: 10–100×). Use more data or reduce rank.`
    );
  }
  if (ratio <= 100) {
    return at(
      "success",
      `**Within LoRA sweet spot.** ${toFixedHalfEven(ratio, 1)} tok/adapter_param — good range.`
    );
  }
  return at(
    "info",
    `**Above LoRA sweet spot.** ${toFixedHalfEven(ratio, 1)} tok/adapter_param. ` +
      `Consider increasing rank or reducing dataset.`
  );
}

export function assessChinchillaRatio(
  ratio: number,
  optimalTokens: number,
  fixHint = ""
): Assessment {
  const suffix = fixHint ? ` ${fixHint}` : "";

  if (ratio < 1) {
    return at(
      "error",
      `**Extremely data-sparse.** ${toFixedHalfEven(ratio, 2)} tok/param — fewer tokens than parameters.` +
        suffix
    );
  }
  if (ratio < 10) {
    return at(
      "warning",
      `**Undertrained vs. Chinchilla-optimal.** ${toFixedHalfEven(ratio, 1)} tok/param. ` +
        `For compute-efficient training aim for ≥ ${CHINCHILLA_OPTIMAL_RATIO} tok/param ` +
        `(**${fmtTokens(optimalTokens)} tokens**).` +
        suffix
    );
  }
  if (ratio <= 30) {
    return at(
      "success",
      `**Chinchilla-optimal.** ${toFixedHalfEven(ratio, 1)} tok/param — within the 10–30× compute-optimal zone ` +
        `This is a well-balanced configuration.`
    );
  }
  if (ratio <= 200) {
    return at(
      "info",
      `**Inference-optimal (over-trained vs. Chinchilla).** ${toFixedHalfEven(ratio, 1)} tok/param. ` +
        `Training smaller models longer reduces inference cost — the LLaMA / Mistral strategy.`
    );
  }
  return at(
    "warning",
    `**Heavily over-trained relative to model size.** ${toFixedHalfEven(ratio, 0)} tok/param (> 200×). ` +
      `Diminishing returns; consider scaling the model up.` +
      suffix
  );
}
