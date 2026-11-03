/**
 * Scaling-law health assessments.
 *
 * The fixtures compare these messages as whole strings, so every character is load-bearing
 * — including any non-ASCII such as –. Editing the wording is a fixture change.
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
      `**Far too little data.** ` +
        `${toFixedHalfEven(ratio, 2)} tokens per parameter, fewer tokens than parameters. ` +
        `Aim for **${fmtTokens(optimalTokens)} tokens** (${CHINCHILLA_OPTIMAL_RATIO} per parameter). ` +
        `Training will likely fail.`
    );
  }
  if (ratio < 10) {
    return at(
      "warning",
      `**Not enough data.** ` +
        `${toFixedHalfEven(ratio, 1)} tokens per parameter. Aim for ${CHINCHILLA_OPTIMAL_RATIO}, ` +
        `or **${fmtTokens(optimalTokens)} tokens**. Add more data or use a smaller model.`
    );
  }
  if (ratio <= 30) {
    return at(
      "success",
      `**Good balance.** ` +
        `${toFixedHalfEven(ratio, 1)} tokens per parameter, in the optimal 10–30 range.`
    );
  }
  if (ratio <= 200) {
    return at(
      "info",
      `**More data than optimal.** ` +
        `${toFixedHalfEven(ratio, 1)} tokens per parameter. Training costs more, but the smaller ` +
        `model is cheaper to run. Worth it if the model will see heavy use.`
    );
  }
  return at(
    "warning",
    `**More data than this model can use.** ` +
      `${toFixedHalfEven(ratio, 0)} tokens per parameter (over 200). ` +
      `Extra data adds little now; consider a bigger model.`
  );
}

/**
 * Full fine-tuning starts from a trained base; 1–5 tok/param is the sweet spot.
 * Much above 10 tok/param risks catastrophic forgetting.
 */
export function assessFullFineTuning(totalTokens: number, numParams: number): Assessment | null {
  if (totalTokens <= 0 || numParams <= 0) return null;

  const ratio = totalTokens / numParams;

  if (ratio < 0.1) {
    return at(
      "error",
      `**Too little data for full fine-tuning.** ` +
        `${toFixedHalfEven(ratio, 3)} tokens per parameter. You need at least ` +
        `**${fmtTokens(Math.trunc(numParams * 0.5))} tokens** (0.5 per parameter) to see a real change.`
    );
  }
  if (ratio < 1) {
    return at(
      "warning",
      `**Probably not enough data.** ` +
        `${toFixedHalfEven(ratio, 2)} tokens per parameter. Aim for 1–5 per parameter ` +
        `(**${fmtTokens(Math.trunc(numParams))}–${fmtTokens(Math.trunc(numParams * 5))} tokens**) ` +
        `for stable results.`
    );
  }
  if (ratio <= 5) {
    return at(
      "success",
      `**Good amount of data for full fine-tuning.** ` +
        `${toFixedHalfEven(ratio, 1)} tokens per parameter, in the typical range.`
    );
  }
  if (ratio <= 20) {
    return at(
      "info",
      `**Plenty of data.** ` +
        `${toFixedHalfEven(ratio, 1)} tokens per parameter. This works, but watch that the ` +
        `model doesn't forget what it already knew.`
    );
  }
  return at(
    "warning",
    `**Too much data for full fine-tuning** (${toFixedHalfEven(ratio, 0)} tokens per parameter). ` +
      `Above 20 per parameter, the model starts to forget what it already knew. ` +
      `LoRA handles large datasets better.`
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
  const pctTrainable = (numAdapterParams / Math.trunc(baseParams)) * 100;
  const trained = `${toFixedHalfEven(pctTrainable, 2)}% of the model is trained`;

  if (ratioAdapter < 10) {
    return at(
      "warning",
      `**Possibly too little data for ${methodLabel}.** ` +
        `${toFixedHalfEven(ratioAdapter, 1)} tokens per adapter parameter (${trained}). ` +
        `Aim for at least 10, or **${fmtTokens(Math.trunc(numAdapterParams * 10))} tokens**.`
    );
  }
  if (ratioAdapter <= 100) {
    return at(
      "success",
      `**Good amount of data for ${methodLabel}.** ` +
        `${toFixedHalfEven(ratioAdapter, 0)} tokens per adapter parameter (${trained}). ` +
        `Adapters need far less data than training a model from scratch.`
    );
  }
  if (ratioAdapter <= 1000) {
    return at(
      "info",
      `**Plenty of data for ${methodLabel}.** ` +
        `${toFixedHalfEven(ratioAdapter, 0)} tokens per adapter parameter. That's fine: adapters ` +
        `don't make the model forget what it knew. A higher rank would let the adapter learn more from it.`
    );
  }
  return at(
    "info",
    `**More data than the ${methodLabel} adapter can use** ` +
      `(${toFixedHalfEven(ratioAdapter, 0)} tokens per adapter parameter). ` +
      `Raise the rank a lot, or switch to full fine-tuning.`
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
      `**Below the ideal range.** ${toFixedHalfEven(ratio, 1)} tokens per adapter parameter ` +
        `(aim for 10–100). Add data or lower the rank.`
    );
  }
  if (ratio <= 100) {
    return at(
      "success",
      `**In the ideal range.** ${toFixedHalfEven(ratio, 1)} tokens per adapter parameter.`
    );
  }
  return at(
    "info",
    `**Above the ideal range.** ${toFixedHalfEven(ratio, 1)} tokens per adapter parameter ` +
      `(aim for 10–100). Raise the rank or use less data.`
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
      `**Far too little data.** ${toFixedHalfEven(ratio, 2)} tokens per parameter, fewer tokens than parameters.` +
        suffix
    );
  }
  if (ratio < 10) {
    return at(
      "warning",
      `**Not enough data.** ${toFixedHalfEven(ratio, 1)} tokens per parameter. ` +
        `Aim for ${CHINCHILLA_OPTIMAL_RATIO}, or **${fmtTokens(optimalTokens)} tokens**.` +
        suffix
    );
  }
  if (ratio <= 30) {
    return at(
      "success",
      `**Good balance.** ${toFixedHalfEven(ratio, 1)} tokens per parameter, in the optimal 10–30 range.`
    );
  }
  if (ratio <= 200) {
    return at(
      "info",
      `**More data than optimal.** ${toFixedHalfEven(ratio, 1)} tokens per parameter. ` +
        `Training costs more, but the smaller model is cheaper to run.`
    );
  }
  return at(
    "warning",
    `**More data than this model can use.** ${toFixedHalfEven(ratio, 0)} tokens per parameter (over 200). ` +
      `Consider a bigger model.` +
      suffix
  );
}
