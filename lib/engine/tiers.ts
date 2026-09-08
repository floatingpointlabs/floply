/**
 * Scaling-law tier definitions for the Minimum Data Calculator.
 * Port of the tier tables in src/app/config.py.
 *
 * Each tier describes a token-per-parameter (or tok/adapter_param) band:
 *   ratioMin / ratioMax  — the zone boundaries
 *   ratioMinChart        — log-scale chart lower bound, which must be > 0
 */

import { CHART_COLORS } from "./constants";

export interface Tier {
  tier: string;
  ratio_min: number;
  ratio_max: number;
  ratio_min_chart: number;
  color: string;
  label: string;
  meaning: string;
}

export const PRE_TRAINING_TIERS: Tier[] = [
  {
    tier: "Hard floor",
    ratio_min: 0, ratio_max: 1, ratio_min_chart: 0.1,
    color: CHART_COLORS.warning,
    label: "< 1 tok/param",
    meaning:
      "Fewer tokens than parameters — training will likely diverge or severely underfit.",
  },
  {
    tier: "Practical minimum",
    ratio_min: 1, ratio_max: 10, ratio_min_chart: 1,
    color: CHART_COLORS.checkpoint,
    label: "1–10 tok/param",
    meaning:
      "Usable but significantly undertrained — expect poor generalisation and high loss.",
  },
  {
    tier: "Compute-optimal",
    ratio_min: 10, ratio_max: 30, ratio_min_chart: 10,
    color: CHART_COLORS.success,
    label: "10–30 tok/param",
    meaning: "Chinchilla-optimal zone (Hoffmann et al. 2022) — best loss per FLOP.",
  },
  {
    tier: "Inference-optimal",
    ratio_min: 30, ratio_max: 200, ratio_min_chart: 30,
    color: CHART_COLORS.storage,
    label: "30–200 tok/param",
    meaning:
      "Training smaller models longer reduces inference cost — the LLaMA / Mistral strategy.",
  },
];

export const FULL_FT_TIERS: Tier[] = [
  {
    tier: "Too small",
    ratio_min: 0, ratio_max: 0.5, ratio_min_chart: 0.05,
    color: CHART_COLORS.warning,
    label: "< 0.5 tok/param",
    meaning:
      "Likely insufficient for meaningful task adaptation — model may not converge on the target task.",
  },
  {
    tier: "Minimum viable",
    ratio_min: 0.5, ratio_max: 1, ratio_min_chart: 0.5,
    color: CHART_COLORS.checkpoint,
    label: "0.5–1 tok/param",
    meaning:
      "Lower bound for stable full-parameter fine-tuning — expect some instability.",
  },
  {
    tier: "Sweet spot",
    ratio_min: 1, ratio_max: 5, ratio_min_chart: 1,
    color: CHART_COLORS.success,
    label: "1–5 tok/param",
    meaning:
      "Standard supervised fine-tuning range for large models — stable and effective.",
  },
  {
    tier: "Forgetting risk",
    ratio_min: 5, ratio_max: 20, ratio_min_chart: 5,
    color: CHART_COLORS.storage,
    label: "5–20 tok/param",
    meaning:
      "Generous dataset — effective, but monitor for catastrophic forgetting of base model capabilities.",
  },
];

export const LORA_TIERS: Tier[] = [
  {
    tier: "Too small",
    ratio_min: 0, ratio_max: 10, ratio_min_chart: 1,
    color: CHART_COLORS.warning,
    label: "< 10 tok/adapter_param",
    meaning:
      "Adapter likely won't converge reliably — insufficient signal for low-rank matrices.",
  },
  {
    tier: "Minimum viable",
    ratio_min: 10, ratio_max: 50, ratio_min_chart: 10,
    color: CHART_COLORS.checkpoint,
    label: "10–50 tok/adapter_param",
    meaning:
      "Lower bound of the convergence zone — reliable but not yet in the ideal range.",
  },
  {
    tier: "Sweet spot",
    ratio_min: 50, ratio_max: 100, ratio_min_chart: 50,
    color: CHART_COLORS.success,
    label: "50–100 tok/adapter_param",
    meaning:
      "Practical sweet spot for task adaptation — strong convergence with low forgetting risk.",
  },
  {
    tier: "Consider full FT",
    ratio_min: 100, ratio_max: 1000, ratio_min_chart: 100,
    color: CHART_COLORS.storage,
    label: "100–1000 tok/adapter_param",
    meaning:
      "Large dataset relative to adapter size — consider increasing LoRA rank or switching to full fine-tuning.",
  },
];
