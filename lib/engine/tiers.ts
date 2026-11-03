/**
 * Data-size tier definitions for the "Data needed" page.
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
    tier: "Too little",
    ratio_min: 0,
    ratio_max: 1,
    ratio_min_chart: 0.1,
    color: CHART_COLORS.warning,
    label: "Under 1 token per parameter",
    meaning: "Fewer tokens than parameters. Training will likely fail."
  },
  {
    tier: "Bare minimum",
    ratio_min: 1,
    ratio_max: 10,
    ratio_min_chart: 1,
    color: CHART_COLORS.checkpoint,
    label: "1–10 tokens per parameter",
    meaning: "Trains, but the model will be weak."
  },
  {
    tier: "Optimal",
    ratio_min: 10,
    ratio_max: 30,
    ratio_min_chart: 10,
    color: CHART_COLORS.success,
    label: "10–30 tokens per parameter",
    meaning: "The best model for the compute you spend."
  },
  {
    tier: "Extended",
    ratio_min: 30,
    ratio_max: 200,
    ratio_min_chart: 30,
    color: CHART_COLORS.storage,
    label: "30–200 tokens per parameter",
    meaning: "Costs more to train, but gives a smaller model that is cheaper to run."
  }
];

export const FULL_FT_TIERS: Tier[] = [
  {
    tier: "Too little",
    ratio_min: 0,
    ratio_max: 0.5,
    ratio_min_chart: 0.05,
    color: CHART_COLORS.warning,
    label: "Under 0.5 tokens per parameter",
    meaning: "Probably not enough for the model to learn the new task."
  },
  {
    tier: "Bare minimum",
    ratio_min: 0.5,
    ratio_max: 1,
    ratio_min_chart: 0.5,
    color: CHART_COLORS.checkpoint,
    label: "0.5–1 tokens per parameter",
    meaning: "Enough to work, but results may be unstable."
  },
  {
    tier: "Optimal",
    ratio_min: 1,
    ratio_max: 5,
    ratio_min_chart: 1,
    color: CHART_COLORS.success,
    label: "1–5 tokens per parameter",
    meaning: "The typical range for fine-tuning. Stable and effective."
  },
  {
    tier: "Lots of data",
    ratio_min: 5,
    ratio_max: 20,
    ratio_min_chart: 5,
    color: CHART_COLORS.storage,
    label: "5–20 tokens per parameter",
    meaning: "Works, but the model may start to forget what it already knew."
  }
];

export const LORA_TIERS: Tier[] = [
  {
    tier: "Too little",
    ratio_min: 0,
    ratio_max: 10,
    ratio_min_chart: 1,
    color: CHART_COLORS.warning,
    label: "Under 10 tokens per adapter parameter",
    meaning: "Not enough data for the adapter to learn reliably."
  },
  {
    tier: "Bare minimum",
    ratio_min: 10,
    ratio_max: 50,
    ratio_min_chart: 10,
    color: CHART_COLORS.checkpoint,
    label: "10–50 tokens per adapter parameter",
    meaning: "Enough to learn, but below the ideal range."
  },
  {
    tier: "Optimal",
    ratio_min: 50,
    ratio_max: 100,
    ratio_min_chart: 50,
    color: CHART_COLORS.success,
    label: "50–100 tokens per adapter parameter",
    meaning: "The ideal range for LoRA."
  },
  {
    tier: "Outgrowing LoRA",
    ratio_min: 100,
    ratio_max: 1000,
    ratio_min_chart: 100,
    color: CHART_COLORS.storage,
    label: "100–1,000 tokens per adapter parameter",
    meaning: "More data than the adapter can use. Raise the rank or switch to full fine-tuning."
  }
];
