/**
 * Maps a fixture group name to the engine call it pins.
 *
 * Two callers: `fixtures.test.ts` replays the committed fixtures through this, and
 * `scripts/gen-fixtures.mjs` generates them through it. One table means a regenerated
 * fixture and the assertion that checks it can never drift apart.
 */

import * as bo from "./budgetOptimizer";
import * as calc from "./calculator";
import * as ds from "./dataset";
import * as fmt from "./formatting";
import * as sl from "./scalingLaws";
import {
  MODELS,
  displayName,
  effectiveParameterCount,
  getGpuInstance,
  peakFlopsForPrecision
} from "./gpuSpecs";
import { FULL_FT_TIERS, LORA_TIERS, PRE_TRAINING_TIERS } from "./tiers";
import type { Tier } from "./tiers";
import { estimateTrainingBudget } from "./trainingBudget";

export type Case = {
  args: Record<string, any>;
  expect?: unknown;
  raises?: string;
  arg_type?: string;
  name?: string;
  selection?: Record<string, any>;
};

const A = (c: Case) => c.args;

const TIER_TABLES: Record<string, Tier[]> = {
  PRE_TRAINING_TIERS,
  FULL_FT_TIERS,
  LORA_TIERS
};

/**
 * Colour is deliberately excluded: the thresholds are engine behaviour and must not drift,
 * but the hex is presentation and has to be free to change with branding. Pinning it would
 * make a re-theme fail the maths tests.
 */
export const tiersWithoutColor = (table: string) =>
  TIER_TABLES[table].map(({ color: _color, ...rest }) => rest);

const modelsBySlug = new Map(MODELS.map((m) => [m.slug, m]));

export const modelDefinition = (slug: string) => {
  const m = modelsBySlug.get(slug)!;
  return {
    name: m.name,
    family: m.family,
    parameter_count: m.parameter_count,
    effective_parameter_count: effectiveParameterCount(m),
    display_name: displayName(m)
  };
};

/** The two-stage budget solve, with `ft_base_model` given as a slug rather than an object. */
export const solveCase = (args: Record<string, any>, selection: Record<string, any>) => {
  const resolved = { ...args } as bo.OptimizerInputs & { ft_base_model?: any };
  if (typeof resolved.ft_base_model === "string") {
    resolved.ft_base_model = modelsBySlug.get(resolved.ft_base_model)!;
  }
  const inputs: bo.OptimizerInputs = { ...resolved, hardware: bo.autoConfigureHardware() };
  const optimum = bo.solveBudgetOptimum(inputs);
  return { optimum, selection: bo.resolveSelection(inputs, optimum, selection) };
};

export const DISPATCH: Record<string, (c: Case) => unknown> = {
  _get_flops_multiplier: (c) =>
    calc.getFlopsMultiplier(A(c).architecture, A(c).gradient_checkpointing),
  calculate_training_flops: (c) =>
    calc.calculateTrainingFlops(
      A(c).parameter_count,
      A(c).training_tokens,
      A(c).architecture,
      A(c).epochs,
      A(c).gradient_checkpointing
    ),
  estimate_compute_cost: (c) =>
    calc.estimateComputeCost(
      A(c).total_flops,
      A(c).peak_flops_per_gpu,
      A(c).mfu,
      A(c).total_gpus,
      A(c).num_instances,
      A(c).hourly_cost
    ),
  solve_for_parameter_count: (c) =>
    calc.solveForParameterCount(
      A(c).compute_budget_usd,
      A(c).training_tokens,
      A(c).peak_flops_per_gpu,
      A(c).mfu,
      A(c).total_gpus,
      A(c).num_instances,
      A(c).hourly_cost,
      A(c).architecture,
      A(c).epochs,
      A(c).gradient_checkpointing
    ),
  solve_for_training_tokens: (c) =>
    calc.solveForTrainingTokens(
      A(c).compute_budget_usd,
      A(c).parameter_count,
      A(c).peak_flops_per_gpu,
      A(c).mfu,
      A(c).total_gpus,
      A(c).num_instances,
      A(c).hourly_cost,
      A(c).architecture,
      A(c).epochs,
      A(c).gradient_checkpointing
    ),
  estimate_gpu_memory_gb: (c) =>
    calc.estimateGpuMemoryGb(
      A(c).effective_params,
      A(c).trainable_params,
      A(c).ft_method,
      A(c).total_gpus,
      A(c).rl_multiplier,
      A(c).d_model,
      A(c).num_layers,
      A(c).seq_len,
      A(c).batch_size,
      A(c).gradient_checkpointing
    ),
  calculate_lora_trainable_params: (c) =>
    calc.calculateLoraTrainableParams(
      A(c).ft_method,
      A(c).base_params,
      A(c).target_modules,
      A(c).d_model,
      A(c).num_layers,
      A(c).lora_rank,
      A(c).architecture
    ),
  calculate_checkpoint_storage_tb: (c) =>
    calc.calculateCheckpointStorageTb(
      A(c).checkpoint_params,
      A(c).num_checkpoints,
      A(c).num_training_runs,
      A(c).num_hp_trials,
      A(c).num_ablations
    ),
  calculate_project_compute_cost: (c) =>
    calc.calculateProjectComputeCost(
      A(c).single_run_cost,
      A(c).num_training_runs,
      A(c).num_hp_trials,
      A(c).hp_fraction,
      A(c).num_ablations,
      A(c).ablation_fraction
    ),
  calculate_storage_cost: (c) =>
    calc.calculateStorageCost(
      A(c).dataset_size_tb,
      A(c).storage_duration_months,
      A(c).storage_class
    ),
  peak_flops_for_precision: (c) =>
    peakFlopsForPrecision(getGpuInstance(A(c).instance_type), A(c).mixed_precision),
  tokens_per_text_sample: (c) => ds.tokensPerTextSample(A(c).avg_seq_len_words),
  bytes_per_text_sample: (c) => ds.bytesPerTextSample(A(c).tokens),
  tokens_per_image_sample: (c) => ds.tokensPerImageSample(A(c).resolution, A(c).patch_size),
  bytes_per_image_sample: (c) => ds.bytesPerImageSample(A(c).resolution),
  tokens_per_audio_sample: (c) =>
    ds.tokensPerAudioSample(A(c).clip_duration, A(c).base_rate, A(c).num_codebooks),
  bytes_per_audio_sample: (c) => ds.bytesPerAudioSample(A(c).clip_duration),
  tokens_per_video_sample: (c) =>
    ds.tokensPerVideoSample(A(c).duration, A(c).fps, A(c).resolution, A(c).patch_size),
  bytes_per_video_sample: (c) => ds.bytesPerVideoSample(A(c).duration, A(c).fps, A(c).resolution),
  fmt_tokens: (c) => fmt.fmtTokens(A(c).n),
  fmt_samples: (c) => fmt.fmtSamples(A(c).n),
  format_wall_clock_time: (c) => fmt.formatWallClockTime(A(c).wall_clock_days),
  format_gpu_hours: (c) => fmt.formatGpuHours(A(c).gpu_hours),
  // JSON can't distinguish Python's 1000.0 from 1000; arg_type carries the source type.
  display_value: (c) => fmt.displayValue(A(c).val, c.arg_type === "float"),
  assess_training_config: (c) =>
    sl.assessTrainingConfig(
      A(c).total_tokens,
      A(c).ft_method,
      A(c).base_params,
      A(c).pre_params,
      A(c).adapter_params
    ),
  assess_lora_ratio: (c) => sl.assessLoraRatio(A(c).ratio),
  assess_chinchilla_ratio: (c) =>
    sl.assessChinchillaRatio(A(c).ratio, A(c).optimal_tokens, A(c).fix_hint),
  training_budget: (c) => estimateTrainingBudget(c.args),
  tiers: (c) => tiersWithoutColor(A(c).table),
  model_definitions: (c) => modelDefinition(A(c).slug),
  derive_schedule_defaults: (c) =>
    bo.deriveScheduleDefaults(
      A(c).compute_budget,
      A(c).modality,
      A(c).training_type,
      A(c).ft_method,
      A(c).quick_n
    ),
  logspace: (c) => bo.logspace(A(c).start, A(c).stop, A(c).num),
  budget_curve_n: (c) =>
    bo.budgetCurveN(
      A(c).d_tokens,
      A(c).budget,
      A(c).peak_flops_per_gpu,
      A(c).mfu,
      A(c).gpus_per_instance,
      A(c).hourly_cost,
      A(c).multiplier,
      A(c).epochs
    )
};
