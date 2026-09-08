/**
 * Budget Optimizer solve. Port of src/cost_modelling/budget_optimizer.py.
 *
 * In Selection, `undefined` means "not overridden" and the derived default applies. That
 * is the port of Streamlit's `session_state.pop(key)`: absence, never a written-in
 * default. Writing the derived value into state looks identical on first render and is
 * wrong forever after.
 */

import {
  BYTES_PER_PARAM_CHECKPOINT,
  calculateCheckpointStorageTb,
  calculateLoraTrainableParams,
  calculateStorageCost,
  calculateTrainingFlops,
  estimateComputeCost,
  solveForParameterCount,
  solveForTrainingTokens
} from "./calculator";
import {
  effectiveParameterCount,
  getGpuInstance,
  getStorageCost,
  listAvailableInstances,
  peakFlopsForPrecision
} from "./gpuSpecs";
import { CHINCHILLA_OPTIMAL_RATIO, LORA_OPTIMAL_RATIO } from "./constants";
import type { InstanceSpec, ModelDefinition } from "./types";

export { CHINCHILLA_OPTIMAL_RATIO, LORA_OPTIMAL_RATIO };

export const ARCH_MULTIPLIERS: Record<string, number> = {
  Transformer: 6.0,
  CNN: 4.0,
  RNN: 8.0,
  ViT: 6.0,
  Diffusion: 6.5
};

export interface ModalityConfig {
  arch: string;
  bytes_per_token: number;
  tokens_per_sample: number;
  sample_noun: string;
}

export const MODALITY_DEFAULTS: Record<string, ModalityConfig> = {
  "Text (LLM)": {
    arch: "Transformer",
    bytes_per_token: 4,
    tokens_per_sample: 512, // ~512 tokens per web document / article
    sample_noun: "documents"
  },
  "Vision (ViT / CLIP)": {
    arch: "ViT",
    bytes_per_token: 1536,
    tokens_per_sample: 196, // 14x14 patches for 224x224 at 16x16 patch size
    sample_noun: "images"
  },
  Audio: {
    arch: "Transformer",
    bytes_per_token: 8,
    tokens_per_sample: 750, // 10-second clip at 75 tok/s (EnCodec / Whisper rate)
    sample_noun: "clips"
  },
  "Multimodal (VLM)": {
    arch: "Transformer",
    bytes_per_token: 512,
    tokens_per_sample: 512, // mixed text + image patches per caption/pair
    sample_noun: "samples"
  },
  "Diffusion (Image Gen)": {
    arch: "Diffusion",
    bytes_per_token: 2048,
    tokens_per_sample: 1024, // 32x32 latent patches for a 512x512 image (8x VAE)
    sample_noun: "images"
  }
};

export const CHECKPOINTS_PER_RUN = 5;

export const TARGET_WALL_CLOCK_DAYS = 60;

/** Hard cap on recommended instances — a realistic large-scale cluster (~8,192 H100s). */
export const MAX_INSTANCES = 1_024;

export const DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"];

export const EXPLORE_DATASET = "Dataset size → token/param ratio";
export const EXPLORE_LORA_RANK = "LoRA Rank → dataset size";
export const EXPLORE_TOKENS_TO_MODEL = "Token count → model size";
export const EXPLORE_MODEL_TO_TOKENS = "Model size → token count";

export interface Hardware {
  instance_type: string;
  instance_spec: InstanceSpec;
  peak_flops_per_gpu: number;
  mfu: number;
  gpus_per_instance: number;
  hourly_cost: number;
  gradient_checkpointing: boolean;
}

let cachedHardware: Hardware | null = null;

export function autoConfigureHardware(): Hardware {
  if (cachedHardware) return cachedHardware;

  let bestInstance = "";
  let bestEfficiency = -1.0;

  for (const instanceType of listAvailableInstances()) {
    const spec = getGpuInstance(instanceType);
    const flopsPerDollar =
      (spec.peak_flops_fp16 * spec.gpu_count * spec.typical_mfu) / spec.hourly_cost;
    if (flopsPerDollar > bestEfficiency) {
      bestEfficiency = flopsPerDollar;
      bestInstance = instanceType;
    }
  }

  const spec = getGpuInstance(bestInstance);
  // Frozen because every caller now shares this instance; a stray mutation would
  // otherwise leak into unrelated calculations.
  cachedHardware = Object.freeze({
    instance_type: bestInstance,
    instance_spec: spec,
    peak_flops_per_gpu: peakFlopsForPrecision(spec, "bf16"),
    mfu: spec.typical_mfu,
    gpus_per_instance: spec.gpu_count,
    hourly_cost: spec.hourly_cost,
    gradient_checkpointing: false
  });
  return cachedHardware;
}

/**
 * Estimate parameter count from budget alone (1-epoch Chinchilla, auto hardware).
 * Only used for deriving Training Schedule defaults, so no storage correction.
 */
export function quickNEstimate(
  computeBudget: number,
  archMultiplier: number,
  hw: Hardware
): number {
  const quickCostPerTp =
    (archMultiplier / (hw.peak_flops_per_gpu * hw.mfu * hw.gpus_per_instance * 3600)) *
    hw.hourly_cost;
  if (quickCostPerTp <= 0) return 0.0;
  return Math.sqrt(computeBudget / (CHINCHILLA_OPTIMAL_RATIO * quickCostPerTp));
}

export interface ScheduleDefaults {
  epochs: number;
  hp_trials: number;
  hp_fraction_pct: number;
  storage_months: number;
  storage_class: string;
}

export function deriveScheduleDefaults(
  computeBudget: number,
  modality: string,
  trainingType: string,
  ftMethod: string | null,
  quickN: number
): ScheduleDefaults {
  const isLora = ftMethod === "LoRA" || ftMethod === "QLoRA";

  // Pre-training is 1-epoch optimal under Chinchilla; FT benefits from more passes.
  let epochs: number;
  if (trainingType === "Pre-Training") epochs = 1;
  else if (isLora) epochs = 5;
  else epochs = 3;

  let hpTrials: number;
  if (computeBudget < 5_000) hpTrials = 0;
  else if (computeBudget < 50_000) hpTrials = 2;
  else if (computeBudget < 500_000) hpTrials = 5;
  else if (computeBudget < 5_000_000) hpTrials = 10;
  else hpTrials = 20;

  let hpFractionPct: number;
  if (quickN >= 30e9) hpFractionPct = 5;
  else if (quickN >= 3e9) hpFractionPct = 10;
  else if (quickN >= 500e6) hpFractionPct = 15;
  else hpFractionPct = 25;

  const bytesPerToken = MODALITY_DEFAULTS[modality].bytes_per_token;
  const storageHeavy = bytesPerToken > 100; // Vision, Multimodal, Diffusion

  let baseMonths: number;
  if (computeBudget < 10_000) baseMonths = 1;
  else if (computeBudget < 100_000) baseMonths = 3;
  else if (computeBudget < 1_000_000) baseMonths = 6;
  else baseMonths = 12;

  // Python `//` — Math.floor, not `/`
  const storageMonths = Math.max(1, storageHeavy ? Math.floor(baseMonths / 2) : baseMonths);

  return {
    epochs,
    hp_trials: hpTrials,
    hp_fraction_pct: hpFractionPct,
    storage_months: storageMonths,
    storage_class: storageMonths > 3 ? "standard_ia" : "standard"
  };
}

export function budgetCurveN(
  dTokens: number[],
  budget: number,
  peakFlopsPerGpu: number,
  mfu: number,
  gpusPerInstance: number,
  hourlyCost: number,
  multiplier: number,
  epochs: number
): number[] {
  const costPerTokenParam =
    ((epochs * multiplier) / (peakFlopsPerGpu * mfu * gpusPerInstance * 3600)) * hourlyCost;
  return dTokens.map((d) => budget / (costPerTokenParam * d));
}

/** Stand-in for numpy.logspace — evenly spaced on a log10 scale, endpoint included. */
export function logspace(start: number, stop: number, num: number): number[] {
  if (num < 2) return new Array(Math.max(num, 0)).fill(10.0 ** start);
  const step = (stop - start) / (num - 1);
  return Array.from({ length: num }, (_, i) => 10.0 ** (start + i * step));
}

export interface OptimizerInputs {
  compute_budget: number;
  modality: string;
  training_type: string;
  ft_method?: string | null;
  ft_base_model?: ModelDefinition | null;
  lora_rank?: number;
  target_modules?: string[];
  epochs?: number;
  num_hp_trials?: number;
  hp_fraction_pct?: number;
  storage_duration_months?: number;
  storage_class?: string;
  hardware?: Hardware;
}

interface ResolvedInputs extends Required<Omit<OptimizerInputs, "ft_base_model">> {
  ft_base_model: ModelDefinition | null;
}

function resolveInputs(inputs: OptimizerInputs): ResolvedInputs {
  return {
    compute_budget: inputs.compute_budget,
    modality: inputs.modality,
    training_type: inputs.training_type,
    ft_method: inputs.ft_method ?? null,
    ft_base_model: inputs.ft_base_model ?? null,
    lora_rank: inputs.lora_rank ?? 16,
    target_modules: inputs.target_modules ?? [...DEFAULT_TARGET_MODULES],
    epochs: inputs.epochs ?? 1,
    num_hp_trials: inputs.num_hp_trials ?? 0,
    hp_fraction_pct: inputs.hp_fraction_pct ?? 10,
    storage_duration_months: inputs.storage_duration_months ?? 3,
    storage_class: inputs.storage_class ?? "standard",
    hardware: inputs.hardware ?? autoConfigureHardware()
  };
}

export const isLoraMethod = (ftMethod: string | null): boolean =>
  ftMethod === "LoRA" || ftMethod === "QLoRA";

export const isFtWithBase = (i: ResolvedInputs): boolean =>
  i.training_type === "Fine-Tuning" && i.ft_base_model !== null;

/** Total params — checkpoints, VRAM, adapter percentages. */
export const baseParamsTotal = (i: ResolvedInputs): number =>
  isFtWithBase(i) ? i.ft_base_model!.parameter_count : 0;

/** Active params — compute only. Differs from the total for MoE models. */
export const baseParamsActive = (i: ResolvedInputs): number =>
  isFtWithBase(i) ? effectiveParameterCount(i.ft_base_model!) : 0;

export function adapterParams(i: ResolvedInputs, loraRank?: number): number {
  if (!isFtWithBase(i)) return 0;
  if (!isLoraMethod(i.ft_method)) return baseParamsTotal(i); // SFT: all params trainable
  const arch = i.ft_base_model!.architecture;
  return (
    calculateLoraTrainableParams(
      i.ft_method!,
      baseParamsTotal(i),
      i.target_modules ?? [],
      arch.d_model ?? 0,
      arch.num_layers ?? 0,
      loraRank ?? i.lora_rank,
      arch
    ) ?? 0
  );
}

export function exploreOptions(i: ResolvedInputs): string[] {
  if (isFtWithBase(i) && isLoraMethod(i.ft_method)) {
    return [EXPLORE_DATASET, EXPLORE_LORA_RANK];
  }
  if (isFtWithBase(i)) return [EXPLORE_DATASET];
  return [EXPLORE_TOKENS_TO_MODEL, EXPLORE_MODEL_TO_TOKENS];
}

export interface BudgetOptimum {
  n_opt: number;
  d_opt: number;
  n_adapter: number;
  n_flops_base: number;
  /** The two competing LoRA bounds on D; both 0 outside the LoRA-with-base path. */
  d_eff: number;
  d_budget: number;
  opt_compute_cost_total: number;
  opt_dataset_storage_cost: number;
  opt_ckpt_storage_cost: number;
  opt_storage_cost: number;
  total_project_cost: number;
  project_multiplier: number;
  cost_per_token_param: number;
  ckpt_cost_per_param: number;
  storage_cost_per_token: number;
  cost_per_tb_month: number;
  optimal_ratio: number;
}

export function solveBudgetOptimum(rawInputs: OptimizerInputs): BudgetOptimum {
  const i = resolveInputs(rawInputs);
  const hw = i.hardware;

  const computeBudget = i.compute_budget;
  const modalityCfg = MODALITY_DEFAULTS[i.modality];
  const bytesPerToken = modalityCfg.bytes_per_token;
  const architecture = modalityCfg.arch;
  const multiplier = ARCH_MULTIPLIERS[architecture] ?? 6.0;
  const isLora = isLoraMethod(i.ft_method);
  const ftWithBase = isFtWithBase(i);
  const nAdapter = adapterParams(i);

  const hpFraction = i.hp_fraction_pct / 100.0;
  const projectMultiplier = 1.0 + i.num_hp_trials * hpFraction;
  const costPerTbMonth = getStorageCost(i.storage_class);
  const storageCostPerToken = (bytesPerToken / 1e12) * costPerTbMonth * i.storage_duration_months;

  const optimalRatio = isLora ? LORA_OPTIMAL_RATIO : CHINCHILLA_OPTIMAL_RATIO;

  // cost per (parameter x token) — numInstances cancels in the cost formula
  const costPerTokenParam =
    ((i.epochs * multiplier) / (hw.peak_flops_per_gpu * hw.mfu * hw.gpus_per_instance * 3600)) *
    hw.hourly_cost;

  const ckptCostPerParam =
    (BYTES_PER_PARAM_CHECKPOINT / 1e12) *
    costPerTbMonth *
    i.storage_duration_months *
    (CHECKPOINTS_PER_RUN + i.num_hp_trials);

  let nFlopsBase: number;
  let nOpt: number;
  let dOpt: number;
  let dEff = 0.0;
  let dBudget = 0.0;

  if (ftWithBase && nAdapter > 0) {
    // LoRA/QLoRA run the forward/backward pass through the full base model, so compute
    // uses the base params; the adapter count is for checkpoints only.
    //
    // Two competing bounds on D:
    //   1) efficiency: D_eff = LORA_OPTIMAL_RATIO x n_adapter (dominant at large budgets)
    //   2) budget:     D_budget = (B - ckpt) / (cost_per_token_param x N_base + storage)
    // D_opt = min of the two.
    nFlopsBase = baseParamsActive(i);
    nOpt = nAdapter;
    const ckptTotal = ckptCostPerParam * nAdapter;
    const denom = projectMultiplier * costPerTokenParam * nFlopsBase + storageCostPerToken;
    dBudget = denom > 0 ? Math.max(computeBudget - ckptTotal, 0.0) / denom : 0.0;
    dEff = isLora ? LORA_OPTIMAL_RATIO * nAdapter : dBudget;
    dOpt = Math.min(dEff, dBudget);
  } else {
    // Pre-training (or FT without a base): quadratic solve.
    // B = compute + dataset storage + checkpoint storage, with D = ratio x N:
    //   a·N² + b·N = B  →  N* = (−b + √(b² + 4aB)) / 2a
    nFlopsBase = 0;
    const aCoeff = optimalRatio * projectMultiplier * costPerTokenParam;
    const bCoeff = optimalRatio * storageCostPerToken + ckptCostPerParam;
    if (aCoeff > 0) {
      const discriminant = bCoeff ** 2 + 4 * aCoeff * computeBudget;
      nOpt = (-bCoeff + Math.sqrt(Math.max(discriminant, 0.0))) / (2 * aCoeff);
      dOpt = optimalRatio * nOpt;
    } else {
      nOpt = 0.0;
      dOpt = 0.0;
    }
  }

  // Cost breakdown at the optimal point. FLOPs use the base model's active params when
  // fine-tuning; checkpoints use nOpt (adapter params, or pre-training N).
  const optFlopsN = Math.max(Math.trunc(ftWithBase && nFlopsBase ? baseParamsActive(i) : nOpt), 1);
  const optFlops = calculateTrainingFlops(
    optFlopsN,
    Math.max(Math.trunc(dOpt), 1),
    architecture.toLowerCase(),
    i.epochs,
    hw.gradient_checkpointing
  );
  const optRun = estimateComputeCost(
    optFlops,
    hw.peak_flops_per_gpu,
    hw.mfu,
    hw.gpus_per_instance,
    1,
    hw.hourly_cost
  );
  const optComputeCostTotal = optRun.compute_cost * projectMultiplier;
  const optDatasetStorageCost = calculateStorageCost(
    (Math.max(Math.trunc(dOpt), 1) * bytesPerToken) / 1e12,
    i.storage_duration_months,
    i.storage_class
  );
  const optCkptStorageCost = calculateStorageCost(
    calculateCheckpointStorageTb(
      Math.max(Math.trunc(nOpt), 1),
      CHECKPOINTS_PER_RUN,
      1,
      i.num_hp_trials,
      0
    ),
    i.storage_duration_months,
    i.storage_class
  );
  const optStorageCost = optDatasetStorageCost + optCkptStorageCost;

  return {
    n_opt: nOpt,
    d_opt: dOpt,
    n_adapter: nAdapter,
    n_flops_base: nFlopsBase,
    d_eff: dEff,
    d_budget: dBudget,
    opt_compute_cost_total: optComputeCostTotal,
    opt_dataset_storage_cost: optDatasetStorageCost,
    opt_ckpt_storage_cost: optCkptStorageCost,
    opt_storage_cost: optStorageCost,
    total_project_cost: optComputeCostTotal + optStorageCost,
    project_multiplier: projectMultiplier,
    cost_per_token_param: costPerTokenParam,
    ckpt_cost_per_param: ckptCostPerParam,
    storage_cost_per_token: storageCostPerToken,
    cost_per_tb_month: costPerTbMonth,
    optimal_ratio: optimalRatio
  };
}

/** User overrides. `undefined` means "not overridden" — use the derived default. */
export interface Selection {
  explore_dir?: string;
  log_d?: number;
  log_n?: number;
  rank?: number;
  num_instances?: number;
}

export interface SelectionResult {
  explore_dir: string;
  explore_options: string[];
  selected_tokens: number;
  selected_params: number;
  flops_params: number;
  n_axis_title: string;
  n_hover: string;
  total_flops_sel: number;
  compute_cost_sel: number;
  wall_clock_days_sel: number;
  wall_clock_hours_1inst: number;
  recommended_instances: number;
  num_instances: number;
  sel_dataset_storage: number;
  sel_ckpt_storage: number;
  sel_compute_cost_total: number;
  sel_total_cost: number;
  sel_wall_clock_days: number;
  sel_samples: number;
  current_ratio: number;
  budget_used_pct: number;
  d_slider_min: number;
  d_slider_max: number;
  n_slider_min: number;
  n_slider_max: number;
  d_opt_log_default: number;
  n_opt_log_default: number;
  slider_effective_budget: number;
  tokens_per_sample: number;
  sample_noun: string;
}

export function resolveSelection(
  rawInputs: OptimizerInputs,
  optimum: BudgetOptimum,
  selection: Selection = {}
): SelectionResult {
  const i = resolveInputs(rawInputs);
  const hw = i.hardware;

  const computeBudget = i.compute_budget;
  const modalityCfg = MODALITY_DEFAULTS[i.modality];
  const bytesPerToken = modalityCfg.bytes_per_token;
  const architecture = modalityCfg.arch;
  const isLora = isLoraMethod(i.ft_method);
  const ftWithBase = isFtWithBase(i);

  const { d_opt: dOpt, n_opt: nOpt, n_adapter: nAdapter } = optimum;
  const projectMultiplier = optimum.project_multiplier;

  const tokensPerSample = modalityCfg.tokens_per_sample;
  const sampleNoun = modalityCfg.sample_noun;

  const sliderComputeBudget = Math.max(
    computeBudget - optimum.opt_storage_cost,
    computeBudget * 0.5
  );
  const sliderEffectiveBudget = sliderComputeBudget / projectMultiplier;

  // For LoRA, d_opt can sit well below 1B tokens, so anchor ~2 decades below it rather
  // than hardcoding 1B.
  const dSliderMin =
    ftWithBase && isLora && dOpt > 0
      ? Math.max(6.0, Math.floor(Math.log10(Math.max(dOpt, 1e6)) / 0.05) * 0.05 - 2.0)
      : 9.0;

  const dLogOpt = Math.log10(Math.max(dOpt, 10 ** dSliderMin));
  const dSliderMax = Math.max(dSliderMin + 4.0, Math.ceil(dLogOpt / 0.05) * 0.05 + 0.05);

  const nLogOpt = Math.log10(Math.max(nOpt, 1e7));
  const nSliderMin = 7.0;
  const nSliderMax = Math.max(11.0, Math.ceil(nLogOpt / 0.05) * 0.05 + 0.05);

  const dOptLogDefault =
    Math.round(Math.max(dSliderMin, Math.min(dSliderMax, dLogOpt)) / 0.05) * 0.05;
  const nOptLogDefault =
    Math.round(Math.max(nSliderMin, Math.min(nSliderMax, nLogOpt)) / 0.05) * 0.05;

  const options = exploreOptions(i);
  const exploreDir =
    selection.explore_dir !== undefined && options.includes(selection.explore_dir)
      ? selection.explore_dir
      : options[0];

  let selectedTokens: number;
  let selectedParams: number;
  let flopsParams: number;
  let nAxisTitle: string;
  let nHover: string;

  if (ftWithBase && isLora) {
    if (exploreDir === EXPLORE_LORA_RANK) {
      const selRank = Math.trunc(selection.rank ?? i.lora_rank);
      const selNAdapter = adapterParams(i, selRank) || 1;
      const denom =
        projectMultiplier * optimum.cost_per_token_param * baseParamsActive(i) +
        optimum.storage_cost_per_token;
      const effBgt = Math.max(computeBudget - optimum.ckpt_cost_per_param * selNAdapter, 0.0);
      selectedTokens = denom > 0 ? Math.max(Math.trunc(effBgt / denom), 1) : 1;
      selectedParams = selNAdapter;
    } else {
      const logD = selection.log_d ?? dOptLogDefault;
      selectedTokens = Math.trunc(10 ** logD);
      selectedParams = Math.max(nAdapter, 1);
    }
    // FLOPs always run through the full base model (frozen forward/backward pass)
    flopsParams = baseParamsActive(i);
    nAxisTitle = "Adapter params";
    nHover = "Adapter N";
  } else if (ftWithBase) {
    const logD = selection.log_d ?? dOptLogDefault;
    selectedTokens = Math.trunc(10 ** logD);
    selectedParams = baseParamsTotal(i);
    flopsParams = baseParamsActive(i);
    nAxisTitle = "Model size (params)";
    nHover = "N";
  } else {
    if (exploreDir === EXPLORE_TOKENS_TO_MODEL) {
      const logD = selection.log_d ?? dOptLogDefault;
      selectedTokens = Math.trunc(10 ** logD);
      selectedParams = solveForParameterCount(
        sliderEffectiveBudget,
        selectedTokens,
        hw.peak_flops_per_gpu,
        hw.mfu,
        hw.gpus_per_instance,
        1,
        hw.hourly_cost,
        architecture.toLowerCase(),
        i.epochs,
        hw.gradient_checkpointing
      );
    } else {
      const logN = selection.log_n ?? nOptLogDefault;
      selectedParams = Math.trunc(10 ** logN);
      selectedTokens = solveForTrainingTokens(
        sliderEffectiveBudget,
        selectedParams,
        hw.peak_flops_per_gpu,
        hw.mfu,
        hw.gpus_per_instance,
        1,
        hw.hourly_cost,
        architecture.toLowerCase(),
        i.epochs,
        hw.gradient_checkpointing
      );
    }
    flopsParams = selectedParams;
    nAxisTitle = "Max model size (params)";
    nHover = "Max N";
  }

  const totalFlopsSel = calculateTrainingFlops(
    Math.max(flopsParams, 1),
    Math.max(selectedTokens, 1),
    architecture.toLowerCase(),
    i.epochs,
    hw.gradient_checkpointing
  );
  const selRun = estimateComputeCost(
    totalFlopsSel,
    hw.peak_flops_per_gpu,
    hw.mfu,
    hw.gpus_per_instance,
    1,
    hw.hourly_cost
  );

  // Derived after the slider solve so the hardware card and the wall-clock metric always
  // come from the same FLOP count.
  const wallClockHours1Inst =
    totalFlopsSel / (hw.peak_flops_per_gpu * hw.mfu * hw.gpus_per_instance * 3600);
  const recommendedInstances = Math.min(
    MAX_INSTANCES,
    Math.max(1, Math.ceil(wallClockHours1Inst / (TARGET_WALL_CLOCK_DAYS * 24)))
  );
  const numInstances = selection.num_instances ?? recommendedInstances;

  const selDatasetStorage = calculateStorageCost(
    (Math.max(selectedTokens, 1) * bytesPerToken) / 1e12,
    i.storage_duration_months,
    i.storage_class
  );
  const selCkptStorage = calculateStorageCost(
    calculateCheckpointStorageTb(
      Math.max(selectedParams, 1),
      CHECKPOINTS_PER_RUN,
      1,
      i.num_hp_trials,
      0
    ),
    i.storage_duration_months,
    i.storage_class
  );
  const selComputeCostTotal = selRun.compute_cost * projectMultiplier;
  const selTotalCost = selComputeCostTotal + selDatasetStorage + selCkptStorage;

  return {
    explore_dir: exploreDir,
    explore_options: options,
    selected_tokens: selectedTokens,
    selected_params: selectedParams,
    flops_params: flopsParams,
    n_axis_title: nAxisTitle,
    n_hover: nHover,
    total_flops_sel: totalFlopsSel,
    compute_cost_sel: selRun.compute_cost,
    wall_clock_days_sel: selRun.wall_clock_days,
    wall_clock_hours_1inst: wallClockHours1Inst,
    recommended_instances: recommendedInstances,
    num_instances: numInstances,
    sel_dataset_storage: selDatasetStorage,
    sel_ckpt_storage: selCkptStorage,
    sel_compute_cost_total: selComputeCostTotal,
    sel_total_cost: selTotalCost,
    sel_wall_clock_days: wallClockHours1Inst / (numInstances * 24),
    sel_samples: selectedTokens / Math.max(tokensPerSample, 1),
    current_ratio: selectedTokens / Math.max(selectedParams, 1),
    budget_used_pct: Math.min(selTotalCost / Math.max(computeBudget, 1e-9), 1.0),
    d_slider_min: dSliderMin,
    d_slider_max: dSliderMax,
    n_slider_min: nSliderMin,
    n_slider_max: nSliderMax,
    d_opt_log_default: dOptLogDefault,
    n_opt_log_default: nOptLogDefault,
    slider_effective_budget: sliderEffectiveBudget,
    tokens_per_sample: tokensPerSample,
    sample_noun: sampleNoun
  };
}
