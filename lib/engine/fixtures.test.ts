/**
 * The fixtures were generated from the Python implementation and verified against the
 * running Streamlit app, so this is the contract the port has to meet. The Python twin of
 * this file is tests/test_fixtures.py; both must stay green until Python is deleted.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import * as calc from "./calculator";
import * as bo from "./budgetOptimizer";
import * as ds from "./dataset";
import * as fmt from "./formatting";
import * as sl from "./scalingLaws";
import { MODELS, displayName, effectiveParameterCount, getGpuInstance, peakFlopsForPrecision } from "./gpuSpecs";
import { FULL_FT_TIERS, LORA_TIERS, PRE_TRAINING_TIERS } from "./tiers";
import type { Tier } from "./tiers";
import { estimateTrainingBudget } from "./trainingBudget";
import type { ModelDefinition } from "./types";

const FIXTURES = join(process.cwd(), "fixtures");
const load = (name: string) =>
  JSON.parse(readFileSync(join(FIXTURES, name), "utf8")) as Record<string, any>;

const REL_TOL = 1e-12;
// Cases flagged as exceeding 2^53: Python keeps exact integers there, float64 cannot.
const UNSAFE_TOL = 1e-9;

type Case = {
  args: Record<string, any>;
  expect?: unknown;
  raises?: string;
  exceeds_float64_safe_int?: boolean;
  arg_type?: string;
  name?: string;
  selection?: Record<string, any>;
};

function approx(got: unknown, want: unknown, tol: number, path = ""): void {
  if (typeof want === "number" && typeof got === "number") {
    if (Number.isNaN(want) || Number.isNaN(got)) {
      expect(Number.isNaN(got), path).toBe(Number.isNaN(want));
      return;
    }
    if (want === 0) {
      expect(Math.abs(got), path).toBeLessThanOrEqual(tol);
      return;
    }
    expect(Math.abs(got - want) / Math.abs(want), `${path} (got ${got}, want ${want})`)
      .toBeLessThanOrEqual(tol);
    return;
  }
  if (Array.isArray(want)) {
    expect(Array.isArray(got), path).toBe(true);
    const g = got as unknown[];
    expect(g.length, path).toBe(want.length);
    want.forEach((w, i) => approx(g[i], w, tol, `${path}[${i}]`));
    return;
  }
  if (want !== null && typeof want === "object") {
    expect(got !== null && typeof got === "object", path).toBe(true);
    const g = got as Record<string, unknown>;
    for (const [k, w] of Object.entries(want as Record<string, unknown>)) {
      approx(g[k], w, tol, `${path}.${k}`);
    }
    return;
  }
  expect(got, path).toEqual(want);
}

const A = (c: Case) => c.args;

const DISPATCH: Record<string, (c: Case) => unknown> = {
  _get_flops_multiplier: (c) =>
    calc.getFlopsMultiplier(A(c).architecture, A(c).gradient_checkpointing),
  calculate_training_flops: (c) =>
    calc.calculateTrainingFlops(
      A(c).parameter_count, A(c).training_tokens, A(c).architecture,
      A(c).epochs, A(c).gradient_checkpointing,
    ),
  estimate_compute_cost: (c) =>
    calc.estimateComputeCost(
      A(c).total_flops, A(c).peak_flops_per_gpu, A(c).mfu,
      A(c).total_gpus, A(c).num_instances, A(c).hourly_cost,
    ),
  solve_for_parameter_count: (c) =>
    calc.solveForParameterCount(
      A(c).compute_budget_usd, A(c).training_tokens, A(c).peak_flops_per_gpu,
      A(c).mfu, A(c).total_gpus, A(c).num_instances, A(c).hourly_cost,
      A(c).architecture, A(c).epochs, A(c).gradient_checkpointing,
    ),
  solve_for_training_tokens: (c) =>
    calc.solveForTrainingTokens(
      A(c).compute_budget_usd, A(c).parameter_count, A(c).peak_flops_per_gpu,
      A(c).mfu, A(c).total_gpus, A(c).num_instances, A(c).hourly_cost,
      A(c).architecture, A(c).epochs, A(c).gradient_checkpointing,
    ),
  estimate_gpu_memory_gb: (c) =>
    calc.estimateGpuMemoryGb(
      A(c).effective_params, A(c).trainable_params, A(c).ft_method,
      A(c).total_gpus, A(c).rl_multiplier, A(c).d_model, A(c).num_layers,
      A(c).seq_len, A(c).batch_size, A(c).gradient_checkpointing,
    ),
  calculate_lora_trainable_params: (c) =>
    calc.calculateLoraTrainableParams(
      A(c).ft_method, A(c).base_params, A(c).target_modules,
      A(c).d_model, A(c).num_layers, A(c).lora_rank, A(c).architecture,
    ),
  calculate_checkpoint_storage_tb: (c) =>
    calc.calculateCheckpointStorageTb(
      A(c).checkpoint_params, A(c).num_checkpoints, A(c).num_training_runs,
      A(c).num_hp_trials, A(c).num_ablations,
    ),
  calculate_project_compute_cost: (c) =>
    calc.calculateProjectComputeCost(
      A(c).single_run_cost, A(c).num_training_runs, A(c).num_hp_trials,
      A(c).hp_fraction, A(c).num_ablations, A(c).ablation_fraction,
    ),
  calculate_storage_cost: (c) =>
    calc.calculateStorageCost(
      A(c).dataset_size_tb, A(c).storage_duration_months, A(c).storage_class,
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
  bytes_per_video_sample: (c) =>
    ds.bytesPerVideoSample(A(c).duration, A(c).fps, A(c).resolution),
  fmt_tokens: (c) => fmt.fmtTokens(A(c).n),
  fmt_samples: (c) => fmt.fmtSamples(A(c).n),
  format_wall_clock_time: (c) => fmt.formatWallClockTime(A(c).wall_clock_days),
  format_gpu_hours: (c) => fmt.formatGpuHours(A(c).gpu_hours),
  // JSON can't distinguish Python's 1000.0 from 1000; arg_type carries the source type.
  display_value: (c) => fmt.displayValue(A(c).val, c.arg_type === "float"),
  assess_training_config: (c) =>
    sl.assessTrainingConfig(
      A(c).total_tokens, A(c).ft_method, A(c).base_params,
      A(c).pre_params, A(c).adapter_params,
    ),
  assess_lora_ratio: (c) => sl.assessLoraRatio(A(c).ratio),
  assess_chinchilla_ratio: (c) =>
    sl.assessChinchillaRatio(A(c).ratio, A(c).optimal_tokens, A(c).fix_hint),
  derive_schedule_defaults: (c) =>
    bo.deriveScheduleDefaults(
      A(c).compute_budget, A(c).modality, A(c).training_type,
      A(c).ft_method, A(c).quick_n,
    ),
  logspace: (c) => bo.logspace(A(c).start, A(c).stop, A(c).num),
  budget_curve_n: (c) =>
    bo.budgetCurveN(
      A(c).d_tokens, A(c).budget, A(c).peak_flops_per_gpu, A(c).mfu,
      A(c).gpus_per_instance, A(c).hourly_cost, A(c).multiplier, A(c).epochs,
    ),
};

describe("engine.json", () => {
  const data = load("engine.json");
  const models = new Map(MODELS.map((m) => [m.slug, m]));

  for (const [group, cases] of Object.entries(data)) {
    if (group === "_meta") continue;

    it(`${group} (${(cases as Case[]).length} cases)`, () => {
      (cases as Case[]).forEach((c, i) => {
        const where = `${group}[${i}]`;

        if (group === "training_budget") {
          approx(estimateTrainingBudget(c.args), c.expect, REL_TOL, where);
          return;
        }

        if (group === "tiers") {
          const tables: Record<string, Tier[]> = {
            PRE_TRAINING_TIERS, FULL_FT_TIERS, LORA_TIERS,
          };
          // Colour is presentation, not behaviour — see gen_fixtures.py.
          const got = tables[c.args.table].map(({ color: _color, ...rest }) => rest);
          expect(got, where).toEqual(c.expect);
          return;
        }

        if (group === "model_definitions") {
          const m = models.get(c.args.slug) as ModelDefinition;
          expect(
            {
              name: m.name, family: m.family,
              parameter_count: m.parameter_count,
              effective_parameter_count: effectiveParameterCount(m),
              display_name: displayName(m),
            },
            where,
          ).toEqual(c.expect);
          return;
        }

        if (c.raises) {
          // Python raises; the TS port deliberately returns zeroes for the division
          // guards so the UI can't render "$Infinity". Only the lookup errors throw.
          if (c.raises === "ValueError") {
            expect(() => DISPATCH[group](c), where).toThrow();
          } else {
            const got = DISPATCH[group](c);
            const values =
              typeof got === "object" && got !== null
                ? Object.values(got as Record<string, unknown>)
                : [got];
            expect(values.length, `${where} produced nothing to check`).toBeGreaterThan(0);
            values.forEach((v) =>
              expect(
                typeof v === "number" && Number.isFinite(v),
                `${where} must be a finite number, not Infinity/NaN`,
              ).toBe(true),
            );
          }
          return;
        }

        const tol = c.exceeds_float64_safe_int ? UNSAFE_TOL : REL_TOL;
        approx(DISPATCH[group](c), c.expect, tol, where);
      });
    });
  }
});

describe("budget_optimizer.json", () => {
  const data = load("budget_optimizer.json");
  const models = new Map(MODELS.map((m) => [m.slug, m]));
  const hardware = bo.autoConfigureHardware();

  it(`solve (${data.solve.length} scenarios)`, () => {
    for (const c of data.solve as Case[]) {
      const args = { ...c.args } as bo.OptimizerInputs & { ft_base_model?: any };
      if (typeof args.ft_base_model === "string") {
        args.ft_base_model = models.get(args.ft_base_model)!;
      }
      const inputs: bo.OptimizerInputs = { ...args, hardware };
      const optimum = bo.solveBudgetOptimum(inputs);
      const sel = bo.resolveSelection(inputs, optimum, c.selection ?? {});

      const tol = c.exceeds_float64_safe_int ? UNSAFE_TOL : REL_TOL;
      approx({ optimum, selection: sel }, c.expect, tol, c.name ?? "");
    }
  });

  for (const group of ["derive_schedule_defaults", "logspace", "budget_curve_n"]) {
    it(`${group} (${data[group].length} cases)`, () => {
      (data[group] as Case[]).forEach((c, i) => {
        if (c.raises) return;
        approx(DISPATCH[group](c), c.expect, REL_TOL, `${group}[${i}]`);
      });
    });
  }
});
