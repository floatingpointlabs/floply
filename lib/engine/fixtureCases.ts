/**
 * Builds the golden fixtures from the TypeScript engine.
 *
 * Ported from the original scripts/gen_fixtures.py, which generated them from the Python
 * implementation. The case enumeration is deliberately identical: it was ported alongside
 * the engine and validated by regenerating and diffing against the Python-produced files
 * while both stacks still existed.
 *
 * Conventions carried over from the Python original:
 *   - Args are always named, never positional, so a fixture diff is readable.
 *   - A case that throws records `raises` instead of `expect`.
 *   - Budgets are capped at $1e12, above which the numbers stop meaning anything.
 */

import { DISPATCH, solveCase } from "./fixtureDispatch";
import type { Case } from "./fixtureDispatch";
import { INSTANCE_ORDER, MODELS } from "./gpuSpecs";
import { EXPLORE_LORA_RANK, EXPLORE_MODEL_TO_TOKENS } from "./budgetOptimizer";

type Payload = Record<string, Case[]>;

const ARCHES = ["transformer", "cnn", "rnn", "vit", "diffusion", "definitely_not_an_arch"];
const PRECISIONS = ["fp4", "int8", "fp8", "bf16", "fp16", "tf32", "fp32"];
const STORAGE_CLASSES = [
  "standard",
  "intelligent_tiering",
  "standard_ia",
  "one_zone_ia",
  "glacier"
];
const LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"];

/** Call the engine through the shared dispatch, recording the result or the throw. */
function record(group: string, args: Record<string, any>, extra: Record<string, any> = {}): Case {
  const probe: Case = { args, ...extra };
  try {
    return { args, expect: DISPATCH[group](probe), ...extra };
  } catch (err) {
    return { args, raises: (err as Error)?.constructor?.name ?? "Error", ...extra };
  }
}

export function buildEngine(): Payload {
  const out: Payload = {};

  out._get_flops_multiplier = ARCHES.flatMap((architecture) =>
    [false, true].map((gradient_checkpointing) =>
      record("_get_flops_multiplier", { architecture, gradient_checkpointing })
    )
  );

  out.calculate_training_flops = [
    [7_000_000_000, 140_000_000_000],
    [1, 1],
    [0, 1_000],
    [1_000, 0]
  ].flatMap(([parameter_count, training_tokens]) =>
    ["transformer", "diffusion"].flatMap((architecture) =>
      [1.0, 3.0].flatMap((epochs) =>
        [false, true].map((gradient_checkpointing) =>
          record("calculate_training_flops", {
            parameter_count,
            training_tokens,
            architecture,
            epochs,
            gradient_checkpointing
          })
        )
      )
    )
  );

  const cost = (total_flops: number, mfu: number, total_gpus: number, num_instances: number) =>
    record("estimate_compute_cost", {
      total_flops,
      peak_flops_per_gpu: 9.89e14,
      mfu,
      total_gpus,
      num_instances,
      hourly_cost: 66.64
    });

  out.estimate_compute_cost = [5.88e21, 1.0].flatMap((f) =>
    [0.05, 0.55].flatMap((m) =>
      [
        [8, 1],
        [1024, 128]
      ].map(([gpus, inst]) => cost(f, m, gpus, inst))
    )
  );
  // Divergence markers: Python raised ZeroDivisionError here; the TS port returns zeroes
  // so the UI can't render "$Infinity", so these now carry an ordinary expectation.
  out.estimate_compute_cost.push(cost(1e21, 0.0, 8, 1), cost(1e21, 0.5, 0, 1));

  const solverCommon = {
    peak_flops_per_gpu: 9.89e14,
    mfu: 0.55,
    total_gpus: 8,
    num_instances: 1,
    hourly_cost: 66.64,
    architecture: "transformer",
    gradient_checkpointing: false
  };

  out.solve_for_parameter_count = [
    ...[
      [10_000.0, 140_000_000_000, 1.0],
      [1_000_000.0, 1_000_000_000_000, 1.0],
      [1_000_000_000_000.0, 1_000_000_000_000, 1.0], // $1e12 cap
      [0.01, 1_000_000_000_000, 1.0], // tiny budget → 0
      [10_000.0, 0, 1.0], // zero-guard: tokens
      [10_000.0, 140_000_000_000, 0.0], // zero-guard: epochs→tokens
      [10_000.0, 140_000_000_000, 3.0],
      [10_000.0, 140_000_000_000, 0.5]
    ].map(([compute_budget_usd, training_tokens, epochs]) =>
      record("solve_for_parameter_count", {
        compute_budget_usd,
        training_tokens,
        epochs,
        ...solverCommon
      })
    ),
    record("solve_for_parameter_count", {
      compute_budget_usd: 10_000.0,
      training_tokens: 1_000_000_000,
      epochs: 1.0,
      ...solverCommon,
      hourly_cost: 0.0
    }),
    record("solve_for_parameter_count", {
      compute_budget_usd: 10_000.0,
      training_tokens: 1_000_000_000,
      epochs: 1.0,
      ...solverCommon,
      num_instances: 0
    })
  ];

  out.solve_for_training_tokens = [
    ...[
      [10_000.0, 7_000_000_000, 1.0],
      [1_000_000.0, 70_000_000_000, 1.0],
      [1_000_000_000_000.0, 7_000_000_000, 1.0],
      [0.01, 7_000_000_000, 1.0],
      [10_000.0, 0, 1.0], // zero-guard: params
      [10_000.0, 7_000_000_000, 3.0],
      [10_000.0, 7_000_000_000, 0.5], // exercises max(epochs, 1.0)
      [10_000.0, 7_000_000_000, 0.0]
    ].map(([compute_budget_usd, parameter_count, epochs]) =>
      record("solve_for_training_tokens", {
        compute_budget_usd,
        parameter_count,
        epochs,
        ...solverCommon
      })
    ),
    record("solve_for_training_tokens", {
      compute_budget_usd: 10_000.0,
      parameter_count: 7_000_000_000,
      epochs: 1.0,
      ...solverCommon,
      hourly_cost: 0.0
    }),
    record("solve_for_training_tokens", {
      compute_budget_usd: 10_000.0,
      parameter_count: 7_000_000_000,
      epochs: 1.0,
      ...solverCommon,
      num_instances: 0
    })
  ];

  out.estimate_gpu_memory_gb = [
    ["QLoRA", 13_631_488],
    ["LoRA", 13_631_488],
    ["Full Fine-Tuning", 7_000_000_000],
    [null, 7_000_000_000]
  ].flatMap(([ft_method, trainable_params]) =>
    [
      [4096, false],
      [4096, true],
      [0, false]
    ].flatMap(([d_model, gradient_checkpointing]) =>
      [1, 2, 4].map((rl_multiplier) =>
        record("estimate_gpu_memory_gb", {
          effective_params: 7_000_000_000,
          trainable_params,
          ft_method,
          total_gpus: 8,
          rl_multiplier,
          d_model,
          num_layers: 32,
          seq_len: 4096,
          batch_size: 8,
          gradient_checkpointing
        })
      )
    )
  );

  const llama70b = {
    num_layers: 80,
    d_model: 8192,
    num_heads: 64,
    num_kv_heads: 8,
    ffn_intermediate: 28672
  };
  const mha = {
    num_layers: 32,
    d_model: 4096,
    num_heads: 32,
    num_kv_heads: 32,
    ffn_intermediate: 11008
  };
  const noHeads = { num_layers: 32, d_model: 4096, num_heads: 0 };
  // 4096 // 30 == 136, but 4096 / 30 == 136.533… — the floor-division tripwire.
  const unevenHeads = {
    num_layers: 32,
    d_model: 4096,
    num_heads: 30,
    num_kv_heads: 6,
    ffn_intermediate: 11008
  };

  out.calculate_lora_trainable_params = [
    // Full FT short-circuits and returns base_params
    {
      ft_method: "Full Fine-Tuning",
      base_params: 7_000_000_000,
      target_modules: [],
      d_model: 0,
      num_layers: 0,
      lora_rank: 0,
      architecture: {}
    },
    // Each null-return path, one at a time
    {
      ft_method: "LoRA",
      base_params: 7e9,
      target_modules: [],
      d_model: 4096,
      num_layers: 32,
      lora_rank: 16,
      architecture: mha
    },
    {
      ft_method: "LoRA",
      base_params: 7e9,
      target_modules: ["q_proj"],
      d_model: 0,
      num_layers: 32,
      lora_rank: 16,
      architecture: {}
    },
    {
      ft_method: "LoRA",
      base_params: 7e9,
      target_modules: ["q_proj"],
      d_model: 4096,
      num_layers: 0,
      lora_rank: 16,
      architecture: mha
    },
    {
      ft_method: "LoRA",
      base_params: 7e9,
      target_modules: ["q_proj"],
      d_model: 4096,
      num_layers: 32,
      lora_rank: 0,
      architecture: mha
    },
    // GQA: num_kv_heads < num_heads, exercising the k/v_proj kv_dim path
    {
      ft_method: "LoRA",
      base_params: 70e9,
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"],
      d_model: 8192,
      num_layers: 80,
      lora_rank: 16,
      architecture: llama70b
    },
    // Synthetic: d_model % num_heads != 0, so `//` and `/` genuinely disagree. Every real
    // model divides evenly, so nothing realistic pins this — but a port using `/` instead
    // of Math.floor produces a different number here.
    {
      ft_method: "LoRA",
      base_params: 7e9,
      target_modules: ["k_proj", "v_proj"],
      d_model: 4096,
      num_layers: 32,
      lora_rank: 16,
      architecture: unevenHeads
    },
    // Uniform fallback for custom models (architecture={})
    {
      ft_method: "LoRA",
      base_params: 7e9,
      target_modules: ["q_proj", "k_proj"],
      d_model: 4096,
      num_layers: 32,
      lora_rank: 16,
      architecture: {}
    },
    // num_heads=0 → headDim falls back to d
    {
      ft_method: "QLoRA",
      base_params: 7e9,
      target_modules: ["k_proj"],
      d_model: 4096,
      num_layers: 32,
      lora_rank: 8,
      architecture: noHeads
    },
    // Unknown module name → (d, d)
    {
      ft_method: "LoRA",
      base_params: 7e9,
      target_modules: ["mystery_proj"],
      d_model: 4096,
      num_layers: 32,
      lora_rank: 16,
      architecture: mha
    },
    // All six modules; ffn_intermediate default (4 × d_model) when absent
    {
      ft_method: "LoRA",
      base_params: 7e9,
      target_modules: LORA_MODULES,
      d_model: 4096,
      num_layers: 32,
      lora_rank: 64,
      architecture: mha
    },
    {
      ft_method: "LoRA",
      base_params: 7e9,
      target_modules: ["up_proj", "down_proj"],
      d_model: 4096,
      num_layers: 32,
      lora_rank: 16,
      architecture: { d_model: 4096, num_heads: 32 }
    }
  ].map((c) => record("calculate_lora_trainable_params", c));

  out.calculate_checkpoint_storage_tb = [
    [7_000_000_000, 5, 1, 0, 0],
    [7_000_000_000, 5, 3, 10, 2],
    [13_631_488, 5, 1, 5, 0],
    [0, 5, 1, 0, 0]
  ].map(([checkpoint_params, num_checkpoints, num_training_runs, num_hp_trials, num_ablations]) =>
    record("calculate_checkpoint_storage_tb", {
      checkpoint_params,
      num_checkpoints,
      num_training_runs,
      num_hp_trials,
      num_ablations
    })
  );

  out.calculate_project_compute_cost = [
    [10_000.0, 1, 0, 0.1, 0, 0.5],
    [10_000.0, 3, 10, 0.1, 2, 0.5],
    [0.0, 1, 1, 0.1, 1, 0.5],
    [1.0, 1, 20, 0.05, 0, 0.5]
  ].map(
    ([
      single_run_cost,
      num_training_runs,
      num_hp_trials,
      hp_fraction,
      num_ablations,
      ablation_fraction
    ]) =>
      record("calculate_project_compute_cost", {
        single_run_cost,
        num_training_runs,
        num_hp_trials,
        hp_fraction,
        num_ablations,
        ablation_fraction
      })
  );

  out.calculate_storage_cost = [
    ...[...STORAGE_CLASSES, "not_a_storage_class"].map((storage_class) =>
      record("calculate_storage_cost", {
        dataset_size_tb: 10.0,
        storage_duration_months: 3.0,
        storage_class
      })
    ),
    record("calculate_storage_cost", {
      dataset_size_tb: 0.0,
      storage_duration_months: 1.0,
      storage_class: "standard"
    })
  ];

  out.peak_flops_for_precision = [
    ...INSTANCE_ORDER.flatMap((instance_type) =>
      PRECISIONS.map((mixed_precision) =>
        record("peak_flops_for_precision", { instance_type, mixed_precision })
      )
    ),
    record("peak_flops_for_precision", {
      instance_type: "p5.48xlarge",
      mixed_precision: "not_a_precision"
    })
  ];

  out.tokens_per_text_sample = [0, 1, 100, 500, 999].map((avg_seq_len_words) =>
    record("tokens_per_text_sample", { avg_seq_len_words })
  );
  out.bytes_per_text_sample = [0, 1, 650, 1_000_000].map((tokens) =>
    record("bytes_per_text_sample", { tokens })
  );
  out.tokens_per_image_sample = [
    // 336 // 32 = 10 → 100, NOT 110.25
    [224, 16],
    [336, 32],
    [224, 14],
    [512, 16],
    [255, 16],
    [1, 16]
  ].map(([resolution, patch_size]) =>
    record("tokens_per_image_sample", { resolution, patch_size })
  );
  out.bytes_per_image_sample = [224, 336, 512, 1, 3].map((resolution) =>
    record("bytes_per_image_sample", { resolution })
  );
  out.tokens_per_audio_sample = [
    [10.0, 75, 1],
    [10.5, 75, 4],
    [0.5, 50, 1],
    [3.33, 50, 8],
    [0.0, 75, 1]
  ].map(([clip_duration, base_rate, num_codebooks]) =>
    record("tokens_per_audio_sample", { clip_duration, base_rate, num_codebooks })
  );
  out.bytes_per_audio_sample = [10.0, 0.5, 3.33, 0.0].map((clip_duration) =>
    record("bytes_per_audio_sample", { clip_duration })
  );
  out.tokens_per_video_sample = [
    [10.0, 30, 224, 16],
    [2.5, 24, 336, 32],
    [0.5, 30, 224, 16],
    [0.0, 30, 224, 16]
  ].map(([duration, fps, resolution, patch_size]) =>
    record("tokens_per_video_sample", { duration, fps, resolution, patch_size })
  );
  out.bytes_per_video_sample = [
    [10.0, 30, 224],
    [2.5, 24, 336],
    [0.04, 30, 512]
  ].map(([duration, fps, resolution]) =>
    record("bytes_per_video_sample", { duration, fps, resolution })
  );

  const tokenVals = [
    0, 1, 999, 1_000, 1_001, 999_999, 1_000_000, 1_500_000, 999_999_999, 1_000_000_000,
    7_000_000_000, 999_999_999_999, 1_000_000_000_000, 671_000_000_000
  ];
  out.fmt_tokens = tokenVals.map((n) => record("fmt_tokens", { n }));
  out.fmt_samples = tokenVals.map((n) => record("fmt_samples", { n }));
  out.format_wall_clock_time = [
    0.0,
    0.0005,
    1 / 24 - 1e-9,
    1 / 24,
    0.5,
    0.999,
    1.0,
    5.176,
    365.0
  ].map((wall_clock_days) => record("format_wall_clock_time", { wall_clock_days }));
  out.format_gpu_hours = [0.0, 1.0, 999.9, 1_000.0, 999_999.0, 1_000_000.0, 2_500_000.0].map(
    (gpu_hours) => record("format_gpu_hours", { gpu_hours })
  );

  // JSON cannot distinguish Python's 1000.0 from 1000, but displayValue formats them
  // differently ("1.0K" vs "1,000"). The source type is recorded so the port can too.
  out.display_value = [
    [0.5, "float"],
    [3.7, "float"],
    [999.0, "float"],
    [1_000.0, "float"],
    [1.5e6, "float"],
    [2.5e9, "float"],
    [1e12, "float"],
    [5e12, "float"],
    [true, "bool"],
    [false, "bool"],
    [0, "int"],
    [42, "int"],
    [999, "int"],
    [1_000, "int"],
    [1_500_000, "int"],
    ["standard", "str"],
    ["p5.48xlarge", "str"]
  ].map(([val, arg_type]) => record("display_value", { val }, { arg_type }));

  const ratios = [
    0.05, 0.09, 0.1, 0.5, 0.99, 1.0, 1.5, 5.0, 9.99, 10.0, 20.0, 30.0, 30.1, 100.0, 200.0, 200.1,
    999.0, 1000.0, 1000.1, 5000.0
  ];
  const n = 1_000_000_000;
  out.assess_training_config = [
    ...ratios.flatMap((r) =>
      [
        [null, 0, n, 0],
        ["Full Fine-Tuning", n, 0, 0],
        ["LoRA", 8_000_000_000, 0, n],
        ["QLoRA", 8_000_000_000, 0, n]
      ].map(([ft_method, base_params, pre_params, adapter_params]) =>
        record("assess_training_config", {
          total_tokens: Math.trunc(r * n),
          ft_method,
          base_params,
          pre_params,
          adapter_params
        })
      )
    ),
    ...[
      [0, null, 0, 1_000, 0],
      [-5, null, 0, 1_000, 0],
      [100, null, 0, 0, 0],
      [100, "Full Fine-Tuning", 0, 0, 0],
      [100, "LoRA", 0, 0, 0],
      [100, "LoRA", 1_000, 0, 0]
    ].map(([total_tokens, ft_method, base_params, pre_params, adapter_params]) =>
      record("assess_training_config", {
        total_tokens,
        ft_method,
        base_params,
        pre_params,
        adapter_params
      })
    )
  ];

  out.assess_lora_ratio = [0.5, 5.0, 9.99, 10.0, 25.0, 50.0, 99.9, 100.0, 100.1, 500.0, 5000.0].map(
    (ratio) => record("assess_lora_ratio", { ratio })
  );

  out.assess_chinchilla_ratio = [
    0.5, 0.99, 1.0, 5.0, 9.99, 10.0, 25.0, 30.0, 30.1, 150.0, 200.0, 200.1, 1e4
  ].flatMap((ratio) =>
    ["", "Move the slider toward the star."].map((fix_hint) =>
      record("assess_chinchilla_ratio", {
        ratio,
        optimal_tokens: 20_000_000_000,
        fix_hint
      })
    )
  );

  // Pins the wiring, not the maths. Verified against the running Streamlit Tab 3
  // (10/10 rendered metrics) before these were first generated.
  out.training_budget = [
    // The scenario checked against the live app.
    {
      modality: "Text",
      dataset_size: 100_000_000,
      tokens_per_sample: 520,
      bytes_per_sample: 2080,
      storage_months: 3,
      architecture: "Transformer",
      parameter_count: 7_000_000_000,
      d_model: 4096,
      num_layers: 32,
      seq_len: 520,
      epochs: 1,
      instance_type: "p4d.24xlarge",
      num_instances: 1,
      mixed_precision: "bf16",
      mfu: 0.3,
      batch_size: 1024
    },
    // Full fine-tuning off a base model.
    {
      modality: "Text",
      dataset_size: 1_000_000,
      tokens_per_sample: 650,
      bytes_per_sample: 2600,
      storage_months: 6,
      architecture: "Transformer",
      base_params: 7_000_000_000,
      base_flops_params: 7_000_000_000,
      ft_method: "Full Fine-Tuning",
      trainable_params: 7_000_000_000,
      d_model: 4096,
      num_layers: 32,
      seq_len: 650,
      epochs: 3,
      instance_type: "p5.48xlarge",
      num_instances: 2,
      mixed_precision: "fp8",
      mfu: 0.45,
      batch_size: 64,
      num_training_runs: 2,
      num_hp_trials: 4,
      storage_class: "standard_ia"
    },
    // LoRA — checkpoints hold adapters only, not the base model.
    {
      modality: "Text",
      dataset_size: 500_000,
      tokens_per_sample: 520,
      bytes_per_sample: 2080,
      storage_months: 1,
      architecture: "Transformer",
      base_params: 70_000_000_000,
      base_flops_params: 70_000_000_000,
      ft_method: "LoRA",
      trainable_params: 13_631_488,
      d_model: 8192,
      num_layers: 80,
      seq_len: 520,
      epochs: 5,
      instance_type: "p5.48xlarge",
      num_instances: 4,
      mixed_precision: "bf16",
      mfu: 0.4,
      batch_size: 16,
      num_hp_trials: 8,
      num_ablations: 2
    },
    // MoE base: FLOPs on active params, checkpoints and VRAM on the total.
    {
      modality: "Text",
      dataset_size: 2_000_000,
      tokens_per_sample: 520,
      bytes_per_sample: 2080,
      storage_months: 12,
      architecture: "Transformer",
      base_params: 671_000_000_000,
      base_flops_params: 37_000_000_000,
      ft_method: "QLoRA",
      trainable_params: 27_262_976,
      d_model: 7168,
      num_layers: 61,
      seq_len: 520,
      epochs: 2,
      instance_type: "p5.48xlarge",
      num_instances: 8,
      mixed_precision: "bf16",
      mfu: 0.5,
      batch_size: 8,
      storage_class: "glacier"
    },
    // Vision, RL memory multiplier, and a V100 cluster.
    {
      modality: "Image",
      dataset_size: 10_000_000,
      tokens_per_sample: 196,
      bytes_per_sample: 15052,
      storage_months: 3,
      architecture: "ViT",
      parameter_count: 1_000_000_000,
      d_model: 1024,
      num_layers: 24,
      seq_len: 196,
      rl_multiplier: 4,
      epochs: 1,
      instance_type: "p3.16xlarge",
      num_instances: 16,
      mixed_precision: "fp16",
      mfu: 0.25,
      batch_size: 256,
      num_training_runs: 3,
      num_ablations: 5,
      ablation_fraction: 0.25
    },
    // Degenerate: no data at all.
    {
      modality: "Text",
      dataset_size: 0,
      tokens_per_sample: 0,
      bytes_per_sample: 0,
      parameter_count: 1_000_000_000
    }
  ].map((c) => record("training_budget", c));

  out.tiers = ["PRE_TRAINING_TIERS", "FULL_FT_TIERS", "LORA_TIERS"].map((table) =>
    record("tiers", { table })
  );

  out.model_definitions = MODELS.map((m) => record("model_definitions", { slug: m.slug }));

  return out;
}

export function buildBudgetOptimizer(): Payload {
  const setups: [string, Record<string, any>][] = [
    [
      "pretrain_text_10k",
      { compute_budget: 10_000.0, modality: "Text (LLM)", training_type: "Pre-Training" }
    ],
    [
      "pretrain_text_1e12_cap",
      { compute_budget: 1e12, modality: "Text (LLM)", training_type: "Pre-Training" }
    ],
    [
      "pretrain_vision_500k",
      {
        compute_budget: 500_000.0,
        modality: "Vision (ViT / CLIP)",
        training_type: "Pre-Training",
        num_hp_trials: 10
      }
    ],
    [
      "pretrain_diffusion_1m",
      {
        compute_budget: 1_000_000.0,
        modality: "Diffusion (Image Gen)",
        training_type: "Pre-Training"
      }
    ],
    [
      "pretrain_audio_250k",
      {
        compute_budget: 250_000.0,
        modality: "Audio",
        training_type: "Pre-Training",
        epochs: 4,
        num_hp_trials: 7,
        hp_fraction_pct: 20,
        storage_duration_months: 9,
        storage_class: "glacier"
      }
    ],
    [
      "pretrain_multimodal_2m",
      {
        compute_budget: 2_000_000.0,
        modality: "Multimodal (VLM)",
        training_type: "Pre-Training"
      }
    ],
    [
      "lora_llama8b_5k",
      {
        compute_budget: 5_000.0,
        modality: "Text (LLM)",
        training_type: "Fine-Tuning",
        ft_method: "LoRA",
        ft_base_model: "llama3_8b",
        lora_rank: 16,
        epochs: 5
      }
    ],
    [
      "qlora_llama70b_50k",
      {
        compute_budget: 50_000.0,
        modality: "Text (LLM)",
        training_type: "Fine-Tuning",
        ft_method: "QLoRA",
        ft_base_model: "llama3_70b",
        lora_rank: 128,
        epochs: 5
      }
    ],
    [
      "sft_mistral7b_100k",
      {
        compute_budget: 100_000.0,
        modality: "Text (LLM)",
        training_type: "Fine-Tuning",
        ft_method: "Full Fine-Tuning (SFT)",
        ft_base_model: "mistral_7b",
        epochs: 3
      }
    ],
    [
      "lora_moe_deepseek_1m",
      {
        compute_budget: 1_000_000.0,
        modality: "Text (LLM)",
        training_type: "Fine-Tuning",
        ft_method: "LoRA",
        ft_base_model: "deepseek_v3",
        lora_rank: 32,
        epochs: 5
      }
    ],
    [
      "sft_moe_deepseek_10m",
      {
        compute_budget: 10_000_000.0,
        modality: "Text (LLM)",
        training_type: "Fine-Tuning",
        ft_method: "Full Fine-Tuning (SFT)",
        ft_base_model: "deepseek_v3",
        epochs: 3
      }
    ],
    [
      "lora_no_target_modules",
      {
        compute_budget: 50_000.0,
        modality: "Text (LLM)",
        training_type: "Fine-Tuning",
        ft_method: "LoRA",
        ft_base_model: "llama3_8b",
        lora_rank: 16,
        target_modules: []
      }
    ],
    [
      "lora_gqa_qwen72b",
      {
        compute_budget: 2_000_000.0,
        modality: "Multimodal (VLM)",
        training_type: "Fine-Tuning",
        ft_method: "QLoRA",
        ft_base_model: "qwen25_72b",
        lora_rank: 8,
        target_modules: ["q_proj", "up_proj", "down_proj"]
      }
    ],
    [
      "ft_custom_no_base",
      {
        compute_budget: 50_000.0,
        modality: "Text (LLM)",
        training_type: "Fine-Tuning",
        ft_method: "LoRA"
      }
    ],
    ["tiny_budget", { compute_budget: 1.0, modality: "Text (LLM)", training_type: "Pre-Training" }]
  ];

  const selections: [string, Record<string, any>][] = [
    ["default", {}],
    ["dataset_slider", { log_d: 11.35 }],
    ["model_slider", { explore_dir: EXPLORE_MODEL_TO_TOKENS, log_n: 9.75 }],
    ["rank_slider", { explore_dir: EXPLORE_LORA_RANK, rank: 64 }],
    ["instances_override", { num_instances: 37 }]
  ];

  const solve: Case[] = [];
  for (const [setupName, args] of setups) {
    for (const [selName, selection] of selections) {
      solve.push({
        name: `${setupName}__${selName}`,
        args,
        selection,
        expect: solveCase(args, selection)
      });
    }
  }

  const schedule = [
    1_000.0, 5_000.0, 10_000.0, 50_000.0, 100_000.0, 500_000.0, 1_000_000.0, 5_000_000.0,
    10_000_000.0
  ].flatMap((compute_budget) =>
    ["Text (LLM)", "Diffusion (Image Gen)"].flatMap((modality) =>
      [
        ["Pre-Training", null],
        ["Fine-Tuning", "LoRA"],
        ["Fine-Tuning", "Full Fine-Tuning (SFT)"]
      ].flatMap(([training_type, ft_method]) =>
        [1e8, 6e8, 5e9, 50e9].map((quick_n) =>
          record("derive_schedule_defaults", {
            compute_budget,
            modality,
            training_type,
            ft_method,
            quick_n
          })
        )
      )
    )
  );

  return {
    solve,
    derive_schedule_defaults: schedule,
    logspace: [
      [1.0, 3.0, 3],
      [6.0, 12.0, 5],
      [9.0, 13.0, 2],
      [1.0, 2.0, 1],
      [1.0, 2.0, 0]
    ].map(([start, stop, num]) => record("logspace", { start, stop, num })),
    budget_curve_n: [
      record("budget_curve_n", {
        d_tokens: [1e9, 1e11, 1e13],
        budget: 1e5,
        peak_flops_per_gpu: 9.89e14,
        mfu: 0.55,
        gpus_per_instance: 8,
        hourly_cost: 66.64,
        multiplier: 6.0,
        epochs: 1
      })
    ]
  };
}
