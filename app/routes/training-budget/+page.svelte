<script lang="ts">
  import Assessment from "$lib/components/Assessment.svelte";
  import CostBars from "$lib/components/CostBars.svelte";
  import CostDonut from "$lib/components/CostDonut.svelte";
  import Field from "$lib/components/Field.svelte";
  import Figure from "$lib/components/Figure.svelte";
  import ScaledNumber from "$lib/components/ScaledNumber.svelte";
  import { calculateLoraTrainableParams } from "$lib/engine/calculator";
  import {
    ARCHITECTURE_OPTIONS,
    CHART_COLORS,
    MIXED_PRECISION_OPTIONS,
  } from "$lib/engine/constants";
  import { DEFAULT_TARGET_MODULES } from "$lib/engine/budgetOptimizer";
  import {
    ENCODEC_TOKENS_PER_SECOND,
    SOUNDSTREAM_TOKENS_PER_SECOND,
    WHISPER_TOKENS_PER_SECOND,
    bytesPerAudioSample,
    bytesPerImageSample,
    bytesPerTextSample,
    bytesPerVideoSample,
    tokensPerAudioSample,
    tokensPerImageSample,
    tokensPerTextSample,
    tokensPerVideoSample,
  } from "$lib/engine/dataset";
  import {
    UNIT_MULTIPLIERS,
    fmtTokens,
    formatE,
    formatGpuHours,
    formatWallClockTime,
    money,
  } from "$lib/engine/formatting";

  // Line items carry cents; the headline figures elsewhere do not.
  const moneyCents = (n: number) => money(n, true);
  import {
    INSTANCE_ORDER,
    MODELS,
    S3_STORAGE_PRICING,
    getGpuInstance,
    peakFlopsForPrecision,
  } from "$lib/engine/gpuSpecs";
  import { assessTrainingConfig } from "$lib/engine/scalingLaws";
  import { estimateTrainingBudget } from "$lib/engine/trainingBudget";
  import type { FineTuningMethod } from "$lib/engine/types";

  let modality = $state<"Text" | "Image" | "Audio" | "Video" | "">("");
  let datasetSize = $state(100);
  let datasetUnit = $state("M");
  let storageMonths = $state(3);

  let avgWords = $state(400);
  let resolution = $state(224);
  let patchSize = $state(16);
  let clipDuration = $state(30);
  let audioTokenizer = $state("Whisper");
  let codebooks = $state(8);
  let vidDuration = $state(10);
  let sampledFps = $state(1);
  let vidResolution = $state(224);
  let vidPatchSize = $state(16);

  const samples = $derived(Math.trunc(datasetSize * UNIT_MULTIPLIERS[datasetUnit]));

  const perSample = $derived.by(() => {
    switch (modality) {
      case "Text": {
        const t = tokensPerTextSample(avgWords);
        return { tokens: t, bytes: bytesPerTextSample(t) };
      }
      case "Image":
        return {
          tokens: tokensPerImageSample(resolution, patchSize),
          bytes: bytesPerImageSample(resolution),
        };
      case "Audio": {
        const tokens =
          audioTokenizer === "Whisper"
            ? tokensPerAudioSample(clipDuration, WHISPER_TOKENS_PER_SECOND)
            : tokensPerAudioSample(
                clipDuration,
                audioTokenizer === "EnCodec"
                  ? ENCODEC_TOKENS_PER_SECOND
                  : SOUNDSTREAM_TOKENS_PER_SECOND,
                codebooks,
              );
        return { tokens, bytes: bytesPerAudioSample(clipDuration) };
      }
      case "Video":
        return {
          tokens: tokensPerVideoSample(vidDuration, sampledFps, vidResolution, vidPatchSize),
          bytes: bytesPerVideoSample(vidDuration, sampledFps, vidResolution),
        };
      default:
        return { tokens: 0, bytes: 0 };
    }
  });

  const hasDataset = $derived(modality !== "" && samples > 0 && perSample.tokens > 0);

  let trainingType = $state<"Pre-Training" | "Fine-Tuning">("Pre-Training");
  let architecture = $state("Transformer");
  let preParams = $state(7);
  let preUnit = $state("B");
  let dModel = $state(4096);
  let numLayers = $state(32);

  let baseModelName = $state(MODELS[0].name);
  let ftMethod = $state<Exclude<FineTuningMethod, null>>("Full Fine-Tuning");
  let loraRank = $state(16);
  let rlAlgorithm = $state("Supervised");

  const isFt = $derived(trainingType === "Fine-Tuning");
  const baseModel = $derived(MODELS.find((m) => m.name === baseModelName) ?? MODELS[0]);
  const isAdapter = $derived(isFt && (ftMethod === "LoRA" || ftMethod === "QLoRA"));

  const trainableParams = $derived(
    !isFt
      ? 0
      : isAdapter
        ? (calculateLoraTrainableParams(
            ftMethod, baseModel.parameter_count,
            DEFAULT_TARGET_MODULES,
            baseModel.architecture.d_model ?? 0,
            baseModel.architecture.num_layers ?? 0,
            loraRank, baseModel.architecture,
          ) ?? 0)
        : baseModel.parameter_count,
  );

  // DPO keeps a reference model in memory; PPO adds a value and reward model too.
  const rlMultiplier = $derived(
    rlAlgorithm === "DPO" ? 2 : rlAlgorithm === "PPO" ? 4 : 1,
  );

  let epochs = $state(1);
  let gradientCheckpointing = $state(false);
  let instanceType = $state(INSTANCE_ORDER[0]);
  let numInstances = $state(1);
  let mixedPrecision = $state("bf16");
  let mfuPct = $state(30);
  let batchSize = $state(1024);

  const preParamCount = $derived(Math.trunc(preParams * UNIT_MULTIPLIERS[preUnit]));
  const instanceSpec = $derived(getGpuInstance(instanceType));

  /**
   * Whether this GPU actually has hardware for a precision. The estimate already falls
   * back to the FP16 rate when it doesn't, but silently — so say so in the selector
   * rather than letting someone pick a format expecting a speedup that cannot happen.
   */
  const acceleratedOn = (precision: string) =>
      !["fp4", "fp8", "int8"].includes(precision) ||
    peakFlopsForPrecision(instanceSpec, precision) >
      peakFlopsForPrecision(instanceSpec, "fp16");

  const precisionNote = $derived(
    acceleratedOn(mixedPrecision)
      ? undefined
      : `${instanceSpec.gpu} has no ${mixedPrecision} path, so this is costed at the fp16 rate.`,
  );

  let numTrainingRuns = $state(1);
  let numCheckpoints = $state(5);
  let numHpTrials = $state(0);
  let hpFraction = $state(0.3);
  let numAblations = $state(0);
  let ablationFraction = $state(0.5);
  let storageClass = $state("standard");

  const budget = $derived(
    estimateTrainingBudget({
      modality: modality || "Text",
      dataset_size: samples,
      tokens_per_sample: perSample.tokens,
      bytes_per_sample: perSample.bytes,
      storage_months: storageMonths,
      architecture,
      parameter_count: isFt ? 0 : preParamCount,
      base_params: isFt ? baseModel.parameter_count : 0,
      base_flops_params: isFt
        ? (baseModel.architecture.moe?.active_parameter_count ?? baseModel.parameter_count)
        : 0,
      ft_method: isFt ? ftMethod : null,
      trainable_params: trainableParams,
      d_model: isFt ? (baseModel.architecture.d_model ?? 0) : dModel,
      num_layers: isFt ? (baseModel.architecture.num_layers ?? 0) : numLayers,
      seq_len: perSample.tokens,
      rl_multiplier: rlMultiplier,
      epochs,
      gradient_checkpointing: gradientCheckpointing,
      instance_type: instanceType,
      num_instances: numInstances,
      mixed_precision: mixedPrecision,
      mfu: mfuPct / 100,
      batch_size: batchSize,
      num_training_runs: numTrainingRuns,
      num_checkpoints: numCheckpoints,
      num_hp_trials: numHpTrials,
      hp_fraction: hpFraction,
      num_ablations: numAblations,
      ablation_fraction: ablationFraction,
      storage_class: storageClass,
    }),
  );

  // Assessed on dataset tokens, not tokens-seen: epochs are applied inside the FLOPs
  // calculation, and the scaling-law thresholds are defined against dataset size.
  const assessment = $derived(
    assessTrainingConfig(
      budget.total_tokens,
      isFt ? ftMethod : null,
      isFt ? baseModel.parameter_count : 0,
      isFt ? 0 : preParamCount,
      trainableParams,
    ),
  );



  const sizeLabel = $derived.by(() => {
    const tb = budget.dataset_size_tb;
    if (tb >= 1) return `${tb.toFixed(2)} TB`;
    if (tb >= 0.001) return `${(tb * 1000).toFixed(2)} GB`;
    return `${(tb * 1e6).toFixed(2)} MB`;
  });

  const runTypeSegments = $derived([
    {
      label: "Full runs",
      value: budget.compute_cost * numTrainingRuns,
      detail: `${numTrainingRuns} × full training run`,
      color: CHART_COLORS.compute,
    },
    {
      label: "HP trials",
      value: budget.compute_cost * hpFraction * numHpTrials,
      detail: numHpTrials
        ? `${numHpTrials} × ${(hpFraction * 100).toFixed(0)}% of a run`
        : "none planned",
      color: CHART_COLORS.warning,
    },
    {
      label: "Ablations",
      value: budget.compute_cost * ablationFraction * numAblations,
      detail: numAblations
        ? `${numAblations} × ${(ablationFraction * 100).toFixed(0)}% of a run`
        : "none planned",
      color: CHART_COLORS.success,
    },
  ]);

  /**
   * Gradient accumulation and effective batch size are deliberately absent: Streamlit
   * collected them but they feed no calculation (memory uses batch_size, not the
   * effective batch), and an input that changes no number is worse than no input.
   */
  const recap = $derived([
    ["Dataset", "Modality", modality],
    ["Dataset", "Samples", samples.toLocaleString("en-US")],
    ["Dataset", "Tokens per sample", fmtTokens(perSample.tokens)],
    ["Dataset", "Total tokens", fmtTokens(budget.total_tokens)],
    ["Dataset", "Estimated size", sizeLabel],
    ["Dataset", "Storage duration", `${storageMonths} months`],

    ["Model", "Training type", trainingType],
    ["Model", "Model / base", isFt ? baseModelName : architecture],
    [
      "Model",
      "Parameter count",
      fmtTokens(isFt ? baseModel.parameter_count : preParamCount),
    ],
    ["Model", "Fine-tuning method", isFt ? ftMethod : "N/A"],
    ["Model", "Trainable parameters", isFt ? fmtTokens(trainableParams) : "all"],
    ["Model", "Alignment", isFt ? rlAlgorithm : "N/A"],

    ["Cluster", "Instance type", instanceType],
    ["Cluster", "Instances", String(numInstances)],
    ["Cluster", "Total GPUs", String(budget.total_gpus)],
    ["Cluster", "Mixed precision", mixedPrecision],
    ["Cluster", "MFU", `${mfuPct}%`],
    ["Cluster", "Epochs", String(epochs)],
    ["Cluster", "Batch size", batchSize.toLocaleString("en-US")],
    ["Cluster", "Gradient checkpointing", gradientCheckpointing ? "Yes" : "No"],

    ["Run", "Total FLOPs", formatE(budget.total_flops, 2)],
    ["Run", "GPU-hours (1 run)", formatGpuHours(budget.gpu_hours)],
    ["Run", "Wall-clock (1 run)", formatWallClockTime(budget.wall_clock_days)],
    ["Run", "Memory per GPU", `${budget.memory_per_gpu_gb.toFixed(1)} GB`],
    ["Run", "VRAM available", `${budget.vram_per_gpu} GB`],

    ["Experiment", "Full training runs", String(numTrainingRuns)],
    ["Experiment", "Checkpoints per run", String(numCheckpoints)],
    ["Experiment", "HP tuning trials", String(numHpTrials)],
    ["Experiment", "HP trial length", `${(hpFraction * 100).toFixed(0)}% of a run`],
    ["Experiment", "Ablation studies", String(numAblations)],
    ["Experiment", "Ablation length", `${(ablationFraction * 100).toFixed(0)}% of a run`],
    ["Experiment", "Total experiment runs", String(budget.total_experiment_runs)],
    ["Experiment", "Checkpoint storage", `${budget.checkpoint_storage_tb.toFixed(4)} TB`],
    ["Experiment", "S3 storage class", storageClass.replace(/_/g, " ")],

    ["Cost", "Compute (all runs)", moneyCents(budget.total_compute_cost)],
    ["Cost", "Dataset storage", moneyCents(budget.dataset_storage_cost)],
    ["Cost", "Checkpoint storage", moneyCents(budget.checkpoint_storage_cost)],
    ["Cost", "Total project cost", moneyCents(budget.total_project_cost)],
  ] as const);

</script>

<svelte:head>
  <title>Training budget — Floply</title>
  <meta
    name="description"
    content="Estimate the full cost of a training project: GPU compute, dataset storage, checkpoints, hyperparameter trials and ablations."
  />
</svelte:head>

<h1 class="text-2xl font-semibold tracking-tight">What will this project cost?</h1>
<p class="mt-2 max-w-[62ch] text-sm leading-relaxed text-[var(--color-ink-muted)]">
  Describe the dataset, the model and the cluster. Each section unlocks as the one above
  it is answered.
</p>

<section class="mt-8 border-t border-[var(--color-rule)] pt-6">
  <h2 class="text-lg font-semibold tracking-tight">Dataset</h2>
  <div class="mt-4 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
    <Field label="Modality" id="modality">
      <select id="modality" bind:value={modality} class="control">
        <option value="" disabled>Choose one…</option>
        <option>Text</option>
        <option>Image</option>
        <option>Audio</option>
        <option>Video</option>
      </select>
    </Field>

    {#if modality}
      <Field label="Dataset size (samples)" id="dataset-size">
        <ScaledNumber
          id="dataset-size"
          bind:value={datasetSize}
          bind:unit={datasetUnit}
          units={["K", "M", "B", "T"]}
        />
      </Field>

      {#if modality === "Text"}
        <Field label="Average length (words)" id="avg-words" hint="1 word ≈ 1.3 BPE tokens.">
          <input id="avg-words" type="number" bind:value={avgWords} min="1" max="100000" step="50" class="control num" />
        </Field>
      {:else if modality === "Image"}
        <Field label="Resolution (px)" id="resolution">
          <select id="resolution" bind:value={resolution} class="control num">
            {#each [224, 336, 512, 1024] as r (r)}<option value={r}>{r}</option>{/each}
          </select>
        </Field>
        <Field label="Patch size (px)" id="patch">
          <select id="patch" bind:value={patchSize} class="control num">
            {#each [14, 16, 32] as p (p)}<option value={p}>{p}</option>{/each}
          </select>
        </Field>
      {:else if modality === "Audio"}
        <Field label="Clip duration (seconds)" id="clip">
          <input id="clip" type="number" bind:value={clipDuration} min="0.1" max="3600" step="5" class="control num" />
        </Field>
        <Field label="Tokenizer" id="tokenizer">
          <select id="tokenizer" bind:value={audioTokenizer} class="control">
            <option value="Whisper">Whisper (50 tok/sec)</option>
            <option value="EnCodec">EnCodec 24kHz (75 tok/sec × codebooks)</option>
            <option value="SoundStream">SoundStream (50 tok/sec × codebooks)</option>
          </select>
        </Field>
        {#if audioTokenizer !== "Whisper"}
          <Field label="Codebooks" id="codebooks">
            <input id="codebooks" type="number" bind:value={codebooks} min="1" max="16" class="control num" />
          </Field>
        {/if}
      {:else if modality === "Video"}
        <Field label="Clip duration (seconds)" id="vid-dur">
          <input id="vid-dur" type="number" bind:value={vidDuration} min="0.1" max="3600" step="1" class="control num" />
        </Field>
        <Field label="Sampled FPS" id="fps" hint="Frames sampled per second, not source frame rate.">
          <input id="fps" type="number" bind:value={sampledFps} min="1" max="60" class="control num" />
        </Field>
        <Field label="Frame resolution (px)" id="vid-res">
          <select id="vid-res" bind:value={vidResolution} class="control num">
            {#each [224, 336, 512] as r (r)}<option value={r}>{r}</option>{/each}
          </select>
        </Field>
        <Field label="Patch size (px)" id="vid-patch">
          <select id="vid-patch" bind:value={vidPatchSize} class="control num">
            {#each [14, 16, 32] as p (p)}<option value={p}>{p}</option>{/each}
          </select>
        </Field>
      {/if}

      <Field label="Storage duration (months)" id="storage-months">
        <input id="storage-months" type="range" bind:value={storageMonths} min="1" max="36" />
        <span class="num text-sm">{storageMonths}</span>
      </Field>
    {/if}
  </div>

  {#if hasDataset}
    <dl class="mt-6 grid gap-6 border-t border-[var(--color-rule)] pt-5 sm:grid-cols-3">
      <Figure label="Tokens per sample" value={fmtTokens(perSample.tokens)} />
      <Figure label="Total tokens" value={fmtTokens(budget.total_tokens)} />
      <Figure label="Dataset size (est.)" value={sizeLabel} />
    </dl>
  {:else}
    <p class="mt-4 text-sm text-[var(--color-ink-muted)]">
      Choose a modality and dataset size to continue.
    </p>
  {/if}
</section>

{#if hasDataset}
  <section class="mt-10 border-t border-[var(--color-rule)] pt-6">
    <h2 class="text-lg font-semibold tracking-tight">Model</h2>
    <div class="mt-4 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
      <Field label="Training type" id="training-type">
        <select id="training-type" bind:value={trainingType} class="control">
          <option>Pre-Training</option>
          <option>Fine-Tuning</option>
        </select>
      </Field>

      {#if isFt}
        <Field label="Base model" id="base-model">
          <select id="base-model" bind:value={baseModelName} class="control">
            {#each MODELS as m (m.slug)}<option value={m.name}>{m.name}</option>{/each}
          </select>
        </Field>
        <Field label="Fine-tuning method" id="ft-method">
          <select id="ft-method" bind:value={ftMethod} class="control">
            <option>Full Fine-Tuning</option>
            <option>LoRA</option>
            <option>QLoRA</option>
          </select>
        </Field>
        {#if isAdapter}
          <Field label="LoRA rank (r)" id="rank">
            <input id="rank" type="range" bind:value={loraRank} min="1" max="128" />
            <span class="num text-sm">{loraRank}</span>
          </Field>
        {/if}
        <Field label="Alignment" id="rl" hint="DPO holds a reference model in memory; PPO adds value and reward models.">
          <select id="rl" bind:value={rlAlgorithm} class="control">
            <option>Supervised</option>
            <option>DPO</option>
            <option>PPO</option>
          </select>
        </Field>
      {:else}
        <Field label="Architecture" id="arch">
          <select id="arch" bind:value={architecture} class="control">
            {#each ARCHITECTURE_OPTIONS as a (a)}<option>{a}</option>{/each}
          </select>
        </Field>
        <Field label="Parameter count" id="pre-params">
          <ScaledNumber id="pre-params" bind:value={preParams} bind:unit={preUnit} units={["M", "B", "T"]} />
        </Field>
        <Field label="Hidden dimension" id="d-model">
          <input id="d-model" type="number" bind:value={dModel} min="64" max="65536" step="64" class="control num" />
        </Field>
        <Field label="Layers" id="layers">
          <input id="layers" type="number" bind:value={numLayers} min="1" max="512" class="control num" />
        </Field>
      {/if}
    </div>

    {#if isAdapter}
      <p class="mt-4 text-xs text-[var(--color-ink-faint)]">
        <span class="num">{fmtTokens(trainableParams)}</span> trainable adapter parameters
        &mdash; checkpoints store these, not the {fmtTokens(baseModel.parameter_count)}-parameter base.
      </p>
    {/if}

    <Assessment {assessment} class="mt-5" />
  </section>

  <section class="mt-10 border-t border-[var(--color-rule)] pt-6">
    <h2 class="text-lg font-semibold tracking-tight">Cluster</h2>
    <div class="mt-4 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
      <Field label="Instance type" id="instance">
        <select id="instance" bind:value={instanceType} class="control">
          {#each INSTANCE_ORDER as t (t)}
            <option value={t}>{getGpuInstance(t).display_name} — ${getGpuInstance(t).hourly_cost.toFixed(2)}/hr</option>
          {/each}
        </select>
      </Field>
      <Field label="Instances" id="instances" hint="{instanceSpec.gpu_count} GPUs each, {budget.total_gpus} total.">
        <input id="instances" type="number" bind:value={numInstances} min="1" max="512" class="control num" />
      </Field>
      <Field label="Epochs" id="epochs">
        <input id="epochs" type="number" bind:value={epochs} min="1" max="100" class="control num" />
      </Field>
      <Field
        label="Mixed precision"
        id="precision"
        hint={precisionNote}
      >
        <select id="precision" bind:value={mixedPrecision} class="control">
          {#each MIXED_PRECISION_OPTIONS as p (p)}
            <option value={p}>
              {p}{acceleratedOn(p) ? "" : " — not accelerated on this GPU"}
            </option>
          {/each}
        </select>
      </Field>
      <Field label="Model FLOPs utilisation" id="mfu" hint="Share of peak throughput actually achieved. 30% is a conservative real-world default.">
        <input id="mfu" type="range" bind:value={mfuPct} min="5" max="100" step="5" />
        <span class="num text-sm">{mfuPct}%</span>
      </Field>
      <Field label="Batch size" id="batch">
        <input id="batch" type="number" bind:value={batchSize} min="1" max="65536" class="control num" />
      </Field>
      <Field label="Gradient checkpointing" id="grad-ckpt" hint="Recomputes activations to save memory. Adds ~33% FLOPs.">
        <label class="flex items-center gap-2 text-sm">
          <input id="grad-ckpt" type="checkbox" bind:checked={gradientCheckpointing} />
          Enabled
        </label>
      </Field>
    </div>

    <dl class="mt-6 grid gap-6 border-t border-[var(--color-rule)] pt-5 sm:grid-cols-2 lg:grid-cols-4">
      <Figure label="Total FLOPs" value={formatE(budget.total_flops, 2)} />
      <Figure label="Wall-clock (1 run)" value={formatWallClockTime(budget.wall_clock_days)} />
      <Figure label="GPU-hours (1 run)" value="{formatGpuHours(budget.gpu_hours)} GPU-hrs" />
      <Figure
        label="Memory per GPU"
        value="{budget.memory_per_gpu_gb.toFixed(1)} GB"
        note="{budget.vram_per_gpu} GB available"
        tone={budget.memory_fits ? "success" : "warning"}
      />
    </dl>
    {#if !budget.memory_fits}
      <p class="mt-3 text-sm text-[var(--color-warning)]">
        This configuration exceeds the GPU's memory. Reduce batch size, enable gradient
        checkpointing, or switch to a larger-memory instance.
      </p>
    {/if}
  </section>

  <section class="mt-10 border-t border-[var(--color-rule)] pt-6">
    <h2 class="text-lg font-semibold tracking-tight">Experiments</h2>
    <p class="mt-1.5 max-w-[62ch] text-sm text-[var(--color-ink-muted)]">
      Real projects are never one run. Hyperparameter trials and ablations usually cost
      more than the final model.
    </p>
    <div class="mt-4 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
      <Field label="Full training runs" id="runs">
        <input id="runs" type="number" bind:value={numTrainingRuns} min="1" max="100" class="control num" />
      </Field>
      <Field label="Checkpoints per run" id="ckpts">
        <input id="ckpts" type="number" bind:value={numCheckpoints} min="1" max="200" class="control num" />
      </Field>
      <Field label="HP tuning trials" id="hp">
        <input id="hp" type="number" bind:value={numHpTrials} min="0" max="1000" class="control num" />
      </Field>
      <Field label="HP trial length" id="hp-frac" hint="Fraction of a full run's cost.">
        <input id="hp-frac" type="number" bind:value={hpFraction} min="0.05" max="1" step="0.05" class="control num" />
      </Field>
      <Field label="Ablation studies" id="abl">
        <input id="abl" type="number" bind:value={numAblations} min="0" max="100" class="control num" />
      </Field>
      <Field label="Ablation length" id="abl-frac" hint="Fraction of a full run's cost.">
        <input id="abl-frac" type="number" bind:value={ablationFraction} min="0.05" max="1" step="0.05" class="control num" />
      </Field>
      <Field label="S3 storage class" id="storage-class">
        <select id="storage-class" bind:value={storageClass} class="control">
          {#each Object.keys(S3_STORAGE_PRICING) as c (c)}
            <option value={c}>{c.replace(/_/g, " ")}</option>
          {/each}
        </select>
      </Field>
    </div>
  </section>

  <section class="mt-10 border-t border-[var(--color-rule-strong)] pt-6">
    <h2 class="text-lg font-semibold tracking-tight">Total project cost</h2>

    <dl class="mt-5 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
      <Figure label="Compute (1 run)" value={moneyCents(budget.compute_cost)} />
      <Figure
        label="Compute (all {budget.total_experiment_runs} runs)"
        value={moneyCents(budget.total_compute_cost)}
      />
      <Figure label="Storage" value={moneyCents(budget.dataset_storage_cost + budget.checkpoint_storage_cost)} />
      <Figure label="Total" value={moneyCents(budget.total_project_cost)} tone="accent" />
    </dl>

    <div class="mt-8">
      <CostDonut
        total={budget.total_project_cost}
        segments={[
          { label: "Compute", value: budget.total_compute_cost, color: CHART_COLORS.compute },
          { label: "Dataset storage", value: budget.dataset_storage_cost, color: CHART_COLORS.storage },
          { label: "Checkpoint storage", value: budget.checkpoint_storage_cost, color: CHART_COLORS.checkpoint },
        ]}
      />
    </div>

    <p class="mt-6 text-xs text-[var(--color-ink-faint)]">
      {budget.total_gpus} GPUs · {instanceSpec.gpu} · ${instanceSpec.hourly_cost.toFixed(2)}/hr
      per instance · {(budget.peak_flops_per_gpu / 1e12).toFixed(0)} TFLOPS peak
      ({mixedPrecision}) · {budget.checkpoint_storage_tb.toFixed(2)} TB of checkpoints
    </p>
  </section>

  <section class="mt-10 border-t border-[var(--color-rule)] pt-6">
    <h2 class="text-lg font-semibold tracking-tight">Compute by run type</h2>
    <p class="mt-1.5 max-w-[62ch] text-sm text-[var(--color-ink-muted)]">
      Sweeps and ablations often cost more than the model you keep. A single total hides
      that.
    </p>
    <div class="mt-5">
      <CostBars segments={runTypeSegments} />
    </div>
  </section>

  <section class="mt-10 border-t border-[var(--color-rule)] pt-6">
    <h2 class="text-lg font-semibold tracking-tight">Run configuration</h2>
    <p class="mt-1.5 max-w-[62ch] text-sm text-[var(--color-ink-muted)]">
      Everything this estimate assumed, in one place.
    </p>

    <div class="mt-5 overflow-x-auto">
      <table class="w-full text-sm">
        <tbody>
          {#each recap as [category, label, value], i (category + label)}
            {@const first = i === 0 || recap[i - 1][0] !== category}
            <tr class={first && i > 0 ? "border-t border-[var(--color-rule-strong)]" : ""}>
              <td
                class="w-28 py-1.5 pr-4 align-top text-[var(--color-ink-faint)]"
                >{first ? category : ""}</td
              >
              <td class="py-1.5 pr-4 align-top text-[var(--color-ink-muted)]">{label}</td>
              <td class="num py-1.5 align-top text-[var(--color-ink)]">{value}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>
{/if}
