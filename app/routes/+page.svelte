<script lang="ts">
  /**
   * The faithful port is `override ?? derived`, with an effect that clears the overrides
   * on a setup change. Note this is deliberately NOT Svelte's reassign-a-$derived idiom:
   * that reverts on *any* dependency change, so editing epochs would silently discard a
   * slider override that Streamlit keeps. What Svelte does buy here is that the derived
   * defaults track their inputs automatically — no dependency array to get wrong.
   */
  import Assessment from "$lib/components/Assessment.svelte";
  import BudgetFrontier from "$lib/components/BudgetFrontier.svelte";
  import Field from "$lib/components/Field.svelte";
  import Figure from "$lib/components/Figure.svelte";
  import ScaledNumber from "$lib/components/ScaledNumber.svelte";
  import {
    ARCH_MULTIPLIERS,
    EXPLORE_LORA_RANK,
    EXPLORE_MODEL_TO_TOKENS,
    MODALITY_DEFAULTS,
    autoConfigureHardware,
    budgetCurveN,
    deriveScheduleDefaults,
    logspace,
    quickNEstimate,
    resolveSelection,
    solveBudgetOptimum
  } from "$lib/engine/budgetOptimizer";
  import { CHINCHILLA_OPTIMAL_RATIO, LORA_OPTIMAL_RATIO } from "$lib/engine/constants";
  import { UNIT_MULTIPLIERS, fmtTokens, formatWallClockTime, money } from "$lib/engine/formatting";
  import { MODELS, S3_STORAGE_PRICING } from "$lib/engine/gpuSpecs";
  import { assessChinchillaRatio, assessLoraRatio } from "$lib/engine/scalingLaws";

  const hardware = autoConfigureHardware();

  let budgetValue = $state(10);
  let budgetUnit = $state("K");
  let modality = $state("Text (LLM)");
  let trainingType = $state<"Pre-Training" | "Fine-Tuning">("Pre-Training");
  let ftMethod = $state<"Full Fine-Tuning (SFT)" | "LoRA" | "QLoRA">("Full Fine-Tuning (SFT)");
  let baseModelName = $state<string>("");
  let loraRank = $state(16);

  const computeBudget = $derived(budgetValue * UNIT_MULTIPLIERS[budgetUnit]);
  const archMultiplier = $derived(ARCH_MULTIPLIERS[MODALITY_DEFAULTS[modality].arch]);
  const baseModel = $derived(MODELS.find((m) => m.name === baseModelName) ?? null);
  const isFt = $derived(trainingType === "Fine-Tuning");
  const isLora = $derived(ftMethod === "LoRA" || ftMethod === "QLoRA");

  const quickN = $derived(quickNEstimate(computeBudget, archMultiplier, hardware));
  const scheduleDefaults = $derived(
    deriveScheduleDefaults(computeBudget, modality, trainingType, isFt ? ftMethod : null, quickN)
  );

  let epochsOverride = $state<number | undefined>(undefined);
  let hpTrialsOverride = $state<number | undefined>(undefined);
  let hpPctOverride = $state<number | undefined>(undefined);
  let storageMonthsOverride = $state<number | undefined>(undefined);
  let storageClassOverride = $state<string | undefined>(undefined);

  const epochs = $derived(epochsOverride ?? scheduleDefaults.epochs);
  const hpTrials = $derived(hpTrialsOverride ?? scheduleDefaults.hp_trials);
  const hpPct = $derived(hpPctOverride ?? scheduleDefaults.hp_fraction_pct);
  const storageMonths = $derived(storageMonthsOverride ?? scheduleDefaults.storage_months);
  const storageClass = $derived(storageClassOverride ?? scheduleDefaults.storage_class);

  let exploreDirOverride = $state<string | undefined>(undefined);
  let logDOverride = $state<number | undefined>(undefined);
  let logNOverride = $state<number | undefined>(undefined);
  let rankOverride = $state<number | undefined>(undefined);
  let instancesOverride = $state<number | undefined>(undefined);

  // Exactly the setup_sig tuple, including the `or ""` coalescing.
  const setupSig = $derived(
    JSON.stringify([
      computeBudget,
      modality,
      trainingType,
      isFt ? ftMethod : "",
      baseModelName || "",
      loraRank
    ])
  );

  // Plain variable, not $state: writing it must not itself trigger reactivity.
  let appliedSig = "";

  $effect(() => {
    // Comparing (rather than a bare reference) makes the dependency unmistakable, and
    // the early return means a re-run from any other cause can't wipe an override.
    if (setupSig === appliedSig) return;
    appliedSig = setupSig;

    epochsOverride = undefined;
    hpTrialsOverride = undefined;
    hpPctOverride = undefined;
    storageMonthsOverride = undefined;
    storageClassOverride = undefined;
    logDOverride = undefined;
    logNOverride = undefined;
    rankOverride = undefined;
    instancesOverride = undefined;
  });

  const inputs = $derived({
    compute_budget: computeBudget,
    modality,
    training_type: trainingType,
    ft_method: isFt ? ftMethod : null,
    ft_base_model: isFt ? baseModel : null,
    lora_rank: loraRank,
    epochs,
    num_hp_trials: hpTrials,
    hp_fraction_pct: hpPct,
    storage_duration_months: storageMonths,
    storage_class: storageClass,
    hardware
  });

  const optimum = $derived(solveBudgetOptimum(inputs));
  const sel = $derived(
    resolveSelection(inputs, optimum, {
      explore_dir: exploreDirOverride,
      log_d: logDOverride,
      log_n: logNOverride,
      rank: rankOverride,
      num_instances: instancesOverride
    })
  );

  // Read the bounds as scalars first. `sel` is a fresh object on every slider tick, so
  // deriving straight from it defeats Svelte's referential check and rebuilds the whole
  // 240-point chart pipeline for an axis that did not move.
  const dMin = $derived(sel.d_slider_min);
  const dMax = $derived(sel.d_slider_max);
  const dRange = $derived(logspace(dMin, dMax, 240));
  const curve = $derived(
    budgetCurveN(
      dRange,
      sel.slider_effective_budget,
      hardware.peak_flops_per_gpu,
      hardware.mfu,
      hardware.gpus_per_instance,
      hardware.hourly_cost,
      archMultiplier,
      epochs
    )
  );

  const isFtWithBase = $derived(isFt && baseModel !== null);
  const optimalLabel = $derived(
    isLora && isFtWithBase
      ? `LoRA sweet spot (D / ${LORA_OPTIMAL_RATIO})`
      : `Chinchilla optimal (N = D / ${CHINCHILLA_OPTIMAL_RATIO})`
  );

  // Fine-tuning with a base model is judged against adapter size, not Chinchilla.
  const ratioAssessment = $derived(
    isFtWithBase && isLora
      ? assessLoraRatio(sel.current_ratio)
      : isFtWithBase
        ? null
        : assessChinchillaRatio(
            sel.current_ratio,
            Math.trunc(sel.selected_params * CHINCHILLA_OPTIMAL_RATIO),
            "Move the slider toward the star to return to Chinchilla-optimal."
          )
  );

  const budgetUsedTone = $derived(sel.budget_used_pct >= 0.999 ? "warning" : ("accent" as const));
</script>

<svelte:head>
  <title>Floply — training budget optimizer</title>
  <meta
    name="description"
    content="Given a budget, find the largest model and dataset you can afford to train, and where the scaling-law optimum sits." />
</svelte:head>

<h1 class="text-2xl font-semibold tracking-tight">What can this budget buy?</h1>
<p class="mt-2 max-w-[62ch] text-sm leading-relaxed text-[var(--color-ink-muted)]">
  For a fixed spend, a bigger model means less data. Set your budget and see the frontier — and
  where the scaling laws say the sweet spot is.
</p>

<section class="mt-8 border-t border-[var(--color-rule)] pt-6">
  <div class="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
    <Field label="Total budget (USD)" id="budget" hint="GPU compute, tuning runs and storage.">
      <ScaledNumber
        id="budget"
        bind:value={budgetValue}
        bind:unit={budgetUnit}
        units={["K", "M", "B", "T"]} />
    </Field>

    <Field label="Modality" id="modality">
      <select id="modality" bind:value={modality} class="control">
        {#each Object.keys(MODALITY_DEFAULTS) as m (m)}<option>{m}</option>{/each}
      </select>
    </Field>

    <Field label="Training type" id="training-type">
      <select id="training-type" bind:value={trainingType} class="control">
        <option>Pre-Training</option>
        <option>Fine-Tuning</option>
      </select>
    </Field>

    {#if isFt}
      <Field label="Fine-tuning method" id="ft-method">
        <select id="ft-method" bind:value={ftMethod} class="control">
          <option>Full Fine-Tuning (SFT)</option>
          <option>LoRA</option>
          <option>QLoRA</option>
        </select>
      </Field>

      <Field label="Base model" id="base-model">
        <select id="base-model" bind:value={baseModelName} class="control">
          <option value="">Custom / unspecified</option>
          {#each MODELS as m (m.slug)}<option value={m.name}>{m.name}</option>{/each}
        </select>
      </Field>

      {#if isLora}
        <Field label="LoRA rank (r)" id="rank">
          <input id="rank" type="range" bind:value={loraRank} min="1" max="256" />
          <span class="num text-sm">{loraRank}</span>
        </Field>
      {/if}
    {/if}
  </div>
</section>

<section class="mt-10 border-t border-[var(--color-rule)] pt-6">
  <h2 class="text-lg font-semibold tracking-tight">The frontier</h2>
  <div class="mt-4">
    <BudgetFrontier
      {dRange}
      budgetCurve={curve}
      optimalRatio={optimum.optimal_ratio}
      {optimalLabel}
      adapterParams={isFtWithBase && isLora ? optimum.n_adapter : null}
      optimum={optimum.d_opt > 0
        ? { d: optimum.d_opt, n: isFtWithBase && isLora ? optimum.n_adapter : optimum.n_opt }
        : null}
      selection={{ d: sel.selected_tokens, n: sel.selected_params }}
      yAxisTitle={sel.n_axis_title} />
  </div>

  <div
    class="mt-6 grid gap-5 border-t border-[var(--color-rule)] pt-5 lg:grid-cols-[minmax(0,20rem)_1fr]">
    <Field label="Explore by" id="explore">
      <select
        id="explore"
        value={sel.explore_dir}
        onchange={(e) => (exploreDirOverride = e.currentTarget.value)}
        class="control">
        {#each sel.explore_options as o (o)}<option>{o}</option>{/each}
      </select>
    </Field>

    {#if sel.explore_dir === EXPLORE_LORA_RANK}
      <Field
        label="LoRA rank"
        id="rank-slider"
        hint="A larger adapter needs more data to be worth it.">
        <input
          id="rank-slider"
          type="range"
          min="1"
          max="256"
          step="1"
          value={rankOverride ?? loraRank}
          oninput={(e) => (rankOverride = +e.currentTarget.value)} />
        <span class="num text-sm">rank {rankOverride ?? loraRank}</span>
      </Field>
    {:else if sel.explore_dir === EXPLORE_MODEL_TO_TOKENS}
      <Field label="Model size" id="n-slider" hint="Bigger model, less data for the same spend.">
        <input
          id="n-slider"
          type="range"
          min={sel.n_slider_min}
          max={sel.n_slider_max}
          step="0.05"
          value={logNOverride ?? sel.n_opt_log_default}
          oninput={(e) => (logNOverride = +e.currentTarget.value)} />
        <span class="num text-sm">{fmtTokens(sel.selected_params)} params</span>
      </Field>
    {:else}
      <Field label="Dataset size" id="d-slider" hint="More data, smaller model for the same spend.">
        <input
          id="d-slider"
          type="range"
          min={sel.d_slider_min}
          max={sel.d_slider_max}
          step="0.05"
          value={logDOverride ?? sel.d_opt_log_default}
          oninput={(e) => (logDOverride = +e.currentTarget.value)} />
        <span class="num text-sm">{fmtTokens(sel.selected_tokens)} tokens</span>
      </Field>
    {/if}
  </div>
</section>

<section class="mt-10 border-t border-[var(--color-rule)] pt-6">
  <h2 class="text-lg font-semibold tracking-tight">At this point on the curve</h2>

  <dl class="mt-5 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
    <Figure label={sel.n_axis_title} value={fmtTokens(sel.selected_params)} />
    <Figure label="Dataset" value="{fmtTokens(sel.selected_tokens)} tokens" />
    <Figure
      label="Tokens per parameter"
      value={sel.current_ratio.toFixed(1)}
      note={isLora && isFtWithBase ? "against adapter params" : "against model params"} />
    <Figure label="Wall-clock" value={formatWallClockTime(sel.sel_wall_clock_days)} />
  </dl>

  <dl
    class="mt-8 grid gap-6 border-t border-[var(--color-rule)] pt-5 sm:grid-cols-2 lg:grid-cols-4">
    <Figure label="Compute" value={money(sel.sel_compute_cost_total)} />
    <Figure label="Storage" value={money(sel.sel_dataset_storage + sel.sel_ckpt_storage)} />
    <Figure label="Total" value={money(sel.sel_total_cost)} tone={budgetUsedTone} />
    <Figure
      label="Budget used"
      value="{(sel.budget_used_pct * 100).toFixed(0)}%"
      note="of {money(computeBudget)}"
      tone={budgetUsedTone} />
  </dl>

  <Assessment assessment={ratioAssessment} class="mt-6" />
</section>

<section class="mt-10 border-t border-[var(--color-rule)] pt-6">
  <h2 class="text-lg font-semibold tracking-tight">Training schedule &amp; storage</h2>
  <p class="mt-1.5 max-w-[62ch] text-sm text-[var(--color-ink-muted)]">
    Derived from your setup. Change any of these and they hold until you edit the project setup
    above.
  </p>

  <div class="mt-5 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
    <Field label="Epochs" id="epochs">
      <input
        id="epochs"
        type="number"
        min="1"
        max="100"
        value={epochs}
        oninput={(e) => (epochsOverride = +e.currentTarget.value)}
        class="control num" />
    </Field>
    <Field label="HP tuning trials" id="hp-trials">
      <input
        id="hp-trials"
        type="number"
        min="0"
        max="1000"
        value={hpTrials}
        oninput={(e) => (hpTrialsOverride = +e.currentTarget.value)}
        class="control num" />
    </Field>
    <Field label="HP trial cost (%)" id="hp-pct" hint="Share of a full run's cost per trial.">
      <input
        id="hp-pct"
        type="number"
        min="1"
        max="100"
        value={hpPct}
        oninput={(e) => (hpPctOverride = +e.currentTarget.value)}
        class="control num" />
    </Field>
    <Field label="Storage duration (months)" id="months">
      <input
        id="months"
        type="number"
        min="1"
        max="36"
        value={storageMonths}
        oninput={(e) => (storageMonthsOverride = +e.currentTarget.value)}
        class="control num" />
    </Field>
    <Field label="S3 storage class" id="storage-class">
      <select
        id="storage-class"
        value={storageClass}
        onchange={(e) => (storageClassOverride = e.currentTarget.value)}
        class="control">
        {#each Object.keys(S3_STORAGE_PRICING) as c (c)}
          <option value={c}>{c.replace(/_/g, " ")}</option>
        {/each}
      </select>
    </Field>
    <Field
      label="Instances"
      id="instances"
      hint="Auto-recommended: {sel.recommended_instances} for a ≤60 day run.">
      <input
        id="instances"
        type="number"
        min="1"
        max="1024"
        value={sel.num_instances}
        oninput={(e) => (instancesOverride = +e.currentTarget.value)}
        class="control num" />
    </Field>
  </div>

  <p class="mt-6 text-xs text-[var(--color-ink-faint)]">
    {hardware.instance_spec.display_name} · bf16 · MFU {(hardware.mfu * 100).toFixed(0)}% · ${hardware.hourly_cost.toFixed(
      2
    )}/hr per instance ·
    {sel.num_instances * hardware.gpus_per_instance} GPUs total
  </p>
</section>
