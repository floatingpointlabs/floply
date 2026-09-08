<script lang="ts">
  import Field from "$lib/components/Field.svelte";
  import Figure from "$lib/components/Figure.svelte";
  import ScaledNumber from "$lib/components/ScaledNumber.svelte";
  import TierChart from "$lib/components/TierChart.svelte";
  import { calculateLoraTrainableParams } from "$lib/engine/calculator";
  import {
    ENCODEC_TOKENS_PER_SECOND,
    SOUNDSTREAM_TOKENS_PER_SECOND,
    WHISPER_TOKENS_PER_SECOND,
    tokensPerAudioSample,
    tokensPerImageSample,
    tokensPerTextSample,
    tokensPerVideoSample,
  } from "$lib/engine/dataset";
  import { UNIT_MULTIPLIERS, fmtSamples, fmtTokens } from "$lib/engine/formatting";
  import { DEFAULT_TARGET_MODULES } from "$lib/engine/budgetOptimizer";
  import { MODELS } from "$lib/engine/gpuSpecs";
  import { CHINCHILLA_OPTIMAL_RATIO } from "$lib/engine/constants";
  import { FULL_FT_TIERS, LORA_TIERS, PRE_TRAINING_TIERS } from "$lib/engine/tiers";

  let trainingType = $state<"Pre-Training" | "Fine-Tuning">("Pre-Training");

  let preParams = $state(7);
  let preUnit = $state("B");

  let baseModelName = $state<string>(MODELS[0].name);
  let customParams = $state(7);
  let customUnit = $state("B");
  let customDModel = $state(4096);
  let customLayers = $state(32);
  let ftMethod = $state<"Full Fine-Tuning" | "LoRA" | "QLoRA">("Full Fine-Tuning");
  let loraRank = $state(16);
  let targetModules = $state<string[]>([...DEFAULT_TARGET_MODULES]);

  const ALL_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"];

  const selectedModel = $derived(MODELS.find((m) => m.name === baseModelName) ?? null);
  const isCustom = $derived(baseModelName === "Custom");

  const ftParams = $derived(
    isCustom
      ? Math.trunc(customParams * UNIT_MULTIPLIERS[customUnit])
      : (selectedModel?.parameter_count ?? 0),
  );
  const ftArch = $derived(selectedModel?.architecture ?? {});
  const ftDModel = $derived(isCustom ? customDModel : (ftArch.d_model ?? 4096));
  const ftLayers = $derived(isCustom ? customLayers : (ftArch.num_layers ?? 32));

  const trainableParams = $derived(
    ftMethod === "Full Fine-Tuning"
      ? ftParams
      : (calculateLoraTrainableParams(
          ftMethod, ftParams, targetModules, ftDModel, ftLayers, loraRank, ftArch,
        ) ?? 0),
  );

  const preParamCount = $derived(Math.trunc(preParams * UNIT_MULTIPLIERS[preUnit]));

  const isLora = $derived(ftMethod === "LoRA" || ftMethod === "QLoRA");

  const tiers = $derived(
    trainingType === "Pre-Training"
      ? PRE_TRAINING_TIERS
      : isLora
        ? LORA_TIERS
        : FULL_FT_TIERS,
  );

  const effectiveCount = $derived(
    trainingType === "Pre-Training"
      ? preParamCount
      : isLora
        ? trainableParams
        : ftParams,
  );

  let modality = $state<"Text" | "Image" | "Audio" | "Video">("Text");

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

  const tokensPerSample = $derived.by(() => {
    switch (modality) {
      case "Text":
        return tokensPerTextSample(avgWords);
      case "Image":
        return tokensPerImageSample(resolution, patchSize);
      case "Audio":
        if (audioTokenizer === "Whisper") {
          return tokensPerAudioSample(clipDuration, WHISPER_TOKENS_PER_SECOND);
        }
        return tokensPerAudioSample(
          clipDuration,
          audioTokenizer === "EnCodec"
            ? ENCODEC_TOKENS_PER_SECOND
            : SOUNDSTREAM_TOKENS_PER_SECOND,
          codebooks,
        );
      case "Video":
        return tokensPerVideoSample(vidDuration, sampledFps, vidResolution, vidPatchSize);
    }
  });

  const sampleUnit = $derived(
    { Text: "text examples", Image: "images", Audio: "audio clips", Video: "video clips" }[
      modality
    ],
  );

  const sampleRows = $derived(
    tokensPerSample <= 0 ? [] :
    tiers.map((tier) => {
      const tokMin = Math.trunc(effectiveCount * tier.ratio_min_chart);
      const tokMax = Math.trunc(effectiveCount * tier.ratio_max);
      // Python `//` — floor, so a partial sample never counts.
      const sampMin = Math.max(1, Math.floor(tokMin / tokensPerSample));
      const sampMax = Math.max(1, Math.floor(tokMax / tokensPerSample));
      return {
        tier,
        label:
          tier.ratio_min === 0
            ? `< ${fmtSamples(sampMax)}`
            : `${fmtSamples(sampMin)}–${fmtSamples(sampMax)}`,
        note: `${fmtTokens(tokMin)}–${fmtTokens(tokMax)} tokens`,
      };
    }),
  );
</script>

<svelte:head>
  <title>Minimum data — Floply</title>
  <meta
    name="description"
    content="How much training data a model needs, from Chinchilla scaling laws and practical fine-tuning thresholds."
  />
</svelte:head>

<h1 class="text-2xl font-semibold tracking-tight">How much data do I need?</h1>
<p class="mt-2 max-w-[62ch] text-sm leading-relaxed text-[var(--color-ink-muted)]">
  Scaling laws set a floor and a ceiling on useful dataset size. Pick a model and see
  where your data lands.
</p>

<section class="mt-8 border-t border-[var(--color-rule)] pt-6">
  <div class="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
    <Field label="Training type" id="training-type">
      <select id="training-type" bind:value={trainingType} class="control">
        <option>Pre-Training</option>
        <option>Fine-Tuning</option>
      </select>
    </Field>

    {#if trainingType === "Pre-Training"}
      <Field label="Parameter count" id="pre-params">
        <ScaledNumber id="pre-params" bind:value={preParams} bind:unit={preUnit} />
      </Field>
    {:else}
      <Field label="Base model" id="base-model">
        <select id="base-model" bind:value={baseModelName} class="control">
          {#each MODELS as model (model.slug)}
            <option value={model.name}>{model.name}</option>
          {/each}
          <option value="Custom">Custom</option>
        </select>
      </Field>

      <Field label="Fine-tuning method" id="ft-method">
        <select id="ft-method" bind:value={ftMethod} class="control">
          <option>Full Fine-Tuning</option>
          <option>LoRA</option>
          <option>QLoRA</option>
        </select>
      </Field>

      {#if isCustom}
        <Field label="Parameter count" id="custom-params">
          <ScaledNumber id="custom-params" bind:value={customParams} bind:unit={customUnit} />
        </Field>
        <Field label="Hidden dimension" id="custom-d-model">
          <input id="custom-d-model" type="number" bind:value={customDModel} min="64" max="65536" step="64" class="control num" />
        </Field>
        <Field label="Layers" id="custom-layers">
          <input id="custom-layers" type="number" bind:value={customLayers} min="1" max="512" class="control num" />
        </Field>
      {/if}

      {#if isLora}
        <Field label="LoRA rank (r)" id="lora-rank" hint="Higher rank is more expressive, and more parameters to train.">
          <input id="lora-rank" type="range" bind:value={loraRank} min="1" max="128" step="1" />
          <span class="num text-sm">{loraRank}</span>
        </Field>

        <Field label="Target modules" id="target-modules">
          <div id="target-modules" class="flex flex-wrap gap-1.5">
            {#each ALL_MODULES as mod (mod)}
              {@const on = targetModules.includes(mod)}
              <button
                type="button"
                aria-pressed={on}
                onclick={() =>
                  (targetModules = on
                    ? targetModules.filter((m) => m !== mod)
                    : [...targetModules, mod])}
                class="pill transition-colors
                       {on
                  ? 'border-[var(--color-violet)] text-[var(--color-ink)]'
                  : 'hover:text-[var(--color-ink-muted)]'}"
              >
                {mod}
              </button>
            {/each}
          </div>
        </Field>
      {/if}
    {/if}
  </div>

  {#if trainingType === "Fine-Tuning" && isLora && trainableParams > 0}
    <p class="mt-4 text-xs text-[var(--color-ink-faint)]">
      <span class="num">{fmtTokens(trainableParams)}</span> trainable adapter parameters
      &mdash; <span class="num">{((trainableParams / ftParams) * 100).toFixed(2)}%</span>
      of the {fmtTokens(ftParams)}-parameter base.
    </p>
  {/if}
</section>

<section class="mt-10 border-t border-[var(--color-rule)] pt-6">
  {#if effectiveCount > 0}
    <h2 class="text-lg font-semibold tracking-tight">
      {trainingType === "Pre-Training"
        ? "Pre-training data requirements"
        : isLora
          ? `${ftMethod} data requirements`
          : "Full fine-tuning data requirements"}
    </h2>
    <p class="mt-1.5 max-w-[68ch] text-sm text-[var(--color-ink-muted)]">
      {#if trainingType === "Pre-Training"}
        Chinchilla scaling laws (Hoffmann et al. 2022) for a
        <span class="num">{fmtTokens(effectiveCount)}</span>-parameter model.
      {:else if isLora}
        Thresholds measured against adapter parameters, not the full base model. The frozen
        base provides strong priors, so far less data is needed than Chinchilla's
        {CHINCHILLA_OPTIMAL_RATIO} tok/param pre-training recommendation.
      {:else}
        Practical thresholds for a
        <span class="num">{fmtTokens(effectiveCount)}</span>-parameter base model.
      {/if}
    </p>

    <div class="mt-5">
      <TierChart {tiers} {effectiveCount} />
    </div>
  {:else}
    <p class="text-sm text-[var(--color-ink-muted)]">
      Choose a larger model or add target modules to see data requirements.
    </p>
  {/if}
</section>

{#if effectiveCount > 0}
  <section class="mt-10 border-t border-[var(--color-rule)] pt-6">
    <h2 class="text-lg font-semibold tracking-tight">In real examples</h2>
    <p class="mt-1.5 max-w-[62ch] text-sm text-[var(--color-ink-muted)]">
      Token counts are hard to picture. Convert them into the thing you actually collect.
    </p>

    <div class="mt-5 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
      <Field label="Modality" id="modality">
        <select id="modality" bind:value={modality} class="control">
          <option>Text</option>
          <option>Image</option>
          <option>Audio</option>
          <option>Video</option>
        </select>
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
        <Field label="Patch size (px)" id="patch-size" hint="Smaller patches mean more tokens per image.">
          <select id="patch-size" bind:value={patchSize} class="control num">
            {#each [14, 16, 32] as p (p)}<option value={p}>{p}</option>{/each}
          </select>
        </Field>
      {:else if modality === "Audio"}
        <Field label="Clip duration (seconds)" id="clip-duration">
          <input id="clip-duration" type="number" bind:value={clipDuration} min="0.1" max="3600" step="5" class="control num" />
        </Field>
        <Field label="Tokenizer" id="audio-tokenizer">
          <select id="audio-tokenizer" bind:value={audioTokenizer} class="control">
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
      {:else}
        <Field label="Clip duration (seconds)" id="vid-duration">
          <input id="vid-duration" type="number" bind:value={vidDuration} min="0.1" max="3600" step="1" class="control num" />
        </Field>
        <Field label="Sampled FPS" id="sampled-fps" hint="Frames sampled per second, not source frame rate.">
          <input id="sampled-fps" type="number" bind:value={sampledFps} min="1" max="60" class="control num" />
        </Field>
        <Field label="Frame resolution (px)" id="vid-resolution">
          <select id="vid-resolution" bind:value={vidResolution} class="control num">
            {#each [224, 336, 512] as r (r)}<option value={r}>{r}</option>{/each}
          </select>
        </Field>
        <Field label="Patch size (px)" id="vid-patch-size">
          <select id="vid-patch-size" bind:value={vidPatchSize} class="control num">
            {#each [14, 16, 32] as p (p)}<option value={p}>{p}</option>{/each}
          </select>
        </Field>
      {/if}
    </div>

    {#if tokensPerSample > 0}
      <p class="mt-5 text-sm text-[var(--color-ink-muted)]">
        <span class="num text-[var(--color-ink)]">{fmtTokens(tokensPerSample)}</span>
        tokens per {sampleUnit.replace(/s$/, "")}.
      </p>

      <dl class="mt-5 grid gap-6 border-t border-[var(--color-rule)] pt-5 sm:grid-cols-2 lg:grid-cols-4">
        {#each sampleRows as row (row.tier.tier)}
          <Figure label={row.tier.tier} value={row.label} note={row.note} />
        {/each}
      </dl>
      <p class="mt-4 text-xs text-[var(--color-ink-faint)]">
        {sampleUnit.replace(/^./, (c) => c.toUpperCase())} needed to reach each tier.
      </p>
    {/if}
  </section>
{/if}
