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
    tokensPerVideoSample
  } from "$lib/engine/dataset";
  import { UNIT_MULTIPLIERS, fmtSamples, fmtTokens } from "$lib/engine/formatting";
  import { DEFAULT_TARGET_MODULES } from "$lib/engine/budgetOptimizer";
  import { DEFAULT_MODEL, MODELS, MODEL_GROUPS } from "$lib/engine/gpuSpecs";
  import { FULL_FT_TIERS, LORA_TIERS, PRE_TRAINING_TIERS } from "$lib/engine/tiers";

  let trainingType = $state<"Pre-Training" | "Fine-Tuning">("Pre-Training");

  let preParams = $state(7);
  let preUnit = $state("B");

  let baseModelName = $state<string>(DEFAULT_MODEL.name);
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
      : (selectedModel?.parameter_count ?? 0)
  );
  const ftArch = $derived(selectedModel?.architecture ?? {});
  const ftDModel = $derived(isCustom ? customDModel : (ftArch.d_model ?? 4096));
  const ftLayers = $derived(isCustom ? customLayers : (ftArch.num_layers ?? 32));

  const trainableParams = $derived(
    ftMethod === "Full Fine-Tuning"
      ? ftParams
      : (calculateLoraTrainableParams(
          ftMethod,
          ftParams,
          targetModules,
          ftDModel,
          ftLayers,
          loraRank,
          ftArch
        ) ?? 0)
  );

  const preParamCount = $derived(Math.trunc(preParams * UNIT_MULTIPLIERS[preUnit]));

  const isLora = $derived(ftMethod === "LoRA" || ftMethod === "QLoRA");

  const tiers = $derived(
    trainingType === "Pre-Training" ? PRE_TRAINING_TIERS : isLora ? LORA_TIERS : FULL_FT_TIERS
  );

  const effectiveCount = $derived(
    trainingType === "Pre-Training" ? preParamCount : isLora ? trainableParams : ftParams
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
          audioTokenizer === "EnCodec" ? ENCODEC_TOKENS_PER_SECOND : SOUNDSTREAM_TOKENS_PER_SECOND,
          codebooks
        );
      case "Video":
        return tokensPerVideoSample(vidDuration, sampledFps, vidResolution, vidPatchSize);
    }
  });

  const sampleUnit = $derived(
    { Text: "text examples", Image: "images", Audio: "audio clips", Video: "video clips" }[modality]
  );

  const sampleRows = $derived(
    tokensPerSample <= 0
      ? []
      : tiers.map((tier) => {
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
            note: `${fmtTokens(tokMin)}–${fmtTokens(tokMax)} tokens`
          };
        })
  );
</script>

<svelte:head>
  <title>Data needed — Floply</title>
  <meta name="description" content="Find out how much training data your model needs." />
</svelte:head>

<h1>How much data do I need?</h1>
<p class="mt-2 max-w-[62ch] text-[var(--color-ink-muted)]">
  Pick a model to see how much training data it needs, from too little to more than enough.
</p>

<section class="mt-8 border-t border-[var(--color-soft)] pt-6">
  <div class="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
    <Field label="Training type" id="training-type">
      <select id="training-type" bind:value={trainingType} class="control">
        <option value="Pre-Training">Pre-training</option>
        <option value="Fine-Tuning">Fine-tuning</option>
      </select>
    </Field>

    {#if trainingType === "Pre-Training"}
      <Field label="Model size (parameters)" id="pre-params">
        <ScaledNumber id="pre-params" bind:value={preParams} bind:unit={preUnit} />
      </Field>
    {:else}
      <Field label="Base model" id="base-model">
        <select id="base-model" bind:value={baseModelName} class="control">
          {#each MODEL_GROUPS as g (g.developer)}
            <optgroup label={g.developer}>
              {#each g.models as m (m.slug)}<option value={m.name}>{m.name}</option>{/each}
            </optgroup>
          {/each}
          <option value="Custom">Custom</option>
        </select>
      </Field>

      <Field label="Fine-tuning method" id="ft-method">
        <select id="ft-method" bind:value={ftMethod} class="control">
          <option value="Full Fine-Tuning">Full fine-tuning</option>
          <option>LoRA</option>
          <option>QLoRA</option>
        </select>
      </Field>

      {#if isCustom}
        <Field label="Model size (parameters)" id="custom-params">
          <ScaledNumber id="custom-params" bind:value={customParams} bind:unit={customUnit} />
        </Field>
        <Field label="Hidden size" id="custom-d-model">
          <input
            id="custom-d-model"
            type="number"
            bind:value={customDModel}
            min="64"
            max="65536"
            step="64"
            class="control num" />
        </Field>
        <Field label="Layers" id="custom-layers">
          <input
            id="custom-layers"
            type="number"
            bind:value={customLayers}
            min="1"
            max="512"
            class="control num" />
        </Field>
      {/if}

      {#if isLora}
        <Field
          label="LoRA rank (r)"
          id="lora-rank"
          hint="Higher rank learns more, but trains more parameters.">
          <input id="lora-rank" type="range" bind:value={loraRank} min="1" max="128" step="1" />
          <span class="num text-sm">{loraRank}</span>
        </Field>

        <Field label="Target modules" id="target-modules" hint="The layers that get an adapter.">
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
                class="pill">
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
      <span class="num">{fmtTokens(trainableParams)}</span>
      parameters to train,
      <span class="num">{((trainableParams / ftParams) * 100).toFixed(2)}%</span>
      of the full {fmtTokens(ftParams)} model.
    </p>
  {/if}
</section>

<section class="mt-10 border-t border-[var(--color-soft)] pt-6">
  {#if effectiveCount > 0}
    <h2>
      {trainingType === "Pre-Training"
        ? "Data needed for pre-training"
        : isLora
          ? `Data needed for ${ftMethod}`
          : "Data needed for full fine-tuning"}
    </h2>
    <!-- The count runs straight into "-parameter", so a newline after the span would
         render as "7.0B -parameter model". -->
    <!-- prettier-ignore -->
    <p class="mt-1.5 max-w-[68ch] text-sm text-[var(--color-ink-muted)]">
      {#if trainingType === "Pre-Training"}
        For a <span class="num">{fmtTokens(effectiveCount)}</span>-parameter model. Aim for
        the Optimal range.
      {:else if isLora}
        Based on the <span class="num">{fmtTokens(effectiveCount)}</span> adapter parameters,
        not the full model, so far less data is needed. Aim for the Optimal range.
      {:else}
        For a <span class="num">{fmtTokens(effectiveCount)}</span>-parameter base model. Aim for
        the Optimal range.
      {/if}
    </p>

    <div class="mt-5">
      <TierChart {tiers} {effectiveCount} />
    </div>
  {:else}
    <p class="text-sm text-[var(--color-ink-muted)]">
      Enter a model size, or select at least one target module.
    </p>
  {/if}
</section>

{#if effectiveCount > 0}
  <section class="mt-10 border-t border-[var(--color-soft)] pt-6">
    <h2>In examples</h2>
    <p class="mt-1.5 max-w-[62ch] text-sm text-[var(--color-ink-muted)]">
      The same ranges as a number of texts, images or clips.
    </p>

    <div class="mt-5 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
      <Field label="Data type" id="modality">
        <select id="modality" bind:value={modality} class="control">
          <option>Text</option>
          <option>Image</option>
          <option>Audio</option>
          <option>Video</option>
        </select>
      </Field>

      {#if modality === "Text"}
        <Field label="Average length (words)" id="avg-words" hint="About 1.3 tokens per word.">
          <input
            id="avg-words"
            type="number"
            bind:value={avgWords}
            min="1"
            max="100000"
            step="50"
            class="control num" />
        </Field>
      {:else if modality === "Image"}
        <Field label="Resolution (px)" id="resolution">
          <select id="resolution" bind:value={resolution} class="control num">
            {#each [224, 336, 512, 1024] as r (r)}<option value={r}>{r}</option>{/each}
          </select>
        </Field>
        <Field
          label="Patch size (px)"
          id="patch-size"
          hint="Smaller patches mean more tokens per image.">
          <select id="patch-size" bind:value={patchSize} class="control num">
            {#each [14, 16, 32] as p (p)}<option value={p}>{p}</option>{/each}
          </select>
        </Field>
      {:else if modality === "Audio"}
        <Field label="Clip duration (seconds)" id="clip-duration">
          <input
            id="clip-duration"
            type="number"
            bind:value={clipDuration}
            min="0.1"
            max="3600"
            step="5"
            class="control num" />
        </Field>
        <Field label="Tokenizer" id="audio-tokenizer">
          <select id="audio-tokenizer" bind:value={audioTokenizer} class="control">
            <option value="Whisper">Whisper (50 tokens/sec)</option>
            <option value="EnCodec">EnCodec 24 kHz (75 tokens/sec per codebook)</option>
            <option value="SoundStream">SoundStream (50 tokens/sec per codebook)</option>
          </select>
        </Field>
        {#if audioTokenizer !== "Whisper"}
          <Field label="Codebooks" id="codebooks">
            <input
              id="codebooks"
              type="number"
              bind:value={codebooks}
              min="1"
              max="16"
              class="control num" />
          </Field>
        {/if}
      {:else}
        <Field label="Clip duration (seconds)" id="vid-duration">
          <input
            id="vid-duration"
            type="number"
            bind:value={vidDuration}
            min="0.1"
            max="3600"
            step="1"
            class="control num" />
        </Field>
        <Field
          label="Frames per second"
          id="sampled-fps"
          hint="Frames you sample each second, not the video's frame rate.">
          <input
            id="sampled-fps"
            type="number"
            bind:value={sampledFps}
            min="1"
            max="60"
            class="control num" />
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

      <dl
        class="mt-5 grid gap-6 border-t border-[var(--color-soft)] pt-5 sm:grid-cols-2 lg:grid-cols-4">
        {#each sampleRows as row (row.tier.tier)}
          <Figure label={row.tier.tier} value={row.label} note={row.note} />
        {/each}
      </dl>
      <p class="mt-4 text-xs text-[var(--color-ink-faint)]">
        {sampleUnit.replace(/^./, (c) => c.toUpperCase())} needed for each range.
      </p>
    {/if}
  </section>
{/if}
