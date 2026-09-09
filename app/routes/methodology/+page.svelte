<script lang="ts">
  import { getGpuInstance } from "$lib/engine/gpuSpecs";
  import type { PageProps } from "./$types";

  let { data }: PageProps = $props();

  const sections = [
    { id: "overview", title: "Overview" },
    { id: "tokenisation", title: "Tokenisation" },
    { id: "scaling-laws", title: "Scaling-law thresholds" },
    { id: "flops", title: "Training FLOPs" },
    { id: "lora", title: "LoRA parameter counts" },
    { id: "throughput", title: "GPU time and cost" },
    { id: "memory", title: "GPU memory" },
    { id: "storage", title: "Storage costs" },
    { id: "experiments", title: "Experiment rollup" },
    { id: "instances", title: "Instance reference" },
    { id: "limitations", title: "Limitations" }
  ];

  const instances = $derived(
    data.catalog.order.map((t) => ({ type: t, ...getGpuInstance(data.catalog, t) }))
  );
</script>

<svelte:head>
  <title>Methodology — Floply</title>
  <meta
    name="description"
    content="How Floply estimates ML training cost: tokenisation per modality, the C = 6ND FLOPs model, scaling-law thresholds, GPU memory, and S3 storage." />
</svelte:head>

<div class="lg:grid lg:grid-cols-[1fr_14rem] lg:gap-12">
  <article class="methodology max-w-[68ch]">
    <h1 class="text-2xl font-semibold tracking-tight">Methodology</h1>
    <p class="mt-2 text-sm text-[var(--color-ink-muted)]">
      Every formula behind the estimates, and what they assume. Pricing current as of February 2026.
    </p>

    <section id="overview">
      <h2>Overview</h2>
      <p>
        Floply builds a bottom-up estimate in five stages: dataset dimensions, model configuration,
        cluster, experiment plan, and cost rollup. Each stage gates the next, so compute settings
        only appear once a model and dataset exist.
      </p>
      <p>
        Nothing here is a benchmark. These are analytical estimates from published scaling
        relationships and vendor datasheets — useful for deciding whether a project is affordable,
        not for predicting a specific run to the dollar.
      </p>
    </section>

    <section id="tokenisation">
      <h2>Tokenisation</h2>
      <p>Token counts and raw storage are derived per modality.</p>

      <h3>Text</h3>
      <pre><code>tokens_per_sample = words × 1.3   # BPE subword average
bytes_per_sample  = tokens × 4    # raw UTF-8, ~4 bytes/token</code></pre>
      <p>1.3 is the empirical BPE inflation ratio for English.</p>

      <h3>Image (ViT patch tokenisation)</h3>
      <pre><code>tokens_per_sample = (resolution ÷ patch_size)²
bytes_per_sample  = resolution² × 3 ÷ 10   # JPEG ~10:1</code></pre>
      <p>
        A 224px image at 16px patches gives 196 tokens. Note the division floors: 336 ÷ 32 is 10
        patches per side, not 10.5, so the count is 100 rather than 110.
      </p>

      <h3>Audio</h3>
      <table>
        <thead>
          <tr>
            <th>Tokeniser</th>
            <th>Rate</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Whisper</td>
            <td>50 tok/sec</td>
            <td>mel-spectrogram frames</td>
          </tr>
          <tr>
            <td>EnCodec 24kHz</td>
            <td>75 tok/sec × codebooks</td>
            <td>residual vector quantisation</td>
          </tr>
          <tr>
            <td>SoundStream</td>
            <td>50 tok/sec × codebooks</td>
            <td>residual vector quantisation</td>
          </tr>
        </tbody>
      </table>
      <p>
        Raw storage is always <code>clip_duration × 32,000 bytes</code>
        (16 kHz mono 16-bit PCM).
      </p>

      <h3>Video</h3>
      <pre><code>num_frames        = clip_duration × sampled_fps
tokens_per_frame  = (resolution ÷ patch_size)²
tokens_per_sample = tokens_per_frame × num_frames
bytes_per_sample  = (resolution² × 3 ÷ 10) × num_frames</code></pre>
      <p>
        Storage is modelled as JPEG-encoded extracted frames, matching how datasets like Kinetics
        are stored on disk.
      </p>
    </section>

    <section id="scaling-laws">
      <h2>Scaling-law thresholds</h2>
      <p>The health check adapts its thresholds to the training method.</p>

      <h3>Pre-training</h3>
      <p>
        Hoffmann et al. 2022 (Chinchilla): <strong>D* ≈ 20 × N</strong>
        .
      </p>
      <table>
        <thead>
          <tr>
            <th>tok / param</th>
            <th>Reading</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>&lt; 1</td>
            <td>Critically undersized — likely to diverge</td>
          </tr>
          <tr>
            <td>1–9</td>
            <td>Undertrained, below compute-optimal</td>
          </tr>
          <tr>
            <td>10–30</td>
            <td>
              <strong>Chinchilla-optimal</strong>
              — best loss per FLOP
            </td>
          </tr>
          <tr>
            <td>31–200</td>
            <td>Inference-optimal — the LLaMA / Mistral over-train strategy</td>
          </tr>
          <tr>
            <td>&gt; 200</td>
            <td>Heavily over-trained, diminishing returns</td>
          </tr>
        </tbody>
      </table>

      <h3>Full fine-tuning</h3>
      <p>
        Starts from a converged base, so far less data is needed. Chinchilla does not apply; the
        risk is catastrophic forgetting rather than underfitting.
      </p>
      <table>
        <thead>
          <tr>
            <th>tok / base_param</th>
            <th>Reading</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>&lt; 0.1</td>
            <td>Too small to shift behaviour meaningfully</td>
          </tr>
          <tr>
            <td>0.1–1</td>
            <td>Likely undertrained for stable adaptation</td>
          </tr>
          <tr>
            <td>1–5</td>
            <td><strong>Standard SFT range</strong></td>
          </tr>
          <tr>
            <td>5–20</td>
            <td>Generous — watch for forgetting</td>
          </tr>
          <tr>
            <td>&gt; 20</td>
            <td>Consider LoRA instead</td>
          </tr>
        </tbody>
      </table>

      <h3>LoRA and QLoRA</h3>
      <p>
        The frozen base provides strong priors, so the relevant count is the
        <strong>adapter size</strong>
        , not the full model.
      </p>
      <table>
        <thead>
          <tr>
            <th>tok / adapter_param</th>
            <th>Reading</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>&lt; 10</td>
            <td>Adapter may underfit</td>
          </tr>
          <tr>
            <td>10–100</td>
            <td><strong>Standard for task adaptation</strong></td>
          </tr>
          <tr>
            <td>100–1000</td>
            <td>Large dataset — consider a higher rank</td>
          </tr>
          <tr>
            <td>&gt; 1000</td>
            <td>Adapter is the bottleneck; consider full fine-tuning</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section id="flops">
      <h2>Training FLOPs</h2>
      <p>Kaplan et al. 2020:</p>
      <pre><code>C = 6 × N × D</code></pre>
      <ul>
        <li>
          <strong>C</strong>
          — total floating-point operations
        </li>
        <li>
          <strong>N</strong>
          — trainable parameters
        </li>
        <li>
          <strong>D</strong>
          — training tokens (dataset tokens × epochs)
        </li>
        <li>
          <strong>6</strong>
          — forward pass (2N) plus backward pass (4N)
        </li>
      </ul>
      <p>
        Other architectures use different multipliers: CNN 4, RNN 8, ViT 6, diffusion 6.5. Gradient
        checkpointing recomputes activations, adding 2 to the multiplier.
      </p>

      <h3>Mixture-of-Experts</h3>
      <p>
        MoE models route each token through a subset of experts, so FLOPs scale with
        <strong>active</strong>
        parameters while checkpoints and VRAM scale with the total. DeepSeek V3 is costed at its 37B active
        parameters, not its 671B total.
      </p>
    </section>

    <section id="lora">
      <h2>LoRA parameter counts</h2>
      <p>
        For models in the library, trainable parameters are computed per-module from the real
        architecture, handling Grouped Query Attention correctly.
      </p>
      <pre><code>MODULE_DIMS = &lbrace;
    "q_proj":    (d_model, d_model),
    "k_proj":    (d_model, kv_dim),   # kv_dim = num_kv_heads × head_dim
    "v_proj":    (d_model, kv_dim),
    "o_proj":    (d_model, d_model),
    "up_proj":   (d_model, ffn_intermediate),
    "down_proj": (ffn_intermediate, d_model),
&rbrace;

trainable_params = Σ[rank × (in_dim + out_dim)] × num_layers</code></pre>
      <p>Custom models fall back to a uniform approximation:</p>
      <pre><code>trainable_params ≈ num_modules × 2 × rank × d_model × num_layers</code></pre>

      <h3>LoRA versus QLoRA</h3>
      <p>
        Both train an identical number of parameters. They differ only in how the frozen base is
        stored.
      </p>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Base dtype</th>
            <th>Memory</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Full fine-tuning</td>
            <td>fp32/bf16</td>
            <td>params × 16</td>
          </tr>
          <tr>
            <td>LoRA</td>
            <td>bf16</td>
            <td>base × 2 + adapters × 16</td>
          </tr>
          <tr>
            <td>QLoRA</td>
            <td>int4</td>
            <td>base × 0.5 + adapters × 16</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section id="throughput">
      <h2>GPU time and cost</h2>
      <pre><code>cluster_flops_per_s = peak_flops × MFU × total_GPUs
wall_clock_hours    = total_flops ÷ cluster_flops_per_s ÷ 3600
gpu_hours           = wall_clock_hours × total_GPUs
compute_cost        = wall_clock_hours × hourly_rate × num_instances</code></pre>

      <h3>Precision multipliers, relative to FP16</h3>
      <table>
        <thead>
          <tr>
            <th>Precision</th>
            <th>A100</th>
            <th>H100</th>
            <th>V100</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>fp4</td>
            <td>1×</td>
            <td>1×</td>
            <td>1×</td>
          </tr>
          <tr>
            <td>int8</td>
            <td>2×</td>
            <td>2×</td>
            <td>0.9×</td>
          </tr>
          <tr>
            <td>fp8</td>
            <td>1×</td>
            <td><strong>2×</strong></td>
            <td>1×</td>
          </tr>
          <tr>
            <td>bf16 / fp16</td>
            <td>1× (baseline)</td>
            <td>1× (baseline)</td>
            <td>1× (baseline)</td>
          </tr>
          <tr>
            <td>tf32</td>
            <td>0.5×</td>
            <td>0.5×</td>
            <td>0.5×</td>
          </tr>
          <tr>
            <td>fp32</td>
            <td colspan="3">uses the fp32 datasheet figure</td>
          </tr>
        </tbody>
      </table>
      <p>
        FP4 is Blackwell-only (B100/B200/GB200), so none of the instances above accelerate it, and
        FP8 arrived with Hopper. A precision the GPU cannot accelerate is costed at its FP16 rate —
        inventing a speedup for absent hardware would understate cost, which is the wrong direction
        to be wrong in.
      </p>

      <h3>Model FLOPs Utilisation</h3>
      <p>
        MFU is the share of theoretical peak actually achieved, and it is adjustable because it
        depends on your setup: architecture and batch size, multi-node communication, data-loading
        bubbles, and framework overhead. The 30% default is deliberately conservative; well-tuned
        production LLM training reaches 40–55%.
      </p>
    </section>

    <section id="memory">
      <h2>GPU memory</h2>
      <h3>Weights</h3>
      <pre><code>QLoRA:  base × 0.5 + adapters × 16   # int4 base
LoRA:   base × 2   + adapters × 16   # bf16 base
Full:   params × 16                  # weights + grads + Adam states</code></pre>

      <h3>Alignment overhead</h3>
      <p>
        DPO holds a frozen reference model alongside the policy (
        <strong>2×</strong>
        ). PPO adds a reward model and a critic (
        <strong>4×</strong>
        ). The multiplier applies to weight memory before dividing across GPUs.
      </p>

      <h3>Activations</h3>
      <pre><code>without checkpointing: batch × seq_len × d_model × num_layers × 4 × 2
with checkpointing:    batch × seq_len × d_model × 4 × 2</code></pre>
      <p>
        The 4 covers QKV projections, attention scores and MLP intermediates; the 2 is bf16. With
        checkpointing only one layer of activations is live at a time.
      </p>
      <pre><code>memory_per_GPU_GB = (weight_bytes + activation_bytes) ÷ total_GPUs ÷ 1e9</code></pre>
    </section>

    <section id="storage">
      <h2>Storage costs</h2>
      <pre><code>dataset_TB   = bytes_per_sample × num_samples ÷ 1e12
dataset_cost = price_per_TB_month × dataset_TB × months</code></pre>
      <p>
        Each checkpoint stores trainable parameters in mixed precision, 14 bytes per parameter —
        bf16 weights plus fp32 optimiser state.
      </p>
      <pre><code>checkpoint_TB = params × 14 × (
      checkpoints_per_run × full_runs
    + hp_trials        # one final checkpoint each
    + ablations        # one final checkpoint each
  ) ÷ 1e12</code></pre>
      <p>
        Sweeps are where this bites: dozens of trials each leaving a final checkpoint adds up faster
        than the runs themselves.
      </p>
    </section>

    <section id="experiments">
      <h2>Experiment rollup</h2>
      <pre><code>single_run   = wall_clock_hours × hourly_rate × num_instances

total_compute = single_run × full_runs
              + single_run × hp_fraction × hp_trials
              + single_run × ablation_fraction × ablations

total_project = total_compute + dataset_storage + checkpoint_storage</code></pre>
    </section>

    <section id="instances">
      <h2>Instance reference</h2>
      <p>On-demand averages across US regions.</p>
      <table>
        <thead>
          <tr>
            <th>Instance</th>
            <th>GPU</th>
            <th>Count</th>
            <th>VRAM</th>
            <th>FP16</th>
            <th>Cost/hr</th>
          </tr>
        </thead>
        <tbody>
          {#each instances as i (i.type)}
            <tr>
              <td class="num">{i.type}</td>
              <td>{i.gpu}</td>
              <td class="num">{i.gpu_count}</td>
              <td class="num">{i.memory_per_gpu} GB</td>
              <td class="num">{(i.peak_flops_fp16 / 1e12).toFixed(0)} TFLOPS</td>
              <td class="num">${i.hourly_cost.toFixed(2)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      <p>
        All use NVLink or NVSwitch within a node. Multi-node scaling is modelled as linear, with no
        communication penalty.
      </p>
    </section>

    <section id="limitations">
      <h2>Limitations</h2>
      <p>
        Where the model is knowingly simplified. Read this before using an estimate to justify a
        budget.
      </p>
      <table>
        <thead>
          <tr>
            <th>Area</th>
            <th>Assumption</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Scaling</td>
            <td>Linear across GPUs and nodes; no NCCL or EFA communication overhead.</td>
          </tr>
          <tr>
            <td>Pricing</td>
            <td>On-demand only. Spot can be 60–90% cheaper, reserved 30–40%.</td>
          </tr>
          <tr>
            <td>Cloud</td>
            <td>AWS only. GCP TPUs, Azure NDv5 and CoreWeave are not modelled.</td>
          </tr>
          <tr>
            <td>Optimiser</td>
            <td>Adam, at 16 bytes/param. Lion or Adafactor would use less.</td>
          </tr>
          <tr>
            <td>Startup</td>
            <td>Steady state only — no spin-up, prefetch or compilation time.</td>
          </tr>
          <tr>
            <td>Data pipeline</td>
            <td>I/O bottlenecks unmodelled; a starved cluster has lower real MFU.</td>
          </tr>
          <tr>
            <td>Mixed precision</td>
            <td>Multipliers are theoretical peaks; real kernels rarely saturate them.</td>
          </tr>
          <tr>
            <td>LoRA accuracy</td>
            <td>GQA-aware for library models; custom models use the uniform approximation.</td>
          </tr>
          <tr>
            <td>Storage</td>
            <td>S3 only. No EFS, FSx for Lustre, or local NVMe during training.</td>
          </tr>
        </tbody>
      </table>
    </section>
  </article>

  <nav class="mt-12 hidden lg:mt-0 lg:block" aria-label="On this page">
    <ul class="sticky top-8 space-y-2 border-l border-[var(--color-rule)] pl-4 text-sm">
      {#each sections as s (s.id)}
        <li>
          <a
            href="#{s.id}"
            class="text-[var(--color-ink-faint)] transition-colors hover:text-[var(--color-ink)]">
            {s.title}
          </a>
        </li>
      {/each}
    </ul>
  </nav>
</div>

<style>
  .methodology :global(section) {
    margin-top: 2.75rem;
    scroll-margin-top: 2rem;
  }
  .methodology :global(h2) {
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--color-rule);
  }
  .methodology :global(h3) {
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 1.75rem;
    color: var(--color-ink);
  }
  .methodology :global(p) {
    margin-top: 0.85rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--color-ink-muted);
  }
  .methodology :global(ul) {
    margin-top: 0.85rem;
    padding-left: 1.1rem;
    list-style: disc;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--color-ink-muted);
  }
  .methodology :global(strong) {
    color: var(--color-ink);
    font-weight: 600;
  }
  .methodology :global(pre) {
    margin-top: 1rem;
    overflow-x: auto;
    border-left: 2px solid var(--color-rule-strong);
    background: var(--color-surface);
    padding: 0.85rem 1rem;
    font-size: 0.8rem;
    line-height: 1.6;
  }
  .methodology :global(code) {
    font-family: var(--font-mono);
    color: var(--color-ink);
  }
  .methodology :global(p code) {
    font-size: 0.85em;
    background: var(--color-surface);
    padding: 0.1rem 0.3rem;
  }
  .methodology :global(table) {
    margin-top: 1rem;
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }
  .methodology :global(th) {
    text-align: left;
    font-weight: 500;
    color: var(--color-ink-faint);
    border-bottom: 1px solid var(--color-rule-strong);
    padding: 0.4rem 0.75rem 0.4rem 0;
  }
  .methodology :global(td) {
    border-bottom: 1px solid var(--color-rule);
    padding: 0.45rem 0.75rem 0.45rem 0;
    color: var(--color-ink-muted);
    vertical-align: top;
  }
</style>
