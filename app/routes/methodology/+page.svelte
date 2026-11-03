<script lang="ts">
  import { getGpuInstance } from "$lib/engine/gpuSpecs";
  import type { PageProps } from "./$types";

  let { data }: PageProps = $props();

  const sections = [
    { id: "overview", title: "Overview" },
    { id: "tokenisation", title: "Tokens per sample" },
    { id: "scaling-laws", title: "How much data is enough" },
    { id: "flops", title: "Training compute" },
    { id: "lora", title: "LoRA parameters" },
    { id: "throughput", title: "Time and cost" },
    { id: "memory", title: "GPU memory" },
    { id: "storage", title: "Storage" },
    { id: "experiments", title: "Total project cost" },
    { id: "instances", title: "Instances" },
    { id: "limitations", title: "Limitations" }
  ];

  const instances = $derived(
    data.catalog.order.map((t) => ({ type: t, ...getGpuInstance(data.catalog, t) }))
  );
</script>

<svelte:head>
  <title>How it works — Floply</title>
  <meta
    name="description"
    content="The formulas Floply uses to estimate data, compute, memory, storage and cost, and what they assume." />
</svelte:head>

<div class="lg:grid lg:grid-cols-[1fr_14rem] lg:gap-12">
  <article class="methodology max-w-[68ch]">
    <h1>How it works</h1>
    <p class="mt-2 text-sm text-[var(--color-ink-muted)]">
      The formulas behind every estimate, and what they assume. Prices come live from AWS for the
      region you pick.
    </p>

    <section id="overview">
      <h2>Overview</h2>
      <p>
        Floply builds an estimate in five steps: dataset, model, hardware, experiments, then total
        cost. Each step needs the one before it.
      </p>
      <p>
        These are calculations, not benchmarks. Use them to judge whether a project is affordable,
        not to predict a run to the dollar.
      </p>
    </section>

    <section id="tokenisation">
      <h2>Tokens per sample</h2>
      <p>Each data type is converted into tokens and an estimated file size.</p>

      <h3>Text</h3>
      <pre><code>tokens_per_sample = words × 1.3   # average for English
bytes_per_sample  = tokens × 4    # plain text, ~4 bytes per token</code></pre>

      <h3>Image</h3>
      <pre><code>tokens_per_sample = (resolution ÷ patch_size)²
bytes_per_sample  = resolution² × 3 ÷ 10   # JPEG, ~10:1 compression</code></pre>
      <p>
        A 224px image with 16px patches is 196 tokens. Partial patches don't count: at 336px with
        32px patches, that's 10 per side, so 100 tokens, not 110.
      </p>

      <h3>Audio</h3>
      <table>
        <thead>
          <tr>
            <th>Tokenizer</th>
            <th>Rate</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Whisper</td>
            <td>50 tokens/sec</td>
          </tr>
          <tr>
            <td>EnCodec 24 kHz</td>
            <td>75 tokens/sec per codebook</td>
          </tr>
          <tr>
            <td>SoundStream</td>
            <td>50 tokens/sec per codebook</td>
          </tr>
        </tbody>
      </table>
      <p>
        File size is always <code>clip_duration × 32,000 bytes</code>
        (16 kHz, mono, 16-bit).
      </p>

      <h3>Video</h3>
      <pre><code>num_frames        = clip_duration × sampled_fps
tokens_per_frame  = (resolution ÷ patch_size)²
tokens_per_sample = tokens_per_frame × num_frames
bytes_per_sample  = (resolution² × 3 ÷ 10) × num_frames</code></pre>
      <p>Video is stored as JPEG frames, the way most video datasets are kept on disk.</p>
    </section>

    <section id="scaling-laws">
      <h2>How much data is enough</h2>
      <p>
        The health check compares your dataset to the number of parameters being trained. The ranges
        depend on how you train.
      </p>

      <h3>Pre-training</h3>
      <p>
        The ideal is about <strong>20 tokens per parameter</strong>
        .
      </p>
      <table>
        <thead>
          <tr>
            <th>Tokens per parameter</th>
            <th>What it means</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Under 1</td>
            <td>Far too little. Training will likely fail.</td>
          </tr>
          <tr>
            <td>1–10</td>
            <td>Not enough. The model will be weak.</td>
          </tr>
          <tr>
            <td>10–30</td>
            <td>
              <strong>Optimal.</strong>
              The best model for the compute spent.
            </td>
          </tr>
          <tr>
            <td>30–200</td>
            <td>
              More than optimal. Costs more to train, but the smaller model is cheaper to run.
            </td>
          </tr>
          <tr>
            <td>Over 200</td>
            <td>More than the model can use. Consider a bigger model.</td>
          </tr>
        </tbody>
      </table>

      <h3>Full fine-tuning</h3>
      <p>
        The model has already learned a lot, so it needs much less data. Too much data can make it
        forget what it knew.
      </p>
      <table>
        <thead>
          <tr>
            <th>Tokens per parameter</th>
            <th>What it means</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Under 0.1</td>
            <td>Too little to change the model.</td>
          </tr>
          <tr>
            <td>0.1–1</td>
            <td>Probably not enough for stable results.</td>
          </tr>
          <tr>
            <td>1–5</td>
            <td>
              <strong>Optimal.</strong>
              The typical range.
            </td>
          </tr>
          <tr>
            <td>5–20</td>
            <td>Plenty. Watch that the model doesn't forget what it knew.</td>
          </tr>
          <tr>
            <td>Over 20</td>
            <td>Too much. Use LoRA instead.</td>
          </tr>
        </tbody>
      </table>

      <h3>LoRA and QLoRA</h3>
      <p>
        LoRA freezes the model and trains small add-on layers called adapters. The ranges are based
        on <strong>adapter parameters</strong>
        , not the full model.
      </p>
      <table>
        <thead>
          <tr>
            <th>Tokens per adapter parameter</th>
            <th>What it means</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Under 10</td>
            <td>Possibly too little.</td>
          </tr>
          <tr>
            <td>10–100</td>
            <td>
              <strong>Optimal.</strong>
              The typical range.
            </td>
          </tr>
          <tr>
            <td>100–1,000</td>
            <td>Plenty. A higher rank would help.</td>
          </tr>
          <tr>
            <td>Over 1,000</td>
            <td>More than the adapter can use. Consider full fine-tuning.</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section id="flops">
      <h2>Training compute</h2>
      <p>Total compute is measured in floating-point operations (FLOPs):</p>
      <pre><code>C = 6 × N × D</code></pre>
      <ul>
        <li>
          <strong>C</strong>
          — total FLOPs
        </li>
        <li>
          <strong>N</strong>
          — parameters being trained
        </li>
        <li>
          <strong>D</strong>
          — training tokens (dataset tokens × epochs)
        </li>
        <li>
          <strong>6</strong>
          — 2 for the forward pass, 4 for the backward pass
        </li>
      </ul>
      <p>
        Other architectures use a different multiplier: CNN 4, RNN 8, ViT 6, diffusion 6.5. Gradient
        checkpointing adds 2.
      </p>

      <h3>Mixture-of-experts models</h3>
      <p>
        These models only use some of their parameters for each token. Compute is based on the
        <strong>active</strong>
        parameters; memory and checkpoints use the total. DeepSeek V3 is costed at 37B active parameters,
        not 671B.
      </p>
    </section>

    <section id="lora">
      <h2>LoRA parameters</h2>
      <p>
        For models in the list, adapter size is calculated layer by layer from the real
        architecture.
      </p>
      <pre><code>MODULE_DIMS = &lbrace;
    "q_proj":    (d_model, q_dim),    # q_dim  = num_heads × head_dim
    "k_proj":    (d_model, kv_dim),   # kv_dim = num_kv_heads × head_dim
    "v_proj":    (d_model, kv_dim),
    "o_proj":    (q_dim, d_model),
    "up_proj":   (d_model, ffn_intermediate),
    "down_proj": (ffn_intermediate, d_model),
&rbrace;

trainable_params = Σ[rank × (in_dim + out_dim)] × num_layers</code></pre>
      <p>Custom models use an approximation:</p>
      <pre><code>trainable_params ≈ num_modules × 2 × rank × d_model × num_layers</code></pre>

      <h3>LoRA vs. QLoRA</h3>
      <p>
        Both train the same number of parameters. QLoRA stores the frozen model in 4-bit, so it uses
        less memory.
      </p>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Frozen model stored as</th>
            <th>Memory (bytes)</th>
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
      <h2>Time and cost</h2>
      <pre><code>cluster_flops_per_s = peak_flops × utilisation × total_GPUs
wall_clock_hours    = total_flops ÷ cluster_flops_per_s ÷ 3600
gpu_hours           = wall_clock_hours × total_GPUs
compute_cost        = wall_clock_hours × hourly_rate × num_instances</code></pre>

      <h3>Speed by precision, compared to FP16</h3>
      <table>
        <thead>
          <tr>
            <th>Precision</th>
            <th>V100</th>
            <th>A100</th>
            <th>H100 / H200</th>
            <th>B200</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>fp4</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>4×</td>
          </tr>
          <tr>
            <td>fp8</td>
            <td>—</td>
            <td>—</td>
            <td>2×</td>
            <td>2×</td>
          </tr>
          <tr>
            <td>int8</td>
            <td>0.5×</td>
            <td>2×</td>
            <td>2×</td>
            <td>2×</td>
          </tr>
          <tr>
            <td>bf16</td>
            <td>—</td>
            <td>1×</td>
            <td>1×</td>
            <td>1×</td>
          </tr>
          <tr>
            <td>fp16</td>
            <td>1×</td>
            <td>1×</td>
            <td>1×</td>
            <td>1×</td>
          </tr>
          <tr>
            <td>tf32</td>
            <td>—</td>
            <td>0.5×</td>
            <td>0.5×</td>
            <td>0.5×</td>
          </tr>
          <tr>
            <td>fp32</td>
            <td colspan="4">uses the GPU's listed fp32 speed</td>
          </tr>
        </tbody>
      </table>
      <p>
        A dash means the GPU has no hardware support for that format, so it isn't offered for that
        instance.
      </p>

      <h3>GPU utilisation</h3>
      <p>
        No training run reaches a GPU's peak speed. Utilisation is the share you actually get, and
        it depends on your model, batch size, networking and code. The 30% default is cautious;
        well-tuned large runs reach 40–55%.
      </p>
    </section>

    <section id="memory">
      <h2>GPU memory</h2>
      <h3>Model</h3>
      <pre><code>QLoRA:  base × 0.5 + adapters × 16   # 4-bit base
LoRA:   base × 2   + adapters × 16   # bf16 base
Full:   params × 16                  # weights, gradients and optimizer state</code></pre>

      <h3>DPO and PPO</h3>
      <p>
        DPO keeps a second copy of the model for reference (
        <strong>2×</strong>
        model memory). PPO adds a reward model and a critic (
        <strong>4×</strong>
        ).
      </p>

      <h3>Activations</h3>
      <pre><code>without checkpointing: batch × seq_len × d_model × num_layers × 4 × 2
with checkpointing:    batch × seq_len × d_model × 4 × 2</code></pre>
      <p>
        The 4 covers the main intermediate results in each layer; the 2 is bytes per value in bf16.
        With gradient checkpointing, only one layer is held in memory at a time.
      </p>
      <pre><code>memory_per_GPU_GB = (model_bytes + activation_bytes) ÷ total_GPUs ÷ 1e9</code></pre>
    </section>

    <section id="storage">
      <h2>Storage</h2>
      <pre><code>dataset_TB   = bytes_per_sample × num_samples ÷ 1e12
dataset_cost = price_per_TB_month × dataset_TB × months</code></pre>
      <p>Each checkpoint saves 14 bytes per trained parameter.</p>
      <pre><code>checkpoint_TB = params × 14 × (
      checkpoints_per_run × full_runs
    + tuning_runs      # one checkpoint each
    + ablation_runs    # one checkpoint each
  ) ÷ 1e12</code></pre>
      <p>
        Tuning runs add up fast: dozens of runs, each leaving a checkpoint, can outweigh the main
        runs.
      </p>
    </section>

    <section id="experiments">
      <h2>Total project cost</h2>
      <pre><code>single_run    = wall_clock_hours × hourly_rate × num_instances

total_compute = single_run × full_runs
              + single_run × tuning_run_length × tuning_runs
              + single_run × ablation_run_length × ablation_runs

total_project = total_compute + dataset_storage + checkpoint_storage</code></pre>
    </section>

    <section id="instances">
      <h2>Instances</h2>
      <p>On-demand prices for the selected region.</p>
      <table>
        <thead>
          <tr>
            <th>Instance</th>
            <th>GPU</th>
            <th>GPUs</th>
            <th>Memory</th>
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
        Adding more GPUs or instances is assumed to speed training up perfectly, with no networking
        slowdown.
      </p>
    </section>

    <section id="limitations">
      <h2>Limitations</h2>
      <p>Where the estimate is simplified. Read this before using a number to justify a budget.</p>
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
            <td>More GPUs always means proportionally faster. No networking overhead.</td>
          </tr>
          <tr>
            <td>Pricing</td>
            <td>On-demand only. Spot can be 60–90% cheaper, reserved 30–40%.</td>
          </tr>
          <tr>
            <td>Cloud</td>
            <td>AWS only. Google Cloud, Azure and others aren't included.</td>
          </tr>
          <tr>
            <td>Optimizer</td>
            <td>Adam, at 16 bytes per parameter. Others may use less memory.</td>
          </tr>
          <tr>
            <td>Startup</td>
            <td>No time for setup, data loading or compilation.</td>
          </tr>
          <tr>
            <td>Data loading</td>
            <td>Assumed never to slow training. If it does, utilisation drops.</td>
          </tr>
          <tr>
            <td>Precision</td>
            <td>Speedups are best-case. Real code rarely reaches them.</td>
          </tr>
          <tr>
            <td>LoRA</td>
            <td>Exact for listed models; custom models use an approximation.</td>
          </tr>
          <tr>
            <td>Storage</td>
            <td>S3 only. No other file systems or local disks.</td>
          </tr>
        </tbody>
      </table>
    </section>
  </article>

  <nav class="mt-12 hidden lg:mt-0 lg:block" aria-label="On this page">
    <ul
      class="sticky top-[calc(var(--nav-height)+1.5rem)] space-y-2 border-l border-[var(--color-soft)] pl-4 text-sm">
      {#each sections as s (s.id)}
        <li>
          <a
            href="#{s.id}"
            class="text-[var(--color-ink-muted)] transition-colors hover:text-[var(--color-ink)]">
            {s.title}
          </a>
        </li>
      {/each}
    </ul>
  </nav>
</div>

<style>
  .methodology :global(section) {
    margin-top: 2.5rem;
    scroll-margin-top: calc(var(--nav-height) + 1rem);
  }
  /* Sections sit under a Soft line, with the space above. */
  .methodology :global(h2) {
    border-top: 1px solid var(--color-soft);
    padding-top: 1rem;
  }
  .methodology :global(h3) {
    margin-top: 1.75rem;
  }
  .methodology :global(p),
  .methodology :global(ul) {
    margin-top: 0.75rem;
    line-height: 1.7;
    color: var(--color-ink-muted);
  }
  .methodology :global(ul) {
    padding-left: 1.25rem;
    list-style: disc;
  }
  .methodology :global(strong) {
    color: var(--color-ink);
    font-weight: 600;
  }
  .methodology :global(pre) {
    margin-top: 1rem;
    overflow-x: auto;
    background: var(--color-well);
    border: 1px solid var(--color-soft);
    border-radius: var(--radius-screen);
    padding: 0.75rem 1rem;
    line-height: 1.6;
  }
  .methodology :global(code) {
    color: var(--color-ink);
  }
  .methodology :global(p code),
  .methodology :global(li code),
  .methodology :global(td code) {
    background: var(--color-surface);
    border-radius: 3px;
    padding: 0.05em 0.3em;
  }
  /* A table on the page is a raised Panel with 14px corners; it scrolls inside itself. */
  .methodology :global(table) {
    display: block;
    margin-top: 1rem;
    overflow-x: auto;
    border-collapse: collapse;
    background: var(--color-panel);
    border: 1px solid var(--color-soft);
    border-radius: var(--radius-screen);
    box-shadow: var(--shadow-raised);
    font-size: 0.875rem;
    font-variant-numeric: tabular-nums;
  }
  .methodology :global(th),
  .methodology :global(td) {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--color-soft);
    text-align: left;
    vertical-align: top;
  }
  .methodology :global(th) {
    font-weight: 600;
    color: var(--color-ink-muted);
  }
  .methodology :global(td) {
    color: var(--color-ink-muted);
  }
  .methodology :global(tbody tr:last-child td) {
    border-bottom: 0;
  }
</style>
