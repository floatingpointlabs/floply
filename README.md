# Floply

Floply is an open-source ML training cost estimator. Given a dataset's dimensions, a model architecture, and a compute cluster, it calculates total GPU hours, wall-clock time, memory requirements, storage costs, and full project cost across training runs, hyperparameter trials, and ablations — all without writing any code.

Four tools:

|                      |                                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| **Minimum data**     | How much data a model needs, from Chinchilla and practical fine-tuning thresholds                    |
| **Budget optimizer** | Given a budget, the largest model and dataset you can afford, and where the scaling-law optimum sits |
| **Training budget**  | Bottom-up cost for a whole project: compute, storage, sweeps and ablations                           |
| **Methodology**      | Every formula behind the estimates, and what they assume                                             |

## Setup

**Prerequisites:** Node 22+ and [pnpm](https://pnpm.io/) (via `corepack enable`).

```bash
git clone https://github.com/floatingpointlabs/floply.git
cd floply
pnpm install
pnpm dev
```

Then open `http://localhost:5173`.

## Commands

```bash
pnpm dev        # dev server
pnpm build      # production build (adapter-node)
pnpm preview    # serve the production build
pnpm test       # engine tests against the golden fixtures
pnpm check      # svelte-check + TypeScript
```

`pnpm build-data` regenerates `lib/data/generated.ts` from the YAML in `data/`. It runs
automatically before `dev`, `build`, `test` and `check`, and the generated file is
gitignored — never commit it.

`pnpm gen-fixtures` rewrites `fixtures/*.json` from the current engine. Unlike the above,
this one is **not** automatic and should not be. The fixtures are committed, so `pnpm test`
compares the engine against the behaviour it had when they were last generated — which is
the only thing standing between a deliberate change to the maths and a silent one.
Regenerating and committing without reading the diff blesses whatever changed.

## With Docker

```bash
docker build -t floply .
docker run -p 3000:3000 \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION \
  -v floply-cache:/app/.cache \
  floply
```

Then open `http://localhost:3000`. Health check is at `/api/health`.

The volume holds the fetched price cache — without it every container replacement
re-fetches from AWS, and an AWS outage during one leaves the app unable to price.
To run with no credentials at all, set `FLOPLY_PRICING_FIXTURE=1` (see below).

Analytics are optional and read at **request** time, so one image can be deployed to
several environments:

```bash
docker run -p 3000:3000 \
  -e UMAMI_URL=https://analytics.example.com/script.js \
  -e UMAMI_WEBSITE_ID=your-site-id \
  floply
```

## How it's built

- **`lib/engine/`** — the cost model: FLOPs, scaling laws, memory, storage, pricing. Pure TypeScript with no UI dependencies.
- **`lib/components/`** — shared UI. Charts are drawn directly with `d3-scale` and SVG; there is no charting library.
- **`lib/server/`** — the AWS pricing layer: fetch, validate, cache, resolve. Server-only; never imported by a page.
- **`app/routes/`** — the four pages.
- **`data/`** — model definitions and curated GPU facts as YAML, compiled to TypeScript at build time. Prices are _not_ here — they come from AWS at request time.
- **`fixtures/`** — golden test fixtures pinning every engine function's output. `pnpm test` replays them.

The engine is deliberately framework-agnostic: it is exercised by the fixtures with no DOM, so the UI layer can change without touching the maths. It takes the priced `Catalog` as an argument rather than importing one, because pages are server-rendered and a module-level catalog would leak one visitor's region into another's request.

Both paths need AWS credentials — see below.

## AWS pricing

Floply reads instance prices and hardware specs live from the AWS Price List and
EC2 APIs. **There is no bundled price snapshot**: a cost estimator quoting
months-old prices is worse than one that says it cannot price at all.

Pricing is resolved **server-side** in the layout load function and handed to the
pages as plain data, so credentials never reach the browser.

### Credentials

Standard AWS SDK resolution — environment variables, `~/.aws/credentials`, or an
instance/task role. The required policy is read-only and grants no access to
your account's data:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["pricing:GetProducts"], "Resource": "*" },
    {
      "Effect": "Allow",
      "Action": ["ec2:DescribeInstanceTypes", "ec2:DescribeInstanceTypeOfferings"],
      "Resource": "*"
    }
  ]
}
```

A bad policy surfaces on the first page load: the region chip in the header shows
where the numbers came from, and expands to the full warning and quarantine list.

### Caching

Results are cached per region (30 days by default) with hardware specs cached
separately (90 days), which keeps a cold start to one region's worth of calls
rather than all of them. The first request after a cold start pays that fetch;
everything after it is served from the memoised catalog.

The TTL means _"try to refresh"_, never _"delete"_. If AWS is unreachable, cached
prices keep being served behind a staleness warning and the refresh happens off
the render path, so no request waits on a failing endpoint; the app only refuses
to price when it has nothing cached at all. **Mount a volume over `/app/.cache`**
so this survives container replacement.

### Configuration

All settings are environment variables; the common ones:

| Variable                           | Default            | Purpose                                            |
| ---------------------------------- | ------------------ | -------------------------------------------------- |
| `FLOPLY_AWS_REGIONS`               | six common regions | Regions offered in the selector                    |
| `FLOPLY_DEFAULT_REGION`            | `us-east-1`        | Initial selection                                  |
| `FLOPLY_PRICING_TTL_DAYS`          | `30`               | Price refresh threshold                            |
| `FLOPLY_SPECS_TTL_DAYS`            | `90`               | Hardware spec refresh threshold                    |
| `FLOPLY_CACHE_DIR`                 | `.cache/floply`    | Cache location                                     |
| `FLOPLY_INSTANCE_FAMILY_ALLOWLIST` | `p`                | Instance families eligible for discovery           |
| `FLOPLY_DISCOVER_MIN_GPUS`         | `4`                | Minimum GPUs for a discovered instance             |
| `FLOPLY_PRICING_FIXTURE`           | unset              | Serve frozen fixture prices instead of calling AWS |

See `buildSettings()` in `lib/server/pricing.ts` for the full list.

`FLOPLY_PRICING_FIXTURE=1` is what lets the e2e suite assert exact dollar figures,
and lets you run the app with no credentials. It is opt-in and never a fallback —
a misconfigured deployment shows _"pricing unavailable"_ rather than quietly
serving numbers that look real and are not.

> Prices shown are **on-demand list prices** (Linux, shared tenancy). Real
> training spend usually goes through Capacity Blocks, Savings Plans, or Spot,
> which are substantially cheaper — treat the output as an upper bound.

## Adding a GPU

Throughput (`peak_flops_*`) and `typical_mfu` come from NVIDIA datasheets and
empirical measurement — AWS publishes neither, so they are curated in
`data/gpu_hardware.yaml`, keyed by GPU model as `ec2:DescribeInstanceTypes`
reports it. Adding a GPU generation is a YAML edit, not a code change.

This is what makes instance discovery safe: when AWS starts offering a new
family, Floply picks up its specs and price automatically, but **quarantines it**
until its GPU has a curated entry. A quarantined instance is listed behind the
region chip in the header with the exact key to add, rather than appearing with
guessed throughput.

An absent entry in `precision_multipliers` means the die has _no hardware
support_ for that format — not that it runs at 1×. That distinction is why
selecting fp4 on an A100 is now rejected instead of silently reporting double
the real throughput.

## Adding a model

Model definitions live in `data/models/`. Each file is a YAML that describes an architecture. The app picks up every `.yaml` in that directory automatically — no TypeScript changes needed.

To contribute a new model:

1. Create `data/models/<slug>.yaml` following the format below
2. Run `pnpm dev` and confirm it appears in the **Base model** dropdown
3. Open a pull request containing only the YAML

### Minimal example (dense transformer)

```yaml
# data/models/my_model_7b.yaml
name: "My Model 7B"
slug: "my_model_7b"
family: "transformer"
parameter_count: 7000000000

architecture:
  num_layers: 32
  d_model: 4096
  num_heads: 32
  num_kv_heads: 8 # omit if not GQA — defaults to num_heads
  ffn_intermediate: 11008
  ffn_type: "swiglu"
  vocab_size: 32000

source: "https://huggingface.co/..."
notes: "Short description shown as a caption in the UI."
```

### Mixture-of-Experts example

MoE models must include `moe.active_parameter_count`. Floply uses this value — not `parameter_count` — when computing FLOPs, since only a subset of experts is active per forward pass. Checkpoint size and GPU memory still use the total.

```yaml
name: "My MoE Model"
slug: "my_moe_model"
family: "transformer"
parameter_count: 140000000000 # total parameters across all experts

architecture:
  num_layers: 32
  d_model: 4096
  num_heads: 32
  num_kv_heads: 8
  ffn_intermediate: 14336
  ffn_type: "swiglu"
  vocab_size: 32000
  moe:
    num_experts: 64
    active_experts: 2
    active_parameter_count: 12000000000 # params active per token — used for FLOPs

source: "https://huggingface.co/..."
notes: "Brief description."
```

The build validates these files and fails loudly on a malformed one — a missing
`active_parameter_count` on an MoE model, a slug that doesn't match the filename, or a
numeric field that parsed as a string.

### Field reference

#### Required

| Field             | Type    | Used for                                                                            |
| ----------------- | ------- | ----------------------------------------------------------------------------------- |
| `name`            | string  | Display name in the UI dropdown                                                     |
| `slug`            | string  | Unique identifier; must match the filename without `.yaml`                          |
| `family`          | string  | FLOPs multiplier (`transformer` 6×, `cnn` 4×, `rnn` 8×, `vit` 6×, `diffusion` 6.5×) |
| `parameter_count` | integer | FLOPs, checkpoint storage size, memory estimation                                   |

#### Optional — `architecture`

Including these fields enables GQA-aware LoRA parameter counting and accurate activation memory estimation. Without them, Floply falls back to uniform approximations.

| Field              | Used for                                        | Default if omitted          |
| ------------------ | ----------------------------------------------- | --------------------------- |
| `num_layers`       | Activation memory, LoRA trainable param count   | Falls back to approximation |
| `d_model`          | Activation memory, LoRA adapter sizing          | Falls back to approximation |
| `num_heads`        | Per-head dimension for `q_proj` / `o_proj` LoRA | Falls back to approximation |
| `num_kv_heads`     | GQA-aware `k_proj` / `v_proj` LoRA sizing       | Defaults to `num_heads`     |
| `ffn_intermediate` | LoRA `up_proj` / `down_proj` adapter sizing     | Defaults to `4 × d_model`   |
| `ffn_type`         | Informational only                              | —                           |
| `vocab_size`       | Informational only                              | —                           |

#### Optional — `architecture.moe` (MoE models only)

| Field                    | Used for                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `num_experts`            | Informational                                                                       |
| `active_experts`         | Informational                                                                       |
| `active_parameter_count` | **FLOPs calculation** — overrides `parameter_count` when computing training compute |

#### Other top-level

| Field    | Used for                                                      |
| -------- | ------------------------------------------------------------- |
| `source` | URL to the HuggingFace model page or paper (informational)    |
| `notes`  | Short description shown as a caption below the model selector |

## Adding an instance

You don't. Instance types and prices are discovered from AWS — add a GPU to
`data/gpu_hardware.yaml` (above) and any instance carrying it is picked up in
every region that sells it, in the cluster selector, in the methodology reference
table, and in every estimate.

Low-precision support is declared per GPU in that same file, as
`precision_multipliers` relative to `peak_flops_fp16`. `null` means "use
`peak_flops_fp32`"; an **absent** key means the die has no hardware for that
format, and asking for it raises rather than quietly falling back.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
