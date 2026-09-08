# Floply

Floply is an open-source ML training cost estimator. Given a dataset's dimensions, a model architecture, and a compute cluster, it calculates total GPU hours, wall-clock time, memory requirements, storage costs, and full project cost across training runs, hyperparameter trials, and ablations — all without writing any code.

Four tools:

| | |
|---|---|
| **Budget optimizer** | Given a budget, the largest model and dataset you can afford, and where the scaling-law optimum sits |
| **Minimum data** | How much data a model needs, from Chinchilla and practical fine-tuning thresholds |
| **Training budget** | Bottom-up cost for a whole project: compute, storage, sweeps and ablations |
| **Methodology** | Every formula behind the estimates, and what they assume |

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

## With Docker

```bash
docker build -f Dockerfile.web -t floply .
docker run -p 3000:3000 floply
```

Then open `http://localhost:3000`. Health check is at `/api/health`.

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
- **`app/routes/`** — the four pages.
- **`data/`** — model and provider definitions as YAML, compiled to TypeScript at build time.
- **`fixtures/`** — golden test fixtures pinning every engine function's output. `pnpm test` replays them.

The engine is deliberately framework-agnostic: it is exercised by the fixtures with no DOM, so the UI layer can change without touching the maths.

Both paths need AWS credentials — see below.

## AWS pricing

Floply reads instance prices and hardware specs live from the AWS Price List and
EC2 APIs. **There is no bundled price snapshot**: a cost estimator quoting
months-old prices is worse than one that says it cannot price at all.

### Credentials

Standard boto3 resolution — environment variables, `~/.aws/credentials`, or an
instance/task role. The required policy is read-only and grants no access to
your account's data:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["pricing:GetProducts"], "Resource": "*"},
    {"Effect": "Allow",
     "Action": ["ec2:DescribeInstanceTypes", "ec2:DescribeInstanceTypeOfferings"],
     "Resource": "*"}
  ]
}
```

A cheap credential check runs at startup so a bad policy surfaces at deploy time
rather than to a user mid-form.

### Caching

Prices are fetched lazily — nothing calls AWS until a price is actually needed,
so loading the app or reading the About tab costs nothing. Results are cached per
region (30 days by default) with hardware specs cached separately (90 days),
which keeps a cold start to one region's worth of calls rather than all of them.

The TTL means *"try to refresh"*, never *"delete"*. If AWS is unreachable, cached
prices keep being served behind a prominent staleness warning; the app only
refuses to price when it has nothing cached at all. **Mount a volume over
`/app/.cache`** so this survives container replacement.

Prime or inspect the cache directly:

```bash
python -m src.cost_modelling.pricing.refresh_cli            # refresh stale entries
python -m src.cost_modelling.pricing.refresh_cli --force    # refresh everything
python -m src.cost_modelling.pricing.refresh_cli --probe-only  # check credentials
```

### Configuration

All settings are environment variables; the common ones:

| Variable | Default | Purpose |
|---|---|---|
| `FLOPLY_AWS_REGIONS` | six common regions | Regions offered in the selector |
| `FLOPLY_DEFAULT_REGION` | `us-east-1` | Initial selection |
| `FLOPLY_PRICING_TTL_DAYS` | `30` | Price refresh threshold |
| `FLOPLY_SPECS_TTL_DAYS` | `90` | Hardware spec refresh threshold |
| `FLOPLY_CACHE_DIR` | `.cache/floply` | Cache location |
| `FLOPLY_INSTANCE_FAMILY_ALLOWLIST` | `p` | Instance families eligible for discovery |
| `FLOPLY_DISCOVER_MIN_GPUS` | `4` | Minimum GPUs for a discovered instance |

See `src/cost_modelling/pricing/settings.py` for the full list.

> Prices shown are **on-demand list prices** (Linux, shared tenancy). Real
> training spend usually goes through Capacity Blocks, Savings Plans, or Spot,
> which are substantially cheaper — treat the output as an upper bound.

## Adding a GPU

Throughput (`peak_flops_*`) and `typical_mfu` come from NVIDIA datasheets and
empirical measurement — AWS publishes neither, so they are curated in
`src/data/gpu_hardware.yaml`, keyed by GPU model.

This is what makes instance discovery safe: when AWS starts offering a new
family, Floply picks up its specs and price automatically, but **quarantines it**
until its GPU has a curated entry. A quarantined instance is listed in the
sidebar's *Pricing data source* panel with the exact key to add, rather than
appearing with guessed throughput.

An absent entry in `precision_multipliers` means the die has *no hardware
support* for that format — not that it runs at 1×. That distinction is why
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
  num_kv_heads: 8        # omit if not GQA — defaults to num_heads
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
parameter_count: 140000000000   # total parameters across all experts

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
    active_parameter_count: 12000000000   # params active per token — used for FLOPs

source: "https://huggingface.co/..."
notes: "Brief description."
```

The build validates these files and fails loudly on a malformed one — a missing
`active_parameter_count` on an MoE model, a slug that doesn't match the filename, or a
numeric field that parsed as a string.

### Field reference

#### Required

| Field | Type | Used for |
|---|---|---|
| `name` | string | Display name in the UI dropdown |
| `slug` | string | Unique identifier; must match the filename without `.yaml` |
| `family` | string | FLOPs multiplier (`transformer` 6×, `cnn` 4×, `rnn` 8×, `vit` 6×, `diffusion` 6.5×) |
| `parameter_count` | integer | FLOPs, checkpoint storage size, memory estimation |

#### Optional — `architecture`

Including these fields enables GQA-aware LoRA parameter counting and accurate activation memory estimation. Without them, Floply falls back to uniform approximations.

| Field | Used for | Default if omitted |
|---|---|---|
| `num_layers` | Activation memory, LoRA trainable param count | Falls back to approximation |
| `d_model` | Activation memory, LoRA adapter sizing | Falls back to approximation |
| `num_heads` | Per-head dimension for `q_proj` / `o_proj` LoRA | Falls back to approximation |
| `num_kv_heads` | GQA-aware `k_proj` / `v_proj` LoRA sizing | Defaults to `num_heads` |
| `ffn_intermediate` | LoRA `up_proj` / `down_proj` adapter sizing | Defaults to `4 × d_model` |
| `ffn_type` | Informational only | — |
| `vocab_size` | Informational only | — |

#### Optional — `architecture.moe` (MoE models only)

| Field | Used for |
|---|---|
| `num_experts` | Informational |
| `active_experts` | Informational |
| `active_parameter_count` | **FLOPs calculation** — overrides `parameter_count` when computing training compute |

#### Other top-level

| Field | Used for |
|---|---|
| `source` | URL to the HuggingFace model page or paper (informational) |
| `notes` | Short description shown as a caption below the model selector |

## Adding a provider or instance

Instance pricing and specs live in `data/providers/*.yaml`. Add an instance there and it
appears in the cluster selector, in the methodology reference table, and in every
estimate — there is no second place to update.

Note that low-precision speedups are per-GPU-family and declared in
`lib/engine/gpuSpecs.ts`: FP8 requires Hopper, FP4 requires Blackwell. A precision the
GPU cannot accelerate is costed at its FP16 rate rather than being given a speedup it
does not have.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
