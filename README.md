# Floply

Floply is an open-source, Streamlit-based ML training cost estimator. Given a dataset's dimensions, a model architecture, and a compute cluster, it calculates total GPU hours, wall-clock time, memory requirements, storage costs, and full project cost across training runs, hyperparameter trials, and ablations — all without writing any code.

## Setup

**Prerequisites:** [pyenv](https://github.com/pyenv/pyenv), [direnv](https://direnv.net/), and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/your-org/floply.git
cd floply

# Install Python 3.12.6 via pyenv if not already present
pyenv install 3.12.6

# Allow direnv — this creates .venv, activates it, and runs poetry install automatically
direnv allow
```

direnv reads `.envrc` on every `cd` into the project, so the virtualenv stays active and dependencies stay in sync without any manual steps.

## Quick start

**Run locally** (after setup above):

```bash
streamlit run src/app/app.py
```

**With Docker** (no local Python setup needed):

```bash
docker build -t floply .
docker run -p 8501:8501 \
  -v floply-cache:/app/.cache \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  floply
```

Then open `http://localhost:8501` in your browser.

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

Model definitions live in `src/data/models/`. Each file is a YAML that describes an architecture. The app picks up every `.yaml` in that directory automatically — no Python changes needed.

To contribute a new model:

1. Create `src/data/models/<slug>.yaml` following the format below
2. Verify it appears in the **Fine-Tuning → Base Model** dropdown when you run the app
3. Open a pull request

### Minimal example (dense transformer)

```yaml
# src/data/models/my_model_7b.yaml
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

MoE models must include `moe.active_parameter_count`. Floply uses this value — not `parameter_count` — when computing FLOPs, since only a subset of experts is active per forward pass.

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
