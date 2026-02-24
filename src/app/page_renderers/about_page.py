import streamlit as st


def render_about_page():
    """Render the about page with methodology and assumptions."""
    st.header("About Floply")
    st.caption("A compute & storage cost estimator for ML training — February 2026")

    # ── 1. Overview ────────────────────────────────────────────────────────────
    with st.expander("Overview", expanded=True):
        st.markdown("""
Floply walks you through five sequential stages to produce a bottom-up cost estimate
for a complete ML training project:

1. **Dataset Dimensions** — tokenisation, dataset size, and raw storage
2. **Training Configuration** — pre-training or fine-tuning model settings
3. **Compute Config** — instance selection, batch size, MFU, and memory
4. **Experiment & Evaluation** — full runs, HP sweeps, ablations, checkpointing
5. **Cost Summary** — aggregated cost breakdown with charts

Each stage gates the next, so you only see compute settings once a model size
and dataset have been defined.
        """)

    # ── 2. Tokenisation ────────────────────────────────────────────────────────
    with st.expander("Tokenisation — how tokens are counted per modality"):
        st.markdown("""
### Text
```
tokens_per_sample = words x 1.3   # BPE subword average
bytes_per_sample  = tokens x 4    # raw UTF-8 (~4 bytes / token)
```
The 1.3 factor is the empirical BPE inflation ratio for English text.

### Image (ViT-style patch tokenisation)
```
tokens_per_sample = (resolution ÷ patch_size)²
bytes_per_sample  = resolution² x 3 ÷ 10   # JPEG ~10:1 compression
```
Typical values: 224px image with 16px patches → 196 tokens.

### Audio
Three tokeniser styles are supported:

| Style | Rate | Notes |
|---|---|---|
| Whisper | 50 tok/sec | mel-spectrogram frames |
| EnCodec 24kHz | 75 tok/sec x codebooks | residual vector quantisation |
| SoundStream | 50 tok/sec x codebooks | residual vector quantisation |

Raw storage is always estimated as `clip_duration x 32,000 bytes` (16 kHz mono 16-bit PCM).

### Video
```
tokens_per_sample = tokens_per_frame x num_frames
tokens_per_frame  = (resolution ÷ patch_size)²
num_frames        = clip_duration x sampled_fps
bytes_per_frame   = resolution² x 3 ÷ 10   # JPEG per extracted frame
bytes_per_sample  = bytes_per_frame x num_frames
```
Storage is modelled as JPEG-encoded extracted frames, matching how most
ML video datasets (Kinetics, Something-Something, etc.) are stored on disk.
        """)

    # ── 3. Chinchilla / Scaling Law Banner ────────────────────────────────────
    with st.expander("Scaling law health check — Chinchilla, full FT, and LoRA thresholds"):
        st.markdown("""
The banner adapts its thresholds to the training method in use.

---

### Pre-Training
Uses **Hoffmann et al. 2022** (Chinchilla): **D\* ≈ 20 x N**.

| tok / param | Classification |
|---|---|
| < 1 | Dataset critically undersized — will likely diverge |
| 1 - 9 | Undertrained — below Chinchilla-optimal |
| 10 - 30 | **Chinchilla-optimal** — best compute efficiency |
| 31 - 200 | Inference-optimal — LLaMA / Mistral over-train strategy |
| > 200 | Heavily over-trained — diminishing returns |

---

### Full Fine-Tuning
Starts from a converged base, so much less data is needed.
Chinchilla does *not* apply here — the relevant risk is catastrophic forgetting.

| tok / base_param | Classification |
|---|---|
| < 0.1 | Too small — unlikely to shift behaviour meaningfully |
| 0.1 - 1 | Likely undertrained for stable adaptation |
| 1 - 5 | **Good range** — standard SFT sweet spot |
| 5 - 20 | Generous — watch for catastrophic forgetting |
| > 20 | Risk of forgetting base capabilities; consider LoRA instead |

---

### LoRA / QLoRA
The frozen base model provides strong priors. The relevant parameter count is
the **adapter size** (trainable params only), not the full model.

| tok / adapter_param | Classification |
|---|---|
| < 10 | Possibly too little data — adapter may underfit |
| 10 - 100 | **Good range** — standard for task adaptation |
| 100 - 1000 | Large dataset; consider increasing LoRA rank for more capacity |
| > 1000 | Adapter is the bottleneck — consider full fine-tuning or much higher rank |

The banner always shows tok/adapter_param *and* tok/base_param so you can see
both perspectives at once.
        """)

    # ── 4. Training FLOPs ──────────────────────────────────────────────────────
    with st.expander("Training FLOPs — C = 6 x N x D"):
        st.markdown("""
### Base formula (Kaplan et al. 2020)
```
C = 6 x N x D
```
- **C** — total floating-point operations
- **N** — number of *trainable* parameters
- **D** — total training tokens (dataset tokens x epochs)
- **6** — accounts for forward pass (2N) + backward pass (4N)

### Gradient checkpointing
When gradient checkpointing is enabled, recomputed activations add a ≈ 33%
overhead to the forward pass:
```
C_checkpointed ≈ 6 x N x D x 1.33
```

### Fine-tuning with LoRA / QLoRA
Only the trainable adapter parameters are used for N, not the full base model size.
This makes fine-tuning FLOPs dramatically lower than full fine-tuning.
        """)

    # ── 5. LoRA Parameter Count ────────────────────────────────────────────────
    with st.expander("LoRA & QLoRA — trainable parameter calculation"):
        st.markdown("""
For **preset models** (selected from the YAML library), trainable parameters are
computed per-module using the actual architecture dimensions, correctly handling
**Grouped Query Attention (GQA)**:

```python
MODULE_DIMS = {
    "q_proj":    (d_model, d_model),
    "k_proj":    (d_model, kv_dim),   # kv_dim = num_kv_heads x head_dim
    "v_proj":    (d_model, kv_dim),
    "o_proj":    (d_model, d_model),
    "up_proj":   (d_model, ffn_intermediate),
    "down_proj": (ffn_intermediate, d_model),
}

trainable_params = (
    Σ [lora_rank x (in_dim + out_dim) for each selected module]
) x num_layers
```

For **custom models**, the approximate uniform formula is used with a disclaimer:
```
trainable_params ≈ num_modules x 2 x rank x d_model x num_layers
```

### LoRA vs QLoRA — memory difference
Both methods have *identical trainable parameter counts*; they differ in how the
frozen base model weights are stored:

| Method | Base model dtype | Memory formula |
|---|---|---|
| Full Fine-Tuning | fp32/bf16 (16 bytes/param) | params x 16 |
| LoRA | bf16 (2 bytes/param) | base x 2 + adapters x 16 |
| QLoRA | int4 (0.5 bytes/param) | base x 0.5 + adapters x 16 |
        """)

    # ── 6. GPU Time & Cost ─────────────────────────────────────────────────────
    with st.expander("GPU time & compute cost — precision-aware throughput"):
        st.markdown("""
### Effective cluster throughput
```
peak_flops       = f(instance_type, mixed_precision)   # see table below
cluster_flops/s  = peak_flops x MFU x total_GPUs
wall_clock_s     = total_flops ÷ cluster_flops/s
wall_clock_hours = wall_clock_s ÷ 3600
gpu_hours        = wall_clock_hours x total_GPUs
compute_cost     = wall_clock_hours x instance_hourly_rate x num_instances
```

### Precision multipliers (vs FP16 baseline)
Each GPU family has different hardware acceleration for different number formats:

| Precision | A100 | H100 | V100 |
|---|---|---|---|
| fp4 | 2x fp16 | 2x fp16 | 1x fp16 |
| int8 | 2x fp16 | 2x fp16 | 0.9x fp16 |
| fp8 | 1x fp16 | **2x fp16** | 1x fp16 |
| bf16 / fp16 | 1x (baseline) | 1x (baseline) | 1x (baseline) |
| tf32 | 0.5x fp16 | 0.5x fp16 | — |
| fp32 | uses fp32 spec | uses fp32 spec | uses fp32 spec |

H100 is the only GPU with hardware fp8 acceleration.

### Model FLOPs Utilization (MFU)
MFU is user-adjustable (5-100%, default 30%). Real-world MFU depends on:
- Model architecture and batch size (larger batches → higher MFU)
- Communication overhead in multi-node training
- Data loading / pipeline bubbles
- Framework overhead (PyTorch vs JAX vs Triton kernels)

A conservative 30% is recommended for initial estimates. Production LLM training
on well-tuned infrastructure typically achieves 40-55%.
        """)

    # ── 7. GPU Memory Estimate ─────────────────────────────────────────────────
    with st.expander("GPU memory estimation — weights + activations"):
        st.markdown("""
### Weight memory
```
# QLoRA:
model_memory = base_params x 0.5 + adapter_params x 16   # int4 base

# LoRA:
model_memory = base_params x 2   + adapter_params x 16   # bf16 base

# Full fine-tuning / pre-training:
model_memory = params x 16                                 # fp32 weights + grads + Adam states
```

### RL memory multiplier
- **DPO** — policy + frozen reference model → **2x** memory
- **PPO** — actor + reference + reward model + critic → **4x** memory

The multiplier is applied to `model_memory` before dividing across GPUs.

### Activation memory
```
# Without gradient checkpointing:
activation_bytes = batch_size x seq_len x d_model x num_layers x 4 x 2

# With gradient checkpointing (only 1 layer stored at a time):
activation_bytes = batch_size x seq_len x d_model x 4 x 2
```
The factor of 4 accounts for QKV projections, attention scores, and MLP
intermediate activations. The factor of 2 is for bf16 (2 bytes per element).

### Total memory per GPU
```
total_memory_GB = (model_memory_bytes + activation_bytes) ÷ total_GPUs ÷ 1e9
```
The memory metric shows a breakdown: `wts X GB + act Y GB`.
        """)

    # ── 8. Storage Costs ───────────────────────────────────────────────────────
    with st.expander("Storage costs — dataset and checkpoints on S3"):
        st.markdown("""
### Dataset storage
```
dataset_size_TB      = bytes_per_sample x num_samples ÷ 1e12
dataset_storage_cost = S3_price_per_TB_month x dataset_size_TB x storage_months
```

### Checkpoint storage
Each checkpoint stores all trainable parameters in mixed precision (14 bytes/param
≈ bf16 weights + fp32 optimiser states):

```
checkpoint_size_TB = (
    params x 14 x checkpoints_per_run x full_runs    # full training runs
  + params x 14 x 1 x hp_trials                      # 1 final ckpt per HP trial
  + params x 14 x 1 x ablations                      # 1 final ckpt per ablation
) ÷ 1e12
```

HP trials and ablations each save one final checkpoint, which accumulates
meaningfully when running dozens of sweeps.

### S3 storage classes
Standard pricing is applied by default. The storage class selector lets you
model cheaper tiers (Infrequent Access, Glacier) for archival use cases.
        """)

    # ── 9. Experiment & Total Cost ─────────────────────────────────────────────
    with st.expander("Experiment cost — full runs, HP sweeps, ablations"):
        st.markdown("""
### Compute cost rollup
```
single_run_cost  = wall_clock_hours x hourly_rate x num_instances

total_compute_cost = (
    single_run_cost x num_full_runs
  + single_run_cost x hp_run_fraction x num_hp_trials
  + single_run_cost x ablation_fraction x num_ablations
)
```

### Total project cost
```
total_project_cost = total_compute_cost + dataset_storage_cost + checkpoint_storage_cost
```
        """)

    # ── 10. AWS Instance Reference ─────────────────────────────────────────────
    with st.expander("AWS GPU instance reference — pricing & specs"):
        st.markdown("""
Prices are on-demand averages across US regions, as of February 2026.

| Instance | GPU | Count | VRAM/GPU | FP16 TFLOPS | Cost/hr |
|---|---|---|---|---|---|
| p4d.24xlarge | A100 | 8 | 40 GB | 312 | $26.87 |
| p4de.24xlarge | A100 | 8 | 80 GB | 312 | $32.77 |
| p5.48xlarge | H100 | 8 | 80 GB | 989 | $66.64 |
| p3.16xlarge | V100 | 8 | 16 GB | 125 | $24.48 |
| p3dn.24xlarge | V100 | 8 | 32 GB | 125 | $31.22 |

All instances use NVLink / NVSwitch for intra-node GPU communication.
Multi-node scaling is modelled as linear (no communication overhead penalty).
        """)

    # ── 11. Limitations ────────────────────────────────────────────────────────
    with st.expander("Known limitations & assumptions"):
        st.markdown("""
| Area | Assumption / Limitation |
|---|---|
| FLOPs formula | `C = 6ND` applies cleanly to dense transformers. MoE models require a sparsity correction. |
| Scaling | Linear across GPUs and nodes — no modelling of NCCL / EFA communication overhead. |
| Pricing | On-demand only. Spot instances can be 60-90% cheaper; Reserved instances 30-40% cheaper. |
| Cloud | AWS only. GCP (TPUs/A3), Azure (NDv4/NDv5), CoreWeave not modelled. |
| Optimizer | Adam (16 bytes/param for weights + grads + 2 moment estimates). Lion / Adafactor would reduce memory. |
| Startup overhead | Assumes steady-state training; ignores instance spin-up, dataset prefetch, and compilation time. |
| Data pipeline | Data loading bottlenecks are not modelled — actual MFU may be lower if I/O-bound. |
| Mixed precision | Precision multipliers are theoretical peaks; real kernels may not saturate hardware. |
| LoRA accuracy | Per-module GQA-aware calculation for preset models. Custom models use an approximate formula. |
| Storage | S3 only; no EFS, FSx for Lustre (common in production), or local NVMe during training. |
        """)

    st.caption("Floply v2.0 · February 2026 · [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) · [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)")