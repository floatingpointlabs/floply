/** Compute-optimal pre-training requires ~20 tokens per parameter (Hoffmann et al. 2022). */
export const CHINCHILLA_OPTIMAL_RATIO = 20;

/** Empirical sweet spot: ~50 tokens per trainable adapter parameter. */
export const LORA_OPTIMAL_RATIO = 50;

/**
 * All five clear 3:1 against the #0a0a0f ground for graphical use; storage, checkpoint,
 * warning and success also clear 4.5:1, so they are safe as text.
 */
export const CHART_COLORS = {
  compute: "#6366f1",
  storage: "#22d3ee",
  checkpoint: "#f59e0b",
  warning: "#fb7185",
  success: "#34d399",
} as const;

export const ARCHITECTURE_OPTIONS = ["Transformer", "CNN", "RNN", "ViT", "Diffusion"];

/** Mixed-precision formats, ordered fastest/smallest → most precise. */
export const MIXED_PRECISION_OPTIONS = [
  "fp4", "int8", "fp8", "bf16", "fp16", "tf32", "fp32",
];

