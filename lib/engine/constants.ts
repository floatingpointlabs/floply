/** Compute-optimal pre-training requires ~20 tokens per parameter (Hoffmann et al. 2022). */
export const CHINCHILLA_OPTIMAL_RATIO = 20;

/** Empirical sweet spot: ~50 tokens per trainable adapter parameter. */
export const LORA_OPTIMAL_RATIO = 50;

/**
 * The FPL product data palette (blue, cyan, lime, rose, slate): it leaves out the accent
 * violet, signal mint and highlight amber, so a chart never reads as a key or a state. All
 * five clear 4.5:1 against the #15141a ground.
 *
 * Duplicated as --color-compute/storage/checkpoint/warning/success in app/app.css, which
 * the Svelte components use for text tones. Change both.
 */
export const CHART_COLORS = {
  compute: "#6aa8f0",
  storage: "#4dc8e0",
  checkpoint: "#b5d36a",
  warning: "#e88ac6",
  success: "#8fa3bf"
} as const;

export const ARCHITECTURE_OPTIONS = ["Transformer", "CNN", "RNN", "ViT", "Diffusion"];

/** Mixed-precision formats, ordered fastest/smallest → most precise. */
export const MIXED_PRECISION_OPTIONS = ["fp4", "int8", "fp8", "bf16", "fp16", "tf32", "fp32"];
