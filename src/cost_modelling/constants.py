"""Shared domain constants.

These are owned here rather than by whichever module happened to need them first, so
budget_optimizer and scaling_laws can both use them without importing each other.
Mirrors lib/engine/constants.ts.
"""

# Chinchilla scaling-law optimal ratio (Hoffmann et al. 2022).
# Compute-optimal pre-training requires ~20 tokens per parameter.
CHINCHILLA_OPTIMAL_RATIO = 20

# LoRA efficiency optimal token-to-adapter-param ratio.
# Empirical sweet spot: ~50 tokens per trainable adapter parameter.
LORA_OPTIMAL_RATIO = 50
