<script lang="ts">
  interface Props {
    label: string;
    value: string;
    note?: string;
    tone?: "default" | "accent" | "warning" | "success";
  }

  let { label, value, note, tone = "default" }: Props = $props();

  // Data isn't action: a figure is never violet. Signal says it landed, Danger that it won't fit.
  const toneClass = {
    default: "text-[var(--color-ink)]",
    accent: "text-[var(--color-ink)]",
    warning: "text-[var(--color-danger)]",
    success: "text-[var(--color-signal)]"
  } as const;
</script>

<!-- A tile: a raised Plate card, the value at 1.5rem tabular. -->
<div class="card px-4 py-3.5">
  <dt class="text-sm font-medium text-[var(--color-ink-muted)]">{label}</dt>
  <dd class="num mt-1 text-2xl leading-tight font-bold tracking-[-0.01em] {toneClass[tone]}">
    {value}
  </dd>
  {#if note}
    <p class="mt-1 text-xs text-[var(--color-ink-faint)]">{note}</p>
  {/if}
</div>
