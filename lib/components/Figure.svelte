<script lang="ts">
  interface Props {
    label: string;
    value: string;
    note?: string;
    tone?: "default" | "accent" | "warning" | "success";
  }

  let { label, value, note, tone = "default" }: Props = $props();

  const toneClass = {
    default: "text-[var(--color-ink)]",
    accent: "text-[var(--color-violet)]",
    warning: "text-[var(--color-warning)]",
    success: "text-[var(--color-success)]",
  } as const;
</script>

<div>
  <dt class="text-[0.8rem] leading-tight text-[var(--color-ink-muted)]">{label}</dt>
  <dd class="num mt-1 text-[1.6rem] leading-none {toneClass[tone]}">{value}</dd>
  {#if note}
    <p class="mt-1.5 text-xs leading-snug text-[var(--color-ink-faint)]">{note}</p>
  {/if}
</div>
