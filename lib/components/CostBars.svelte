<script lang="ts">
  import { money } from "$lib/engine/formatting";

  interface Segment {
    label: string;
    value: number;
    detail: string;
    color: string;
  }

  interface Props {
    segments: Segment[];
  }

  let { segments }: Props = $props();

  const total = $derived(segments.reduce((a, s) => a + s.value, 0));
  const shown = $derived(segments.filter((s) => s.value > 0));
</script>

{#if total > 0}
  <div
    class="flex h-7 w-full overflow-hidden rounded-sm"
    role="img"
    aria-label="Compute cost split across run types">
    {#each shown as s (s.label)}
      <div
        class="h-full"
        style="width:{(s.value / total) * 100}%; background:{s.color}; opacity:0.9"
        title="{s.label}: {money(s.value)}">
      </div>
    {/each}
  </div>

  <dl class="mt-4 grid gap-4 sm:grid-cols-3">
    {#each segments as s (s.label)}
      <div>
        <dt class="flex items-center gap-2 text-[0.8rem] text-[var(--color-ink-muted)]">
          <span class="h-2.5 w-2.5 shrink-0 rounded-[1px]" style="background:{s.color}"></span>
          {s.label}
        </dt>
        <dd class="num mt-1 text-[1.15rem] leading-none">{money(s.value)}</dd>
        <p class="mt-1 text-xs text-[var(--color-ink-faint)]">{s.detail}</p>
      </div>
    {/each}
  </dl>
{:else}
  <p class="text-sm text-[var(--color-ink-muted)]">No compute cost to break down yet.</p>
{/if}
