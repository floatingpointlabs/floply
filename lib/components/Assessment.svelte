<script lang="ts">
  import type { Assessment } from "$lib/engine/scalingLaws";

  interface Props {
    assessment: Assessment | null;
    class?: string;
  }

  let { assessment, class: className = "" }: Props = $props();

  // A notice, not a callout: Panel with a Soft edge; only the point carries the tone.
  const TONE = {
    error: "bg-[var(--color-danger)]",
    warning: "bg-[var(--color-highlight)]",
    success: "bg-[var(--color-signal)] shadow-[0_0_8px_1px_rgb(127_224_192/0.5)]",
    info: "bg-[var(--color-ink-faint)]"
  } as const;

  /**
   * The alternative is {@html}, which would work today because these messages are built
   * from numbers and a fixed enum — but it bakes in a rule that any future message must
   * never contain user text. Splitting keeps that guarantee structural.
   */
  const parts = $derived(
    (assessment?.message ?? "").split("**").map((text, i) => ({ text, bold: i % 2 === 1 }))
  );
</script>

{#if assessment}
  <p
    class="flex max-w-[66ch] items-baseline gap-3 rounded-[var(--radius-control)] border
           border-[var(--color-soft)] bg-[var(--color-panel)] px-4 py-3 text-[var(--color-ink-muted)]
           {className}">
    <span
      class="relative top-[-0.1em] h-2 w-2 shrink-0 rounded-full {TONE[assessment.level]}"
      aria-hidden="true">
    </span>
    <!-- <strong> is inline: breaking it across lines renders as a space either side of
         every bold run. -->
    <!-- prettier-ignore -->
    <span>{#each parts as part, i (i)}{#if part.bold}<strong class="font-semibold text-[var(--color-ink)]">{part.text}</strong
        >{:else}{part.text}{/if}{/each}</span>
  </p>
{/if}
