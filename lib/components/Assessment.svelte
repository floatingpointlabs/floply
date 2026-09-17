<script lang="ts">
  import type { Assessment } from "$lib/engine/scalingLaws";

  interface Props {
    assessment: Assessment | null;
    class?: string;
  }

  let { assessment, class: className = "" }: Props = $props();

  const TONE = {
    error: "border-[var(--color-warning)] text-[var(--color-warning)]",
    warning: "border-[var(--color-checkpoint)] text-[var(--color-checkpoint)]",
    success: "border-[var(--color-success)] text-[var(--color-success)]",
    info: "border-[var(--color-compute)] text-[var(--color-compute)]"
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
  <!-- <strong> is inline: breaking it across lines renders as a space either side of
       every bold run. -->
  <!-- prettier-ignore -->
  <p class="border-l-2 py-1 pl-3 text-sm leading-relaxed {TONE[assessment.level]} {className}">
    {#each parts as part, i (i)}{#if part.bold}<strong>{part.text}</strong
        >{:else}{part.text}{/if}{/each}
  </p>
{/if}
