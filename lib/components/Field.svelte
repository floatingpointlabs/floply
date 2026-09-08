<script lang="ts">
  interface Props {
    label: string;
    id: string;
    hint?: string;
    children: import("svelte").Snippet;
  }

  let { label, id, hint, children }: Props = $props();
</script>

<div class="flex flex-col gap-1.5">
  <label for={id} class="text-[0.8rem] text-[var(--color-ink-muted)]">{label}</label>
  {@render children()}
  {#if hint}
    <p id="{id}-hint" class="text-xs leading-snug text-[var(--color-ink-faint)]">{hint}</p>
  {/if}
</div>
