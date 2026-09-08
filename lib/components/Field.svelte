<script lang="ts">
  interface Props {
    label: string;
    id: string;
    hint?: string;
    children: import("svelte").Snippet;
  }

  let { label, id, hint, children }: Props = $props();

  // The control is supplied by the caller's snippet, so this component has no reference to
  // it and cannot put `aria-describedby` on it directly. It is addressable by the same id
  // the label already points at. Without this the hint is rendered but never announced.
  $effect(() => {
    const el = document.getElementById(id);
    if (!el) return;
    // Cleared as well as set: a hint can come and go (the precision field only warns on
    // GPUs without a hardware path), and a stale reference to a removed element is worse
    // than none at all.
    if (hint) el.setAttribute("aria-describedby", `${id}-hint`);
    else el.removeAttribute("aria-describedby");
  });
</script>

<div class="flex flex-col gap-1.5">
  <label for={id} class="text-[0.8rem] text-[var(--color-ink-muted)]">{label}</label>
  {@render children()}
  {#if hint}
    <p id="{id}-hint" class="text-xs leading-snug text-[var(--color-ink-faint)]">{hint}</p>
  {/if}
</div>
