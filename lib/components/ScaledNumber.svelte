<script lang="ts">
  import { UNIT_MULTIPLIERS } from "$lib/engine/formatting";

  interface Props {
    id: string;
    value: number;
    unit: string;
    units?: string[];
  }

  let {
    id,
    value = $bindable(),
    unit = $bindable(),
    units = ["M", "B"],
  }: Props = $props();

  const resolved = $derived(Math.trunc(value * UNIT_MULTIPLIERS[unit]));
</script>

<div class="flex gap-2">
  <input
    {id}
    type="number"
    bind:value
    min="0.001"
    max="9999"
    step="0.1"
    class="control num"
  />
  <select
    bind:value={unit}
    aria-label="Magnitude"
    class="control w-auto"
  >
    {#each units as u (u)}
      <option value={u}>{u}</option>
    {/each}
  </select>
</div>
<p class="num text-xs text-[var(--color-ink-faint)]">{resolved.toLocaleString("en-US")}</p>
