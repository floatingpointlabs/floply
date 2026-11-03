<script lang="ts">
  import { goto } from "$app/navigation";
  import { page } from "$app/state";
  import type { Provenance } from "$lib/engine/types";

  let {
    regions,
    provenance
  }: {
    regions: { code: string; label: string }[];
    provenance: Provenance;
  } = $props();

  let open = $state(false);

  async function select(event: Event) {
    const url = new URL(page.url);
    url.searchParams.set("region", (event.currentTarget as HTMLSelectElement).value);
    await goto(url, { invalidateAll: true, noScroll: true });
  }

  const summary = $derived.by(() => {
    if (provenance.source === "unavailable") return `Pricing unavailable · ${provenance.region}`;
    const when = provenance.age_days ? `${provenance.age_days}d ago` : "just now";
    const label = provenance.source === "live" ? "Live" : "Cached";
    const suffix = provenance.stale && provenance.last_error ? " · refresh failed" : "";
    return `${label} · fetched ${when}${suffix}`;
  });

  const tone = $derived(
    provenance.source === "unavailable"
      ? "text-[var(--color-danger)]"
      : provenance.stale
        ? "text-[var(--color-highlight)]"
        : "text-[var(--color-ink-faint)]"
  );

  const detailCount = $derived(
    Object.keys(provenance.quarantined).length + provenance.warnings.length
  );
</script>

<div class="relative flex flex-col items-start gap-1.5 sm:items-end">
  <label class="flex items-center gap-2 text-sm">
    <span class="font-medium text-[var(--color-ink-muted)]">Region</span>
    <select
      value={provenance.region}
      onchange={select}
      class="control w-auto !py-1.5 text-sm"
      title="Instance and S3 prices are region-specific.">
      {#each regions as region (region.code)}
        <option value={region.code}>
          {region.label ? `${region.code} — ${region.label}` : region.code}
        </option>
      {/each}
    </select>
  </label>

  <p class="px-1 text-sm leading-snug {tone}">
    {#if !provenance.stale && provenance.source !== "unavailable"}
      <span class="point mr-1.5 align-[0.05em]" aria-hidden="true"></span>
    {/if}
    {summary}
    {#if detailCount}
      <button
        type="button"
        class="font-medium text-[var(--color-accent-lit)] hover:text-[var(--color-ink)]"
        aria-expanded={open}
        onclick={() => (open = !open)}>
        {detailCount} note{detailCount === 1 ? "" : "s"}
      </button>
    {/if}
  </p>

  {#if open}
    <div
      class="popover settle absolute top-full left-0 z-30 mt-2 w-[min(22rem,90vw)] p-4 text-sm
             leading-relaxed text-[var(--color-ink-muted)] sm:right-0 sm:left-auto">
      <p>
        On-demand list price, Linux, shared tenancy. Cache: <code>{provenance.cache_location}</code>
      </p>
      {#if provenance.last_error}
        <p class="mt-2">Last error: {provenance.last_error}</p>
      {/if}
      {#if Object.keys(provenance.quarantined).length}
        <p class="mt-2 font-semibold text-[var(--color-ink)]">Discovered but unusable</p>
        <ul class="list-disc pl-4">
          {#each Object.entries(provenance.quarantined) as [instanceType, reason] (instanceType)}
            <li>
              <code>{instanceType}</code>
              — {reason}
            </li>
          {/each}
        </ul>
      {/if}
      {#if provenance.warnings.length}
        <p class="mt-2 font-semibold text-[var(--color-ink)]">Warnings</p>
        <ul class="list-disc pl-4">
          <!-- Keyed by index, not by text: two warnings can legitimately read the same
               (one per region, one per storage class), and a duplicate key would crash the
               very panel that exists to report problems. -->
          {#each provenance.warnings as warning, i (i)}
            <li>{warning}</li>
          {/each}
        </ul>
      {/if}
    </div>
  {/if}
</div>
