<script lang="ts">
  import { logAxis } from "$lib/logAxis";
  import { fmtTokens } from "$lib/engine/formatting";
  import type { Tier } from "$lib/engine/tiers";

  interface Props {
    tiers: Tier[];
    /** Parameter (or adapter-parameter) count the ratios are measured against. */
    effectiveCount: number;
  }

  let { tiers, effectiveCount }: Props = $props();

  const ROW_HEIGHT = 44;
  const LABEL_WIDTH = 148;
  const AXIS_HEIGHT = 28;
  const PAD_RIGHT = 16;

  let width = $state(760);

  const rows = $derived(
    tiers.map((tier) => ({
      tier,
      start: effectiveCount * tier.ratio_min_chart,
      end: effectiveCount * tier.ratio_max
    }))
  );

  const domain = $derived<[number, number]>([
    Math.min(...rows.map((r) => r.start)),
    Math.max(...rows.map((r) => r.end))
  ]);
  const plotWidth = $derived(Math.max(120, width - LABEL_WIDTH - PAD_RIGHT));
  const axis = $derived(logAxis(domain, [0, plotWidth]));
  const renderable = $derived(axis.ok);
  const x = $derived(axis.scale);
  const ticks = $derived(axis.ticks);
  const height = $derived(rows.length * ROW_HEIGHT + AXIS_HEIGHT);
</script>

<figure class="m-0" bind:clientWidth={width}>
  {#if !renderable}
    <p class="text-sm text-[var(--color-ink-muted)]">
      No parameter count yet, so there is nothing to plot.
    </p>
  {:else}
    <svg
      {width}
      {height}
      role="img"
      aria-label="Token requirements by scaling-law tier, on a logarithmic scale">
      <!-- Gridlines first so bars sit on top of them -->
      <g transform="translate({LABEL_WIDTH},0)">
        {#each ticks as tick (tick)}
          <line
            x1={x(tick)}
            x2={x(tick)}
            y1="0"
            y2={rows.length * ROW_HEIGHT}
            stroke="var(--color-rule)"
            stroke-width="1" />
          <text
            x={x(tick)}
            y={rows.length * ROW_HEIGHT + 18}
            text-anchor="middle"
            class="num fill-[var(--color-ink-faint)] text-[11px]">
            {fmtTokens(tick)}
          </text>
        {/each}
      </g>

      {#each rows as row, i (row.tier.tier)}
        {@const y = i * ROW_HEIGHT}
        {@const barX = LABEL_WIDTH + x(row.start)}
        {@const barW = Math.max(2, x(row.end) - x(row.start))}
        <g>
          <text
            x={LABEL_WIDTH - 12}
            y={y + ROW_HEIGHT / 2}
            text-anchor="end"
            dominant-baseline="middle"
            class="fill-[var(--color-ink-muted)] text-[13px]">
            {row.tier.tier}
          </text>

          <rect
            x={barX}
            y={y + 9}
            width={barW}
            height={ROW_HEIGHT - 20}
            rx="2"
            fill={row.tier.color}
            fill-opacity="0.85">
            <title>{row.tier.tier} — {row.tier.label}. {row.tier.meaning}</title>
          </rect>

          <text
            x={barX + barW + 8}
            y={y + ROW_HEIGHT / 2}
            dominant-baseline="middle"
            class="num fill-[var(--color-ink-muted)] text-[11px]">
            {fmtTokens(Math.trunc(row.start))}–{fmtTokens(Math.trunc(row.end))}
          </text>
        </g>
      {/each}
    </svg>
  {/if}

  <figcaption class="sr-only">
    Each bar spans the token range for one tier, measured against
    {fmtTokens(effectiveCount)} parameters.
  </figcaption>
</figure>
