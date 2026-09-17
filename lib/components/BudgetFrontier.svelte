<script lang="ts">
  import { logAxis } from "$lib/logAxis";
  import { CHART_COLORS } from "$lib/engine/constants";
  import { fmtTokens } from "$lib/engine/formatting";

  interface Props {
    dRange: number[];
    budgetCurve: number[];
    optimalRatio: number;
    optimalLabel: string;
    adapterParams: number | null;
    optimum: { d: number; n: number } | null;
    selection: { d: number; n: number };
    yAxisTitle: string;
    height?: number;
  }

  let {
    dRange,
    budgetCurve,
    optimalRatio,
    optimalLabel,
    adapterParams,
    optimum,
    selection,
    yAxisTitle,
    height = 380
  }: Props = $props();

  const M = { top: 16, right: 18, bottom: 42, left: 66 };

  let width = $state(760);

  const optimalLine = $derived(dRange.map((d) => d / optimalRatio));

  const positive = (xs: number[]) => xs.filter((v) => Number.isFinite(v) && v > 0);

  // logspace is strictly ascending, so the ends are the domain — no need to scan, and
  // no 240-argument spread with its hard argument-count ceiling.
  const xDomain = $derived<[number, number]>([dRange[0] ?? 0, dRange[dRange.length - 1] ?? 0]);

  const yDomain = $derived.by(() => {
    const all = [
      ...positive(budgetCurve),
      ...positive(optimalLine),
      ...positive([adapterParams ?? 0, optimum?.n ?? 0, selection.n])
    ];
    if (!all.length) return [1, 10] as [number, number];
    return [Math.min(...all), Math.max(...all)] as [number, number];
  });

  const innerW = $derived(Math.max(120, width - M.left - M.right));
  const innerH = $derived(height - M.top - M.bottom);

  const xAxis = $derived(logAxis(xDomain, [0, innerW]));
  const yAxis = $derived(logAxis(yDomain, [innerH, 0]));
  const x = $derived(xAxis.scale);
  const y = $derived(yAxis.scale);
  const xTicks = $derived(xAxis.ticks);
  const yTicks = $derived(yAxis.ticks);
  const renderable = $derived(dRange.length > 1 && xAxis.ok && yAxis.ok);

  /** One pass: map+filter would allocate two throwaway 240-element arrays per polyline. */
  const path = (ys: number[]) => {
    const points: string[] = [];
    for (let i = 0; i < dRange.length; i++) {
      const v = ys[i];
      if (v > 0 && Number.isFinite(v)) points.push(`${x(dRange[i])},${y(v)}`);
    }
    return points.join(" ");
  };
</script>

<figure class="m-0" bind:clientWidth={width}>
  {#if !renderable}
    <p class="text-sm text-[var(--color-ink-muted)]">Enter a budget to plot the frontier.</p>
  {:else}
    <svg
      {width}
      {height}
      role="img"
      aria-label="Affordable model size against dataset size, on logarithmic axes">
      <g transform="translate({M.left},{M.top})">
        {#each xTicks as t (t)}
          <line x1={x(t)} x2={x(t)} y1="0" y2={innerH} stroke="var(--color-rule)" />
          <text
            x={x(t)}
            y={innerH + 18}
            text-anchor="middle"
            class="num fill-[var(--color-ink-faint)] text-[11px]">
            {fmtTokens(t)}
          </text>
        {/each}
        {#each yTicks as t (t)}
          <line x1="0" x2={innerW} y1={y(t)} y2={y(t)} stroke="var(--color-rule)" />
          <text
            x="-10"
            y={y(t)}
            text-anchor="end"
            dominant-baseline="middle"
            class="num fill-[var(--color-ink-faint)] text-[11px]">
            {fmtTokens(t)}
          </text>
        {/each}

        <polyline
          points={path(optimalLine)}
          fill="none"
          stroke={CHART_COLORS.success}
          stroke-width="2"
          stroke-dasharray="6 4" />

        <polyline
          points={path(budgetCurve)}
          fill="none"
          stroke={CHART_COLORS.compute}
          stroke-width="2.5" />

        {#if adapterParams && adapterParams > 0}
          <line
            x1="0"
            x2={innerW}
            y1={y(adapterParams)}
            y2={y(adapterParams)}
            stroke={CHART_COLORS.checkpoint}
            stroke-width="1"
            stroke-dasharray="2 3" />
        {/if}

        {#if optimum && optimum.d > 0 && optimum.n > 0}
          <path
            transform="translate({x(optimum.d)},{y(optimum.n)})"
            d="M0,-9 L2.6,-2.9 L9,-2.6 L4,1.6 L5.6,8 L0,4.5 L-5.6,8 L-4,1.6 L-9,-2.6 L-2.6,-2.9 Z"
            fill={CHART_COLORS.checkpoint}>
            <title>
              Budget-optimal: {fmtTokens(Math.trunc(optimum.d))} tokens, {fmtTokens(
                Math.trunc(optimum.n)
              )}
            </title>
          </path>
        {/if}

        {#if selection.d > 0 && selection.n > 0}
          <circle
            cx={x(selection.d)}
            cy={y(selection.n)}
            r="6"
            fill="var(--color-ink)"
            stroke={CHART_COLORS.compute}
            stroke-width="2">
            <title>Your selection: {fmtTokens(selection.d)} tokens, {fmtTokens(selection.n)}</title>
          </circle>
        {/if}

        <text
          x={innerW / 2}
          y={innerH + 36}
          text-anchor="middle"
          class="fill-[var(--color-ink-muted)] text-[12px]">
          Dataset size (tokens)
        </text>
        <text
          transform="translate({-M.left + 14},{innerH / 2}) rotate(-90)"
          text-anchor="middle"
          class="fill-[var(--color-ink-muted)] text-[12px]">
          {yAxisTitle}
        </text>
      </g>
    </svg>

    <figcaption class="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs text-[var(--color-ink-muted)]">
      <span class="flex items-center gap-1.5">
        <span class="inline-block h-0.5 w-5" style="background:{CHART_COLORS.compute}"></span>
        What the budget buys
      </span>
      <span class="flex items-center gap-1.5">
        <span
          class="inline-block h-0.5 w-5"
          style="background:repeating-linear-gradient(90deg,{CHART_COLORS.success} 0 6px,transparent 6px 10px)">
        </span>
        {optimalLabel}
      </span>
      {#if adapterParams && adapterParams > 0}
        <span class="flex items-center gap-1.5">
          <span
            class="inline-block h-0.5 w-5"
            style="background:repeating-linear-gradient(90deg,{CHART_COLORS.checkpoint} 0 2px,transparent 2px 5px)">
          </span>
          Adapter size
        </span>
      {/if}
    </figcaption>
  {/if}
</figure>
