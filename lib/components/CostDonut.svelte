<script lang="ts">
  import { money } from "$lib/engine/formatting";

  interface Segment {
    label: string;
    value: number;
    color: string;
  }

  interface Props {
    segments: Segment[];
    total: number;
    size?: number;
  }

  let { segments, total, size = 190 }: Props = $props();

  const OUTER = $derived(size / 2);
  const INNER = $derived(OUTER * 0.62);

  const sum = $derived(segments.reduce((a, s) => a + s.value, 0));

  /**
   * Annular sector between two angles, measured clockwise from 12 o'clock.
   */
  function arc(
    from: number,
    to: number,
    outer: number,
    inner: number,
    centre: number,
  ): string {
    const point = (r: number, a: number) => [
      centre + r * Math.sin(a),
      centre - r * Math.cos(a),
    ];
    const large = to - from > Math.PI ? 1 : 0;
    const [x1, y1] = point(outer, from);
    const [x2, y2] = point(outer, to);
    const [x3, y3] = point(inner, to);
    const [x4, y4] = point(inner, from);
    return [
      `M ${x1} ${y1}`,
      `A ${outer} ${outer} 0 ${large} 1 ${x2} ${y2}`,
      `L ${x3} ${y3}`,
      `A ${inner} ${inner} 0 ${large} 0 ${x4} ${y4}`,
      "Z",
    ].join(" ");
  }

  const wedges = $derived.by(() => {
    if (sum <= 0) return [];
    let angle = 0;
    return segments.map((s) => {
      const sweep = (s.value / sum) * Math.PI * 2;
      // A full circle can't be drawn as one arc; nudge it just short of 360°.
      const end = angle + Math.min(sweep, Math.PI * 2 - 1e-6);
      const d = arc(angle, end, OUTER, INNER, OUTER);
      angle = end;
      return { ...s, d, share: s.value / sum };
    });
  });
</script>

<div class="flex flex-wrap items-center gap-6">
  {#if wedges.length}
    <svg width={size} height={size} role="img" aria-label="Cost distribution">
      {#each wedges as w (w.label)}
        <path d={w.d} fill={w.color} fill-opacity="0.9">
          <title>{w.label}: {money(w.value)} ({(w.share * 100).toFixed(1)}%)</title>
        </path>
      {/each}
      <text
        x={OUTER}
        y={OUTER - 2}
        text-anchor="middle"
        class="num fill-[var(--color-ink)] text-[1.15rem]"
      >
        {money(total)}
      </text>
      <text
        x={OUTER}
        y={OUTER + 16}
        text-anchor="middle"
        class="fill-[var(--color-ink-faint)] text-[11px]"
      >
        total
      </text>
    </svg>
  {/if}

  <dl class="min-w-[13rem] flex-1 space-y-2.5">
    {#each segments as s (s.label)}
      <div class="flex items-baseline justify-between gap-4">
        <dt class="flex items-center gap-2 text-sm text-[var(--color-ink-muted)]">
          <span class="h-2.5 w-2.5 shrink-0 rounded-[1px]" style="background:{s.color}"></span>
          {s.label}
        </dt>
        <dd class="num text-sm">
          {money(s.value)}
          <span class="ml-1.5 text-[var(--color-ink-faint)]">
            {sum > 0 ? ((s.value / sum) * 100).toFixed(1) : "0.0"}%
          </span>
        </dd>
      </div>
    {/each}
  </dl>
</div>
