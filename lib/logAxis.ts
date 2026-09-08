import { scaleLog } from "d3-scale";

export interface LogAxis {
  scale: (v: number) => number;
  ticks: number[];
  /** False when the domain is degenerate — a log scale cannot take 0 or a zero width. */
  ok: boolean;
}

export function logAxis(
  domain: [number, number],
  range: [number, number],
): LogAxis {
  const ok = domain[0] > 0 && domain[1] > domain[0];
  const scale = scaleLog().domain(domain).range(range).clamp(true);

  // Decade ticks only. d3's log ticks include every minor tick (1..9 per decade), which
  // over a 3-4 decade span is ~30 overlapping labels.
  const first = Math.ceil(Math.log10(domain[0]));
  const last = Math.floor(Math.log10(domain[1]));
  const ticks: number[] = [];
  if (ok && Number.isFinite(first) && Number.isFinite(last)) {
    // log10(0) is -Infinity, and incrementing -Infinity never terminates.
    for (let e = first; e <= last; e++) ticks.push(10 ** e);
  }

  return { scale, ticks, ok };
}
