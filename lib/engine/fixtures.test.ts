import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { DISPATCH, solveCase } from "./fixtureDispatch";
import type { Case } from "./fixtureDispatch";

const FIXTURES = join(process.cwd(), "fixtures");
const load = (name: string) =>
  JSON.parse(readFileSync(join(FIXTURES, name), "utf8")) as Record<string, any>;

const REL_TOL = 1e-12;

/**
 * Compared with toEqual rather than approx(): approx only walks the keys present in the
 * expectation, so it cannot see an extra key the engine started returning. These two groups
 * are whole-object snapshots where that matters.
 */
const EXACT_GROUPS = new Set(["tiers", "model_definitions"]);

function approx(got: unknown, want: unknown, path = ""): void {
  if (typeof want === "number" && typeof got === "number") {
    if (Number.isNaN(want) || Number.isNaN(got)) {
      expect(Number.isNaN(got), path).toBe(Number.isNaN(want));
      return;
    }
    if (want === 0) {
      expect(Math.abs(got), path).toBeLessThanOrEqual(REL_TOL);
      return;
    }
    expect(
      Math.abs(got - want) / Math.abs(want),
      `${path} (got ${got}, want ${want})`
    ).toBeLessThanOrEqual(REL_TOL);
    return;
  }
  if (Array.isArray(want)) {
    expect(Array.isArray(got), path).toBe(true);
    const g = got as unknown[];
    expect(g.length, path).toBe(want.length);
    want.forEach((w, i) => approx(g[i], w, `${path}[${i}]`));
    return;
  }
  if (want !== null && typeof want === "object") {
    expect(got !== null && typeof got === "object", path).toBe(true);
    const g = got as Record<string, unknown>;
    for (const [k, w] of Object.entries(want as Record<string, unknown>)) {
      approx(g[k], w, `${path}.${k}`);
    }
    return;
  }
  expect(got, path).toEqual(want);
}

describe("engine.json", () => {
  const data = load("engine.json");

  for (const [group, cases] of Object.entries(data)) {
    if (group === "_meta") continue;

    it(`${group} (${(cases as Case[]).length} cases)`, () => {
      (cases as Case[]).forEach((c, i) => {
        const where = `${group}[${i}]`;

        if (EXACT_GROUPS.has(group)) {
          expect(DISPATCH[group](c), where).toEqual(c.expect);
          return;
        }

        if (c.raises) {
          expect(() => DISPATCH[group](c), where).toThrow();
          return;
        }

        approx(DISPATCH[group](c), c.expect, where);
      });
    });
  }
});

describe("budget_optimizer.json", () => {
  const data = load("budget_optimizer.json");

  it(`solve (${data.solve.length} scenarios)`, () => {
    for (const c of data.solve as Case[]) {
      approx(solveCase(c.args, c.selection ?? {}), c.expect, c.name ?? "");
    }
  });

  for (const group of ["derive_schedule_defaults", "logspace", "budget_curve_n"]) {
    it(`${group} (${data[group].length} cases)`, () => {
      (data[group] as Case[]).forEach((c, i) => {
        approx(DISPATCH[group](c), c.expect, `${group}[${i}]`);
      });
    });
  }
});
