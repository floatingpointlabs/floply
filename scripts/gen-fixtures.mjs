/**
 * Regenerates fixtures/*.json from the TypeScript engine.
 *
 * Run: pnpm gen-fixtures [outDir]
 *
 * The engine is TypeScript with extensionless relative imports, which Node cannot resolve
 * on its own, so the modules are loaded through Vite — the same resolution the app and the
 * test suite use, which means the generator cannot drift from what ships.
 *
 * Regenerating is a deliberate act: the fixtures are the only thing standing between a
 * behaviour change and a silent one, so read the diff before committing it.
 */

import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outDir = resolve(process.argv[2] ?? join(ROOT, "fixtures"));

function gitSha() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT }).toString().trim();
  } catch {
    return "unknown";
  }
}

const server = await createServer({
  root: ROOT,
  configFile: false,
  server: { middlewareMode: true },
  appType: "custom",
  logLevel: "error"
});

try {
  const { buildEngine, buildBudgetOptimizer } = await server.ssrLoadModule("/scripts/fixtures.ts");

  const meta = {
    generated_from: gitSha(),
    node: process.versions.node,
    note: "Golden fixtures for the engine. Regenerate with `pnpm gen-fixtures`, and read the diff."
  };

  mkdirSync(outDir, { recursive: true });

  for (const [filename, payload] of [
    ["engine.json", buildEngine()],
    ["budget_optimizer.json", buildBudgetOptimizer()]
  ]) {
    const path = join(outDir, filename);
    writeFileSync(path, JSON.stringify({ _meta: meta, ...payload }, null, 2) + "\n");

    const groups = Object.keys(payload).length;
    const cases = Object.values(payload).reduce((a, g) => a + g.length, 0);
    console.log(`  ${relative(ROOT, path)}: ${groups} groups, ${cases} cases`);
  }
} finally {
  await server.close();
}
