import { defineConfig, devices } from "@playwright/test";

/**
 * Smoke tests only. These run against a *production build*, not the dev server — the
 * failures worth catching here (broken hydration, a missing runtime dependency in the
 * bundle) are exactly the ones dev mode papers over.
 */
export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:4321",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "pnpm build && PORT=4321 node build/index.js",
    port: 4321,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
