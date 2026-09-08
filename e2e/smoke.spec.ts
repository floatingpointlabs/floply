import { expect, test, type Page } from "@playwright/test";

const ROUTES = ["/", "/minimum-data", "/training-budget", "/methodology"];

/** Read a readout by its label. Figure.svelte renders <dt>label</dt><dd>value</dd>. */
function figure(page: Page, label: string) {
  return page.getByText(label, { exact: true }).locator("xpath=following-sibling::dd[1]");
}

test("every route loads and hydrates without console errors", async ({ page }) => {
  // Listeners are registered once and the buffer cleared per route; registering inside
  // the loop would stack a new pair on every iteration and attribute errors oddly.
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));

  for (const route of ROUTES) {
    errors.length = 0;
    const response = await page.goto(route);
    expect(response?.status(), route).toBe(200);
    await expect(page.locator("h1")).toBeVisible();
    // Hydration is async; give it a beat to throw if it is going to.
    await page.waitForLoadState("networkidle");
    expect(errors, `console errors on ${route}`).toEqual([]);
  }
});

test("budget optimizer shows the values verified against Streamlit", async ({ page }) => {
  await page.goto("/");
  // The same baseline captured from the Streamlit app at the start of the migration.
  await expect(figure(page, "Max model size (params)")).toHaveText("4.09B");
  await expect(figure(page, "Dataset")).toHaveText("79.43B tokens");
  await expect(figure(page, "Total")).toHaveText("$10,000");
  await expect(figure(page, "Budget used")).toHaveText("100%");
});

test("the dataset slider recomputes the answer", async ({ page }) => {
  await page.goto("/");
  const dataset = figure(page, "Dataset");
  const before = await dataset.innerText();

  await page.locator("#d-slider").fill("12");
  await expect(dataset).not.toHaveText(before);
  await expect(dataset).toHaveText("1.00T tokens");

  const used = await figure(page, "Budget used").innerText();
  expect(Number.parseInt(used, 10)).toBeLessThanOrEqual(100);
});

test("schedule overrides survive a schedule edit but reset on a setup change", async ({ page }) => {
  await page.goto("/");
  const epochs = page.locator("#epochs");
  await expect(epochs).toHaveValue("1"); // derived default for pre-training

  await epochs.fill("7");
  await expect(epochs).toHaveValue("7");

  await page.locator("#months").fill("9");
  await expect(epochs).toHaveValue("7");

  await page.locator("#modality").selectOption("Vision (ViT / CLIP)");
  await expect(epochs).toHaveValue("1");
});

test("training budget gates each section until the one above is answered", async ({ page }) => {
  await page.goto("/training-budget");
  await expect(page.getByText("Choose a modality and dataset size to continue.")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Total project cost" })).toBeHidden();

  await page.locator("#modality").selectOption("Text");

  await expect(page.getByRole("heading", { name: "Total project cost" })).toBeVisible();
  await expect(figure(page, "Tokens per sample")).toHaveText("520");
});

test("training budget breaks compute down by run type and recaps the config", async ({ page }) => {
  await page.goto("/training-budget");
  await page.locator("#modality").selectOption("Text");

  await expect(page.getByRole("heading", { name: "Compute by run type" })).toBeVisible();
  await expect(figure(page, "Full runs")).toHaveText("$21,770");
  await expect(figure(page, "HP trials")).toHaveText("$0");

  // Ten trials at 30% of a run cost 3x the model you actually keep — the point of the
  // whole section, and previously invisible.
  await page.locator("#hp").fill("10");
  await expect(figure(page, "HP trials")).toHaveText("$65,309");

  await expect(page.getByRole("heading", { name: "Run configuration" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "HP tuning trials" })).toBeVisible();
});

test("minimum data swaps tier tables between pre-training and LoRA", async ({ page }) => {
  await page.goto("/minimum-data");
  // Scoped to the chart: these tier names also appear as readout labels below it.
  const chart = page.getByRole("img", { name: /Token requirements/ });

  await expect(chart.getByText("Compute-optimal", { exact: true })).toBeVisible();

  await page.locator("#training-type").selectOption("Fine-Tuning");
  await page.locator("#ft-method").selectOption("LoRA");

  await expect(chart.getByText("Sweet spot", { exact: true })).toBeVisible();
  await expect(chart.getByText("Compute-optimal", { exact: true })).toBeHidden();
});
