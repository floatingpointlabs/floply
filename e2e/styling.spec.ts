import { expect, test } from "@playwright/test";

test("the unit select next to a scaled number is not stretched to full width", async ({ page }) => {
  await page.goto("/");
  const wrapper = page.locator("#budget").locator("xpath=..");
  const numberBox = await page.locator("#budget").boundingBox();
  const unitBox = await wrapper.locator("select").boundingBox();

  expect(numberBox, "number input should render").toBeTruthy();
  expect(unitBox, "unit select should render").toBeTruthy();
  // The unit select is a short K/M/B/T picker; if `.control`'s width:100% wins over
  // `w-auto` it ends up as wide as the number field beside it.
  expect(unitBox!.width).toBeLessThan(numberBox!.width);
});

test("figures keep their tabular monospace inside a .control input", async ({ page }) => {
  await page.goto("/");
  // `.control` sets `font-family: inherit`; `.num` must still win, or every numeric
  // field silently loses its tabular figures and jitters as values change.
  const family = await page.locator("#budget").evaluate((el) => getComputedStyle(el).fontFamily);
  expect(family.toLowerCase()).toContain("mono");
});

test("a field hint is announced, not just rendered", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#budget")).toHaveAttribute("aria-describedby", "budget-hint");
  await expect(page.locator("#budget-hint")).toHaveText(/GPU compute/);
});

test("a hint that comes and goes takes its aria reference with it", async ({ page }) => {
  await page.goto("/training-budget");
  await page.locator("#modality").selectOption("Text");

  const precision = page.locator("#precision");
  await expect(precision).not.toHaveAttribute("aria-describedby");

  await precision.selectOption("fp4");
  await expect(precision).toHaveAttribute("aria-describedby", "precision-hint");
  await expect(page.locator("#precision-hint")).toBeVisible();

  await precision.selectOption("bf16");
  await expect(page.locator("#precision-hint")).toHaveCount(0);
  await expect(precision).not.toHaveAttribute("aria-describedby");
});

test("a selected LoRA module pill actually looks selected", async ({ page }) => {
  await page.goto("/minimum-data");
  await page.locator("#training-type").selectOption("Fine-Tuning");
  await page.locator("#ft-method").selectOption("LoRA");

  const on = page.getByRole("button", { name: "q_proj", exact: true });
  const off = page.getByRole("button", { name: "up_proj", exact: true });
  await expect(on).toHaveAttribute("aria-pressed", "true");
  await expect(off).toHaveAttribute("aria-pressed", "false");

  const colourOf = (el: typeof on) => el.evaluate((n) => getComputedStyle(n).color);
  const borderOf = (el: typeof on) => el.evaluate((n) => getComputedStyle(n).borderTopColor);

  expect(await colourOf(on)).not.toBe(await colourOf(off));
  expect(await borderOf(on)).not.toBe(await borderOf(off));
});
