<script lang="ts">
  import { page } from "$app/state";
  import Logo from "$lib/components/Logo.svelte";
  import RegionPicker from "$lib/components/RegionPicker.svelte";
  import "../app.css";
  import type { LayoutProps } from "./$types";

  let { children, data }: LayoutProps = $props();

  const pricingUsable = $derived(data.catalog.provenance.source !== "unavailable");

  const tabs = [
    { href: "/", label: "Data needed" },
    { href: "/budget-optimizer", label: "Budget optimizer" },
    { href: "/training-budget", label: "Project cost" },
    { href: "/methodology", label: "How it works" }
  ];

  const current = $derived(page.url.pathname);
</script>

<svelte:head>
  <meta property="og:site_name" content="Floply" />
  <meta property="og:type" content="website" />
  <meta property="og:title" content="Floply — estimate what it costs to train an ML model" />
  <meta
    property="og:description"
    content="Estimate the data, GPU time, memory and total cost of training an ML model." />
  <meta name="twitter:card" content="summary" />

  {#if data.umami}
    <script async defer src={data.umami.url} data-website-id={data.umami.websiteId}></script>
  {/if}
</svelte:head>

<div class="flex min-h-screen flex-col">
  <header class="border-b border-[var(--color-rule)]">
    <div class="mx-auto flex max-w-6xl flex-wrap items-baseline gap-x-6 gap-y-2 px-6 py-6">
      <!-- The wordmark splits mid-word: a newline before the span renders as
           "flop ly." -->
      <!-- prettier-ignore -->
      <a
        href="/"
        class="text-[1.6rem] font-semibold lowercase leading-none tracking-[-0.025em]
               text-[var(--color-ink)] no-underline">
        flop<span class="text-[var(--color-indigo)]">ly</span>
      </a>
      <p class="text-sm text-[var(--color-ink-muted)]">
        Estimate what it costs to train an ML model.
      </p>
      <div class="ms-auto">
        <RegionPicker regions={data.regions} provenance={data.catalog.provenance} />
      </div>
    </div>

    <nav class="mx-auto max-w-6xl px-6" aria-label="Sections">
      <ul class="-mb-px flex flex-wrap gap-x-1">
        {#each tabs as tab (tab.href)}
          {@const active = current === tab.href}
          <li>
            <a
              href={tab.href}
              aria-current={active ? "page" : undefined}
              class="relative block px-3 py-2.5 text-sm no-underline transition-colors
                     {active
                ? 'text-[var(--color-ink)]'
                : 'text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]'}">
              {tab.label}
              {#if active}
                <span class="absolute inset-x-0 bottom-0 block h-0.5 bg-[var(--color-indigo)]">
                </span>
              {/if}
            </a>
          </li>
        {/each}
      </ul>
    </nav>
  </header>

  <main class="mx-auto w-full max-w-6xl grow px-6 py-10">
    {#if pricingUsable}
      {@render children()}
    {:else}
      <section class="max-w-2xl">
        <h1 class="text-xl font-semibold">Can't load prices for {data.catalog.region}.</h1>
        <p class="mt-3 text-sm text-[var(--color-ink-muted)]">
          {data.catalog.provenance.last_error}
        </p>
        <p class="mt-3 text-sm text-[var(--color-ink-muted)]">
          Floply uses live AWS prices and has no offline copy. Check that AWS credentials are set
          and that the IAM policy allows these read-only actions:
        </p>
        <ul class="mt-2 list-disc pl-5 text-sm text-[var(--color-ink-muted)]">
          <li><code>pricing:GetProducts</code></li>
          <li><code>ec2:DescribeInstanceTypes</code></li>
          <li><code>ec2:DescribeInstanceTypeOfferings</code></li>
        </ul>
      </section>
    {/if}
  </main>

  <footer class="border-t border-[var(--color-rule)]">
    <div class="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-7">
      <a
        href="https://floatingpointlabs.ca"
        class="flex items-center gap-2.5 text-sm text-[var(--color-ink-muted)] no-underline
               transition-colors hover:text-[var(--color-ink)]">
        <Logo idPrefix="footer" size={30} />
        <span class="tracking-[-0.01em]">
          A <span class="font-semibold text-[var(--color-ink)]">Floating Point Labs</span>
          project
        </span>
      </a>

      <p class="max-w-[52ch] text-xs leading-relaxed text-[var(--color-ink-faint)]">
        Estimates only. They assume ideal conditions, so real runs usually take longer and cost
        more.
      </p>
    </div>
  </footer>
</div>
