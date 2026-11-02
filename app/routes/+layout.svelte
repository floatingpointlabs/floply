<script lang="ts">
  import { page } from "$app/state";
  import Logo from "$lib/components/Logo.svelte";
  import RegionPicker from "$lib/components/RegionPicker.svelte";
  import Sky from "$lib/components/Sky.svelte";
  import "../app.css";
  import type { LayoutProps } from "./$types";

  let { children, data }: LayoutProps = $props();

  const pricingUsable = $derived(data.catalog.provenance.source !== "unavailable");

  const tabs = [
    { href: "/", label: "Minimum data" },
    { href: "/budget-optimizer", label: "Budget optimizer" },
    { href: "/training-budget", label: "Training budget" },
    { href: "/methodology", label: "Methodology" }
  ];

  const current = $derived(page.url.pathname);
</script>

<svelte:head>
  <meta property="og:site_name" content="Floply" />
  <meta property="og:type" content="website" />
  <meta
    property="og:title"
    content="Floply — what a training run costs, before you commit the budget" />
  <meta
    property="og:description"
    content="Estimate GPU hours, wall-clock time, memory and total cost for an ML training project — compute, storage, sweeps and ablations." />
  <meta name="twitter:card" content="summary" />

  {#if data.umami}
    <script async defer src={data.umami.url} data-website-id={data.umami.websiteId}></script>
  {/if}
</svelte:head>

<Sky />

<div class="relative z-[2] flex min-h-screen flex-col">
  <header class="site-nav">
    <div
      class="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-8 gap-y-1 px-6 py-3">
      <a href="/" class="flex items-center gap-2.5 py-1 no-underline">
        <Logo idPrefix="nav" size={36} />
        <!-- The wordmark splits mid-word: a newline before the span renders as
             "flop ly." -->
        <!-- prettier-ignore -->
        <span class="wordmark">flop<span class="text-[var(--color-indigo)]">ly</span></span>
      </a>

      <!-- Below md the tabs take their own row and scroll sideways rather than wrap. -->
      <nav
        class="-mx-6 w-[calc(100%+3rem)] overflow-x-auto px-6 pb-1 md:mx-0 md:w-auto md:p-0 max-md:[mask-image:linear-gradient(to_right,#000_85%,transparent)]"
        aria-label="Sections">
        <ul class="flex gap-x-6">
          {#each tabs as tab (tab.href)}
            <li>
              <a
                href={tab.href}
                aria-current={current === tab.href ? "page" : undefined}
                class="nav-link">
                {tab.label}
              </a>
            </li>
          {/each}
        </ul>
      </nav>
    </div>
  </header>

  <div
    class="mx-auto flex w-full max-w-6xl flex-wrap items-start justify-between gap-x-6 gap-y-3 px-6 pt-6">
    <p class="max-w-[60ch] pt-1 text-sm text-[var(--color-ink-muted)]">
      What a training run costs, before you commit the budget.
    </p>
    <RegionPicker regions={data.regions} provenance={data.catalog.provenance} />
  </div>

  <main class="mx-auto w-full max-w-6xl grow px-6 pt-10 pb-20">
    {#if pricingUsable}
      {@render children()}
    {:else}
      <section class="max-w-2xl">
        <h1 class="page-title">Pricing unavailable for {data.catalog.region}.</h1>
        <p class="mt-3 text-sm text-[var(--color-ink-muted)]">
          {data.catalog.provenance.last_error}
        </p>
        <p class="mt-3 text-sm text-[var(--color-ink-muted)]">
          Floply reads live AWS pricing and has no bundled fallback. Check that credentials are
          configured and that the IAM policy allows these read-only actions:
        </p>
        <ul class="mt-2 list-disc pl-5 text-sm text-[var(--color-ink-muted)]">
          <li><code>pricing:GetProducts</code></li>
          <li><code>ec2:DescribeInstanceTypes</code></li>
          <li><code>ec2:DescribeInstanceTypeOfferings</code></li>
        </ul>
      </section>
    {/if}
  </main>

  <footer
    class="border-t border-[var(--color-groove)] bg-[var(--color-well)] shadow-[0_-1px_0_var(--color-edge)]">
    <div class="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-7">
      <a
        href="https://floatingpointlabs.ca"
        class="flex items-center gap-2.5 text-sm text-[var(--color-ink-muted)] no-underline
               transition-colors hover:text-[var(--color-ink)]">
        <Logo idPrefix="footer" size={30} />
        <span class="tracking-[-0.01em]">
          A
          <span class="font-semibold text-[var(--color-ink)]">
            Floating Point <span class="text-[var(--color-indigo)]">Labs</span>
          </span>
          project
        </span>
      </a>

      <p class="max-w-[52ch] text-xs leading-relaxed text-[var(--color-ink-faint)]">
        Estimates only. FLOPs-based modelling assumes ideal scaling; real runs vary with data
        loading, checkpointing overhead, and cluster utilisation.
      </p>
    </div>
  </footer>
</div>
