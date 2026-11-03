<script lang="ts">
  import { onNavigate } from "$app/navigation";
  import { page } from "$app/state";
  import Logo from "$lib/components/Logo.svelte";
  import RegionPicker from "$lib/components/RegionPicker.svelte";
  import "../app.css";
  import type { LayoutProps } from "./$types";

  let { children, data }: LayoutProps = $props();

  const pricingUsable = $derived(data.catalog.provenance.source !== "unavailable");

  // Icons are Lucide's (ISC), inlined: four glyphs don't justify a dependency.
  const links = [
    { href: "/", label: "Minimum data", icon: "database" },
    { href: "/budget-optimizer", label: "Budget optimizer", icon: "trending-up" },
    { href: "/training-budget", label: "Training budget", icon: "calculator" },
    { href: "/methodology", label: "Methodology", icon: "book-open" }
  ] as const;

  const current = $derived(page.url.pathname);

  // The top bar and footer hold still while the page cross-fades (see ::view-transition-* in app.css).
  onNavigate((navigation) => {
    if (
      !document.startViewTransition ||
      navigation.from?.url.pathname === navigation.to?.url.pathname
    )
      return;
    return new Promise((resolve) => {
      document.startViewTransition(async () => {
        resolve();
        await navigation.complete;
      });
    });
  });
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

{#snippet icon(name: (typeof links)[number]["icon"])}
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-linecap="round"
    stroke-linejoin="round">
    {#if name === "database"}
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5V19A9 3 0 0 0 21 19V5" />
      <path d="M3 12A9 3 0 0 0 21 12" />
    {:else if name === "trending-up"}
      <path d="M16 7h6v6" />
      <path d="m22 7-8.5 8.5-5-5L2 17" />
    {:else if name === "calculator"}
      <rect width="16" height="20" x="4" y="2" rx="2" />
      <line x1="8" x2="16" y1="6" y2="6" />
      <line x1="16" x2="16" y1="14" y2="18" />
      <path d="M16 10h.01M12 10h.01M8 10h.01M12 14h.01M8 14h.01M12 18h.01M8 18h.01" />
    {:else}
      <path d="M12 5v16" />
      <path
        d="M20.001 19A2 2 0 0022 17V5a2 2 0 00-1.999-2L16 3.002A5 5 0 0012 5a5 5 0 00-4-2H4a2 2 0 00-2 2v12a2 2 0 001.999 2H8a5 5 0 014 2 5 5 0 014-2z" />
    {/if}
  </svg>
{/snippet}

<div class="flex min-h-screen flex-col">
  <header class="topbar">
    <div
      class="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-8 gap-y-2 px-6 py-3">
      <!-- The wordmark splits mid-word: a newline before the span renders as "flop ly." -->
      <!-- prettier-ignore -->
      <a href="/" class="wordmark brand">flop<span class="text-[var(--color-accent-lit)]">ly</span></a>

      <!-- Below md the links take their own row and scroll sideways rather than wrap. -->
      <nav class="links" aria-label="Sections">
        <ul class="flex gap-1">
          {#each links as link (link.href)}
            <li>
              <a
                href={link.href}
                aria-current={current === link.href ? "page" : undefined}
                class="nav-link">
                {@render icon(link.icon)}
                {link.label}
              </a>
            </li>
          {/each}
        </ul>
      </nav>
    </div>
  </header>

  <div
    class="mx-auto flex w-full max-w-6xl flex-wrap items-start justify-between gap-x-6 gap-y-3 px-6 pt-6">
    <p class="max-w-[60ch] pt-1.5 text-[var(--color-ink-muted)]">
      What a training run costs, before you commit the budget.
    </p>
    <RegionPicker regions={data.regions} provenance={data.catalog.provenance} />
  </div>

  <main class="page mx-auto w-full max-w-6xl grow px-6 pt-8 pb-16">
    {#if pricingUsable}
      {@render children()}
    {:else}
      <section class="max-w-2xl">
        <h1>Pricing unavailable for {data.catalog.region}.</h1>
        <p class="mt-3 text-[var(--color-ink-muted)]">
          {data.catalog.provenance.last_error}
        </p>
        <p class="mt-3 text-[var(--color-ink-muted)]">
          Floply reads live AWS pricing and has no bundled fallback. Check that credentials are
          configured and that the IAM policy allows these read-only actions:
        </p>
        <ul class="mt-2 list-disc pl-5 text-[var(--color-ink-muted)]">
          <li><code>pricing:GetProducts</code></li>
          <li><code>ec2:DescribeInstanceTypes</code></li>
          <li><code>ec2:DescribeInstanceTypeOfferings</code></li>
        </ul>
      </section>
    {/if}
  </main>

  <footer class="footer">
    <div
      class="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-8 gap-y-4 px-6 py-6">
      <a href="https://floatingpointlabs.ca" class="maker">
        <Logo idPrefix="footer" size={24} />
        A Floating Point Labs project
      </a>
      <p class="max-w-[60ch] text-sm leading-relaxed text-[var(--color-ink-faint)]">
        Estimates only. FLOPs-based modelling assumes ideal scaling; real runs vary with data
        loading, checkpointing overhead, and cluster utilisation.
      </p>
    </div>
  </footer>
</div>

<style>
  /* Chrome sits on the Well, one step below the page, with a Soft edge. */
  .topbar {
    background: var(--color-well);
    border-bottom: 1px solid var(--color-soft);
    position: sticky;
    top: 0;
    z-index: 20;
  }

  .footer {
    background: var(--color-well);
    border-top: 1px solid var(--color-soft);
  }

  .brand:hover {
    text-decoration: none;
  }

  .maker {
    align-items: center;
    color: var(--color-ink-muted);
    display: flex;
    font-size: 0.875rem;
    gap: 0.5rem;
  }

  .maker:hover {
    color: var(--color-ink);
    text-decoration: none;
  }

  .page {
    view-transition-name: page;
  }

  @media (max-width: 767px) {
    .links {
      margin-inline: -1.5rem;
      mask-image: linear-gradient(to right, #000 85%, transparent);
      overflow-x: auto;
      padding: 0 1.5rem 0.25rem;
      width: calc(100% + 3rem);
    }
  }
</style>
