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

  // The sidebar holds still while the page cross-fades (see ::view-transition-* in app.css).
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

<div class="shell">
  <nav class="sidebar" aria-label="Floply">
    <!-- The wordmark splits mid-word: a newline before the span renders as "flop ly." -->
    <!-- prettier-ignore -->
    <a href="/" class="wordmark brand">flop<span class="text-[var(--color-accent-lit)]">ly</span></a>

    <div class="region">
      <RegionPicker regions={data.regions} provenance={data.catalog.provenance} />
    </div>

    <ul class="links">
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

    <a href="https://floatingpointlabs.ca" class="maker">
      <Logo idPrefix="nav" size={20} />
      A Floating Point Labs project
    </a>
  </nav>

  <main>
    <div class="page">
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
    </div>

    <footer class="mt-16 max-w-[66ch] text-sm leading-relaxed text-[var(--color-ink-faint)]">
      What a training run costs, before you commit the budget. Estimates only: FLOPs-based modelling
      assumes ideal scaling; real runs vary with data loading, checkpointing overhead, and cluster
      utilisation.
    </footer>
  </main>
</div>

<style>
  /* The sidebar shell: a sticky Well column beside the page. 13rem rather than the
     product's 11, so "Budget optimizer" and the region fit. */
  .shell {
    display: grid;
    grid-template-columns: 13rem minmax(0, 1fr);
    min-height: 100vh;
  }

  .sidebar {
    background: var(--color-well);
    border-right: 1px solid var(--color-soft);
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    height: 100vh;
    padding: 1.25rem 0.75rem;
    position: sticky;
    top: 0;
    /* Above the page, so the region notes popover can overlap it. */
    z-index: 20;
  }

  .brand {
    padding: 0 0.5rem 1rem;
  }

  .brand:hover {
    text-decoration: none;
  }

  /* Like the product's active-agent switch: which prices are in use, under the wordmark. */
  .region {
    margin-bottom: 1rem;
    padding-inline: 0.25rem;
  }

  .links {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .maker {
    align-items: center;
    color: var(--color-ink-faint);
    display: flex;
    font-size: 0.875rem;
    gap: 0.5rem;
    line-height: 1.3;
    margin-top: auto;
    padding: 0.25rem 0.5rem;
  }

  .maker:hover {
    color: var(--color-ink);
    text-decoration: none;
  }

  main {
    min-width: 0;
    padding: 1.5rem clamp(1rem, 3vw, 2.5rem) 2.5rem;
  }

  .page {
    view-transition-name: page;
  }

  /* Under 720px the sidebar becomes a wrapping bar over the page. */
  @media (max-width: 720px) {
    .shell {
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: auto 1fr;
    }

    .sidebar {
      align-items: center;
      border-bottom: 1px solid var(--color-soft);
      border-right: 0;
      flex-flow: row wrap;
      gap: 0.5rem;
      height: auto;
      padding: 0.75rem;
      position: relative;
    }

    .brand {
      padding: 0.25rem 0.5rem;
    }

    .region {
      margin: 0 0 0 auto;
    }

    .links {
      flex-flow: row wrap;
      flex-basis: 100%;
    }

    .maker {
      margin-top: 0;
    }
  }
</style>
