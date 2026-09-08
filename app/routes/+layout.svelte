<script lang="ts">
  import { page } from "$app/state";
  import Logo from "$lib/components/Logo.svelte";
  import "../app.css";
  import type { LayoutProps } from "./$types";

  let { children, data }: LayoutProps = $props();

  const tabs = [
    { href: "/", label: "Budget optimizer" },
    { href: "/minimum-data", label: "Minimum data" },
    { href: "/training-budget", label: "Training budget" },
    { href: "/methodology", label: "Methodology" },
  ];

  const current = $derived(page.url.pathname);
</script>

<svelte:head>
  <meta property="og:site_name" content="Floply" />
  <meta property="og:type" content="website" />
  <meta
    property="og:title"
    content="Floply — what a training run costs, before you commit the budget"
  />
  <meta
    property="og:description"
    content="Estimate GPU hours, wall-clock time, memory and total cost for an ML training project — compute, storage, sweeps and ablations."
  />
  <meta name="twitter:card" content="summary" />

  {#if data.umami}
    <script async defer src={data.umami.url} data-website-id={data.umami.websiteId}
    ></script>
  {/if}
</svelte:head>

<div class="flex min-h-screen flex-col">
  <header class="border-b border-[var(--color-rule)]">
    <div class="mx-auto flex max-w-6xl flex-wrap items-baseline gap-x-6 gap-y-2 px-6 py-6">
      <a
        href="/"
        class="font-[family-name:var(--font-display)] text-[1.6rem] font-bold lowercase
               leading-none tracking-[-0.03em] no-underline"
      >
        flop<span class="brand-gradient-text">ly.</span>
      </a>
      <p class="text-sm text-[var(--color-ink-muted)]">
        What a training run costs, before you commit the budget.
      </p>
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
                : 'text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]'}"
            >
              {tab.label}
              {#if active}
                <span
                  class="brand-gradient absolute inset-x-0 bottom-0 block h-0.5 rounded-full"
                ></span>
              {/if}
            </a>
          </li>
        {/each}
      </ul>
    </nav>
  </header>

  <main class="mx-auto w-full max-w-6xl grow px-6 py-10">
    {@render children()}
  </main>

  <footer class="border-t border-[var(--color-rule)]">
    <div
      class="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-7"
    >
      <a
        href="https://floatingpointlabs.github.io"
        class="flex items-center gap-2.5 text-sm text-[var(--color-ink-muted)] no-underline
               transition-colors hover:text-[var(--color-ink)]"
      >
        <Logo idPrefix="footer" size={30} />
        <span class="font-[family-name:var(--font-display)] lowercase tracking-[-0.02em]">
          A <span class="text-[var(--color-ink)]">Floating Point Labs</span> project
        </span>
      </a>

      <p class="max-w-[52ch] text-xs leading-relaxed text-[var(--color-ink-faint)]">
        Estimates only. FLOPs-based modelling assumes ideal scaling; real runs vary with
        data loading, checkpointing overhead, and cluster utilisation.
      </p>
    </div>
  </footer>
</div>
