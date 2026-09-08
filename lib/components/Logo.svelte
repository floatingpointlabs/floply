<script lang="ts">
  /**
   * Gradient ids are namespaced because the mark can render more than once per page and
   * duplicate ids are invalid — worse, `url(#…)` then resolves to whichever parsed first.
   */
  interface Props {
    idPrefix: string;
    size?: number;
  }

  let { idPrefix, size = 32 }: Props = $props();

  const wave = $derived(`${idPrefix}-wave`);
  const dot = $derived(`${idPrefix}-dot`);
</script>

<svg
  aria-hidden="true"
  focusable="false"
  width={size}
  height={size}
  viewBox="0 0 64 64"
  xmlns="http://www.w3.org/2000/svg"
  style="overflow: visible"
>
  <defs>
    <linearGradient id={wave} x1="0%" x2="100%" y1="0%" y2="0%">
      <stop offset="0%" stop-color="var(--color-indigo)" />
      <stop offset="100%" stop-color="var(--color-violet)" />
    </linearGradient>
    <radialGradient cx="38%" cy="32%" id={dot} r="65%">
      <stop offset="0%" stop-color="var(--color-violet-pale)" />
      <stop offset="100%" stop-color="var(--color-purple-deep)" />
    </radialGradient>
  </defs>

  <path
    d="M14 41q8.2-18.8 17.6-11.1t18 .3"
    fill="none"
    stroke="url(#{wave})"
    stroke-linejoin="round"
    stroke-width="8.8"
  />
  <circle cx="40.8" cy="22.2" r="5.1" fill="url(#{dot})" />
</svg>
