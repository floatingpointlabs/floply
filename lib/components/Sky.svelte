<script lang="ts">
  /**
   * The FPL night sky (website repo, Sky.astro): a violet dot field, a seeded star canvas and
   * a top wash, fixed behind the page, plus the aurora at the top of the document. Same seed
   * as the website, so both skies are the same sky.
   */

  const COLOURS = ["#dfe4ff", "#dfe4ff", "#eef1ff", "#c9d2ff", "#ffe3cc", "#ffd2b0"];

  let canvas: HTMLCanvasElement;

  // mulberry32: the sky is a place, not an effect, so it never reshuffles.
  function seeded(seed: number) {
    let a = seed;
    return () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  $effect(() => {
    const context = canvas.getContext("2d");
    if (!context) return;

    const draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = window.innerWidth;
      const height = window.innerHeight;

      canvas.width = width * dpr;
      canvas.height = height * dpr;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);

      const random = seeded(20260911);
      const count = Math.round((width * height) / 7000);

      for (let i = 0; i < count; i++) {
        const x = random() * width;
        const y = random() * height;
        const magnitude = random() ** 3;

        context.globalAlpha = 0.16 + magnitude * 0.54;
        context.fillStyle = COLOURS[Math.floor(random() * COLOURS.length)];
        context.beginPath();
        context.arc(x, y, 0.35 + magnitude * 1.4, 0, Math.PI * 2);
        context.fill();
      }
    };

    draw();

    let pending = 0;
    const onResize = () => {
      window.clearTimeout(pending);
      pending = window.setTimeout(draw, 150);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  });
</script>

<div aria-hidden="true" class="sky">
  <div class="sky__dots"></div>
  <canvas bind:this={canvas} class="sky__stars"></canvas>
  <div class="sky__wash"></div>
</div>

<div aria-hidden="true" class="aurora"></div>
