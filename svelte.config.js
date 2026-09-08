import adapter from "@sveltejs/adapter-node";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import('@sveltejs/kit').Config} */
export default {
  preprocess: vitePreprocess(),
  kit: {
    // adapter-node, not adapter-static: the Umami analytics vars are read at request
    // time so one image can be deployed with different env per environment — the same
    // property the Streamlit deployment has today.
    adapter: adapter(),

    // Non-default locations: the app grew up beside a Python package that owned src/,
    // and the paths stayed once it was removed. Renaming to SvelteKit's defaults would
    // touch every import for no behavioural gain.
    files: {
      routes: "app/routes",
      lib: "lib",
      appTemplate: "app/app.html",
      errorTemplate: "app/error.html",
      assets: "static"
    }
  }
};
