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

    // SvelteKit defaults to src/routes and src/lib, but src/ is the Python package
    // until it is deleted in the final cleanup. Point the app elsewhere so the two
    // coexist; lib/ already holds the ported engine, so $lib/engine/* resolves to it.
    files: {
      routes: "app/routes",
      lib: "lib",
      appTemplate: "app/app.html",
      errorTemplate: "app/error.html",
      assets: "static"
    }
  }
};
