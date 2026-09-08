import { env } from "$env/dynamic/private";

/**
 * Analytics config is read at *request* time, not build time.
 *
 * $env/dynamic/private reads the running process's environment, so one image can be
 * deployed to several environments with different analytics — the property the Streamlit
 * deployment had. Using $env/static/public (or NEXT_PUBLIC_-style inlining) would bake
 * whatever the build machine happened to have.
 */
export const load = () => ({
  umami:
    env.UMAMI_URL && env.UMAMI_WEBSITE_ID
      ? { url: env.UMAMI_URL, websiteId: env.UMAMI_WEBSITE_ID }
      : null
});
