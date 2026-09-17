import { env } from "$env/dynamic/private";
import { AWS_REGION_LABELS, getCatalog, getSettings, listRegions } from "$lib/server/pricing";
import type { LayoutServerLoad } from "./$types";

const REGION_COOKIE = "floply_region";

export const load: LayoutServerLoad = async ({ url, cookies }) => {
  const settings = getSettings();
  const regions = listRegions(settings);

  const requested = url.searchParams.get("region") ?? cookies.get(REGION_COOKIE);
  const region = requested && regions.includes(requested) ? requested : settings.defaultRegion;

  if (cookies.get(REGION_COOKIE) !== region) {
    cookies.set(REGION_COOKIE, region, { path: "/", sameSite: "lax" });
  }

  return {
    catalog: await getCatalog(region, settings),
    regions: regions.map((code) => ({ code, label: AWS_REGION_LABELS[code] ?? "" })),
    umami:
      env.UMAMI_URL && env.UMAMI_WEBSITE_ID
        ? { url: env.UMAMI_URL, websiteId: env.UMAMI_WEBSITE_ID }
        : null
  };
};
