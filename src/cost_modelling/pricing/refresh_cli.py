"""Prime or refresh the pricing cache from the command line.

    python -m src.cost_modelling.pricing.refresh_cli --help

Useful for warming a fresh volume before first use, for a cron/systemd timer
in deployments that prefer scheduled refreshes over the lazy TTL, and for
inspecting exactly what the filters return without going through the UI.
"""

import argparse
import logging
import sys

from src.cost_modelling.pricing import refresh
from src.cost_modelling.pricing.settings import get_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refresh_cli",
        description="Fetch AWS pricing and hardware specs into the local cache.",
    )
    parser.add_argument(
        "--regions", help="Comma-separated regions to refresh (default: configured list).")
    parser.add_argument(
        "--force", action="store_true",
        help="Refresh even when the cached entries are still fresh.")
    parser.add_argument(
        "--probe-only", action="store_true",
        help="Check credentials and permissions, then exit without fetching.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    if args.regions:
        regions = tuple(r.strip() for r in args.regions.split(",") if r.strip())
    else:
        regions = settings.regions

    probe = refresh.probe(settings)
    print(f"credentials: {'ok' if probe.ok else 'FAILED'} — {probe.detail}")
    if probe.missing_action:
        print(f"  missing IAM action: {probe.missing_action}")
    if not probe.ok:
        return 2
    if args.probe_only:
        return 0

    result = refresh.refresh_all(regions=regions, force=args.force, settings=settings)

    print(f"cache: {result.store_description}")
    print(f"specs: {len(result.specs.instances) if result.specs else 0} GPU instance types")
    for region in regions:
        entry = result.regions.get(region)
        if entry is None:
            print(f"  {region}: not refreshed")
            continue
        print(f"  {region}: {len(entry.prices)} prices, {len(entry.storage)} storage classes")
        for instance_type, price in sorted(entry.prices.items()):
            print(f"      {instance_type:<20} ${price:>9,.2f}/hr")

    if result.quarantined:
        print("\nquarantined (no curated FLOPs — add to src/data/gpu_hardware.yaml):")
        for instance_type, reason in sorted(result.quarantined.items()):
            print(f"  {instance_type}: {reason}")

    if result.warnings:
        print("\nwarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
