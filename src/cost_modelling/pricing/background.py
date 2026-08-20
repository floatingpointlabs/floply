"""Background refresh with failure backoff.

Page rendering must never wait on the network when there is something to serve.
A cold cache blocks once (there is no alternative); a stale cache is served
immediately and refreshed on a daemon thread, so the update lands on the next
Streamlit rerun — which any interaction triggers.

Threads here must never touch ``st.*``: they have no ScriptRunContext.
"""

import logging
import threading
import time
from dataclasses import dataclass

from src.cost_modelling.pricing import refresh
from src.cost_modelling.pricing.settings import PricingSettings, get_settings
from src.cost_modelling.pricing.store import CacheStore, build_store

log = logging.getLogger(__name__)


@dataclass
class _Attempt:
    at: float
    ok: bool
    error: str | None = None


_lock = threading.RLock()
_in_flight: set[str] = set()
_last_attempt: dict[str, _Attempt] = {}


def last_error(region: str) -> str | None:
    """Error from the most recent failed refresh of this region, if any."""
    with _lock:
        attempt = _last_attempt.get(region)
    return None if attempt is None or attempt.ok else attempt.error


def _in_backoff(region: str, settings: PricingSettings) -> bool:
    """Whether a recent failure should suppress another attempt.

    Without this, every Streamlit rerun would retry a failing endpoint —
    turning one outage into a request storm.
    """
    with _lock:
        attempt = _last_attempt.get(region)
    if attempt is None or attempt.ok:
        return False
    return (time.monotonic() - attempt.at) < settings.retry_backoff_s


def _record(region: str, ok: bool, error: str | None = None) -> None:
    with _lock:
        _last_attempt[region] = _Attempt(at=time.monotonic(), ok=ok, error=error)
        _in_flight.discard(region)


def _run(region: str, settings: PricingSettings, store: CacheStore) -> None:
    from src.cost_modelling.pricing import catalog

    try:
        specs, specs_warnings = refresh.refresh_specs(store, settings, force=True)
        if specs is None:
            _record(region, False, "; ".join(specs_warnings) or "specs unavailable")
            return

        usable, _ = refresh.partition_by_curation(specs)
        pricing, warnings = refresh.refresh_region(
            store, region, usable, settings, force=True)
        if pricing is None:
            _record(region, False, "; ".join(warnings) or "pricing unavailable")
            return

        _record(region, True)
        # Drop the memo so the next rerun picks up the refreshed numbers.
        catalog.invalidate(region)
        log.info("Background refresh of %s complete", region)
    except Exception as exc:                       # noqa: BLE001 - thread must not die loudly
        log.warning("Background refresh of %s failed: %s", region, exc)
        _record(region, False, f"{type(exc).__name__}: {exc}")


def schedule_refresh(
    region: str,
    settings: PricingSettings | None = None,
    store: CacheStore | None = None,
) -> bool:
    """Start a background refresh unless one is running or we are backing off.

    Returns True if a thread was started.
    """
    settings = settings or get_settings()

    with _lock:
        if region in _in_flight:
            return False
        if _in_backoff(region, settings):
            return False
        _in_flight.add(region)

    store = store or build_store(settings)
    thread = threading.Thread(
        target=_run,
        args=(region, settings, store),
        name=f"floply-refresh-{region}",
        daemon=True,
    )
    thread.start()
    return True


def reset() -> None:
    """Clear in-flight and backoff state. For tests."""
    with _lock:
        _in_flight.clear()
        _last_attempt.clear()
