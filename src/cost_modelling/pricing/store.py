"""Pluggable key/value backing for the pricing cache.

Nothing above this module knows whether bytes came from a file, memory, or
(later) Redis. Two rules the file implementation exists to guarantee:

- A reader never observes a half-written payload.
- A failed write never destroys a good previous value.

If Redis is ever added, note that its TTL *deletes* keys, which would break the
serve-stale-when-AWS-is-down policy. Staleness is computed from the payload's
own ``fetched_at`` (see cache.py); any Redis TTL should be a long GC backstop
only, and persistence (AOF or RDB) must be enabled or a restart takes the sole
pricing source with it.
"""

import json
import logging
import os
import tempfile
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

log = logging.getLogger(__name__)


@runtime_checkable
class CacheStore(Protocol):
    """Minimal contract for a pricing cache backend."""

    def read(self, key: str) -> dict[str, Any] | None:
        """Return the stored payload, or None when absent or unreadable."""
        ...

    def write(self, key: str, payload: dict[str, Any]) -> bool:
        """Persist a payload atomically. Returns False if the write was skipped."""
        ...

    def lock(self, key: str) -> AbstractContextManager[bool]:
        """Hold an exclusive lock, yielding False if another holder has it."""
        ...

    @property
    def description(self) -> str:
        """Human-readable location, for the provenance panel."""
        ...


class InMemoryStore:
    """Process-local store. Used by tests and as the last-resort fallback."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def read(self, key: str) -> dict[str, Any] | None:
        payload = self._data.get(key)
        return json.loads(json.dumps(payload)) if payload is not None else None

    def write(self, key: str, payload: dict[str, Any]) -> bool:
        self._data[key] = json.loads(json.dumps(payload))
        return True

    @contextmanager
    def lock(self, key: str) -> Iterator[bool]:
        yield True

    @property
    def description(self) -> str:
        return "in-memory (not persisted)"


class FileStore:
    """JSON-file store, one file per key, written atomically."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def read(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        try:
            with open(path) as f:
                payload = json.load(f)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            # Corrupt cache is a cache miss, never a crash. Keep the file for
            # diagnosis rather than deleting evidence.
            log.warning("Discarding unreadable cache entry %s: %s", path, exc)
            try:
                path.replace(path.with_suffix(".json.corrupt"))
            except OSError:
                pass
            return None

        if not isinstance(payload, dict):
            log.warning("Cache entry %s is not an object; ignoring", path)
            return None
        return payload

    def write(self, key: str, payload: dict[str, Any]) -> bool:
        target = self._path(key)
        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self.directory, prefix=f".{key}-", suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)      # atomic within a filesystem
            tmp_path = None
            return True
        except OSError as exc:
            log.warning("Could not write cache entry %s: %s", target, exc)
            return False
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @contextmanager
    def lock(self, key: str) -> Iterator[bool]:
        """Best-effort exclusive lock, for a concurrent refresh_cli run.

        Yields False when another holder has it, so the caller keeps its
        in-memory result instead of racing the write.
        """
        try:
            import fcntl
        except ImportError:                   # non-POSIX; single-container anyway
            yield True
            return

        lock_path = self.directory / f"{key}.lock"
        handle = None
        try:
            handle = open(lock_path, "w")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
            yield True
        except OSError as exc:
            log.warning("Could not acquire lock %s: %s", lock_path, exc)
            yield True
        finally:
            if handle is not None:
                handle.close()

    @property
    def description(self) -> str:
        return str(self.directory)


def _candidate_dirs(configured: Path) -> list[Path]:
    candidates = [configured]
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        candidates.append(Path(xdg) / "floply")
    candidates.append(Path.home() / ".cache" / "floply")
    return candidates


def build_store(settings) -> CacheStore:
    """Best available store for the configured backend.

    Falls back through the candidate directories, then to memory. A read-only
    filesystem degrades to "fetch works, nothing persists" rather than failing.
    """
    if settings.cache_backend != "file":
        log.warning(
            "Unknown FLOPLY_CACHE_BACKEND %r; falling back to file.",
            settings.cache_backend,
        )

    for directory in _candidate_dirs(settings.cache_dir):
        try:
            store = FileStore(directory)
            probe = directory / ".write-test"
            probe.write_text("")
            probe.unlink()
            return store
        except OSError:
            continue

    log.warning(
        "No writable cache directory (tried %s); pricing will not persist "
        "across restarts.",
        ", ".join(str(d) for d in _candidate_dirs(settings.cache_dir)),
    )
    return InMemoryStore()
