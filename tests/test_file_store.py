"""FileStore durability guarantees.

With no bundled fallback, this file *is* the availability layer. Two properties
matter above all: a reader never sees a half-written payload, and a failed
write never destroys a good previous value.
"""

import json
import os
from pathlib import Path

import pytest

from src.cost_modelling.pricing.store import FileStore, InMemoryStore, build_store
from src.cost_modelling.pricing.settings import build_settings


@pytest.fixture
def store(tmp_path) -> FileStore:
    return FileStore(tmp_path / "cache")


# -- Round trip -------------------------------------------------------------

def test_write_then_read(store):
    assert store.write("catalog_v1_us-east-1", {"prices": {"p5.48xlarge": 55.04}})
    assert store.read("catalog_v1_us-east-1") == {"prices": {"p5.48xlarge": 55.04}}


def test_missing_key_reads_as_none(store):
    assert store.read("never_written") is None


def test_creates_its_directory(tmp_path):
    target = tmp_path / "deep" / "nested" / "cache"
    FileStore(target).write("k", {"a": 1})
    assert (target / "k.json").exists()


# -- Corruption -------------------------------------------------------------

def test_corrupt_json_is_a_miss_not_a_crash(store):
    (store.directory / "broken.json").write_text("{not valid json")
    assert store.read("broken") is None


def test_corrupt_file_is_kept_for_diagnosis(store):
    (store.directory / "broken.json").write_text("{not valid json")
    store.read("broken")
    assert (store.directory / "broken.json.corrupt").exists()
    assert not (store.directory / "broken.json").exists()


def test_non_object_payload_is_rejected(store):
    (store.directory / "listy.json").write_text("[1, 2, 3]")
    assert store.read("listy") is None


# -- Atomicity --------------------------------------------------------------

def test_write_leaves_no_temp_files(store):
    store.write("k", {"a": 1})
    assert [p.name for p in store.directory.iterdir()] == ["k.json"]


def test_failed_serialization_leaves_no_temp_file_and_keeps_the_old_value(store):
    store.write("k", {"good": 1})

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        store.write("k", {"bad": Unserializable()})

    assert store.read("k") == {"good": 1}
    leftovers = [p.name for p in store.directory.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_overwrite_is_atomic_via_replace(store):
    store.write("k", {"version": 1})
    store.write("k", {"version": 2})
    assert store.read("k") == {"version": 2}


def test_unwritable_directory_returns_false_rather_than_raising(tmp_path):
    directory = tmp_path / "readonly"
    store = FileStore(directory)
    os.chmod(directory, 0o500)
    try:
        assert store.write("k", {"a": 1}) is False
    finally:
        os.chmod(directory, 0o700)


# -- Locking ----------------------------------------------------------------

def test_lock_is_acquired_when_free(store):
    with store.lock("k") as acquired:
        assert acquired


def test_lock_is_exclusive_across_handles(store):
    with store.lock("k") as first:
        assert first
        with store.lock("k") as second:
            assert second is False, "a second holder must not get the same lock"


def test_lock_is_released_on_exit(store):
    with store.lock("k"):
        pass
    with store.lock("k") as acquired:
        assert acquired


# -- InMemoryStore ----------------------------------------------------------

def test_in_memory_round_trip():
    store = InMemoryStore()
    store.write("k", {"a": 1})
    assert store.read("k") == {"a": 1}


def test_in_memory_store_deep_copies():
    """Callers must not be able to mutate cached state by holding a reference."""
    store = InMemoryStore()
    payload = {"prices": {"p5": 1.0}}
    store.write("k", payload)
    payload["prices"]["p5"] = 999.0
    assert store.read("k") == {"prices": {"p5": 1.0}}

    fetched = store.read("k")
    fetched["prices"]["p5"] = 123.0
    assert store.read("k") == {"prices": {"p5": 1.0}}


# -- Backend selection ------------------------------------------------------

def test_build_store_uses_the_configured_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOPLY_CACHE_DIR", str(tmp_path / "floply"))
    store = build_store(build_settings())
    assert isinstance(store, FileStore)
    assert store.directory == tmp_path / "floply"


def test_build_store_falls_back_to_memory_when_nothing_is_writable(monkeypatch, tmp_path):
    """A read-only filesystem must degrade to 'nothing persists', not a crash."""
    monkeypatch.setattr(
        "src.cost_modelling.pricing.store._candidate_dirs", lambda _: [])
    store = build_store(build_settings())
    assert isinstance(store, InMemoryStore)
    assert "not persisted" in store.description


def test_cache_dir_is_not_inside_src(monkeypatch):
    """A cache under src/ would be baked into the Docker image by COPY src."""
    settings = build_settings()
    assert "src" not in settings.cache_dir.parts
