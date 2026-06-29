"""Tests for ``care.runtime.run_state`` (TODO §1.2 P2).

The store is best-effort: any malformed file means "no state",
not "raise". Coverage layers:

1. **RunState shape** — to_dict / from_dict round-trip; default
   factories don't share state; frozen.
2. **Atomic save** — the file appears on disk with the expected
   JSON; a second save overwrites cleanly without leaving
   leftover tempfiles; parent directories are created.
3. **load() tolerance** — missing file, malformed JSON,
   schema-version mismatch, payload not a dict, missing required
   fields → all return ``None``.
4. **clear() semantics** — returns bool, idempotent, removes the
   file.
5. **Concurrency** — 16 threads racing to save don't corrupt
   the file (atomic write).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from care.runtime.run_state import (
    DEFAULT_RUN_STATE_PATH,
    SCHEMA_VERSION,
    RunState,
    RunStateStore,
)


def _make_state(run_id: str = "r-1", **overrides) -> RunState:
    base = {
        "run_id": run_id,
        "kind": "carl_execution",
        "label": "Forecast run",
        "started_at": 1_700_000_000.0,
        "payload": {"chain_entity_id": "ent-42"},
    }
    base.update(overrides)
    return RunState(**base)


# ---------------------------------------------------------------------------
# RunState shape
# ---------------------------------------------------------------------------


class TestRunStateShape:
    def test_to_dict_includes_all_fields(self):
        state = _make_state()
        d = state.to_dict()
        assert d["run_id"] == "r-1"
        assert d["kind"] == "carl_execution"
        assert d["label"] == "Forecast run"
        assert d["started_at"] == 1_700_000_000.0
        assert d["payload"] == {"chain_entity_id": "ent-42"}
        assert d["schema_version"] == SCHEMA_VERSION

    def test_from_dict_inverse_of_to_dict(self):
        state = _make_state()
        restored = RunState.from_dict(state.to_dict())
        assert restored == state

    def test_default_payload_is_independent(self):
        a = RunState(run_id="a", kind="x", label="A")
        b = RunState(run_id="b", kind="x", label="B")
        a.payload["mutated"] = True
        assert b.payload == {}

    def test_default_started_at_is_now_ish(self):
        before = 1_000_000.0
        state = RunState(run_id="x", kind="k", label="l")
        # Default factory ran something — sanity check it's a float.
        assert isinstance(state.started_at, float)
        assert state.started_at > before

    def test_frozen(self):
        state = _make_state()
        with pytest.raises(Exception):
            state.run_id = "other"  # type: ignore[misc]

    def test_from_dict_requires_run_id(self):
        with pytest.raises(KeyError):
            RunState.from_dict({"kind": "x", "label": "y"})

    def test_from_dict_coerces_types(self):
        # Numeric started_at as string; payload as None.
        restored = RunState.from_dict(
            {
                "run_id": "r",
                "kind": "x",
                "label": "y",
                "started_at": "12345.0",
                "payload": None,
            }
        )
        assert restored.started_at == 12345.0
        assert restored.payload == {}


# ---------------------------------------------------------------------------
# Save + load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path):
        store = RunStateStore(tmp_path / "state.json")
        state = _make_state()
        store.save(state)
        loaded = store.load()
        assert loaded == state

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c" / "state.json"
        store = RunStateStore(nested)
        store.save(_make_state())
        assert nested.exists()

    def test_save_writes_valid_json(self, tmp_path: Path):
        path = tmp_path / "state.json"
        store = RunStateStore(path)
        store.save(_make_state())
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run_id"] == "r-1"
        assert data["schema_version"] == SCHEMA_VERSION

    def test_save_overwrites_existing(self, tmp_path: Path):
        store = RunStateStore(tmp_path / "state.json")
        store.save(_make_state(run_id="first"))
        store.save(_make_state(run_id="second"))
        loaded = store.load()
        assert loaded is not None
        assert loaded.run_id == "second"

    def test_save_leaves_no_tempfiles(self, tmp_path: Path):
        store = RunStateStore(tmp_path / "state.json")
        store.save(_make_state())
        store.save(_make_state(run_id="updated"))
        leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".run_state-")]
        assert leftover == []

    def test_default_path_constant(self):
        # Pin the documented default so docs / migration scripts
        # can reference it.
        assert str(DEFAULT_RUN_STATE_PATH).endswith("run_state.json")

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        store = RunStateStore("~/state.json")
        assert store.path == tmp_path / "state.json"


# ---------------------------------------------------------------------------
# load() tolerance
# ---------------------------------------------------------------------------


class TestLoadTolerance:
    def test_missing_file_returns_none(self, tmp_path: Path):
        store = RunStateStore(tmp_path / "state.json")
        assert store.load() is None

    def test_malformed_json_returns_none(self, tmp_path: Path):
        path = tmp_path / "state.json"
        path.write_text("{ invalid }", encoding="utf-8")
        store = RunStateStore(path)
        assert store.load() is None

    def test_not_a_dict_returns_none(self, tmp_path: Path):
        path = tmp_path / "state.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        store = RunStateStore(path)
        assert store.load() is None

    def test_version_mismatch_returns_none(self, tmp_path: Path):
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": "r",
                    "kind": "x",
                    "label": "y",
                    "started_at": 1.0,
                    "payload": {},
                    "schema_version": SCHEMA_VERSION + 999,
                }
            ),
            encoding="utf-8",
        )
        store = RunStateStore(path)
        assert store.load() is None

    def test_missing_required_field_returns_none(self, tmp_path: Path):
        path = tmp_path / "state.json"
        # No `run_id` — `from_dict` would raise KeyError; store
        # converts that into a `None`.
        path.write_text(
            json.dumps(
                {
                    "kind": "x",
                    "label": "y",
                    "schema_version": SCHEMA_VERSION,
                }
            ),
            encoding="utf-8",
        )
        store = RunStateStore(path)
        assert store.load() is None


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_returns_true_after_save(self, tmp_path: Path):
        store = RunStateStore(tmp_path / "state.json")
        store.save(_make_state())
        assert store.clear() is True
        assert not (tmp_path / "state.json").exists()

    def test_clear_returns_false_when_missing(self, tmp_path: Path):
        store = RunStateStore(tmp_path / "state.json")
        assert store.clear() is False

    def test_clear_is_idempotent(self, tmp_path: Path):
        store = RunStateStore(tmp_path / "state.json")
        store.save(_make_state())
        assert store.clear() is True
        assert store.clear() is False

    def test_load_after_clear_returns_none(self, tmp_path: Path):
        store = RunStateStore(tmp_path / "state.json")
        store.save(_make_state())
        store.clear()
        assert store.load() is None


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_saves_dont_corrupt(self, tmp_path: Path):
        store = RunStateStore(tmp_path / "state.json")

        def writer(i: int) -> None:
            store.save(_make_state(run_id=f"writer-{i}"))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After the storm, the file is well-formed and matches one
        # of the writes (not a torn mix).
        loaded = store.load()
        assert loaded is not None
        assert loaded.run_id.startswith("writer-")

    def test_save_then_clear_then_save_does_not_race(self, tmp_path: Path):
        store = RunStateStore(tmp_path / "state.json")

        def cycler() -> None:
            for i in range(20):
                store.save(_make_state(run_id=f"c-{i}"))
                store.clear()

        ts = [threading.Thread(target=cycler) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

        # Final state may be present or absent (race-dependent).
        # The contract: no crash, no corruption — load returns
        # `None` or a well-formed state.
        loaded = store.load()
        assert loaded is None or loaded.run_id.startswith("c-")


# ---------------------------------------------------------------------------
# Re-export check
# ---------------------------------------------------------------------------


class TestRuntimeReExport:
    def test_runtime_package_exports_run_state_symbols(self):
        from care.runtime import (
            DEFAULT_RUN_STATE_PATH as exported_default,
        )
        from care.runtime import RunState as ExportedRunState
        from care.runtime import RunStateStore as ExportedRunStateStore

        assert exported_default == DEFAULT_RUN_STATE_PATH
        assert ExportedRunState is RunState
        assert ExportedRunStateStore is RunStateStore
