"""Tests for `care.runtime.session_persistence` (TODO §3 P1)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from care.runtime.session_artifacts import (
    SessionArtifactStore,
    replay_into,
)
from care.runtime.session_persistence import (
    PersistenceHandle,
    SessionInfo,
    attach_persistence,
    list_sessions,
    load_session,
    make_session_id,
    session_path,
    sessions_dir,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestMakeSessionId:
    def test_returns_well_formed_id(self) -> None:
        sid = make_session_id()
        # YYYYMMDD-HHMMSS-<6 hex>
        date, hms, suffix = sid.split("-")
        assert len(date) == 8 and date.isdigit()
        assert len(hms) == 6 and hms.isdigit()
        assert len(suffix) == 6
        int(suffix, 16)  # parses as hex

    def test_two_calls_yield_distinct_ids(self) -> None:
        a = make_session_id()
        b = make_session_id()
        assert a != b


class TestSessionPath:
    def test_resolves_under_cache_dir(self, tmp_path: Path) -> None:
        path = session_path("abc-123", cache_dir=tmp_path)
        assert path == tmp_path / "sessions" / "abc-123.jsonl"

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            session_path("../etc/passwd", cache_dir=tmp_path)

    def test_rejects_empty_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            session_path("", cache_dir=tmp_path)

    def test_accepts_underscore_and_dash(self, tmp_path: Path) -> None:
        path = session_path("my_session-1", cache_dir=tmp_path)
        assert path.name == "my_session-1.jsonl"


# ---------------------------------------------------------------------------
# attach_persistence
# ---------------------------------------------------------------------------


class TestAttachPersistence:
    def test_attach_writes_on_every_append(
        self, tmp_path: Path,
    ) -> None:
        store = SessionArtifactStore()
        path = tmp_path / "session.jsonl"
        handle = attach_persistence(store, path)
        try:
            store.append_chain(
                chain={"name": "demo"},
                title="A",
                summary="first",
            )
            store.append_chain(
                chain={"name": "demo2"},
                title="B",
                summary="second",
            )
            assert path.is_file()
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 2
        finally:
            handle.detach()

    def test_attach_is_idempotent(self, tmp_path: Path) -> None:
        store = SessionArtifactStore()
        path = tmp_path / "session.jsonl"
        handle = attach_persistence(store, path)
        # Second attach is a no-op.
        handle.attach()
        try:
            store.append_chain(
                chain={"k": "v"}, title="t", summary="s",
            )
            # One listener → one dump → JSONL has only this artifact.
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 1
        finally:
            handle.detach()

    def test_detach_stops_persistence(self, tmp_path: Path) -> None:
        store = SessionArtifactStore()
        path = tmp_path / "session.jsonl"
        handle = attach_persistence(store, path)
        store.append_chain(chain={"a": 1}, title="A", summary="x")
        handle.detach()
        # Subsequent appends shouldn't update the file.
        mtime_before = path.stat().st_mtime
        time.sleep(0.01)
        store.append_chain(chain={"b": 2}, title="B", summary="y")
        assert path.stat().st_mtime == mtime_before
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_flush_writes_even_when_detached(
        self, tmp_path: Path,
    ) -> None:
        store = SessionArtifactStore()
        store.append_chain(chain={"a": 1}, title="A", summary="x")
        path = tmp_path / "session.jsonl"
        handle = PersistenceHandle(store=store, path=path)
        # Never attached; flush still writes.
        handle.flush()
        assert path.is_file()
        assert path.read_text().count("\n") == 1

    def test_mark_saved_triggers_redump(self, tmp_path: Path) -> None:
        store = SessionArtifactStore()
        path = tmp_path / "session.jsonl"
        handle = attach_persistence(store, path)
        try:
            art = store.append_chain(
                chain={"k": "v"}, title="t", summary="s",
            )
            store.mark_saved(art.id, memory_entity_id="mem-42")
            payload = path.read_text()
            assert "mem-42" in payload
            assert "saved_to_memory" in payload
        finally:
            handle.detach()


# ---------------------------------------------------------------------------
# list + load
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_empty_when_dir_missing(self, tmp_path: Path) -> None:
        # tmp_path itself exists but `sessions/` underneath doesn't.
        assert list_sessions(cache_dir=tmp_path) == []

    def test_returns_known_sessions_newest_first(
        self, tmp_path: Path,
    ) -> None:
        sdir = sessions_dir(cache_dir=tmp_path)
        sdir.mkdir(parents=True)
        older = sdir / "alpha.jsonl"
        newer = sdir / "beta.jsonl"
        older.write_text('{"id": "1", "kind": "chain"}\n')
        newer.write_text('{"id": "2", "kind": "chain"}\n')
        # Force a deterministic mtime ordering.
        import os

        os.utime(older, (1000.0, 1000.0))
        os.utime(newer, (2000.0, 2000.0))
        rows = list_sessions(cache_dir=tmp_path)
        assert [r.session_id for r in rows] == ["beta", "alpha"]
        assert isinstance(rows[0], SessionInfo)
        assert rows[0].path == newer
        assert rows[0].mtime == 2000.0

    def test_skips_non_jsonl_files(self, tmp_path: Path) -> None:
        sdir = sessions_dir(cache_dir=tmp_path)
        sdir.mkdir(parents=True)
        (sdir / "x.txt").write_text("not a session")
        (sdir / "y.jsonl").write_text("")
        rows = list_sessions(cache_dir=tmp_path)
        assert [r.session_id for r in rows] == ["y"]


class TestLoadSession:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = SessionArtifactStore()
        path = session_path("roundtrip", cache_dir=tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = attach_persistence(store, path)
        try:
            store.append_chain(
                chain={"name": "demo"}, title="A", summary="first",
            )
            store.append_chain(
                chain={"name": "demo2"}, title="B", summary="second",
            )
        finally:
            handle.detach()

        # Hydrate a fresh store.
        loaded = load_session("roundtrip", cache_dir=tmp_path)
        assert [a.title for a in loaded] == ["A", "B"]

        rehydrated = SessionArtifactStore()
        replay_into(rehydrated, loaded)
        assert len(rehydrated) == 2
        titles = [a.title for a in rehydrated.list_artifacts(
            newest_first=False,
        )]
        assert titles == ["A", "B"]

    def test_missing_session_returns_empty(
        self, tmp_path: Path,
    ) -> None:
        assert load_session("never-existed", cache_dir=tmp_path) == []
