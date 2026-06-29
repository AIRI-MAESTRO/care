"""Tests for `care.runtime.session_artifacts` (TODO §3 P0).

Pure data-layer tests — no Textual, no Memory, no network. The
store is in-memory + thread-safe and these tests confirm every
documented surface:

* Construction / append (one path per ``append_*`` helper).
* mark_saved transitions an entry's `saved_to_memory` +
  `memory_entity_id` and emits a listener event.
* forget / clear shrink the store correctly.
* Snapshots are stable (frozen dataclass; listing returns a
  copy so the caller can mutate freely).
* Listener fan-out swallows exceptions so one bad subscriber
  doesn't break the chain.
* JSON-Lines dump / load round-trip preserves every field.
* Functional `dump_jsonl(store, path)` alias matches the bound
  method.
* Convenience `replay_into` rejects a non-empty store.
"""

from __future__ import annotations

import json
import threading

import pytest

from care.runtime.session_artifacts import (
    SessionArtifact,
    SessionArtifactStore,
    SessionArtifactStoreError,
    dump_jsonl,
    load_jsonl_artifacts,
    replay_into,
)


class TestAppend:
    def test_append_chain_returns_frozen_snapshot(self):
        store = SessionArtifactStore()
        artifact = store.append_chain(
            chain={"steps": []},
            title="weather agent",
            summary="3-step weather flow",
            origin={"turn_index": 1},
        )
        assert artifact.kind == "chain"
        assert artifact.title == "weather agent"
        assert artifact.payload == {"steps": []}
        assert artifact.origin == {"turn_index": 1}
        assert artifact.saved_to_memory is False
        assert artifact.memory_entity_id is None
        assert isinstance(artifact, SessionArtifact)
        # Frozen — direct mutation should raise.
        with pytest.raises(Exception):
            artifact.title = "mutated"  # type: ignore[misc]

    def test_append_each_kind_helper(self):
        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c", summary="s")
        store.append_stage_payload(stage="critique", payload={"ok": True})
        store.append_tool_output(tool="ls", output="a\nb")
        store.append_dataset_row(row={"input": "x", "expected": "y"})
        store.append_synthesised_answer(answer="The answer is 42.\nMore.")
        kinds = {a.kind for a in store.list_artifacts()}
        assert kinds == {
            "chain", "stage_payload", "tool_output",
            "dataset_row", "synthesised_answer",
        }

    def test_append_unknown_kind_rejected(self):
        store = SessionArtifactStore()
        with pytest.raises(SessionArtifactStoreError):
            store.append(
                kind="bogus",  # type: ignore[arg-type]
                title="x", summary="x", payload={},
            )

    def test_stage_origin_carries_stage_label(self):
        store = SessionArtifactStore()
        a = store.append_stage_payload(stage="dag", payload={"nodes": 5})
        assert a.origin["stage"] == "dag"
        assert "Agent chain generator dag" in a.title

    def test_synthesised_summary_is_first_line(self):
        store = SessionArtifactStore()
        a = store.append_synthesised_answer(
            answer="The capital of France is Paris.\nWith population 2.1M.",
        )
        assert a.summary.startswith("The capital of France")
        assert "\n" not in a.summary

    def test_append_chain_carries_tags(self):
        """§3 P3 — `append_chain(tags=...)` lands the cleaned
        tuple on `SessionArtifact.tags`."""
        store = SessionArtifactStore()
        a = store.append_chain(
            chain={}, title="t", summary="s",
            tags=("ml", "urgent", "production"),
        )
        assert a.tags == ("ml", "urgent", "production")

    def test_append_tags_dedupes_and_strips(self):
        store = SessionArtifactStore()
        a = store.append_chain(
            chain={}, title="t", summary="s",
            tags=(" ml ", "ml", "", "urgent", "  ", "urgent"),
        )
        # Whitespace stripped, empties dropped, dupes removed.
        assert a.tags == ("ml", "urgent")

    def test_append_default_tags_empty_tuple(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="t", summary="s")
        assert a.tags == ()

    def test_to_dict_includes_tags(self):
        store = SessionArtifactStore()
        a = store.append_chain(
            chain={}, title="t", summary="s",
            tags=("alpha", "beta"),
        )
        out = a.to_dict()
        assert out["tags"] == ["alpha", "beta"]

    def test_load_jsonl_restores_tags(self, tmp_path):
        from care.runtime.session_artifacts import (
            load_jsonl_artifacts,
        )

        store = SessionArtifactStore()
        store.append_chain(
            chain={}, title="t", summary="s",
            tags=("alpha", "beta"),
        )
        dump_path = tmp_path / "dump.jsonl"
        store.dump_jsonl(dump_path)
        loaded = load_jsonl_artifacts(dump_path)
        assert len(loaded) == 1
        assert loaded[0].tags == ("alpha", "beta")

    def test_ids_are_unique(self):
        store = SessionArtifactStore()
        for _ in range(30):
            store.append_chain(chain={}, title="t", summary="s")
        ids = [a.id for a in store.list_artifacts()]
        assert len(set(ids)) == len(ids)


class TestQueries:
    def _seed(self) -> SessionArtifactStore:
        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c1", summary="s")
        store.append_tool_output(tool="grep", output="found")
        c2 = store.append_chain(chain={"steps": [1]}, title="c2", summary="s")
        store.mark_saved(c2.id, memory_entity_id="ENT-99")
        return store

    def test_list_newest_first(self):
        store = self._seed()
        rows = store.list_artifacts()
        # Newest first → c2 (last appended) leads.
        assert rows[0].title == "c2"
        assert rows[-1].title == "c1"

    def test_list_kind_filter(self):
        store = self._seed()
        chains = store.list_artifacts(kind="chain")
        assert {a.title for a in chains} == {"c1", "c2"}
        tools = store.list_artifacts(kind="tool_output")
        assert {a.payload for a in tools} == {"found"}

    def test_unsaved_only_returns_pending(self):
        store = self._seed()
        rows = store.unsaved()
        assert all(not a.saved_to_memory for a in rows)
        # The seeded chain "c1" + the tool helper which titles
        # itself "Tool: grep" — both unsaved. "c2" was marked
        # saved so it's excluded.
        assert {a.title for a in rows} == {"c1", "Tool: grep"}
        assert all(a.title != "c2" for a in rows)

    def test_counts_projection(self):
        store = self._seed()
        c = store.counts()
        assert c["total"] == 3
        assert c["saved"] == 1
        assert c["unsaved"] == 2
        assert c["kind:chain"] == 2
        assert c["kind:tool_output"] == 1


class TestMutation:
    def test_mark_saved_updates_snapshot(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c", summary="s")
        assert not a.saved_to_memory
        updated = store.mark_saved(a.id, memory_entity_id="ENT-7")
        assert updated.saved_to_memory is True
        assert updated.memory_entity_id == "ENT-7"
        # Subsequent lookup also sees the new state.
        again = store.get(a.id)
        assert again.saved_to_memory is True

    def test_mark_saved_rejects_unknown_id(self):
        store = SessionArtifactStore()
        with pytest.raises(SessionArtifactStoreError):
            store.mark_saved("does-not-exist", memory_entity_id="X")

    def test_mark_saved_requires_entity_id(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c", summary="s")
        with pytest.raises(SessionArtifactStoreError):
            store.mark_saved(a.id, memory_entity_id="")

    def test_forget_removes_entry(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c", summary="s")
        b = store.append_chain(chain={}, title="d", summary="s")
        removed = store.forget(a.id)
        assert removed.id == a.id
        assert a.id not in store
        assert b.id in store

    def test_forget_unknown_raises(self):
        store = SessionArtifactStore()
        with pytest.raises(SessionArtifactStoreError):
            store.forget("missing")

    def test_clear_returns_count(self):
        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c", summary="s")
        store.append_chain(chain={}, title="d", summary="s")
        n = store.clear()
        assert n == 2
        assert len(store) == 0


class TestListeners:
    def test_listener_fires_on_append_and_mark_saved(self):
        store = SessionArtifactStore()
        seen: list[tuple[str, bool]] = []
        store.add_listener(
            lambda a: seen.append((a.id, a.saved_to_memory))
        )
        a = store.append_chain(chain={}, title="c", summary="s")
        store.mark_saved(a.id, memory_entity_id="ENT")
        # First event: appended (unsaved). Second: marked saved.
        assert seen == [(a.id, False), (a.id, True)]

    def test_listener_dedup_on_add(self):
        store = SessionArtifactStore()
        seen: list[str] = []

        def listener(a):  # noqa: ANN001 — local stub
            seen.append(a.id)

        store.add_listener(listener)
        store.add_listener(listener)  # duplicate — should not re-register
        store.append_chain(chain={}, title="c", summary="s")
        assert len(seen) == 1

    def test_listener_failure_is_swallowed(self, caplog):
        store = SessionArtifactStore()

        def bad(_a):  # noqa: ANN001
            raise RuntimeError("boom")

        good: list[str] = []
        store.add_listener(bad)
        store.add_listener(lambda a: good.append(a.id))
        with caplog.at_level("WARNING", logger="care.session_artifacts"):
            store.append_chain(chain={}, title="c", summary="s")
        # Good listener still fired despite bad one raising.
        assert len(good) == 1
        # Warning logged about the failing listener.
        assert any("listener" in r.message for r in caplog.records)

    def test_remove_listener_is_silent_no_op_when_unregistered(self):
        store = SessionArtifactStore()
        store.remove_listener(lambda _a: None)  # must not raise


class TestThreadSafety:
    def test_concurrent_appends_dont_corrupt(self):
        store = SessionArtifactStore()

        def worker(n: int) -> None:
            for i in range(n):
                store.append_chain(chain={"i": i}, title=f"c{i}", summary="s")

        threads = [
            threading.Thread(target=worker, args=(20,)) for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All appends landed, ids unique.
        rows = store.list_artifacts()
        assert len(rows) == 8 * 20
        assert len({a.id for a in rows}) == len(rows)


class TestPersistence:
    def test_dump_load_round_trip(self, tmp_path):
        store = SessionArtifactStore()
        a = store.append_chain(
            chain={"steps": [{"id": "s1"}]},
            title="weather", summary="3-step",
            origin={"turn_index": 1},
        )
        store.append_tool_output(tool="ls", output=["a", "b"])
        store.mark_saved(a.id, memory_entity_id="ENT-1")
        path = tmp_path / "session.jsonl"
        n = store.dump_jsonl(path)
        assert n == 2
        # Each row is a single JSON object.
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        json.loads(lines[0])  # raises on malformed
        # Functional alias matches the bound method.
        path2 = tmp_path / "alias.jsonl"
        n_alias = dump_jsonl(store, path2)
        assert n_alias == 2
        # Round-trip via load_jsonl_artifacts.
        loaded = load_jsonl_artifacts(path)
        assert {a.title for a in loaded} == {"weather", "Tool: ls"}
        # Saved flag survived.
        saved_loaded = [a for a in loaded if a.title == "weather"][0]
        assert saved_loaded.saved_to_memory is True
        assert saved_loaded.memory_entity_id == "ENT-1"

    def test_load_missing_file_returns_empty(self, tmp_path):
        path = tmp_path / "absent.jsonl"
        assert load_jsonl_artifacts(path) == []

    def test_load_skips_malformed_rows(self, tmp_path, caplog):
        path = tmp_path / "broken.jsonl"
        # Valid row, then a malformed one.
        good = {
            "id": "abc123",
            "kind": "chain",
            "created_at": "2026-06-03T12:00:00+00:00",
            "title": "good",
            "summary": "",
            "payload": {},
            "origin": {},
            "saved_to_memory": False,
            "memory_entity_id": None,
        }
        path.write_text(
            json.dumps(good) + "\n" + "{not json}\n",
            encoding="utf-8",
        )
        with caplog.at_level("WARNING", logger="care.session_artifacts"):
            rows = load_jsonl_artifacts(path)
        assert len(rows) == 1
        assert rows[0].id == "abc123"
        assert any("malformed" in r.message for r in caplog.records)

    def test_replay_into_requires_empty_store(self, tmp_path):
        path = tmp_path / "session.jsonl"
        seed = SessionArtifactStore()
        seed.append_chain(chain={}, title="x", summary="")
        seed.dump_jsonl(path)
        loaded = load_jsonl_artifacts(path)

        target = SessionArtifactStore()
        replay_into(target, loaded)
        assert len(target) == 1

        target_again = SessionArtifactStore()
        target_again.append_chain(chain={}, title="pre", summary="")
        with pytest.raises(SessionArtifactStoreError):
            replay_into(target_again, loaded)


class TestRuntimeExport:
    def test_top_level_exports_resolve(self):
        from care import runtime as r
        assert r.SessionArtifact is SessionArtifact
        assert r.SessionArtifactStore is SessionArtifactStore
        assert r.SessionArtifactStoreError is SessionArtifactStoreError
        # Functional alias re-exported under a disambiguated name.
        assert r.dump_session_artifacts_jsonl is dump_jsonl
        assert r.load_jsonl_artifacts is load_jsonl_artifacts
