"""Tests for the bulk-library-operations data layer (TODO §1.3 P1).

The Textual key bindings + tag-editor modal are gated on §1 P0;
this suite pins the contract the modal binds to.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import FrozenInstanceError

import pytest

from care.runtime.bulk_ops import (
    BulkOperationError,
    BulkOperationOutcome,
    BulkOperationResult,
    BulkSelection,
    BulkTarget,
    apply_delete,
    apply_favourite,
    apply_tag_edits,
    merge_tags,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _target(
    entity_id: str = "id-1",
    *,
    entity_type: str = "chain",
    tags: tuple[str, ...] = (),
    name: str | None = None,
) -> BulkTarget:
    return BulkTarget(
        entity_id=entity_id,
        entity_type=entity_type,
        current_tags=tags,
        display_name=name,
    )


class _StubClient:
    """Mimics the SDK base — `_mark_favourite` / `_update_metadata`
    / `_delete_entity` / `_get_entity`."""

    def __init__(
        self,
        *,
        fail_ids: set[str] | None = None,
        slow_ids: set[str] | None = None,
        slow_seconds: float = 0.0,
        get_response: dict | None = None,
    ):
        self.calls: list[tuple[str, str, str, dict]] = []
        self._fail_ids = fail_ids or set()
        self._slow_ids = slow_ids or set()
        self._slow_seconds = slow_seconds
        self._get_response = get_response or {"meta": {"tags": []}}
        self._lock = threading.Lock()

    def _record(self, op: str, entity_type: str, entity_id: str, **extra) -> None:
        with self._lock:
            self.calls.append((op, entity_type, entity_id, dict(extra)))
        if entity_id in self._slow_ids:
            time.sleep(self._slow_seconds)
        if entity_id in self._fail_ids:
            raise RuntimeError(f"503 for {entity_id}")

    def _mark_favourite(self, entity_type, entity_id, value=True):
        self._record(
            "favourite", entity_type, entity_id, value=value
        )
        return {"ok": True}

    def _update_metadata(
        self, entity_type, entity_id, *, display_name=None,
        description=None, tags=None, favourite=None,
    ):
        self._record(
            "patch", entity_type, entity_id,
            tags=list(tags) if tags is not None else None,
            favourite=favourite,
        )
        return {"ok": True}

    def _delete_entity(self, entity_type, entity_id):
        self._record("delete", entity_type, entity_id)
        return True

    def _get_entity(self, entity_type, entity_id):
        self._record("get", entity_type, entity_id)
        return self._get_response


class _StubMemory:
    def __init__(self, client):
        self.client = client


# ---------------------------------------------------------------------------
# BulkSelection
# ---------------------------------------------------------------------------


class TestBulkSelection:
    def test_empty_predicates(self):
        sel = BulkSelection()
        assert sel.is_empty is True
        assert len(sel) == 0
        assert sel.entity_ids == ()
        assert "x" not in sel

    def test_add_appends_and_dedupes(self):
        sel = BulkSelection().add(_target("a")).add(_target("b"))
        assert len(sel) == 2
        assert sel.entity_ids == ("a", "b")
        # Same id again → no-op.
        same = sel.add(_target("a"))
        assert same is sel

    def test_remove_drops_when_present(self):
        sel = BulkSelection().add(_target("a")).add(_target("b"))
        smaller = sel.remove("a")
        assert smaller.entity_ids == ("b",)
        # Removing unknown id → no-op (same instance).
        same = sel.remove("missing")
        assert same is sel

    def test_toggle_round_trip(self):
        sel = BulkSelection()
        t = _target("x")
        on = sel.toggle(t)
        assert "x" in on
        off = on.toggle(t)
        assert "x" not in off

    def test_clear_resets(self):
        sel = BulkSelection().add(_target("a"))
        assert sel.clear().is_empty

    def test_find(self):
        sel = BulkSelection().add(_target("a", name="alpha"))
        assert sel.find("a").display_name == "alpha"
        assert sel.find("z") is None

    def test_iter(self):
        sel = BulkSelection().add(_target("a")).add(_target("b"))
        assert [t.entity_id for t in sel] == ["a", "b"]

    def test_contains_only_strings(self):
        sel = BulkSelection().add(_target("a"))
        assert "a" in sel
        # Non-string lookups don't crash; just return False.
        assert 42 not in sel  # type: ignore[operator]

    def test_target_is_frozen(self):
        target = _target()
        with pytest.raises(FrozenInstanceError):
            target.entity_id = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# merge_tags
# ---------------------------------------------------------------------------


class TestMergeTags:
    def test_pure_add(self):
        assert merge_tags(["a"], add=["b", "c"]) == ["a", "b", "c"]

    def test_pure_remove(self):
        assert merge_tags(["a", "b", "c"], remove=["b"]) == ["a", "c"]

    def test_add_and_remove(self):
        assert merge_tags(["x", "y"], add=["z"], remove=["x"]) == ["y", "z"]

    def test_dedup_existing(self):
        # If the caller passed duplicates, dedup them on the way through.
        assert merge_tags(["a", "a", "b"], add=["c"]) == ["a", "b", "c"]

    def test_add_overlapping_existing(self):
        # Adding what's already there is a no-op.
        assert merge_tags(["a", "b"], add=["a"]) == ["a", "b"]

    def test_remove_then_re_add(self):
        # remove wins over add for the same tag — predictable
        # behaviour for the modal's combined add+remove input.
        assert merge_tags(["a", "b"], add=["b"], remove=["b"]) == ["a"]

    def test_whitespace_stripped(self):
        assert merge_tags(["a"], add=["  b  ", "\tc"]) == ["a", "b", "c"]

    def test_empty_strings_skipped(self):
        assert merge_tags(["a"], add=["", "   "]) == ["a"]

    def test_case_sensitive(self):
        assert merge_tags(["Tag"], add=["tag"]) == ["Tag", "tag"]


# ---------------------------------------------------------------------------
# apply_favourite
# ---------------------------------------------------------------------------


class TestApplyFavourite:
    def test_empty_selection_returns_empty_result(self):
        memory = _StubMemory(_StubClient())
        result = asyncio.run(apply_favourite(memory, BulkSelection()))
        assert result.total == 0
        assert result.operation == "favourite"
        assert memory.client.calls == []

    def test_all_succeed(self):
        client = _StubClient()
        memory = _StubMemory(client)
        sel = (
            BulkSelection()
            .add(_target("c1"))
            .add(_target("c2", entity_type="agent"))
        )
        result = asyncio.run(apply_favourite(memory, sel))
        assert result.total == 2
        assert result.succeeded == 2
        assert result.failed == 0
        assert result.all_succeeded
        # Each call hit `_mark_favourite` with the right typed router.
        ops = [(c[0], c[1], c[2]) for c in client.calls]
        assert ("favourite", "chain", "c1") in ops
        assert ("favourite", "agent", "c2") in ops

    def test_partial_failure(self):
        client = _StubClient(fail_ids={"c2"})
        memory = _StubMemory(client)
        sel = BulkSelection().add(_target("c1")).add(_target("c2"))
        result = asyncio.run(apply_favourite(memory, sel))
        assert result.succeeded == 1
        assert result.failed == 1
        assert result.any_failed
        # The failed outcome carries the error.
        failures = result.failures
        assert len(failures) == 1
        assert "503" in failures[0].error
        assert failures[0].entity_id == "c2"

    def test_unfavourite_passes_false(self):
        client = _StubClient()
        memory = _StubMemory(client)
        sel = BulkSelection().add(_target("c1"))
        asyncio.run(apply_favourite(memory, sel, favourite=False))
        assert client.calls[0][3] == {"value": False}

    def test_missing_client_raises(self):
        with pytest.raises(BulkOperationError, match="client"):
            asyncio.run(apply_favourite(object(), BulkSelection().add(_target("a"))))

    def test_missing_method_raises(self):
        class _Empty:
            pass

        memory = _StubMemory(_Empty())
        with pytest.raises(BulkOperationError, match="_mark_favourite"):
            asyncio.run(apply_favourite(memory, BulkSelection().add(_target("a"))))

    def test_outcomes_preserve_order(self):
        client = _StubClient()
        memory = _StubMemory(client)
        sel = (
            BulkSelection()
            .add(_target("a"))
            .add(_target("b"))
            .add(_target("c"))
        )
        result = asyncio.run(apply_favourite(memory, sel))
        assert [o.entity_id for o in result] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# apply_tag_edits
# ---------------------------------------------------------------------------


class TestApplyTagEdits:
    def test_no_op_when_no_tags(self):
        memory = _StubMemory(_StubClient())
        sel = BulkSelection().add(_target("a"))
        result = asyncio.run(apply_tag_edits(memory, sel))
        assert result.total == 0
        assert memory.client.calls == []

    def test_no_op_on_whitespace_only_tags(self):
        memory = _StubMemory(_StubClient())
        sel = BulkSelection().add(_target("a"))
        result = asyncio.run(apply_tag_edits(memory, sel, add_tags=["   "]))
        assert result.total == 0

    def test_add_tags_merged_with_existing(self):
        client = _StubClient()
        memory = _StubMemory(client)
        sel = BulkSelection().add(_target("a", tags=("existing",)))
        result = asyncio.run(apply_tag_edits(memory, sel, add_tags=["new"]))
        assert result.all_succeeded
        # PATCH call carries the merged tag set.
        patch_calls = [c for c in client.calls if c[0] == "patch"]
        assert patch_calls[0][3]["tags"] == ["existing", "new"]

    def test_remove_tags_with_known_current_tags(self):
        client = _StubClient()
        memory = _StubMemory(client)
        sel = BulkSelection().add(
            _target("a", tags=("keep", "drop"))
        )
        asyncio.run(apply_tag_edits(memory, sel, remove_tags=["drop"]))
        patch_calls = [c for c in client.calls if c[0] == "patch"]
        assert patch_calls[0][3]["tags"] == ["keep"]
        # No GET should fire because current_tags was supplied.
        assert not any(c[0] == "get" for c in client.calls)

    def test_remove_tags_fetches_when_current_unknown(self):
        # Row was selected before tags were loaded → bulk helper
        # GETs the entity to learn the current set.
        client = _StubClient(
            get_response={"meta": {"tags": ["server-known", "to-drop"]}}
        )
        memory = _StubMemory(client)
        sel = BulkSelection().add(_target("a", tags=()))
        asyncio.run(apply_tag_edits(memory, sel, remove_tags=["to-drop"]))
        # GET called once, then PATCH.
        ops = [c[0] for c in client.calls]
        assert ops == ["get", "patch"]
        patch = [c for c in client.calls if c[0] == "patch"][0]
        assert patch[3]["tags"] == ["server-known"]

    def test_remove_only_no_fetch_when_current_tags_present(self):
        # current_tags is empty BUT only adding → no GET needed.
        client = _StubClient()
        memory = _StubMemory(client)
        sel = BulkSelection().add(_target("a"))
        asyncio.run(apply_tag_edits(memory, sel, add_tags=["foo"]))
        ops = [c[0] for c in client.calls]
        assert ops == ["patch"]
        patch = [c for c in client.calls if c[0] == "patch"][0]
        assert patch[3]["tags"] == ["foo"]

    def test_partial_failure(self):
        client = _StubClient(fail_ids={"b"})
        memory = _StubMemory(client)
        sel = (
            BulkSelection()
            .add(_target("a", tags=("x",)))
            .add(_target("b", tags=("x",)))
        )
        result = asyncio.run(apply_tag_edits(memory, sel, add_tags=["y"]))
        assert result.succeeded == 1
        assert result.failed == 1

    def test_operation_name(self):
        memory = _StubMemory(_StubClient())
        sel = BulkSelection().add(_target("a", tags=("x",)))
        result = asyncio.run(
            apply_tag_edits(memory, sel, add_tags=["y"], remove_tags=["x"])
        )
        # `+1 tag` and `-1 tag` in the operation name.
        assert "+1 tag" in result.operation
        assert "-1 tag" in result.operation


# ---------------------------------------------------------------------------
# apply_delete
# ---------------------------------------------------------------------------


class TestApplyDelete:
    def test_calls_delete_for_each(self):
        client = _StubClient()
        memory = _StubMemory(client)
        sel = (
            BulkSelection()
            .add(_target("a", entity_type="chain"))
            .add(_target("b", entity_type="agent"))
        )
        result = asyncio.run(apply_delete(memory, sel))
        assert result.total == 2
        assert result.all_succeeded
        ops = [(c[0], c[1], c[2]) for c in client.calls]
        assert ("delete", "chain", "a") in ops
        assert ("delete", "agent", "b") in ops

    def test_empty_selection_no_call(self):
        memory = _StubMemory(_StubClient())
        result = asyncio.run(apply_delete(memory, BulkSelection()))
        assert result.total == 0

    def test_partial_failure(self):
        client = _StubClient(fail_ids={"b"})
        memory = _StubMemory(client)
        sel = BulkSelection().add(_target("a")).add(_target("b"))
        result = asyncio.run(apply_delete(memory, sel))
        assert result.succeeded == 1
        assert result.failed == 1


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeouts:
    def test_slow_target_times_out_without_blocking_others(self):
        # `a` is slow, `b` is normal — only `a` times out.
        client = _StubClient(slow_ids={"a"}, slow_seconds=0.5)
        memory = _StubMemory(client)
        sel = BulkSelection().add(_target("a")).add(_target("b"))
        result = asyncio.run(apply_favourite(memory, sel, timeout=0.05))
        # `a` failed with timeout, `b` succeeded.
        a_outcome = next(o for o in result if o.entity_id == "a")
        b_outcome = next(o for o in result if o.entity_id == "b")
        assert a_outcome.success is False
        assert "timed out" in a_outcome.error
        assert b_outcome.success is True


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_bounded_semaphore_caps_in_flight(self):
        # 4 targets, concurrency=2, each takes 0.05s. Total wall-clock
        # should be ~0.1s (two batches of two), not ~0.2s (serial)
        # nor ~0.05s (full parallel).
        client = _StubClient(
            slow_ids={"a", "b", "c", "d"}, slow_seconds=0.05
        )
        memory = _StubMemory(client)
        sel = (
            BulkSelection()
            .add(_target("a"))
            .add(_target("b"))
            .add(_target("c"))
            .add(_target("d"))
        )
        start = time.monotonic()
        result = asyncio.run(
            apply_favourite(memory, sel, concurrency=2, timeout=2.0)
        )
        elapsed = time.monotonic() - start
        assert result.all_succeeded
        assert 0.08 < elapsed < 0.18, (
            f"expected ~0.1s with concurrency=2, got {elapsed:.3f}s"
        )

    def test_concurrency_one_runs_serially(self):
        # 3 targets × 0.03s with concurrency=1 → ~0.09s.
        client = _StubClient(
            slow_ids={"a", "b", "c"}, slow_seconds=0.03
        )
        memory = _StubMemory(client)
        sel = (
            BulkSelection()
            .add(_target("a"))
            .add(_target("b"))
            .add(_target("c"))
        )
        start = time.monotonic()
        asyncio.run(apply_favourite(memory, sel, concurrency=1, timeout=2.0))
        elapsed = time.monotonic() - start
        assert elapsed >= 0.08


# ---------------------------------------------------------------------------
# BulkOperationResult predicates
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_format_summary_all_ok(self):
        outcomes = (
            BulkOperationOutcome(target=_target("a"), success=True),
            BulkOperationOutcome(target=_target("b"), success=True),
        )
        result = BulkOperationResult(outcomes=outcomes, operation="favourite")
        assert "2/2 ok" in result.format_summary()

    def test_format_summary_partial(self):
        outcomes = (
            BulkOperationOutcome(target=_target("a"), success=True),
            BulkOperationOutcome(target=_target("b"), success=False, error="boom"),
        )
        result = BulkOperationResult(outcomes=outcomes, operation="delete")
        text = result.format_summary()
        assert "1/2 ok" in text
        assert "1 failed" in text

    def test_format_summary_empty(self):
        assert "nothing to do" in BulkOperationResult().format_summary()

    def test_iter(self):
        outcomes = (
            BulkOperationOutcome(target=_target("a"), success=True),
        )
        result = BulkOperationResult(outcomes=outcomes)
        assert list(result) == list(outcomes)

    def test_outcome_is_frozen(self):
        outcome = BulkOperationOutcome(target=_target("a"), success=True)
        with pytest.raises(FrozenInstanceError):
            outcome.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            BulkSelection as S,
            BulkTarget as T,
            BulkOperationResult as R,
            apply_favourite as fav,
            apply_tag_edits as tag,
            apply_delete as delete,
            merge_tags as merge,
        )

        assert S is BulkSelection
        assert T is BulkTarget
        assert R is BulkOperationResult
        assert fav is apply_favourite
        assert tag is apply_tag_edits
        assert delete is apply_delete
        assert merge is merge_tags
