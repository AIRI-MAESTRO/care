"""Tests for ``care.runtime.task_registry`` (TODO §1.2 P1).

Pure-Python state model — no upstream deps. Coverage layers:

1. ``TaskRecord`` shape: frozen + `is_terminal` predicate +
   `duration_seconds` lifecycle.
2. ``register`` issues fresh ids, attaches a default token,
   rejects duplicates.
3. State machine: pending → running → completed / failed work;
   illegal transitions raise; cancel from any non-terminal state.
4. ``cancel`` flips the bound token AND transitions the record.
5. ``clear`` / ``clear_finished`` only touch terminal tasks.
6. ``list_tasks`` filters + ordering.
7. Listeners fire on every event, exceptions are swallowed,
   unsubscribe works.
8. Thread-safety smoke: 8 concurrent registrations all land
   without losing entries.
"""

from __future__ import annotations

import threading
import time

import pytest

from care.runtime import (
    CancellationToken,
    TaskRecord,
    TaskRegistry,
    TaskRegistryError,
)


# ---------------------------------------------------------------------------
# TaskRecord shape
# ---------------------------------------------------------------------------


class TestTaskRecordShape:
    def test_frozen(self):
        rec = TaskRecord(id="t-1", kind="mage_generation", label="X")
        with pytest.raises(AttributeError):
            rec.status = "completed"  # type: ignore[misc]

    def test_is_terminal_predicate(self):
        for status in ("pending", "running"):
            rec = TaskRecord(id="t", kind="other", label="X", status=status)
            assert rec.is_terminal is False
        for status in ("completed", "failed", "cancelled"):
            rec = TaskRecord(id="t", kind="other", label="X", status=status)
            assert rec.is_terminal is True

    def test_duration_seconds_returns_none_until_finished(self):
        rec = TaskRecord(id="t", kind="other", label="X")
        assert rec.duration_seconds is None
        rec = TaskRecord(id="t", kind="other", label="X", started_at=10.0)
        assert rec.duration_seconds is None
        rec = TaskRecord(
            id="t", kind="other", label="X", started_at=10.0, finished_at=12.5
        )
        assert rec.duration_seconds == 2.5

    def test_records_compare_equal_ignoring_token(self):
        """The token field is excluded from __eq__ so two snapshots
        of the same task (same id + state) compare equal even when
        each carries a distinct CancellationToken object."""
        rec_a = TaskRecord(
            id="t-1", kind="other", label="X", token=CancellationToken()
        )
        rec_b = TaskRecord(
            id="t-1", kind="other", label="X", token=CancellationToken()
        )
        assert rec_a == rec_b


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_issues_fresh_id(self):
        reg = TaskRegistry()
        a = reg.register(kind="other", label="A")
        b = reg.register(kind="other", label="B")
        assert a.id != b.id
        assert a.status == "pending"

    def test_register_attaches_default_token(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        assert isinstance(rec.token, CancellationToken)
        assert rec.token.is_cancelled is False

    def test_register_honours_caller_token(self):
        reg = TaskRegistry()
        tok = CancellationToken(reason="custom")
        rec = reg.register(kind="other", label="X", token=tok)
        assert rec.token is tok

    def test_register_caller_supplied_id(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X", task_id="t-fixed")
        assert rec.id == "t-fixed"

    def test_duplicate_id_raises(self):
        reg = TaskRegistry()
        reg.register(kind="other", label="X", task_id="t-1")
        with pytest.raises(TaskRegistryError, match="already registered"):
            reg.register(kind="other", label="Y", task_id="t-1")


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_pending_to_running(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        running = reg.mark_running(rec.id)
        assert running.status == "running"
        assert running.started_at is not None

    def test_running_to_completed(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        reg.mark_running(rec.id)
        done = reg.mark_completed(rec.id, metadata={"chain_id": "c-7"})
        assert done.status == "completed"
        assert done.finished_at is not None
        assert done.metadata == {"chain_id": "c-7"}

    def test_running_to_failed_with_error_message(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        reg.mark_running(rec.id)
        failed = reg.mark_failed(rec.id, "step 1 timed out")
        assert failed.status == "failed"
        assert failed.error_message == "step 1 timed out"

    def test_completed_can_be_reached_from_pending(self):
        """A task that completes synchronously without ever
        running through the registry's mark_running call should
        still transition cleanly."""
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        done = reg.mark_completed(rec.id)
        assert done.status == "completed"

    def test_cannot_resurrect_terminal_task(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        reg.mark_completed(rec.id)
        with pytest.raises(TaskRegistryError, match="cannot transition"):
            reg.mark_running(rec.id)

    def test_unknown_id_raises(self):
        reg = TaskRegistry()
        with pytest.raises(TaskRegistryError, match="no task"):
            reg.mark_running("does-not-exist")

    def test_metadata_merged_not_replaced(self):
        reg = TaskRegistry()
        rec = reg.register(
            kind="other", label="X", metadata={"phase": "init"}
        )
        reg.mark_running(rec.id)
        done = reg.mark_completed(rec.id, metadata={"chain_id": "c-7"})
        assert done.metadata == {"phase": "init", "chain_id": "c-7"}


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_flips_token_and_transitions_record(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        cancelled = reg.cancel(rec.id, reason="user esc")
        assert cancelled.status == "cancelled"
        assert cancelled.token is not None
        assert cancelled.token.is_cancelled is True
        assert cancelled.metadata.get("cancel_reason") == "user esc"

    def test_cancel_is_idempotent_on_terminal_tasks(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        reg.mark_completed(rec.id)
        # Second call is a no-op — returns the completed record.
        out = reg.cancel(rec.id)
        assert out.status == "completed"

    def test_cancel_unknown_raises(self):
        reg = TaskRegistry()
        with pytest.raises(TaskRegistryError, match="no task"):
            reg.cancel("missing")

    def test_cancel_running_task(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        reg.mark_running(rec.id)
        cancelled = reg.cancel(rec.id)
        assert cancelled.status == "cancelled"
        assert cancelled.finished_at is not None


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_removes_terminal_task(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        reg.mark_completed(rec.id)
        removed = reg.clear(rec.id)
        assert removed is not None
        assert removed.id == rec.id
        assert rec.id not in reg

    def test_clear_unknown_returns_none(self):
        reg = TaskRegistry()
        assert reg.clear("missing") is None

    def test_clear_refuses_non_terminal(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        with pytest.raises(TaskRegistryError, match="cancel it before"):
            reg.clear(rec.id)

    def test_clear_finished_drops_all_terminal(self):
        reg = TaskRegistry()
        a = reg.register(kind="other", label="A")
        b = reg.register(kind="other", label="B")
        c = reg.register(kind="other", label="C")
        reg.mark_completed(a.id)
        reg.mark_failed(b.id, "boom")
        # c stays pending
        n = reg.clear_finished()
        assert n == 2
        assert len(reg) == 1
        assert c.id in reg

    def test_clear_finished_on_empty_returns_zero(self):
        reg = TaskRegistry()
        assert reg.clear_finished() == 0


# ---------------------------------------------------------------------------
# list_tasks filters + ordering
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_empty_registry_returns_empty_list(self):
        assert TaskRegistry().list_tasks() == []

    def test_ordering_pending_first_then_by_started_at(self):
        reg = TaskRegistry()
        a = reg.register(kind="other", label="A")
        b = reg.register(kind="other", label="B")
        reg.register(kind="other", label="C")  # stays pending
        reg.mark_running(b.id, started_at=1.0)
        reg.mark_running(a.id, started_at=2.0)
        records = reg.list_tasks()
        labels = [r.label for r in records]
        # Pending (started_at=None) first, then by started_at asc.
        assert labels == ["C", "B", "A"]

    def test_filter_by_kind(self):
        reg = TaskRegistry()
        reg.register(kind="mage_generation", label="MAGE-1")
        reg.register(kind="carl_execution", label="CARL-1")
        reg.register(kind="mage_generation", label="MAGE-2")
        mage = reg.list_tasks(kind="mage_generation")
        assert [r.label for r in mage] == ["MAGE-1", "MAGE-2"]

    def test_filter_by_status(self):
        reg = TaskRegistry()
        a = reg.register(kind="other", label="A")
        b = reg.register(kind="other", label="B")
        reg.mark_completed(a.id)
        running = reg.list_tasks(status="pending")
        assert [r.id for r in running] == [b.id]

    def test_active_only_filter_drops_terminal(self):
        reg = TaskRegistry()
        a = reg.register(kind="other", label="A")
        b = reg.register(kind="other", label="B")
        c = reg.register(kind="other", label="C")
        reg.mark_completed(a.id)
        reg.cancel(b.id)
        active = reg.list_tasks(active_only=True)
        assert [r.id for r in active] == [c.id]


# ---------------------------------------------------------------------------
# Listeners
# ---------------------------------------------------------------------------


class TestListeners:
    def test_fires_on_register_and_updates(self):
        reg = TaskRegistry()
        events: list[tuple] = []
        reg.on_change(lambda kind, rec: events.append((kind, rec.id, rec.status)))
        rec = reg.register(kind="other", label="X")
        reg.mark_running(rec.id)
        reg.mark_completed(rec.id)
        # 1 registered + 2 updated.
        kinds = [k for k, _, _ in events]
        assert kinds == ["registered", "updated", "updated"]
        statuses = [s for _, _, s in events]
        assert statuses == ["pending", "running", "completed"]

    def test_listener_exception_swallowed(self):
        reg = TaskRegistry()
        good_events: list[str] = []

        def bad(kind, rec):
            raise RuntimeError("bad listener")

        reg.on_change(bad)
        reg.on_change(lambda k, r: good_events.append(k))
        reg.register(kind="other", label="X")
        # The good listener still fired despite the bad one.
        assert good_events == ["registered"]

    def test_unsubscribe(self):
        reg = TaskRegistry()
        events: list[str] = []
        unsub = reg.on_change(lambda k, r: events.append(k))
        reg.register(kind="other", label="X")
        unsub()
        reg.register(kind="other", label="Y")
        assert events == ["registered"]

    def test_unsubscribe_twice_is_safe(self):
        reg = TaskRegistry()
        unsub = reg.on_change(lambda k, r: None)
        unsub()
        unsub()  # no exception

    def test_remove_event_fires_on_clear(self):
        reg = TaskRegistry()
        events: list[tuple] = []
        reg.on_change(lambda kind, rec: events.append((kind, rec.id)))
        rec = reg.register(kind="other", label="X")
        reg.mark_completed(rec.id)
        reg.clear(rec.id)
        kinds = [k for k, _ in events]
        assert kinds == ["registered", "updated", "removed"]


# ---------------------------------------------------------------------------
# Lookup edge cases
# ---------------------------------------------------------------------------


class TestLookup:
    def test_contains_operator(self):
        reg = TaskRegistry()
        rec = reg.register(kind="other", label="X")
        assert rec.id in reg
        assert "missing" not in reg
        # Non-string keys don't crash.
        assert 42 not in reg  # type: ignore[operator]

    def test_len(self):
        reg = TaskRegistry()
        assert len(reg) == 0
        reg.register(kind="other", label="A")
        reg.register(kind="other", label="B")
        assert len(reg) == 2

    def test_get_raises_on_missing(self):
        reg = TaskRegistry()
        with pytest.raises(TaskRegistryError, match="no task"):
            reg.get("nope")


# ---------------------------------------------------------------------------
# Concurrency smoke
# ---------------------------------------------------------------------------


def test_concurrent_registrations_land_without_loss():
    reg = TaskRegistry()
    barrier = threading.Barrier(8)
    ids: list[str] = []
    lock = threading.Lock()

    def register():
        barrier.wait()
        rec = reg.register(kind="other", label="X")
        with lock:
            ids.append(rec.id)

    threads = [threading.Thread(target=register) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every registration produced a unique id; nothing got lost.
    assert len(ids) == 8
    assert len(set(ids)) == 8
    assert len(reg) == 8


def test_concurrent_cancels_safe_with_concurrent_listeners():
    """A bad listener + concurrent cancels should not crash."""
    reg = TaskRegistry()
    rec = reg.register(kind="other", label="X")
    fired: list[int] = []

    def bad(kind, record):
        time.sleep(0.001)
        raise RuntimeError("ow")

    reg.on_change(bad)
    reg.on_change(lambda k, r: fired.append(1))

    threads = [
        threading.Thread(target=reg.cancel, args=(rec.id,)) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert reg.get(rec.id).status == "cancelled"
    # Listeners registered AFTER the task, so only the cancel
    # path produced events. The first cancel transitions the task
    # to cancelled and fires one update; the remaining three
    # cancels short-circuit on the now-terminal task and fire
    # nothing.
    assert sum(fired) == 1
