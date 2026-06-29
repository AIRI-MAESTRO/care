"""In-session background-task registry (TODO §1.2 P1).

CARE runs several long-lived background tasks at once — a MAGE
generation, one or more CARL executions, a Platform evolution
stream — and the user wants a single place to see them all,
switch focus between them, or cancel any of them. The
``TaskList`` widget is the UI; this module owns the **state**
behind it.

Design:

* :class:`TaskRecord` — frozen snapshot of one task's identity
  + status + timing. Frozen so listeners can hold one without
  worrying about mutation; the registry hands out fresh records
  on every update.
* :class:`TaskRegistry` — mutable in-memory store. Add a task
  via :meth:`register`, mark progress with
  :meth:`mark_running` / :meth:`mark_completed` / :meth:`mark_failed`,
  cancel via :meth:`cancel` (which flips the bound
  :class:`CancellationToken`). Listeners subscribe via
  :meth:`on_change` and get called with every transition.

The registry is **session-local** — entries die with the process.
Resumable runs (TODO §1.2 P2) layer persistence on top later.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

from care.runtime.cancellation import CancellationToken

TaskKind = Literal[
    "mage_generation",
    "carl_execution",
    "platform_evolution",
    "memory_sync",
    "other",
]
"""Canonical categories the TaskList groups by. ``"other"`` is
the catch-all so module experimentation doesn't need a registry
bump."""

TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
"""State machine: ``pending → running → completed | failed |
cancelled``. Terminal transitions don't fire onward — once a
task is done it stays in its terminal state until the user clears
it via :meth:`TaskRegistry.clear`."""

_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "cancelled"}
)


@dataclass(frozen=True)
class TaskRecord:
    """One row in the task list.

    Frozen so the UI can hold snapshots safely. ``token`` is the
    cancellation handle the task itself polls — the registry
    flips it on :meth:`TaskRegistry.cancel`. ``token`` is excluded
    from equality so two snapshots of the same task compare equal
    regardless of which one came first.
    """

    id: str
    kind: TaskKind
    label: str
    status: TaskStatus = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    token: CancellationToken | None = field(default=None, compare=False)

    @property
    def is_terminal(self) -> bool:
        """``True`` for tasks that have left the running state —
        the UI can stop polling them."""
        return self.status in _TERMINAL_STATUSES

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock seconds from start to finish. ``None`` when
        the task hasn't started yet OR is still running."""
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


class TaskRegistryError(RuntimeError):
    """Raised on registry misuse — registering a duplicate id,
    transitioning a terminal task, looking up an unknown id."""


ListenerKind = Literal["registered", "updated", "removed"]
"""Listener-event kinds.

* ``registered``: a new task appeared.
* ``updated``: an existing task transitioned (any field change).
* ``removed``: ``clear()`` dropped the task from the registry.
"""

Listener = Callable[[ListenerKind, TaskRecord], None]


class TaskRegistry:
    """Mutable in-memory store of active + recently-finished tasks.

    Thread-safe: a `threading.Lock` guards all reads + writes so
    the TUI thread + worker threads can both touch it without
    races. Listener callbacks fire **outside the lock** so a slow
    listener can't deadlock the registry.

    All mutators return the post-mutation :class:`TaskRecord` so
    callers can chain assertions / propagate the new state into
    UI messages without a second lookup.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: object) -> bool:
        return isinstance(task_id, str) and task_id in self._tasks

    def get(self, task_id: str) -> TaskRecord:
        """Return the record for ``task_id`` or raise
        :class:`TaskRegistryError` when unknown.

        Use this over ``registry[task_id]`` because the loud
        failure tells the screen "you held a stale id"."""
        with self._lock:
            try:
                return self._tasks[task_id]
            except KeyError as exc:
                raise TaskRegistryError(
                    f"no task registered with id {task_id!r}"
                ) from exc

    def list_tasks(
        self,
        *,
        kind: TaskKind | None = None,
        status: TaskStatus | None = None,
        active_only: bool = False,
    ) -> list[TaskRecord]:
        """Snapshot of every record matching the optional filters.

        Sorted by ``started_at`` (None first, then ascending) so
        the TaskList can render in arrival order. ``active_only``
        drops terminal tasks regardless of the ``status`` filter
        — useful for the always-visible header counter."""
        with self._lock:
            records = list(self._tasks.values())
        if kind is not None:
            records = [r for r in records if r.kind == kind]
        if status is not None:
            records = [r for r in records if r.status == status]
        if active_only:
            records = [r for r in records if not r.is_terminal]
        records.sort(
            key=lambda r: (r.started_at is not None, r.started_at or 0.0)
        )
        return records

    # ------------------------------------------------------------------
    # Mutate
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        kind: TaskKind,
        label: str,
        token: CancellationToken | None = None,
        metadata: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> TaskRecord:
        """Add a new task in the ``pending`` state.

        Args:
            kind: Task category (see :data:`TaskKind`).
            label: Human-friendly description rendered in the
                TaskList ("Generating chain for 'weather report'").
            token: Optional :class:`CancellationToken` the task
                itself polls. The registry's :meth:`cancel` flips
                this. ``None`` creates a fresh token automatically.
            metadata: Free-form dict the UI can inspect (e.g.
                ``{"chain_id": "e-7"}`` for execution tasks).
            task_id: Override the generated id. Mostly for tests;
                production callers leave it None.

        Returns:
            The fresh :class:`TaskRecord`.
        """
        assigned_id = task_id or uuid.uuid4().hex
        record = TaskRecord(
            id=assigned_id,
            kind=kind,
            label=label,
            status="pending",
            token=token or CancellationToken(),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            if assigned_id in self._tasks:
                raise TaskRegistryError(
                    f"task id {assigned_id!r} already registered"
                )
            self._tasks[assigned_id] = record
        self._notify("registered", record)
        return record

    def mark_running(
        self,
        task_id: str,
        *,
        started_at: float | None = None,
    ) -> TaskRecord:
        """Transition ``pending → running``. ``started_at`` defaults
        to ``time.monotonic()``."""
        return self._transition(
            task_id,
            allowed_from=("pending",),
            new_status="running",
            started_at=started_at or time.monotonic(),
        )

    def mark_completed(
        self,
        task_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        finished_at: float | None = None,
    ) -> TaskRecord:
        """Transition ``pending|running → completed``. Optional
        ``metadata`` is merged into the record (so the UI can pin
        the result entity id, total tokens, etc.)."""
        return self._transition(
            task_id,
            allowed_from=("pending", "running"),
            new_status="completed",
            finished_at=finished_at or time.monotonic(),
            metadata_merge=metadata,
        )

    def mark_failed(
        self,
        task_id: str,
        error_message: str,
        *,
        metadata: dict[str, Any] | None = None,
        finished_at: float | None = None,
    ) -> TaskRecord:
        """Transition to ``failed`` with an error message the UI
        renders verbatim in the failure toast / detail pane."""
        return self._transition(
            task_id,
            allowed_from=("pending", "running"),
            new_status="failed",
            finished_at=finished_at or time.monotonic(),
            error_message=error_message,
            metadata_merge=metadata,
        )

    def cancel(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskRecord:
        """Flip the task's cancellation token AND transition the
        record to ``cancelled``.

        Already-terminal tasks return their current record
        unchanged (idempotent; the user smashing Esc twice
        shouldn't 500).
        """
        with self._lock:
            try:
                current = self._tasks[task_id]
            except KeyError as exc:
                raise TaskRegistryError(
                    f"no task registered with id {task_id!r}"
                ) from exc
            if current.is_terminal:
                return current
            if current.token is not None:
                if reason and not current.token.is_cancelled:
                    # Refresh the reason on the token if the
                    # caller supplied one (only matters for the
                    # first cancel — token reason is immutable
                    # after that, but we set it once).
                    current = replace(
                        current,
                        metadata={**current.metadata, "cancel_reason": reason},
                    )
                    self._tasks[task_id] = current
                current.token.cancel()
            updated = replace(
                current,
                status="cancelled",
                finished_at=time.monotonic(),
            )
            self._tasks[task_id] = updated
        self._notify("updated", updated)
        return updated

    def clear(self, task_id: str) -> TaskRecord | None:
        """Remove a terminal task from the registry. Returns the
        removed record, or ``None`` when nothing was there.

        Refuses to clear a non-terminal task — call :meth:`cancel`
        first."""
        with self._lock:
            current = self._tasks.get(task_id)
            if current is None:
                return None
            if not current.is_terminal:
                raise TaskRegistryError(
                    f"task {task_id!r} is still {current.status!r}; "
                    "cancel it before clearing"
                )
            del self._tasks[task_id]
        self._notify("removed", current)
        return current

    def clear_finished(self) -> int:
        """Drop every terminal task. Returns the number removed.

        Convenience for the "Clean up" command-palette action."""
        removed: list[TaskRecord] = []
        with self._lock:
            for tid, rec in list(self._tasks.items()):
                if rec.is_terminal:
                    removed.append(rec)
                    del self._tasks[tid]
        for rec in removed:
            self._notify("removed", rec)
        return len(removed)

    # ------------------------------------------------------------------
    # Listener API
    # ------------------------------------------------------------------

    def on_change(self, listener: Listener) -> Callable[[], None]:
        """Register a callback that fires on every registry event.

        Returns an unsubscribe function — call it from the screen's
        ``on_unmount`` to avoid leaks. Listener exceptions are
        swallowed so one bad subscriber can't break the rest."""
        with self._lock:
            self._listeners.append(listener)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(listener)
                except ValueError:
                    pass

        return _unsubscribe

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _transition(
        self,
        task_id: str,
        *,
        allowed_from: tuple[TaskStatus, ...],
        new_status: TaskStatus,
        started_at: float | None = None,
        finished_at: float | None = None,
        error_message: str | None = None,
        metadata_merge: dict[str, Any] | None = None,
    ) -> TaskRecord:
        with self._lock:
            try:
                current = self._tasks[task_id]
            except KeyError as exc:
                raise TaskRegistryError(
                    f"no task registered with id {task_id!r}"
                ) from exc
            if current.status not in allowed_from:
                raise TaskRegistryError(
                    f"cannot transition task {task_id!r} from "
                    f"{current.status!r} to {new_status!r}"
                )
            new_metadata = current.metadata
            if metadata_merge:
                new_metadata = {**current.metadata, **metadata_merge}
            updated = replace(
                current,
                status=new_status,
                started_at=started_at if started_at is not None else current.started_at,
                finished_at=finished_at if finished_at is not None else current.finished_at,
                error_message=error_message if error_message is not None else current.error_message,
                metadata=new_metadata,
            )
            self._tasks[task_id] = updated
        self._notify("updated", updated)
        return updated

    def _notify(self, kind: ListenerKind, record: TaskRecord) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(kind, record)
            except Exception:  # noqa: BLE001
                # One bad listener mustn't break the others or
                # poison the calling mutator.
                pass


__all__ = [
    "Listener",
    "ListenerKind",
    "TaskKind",
    "TaskRecord",
    "TaskRegistry",
    "TaskRegistryError",
    "TaskStatus",
]
