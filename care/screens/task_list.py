"""TaskList drawer (TODO §1.1 P0.36).

Pushed by `Ctrl+B` from anywhere in the app. Renders the
active + recently-finished entries from
``app.task_registry`` as a `DataTable`. Per-row affordances:

* `Switch to` — posts :class:`TaskList.SwitchRequested` so
  the host screen routes (e.g. to ExecutionScreen for an
  in-flight chain).
* `Cancel` — calls
  :meth:`TaskRegistry.cancel(task_id)` and refreshes the
  table.

The drawer subscribes to
:meth:`TaskRegistry.on_change` so live updates land without
polling.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from care.runtime.i18n import t
from care.runtime.task_registry import (
    TaskRecord,
    TaskRegistry,
)


_STATUS_BADGES: dict[str, str] = {
    "pending": "·",
    "running": "▶",
    "completed": "✓",
    "failed": "✗",
    "cancelled": "⊘",
}


class TaskListDrawer(ModalScreen[None]):
    """Drawer-style modal listing the active task registry.

    Construct with the :class:`TaskRegistry` to render. The
    drawer mounts a DataTable, subscribes to
    `registry.on_change`, and refreshes on every event.
    """

    DEFAULT_CSS = """
    TaskListDrawer {
        align: right top;
    }
    TaskListDrawer #task-list-box {
        width: 60;
        max-width: 80%;
        height: 30;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    TaskListDrawer #task-list-title {
        text-style: bold;
        padding-bottom: 1;
    }
    TaskListDrawer #task-list-table {
        height: 1fr;
    }
    TaskListDrawer #task-list-actions {
        height: 3;
        align-horizontal: right;
    }
    TaskListDrawer Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("ctrl+b", "close", "Close", show=False),
    ]

    class SwitchRequested(Message):
        """Posted when the user picks ``Switch to`` for a row.
        The host app maps `record.kind` + `record.metadata` to
        the right destination screen."""

        def __init__(self, record: TaskRecord) -> None:
            super().__init__()
            self.record = record

    def __init__(
        self,
        registry: TaskRegistry,
        *,
        active_only: bool = True,
    ) -> None:
        super().__init__()
        self._registry = registry
        self.active_only = active_only
        self.records: tuple[TaskRecord, ...] = ()
        self._unsubscribe: Any = None
        # Selected row id — updated by DataTable.RowSelected.
        self.selected_task_id: str | None = None
        # Last cancel target — exposed for tests.
        self.last_cancelled_id: str | None = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="task-list-box"):
            yield Static(
                self._title_text(), id="task-list-title",
            )
            yield DataTable(id="task-list-table")
            with Horizontal(id="task-list-actions"):
                yield Button(t("common.close"), id="task-list-btn-close")
                yield Button(
                    t("taskList.cancelSelected"),
                    id="task-list-btn-cancel",
                )
                yield Button(
                    t("taskList.switchTo"),
                    id="task-list-btn-switch",
                    variant="primary",
                )

    def _title_text(self) -> str:
        suffix = t("taskList.activeSuffix") if self.active_only else ""
        return t("taskList.title", suffix=suffix)

    def on_mount(self) -> None:
        try:
            table = self.query_one("#task-list-table", DataTable)
            table.add_columns("", "Kind", "Label", "Status", "Elapsed")
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        self._refresh()
        # Subscribe to registry changes for live updates.
        try:
            self._unsubscribe = self._registry.on_change(
                self._on_registry_event,
            )
        except Exception:
            self._unsubscribe = None

    def on_unmount(self) -> None:
        unsub = self._unsubscribe
        if callable(unsub):
            try:
                unsub()
            except Exception:
                pass
        self._unsubscribe = None

    # ------------------------------------------------------------------
    # Refresh + render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        try:
            self.records = tuple(
                self._registry.list_tasks(active_only=self.active_only)
            )
        except Exception:
            self.records = ()
        self._render_table()

    def _render_table(self) -> None:
        try:
            table = self.query_one("#task-list-table", DataTable)
        except Exception:
            return
        try:
            table.clear()
        except Exception:
            pass
        for record in self.records:
            table.add_row(*self._row_cells(record), key=record.id)

    @staticmethod
    def _row_cells(record: TaskRecord) -> tuple[str, ...]:
        badge = _STATUS_BADGES.get(record.status, "?")
        duration = record.duration_seconds
        elapsed = (
            f"{duration:.1f}s"
            if duration is not None
            else ("running" if record.status == "running" else "—")
        )
        return (
            badge,
            record.kind,
            record.label[:40],
            record.status,
            elapsed,
        )

    def _on_registry_event(self, event_kind, record):
        """Listener — fires from worker threads, so hop back to
        the Textual loop before mutating widgets."""
        try:
            self.app.call_from_thread(self._refresh)
        except Exception:
            # The screen may have been unmounted between the
            # event firing and the call landing — best-effort.
            try:
                self._refresh()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Selection + actions
    # ------------------------------------------------------------------

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected,
    ) -> None:
        if event.data_table.id != "task-list-table":
            return
        try:
            self.selected_task_id = str(event.row_key.value or "")
        except Exception:
            self.selected_task_id = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "task-list-btn-close":
            self.action_close()
        elif bid == "task-list-btn-cancel":
            self.action_cancel_selected()
        elif bid == "task-list-btn-switch":
            self.action_switch_to_selected()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_cancel_selected(self) -> None:
        task_id = self.selected_task_id or self._first_task_id()
        if task_id is None:
            return
        try:
            self._registry.cancel(task_id)
        except Exception:
            return
        self.last_cancelled_id = task_id
        self._refresh()

    def action_switch_to_selected(self) -> None:
        task_id = self.selected_task_id or self._first_task_id()
        if task_id is None:
            return
        record = next(
            (r for r in self.records if r.id == task_id), None,
        )
        if record is None:
            return
        self.post_message(self.SwitchRequested(record))
        self.dismiss(None)

    def _first_task_id(self) -> str | None:
        return self.records[0].id if self.records else None


__all__ = ["TaskListDrawer"]
