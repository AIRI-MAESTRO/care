"""SaveReport modal — post-mortem for save-all batches
(TODO §3 P1).

Pushed by :meth:`ArtifactsScreen._save_all_worker` when the
batch was large (≥ 5 artifacts) or any row failed. For
smaller all-success batches the existing per-row + summary
toasts are sufficient.

The modal renders a DataTable of `(title, status, entity_id,
error)` rows so the user can scan every artifact's outcome
in one view. Bindings:

* ``Esc`` / close button — dismiss.
* ``Enter`` — show the highlighted row's UseItNowModal
  (only meaningful for successful saves; surfaces a toast
  for failures).

Pure presentation — no Memory side-effects, no retry plumbing
yet (filed as a §3 P2 follow-up).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Label, Static

from care.runtime.i18n import t
from care.screens._animated_modal import AnimatedModalScreen
from care.widgets.footer import CareFooter

_log = logging.getLogger("care.screen.save_report")


_COLUMNS: tuple[str, ...] = (
    "Title",
    "Status",
    "Entity ID",
    "Error",
)


@dataclass(frozen=True)
class SaveReportRow:
    """One row in the report — the artifact's title, the
    save outcome (`success` / `failure`), the resulting
    Memory entity_id when known, and a friendly error
    string (empty on success)."""

    artifact_id: str
    title: str
    status: str  # "success" | "failure"
    entity_id: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "success"


@dataclass(frozen=True)
class SaveReportResult:
    """Dismiss envelope. ``show_id`` carries the artifact id
    the user picked via Enter (for the host to open
    UseItNowModal). Empty when the user dismissed without
    selecting."""

    closed: bool = True
    show_id: str = ""


class SaveReport(AnimatedModalScreen[SaveReportResult]):
    """Post-mortem view of a save-all batch.

    Construct with the list of :class:`SaveReportRow`
    snapshots the worker accumulated; the modal renders
    them in arrival order so the user reads through the
    same sequence the save attempted.
    """

    DEFAULT_CSS = """
    SaveReport {
        align: center middle;
    }
    SaveReport #save-report-box {
        width: 100;
        max-width: 95%;
        height: 30;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    SaveReport #save-report-title {
        text-style: bold;
        padding-bottom: 1;
    }
    SaveReport #save-report-summary {
        color: $accent;
        margin-bottom: 1;
    }
    SaveReport #save-report-table {
        height: 1fr;
    }
    SaveReport #save-report-actions {
        height: 3;
        align-horizontal: right;
        margin-top: 1;
    }
    SaveReport Button {
        margin-left: 1;
    }
    """

    ANIM_BOX_ID = "save-report-box"

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("enter", "show_id", "Show ID", show=True),
    ]

    def __init__(
        self,
        rows: tuple[SaveReportRow, ...] | list[SaveReportRow],
    ) -> None:
        super().__init__()
        self.rows: tuple[SaveReportRow, ...] = tuple(rows)
        self.action_log: list[tuple[str, str]] = []

    @property
    def saved_count(self) -> int:
        return sum(1 for r in self.rows if r.ok)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.rows if not r.ok)

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="save-report-box"):
            yield Label(
                t("saveReport.title"), id="save-report-title",
            )
            yield Static(
                self._summary_text(),
                id="save-report-summary",
            )
            yield DataTable(id="save-report-table")
            with Horizontal(id="save-report-actions"):
                yield Button(t("common.close"), id="save-report-btn-close")
        yield CareFooter()

    def on_mount(self) -> None:
        self._animate_modal_in()
        try:
            table = self.query_one(
                "#save-report-table", DataTable,
            )
            for col in _COLUMNS:
                table.add_column(col, key=col)
            table.cursor_type = "row"
            table.zebra_stripes = True
            for row in self.rows:
                table.add_row(
                    row.title or "(untitled)",
                    _format_status(row),
                    _format_entity_id(row.entity_id),
                    _format_error(row.error),
                    key=row.artifact_id,
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summary_text(self) -> str:
        if not self.rows:
            return t("saveReport.noAttempts")
        total = len(self.rows)
        if self.failed_count == 0:
            return t("saveReport.savedAll", total=total)
        return t(
            "saveReport.savedSome",
            saved=self.saved_count,
            total=total,
            failed=self.failed_count,
        )

    @property
    def current_row(self) -> SaveReportRow | None:
        if not self.rows:
            return None
        try:
            table = self.query_one(
                "#save-report-table", DataTable,
            )
        except Exception:
            return None
        idx = table.cursor_row
        if idx is None or idx < 0 or idx >= len(self.rows):
            return None
        return self.rows[idx]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_close(self) -> None:
        self.action_log.append(("close", ""))
        self.dismiss(SaveReportResult(closed=True, show_id=""))

    def action_show_id(self) -> None:
        row = self.current_row
        if row is None:
            return
        if not row.ok:
            self.action_log.append(("show_id_failed", row.artifact_id))
            self._toast(
                t("saveReport.rowFailed"),
                severity="info",
            )
            return
        if not row.entity_id:
            self.action_log.append(
                ("show_id_no_entity", row.artifact_id),
            )
            self._toast(
                t("saveReport.noEntityId"),
                severity="info",
            )
            return
        self.action_log.append(
            ("show_id", row.artifact_id),
        )
        self.dismiss(SaveReportResult(
            closed=False, show_id=row.artifact_id,
        ))

    def on_button_pressed(
        self, event: Button.Pressed,
    ) -> None:
        if event.button.id == "save-report-btn-close":
            self.action_close()

    def _toast(self, message: str, *, severity: str = "info") -> None:
        push = getattr(self.app, "push_toast", None)
        if callable(push):
            try:
                push(message, severity=severity)
                return
            except Exception:
                pass
        _log.info(
            "SaveReport toast [%s]: %s", severity, message,
        )


# ---------------------------------------------------------------------------
# Pure formatters
# ---------------------------------------------------------------------------


def _format_status(row: SaveReportRow) -> str:
    badge = "✓" if row.ok else "✗"
    return f"{badge} {row.status}"


def _format_entity_id(entity_id: str) -> str:
    if not entity_id:
        return "—"
    if len(entity_id) <= 16:
        return entity_id
    return entity_id[:14] + "…"


def _format_error(error: str) -> str:
    if not error:
        return ""
    if len(error) <= 56:
        return error
    return error[:53] + "…"


__all__ = [
    "SaveReport",
    "SaveReportResult",
    "SaveReportRow",
]
