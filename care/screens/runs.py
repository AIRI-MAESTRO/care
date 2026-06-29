"""RunsScreen — local run history viewer (TODO §6 P1).

Reads `~/.cache/care/runs/<YYYY-MM-DD>.jsonl` (one row per
recorded chain execution) and renders the rows newest-first
in a DataTable.

Bindings:

* ``r`` — re-read the cache directory.
* ``Esc`` — pop the screen.
* ``Enter`` — drill into the highlighted row (currently
  shows a friendly toast pointing at the run_id + chain_id;
  ReplayScreen integration is filed as a follow-up since
  the existing ReplayScreen consumes a per-chain log
  format).

The screen is read-only — the recording side lives in
:mod:`care.runtime.local_run_history`; this screen only
displays what's already on disk.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Static

from care.runtime.i18n import t
from care.runtime.local_run_history import (
    LocalRunEntry,
    load_local_runs,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader

_log = logging.getLogger("care.screen.runs")


_COLUMN_KEYS: tuple[str, ...] = (
    "When",
    "Chain",
    "Status",
    "Duration",
    "Tokens",
    "Cost",
)


def _columns() -> tuple[str, ...]:
    return (
        t("runs.colWhen"),
        t("runs.colChain"),
        t("runs.colStatus"),
        t("runs.colDuration"),
        t("runs.colTokens"),
        t("runs.colCost"),
    )


class RunsScreen(Screen):
    """Local run-history viewer.

    Construct without args; reads the cache root from
    :data:`CARE_CACHE_DIR` by default. Tests pass
    ``cache_dir=tmp_path`` to drive a controlled
    hierarchy.
    """

    DEFAULT_CSS = """
    RunsScreen {
        layout: vertical;
    }
    RunsScreen #runs-body {
        height: 1fr;
        padding: 0 1;
    }
    RunsScreen #runs-table {
        height: 1fr;
    }
    RunsScreen #runs-empty {
        padding: 1 2;
        color: $text-muted;
    }
    RunsScreen #runs-status {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("escape", "back", "Back", show=True),
        Binding("enter", "open_run", "Inspect", show=True),
    ]

    DEFAULT_LIMIT: int = 200
    """Cap rows so a heavy user's table stays responsive.
    Newest-first sort guarantees the latest runs win the
    cap."""

    def __init__(
        self, *, cache_dir: Any = None, limit: int | None = None,
    ) -> None:
        super().__init__()
        self._cache_dir = cache_dir
        self._limit = (
            self.DEFAULT_LIMIT if limit is None else limit
        )
        self.rows: list[LocalRunEntry] = []
        self.last_error: str | None = None
        self.action_log: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Vertical(id="runs-body"):
            yield DataTable(id="runs-table")
            yield Static(" ", id="runs-empty")
        yield Static(" ", id="runs-status")
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="RunsScreen",
                breadcrumb=(t("header.breadcrumb.runs"),),
            )
        except Exception:
            pass
        try:
            table = self.query_one("#runs-table", DataTable)
            for label, col_key in zip(_columns(), _COLUMN_KEYS):
                table.add_column(label, key=col_key)
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        # Defer the first refresh so query_one finds the body.
        self.app.call_after_refresh(self.refresh_rows)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_rows(self) -> None:
        """Reload rows from the cache directory + repaint.

        Runs synchronously since `load_local_runs` reads
        local files (no network). Heavy users (>200 rows)
        are capped at `DEFAULT_LIMIT` for table-render
        responsiveness; the cap is applied at the
        data-layer call so we don't slurp everything into
        memory only to throw it away.
        """
        try:
            self.rows = load_local_runs(
                cache_dir=self._cache_dir, limit=self._limit,
            )
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.rows = []
            self.last_error = f"{type(exc).__name__}: {exc}"
        self._apply_view()

    def action_refresh(self) -> None:
        self.action_log.append(("refresh", ""))
        self.refresh_rows()

    def action_back(self) -> None:
        self.action_log.append(("back", ""))
        try:
            self.app.pop_screen()
        except Exception:
            pass

    def action_open_run(self) -> None:
        run = self.current_run
        if run is None:
            return
        self.action_log.append(("open_run", run.run_id))
        # §6 P1 — drill into ReplayScreen when a replay
        # sidecar was written. Falls back to the legacy
        # info-toast when the row has no replay_path (older
        # runs / executor that crashed before producing a
        # `ReasoningResult`).
        if run.replay_path:
            try:
                from pathlib import Path as _Path

                from care.screens.replay import ReplayScreen

                # `load_replay` accepts JSON strings; read
                # the sidecar + push the screen with the body.
                body = _Path(run.replay_path).read_text(
                    encoding="utf-8",
                )
                self.app.push_screen(
                    ReplayScreen(source=body),
                )
                return
            except FileNotFoundError:
                self._toast(
                    t("runs.replayMissing", path=run.replay_path),
                    severity="warning",
                )
                return
            except Exception as exc:  # noqa: BLE001
                self._toast(
                    t("runs.replayFailed", error=exc),
                    severity="error",
                )
                return
        if run.chain_id:
            msg = t(
                "runs.inspectRunChain",
                run_id=run.run_id,
                chain_id=run.chain_id,
            )
        else:
            msg = t(
                "runs.inspectRun",
                run_id=run.run_id,
                chain_id="?",
            )
        self._toast(msg, severity="info")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def current_run(self) -> LocalRunEntry | None:
        if not self.rows:
            return None
        try:
            table = self.query_one("#runs-table", DataTable)
        except Exception:
            return None
        idx = table.cursor_row
        if idx is None or idx < 0 or idx >= len(self.rows):
            return None
        return self.rows[idx]

    def _apply_view(self) -> None:
        try:
            table = self.query_one("#runs-table", DataTable)
            empty = self.query_one("#runs-empty", Static)
            status = self.query_one("#runs-status", Static)
        except Exception:
            return
        table.clear()
        for row in self.rows:
            table.add_row(
                _format_when(row.started_at),
                _format_chain(row),
                _format_status(row),
                _format_duration(row.duration_seconds),
                _format_tokens(row),
                _format_cost(row.cost_usd),
                key=row.run_id,
            )
        is_empty = not self.rows
        empty.display = is_empty and not self.last_error
        if is_empty and not self.last_error:
            empty.update(t("runs.empty"))
        else:
            empty.update(" ")
        if self.last_error:
            status.update(f"⚠ {self.last_error}")
        else:
            count = len(self.rows)
            key = (
                "runs.statusOne" if count == 1 else "runs.statusMany"
            )
            status.update(t(key, n=count, limit=self._limit))

    def _toast(self, message: str, *, severity: str = "info") -> None:
        push = getattr(self.app, "push_toast", None)
        if callable(push):
            try:
                push(message, severity=severity)
                return
            except Exception:
                pass
        _log.info("RunsScreen toast [%s]: %s", severity, message)


# ---------------------------------------------------------------------------
# Pure formatters (testable without Textual)
# ---------------------------------------------------------------------------


def _format_when(started: float) -> str:
    if not started:
        return "—"
    return time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(started),
    )


def _format_chain(row: LocalRunEntry) -> str:
    if row.chain_name and row.chain_id:
        return f"{row.chain_name} ({row.chain_id[:10]})"
    return row.chain_name or row.chain_id or "—"


def _format_status(row: LocalRunEntry) -> str:
    badge = "✓" if row.status == "success" else "✗"
    return f"{badge} {row.status}"


def _format_duration(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 60:
        return f"{value:.1f}s"
    mins, secs = divmod(int(value), 60)
    return f"{mins}m {secs}s"


def _format_tokens(row: LocalRunEntry) -> str:
    total = row.tokens_total
    if total is None:
        return "—"
    return str(total)


def _format_cost(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.4f}"


__all__ = [
    "RunsScreen",
]
