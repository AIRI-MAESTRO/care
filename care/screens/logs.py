"""LogsScreen — in-app log viewer (TODO §6 P2).

Tails the active app log file (`CARE_LOG_FILE` env var or
the attached `care-app-file` handler's path) and shows
the most-recent records inside the TUI. Replaces the
`make run LOG=1` + external editor dance.

Bindings:

* ``r`` — re-read the log file.
* ``l`` — cycle the level floor (ALL → DEBUG → INFO →
  WARNING → ERROR → ALL).
* ``Esc`` — pop the screen.

The viewer reads up to ``DEFAULT_TAIL_LINES`` lines off the
file end (5000 by default) and renders them inside a
:class:`VerticalScroll`. Each line lands as its own
:class:`Static` so the scroll snaps to the bottom on
refresh (matches the StatusBar's append semantics).
"""

from __future__ import annotations

import logging
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Input, Static

from care.runtime.i18n import t
from care.runtime.log_discovery import (
    active_log_path,
    tail_log_lines,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader

_log = logging.getLogger("care.screen.logs")


_LEVEL_CYCLE: tuple[str | None, ...] = (
    None, "DEBUG", "INFO", "WARNING", "ERROR",
)
"""Cycle order for the `l` binding. ``None`` means show
every line."""


class LogsScreen(Screen):
    """In-app log viewer."""

    DEFAULT_TAIL_LINES: int = 5000

    DEFAULT_CSS = """
    LogsScreen {
        layout: vertical;
    }
    LogsScreen #logs-meta {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    LogsScreen #logs-body {
        height: 1fr;
        padding: 0 1;
    }
    LogsScreen #logs-empty {
        padding: 1 2;
        color: $text-muted;
    }
    LogsScreen #logs-status {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    LogsScreen #logs-filter-input {
        height: auto;
        margin: 0 1;
        display: none;
    }
    LogsScreen #logs-filter-input.-visible {
        display: block;
    }
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("l", "cycle_level", "Level", show=True),
        Binding("m", "toggle_module_filter", "Module filter", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(
        self,
        *,
        log_path: Path | None = None,
        max_lines: int | None = None,
    ) -> None:
        super().__init__()
        self._explicit_path = log_path
        self._max_lines = (
            self.DEFAULT_TAIL_LINES if max_lines is None
            else max_lines
        )
        self.level_floor: str | None = None
        self.module_substr: str = ""
        self.lines: list[str] = []
        self.last_error: str | None = None
        self.resolved_path: Path | None = None
        self.action_log: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Vertical():
            yield Static(" ", id="logs-meta")
            yield Input(
                placeholder=t("logs.filterPlaceholder"),
                id="logs-filter-input",
            )
            with VerticalScroll(id="logs-body"):
                yield Static(
                    t("common.loading"), id="logs-content", markup=False,
                )
        yield Static(" ", id="logs-status")
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="LogsScreen",
                breadcrumb=(t("logs.breadcrumb"),),
            )
        except Exception:
            pass
        self.app.call_after_refresh(self.refresh_log)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_log(self) -> None:
        """Re-read the log file + repaint.

        Resolution order: explicit `log_path` kwarg →
        :func:`active_log_path` (env / handler). Empty
        result lands as the empty-state placeholder.
        """
        self.resolved_path = (
            self._explicit_path or active_log_path()
        )
        if self.resolved_path is None:
            self.lines = []
            self.last_error = None
            self._apply_view()
            return
        try:
            self.lines = tail_log_lines(
                self.resolved_path,
                max_lines=self._max_lines,
                level_floor=self.level_floor,
                module_substr=self.module_substr,
            )
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.lines = []
            self.last_error = f"{type(exc).__name__}: {exc}"
        self._apply_view()

    def action_refresh(self) -> None:
        self.action_log.append(("refresh", ""))
        self.refresh_log()

    def action_back(self) -> None:
        self.action_log.append(("back", ""))
        try:
            self.app.pop_screen()
        except Exception:
            pass

    def action_cycle_level(self) -> None:
        idx = _LEVEL_CYCLE.index(self.level_floor)
        nxt = _LEVEL_CYCLE[(idx + 1) % len(_LEVEL_CYCLE)]
        self.level_floor = nxt
        label = nxt or "ALL"
        self.action_log.append(("cycle_level", label))
        self.refresh_log()

    def action_toggle_module_filter(self) -> None:
        """`m` — toggle the inline module-filter Input. When
        opening, focuses the input + pre-fills with the
        current filter; when closing, clears the filter +
        refreshes."""
        try:
            inp = self.query_one("#logs-filter-input", Input)
        except Exception:
            return
        if inp.has_class("-visible"):
            # Hiding → clear filter.
            inp.remove_class("-visible")
            cleared = bool(self.module_substr)
            self.module_substr = ""
            self.action_log.append(
                ("toggle_module_filter", "hidden"),
            )
            if cleared:
                self.refresh_log()
        else:
            inp.add_class("-visible")
            inp.value = self.module_substr
            inp.focus()
            self.action_log.append(
                ("toggle_module_filter", "visible"),
            )

    def on_input_submitted(
        self, event: "Input.Submitted",
    ) -> None:
        """Apply the module filter from the inline Input."""
        if event.input.id != "logs-filter-input":
            return
        new_substr = (event.value or "").strip()
        changed = new_substr != self.module_substr
        self.module_substr = new_substr
        self.action_log.append(
            ("apply_module_filter", new_substr),
        )
        # Hide the input after submit so the body reclaims
        # the row; user can re-open with `m` if they want
        # to refine.
        try:
            inp = self.query_one("#logs-filter-input", Input)
            inp.remove_class("-visible")
        except Exception:
            pass
        if changed:
            self.refresh_log()
        else:
            self._apply_view()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _apply_view(self) -> None:
        try:
            meta = self.query_one("#logs-meta", Static)
            body = self.query_one("#logs-body", VerticalScroll)
            content = self.query_one("#logs-content", Static)
            status = self.query_one("#logs-status", Static)
        except Exception:
            return
        meta.update(self._meta_text())
        if self.last_error:
            content.update(f"⚠ {self.last_error}")
        elif not self.lines:
            empty_msg = (
                t("logs.emptyNoFile")
                if self.resolved_path is None
                else t(
                    "logs.emptyFiltered", path=self.resolved_path,
                )
            )
            content.update(empty_msg)
        else:
            # Lines may carry markup chars (square brackets,
            # backticks) that Textual's Static renderer would
            # otherwise interpret as console markup. The Static
            # was constructed with `markup=False` so the body
            # renders verbatim.
            content.update("\n".join(self.lines))
            try:
                body.scroll_end(animate=False)
            except Exception:
                pass
        status.update(self._status_text())

    def _meta_text(self) -> str:
        if self.resolved_path is None:
            return t("logs.metaNone")
        return t("logs.metaPath", path=self.resolved_path)

    def _status_text(self) -> str:
        if self.last_error:
            return f"⚠ {self.last_error}"
        floor = self.level_floor or "ALL"
        line_count = len(self.lines)
        lines_key = (
            "logs.statusLinesOne"
            if line_count == 1
            else "logs.statusLinesMany"
        )
        parts = [
            t(lines_key, n=line_count),
            t("logs.statusLevel", floor=floor),
        ]
        if self.module_substr:
            parts.append(
                t("logs.statusModule", substr=self.module_substr),
            )
        parts.append(t("logs.statusHint"))
        return " · ".join(parts)


__all__ = [
    "LogsScreen",
]
