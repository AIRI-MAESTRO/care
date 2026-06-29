"""ReplayScreen — step through a stored `ReasoningResult`
(TODO §6 replay mode).

Pushed by the InspectionScreen's `Run history` row dispatch
when the user picks an entry, or by the `care replay` CLI
subcommand. Wraps the shipped
:func:`care.load_replay(source)` → :class:`ReplaySession`
data layer and adds keyboard navigation + per-step render.

* `Right` / `n` → next step
* `Left` / `p` → previous step
* `Home` / `r` → restart
* `Esc` → pop

The screen is purely presentational; the host owns the
source (a `memory_card` payload, a `RunRecord`, or a
`ReasoningResult` directly).
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Static

from care.replay import ReplaySession, ReplayStep, load_replay
from care.runtime.i18n import t
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


class ReplayScreen(Screen):
    """Step-through inspector for a saved chain run.

    Construct with ``source`` (any
    :func:`care.load_replay`-accepted shape). The screen
    builds the session, renders the step list + a detail
    pane, and exposes keyboard navigation."""

    DEFAULT_CSS = """
    ReplayScreen {
        layout: vertical;
    }
    ReplayScreen #replay-body {
        height: 1fr;
    }
    ReplayScreen #replay-steps {
        width: 1fr;
        padding: 1 2;
    }
    ReplayScreen #replay-detail {
        width: 2fr;
        padding: 1 2;
    }
    ReplayScreen .pane-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    ReplayScreen #replay-status {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("right", "next_step", "Next", show=True),
        Binding("n", "next_step", "Next", show=False),
        Binding("left", "previous_step", "Prev", show=True),
        Binding("p", "previous_step", "Prev", show=False),
        Binding("home", "restart", "Restart", show=True),
        Binding("r", "restart", "Restart", show=False),
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(self, source: Any = None) -> None:
        super().__init__()
        if isinstance(source, ReplaySession):
            self.session: ReplaySession = source
            return
        try:
            self.session = load_replay(source)
        except Exception:
            self.session = ReplaySession()

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Horizontal(id="replay-body"):
            with Vertical(id="replay-steps"):
                yield Label(t("replay.steps"), classes="pane-title")
                yield ListView(id="replay-step-list")
            with Vertical(id="replay-detail"):
                yield Label(t("replay.detail"), classes="pane-title")
                yield VerticalScroll(id="replay-detail-body")
        yield Static("", id="replay-status")
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="ReplayScreen",
                breadcrumb=(
                    t("header.breadcrumb.library"),
                    t("header.breadcrumb.replay"),
                    self.session.chain_title or self.session.chain_id or "",
                ),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="ReplayScreen",
                scope="screen",
            )
        except Exception:
            pass
        self._refresh_panes()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def action_next_step(self) -> None:
        self.session.next()
        self._refresh_panes()

    def action_previous_step(self) -> None:
        self.session.previous()
        self._refresh_panes()

    def action_restart(self) -> None:
        self.session.restart()
        self._refresh_panes()

    def action_back(self) -> None:
        try:
            self.app.pop_screen()
        except Exception:
            pass

    def on_list_view_highlighted(
        self, event: ListView.Highlighted,
    ) -> None:
        if event.list_view.id != "replay-step-list":
            return
        idx = event.list_view.index
        if isinstance(idx, int):
            self.session.seek(idx)
            self._render_detail()
            self._render_status()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh_panes(self) -> None:
        # Don't gate on `self.is_mounted` — Screen's flag stays
        # `False` during `on_mount` even though children are
        # already queryable (same gotcha hit in
        # CommandPaletteModal). Each sub-renderer does its own
        # `query_one` probe and bails if the widget isn't
        # composed yet.
        self._render_step_list()
        self._render_detail()
        self._render_status()

    def _render_step_list(self) -> None:
        try:
            listview = self.query_one("#replay-step-list", ListView)
        except Exception:
            return
        try:
            listview.clear()
        except Exception:
            pass
        if self.session.is_empty:
            listview.append(ListItem(Label(t("replay.noSteps"))))
            return
        for idx, step in enumerate(self.session.steps):
            label = self._step_label(idx, step)
            listview.append(ListItem(Label(label)))

    @staticmethod
    def _step_label(index: int, step: ReplayStep) -> str:
        badge = "✓"
        if step.skipped:
            badge = "·"
        elif not step.success:
            badge = "✗"
        title = step.step_title or step.step_type or f"step-{index + 1}"
        return f"{badge} {title}"

    def _render_detail(self) -> None:
        try:
            container = self.query_one(
                "#replay-detail-body", VerticalScroll,
            )
        except Exception:
            return
        try:
            for child in list(container.children):
                child.remove()
        except Exception:
            pass
        current = self.session.current()
        if current is None:
            container.mount(Static(t("replay.noStepSelected")))
            return
        elapsed = (
            f"{current.execution_time_s:.3f}s"
            if current.execution_time_s is not None
            else "—"
        )
        status = (
            t("replay.statusOk") if current.success
            else (
                t("replay.statusSkipped") if current.skipped
                else t("replay.statusFailed")
            )
        )
        lines = [
            f"step: {current.step_number}",
            f"type: {current.step_type or '—'}",
            f"title: {current.step_title or '—'}",
            f"status: {status}",
            f"time: {elapsed}",
        ]
        if current.error_message:
            lines.append(f"error: {current.error_message}")
        if current.result_preview:
            lines.append("")
            lines.append("result:")
            lines.append(current.result_preview)
            if current.result_truncated:
                lines.append(t("replay.truncated"))
        container.mount(Static("\n".join(lines)))

    def _render_status(self) -> None:
        try:
            target = self.query_one("#replay-status", Static)
        except Exception:
            return
        if self.session.is_empty:
            target.update(t("replay.emptySession"))
            return
        target.update(
            f"step {self.session.cursor + 1} / {self.session.step_count}"
            + (
                f"  ·  chain {self.session.chain_id}"
                if self.session.chain_id else ""
            )
        )


__all__ = ["ReplayScreen"]
