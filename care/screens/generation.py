"""GenerationScreen — split layout consuming MagePoster events
(TODO §1.1 P0.16).

Pushed by `QueryScreen` after the user submits a task. Hosts:

* **Left pane** — per-stage status log (`pending` / `running` /
  `done` / `failed`) driven by the
  :class:`care.runtime.mage_poster` message family
  (`StageStarted`, `StageCompleted`, `StageError`,
  `StageRetry`). Each row carries the stage label + the
  elapsed time since `StageStarted`.
* **Right pane** — incremental DAG preview that fills in as
  `StageProgress` artifacts land (one row per artifact). The
  Inspection screen reads the final chain from
  ``last_completion``; until that lands, the right pane is
  the user-visible record of what got built.

The screen is wired entirely through `MagePoster` messages so a
test can fire them by hand and assert the rendered state — no
real MAGE process required. The actual worker that fires the
messages lands alongside `QueryScreen.GenerateRequested`
plumbing once the MAGE bring-up worker is wired (later
sub-task); the screen accepts the events from any source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Label, Static

from care.mage_summary import MetadataSummary, summarise_mage_result
from care.runtime.i18n import t
from care.runtime.mage_poster import (
    StageCompleted,
    StageError,
    StageProgress,
    StageRetry,
    StageStarted,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


StageStatus = Literal["pending", "running", "done", "failed"]
"""Canonical states the left pane renders. `pending` is the
default before any event lands; `running` is set on
`StageStarted`; `done` on `StageCompleted`; `failed` on
`StageError`."""


@dataclass
class StageState:
    """One stage's bookkeeping for the left pane."""

    name: str
    status: StageStatus = "pending"
    started_at: float | None = None
    ended_at: float | None = None
    error: str | None = None
    retries: int = 0

    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    def format_row(self) -> str:
        elapsed = self.elapsed()
        elapsed_str = f"{elapsed:0.1f}s" if elapsed is not None else "--"
        badge = _STATUS_BADGE.get(self.status, "·")
        retry_str = f" (retry {self.retries})" if self.retries else ""
        suffix = f" — {self.error}" if self.error else ""
        return f"{badge} {self.name}  {elapsed_str}{retry_str}{suffix}"


_STATUS_BADGE: dict[StageStatus, str] = {
    "pending": "·",
    "running": "▶",
    "done": "✓",
    "failed": "✗",
}


@dataclass
class GenerationProgress:
    """Aggregate the screen exposes for tests + the future
    SaveAgentModal handoff. Snapshots the per-stage state +
    the streaming-DAG artifact list."""

    stages: dict[str, StageState] = field(default_factory=dict)
    artifacts: list[tuple[str, Any]] = field(default_factory=list)

    @property
    def has_failure(self) -> bool:
        return any(s.status == "failed" for s in self.stages.values())

    @property
    def is_complete(self) -> bool:
        return bool(self.stages) and all(
            s.status in {"done", "failed"} for s in self.stages.values()
        )


class GenerationScreen(Screen):
    """Watch a MAGE run unfold via :class:`MagePoster` events.

    The screen owns no worker — callers pass it the same
    :class:`MagePoster` target (it accepts the messages
    posted via `Screen.post_message`). The split layout is
    composed in :meth:`compose`; per-event state mutations
    live in the `on_stage_*` handlers.
    """

    DEFAULT_CSS = """
    GenerationScreen {
        layout: vertical;
    }
    GenerationScreen #generation-body {
        height: 1fr;
    }
    GenerationScreen #generation-stages {
        width: 1fr;
        padding: 1 2;
    }
    GenerationScreen #generation-dag {
        width: 2fr;
        padding: 1 2;
        border-left: solid $primary 30%;
    }
    GenerationScreen .pane-title {
        text-style: bold;
        color: $accent;
    }
    GenerationScreen .pane-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    GenerationScreen #stage-rows {
        height: 1fr;
    }
    GenerationScreen #dag-rows {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel_generate", "Cancel", show=True),
    ]

    def __init__(self, *, task_preview: str = "") -> None:
        super().__init__()
        self._task_preview = task_preview
        self.progress = GenerationProgress()
        # P0.17 cancel state — flips `True` when the user
        # presses `Esc`. Worker callbacks and tests read this
        # to gate post-cancel side effects (toast, telemetry).
        self.cancelled: bool = False
        # MAGE metadata summary — projected on chain
        # completion via :func:`summarise_mage_result` and
        # rendered into `#generation-metadata` (the §1.2
        # `[DONE — data layer]` metadata bullet's UI half).
        self.metadata_summary: MetadataSummary | None = None

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Horizontal(id="generation-body"):
            with Vertical(id="generation-stages"):
                yield Label(t("generation.stages"), classes="pane-title")
                if self._task_preview:
                    yield Label(self._task_preview, classes="pane-hint")
                yield VerticalScroll(id="stage-rows")
            with Vertical(id="generation-dag"):
                yield Label(t("generation.streamingDag"), classes="pane-title")
                yield Label(
                    t("generation.dagHint"),
                    classes="pane-hint",
                )
                yield VerticalScroll(id="dag-rows")
        yield Static("", id="generation-metadata")
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="GenerationScreen",
                breadcrumb=(t("header.breadcrumb.library"), t("header.breadcrumb.generate")),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="GenerationScreen",
                scope="screen",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Stage log (left pane)
    # ------------------------------------------------------------------

    def _stage_row_id(self, stage: str) -> str:
        # IDs only allow [a-zA-Z0-9_-]; squash everything else.
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in stage)
        return f"stage-row-{slug}"

    def _upsert_stage(self, stage: str) -> StageState:
        state = self.progress.stages.get(stage)
        if state is None:
            state = StageState(name=stage)
            self.progress.stages[stage] = state
        return state

    def _render_stage_row(self, state: StageState) -> None:
        if not self.is_mounted:
            return
        try:
            container = self.query_one("#stage-rows", VerticalScroll)
        except Exception:
            return
        row_id = self._stage_row_id(state.name)
        try:
            existing = container.query_one(f"#{row_id}", Static)
            existing.update(state.format_row())
        except Exception:
            container.mount(Static(state.format_row(), id=row_id))

    def on_stage_started(self, event: StageStarted) -> None:
        state = self._upsert_stage(event.stage)
        state.status = "running"
        state.started_at = time.monotonic()
        state.ended_at = None
        state.error = None
        self._render_stage_row(state)

    def on_stage_completed(self, event: StageCompleted) -> None:
        state = self._upsert_stage(event.stage)
        state.status = "done"
        state.ended_at = time.monotonic()
        self._render_stage_row(state)

    def on_stage_error(self, event: StageError) -> None:
        state = self._upsert_stage(event.stage)
        state.status = "failed"
        state.ended_at = time.monotonic()
        state.error = f"{type(event.error).__name__}: {event.error}"
        self._render_stage_row(state)

    def on_stage_retry(self, event: StageRetry) -> None:
        state = self._upsert_stage(event.stage)
        state.retries = event.attempt
        self._render_stage_row(state)

    # ------------------------------------------------------------------
    # DAG preview (right pane)
    # ------------------------------------------------------------------

    def on_stage_progress(self, event: StageProgress) -> None:
        artifact = event.artifact
        self.progress.artifacts.append((event.stage, artifact))
        if not self.is_mounted:
            return
        try:
            container = self.query_one("#dag-rows", VerticalScroll)
        except Exception:
            return
        container.mount(
            Static(self._format_artifact_row(event.stage, artifact))
        )

    @staticmethod
    def _format_artifact_row(stage: str, artifact: Any) -> str:
        label = _read_artifact_label(artifact)
        return f"[{stage}] {label}" if label else f"[{stage}]"

    # ------------------------------------------------------------------
    # MAGE result handoff (§1.2 metadata footer)
    # ------------------------------------------------------------------

    def record_mage_result(self, result: Any) -> None:
        """Stamp the finished MAGE generation onto the
        screen.

        The host worker (the MAGE bring-up that wires
        :class:`MagePoster`) calls this with the
        :class:`mmar_mage.MAGEResult` after
        :meth:`MAGEGenerator.generate` returns. The screen
        projects via :func:`care.summarise_mage_result` and
        renders the multi-line summary into the
        `#generation-metadata` Static so the user sees the
        domain / stages / quality block before the
        SaveAgentModal pops."""
        try:
            self.metadata_summary = summarise_mage_result(result)
        except Exception:
            self.metadata_summary = None
            return
        self._render_metadata()

    def _render_metadata(self) -> None:
        if not self.is_mounted:
            return
        try:
            target = self.query_one("#generation-metadata", Static)
        except Exception:
            return
        if self.metadata_summary is None:
            target.update("")
            return
        target.update(self.metadata_summary.format_text())

    # ------------------------------------------------------------------
    # Cancel binding (P0.17)
    # ------------------------------------------------------------------

    def action_cancel_generate(self) -> None:
        """`Esc` → cancel every worker in the ``generate``
        group. Matches the convention shipped on
        :class:`DemoScreen` (and the future ExecutionScreen)
        so the cancel gesture stays uniform across MAGE / CARL
        screens. Stays on the screen so the user can read
        partial state; an explicit `BackRequested` /
        `pop_screen` is the way out.

        The :attr:`cancelled` flag flips true when the user
        invokes this — tests + future telemetry read it
        without scraping worker state."""
        self.cancelled = True
        try:
            self.workers.cancel_group(self, "generate")
        except Exception:
            pass


def _read_artifact_label(artifact: Any) -> str:
    """Best-effort one-liner for the right-pane artifact row.

    MAGE's `StageProgress.artifact` varies by stage — pydantic
    models (`StepPlan`, `DAGStructure`, `CARLStepSchema`), dicts,
    or bare strings all flow through. The function reads the
    most common label fields without importing MAGE-side types.
    """
    if artifact is None:
        return ""
    if isinstance(artifact, str):
        return artifact
    for attr in ("name", "step_name", "label", "title", "stage"):
        value = getattr(artifact, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(artifact, dict):
        for key in ("name", "step_name", "label", "title"):
            value = artifact.get(key)
            if isinstance(value, str) and value:
                return value
    return type(artifact).__name__


__all__ = [
    "GenerationProgress",
    "GenerationScreen",
    "StageState",
    "StageStatus",
]
