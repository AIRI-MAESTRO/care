"""ExecutionScreen — render a live CARL chain run
(TODO §1.1 P0.22).

Pushed by the "Run" action on a saved agent (via
:class:`RunContextModal` for re-runs or directly from
LibraryScreen for the "Replay with same inputs" gesture).
Consumes the :mod:`care.runtime.carl_streamer` message family
to render three panes:

* **Step log** — one row per ``StepStarted`` / ``StepCompleted``
  with status badge + duration + step title.
* **LLM stream** — accumulating chunks for the focused step
  (any ``LlmChunk`` whose ``step_number`` matches).
* **Telemetry** — chain-level progress + token / step counts.

The screen is wired entirely through `CarlStreamer` messages
so tests can fire them by hand. The real worker driving the
events lives elsewhere — the host pushes the screen with a
`RunPlan` from `load_run_plan`, and on_mount fires the
`execute_library_run` worker (data layer already shipped).
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

from care.profiling import ProfilingSummary, project_profiling
from care.runtime.i18n import t
from care.runtime.carl_streamer import (
    ChainCompleted,
    HumanInputRequested,
    LlmChunk,
    Progress,
    StepCompleted,
    StepStarted,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


StepStatus = Literal["pending", "running", "done"]

# Braille spinner frames for the in-flight step (animated by the screen's
# `_tick_spinner` interval).
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class StepRecord:
    """Per-step bookkeeping the screen renders."""

    step_number: int
    title: str = ""
    status: StepStatus = "pending"
    started_at: float | None = None
    ended_at: float | None = None
    result_summary: str = ""

    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    def format_row(self, *, spinner: str = "▶") -> str:
        # The running step shows an animated spinner frame (passed in by the
        # screen's ticker) instead of a static glyph, so it's obvious which
        # step is in flight.
        badge = {"pending": "·", "running": spinner, "done": "✓"}[self.status]
        elapsed = self.elapsed()
        elapsed_str = f"{elapsed:0.1f}s" if elapsed is not None else "--"
        return f"{badge} step {self.step_number} {self.title}  {elapsed_str}"


@dataclass
class ExecutionState:
    """Aggregate the screen exposes to tests + downstream
    SaveAgentModal handoff."""

    steps: dict[int, StepRecord] = field(default_factory=dict)
    completed: int = 0
    total: int = 0
    finished: bool = False
    chain_result: Any = None
    pending_human_prompt: str | None = None

    @property
    def progress_fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.completed / self.total)


class ExecutionScreen(Screen):
    """Live run renderer.

    Construct with a ``title`` (used in the header / breadcrumb)
    + an optional ``total_steps`` hint so the progress bar
    starts with a denominator. The screen ships the bindings
    and the message subscriptions; the actual worker that
    drives the chain lives in the host."""

    DEFAULT_CSS = """
    ExecutionScreen {
        layout: vertical;
    }
    ExecutionScreen #execution-body {
        height: 1fr;
    }
    ExecutionScreen #execution-steps {
        width: 1fr;
        padding: 1 2;
    }
    ExecutionScreen #execution-dag-section {
        height: auto;
        max-height: 55%;
        margin-bottom: 1;
    }
    ExecutionScreen #execution-dag {
        height: auto;
        max-height: 14;
    }
    ExecutionScreen #execution-dag-text {
        width: auto;
    }
    ExecutionScreen #execution-stream {
        width: 2fr;
        padding: 1 2;
    }
    ExecutionScreen #execution-stream-body {
        width: 1fr;
        height: 1fr;
    }
    /* Fill the column so streamed text wraps at the full pane width
       instead of breaking early and leaving the rest of the column
       blank. */
    ExecutionScreen #execution-stream-text {
        width: 1fr;
    }
    ExecutionScreen #execution-telemetry {
        width: 1fr;
        padding: 1 2;
    }
    ExecutionScreen .pane-title {
        text-style: bold;
        color: $accent;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel_execute", "Cancel", show=True),
        Binding("m", "cycle_dag_overlay", "DAG: status/latency", show=True),
    ]

    def __init__(
        self,
        *,
        title: str = "Execution",
        total_steps: int = 0,
        chain_steps: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self.state = ExecutionState(total=total_steps)
        # Projected step dicts (number / title / step_type / dependencies)
        # the live DAG overlay draws. Empty → the FLOW pane stays hidden,
        # so re-runs that can't surface a chain shape degrade gracefully
        # to the flat step log.
        self._chain_steps: list[dict] = list(chain_steps) if chain_steps else []
        # DAG overlay mode: "status" (live per-step state) or "latency"
        # (post-run wall-clock heat). Flipped with the `m` binding.
        self._dag_overlay: str = "status"
        self.focused_step: int | None = None
        # P0.22 cancel flag (mirrors GenerationScreen pattern).
        self.cancelled: bool = False
        # Per-step LLM-stream text, keyed by step number. The centre pane
        # renders ALL steps as a scrollable transcript ("── Step N ──"
        # headers + each step's output) so the user can review the run
        # step by step — not just the final step's stream.
        self._step_streams: dict[int, str] = {}
        # Profiling summary (§5 [DONE — data layer] → fully
        # DONE). Populated by `record_chain_result(result)`
        # after the chain finishes; rendered into the
        # `#execution-profiling` Static below the telemetry
        # pane.
        self.profiling: ProfilingSummary = ProfilingSummary()
        # Braille spinner cursor for the currently-running step row; advanced
        # by the `_tick_spinner` interval, which runs only while a step is
        # actually running (see `_sync_spinner_timer`).
        self._spinner_idx: int = 0
        # Handle for the adaptive spinner interval. `None` while no step is
        # running (or under reduced motion) so the screen doesn't repaint
        # 10x/sec once the run has finished.
        self._spinner_timer: Any = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Horizontal(id="execution-body"):
            with Vertical(id="execution-steps"):
                with Vertical(id="execution-dag-section"):
                    yield Label(
                        t("execution.flow"),
                        classes="pane-title",
                        id="execution-flow-title",
                    )
                    with VerticalScroll(id="execution-dag"):
                        yield Static("", id="execution-dag-text", markup=False)
                yield Label(t("execution.steps"), classes="pane-title")
                yield VerticalScroll(id="execution-step-rows")
            with Vertical(id="execution-stream"):
                yield Label(t("execution.llmStream"), classes="pane-title")
                with VerticalScroll(id="execution-stream-body"):
                    yield Static("", id="execution-stream-text", markup=False)
            with Vertical(id="execution-telemetry"):
                yield Label(t("execution.telemetry"), classes="pane-title")
                yield Static(t("execution.stepCount"), id="execution-telemetry-text")
                yield Static("", id="execution-profiling")
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="ExecutionScreen",
                breadcrumb=(t("header.breadcrumb.library"), self._title),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="ExecutionScreen",
                scope="screen",
            )
        except Exception:
            pass
        self._refresh_telemetry()
        # Animate the running step's spinner + keep its elapsed clock ticking.
        # Adaptive: the interval only runs while a step is actually running
        # (and motion is enabled), so an idle / finished run doesn't repaint
        # 10x/sec. Started/stopped from the step-state transitions.
        self._sync_spinner_timer()
        # The DAG overlay is only meaningful when we know the chain's
        # shape; otherwise hide the pane so the left column is just the
        # flat step log (the re-run shape that predates this overlay).
        if not self._chain_steps:
            try:
                self.query_one("#execution-dag-section").display = False
            except Exception:
                pass
            return
        # Paint after the first refresh: pre-layout the pane width is 0,
        # which would mis-size the graph (and the initial paint at mount
        # time doesn't stick). One refresh later the width is real.
        self.call_after_refresh(self._render_dag)

    # ------------------------------------------------------------------
    # CarlStreamer message handlers
    # ------------------------------------------------------------------

    def on_step_started(self, event: StepStarted) -> None:
        rec = self.state.steps.get(event.step_number)
        if rec is None:
            rec = StepRecord(step_number=event.step_number)
            self.state.steps[event.step_number] = rec
        rec.title = event.step_title or rec.title
        rec.status = "running"
        rec.started_at = time.monotonic()
        rec.ended_at = None
        self.focused_step = event.step_number
        # Register the step in the transcript (so its header shows even
        # before any output streams in).
        self._step_streams.setdefault(event.step_number, "")
        self._render_step_row(rec)
        self._render_stream()
        self._refresh_telemetry()
        self._render_dag()
        # A step is now running — start the spinner interval (motion-gated).
        self._sync_spinner_timer()

    def on_step_completed(self, event: StepCompleted) -> None:
        step_number = self._infer_step_number(event.result)
        if step_number is None and self.focused_step is not None:
            step_number = self.focused_step
        if step_number is None:
            return
        rec = self.state.steps.get(step_number)
        if rec is None:
            rec = StepRecord(step_number=step_number)
            self.state.steps[step_number] = rec
        rec.status = "done"
        rec.ended_at = time.monotonic()
        rec.result_summary = self._summarise_step_result(event.result)
        self.state.completed = sum(
            1 for r in self.state.steps.values() if r.status == "done"
        )
        self._step_streams.setdefault(step_number, "")
        self._render_step_row(rec)
        # Re-render so a step that streamed nothing (tool / code steps)
        # shows its result summary in the transcript.
        self._render_stream()
        self._refresh_telemetry()
        self._render_dag()
        # This step finished — stop the spinner if nothing else is running.
        self._sync_spinner_timer()

    def on_chain_completed(self, event: ChainCompleted) -> None:
        self.state.finished = True
        self.state.chain_result = event.result
        self.record_chain_result(event.result)
        self._refresh_telemetry()
        # Profiling is now available, so a latency overlay can paint.
        self._render_dag()
        # Run is done — make sure the spinner interval is stopped.
        self._sync_spinner_timer()

    def record_chain_result(self, result: Any) -> None:
        """Project the chain's `ReasoningResult` into the
        per-step profiling pane.

        Called from `on_chain_completed`; exposed publicly so
        the host can replay a stored
        :class:`mmar_carl.RunRecord` against the screen
        without firing a Textual message."""
        try:
            self.profiling = project_profiling(result)
        except Exception:
            return
        self._render_profiling()

    def _render_profiling(self) -> None:
        if not self.is_mounted:
            return
        try:
            target = self.query_one("#execution-profiling", Static)
        except Exception:
            return
        if self.profiling.is_empty:
            target.update("")
            return
        target.update(self.profiling.format_text())

    def on_progress(self, event: Progress) -> None:
        self.state.completed = event.completed
        if event.total > 0:
            self.state.total = event.total
        self._refresh_telemetry()

    def on_llm_chunk(self, event: LlmChunk) -> None:
        step_number = event.step_number
        if step_number is not None and step_number != self.focused_step:
            # Stream is per-step; only the focused step's
            # chunks land in the centre pane.
            return
        # Attribute the chunk to its step (falling back to the focused
        # step when the chunk carries no number).
        step = step_number if step_number is not None else self.focused_step
        if step is None:
            return
        self._step_streams[step] = self._step_streams.get(step, "") + event.chunk
        self._render_stream()

    def _render_stream(self) -> None:
        """Render every step as a scrollable transcript so the run reads
        step by step: a ``── Step N ──`` header per step followed by its
        streamed output (or its result summary when the step streamed
        nothing — e.g. tool / code steps). One markup-free Static so raw
        model output with ``[...]`` can't raise MarkupError."""
        if not self.is_mounted:
            return
        try:
            target = self.query_one("#execution-stream-text", Static)
        except Exception:
            return
        sections: list[str] = []
        for n in sorted(self._step_streams):
            rec = self.state.steps.get(n)
            title = (rec.title if rec and rec.title else "").strip()
            header = t("execution.stepHeader", n=n)
            if title:
                header = f"{header} · {title}"
            body = self._step_streams.get(n, "")
            if not body.strip():
                body = (
                    rec.result_summary
                    if rec and rec.result_summary
                    else t("execution.noOutput")
                )
            sections.append(f"── {header} ──\n{body}")
        target.update("\n\n".join(sections))
        try:
            self.query_one(
                "#execution-stream-body", VerticalScroll,
            ).scroll_end(animate=False)
        except Exception:
            pass

    def on_human_input_requested(
        self, event: HumanInputRequested,
    ) -> None:
        """Surface the request so tests + future modal wiring
        can read it. The actual modal push lands when the
        HumanInputModal ships (later sub-task)."""
        self.state.pending_human_prompt = event.prompt

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def action_cancel_execute(self) -> None:
        """`Esc` → cancel the in-flight run and return to the previous
        screen (Inspect / Library). Previously this only flagged
        ``cancelled`` + cancelled the local ``execute`` worker group,
        leaving the user stranded on the run screen with no way back."""
        self.cancelled = True
        try:
            self.workers.cancel_group(self, "execute")
        except Exception:
            pass
        # The real run worker lives on the app (group ``library_run``,
        # spawned by ``CareApp._push_run_for``); cancel it too so a
        # half-finished run stops driving this now-departing screen.
        try:
            self.app.workers.cancel_group(self.app, "library_run")
        except Exception:
            pass
        # Pop back to whatever pushed us. Guarded: popping the last
        # screen raises, which we treat as a no-op (e.g. test hosts).
        try:
            self.app.pop_screen()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _spinner_frame(self) -> str:
        return _SPINNER_FRAMES[self._spinner_idx % len(_SPINNER_FRAMES)]

    def _motion_enabled(self) -> bool:
        """True when the app permits the spinner to animate. Reduced-motion
        (``animation_level == "none"``) freezes the spinner on its first
        frame — the running indicator still shows, it just doesn't spin."""
        try:
            return getattr(self.app, "animation_level", "none") != "none"
        except Exception:
            return False

    def _has_running_step(self) -> bool:
        return any(
            rec.status == "running" for rec in self.state.steps.values()
        )

    def _sync_spinner_timer(self) -> None:
        """Run the spinner interval only while a step is running AND motion is
        enabled; stop/standby it otherwise so a finished or idle run stops
        repainting 10x/sec (mirrors ChatScreen's `_sync_status_anim_timer`).
        """
        want = self._has_running_step() and self._motion_enabled()
        timer = self._spinner_timer
        if want and timer is None:
            try:
                self._spinner_timer = self.set_interval(0.1, self._tick_spinner)
            except Exception:
                self._spinner_timer = None
        elif not want and timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            self._spinner_timer = None

    def _tick_spinner(self) -> None:
        """Advance the spinner + re-render every running step row so the
        in-flight step shows a live loading animation (and its elapsed clock
        keeps counting). Stops the timer when nothing is running."""
        running = [
            rec for rec in self.state.steps.values() if rec.status == "running"
        ]
        if not running:
            self._sync_spinner_timer()
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER_FRAMES)
        for rec in running:
            self._render_step_row(rec)

    def _render_step_row(self, rec: StepRecord) -> None:
        if not self.is_mounted:
            return
        try:
            container = self.query_one(
                "#execution-step-rows", VerticalScroll,
            )
        except Exception:
            return
        row_id = f"execution-step-{rec.step_number}"
        text = rec.format_row(spinner=self._spinner_frame())
        try:
            existing = container.query_one(f"#{row_id}", Static)
            existing.update(text)
        except Exception:
            # markup=False: step titles can carry `[...]` (e.g. config
            # reprs / indexed names) that aren't Rich markup.
            container.mount(Static(text, id=row_id, markup=False))

    def _render_dag(self) -> None:
        """Paint the chain's DAG tinted by live per-step status, so the
        run reads as a graph lighting up — pending steps muted, the
        running step amber, finished steps green. Best-effort: an
        unprojectable chain leaves the (hidden) pane empty rather than
        breaking the run."""
        if not self.is_mounted or not self._chain_steps:
            return
        try:
            target = self.query_one("#execution-dag-text", Static)
        except Exception:
            return
        from rich.text import Text

        from care.runtime.dag_view import dag_display_opts, render_dag_styled

        # Latency overlay once profiling exists; otherwise live status.
        overlay: dict[str, Any] = {}
        if self._dag_overlay == "latency" and not self.profiling.is_empty:
            from care.profiling import profiling_metric

            overlay["metric_by_ref"] = profiling_metric(self.profiling, "time")
        else:
            overlay["status_by_ref"] = {
                str(num): rec.status for num, rec in self.state.steps.items()
            }
        try:
            lines = render_dag_styled(
                self._chain_steps,
                max_graph_width=self._dag_graph_width(),
                **overlay,
                **dag_display_opts(getattr(self.app, "config", None)),
            )
        except Exception:  # noqa: BLE001
            lines = []
        target.update(Text("\n").join(lines) if lines else "")

    def action_cycle_dag_overlay(self) -> None:
        """`m` flips the DAG between the live status tint and a post-run
        latency heat-map (slow steps red). No-op until the chain has
        finished and profiling is available."""
        if not self._chain_steps:
            return
        nxt = "latency" if self._dag_overlay == "status" else "status"
        if nxt == "latency" and self.profiling.is_empty:
            return  # nothing to heat-map yet
        self._dag_overlay = nxt
        suffix = t("execution.flowLatency") if nxt == "latency" else ""
        try:
            self.query_one("#execution-flow-title", Label).update(
                t("execution.flow") + suffix
            )
        except Exception:
            pass
        self._render_dag()

    def _dag_graph_width(self) -> int:
        """Width budget for the DAG before it collapses to the compact
        number-box + legend variant. Tracks the FLOW pane's live width
        (minus its gutter); falls back to a sane default pre-layout."""
        from care.runtime.dag_view import _DEFAULT_MAX_GRAPH_WIDTH

        try:
            width = int(self.query_one("#execution-dag").size.width)
        except Exception:
            return _DEFAULT_MAX_GRAPH_WIDTH
        if width <= 0:
            return _DEFAULT_MAX_GRAPH_WIDTH
        return max(20, width - 4)

    def _refresh_telemetry(self) -> None:
        if not self.is_mounted:
            return
        try:
            target = self.query_one(
                "#execution-telemetry-text", Static,
            )
        except Exception:
            return
        parts = [f"{self.state.completed}/{self.state.total or '?'} steps"]
        if self.state.finished:
            parts.append("done")
        elif self.cancelled:
            parts.append("cancelled")
        if self.state.pending_human_prompt:
            parts.append("waiting on user")
        target.update("  ·  ".join(parts))

    @staticmethod
    def _infer_step_number(result: Any) -> int | None:
        for attr in ("step_number", "step_index", "index", "step"):
            val = getattr(result, attr, None)
            if isinstance(val, int):
                return val
        if isinstance(result, dict):
            for key in ("step_number", "step_index", "index", "step"):
                v = result.get(key)
                if isinstance(v, int):
                    return v
        return None

    @staticmethod
    def _summarise_step_result(result: Any) -> str:
        if result is None:
            return ""
        for attr in ("summary", "title", "name", "output"):
            val = getattr(result, attr, None)
            if isinstance(val, str):
                return val[:80]
        if isinstance(result, dict):
            for key in ("summary", "title", "name", "output"):
                v = result.get(key)
                if isinstance(v, str):
                    return v[:80]
        return type(result).__name__


def project_chain_steps(chain: Any) -> list[dict]:
    """Project a loaded chain into plain step dicts the DAG renderer can
    consume — each carrying ``number`` / ``title`` / ``step_type`` /
    ``dependencies``.

    Tolerates the shapes CARE sees in practice: a CARL ``ReasoningChain``
    object (serialised via ``model_dump`` / ``to_dict``), a
    ``{"steps": [...]}`` mapping, or a bare list of step dicts/objects.
    Returns ``[]`` for an unrecognised shape so the live DAG overlay
    simply stays hidden instead of raising mid-run.
    """
    out: list[dict] = []
    for idx, step in enumerate(_raw_chain_steps(chain)):
        if isinstance(step, dict):
            out.append(step)
            continue
        projected = _dump_obj(step)
        if isinstance(projected, dict):
            out.append(projected)
            continue
        # Last resort — pull the handful of attributes the DAG needs.
        out.append(
            {
                "number": getattr(step, "number", idx + 1),
                "title": getattr(step, "title", "")
                or getattr(step, "name", ""),
                "step_type": getattr(step, "step_type", "")
                or getattr(step, "type", ""),
                "dependencies": list(getattr(step, "dependencies", []) or []),
            }
        )
    return out


def _raw_chain_steps(chain: Any) -> list:
    """Pull the raw step sequence off a chain in any of its shapes."""
    steps = getattr(chain, "steps", None)
    if isinstance(steps, (list, tuple)):
        return list(steps)
    if isinstance(chain, (list, tuple)):
        return list(chain)
    if isinstance(chain, dict):
        s = chain.get("steps")
        return list(s) if isinstance(s, (list, tuple)) else []
    dumped = _dump_obj(chain)
    if isinstance(dumped, dict):
        content = dumped.get("content")
        body = content if isinstance(content, dict) else dumped
        s = body.get("steps")
        return list(s) if isinstance(s, (list, tuple)) else []
    return []


def _dump_obj(obj: Any) -> Any:
    """Best-effort Pydantic/CARL serialisation of a chain or step object."""
    for serialiser in ("model_dump", "to_dict"):
        fn = getattr(obj, serialiser, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                continue
    return None


__all__ = [
    "ExecutionScreen",
    "ExecutionState",
    "StepRecord",
    "StepStatus",
    "project_chain_steps",
]
