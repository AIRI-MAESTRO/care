"""QueryScreen — task description + optional generation hints
(TODO §1.1 P0.15).

The "+ New agent" entry-point. Migrates the existing
:class:`TaskSetup` widget under a proper Textual `Screen`,
augments it with optional generation hints (domain, target
runtime, max steps), and routes both `Ctrl+G` and the embedded
"Generate pipeline" button into a single
:class:`QueryScreen.GenerateRequested` message the future
GenerationScreen (P0.16) listens for.

The message carries the full :class:`QuerySubmission` payload so
downstream screens don't need to re-query widgets. `Ctrl+L`
pops the screen — the canonical "back to library" gesture per
the §1.1 design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Checkbox,
    Input,
    RadioSet,
    TextArea,
)

from care.runtime.i18n import t
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader
from care.widgets.task_setup import TaskSetup


TargetRuntime = Literal["local", "docker", "e2b"]
"""Runtime the user wants MAGE to favour when generating a
chain. Mirrors the runtime registry shipped in CARL §5.1 —
the value rides on the request so the future GenerationScreen
can propagate it through `MAGEConfig`."""


MageMode = Literal["fast", "deep"]
"""MAGE generation mode. ``"deep"`` (default) runs the full
pipeline (memory research → plan → describe → critique →
verify → refine); ``"fast"`` short-circuits to single-shot
generation. The toggle rides on the
:class:`QuerySubmission` so the downstream worker forwards
to :func:`care.build_mage_generator(..., mode=)`."""


_RUNTIME_BY_BUTTON_ID = {
    "query-runtime-local": "local",
    "query-runtime-docker": "docker",
    "query-runtime-e2b": "e2b",
}
_BUTTON_ID_BY_RUNTIME = {v: k for k, v in _RUNTIME_BY_BUTTON_ID.items()}


@dataclass(frozen=True)
class QuerySubmission:
    """Frozen snapshot of every QueryScreen field at submit time.

    Frozen so the GenerateRequested message can flow through
    Textual's queue without defensive copies. Optional fields
    default to ``None`` / empty when the user leaves the chip
    unset; the downstream worker treats those as "MAGE picks".
    """

    task: str
    files: tuple[Path, ...] = ()
    domain_hint: str | None = None
    target_runtime: TargetRuntime = "local"
    max_steps: int | None = None
    mage_mode: MageMode = "deep"

    def has_task(self) -> bool:
        """``True`` when the task description is non-empty.
        Drives the future "Disable Generate when blank"
        button-state — exposed here so the future
        GenerationScreen can also gate the worker."""
        return bool(self.task.strip())


class QueryScreen(Screen):
    """+ New agent flow.

    Hosts :class:`TaskSetup` for task description + context
    files plus a short "Generation hints" pane for the
    optional fields. Submitting either via the embedded
    "Generate pipeline" button or `Ctrl+G` posts a single
    :class:`GenerateRequested` message.
    """

    DEFAULT_CSS = """
    QueryScreen {
        layout: vertical;
    }
    QueryScreen TaskSetup {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+g", "submit", "Generate", show=True),
        Binding("ctrl+l", "back_to_library", "Library", show=True),
    ]

    class GenerateRequested(Message):
        """Posted when the user submits the form (via
        `Ctrl+G` or the embedded button). Carries a frozen
        :class:`QuerySubmission` — every field as visible to
        the user at submit time."""

        def __init__(self, submission: QuerySubmission) -> None:
            super().__init__()
            self.submission = submission

    class BackRequested(Message):
        """Posted on `Ctrl+L`. The host app dismisses the
        screen — keeps the navigation gesture explicit so
        the future cross-screen telemetry has a single
        hook."""

    def __init__(
        self,
        *,
        initial_task: str | None = None,
        initial_runtime: TargetRuntime = "local",
        initial_mode: MageMode = "deep",
    ) -> None:
        super().__init__()
        self._initial_task = initial_task
        self._initial_runtime: TargetRuntime = initial_runtime
        self._initial_mode: MageMode = initial_mode
        # Last submitted :class:`QuerySubmission` — exposed so
        # tests + future telemetry can read the latest snapshot
        # without scraping widgets.
        self.last_submission: QuerySubmission | None = None

    def compose(self) -> ComposeResult:
        yield CareHeader()
        yield TaskSetup(
            id="query-task-setup",
            initial_runtime=self._initial_runtime,
            initial_mode=self._initial_mode,
        )
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="QueryScreen",
                breadcrumb=(t("header.breadcrumb.library"), t("header.breadcrumb.newAgent")),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="QueryScreen",
                scope="screen",
            )
        except Exception:
            pass
        if self._initial_task is not None:
            try:
                self.query_one("#task-input", TextArea).load_text(
                    self._initial_task,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Submission flow
    # ------------------------------------------------------------------

    def current_submission(self) -> QuerySubmission:
        """Read every field on the screen + return a frozen
        :class:`QuerySubmission`. Exposed at instance scope so
        tests + future telemetry can grab the snapshot without
        firing the message."""
        try:
            task_widget = self.query_one("#task-input", TextArea)
            task = task_widget.text.strip()
        except Exception:
            task = ""
        try:
            ts = self.query_one(TaskSetup)
            files = tuple(getattr(ts, "_files", ()) or ())
        except Exception:
            files = ()
        domain_hint = self._read_input("#query-domain-hint")
        max_steps = self._read_max_steps()
        runtime = self._read_runtime()
        mode = self._read_mage_mode()
        return QuerySubmission(
            task=task,
            files=files,
            domain_hint=domain_hint or None,
            target_runtime=runtime,
            max_steps=max_steps,
            mage_mode=mode,
        )

    def _read_mage_mode(self) -> MageMode:
        """Return ``"fast"`` when the Fast checkbox is set,
        else ``"deep"``. Drives
        :func:`care.build_mage_generator(..., mode=)` so the
        downstream worker picks the single-shot vs. deep
        pipeline."""
        try:
            cb = self.query_one("#query-mage-fast", Checkbox)
        except Exception:
            return self._initial_mode
        return "fast" if cb.value else "deep"

    def _read_input(self, selector: str) -> str:
        try:
            widget = self.query_one(selector, Input)
        except Exception:
            return ""
        return (widget.value or "").strip()

    def _read_max_steps(self) -> int | None:
        raw = self._read_input("#query-max-steps")
        if not raw:
            return None
        try:
            parsed = int(raw)
        except ValueError:
            return None
        if parsed <= 0:
            return None
        return parsed

    def _read_runtime(self) -> TargetRuntime:
        try:
            rs = self.query_one("#query-runtime", RadioSet)
        except Exception:
            return self._initial_runtime
        pressed = rs.pressed_button
        pressed_id = pressed.id if pressed is not None else None
        return _RUNTIME_BY_BUTTON_ID.get(
            pressed_id or "",
            self._initial_runtime,
        )

    def action_submit(self) -> None:
        """`Ctrl+G` → fire `GenerateRequested` with the current
        snapshot."""
        self._submit()

    def action_back_to_library(self) -> None:
        """`Ctrl+L` → emit `BackRequested` so the host app
        dismisses the screen."""
        self.post_message(self.BackRequested())

    def on_task_setup_generate_requested(
        self, event: TaskSetup.GenerateRequested,
    ) -> None:
        """The TaskSetup button still works — we re-emit the
        unified message with the hint fields filled in."""
        event.stop()
        self._submit()

    def _submit(self) -> None:
        submission = self.current_submission()
        self.last_submission = submission
        self.post_message(self.GenerateRequested(submission))


__all__ = [
    "MageMode",
    "QueryScreen",
    "QuerySubmission",
    "TargetRuntime",
]


# Re-export the field helper so future dataclass expansions don't
# need a separate import in this module (`max_steps` already uses
# default=None; the helper anchors the import).
_ = field
