"""Demo screen — task setup → generated agent pipeline.

A scratch layout for the CARE TUI: user describes a task on the left,
picks context documents from local files, then generates an agent
pipeline that appears on the right.

Pipeline rendering runs through a Textual ``Worker(thread=False)``
so the event loop stays responsive while synthesis runs and a
fresh Ctrl+G mid-flight cancels the previous render (TODO §1.2 P0).
Future MAGE-driven generation lands on the same worker shape — only
the work function changes.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header
from textual.worker import Worker

from care.widgets import PipelinePreview, TaskSetup


class DemoScreen(Screen):
    """Two-pane demo: task setup on the left, pipeline preview on the right."""

    CSS_PATH = "../styles/demo.tcss"

    BINDINGS = [
        Binding("ctrl+g", "generate", "Generate pipeline"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("escape", "cancel_generate", "Cancel"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-row"):
            yield TaskSetup(id="task-setup")
            yield PipelinePreview(id="pipeline-preview")
        yield Footer()

    def on_mount(self) -> None:
        from care.app import CareApp

        self.title = CareApp.TITLE
        self.sub_title = CareApp.SUB_TITLE

    def on_task_setup_generate_requested(
        self, event: TaskSetup.GenerateRequested
    ) -> None:
        """Kick the synthesis worker.

        Returns control to Textual immediately; the actual
        rendering runs inside :meth:`_render_pipeline_worker`.
        ``exclusive=True`` means a fresh Ctrl+G while a
        previous worker is still running cancels the old one
        before launching the new one — same UX `Worker` gives
        every future MAGE-driven screen.
        """
        self._render_pipeline_worker(event.task, event.files)

    # `@work` provides the worker decorator; using the imperative
    # `run_worker` keeps the screen Textual-version-agnostic.
    def _render_pipeline_worker(
        self, task: str, files: tuple[Path, ...]
    ) -> Worker:
        preview = self.query_one(PipelinePreview)
        return self.run_worker(
            preview.show_pipeline(task, files),
            name="pipeline_synthesis",
            group="generate",
            exclusive=True,
            exit_on_error=False,
        )

    def action_generate(self) -> None:
        self.query_one(TaskSetup).query_one("#btn-generate").press()

    def action_clear(self) -> None:
        self.query_one(TaskSetup).query_one("#btn-clear").press()

    def action_cancel_generate(self) -> None:
        """Cancel any in-flight synthesis worker — wired to
        ``Esc``. Matches the convention future MAGE-driven
        screens will use (GenerationScreen / ExecutionScreen
        bind Esc to cancel via the same group)."""
        self.workers.cancel_group(self, "generate")
