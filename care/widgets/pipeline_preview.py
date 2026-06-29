"""Pipeline preview — visualizes the generated agent pipeline.

Rendering runs through an async iterator + a Textual ``Worker``
so a slow synthesis (or a future MAGE-driven generation) doesn't
block the event loop and successive ``Ctrl+G`` presses cancel
the previous worker (TODO §1.2 P0).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Label, Static

from care.runtime.i18n import t


@dataclass(frozen=True)
class PipelineStep:
    name: str
    role: str
    inputs: tuple[str, ...]
    accent: str  # CSS color name


class PipelineStepCard(Static):
    DEFAULT_CSS = """
    PipelineStepCard {
        height: auto;
        padding: 0 1;
        border: round $primary 40%;
        background: $panel;
    }
    PipelineStepCard .step-name {
        text-style: bold;
    }
    PipelineStepCard .step-role {
        color: $text-muted;
    }
    PipelineStepCard .step-inputs {
        color: $accent;
    }
    """

    def __init__(self, step: PipelineStep, index: int) -> None:
        super().__init__()
        self.step = step
        self.index = index

    def compose(self) -> ComposeResult:
        yield Label(f"[{self.index}] {self.step.name}", classes="step-name")
        yield Label(self.step.role, classes="step-role")
        if self.step.inputs:
            yield Label("← " + ", ".join(self.step.inputs), classes="step-inputs")


class PipelineArrow(Static):
    DEFAULT_CSS = """
    PipelineArrow {
        height: 1;
        content-align: center middle;
        color: $accent;
    }
    """

    def __init__(self) -> None:
        super().__init__("▼")


class PipelinePreview(Widget):
    """Right pane — shows the generated agent pipeline."""

    DEFAULT_CSS = """
    PipelinePreview {
        width: 1fr;
        padding: 1 2;
        border-left: tall $primary 30%;
    }
    PipelinePreview > .pane-title {
        text-style: bold;
        color: $accent;
    }
    PipelinePreview > .pane-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    PipelinePreview > #pipeline-body {
        height: 1fr;
    }
    PipelinePreview .empty {
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label(t("pipelinePreview.title"), classes="pane-title")
        yield Label(
            t("pipelinePreview.hint"),
            classes="pane-hint",
        )
        with VerticalScroll(id="pipeline-body"):
            yield Static(
                t("pipelinePreview.emptyState"),
                classes="empty",
                id="empty-state",
            )

    async def show_pipeline(
        self, task: str, files: tuple[Path, ...]
    ) -> None:
        """Render the synthesised pipeline incrementally.

        The synthesis runs through :meth:`_iter_pipeline` — an
        async iterator that yields one :class:`PipelineStep` at
        a time with a small ``await asyncio.sleep(0)`` between
        yields. The await both gives the event loop a chance to
        process cancellation **and** mirrors the future MAGE
        stage-by-stage flow (`on_stage_started`, etc.) so the
        screen widget pattern stays uniform across the heuristic
        path and the real-MAGE path.

        Cancellable: when the enclosing Textual worker is
        cancelled (Ctrl+G again, Esc, etc.), the `CancelledError`
        propagates here, the partially-rendered pane stays
        intact, and the caller's worker handles the rest.
        """
        body = self.query_one("#pipeline-body", VerticalScroll)
        await body.remove_children()

        header_lines = []
        if task:
            preview = task if len(task) <= 70 else task[:67] + "..."
            header_lines.append(t("pipelinePreview.taskLine", task=preview))
        header_lines.append(t("pipelinePreview.contextLine", count=len(files)))
        await body.mount(
            Static("\n".join(header_lines), classes="pane-hint", id="run-header")
        )

        previous: PipelineStep | None = None
        index = 0
        async for step in self._iter_pipeline(task, files):
            index += 1
            if previous is not None:
                await body.mount(PipelineArrow())
            await body.mount(PipelineStepCard(step, index))
            previous = step

    async def _iter_pipeline(
        self, task: str, files: tuple[Path, ...]
    ) -> AsyncIterator[PipelineStep]:
        """Yield each synthesised :class:`PipelineStep` with a
        small await between, so the worker can be cancelled
        mid-render and the event loop stays responsive.

        Wraps :meth:`_synthesize_pipeline` — the heuristic
        itself stays a pure synchronous function so it's
        trivially testable.
        """
        for step in self._synthesize_pipeline(task, files):
            # Yield control to the loop so cancellation can
            # land + the UI can repaint.
            await asyncio.sleep(0)
            yield step

    @staticmethod
    def _synthesize_pipeline(
        task: str, files: tuple[Path, ...]
    ) -> list[PipelineStep]:
        """Toy heuristic — picks plausible agents based on task keywords.

        Real orchestration logic will replace this later.
        """
        task_lower = task.lower()
        file_names = tuple(p.name for p in files)
        context_ref = ("task brief",) + (file_names if file_names else ())

        steps: list[PipelineStep] = [
            PipelineStep(
                "Planner",
                "Decompose the task into ordered sub-goals.",
                ("task brief",),
                "cyan",
            )
        ]

        if files:
            steps.append(
                PipelineStep(
                    "Ingestor",
                    "Parse, chunk and embed the context documents.",
                    file_names,
                    "magenta",
                )
            )
            steps.append(
                PipelineStep(
                    "Researcher",
                    "Retrieve evidence from the ingested corpus.",
                    ("plan", "embeddings"),
                    "magenta",
                )
            )
        else:
            steps.append(
                PipelineStep(
                    "Researcher",
                    "Gather background from general knowledge.",
                    ("plan",),
                    "magenta",
                )
            )

        if any(k in task_lower for k in ("experiment", "propose", "design", "hypothes")):
            steps.append(
                PipelineStep(
                    "Hypothesis Generator",
                    "Draft candidate hypotheses or experiment designs.",
                    ("research notes",),
                    "yellow",
                )
            )

        steps.append(
            PipelineStep(
                "Critic",
                "Stress-test claims, surface weak assumptions.",
                ("drafts",),
                "yellow",
            )
        )
        steps.append(
            PipelineStep(
                "Synthesizer",
                "Merge perspectives into the final answer.",
                ("critique", "research notes"),
                "green",
            )
        )

        if any(k in task_lower for k in ("execute", "run ", "implement", "deploy", "apply")):
            steps.append(
                PipelineStep(
                    "Executor",
                    "Carry out the approved plan in the target system.",
                    ("final plan",),
                    "bright_blue",
                )
            )
        else:
            steps.append(
                PipelineStep(
                    "Reporter",
                    "Format the deliverable for the user.",
                    ("synthesis",),
                    "bright_blue",
                )
            )

        # context_ref is unused for now — kept to show how context flows.
        _ = context_ref
        return steps
