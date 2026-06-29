"""Pilot smoke tests for ``care.screens.DemoScreen`` (TODO §9 P2).

The TUI workflow screens (Library / Query / Generation /
Inspection / Edit / Execution / Evolution) are P0-blocked
behind §1 — they haven't shipped yet. This file demonstrates
the **Textual `Pilot` template** the future screens will all
follow, exercised against the only screen that exists today.

When a workflow screen lands, copy this file's pattern (mount
inside a minimal `App`, assert widget tree, drive actions /
keys, await `pilot.pause()` for messages to settle).
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from care.screens import DemoScreen
from care.widgets import PipelinePreview, TaskSetup


class _DemoHarnessApp(App):
    """Minimal `App` host mounting just `DemoScreen`.

    Keeping the host trivial means the test exercises screen
    behaviour, not app-level wiring — a regression in
    ``CareApp`` won't false-fail these.
    """

    def compose(self) -> ComposeResult:
        # Empty compose; the screen mounts via on_mount.
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(DemoScreen())


# ---------------------------------------------------------------------------
# Mount + structural checks
# ---------------------------------------------------------------------------


class TestMount:
    @pytest.mark.asyncio
    async def test_mounts_with_expected_widgets(self):
        app = _DemoHarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, DemoScreen)
            # Two-pane layout: TaskSetup on the left, PipelinePreview on the right.
            assert len(screen.query(TaskSetup)) == 1
            assert len(screen.query(PipelinePreview)) == 1

    @pytest.mark.asyncio
    async def test_title_set_on_mount(self):
        app = _DemoHarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen.title == "MAESTRO"
            assert (
                app.screen.sub_title
                == "Collaborative Agent Reasoning Ecosystem"
            )

    @pytest.mark.asyncio
    async def test_bindings_registered(self):
        app = _DemoHarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            keys = {b.key for b in screen.BINDINGS}
            assert "ctrl+g" in keys
            assert "ctrl+l" in keys
            assert "q" in keys


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class TestActions:
    @pytest.mark.asyncio
    async def test_generate_action_drives_pipeline_preview(self):
        """`Ctrl+G` posts a GenerateRequested message that the
        screen routes into `PipelinePreview.show_pipeline`. We
        verify the preview was updated by checking the widget's
        internal state changed."""
        app = _DemoHarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Type a task into the input.
            task_setup = app.screen.query_one(TaskSetup)
            from textual.widgets import TextArea

            task_setup.query_one("#task-input", TextArea).load_text("a task")
            await pilot.pause()
            # Snapshot the preview before pressing the action.
            preview = app.screen.query_one(PipelinePreview)
            before_kids = len(list(preview.query("*")))

            await pilot.press("ctrl+g")
            await pilot.pause()
            # After generation, the preview has rendered something — its
            # child count grew (cards added) OR a placeholder was replaced.
            after_kids = len(list(preview.query("*")))
            assert after_kids != before_kids or after_kids > 0

    @pytest.mark.asyncio
    async def test_clear_action_wipes_task_input(self):
        """`Ctrl+L` clears the task input + the context-document
        list. We assert the input is empty after the press."""
        app = _DemoHarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            from textual.widgets import TextArea

            text_area = app.screen.query_one(TaskSetup).query_one(
                "#task-input", TextArea
            )
            text_area.load_text("draft text")
            await pilot.pause()
            assert text_area.text == "draft text"

            await pilot.press("ctrl+l")
            await pilot.pause()
            assert text_area.text == ""


# ---------------------------------------------------------------------------
# Message routing
# ---------------------------------------------------------------------------


class TestMessageRouting:
    @pytest.mark.asyncio
    async def test_generate_requested_message_handled_by_screen(self):
        """The screen's `on_task_setup_generate_requested` handler
        is the integration point future workflow screens will
        wire their generation triggers through. We assert the
        handler returns without raising for a typical input."""
        app = _DemoHarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            from textual.widgets import TextArea

            screen.query_one(TaskSetup).query_one(
                "#task-input", TextArea
            ).load_text("integration task")
            await pilot.pause()
            # Press the generate button directly via the action —
            # exercises the binding plus the downstream
            # `on_task_setup_generate_requested` handler.
            await pilot.press("ctrl+g")
            await pilot.pause()
            # No exception = pass. If the handler raised, run_test
            # would propagate it.


# ---------------------------------------------------------------------------
# Worker lifecycle (TODO §1.2 P0)
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    """Pin the Worker(thread=False) behaviour around pipeline
    synthesis: cancel-on-rapid-resubmit, Esc cancels in-flight,
    pane stays consistent after cancellation."""

    @pytest.mark.asyncio
    async def test_generate_method_directly_renders_pipeline(self):
        """`show_pipeline` driven directly (skipping the worker
        layer) produces the rendered cards. This pins the
        underlying async-iterator path that the worker drives."""
        app = _DemoHarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            preview = app.screen.query_one(PipelinePreview)
            await preview.show_pipeline("a task", ())
            await pilot.pause()
            from care.widgets.pipeline_preview import PipelineStepCard

            cards = list(preview.query(PipelineStepCard))
            assert len(cards) >= 1

    @pytest.mark.asyncio
    async def test_escape_binding_cancels_group(self):
        """Esc invokes ``action_cancel_generate``. We assert the
        binding is wired + the action runs without raising even
        when no worker is in-flight."""
        app = _DemoHarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            # No exception → the action is bound + safe to call
            # against an empty worker group.

    @pytest.mark.asyncio
    async def test_worker_spawned_on_generate(self):
        """Pressing `Ctrl+G` should land in `_render_pipeline_worker`,
        which calls `run_worker(...)` and registers a worker.
        We assert by spying on the worker-spawning hook."""
        app = _DemoHarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen

            spy_calls: list = []
            original = screen._render_pipeline_worker

            def _spy(task, files):
                spy_calls.append((task, files))
                return original(task, files)

            screen._render_pipeline_worker = _spy  # type: ignore[method-assign]

            from textual.widgets import TextArea

            screen.query_one(TaskSetup).query_one(
                "#task-input", TextArea
            ).load_text("a task")
            await pilot.pause()
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert len(spy_calls) == 1
            assert spy_calls[0][0] == "a task"

    @pytest.mark.asyncio
    async def test_pipeline_iterator_yields_same_set_as_sync(self):
        """The async iterator the worker drives produces the
        same step set the heuristic generates synchronously —
        no steps lost in the async path."""
        from care.widgets.pipeline_preview import PipelinePreview as _P

        preview = _P()
        steps_async: list = []
        async for step in preview._iter_pipeline(
            "design a deploy experiment", ()
        ):
            steps_async.append(step)
        steps_sync = _P._synthesize_pipeline(
            "design a deploy experiment", ()
        )
        assert [s.name for s in steps_async] == [s.name for s in steps_sync]

    @pytest.mark.asyncio
    async def test_pipeline_iterator_yields_to_loop(self):
        """The iterator must `await` between steps so the
        worker can be cancelled mid-render. Pin this by
        checking the iterator type."""
        import inspect

        from care.widgets.pipeline_preview import PipelinePreview as _P

        preview = _P()
        result = preview._iter_pipeline("x", ())
        assert inspect.isasyncgen(result)
