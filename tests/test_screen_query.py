"""Pilot tests for QueryScreen (TODO §1.1 P0.15).

Exercises:
* Composition — `TaskSetup` + hint pane (`domain`, `max steps`,
  runtime radio set) all mount.
* `Ctrl+G` submits the form via :class:`GenerateRequested`.
* The embedded TaskSetup `Generate pipeline` button submits
  through the same message — no duplicate dispatch.
* Optional fields land on the submission verbatim.
* `Ctrl+L` posts :class:`BackRequested`.
* LibraryScreen `create_first_agent` CTA pushes QueryScreen.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, RadioSet, TextArea

from care.screens.library import LibraryScreen
from care.screens.query import QueryScreen, QuerySubmission, TargetRuntime
from care.widgets.task_setup import TaskSetup


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _QueryHost(App):
    def __init__(self, *, screen: QueryScreen | None = None) -> None:
        super().__init__()
        self._initial = screen or QueryScreen()
        self.submitted: list[QuerySubmission] = []
        self.back_requested: int = 0

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self._initial)

    def on_query_screen_generate_requested(
        self, event: QueryScreen.GenerateRequested,
    ) -> None:
        self.submitted.append(event.submission)

    def on_query_screen_back_requested(
        self, event: QueryScreen.BackRequested,
    ) -> None:
        self.back_requested += 1


# ---------------------------------------------------------------------------
# Submission dataclass
# ---------------------------------------------------------------------------


class TestQuerySubmission:
    def test_defaults(self):
        sub = QuerySubmission(task="hello")
        assert sub.task == "hello"
        assert sub.files == ()
        assert sub.domain_hint is None
        assert sub.target_runtime == "local"
        assert sub.max_steps is None
        assert sub.has_task() is True

    def test_blank_task_predicate(self):
        sub = QuerySubmission(task="   ")
        assert sub.has_task() is False


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_renders_task_setup_plus_hints(self):
        app = _QueryHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            assert screen.query_one(TaskSetup) is not None
            assert screen.query_one("#query-domain-hint", Input) is not None
            assert screen.query_one("#query-max-steps", Input) is not None
            assert screen.query_one("#query-runtime", RadioSet) is not None


# ---------------------------------------------------------------------------
# Submission flow
# ---------------------------------------------------------------------------


class TestSubmit:
    @pytest.mark.asyncio
    async def test_ctrl_g_submits(self):
        app = _QueryHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            screen.query_one("#task-input", TextArea).load_text("evaluate storms")
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            await pilot.pause()
            assert len(app.submitted) == 1
            assert app.submitted[0].task == "evaluate storms"
            assert app.submitted[0].target_runtime == "local"

    @pytest.mark.asyncio
    async def test_task_setup_button_routes_through_generate_requested(self):
        app = _QueryHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            screen.query_one("#task-input", TextArea).load_text("press button")
            await pilot.pause()
            screen.query_one("#btn-generate").press()
            await pilot.pause()
            await pilot.pause()
            # Exactly one GenerateRequested, even though
            # TaskSetup also emits its own message — QueryScreen
            # stops + re-emits with hints attached.
            assert len(app.submitted) == 1
            assert app.submitted[0].task == "press button"

    @pytest.mark.asyncio
    async def test_optional_fields_land_on_submission(self):
        app = _QueryHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            screen.query_one("#task-input", TextArea).load_text("with hints")
            screen.query_one("#query-domain-hint", Input).value = "weather"
            screen.query_one("#query-max-steps", Input).value = "12"
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            await pilot.pause()
            sub = app.submitted[0]
            assert sub.domain_hint == "weather"
            assert sub.max_steps == 12

    @pytest.mark.asyncio
    async def test_invalid_max_steps_drops_to_none(self):
        app = _QueryHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            screen.query_one("#query-max-steps", Input).value = "not-a-number"
            await pilot.pause()
            sub = screen.current_submission()
            assert sub.max_steps is None

    @pytest.mark.asyncio
    async def test_runtime_radio_changes_submission(self):
        initial: TargetRuntime = "docker"
        app = _QueryHost(screen=QueryScreen(initial_runtime=initial))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            screen.query_one("#task-input", TextArea).load_text("docker task")
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            await pilot.pause()
            assert app.submitted[0].target_runtime == "docker"

    @pytest.mark.asyncio
    async def test_last_submission_stored(self):
        app = _QueryHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            screen.query_one("#task-input", TextArea).load_text("snapshot")
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            assert screen.last_submission is not None
            assert screen.last_submission.task == "snapshot"


# ---------------------------------------------------------------------------
# Back-to-library
# ---------------------------------------------------------------------------


class TestBack:
    @pytest.mark.asyncio
    async def test_ctrl_l_posts_back_requested(self):
        app = _QueryHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            screen.action_back_to_library()
            await pilot.pause()
            await pilot.pause()
            assert app.back_requested == 1


# ---------------------------------------------------------------------------
# LibraryScreen integration
# ---------------------------------------------------------------------------


class TestLibraryIntegration:
    @pytest.mark.asyncio
    async def test_create_first_agent_pushes_query_screen(self):
        class _LibHost(App):
            memory = None

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(LibraryScreen(restore_state=False))

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            from textual.widgets import Button

            library = app.screen_stack[-1]
            assert isinstance(library, LibraryScreen)
            cta = library.query_one("#empty-state-cta", Button)
            cta.press()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], QueryScreen)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports_query_screen(self):
        from care.screens import QueryScreen as ReExported
        from care.screens import QuerySubmission as ReExportedSub

        assert ReExported is QueryScreen
        assert ReExportedSub is QuerySubmission


# ---------------------------------------------------------------------------
# MAGE Fast/Deep mode toggle
# ---------------------------------------------------------------------------


class TestMageMode:
    @pytest.mark.asyncio
    async def test_default_mode_is_deep(self):
        app = _QueryHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            screen.query_one("#task-input", TextArea).load_text("hello")
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            await pilot.pause()
            assert app.submitted[0].mage_mode == "deep"

    @pytest.mark.asyncio
    async def test_fast_checkbox_flips_mode(self):
        from textual.widgets import Checkbox

        app = _QueryHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            screen.query_one(
                "#query-mage-fast", Checkbox,
            ).value = True
            screen.query_one("#task-input", TextArea).load_text("fast")
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            await pilot.pause()
            assert app.submitted[0].mage_mode == "fast"

    @pytest.mark.asyncio
    async def test_initial_mode_kwarg_sets_default(self):
        app = _QueryHost(screen=QueryScreen(initial_mode="fast"))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, QueryScreen)
            sub = screen.current_submission()
            assert sub.mage_mode == "fast"
