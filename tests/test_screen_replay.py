"""Pilot tests for ReplayScreen (§6 [DONE — data layer] → fully DONE).

Wires `care.load_replay` into a `Screen` that lets the user
step through a stored `ReasoningResult`.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ListItem, ListView

from care.replay import ReplaySession, ReplayStep
from care.screens.replay import ReplayScreen


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def _sample_session() -> ReplaySession:
    return ReplaySession(
        steps=(
            ReplayStep(
                step_number=1,
                step_title="fetch",
                step_type="llm",
                result_preview="fetched payload",
                success=True,
                execution_time_s=0.42,
            ),
            ReplayStep(
                step_number=2,
                step_title="summarise",
                step_type="llm",
                result_preview="summary",
                success=True,
                execution_time_s=1.1,
            ),
            ReplayStep(
                step_number=3,
                step_title="verify",
                step_type="llm",
                success=False,
                error_message="boom",
                execution_time_s=0.05,
            ),
        ),
        chain_id="agent-x",
        chain_title="Storm Watcher",
    )


class _Host(App):
    def __init__(self, *, source=None) -> None:
        super().__init__()
        self._source = source

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(ReplayScreen(self._source))


def _screen(app: App) -> ReplayScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, ReplayScreen)
    return s


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_empty_session_renders_placeholder(self):
        app = _Host(source=None)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.session.is_empty is True
            listview = screen.query_one("#replay-step-list", ListView)
            items = list(listview.query(ListItem))
            assert len(items) >= 1

    @pytest.mark.asyncio
    async def test_session_populates_list(self):
        app = _Host(source=_sample_session())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.session.step_count == 3
            listview = screen.query_one("#replay-step-list", ListView)
            assert len(list(listview.query(ListItem))) == 3


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


class TestNavigation:
    @pytest.mark.asyncio
    async def test_next_advances_cursor(self):
        app = _Host(source=_sample_session())
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.session.cursor == 0
            screen.action_next_step()
            await pilot.pause()
            assert screen.session.cursor == 1
            screen.action_next_step()
            await pilot.pause()
            assert screen.session.cursor == 2
            # Clamped at end.
            screen.action_next_step()
            await pilot.pause()
            assert screen.session.cursor == 2

    @pytest.mark.asyncio
    async def test_previous_rewinds_cursor(self):
        app = _Host(source=_sample_session())
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.session.seek(2)
            screen.action_previous_step()
            await pilot.pause()
            assert screen.session.cursor == 1
            screen.action_previous_step()
            await pilot.pause()
            assert screen.session.cursor == 0
            # Clamped at start.
            screen.action_previous_step()
            await pilot.pause()
            assert screen.session.cursor == 0

    @pytest.mark.asyncio
    async def test_restart_jumps_to_start(self):
        app = _Host(source=_sample_session())
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.session.seek(2)
            screen.action_restart()
            await pilot.pause()
            assert screen.session.cursor == 0

    @pytest.mark.asyncio
    async def test_escape_pops_screen(self):
        app = _Host(source=_sample_session())
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            initial_depth = len(app.screen_stack)
            screen.action_back()
            await pilot.pause()
            assert len(app.screen_stack) < initial_depth


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestStepLabel:
    def test_success_badge(self):
        step = ReplayStep(
            step_number=1, step_title="fetch", step_type="llm",
            success=True,
        )
        assert ReplayScreen._step_label(0, step).startswith("✓")

    def test_failure_badge(self):
        step = ReplayStep(
            step_number=1, step_title="bad", step_type="llm",
            success=False,
        )
        assert ReplayScreen._step_label(0, step).startswith("✗")

    def test_skipped_badge(self):
        step = ReplayStep(
            step_number=1, step_title="cond", step_type="conditional",
            success=True, skipped=True,
        )
        assert ReplayScreen._step_label(0, step).startswith("·")

    def test_fallback_title(self):
        step = ReplayStep(
            step_number=4, step_title="", step_type="",
        )
        assert "step-1" in ReplayScreen._step_label(0, step)


# ---------------------------------------------------------------------------
# Source variations (load_replay handles dicts + JSON strings)
# ---------------------------------------------------------------------------


class TestSourceVariations:
    @pytest.mark.asyncio
    async def test_dict_source(self):
        source = {
            "step_results": [
                {
                    "step_number": 1,
                    "step_title": "fetch",
                    "step_type": "llm",
                    "result": "hello",
                    "success": True,
                },
            ],
        }
        app = _Host(source=source)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.session.step_count == 1

    @pytest.mark.asyncio
    async def test_malformed_source_silent(self):
        # Bogus JSON-string source — `load_replay` raises
        # `ReplayError`. The constructor swallows; the screen
        # composes with an empty session.
        app = _Host(source="{not valid json")
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.session.is_empty is True


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import ReplayScreen as R

        assert R is ReplayScreen
