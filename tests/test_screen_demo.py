"""Smoke tests for `DemoScreen` (TODO §8 P1).

Goal: compose without error + one navigation assertion. The
demo screen is a scratch surface from early scaffolding —
not on the user's main flow — but it's still part of the
shipped wheel, so a smoke test guards against drift.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from care.screens.demo import DemoScreen


class _Host(App):
    def __init__(self):
        super().__init__()

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(DemoScreen())


class TestCompose:
    @pytest.mark.asyncio
    async def test_mount_does_not_raise(self) -> None:
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, DemoScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_title_and_subtitle_set_on_mount(self) -> None:
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = next(
                s for s in app.screen_stack
                if isinstance(s, DemoScreen)
            )
            # `on_mount` sets the screen-level title/subtitle.
            assert screen.title == "MAESTRO"
            assert "Collaborative Agent" in screen.sub_title


class TestActions:
    @pytest.mark.asyncio
    async def test_cancel_generate_does_not_raise(self) -> None:
        # `Esc` → `action_cancel_generate` cancels in-flight
        # workers via `workers.cancel_group`. Calling with no
        # workers attached should be a clean no-op.
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = next(
                s for s in app.screen_stack
                if isinstance(s, DemoScreen)
            )
            # Should not raise regardless of worker state.
            screen.action_cancel_generate()
            await pilot.pause()
