"""Smoke + behaviour tests for `MultiLineComposer` (TODO §8 P1)."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import TextArea

from care.screens.multi_compose import MultiLineComposer


class _Host(App):
    def __init__(self, *, initial_text: str = ""):
        super().__init__()
        self._initial_text = initial_text
        self.dismissed: list[str | None] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(
            MultiLineComposer(initial_text=self._initial_text),
            self._on_dismiss,
        )

    def _on_dismiss(self, result: str | None) -> None:
        self.dismissed.append(result)


def _composer(app: _Host) -> MultiLineComposer:
    for s in app.screen_stack:
        if isinstance(s, MultiLineComposer):
            return s
    raise AssertionError("MultiLineComposer not on stack")


class TestCompose:
    @pytest.mark.asyncio
    async def test_mount_does_not_raise(self) -> None:
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, MultiLineComposer)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_initial_text_prefills_textarea(self) -> None:
        app = _Host(initial_text="hello world")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            composer = _composer(app)
            ta = composer.query_one("#composer-input", TextArea)
            assert ta.text == "hello world"


class TestActions:
    @pytest.mark.asyncio
    async def test_cancel_dismisses_with_none(self) -> None:
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            composer = _composer(app)
            composer.action_cancel()
            for _ in range(4):
                await pilot.pause()
            assert app.dismissed == [None]

    @pytest.mark.asyncio
    async def test_submit_empty_dismisses_with_none(self) -> None:
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            composer = _composer(app)
            composer.action_submit()
            for _ in range(4):
                await pilot.pause()
            # Empty / whitespace-only submission collapses to
            # None so the caller treats it as a cancel.
            assert app.dismissed == [None]

    @pytest.mark.asyncio
    async def test_submit_text_dismisses_with_text(self) -> None:
        app = _Host(initial_text="actual task body")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            composer = _composer(app)
            composer.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert app.dismissed == ["actual task body"]


class TestReExports:
    def test_screens_re_exports(self) -> None:
        # MultiLineComposer is not re-exported by
        # `care.screens` today — assert the import path
        # stays stable so a refactor doesn't silently break
        # the `/multi` slash command.
        from care.screens.multi_compose import MultiLineComposer as M

        assert M is MultiLineComposer
