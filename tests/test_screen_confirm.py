"""Pilot tests for ConfirmModal (TODO §1.1 P0.29).

The modal was first shipped during P0.11 (LibraryScreen
per-row delete confirm); P0.29 adds the dedicated test suite
the §1.1 spec calls for.

Exercises:
* Composition — title + body + confirm/cancel buttons mount.
* Button presses dismiss with the right boolean.
* `Y` / `Enter` confirm; `N` / `Esc` cancel.
* Pilot key-press flow (`Enter` → handler runs;
  `Esc` → no-op).
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from care.screens.confirm import ConfirmModal


class _Host(App):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self.dismissed: list[bool] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(result):
            self.dismissed.append(bool(result))

        self.push_screen(ConfirmModal(**self._kwargs), _on_dismiss)


def _modal(app: App) -> ConfirmModal:
    s = app.screen_stack[-1]
    assert isinstance(s, ConfirmModal)
    return s


class TestCompose:
    @pytest.mark.asyncio
    async def test_renders_title_body_and_buttons(self):
        app = _Host(title="Delete?", body="row-1")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.query_one("#confirm-title", Static) is not None
            assert modal.query_one("#confirm-body", Static) is not None
            assert modal.query_one("#confirm-ok", Button) is not None
            assert modal.query_one("#confirm-cancel", Button) is not None

    @pytest.mark.asyncio
    async def test_blank_body_skips_static(self):
        app = _Host(title="Discard?")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            statics = list(modal.query("#confirm-body"))
            assert statics == []


class TestButtonDismiss:
    @pytest.mark.asyncio
    async def test_ok_dismisses_true(self):
        app = _Host(title="Delete?")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#confirm-ok", Button).press()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed == [True]

    @pytest.mark.asyncio
    async def test_cancel_dismisses_false(self):
        app = _Host(title="Delete?")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#confirm-cancel", Button).press()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed == [False]


class TestActionBindings:
    @pytest.mark.asyncio
    async def test_action_confirm_dismisses_true(self):
        app = _Host(title="Delete?")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_confirm()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed == [True]

    @pytest.mark.asyncio
    async def test_action_cancel_dismisses_false(self):
        app = _Host(title="Delete?")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_cancel()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed == [False]


class TestPilotKeys:
    @pytest.mark.asyncio
    async def test_enter_key_confirms(self):
        app = _Host(title="Delete?")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed == [True]

    @pytest.mark.asyncio
    async def test_y_key_confirms(self):
        app = _Host(title="Delete?")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("y")
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed == [True]

    @pytest.mark.asyncio
    async def test_escape_key_cancels(self):
        app = _Host(title="Delete?")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed == [False]

    @pytest.mark.asyncio
    async def test_n_key_cancels(self):
        app = _Host(title="Delete?")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("n")
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed == [False]


class TestLabelsCustomisable:
    @pytest.mark.asyncio
    async def test_confirm_label_renders(self):
        app = _Host(title="Erase?", confirm_label="Erase forever")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            confirm = modal.query_one("#confirm-ok", Button)
            assert str(confirm.label) == "Erase forever"

    @pytest.mark.asyncio
    async def test_cancel_label_renders(self):
        app = _Host(title="Erase?", cancel_label="Keep")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            cancel = modal.query_one("#confirm-cancel", Button)
            assert str(cancel.label) == "Keep"


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import ConfirmModal as Re

        assert Re is ConfirmModal
