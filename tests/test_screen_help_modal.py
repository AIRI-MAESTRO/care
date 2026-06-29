"""Pilot tests for HelpModal — the Chat header «Help» action menu."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from care.screens.help_modal import HelpModal


class _Host(App):
    def __init__(self) -> None:
        super().__init__()
        self.dismissed: list[object] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(HelpModal(), self.dismissed.append)


def _modal(app: App) -> HelpModal:
    for s in app.screen_stack:
        if isinstance(s, HelpModal):
            return s
    raise AssertionError("HelpModal not on stack")


def _press(modal: HelpModal, bid: str) -> None:
    modal.on_button_pressed(Button.Pressed(modal.query_one(f"#{bid}", Button)))


class TestHelpModal:
    @pytest.mark.asyncio
    async def test_renders_data_skill_close_buttons(self):
        from care.runtime.i18n import t

        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal = _modal(app)
            labels = [str(b.label) for b in modal.query(Button)]
            assert t("helpModal.data") in labels
            assert t("helpModal.skill") in labels
            assert t("common.close") in labels

    @pytest.mark.asyncio
    async def test_data_dismisses_with_data(self):
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            _press(_modal(app), "help-modal-data")
            await pilot.pause()
            assert app.dismissed == ["data"]

    @pytest.mark.asyncio
    async def test_skill_dismisses_with_skill(self):
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            _press(_modal(app), "help-modal-skill")
            await pilot.pause()
            assert app.dismissed == ["skill"]

    @pytest.mark.asyncio
    async def test_close_dismisses_with_none(self):
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            _press(_modal(app), "help-modal-close")
            await pilot.pause()
            assert app.dismissed == [None]
