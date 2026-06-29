"""Pilot tests for the InspectionScreen → Export-to-Markdown affordance.

The `m` binding / "Export MD" button opens the shared `ExportChainModal`
defaulted to Markdown, carrying the inspected chain.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, RadioSet

from care.screens.export_chain import ExportChainModal
from care.screens.inspection import InspectionPayload, InspectionScreen


class _InspHost(App):
    memory = None

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(InspectionScreen("agent-x"))


def _screen(app: App) -> InspectionScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, InspectionScreen)
    return s


_PAYLOAD = InspectionPayload(
    entity_id="agent-x",
    display_name="Weather Brief",
    domain="reporting",
    description="Fetch + summarize.",
    steps=(
        {"number": 1, "type": "tool", "title": "Fetch", "config": {"tool_name": "web_search"}},
        {"number": 2, "type": "llm", "title": "Summarize", "aim": "Write a brief.", "dependencies": [1]},
    ),
)


class TestInspectionExport:
    @pytest.mark.asyncio
    async def test_button_opens_export_modal_in_markdown(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.state.payload = _PAYLOAD
            # Click the Export MD button.
            screen.on_button_pressed(
                Button.Pressed(screen.query_one("#inspection-btn-export", Button)),
            )
            for _ in range(4):
                await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ExportChainModal)
            # The Markdown radio is the pre-selected format.
            radio = modal.query_one("#export-chain-format", RadioSet)
            assert (radio.pressed_button.id or "").endswith("-markdown")
            assert modal._read_format() == "markdown"

    @pytest.mark.asyncio
    async def test_binding_action_no_chain_does_not_open_modal(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.state.payload = None  # nothing loaded
            depth = len(app.screen_stack)
            screen.action_export_markdown()
            await pilot.pause()
            # No modal pushed when there's no chain.
            assert len(app.screen_stack) == depth

    @pytest.mark.asyncio
    async def test_export_button_mounts(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.query_one("#inspection-btn-export", Button) is not None
