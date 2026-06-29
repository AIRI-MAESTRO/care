"""SaveChainNameModal — name prompt before library save."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Input

from care.screens.save_chain_name import SaveChainNameModal


class _Host(App):
    """Minimal runnable app to push the modal onto (no CareApp facades
    needed — the modal only seeds its own Input)."""


@pytest.mark.asyncio
async def test_save_chain_name_modal_seeds_default_name():
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = SaveChainNameModal(default_name="My chain")
        app.push_screen(modal)
        await pilot.pause()
        assert isinstance(app.screen, SaveChainNameModal)
        assert app.screen.query_one("#save-chain-name-input", Input).value == "My chain"


def test_save_chain_name_modal_confirm_sanitizes_name():
    modal = SaveChainNameModal(default_name="fallback")
    captured: list[str | None] = []
    modal.dismiss = lambda value=None: captured.append(value)  # type: ignore[method-assign]
    modal.query_one = lambda _sel, _typ: type("I", (), {"value": "  Renamed  "})()  # type: ignore[method-assign, assignment]
    modal.action_confirm()
    assert captured == ["Renamed"]
