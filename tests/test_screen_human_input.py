"""Pilot tests for HumanInputModal (TODO §1.1 P0.33).

Exercises:
* Free-form prompt renders an Input pre-filled with `default`.
* Discrete `options` render an OptionList.
* Submit calls `broker.resolve(request_id, value)` and
  dismisses with `submitted=True`.
* Cancel calls `broker.cancel(request_id, ...)` and dismisses
  with `submitted=False`.
* Enter on the Input submits.
"""

from __future__ import annotations

from concurrent.futures import Future

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, OptionList

from care.runtime.human_input import (
    HumanInputBroker,
    HumanInputCancelled,
)
from care.screens.human_input import (
    HumanInputModal,
    HumanInputResult,
)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(
        self,
        *,
        broker: HumanInputBroker,
        prompt: str = "what is your name?",
        default: str = "",
        options=(),
    ) -> None:
        super().__init__()
        self._broker = broker
        self._prompt = prompt
        self._default = default
        self._options = tuple(options)
        self.dismissed: list[HumanInputResult] = []
        self._future: Future = Future()

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        request = self._broker.submit(
            self._prompt,
            future=self._future,
            default=self._default,
            options=list(self._options) if self._options else None,
        )

        def _on_dismiss(result):
            self.dismissed.append(result)

        self.push_screen(
            HumanInputModal(broker=self._broker, request=request),
            _on_dismiss,
        )


def _modal(app: App) -> HumanInputModal:
    s = app.screen_stack[-1]
    assert isinstance(s, HumanInputModal)
    return s


# ---------------------------------------------------------------------------
# Free-form prompt
# ---------------------------------------------------------------------------


class TestFreeForm:
    @pytest.mark.asyncio
    async def test_input_renders_with_default(self):
        broker = HumanInputBroker()
        app = _Host(broker=broker, default="seed")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            text = modal.query_one("#human-input-text", Input)
            assert text.value == "seed"

    @pytest.mark.asyncio
    async def test_submit_resolves_future(self):
        broker = HumanInputBroker()
        app = _Host(broker=broker)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#human-input-text", Input,
            ).value = "world"
            await pilot.pause()
            modal.action_submit()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].submitted is True
            assert app.dismissed[0].value == "world"
            assert app._future.done()
            assert app._future.result() == "world"

    @pytest.mark.asyncio
    async def test_cancel_cancels_future(self):
        broker = HumanInputBroker()
        app = _Host(broker=broker)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_cancel()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].submitted is False
            # Future was cancelled via broker; reading it
            # raises `HumanInputCancelled`.
            with pytest.raises(HumanInputCancelled):
                app._future.result(timeout=0)

    @pytest.mark.asyncio
    async def test_enter_on_input_submits(self):
        broker = HumanInputBroker()
        app = _Host(broker=broker)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            input_widget = modal.query_one("#human-input-text", Input)
            input_widget.value = "via enter"
            await pilot.pause()
            input_widget.post_message(
                Input.Submitted(input_widget, value="via enter"),
            )
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed[0].value == "via enter"


# ---------------------------------------------------------------------------
# Discrete options
# ---------------------------------------------------------------------------


class TestOptions:
    @pytest.mark.asyncio
    async def test_option_list_renders(self):
        broker = HumanInputBroker()
        app = _Host(broker=broker, options=("yes", "no", "skip"))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            options = modal.query_one(
                "#human-input-options", OptionList,
            )
            assert options.option_count == 3

    @pytest.mark.asyncio
    async def test_option_select_submits(self):
        broker = HumanInputBroker()
        app = _Host(broker=broker, options=("yes", "no", "skip"))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            options = modal.query_one(
                "#human-input-options", OptionList,
            )
            # Highlight the second option ("no") then submit.
            options.highlighted = 1
            await pilot.pause()
            modal.action_submit()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed[0].value == "no"


# ---------------------------------------------------------------------------
# Button dismiss
# ---------------------------------------------------------------------------


class TestButtonDismiss:
    @pytest.mark.asyncio
    async def test_cancel_button_cancels(self):
        broker = HumanInputBroker()
        app = _Host(broker=broker)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#human-input-cancel", Button).press()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed[0].submitted is False

    @pytest.mark.asyncio
    async def test_submit_button_resolves(self):
        broker = HumanInputBroker()
        app = _Host(broker=broker, default="ok")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#human-input-submit", Button).press()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed[0].submitted is True
            assert app.dismissed[0].value == "ok"


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import HumanInputModal as M
        from care.screens import HumanInputResult as R

        assert M is HumanInputModal
        assert R is HumanInputResult
