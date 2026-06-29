"""ChatInput — Enter submits, Shift+Enter inserts a soft newline.

The real-terminal caveat (a bare CR for both chords on terminals without the
Kitty keyboard protocol) is a terminal limitation, not a logic one: Textual's
test pilot delivers ``shift+enter`` as a distinct chord, so the binding wiring
is verifiable here regardless of the host terminal.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input

from care.widgets.chat_input import ChatInput


class _Host(App):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def compose(self) -> ComposeResult:
        yield ChatInput(id="chat-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.submitted.append(event.value)


@pytest.mark.asyncio
async def test_shift_enter_inserts_newline_and_does_not_submit() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        inp = app.query_one("#chat-input", ChatInput)
        inp.focus()
        await pilot.pause()
        inp.value = "line1"
        await pilot.pause()
        await pilot.press("shift+enter")
        await pilot.pause()
        # Soft newline landed in the buffer; nothing was submitted.
        assert inp.value == "line1\n"
        assert app.submitted == []


@pytest.mark.asyncio
async def test_enter_still_submits() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        inp = app.query_one("#chat-input", ChatInput)
        inp.focus()
        await pilot.pause()
        inp.value = "hello"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.submitted == ["hello"]
        # Enter must NOT leave a stray newline in the buffer.
        assert "\n" not in inp.value


@pytest.mark.asyncio
async def test_action_newline_inserts_at_cursor() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        inp = app.query_one("#chat-input", ChatInput)
        inp.value = "ab"
        inp.cursor_position = 1  # caret between 'a' and 'b'
        await pilot.pause()
        inp.action_newline()
        await pilot.pause()
        assert inp.value == "a\nb"
        assert app.submitted == []
