"""HumanInputModal — answer a CARL `HumanInputStep`
(TODO §1.1 P0.33).

Pushed by ExecutionScreen when the
:class:`care.runtime.carl_streamer.HumanInputRequested`
message lands, or by any other screen that subscribes to
:meth:`HumanInputBroker.on_request`. The modal renders the
pending :class:`HumanInputRequest` prompt + a free-form Input
(or an OptionList when discrete `options` were supplied) and
resolves the broker on Submit.

Cancelling the modal calls
:meth:`HumanInputBroker.cancel(request_id, reason=...)` —
CARL's chain will then surface a :class:`HumanInputCancelled`
the executor can handle as a skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from care.runtime.human_input import (
    HumanInputBroker,
    HumanInputRequest,
)
from care.runtime.i18n import t


@dataclass(frozen=True)
class HumanInputResult:
    """Dismiss envelope.

    ``submitted`` is ``True`` when the user resolved the
    request (broker.resolve called); ``False`` when the user
    cancelled (broker.cancel called). ``value`` is the
    submitted text (empty for cancel)."""

    submitted: bool
    value: str
    request_id: str


class HumanInputModal(ModalScreen[HumanInputResult]):
    """Free-form prompt + optional discrete picker.

    Construct with the :class:`HumanInputBroker` + the
    :class:`HumanInputRequest` the user should answer. On
    Submit the modal calls ``broker.resolve(request_id,
    value)`` so CARL's future fires immediately; on Cancel it
    calls ``broker.cancel(request_id, reason="user
    cancelled")``."""

    DEFAULT_CSS = """
    HumanInputModal {
        align: center middle;
    }
    HumanInputModal #human-input-box {
        width: 70;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    HumanInputModal #human-input-title {
        text-style: bold;
        padding-bottom: 1;
    }
    HumanInputModal #human-input-prompt {
        padding-bottom: 1;
    }
    HumanInputModal #human-input-options {
        height: 10;
        margin-bottom: 1;
    }
    HumanInputModal #human-input-text {
        margin-bottom: 1;
    }
    HumanInputModal #human-input-buttons {
        height: auto;
        align-horizontal: right;
    }
    HumanInputModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+enter", "submit", "Submit", show=False),
    ]

    def __init__(
        self,
        *,
        broker: HumanInputBroker,
        request: HumanInputRequest,
    ) -> None:
        super().__init__()
        self._broker = broker
        self.request = request

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="human-input-box"):
            yield Label(t("humanInput.title"), id="human-input-title")
            yield Static(
                self.request.prompt or t("humanInput.noPrompt"),
                id="human-input-prompt",
            )
            if self.request.options:
                opts = [
                    Option(o, id=self._option_id(i))
                    for i, o in enumerate(self.request.options)
                ]
                yield OptionList(*opts, id="human-input-options")
            else:
                yield Input(
                    value=self.request.default,
                    placeholder=t("humanInput.answerPlaceholder"),
                    id="human-input-text",
                )
            with Horizontal(id="human-input-buttons"):
                yield Button(t("common.cancel"), id="human-input-cancel")
                yield Button(
                    t("common.submit"),
                    id="human-input-submit",
                    variant="primary",
                )

    def on_mount(self) -> None:
        try:
            if self.request.options:
                self.query_one("#human-input-options", OptionList).focus()
            else:
                self.query_one("#human-input-text", Input).focus()
        except Exception:
            pass

    @staticmethod
    def _option_id(index: int) -> str:
        return f"hi-opt-{index}"

    # ------------------------------------------------------------------
    # Field readers
    # ------------------------------------------------------------------

    def _read_value(self) -> str:
        if self.request.options:
            try:
                option_list = self.query_one(
                    "#human-input-options", OptionList,
                )
            except Exception:
                return self.request.default
            idx = option_list.highlighted
            if idx is None or idx < 0:
                return self.request.default
            try:
                option = option_list.get_option_at_index(idx)
            except Exception:
                return self.request.default
            try:
                pos = int((option.id or "hi-opt-0").rsplit("-", 1)[1])
            except (ValueError, IndexError):
                return self.request.default
            if 0 <= pos < len(self.request.options):
                return self.request.options[pos]
            return self.request.default
        try:
            return (
                self.query_one("#human-input-text", Input).value
                or self.request.default
            )
        except Exception:
            return self.request.default

    # ------------------------------------------------------------------
    # Submit / cancel
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "human-input-cancel":
            self.action_cancel()
        elif bid == "human-input-submit":
            self.action_submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "human-input-text":
            return
        self.action_submit()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected,
    ) -> None:
        if event.option_list.id != "human-input-options":
            return
        self.action_submit()

    def action_submit(self) -> None:
        value = self._read_value()
        try:
            self._broker.resolve(self.request.id, value)
        except Exception:
            pass
        self.dismiss(
            HumanInputResult(
                submitted=True,
                value=value,
                request_id=self.request.id,
            ),
        )

    def action_cancel(self) -> None:
        try:
            self._broker.cancel(
                self.request.id, reason="user cancelled",
            )
        except Exception:
            pass
        self.dismiss(
            HumanInputResult(
                submitted=False,
                value="",
                request_id=self.request.id,
            ),
        )


def _ensure_any(_: Any) -> None:
    """Anchor the `Any` import for future expansion."""


__all__ = ["HumanInputModal", "HumanInputResult"]
