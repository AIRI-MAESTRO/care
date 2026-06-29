"""HelpModal — secondary actions surfaced from the Chat header's «Help».

Groups the lower-traffic affordances that used to sit in the mode-row
quick-action strip: «Working with data» (the data primer) and «Add as a
coding-agent skill» (the skill-export stub). Dismisses with the chosen
action so the host (ChatScreen) routes to the existing handlers.
"""

from __future__ import annotations

from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from care.runtime.i18n import t
from care.screens._animated_modal import AnimatedModalScreen

HelpAction = Literal["data", "skill"]


class HelpModal(AnimatedModalScreen["HelpAction | None"]):
    """Small action menu: «Working with data» / «Add as a coding-agent
    skill». Dismisses with the chosen :data:`HelpAction`, or ``None`` on
    cancel."""

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
    }
    HelpModal #help-modal-box {
        width: 60;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: solid $accent;
    }
    HelpModal #help-modal-title {
        text-style: bold;
        margin-bottom: 1;
    }
    HelpModal .help-modal-action {
        width: 100%;
        margin-bottom: 1;
    }
    HelpModal #help-modal-buttons {
        height: auto;
        align-horizontal: right;
    }
    """

    ANIM_BOX_ID = "help-modal-box"

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-modal-box"):
            yield Static(t("helpModal.title"), id="help-modal-title")
            yield Button(
                t("helpModal.data"),
                id="help-modal-data",
                classes="help-modal-action",
            )
            yield Button(
                t("helpModal.skill"),
                id="help-modal-skill",
                classes="help-modal-action",
            )
            with Horizontal(id="help-modal-buttons"):
                yield Button(t("common.close"), id="help-modal-close")

    def on_mount(self) -> None:
        self._animate_modal_in()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "help-modal-data":
            self.dismiss("data")
        elif bid == "help-modal-skill":
            self.dismiss("skill")
        elif bid == "help-modal-close":
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


__all__ = ["HelpAction", "HelpModal"]
