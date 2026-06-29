"""SaveChainNameModal — prompt for a library name before saving a chain."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from care.runtime.i18n import t
from care.runtime.save_agent_form import sanitize_chain_name
from care.screens._animated_modal import AnimatedModalScreen


class SaveChainNameModal(AnimatedModalScreen[str | None]):
    """Ask the user to confirm/edit the chain name before Memory save."""

    DEFAULT_CSS = """
    SaveChainNameModal {
        align: center middle;
    }
    SaveChainNameModal #save-chain-name-box {
        width: 60;
        max-width: 90%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    SaveChainNameModal #save-chain-name-title {
        text-style: bold;
        padding-bottom: 1;
    }
    SaveChainNameModal #save-chain-name-hint {
        padding-bottom: 1;
        color: $text-muted;
    }
    SaveChainNameModal #save-chain-name-input {
        margin-bottom: 1;
    }
    SaveChainNameModal #save-chain-name-buttons {
        height: auto;
        align-horizontal: right;
    }
    SaveChainNameModal Button {
        margin-left: 1;
    }
    """

    ANIM_BOX_ID = "save-chain-name-box"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "confirm", "Save", show=False),
    ]

    def __init__(
        self,
        *,
        default_name: str = "",
        title_key: str = "chat.saveName.title",
        hint_key: str = "chat.saveName.hint",
        confirm_key: str = "chat.saveName.confirm",
    ) -> None:
        super().__init__()
        self._default_name = default_name or ""
        self._title_key = title_key
        self._hint_key = hint_key
        self._confirm_key = confirm_key

    def compose(self) -> ComposeResult:
        with Vertical(id="save-chain-name-box"):
            yield Static(t(self._title_key), id="save-chain-name-title")
            yield Static(t(self._hint_key), id="save-chain-name-hint")
            yield Input(
                value=self._default_name,
                placeholder=t("chat.saveName.placeholder"),
                id="save-chain-name-input",
            )
            with Horizontal(id="save-chain-name-buttons"):
                yield Button(t("common.cancel"), id="save-chain-name-cancel")
                yield Button(
                    t(self._confirm_key),
                    id="save-chain-name-ok",
                    variant="primary",
                )

    def on_mount(self) -> None:
        try:
            self.query_one("#save-chain-name-input", Input).focus()
        except Exception:
            pass
        self._animate_modal_in()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "save-chain-name-ok":
            self.action_confirm()
        elif bid == "save-chain-name-cancel":
            self.action_cancel()

    def action_confirm(self) -> None:
        try:
            raw = self.query_one("#save-chain-name-input", Input).value
        except Exception:
            raw = self._default_name
        name = sanitize_chain_name(str(raw or "").strip())
        if not name:
            name = sanitize_chain_name(self._default_name) or str(raw or "").strip()
        if not name:
            return
        self.dismiss(name)

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["SaveChainNameModal"]
