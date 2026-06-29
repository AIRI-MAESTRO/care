"""ConfirmModal — minimal Yes/No prompt (TODO §1.1 P0.11).

A :class:`ModalScreen` that asks the user to confirm a
destructive action. Pushed by :class:`LibraryScreen` (and any
future screen) ahead of `delete_row` / bulk-delete style
gestures. Dismissed with `True` on confirm, `False` on cancel
(also `Escape` / `N` keys cancel; `Enter` / `Y` confirm).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from care.runtime.i18n import t
from care.screens._animated_modal import AnimatedModalScreen


class ConfirmModal(AnimatedModalScreen[bool]):
    """Modal confirmation prompt.

    Args:
        title: Headline rendered at the top of the modal.
        body: Sub-line — usually the specific entity name or
            action target.
        confirm_label: Text on the destructive button.
        cancel_label: Text on the safe button.
    """

    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
    }
    ConfirmModal #confirm-box {
        width: 60;
        max-width: 80%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    ConfirmModal #confirm-title {
        text-style: bold;
        padding-bottom: 1;
    }
    ConfirmModal #confirm-body {
        padding-bottom: 1;
        color: $text-muted;
        max-height: 18;
        overflow-y: auto;
    }
    ConfirmModal #confirm-buttons {
        height: auto;
        align-horizontal: right;
    }
    ConfirmModal Button {
        margin-left: 1;
    }
    """

    ANIM_BOX_ID = "confirm-box"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("n", "cancel", "Cancel", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("y", "confirm", "Confirm", show=False),
    ]

    def __init__(
        self,
        *,
        title: str,
        body: str = "",
        confirm_label: str | None = None,
        cancel_label: str | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._body = body
        self._confirm_label = (
            confirm_label if confirm_label is not None else t("common.confirm")
        )
        self._cancel_label = (
            cancel_label if cancel_label is not None else t("common.cancel")
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(self._title, id="confirm-title")
            if self._body:
                yield Static(self._body, id="confirm-body")
            with Horizontal(id="confirm-buttons"):
                yield Button(
                    self._cancel_label, id="confirm-cancel",
                )
                yield Button(
                    self._confirm_label,
                    id="confirm-ok",
                    variant="error",
                )

    def on_mount(self) -> None:
        # Focus the destructive button by default so `Enter`
        # confirms via the `BINDINGS` table without bouncing
        # off a focused Cancel button. Mirrors the convention
        # other CARE modals use (modal action_id wins over
        # the implicit Button.press(Enter)).
        try:
            self.query_one("#confirm-ok", Button).focus()
        except Exception:
            pass
        self._animate_modal_in()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-ok":
            self.dismiss(True)
        elif event.button.id == "confirm-cancel":
            self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


__all__ = ["ConfirmModal"]
