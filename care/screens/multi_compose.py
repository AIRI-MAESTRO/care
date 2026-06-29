"""MultiLineComposer — Phase 8 P0 #1 multi-line task composer.

A :class:`ModalScreen` that gives the user a multi-line
``TextArea`` for composing a long task description without
fighting the single-line ``Input`` in the main chat surface.
Triggered via the ``/multi`` slash command (and optionally
Ctrl+M in future iterations).

Submission contract:

* ``Ctrl+J`` / ``Ctrl+S`` — submit (dismiss with the typed text).
* ``Escape`` — cancel (dismiss with ``None`` so the caller
  can detect the difference).
* Click ``Submit`` / ``Cancel`` buttons for mouse users.

The caller is :class:`care.screens.chat.ChatScreen`'s
``/multi`` handler, which awaits the dismiss value and feeds
the string to ``_handle_task`` if non-empty.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from care.runtime.i18n import t


class MultiLineComposer(ModalScreen["str | None"]):
    """Modal multi-line composer for long task descriptions.

    Construct with ``initial_text`` to pre-fill the area
    (useful when the user typed `/multi` after starting a
    prompt and wants to keep the partial input).
    """

    DEFAULT_CSS = """
    MultiLineComposer {
        align: center middle;
    }
    MultiLineComposer #composer-box {
        width: 80%;
        max-width: 120;
        height: 60%;
        max-height: 30;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    MultiLineComposer #composer-title {
        text-style: bold;
        padding-bottom: 1;
    }
    MultiLineComposer #composer-hint {
        color: $text-muted;
        text-style: italic;
        padding-bottom: 1;
    }
    MultiLineComposer #composer-input {
        height: 1fr;
        border: solid $primary 30%;
    }
    MultiLineComposer #composer-buttons {
        height: auto;
        align-horizontal: right;
        padding-top: 1;
    }
    MultiLineComposer Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+j", "submit", "Submit", show=True),
        Binding("ctrl+s", "submit", "Submit", show=False),
    ]

    def __init__(self, *, initial_text: str = "") -> None:
        super().__init__()
        self._initial_text = initial_text

    def compose(self) -> ComposeResult:
        with Vertical(id="composer-box"):
            yield Static(
                t("multiCompose.title"),
                id="composer-title",
            )
            yield Static(t("multiCompose.hint"), id="composer-hint")
            yield TextArea(
                self._initial_text,
                id="composer-input",
            )
            with Horizontal(id="composer-buttons"):
                yield Button(t("common.cancel"), id="composer-cancel")
                yield Button(
                    t("common.submit"), id="composer-submit", variant="primary",
                )

    def on_mount(self) -> None:
        try:
            ta = self.query_one("#composer-input", TextArea)
            ta.focus()
            # Land cursor at the END of any pre-filled text so
            # the user types continuation rather than overwriting.
            ta.cursor_location = ta.document.end
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "composer-submit":
            self.action_submit()
        elif event.button.id == "composer-cancel":
            self.action_cancel()

    def action_submit(self) -> None:
        try:
            ta = self.query_one("#composer-input", TextArea)
            text = ta.text
        except Exception:
            text = ""
        # Dismiss with None when the area is empty (whitespace-
        # only) so the caller can treat it as a cancel — no
        # point firing a generation with no task.
        cleaned = (text or "").strip()
        if not cleaned:
            self.dismiss(None)
            return
        self.dismiss(text)

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["MultiLineComposer"]
