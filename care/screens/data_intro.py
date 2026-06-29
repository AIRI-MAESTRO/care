"""DataIntroModal — primer on @-files, re-runs, and datasets.

Opened from the chat mode-row button «Working with data» /
«Работа с данными», or automatically once on the first
successful ``@`` attachment or first Library Run (tutorial
sidecar ``data_intro_shown``).
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from care.runtime.i18n import t


@dataclass(frozen=True)
class DataIntroResult:
    """Dismissal payload — ``open_help`` when the user wants /help."""

    open_help: bool = False


class DataIntroModal(ModalScreen[DataIntroResult | None]):
    """Informational modal: what / why / how of user data in MAESTRO."""

    DEFAULT_CSS = """
    DataIntroModal {
        align: center middle;
    }
    DataIntroModal #data-intro-box {
        width: 78%;
        max-width: 92;
        height: auto;
        max-height: 85%;
        padding: 1 2;
        background: $surface;
        border: solid $accent;
    }
    DataIntroModal #data-intro-title {
        text-style: bold;
        margin-bottom: 1;
    }
    DataIntroModal #data-intro-scroll {
        height: 1fr;
        max-height: 28;
        margin: 1 0;
    }
    DataIntroModal .intro-section-title {
        text-style: bold;
        margin-top: 1;
        color: $accent;
    }
    DataIntroModal .intro-section-body {
        margin-top: 0;
    }
    DataIntroModal #data-intro-buttons {
        margin-top: 1;
        align-horizontal: right;
        height: auto;
    }
    DataIntroModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="data-intro-box"):
            yield Static(t("dataIntro.title"), id="data-intro-title")
            with VerticalScroll(id="data-intro-scroll"):
                yield Static(
                    t("dataIntro.whatTitle"),
                    classes="intro-section-title",
                )
                yield Static(
                    t("dataIntro.whatBody"),
                    classes="intro-section-body",
                )
                yield Static(
                    t("dataIntro.whyTitle"),
                    classes="intro-section-title",
                )
                yield Static(
                    t("dataIntro.whyBody"),
                    classes="intro-section-body",
                )
                yield Static(
                    t("dataIntro.howTitle"),
                    classes="intro-section-title",
                )
                yield Static(
                    t("dataIntro.howBody"),
                    classes="intro-section-body",
                )
            with Horizontal(id="data-intro-buttons"):
                yield Button(
                    t("dataIntro.openHelp"),
                    id="data-intro-open-help",
                    variant="primary",
                )
                yield Button(
                    t("dataIntro.close"),
                    id="data-intro-close",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "data-intro-open-help":
            self.dismiss(DataIntroResult(open_help=True))
        elif bid == "data-intro-close":
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


__all__ = [
    "DataIntroModal",
    "DataIntroResult",
]
