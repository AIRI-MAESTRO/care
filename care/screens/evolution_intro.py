"""EvolutionIntroModal — short primer on chain evolution.

Opened from the chat mode row button «More about evolution» /
«Подробнее про эволюцию», or as the first step of bare
``/evolution``. Explains what evolution is, why it helps, and
how to launch a run in MAESTRO before optionally handing off to
:class:`EvolutionDashboard`.
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
class EvolutionIntroResult:
    """Dismissal payload — ``open_dashboard`` when the user wants
    the runs list after reading the intro."""

    open_dashboard: bool = False


class EvolutionIntroModal(ModalScreen[EvolutionIntroResult | None]):
    """Informational modal: what / why / how of chain evolution."""

    def __init__(self, *, dismiss_to_dashboard: bool = False) -> None:
        super().__init__()
        # When True (e.g. bare ``/evolution``), «Got it» also hands
        # off to :class:`EvolutionDashboard`; Esc still cancels.
        self._dismiss_to_dashboard = dismiss_to_dashboard

    DEFAULT_CSS = """
    EvolutionIntroModal {
        align: center middle;
    }
    EvolutionIntroModal #intro-box {
        width: 78%;
        max-width: 92;
        height: auto;
        max-height: 85%;
        padding: 1 2;
        background: $surface;
        border: solid $accent;
    }
    EvolutionIntroModal #intro-title {
        text-style: bold;
        margin-bottom: 1;
    }
    EvolutionIntroModal #intro-scroll {
        height: 1fr;
        max-height: 28;
        margin: 1 0;
    }
    EvolutionIntroModal .intro-section-title {
        text-style: bold;
        margin-top: 1;
        color: $accent;
    }
    EvolutionIntroModal .intro-section-body {
        margin-top: 0;
    }
    EvolutionIntroModal #intro-buttons {
        margin-top: 1;
        align-horizontal: right;
        height: auto;
    }
    EvolutionIntroModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="intro-box"):
            yield Static(t("evolutionIntro.title"), id="intro-title")
            with VerticalScroll(id="intro-scroll"):
                yield Static(
                    t("evolutionIntro.whatTitle"),
                    classes="intro-section-title",
                )
                yield Static(
                    t("evolutionIntro.whatBody"),
                    classes="intro-section-body",
                )
                yield Static(
                    t("evolutionIntro.whyTitle"),
                    classes="intro-section-title",
                )
                yield Static(
                    t("evolutionIntro.whyBody"),
                    classes="intro-section-body",
                )
                yield Static(
                    t("evolutionIntro.howTitle"),
                    classes="intro-section-title",
                )
                yield Static(
                    t("evolutionIntro.howBody"),
                    classes="intro-section-body",
                )
            with Horizontal(id="intro-buttons"):
                yield Button(
                    t("evolutionIntro.openDashboard"),
                    id="intro-open-dashboard",
                    variant="primary",
                )
                yield Button(
                    t("evolutionIntro.close"),
                    id="intro-close",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "intro-open-dashboard":
            self.dismiss(EvolutionIntroResult(open_dashboard=True))
        elif bid == "intro-close":
            if self._dismiss_to_dashboard:
                self.dismiss(EvolutionIntroResult(open_dashboard=True))
            else:
                self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
