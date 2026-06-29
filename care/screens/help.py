"""HelpScreen — tutorial walkthrough + key cheat-sheet
(TODO §9 P3 [DONE — data layer] → fully DONE).

Pushed by the global ``?`` binding (and by the WelcomeScreen's
"Show tutorial" CTA). Reads the shipped
:func:`care.build_registry` helper to assemble the canonical
tutorial steps + every documented key binding (defaults +
plugin extensions), then renders two panes:

* **Walkthrough** — every :class:`care.TutorialStep` as a
  card (title, body, optional hint key, optional screen).
* **Key bindings** — every :class:`care.KeyBinding` grouped by
  category (``global`` / ``library`` / ``generation`` /
  ``execution`` / ``evolution``).

Pure presentation; no Memory side-effects, no facade calls.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Label, Static

from care.help import HelpRegistry, KeyBinding, TutorialStep, build_registry
from care.runtime.i18n import t
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


_CATEGORIES: tuple[str, ...] = (
    "global",
    "chat",
    "library",
    "generation",
    "execution",
    "evolution",
)


def _populate_slash_commands(registry: HelpRegistry) -> None:
    """Mirror every `@_register(...)` slash command into the
    help registry under category ``"chat"`` (TODO §2 P1).

    The chat-side `_COMMAND_HANDLERS` dict + `_COMMAND_BLURBS`
    on `ChatScreen` are the single source of truth — pulling
    them in here means the help screen / `care help` CLI list
    every command without a parallel registration step.
    Idempotent: a command that's already bound (same key) is
    skipped so re-mounting the help screen doesn't duplicate
    rows.

    Lazy import of `care.screens.chat` so this module stays
    cheap to import + so the help registry can be exercised
    in tests that don't pull the chat surface.
    """
    try:
        from care.screens.chat import (
            ChatScreen,
            _COMMAND_HANDLERS,
        )
    except Exception:
        return
    existing_keys = {b.key for b in registry.bindings()}
    blurbs = getattr(ChatScreen, "_COMMAND_BLURBS", {}) or {}
    for name in sorted(_COMMAND_HANDLERS.keys()):
        key = f"/{name}"
        if key in existing_keys:
            continue
        action = blurbs.get(name) or ""
        registry.add_binding(
            KeyBinding(
                key=key,
                action=action,
                category="chat",
            ),
        )


class HelpScreen(Screen):
    """Walkthrough + cheat-sheet screen.

    Construct with an optional pre-built :class:`HelpRegistry`
    (mostly for tests that want to inject a custom registry).
    Defaults to :func:`care.build_registry` so plugin extensions
    show up automatically.
    """

    DEFAULT_CSS = """
    HelpScreen {
        layout: vertical;
    }
    HelpScreen #help-body {
        height: 1fr;
    }
    HelpScreen #help-tutorial {
        width: 3fr;
        padding: 1 2;
    }
    HelpScreen #help-bindings {
        width: 2fr;
        padding: 1 2;
        border-left: solid $primary;
    }
    HelpScreen .pane-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    HelpScreen .help-step-title {
        text-style: bold;
    }
    HelpScreen .help-step-hint {
        color: $accent;
    }
    HelpScreen .help-step-screen {
        color: $text-muted;
    }
    HelpScreen .help-binding-category {
        text-style: bold;
        margin-top: 1;
    }
    HelpScreen .help-binding-row {
        color: $text;
    }
    HelpScreen #help-empty {
        height: 1;
        color: $text-muted;
        padding: 0 2;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("q", "back", "Back", show=False),
    ]

    def __init__(
        self,
        registry: HelpRegistry | None = None,
        *,
        include_slash_commands: bool = True,
    ) -> None:
        super().__init__()
        self.registry: HelpRegistry = (
            registry if registry is not None else build_registry()
        )
        if include_slash_commands:
            _populate_slash_commands(self.registry)

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Horizontal(id="help-body"):
            with Vertical(id="help-tutorial"):
                yield Label(t("help.walkthrough"), classes="pane-title")
                yield VerticalScroll(id="help-tutorial-body")
            with Vertical(id="help-bindings"):
                yield Label(t("help.keyBindings"), classes="pane-title")
                yield VerticalScroll(id="help-bindings-body")
        yield Static("", id="help-empty")
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="HelpScreen",
                breadcrumb=(t("header.breadcrumb.help"),),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="HelpScreen",
                scope="screen",
            )
        except Exception:
            pass
        self._render_tutorial()
        self._render_bindings()
        self._render_empty_state()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_tutorial(self) -> None:
        try:
            container = self.query_one(
                "#help-tutorial-body", VerticalScroll,
            )
        except Exception:
            return
        for child in list(container.children):
            try:
                child.remove()
            except Exception:
                pass
        for index, step in enumerate(self.registry.steps(), 1):
            container.mount(
                Static(
                    f"{index}. {step.title}",
                    classes="help-step-title",
                ),
            )
            container.mount(Static(step.body))
            if step.hint_key:
                container.mount(
                    Static(
                        f"  ⌨ {step.hint_key}",
                        classes="help-step-hint",
                    ),
                )
            if step.screen:
                container.mount(
                    Static(
                        f"  ↳ {step.screen}",
                        classes="help-step-screen",
                    ),
                )
            container.mount(Static(""))

    def _render_bindings(self) -> None:
        try:
            container = self.query_one(
                "#help-bindings-body", VerticalScroll,
            )
        except Exception:
            return
        for child in list(container.children):
            try:
                child.remove()
            except Exception:
                pass
        for category in _CATEGORIES:
            rows = self.registry.by_category(category)  # type: ignore[arg-type]
            if not rows:
                continue
            container.mount(
                Static(
                    f"# {category}",
                    classes="help-binding-category",
                ),
            )
            for binding in rows:
                container.mount(
                    Static(
                        self._format_binding(binding),
                        classes="help-binding-row",
                    ),
                )

    @staticmethod
    def _format_binding(binding: KeyBinding) -> str:
        suffix = f"  ({binding.screen})" if binding.screen else ""
        return f"  {binding.key:<10} {binding.action}{suffix}"

    def _render_empty_state(self) -> None:
        try:
            target = self.query_one("#help-empty", Static)
        except Exception:
            return
        if self.registry.steps() or self.registry.bindings():
            target.update("")
            return
        target.update(t("help.empty"))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_back(self) -> None:
        try:
            self.app.pop_screen()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public helpers (for tests / instrumentation)
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Number of tutorial steps the screen renders."""
        return len(self.registry.steps())

    @property
    def binding_count(self) -> int:
        """Number of bindings the screen renders across all
        categories."""
        return len(self.registry.bindings())

    def _step_titles(self) -> tuple[str, ...]:
        return tuple(s.title for s in self.registry.steps())

    def _binding_keys(self) -> tuple[str, ...]:
        return tuple(b.key for b in self.registry.bindings())

    def _registry_for_tests(self) -> HelpRegistry:
        """Test hook — return the registry the screen rendered
        against. Equivalent to ``self.registry`` but explicit
        about intent."""
        return self.registry

    @staticmethod
    def _tutorial_step_dummy() -> TutorialStep:
        """Test convenience — a minimal step factory. Kept on the
        screen to avoid a public re-export from `care.help`."""
        return TutorialStep(title="", body="")


__all__ = ["HelpScreen"]
