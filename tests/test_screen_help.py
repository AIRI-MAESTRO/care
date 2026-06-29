"""Pilot tests for HelpScreen (§9 P3 [DONE — data layer] → fully DONE).

Wires :func:`care.build_registry` into a Textual ``Screen`` that
renders the canonical tutorial + key bindings cheat-sheet. Tests
exercise:

* Compose — both panes mount and populate.
* Default registry has every documented step + binding.
* Custom registry is honoured (test injects a tiny one).
* Empty registry shows the empty-state line.
* `Esc` / `q` pop the screen.
* `?` global binding on `CareApp` pushes HelpScreen.
* Re-exports.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from care.app import CareApp
from care.help import HelpRegistry, KeyBinding, TutorialStep
from care.screens.help import HelpScreen


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(
        self,
        *,
        registry: HelpRegistry | None = None,
        include_slash_commands: bool = False,
    ) -> None:
        super().__init__()
        # Don't shadow `App._registry` — Textual's internal DOM
        # tracker lives at that slot.
        self._help_registry = registry
        self._include_slash = include_slash_commands

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(
            HelpScreen(
                self._help_registry,
                include_slash_commands=self._include_slash,
            ),
        )


def _screen(app: App) -> HelpScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, HelpScreen)
    return s


def _small_registry() -> HelpRegistry:
    reg = HelpRegistry()
    reg.add_step(TutorialStep(
        title="First step",
        body="Body of the first step.",
        hint_key="Ctrl+G",
        screen="QueryScreen",
    ))
    reg.add_step(TutorialStep(
        title="Second step",
        body="Body of the second step.",
    ))
    reg.add_binding(KeyBinding(key="Ctrl+G", action="Generate"))
    reg.add_binding(KeyBinding(
        key="Enter", action="Open", category="library",
        screen="LibraryScreen",
    ))
    return reg


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_both_panes_mount(self):
        app = _Host(registry=_small_registry())
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            screen = _screen(app)
            assert screen.query_one("#help-tutorial-body") is not None
            assert screen.query_one("#help-bindings-body") is not None

    @pytest.mark.asyncio
    async def test_tutorial_pane_has_step_titles(self):
        app = _Host(registry=_small_registry())
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            screen = _screen(app)
            statics = list(
                screen.query("#help-tutorial-body Static"),
            )
            joined = "\n".join(str(s.content) for s in statics)
            assert "First step" in joined
            assert "Second step" in joined
            assert "Ctrl+G" in joined

    @pytest.mark.asyncio
    async def test_bindings_pane_groups_by_category(self):
        app = _Host(registry=_small_registry())
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            screen = _screen(app)
            statics = list(
                screen.query("#help-bindings-body Static"),
            )
            joined = "\n".join(str(s.content) for s in statics)
            assert "# global" in joined
            assert "# library" in joined
            assert "Ctrl+G" in joined
            assert "Enter" in joined
            assert "LibraryScreen" in joined


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


class TestDefaults:
    @pytest.mark.asyncio
    async def test_default_registry_used_when_none_supplied(self):
        app = _Host(registry=None)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            screen = _screen(app)
            assert screen.step_count >= 7
            assert screen.binding_count >= 17

    @pytest.mark.asyncio
    async def test_default_registry_includes_canonical_steps(self):
        app = _Host(registry=None)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            screen = _screen(app)
            titles = screen._step_titles()
            assert "Welcome to CARE" in titles
            assert any("library" in t.lower() for t in titles)

    @pytest.mark.asyncio
    async def test_default_registry_includes_documented_keys(self):
        app = _Host(registry=None)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            screen = _screen(app)
            keys = screen._binding_keys()
            for required in ("Ctrl+P", "Ctrl+Q", "Esc", "?"):
                assert required in keys


# ---------------------------------------------------------------------------
# Empty registry
# ---------------------------------------------------------------------------


class TestEmptyState:
    @pytest.mark.asyncio
    async def test_empty_registry_shows_empty_line(self):
        empty = HelpRegistry()
        app = _Host(registry=empty)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            screen = _screen(app)
            empty_line = screen.query_one("#help-empty", Static)
            assert "(no help content" in str(empty_line.content)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


class TestNavigation:
    @pytest.mark.asyncio
    async def test_escape_pops_screen(self):
        app = _Host(registry=_small_registry())
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            depth_before = len(app.screen_stack)
            screen = _screen(app)
            screen.action_back()
            await pilot.pause()
            assert len(app.screen_stack) < depth_before

    @pytest.mark.asyncio
    async def test_action_open_help_pushes_helpscreen(self):
        # Bypass the WelcomeScreen splash race by pushing directly
        # via a CareApp-side action call. Some prior tests in the
        # full suite leave Textual state that throws off boot
        # timing, so we test the action method (not the keyboard
        # binding) and check that *some* HelpScreen is in the
        # stack after a generous quiet period.
        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(12):
                await pilot.pause()
            app.action_open_help()
            for _ in range(12):
                await pilot.pause()
            assert any(
                isinstance(s, HelpScreen) for s in app.screen_stack
            )

    def test_app_has_help_binding(self):
        # Make sure the `?` key is wired on the app class so
        # the global binding fires action_open_help.
        actions = {b.action for b in CareApp.BINDINGS}
        assert "open_help" in actions


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestFormatBinding:
    def test_global_no_screen_suffix(self):
        out = HelpScreen._format_binding(
            KeyBinding(key="Ctrl+Q", action="Quit"),
        )
        assert "Ctrl+Q" in out
        assert "Quit" in out
        assert "(" not in out

    def test_scoped_includes_screen_suffix(self):
        out = HelpScreen._format_binding(
            KeyBinding(
                key="Enter",
                action="Open",
                category="library",
                screen="LibraryScreen",
            ),
        )
        assert "(LibraryScreen)" in out


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestSlashCommandPopulation:
    """§2 P1 — HelpScreen surfaces every registered slash
    command under the `"chat"` category so users can scan
    the cheat-sheet for available commands."""

    def test_populate_adds_one_binding_per_command(self):
        from care.screens.chat import _COMMAND_HANDLERS
        from care.screens.help import _populate_slash_commands

        reg = HelpRegistry()
        _populate_slash_commands(reg)
        slash_bindings = [
            b for b in reg.bindings() if b.category == "chat"
        ]
        assert len(slash_bindings) == len(_COMMAND_HANDLERS)
        # Every registered cmd is present.
        registered_keys = {
            f"/{name}" for name in _COMMAND_HANDLERS
        }
        rendered_keys = {b.key for b in slash_bindings}
        assert registered_keys == rendered_keys

    def test_populate_uses_blurb_when_available(self):
        from care.screens.chat import ChatScreen
        from care.screens.help import _populate_slash_commands

        reg = HelpRegistry()
        _populate_slash_commands(reg)
        # `/help` has a well-known blurb.
        help_row = next(
            b for b in reg.bindings() if b.key == "/help"
        )
        assert help_row.action == (
            ChatScreen._COMMAND_BLURBS["help"]
        )

    def test_populate_is_idempotent_on_repeat_calls(self):
        from care.screens.help import _populate_slash_commands

        reg = HelpRegistry()
        _populate_slash_commands(reg)
        first_count = len(reg.bindings())
        _populate_slash_commands(reg)
        # No duplicates added — same key skips.
        assert len(reg.bindings()) == first_count

    def test_screen_default_includes_slash_commands(self):
        # Constructing without registry pulls
        # `build_registry()` + `_populate_slash_commands`.
        from care.screens.chat import _COMMAND_HANDLERS

        screen = HelpScreen()
        slash_keys = {
            b.key for b in screen.registry.bindings()
            if b.category == "chat"
        }
        # Every registered cmd appears in the screen's
        # registry.
        for name in _COMMAND_HANDLERS:
            assert f"/{name}" in slash_keys

    def test_include_slash_commands_false_skips_population(self):
        screen = HelpScreen(
            registry=HelpRegistry(),
            include_slash_commands=False,
        )
        chat_bindings = [
            b for b in screen.registry.bindings()
            if b.category == "chat"
        ]
        assert chat_bindings == []


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import HelpScreen as H

        assert H is HelpScreen
