"""Tests for ``care.help`` (TODO §9 P3).

Coverage:

1. **Frozen dataclasses** — `KeyBinding` + `TutorialStep`.
2. **Registry mutation** — `add_step` / `add_binding` /
   `steps` / `bindings` ordering.
3. **Filters** — `by_category` / `by_screen`.
4. **Default content** — canonical-flow tutorial covers
   every screen the README lists; every documented global
   binding is registered.
5. **Plugin extensions** — `register_help_extension` /
   `build_registry` / `unregister_help_extension`; buggy
   extension doesn't break the registry.
6. **Rendering** — `format_text` includes tutorial headers
   + grouped bindings; `format_markdown` produces
   copy-paste-ready output.
"""

from __future__ import annotations

import pytest

from care.help import (
    HelpRegistry,
    KeyBinding,
    TutorialStep,
    build_registry,
    default_registry,
    register_help_extension,
    unregister_help_extension,
)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


class TestFrozenShape:
    def test_key_binding_frozen(self):
        b = KeyBinding(key="Ctrl+G", action="Generate")
        with pytest.raises(Exception):
            b.action = "Other"  # type: ignore[misc]

    def test_tutorial_step_frozen(self):
        s = TutorialStep(title="Hi", body="body")
        with pytest.raises(Exception):
            s.title = "Other"  # type: ignore[misc]

    def test_key_binding_defaults(self):
        b = KeyBinding(key="X", action="do thing")
        assert b.category == "global"
        assert b.screen == ""

    def test_tutorial_step_defaults(self):
        s = TutorialStep(title="t", body="b")
        assert s.hint_key == ""
        assert s.screen == ""


# ---------------------------------------------------------------------------
# Registry mutation
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_empty_registry(self):
        reg = HelpRegistry()
        assert reg.steps() == ()
        assert reg.bindings() == ()
        assert reg.step_titles() == ()
        # format_text on an empty registry returns empty string.
        assert reg.format_text() == ""

    def test_add_step_preserves_order(self):
        reg = HelpRegistry()
        reg.add_step(TutorialStep(title="A", body="a"))
        reg.add_step(TutorialStep(title="B", body="b"))
        reg.add_step(TutorialStep(title="C", body="c"))
        titles = reg.step_titles()
        assert titles == ("A", "B", "C")

    def test_add_binding_preserves_order(self):
        reg = HelpRegistry()
        reg.add_binding(KeyBinding(key="X", action="x"))
        reg.add_binding(KeyBinding(key="Y", action="y"))
        keys = [b.key for b in reg.bindings()]
        assert keys == ["X", "Y"]

    def test_by_category_filters(self):
        reg = HelpRegistry()
        reg.add_binding(KeyBinding(key="X", action="x", category="global"))
        reg.add_binding(KeyBinding(key="Y", action="y", category="library"))
        reg.add_binding(KeyBinding(key="Z", action="z", category="library"))
        glob = reg.by_category("global")
        lib = reg.by_category("library")
        assert [b.key for b in glob] == ["X"]
        assert [b.key for b in lib] == ["Y", "Z"]

    def test_by_screen_filters(self):
        reg = HelpRegistry()
        reg.add_binding(
            KeyBinding(key="A", action="a", screen="LibraryScreen", category="library")
        )
        reg.add_binding(
            KeyBinding(key="B", action="b", screen="LibraryScreen", category="library")
        )
        reg.add_binding(KeyBinding(key="C", action="c", screen="EvolutionScreen", category="evolution"))
        lib = reg.by_screen("LibraryScreen")
        assert [b.key for b in lib] == ["A", "B"]


# ---------------------------------------------------------------------------
# Default content
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_default_has_tutorial_steps(self):
        reg = default_registry()
        # The canonical-flow tutorial has at least the
        # documented seven steps.
        assert len(reg.steps()) >= 7

    def test_default_includes_welcome_step(self):
        reg = default_registry()
        titles = reg.step_titles()
        # First step is the welcome banner.
        assert "Welcome to CARE" in titles[0]

    def test_default_covers_canonical_flow_screens(self):
        # Every screen in the README's canonical user flow has
        # at least one mention.
        reg = default_registry()
        screens = {s.screen for s in reg.steps()}
        expected = {
            "QueryScreen",
            "GenerationScreen",
            "SaveAgentModal",
            "LibraryScreen",
            "EvolutionScreen",
        }
        assert expected.issubset(screens)

    def test_default_global_bindings_present(self):
        reg = default_registry()
        globals_ = reg.by_category("global")
        keys = {b.key for b in globals_}
        # Documented global key set.
        for expected in ("Ctrl+P", "Ctrl+Q", "Ctrl+S", "Ctrl+R", "Esc"):
            assert expected in keys

    def test_default_per_screen_bindings_present(self):
        reg = default_registry()
        library_keys = {b.key for b in reg.by_category("library")}
        # README documents R / E / F / Enter / Del on the LibraryScreen.
        for key in ("Enter", "R", "E", "F", "Del"):
            assert key in library_keys

    def test_default_returns_fresh_instance_each_call(self):
        a = default_registry()
        b = default_registry()
        # Different instances; mutating one doesn't bleed.
        a.add_step(TutorialStep(title="new", body=""))
        assert len(a.steps()) > len(b.steps())


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------


class TestExtensions:
    def teardown_method(self):
        # Clean up any registrations the tests left behind.
        for ext in self._installed:
            unregister_help_extension(ext)
        self._installed.clear()

    def setup_method(self):
        self._installed: list = []

    def _install(self, ext):
        register_help_extension(ext)
        self._installed.append(ext)

    def test_extension_appends_steps(self):
        def my_ext(reg: HelpRegistry) -> None:
            reg.add_step(TutorialStep(title="plugin step", body="plugin body"))

        self._install(my_ext)
        reg = build_registry()
        titles = reg.step_titles()
        assert "plugin step" in titles
        # Plugin step lands after the defaults.
        assert titles[-1] == "plugin step"

    def test_extension_appends_binding(self):
        def my_ext(reg: HelpRegistry) -> None:
            reg.add_binding(
                KeyBinding(key="Ctrl+T", action="Toggle telemetry")
            )

        self._install(my_ext)
        reg = build_registry()
        keys = {b.key for b in reg.bindings()}
        assert "Ctrl+T" in keys

    def test_buggy_extension_does_not_break_registry(self):
        def bad_ext(reg: HelpRegistry) -> None:
            raise RuntimeError("plugin error")

        def good_ext(reg: HelpRegistry) -> None:
            reg.add_step(TutorialStep(title="good", body="b"))

        self._install(bad_ext)
        self._install(good_ext)
        reg = build_registry()
        # Bad extension's exception was swallowed; good one
        # still ran.
        assert "good" in reg.step_titles()

    def test_unregister_returns_bool(self):
        def ext(reg: HelpRegistry) -> None:
            return

        register_help_extension(ext)
        self._installed.append(ext)
        assert unregister_help_extension(ext) is True
        # Already gone.
        assert unregister_help_extension(ext) is False
        # Won't fire a second time after explicit unregister.
        # Re-clean so teardown doesn't try to remove it again.
        self._installed.remove(ext)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestChatModeContent:
    """Phase 7 P1 — help registry covers the chat-mode slash
    commands `/mode`, `/dataset`, `/evolution` so `care help`
    surfaces them alongside the legacy screen flows."""

    def test_chat_tutorial_steps_reference_each_slash_command(self):
        reg = default_registry()
        bodies = " ".join(s.body for s in reg.steps())
        titles = " ".join(s.title for s in reg.steps())
        for token in ("/mode", "/dataset", "/evolution"):
            assert token in bodies, (
                f"chat help blurb should mention {token}"
            )
        assert "Ad-Hoc" in bodies or "Interactive" in bodies or "Interactive" in bodies
        assert "Production" in bodies or "Production" in titles

    def test_chat_screen_steps_exist(self):
        reg = default_registry()
        chat_steps = [s for s in reg.steps() if s.screen == "ChatScreen"]
        # At least one tutorial step talks about the ChatScreen,
        # and at least one mentions each of /mode, /dataset,
        # /evolution. Allowed to be one consolidated step OR
        # several focused steps.
        assert chat_steps, "no ChatScreen tutorial steps registered"
        combined = " ".join(s.body for s in chat_steps)
        for token in ("/mode", "/dataset", "/evolution"):
            assert token in combined

    def test_chat_category_bindings_cover_slash_commands(self):
        reg = default_registry()
        chat_bindings = reg.by_category("chat")
        keys = {b.key for b in chat_bindings}
        assert "/mode" in keys
        assert "/help" in keys
        assert any(k.startswith("/dataset") for k in keys)
        assert any(k.startswith("/evolution") for k in keys)

    def test_chat_bindings_tagged_to_chat_screen(self):
        reg = default_registry()
        chat_bindings = reg.by_category("chat")
        assert chat_bindings, "no chat-category bindings registered"
        for binding in chat_bindings:
            assert binding.screen == "ChatScreen", (
                f"chat binding {binding.key!r} missing ChatScreen tag"
            )

    def test_format_text_renders_chat_section(self):
        reg = default_registry()
        text = reg.format_text()
        # New per-category section heading.
        assert "## chat" in text
        # Each canonical slash command appears in the rendered
        # output (either in the tutorial section or the bindings
        # section).
        assert "/mode" in text
        assert "/dataset" in text
        assert "/evolution" in text

    def test_format_markdown_renders_chat_section(self):
        reg = default_registry()
        md = reg.format_markdown()
        assert "**chat**" in md
        # Slash commands wrapped in backticks per the markdown
        # rule (`Ctrl+Q` style).
        assert "`/mode`" in md

    def test_chat_category_orders_after_global_before_library(self):
        """The format renderers iterate categories in a fixed
        order; chat sits right after global so it surfaces high
        on the help page."""
        reg = default_registry()
        text = reg.format_text()
        global_pos = text.find("## global")
        chat_pos = text.find("## chat")
        library_pos = text.find("## library")
        assert global_pos != -1
        assert chat_pos != -1
        assert library_pos != -1
        assert global_pos < chat_pos < library_pos


class TestFormatText:
    def test_includes_tutorial_section(self):
        reg = default_registry()
        text = reg.format_text()
        assert "# Tutorial" in text
        assert "1. Welcome to CARE" in text

    def test_includes_bindings_section(self):
        reg = default_registry()
        text = reg.format_text()
        assert "# Key bindings" in text
        # Globals section header.
        assert "## global" in text
        assert "Ctrl+Q" in text

    def test_per_screen_suffix_rendered(self):
        reg = HelpRegistry()
        reg.add_binding(KeyBinding(key="R", action="Run", category="library", screen="LibraryScreen"))
        text = reg.format_text()
        assert "(LibraryScreen)" in text

    def test_hint_key_rendered_under_step(self):
        reg = HelpRegistry()
        reg.add_step(TutorialStep(title="t", body="b", hint_key="Ctrl+G"))
        text = reg.format_text()
        assert "Ctrl+G" in text

    def test_empty_registry_returns_empty_string(self):
        reg = HelpRegistry()
        assert reg.format_text() == ""


class TestFormatMarkdown:
    def test_uses_markdown_headings(self):
        reg = default_registry()
        md = reg.format_markdown()
        assert "## Walkthrough" in md
        assert "## Keys" in md

    def test_keystrokes_in_backticks(self):
        reg = default_registry()
        md = reg.format_markdown()
        assert "`Ctrl+Q`" in md

    def test_hint_key_as_blockquote(self):
        reg = HelpRegistry()
        reg.add_step(TutorialStep(title="t", body="b", hint_key="Ctrl+G"))
        md = reg.format_markdown()
        assert "> Try: `Ctrl+G`" in md

    def test_per_screen_italic_suffix(self):
        reg = HelpRegistry()
        reg.add_binding(
            KeyBinding(key="R", action="Run", category="library", screen="LibraryScreen")
        )
        md = reg.format_markdown()
        assert "*(LibraryScreen)*" in md

    def test_empty_registry_markdown_is_trailing_newline_only(self):
        reg = HelpRegistry()
        md = reg.format_markdown()
        assert md == "\n"


# ---------------------------------------------------------------------------
# build_registry vs default_registry
# ---------------------------------------------------------------------------


class TestBuildRegistry:
    def test_build_registry_returns_help_registry(self):
        reg = build_registry()
        assert isinstance(reg, HelpRegistry)

    def test_build_registry_includes_defaults(self):
        reg = build_registry()
        assert len(reg.steps()) >= 7
        # Same set of globals.
        keys = {b.key for b in reg.by_category("global")}
        assert "Ctrl+Q" in keys
