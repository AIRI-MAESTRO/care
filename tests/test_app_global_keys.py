"""Pilot tests for `CareApp` global key bindings (TODO §1.1 P0.5).

Mounts the app via `run_test()`, fires every binding, and
asserts the right action method ran + the right Message
posted.
"""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.screen import Screen

from care.app import CareApp, _build_textual_bindings
from care.runtime.global_bindings import default_global_bindings


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_config_path(tmp_path, monkeypatch):
    """Point DEFAULT_CONFIG_PATH at a tmp dir so the app
    constructor doesn't touch the user's real config."""
    from care import app as app_module
    from care import config as config_module

    fake_path = tmp_path / "config.toml"
    monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
    # Also speed up the WelcomeScreen splash.
    from care.screens.welcome import WelcomeScreen

    monkeypatch.setattr(WelcomeScreen, "DEFAULT_SPLASH_SECONDS", 0.0)


# ---------------------------------------------------------------------------
# BINDINGS construction
# ---------------------------------------------------------------------------


class TestBindingsConstruction:
    def test_build_textual_bindings_one_per_default(self):
        bindings = _build_textual_bindings()
        assert len(bindings) == len(default_global_bindings())

    def test_action_names_namespaced(self):
        bindings = _build_textual_bindings()
        for b in bindings:
            assert b.action.startswith("global_")

    def test_textual_keys_match_canonical(self):
        # Bindings use the lowercased / dashed form Textual
        # expects.
        bindings = {b.action: b for b in _build_textual_bindings()}
        assert bindings["global_open_command_palette"].key == "ctrl+p"
        assert bindings["global_quit"].key == "ctrl+q"
        assert bindings["global_save_artifact"].key == "ctrl+s"
        assert bindings["global_rerun_artifact"].key == "ctrl+r"
        assert bindings["global_back"].key == "escape"

    def test_app_bindings_match_builder(self):
        # `CareApp.BINDINGS` should at minimum contain every
        # binding the global registry projects. Extra app-level
        # bindings (P0.36 Ctrl+B task list, etc.) are allowed.
        builder = {b.action_id for b in default_global_bindings()}
        app_actions = {b.action for b in CareApp.BINDINGS}
        for action in builder:
            assert f"global_{action}" in app_actions
        assert len(CareApp.BINDINGS) >= len(default_global_bindings())


# ---------------------------------------------------------------------------
# Action method dispatch
# ---------------------------------------------------------------------------


class TestActionDispatch:
    @pytest.mark.asyncio
    async def test_ctrl_p_fires_open_command_palette(self):
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.global_action_log.clear()
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert "open_command_palette" in app.global_action_log

    @pytest.mark.asyncio
    async def test_ctrl_s_fires_save_artifact(self):
        # Push LibraryScreen explicitly so the gesture isn't
        # swallowed by ChatScreen's Input (default boot for a
        # configured returning user) or SettingsScreen's own
        # Ctrl+S → Save binding.
        from care.screens.library import LibraryScreen

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.push_screen(LibraryScreen())
            for _ in range(3):
                await pilot.pause()
            app.global_action_log.clear()
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert "save_artifact" in app.global_action_log

    @pytest.mark.asyncio
    async def test_ctrl_r_fires_rerun_artifact(self):
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.global_action_log.clear()
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert "rerun_artifact" in app.global_action_log

    @pytest.mark.asyncio
    async def test_ctrl_q_exits_app(self):
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.global_action_log.clear()
            await pilot.press("ctrl+q")
            await pilot.pause()
            assert "quit" in app.global_action_log
            # `exit()` sets _exit; we don't assert the app is
            # actually torn down (Pilot teardown handles that).

    @pytest.mark.asyncio
    async def test_escape_pops_screen_when_stack_deep(self):
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # Push an extra screen so the stack is deep.
            class _Dummy(Screen):
                def compose(self) -> ComposeResult:
                    yield from ()

            app.push_screen(_Dummy())
            await pilot.pause()
            initial_depth = len(app.screen_stack)
            app.global_action_log.clear()
            await pilot.press("escape")
            await pilot.pause()
            assert "back" in app.global_action_log
            # Stack shrank by one.
            assert len(app.screen_stack) == initial_depth - 1


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestMessages:
    @pytest.mark.asyncio
    async def test_save_request_message(self):
        # Subscribe to the message and assert it fires.
        observed: list[str] = []

        class _ListenerScreen(Screen):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_care_app_save_requested(self, event) -> None:
                observed.append("save")

        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.push_screen(_ListenerScreen())
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.pause()
            assert observed == ["save"]

    @pytest.mark.asyncio
    async def test_rerun_request_message(self):
        observed: list[str] = []

        class _ListenerScreen(Screen):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_care_app_rerun_requested(self, event) -> None:
                observed.append("rerun")

        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.push_screen(_ListenerScreen())
            await pilot.pause()
            await pilot.press("ctrl+r")
            await pilot.pause()
            await pilot.pause()
            assert observed == ["rerun"]

    @pytest.mark.asyncio
    async def test_command_palette_request_message(self):
        observed: list[str] = []

        class _ListenerScreen(Screen):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_care_app_command_palette_requested(self, event) -> None:
                observed.append("palette")

        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.push_screen(_ListenerScreen())
            await pilot.pause()
            await pilot.press("ctrl+p")
            await pilot.pause()
            await pilot.pause()
            assert observed == ["palette"]


# ---------------------------------------------------------------------------
# Action log
# ---------------------------------------------------------------------------


class TestActionLog:
    def test_action_log_initialised_empty(self):
        app = CareApp()
        assert app.global_action_log == []

    @pytest.mark.asyncio
    async def test_action_log_records_every_fire(self):
        # Push LibraryScreen explicitly so the Ctrl+S / Ctrl+R
        # / Ctrl+P globals aren't shadowed by ChatScreen's Input
        # widget (default boot for a configured returning user)
        # or SettingsScreen's own Ctrl+S → Save binding.
        from care.screens.library import LibraryScreen

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.push_screen(LibraryScreen())
            for _ in range(3):
                await pilot.pause()
            app.global_action_log.clear()
            await pilot.press("ctrl+s")
            await pilot.press("ctrl+r")
            await pilot.press("ctrl+p")
            await pilot.pause()
            # Every action that fired (in arbitrary order) is
            # represented.
            assert "save_artifact" in app.global_action_log
            assert "rerun_artifact" in app.global_action_log
            assert "open_command_palette" in app.global_action_log
