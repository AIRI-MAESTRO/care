"""Tests for the `CareApp` shell (TODO §1.1 P0.1).

Pilots the app via Textual's `run_test()` and asserts the
facade slots + mode reactive land correctly. Future workflow-
screen tests follow this pattern (mount inside a minimal
host App, run_test pause, assert state).
"""

from __future__ import annotations

import pytest

from care.app import CareApp
from care.config import CareConfig, MageConfig
from care.runtime import SessionTokenCounter, TaskRegistry, ThemePreference


# ---------------------------------------------------------------------------
# Construction (no Pilot — just exercise __init__)
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self, tmp_path, monkeypatch):
        # Point DEFAULT_CONFIG_PATH at a tmp dir to assert the
        # first-run-vs-returning split without touching the
        # user's real config.
        from care import app as app_module
        from care import config as config_module

        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        app = CareApp()
        # No config file exists → first_run.
        assert app._initial_mode == "first_run"
        # Facade slots populated with sensible defaults.
        assert isinstance(app.config, CareConfig)
        assert app.memory is None
        assert app.platform is None
        assert isinstance(app.task_registry, TaskRegistry)
        assert isinstance(app.token_counter, SessionTokenCounter)
        assert isinstance(app.theme_pref, ThemePreference)

    def test_returning_when_config_file_exists(self, tmp_path, monkeypatch):
        from care import app as app_module
        from care import config as config_module

        fake_path = tmp_path / "config.toml"
        fake_path.write_text("")  # empty file is fine; pydantic defaults fill in
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        app = CareApp()
        assert app._initial_mode == "returning"

    def test_explicit_mode_override(self, tmp_path, monkeypatch):
        from care import app as app_module
        from care import config as config_module

        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        # File doesn't exist → would normally be first_run.
        app = CareApp(mode="returning")
        assert app._initial_mode == "returning"

    def test_explicit_config_short_circuits_load(self):
        cfg = CareConfig(mage=MageConfig(api_key="sk-explicit", model="m"))
        app = CareApp(config=cfg)
        assert app.config is cfg
        assert app.config.mage.api_key == "sk-explicit"

    def test_explicit_facades_used(self):
        registry = TaskRegistry()
        counter = SessionTokenCounter()
        pref = ThemePreference(theme_name="dark")
        app = CareApp(
            task_registry=registry,
            token_counter=counter,
            theme_pref=pref,
        )
        assert app.task_registry is registry
        assert app.token_counter is counter
        assert app.theme_pref is pref

    def test_memory_platform_default_none(self):
        # No credentials wired → screens that need Memory will
        # explicitly populate the slot later.
        app = CareApp()
        assert app.memory is None
        assert app.platform is None

    def test_memory_platform_explicit(self):
        sentinel_memory = object()
        sentinel_platform = object()
        app = CareApp(
            memory=sentinel_memory, platform=sentinel_platform,
        )
        assert app.memory is sentinel_memory
        assert app.platform is sentinel_platform

    def test_theme_pref_fallback_to_default(self, tmp_path, monkeypatch):
        # Point the theme store at a path that doesn't exist →
        # falls back to the default ThemePreference.
        from care.runtime import theme as theme_module

        monkeypatch.setattr(
            theme_module, "DEFAULT_THEME_PATH", tmp_path / "theme.json",
        )
        app = CareApp()
        assert app.theme_pref.theme_name == "auto"


# ---------------------------------------------------------------------------
# Pilot — mount + screen-stack
# ---------------------------------------------------------------------------


class TestMount:
    @pytest.mark.asyncio
    async def test_pushes_boot_screen(self, tmp_path, monkeypatch):
        from care import app as app_module
        from care import config as config_module

        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Screen stack: default + boot screen on top.
            assert len(app.screen_stack) >= 1
            assert app.screen is not None

    @pytest.mark.asyncio
    async def test_mode_reactive_set_on_mount(self, tmp_path, monkeypatch):
        from care import app as app_module
        from care import config as config_module

        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.mode == "first_run"

    @pytest.mark.asyncio
    async def test_returning_mode_pilot(self, tmp_path, monkeypatch):
        from care import app as app_module
        from care import config as config_module

        fake_path = tmp_path / "config.toml"
        fake_path.write_text("")
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.mode == "returning"

    @pytest.mark.asyncio
    async def test_screen_stack_push_pop(self, tmp_path, monkeypatch):
        from care import app as app_module
        from care import config as config_module

        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        from textual.screen import Screen

        class _DummyScreen(Screen):
            pass

        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            initial_depth = len(app.screen_stack)
            app.push_screen(_DummyScreen())
            await pilot.pause()
            assert len(app.screen_stack) == initial_depth + 1
            app.pop_screen()
            await pilot.pause()
            assert len(app.screen_stack) == initial_depth


# ---------------------------------------------------------------------------
# App-wide copy-on-selection
# ---------------------------------------------------------------------------


class TestCopyOnSelection:
    """The app-level `on_text_selected` handler gives every screen
    the chat's copy-on-drag-release gesture."""

    def test_copy_selection_helper_copies_and_toasts(self, monkeypatch):
        from care.runtime import clipboard

        captured: dict[str, str] = {}
        monkeypatch.setattr(
            clipboard, "copy_text",
            lambda app, text: captured.setdefault("text", text) or True,
        )
        toasts: list[str] = []

        class _App:
            def notify(self, message, **kw):
                toasts.append(message)

        class _Source:
            def get_selected_text(self):
                return "hello world"

        chars = clipboard.copy_selection(_App(), _Source())
        assert chars == len("hello world")
        assert captured["text"] == "hello world"
        assert toasts and "Copied 11 chars" in toasts[0]

    def test_copy_selection_helper_noop_on_empty(self, monkeypatch):
        from care.runtime import clipboard

        called: list[str] = []
        monkeypatch.setattr(
            clipboard, "copy_text",
            lambda app, text: called.append(text) or True,
        )

        class _Source:
            def get_selected_text(self):
                return "   \n"

        assert clipboard.copy_selection(object(), _Source()) is None
        assert called == []

    @pytest.mark.asyncio
    async def test_app_handler_copies_active_screen_selection(
        self, monkeypatch,
    ):
        from textual import events
        from care.runtime import clipboard

        captured: dict[str, str] = {}
        monkeypatch.setattr(
            clipboard, "copy_text",
            lambda app, text: captured.setdefault("text", text) or True,
        )
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            monkeypatch.setattr(
                app.screen, "get_selected_text",
                lambda: "selected on a non-chat screen",
            )
            app.on_text_selected(events.TextSelected())
            await pilot.pause()
        assert captured.get("text") == "selected on a non-chat screen"


# ---------------------------------------------------------------------------
# Palette navigation dispatch (TODO §2 P0 Screens group)
# ---------------------------------------------------------------------------


class TestPaletteScreensDispatch:
    """Locks the wiring between the Screens-group palette
    entries (`open_chat` / `open_artifacts` / `show_library`
    / `open_evolution` / `open_settings`) and the matching
    `action_palette_*` methods on `CareApp`. Asserts both
    sides exist + that the dispatch table maps each id to a
    callable.
    """

    def test_every_screens_action_has_dispatcher(self):
        from care.app import _PALETTE_ACTION_DISPATCH

        for action_id in (
            "open_chat",
            "open_artifacts",
            "show_library",
            "open_evolution",
            "open_settings",
        ):
            assert action_id in _PALETTE_ACTION_DISPATCH, (
                f"{action_id} missing from _PALETTE_ACTION_DISPATCH"
            )
            assert callable(_PALETTE_ACTION_DISPATCH[action_id])

    def _zero_welcome_splash(self, monkeypatch) -> None:
        """Pin `WelcomeScreen.DEFAULT_SPLASH_SECONDS = 0.0` so
        the boot routing fires via `call_later` (next event-loop
        tick) instead of a 200 ms `set_timer`. That makes the
        boot stack deterministic across a single
        `await pilot.pause()` — without it, `pilot.pause` may
        return before the splash timer fires, and
        `switch_screen(...)` later pops a screen the test just
        pushed, producing flaky assertions."""
        from care.screens.welcome import WelcomeScreen

        monkeypatch.setattr(
            WelcomeScreen, "DEFAULT_SPLASH_SECONDS", 0.0,
        )

    async def _settle_boot(self, pilot) -> None:
        """Drain `WelcomeScreen`'s splash-routing path. Used in
        tandem with :meth:`_zero_welcome_splash` so the
        sequence is deterministic — the call-later landed on
        the queue, and these pauses drain it + the destination
        screen's own mount cycle."""
        for _ in range(6):
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_palette_open_chat_pops_to_chat(self, tmp_path, monkeypatch):
        from textual.screen import Screen
        from care import app as app_module
        from care import config as config_module
        from care.screens.chat import ChatScreen

        fake_path = tmp_path / "config.toml"
        fake_path.write_text("")  # returning mode → boots to chat
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)

        class _Cover(Screen):
            pass

        app = CareApp()
        async with app.run_test() as pilot:
            await self._settle_boot(pilot)
            # The boot screen is WelcomeScreen, not ChatScreen.
            # Push a ChatScreen to simulate "user is on chat",
            # then layer a cover screen on top so the dispatcher
            # has something to pop.
            app.push_screen(ChatScreen())
            await pilot.pause()
            app.push_screen(_Cover())
            await pilot.pause()
            assert not isinstance(app.screen, ChatScreen)
            app.action_palette_open_chat()
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)

    @pytest.mark.asyncio
    async def test_palette_open_chat_pushes_when_missing(
        self, tmp_path, monkeypatch,
    ):
        from care import app as app_module
        from care import config as config_module
        from care.screens.chat import ChatScreen

        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        self._zero_welcome_splash(monkeypatch)
        app = CareApp()
        async with app.run_test() as pilot:
            await self._settle_boot(pilot)
            # `ChatScreen` may already be on the stack as
            # part of the first-run boot routing (SettingsScreen
            # → ChatScreen happens in some paths). The contract
            # of this test is that, regardless of starting
            # state, calling the dispatcher *guarantees* a
            # ChatScreen is reachable afterwards. Drop the
            # existing chat (if any) so we exercise the push
            # branch deterministically.
            while any(
                isinstance(s, ChatScreen) for s in app.screen_stack
            ):
                if len(app.screen_stack) <= 1:
                    break
                app.pop_screen()
                await pilot.pause()
            chat_present_before = any(
                isinstance(s, ChatScreen) for s in app.screen_stack
            )
            app.action_palette_open_chat()
            await pilot.pause()
            assert any(
                isinstance(s, ChatScreen) for s in app.screen_stack
            ), (
                f"open_chat dispatcher should reach ChatScreen "
                f"(chat_present_before={chat_present_before}, "
                f"stack={[type(s).__name__ for s in app.screen_stack]})"
            )

    @pytest.mark.asyncio
    async def test_palette_open_settings_pushes_settings_screen(
        self, tmp_path, monkeypatch,
    ):
        from care import app as app_module
        from care import config as config_module
        from care.screens.settings import SettingsScreen

        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        self._zero_welcome_splash(monkeypatch)
        app = CareApp()
        async with app.run_test() as pilot:
            await self._settle_boot(pilot)
            app.action_palette_open_settings()
            await pilot.pause()
            assert any(
                isinstance(s, SettingsScreen) for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_palette_open_library_pushes_library_screen(
        self, tmp_path, monkeypatch,
    ):
        from care import app as app_module
        from care import config as config_module
        from care.screens.library import LibraryScreen

        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        self._zero_welcome_splash(monkeypatch)
        app = CareApp()
        async with app.run_test() as pilot:
            await self._settle_boot(pilot)
            app.action_palette_open_library()
            await pilot.pause()
            assert any(
                isinstance(s, LibraryScreen) for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_palette_open_artifacts_warns_without_chat(
        self, tmp_path, monkeypatch,
    ):
        # When no ChatScreen is mounted (e.g. palette opened
        # from a fresh first_run boot), the palette opener
        # warns the user to open chat first — there's no
        # session artifact store to read otherwise. The real
        # ArtifactsScreen shipped in §3 P0 (iter 14); the
        # chat-mounted path is exercised in
        # `tests/test_screen_artifacts.py`.
        from care import app as app_module
        from care import config as config_module

        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        self._zero_welcome_splash(monkeypatch)
        app = CareApp()
        toasts: list[tuple[str, str]] = []

        async with app.run_test() as pilot:
            await self._settle_boot(pilot)
            original = app.push_toast

            def _spy(message: str, *, severity: str = "info", ttl=None):  # type: ignore[no-redef]
                toasts.append((message, severity))
                return original(message, severity=severity, ttl=ttl)

            app.push_toast = _spy  # type: ignore[method-assign]
            app.action_palette_open_artifacts()
            await pilot.pause()
        assert any(
            "Open the chat first" in m for m, _ in toasts
        )

    @pytest.mark.asyncio
    async def test_palette_open_evolution_toasts(self, tmp_path, monkeypatch):
        # With §5 P0 EvolutionDashboard landed, the no-platform
        # branch surfaces a "needs a configured Platform facade"
        # warning toast rather than pushing the dashboard.
        from care import app as app_module
        from care import config as config_module

        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        app = CareApp()
        toasts: list[tuple[str, str]] = []
        async with app.run_test() as pilot:
            await self._settle_boot(pilot)
            original = app.push_toast

            def _spy(message: str, *, severity: str = "info", ttl=None):  # type: ignore[no-redef]
                toasts.append((message, severity))
                return original(message, severity=severity, ttl=ttl)

            app.push_toast = _spy  # type: ignore[method-assign]
            # The app now always builds a facade from config; null it to
            # exercise the no-platform warning path this test targets.
            app.platform = None
            app.action_palette_open_evolution()
            await pilot.pause()
        assert any(
            "Platform facade" in m for m, _ in toasts
        )


# ---------------------------------------------------------------------------
# Title / sub-title metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_title(self):
        assert CareApp.TITLE == "MAESTRO"

    def test_subtitle(self):
        assert "Collaborative" in CareApp.SUB_TITLE


# ---------------------------------------------------------------------------
# Run entry point
# ---------------------------------------------------------------------------


class TestRunEntryPoint:
    def test_run_callable_exists(self):
        from care.app import run

        assert callable(run)


# ---------------------------------------------------------------------------
# Settings-saved → chat breadcrumb
# ---------------------------------------------------------------------------


class _RecordingChat:
    """Stand-in for the live ChatScreen — records the breadcrumb the
    save handler posts so the test asserts the wiring, not ChatScreen
    internals."""

    def __init__(self) -> None:
        self.changes: list[str] | None = None
        self.calls = 0
        self.relocalized = 0

    def post_settings_updated(self, changes):
        self.calls += 1
        self.changes = changes

    def relocalize(self):
        self.relocalized += 1


class TestSettingsSavedBreadcrumb:
    """`on_settings_screen_saved` posts a `Settings were updated!`
    breadcrumb (with a masked field diff) into the chat after a save."""

    @pytest.mark.asyncio
    async def test_save_posts_masked_diff_to_chat(self, tmp_path, monkeypatch):
        from care import app as app_module
        from care import config as config_module
        from care.screens.settings import SettingsScreen, SettingsSnapshot

        fake_path = tmp_path / "config.toml"
        fake_path.write_text("")
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)

        old_cfg = CareConfig()
        # Deterministic reload so the handler can't early-return on a
        # malformed/env-polluted load.
        monkeypatch.setattr(
            CareConfig, "load", classmethod(lambda cls, *a, **k: old_cfg),
        )

        app = CareApp(config=old_cfg, mode="returning")
        chat = _RecordingChat()
        monkeypatch.setattr(app, "_find_chat_screen", lambda: chat)

        async with app.run_test() as pilot:
            await pilot.pause()
            app.config = old_cfg
            new_cfg = old_cfg.model_copy(update={
                "defaults": old_cfg.defaults.model_copy(
                    update={"ui_language": "en"},
                ),
                "mage": old_cfg.mage.model_copy(
                    update={"api_key": "sk-super-secret"},
                ),
            })
            app.on_settings_screen_saved(
                SettingsScreen.Saved(
                    SettingsSnapshot(
                        config=new_cfg, theme_name=None, report=None,
                    ),
                ),
            )
            await pilot.pause()

        assert chat.calls == 1
        rows = chat.changes or []
        assert "defaults.ui_language: ru → en" in rows
        # The credential edit is surfaced but masked.
        assert "mage.api_key: set" in rows
        assert all("sk-super-secret" not in r for r in rows)
        # ui_language flipped → the chat chrome was re-localized.
        assert chat.relocalized == 1

    @pytest.mark.asyncio
    async def test_no_language_change_skips_relocalize(
        self, tmp_path, monkeypatch,
    ):
        from care import app as app_module
        from care import config as config_module
        from care.screens.settings import SettingsScreen, SettingsSnapshot

        fake_path = tmp_path / "config.toml"
        fake_path.write_text("")
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)

        old_cfg = CareConfig()
        monkeypatch.setattr(
            CareConfig, "load", classmethod(lambda cls, *a, **k: old_cfg),
        )
        app = CareApp(config=old_cfg, mode="returning")
        chat = _RecordingChat()
        monkeypatch.setattr(app, "_find_chat_screen", lambda: chat)

        async with app.run_test() as pilot:
            await pilot.pause()
            app.config = old_cfg
            # Edit a non-language field — same ui_language.
            new_cfg = old_cfg.model_copy(update={
                "mage": old_cfg.mage.model_copy(update={"model": "x/y"}),
            })
            app.on_settings_screen_saved(
                SettingsScreen.Saved(
                    SettingsSnapshot(
                        config=new_cfg, theme_name=None, report=None,
                    ),
                ),
            )
            await pilot.pause()

        # Breadcrumb still posts, but no re-localize when language held.
        assert chat.calls == 1
        assert chat.relocalized == 0

    @pytest.mark.asyncio
    async def test_no_chat_on_stack_is_safe(self, tmp_path, monkeypatch):
        # First-run path: no ChatScreen underneath Settings → handler
        # must not blow up trying to post a breadcrumb.
        from care import app as app_module
        from care import config as config_module
        from care.screens.settings import SettingsScreen, SettingsSnapshot

        fake_path = tmp_path / "config.toml"
        fake_path.write_text("")
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)

        old_cfg = CareConfig()
        monkeypatch.setattr(
            CareConfig, "load", classmethod(lambda cls, *a, **k: old_cfg),
        )
        app = CareApp(config=old_cfg, mode="returning")
        monkeypatch.setattr(app, "_find_chat_screen", lambda: None)

        async with app.run_test() as pilot:
            await pilot.pause()
            app.config = old_cfg
            # Should be a no-op (no chat), not a crash.
            app.on_settings_screen_saved(
                SettingsScreen.Saved(
                    SettingsSnapshot(
                        config=old_cfg, theme_name=None, report=None,
                    ),
                ),
            )
            await pilot.pause()


class TestInspectionRunAction:
    """Library → Inspect 'Run' must actually execute the saved chain
    (not no-op): the InspectionScreen posts ``ActionRequested('run',
    id)`` and the app routes it through the library-run pipeline,
    pushing an ExecutionScreen for live progress."""

    def _boot(self, tmp_path, monkeypatch) -> None:
        from care import app as app_module
        from care import config as config_module
        from care.screens.welcome import WelcomeScreen

        fake = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake)
        monkeypatch.setattr(
            WelcomeScreen, "DEFAULT_SPLASH_SECONDS", 0.0,
        )

    @staticmethod
    def _evt(action: str, entity_id: str):
        from types import SimpleNamespace

        return SimpleNamespace(action=action, entity_id=entity_id)

    @staticmethod
    def _spy_toasts(app, toasts) -> None:
        original = app.push_toast

        def _spy(message, *, severity="info", ttl=None):
            toasts.append((message, severity))
            return original(message, severity=severity, ttl=ttl)

        app.push_toast = _spy  # type: ignore[method-assign]

    @pytest.mark.asyncio
    async def test_run_without_memory_toasts_error(
        self, tmp_path, monkeypatch,
    ):
        self._boot(tmp_path, monkeypatch)
        app = CareApp()
        toasts: list[tuple[str, str]] = []
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            app.memory = None
            self._spy_toasts(app, toasts)
            app.on_inspection_screen_action_requested(
                self._evt("run", "agent-1"),
            )
            await pilot.pause()
        assert any("Memory facade" in m for m, _ in toasts)

    @pytest.mark.asyncio
    async def test_run_executes_pipeline_and_pushes_execution_screen(
        self, tmp_path, monkeypatch,
    ):
        self._boot(tmp_path, monkeypatch)
        from types import SimpleNamespace

        import care.runtime.carl_streamer as cs_mod
        import care.runtime.library_run as lr_mod
        import care.runtime.llm_client as llm_mod

        # Stub the heavy pipeline the worker imports lazily so the run
        # resolves deterministically offline.
        monkeypatch.setattr(
            llm_mod, "build_carl_llm_client", lambda *a, **k: object(),
        )
        monkeypatch.setattr(cs_mod, "CarlStreamer", lambda target: object())

        async def _fake_load(memory, entity_id, **k):
            return SimpleNamespace(
                draft=object(), chain=object(), entity_id=entity_id,
                display_name="Demo",
            )

        summary = SimpleNamespace(
            success=True, duration_seconds=1.5, step_count=3,
            error_message="",
        )

        async def _fake_exec(memory, plan, draft, **k):
            return SimpleNamespace(summary=summary, run_id="run-x")

        monkeypatch.setattr(lr_mod, "load_run_plan", _fake_load)
        monkeypatch.setattr(lr_mod, "execute_library_run", _fake_exec)

        app = CareApp(memory=SimpleNamespace())
        toasts: list[tuple[str, str]] = []
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            # The RunContextModal collects the task — stub it as a
            # confirmed submit so the worker proceeds to execution.
            async def _fake_wait(screen):
                return SimpleNamespace(submitted=True, draft=object())

            monkeypatch.setattr(app, "push_screen_wait", _fake_wait)
            self._spy_toasts(app, toasts)
            app.on_inspection_screen_action_requested(
                self._evt("run", "agent-1"),
            )
            for _ in range(8):
                await pilot.pause()
            from care.screens.execution import ExecutionScreen

            assert isinstance(app.screen_stack[-1], ExecutionScreen)
        assert any("Run succeeded" in m for m, _ in toasts)

    @pytest.mark.asyncio
    async def test_run_cancelled_in_modal_does_not_execute(
        self, tmp_path, monkeypatch,
    ):
        self._boot(tmp_path, monkeypatch)
        from types import SimpleNamespace

        import care.runtime.library_run as lr_mod
        import care.runtime.llm_client as llm_mod

        monkeypatch.setattr(
            llm_mod, "build_carl_llm_client", lambda *a, **k: object(),
        )

        async def _fake_load(memory, entity_id, **k):
            return SimpleNamespace(
                draft=object(), chain=object(), entity_id=entity_id,
                display_name="Demo",
            )

        exec_calls: list[str] = []

        async def _fake_exec(memory, plan, draft, **k):
            exec_calls.append("called")
            return SimpleNamespace(
                summary=SimpleNamespace(success=True), run_id="x",
            )

        monkeypatch.setattr(lr_mod, "load_run_plan", _fake_load)
        monkeypatch.setattr(lr_mod, "execute_library_run", _fake_exec)

        app = CareApp(memory=SimpleNamespace())
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()

            async def _fake_wait(screen):
                return SimpleNamespace(submitted=False, draft=object())

            monkeypatch.setattr(app, "push_screen_wait", _fake_wait)
            app.on_inspection_screen_action_requested(
                self._evt("run", "agent-1"),
            )
            for _ in range(8):
                await pilot.pause()
            from care.screens.execution import ExecutionScreen

            # Cancel ⇒ no execution + no ExecutionScreen pushed.
            assert exec_calls == []
            assert not isinstance(app.screen_stack[-1], ExecutionScreen)

    @pytest.mark.asyncio
    async def test_run_handler_routes_to_push_run_for(
        self, tmp_path, monkeypatch,
    ):
        # The action-bar click path (ActionRequested) lands on the run
        # helper — not the old no-op stub toast.
        from types import SimpleNamespace

        self._boot(tmp_path, monkeypatch)
        app = CareApp(memory=SimpleNamespace())
        seen: list[str] = []
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            app._push_run_for = lambda eid: seen.append(eid)  # type: ignore[method-assign]
            app.on_inspection_screen_action_requested(
                self._evt("run", "agent-9"),
            )
            await pilot.pause()
        assert seen == ["agent-9"]
