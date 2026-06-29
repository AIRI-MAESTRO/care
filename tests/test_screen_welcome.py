"""Pilot tests for `WelcomeScreen` (TODO §1.1 P0.2).

Exercises the splash + mode-based routing without timing out:
tests pass ``splash_seconds=0.0`` and a custom
``next_screen_factory`` that records the mode.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.screen import Screen

from care.screens.welcome import WelcomeScreen, default_next_screen


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubTargetScreen(Screen):
    """Lightweight stand-in for the SettingsScreen /
    LibraryScreen that aren't shipped yet."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode_received = mode


def _factory(observed: list[str]):
    """Build a `next_screen_factory` that records every mode
    it's called with."""

    def _build(mode: str) -> Screen:
        observed.append(mode)
        return _StubTargetScreen(mode)

    return _build


class _WelcomeHarnessApp(App):
    """Minimal host that mounts a custom WelcomeScreen on
    boot. Mode is settable via constructor kwarg so each test
    can pin the branch under exercise."""

    def __init__(
        self,
        *,
        mode: str = "returning",
        next_screen_factory=None,
        splash_seconds: float = 0.0,
    ) -> None:
        super().__init__()
        self._initial_mode = mode
        self._next_screen_factory = next_screen_factory
        self._splash_seconds = splash_seconds

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        # Stamp the mode reactive-ish attribute the screen
        # reads.
        self.mode = self._initial_mode  # type: ignore[attr-defined]
        self.push_screen(
            WelcomeScreen(
                splash_seconds=self._splash_seconds,
                next_screen_factory=self._next_screen_factory,
            )
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_splash_seconds(self):
        screen = WelcomeScreen()
        assert screen.splash_seconds == WelcomeScreen.DEFAULT_SPLASH_SECONDS

    def test_explicit_splash_seconds(self):
        screen = WelcomeScreen(splash_seconds=0.5)
        assert screen.splash_seconds == 0.5

    def test_zero_splash_seconds(self):
        # Zero is valid — used in tests to skip the splash.
        screen = WelcomeScreen(splash_seconds=0.0)
        assert screen.splash_seconds == 0.0

    def test_default_next_screen_factory_none(self):
        screen = WelcomeScreen()
        assert screen._next_screen_factory is None

    def test_routed_starts_false(self):
        screen = WelcomeScreen()
        assert screen._routed is False
        assert screen.routed_to_mode is None


# ---------------------------------------------------------------------------
# Pilot — mode-based routing
# ---------------------------------------------------------------------------


class TestRouting:
    @pytest.mark.asyncio
    async def test_returning_mode_routes(self):
        observed: list[str] = []
        app = _WelcomeHarnessApp(
            mode="returning",
            next_screen_factory=_factory(observed),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            # Give the timer a beat to fire (splash_seconds=0).
            await pilot.pause()
            assert observed == ["returning"]
            # The screen on top is the routed target.
            assert isinstance(app.screen, _StubTargetScreen)
            assert app.screen.mode_received == "returning"

    @pytest.mark.asyncio
    async def test_first_run_mode_routes(self):
        observed: list[str] = []
        app = _WelcomeHarnessApp(
            mode="first_run",
            next_screen_factory=_factory(observed),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert observed == ["first_run"]
            assert isinstance(app.screen, _StubTargetScreen)
            assert app.screen.mode_received == "first_run"

    @pytest.mark.asyncio
    async def test_unknown_mode_passes_through(self):
        # The screen doesn't validate mode strings — it just
        # forwards whatever the app exposes. Lets future
        # custom modes plug in without changing this layer.
        observed: list[str] = []
        app = _WelcomeHarnessApp(
            mode="custom",
            next_screen_factory=_factory(observed),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert observed == ["custom"]

    @pytest.mark.asyncio
    async def test_missing_app_mode_defaults_to_returning(self):
        # Host App doesn't set `mode` at all — the screen
        # falls back to "returning" rather than crashing.
        class _BareHostApp(App):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(
                    WelcomeScreen(
                        splash_seconds=0.0,
                        next_screen_factory=lambda m: _StubTargetScreen(m),
                    )
                )

        app = _BareHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, _StubTargetScreen)
            assert app.screen.mode_received == "returning"


# ---------------------------------------------------------------------------
# Re-entry guard
# ---------------------------------------------------------------------------


class TestRouteGuard:
    @pytest.mark.asyncio
    async def test_route_idempotent(self):
        # Calling `_route` twice (e.g. double-fired timer)
        # only switches once.
        observed: list[str] = []
        screen = WelcomeScreen(
            splash_seconds=0.0,
            next_screen_factory=_factory(observed),
        )

        class _ProbeApp(App):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.mode = "returning"  # type: ignore[attr-defined]
                self.push_screen(screen)

        app = _ProbeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # Fire the route a second time — should be no-op.
            screen._route()
            screen._route()
            assert observed == ["returning"]


# ---------------------------------------------------------------------------
# Splash composition
# ---------------------------------------------------------------------------


class TestComposition:
    @pytest.mark.asyncio
    async def test_renders_title_and_subtitle(self):
        # Use a longer splash to keep the screen visible long
        # enough to query, then assert the widgets are there.
        observed: list[str] = []

        class _NoRouteApp(App):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.mode = "returning"  # type: ignore[attr-defined]
                # Use a long splash so the screen stays mounted
                # for the assertion.
                self.push_screen(
                    WelcomeScreen(
                        splash_seconds=10.0,
                        next_screen_factory=_factory(observed),
                    )
                )

        app = _NoRouteApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            from textual.widgets import Static

            statics = app.screen.query(Static)
            titles = [s for s in statics if s.id == "welcome-title"]
            subs = [s for s in statics if s.id == "welcome-subtitle"]
            assert len(titles) == 1
            assert len(subs) == 1
            # Splash hasn't fired yet, no routing.
            assert observed == []


# ---------------------------------------------------------------------------
# default_next_screen factory
# ---------------------------------------------------------------------------


class TestDefaultNextScreen:
    def test_returning_returns_screen(self):
        from textual.screen import Screen as TextualScreen

        result = default_next_screen("returning")
        assert isinstance(result, TextualScreen)

    def test_first_run_returns_screen(self):
        from textual.screen import Screen as TextualScreen

        result = default_next_screen("first_run")
        assert isinstance(result, TextualScreen)


# ---------------------------------------------------------------------------
# Wired into CareApp._build_boot_screen
# ---------------------------------------------------------------------------


class TestCareAppIntegration:
    @pytest.mark.asyncio
    async def test_care_app_boots_into_welcome_then_routes(
        self, tmp_path, monkeypatch,
    ):
        from care import app as app_module
        from care import config as config_module

        # No config file → first_run mode.
        fake_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_path)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        # Speed up the splash for the test.
        monkeypatch.setattr(
            WelcomeScreen, "DEFAULT_SPLASH_SECONDS", 0.0,
        )

        from care.app import CareApp

        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Welcome should have mounted then immediately
            # routed to the demo screen (placeholder for
            # SettingsScreen / LibraryScreen).
            await pilot.pause()
            # The screen on top is no longer WelcomeScreen.
            assert not isinstance(app.screen, WelcomeScreen)
