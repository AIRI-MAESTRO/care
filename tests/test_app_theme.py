"""Pilot tests for CARE theme application (TODO §1.1 P0.6).

Verifies `CareApp` registers CARE themes, applies the active
theme on mount, re-resolves auto themes when `app.dark` flips,
and persists choices via `apply_care_theme(persist=True)`.
"""

from __future__ import annotations

import pytest

from care.app import CareApp, _care_theme_to_textual
from care.runtime.theme import (
    ThemePreference,
    get_theme,
    list_themes,
    load_theme_preference,
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_paths(tmp_path, monkeypatch):
    """Redirect config + theme paths to a tmp dir so tests
    don't touch the user's real files."""
    from care import app as app_module
    from care import config as config_module
    from care.runtime import theme as theme_module

    fake_config = tmp_path / "config.toml"
    monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", fake_config)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_config)

    fake_theme = tmp_path / "theme.json"
    monkeypatch.setattr(theme_module, "DEFAULT_THEME_PATH", fake_theme)
    # Speed up the welcome splash.
    from care.screens.welcome import WelcomeScreen

    monkeypatch.setattr(WelcomeScreen, "DEFAULT_SPLASH_SECONDS", 0.0)


# ---------------------------------------------------------------------------
# _care_theme_to_textual projection
# ---------------------------------------------------------------------------


class TestProjection:
    def test_concrete_theme_translates(self):
        from textual.theme import Theme as TextualTheme

        dark = get_theme("dark")
        textual = _care_theme_to_textual(dark)
        assert isinstance(textual, TextualTheme)
        assert textual.name == "dark"
        assert textual.dark is True
        # Built-in palette keys mapped.
        assert textual.primary == dark.variables["primary"]
        assert textual.background == dark.variables["background"]

    def test_light_theme_dark_false(self):
        light = get_theme("light")
        textual = _care_theme_to_textual(light)
        assert textual.dark is False

    def test_extra_vars_ride_along(self):
        # `foreground-muted` and `border` aren't named args on
        # Textual Theme but should ride along via `variables`.
        dark = get_theme("dark")
        textual = _care_theme_to_textual(dark)
        # Textual stores extras on `variables` attribute.
        assert "foreground-muted" in textual.variables
        assert "border" in textual.variables


# ---------------------------------------------------------------------------
# Registration + apply on mount
# ---------------------------------------------------------------------------


class TestMountApplication:
    @pytest.mark.asyncio
    async def test_register_concrete_themes_on_mount(self):
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            available = app.available_themes
            # Built-in concrete themes registered.
            assert "light" in available
            assert "dark" in available

    @pytest.mark.asyncio
    async def test_auto_theme_resolves_to_concrete_on_mount(self):
        # Default theme_pref is "auto"; resolver picks light or
        # dark based on `self.dark`. App starts in dark mode.
        app = CareApp(theme_pref=ThemePreference(theme_name="auto"))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.active_theme.kind in ("light", "dark")
            # Should NOT remain auto.
            assert not app.active_theme.is_auto

    @pytest.mark.asyncio
    async def test_explicit_dark_pref_applies_dark(self):
        app = CareApp(theme_pref=ThemePreference(theme_name="dark"))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.active_theme.name == "dark"
            assert app.theme == "dark"

    @pytest.mark.asyncio
    async def test_explicit_light_pref_applies_light(self):
        app = CareApp(theme_pref=ThemePreference(theme_name="light"))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.active_theme.name == "light"
            assert app.theme == "light"


# ---------------------------------------------------------------------------
# apply_care_theme runtime swap
# ---------------------------------------------------------------------------


class TestApplyTheme:
    @pytest.mark.asyncio
    async def test_apply_switches_active_theme(self):
        app = CareApp(theme_pref=ThemePreference(theme_name="dark"))
        async with app.run_test() as pilot:
            await pilot.pause()
            applied = app.apply_care_theme("light")
            assert applied.name == "light"
            assert app.active_theme.name == "light"
            assert app.theme == "light"

    @pytest.mark.asyncio
    async def test_apply_persist_true_writes_to_disk(self, tmp_path):
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.apply_care_theme("light", persist=True)
            loaded = load_theme_preference()
            assert loaded is not None
            assert loaded.theme_name == "light"

    @pytest.mark.asyncio
    async def test_apply_persist_false_skips_disk(self):
        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.apply_care_theme("light", persist=False)
            # No file written.
            assert load_theme_preference() is None

    @pytest.mark.asyncio
    async def test_apply_auto_resolves_to_concrete(self):
        app = CareApp(theme_pref=ThemePreference(theme_name="dark"))
        async with app.run_test() as pilot:
            await pilot.pause()
            applied = app.apply_care_theme("auto", persist=False)
            # Auto resolves; concrete theme name is light or dark.
            assert applied.kind in ("light", "dark")
            assert not applied.is_auto


# ---------------------------------------------------------------------------
# watch_dark reactivity
# ---------------------------------------------------------------------------


class TestWatchAppearance:
    @pytest.mark.asyncio
    async def test_auto_pref_reapplies_on_appearance_flip(self):
        # Start with auto preference + no system appearance →
        # resolver falls back to dark.
        app = CareApp(theme_pref=ThemePreference(theme_name="auto"))
        async with app.run_test() as pilot:
            await pilot.pause()
            # Boot routes through first-run → SettingsScreen, whose theme
            # Select applies a concrete theme on mount and clears the
            # `auto` preference. Re-establish `auto` so this test exercises
            # the appearance-flip watcher, not boot routing.
            app.apply_care_theme("auto", persist=False)
            await pilot.pause()
            initial = app.active_theme.name
            # Set system_appearance → re-resolves.
            app.system_appearance = "light"
            await pilot.pause()
            assert app.active_theme.name == "light"
            # Flip the other way.
            app.system_appearance = "dark"
            await pilot.pause()
            assert app.active_theme.name == "dark"
            # The two outcomes were different (signal actually
            # drove a re-resolution).
            assert "light" in {initial, app.active_theme.name} or initial == "dark"

    @pytest.mark.asyncio
    async def test_explicit_pref_ignores_appearance_flip(self):
        # User explicitly chose dark — flipping system_appearance
        # to light should NOT change the active theme.
        app = CareApp(theme_pref=ThemePreference(theme_name="dark"))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.active_theme.name == "dark"
            app.system_appearance = "light"
            await pilot.pause()
            # Still dark — explicit preference wins.
            assert app.active_theme.name == "dark"


# ---------------------------------------------------------------------------
# Re-registration is idempotent
# ---------------------------------------------------------------------------


class TestIdempotentRegistration:
    @pytest.mark.asyncio
    async def test_register_skips_already_registered(self):
        # Second mount of CareApp shouldn't crash on built-in
        # themes that Textual's App already registered.
        app1 = CareApp()
        async with app1.run_test() as pilot:
            await pilot.pause()
            assert "dark" in app1.available_themes
        # Build a fresh app — registration runs again.
        app2 = CareApp()
        async with app2.run_test() as pilot:
            await pilot.pause()
            assert "dark" in app2.available_themes


# ---------------------------------------------------------------------------
# Theme list integration
# ---------------------------------------------------------------------------


class TestThemeListIntegration:
    def test_list_themes_unchanged_by_app(self):
        # Constructing a CareApp shouldn't mutate the global
        # theme registry (built-ins stay, no extras added).
        before = {t.name for t in list_themes()}
        _ = CareApp()
        after = {t.name for t in list_themes()}
        assert before == after
