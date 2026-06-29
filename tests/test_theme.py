"""Tests for the theming data layer (TODO §1 P2).

The toggle UI is gated on §1 P0; this suite pins the registry +
resolver + persistence contract.
"""

from __future__ import annotations

import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from care.runtime.theme import (
    DEFAULT_THEMES,
    DEFAULT_THEME_PATH,
    Theme,
    ThemeError,
    ThemePreference,
    ThemePreferenceStore,
    get_theme,
    list_themes,
    load_theme_preference,
    register_theme,
    resolve_active_theme,
    save_theme_preference,
    theme_to_tcss_vars,
    unregister_theme,
)


# ---------------------------------------------------------------------------
# Fixture — keep the registry pristine between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Snapshot the registry before each test, restore after.
    Lets test cases register / unregister custom themes
    without leaking state between tests."""
    from care.runtime.theme import _REGISTRY

    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_themes_includes_auto_light_dark(self):
        names = {t.name for t in DEFAULT_THEMES}
        assert names == {"auto", "light", "dark"}

    def test_default_themes_first_is_auto(self):
        assert DEFAULT_THEMES[0].name == "auto"

    def test_get_theme_known(self):
        assert get_theme("light") is not None
        assert get_theme("dark") is not None
        assert get_theme("auto") is not None

    def test_get_theme_unknown(self):
        assert get_theme("xyzzy") is None

    def test_theme_is_frozen(self):
        t = Theme(name="x", kind="dark")
        with pytest.raises(FrozenInstanceError):
            t.name = "y"  # type: ignore[misc]

    def test_default_path_constant(self):
        assert str(DEFAULT_THEME_PATH).endswith("/care/theme.json")

    def test_light_has_required_variables(self):
        light = get_theme("light")
        assert "background" in light.variables
        assert "foreground" in light.variables
        assert "primary" in light.variables

    def test_dark_has_required_variables(self):
        dark = get_theme("dark")
        assert "background" in dark.variables
        assert "foreground" in dark.variables
        assert "primary" in dark.variables

    def test_auto_is_kind_auto(self):
        assert get_theme("auto").is_auto


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_custom_theme(self):
        custom = Theme(name="ocean", kind="dark", variables={"primary": "#3399ff"})
        register_theme(custom)
        assert get_theme("ocean") is custom

    def test_register_empty_name_raises(self):
        with pytest.raises(ThemeError, match="empty"):
            register_theme(Theme(name="   ", kind="dark"))

    def test_register_duplicate_custom_raises(self):
        register_theme(Theme(name="ocean", kind="dark"))
        with pytest.raises(ThemeError, match="already registered"):
            register_theme(Theme(name="ocean", kind="dark"))

    def test_register_built_in_raises(self):
        with pytest.raises(ThemeError, match="built-in"):
            register_theme(Theme(name="dark", kind="dark"))

    def test_unregister_custom_succeeds(self):
        register_theme(Theme(name="ocean", kind="dark"))
        assert unregister_theme("ocean") is True
        assert get_theme("ocean") is None
        # Idempotent — second call returns False.
        assert unregister_theme("ocean") is False

    def test_unregister_built_in_raises(self):
        with pytest.raises(ThemeError, match="built-in"):
            unregister_theme("dark")

    def test_list_themes_built_in_order_first(self):
        register_theme(Theme(name="ocean", kind="dark"))
        register_theme(Theme(name="amber", kind="light"))
        names = [t.name for t in list_themes()]
        # Built-ins always first in `auto/light/dark` order.
        assert names[:3] == ["auto", "light", "dark"]
        # Custom themes sorted alphabetically after.
        assert names[3:] == ["amber", "ocean"]


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class TestResolver:
    def test_resolve_light_direct(self):
        resolved = resolve_active_theme("light")
        assert resolved.name == "light"
        assert resolved.kind == "light"

    def test_resolve_dark_direct(self):
        resolved = resolve_active_theme("dark")
        assert resolved.name == "dark"
        assert resolved.kind == "dark"

    def test_resolve_auto_light_appearance(self):
        resolved = resolve_active_theme(
            "auto", system_appearance="light",
        )
        assert resolved.name == "light"

    def test_resolve_auto_dark_appearance(self):
        resolved = resolve_active_theme(
            "auto", system_appearance="dark",
        )
        assert resolved.name == "dark"

    def test_resolve_auto_no_appearance_uses_fallback(self):
        # Default fallback is dark.
        resolved = resolve_active_theme("auto", system_appearance=None)
        assert resolved.name == "dark"

    def test_resolve_auto_custom_fallback(self):
        resolved = resolve_active_theme(
            "auto", system_appearance=None, fallback="light",
        )
        assert resolved.name == "light"

    def test_resolve_unknown_name_falls_back_to_auto_then_appearance(self):
        # Unknown name → falls through `auto` → appearance.
        resolved = resolve_active_theme(
            "not-real", system_appearance="light",
        )
        assert resolved.name == "light"

    def test_resolve_auto_with_missing_partner_falls_back(self):
        # Custom auto theme that references a non-existent partner.
        register_theme(
            Theme(
                name="custom-auto",
                kind="auto",
                light_pair="ghost-light",
                dark_pair="ghost-dark",
            )
        )
        resolved = resolve_active_theme(
            "custom-auto", system_appearance="light",
        )
        # Falls back to built-in light.
        assert resolved.name == "light"

    def test_resolve_auto_returns_concrete_theme(self):
        # Resolver should NEVER return an auto-kind theme to
        # the consumer (TCSS projection would be empty).
        for appearance in ("light", "dark", None):
            resolved = resolve_active_theme("auto", system_appearance=appearance)
            assert resolved.kind in ("light", "dark")
            assert not resolved.is_auto


# ---------------------------------------------------------------------------
# TCSS projection
# ---------------------------------------------------------------------------


class TestTcssProjection:
    def test_dark_projection_includes_keys_with_dollar(self):
        vars_ = theme_to_tcss_vars(get_theme("dark"))
        assert "$background" in vars_
        assert "$primary" in vars_
        # Values are colour strings.
        assert vars_["$background"].startswith("#")

    def test_auto_projection_is_empty(self):
        # Defensive — caller should have resolved first; we
        # surface empty rather than throw so the bug is
        # detectable.
        assert theme_to_tcss_vars(get_theme("auto")) == {}

    def test_projection_preserves_variable_values(self):
        custom = Theme(
            name="ocean",
            kind="dark",
            variables={"primary": "#3399ff", "background": "#001122"},
        )
        register_theme(custom)
        vars_ = theme_to_tcss_vars(custom)
        assert vars_["$primary"] == "#3399ff"
        assert vars_["$background"] == "#001122"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestThemePreferenceStore:
    def test_preference_is_frozen(self):
        pref = ThemePreference()
        with pytest.raises(FrozenInstanceError):
            pref.theme_name = "x"  # type: ignore[misc]

    def test_default_preference_is_auto(self):
        assert ThemePreference().theme_name == "auto"

    def test_save_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "theme.json"
        save_theme_preference(
            ThemePreference(theme_name="dark"), path=path,
        )
        loaded = load_theme_preference(path)
        assert loaded is not None
        assert loaded.theme_name == "dark"

    def test_load_missing_returns_none(self, tmp_path: Path):
        assert load_theme_preference(tmp_path / "nope.json") is None

    def test_load_malformed_returns_none(self, tmp_path: Path):
        path = tmp_path / "theme.json"
        path.write_text("not json")
        assert load_theme_preference(path) is None

    def test_load_non_dict_returns_none(self, tmp_path: Path):
        path = tmp_path / "theme.json"
        path.write_text('["dark"]')
        assert load_theme_preference(path) is None

    def test_load_schema_mismatch_returns_none(self, tmp_path: Path):
        path = tmp_path / "theme.json"
        path.write_text('{"schema_version": 999, "theme_name": "dark"}')
        assert load_theme_preference(path) is None

    def test_load_missing_theme_name_returns_none(self, tmp_path: Path):
        path = tmp_path / "theme.json"
        path.write_text('{"schema_version": 1}')
        assert load_theme_preference(path) is None

    def test_load_empty_theme_name_returns_none(self, tmp_path: Path):
        path = tmp_path / "theme.json"
        path.write_text('{"schema_version": 1, "theme_name": "  "}')
        assert load_theme_preference(path) is None

    def test_atomic_write_no_leftovers(self, tmp_path: Path):
        path = tmp_path / "theme.json"
        for _ in range(3):
            save_theme_preference(ThemePreference(theme_name="dark"), path=path)
        leftovers = list(tmp_path.glob(".theme-*"))
        assert leftovers == []

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "deeper" / "theme.json"
        save_theme_preference(ThemePreference(), path=path)
        assert path.exists()

    def test_clear_idempotent(self, tmp_path: Path):
        path = tmp_path / "theme.json"
        save_theme_preference(ThemePreference(), path=path)
        store = ThemePreferenceStore(path)
        assert store.clear() is True
        assert store.clear() is False  # already gone

    def test_concurrent_save_no_corruption(self, tmp_path: Path):
        path = tmp_path / "theme.json"
        store = ThemePreferenceStore(path)

        def hammer(idx):
            store.save(ThemePreference(theme_name=f"thread-{idx}"))

        threads = [
            threading.Thread(target=hammer, args=(i,)) for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        loaded = store.load()
        assert loaded is not None
        assert loaded.theme_name.startswith("thread-")


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            DEFAULT_THEMES as defaults,
            DEFAULT_THEME_PATH as path_const,
            Theme as T,
            ThemeError as Err,
            ThemePreference as P,
            ThemePreferenceStore as Store,
            get_theme as get_t,
            list_themes as list_t,
            load_theme_preference as load,
            register_theme as register,
            resolve_active_theme as resolve,
            save_theme_preference as save,
            theme_to_tcss_vars as project,
            unregister_theme as unregister,
        )

        assert T is Theme
        assert Err is ThemeError
        assert P is ThemePreference
        assert Store is ThemePreferenceStore
        assert get_t is get_theme
        assert list_t is list_themes
        assert load is load_theme_preference
        assert register is register_theme
        assert resolve is resolve_active_theme
        assert save is save_theme_preference
        assert project is theme_to_tcss_vars
        assert unregister is unregister_theme
        assert defaults is DEFAULT_THEMES
        assert path_const == DEFAULT_THEME_PATH
