"""Pilot tests for SettingsScreen (TODO §1.1 P0.32).

Exercises:
* Composition — three nested config sections + theme picker
  + report Static + action bar.
* Field edits land on `current_config()` via nested
  `model_copy(update=)`.
* `Validate` runs `run_all_probes` (monkey-patched stub) and
  renders the report.
* `Save` posts the `Saved` envelope with a frozen
  `SettingsSnapshot`.
* `Escape` / `Cancel` post `Cancelled` and pop.
* Theme `save_theme_preference` is called on Save.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Select, Static

from care.config import CareConfig
from care.first_run import FirstRunReport, ProbeResult
from care.screens.settings import SettingsScreen, SettingsSnapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env_file(monkeypatch, tmp_path):
    """Redirect the SettingsScreen `.env` write into tmp_path so a
    Save in tests never clobbers the repo's real ./.env. Also keeps
    os.environ untouched (apply_to_environ=False) for test hygiene."""
    from care import dotenv as _dotenv

    real_update = _dotenv.update_env_file
    env_target = tmp_path / ".env"

    def _redirected(updates, path=None, *, apply_to_environ=True):
        return real_update(
            updates, path=env_target, apply_to_environ=False,
        )

    monkeypatch.setattr(
        "care.screens.settings.update_env_file", _redirected,
    )
    return env_target


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _fake_report(*, all_ok: bool = True) -> FirstRunReport:
    return FirstRunReport(
        memory=ProbeResult(
            service="memory",
            status="ok" if all_ok else "failed",
            latency_ms=10.0,
        ),
        mage=ProbeResult(
            service="mage", status="ok" if all_ok else "skipped",
        ),
        platform=ProbeResult(service="platform", status="ok"),
    )


class _Host(App):
    def __init__(self, config: CareConfig | None = None) -> None:
        super().__init__()
        self._initial = config or CareConfig()
        self.saved: list[SettingsSnapshot] = []
        self.cancelled = 0

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(SettingsScreen(self._initial))

    def on_settings_screen_saved(
        self, event: SettingsScreen.Saved,
    ) -> None:
        self.saved.append(event.snapshot)

    def on_settings_screen_cancelled(
        self, event: SettingsScreen.Cancelled,
    ) -> None:
        self.cancelled += 1


def _screen(app: App) -> SettingsScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, SettingsScreen)
    return s


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_form_widgets_mount(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.query_one("#settings-memory-url", Input) is not None
            assert screen.query_one("#settings-platform-url", Input) is not None
            assert screen.query_one("#settings-mage-base-url", Input) is not None
            assert screen.query_one("#settings-theme", Select) is not None
            assert screen.query_one("#settings-report", Static) is not None

    @pytest.mark.asyncio
    async def test_initial_values_pre_filled(self):
        cfg = CareConfig()
        cfg.memory.base_url = "https://mem.example"
        app = _Host(cfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            mem_input = screen.query_one("#settings-memory-url", Input)
            assert mem_input.value == "https://mem.example"


class TestClearButtons:
    """Each text field has an inline `✕` clear button to its right."""

    @pytest.mark.asyncio
    async def test_every_input_has_a_clear_button(self):
        from textual.widgets import Button

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            for inp in screen.query(Input):
                clear = screen.query_one(f"#{inp.id}-clear", Button)
                assert clear is not None

    @pytest.mark.asyncio
    async def test_clear_button_empties_only_its_field(self):
        from textual.widgets import Button

        cfg = CareConfig()
        cfg.mage.api_key = "sk-secret"
        cfg.mage.base_url = "https://keep-me"
        app = _Host(cfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert (
                screen.query_one("#settings-mage-key", Input).value
                == "sk-secret"
            )
            # `.press()` posts Button.Pressed regardless of scroll
            # position (the tab pane may have scrolled the field out
            # of view, which would make a pixel click miss).
            screen.query_one("#settings-mage-key-clear", Button).press()
            await pilot.pause()
            # Target field emptied…
            assert screen.query_one("#settings-mage-key", Input).value == ""
            # …its neighbour untouched.
            assert (
                screen.query_one("#settings-mage-base-url", Input).value
                == "https://keep-me"
            )
            # The cleared empty lands on the saved config.
            assert screen.current_config().mage.api_key is None


class TestTabScrolling:
    """A tab whose fields exceed the viewport must scroll, not clip —
    TabPane/ContentSwitcher default to `height: auto`, so the screen
    pins them to `1fr` + `overflow-y: auto` to make that work."""

    @pytest.mark.asyncio
    async def test_overflowing_tab_is_scrollable(self):
        from textual.widgets import TabbedContent, TabPane

        app = _Host()
        # Short terminal so the Platform tab (the most fields) overflows.
        async with app.run_test(size=(90, 14)) as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one("#settings-tabs", TabbedContent).active = (
                "settings-tab-platform"
            )
            await pilot.pause()
            pane = screen.query_one("#settings-tab-platform", TabPane)
            # Pane is bounded to the viewport (not grown to fit content)…
            assert pane.styles.overflow_y == "auto"
            assert pane.region.height < pane.virtual_size.height
            # …so the overflowing fields are reachable via scrolling.
            assert pane.max_scroll_y > 0


class TestLocalizationAndLayout:
    """B3 — Generating tab first, interface-language toggle, per-section
    descriptions."""

    def test_mage_tab_is_first(self):
        assert SettingsScreen.TAB_IDS[0] == "settings-tab-mage"

    @pytest.mark.asyncio
    async def test_language_select_reflects_active_language(self):
        from care.runtime import i18n

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            sel = screen.query_one("#settings-ui-language", Select)
            # conftest pins the suite to English.
            assert sel.value == i18n.get_ui_language()

    @pytest.mark.asyncio
    async def test_each_section_has_a_description(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            # Top keys-hint + one per tab (generating/memory/platform/
            # search/theme).
            assert len(screen.query(".settings-hint")) >= 6

    @pytest.mark.asyncio
    async def test_language_change_lands_on_config(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one("#settings-ui-language", Select).value = "ru"
            await pilot.pause()
            cfg = screen.current_config()
            assert cfg.defaults.ui_language == "ru"
            # The agent's answer language is a separate field, untouched.
            assert cfg.defaults.language == screen.config.defaults.language

    @pytest.mark.asyncio
    async def test_interactive_run_change_lands_on_config(self):
        """Modes redesign — the Interactive RUN Select folds into
        `chat.mode.interactive.run`."""
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one("#settings-interactive-run", Select).value = "auto"
            await pilot.pause()
            cfg = screen.current_config()
            assert cfg.chat.mode.interactive.run == "auto"

    @pytest.mark.asyncio
    async def test_interactive_run_default_is_ask(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            sel = screen.query_one("#settings-interactive-run", Select)
            assert sel.value == "ask"  # preset default

    @pytest.mark.asyncio
    async def test_language_change_rerenders_screen_immediately(self):
        """Changing the language Select flips the live UI language and
        re-renders the screen, so labels switch without a Save."""
        from textual.widgets import Button

        from care.runtime import i18n

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            # Pinned to English by conftest — the save button proves it.
            assert str(screen.query_one("#settings-btn-save", Button).label) == "Save"

            screen.query_one("#settings-ui-language", Select).value = "ru"
            # Let the scheduled recompose flush.
            await pilot.pause()
            await pilot.pause()

            assert i18n.get_ui_language() == "ru"
            save_btn = screen.query_one("#settings-btn-save", Button)
            assert str(save_btn.label) == i18n.t("settings.action.save")
            assert str(save_btn.label) == "Сохранить"

    @pytest.mark.asyncio
    async def test_language_change_preserves_active_tab_and_edits(self):
        """The live re-render keeps the user on their current tab and
        does not revert in-progress field edits back to on-open values."""
        from textual.widgets import TabbedContent

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            # Move off the first tab and type an unsaved edit.
            screen.query_one("#settings-tabs", TabbedContent).active = (
                "settings-tab-theme"
            )
            screen.query_one("#settings-mage-model", Input).value = "edited/model"
            await pilot.pause()

            screen.query_one("#settings-ui-language", Select).value = "ru"
            await pilot.pause()
            await pilot.pause()

            assert (
                screen.query_one("#settings-tabs", TabbedContent).active
                == "settings-tab-theme"
            )
            # The edit survives the recompose.
            assert (
                screen.query_one("#settings-mage-model", Input).value
                == "edited/model"
            )


# ---------------------------------------------------------------------------
# current_config
# ---------------------------------------------------------------------------


class TestCurrentConfig:
    @pytest.mark.asyncio
    async def test_field_edits_land_on_snapshot(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one(
                "#settings-memory-url", Input,
            ).value = "https://memnew.example"
            screen.query_one(
                "#settings-platform-url", Input,
            ).value = "https://platnew.example"
            screen.query_one(
                "#settings-mage-model", Input,
            ).value = "gpt-4o"
            await pilot.pause()
            cfg = screen.current_config()
            assert cfg.memory.base_url == "https://memnew.example"
            assert cfg.platform.base_url == "https://platnew.example"
            assert cfg.mage.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_memory_platform_api_keys_preserved_on_save(self):
        # Memory + Platform API keys aren't editable on the
        # SettingsScreen (env-driven). A Save round-trip
        # preserves whatever the source config carried.
        from care.config import CareConfig, MemoryConfig, PlatformConfig

        config = CareConfig(
            memory=MemoryConfig(
                base_url="https://m.example",
                api_key="pre-existing-mem-key",
            ),
            platform=PlatformConfig(
                base_url="https://p.example",
                api_key="pre-existing-plat-key",
            ),
        )

        class _Host2(App):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(SettingsScreen(config))

        app = _Host2()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, SettingsScreen)
            cfg = screen.current_config()
            assert cfg.memory.api_key == "pre-existing-mem-key"
            assert cfg.platform.api_key == "pre-existing-plat-key"

    @pytest.mark.asyncio
    async def test_memory_and_platform_key_inputs_absent(self):
        # The two API-key Inputs are removed from the form;
        # querying for them raises NoMatches.
        from textual.css.query import NoMatches

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            with pytest.raises(NoMatches):
                screen.query_one("#settings-memory-key", Input)
            with pytest.raises(NoMatches):
                screen.query_one("#settings-platform-key", Input)

    @pytest.mark.asyncio
    async def test_keys_hint_renders(self):
        # The env-var hint replaces the removed API-key inputs
        # so first-run users know where Memory + Platform
        # credentials come from.
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            hint = screen.query_one("#settings-keys-hint", Static)
            text = str(hint.content)
            assert "CARE_MEMORY__API_KEY" in text
            assert "CARE_PLATFORM__API_KEY" in text
            assert "care.toml" in text


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


class TestValidate:
    @pytest.mark.asyncio
    async def test_validate_runs_probes(self, monkeypatch):
        called: list = []

        async def _stub_run_all_probes(config, **kw):
            called.append(config)
            return _fake_report(all_ok=True)

        monkeypatch.setattr(
            "care.screens.settings.run_all_probes",
            _stub_run_all_probes,
        )
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.action_validate()
            for _ in range(6):
                await pilot.pause()
            assert called != []
            assert screen.report is not None
            assert screen.validated_ok is True

    @pytest.mark.asyncio
    async def test_validate_failed_sets_validated_ok_false(self, monkeypatch):
        async def _stub_run_all_probes(config, **kw):
            return _fake_report(all_ok=False)

        monkeypatch.setattr(
            "care.screens.settings.run_all_probes",
            _stub_run_all_probes,
        )
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.action_validate()
            for _ in range(6):
                await pilot.pause()
            assert screen.report is not None
            assert screen.validated_ok is False


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


class TestSave:
    @pytest.mark.asyncio
    async def test_save_posts_envelope(self, monkeypatch, tmp_path):
        # Stub save_theme_preference so the test doesn't write
        # the user's real preference file. Redirect the config
        # save path so it lands in tmp_path rather than the real
        # `~/.config/care/config.toml`.
        called: list[str] = []

        def _stub_save_theme(name):
            called.append(name)

        monkeypatch.setattr(
            "care.screens.settings.save_theme_preference",
            _stub_save_theme,
        )
        monkeypatch.setattr(
            "care.config.DEFAULT_CONFIG_PATH",
            tmp_path / "care.toml",
        )
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one(
                "#settings-memory-url", Input,
            ).value = "https://memnew.example"
            await pilot.pause()
            screen.action_save_config()
            for _ in range(3):
                await pilot.pause()
            assert len(app.saved) == 1
            snap = app.saved[0]
            assert isinstance(snap, SettingsSnapshot)
            assert snap.config.memory.base_url == "https://memnew.example"
            # save_theme_preference was invoked for the
            # default theme.
            assert called != []

    @pytest.mark.asyncio
    async def test_save_writes_to_disk(self, monkeypatch, tmp_path):
        # The Save action now persists to disk so first-run
        # users' edits survive a session. Verify a TOML file
        # lands at the configured path and contains the edited
        # value.
        from care.config import CareConfig

        target = tmp_path / "care.toml"
        monkeypatch.setattr(
            "care.screens.settings.save_theme_preference",
            lambda name: None,
        )
        monkeypatch.setattr(
            "care.config.DEFAULT_CONFIG_PATH", target,
        )
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one(
                "#settings-memory-url", Input,
            ).value = "https://memnew.example"
            await pilot.pause()
            screen.action_save_config()
            for _ in range(3):
                await pilot.pause()
            # File landed.
            assert target.exists()
            reloaded = CareConfig.load(path=target, env={})
            assert reloaded.memory.base_url == "https://memnew.example"


# ---------------------------------------------------------------------------
# Search engine + API key
# ---------------------------------------------------------------------------


class TestSearchEngine:
    @pytest.mark.asyncio
    async def test_search_widgets_mount(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.query_one(
                "#settings-search-provider", Select,
            ) is not None
            assert screen.query_one(
                "#settings-search-key", Input,
            ) is not None

    @pytest.mark.asyncio
    async def test_initial_provider_prefilled(self):
        cfg = CareConfig()
        cfg.tools.web_search_provider = "brave"
        app = _Host(cfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            sel = screen.query_one("#settings-search-provider", Select)
            assert sel.value == "brave"

    @pytest.mark.asyncio
    async def test_edits_land_on_tools_and_mage(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one(
                "#settings-search-provider", Select,
            ).value = "serpapi"
            screen.query_one(
                "#settings-search-key", Input,
            ).value = "tvly-secret"
            await pilot.pause()
            cfg = screen.current_config()
            assert cfg.tools.web_search_provider == "serpapi"
            assert cfg.tools.web_search_api_key == "tvly-secret"
            assert cfg.mage.web_search_provider == "serpapi"
            assert cfg.mage.web_search_api_key == "tvly-secret"


class TestEnvWrite:
    @pytest.mark.asyncio
    async def test_save_writes_env_file(
        self, monkeypatch, tmp_path, _isolate_env_file,
    ):
        monkeypatch.setattr(
            "care.screens.settings.save_theme_preference",
            lambda name: None,
        )
        monkeypatch.setattr(
            "care.config.DEFAULT_CONFIG_PATH", tmp_path / "care.toml",
        )
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one(
                "#settings-mage-model", Input,
            ).value = "gpt-4o"
            screen.query_one(
                "#settings-search-provider", Select,
            ).value = "brave"
            screen.query_one(
                "#settings-search-key", Input,
            ).value = "brave-key"
            await pilot.pause()
            screen.action_save_config()
            for _ in range(3):
                await pilot.pause()
            assert _isolate_env_file.exists()
            text = _isolate_env_file.read_text(encoding="utf-8")
            assert "CARE_MAGE__MODEL=gpt-4o" in text
            assert "CARE_TOOLS__WEB_SEARCH_PROVIDER=brave" in text
            assert "CARE_TOOLS__WEB_SEARCH_API_KEY=brave-key" in text
            assert "CARE_MAGE__WEB_SEARCH_API_KEY=brave-key" in text

    @pytest.mark.asyncio
    async def test_existing_keys_rewritten_in_place(
        self, monkeypatch, tmp_path, _isolate_env_file,
    ):
        # Pre-seed an .env with a comment + an existing key; the
        # save should rewrite the key's value while keeping the
        # comment, not duplicate the key.
        _isolate_env_file.write_text(
            "# keep me\nCARE_MAGE__MODEL=old-model\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "care.screens.settings.save_theme_preference",
            lambda name: None,
        )
        monkeypatch.setattr(
            "care.config.DEFAULT_CONFIG_PATH", tmp_path / "care.toml",
        )
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one(
                "#settings-mage-model", Input,
            ).value = "new-model"
            await pilot.pause()
            screen.action_save_config()
            for _ in range(3):
                await pilot.pause()
            text = _isolate_env_file.read_text(encoding="utf-8")
            assert "# keep me" in text
            assert "CARE_MAGE__MODEL=new-model" in text
            assert "old-model" not in text
            assert text.count("CARE_MAGE__MODEL=") == 1


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class TestCancel:
    @pytest.mark.asyncio
    async def test_action_cancel_posts_and_pops(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            initial_depth = len(app.screen_stack)
            screen.action_cancel()
            for _ in range(3):
                await pilot.pause()
            assert app.cancelled == 1
            assert len(app.screen_stack) < initial_depth


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import SettingsScreen as S
        from care.screens import SettingsSnapshot as Snap

        assert S is SettingsScreen
        assert Snap is SettingsSnapshot
