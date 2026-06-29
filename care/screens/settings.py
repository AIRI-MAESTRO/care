"""SettingsScreen — edit CARE config + validate connectivity
(TODO §1.1 P0.32).

The first-run wizard (and the in-app "Settings" entry) lands
here. Renders a form bound to :class:`CareConfig` nested
models — Memory / Platform / MAGE / Theme — and exposes:

* `Validate` button → :func:`care.first_run.run_all_probes`,
  rendering the :class:`FirstRunReport` inline.
* `Theme` picker → reads :func:`list_themes`; persists via
  :func:`save_theme_preference` on selection.
* `Save` → emits a :class:`SettingsScreen.Saved` envelope so
  the host writes the TOML.

Validate is the gating step: until all three probes return
`ok`, the `Save & continue` button stays disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from care.config import CareConfig, StagePolicy
from care.dotenv import update_env_file
from care.first_run import FirstRunReport, run_all_probes
from care.runtime.i18n import get_ui_language, set_ui_language, t
from care.runtime.theme import (
    ThemePreference,
    list_themes,
    load_theme_preference,
    save_theme_preference,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


@dataclass(frozen=True)
class SettingsSnapshot:
    """Frozen snapshot the host receives on Save.

    Only the fields the form actually edits ride on the
    snapshot; nested config sections the modal didn't expose
    stay at whatever the input :class:`CareConfig` carried.
    """

    config: CareConfig
    theme_name: str | None
    report: FirstRunReport | None


class SettingsScreen(Screen):
    """Form-bound `CareConfig` editor.

    Construct with the initial :class:`CareConfig` (typically
    `app.config`). The screen reads / writes the nested
    Memory / Platform / MAGE blocks via `model_copy(update=)`
    so the original instance stays untouched until Save
    fires."""

    DEFAULT_CSS = """
    SettingsScreen {
        layout: vertical;
    }
    SettingsScreen #settings-keys-hint {
        margin: 1 2;
    }
    SettingsScreen TabbedContent {
        height: 1fr;
    }
    SettingsScreen TabbedContent ContentSwitcher {
        height: 1fr;
    }
    SettingsScreen TabPane {
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }
    SettingsScreen .section-title {
        text-style: bold;
        color: $accent;
    }
    SettingsScreen .settings-hint {
        color: $text-muted;
        margin-bottom: 1;
        padding: 0 1;
        border-left: thick $accent;
    }
    SettingsScreen Input {
        width: 100%;
        margin-bottom: 1;
    }
    /* Input + inline clear (✕) button on one row. The Input flexes to
       fill; the button pins to the right at content width. */
    SettingsScreen .settings-input-row {
        height: auto;
        width: 100%;
        margin-bottom: 1;
    }
    SettingsScreen .settings-input-row Input {
        width: 1fr;
        margin-bottom: 0;
    }
    SettingsScreen .settings-clear-btn {
        width: auto;
        min-width: 5;
        margin-left: 1;
    }
    SettingsScreen Select {
        width: 100%;
        margin-bottom: 1;
    }
    SettingsScreen Label {
        width: 100%;
    }
    SettingsScreen #settings-report {
        color: $text-muted;
        padding: 0 2;
        height: auto;
    }
    SettingsScreen #settings-actions {
        height: auto;
        min-height: 3;
        padding: 0 1;
        layout: horizontal;
    }
    SettingsScreen #settings-actions Button {
        width: 1fr;
        min-width: 8;
        margin-left: 1;
    }
    SettingsScreen #settings-actions Button:first-of-type {
        margin-left: 0;
    }
    """

    TAB_IDS: tuple[str, ...] = (
        "settings-tab-mage",
        "settings-tab-memory",
        "settings-tab-platform",
        "settings-tab-search",
        "settings-tab-theme",
    )

    SEARCH_PROVIDERS: tuple[str, ...] = (
        "tavily", "serper", "exa", "duckduckgo", "serpapi", "brave",
    )

    BINDINGS = [
        Binding("ctrl+s", "save_config", "Save", show=True),
        Binding("ctrl+v", "validate", "Validate", show=True),
        Binding("escape", "cancel", "Back", show=True),
        Binding("ctrl+right", "next_tab", "Next tab", show=False),
        Binding("ctrl+left", "prev_tab", "Prev tab", show=False),
    ]

    class Saved(Message):
        """Posted on a successful Save."""

        def __init__(self, snapshot: SettingsSnapshot) -> None:
            super().__init__()
            self.snapshot = snapshot

    class Cancelled(Message):
        """Posted on Escape / Cancel — the host pops the
        screen."""

    def __init__(self, config: CareConfig) -> None:
        super().__init__()
        self.config: CareConfig = config
        self.report: FirstRunReport | None = None
        self.last_save_snapshot: SettingsSnapshot | None = None
        # Block Save until validate confirms — the user can
        # still override via Ctrl+S without validating, but
        # the gating shows up on the button label.
        self.validated_ok: bool = False
        self.validating: bool = False
        # Tab to restore after a live UI-language re-render (recompose
        # rebuilds the TabbedContent, resetting it to the first tab).
        self._pending_active_tab: str | None = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        yield Static(
            t("settings.keysHint"),
            id="settings-keys-hint",
            classes="settings-hint",
        )
        with TabbedContent(id="settings-tabs"):
            with TabPane(t("settings.tab.mage"), id="settings-tab-mage"):
                yield Static(
                    t("settings.hint.mage"),
                    classes="settings-hint",
                )
                yield Label(t("settings.field.baseUrl"))
                yield self._clearable_input(
                    value=self.config.mage.base_url or "",
                    placeholder="https://openrouter.ai/api/v1",
                    id="settings-mage-base-url",
                )
                yield Label(t("settings.field.model"))
                yield self._clearable_input(
                    value=self.config.mage.model or "",
                    placeholder="anthropic/claude-3.5-sonnet",
                    id="settings-mage-model",
                )
                yield Label(t("settings.field.apiKey"))
                yield self._clearable_input(
                    value=self.config.mage.api_key or "",
                    password=True,
                    id="settings-mage-key",
                )
                # Modes redesign — Interactive RUN policy: confirm before
                # running a generated chain (`ask`) or run immediately
                # (`auto`). Persists to CARE_CHAT__MODE__INTERACTIVE__RUN.
                yield Label(t("settings.field.interactiveRun"))
                yield Select(
                    [
                        (t("settings.field.interactiveRunAsk"), "ask"),
                        (t("settings.field.interactiveRunAuto"), "auto"),
                    ],
                    value=self._initial_interactive_run(),
                    id="settings-interactive-run",
                    allow_blank=False,
                )

            with TabPane(t("settings.tab.memory"), id="settings-tab-memory"):
                yield Static(
                    t("settings.hint.memory"),
                    classes="settings-hint",
                )
                yield Label(t("settings.field.baseUrl"))
                yield self._clearable_input(
                    value=self.config.memory.base_url,
                    id="settings-memory-url",
                )

            with TabPane(t("settings.tab.platform"), id="settings-tab-platform"):
                yield Static(
                    t("settings.hint.platform"),
                    classes="settings-hint",
                )
                yield Label(t("settings.field.baseUrl"))
                yield self._clearable_input(
                    value=self.config.platform.base_url,
                    id="settings-platform-url",
                )
                yield Label(t("settings.field.mutationBaseUrl"))
                yield self._clearable_input(
                    value=self.config.platform.mutation_base_url or "",
                    placeholder="https://openrouter.ai/api/v1",
                    id="settings-platform-mutation-url",
                )
                yield Label(t("settings.field.mutationModel"))
                yield self._clearable_input(
                    value=self.config.platform.mutation_model or "",
                    placeholder="tngtech/deepseek-r1t-chimera:free",
                    id="settings-platform-mutation-model",
                )
                yield Label(t("settings.field.mutationApiKey"))
                yield self._clearable_input(
                    value=self.config.platform.mutation_api_key or "",
                    password=True,
                    id="settings-platform-mutation-key",
                )
                yield Label(t("settings.field.validationBaseUrl"))
                yield self._clearable_input(
                    value=self.config.platform.validation_base_url or "",
                    placeholder="https://openrouter.ai/api/v1",
                    id="settings-platform-validation-url",
                )
                yield Label(t("settings.field.validationModel"))
                yield self._clearable_input(
                    value=self.config.platform.validation_model or "",
                    placeholder="tngtech/deepseek-r1t-chimera:free",
                    id="settings-platform-validation-model",
                )
                yield Label(t("settings.field.validationApiKey"))
                yield self._clearable_input(
                    value=self.config.platform.validation_api_key or "",
                    password=True,
                    id="settings-platform-validation-key",
                )

            with TabPane(t("settings.tab.search"), id="settings-tab-search"):
                yield Static(
                    t("settings.hint.search"),
                    classes="settings-hint",
                )
                yield Label(t("settings.field.searchEngine"))
                yield Select(
                    [(name.capitalize(), name) for name in self.SEARCH_PROVIDERS],
                    value=self._initial_search_provider(),
                    id="settings-search-provider",
                    allow_blank=False,
                )
                yield Label(t("settings.field.searchKey"))
                yield self._clearable_input(
                    value=self.config.tools.web_search_api_key
                    or self.config.mage.web_search_api_key
                    or "",
                    password=True,
                    placeholder="tvly-… / serpapi / brave key",
                    id="settings-search-key",
                )

            with TabPane(t("settings.tab.theme"), id="settings-tab-theme"):
                yield Static(
                    t("settings.hint.theme"),
                    classes="settings-hint",
                )
                yield Label(t("settings.field.uiLanguage"))
                yield Select(
                    [("Русский", "ru"), ("English", "en")],
                    value=get_ui_language(),
                    id="settings-ui-language",
                    allow_blank=False,
                )
                yield Label(t("settings.field.theme"))
                theme_options = self._theme_options()
                yield Select(
                    theme_options,
                    value=self._default_theme_value(theme_options),
                    id="settings-theme",
                    allow_blank=False,
                )

        yield Static("", id="settings-report")
        with Horizontal(id="settings-actions"):
            yield Button(t("settings.action.back"), id="settings-btn-back")
            yield Button(t("settings.action.prev"), id="settings-btn-prev")
            yield Button(t("settings.action.next"), id="settings-btn-next")
            yield Button(t("settings.action.validate"), id="settings-btn-validate")
            yield Button(
                t("settings.action.save"),
                id="settings-btn-save",
                variant="primary",
            )
        yield CareFooter()

    def _clearable_input(self, **input_kwargs: Any) -> Horizontal:
        """An :class:`Input` paired with an inline ``✕`` clear button to
        its right. The Input keeps its own id (so field readers + tests
        are unaffected); the button gets ``<input-id>-clear`` and is
        handled centrally in :meth:`on_button_pressed`. Handy for the
        long credential / URL fields where selecting-all to delete is
        fiddly."""
        input_id = str(input_kwargs.get("id", ""))
        return Horizontal(
            Input(**input_kwargs),
            Button(
                "✕",
                id=f"{input_id}-clear",
                classes="settings-clear-btn",
                tooltip=t("settings.action.clear"),
            ),
            classes="settings-input-row",
        )

    def on_mount(self) -> None:
        self._refresh_chrome()

    def _refresh_chrome(self) -> None:
        """(Re)apply the header breadcrumb + footer bindings for this
        screen. Split out of :meth:`on_mount` so it can run again after a
        live UI-language re-render — :meth:`recompose` rebuilds the
        children but does NOT re-fire ``on_mount``."""
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="SettingsScreen",
                breadcrumb=(t("header.breadcrumb.settings"),),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="SettingsScreen",
                scope="screen",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------

    def _theme_options(self) -> list[tuple[str, str]]:
        """List every theme the host app has registered. Prefers
        Textual's full registry (``app.available_themes``) so the
        built-in palette (catppuccin, dracula, nord, …) is offered
        alongside CARE's own themes — matching what the `/theme`
        command lists. Falls back to ``list_themes()`` (CARE-only)
        when no app is wired (bare-host tests).
        """
        names: list[str] = []
        seen: set[str] = set()
        registry = getattr(self.app, "available_themes", None) or {}
        try:
            for name in sorted(registry):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        except Exception:
            pass
        if not names:
            try:
                for theme in list_themes():
                    name = getattr(theme, "name", str(theme))
                    if name not in seen:
                        seen.add(name)
                        names.append(name)
            except Exception:
                pass
        if not names:
            names.append("default")
        return [(name, name) for name in names]

    def _default_theme_value(self, options: list[tuple[str, str]]) -> str:
        """Pre-select the user's current preference. Falls back
        to the app's live `theme_pref`, then the persisted
        preference on disk, then the first option."""
        valid = {value for _, value in options}
        candidates: list[str] = []
        current = getattr(self.app, "theme", None)
        if isinstance(current, str):
            candidates.append(current)
        pref = getattr(self.app, "theme_pref", None)
        if pref is not None:
            candidates.append(getattr(pref, "theme_name", ""))
        try:
            persisted = load_theme_preference()
        except Exception:
            persisted = None
        if persisted is not None:
            candidates.append(persisted.theme_name)
        for name in candidates:
            if name and name in valid:
                return name
        return options[0][1] if options else "default"

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def _initial_search_provider(self) -> str:
        provider = (
            getattr(self.config.tools, "web_search_provider", None)
            or getattr(self.config.mage, "web_search_provider", None)
            or "tavily"
        )
        return provider if provider in self.SEARCH_PROVIDERS else "tavily"

    def _read_search_provider(self) -> str:
        try:
            select = self.query_one("#settings-search-provider", Select)
        except Exception:
            return self._initial_search_provider()
        value = select.value
        if value in (None, Select.BLANK):
            return self._initial_search_provider()
        return str(value)

    # ------------------------------------------------------------------
    # Field readers
    # ------------------------------------------------------------------

    def _read_input(self, selector: str) -> str:
        try:
            return (self.query_one(selector, Input).value or "").strip()
        except Exception:
            return ""

    def _read_theme(self) -> str:
        try:
            select = self.query_one("#settings-theme", Select)
        except Exception:
            return ""
        value = select.value
        if value in (None, Select.BLANK):
            return ""
        return str(value)

    def current_config(self) -> CareConfig:
        """Snapshot the form into a fresh :class:`CareConfig`
        via nested `model_copy(update=)` so the source instance
        stays untouched.

        Memory + Platform `api_key` values are NOT edited via
        this form — they're sourced from env vars
        (`CARE_MEMORY__API_KEY` / `CARE_PLATFORM__API_KEY`) or
        the on-disk `care.toml`. The snapshot preserves the
        current values so a Save round-trip doesn't accidentally
        wipe them.
        """
        memory = self.config.memory.model_copy(update={
            "base_url": self._read_input("#settings-memory-url") or self.config.memory.base_url,
        })
        platform = self.config.platform.model_copy(update={
            "base_url": self._read_input("#settings-platform-url") or self.config.platform.base_url,
            "mutation_base_url": self._read_input("#settings-platform-mutation-url") or self.config.platform.mutation_base_url,
            "mutation_model": self._read_input("#settings-platform-mutation-model") or self.config.platform.mutation_model,
            "mutation_api_key": self._read_input("#settings-platform-mutation-key") or self.config.platform.mutation_api_key,
            "validation_base_url": self._read_input("#settings-platform-validation-url") or self.config.platform.validation_base_url,
            "validation_model": self._read_input("#settings-platform-validation-model") or self.config.platform.validation_model,
            "validation_api_key": self._read_input("#settings-platform-validation-key") or self.config.platform.validation_api_key,
        })
        search_provider = self._read_search_provider()
        search_key = self._read_input("#settings-search-key") or None
        mage = self.config.mage.model_copy(update={
            "base_url": self._read_input("#settings-mage-base-url") or None,
            "model": self._read_input("#settings-mage-model") or None,
            "api_key": self._read_input("#settings-mage-key") or None,
            "web_search_provider": search_provider,
            "web_search_api_key": search_key,
        })
        tools = self.config.tools.model_copy(update={
            "web_search_provider": search_provider,
            "web_search_api_key": search_key,
        })
        defaults = self.config.defaults.model_copy(update={
            "ui_language": self._read_ui_language(),
        })
        # Modes redesign — fold the Interactive RUN policy into the nested
        # chat.mode.interactive surface.
        interactive_stage = self.config.chat.mode.interactive.model_copy(
            update={"run": self._read_interactive_run()},
        )
        chat_mode = self.config.chat.mode.model_copy(
            update={"interactive": interactive_stage},
        )
        chat = self.config.chat.model_copy(update={"mode": chat_mode})
        return self.config.model_copy(update={
            "memory": memory,
            "platform": platform,
            "mage": mage,
            "tools": tools,
            "defaults": defaults,
            "chat": chat,
        })

    def _initial_interactive_run(self) -> str:
        """Current Interactive RUN policy for the Select (preset default
        ``ask`` when unset)."""
        return str(self.config.chat.mode.interactive.run or "ask")

    def _read_interactive_run(self) -> StagePolicy:
        """Read the Interactive RUN Select as a `StagePolicy`, falling back
        to the current config value when the widget isn't mounted.

        Returns the enum (not a bare str) so `model_copy(update=)` — which
        skips validation — stores a proper `StagePolicy` and Pydantic
        serialises it cleanly.
        """
        try:
            value = self.query_one("#settings-interactive-run", Select).value
        except Exception:
            value = self._initial_interactive_run()
        return StagePolicy(value) if value in ("ask", "auto") else StagePolicy.ASK

    def _read_ui_language(self) -> str:
        """Read the interface-language Select, falling back to the current
        config value when the widget isn't mounted."""
        try:
            value = self.query_one("#settings-ui-language", Select).value
        except Exception:
            return self.config.defaults.ui_language
        return value if value in ("ru", "en") else self.config.defaults.ui_language

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.endswith("-clear"):
            self._clear_field(bid[: -len("-clear")])
        elif bid == "settings-btn-back":
            self.action_cancel()
        elif bid == "settings-btn-validate":
            self.action_validate()
        elif bid == "settings-btn-save":
            self.action_save_config()
        elif bid == "settings-btn-prev":
            self.action_prev_tab()
        elif bid == "settings-btn-next":
            self.action_next_tab()

    def _clear_field(self, input_id: str) -> None:
        """Empty the Input paired with an inline ``✕`` button, then
        refocus it so the user can immediately retype."""
        try:
            field = self.query_one(f"#{input_id}", Input)
        except Exception:
            return
        field.value = ""
        try:
            field.focus()
        except Exception:
            pass

    def action_next_tab(self) -> None:
        self._switch_tab(1)

    def action_prev_tab(self) -> None:
        self._switch_tab(-1)

    def _switch_tab(self, delta: int) -> None:
        try:
            tabs = self.query_one("#settings-tabs", TabbedContent)
        except Exception:
            return
        current = tabs.active or self.TAB_IDS[0]
        if current not in self.TAB_IDS:
            return
        idx = self.TAB_IDS.index(current)
        tabs.active = self.TAB_IDS[(idx + delta) % len(self.TAB_IDS)]

    def action_validate(self) -> None:
        if self.validating:
            return
        self.validating = True
        self.run_worker(
            self._validate_worker(),
            name="settings_validate",
            group="settings",
            exclusive=True,
            exit_on_error=False,
        )

    async def _validate_worker(self) -> None:
        config = self.current_config()
        try:
            report = await run_all_probes(config)
        except Exception as exc:  # noqa: BLE001
            self.report = None
            self._render_report_text(f"⚠ probe failed: {exc}")
            self.validating = False
            return
        self.report = report
        self.validated_ok = report.all_ok
        self._render_report_text(report.format_text())
        self.validating = False

    @staticmethod
    def _env_updates(config: CareConfig) -> dict[str, str]:
        """Map the form-edited config onto the ``CARE_*`` env keys
        that the ``.env`` file carries.

        These env vars outrank ``config.toml`` in
        :meth:`CareConfig.load`, so persisting them here (and into
        ``os.environ`` via :func:`update_env_file`) is what makes a
        Save actually take effect — otherwise the startup ``.env``
        values keep masking the edit.
        """
        mage = config.mage
        tools = config.tools
        return {
            "CARE_MEMORY__BASE_URL": config.memory.base_url or "",
            "CARE_PLATFORM__BASE_URL": config.platform.base_url or "",
            "CARE_PLATFORM__MUTATION_BASE_URL": config.platform.mutation_base_url or "",
            "CARE_PLATFORM__MUTATION_MODEL": config.platform.mutation_model or "",
            "CARE_PLATFORM__MUTATION_API_KEY": config.platform.mutation_api_key or "",
            "CARE_PLATFORM__VALIDATION_BASE_URL": config.platform.validation_base_url or "",
            "CARE_PLATFORM__VALIDATION_MODEL": config.platform.validation_model or "",
            "CARE_PLATFORM__VALIDATION_API_KEY": config.platform.validation_api_key or "",
            "CARE_MAGE__BASE_URL": mage.base_url or "",
            "CARE_MAGE__MODEL": mage.model or "",
            "CARE_MAGE__API_KEY": mage.api_key or "",
            "CARE_MAGE__WEB_SEARCH_PROVIDER": (
                getattr(mage, "web_search_provider", "") or ""
            ),
            "CARE_MAGE__WEB_SEARCH_API_KEY": mage.web_search_api_key or "",
            "CARE_TOOLS__WEB_SEARCH_PROVIDER": tools.web_search_provider or "",
            "CARE_TOOLS__WEB_SEARCH_API_KEY": tools.web_search_api_key or "",
            "CARE_DEFAULTS__UI_LANGUAGE": config.defaults.ui_language,
            "CARE_CHAT__MODE__INTERACTIVE__RUN": str(
                config.chat.mode.interactive.run or "ask"
            ),
        }

    def _persist_env(self, config: CareConfig) -> Path | None:
        """Write the form values back to ``.env`` (+ ``os.environ``).
        Returns the path written, or ``None`` on OSError so the
        TOML-save success line still shows."""
        try:
            return update_env_file(self._env_updates(config))
        except OSError:
            return None

    def action_save_config(self) -> None:
        config = self.current_config()
        # The language already flips live on the Select change, but
        # re-assert here so a Save still lands the right language even
        # when it was set programmatically without firing that path.
        set_ui_language(config.defaults.ui_language)
        theme_name = self._read_theme() or None
        if theme_name:
            self._apply_theme(theme_name, persist=True)
        # Persist to ~/.config/care/config.toml so the user's
        # edits survive a session. On OSError keep the screen
        # open so the user can see the error in the report line
        # and retry — the `Saved` message only fires on a clean
        # write so the host can dismiss the screen unambiguously.
        try:
            save_report = config.save_to_disk_with_report()
        except OSError as exc:
            self._render_report_text(
                f"⚠ save failed: {exc}",
            )
            return
        # Mirror the edits into ./.env + os.environ — env vars
        # outrank config.toml on reload, so this is what makes the
        # Save actually visible (and keeps the user's .env current).
        env_path = self._persist_env(config)
        # §1 P2 — name the keystore backend when secrets
        # were offloaded so the user knows where they
        # went (and can audit externally).
        if save_report.stored_slots > 0:
            line = (
                f"✓ saved to {save_report.path}  ·  "
                f"secrets stored in "
                f"{save_report.display_backend} "
                f"({save_report.stored_slots} slot(s))"
            )
        else:
            line = f"✓ saved to {save_report.path}"
        if env_path is not None:
            line += f"  ·  .env updated ({env_path})"
        else:
            line += "  ·  ⚠ .env write failed"
        from care.runtime.platform_llm_sync import try_sync_platform_llm_registry

        plat_sync = try_sync_platform_llm_registry(config)
        if plat_sync is not None and plat_sync.wrote:
            line += "  ·  platform LLM synced"
        elif plat_sync is not None and plat_sync.path is None:
            line += "  ·  ⚠ platform checkout missing (LLM not synced)"
        self._render_report_text(line)
        snapshot = SettingsSnapshot(
            config=config, theme_name=theme_name, report=self.report,
        )
        self.last_save_snapshot = snapshot
        self.post_message(self.Saved(snapshot))

    def on_select_changed(self, event: Select.Changed) -> None:
        """Live-preview Select changes. Theme cycles the host
        palette; UI-language re-renders the screen so every label,
        tab, and button switches immediately. Both persist on Save —
        a cancel-out reverts cleanly via Cancelled handling on the
        host. Other Selects (e.g. search provider) are ignored.
        """
        try:
            select_id = event.select.id
        except Exception:
            return
        value = event.value
        if value in (None, Select.BLANK):
            return
        if select_id == "settings-ui-language":
            self._apply_ui_language_live(str(value))
        elif select_id == "settings-theme":
            self._apply_theme(str(value), persist=False)

    def _apply_ui_language_live(self, language: str) -> None:
        """Switch the active UI language and re-render the screen so
        every label/tab/button reflects it immediately — persistence
        still waits for Save.

        No-ops when the language is unchanged. That guard also absorbs
        the mount-time ``Select.Changed`` echo emitted when
        :meth:`recompose` rebuilds the language Select, which would
        otherwise loop. In-progress edits are folded back into
        ``self.config`` first so the rebuild repopulates them rather
        than reverting to the on-open values.
        """
        if language not in ("ru", "en") or language == get_ui_language():
            return
        set_ui_language(language)
        try:
            self.config = self.current_config()
        except Exception:
            pass
        try:
            self._pending_active_tab = self.query_one(
                "#settings-tabs", TabbedContent,
            ).active
        except Exception:
            self._pending_active_tab = None
        self.app.call_later(self._recompose_then_restore)

    async def _recompose_then_restore(self) -> None:
        """Rebuild the form in the new language, then restore the
        chrome and the tab the user was on — both reset by a bare
        recompose."""
        await self.recompose()
        self._refresh_chrome()
        if self._pending_active_tab:
            try:
                self.query_one(
                    "#settings-tabs", TabbedContent,
                ).active = self._pending_active_tab
            except Exception:
                pass
            finally:
                self._pending_active_tab = None

    def _apply_theme(self, name: str, *, persist: bool) -> None:
        """Switch the host app's active theme. Works for both
        CARE-registered themes (auto/light/dark) and Textual
        built-ins (catppuccin, dracula, nord, …) — anything
        present in ``app.available_themes``.

        For CARE themes, routes through ``apply_care_theme`` so
        the `theme_pref` / on-disk preference round-trips. For
        Textual built-ins, sets ``app.theme`` directly (matching
        the `/theme` command) and best-effort persists the name
        into the preference store so the next launch can restore
        it via the live `available_themes` registry.
        """
        # CARE-registered theme: route through the high-level
        # API so auto-resolution + preference store both update.
        care_names = set()
        try:
            care_names = {t.name for t in list_themes()}
        except Exception:
            pass
        if name in care_names:
            applier = getattr(self.app, "apply_care_theme", None)
            if callable(applier):
                try:
                    applier(name, persist=persist)
                except Exception:
                    pass
                else:
                    if persist:
                        self._sync_chat_theme_sidecar(name)
                    return
        # Generic path: set Textual theme directly.
        try:
            self.app.theme = name
        except Exception:
            return
        if persist:
            try:
                save_theme_preference(
                    ThemePreference(theme_name=name),
                )
            except Exception:
                pass
            self._sync_chat_theme_sidecar(name)

    @staticmethod
    def _sync_chat_theme_sidecar(name: str) -> None:
        """Mirror the picked theme into ChatScreen's sidecar
        (``~/.local/state/care/theme_preference.txt``). ChatScreen
        reads this on mount and applies it — without this write,
        a stale sidecar from a previous `/theme` call would
        clobber the Save the user just made on the next launch.
        """
        try:
            from care.screens.chat import ChatScreen

            ChatScreen._persist_theme_preference(name)
        except Exception:
            pass

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled())
        try:
            self.app.pop_screen()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render_report_text(self, text: str) -> None:
        try:
            target = self.query_one("#settings-report", Static)
        except Exception:
            return
        target.update(text)


__all__ = ["SettingsScreen", "SettingsSnapshot"]


def _ensure_any(_: Any) -> None:
    """Anchor the `Any` import so future field-type expansions
    don't need a separate import."""
