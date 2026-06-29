"""CARE Textual application entry (TODO §1.1 P0.1).

`CareApp` is the screen-stack-driven shell every workflow screen
(Library / Query / Generation / Inspection / Edit / Execution /
Evolution) mounts onto. The app owns:

* A `mode` reactive flagging ``"first_run"`` vs ``"returning"``
  — set in :meth:`__init__` based on whether
  ``~/.config/care/config.toml`` exists. The boot-screen
  selector reads this to pick :class:`WelcomeScreen` (returning)
  vs :class:`SettingsScreen` (first-run; not shipped yet —
  falls back to the existing demo for now).
* Lazy facade slots — `memory`, `platform`, `config`,
  `task_registry`, `token_counter`, `theme_pref`. Constructed
  here so every screen reads `app.memory` (etc.) instead of
  threading the facades through screen kwargs. Slots that
  require credentials (``memory`` / ``platform``) stay
  ``None`` until the first-run wizard populates them.
* Screen-stack helpers inherited from Textual's `App`
  (`push_screen` / `pop_screen` / `switch_screen`).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from textual.app import App
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive

from care.config import DEFAULT_CONFIG_PATH, CareConfig
from care.runtime import (
    SessionTokenCounter,
    SystemAppearance,
    TaskRegistry,
    Theme as CareTheme,
    ThemePreference,
    default_global_bindings,
    list_themes,
    load_theme_preference,
    resolve_active_theme,
    save_theme_preference,
)
from care.runtime.i18n import t
from care.runtime.user_paths import UserPathReport, ensure_user_dirs

_log = logging.getLogger("care.app")


Mode = Literal["first_run", "returning"]
"""App boot mode. ``"first_run"`` triggers the SettingsScreen
flow; ``"returning"`` drops straight into LibraryScreen via the
WelcomeScreen splash."""


def _care_theme_to_textual(theme: CareTheme) -> Any:
    """Project a :class:`care.runtime.Theme` into Textual's
    :class:`textual.theme.Theme` shape.

    Textual's Theme constructor takes named colour args
    (primary, secondary, background, ...); CARE stores the
    same palette in a flat ``variables`` dict keyed by the
    same names. CARE-specific keys without a Textual
    counterpart (``foreground-muted``, ``border``) ride
    along via the ``variables`` kwarg.
    """
    from textual.theme import Theme as TextualTheme

    vars_ = theme.variables
    return TextualTheme(
        name=theme.name,
        primary=vars_.get("primary", "#ffffff"),
        secondary=vars_.get("secondary"),
        warning=vars_.get("warning"),
        error=vars_.get("error"),
        success=vars_.get("success"),
        accent=vars_.get("accent"),
        foreground=vars_.get("foreground"),
        background=vars_.get("background"),
        surface=vars_.get("surface"),
        panel=vars_.get("panel"),
        dark=theme.kind == "dark",
        variables={
            k: v
            for k, v in vars_.items()
            if k not in {
                "primary", "secondary", "warning", "error",
                "success", "accent", "foreground",
                "background", "surface", "panel",
            }
        },
    )


def _build_textual_bindings() -> list[Binding]:
    """Project :func:`default_global_bindings` into Textual
    `Binding` declarations.

    The canonical action id ``"open_command_palette"`` becomes
    the Textual action name ``"global_open_command_palette"`` so
    every action lives under the ``action_global_*`` namespace
    in the app. The footer + tests both rely on the
    ``global_<action_id>`` convention — keep it in lockstep with
    `default_global_bindings()`.
    """
    return [
        Binding(
            b.textual_key,
            f"global_{b.action_id}",
            b.label,
            show=False,  # Footer widget owns the visible hints.
        )
        for b in default_global_bindings()
    ]


# Palette-action → app-method dispatch table. Built at module
# load so the lookup stays O(1) and ``_dispatch_palette_action``
# doesn't have to grow an if/elif chain when new commands land.
# Each handler takes the `CareApp` instance and fires the
# matching `action_*` method (or whatever the palette command
# should do).
_PALETTE_ACTION_DISPATCH: dict[str, Any] = {
    "open_catalog": lambda app: app.action_open_catalog(),
    "open_marketplace": lambda app: app.action_open_marketplace(),
    "show_help": lambda app: app.action_open_help(),
    "open_chat": lambda app: app.action_palette_open_chat(),
    "open_artifacts": lambda app: app.action_palette_open_artifacts(),
    "show_library": lambda app: app.action_palette_open_library(),
    "open_evolution": lambda app: app.action_palette_open_evolution(),
    "open_settings": lambda app: app.action_palette_open_settings(),
    "quit": lambda app: app.exit(),
}


class CareApp(App):
    """Collaborative Agent Reasoning Ecosystem — TUI shell."""

    TITLE = "MAESTRO"
    SUB_TITLE = "Collaborative Agent Reasoning Ecosystem"

    # Textual ships a built-in Ctrl+P command palette that
    # intercepts the chord before our binding fires. Disable
    # it so CARE's own palette (the shipped
    # `CommandPaletteModal` / data layer in §1 P3) owns the
    # gesture.
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = _build_textual_bindings() + [
        Binding("ctrl+b", "open_task_list", "Tasks", show=False),
        Binding("question_mark", "open_help", "Help", show=False),
        Binding("ctrl+k", "open_catalog", "Catalog", show=False),
        # Override Textual's default ctrl+c → action_help_quit (which only
        # shows a "press ctrl+q to quit" toast) so the classic interrupt
        # actually exits CARE. priority=True wins over focused-widget bindings.
        Binding("ctrl+c", "global_quit", "Quit", show=False, priority=True),
    ]

    mode: reactive[Mode] = reactive("returning", init=False)
    # `system_appearance` is CARE's stand-in for the host's
    # `prefers-color-scheme` signal. Textual doesn't expose
    # one natively in our pinned version — apps + tests flip
    # this reactive to emulate the appearance change, and
    # `watch_system_appearance` re-resolves `auto`-kind
    # themes accordingly.
    system_appearance: reactive[SystemAppearance | None] = reactive(
        None, init=False
    )

    # ------------------------------------------------------------------
    # Global action message types
    # ------------------------------------------------------------------

    class SaveRequested(Message):
        """Posted when the user fires Ctrl+S. The active screen
        listens via ``on_care_app_save_requested`` and acts on
        whatever artifact the current view represents."""

    class RerunRequested(Message):
        """Posted when the user fires Ctrl+R. Active screen
        listens for the same reason as :class:`SaveRequested`."""

    class CommandPaletteRequested(Message):
        """Posted when the user fires Ctrl+P. The future
        `CommandPaletteModal` (P0.25) will be pushed by either
        a global handler or a screen-level listener that
        consumes this message."""

    class BackRequested(Message):
        """Posted when Esc fires AND the screen stack has only
        one screen (popping would empty the app). Top-level
        screens can listen + do their own back behaviour."""

    def __init__(
        self,
        *,
        config: CareConfig | None = None,
        memory: Any = None,
        platform: Any = None,
        task_registry: TaskRegistry | None = None,
        token_counter: SessionTokenCounter | None = None,
        theme_pref: ThemePreference | None = None,
        mode: Mode | None = None,
    ) -> None:
        """Construct the app with optional pre-built facades.

        Every kwarg is optional — production callers pass
        ``CareApp()`` and the constructor materialises sensible
        defaults. Tests pass stubs for the relevant slots.

        Args:
            config: Pre-built :class:`CareConfig`. ``None``
                loads via :meth:`CareConfig.load` with the
                standard precedence stack.
            memory: A :class:`CareMemory`-like facade.
                ``None`` until the first-run wizard / config
                resolves credentials — screens that need
                Memory call ``app.ensure_memory()``.
            platform: A :class:`CarePlatform`-like facade.
                Same lazy semantics as ``memory``.
            task_registry: In-session task tracker. ``None``
                builds a fresh :class:`TaskRegistry`.
            token_counter: Session-wide token accumulator.
                ``None`` builds a fresh
                :class:`SessionTokenCounter`.
            theme_pref: Persisted theme preference.
                ``None`` reads :func:`load_theme_preference`
                or falls back to the default
                :class:`ThemePreference` (which is
                ``theme_name="auto"``).
            mode: Override the auto-detected boot mode.
                Useful for tests that want to assert a
                specific branch.
        """
        super().__init__()
        # First-run write paths — make sure ~/.config/care,
        # ~/.cache/care, ~/.local/state/care all exist before
        # any downstream code tries to write into them.
        # Non-fatal: failures land on `user_path_report` and
        # are surfaced to the user via SettingsScreen / log
        # rather than refusing boot.
        self.user_path_report: UserPathReport = ensure_user_dirs()
        # Lazy config load — never crash on a missing file.
        # `CareConfig.load()` raises only on malformed TOML or
        # field-level validation errors, both of which we let
        # propagate (the user needs to see them).
        self._config_path_exists = DEFAULT_CONFIG_PATH.exists()
        self.config: CareConfig = config if config is not None else CareConfig.load()
        # Reduced-motion (TODO §Animations A-0). Textual already seeded
        # `self.animation_level` from the TEXTUAL_ANIMATIONS env in
        # `super().__init__()`; the CARE config can additionally force it off
        # so every `styles.animate()` + CSS `transition` resolves instantly.
        # We only ever tighten (never loosen) the level — an env-disabled
        # session stays disabled even if the config doesn't ask for it.
        try:
            if getattr(self.config.defaults, "reduced_motion", False):
                self.animation_level = "none"
        except Exception:
            pass
        # Set the TUI language before any screen renders so labels / system
        # messages come up in the configured language (Russian by default).
        from care.runtime.i18n import set_ui_language

        set_ui_language(getattr(self.config.defaults, "ui_language", "ru"))
        self.memory: Any = memory
        self.platform: Any = platform
        # `is None` checks (not `or` defaults) — TaskRegistry +
        # SessionTokenCounter both override `__len__`, so a
        # fresh empty instance is falsy and an `or` default
        # would discard the caller's instance.
        self.task_registry: TaskRegistry = (
            task_registry if task_registry is not None else TaskRegistry()
        )
        self.token_counter: SessionTokenCounter = (
            token_counter if token_counter is not None else SessionTokenCounter()
        )
        if theme_pref is not None:
            self.theme_pref: ThemePreference = theme_pref
        else:
            loaded = load_theme_preference()
            self.theme_pref = loaded if loaded is not None else ThemePreference()
        # Mode resolution: explicit kwarg wins; else default to
        # `first_run` when no user config file exists yet, else
        # `returning`. The `_config_path_exists` snapshot is
        # taken before the constructor's load to preserve the
        # signal across in-memory config edits.
        if mode is not None:
            self._initial_mode = mode
        elif self._config_path_exists:
            self._initial_mode = "returning"
        else:
            self._initial_mode = "first_run"
        # Action telemetry — tests + future TaskList drawer
        # read this; production callers rarely need it.
        self.global_action_log: list[str] = []

    def on_mount(self) -> None:
        """Set the reactive `mode`, apply the persisted theme,
        and push the boot screen."""
        self._ensure_facades_from_config()
        _log.info(
            "CareApp boot: mode=%s memory=%s platform=%s config_path_exists=%s",
            self._initial_mode,
            type(self.memory).__name__ if self.memory else None,
            type(self.platform).__name__ if self.platform else None,
            self._config_path_exists,
        )
        self.mode = self._initial_mode
        self._register_care_themes()
        self._apply_active_theme()

    # Default base URLs from `care.config.MemoryConfig` /
    # `care.config.PlatformConfig`. A value DIFFERENT from these
    # is treated as an opt-in signal: the user pointed CARE at
    # a specific local / remote deployment, so we should build
    # the facade even when no api_key is set (Memory's opt-in
    # auth mode accepts anonymous requests). Defaults stay
    # ``None`` so unconfigured CareApp() boots don't try to
    # hit a localhost service that isn't running.
    _DEFAULT_MEMORY_URL = "http://localhost:8000"
    _DEFAULT_PLATFORM_URL = "http://localhost:8001"

    def _ensure_facades_from_config(self) -> None:
        """Lazily build :class:`CareMemory` / :class:`CarePlatform`
        from the loaded config when the slots are still ``None``
        and the user has opted in to a specific deployment —
        either by setting an ``api_key`` or by pointing
        ``base_url`` at a non-default endpoint.

        Without this, returning users with a valid config still
        boot with `app.memory = None` / `app.platform = None` —
        screens that gate on those slots (LibraryScreen,
        EvolutionScreen, etc.) then silently no-op. Build at
        boot so the facades are ready before the first screen
        push reads them.
        """
        # Always opt in: facade construction is a thin client
        # wrapper with no I/O — actual HTTP only happens on call
        # sites, which already handle failures. The old "URL ==
        # default → skip" gate broke users actually running
        # Memory/Platform at the documented default ports, and
        # there's no cost to building a facade that's never used.
        memory_opted_in = True
        platform_opted_in = True
        if self.memory is None and memory_opted_in:
            try:
                from care.memory import CareMemory

                self.memory = CareMemory.from_config(self.config)
                _log.info(
                    "memory facade built: base_url=%s",
                    self.config.memory.base_url,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "memory facade unavailable: %s", exc, exc_info=False,
                )
                self.memory = None
        elif self.memory is None:
            _log.info(
                "memory facade not built: no api_key set and "
                "base_url left at default "
                "(set CARE_MEMORY__API_KEY or point "
                "CARE_MEMORY__BASE_URL at your deployment)",
            )
        if self.platform is None and platform_opted_in:
            try:
                from care.platform import CarePlatform

                self.platform = CarePlatform.from_config(self.config)
                _log.info(
                    "platform facade built: base_url=%s",
                    self.config.platform.base_url,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "platform facade unavailable: %s", exc, exc_info=False,
                )
                self.platform = None
        elif self.platform is None:
            _log.info(
                "platform facade not built: no api_key set and "
                "base_url left at default "
                "(set CARE_PLATFORM__API_KEY or point "
                "CARE_PLATFORM__BASE_URL at your deployment)",
            )
        # Mount the toast host before the boot screen so
        # early failure paths (e.g. probe rejection on first
        # paint) can push a toast immediately.
        from care.widgets.status_bar import StatusBar
        from care.widgets.toast import ToastHost

        self._toast_host = ToastHost()
        self.mount(self._toast_host)
        # P1.1: status strip sits above the toast host so a
        # toast can dock below without obscuring the
        # always-visible health line.
        self._status_bar = StatusBar()
        self.mount(self._status_bar)
        self.push_screen(self._build_boot_screen())

    # ------------------------------------------------------------------
    # Toast / notification host (TODO §1.1 P0.35)
    # ------------------------------------------------------------------

    def action_open_task_list(self) -> None:
        """`Ctrl+B` → push the TaskListDrawer (TODO §1.1 P0.36).

        The drawer reads `app.task_registry` so it always
        reflects the current set of running / recent tasks.
        """
        from care.screens.task_list import TaskListDrawer

        try:
            self.push_screen(TaskListDrawer(self.task_registry))
        except Exception:
            pass

    def action_open_help(self) -> None:
        """`?` → push the HelpScreen (TODO §9 P3).

        Reads `care.build_registry()` so plugin extensions show
        up automatically alongside the canonical-flow tutorial +
        documented bindings.
        """
        from care.screens.help import HelpScreen

        try:
            self.push_screen(HelpScreen())
        except Exception:
            pass

    def action_open_catalog(self) -> None:
        """`Ctrl+K` → push the CatalogScreen (§8 P1).

        Constructs the screen with a lazy `catalog_factory` that
        runs :func:`care.build_catalog` against the config's
        default skill / tool / MCP paths plus the host's memory
        (when configured) so the same data the CLI's `care
        catalog` subcommand returns lands in the TUI.
        """
        self._open_catalog_focused(focus_entry_id=None)

    def _open_catalog_focused(
        self, *, focus_entry_id: str | None,
    ) -> None:
        """Shared push helper for the catalog screen.

        `Ctrl+K` lands with `focus_entry_id=None` (top of the
        list). Palette agent_skill picks land with the picked
        entity_id so the table cursor lands on the matching
        row when one exists.
        """
        from care.catalog import build_catalog
        from care.screens.catalog import CatalogScreen

        memory = self.memory

        def _factory():
            return build_catalog(memory=memory)

        try:
            self.push_screen(
                CatalogScreen(
                    catalog_factory=_factory,
                    focus_entry_id=focus_entry_id,
                ),
            )
        except Exception:
            pass

    def action_open_marketplace(self) -> None:
        """Push the MarketplaceScreen (§8 P2).

        No keyboard binding — opened via the command palette
        ("Browse capability marketplace" entry). The screen
        consumes the configured `app.memory.client` for the
        backend search + install flow; an unconfigured memory
        lands an error in the screen's status pane.
        """
        from care.screens.marketplace import MarketplaceScreen

        memory = self.memory
        # Marketplace search hits Memory's `find_capability_matches`
        # — that lives on the SDK client. Prefer client when
        # available, fall back to the facade itself (matches the
        # `search_marketplace` duck-typing pattern).
        target = getattr(memory, "client", None) or memory
        try:
            self.push_screen(MarketplaceScreen(memory=target))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Palette navigation actions (TODO §2 P0)
    # ------------------------------------------------------------------
    #
    # The Screens-group entries — "Open Chat / Library / Settings /
    # Artifacts / Evolution" — route here from `_PALETTE_ACTION_DISPATCH`.
    # Each handler does best-effort navigation: when the destination
    # isn't reachable from the current state (e.g. Memory unconfigured
    # blocks LibraryScreen) the handler surfaces a toast instead of
    # crashing the palette dismissal flow.

    def _pop_to_screen_type(self, target_cls: Any) -> bool:
        """Pop screens until ``target_cls`` is on top. Returns
        True when a matching screen was found + reached, False
        when nothing on the stack matches. Lets a palette action
        navigate *back* to an open screen instead of stacking a
        fresh duplicate."""
        if not self.screen_stack:
            return False
        for screen in self.screen_stack:
            if isinstance(screen, target_cls):
                # Pop until target is on top. screen_stack[-1]
                # is the active screen; iterate while the top
                # isn't our target.
                guard = 0
                while (
                    len(self.screen_stack) > 1
                    and not isinstance(self.screen_stack[-1], target_cls)
                ):
                    self.pop_screen()
                    guard += 1
                    if guard > 32:
                        # Defensive cap — should never trip; if
                        # the stack is somehow circular, stop
                        # rather than spin.
                        break
                return True
        return False

    def action_palette_open_chat(self) -> None:
        """Palette → Open Chat.

        ChatScreen is the home — pop down to it when it's
        already on the stack; otherwise push a fresh instance.
        Best-effort: any push / pop failure falls through to a
        warning toast.
        """
        try:
            from care.screens.chat import ChatScreen

            if not self._pop_to_screen_type(ChatScreen):
                self.push_screen(ChatScreen())
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Couldn't open chat: {exc}", severity="warning",
            )

    def action_palette_open_library(self) -> None:
        """Palette → Open Library.

        Pops to an existing LibraryScreen when one is on the
        stack; otherwise pushes a fresh one. Symmetric with
        the `/library` slash command so keyboard + palette
        navigation reach the same screen.
        """
        try:
            from care.screens.library import LibraryScreen

            if not self._pop_to_screen_type(LibraryScreen):
                self.push_screen(LibraryScreen())
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Couldn't open library: {exc}", severity="warning",
            )

    def action_palette_open_settings(self) -> None:
        """Palette → Open Settings.

        Pushes the existing SettingsScreen with the live
        :class:`CareConfig`. Settings is always reachable —
        no Memory / Platform dependency — so this is the
        simplest of the five.
        """
        try:
            from care.config import CareConfig
            from care.screens.settings import SettingsScreen

            cfg = self.config if self.config is not None else CareConfig.load()
            self.push_screen(SettingsScreen(cfg))
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Couldn't open settings: {exc}", severity="warning",
            )

    def action_palette_open_artifacts(self) -> None:
        """Palette → Open Artifacts (§3 P0).

        Resolves the underlying `ChatScreen` on the screen
        stack and pushes its `ArtifactsScreen`. When no
        ChatScreen is mounted (eg. palette opened from
        another top-level screen pre-chat), surfaces a toast
        pointing at `/artifacts` so the discovery path still
        nudges the user toward the right action.
        """
        try:
            from care.screens.artifacts import ArtifactsScreen
            from care.screens.chat import ChatScreen
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Couldn't load artifacts screen: {exc}",
                severity="error",
            )
            return
        chat: ChatScreen | None = None
        for screen in self.screen_stack:
            if isinstance(screen, ChatScreen):
                chat = screen
                break
        if chat is None:
            self.push_toast(
                "Open the chat first — `/artifacts` browses the "
                "current chat session's artifacts.",
                severity="warning",
            )
            return
        try:
            self.push_screen(ArtifactsScreen(chat.artifact_store))
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Couldn't open artifacts: {exc}", severity="error",
            )

    def action_palette_open_evolution(self) -> None:
        """Palette → Open Evolution (§5 P0).

        Pushes :class:`EvolutionDashboard` listing every
        recent + active run. Platform-unconfigured paints a
        warning toast — the dashboard itself also surfaces
        the same diagnostic once mounted, but the early
        warning keeps the palette-flow cheap.
        """
        if self.platform is None:
            self.push_toast(
                "Evolution dashboard needs a configured "
                "Platform facade — set CARE_PLATFORM__BASE_URL "
                "first.",
                severity="warning",
            )
            return
        try:
            from care.screens.evolution_dashboard import EvolutionDashboard

            self.push_screen(EvolutionDashboard())
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Couldn't open evolution dashboard: {exc}",
                severity="error",
            )

    # ------------------------------------------------------------------
    # CatalogScreen / MarketplaceScreen message handlers
    # ------------------------------------------------------------------

    def on_catalog_screen_promote_requested(self, event: Any) -> None:
        """Persist a `CatalogScreen.PromoteRequested` to Memory.

        The CatalogScreen stays pure-presentation; the actual
        upload happens here so the host owns the Memory wiring +
        toast surface. Runs the upload on a worker so a slow
        Memory backend doesn't freeze the UI thread.
        """
        entry = getattr(event, "entry", None)
        if entry is None:
            return
        if self.memory is None:
            self.push_toast(
                "Promote needs a configured Memory facade — "
                "check care.toml / CARE_MEMORY__* env vars.",
                severity="error",
            )
            return
        self.run_worker(
            self._promote_catalog_entry(entry),
            name="catalog_promote",
            group="catalog_promote",
            exclusive=False,
            exit_on_error=False,
        )

    async def _promote_catalog_entry(self, entry: Any) -> None:
        """Worker payload — calls
        :func:`care.promote_skill_to_memory` against the entry's
        SKILL.md `source` path. Posts toast on completion."""
        import asyncio as _asyncio

        from care.skills import (
            SkillPromotionError,
            promote_skill_to_memory,
        )

        try:
            entity_id = await _asyncio.to_thread(
                promote_skill_to_memory,
                entry.source,
                self.memory,
                # The catalog entry carries the manifest's
                # display name + tags; promote_skill_to_memory's
                # own fallbacks pick those up from the SKILL.md
                # itself, but passing them through here means
                # the catalog's already-parsed view stays the
                # source of truth (e.g. a tag the user added
                # via `--skills` override).
                name=entry.name,
                tags=list(entry.tags) if entry.tags else None,
            )
        except SkillPromotionError as exc:
            self.push_toast(
                f"Promote failed: {exc}", severity="error",
            )
            return
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Promote crashed: {type(exc).__name__}: {exc}",
                severity="error",
            )
            return
        self.push_toast(
            f"Promoted {entry.name} → {entity_id}",
            severity="success",
        )

    def on_marketplace_screen_installed(self, event: Any) -> None:
        """App-level reaction to a successful marketplace install.

        The `MarketplaceScreen` already pushed its own toast on
        the success path; this handler is the hook for app-level
        side-effects (telemetry, library refresh, etc.). Today
        it's an explicit no-op — the existence of the handler
        prevents Textual from logging "no handler" warnings and
        gives future iterations a single anchor point to wire
        new behaviour."""
        # Intentional no-op. Screen's own toast handles the user
        # feedback; future iterations can hook telemetry / a
        # library-refresh post here without modifying the screen.
        return

    def on_evolution_screen_acceptance_complete(
        self, event: Any,
    ) -> None:
        """App-level reaction to evolution accept-winner.

        The screen posts this after a successful
        `accept_individual` call — the chosen individual was
        promoted to Memory's `stable` channel. We surface a
        success toast so the user gets a final confirmation
        even after they navigate away from the EvolutionScreen,
        and refresh any LibraryScreen on the stack so the
        newly-stable individual appears immediately instead of
        on the user's next manual refresh.
        """
        evolution_id = getattr(event, "evolution_id", None) or "?"
        individual_id = getattr(event, "individual_id", None) or "?"
        # §5 P0 — when the platform shipped version + chain_id
        # in the accept response, render the canonical
        # `<chain_id> v(N) → v(N+1) (now latest)` form so the
        # user sees what was bumped. Older platforms / SDKs
        # leave these as None / "" and we fall back to the
        # legacy "Accepted <individual> from evolution <id>"
        # toast — strictly more information than before, never
        # less.
        chain_id = getattr(event, "chain_id", "") or ""
        previous_version = getattr(event, "previous_version", None)
        new_version = getattr(event, "new_version", None)
        if chain_id and new_version is not None:
            if previous_version is not None:
                message = (
                    f"Accepted: {chain_id} "
                    f"v{previous_version} → v{new_version} "
                    f"(now latest)"
                )
            else:
                message = (
                    f"Accepted: {chain_id} v{new_version} "
                    f"(now latest)"
                )
        else:
            message = (
                f"Accepted {individual_id} from evolution "
                f"{evolution_id} → promoted to stable"
            )
        self.push_toast(message, severity="success")
        self._refresh_library_screens()
        # §5 P1 — surface the accept-winner integration snippet so
        # the user copy-pastes a `gigaevo_client.get_chain(...,
        # version="latest")` (or curl / cli) call into their own
        # service in ≤ 2 keystrokes. Skip silently when the
        # platform didn't ship `chain_id` (older SDK / older
        # platform — the legacy toast above is the best signal we
        # can give in that case).
        if chain_id:
            self._push_use_it_now_for_accepted(
                chain_id=chain_id,
                new_version=new_version,
            )

    def _push_use_it_now_for_accepted(
        self,
        *,
        chain_id: str,
        new_version: int | None,
    ) -> None:
        """Push :class:`UseItNowModal` immediately after a
        successful accept-winner.

        Carries ``channel="stable"`` (the accept flow flips the
        chain to the stable channel) and the new version integer
        in the modal's `version=` field so the rendered snippet
        pins the user's downstream service to the right
        revision. On dismiss with ``evolve_requested=True`` we
        route through the existing
        :meth:`_push_evolution_for` opener so the user can fold
        the just-accepted chain into another evolution run.

        Best-effort: a missing import, no `push_screen` method
        on the host (rare unit-test scaffolds), or an unset
        Memory facade all degrade silently — the legacy
        accept-winner toast above already gave the user the
        success signal.
        """
        if not chain_id:
            return
        push = getattr(self, "push_screen", None)
        if not callable(push):
            return
        try:
            from care.screens.use_it_now import (
                UseItNowModal,
                UseItNowResult,
            )
        except Exception:
            return

        base_url = ""
        memory = getattr(self, "memory", None)
        if memory is not None:
            client = getattr(memory, "client", None)
            base_url = (
                str(getattr(client, "base_url", "") or "")
                or str(getattr(memory, "base_url", "") or "")
            )

        version_str = (
            str(new_version) if new_version is not None else None
        )

        def _on_dismiss(result: UseItNowResult | None) -> None:
            if result is None:
                return
            if result.evolve_requested:
                self._push_evolution_for(chain_id)

        try:
            push(
                UseItNowModal(
                    entity_id=chain_id,
                    version=version_str,
                    channel="stable",
                    display_name=chain_id,
                    memory_base_url=base_url,
                ),
                _on_dismiss,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "Couldn't push UseItNowModal for accepted "
                "chain %s: %s",
                chain_id, exc, exc_info=False,
            )

    def _refresh_library_screens(self) -> None:
        """Trigger :meth:`care.screens.library.LibraryScreen.refresh_library`
        on every `LibraryScreen` currently on the screen stack.

        Used by acceptance flows (evolution accept, future
        promote-from-inspection) to keep the library view
        consistent with Memory after a mutation that lands a
        new stable row. Walks the full stack — typically
        zero or one library is mounted, but the loop costs
        nothing and stays correct if a future flow stacks two
        library views (e.g. comparison split-pane).

        Best-effort: per-library refresh failures (worker
        couldn't start, screen torn down between the lookup
        and the call) are swallowed so a single bad screen
        doesn't block the others — the original toast already
        gave the user the success signal.
        """
        # Lazy import — the canonical pattern in this module
        # (avoids paying the LibraryScreen import cost on every
        # `from care.app import CareApp`).
        from care.screens.library import LibraryScreen

        if not self.screen_stack:
            return
        for screen in list(self.screen_stack):
            if not isinstance(screen, LibraryScreen):
                continue
            try:
                screen.refresh_library()
            except Exception:  # noqa: BLE001
                # Surfaced in the app log via the screen's own
                # worker telemetry; we deliberately swallow
                # here so iteration over the stack continues.
                _log.warning(
                    "LibraryScreen refresh failed after "
                    "evolution-accept; continuing.",
                    exc_info=True,
                )

    def on_inspection_screen_action_requested(self, event: Any) -> None:
        """Route an InspectionScreen action-row click.

        The screen posts ``ActionRequested(action, entity_id)``
        for each of the documented five buttons (``run`` / ``edit``
        / ``evolve`` / ``duplicate`` / ``back``). The host's job is
        to push the destination screen + carry the entity_id
        through.

        ``back`` is the InspectionScreen's own pop affordance —
        the screen already pops itself, so the host emits no
        toast (matches the cancel convention).

        Other actions push the matching destination screen. The
        push is best-effort: a constructor failure (rare; mostly
        a stale entity_id) lands an error toast rather than
        crashing the stack.
        """
        action = getattr(event, "action", None)
        entity_id = getattr(event, "entity_id", None) or ""
        if not action or not entity_id:
            return
        if action == "back":
            return
        if action == "run":
            # `Run` from inspection: load the chain and execute it via
            # CARL, rendering live progress on an ExecutionScreen — the
            # in-TUI sibling of `care run <id> --execute`.
            self._push_run_for(entity_id)
            return
        if action == "edit":
            self._push_edit_agent_for(entity_id)
            return
        if action == "evolve":
            self._push_evolution_for(entity_id)
            return
        if action == "duplicate":
            self.push_toast(
                f"Duplicate {entity_id}: not yet wired — use "
                f"`care memory show {entity_id} --export ...` "
                f"+ `care import` round-trip for now.",
                severity="info",
            )
            return

    def _push_edit_agent_for(self, entity_id: str) -> None:
        """Construct EditAgentScreen from a saved chain.

        EditAgentScreen requires the chain object (not the
        entity_id) — we fetch the chain dict and let the screen's
        own `extract_edit_draft` projector handle it. Memory
        unconfigured → error toast.
        """
        if self.memory is None:
            self.push_toast(
                "Edit needs a configured Memory facade.",
                severity="error",
            )
            return
        try:
            chain_dict = self.memory.get_chain(entity_id)
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Edit failed: {exc}", severity="error",
            )
            return
        try:
            from care.screens.edit_agent import EditAgentScreen

            self.push_screen(
                EditAgentScreen(
                    chain=chain_dict,
                    memory=self.memory,
                    entity_id=entity_id,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Edit failed to open: {exc}", severity="error",
            )

    def _revise_chain_for(self, entity_id: str) -> None:
        """Library → 'Revise (AI)': drop to chat with ``/revise <id> `` seeded.

        The conversational ``/revise`` flow (ChatScreen) owns the actual NL
        edit; this is only the hand-off so the user lands in chat ready to type
        the change. Best-effort — a missing chat surface degrades to a toast.
        """
        try:
            from care.screens.chat import ChatScreen
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                t("app.toast.openChatFailed", error=exc), severity="warning",
            )
            return
        if not self._pop_to_screen_type(ChatScreen):
            self.push_screen(ChatScreen())
        chat: ChatScreen | None = None
        for screen in self.screen_stack:
            if isinstance(screen, ChatScreen):
                chat = screen
                break
        if chat is None:
            self.push_toast(
                t("app.toast.openChatToRevise"), severity="warning",
            )
            return
        seed = getattr(chat, "seed_input", None)
        if callable(seed):
            seed(f"/revise {entity_id} ")

    def _push_evolution_for(self, entity_id: str = "") -> None:
        """Push :class:`EvolutionLaunchModal` for ``entity_id``
        (§4 P0).

        On Submit the modal dismisses with an
        :class:`EvolutionLaunchSpec`; we then push
        :class:`EvolutionScreen` seeded with the user's
        choices. Cancel (dismiss with ``None``) is a no-op.

        ``entity_id`` is optional: the Inspection / Library Evolve
        buttons pass the inspected chain so the modal opens pre-bound,
        while ``/evolution`` from chat may open it cold (empty) so the
        user types the chain to evolve into the modal's Base-chain field.
        Either way the launched run uses ``spec.base_chain_id`` — the
        field's value — as the source of truth.

        Falls back to direct EvolutionScreen push when the
        modal can't be imported (defensive — tests with
        slimmed-down test scaffolds may stub the import).
        """
        if self.platform is None:
            self.push_toast(
                "Evolve needs a configured Platform facade — "
                "check CARE_PLATFORM__BASE_URL.",
                severity="error",
            )
            return

        try:
            from care.screens.evolution_launch import (
                EvolutionLaunchModal,
                EvolutionLaunchSpec,
            )
        except Exception:
            self._push_evolution_screen_direct(entity_id)
            return

        def _on_dismiss(spec: EvolutionLaunchSpec | None) -> None:
            if spec is None or not spec.base_chain_id:
                return
            self._push_evolution_screen_with_spec(spec)

        try:
            self.push_screen(
                EvolutionLaunchModal(base_chain_id=entity_id or ""),
                _on_dismiss,
            )
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Evolve launch modal failed: {exc}",
                severity="error",
            )

    def _push_evolution_screen_direct(self, entity_id: str) -> None:
        """Bypass-the-modal escape hatch — push
        EvolutionScreen with default kwargs. Kept available
        for diagnostic flows + test scaffolds that don't want
        to drive the modal."""
        try:
            from care.screens.evolution import EvolutionScreen

            self.push_screen(
                EvolutionScreen(base_chain_id=entity_id),
            )
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Evolve failed to open: {exc}", severity="error",
            )

    def _push_evolution_screen_with_spec(self, spec: Any) -> None:
        """Push EvolutionScreen seeded with the user's launch
        spec. ``spec`` is an
        :class:`care.screens.evolution_launch.EvolutionLaunchSpec`
        — declared as ``Any`` here to keep the lazy import
        pattern consistent with the rest of the file. The base chain
        comes from ``spec.base_chain_id`` (the modal's editable field)."""
        base_chain_content: dict | None = None
        if self.memory is not None and spec.base_chain_id:
            try:
                fetched = self.memory.get_chain(spec.base_chain_id)
                if isinstance(fetched, dict) and fetched.get("steps"):
                    base_chain_content = fetched
            except Exception:
                pass

        try:
            from care.screens.evolution import EvolutionScreen

            self.push_screen(
                EvolutionScreen(
                    base_chain_id=spec.base_chain_id,
                    base_chain_content=base_chain_content,
                    **spec.to_screen_kwargs(),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Evolve failed to open: {exc}", severity="error",
            )

    def _push_run_for(self, entity_id: str) -> None:
        """Run a saved chain end-to-end via CARL, rendering live
        progress on an :class:`ExecutionScreen` — the in-TUI sibling of
        ``care run <id> --execute``.

        Reuses the same library-run pipeline the promote-gate baseline
        runner uses (`load_run_plan` → `execute_library_run`), driven
        through a `CarlStreamer` so the user watches steps complete in
        real time. The flow first opens a :class:`RunContextModal` so
        the user supplies the task — most saved chains carry no stored
        task, and CARL's executor *requires* one (an empty task fails
        validation, which is what left the run screen blank before).
        Failure modes (no Memory, missing LLM key, load error) surface
        as toasts instead of a silent no-op.
        """
        # Precondition toasts use a long ttl (and stay until dismissed
        # via ttl=0 caps) so the user actually SEES why Run stopped —
        # a default 3s corner toast reads as "nothing happened".
        if self.memory is None:
            self.push_toast(
                "Run needs a configured Memory facade — set "
                "CARE_MEMORY__BASE_URL first.",
                severity="error",
                ttl=0,
            )
            return
        # Build the CARL LLM client up front so a missing api key fails
        # fast with a clear toast rather than mid-run.
        try:
            from care.runtime.llm_client import build_carl_llm_client

            api = build_carl_llm_client(self.config.mage)
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Run unavailable: {exc}", severity="error", ttl=0,
            )
            return
        self.run_worker(
            self._run_saved_chain(entity_id, api),
            name="library_run",
            group="library_run",
            exclusive=False,
            exit_on_error=False,
        )

    async def _run_saved_chain(self, entity_id: str, api: Any) -> None:
        """Worker: load the chain → collect the task via RunContextModal
        → execute with a streamer driving a live ExecutionScreen → toast
        the outcome. The ExecutionScreen is only pushed once the user
        confirms a task, so a cancel or a hard failure never strands the
        user on an empty run screen."""
        from care.runtime.carl_streamer import CarlStreamer
        from care.runtime.library_run import (
            LibraryRunError,
            execute_library_run,
            load_run_plan,
        )
        from care.screens.execution import ExecutionScreen, project_chain_steps
        from care.screens.run_context import RunContextModal

        try:
            plan = await load_run_plan(self.memory, entity_id)
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Run failed to load chain: {exc}",
                severity="error", ttl=0,
            )
            return

        from care.screens.chat import ChatScreen
        from care.screens.data_intro import DataIntroModal

        if not ChatScreen._tutorial_seen("data_intro_shown"):
            ChatScreen._mark_tutorial_seen("data_intro_shown")
            try:
                await self.push_screen_wait(DataIntroModal())
            except Exception:  # noqa: BLE001
                pass

        # Collect / confirm the task + inputs. The modal gates submit on
        # a non-empty task, so the draft we get back always validates.
        try:
            result = await self.push_screen_wait(
                RunContextModal(plan.chain, source_name=plan.display_name),
            )
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                t("app.toast.runUnavailable", error=exc), severity="error",
            )
            return
        if result is None or not getattr(result, "submitted", False):
            return  # user cancelled — nothing to run
        draft = result.draft

        title = plan.display_name or entity_id
        screen = ExecutionScreen(
            title=title,
            total_steps=self._chain_step_count(plan.chain),
            chain_steps=project_chain_steps(plan.chain),
        )
        self.push_screen(screen)
        streamer = CarlStreamer(screen)
        try:
            completion = await execute_library_run(
                self.memory, plan, draft,
                config=self.config, api=api, streamer=streamer,
            )
        except LibraryRunError as exc:
            self._pop_if_top(screen)
            self.push_toast(
                t("app.toast.runFailed", error=exc), severity="error",
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._pop_if_top(screen)
            self.push_toast(
                f"Run failed: {type(exc).__name__}: {exc}",
                severity="error",
            )
            return
        summary = completion.summary
        if getattr(summary, "success", False):
            self.push_toast(
                f"✓ Run succeeded in {summary.duration_seconds:.1f}s "
                f"({summary.step_count} steps).",
                severity="success",
            )
            refresh = getattr(self, "_refresh_library_screens", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception:  # noqa: BLE001
                    pass
        else:
            self.push_toast(
                "Run finished with errors: "
                f"{getattr(summary, 'error_message', None) or 'unsuccessful'}",
                severity="warning",
            )

    @staticmethod
    def _chain_step_count(chain: Any) -> int:
        """Best-effort step count for the ExecutionScreen's progress
        denominator. ``0`` when unknown — Progress events fill it in."""
        steps = getattr(chain, "steps", None)
        if steps is None and isinstance(chain, dict):
            steps = chain.get("steps")
        try:
            return len(steps) if steps is not None else 0
        except Exception:  # noqa: BLE001
            return 0

    def _pop_if_top(self, screen: Any) -> None:
        """Pop ``screen`` only when it's still the active screen — so a
        failed run returns the user to Inspect instead of stranding them
        on a blank ExecutionScreen."""
        try:
            if self.screen is screen:
                self.pop_screen()
        except Exception:  # noqa: BLE001
            pass

    def on_settings_screen_saved(self, event: Any) -> None:
        """SettingsScreen → Save: refresh app's facade objects.

        When the user saves new settings the in-memory `memory`
        / `platform` facades may now be stale (pointing at the
        old base URL / api key). Rebuild them from the new
        config, push a confirmation toast.

        Defensive: any rebuild failure (e.g. malformed config
        the validator missed) lands an error toast and leaves
        the existing facades untouched.
        """
        snapshot = getattr(event, "snapshot", None)
        if snapshot is None:
            return
        from care.config import CareConfig
        from care.memory import CareMemory
        from care.platform import CarePlatform

        # Capture state BEFORE the reload so we can show the user what
        # changed: the pre-save config + the chat surface to post into
        # (resolved now, while it's still under the SettingsScreen on
        # the stack — the boot/first-run path has no chat to post to).
        old_config = self.config
        chat = self._find_chat_screen()

        try:
            self.config = CareConfig.load()
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Settings saved but reload failed: {exc}",
                severity="warning",
            )
            return
        try:
            self.memory = CareMemory.from_config(self.config)
        except Exception:  # noqa: BLE001
            self.memory = None
        try:
            self.platform = CarePlatform.from_config(self.config)
        except Exception:  # noqa: BLE001
            self.platform = None
        self.push_toast(
            "Settings saved · Memory / Platform reloaded",
            severity="success",
        )
        # Dismiss the SettingsScreen so the user lands on
        # something useful. Two cases:
        #
        # * Boot-time SettingsScreen (missing-creds routing):
        #   the stack is `[SettingsScreen]` because
        #   WelcomeScreen used `switch_screen`. A plain
        #   `pop_screen()` would empty the stack and land the
        #   user on Textual's blank default screen — instead
        #   we `switch_screen` to the freshly-resolved next
        #   screen (ChatScreen now that creds are present).
        #
        # * In-app Settings entry (e.g. opened from the
        #   command palette): the stack has at least one
        #   screen underneath the Settings screen, so a normal
        #   `pop_screen()` returns the user to where they
        #   were.
        #
        # Guard the dismiss: only fire when the active screen
        # is actually SettingsScreen so callers that fake a
        # Saved event (tests, telemetry replays) don't blow
        # away unrelated screens.
        try:
            from care.screens.settings import SettingsScreen as _SettingsScreen
            from care.screens.welcome import default_next_screen

            top = self.screen if self.screen_stack else None
            if isinstance(top, _SettingsScreen):
                if len(self.screen_stack) <= 1:
                    # Re-route from a fresh CareConfig so the
                    # post-save creds are reflected (e.g.
                    # missing → present flips routing back to
                    # LibraryScreen).
                    # After a successful save the user has
                    # working config on disk — treat them as
                    # "returning" so routing lands on ChatScreen
                    # / LibraryScreen instead of bouncing back
                    # to SettingsScreen (which is what
                    # `first_run` mode resolves to).
                    self.mode = "returning"
                    next_screen = default_next_screen("returning")
                    self.switch_screen(next_screen)
                else:
                    self.pop_screen()
        except Exception:
            pass

        # Leave a visible breadcrumb in the chat trace so the user sees
        # the `/settings` save landed (and what it touched). Only when a
        # chat surface already existed underneath Settings — the
        # first-run path switches to a fresh ChatScreen with no prior
        # transcript to annotate.
        if chat is not None:
            try:
                from care.config import summarize_config_changes

                changes = (
                    summarize_config_changes(old_config, snapshot.config)
                    if old_config is not None
                    else []
                )
                chat.post_settings_updated(changes)
            except Exception:
                pass
            # A UI-language switch only takes effect on newly-rendered
            # text — re-localize the chat's already-mounted chrome
            # (welcome banner, mode toggle, badge) so it doesn't stay in
            # the old language. `set_ui_language` already ran in the
            # SettingsScreen save, so `t()` now resolves to the new one.
            try:
                language_changed = (
                    old_config is not None
                    and old_config.defaults.ui_language
                    != snapshot.config.defaults.ui_language
                )
                if language_changed:
                    relocalize = getattr(chat, "relocalize", None)
                    if callable(relocalize):
                        relocalize()
            except Exception:
                pass

    def _find_chat_screen(self) -> Any:
        """Return the `ChatScreen` on the stack, or ``None``.

        Several handlers need to post into / act on the live chat
        surface regardless of which screen is on top; centralise the
        lookup so they don't each re-implement the stack walk."""
        try:
            from care.screens.chat import ChatScreen
        except Exception:
            return None
        for screen in self.screen_stack:
            if isinstance(screen, ChatScreen):
                return screen
        return None

    def on_settings_screen_cancelled(self, event: Any) -> None:
        """SettingsScreen → Cancel: silent (cancel paths don't
        toast). Anchor exists so Textual doesn't log
        "no handler" warnings."""
        return

    def on_query_screen_generate_requested(self, event: Any) -> None:
        """QueryScreen → Generate: push GenerationScreen + run
        a MAGE worker that streams stage events into the screen.

        When MAGE isn't available (extra not installed, missing
        API key, malformed config), the worker falls back to a
        short synthetic stage sequence so the user can see the
        screen working and gets a clear "configure MAGE" hint
        in the metadata footer.
        """
        submission = getattr(event, "submission", None)
        if submission is None:
            _log.debug("generate requested with no submission attached; ignoring.")
            return
        task = getattr(submission, "task", None) or ""
        if not task:
            _log.debug("generate requested with empty task; ignoring.")
            return
        mode = getattr(submission, "mage_mode", None) or "deep"
        files = tuple(getattr(submission, "files", ()) or ())
        _log.info(
            "generate requested: mode=%s files=%d task=%r",
            mode, len(files), task[:120],
        )

        from care.screens.generation import GenerationScreen

        preview = task if len(task) <= 80 else task[:77] + "…"
        screen = GenerationScreen(task_preview=preview)
        try:
            self.push_screen(screen)
        except Exception as exc:  # noqa: BLE001
            _log.error("push GenerationScreen failed: %s", exc, exc_info=True)
            self.push_toast(
                f"Couldn't open generation screen: {exc}",
                severity="error",
            )
            return

        try:
            screen.run_worker(
                self._run_mage_generation(screen, task, files, mode),
                name="mage_generate",
                group="generate",
                exclusive=True,
                exit_on_error=False,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("worker spawn failed: %s", exc, exc_info=True)
            self.push_toast(
                f"Couldn't start generation worker: {exc}",
                severity="error",
            )

    async def _run_mage_generation(
        self,
        screen: Any,
        task: str,
        files: tuple[Any, ...],
        mode: str,
    ) -> None:
        """Worker body — drive MAGE end-to-end against the
        GenerationScreen. Cancelled via `Esc` on the screen
        (the screen's `action_cancel_generate` calls
        `workers.cancel_group(..., "generate")`)."""
        from care.generation import (
            GenerationError,
            build_mage_generator,
            run_generation,
        )
        from care.runtime.mage_poster import MagePoster, StageError

        poster = MagePoster(screen)
        _log.info("building MAGE generator (mode=%s)", mode)
        try:
            generator = build_mage_generator(
                self.config, progress=poster, mode=mode,  # type: ignore[arg-type]
            )
        except GenerationError as exc:
            _log.warning("MAGE unavailable: %s", exc)
            await self._simulate_unavailable_generation(screen, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            _log.error("MAGE setup failed: %s", exc, exc_info=True)
            await self._simulate_unavailable_generation(
                screen, f"Setup failed: {exc}",
            )
            return

        context_files = (
            [{"path": str(p)} for p in files] if files else None
        )
        _log.info(
            "MAGE.generate starting (task_len=%d context_files=%d)",
            len(task), len(context_files or ()),
        )
        try:
            result = await run_generation(
                generator, task, context_files=context_files,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("MAGE.generate raised: %s", exc, exc_info=True)
            screen.post_message(StageError("mage", RuntimeError(str(exc))))
            self.push_toast(
                f"Generation failed: {exc}", severity="error",
            )
            return

        _log.info("MAGE.generate succeeded; rendering metadata.")
        try:
            screen.record_mage_result(result)
        except Exception:
            _log.exception("record_mage_result failed")
        self.push_toast(t("app.toast.generationComplete"), severity="success")

    async def _simulate_unavailable_generation(
        self, screen: Any, message: str,
    ) -> None:
        """Fallback when MAGE isn't wired — walk the canonical
        MAGE stage sequence with synthetic artifacts so the
        screen visibly fills in, then surface the underlying
        message in the metadata footer.

        The artifacts are *placeholders* — the right pane shows
        labels mirroring what a real MAGE run would emit so the
        user can verify the screen's data flow end-to-end without
        the upstream package being installed.
        """
        import asyncio

        from care.runtime.mage_poster import (
            StageCompleted,
            StageProgress,
            StageStarted,
        )
        from textual.widgets import Static

        _log.info("MAGE fallback path: %s", message)
        stages: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("preflight", ("config validated", "task length checked")),
            ("memory_research", ("no API key — skipped",)),
            ("plan", ("Step 1: ingest", "Step 2: analyse", "Step 3: report")),
            (
                "describe",
                (
                    "ingest: load source documents",
                    "analyse: extract key claims",
                    "report: synthesise summary",
                ),
            ),
            ("critique", ("placeholder pass — agent chain generator unavailable",)),
            ("verify", ("placeholder chain is well-formed",)),
        )
        for stage, artifacts in stages:
            _log.debug("fallback stage: %s start", stage)
            screen.post_message(StageStarted(stage))
            for artifact in artifacts:
                await asyncio.sleep(0.12)
                screen.post_message(StageProgress(stage, artifact))
            await asyncio.sleep(0.1)
            screen.post_message(StageCompleted(stage, None))
            _log.debug("fallback stage: %s done", stage)

        notice = (
            "Agent chain generator didn't run.\n\n"
            f"{message}\n\n"
            "Install the agent chain generator extra and set CARE_MAGE__API_KEY, "
            "or use the CLI: `care generate \"<task>\"`."
        )
        try:
            target = screen.query_one("#generation-metadata", Static)
            target.update(notice)
        except Exception:
            pass
        self.push_toast(
            "Agent chain generator not configured — see generation screen for details.",
            severity="warning",
        )

    def on_query_screen_back_requested(self, event: Any) -> None:
        """QueryScreen → Ctrl+L (back to library). The screen
        already pops itself on the back gesture; the App
        handler is the telemetry/anchor hook.

        Today: silent no-op anchor — present so Textual doesn't
        log "no handler" warnings and future iterations can
        wire LibraryScreen-refresh logic here without modifying
        the screen.
        """
        return

    def on_task_list_drawer_switch_requested(
        self, event: Any,
    ) -> None:
        """TaskListDrawer → switch focus to a specific running
        task.

        Surfaces an info toast with the picked task's id so the
        user gets visible feedback. Full focus-switch wiring
        (i.e. popping screens until the screen that owns this
        task is on top) requires per-task screen attribution
        the registry doesn't track yet — flagged in the TODO
        history as a follow-up.
        """
        record = getattr(event, "record", None)
        if record is None:
            return
        task_id = getattr(record, "task_id", None) or "?"
        description = getattr(record, "description", None) or ""
        self.push_toast(
            f"Task {task_id}: {description}", severity="info",
        )

    def on_text_selected(self, event: Any) -> None:
        """App-wide copy-on-selection.

        Textual's ``TextSelected`` event bubbles from the screen to
        the app, so handling it here gives every screen the same
        copy-on-drag-release gesture the ChatScreen pioneered — drag
        to highlight, release, and the clipboard is updated with a
        transient toast. ChatScreen keeps its own handler (and calls
        ``event.stop()``) so it isn't double-copied; every other
        screen falls through to this default.
        """
        from care.runtime.clipboard import copy_selection

        copy_selection(self, self.screen)

    def on_welcome_screen_resume_requested(
        self, event: Any,
    ) -> None:
        """WelcomeScreen → Resume: the user picked the
        resume-on-startup card.

        Today: surfaces a toast with the resumed RunState's
        chain_id (when available) so the user sees the pick
        was registered. Full resume push (rebuild ExecutionScreen
        from the snapshot) is gated on the
        `ReasoningContext.restore(snapshot)` runtime wiring the
        CLI's `care run --execute` also blocks on.
        """
        state = getattr(event, "state", None)
        if state is None:
            self.push_toast(
                "Resume: no run state available — try again "
                "after a chain has executed.",
                severity="warning",
            )
            return
        chain_id = getattr(state, "chain_id", None) or "?"
        self.push_toast(
            f"Resume queued for chain {chain_id} — TUI executor "
            "wiring pending.",
            severity="info",
        )

    def on_import_modal_preview_loaded(self, event: Any) -> None:
        """ImportModal → preview ready.

        The screen renders the preview internally; this anchor
        is a no-op so Textual doesn't log "no handler" warnings.
        Future iterations can hook telemetry / library-refresh
        side effects here without modifying the modal.
        """
        return

    def on_welcome_screen_recent_selected(self, event: Any) -> None:
        """Recent-pick on the WelcomeScreen → push InspectionScreen.

        The welcome screen surfaces recent chains in its bottom
        pane; selecting one is the canonical "jump to last
        agent" gesture. The event carries a `row: LibraryRow`;
        we pull `entity_id` off the row.
        """
        row = getattr(event, "row", None)
        if row is None:
            return
        entity_id = getattr(row, "entity_id", None) or ""
        if not entity_id:
            return
        try:
            from care.screens.inspection import InspectionScreen

            self.push_screen(InspectionScreen(entity_id))
        except Exception as exc:  # noqa: BLE001
            self.push_toast(
                f"Recent open failed: {exc}", severity="error",
            )

    def on_edit_agent_screen_submitted(self, event: Any) -> None:
        """App-level reaction to an EditAgentScreen terminal
        action (Save / Promote / Back).

        Surfaces a toast describing the outcome so the user
        gets feedback even after popping the screen. The screen
        itself handles inline validation + per-field error
        rendering; this handler covers the post-dismiss
        signal.
        """
        payload = getattr(event, "payload", None)
        if payload is None:
            return
        action = getattr(payload, "action", None) or "edit"
        if action == "back":
            # User chose Back — no toast (matches every other
            # cancel path in CARE).
            return
        save_result = getattr(payload, "save_result", None)
        promote_result = getattr(payload, "promote_result", None)
        if action == "save":
            if save_result is None or getattr(
                save_result, "error", None,
            ):
                err = (
                    getattr(save_result, "error", None)
                    if save_result is not None else "unknown error"
                )
                self.push_toast(
                    f"Save failed: {err}", severity="error",
                )
                return
            entity_id = getattr(save_result, "entity_id", "?")
            self.push_toast(
                f"Saved → {entity_id}", severity="success",
            )
            # Close the Edit screen so the user lands back on the chain
            # they were inspecting; InspectionScreen.on_screen_resume
            # re-fetches, so the saved edits show immediately instead of
            # the stale pre-edit version.
            self._pop_edit_screen_after_save()
            return
        if action == "promote":
            if promote_result is None or getattr(
                promote_result, "error", None,
            ):
                err = (
                    getattr(promote_result, "error", None)
                    if promote_result is not None else "unknown error"
                )
                self.push_toast(
                    f"Promote failed: {err}", severity="error",
                )
                return
            self.push_toast(
                "Pinned latest → stable channel", severity="success",
            )

    def _pop_edit_screen_after_save(self) -> None:
        """Pop the EditAgentScreen once its save succeeds so the user
        returns to the screen they came from (Inspection / Library).

        Best-effort and idempotent: only pops when an EditAgentScreen is
        actually on top, and never pops the last screen on the stack."""
        try:
            from care.screens.edit_agent import EditAgentScreen
        except Exception:
            return
        try:
            if not isinstance(self.screen, EditAgentScreen):
                return
            if len(self.screen_stack) <= 1:
                return
            self.pop_screen()
        except Exception:
            pass

    def push_toast(
        self,
        message: str,
        *,
        severity: str = "info",
        ttl: float | None = None,
    ) -> None:
        """Append a toast to the app-level :class:`ToastHost`.

        Severity options: ``"info"`` / ``"success"`` /
        ``"warning"`` / ``"error"``. Auto-dismisses after
        ``ttl`` seconds (default 3.0); pass ``ttl=0`` to keep
        the toast until manually dismissed.
        """
        host = getattr(self, "_toast_host", None)
        if host is None:
            return
        try:
            host.push(
                message, severity=severity, ttl=ttl,  # type: ignore[arg-type]
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Theming (TODO §1 P0.6)
    # ------------------------------------------------------------------

    def _register_care_themes(self) -> None:
        """Register every CARE-side concrete theme with Textual's
        theme registry so `App.theme = "<name>"` paints the
        palette.

        Auto-kind themes don't get registered — the resolver
        always returns a concrete light/dark theme.
        """
        already = set(self.available_themes)
        for theme in list_themes():
            if theme.is_auto or theme.name in already:
                continue
            self.register_theme(_care_theme_to_textual(theme))

    def _apply_active_theme(self) -> None:
        """Resolve `self.theme_pref` against the current system
        appearance and switch Textual to the right palette."""
        # Settings → Theme tab lets users pick any Textual-
        # registered theme (catppuccin / dracula / nord / …),
        # which `resolve_active_theme` doesn't know about (it
        # only walks the CARE registry). When the preference
        # matches a live Textual theme, honour it directly so
        # the saved choice actually restores on next launch
        # instead of falling back to dark via the unknown-name
        # branch.
        pref_name = self.theme_pref.theme_name
        care_known = {t.name for t in list_themes()}
        if (
            pref_name not in care_known
            and pref_name in self.available_themes
        ):
            self.theme = pref_name
            return
        resolved = resolve_active_theme(
            pref_name,
            system_appearance=self.system_appearance,
        )
        self.active_theme: CareTheme = resolved
        # Only set Textual's theme when our resolved theme is
        # registered there (built-ins always are; custom themes
        # registered via `register_theme` are too).
        if resolved.name in self.available_themes:
            self.theme = resolved.name

    def apply_care_theme(self, theme_name: str, *, persist: bool = True) -> CareTheme:
        """Public API: switch to a registered theme by name.

        Args:
            theme_name: Name of the CARE theme to apply (must
                exist in :func:`list_themes`).
            persist: When ``True`` (default), saves the choice
                via :func:`save_theme_preference` so the
                preference survives a restart. Pass ``False``
                for transient previews (e.g. the SettingsScreen
                live-preview pane).

        Returns:
            The concrete :class:`CareTheme` that was applied
            (after auto-resolution against the current system
            appearance).
        """
        self.theme_pref = ThemePreference(theme_name=theme_name)
        if persist:
            try:
                save_theme_preference(self.theme_pref)
            except OSError:
                # Best-effort persistence — a read-only config
                # directory shouldn't crash the toggle.
                pass
        self._apply_active_theme()
        return self.active_theme

    def watch_system_appearance(
        self,
        appearance: SystemAppearance | None,
    ) -> None:
        """Re-resolve auto themes when the host appearance flips.

        Concrete light/dark themes don't re-resolve (the user
        explicitly picked one); only `auto`-preference users
        track the host signal.

        Args:
            appearance: New appearance value (passed by the
                Textual reactive system; the watcher itself
                reads via `self.system_appearance`).
        """
        _ = appearance
        if self.theme_pref.theme_name in {t.name for t in list_themes() if t.is_auto}:
            self._apply_active_theme()

    # ------------------------------------------------------------------
    # Boot screen selection
    # ------------------------------------------------------------------

    def _build_boot_screen(self) -> Any:
        """Pick the screen to push at mount time.

        Returning users who are fully set up — required
        credentials present and no interrupted run waiting to
        resume — land directly on :class:`ChatScreen`, the home
        surface, skipping the WelcomeScreen splash. Everyone else
        (first run, missing credentials, or a pending resume
        snapshot) gets :class:`WelcomeScreen`, which owns
        onboarding routing, the recents pane, and the
        resume-on-startup modal.
        """
        # Lazy imports keep the module-load cost low for the
        # `care --help` CLI path that never mounts the TUI.
        if self._can_boot_straight_to_chat():
            from care.screens.chat import ChatScreen

            return ChatScreen()
        from care.screens.welcome import WelcomeScreen

        return WelcomeScreen()

    def _can_boot_straight_to_chat(self) -> bool:
        """True when boot can land on ChatScreen without the
        WelcomeScreen splash.

        Gated on the same three signals the splash routes on so
        the direct path and the splash path can never disagree
        about whether the user is "set up":

        * returning user (a config file already existed at
          launch — i.e. not ``first_run``),
        * no required credentials missing (mirrors the post-Save
          gate in
          :func:`care.screens.welcome._missing_required_creds`),
        * no interrupted-run snapshot pending — that prompt lives
          on WelcomeScreen, so a stored snapshot keeps the
          splash.
        """
        if self._initial_mode != "returning":
            return False
        from care.screens.welcome import _missing_required_creds

        try:
            if _missing_required_creds(self.config):
                return False
        except Exception:
            return False
        try:
            from care.runtime.run_state import RunStateStore

            if RunStateStore().load() is not None:
                return False
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------
    # Global action methods (wired from BINDINGS)
    # ------------------------------------------------------------------

    def _post_to_screen(self, message: Message) -> None:
        """Post ``message`` to the active screen so it bubbles
        through the screen's handler chain before reaching the
        app. The message routes to the active screen's
        ``on_care_app_<name>`` handler (Textual auto-resolves
        the handler name from the nested class qualname).
        """
        screen = self.screen if self.screen_stack else None
        if screen is not None:
            screen.post_message(message)
        else:
            self.post_message(message)

    def action_global_open_command_palette(self) -> None:
        """Ctrl+P → push :class:`CommandPaletteModal` + post
        :class:`CommandPaletteRequested` on the active screen.

        The screen-level message is kept for legacy listeners
        and tests that subscribe to it; the new behaviour is
        for the app itself to push the modal and route the
        dismissed selection via :meth:`_dispatch_palette_action`.
        """
        self.global_action_log.append("open_command_palette")
        self._post_to_screen(self.CommandPaletteRequested())
        try:
            from care.screens.command_palette import CommandPaletteModal

            self.push_screen(
                CommandPaletteModal(memory=self.memory),
                self._dispatch_palette_action,
            )
        except Exception:
            pass

    def _dispatch_palette_action(self, selection: Any) -> None:
        """Route a dismissed :class:`PaletteSelection`.

        Three dispatch paths:

        * ``command`` entries with a known ``command_action``
          route through :data:`_PALETTE_ACTION_DISPATCH` to the
          matching ``action_*`` method.
        * ``chain`` entries push :class:`InspectionScreen`
          seeded with the picked chain's ``entry_id``.
        * ``agent_skill`` entries push :class:`CatalogScreen`
          (no per-skill detail flow yet — landing on the
          catalog gives the user the manifest + Promote
          action).
        * Anything else is a silent no-op.

        Defensive: ``None`` selection, missing entry, unknown
        kind, handler exception — all fall through to a no-op
        rather than crashing the stack on a stale palette entry.
        """
        if selection is None:
            return
        entry = getattr(selection, "entry", None)
        if entry is None:
            return
        kind = getattr(entry, "kind", None)
        if kind == "command":
            action_id = getattr(entry, "command_action", None)
            handler = _PALETTE_ACTION_DISPATCH.get(action_id)
            if handler is None:
                return
            try:
                handler(self)
            except Exception:
                pass
            return
        if kind == "chain":
            entity_id = getattr(entry, "entry_id", None)
            if not entity_id:
                return
            from care.screens.inspection import InspectionScreen

            try:
                self.push_screen(InspectionScreen(entity_id))
            except Exception:
                pass
            return
        if kind == "agent_skill":
            entity_id = getattr(entry, "entry_id", None) or None
            self._open_catalog_focused(focus_entry_id=entity_id)
            return

    def action_global_quit(self) -> None:
        """Ctrl+Q → exit the application."""
        self.global_action_log.append("quit")
        self.exit()

    def action_global_save_artifact(self) -> None:
        """Ctrl+S → posts :class:`SaveRequested` on the active
        screen.

        The active screen interprets "save" against the
        artifact it's showing — InspectionScreen saves to
        Memory via the Edit / Save modals, GenerationScreen
        opens the SaveAgentModal, etc.
        """
        self.global_action_log.append("save_artifact")
        self._post_to_screen(self.SaveRequested())

    def action_global_rerun_artifact(self) -> None:
        """Ctrl+R → posts :class:`RerunRequested` on the active
        screen.

        Active screen re-runs whatever artifact is current
        (chain on InspectionScreen, evolution on
        EvolutionScreen, etc.).
        """
        self.global_action_log.append("rerun_artifact")
        self._post_to_screen(self.RerunRequested())

    def action_global_back(self) -> None:
        """Esc → pop the current screen.

        Pops when the screen stack has more than one entry;
        otherwise posts :class:`BackRequested` on the active
        screen so the top-level screen can implement its own
        back behaviour (e.g. cancel-and-return-to-welcome on
        the LibraryScreen).
        """
        self.global_action_log.append("back")
        if len(self.screen_stack) > 1:
            self.pop_screen()
        else:
            self._post_to_screen(self.BackRequested())


def run() -> None:
    """Console-script entry point."""
    CareApp().run()
