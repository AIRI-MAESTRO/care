"""Header widget (TODO §1.1 P0.3).

Mounts above every screen and renders the
:class:`care.runtime.HeaderModel` projection: app title +
breadcrumb + version. The widget rebuilds its content via the
shipped `build_header(...)` factory whenever the host screen
calls :meth:`refresh_from_app` — typically once on mount and
once per screen-stack push/pop in the wrapping `CareApp`.

Three flat regions inside a `Horizontal` container so the
title sits left, the breadcrumb fills the centre, and the
version pins right:

    ┌──────────────────────────────────────────────────┐
    │ CARE      Library › Weather report     v0.3.0   │
    └──────────────────────────────────────────────────┘

The widget reads `app.config.version` and the breadcrumb /
active-screen attributes that screens expose; defaults
gracefully when those attributes are absent so a test host
without full state still renders something readable.
"""

from __future__ import annotations

from typing import Iterable

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from care.runtime.global_bindings import HeaderModel, build_header
from care.runtime.i18n import t


class CareHeader(Horizontal):
    """`Horizontal` row showing title, breadcrumb, and version.

    Constructed without args — host screens / apps call
    :meth:`refresh_from_app` (or :meth:`set_model`) after
    mount to push fresh data in. Reactive enough for the
    `CareApp.watch_current_screen` hook (P0.1 wiring) to
    trigger a refresh on every screen transition.
    """

    @property
    def BACK_HINT_TEXT(self) -> str:
        """Hint shown next to the breadcrumb on every screen except Chat.
        Esc is the global ``back`` binding (dismiss / pop one level), so
        this is how the user climbs back toward the Chat surface. Resolved
        via :func:`t` at access time so a language change repaints it on the
        next compose."""
        return t("header.backHint")

    DEFAULT_CSS = """
    CareHeader {
        height: 1;
        background: $primary;
        color: $background;
    }
    CareHeader #header-title {
        width: auto;
        padding: 0 1;
        text-style: bold;
    }
    CareHeader #header-breadcrumb {
        width: 1fr;
        content-align: center middle;
    }
    CareHeader #header-back-hint {
        width: auto;
        padding: 0 1;
        text-style: dim;
        /* Collapsed by default; shown for every screen except Chat
           via :meth:`_apply_back_hint` so the user always sees how
           to leave a pushed screen. */
        display: none;
    }
    CareHeader #header-library-btn {
        width: auto;
        padding: 0 1;
        text-style: underline;
        /* Quick-access "My chains" link. Hidden by default (collapsed to
           0 width); shown only on the Chat surface via
           :meth:`set_library_button`. */
        display: none;
    }
    CareHeader #header-library-btn:hover {
        background: $accent;
        color: $background;
        text-style: bold;
    }
    /* Evolution + Help links — same underlined-link style as the library
       link / artifact pill. Shown only on the Chat surface. */
    CareHeader #header-evolution-btn {
        width: auto;
        padding: 0 1;
        text-style: underline;
        display: none;
    }
    CareHeader #header-evolution-btn:hover {
        background: $accent;
        color: $background;
        text-style: bold;
    }
    CareHeader #header-help-btn {
        width: auto;
        padding: 0 1;
        text-style: underline;
        display: none;
    }
    CareHeader #header-help-btn:hover {
        background: $accent;
        color: $background;
        text-style: bold;
    }
    CareHeader #header-badge {
        width: auto;
        padding: 0 1;
        background: $accent;
        color: $background;
        text-style: bold;
        /* `display: none` collapses width to 0 so empty badges
           don't reserve real estate. Toggled at runtime by
           :meth:`set_badge`. */
        display: none;
    }
    CareHeader #header-artifact-pill {
        /* Styled identically to the quick-access library link: an
           underlined text link (no solid pill), accent on hover. */
        width: auto;
        padding: 0 1;
        text-style: underline;
        /* Same display-collapse pattern as the mode badge.
           Toggled by :meth:`set_artifact_pill`. */
        display: none;
    }
    CareHeader #header-artifact-pill:hover {
        background: $accent;
        color: $background;
        text-style: bold;
    }
    CareHeader #header-version {
        width: auto;
        padding: 0 1;
        text-align: right;
        text-style: dim;
    }
    """

    def __init__(
        self,
        model: HeaderModel | None = None,
        *,
        badge: str = "",
        artifact_pill: str = "",
        library_button: bool = False,
    ) -> None:
        super().__init__()
        self._model: HeaderModel = model if model is not None else HeaderModel()
        self._badge: str = badge
        self._artifact_pill: str = artifact_pill
        self._show_library_button: bool = library_button
        self._show_evolution_button: bool = False
        self._show_help_button: bool = False

    def compose(self) -> ComposeResult:
        yield Static(self._model.title, id="header-title")
        yield Static(self._model.breadcrumb_text, id="header-breadcrumb")
        yield Static(self.BACK_HINT_TEXT, id="header-back-hint")
        # Chat nav cluster: "My chains" + "Evolution" sit LEFT of the
        # Artifacts pill; "Help" sits at the end of the cluster.
        yield Static(t("header.library"), id="header-library-btn")
        yield Static(t("header.evolution"), id="header-evolution-btn")
        yield Static(
            self._artifact_pill, id="header-artifact-pill",
        )
        yield Static(t("header.help"), id="header-help-btn")
        yield Static(self._badge, id="header-badge")
        yield Static(self._model.version, id="header-version")

    def on_mount(self) -> None:
        # Compose runs before mount completes, so the initial
        # `display` on the badge is the CSS default (none). Sync
        # to the constructor value here so screens that pass an
        # explicit badge see it on first paint.
        self._apply_badge_display()
        self._apply_artifact_pill_display()
        self._apply_library_button_display()
        self._apply_evolution_button_display()
        self._apply_help_button_display()
        self._apply_back_hint()
        self.relocalize()

    # ------------------------------------------------------------------
    # Refresh hooks
    # ------------------------------------------------------------------

    def set_model(self, model: HeaderModel) -> None:
        """Replace the current model + repaint the three
        Statics. Called by host code on every screen
        transition."""
        self._model = model
        # `query_one` raises if mount hasn't completed yet; the
        # caller is expected to drive `refresh_from_app` /
        # `set_model` AFTER `on_mount`. Skip the paint until
        # then.
        if not self.is_mounted:
            return
        self.query_one("#header-title", Static).update(model.title)
        self.query_one("#header-breadcrumb", Static).update(model.breadcrumb_text)
        self.query_one("#header-version", Static).update(model.version)
        self._apply_back_hint()
        self.relocalize()

    def relocalize(self) -> None:
        """Refresh strings baked in at :meth:`compose` time.

        ``t()`` resolves at call time, but Static widgets keep
        their mounted text until explicitly updated — call this
        after a UI-language switch.
        """
        if not self.is_mounted:
            return
        try:
            btn = self.query_one("#header-library-btn", Static)
            btn.update(t("header.library"))
            btn.tooltip = t("header.libraryTip")
        except Exception:
            pass
        try:
            ev = self.query_one("#header-evolution-btn", Static)
            ev.update(t("header.evolution"))
            ev.tooltip = t("header.evolutionTip")
        except Exception:
            pass
        try:
            hb = self.query_one("#header-help-btn", Static)
            hb.update(t("header.help"))
            hb.tooltip = t("header.helpTip")
        except Exception:
            pass
        try:
            self.query_one("#header-back-hint", Static).update(self.BACK_HINT_TEXT)
        except Exception:
            pass

    def refresh_from_app(
        self,
        *,
        active_screen: str = "",
        breadcrumb: Iterable[str] = (),
        version: str | None = None,
        title: str = "MAESTRO",
    ) -> None:
        """Build a fresh :class:`HeaderModel` via the shipped
        :func:`build_header` factory and apply it.

        Args:
            active_screen: Class name of the screen currently
                on top of the stack (e.g.
                ``"InspectionScreen"``).
            breadcrumb: Iterable of breadcrumb segments. Empty
                strings are filtered out by `build_header`.
            version: App version. ``None`` reads
                ``app.config`` or falls back to empty.
            title: App title (rarely changes; exposed for
                tests + branding overrides).
        """
        if version is None:
            version = self._read_version_from_app()
        model = build_header(
            active_screen=active_screen,
            breadcrumb=breadcrumb,
            version=version,
            title=title,
        )
        self.set_model(model)

    @property
    def model(self) -> HeaderModel:
        """Current header model — read-only snapshot for tests."""
        return self._model

    @property
    def badge(self) -> str:
        """Current badge text — read-only snapshot for tests."""
        return self._badge

    def set_badge(self, text: str) -> None:
        """Replace the right-side badge (e.g. ``"AD-HOC"`` /
        ``"PROD"``). Pass an empty string to hide it.

        Stays a no-op when not yet mounted; the value is stashed
        and re-applied from :meth:`on_mount`.
        """
        self._badge = text or ""
        if not self.is_mounted:
            return
        try:
            badge_widget = self.query_one("#header-badge", Static)
        except Exception:
            return
        badge_widget.update(self._badge)
        self._apply_badge_display()

    @property
    def library_button_visible(self) -> bool:
        """Whether the quick-access Library link is shown — read-only
        snapshot for tests."""
        return self._show_library_button

    def set_library_button(self, visible: bool) -> None:
        """Show / hide the header's Library quick-access link.

        Only the Chat surface turns it on; pushed screens leave it
        hidden (the breadcrumb back-hint covers navigation there).
        Stays a no-op when not yet mounted; the value is stashed and
        re-applied from :meth:`on_mount`.
        """
        self._show_library_button = bool(visible)
        if not self.is_mounted:
            return
        self._apply_library_button_display()

    def _apply_library_button_display(self) -> None:
        try:
            btn = self.query_one("#header-library-btn", Static)
        except Exception:
            return
        if btn.display != self._show_library_button:
            btn.display = self._show_library_button

    @property
    def evolution_button_visible(self) -> bool:
        return self._show_evolution_button

    def set_evolution_button(self, visible: bool) -> None:
        """Show / hide the header's Evolution quick-access link (Chat only)."""
        self._show_evolution_button = bool(visible)
        if not self.is_mounted:
            return
        self._apply_evolution_button_display()

    def _apply_evolution_button_display(self) -> None:
        try:
            btn = self.query_one("#header-evolution-btn", Static)
        except Exception:
            return
        if btn.display != self._show_evolution_button:
            btn.display = self._show_evolution_button

    @property
    def help_button_visible(self) -> bool:
        return self._show_help_button

    def set_help_button(self, visible: bool) -> None:
        """Show / hide the header's Help link (Chat only)."""
        self._show_help_button = bool(visible)
        if not self.is_mounted:
            return
        self._apply_help_button_display()

    def _apply_help_button_display(self) -> None:
        try:
            btn = self.query_one("#header-help-btn", Static)
        except Exception:
            return
        if btn.display != self._show_help_button:
            btn.display = self._show_help_button

    def _apply_back_hint(self) -> None:
        """Show the `Esc to go back` hint for every screen except
        Chat. Driven off ``model.active_screen`` — an empty (unknown)
        screen and ``ChatScreen`` itself stay collapsed; Chat is the
        home surface, so there's nothing to go back to."""
        try:
            hint = self.query_one("#header-back-hint", Static)
        except Exception:
            return
        active = self._model.active_screen
        want_visible = bool(active) and active != "ChatScreen"
        if hint.display != want_visible:
            hint.display = want_visible

    def _apply_badge_display(self) -> None:
        try:
            badge_widget = self.query_one("#header-badge", Static)
        except Exception:
            return
        # `display=False` collapses the widget; True restores the
        # CSS rules. Avoid touching `display` when value matches
        # so we don't trigger unnecessary layout reflows.
        want_visible = bool(self._badge)
        if badge_widget.display != want_visible:
            badge_widget.display = want_visible

    @property
    def artifact_pill(self) -> str:
        """Current artifact-counts pill text — read-only
        snapshot for tests."""
        return self._artifact_pill

    def set_artifact_pill(self, text: str) -> None:
        """Replace the session-artifacts pill (e.g.
        ``"3 · 1 unsaved"``). Pass empty string to hide it.

        Mirrors :meth:`set_badge` — stashes the value when
        the widget isn't mounted yet so the constructor's
        pre-mount default also paints correctly on the first
        compose pass.
        """
        self._artifact_pill = text or ""
        if not self.is_mounted:
            return
        try:
            pill = self.query_one("#header-artifact-pill", Static)
        except Exception:
            return
        pill.update(self._artifact_pill)
        self._apply_artifact_pill_display()

    def _apply_artifact_pill_display(self) -> None:
        try:
            pill = self.query_one("#header-artifact-pill", Static)
        except Exception:
            return
        want_visible = bool(self._artifact_pill)
        if pill.display != want_visible:
            pill.display = want_visible

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_version_from_app(self) -> str:
        """Read the app's version string defensively.

        The host `App` typically exposes one of:

        * ``app.version`` — direct attribute (planned for
          `CareApp` on package metadata wiring).
        * ``app.config.version`` — pydantic config field.
        * Nothing — fall back to empty string.
        """
        app = self.app
        version = getattr(app, "version", None)
        if isinstance(version, str) and version:
            return version
        config = getattr(app, "config", None)
        if config is not None:
            cfg_version = getattr(config, "version", None)
            if isinstance(cfg_version, str) and cfg_version:
                return cfg_version
        return ""


__all__ = ["CareHeader"]
