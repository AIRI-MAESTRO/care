"""WelcomeScreen — boot splash + mode-based routing (TODO §1.1 P0.2).

The first screen `CareApp` mounts. Renders a brief branded
splash plus a "Recents" sidebar (P0.34) showing the five
most-recently-run agents for one-click return, then routes to
the mode-appropriate target:

* ``first_run`` → :class:`SettingsScreen` (P0.32 — shipped).
* ``returning`` → :class:`LibraryScreen` (P0.7 — shipped).

The 200ms splash is a visual transition the CLI experience
expects; tests override `splash_seconds=0.0` to skip the delay.
The routing target is overridable via the ``next_screen_factory``
constructor kwarg so tests can inject a stub destination without
rewriting this module.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import ListItem, ListView, Static

from care.runtime.i18n import t
from care.runtime.library_view import (
    LibraryRow,
    LibrarySort,
    fetch_library_view,
)
from care.runtime.run_state import RunState, RunStateStore


class WelcomeScreen(Screen):
    """Boot splash that auto-routes to the next screen.

    Constructor kwargs are all optional — production callers
    pass ``WelcomeScreen()`` and the screen reads
    ``app.mode`` on mount. Tests pass ``splash_seconds=0.0`` +
    ``next_screen_factory=fake_factory`` to assert routing
    branches without timing out.
    """

    DEFAULT_SPLASH_SECONDS = 0.2
    DEFAULT_RECENTS_LIMIT = 5

    DEFAULT_CSS = """
    WelcomeScreen #welcome-body {
        height: 1fr;
    }
    WelcomeScreen #welcome-splash {
        width: 2fr;
        padding: 2 4;
    }
    WelcomeScreen #welcome-recents {
        width: 1fr;
        padding: 1 2;
        border-left: solid $primary 30%;
    }
    WelcomeScreen #welcome-recents .pane-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    """

    class RecentSelected(Message):
        """Posted when the user picks one of the recents
        entries. The host app routes the gesture through
        :func:`care.runtime.load_run_plan` and pushes the
        ExecutionScreen."""

        def __init__(self, row: LibraryRow) -> None:
            super().__init__()
            self.row = row

    def __init__(
        self,
        *,
        splash_seconds: float | None = None,
        next_screen_factory: Optional[Callable[[str], Any]] = None,
        recents_limit: int = DEFAULT_RECENTS_LIMIT,
        run_state_store: RunStateStore | None = None,
    ) -> None:
        super().__init__()
        self.splash_seconds: float = (
            splash_seconds
            if splash_seconds is not None
            else self.DEFAULT_SPLASH_SECONDS
        )
        self._next_screen_factory = next_screen_factory
        self._routed = False
        # Records the routed mode for tests + telemetry.
        self.routed_to_mode: str | None = None
        # Recents pane state (P0.34) — populated by an async
        # worker, consumed by tests + the future telemetry
        # sink.
        self.recents_limit = recents_limit
        self.recents: tuple[LibraryRow, ...] = ()
        # Resume-on-startup state (P0.37). The store is held so
        # the modal + tests share the same backing path.
        self._run_state_store: RunStateStore = (
            run_state_store
            if run_state_store is not None
            else RunStateStore()
        )
        # Last `ResumeModal` envelope action ("resume" /
        # "discard" / "cancel" / None). Exposed for tests.
        self.resume_action: str | None = None
        self.resume_state: RunState | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="welcome-body"):
            with Vertical(id="welcome-splash"):
                yield Static("MAESTRO", id="welcome-title")
                yield Static(
                    "Collaborative Agent Reasoning Ecosystem",
                    id="welcome-subtitle",
                )
            with Vertical(id="welcome-recents"):
                yield Static(t("welcome.recents"), classes="pane-title")
                yield ListView(id="welcome-recents-list")

    def on_mount(self) -> None:
        # Gentle fade-in of the splash content. Motion-gated: reduced-motion
        # keeps the body fully visible from mount so the splash + routing
        # behaviour is unchanged for tests.
        if self._motion_enabled():
            try:
                body = self.query_one("#welcome-body")
                body.styles.opacity = 0.0
                body.styles.animate(
                    "opacity", value=1.0, duration=0.25, easing="out_cubic",
                )
            except Exception:
                pass
        # P0.37: surface the resume prompt FIRST when a stored
        # snapshot exists — the user makes their decision before
        # the auto-route fires. When no snapshot exists this is a
        # no-op and the normal flow continues.
        try:
            stored = self._run_state_store.load()
        except Exception:
            stored = None
        if stored is not None:
            self.resume_state = stored
            self._push_resume_modal(stored)
        # P0.34: kick the recents fetch BEFORE the route timer so
        # the user can see + click an entry while the splash is
        # visible. The recents pane stays mounted across the
        # auto-route gesture only for tests with
        # splash_seconds=0; production splashes drain the worker
        # via the natural event loop.
        if getattr(self.app, "memory", None) is not None:
            self.run_worker(
                self._load_recents(),
                name="welcome_recents",
                group="welcome",
                exclusive=True,
                exit_on_error=False,
            )
        # Textual's `set_timer` divides by interval internally
        # so a zero / negative splash crashes the timer loop —
        # use `call_later` to route on the next event-loop
        # tick instead. Positive splashes go through the timer
        # so the screen stays visible for the requested
        # duration.
        if self.splash_seconds <= 0:
            self.call_later(self._route)
        else:
            self.set_timer(self.splash_seconds, self._route)

    def _push_resume_modal(self, stored: RunState) -> None:
        from care.screens.resume import ResumeModal, ResumeResult

        def _on_dismiss(result: ResumeResult | None) -> None:
            if result is None:
                self.resume_action = "cancel"
                return
            self.resume_action = result.action
            if result.action == "resume":
                # Re-emit on the screen's message queue so the
                # host app routes via its own handler. We don't
                # try to dispatch here — the modal is a thin
                # signal source.
                self.post_message(
                    self.ResumeRequested(result.state),
                )

        try:
            self.app.push_screen(
                ResumeModal(stored, store=self._run_state_store),
                _on_dismiss,
            )
        except Exception:
            pass

    class ResumeRequested(Message):
        """Posted when the user picks `Resume` from the
        :class:`ResumeModal`. The host app reads `state` and
        re-primes the appropriate runtime (executor /
        generator)."""

        def __init__(self, state: RunState | None) -> None:
            super().__init__()
            self.state = state

    # ------------------------------------------------------------------
    # Recents (P0.34)
    # ------------------------------------------------------------------

    async def _load_recents(self) -> None:
        memory = getattr(self.app, "memory", None)
        if memory is None:
            return
        try:
            view = await fetch_library_view(
                memory,
                sort=LibrarySort(
                    field="last_run_at",
                    direction="desc",
                ),
            )
        except Exception:
            return
        self.recents = tuple(view.rows[: self.recents_limit])
        self._render_recents()

    def _render_recents(self) -> None:
        try:
            listview = self.query_one(
                "#welcome-recents-list", ListView,
            )
        except Exception:
            return
        try:
            listview.clear()
        except Exception:
            pass
        if not self.recents:
            listview.append(ListItem(Static(t("welcome.noRuns"))))
            return
        for row in self.recents:
            listview.append(
                ListItem(
                    Static(self._format_recent_row(row)),
                    id=f"welcome-recent-{_slug(row.entity_id)}",
                )
            )

    @staticmethod
    def _format_recent_row(row: LibraryRow) -> str:
        when = (
            row.last_run_at.strftime("%Y-%m-%d %H:%M")
            if row.last_run_at is not None
            else "—"
        )
        return f"{row.label}  ·  {when}"

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "welcome-recents-list":
            return
        item = event.item
        if item is None or item.id is None:
            return
        if not item.id.startswith("welcome-recent-"):
            return
        slug = item.id[len("welcome-recent-") :]
        for row in self.recents:
            if _slug(row.entity_id) == slug:
                self.post_message(self.RecentSelected(row))
                return

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self) -> None:
        """Switch to the mode-appropriate target screen.

        Guarded against re-entry — a duplicate timer fire (e.g.
        if a test pulses the screen) won't push two screens.
        """
        if self._routed:
            return
        self._routed = True
        # Optional fade-out of the outgoing splash before the switch. Purely
        # cosmetic + motion-gated; the route itself fires unconditionally so
        # the timing / fast-path behaviour is unchanged.
        if self._motion_enabled():
            try:
                self.query_one("#welcome-body").styles.animate(
                    "opacity", value=0.0, duration=0.15,
                )
            except Exception:
                pass
        mode = self._read_app_mode()
        self.routed_to_mode = mode
        next_screen = self._build_next_screen(mode)
        self.app.switch_screen(next_screen)

    def _motion_enabled(self) -> bool:
        """True when the app permits the splash fade. Reduced-motion
        (``animation_level == "none"``) → False so the fade is a no-op and
        the content stays fully visible."""
        try:
            return getattr(self.app, "animation_level", "none") != "none"
        except Exception:
            return False

    def _read_app_mode(self) -> str:
        """Read ``app.mode``, defaulting to ``"returning"``
        when the app doesn't expose one (defensive — every
        :class:`care.app.CareApp` does, but custom hosts in
        tests may not)."""
        return str(getattr(self.app, "mode", "returning"))

    def _build_next_screen(self, mode: str) -> Any:
        """Hook for the routing target. Overridable via the
        ``next_screen_factory`` constructor kwarg + the
        module-level :func:`default_next_screen` factory."""
        if self._next_screen_factory is not None:
            return self._next_screen_factory(mode)
        return default_next_screen(mode)


def default_next_screen(mode: str) -> Any:
    """Default mode-based screen factory.

    Returning users normally land on :class:`LibraryScreen`,
    but when the on-disk config still has empty credentials
    (no MAGE / Memory api key) we route to
    :class:`SettingsScreen` first so the user is prompted to
    fill in what's missing. First-run users always land on
    :class:`SettingsScreen`. If ``CareConfig.load()`` fails
    (malformed on-disk TOML) we fall back to
    :class:`DemoScreen` so the user still sees something.
    """
    # Lazy imports keep the welcome-screen module load cheap
    # for the CLI path that never mounts the TUI.
    from care.config import CareConfig

    if mode == "returning":
        try:
            config = CareConfig.load()
        except Exception:
            from care.screens.demo import DemoScreen

            return DemoScreen()
        if _missing_required_creds(config):
            from care.screens.settings import SettingsScreen

            return SettingsScreen(config)
        from care.screens.chat import ChatScreen

        return ChatScreen()
    # `first_run` (and any unknown mode): SettingsScreen.
    try:
        from care.screens.settings import SettingsScreen

        return SettingsScreen(CareConfig.load())
    except Exception:
        from care.screens.demo import DemoScreen

        return DemoScreen()


_DEFAULT_MEMORY_URL = "http://localhost:8000"


def _missing_required_creds(config: Any) -> bool:
    """Return ``True`` when the loaded config is still missing
    something the user needs to do anything productive in the
    TUI.

    The only true hard gate is **MAGE** — without an LLM key
    the Generate flow can't do real work. Memory is treated
    as opt-in: a deployment running in anonymous mode
    (Memory's ``auth_required=False``) is reachable without
    an api_key, and CARE's ``_ensure_facades_from_config``
    builds the facade whenever ``api_key`` is set OR
    ``base_url`` has been pointed at a non-default deployment.
    We mirror that gate here so the post-Save routing doesn't
    bounce the user back to SettingsScreen when only Memory's
    base_url is customised.

    Platform stays fully optional — evolution is an advanced
    flow gated separately by ``CareApp.platform``.
    """
    try:
        if not (config.mage.api_key or "").strip():
            return True
        memory_opted_in = bool(
            (config.memory.api_key or "").strip()
        ) or (config.memory.base_url != _DEFAULT_MEMORY_URL)
        if not memory_opted_in:
            return True
    except AttributeError:
        return True
    return False


def _slug(value: str) -> str:
    """Project an entity_id into a Textual-id-compatible slug
    so the recents `ListItem` ids stay valid even when the
    underlying entity_id carries colons / slashes."""
    out = []
    for ch in value or "x":
        out.append(ch if ch.isalnum() or ch in "-_" else "-")
    return "".join(out)[:64] or "x"


__all__ = ["WelcomeScreen", "default_next_screen"]
