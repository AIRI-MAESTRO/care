"""Footer widget (TODO §1.1 P0.4).

Sits at the bottom of every screen and renders key-binding
hints projected from :class:`care.runtime.FooterModel`. The
widget rebuilds via :func:`care.runtime.build_footer` whenever
the host screen calls :meth:`refresh_from_app` — typically once
on mount and once per screen-stack push/pop in the wrapping
`CareApp`.

Hint render shape — bracketed key followed by the label,
right-aligned:

    ┌──────────────────────────────────────────────────┐
    │            [Ctrl+P] Palette  [Esc] Back  [Ctrl+Q] Quit │
    └──────────────────────────────────────────────────┘

Each hint becomes its own `Static` child inside a `Horizontal`
container so individual hints can later flicker / highlight
independently (e.g. show a transient highlight on the key the
user just pressed).
"""

from __future__ import annotations

from typing import Iterable, Optional

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from care.runtime.global_bindings import (
    BindingScope,
    FooterHint,
    FooterModel,
    GlobalBinding,
    build_footer,
)


class CareFooter(Horizontal):
    """`Horizontal` row of key-binding hints.

    Constructed without args — host screens / apps call
    :meth:`refresh_from_app` after mount to push fresh data
    in. Reactive enough for the `CareApp.watch_current_screen`
    hook to trigger a refresh on every screen transition.
    """

    DEFAULT_CSS = """
    CareFooter {
        height: 1;
        background: $panel;
        color: $foreground-muted;
        overflow: hidden;
    }
    CareFooter .footer-hint {
        width: auto;
        padding: 0 1;
    }
    CareFooter #footer-spacer {
        width: 1fr;
    }
    CareFooter #footer-status {
        width: auto;
        padding: 0 1;
        color: $accent;
    }
    """

    HINT_CLASS = "footer-hint"
    """CSS class stamped on every hint `Static`. Stable so
    tests + future styling tweaks can query by class."""

    STATUS_ID = "footer-status"
    """Id of the right-most status segment (e.g. "▶ 2 running")."""

    def __init__(self, model: FooterModel | None = None) -> None:
        super().__init__()
        self._model: FooterModel = model if model is not None else FooterModel()
        self._status_text: str = ""

    def compose(self) -> ComposeResult:
        # Leading flexible spacer so hints right-align inside
        # the row.
        yield Static("", id="footer-spacer")
        for hint in self._model.hints:
            yield Static(
                self._format_hint(hint.key, hint.label),
                classes=self.HINT_CLASS,
                id=self._hint_id_for(hint.action_id),
            )
        # Right-most persistent status segment (survives recompose via
        # ``_status_text``). Empty until a host sets it.
        yield Static(self._status_text, id=self.STATUS_ID)

    def set_status(self, text: str) -> None:
        """Set the right-most footer status segment (e.g. evolutions
        running). Persists across hint recomposes. Empty hides it."""
        self._status_text = text or ""
        if not self.is_mounted:
            return
        try:
            self.query_one(f"#{self.STATUS_ID}", Static).update(self._status_text)
        except Exception:
            # Status widget not mounted yet (mid-recompose) — the next
            # compose() picks up ``_status_text``.
            pass

    @property
    def status_text(self) -> str:
        """Current status-segment text — read-only snapshot for tests."""
        return self._status_text

    # ------------------------------------------------------------------
    # Refresh hooks
    # ------------------------------------------------------------------

    def set_model(self, model: FooterModel) -> None:
        """Replace the current model + rebuild the children
        via :meth:`recompose`.

        Textual's `Widget.remove()` is async-deferred, so a
        naive remove-then-mount pattern collides on the
        hint IDs (the old Static is still in the tree when
        the new one mounts). `recompose()` is async; we
        schedule it on the app's event loop via
        `call_later` so callers can stay synchronous.
        """
        self._model = model
        if not self.is_mounted:
            return
        # `recompose()` returns a coroutine; schedule it on
        # the app loop. The Pilot tests `await pilot.pause()`
        # to let the scheduled work flush.
        self.app.call_later(self.recompose)

    def refresh_from_app(
        self,
        *,
        active_screen: str = "",
        scope: BindingScope = "screen",
        registry: Optional[Iterable[GlobalBinding]] = None,
    ) -> None:
        """Build a fresh :class:`FooterModel` via the shipped
        :func:`build_footer` factory and apply it.

        Args:
            active_screen: Class name of the screen currently
                on top of the stack. Stamped onto the model
                for the future conditional-visibility logic.
            scope: ``"screen"`` (default) or ``"modal"``. The
                shipped registry hides screen-only hints
                (`Save` / `Re-run`) on modals automatically.
            registry: Custom binding registry. ``None`` uses
                :func:`default_global_bindings`.
        """
        model = build_footer(
            active_screen=active_screen,
            scope=scope,
            registry=registry,
        )
        self.set_model(model)

    def fit_to_width(self, width: int) -> None:
        """Show as many right-aligned hints as fit; drop leftmost first."""
        if not self.is_mounted:
            return
        hints = self._model.hints
        visible: tuple[FooterHint, ...] = hints
        if width > 0 and hints:
            kept: list[FooterHint] = list(hints)
            while kept:
                total = sum(
                    len(self._format_hint(h.key, h.label)) + 2
                    for h in kept
                )
                if total <= width:
                    break
                kept.pop(0)
            visible = tuple(kept)
        visible_ids = {self._hint_id_for(h.action_id) for h in visible}
        for child in self.query(f".{self.HINT_CLASS}"):
            child.display = child.id in visible_ids

    @property
    def model(self) -> FooterModel:
        """Current footer model — read-only snapshot for tests."""
        return self._model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_hint(key: str, label: str) -> str:
        """Pretty-print one hint as ``[Key] Label``. Matches
        the convention every CARE binding hint uses."""
        return f"[{key}] {label}"

    @staticmethod
    def _hint_id_for(action_id: str) -> str:
        """Stable id per hint so tests can query
        ``#footer-hint-open_command_palette``."""
        return f"footer-hint-{action_id}"


__all__ = ["CareFooter"]
