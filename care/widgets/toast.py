"""Toast / notification host (TODO §1.1 P0.35).

Mounted at the app level by :class:`care.app.CareApp`. Every
screen calls :meth:`CareApp.push_toast(message, severity=...)`;
the host widget appends a toast row and auto-dismisses it
after the configured TTL (default 3s).

Severity drives the badge / styling:

* ``"info"`` (default) — `·` neutral marker.
* ``"success"`` — `✓` accent colour.
* ``"warning"`` — `⚠` warning colour.
* ``"error"`` — `✗` error colour.

The widget is used by every shipped data layer's failure
path: `LibraryViewError`, `LineageError`, `RunHistoryError`,
`AgentDiffError`, `LibraryBundleError`, etc. The screen
catches the typed error and forwards the message to
`app.push_toast(str(exc), severity="error")`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static


ToastSeverity = Literal["info", "success", "warning", "error"]


_BADGES: dict[ToastSeverity, str] = {
    "info": "·",
    "success": "✓",
    "warning": "⚠",
    "error": "✗",
}


@dataclass(frozen=True)
class Toast:
    """Frozen toast snapshot — tests + future telemetry read
    these without scraping the widget tree."""

    message: str
    severity: ToastSeverity = "info"


class ToastHost(Widget):
    """Vertical stack of auto-dismissing toast rows.

    Mount once at the app / top-screen level. Call
    :meth:`push` (or :meth:`CareApp.push_toast` which routes
    here) to append a row.
    """

    DEFAULT_CSS = """
    ToastHost {
        dock: bottom;
        height: auto;
        max-height: 8;
        layer: notifications;
        padding: 0 1;
        background: $surface 0%;
    }
    ToastHost VerticalScroll {
        height: auto;
        max-height: 6;
    }
    ToastHost .toast {
        height: auto;
        padding: 0 1;
        margin-top: 1;
        border: round $primary 30%;
        background: $surface;
    }
    ToastHost .toast.severity-error {
        border: round $error 60%;
    }
    ToastHost .toast.severity-warning {
        border: round $warning 60%;
    }
    ToastHost .toast.severity-success {
        border: round $success 60%;
    }
    """

    DEFAULT_TTL_SECONDS = 3.0

    def __init__(
        self,
        *,
        default_ttl: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        super().__init__()
        self.default_ttl = default_ttl
        # History of toasts pushed — exposed for tests +
        # future telemetry. Bounded to the last 32 entries to
        # avoid unbounded growth.
        self.history: list[Toast] = []
        # Monotonic counter for unique row ids.
        self._counter = 0

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="toast-rows")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        message: str,
        *,
        severity: ToastSeverity = "info",
        ttl: float | None = None,
    ) -> Toast:
        """Append a new toast + schedule auto-dismiss.

        Args:
            message: The user-facing text. Empty is allowed
                (renders a bare badge) but unusual.
            severity: Drives the badge + border colour.
            ttl: Per-toast override; ``None`` uses
                :attr:`default_ttl`.

        Returns:
            The frozen :class:`Toast` snapshot the row was
            built from.
        """
        toast = Toast(message=message, severity=severity)
        self.history.append(toast)
        if len(self.history) > 32:
            self.history = self.history[-32:]
        self._counter += 1
        row_id = f"toast-row-{self._counter}"
        wait_for = self.default_ttl if ttl is None else ttl
        try:
            container = self.query_one("#toast-rows", VerticalScroll)
        except Exception:
            return toast
        # markup=False: a toast renders raw, untrusted strings — e.g. a
        # chain's Pydantic ValidationError full of `[...]` — and the
        # severity colouring comes from CSS classes, not Rich tags. With
        # markup on, a bracketed exception string raises MarkupError when
        # the (possibly long-lived, ttl=0) row is re-rendered, including
        # during the app's shutdown full-arrange pass.
        row = Static(
            self._format_row(toast),
            id=row_id,
            classes=f"toast severity-{severity}",
            markup=False,
        )
        container.mount(row)
        # Entrance fade — only when motion is enabled; reduced-motion keeps
        # the row fully visible from mount so headless pilots see settled
        # state.
        if self._motion_enabled():
            try:
                row.styles.opacity = 0.0
                row.styles.animate(
                    "opacity", value=1.0, duration=0.25, easing="out_cubic",
                )
            except Exception:
                row.styles.opacity = 1.0
        if wait_for > 0:
            self.set_timer(wait_for, lambda: self._dismiss(row_id))
        return toast

    def _dismiss(self, row_id: str) -> None:
        try:
            row = self.query_one(f"#{row_id}", Static)
        except Exception:
            return
        # Motion enabled: fade out, then remove via the animation's
        # on_complete. Reduced-motion: remove immediately so tests see the
        # row gone right away.
        if self._motion_enabled():
            try:
                row.styles.animate(
                    "opacity", value=0.0, duration=0.2, on_complete=row.remove,
                )
                return
            except Exception:
                pass
        try:
            row.remove()
        except Exception:
            pass

    def _motion_enabled(self) -> bool:
        """True when the app permits entrance/exit motion. Reduced-motion
        (``animation_level == "none"``) → False so fades become no-ops."""
        try:
            return getattr(self.app, "animation_level", "none") != "none"
        except Exception:
            return False

    @staticmethod
    def _format_row(toast: Toast) -> str:
        badge = _BADGES.get(toast.severity, "·")
        return f"{badge} {toast.message}" if toast.message else badge


__all__ = ["Toast", "ToastHost", "ToastSeverity"]
