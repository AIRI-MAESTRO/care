"""StatusBar widget (TODO §1 P1 / sub-tasks P1.1 – P1.5).

Mounted at the bottom of `CareApp` above the Footer. Consumes
:func:`care.runtime.status_bar.aggregate_status_bar` to render
a single-line strip with Memory + Platform health, the active
model, session token totals, and the in-flight task summary.

P1.1 ships the scaffold + first-paint placeholder.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static

from care.runtime.i18n import t
from care.runtime.hint_fit import fit_line
from care.runtime.status_bar import (
    StatusBarSnapshot,
    aggregate_status_bar,
    derive_from_task_registry,
)


class StatusBar(Widget):
    """One-line status strip mounted above CareFooter.

    Construct without args — the widget reads `app.config`,
    `app.memory`, `app.platform`, `app.token_counter`, and
    `app.task_registry` directly. Tests that need to drive
    the rendering inject stubs via constructor kwargs
    (`config=`, `memory=`, …) so the widget never has to
    reach into the host app.
    """

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
        layer: status;
    }
    StatusBar Static {
        height: 1;
        content-align: left middle;
    }
    StatusBar #status-bar-text {
        width: 1fr;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    """

    PLACEHOLDER_TEXT = "memory ? · platform ? · …"
    """English source-of-truth for the first-paint placeholder. The
    rendered strip resolves :func:`t` at use-site
    (:meth:`_placeholder_text`) so the live UI is localized; this
    class constant stays the canonical English value for tests."""

    DEFAULT_REFRESH_INTERVAL_SECONDS = 5.0
    """How often the StatusBar re-runs the aggregator. 5s
    matches the §1 P1 spec — short enough to feel live during
    a run, long enough that the probes stay cheap on the
    server."""

    def __init__(
        self,
        *,
        config: Any = None,
        memory: Any = None,
        platform: Any = None,
        token_counter: Any = None,
        task_registry: Any = None,
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL_SECONDS,
    ) -> None:
        super().__init__()
        # Optional explicit overrides — tests pass these; the
        # production constructor leaves them all None so the
        # widget reads from `self.app.*` at refresh time.
        self._explicit_config = config
        self._explicit_memory = memory
        self._explicit_platform = platform
        self._explicit_token_counter = token_counter
        self._explicit_task_registry = task_registry
        self.refresh_interval = refresh_interval
        # Last rendered snapshot — exposed for tests + the
        # future telemetry sink.
        self.last_snapshot: StatusBarSnapshot | None = None
        self.last_error: str | None = None
        self.is_loading: bool = True
        # Counter of completed refreshes — tests + future
        # telemetry can observe how often the timer fired
        # without scraping widget internals.
        self.refresh_count: int = 0
        # Subscription unsubscribe handle (P1.4 wires
        # `task_registry.on_change` to refresh on transitions).
        self._unsubscribe: Any = None
        # Interval timer handle so on_unmount can stop it.
        self._interval_timer: Any = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def _placeholder_text(self) -> str:
        """Resolve the placeholder via :func:`t` at access time so a
        language change repaints the localized strip on the next refresh."""
        return t("statusBar.placeholder")

    def compose(self) -> ComposeResult:
        with Horizontal(id="status-bar-row"):
            # markup=False: the strip is plain text (coloring is via CSS,
            # not Rich tags) and the error path renders raw exception
            # strings — e.g. a chain Pydantic ValidationError full of
            # `[...]` — which would otherwise raise MarkupError mid-render.
            yield Static(
                self._placeholder_text(),
                id="status-bar-text",
                markup=False,
            )

    def on_mount(self) -> None:
        # Kick the first refresh — the widget shows the
        # placeholder until the worker settles.
        self.refresh_snapshot()
        # P1.2: schedule recurring refreshes. `set_interval`
        # divides by interval internally so a non-positive
        # value would crash the timer loop — skip the timer
        # entirely (tests can pass `refresh_interval=0` to
        # disable auto-refresh for deterministic timing).
        if self.refresh_interval > 0:
            self._interval_timer = self.set_interval(
                self.refresh_interval,
                self.refresh_snapshot,
            )
        # P1.4: subscribe to TaskRegistry changes so a run
        # start / completion / cancel refreshes the strip
        # immediately instead of waiting for the next 5s tick.
        registry = self._resolve(
            "_explicit_task_registry", "task_registry",
        )
        if registry is not None and hasattr(registry, "on_change"):
            try:
                self._unsubscribe = registry.on_change(
                    self._on_task_registry_event,
                )
            except Exception:
                self._unsubscribe = None

    def on_unmount(self) -> None:
        timer = self._interval_timer
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._interval_timer = None
        unsub = self._unsubscribe
        if callable(unsub):
            try:
                unsub()
            except Exception:
                pass
        self._unsubscribe = None

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_snapshot(self) -> None:
        """Schedule a fresh aggregator pass. Idempotent — a
        second call while the previous worker is still in
        flight cancels the old one via `exclusive=True`."""
        self.run_worker(
            self._refresh(),
            name="status_bar_refresh",
            group="status_bar",
            exclusive=True,
            exit_on_error=False,
        )

    async def _refresh(self) -> None:
        self.is_loading = True
        try:
            config = self._resolve("_explicit_config", "config")
            memory = self._resolve("_explicit_memory", "memory")
            platform = self._resolve("_explicit_platform", "platform")
            token_counter = self._resolve(
                "_explicit_token_counter", "token_counter",
            )
            registry = self._resolve(
                "_explicit_task_registry", "task_registry",
            )
            active_task = None
            if registry is not None:
                try:
                    active_task = derive_from_task_registry(registry)
                except Exception:
                    active_task = None
            snapshot = await aggregate_status_bar(
                config=config,
                memory=memory,
                platform=platform,
                token_counter=token_counter,
                active_task=active_task,
            )
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.is_loading = False
            # P1.5: surface the aggregator-level error inline so
            # the user sees what went wrong instead of the
            # generic placeholder. The placeholder is reserved
            # for the loading state only.
            self._render_text(self._format_error_text())
            return
        self.last_snapshot = snapshot
        self.last_error = None
        self.is_loading = False
        self.refresh_count += 1
        self._render_text(snapshot.format_text())

    def _on_task_registry_event(self, _event_kind, _record) -> None:
        """Listener — fires from worker threads. Hop back to
        the Textual loop before scheduling the refresh worker
        so the registry's lock-free notify can't race with the
        widget's mount state."""
        try:
            self.app.call_from_thread(self.refresh_snapshot)
        except Exception:
            # Mid-unmount or no live app — best-effort.
            try:
                self.refresh_snapshot()
            except Exception:
                pass

    def _resolve(self, explicit_attr: str, app_attr: str) -> Any:
        """Pick the explicit override when present; else read
        from the host app. ``None`` propagates through so the
        aggregator can render unconfigured slots cleanly."""
        explicit = getattr(self, explicit_attr, None)
        if explicit is not None:
            return explicit
        return getattr(self.app, app_attr, None)

    def _format_error_text(self) -> str:
        """Render the aggregator-level error string when the
        whole call raised (separate from per-probe failures,
        which `StatusBarSnapshot.format_text` already stamps
        onto the strip)."""
        if not self.last_error:
            return self._placeholder_text()
        return t("statusBar.error", err=self.last_error)

    def _render_text(self, text: str) -> None:
        if not self.is_mounted:
            return
        try:
            target = self.query_one("#status-bar-text", Static)
        except Exception:
            return
        try:
            width = int(self.size.width) or int(self.app.size.width)
        except Exception:
            width = 0
        target.update(fit_line(text, max(0, width - 2)))


__all__ = ["StatusBar"]
