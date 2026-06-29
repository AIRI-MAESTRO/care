"""InspectionScreen — read-only chain view (TODO §1.1 P0.19).

Pushed when the user invokes the `Open` row action (or
arrives via Save & Inspect from `SaveAgentModal`). Renders
four panels:

1. **Step list** — selectable list of chain steps. The
   focused step drives the detail pane.
2. **Step detail** — prompt template + dependencies + config
   for the selected step.
3. **DAG view** — box-and-arrow step graph (shared with the chat
   DAG modal), falling back to an ASCII tree on odd payloads.
4. **Memory-key footer** — entity_id + version_id + channel
   so the user can correlate against GigaEvo Memory.

Action bar binds the canonical five-button row: ``Run`` /
``Edit`` / ``Evolve`` / ``Duplicate`` / ``Back to library``.
The destination screens (ExecutionScreen / EditAgentScreen /
EvolutionScreen) don't exist yet, so each action posts a
typed envelope message the host app can route once those
screens land.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Label,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from care.intermediate_artifacts import (
    IntermediateArtifactsView,
    project_intermediate_artifacts,
)
from care.runtime.i18n import t
from care.runtime.run_history import (
    RunHistoryEntry,
    RunHistoryError,
    RunHistorySummary,
    fetch_run_history,
    summarize_run_history,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


InspectionAction = Literal[
    "run", "edit", "evolve", "duplicate", "back"
]
"""Five-button action row per the §1.1 spec."""


@dataclass(frozen=True)
class InspectionPayload:
    """Projection of a single chain into the data the screen
    renders. Frozen so it flows through Textual messages
    without defensive copies."""

    entity_id: str
    channel: str = "latest"
    version_id: str = ""
    display_name: str = ""
    domain: str = ""
    description: str = ""
    steps: tuple[dict, ...] = ()

    def step_label(self, index: int) -> str:
        if index < 0 or index >= len(self.steps):
            return ""
        return _step_label(self.steps[index], index)


@dataclass
class _LoadState:
    """Mutable progress on the chain-fetch worker."""

    loading: bool = True
    error: str | None = None
    payload: InspectionPayload | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class InspectionScreen(Screen):
    """Read-only chain inspector.

    Construct with an `entity_id` (required) + optional
    `channel`. `on_mount` fires the fetch worker that calls
    ``memory.client.get_chain`` (falling back to
    ``get_chain_dict`` / ``get_chain_raw``) and populates the
    panels."""

    DEFAULT_CSS = """
    InspectionScreen {
        layout: vertical;
    }
    InspectionScreen #inspection-tabs {
        height: 1fr;
    }
    InspectionScreen #inspection-body {
        height: 1fr;
    }
    InspectionScreen #inspection-steps {
        width: 1fr;
        padding: 0 2;
        overflow-y: auto;
    }
    InspectionScreen #inspection-step-list {
        height: 1fr;
    }
    /* Step buttons mirror the chat DAG modal's `#dag-steps Button`
       styling — full-width rows the user clicks to drive the detail
       pane. The `-active` tint marks the selected step (the old
       ListView gave that highlight for free). */
    InspectionScreen .inspection-step-btn {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }
    InspectionScreen .inspection-step-btn.-active {
        background: $accent 25%;
    }
    InspectionScreen #inspection-detail {
        width: 2fr;
        padding: 0 2;
    }
    InspectionScreen #inspection-dag {
        width: 1fr;
        padding: 0 2;
        overflow-y: auto;
    }
    InspectionScreen .pane-title {
        text-style: bold;
        color: $accent;
    }
    InspectionScreen .step-detail-header {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    InspectionScreen #inspection-actions {
        height: auto;
        padding: 0 1;
        align-horizontal: left;
    }
    /* Buttons hug their labels (Textual's default `min-width: 16`
       makes 5 buttons overflow ~80-col terminals, clipping the
       rightmost ones). `width: auto` + tight padding keeps the whole
       row visible across terminal widths. */
    InspectionScreen #inspection-actions Button {
        width: auto;
        min-width: 0;
        height: 3;
        margin: 0 1 0 0;
        padding: 0 1;
    }
    InspectionScreen #inspection-memory-footer {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    InspectionScreen #inspection-integration {
        padding: 1 2;
        height: auto;
    }
    InspectionScreen #inspection-integration-id {
        color: $accent;
        margin-bottom: 1;
    }
    InspectionScreen #inspection-integration-meta {
        color: $text-muted;
        margin-bottom: 1;
    }
    InspectionScreen #inspection-integration-lang {
        color: $text-muted;
        margin-bottom: 1;
    }
    InspectionScreen #inspection-integration-snippet {
        background: $boost;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
    }
    InspectionScreen #inspection-integration-hint {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("r", "inspect_run", "Run", show=False),
        Binding("e", "inspect_edit", "Edit", show=False),
        Binding("v", "inspect_evolve", "Evolve", show=False),
        Binding("d", "inspect_duplicate", "Duplicate", show=False),
        # Soft-delete this chain (recoverable from Memory's
        # trash). Confirm-gated; pops back to the Library on
        # success and refreshes any mounted LibraryScreen.
        Binding("delete", "inspect_delete", "Delete", show=True),
        # §4 P0 — toggle the DAG visualisation pane on the
        # Chain tab. Hidden when the user wants more room for
        # the step detail; shown by default since the DAG is
        # the headline navigation aid for multi-step chains.
        Binding("g", "toggle_dag", "Toggle DAG", show=True),
        # §4 P0 — Integration tab affordances. These fire
        # regardless of which tab is active so the user
        # doesn't have to focus the snippet pane first.
        Binding("t", "integration_cycle_lang", "Cycle", show=False),
        Binding("y", "integration_copy_id", "Copy ID", show=False),
        Binding("c", "integration_copy_snippet", "Copy snippet", show=False),
        Binding("L", "integration_open_lineage", "Lineage", show=False),
        # §3 P1 — "Use it now" reveal: pushes the same
        # UseItNowModal the §3 P0 save flow uses, pre-filled
        # with this chain's id / version / lifecycle.
        Binding("u", "integration_use_it_now", "Use it now", show=False),
        # §4 P1 — `R` (uppercase) re-fetches the chain payload
        # to refresh the freshness badge. Lowercase `r` is
        # already bound to "Run" so we shifted to avoid a
        # collision. The Integration pane's badge tooltip
        # spells the chord out so the user doesn't have to
        # remember it.
        Binding(
            "R", "refresh_freshness", "Refresh", show=True,
        ),
        # Export this chain to a Markdown file (human walkthrough +
        # a runnable CARL Python build script).
        Binding("m", "export_markdown", "Export chain", show=True),
        Binding("escape", "inspect_back", "Back", show=True),
    ]

    class ActionRequested(Message):
        """Posted when the user fires an action from the bar
        or its key binding. The host app routes the kind to
        the destination screen (Execution / Edit / Evolution
        screens or `pop_screen`)."""

        def __init__(self, action: InspectionAction, entity_id: str) -> None:
            super().__init__()
            self.action = action
            self.entity_id = entity_id

    def __init__(
        self,
        entity_id: str,
        *,
        channel: str = "latest",
    ) -> None:
        super().__init__()
        self.entity_id = entity_id
        self.channel = channel
        self.state = _LoadState()
        # Selected step index (drives the detail pane).
        self.selected_step: int = 0
        # Signature of the step buttons currently rendered in the STEPS
        # column. Identical refreshes (e.g. a freshness re-fetch of the
        # same chain) skip the remove/mount churn that would otherwise
        # race on duplicate button ids. `None` = nothing rendered yet.
        self._rendered_step_sig: tuple[str, ...] | None = None
        # §4 P0 Integration pane — active snippet language.
        # `t` cycles python → curl → cli; `y`/`c` copy the
        # id / active snippet respectively.
        from care.screens.use_it_now import SnippetLang

        self._integration_lang: SnippetLang = "python"
        # Action log for the Integration pane bindings —
        # tests + telemetry read this rather than scraping.
        self.integration_action_log: list[tuple[str, str]] = []
        # P0.20 run-history tab state.
        self.run_history: tuple[RunHistoryEntry, ...] = ()
        self.run_history_summary: RunHistorySummary = RunHistorySummary()
        self.run_history_error: str | None = None
        self._history_loaded: bool = False
        # Intermediate-artifacts pane (§1.2 [DONE — data layer]
        # → fully DONE). Populated when the host hands the
        # MAGE result to :meth:`record_intermediate_artifacts`.
        self.intermediate_artifacts: IntermediateArtifactsView = (
            IntermediateArtifactsView()
        )
        # §4 P1 — freshness state for the Integration pane.
        # `pinned_version_id` is the version_id the screen
        # bootstrapped with (set from the initial load);
        # `freshness_status` reflects the live poll's
        # comparison against it. Surface as a coloured badge
        # in the Integration meta line.
        self.pinned_version_id: str = ""
        self.freshness_status: str = "unknown"
        # Last error message from the freshness poll (when the
        # SDK call raised). Used by tests + the badge tooltip
        # rendering when the network is offline.
        self.freshness_last_error: str = ""
        # Periodic poll handle; cleared on unmount.
        self._freshness_timer: Any = None
        # Soft-delete outcome — tests read this rather than
        # scraping the toast surface.
        self.last_delete_outcome: Any = None

    FRESHNESS_POLL_SECONDS: float = 10.0
    """How often the freshness poll fires (§4 P1). Falls back
    to a 10-second tick when the upstream
    `watch_chain` SSE helper isn't available — cheap enough
    for an interactive surface, infrequent enough to avoid
    hammering Memory."""

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with TabbedContent(id="inspection-tabs"):
            with TabPane(t("inspection.tabChain"), id="inspection-tab-chain"):
                with Horizontal(id="inspection-body"):
                    with Vertical(id="inspection-steps"):
                        yield Label(t("inspection.steps"), classes="pane-title")
                        yield VerticalScroll(id="inspection-step-list")
                    with Vertical(id="inspection-detail"):
                        yield Label(t("inspection.detail"), classes="pane-title")
                        yield VerticalScroll(id="inspection-detail-body")
                    with Vertical(id="inspection-dag"):
                        yield Label(t("inspection.dag"), classes="pane-title")
                        yield Static("", id="inspection-dag-text", markup=False)
            with TabPane(
                t("inspection.tabRunHistory"), id="inspection-tab-history",
            ):
                yield Static("", id="inspection-history-summary")
                yield DataTable(id="inspection-history-table")
            with TabPane(
                t("inspection.tabArtifacts"), id="inspection-tab-artifacts",
            ):
                yield VerticalScroll(id="inspection-artifacts-body")
            # §4 P0 — Integration pane. One-screen answer to
            # "how do I call this from my service?" — surfaces
            # the stable chain_id + version + lifecycle badge
            # + a language-tabbed snippet (Python / curl /
            # care CLI) that reuses the same projection helper
            # the §3 P0 UseItNowModal renders from.
            with TabPane(
                t("inspection.tabIntegration"),
                id="inspection-tab-integration",
            ):
                with Vertical(id="inspection-integration"):
                    yield Static(
                        "", id="inspection-integration-id",
                    )
                    yield Static(
                        "", id="inspection-integration-meta",
                    )
                    yield Static(
                        "", id="inspection-integration-lang",
                    )
                    # Code snippet via a read-only, syntax-highlighted
                    # TextArea (proper code demonstration; selectable text).
                    yield TextArea(
                        "",
                        read_only=True,
                        id="inspection-integration-snippet",
                    )
                    yield Static(
                        t("inspection.integrationHint"),
                        id="inspection-integration-hint",
                    )
        with Horizontal(id="inspection-actions"):
            yield Button(
                t("inspection.run"),
                id="inspection-btn-run", variant="success",
            )
            yield Button(t("inspection.edit"), id="inspection-btn-edit")
            yield Button(
                t("inspection.evolve"),
                id="inspection-btn-evolve",
                variant="primary",
            )
            yield Button(
                t("inspection.exportMd"),
                id="inspection-btn-export",
            )
            yield Button(
                t("inspection.delete"),
                id="inspection-btn-delete", variant="error",
            )
            yield Button(t("common.back"), id="inspection-btn-back")
        yield Static("", id="inspection-memory-footer")
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="InspectionScreen",
                breadcrumb=(t("library.title"), t("inspection.breadcrumb")),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="InspectionScreen",
                scope="screen",
            )
        except Exception:
            pass
        try:
            table = self.query_one(
                "#inspection-history-table", DataTable,
            )
            table.add_columns(
                t("inspection.col.when"),
                t("inspection.col.status"),
                t("inspection.col.run"),
                t("inspection.col.duration"),
                t("inspection.col.tokens"),
                t("inspection.col.error"),
            )
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        # Native animated loading overlay while the bootstrap fetch is in
        # flight (cleared in `_refresh_panels`, which every `_load` exit
        # path calls). Reduced-motion-safe via Textual's animation level.
        self.loading = True
        self.run_worker(
            self._load(),
            name="inspection_load",
            group="inspection",
            exclusive=True,
            exit_on_error=False,
        )
        # §4 P1 — start the freshness poll. The first tick
        # fires after FRESHNESS_POLL_SECONDS so the bootstrap
        # load can land its pinned_version_id first. Stops on
        # unmount via `_freshness_timer.stop()`.
        if self.FRESHNESS_POLL_SECONDS > 0:
            self._freshness_timer = self.set_interval(
                self.FRESHNESS_POLL_SECONDS,
                self._spawn_freshness_check,
            )

    def on_unmount(self) -> None:
        timer = self._freshness_timer
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._freshness_timer = None

    def on_screen_suspend(self) -> None:
        """Pause the freshness poll while another screen is on top — no
        point re-checking Memory's version for a chain the user can't see.
        Resumed in :meth:`on_screen_resume`."""
        timer = self._freshness_timer
        if timer is not None:
            try:
                timer.pause()
            except Exception:
                pass

    def on_screen_resume(self) -> None:
        """Re-fetch the chain whenever this screen regains focus — e.g.
        after the Edit screen pops following a save — so manual content
        edits show immediately instead of the stale cached version.

        Guarded on an already-loaded payload so the initial activation
        (which pairs with :meth:`on_mount`'s bootstrap load) doesn't
        double-fetch; only genuine returns from a pushed screen reload.
        """
        # Resume the freshness poll paused in `on_screen_suspend` (no-op on
        # the initial activation where it was never paused).
        timer = self._freshness_timer
        if timer is not None:
            try:
                timer.resume()
            except Exception:
                pass
        if self.state.payload is None:
            return
        self.loading = True
        self.run_worker(
            self._load(force=True),
            name="inspection_load",
            group="inspection",
            exclusive=True,
            exit_on_error=False,
        )

    # ------------------------------------------------------------------
    # Fetch worker
    # ------------------------------------------------------------------

    async def _load(self, *, force: bool = False) -> None:
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self.state.loading = False
            self.state.error = "no memory facade configured"
            self._refresh_panels()
            return
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(self._call_get_chain, memory, force=force),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            self.state.loading = False
            self.state.error = "fetch timed out after 10.0s"
            self._refresh_panels()
            return
        except Exception as exc:  # noqa: BLE001
            self.state.loading = False
            self.state.error = f"{type(exc).__name__}: {exc}"
            self._refresh_panels()
            return
        self.state.loading = False
        self.state.payload = _project_payload(
            raw, entity_id=self.entity_id, channel=self.channel,
        )
        # §4 P1 — pin the bootstrap version so the freshness
        # poll has a baseline to compare against. The badge
        # starts as "fresh" once we know the pinned version;
        # later polls flip it to "stale" if a newer version
        # lands in Memory.
        payload = self.state.payload
        if payload is not None and payload.version_id:
            self.pinned_version_id = str(payload.version_id)
            self.freshness_status = "fresh"
            self.freshness_last_error = ""
        self._refresh_panels()

    def _call_get_chain(self, memory: Any, *, force: bool = False) -> Any:
        """Resolve the right SDK call to fetch one chain. The
        TODO names `client.get_chain(entity_id)`; older SDKs
        only expose `get_chain_dict` / `get_chain_raw`. Try
        in fallback order so the screen works against either.

        ``force=True`` bypasses the client's read cache via
        ``force_refresh=True`` (degrading gracefully when a given
        method doesn't accept it) — load-bearing for the
        reload-after-edit path, where the cache would otherwise serve
        the stale pre-save content and a freshly saved version would
        look like it never landed."""
        client = getattr(memory, "client", None) or getattr(memory, "_client", None)
        if client is None:
            raise RuntimeError("memory facade has no .client")
        # Prefer the dict-returning methods: `get_chain` resolves to a
        # `ReasoningChain` object whose serialized form `_project_payload`
        # has to round-trip, whereas `get_chain_dict` hands back the raw
        # content dict (top-level `steps`) the projector reads directly.
        for attr in ("get_chain_dict", "get_chain_raw", "get_chain"):
            fn = getattr(client, attr, None)
            if not callable(fn):
                continue
            # Try richest signature first (channel + cache bypass), then
            # fall back on TypeError so older/narrower methods still work.
            attempts: list[tuple[tuple, dict]] = []
            if force:
                attempts.append(
                    ((self.entity_id, self.channel), {"force_refresh": True}),
                )
                attempts.append(
                    ((self.entity_id,), {"force_refresh": True}),
                )
            attempts.append(((self.entity_id, self.channel), {}))
            attempts.append(((self.entity_id,), {}))
            for args, kwargs in attempts:
                try:
                    return fn(*args, **kwargs)
                except TypeError:
                    continue
        raise RuntimeError("client exposes no get_chain* method")

    # ------------------------------------------------------------------
    # Intermediate artifacts (§1.2 collapsible panes)
    # ------------------------------------------------------------------

    def record_intermediate_artifacts(self, source: Any) -> None:
        """Project a MAGE result / dict / artifacts view into
        the screen's ``Artifacts`` tab.

        The host worker calls this after a successful MAGE
        generation; the projection renders each
        :class:`IntermediateArtifact` as a `Collapsible` pane
        with the header, one-line summary, and the body lines
        from the data layer."""
        try:
            if isinstance(source, IntermediateArtifactsView):
                view = source
            else:
                view = project_intermediate_artifacts(source)
        except Exception:
            return
        self.intermediate_artifacts = view
        self._render_intermediate_artifacts()

    def _render_intermediate_artifacts(self) -> None:
        if not self.is_mounted:
            return
        try:
            container = self.query_one(
                "#inspection-artifacts-body", VerticalScroll,
            )
        except Exception:
            return
        try:
            for child in list(container.children):
                child.remove()
        except Exception:
            pass
        view = self.intermediate_artifacts
        if view.is_empty:
            container.mount(Static(t("inspection.noArtifacts")))
            return
        for art in view.artifacts:
            pane = Collapsible(
                title=f"{art.header}  ·  {art.summary}",
                id=f"art-{_artifact_id(art.stage)}",
            )
            container.mount(pane)
            if art.body:
                try:
                    pane.mount(Static(art.body, markup=False))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Run history (P0.20)
    # ------------------------------------------------------------------

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated,
    ) -> None:
        """Lazy-load the run history when the user opens the
        tab for the first time."""
        try:
            tab_id = event.pane.id if event.pane is not None else None
        except Exception:
            tab_id = None
        if tab_id != "inspection-tab-history":
            return
        if self._history_loaded:
            return
        self.refresh_run_history()

    def refresh_run_history(self) -> None:
        """Fire the run-history fetch worker. Idempotent —
        callers (the tab activator + a future explicit refresh
        button) can invoke without coordinating."""
        self.run_worker(
            self._load_run_history(),
            name="inspection_history",
            group="inspection",
            exclusive=True,
            exit_on_error=False,
        )

    async def _load_run_history(self) -> None:
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self.run_history = ()
            self.run_history_summary = RunHistorySummary()
            self.run_history_error = "no memory facade configured"
            self._history_loaded = True
            self._refresh_history_panel()
            return
        try:
            entries = await fetch_run_history(memory, self.entity_id)
        except RunHistoryError as exc:
            self.run_history = ()
            self.run_history_summary = RunHistorySummary()
            self.run_history_error = str(exc)
            self._history_loaded = True
            self._refresh_history_panel()
            return
        except Exception as exc:  # noqa: BLE001
            self.run_history = ()
            self.run_history_summary = RunHistorySummary()
            self.run_history_error = f"{type(exc).__name__}: {exc}"
            self._history_loaded = True
            self._refresh_history_panel()
            return
        self.run_history = entries
        self.run_history_summary = summarize_run_history(entries)
        self.run_history_error = None
        self._history_loaded = True
        self._refresh_history_panel()

    def _refresh_history_panel(self) -> None:
        if not self.is_mounted:
            return
        try:
            summary = self.query_one(
                "#inspection-history-summary", Static,
            )
            table = self.query_one(
                "#inspection-history-table", DataTable,
            )
        except Exception:
            return
        summary.update(self._format_history_summary())
        try:
            table.clear()
        except Exception:
            pass
        for entry in self.run_history:
            table.add_row(*self._history_row_cells(entry))

    def _format_history_summary(self) -> str:
        if self.run_history_error:
            return f"⚠ {self.run_history_error}"
        s = self.run_history_summary
        if s.total_runs == 0:
            return t("inspection.history.noRuns")
        rate = s.success_rate or 0.0
        avg = s.avg_duration_seconds or 0.0
        parts = [
            t("inspection.history.runs", n=s.total_runs),
            t("inspection.history.ok", n=s.success_count),
            t("inspection.history.failed", n=s.failure_count),
            t("inspection.history.success", pct=f"{rate * 100:.0f}"),
            t("inspection.history.avg", secs=f"{avg:.1f}"),
        ]
        if s.total_tokens:
            parts.append(t("inspection.history.tokens", n=s.total_tokens))
        return "  ·  ".join(parts)

    @staticmethod
    def _history_row_cells(entry: RunHistoryEntry) -> tuple[str, ...]:
        when = (
            entry.finished_at.strftime("%Y-%m-%d %H:%M")
            if entry.finished_at
            else "—"
        )
        status = "✓" if entry.success else "✗"
        duration = (
            f"{entry.duration_seconds:.1f}s"
            if entry.duration_seconds is not None
            else "—"
        )
        tokens = (
            str(entry.total_tokens)
            if entry.total_tokens is not None
            else "—"
        )
        error = entry.error_message or "" if not entry.success else ""
        return (
            when,
            status,
            entry.run_id[:18],
            duration,
            tokens,
            error[:60],
        )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh_panels(self) -> None:
        if not self.is_mounted:
            return
        # The bootstrap / re-fetch worker resolves into here on every path —
        # drop the loading overlay armed at the worker-spawn sites.
        self.loading = False
        try:
            self._render_step_list()
            self._render_detail()
            self._render_dag()
            self._render_footer()
        except Exception:
            pass

    def _render_step_list(self) -> None:
        """Render the STEPS column as clickable step buttons — the same
        affordance the chat DAG modal uses. Clicking a button selects
        that step and drives the detail pane."""
        container = self.query_one("#inspection-step-list", VerticalScroll)
        payload = self.state.payload
        label = (
            t("inspection.loading") if self.state.loading
            else (self.state.error or t("inspection.noChain"))
        )
        # Signature of what we're about to render so identical refreshes
        # short-circuit (just re-highlight) instead of churning the DOM.
        if payload is None:
            sig: tuple[str, ...] = ("__placeholder__", label)
        elif not payload.steps:
            sig = ("__empty__",)
        else:
            sig = tuple(
                self._step_button_label(payload.steps[idx], idx)
                for idx in range(len(payload.steps))
            )
        if sig == self._rendered_step_sig:
            self._highlight_active_step()
            return
        self._rendered_step_sig = sig

        try:
            container.remove_children()
        except Exception:
            pass

        def _mount() -> None:
            # `remove_children()` is async-deferred; mounting synchronously
            # right after races the removal and collides on the stable
            # `inspection-stepbtn-N` ids (DuplicateIds stalls the pump).
            # Defer one refresh so the removal settles first.
            try:
                target = self.query_one(
                    "#inspection-step-list", VerticalScroll,
                )
            except Exception:
                return
            if payload is None:
                target.mount(Static(label, markup=False))
                return
            if not payload.steps:
                target.mount(Static(t("inspection.noSteps"), markup=False))
                return
            for idx in range(len(payload.steps)):
                target.mount(
                    Button(
                        Text(self._step_button_label(payload.steps[idx], idx)),
                        id=f"inspection-stepbtn-{idx}",
                        classes="inspection-step-btn",
                    )
                )
            self._highlight_active_step()

        if self.is_mounted:
            self.call_after_refresh(_mount)
        else:
            _mount()

    @staticmethod
    def _step_button_label(step: dict, idx: int) -> str:
        """``"1. Analyse query (AI)"`` — the numbered step label the chat
        DAG modal renders on its step buttons."""
        number = step.get("number")
        prefix = f"{number}. " if isinstance(number, int) else f"{idx + 1}. "
        return f"{prefix}{_step_label(step, idx)}"

    def _highlight_active_step(self) -> None:
        """Tint the button for :attr:`selected_step` so the active step
        reads at a glance (the ListView gave this for free; buttons need
        an explicit class flip)."""
        payload = self.state.payload
        if payload is None or not payload.steps:
            return
        active = max(0, min(self.selected_step, len(payload.steps) - 1))
        for idx in range(len(payload.steps)):
            try:
                btn = self.query_one(f"#inspection-stepbtn-{idx}", Button)
            except Exception:
                continue
            btn.set_class(idx == active, "-active")

    def _render_detail(self) -> None:
        container = self.query_one("#inspection-detail-body", VerticalScroll)
        # Remove children sync via clear children — old API.
        try:
            for child in list(container.children):
                child.remove()
        except Exception:
            pass
        payload = self.state.payload
        if payload is None:
            container.mount(
                Static(
                    self.state.error or t("inspection.loading"),
                    markup=False,
                )
            )
            return
        if not payload.steps:
            container.mount(Static(t("inspection.noSteps"), markup=False))
            return
        idx = max(0, min(self.selected_step, len(payload.steps) - 1))
        step = payload.steps[idx]
        header, fields = _step_detail_fields(step, idx, payload.steps)
        # markup=False on the header Static: step titles are plain text.
        container.mount(
            Static(header, markup=False, classes="step-detail-header")
        )
        # Build the body as a Rich `Text` so field labels render bold
        # while the values stay literal — `Text.append` never parses
        # markup, so config reprs with bracket syntax (e.g. `memory[-1]`)
        # pass through untouched.
        if not fields:
            container.mount(Static(t("inspection.noDetail"), markup=False))
            return
        body = Text()
        for i, (label, value) in enumerate(fields):
            if i:
                body.append("\n")
            body.append(f"{label}: ", style="bold")
            body.append(value)
        container.mount(Static(body))

    def _render_dag(self) -> None:
        target = self.query_one("#inspection-dag-text", Static)
        payload = self.state.payload
        if payload is None or not payload.steps:
            target.update("(empty)")
            return
        # Reuse the chat DAG modal's box-and-arrow visualisation so the
        # chain's shape (parallel branches, fan-in/out) reads the same
        # everywhere — tinted by step type (AI / Tool / MCP / Code) so a
        # static chain reads at a glance. Falls back to the ASCII tree
        # when the box renderer can't make sense of the payload, so the
        # pane never blanks.
        from rich.text import Text

        from care.runtime.dag_view import dag_display_opts, render_dag_styled

        try:
            lines = render_dag_styled(
                payload.steps,
                max_graph_width=self._dag_graph_width(),
                **dag_display_opts(getattr(self.app, "config", None)),
            )
        except Exception:
            lines = []
        if lines:
            target.update(Text("\n").join(lines))
        else:
            target.update(render_chain_dag(payload.steps))

    def _dag_graph_width(self) -> int:
        """Width budget for the inline box graph before it collapses to
        the compact number-box + legend variant. Tracks the DAG pane's
        live width (minus its horizontal padding) so the graph fits the
        narrow ``1fr`` rail; falls back to a sane default pre-layout."""
        from care.runtime.dag_view import _DEFAULT_MAX_GRAPH_WIDTH

        try:
            width = int(self.query_one("#inspection-dag").size.width)
        except Exception:
            return _DEFAULT_MAX_GRAPH_WIDTH
        if width <= 0:
            return _DEFAULT_MAX_GRAPH_WIDTH
        # `#inspection-dag` carries `padding: 0 2` → 4 cells of gutter.
        return max(20, width - 4)

    def _render_footer(self) -> None:
        target = self.query_one("#inspection-memory-footer", Static)
        payload = self.state.payload
        if payload is None:
            target.update("")
            return
        parts = [
            f"entity_id: {payload.entity_id}",
            f"version: {payload.version_id or '?'}",
            f"channel: {payload.channel}",
        ]
        target.update("  ·  ".join(parts))
        self._render_integration_pane()

    # ------------------------------------------------------------------
    # Integration pane (§4 P0)
    # ------------------------------------------------------------------

    def _resolve_memory_base_url(self) -> str:
        """Best-effort: the curl snippet wants the real
        Memory base URL when known, else fall back to the
        ``${CARE_MEMORY__BASE_URL}`` env-var literal."""
        memory = getattr(self.app, "memory", None)
        if memory is None:
            return ""
        client = getattr(memory, "client", None)
        return (
            str(getattr(client, "base_url", "") or "")
            or str(getattr(memory, "base_url", "") or "")
        )

    def _render_integration_pane(self) -> None:
        """Paint the Integration tab. Reads the same payload
        the chain tab loads from `state.payload`; no extra
        worker."""
        try:
            id_pane = self.query_one(
                "#inspection-integration-id", Static,
            )
            meta_pane = self.query_one(
                "#inspection-integration-meta", Static,
            )
            lang_pane = self.query_one(
                "#inspection-integration-lang", Static,
            )
            snippet_pane = self.query_one(
                "#inspection-integration-snippet", TextArea,
            )
        except Exception:
            return
        from care.screens.use_it_now import (
            lang_indicator,
            render_integration_snippet,
        )

        payload = self.state.payload
        entity_id = (
            payload.entity_id if payload is not None
            else self.entity_id
        )
        version = (
            payload.version_id if payload is not None else ""
        ) or "latest"
        channel = (
            payload.channel if payload is not None
            else self.channel
        ) or "latest"
        id_pane.update(f"chain_id: {entity_id}")
        meta_pane.update(
            f"version: {version}  ·  channel: {channel}  ·  "
            + self._format_freshness_badge(),
        )
        lang_pane.update(lang_indicator(self._integration_lang))
        memory_base_url = (
            self._resolve_memory_base_url()
            or "${CARE_MEMORY__BASE_URL}"
        )
        from care.screens.use_it_now import snippet_language

        snippet_pane.language = snippet_language(self._integration_lang)
        snippet_pane.text = render_integration_snippet(
            self._integration_lang,
            entity_id=entity_id,
            channel=channel,
            memory_base_url=memory_base_url,
        )

    def _format_freshness_badge(self) -> str:
        """§4 P1 — render the freshness state as a coloured
        dot + label. Three states:

        * ``"fresh"`` — green dot + the literal "fresh"
          (the pinned `version_id` matches the latest poll).
        * ``"stale"`` — amber dot + "stale (refresh with R)"
          so the user knows the keystroke.
        * ``"unknown"`` — grey dot + "unknown" (either the
          load hasn't finished or the poll raised; the
          `freshness_last_error` slot has the detail).

        Using textual dots (●) instead of `[fresh]` keeps the
        Rich markup interpreter from swallowing the label —
        same fix-up from iter 67's `★ front` lesson on the
        Pareto pane.
        """
        status = self.freshness_status
        if status == "fresh":
            return "● fresh"
        if status == "stale":
            return "● stale (refresh with R)"
        return "○ unknown"

    def action_integration_cycle_lang(self) -> None:
        from care.screens.use_it_now import cycle_language

        self._integration_lang = cycle_language(
            self._integration_lang,
        )
        self.integration_action_log.append(
            ("cycle_lang", self._integration_lang),
        )
        self._render_integration_pane()

    def action_integration_copy_id(self) -> None:
        payload = self.state.payload
        entity_id = (
            payload.entity_id if payload is not None
            else self.entity_id
        )
        self.integration_action_log.append(
            ("copy_id", entity_id),
        )
        self._copy_to_clipboard(entity_id, "id")

    def action_integration_copy_snippet(self) -> None:
        from care.screens.use_it_now import render_integration_snippet

        payload = self.state.payload
        entity_id = (
            payload.entity_id if payload is not None
            else self.entity_id
        )
        channel = (
            payload.channel if payload is not None
            else self.channel
        ) or "latest"
        body = render_integration_snippet(
            self._integration_lang,
            entity_id=entity_id,
            channel=channel,
            memory_base_url=(
                self._resolve_memory_base_url()
                or "${CARE_MEMORY__BASE_URL}"
            ),
        )
        self.integration_action_log.append(
            ("copy_snippet", self._integration_lang),
        )
        self._copy_to_clipboard(
            body, f"{self._integration_lang} snippet",
        )

    def action_integration_use_it_now(self) -> None:
        """`u` — push the §3 P0 :class:`UseItNowModal`
        pre-filled with this chain's identity. Reuses the
        same component the save flow surfaces, so the
        integration recipes stay consistent across the
        save / inspect / evolution-launch surfaces.

        On dismiss with ``evolve_requested=True``, routes
        through ``app._push_evolution_for(entity_id)``
        (when defined) — same handler the artifacts
        save flow uses."""
        payload = self.state.payload
        entity_id = (
            payload.entity_id if payload is not None
            else self.entity_id
        )
        display_name = (
            payload.display_name if payload is not None
            else ""
        )
        version = (
            payload.version_id if payload is not None else ""
        ) or "latest"
        channel = self.channel or "latest"
        self.integration_action_log.append(
            ("use_it_now", entity_id),
        )
        try:
            from care.screens.use_it_now import (
                UseItNowModal,
                UseItNowResult,
            )
        except Exception as exc:  # noqa: BLE001
            self._integration_toast(
                t("inspection.openUseItNowFailed", error=exc),
                severity="error",
            )
            return

        def _on_dismiss(result: UseItNowResult | None) -> None:
            if result is None or not result.evolve_requested:
                return
            opener = getattr(
                self.app, "_push_evolution_for", None,
            )
            if callable(opener):
                try:
                    opener(entity_id)
                    return
                except Exception as exc:  # noqa: BLE001
                    self._integration_toast(
                        t("inspection.openEvolutionFailed", error=exc),
                        severity="error",
                    )
                    return
            self._integration_toast(
                t("inspection.evolutionHint", id=entity_id),
                severity="info",
            )

        memory_base_url = (
            self._resolve_memory_base_url()
            or "${CARE_MEMORY__BASE_URL}"
        )
        try:
            self.app.push_screen(
                UseItNowModal(
                    entity_id=entity_id,
                    version=version,
                    channel=channel,
                    display_name=display_name,
                    memory_base_url=memory_base_url,
                ),
                _on_dismiss,
            )
        except Exception as exc:  # noqa: BLE001
            self._integration_toast(
                t("inspection.openUseItNowFailed", error=exc),
                severity="error",
            )

    # ------------------------------------------------------------------
    # Freshness poll (§4 P1)
    # ------------------------------------------------------------------

    def _spawn_freshness_check(self) -> None:
        """Timer callback — kicks the async freshness worker.
        Exclusive=True so a slow tick doesn't pile up overlapping
        polls when the network is laggy."""
        self.run_worker(
            self._check_freshness(),
            name="inspection_freshness",
            group="inspection_freshness",
            exclusive=True,
            exit_on_error=False,
        )

    async def _check_freshness(self) -> None:
        """§4 P1 — compare the live ``version_id`` against the
        bootstrap-pinned one. Best-effort:

        * No pinned version yet → skip (load worker hasn't
          stamped one).
        * No memory facade → skip (production gate; the badge
          stays at "unknown").
        * SDK call raises → record the error string, paint the
          badge as "unknown" so the user knows the poll is
          struggling, return.
        * SDK returns a `version_id` matching the pinned →
          status stays "fresh" (the meta line repaints with
          the green dot).
        * SDK returns a different `version_id` → status flips
          to "stale" + the meta line repaints with the amber
          dot + the refresh hint.

        Marshals the SDK call through ``asyncio.to_thread``
        because the gigaevo SDK is sync; the Textual worker
        loop stays responsive.
        """
        if not self.pinned_version_id:
            return
        memory = getattr(self.app, "memory", None)
        if memory is None:
            return
        getter = getattr(memory, "get_entity", None)
        if not callable(getter):
            return
        try:
            entity = await asyncio.wait_for(
                asyncio.to_thread(
                    getter,
                    self.entity_id,
                    entity_type="chain",
                    channel=self.channel or "latest",
                ),
                timeout=5.0,
            )
        except Exception as exc:  # noqa: BLE001
            self.freshness_last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self.freshness_status = "unknown"
            self._render_integration_pane()
            return
        version_id = ""
        if isinstance(entity, dict):
            version_id = str(entity.get("version_id") or "")
        if not version_id:
            self.freshness_last_error = "no version_id in response"
            self.freshness_status = "unknown"
            self._render_integration_pane()
            return
        self.freshness_last_error = ""
        if version_id == self.pinned_version_id:
            self.freshness_status = "fresh"
        else:
            self.freshness_status = "stale"
        self._render_integration_pane()

    def action_refresh_freshness(self) -> None:
        """`R` — re-fetch the chain payload + reset the
        freshness baseline to the latest version. Spawns a
        fresh `_load()` worker (same path the on_mount fires)
        so the entire screen repaints with the newest data."""
        self.integration_action_log.append(("refresh", ""))
        self.loading = True
        self.run_worker(
            self._load(),
            name="inspection_load",
            group="inspection",
            exclusive=True,
            exit_on_error=False,
        )

    def action_integration_open_lineage(self) -> None:
        payload = self.state.payload
        entity_id = (
            payload.entity_id if payload is not None
            else self.entity_id
        )
        self.integration_action_log.append(
            ("open_lineage", entity_id),
        )
        try:
            from care.screens.lineage import LineageModal

            memory = getattr(self.app, "memory", None)
            self.app.push_screen(
                LineageModal(entity_id, memory=memory),
            )
        except Exception as exc:  # noqa: BLE001
            self._integration_toast(
                t("inspection.openLineageFailed", error=exc),
                severity="error",
            )

    def _copy_to_clipboard(self, text: str, label: str) -> None:
        try:
            from care.runtime.clipboard import copy_text

            copy_text(text)
        except Exception as exc:  # noqa: BLE001
            self._integration_toast(
                t("inspection.copyFailed", error=exc), severity="warning",
            )
            return
        self._integration_toast(
            t("inspection.copied", label=label), severity="info",
        )

    def _integration_toast(
        self, message: str, *, severity: str = "info",
    ) -> None:
        push = getattr(self.app, "push_toast", None)
        if callable(push):
            try:
                push(message, severity=severity)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        # Step button → select that step + refresh the detail pane.
        if bid.startswith("inspection-stepbtn-"):
            try:
                idx = int(bid.rsplit("-", 1)[1])
            except ValueError:
                return
            self.selected_step = idx
            self._render_detail()
            self._highlight_active_step()
            return
        if event.button.id == "inspection-btn-delete":
            self.action_inspect_delete()
            return
        if event.button.id == "inspection-btn-export":
            self.action_export_markdown()
            return
        mapping = {
            "inspection-btn-run": "run",
            "inspection-btn-edit": "edit",
            "inspection-btn-evolve": "evolve",
            "inspection-btn-back": "back",
        }
        action = mapping.get(event.button.id or "")
        if action is None:
            return
        self._dispatch(action)  # type: ignore[arg-type]

    def action_inspect_run(self) -> None:
        self._dispatch("run")

    def action_inspect_edit(self) -> None:
        self._dispatch("edit")

    def action_inspect_evolve(self) -> None:
        self._dispatch("evolve")

    def action_inspect_duplicate(self) -> None:
        self._dispatch("duplicate")

    def action_inspect_back(self) -> None:
        self._dispatch("back")

    def action_export_markdown(self) -> None:
        """`m` / Export-MD button — open the shared :class:`ExportChainModal`
        defaulted to Markdown so the user picks a path (and can still switch
        format). The Markdown output is a human-readable ``## Step N`` /
        ``### Aim`` walkthrough that ends with a fenced ``python`` CARL build
        script."""
        payload = self.state.payload
        if payload is None or not payload.steps:
            self._integration_toast(
                t("inspection.exportNoChain"), severity="warning",
            )
            return
        from care.screens.export_chain import (
            ExportChainModal,
            ExportChainResult,
        )

        chain_dict: dict[str, Any] = {
            "name": payload.display_name,
            "domain": payload.domain,
            "description": payload.description,
            "steps": [dict(s) for s in payload.steps],
        }

        def _on_dismiss(result: ExportChainResult | None) -> None:
            if result is None:
                return
            if result.ok and result.path is not None:
                self._integration_toast(
                    t("inspection.exported", path=result.path),
                    severity="info",
                )
            elif result.error:
                self._integration_toast(
                    t("inspection.exportFailed", error=result.error),
                    severity="warning",
                )

        self.app.push_screen(
            ExportChainModal(
                chain=chain_dict,
                display_name=payload.display_name,
                entity_id=self.entity_id,
                version=str(payload.version_id or ""),
                default_format="markdown",
            ),
            _on_dismiss,
        )

    def action_inspect_delete(self) -> None:
        """`Del` / Delete button — confirm-gated soft-delete of
        this chain. Runs on a worker (the confirm modal is
        awaited) so the binding handler stays synchronous."""
        self.run_worker(
            self._run_delete(),
            name="inspection_delete",
            group="inspection_action",
            exclusive=False,
            exit_on_error=False,
        )

    async def _run_delete(self) -> None:
        from care.runtime.row_actions import delete_row
        from care.runtime.library_view import LibraryRow
        from care.screens.confirm import ConfirmModal

        name = self._delete_display_name()
        modal = ConfirmModal(
            title=t("inspection.deleteTitle"),
            body=t("inspection.deleteBody", name=name),
            confirm_label=t("inspection.delete"),
        )
        confirmed = await self.app.push_screen_wait(modal)
        if not confirmed:
            return
        row = LibraryRow(
            entity_id=self.entity_id,
            entity_type="chain",
            channel=self.channel,
            display_name=self._delete_display_name(),
        )
        outcome = await delete_row(getattr(self.app, "memory", None), row)
        self.last_delete_outcome = outcome
        if outcome.success:
            self._integration_toast(
                t("inspection.deleted", name=name), severity="info",
            )
            refresh = getattr(self.app, "_refresh_library_screens", None)
            if callable(refresh):
                refresh()
            try:
                self.app.pop_screen()
            except Exception:
                pass
        else:
            self._integration_toast(
                t("inspection.deleteFailed", error=outcome.error),
                severity="error",
            )

    def _delete_display_name(self) -> str:
        payload = self.state.payload
        if payload is not None and payload.display_name:
            return payload.display_name
        return self.entity_id

    def action_toggle_dag(self) -> None:
        """§4 P0 — flip the DAG pane visibility on the Chain
        tab. Idempotent: sets `display` on `#inspection-dag`
        and (when collapsing) records the toggle so tests
        can pin behaviour without scraping CSS state."""
        try:
            pane = self.query_one("#inspection-dag")
        except Exception:
            return
        new_visible = not pane.display
        pane.display = new_visible
        self._dag_visible = new_visible

    _dag_visible: bool = True
    """Mirrors `#inspection-dag.display`. Tests assert against
    this attribute rather than walking the widget tree."""

    def _dispatch(self, action: InspectionAction) -> None:
        self.post_message(self.ActionRequested(action, self.entity_id))
        if action == "back":
            try:
                self.app.pop_screen()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_payload(
    raw: Any, *, entity_id: str, channel: str,
) -> InspectionPayload:
    """Project the SDK's response into a frozen
    :class:`InspectionPayload`. Accepts a dict (typical
    `get_chain_dict` return) or a pydantic-shaped object."""
    content: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    version_id = ""
    if raw is None:
        return InspectionPayload(entity_id=entity_id, channel=channel)
    if isinstance(raw, dict):
        # EntityResponse wrapper?
        if "content" in raw and isinstance(raw["content"], dict):
            content = raw["content"]
            meta = raw.get("meta") or {}
            version_id = str(raw.get("version_id") or "")
        else:
            content = raw
    else:
        c = getattr(raw, "content", None)
        if isinstance(c, dict):
            content = c
            meta = getattr(raw, "meta", None) or {}
            version_id = str(getattr(raw, "version_id", "") or "")
        else:
            # Pydantic `model_dump` or CARL's `ReasoningChain.to_dict`.
            # The `get_chain` SDK call returns a `ReasoningChain` whose
            # serialized form carries `steps` at the top level (no
            # `content` wrapper), so fall back to `payload` itself.
            for serialiser in ("model_dump", "to_dict"):
                fn = getattr(raw, serialiser, None)
                if not callable(fn):
                    continue
                try:
                    payload = fn()
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(payload, dict):
                    content = payload.get("content") or payload
                    meta = payload.get("meta") or {}
                    version_id = str(payload.get("version_id") or "")
                    break
    steps_raw = content.get("steps") if isinstance(content, dict) else None
    steps: tuple[dict, ...] = tuple(steps_raw) if isinstance(steps_raw, list) else ()
    display_name = str(
        (meta or {}).get("display_name")
        or content.get("display_name")
        or content.get("name")
        or "",
    )
    description = str(
        content.get("description") or (meta or {}).get("description") or ""
    )
    domain = str(
        (meta or {}).get("domain") or content.get("domain") or ""
    )
    return InspectionPayload(
        entity_id=entity_id,
        channel=channel,
        version_id=version_id,
        display_name=display_name,
        domain=domain,
        description=description,
        steps=steps,
    )


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value)) or "(empty)"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items()) or "(empty)"
    return str(value)


# CARL serialises a step's kind under ``step_type`` (older / synthetic
# shapes use ``type``). Map the machine value to the human label the
# inspector shows — "AI" for an LLM reasoning step, etc.
_STEP_TYPE_LABELS: dict[str, str] = {
    "llm": "AI",
    "tool": "Tool",
    "mcp": "MCP",
    "mcp_resource": "MCP Resource",
    "memory": "Memory",
    "transform": "Transform",
    "conditional": "Conditional",
    "structured_output": "Structured Output",
    "agent_skill": "Agent Skill",
    "evaluation": "Evaluation",
    "agent_handoff": "Agent Handoff",
    "parallel_sampling": "Parallel Sampling",
    "tool_discovery": "Tool Discovery",
    "human_input": "Human Input",
    "supervisor": "Supervisor",
    "debate": "Debate",
}


def _step_type_value(step: dict) -> str:
    """Raw step-kind token (``"llm"``, ``"tool"``, …) — empty when
    the step carries no type."""
    return str(step.get("step_type") or step.get("type") or "").strip().lower()


def _step_type_label(step: dict) -> str:
    """Human label for a step's kind (``"AI"`` for ``llm``). Falls
    back to a title-cased version of any unmapped token."""
    raw = _step_type_value(step)
    if not raw:
        return ""
    return _STEP_TYPE_LABELS.get(raw, raw.replace("_", " ").title())


def _step_title(step: dict, index: int) -> str:
    """The step's human title. CARL names it ``title``; older /
    synthetic shapes use ``name`` / ``step_name`` / ``label``."""
    for key in ("title", "name", "step_name", "label", "id"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"step-{index + 1}"


def _format_deps(deps: Any, steps: tuple[dict, ...] | list[dict]) -> str:
    """Render a step's dependency edges. CARL expresses them as a
    list of step ``number`` ints; resolve each to its title so the
    detail pane reads ``"1 (Analyse query)"`` rather than a bare id."""
    if isinstance(deps, (str, int)):
        deps = [deps]
    by_number: dict[str, str] = {}
    for idx, step in enumerate(steps):
        number = step.get("number")
        if number is not None:
            by_number[str(number)] = _step_title(step, idx)
    parts: list[str] = []
    for dep in deps or ():
        key = str(dep)
        title = by_number.get(key)
        parts.append(f"{key} ({title})" if title else key)
    return ", ".join(parts) or "(none)"


def _step_detail_fields(
    step: dict, index: int, steps: tuple[dict, ...] | list[dict] = (),
) -> tuple[str, list[tuple[str, str]]]:
    """Project one step into ``(header, fields)`` where ``fields`` is
    an ordered list of ``(label, value)`` pairs. The header is the
    numbered title plus the friendly kind (``"1. Analyse query  ·
    AI"``). Pure helper so both the string projection
    (:func:`format_step_detail`) and the bold-label renderer share
    one source of truth."""
    type_label = _step_type_label(step)
    title = _step_title(step, index)
    number = step.get("number")
    prefix = f"{number}. " if isinstance(number, int) else ""
    header = f"{prefix}{title}"
    if type_label:
        header = f"{header}  ·  {type_label}"

    fields: list[tuple[str, str]] = []

    def add(label: str, value: Any) -> None:
        if value in (None, "", [], {}, ()):
            return
        fields.append((label, _format_value(value)))

    deps = (
        step.get("dependencies")
        or step.get("deps")
        or step.get("depends_on")
    )
    if deps:
        fields.append(
            (t("inspection.field.dependsOn"), _format_deps(deps, steps))
        )
    add(t("inspection.field.triggeredBy"), step.get("triggered_by"))

    # LLM-flavoured fields.
    add(t("inspection.field.aim"), step.get("aim"))
    add(t("inspection.field.reasoningQuestions"),
        step.get("reasoning_questions"))
    add(t("inspection.field.stageAction"), step.get("stage_action"))
    add(t("inspection.field.exampleReasoning"),
        step.get("example_reasoning"))

    # Per-step LLM config (model / temperature).
    llm_config = step.get("llm_config")
    if isinstance(llm_config, dict):
        add(t("inspection.field.model"), llm_config.get("model"))
        add(t("inspection.field.temperature"), llm_config.get("temperature"))

    # Tool / MCP / generic step config.
    cfg = step.get("config")
    if cfg is None:
        cfg = step.get("step_config")
    if isinstance(cfg, dict):
        for ckey, cval in cfg.items():
            add(ckey.replace("_", " ").capitalize(), cval)
    elif cfg not in (None, "", [], {}):
        add(t("inspection.field.config"), cfg)

    add(t("inspection.field.contextQueries"),
        step.get("step_context_queries"))

    if not fields:
        # Unknown shape — surface whatever the step carries, minus
        # the fields already in the header / known structural noise.
        skip = {"number", "title", "name", "step_name", "label",
                "step_type", "type", "metrics"}
        for key, value in step.items():
            if key in skip:
                continue
            add(key.replace("_", " ").capitalize(), value)

    return header, fields


def format_step_detail(
    step: dict, index: int, steps: tuple[dict, ...] | list[dict] = (),
) -> tuple[str, str]:
    """Project one step into a ``(header, body)`` pair for the
    detail pane — the body is the meaningful fields rendered as
    ``label: value`` lines. Pure helper so tests can pin output
    without driving the screen. The on-screen renderer uses
    :func:`_step_detail_fields` directly so it can embolden the
    labels."""
    header, fields = _step_detail_fields(step, index, steps)
    body = "\n".join(f"{label}: {value}" for label, value in fields)
    return header, body


def _step_label(step: dict, index: int) -> str:
    """Friendly per-step label for the DAG / step-list views:
    ``"Analyse query (AI)"``. Falls back to ``step-N`` when the
    step carries no title."""
    title = _step_title(step, index)
    type_label = _step_type_label(step)
    if type_label and type_label != title:
        return f"{title} ({type_label})"
    return title


def _step_key(step: dict, index: int) -> str:
    """Pick the dependency-graph key for a step. Order
    matters: `id` is the canonical CARL identifier; `name`
    falls back; `step-N` is the synthetic position key shared
    with :func:`_step_label`."""
    for k in ("id", "step_id", "name", "step_name", "label"):
        v = step.get(k)
        if isinstance(v, str) and v:
            return v
    return f"step-{index + 1}"


def render_chain_dag(steps: tuple[dict, ...] | list[dict]) -> str:
    """Render the chain step graph as an ASCII tree (§4 P0).

    Rules:

    * Steps with no declared ``deps`` / ``depends_on`` are
      roots. Steps with deps render as children of every
      declared parent (which means a fan-in step appears more
      than once — the visual cost is acceptable for
      typical CARL chains where fan-in is rare).
    * Unknown dep references (a step depends on a name not in
      the graph) render at the top with a `(?)` suffix so the
      user sees the dangling edge instead of silently
      collapsing the dep.
    * Pure linear chains render as a deep right-leaning tree
      — exactly what a user expects from
      `pipeline` topologies.
    * Cycles are guarded: a step that would re-introduce a
      visited node renders as ``↺ <name>`` and stops the
      branch.

    Pure helper so tests can pin output without driving the
    full screen.
    """
    steps = list(steps)
    if not steps:
        return "(empty)"

    # Index by canonical key.
    by_key: dict[str, tuple[int, dict]] = {}
    for idx, step in enumerate(steps):
        key = _step_key(step, idx)
        # Last-write-wins on duplicate keys (very rare; CARL
        # rejects them upstream).
        by_key[key] = (idx, step)

    # CARL serialises edges as ``dependencies`` referencing each
    # step's ``number`` (an int), not its name. Map number → key
    # so those numeric references resolve to real graph nodes
    # alongside the name-based ``deps`` / ``depends_on`` spellings.
    num_to_key: dict[str, str] = {}
    for idx, step in enumerate(steps):
        num = step.get("number")
        if num is not None:
            num_to_key[str(num)] = _step_key(step, idx)

    children: dict[str, list[str]] = {key: [] for key in by_key}
    roots: list[str] = []
    unknown_deps: list[tuple[str, str]] = []  # (step_key, missing_dep)

    for key, (idx, step) in by_key.items():
        deps = (
            step.get("deps")
            or step.get("depends_on")
            or step.get("dependencies")
            or ()
        )
        if isinstance(deps, str):
            deps = (deps,)
        real_deps: list[str] = []
        for raw_dep in deps:
            dep = str(raw_dep)
            if dep in by_key:
                real_deps.append(dep)
            elif dep in num_to_key:
                real_deps.append(num_to_key[dep])
            else:
                unknown_deps.append((key, dep))
        if not real_deps:
            roots.append(key)
        else:
            for dep in real_deps:
                children[dep].append(key)

    out: list[str] = []

    def _walk(
        key: str,
        prefix: str,
        is_last: bool,
        visited: frozenset[str],
    ) -> None:
        idx, step = by_key[key]
        connector = "└─" if is_last else "├─"
        if key in visited:
            out.append(f"{prefix}{connector} ↺ {_step_label(step, idx)}")
            return
        out.append(f"{prefix}{connector} {_step_label(step, idx)}")
        new_visited = visited | {key}
        child_keys = children[key]
        next_prefix = prefix + ("   " if is_last else "│  ")
        for i, child in enumerate(child_keys):
            _walk(
                child, next_prefix, i == len(child_keys) - 1,
                new_visited,
            )

    # Render dangling-dep nodes first so the user sees them.
    if unknown_deps:
        out.append("(unresolved dependencies)")
        for step_key, missing in unknown_deps:
            idx = by_key[step_key][0]
            label = _step_label(by_key[step_key][1], idx)
            out.append(f"  ⚠ {label} → {missing} (?)")
        out.append("")

    if not roots:
        # Pure cycle / every step has a dep — fall back to
        # rendering everything as a flat list so the user
        # still sees the chain.
        out.append("(cycle detected — flat fallback)")
        for idx, step in enumerate(steps):
            out.append(f"• {_step_label(step, idx)}")
        return "\n".join(out)

    for i, root in enumerate(roots):
        _walk(root, "", i == len(roots) - 1, frozenset())

    return "\n".join(out)


def _artifact_id(stage: str) -> str:
    """Sanitise a stage key so Textual accepts it as a
    widget id (only alnum / `-` / `_`)."""
    out = []
    for ch in stage or "x":
        out.append(ch if ch.isalnum() or ch in "-_" else "-")
    return "".join(out)[:48] or "x"


__all__ = [
    "InspectionAction",
    "InspectionPayload",
    "InspectionScreen",
]
