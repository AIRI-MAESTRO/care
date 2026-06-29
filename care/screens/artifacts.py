"""ArtifactsScreen — current-chat session artifacts (TODO §3 P0).

Two-pane screen that lets a user browse, save, copy, and drop
every artifact the current chat session has produced. Reads
from a :class:`care.runtime.SessionArtifactStore` passed in by
the host (typically `ChatScreen.artifact_store`).

Layout
------

```
┌── header (CareHeader) ──────────────────────────────────────┐
│ CARE      Chat › Artifacts          v0.1.0  [ 3 · 1 unsaved] │
├──────────────────────────────┬──────────────────────────────┤
│ DataTable                    │  Markdown / pre               │
│ ── Time ── Kind ── Title ──★ │  (selected artifact detail)   │
│   12:01    chain   weather   │                               │
│   12:04    tool    grep      │                               │
│   12:09  ★chain    financier │                               │
├──────────────────────────────┴──────────────────────────────┤
│ footer                                                       │
└──────────────────────────────────────────────────────────────┘
```

Bindings (also shown in `/help`)
--------------------------------

* ``Enter`` — push :class:`care.screens.inspection.InspectionScreen`
  for the currently-focused chain artifact. Non-chain rows
  (stage payload / tool output / synthesised answer) toast a
  hint instead.
* ``s`` — **Save to Memory**. Worker calls
  ``CareMemory.save_chain(payload, name=, tags=)`` then
  :meth:`SessionArtifactStore.mark_saved` so the saved-badge
  flips on the row + the chat header pill drops the unsaved
  count by one. No-op + warning toast on a non-chain row or
  when Memory isn't configured.
* ``d`` — **Delete from session**. In-memory drop; the row
  disappears and the pill shrinks by one. Does NOT touch
  Memory (use `/forget` for that).
* ``c`` — **Copy payload to clipboard** via the OSC-52-or-
  fallback helper in :mod:`care.runtime.clipboard`. Chain
  payloads serialise as pretty JSON; tool / text payloads as
  ``str(payload)``.
* ``Esc`` — pop the screen.

The screen subscribes a per-mount listener to the store so an
append / mark_saved / forget happening in a background worker
(or via the chat header pill click) re-paints the table
without forcing a refresh keystroke.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Pretty,
    Static,
    TabbedContent,
    TabPane,
)

from care.runtime.i18n import t
from care.runtime.session_artifacts import (
    SessionArtifact,
    SessionArtifactStore,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader

if TYPE_CHECKING:
    from care.screens.save_report import SaveReportRow

_log = logging.getLogger("care.screen.artifacts")


_COLUMNS: tuple[str, ...] = (
    "Time", "Kind", "Title", "Saved",
)
"""DataTable column labels. ``Saved`` shows ``★`` for persisted
entries + blank for unsaved."""


_TABS: tuple[tuple[str, str], ...] = (
    ("chain", "Chains"),
    ("tool_output", "Tool Output"),
    ("stage_payload", "Stage Payload"),
)
"""Tab order for the artifact browser — chains first, then tool
output, then stage payloads. Each entry is ``(kind, label)`` where
``kind`` matches a :data:`SessionArtifactKind` and drives both the
per-tab DataTable id (``artifacts-table-<kind>``) and the
``store.list_artifacts`` filter."""


def _format_row(artifact: SessionArtifact) -> tuple[str, ...]:
    """Project one artifact into the DataTable row tuple.

    ``Time`` is HH:MM:SS to keep the column narrow; the full
    ISO timestamp lands in the detail pane. ``Saved`` collapses
    to a single ``★`` glyph + the memory entity id so the user
    can correlate against Memory at a glance.
    """
    time_str = artifact.created_at.strftime("%H:%M:%S")
    saved = (
        f"★ {artifact.memory_entity_id}" if artifact.saved_to_memory
        else ""
    )
    return (time_str, artifact.kind, artifact.title, saved)


def _render_detail(artifact: SessionArtifact) -> str:
    """Pretty-print the artifact body for the detail pane.

    * Chain payloads render as 2-space-indent JSON when JSON-
      serialisable, falling back to `repr` for the rest.
    * Tool / synthesised / dataset payloads render as
      ``str(payload)``.

    The header section above the body lists every
    :class:`SessionArtifact` field so the user has the full
    context (id, kind, origin) in one place.
    """
    if artifact.kind == "chain":
        try:
            body = json.dumps(
            artifact.payload, indent=2, default=str, ensure_ascii=False,
        )
        except Exception:
            body = repr(artifact.payload)
    else:
        body = str(artifact.payload)
    return _render_meta(artifact) + body


def _render_meta(artifact: SessionArtifact) -> str:
    """The header block listing every :class:`SessionArtifact` field, with a
    trailing blank line. Shown above the payload — whether the payload then
    renders inline (this fn) or via the `Pretty` widget."""
    lines = [
        f"id: {artifact.id}",
        f"kind: {artifact.kind}",
        f"created_at: {artifact.created_at.isoformat()}",
        f"title: {artifact.title}",
        f"summary: {artifact.summary}",
        f"saved_to_memory: {artifact.saved_to_memory}",
        f"memory_entity_id: {artifact.memory_entity_id}",
        f"origin: {artifact.origin}",
        "",
    ]
    return "\n".join(lines)


def _chain_steps(artifact: SessionArtifact) -> list[dict]:
    """Pull the step list out of a chain artifact's payload.

    CARL serialises the step graph under ``steps`` (a list of
    dicts). Returns an empty list for any other shape so the DAG
    renderer degrades to ``(empty)`` instead of raising."""
    payload = artifact.payload
    if isinstance(payload, dict):
        steps = payload.get("steps")
        if isinstance(steps, list):
            return [s for s in steps if isinstance(s, dict)]
    return []


def _render_dag_detail(artifact: SessionArtifact) -> str:
    """DAG view of a chain artifact — the same header block as
    :func:`_render_detail` followed by the ASCII step graph from
    :func:`care.screens.inspection.render_chain_dag`.

    Reuses the inspection-screen renderer so the two surfaces show
    an identical topology for the same chain."""
    from care.screens.inspection import render_chain_dag

    lines = [
        f"id: {artifact.id}",
        f"kind: {artifact.kind}",
        f"title: {artifact.title}",
        f"summary: {artifact.summary}",
        "",
    ]
    return "\n".join(lines) + render_chain_dag(_chain_steps(artifact))


class ArtifactsScreen(Screen):
    """Two-pane browser over a :class:`SessionArtifactStore`."""

    DEFAULT_CSS = """
    ArtifactsScreen {
        layers: base;
    }
    ArtifactsScreen #artifacts-body {
        height: 1fr;
    }
    ArtifactsScreen #artifacts-tabs {
        width: 1fr;
        min-width: 28;
        height: 1fr;
    }
    ArtifactsScreen #artifacts-tabs DataTable {
        height: 1fr;
    }
    ArtifactsScreen #artifacts-detail-wrap {
        width: 2fr;
        min-width: 22;
        height: 1fr;
        background: $panel;
    }
    ArtifactsScreen #artifacts-detail-controls {
        height: auto;
        padding: 0 1;
        layout: vertical;
    }
    ArtifactsScreen #artifacts-detail-toggle {
        width: 100%;
        min-width: 8;
        margin-bottom: 1;
    }
    ArtifactsScreen #artifacts-detail-toggle.-hidden {
        display: none;
    }
    ArtifactsScreen #artifacts-detail-save {
        width: 100%;
        min-width: 8;
    }
    ArtifactsScreen #artifacts-detail-save.-hidden {
        display: none;
    }
    ArtifactsScreen #artifacts-detail-scroll {
        height: 1fr;
        width: 100%;
        overflow-y: auto;
    }
    ArtifactsScreen #artifacts-detail {
        padding: 1;
        height: auto;
    }
    ArtifactsScreen #artifacts-empty {
        height: 1fr;
        content-align: center middle;
        text-style: dim italic;
    }
    ArtifactsScreen #artifacts-save-all-row {
        height: auto;
        padding: 0 1;
        background: $panel;
    }
    ArtifactsScreen #artifacts-save-all-btn {
        width: 100%;
        min-width: 8;
    }
    ArtifactsScreen #artifacts-save-all-btn.-hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("s", "save", "Save", show=True),
        Binding("S", "save_all_unsaved", "Save all", show=True),
        Binding("p", "promote_stable", "Promote", show=True),
        Binding("d", "delete_from_session", "Drop", show=True),
        Binding("c", "copy_payload", "Copy", show=True),
        Binding("v", "toggle_view", "JSON/DAG", show=True),
        Binding("enter", "inspect", "Inspect", show=True),
        Binding("space", "toggle_select", "Select", show=True),
        Binding("D", "diff_selected", "Diff", show=True),
    ]

    def __init__(self, store: SessionArtifactStore) -> None:
        super().__init__()
        self.store = store
        # Snapshot list — ordered newest-first to match the
        # chat-header pill semantics. Rebuilt on every paint
        # from `store.list_artifacts()` so external mutations
        # (workers, header-pill click, /clear) stay consistent.
        self._rows: list[SessionArtifact] = []
        # Per-tab row slices, keyed by artifact kind. Rebuilt on
        # every paint alongside `_rows` so the active-tab cursor
        # lookup in `current_artifact` stays in sync with what each
        # per-kind DataTable was populated from.
        self._rows_by_kind: dict[str, list[SessionArtifact]] = {
            kind: [] for kind, _label in _TABS
        }
        # §3 P1 — multi-select set for the `D` (diff) binding.
        # Tracks artifact ids the user has spacebar-marked.
        # Cleared on /clear / Escape-with-empty.
        self._selected_ids: set[str] = set()
        # Detail-pane view mode for chain artifacts — "json" dumps
        # the raw payload, "dag" renders the ASCII step graph.
        # Toggled via the `v` binding / the detail-pane button.
        self._detail_view: str = "json"
        # Action log — tests + future telemetry. Mirrors the
        # pattern in LibraryScreen so a binding press is
        # observable without needing to scrape the screen
        # state.
        self.action_log: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Horizontal(id="artifacts-body"):
            with TabbedContent(id="artifacts-tabs"):
                for kind, label in _TABS:
                    with TabPane(label, id=f"artifacts-tab-{kind}"):
                        yield DataTable(id=f"artifacts-table-{kind}")
            with Vertical(id="artifacts-detail-wrap"):
                with Horizontal(id="artifacts-detail-controls"):
                    # Toggle between the raw JSON dump and the
                    # ASCII DAG view. Only meaningful for chain
                    # artifacts — hidden for other kinds.
                    yield Button(
                        t("artifacts.btn.showDag"),
                        id="artifacts-detail-toggle",
                    )
                    yield Button(
                        t("artifacts.btn.saveToLibrary"),
                        id="artifacts-detail-save",
                        variant="primary",
                    )
                # `VerticalScroll` so long chain payloads / DAGs
                # scroll instead of clipping at the pane height —
                # the bare `Static` couldn't scroll on its own.
                with VerticalScroll(id="artifacts-detail-scroll"):
                    # `markup=False` because the detail body is a
                    # plain-text dump (artifact payload repr, stage
                    # data, JSON snippets) that often contains
                    # literal `[...]` tokens (Python lists, CARL
                    # `step_context_queries`, dict reprs). Rich
                    # would try to parse those as markup tags and
                    # raise `MarkupError`. Same workaround pattern
                    # as iter 67's `★ front` fix on the Pareto
                    # pane; here the safer move is to disable
                    # markup parsing entirely since the content is
                    # never decorated.
                    yield Static(
                        "",
                        id="artifacts-detail",
                        markup=False,
                    )
                    # JSON-like payloads (chain dicts, stage payloads) render
                    # through `Pretty` — an indented, syntax-coloured data
                    # structure — below the meta header. Hidden for the DAG
                    # view and for plain-string payloads (the Static above
                    # carries those).
                    yield Pretty({}, id="artifacts-detail-pretty")
        yield Static(
            t("artifacts.empty"),
            id="artifacts-empty",
        )
        with Horizontal(id="artifacts-save-all-row"):
            yield Button(
                t("artifacts.saveAllMany", count=0),
                id="artifacts-save-all-btn",
                variant="primary",
            )
        yield CareFooter()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="ArtifactsScreen",
                breadcrumb=(
                    t("artifacts.breadcrumb.chat"),
                    t("artifacts.breadcrumb.artifacts"),
                ),
            )
        except Exception:
            pass
        # Add columns + cursor type post-mount — calling
        # `add_column` on a not-yet-mounted DataTable silently
        # drops the column in the current Textual.
        for kind, _label in _TABS:
            try:
                table = self.query_one(
                    f"#artifacts-table-{kind}", DataTable,
                )
                for col in _COLUMNS:
                    table.add_column(
                        t(f"artifacts.column.{col.lower()}"), key=col,
                    )
                table.cursor_type = "row"
                table.zebra_stripes = True
            except Exception:
                pass
        self.store.add_listener(self._on_store_event)
        # Defer the initial refresh — `is_mounted` returns
        # False during `on_mount` itself, so a sync call would
        # bail out via the early-return guard inside
        # `refresh_rows`.
        self.app.call_after_refresh(self.refresh_rows)

    def on_unmount(self) -> None:
        # Unsubscribe so a popped screen doesn't keep painting
        # on background events. `remove_listener` is a silent
        # no-op when the listener wasn't registered.
        try:
            self.store.remove_listener(self._on_store_event)
        except Exception:
            pass

    def _on_store_event(self, _artifact: SessionArtifact) -> None:
        """Listener wired in `on_mount`. Marshals the repaint
        to the Textual loop because background workers
        (`append_chain` from a CARL streamer) may fire from a
        non-UI thread."""
        try:
            self.app.call_from_thread(self.refresh_rows)
        except Exception:
            self.refresh_rows()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def refresh_rows(self) -> None:
        """Re-read the store, re-populate the DataTable + empty
        state. Idempotent — called on mount, on store events,
        and after every action that mutates the store."""
        self._rows = self.store.list_artifacts()
        # Partition once per paint so the active-tab cursor lookup
        # in `current_artifact` indexes into the same ordered slice
        # the table was populated from.
        self._rows_by_kind = {
            kind: [a for a in self._rows if a.kind == kind]
            for kind, _label in _TABS
        }
        for kind, _label in _TABS:
            try:
                table = self.query_one(
                    f"#artifacts-table-{kind}", DataTable,
                )
            except Exception:
                continue
            table.clear()
            for artifact in self._rows_by_kind[kind]:
                table.add_row(*_format_row(artifact), key=artifact.id)
        # Toggle empty-state visibility — failures here are
        # cosmetic (e.g. widget not mounted yet during a fast
        # repaint cycle); refresh succeeded as long as the
        # table got the new rows.
        try:
            empty = self.query_one("#artifacts-empty", Static)
            empty.display = not self._rows
        except Exception:
            pass
        self._refresh_save_all_button()
        self._refresh_detail()

    def _refresh_save_all_button(self) -> None:
        """Update the "Save all unsaved (N)" button label +
        visibility from the current store snapshot.

        Hidden when no chain artifact is unsaved (`-hidden` CSS
        class toggle). The save_all bulk worker reads the same
        snapshot when fired, so what the user clicks on matches
        what we attempt to save.
        """
        try:
            btn = self.query_one(
                "#artifacts-save-all-btn", Button,
            )
        except Exception:
            return
        unsaved = [a for a in self._rows if a.kind == "chain" and not a.saved_to_memory]
        count = len(unsaved)
        key = "artifacts.saveAllOne" if count == 1 else "artifacts.saveAllMany"
        btn.label = t(key, count=count)
        if count == 0:
            btn.add_class("-hidden")
        else:
            btn.remove_class("-hidden")

    def _refresh_detail(self) -> None:
        """Sync the detail pane to whatever the table cursor
        points at. Safe to call when the table is empty —
        renders a one-line placeholder. Honors the JSON/DAG view
        toggle for chain artifacts; other kinds always render the
        plain dump (the toggle hides itself for them)."""
        try:
            detail = self.query_one("#artifacts-detail", Static)
        except Exception:
            return
        try:
            pretty = self.query_one("#artifacts-detail-pretty", Pretty)
        except Exception:
            pretty = None
        artifact = self.current_artifact
        self._refresh_detail_toggle(artifact)

        def _show_pretty(visible: bool) -> None:
            if pretty is not None and pretty.display != visible:
                pretty.display = visible

        if artifact is None:
            detail.update("")
            _show_pretty(False)
            return
        if artifact.kind == "chain" and self._detail_view == "dag":
            detail.update(_render_dag_detail(artifact))
            _show_pretty(False)
        elif pretty is not None and isinstance(artifact.payload, dict):
            # JSON-like (dict) payload → meta header in the Static, data
            # structure in the Pretty widget below it.
            detail.update(_render_meta(artifact))
            pretty.update(artifact.payload)
            _show_pretty(True)
        else:
            detail.update(_render_detail(artifact))
            _show_pretty(False)

    def _refresh_detail_toggle(
        self, artifact: SessionArtifact | None,
    ) -> None:
        """Show the JSON/DAG toggle only for chain artifacts and
        sync its label to the *next* view the user would switch
        to. Also drives the sibling "Save to library" button so
        both stay aligned with the active row."""
        self._refresh_save_btn(artifact)
        try:
            btn = self.query_one(
                "#artifacts-detail-toggle", Button,
            )
        except Exception:
            return
        if artifact is None or artifact.kind != "chain":
            btn.add_class("-hidden")
            return
        btn.remove_class("-hidden")
        btn.label = (
            t("artifacts.btn.showJson")
            if self._detail_view == "dag"
            else t("artifacts.btn.showDag")
        )

    def _refresh_save_btn(
        self, artifact: SessionArtifact | None,
    ) -> None:
        """Toggle the detail-pane "Save to library" button so it
        only appears for an unsaved chain row. Mirrors the
        existing `s` keybinding — the button is just a more
        discoverable trigger for the same `action_save` flow.
        Saved chains show a dimmed label so the user can see the
        row's status without losing the slot in the controls bar.
        """
        try:
            btn = self.query_one(
                "#artifacts-detail-save", Button,
            )
        except Exception:
            return
        if artifact is None or artifact.kind != "chain":
            btn.add_class("-hidden")
            return
        btn.remove_class("-hidden")
        if artifact.saved_to_memory:
            btn.label = t("artifacts.btn.savedToLibrary")
            btn.disabled = True
        else:
            btn.label = t("artifacts.btn.saveToLibrary")
            btn.disabled = False

    def _active_kind(self) -> str:
        """The artifact kind backing the currently-selected tab.

        Falls back to the first tab (``chain``) when the
        TabbedContent isn't mounted yet or its active id doesn't
        match a known tab — keeps cursor / detail lookups working
        during early-mount repaints."""
        try:
            tabs = self.query_one("#artifacts-tabs", TabbedContent)
            active = tabs.active
        except Exception:
            return _TABS[0][0]
        for kind, _label in _TABS:
            if active == f"artifacts-tab-{kind}":
                return kind
        return _TABS[0][0]

    @property
    def current_artifact(self) -> SessionArtifact | None:
        """The artifact under the active tab's DataTable cursor, or
        ``None`` when that tab is empty / cursor out of range."""
        kind = self._active_kind()
        rows = self._rows_by_kind.get(kind, [])
        if not rows:
            return None
        try:
            table = self.query_one(
                f"#artifacts-table-{kind}", DataTable,
            )
        except Exception:
            return None
        idx = table.cursor_row
        if idx is None or idx < 0 or idx >= len(rows):
            return None
        return rows[idx]

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted,
    ) -> None:
        """Cursor moved → repaint detail pane."""
        _ = event
        self._refresh_detail()

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected,
    ) -> None:
        """Enter on a row → inspect chain artifacts."""
        _ = event
        self.action_inspect()

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated,
    ) -> None:
        """Switching tabs → repaint the shared detail pane against
        the newly-active tab's cursor."""
        _ = event
        self._refresh_detail()

    # ------------------------------------------------------------------
    # Bindings
    # ------------------------------------------------------------------

    def action_back(self) -> None:
        self.action_log.append(("back", ""))
        try:
            self.app.pop_screen()
        except Exception:
            pass

    def action_inspect(self) -> None:
        artifact = self.current_artifact
        if artifact is None:
            return
        self.action_log.append(("inspect", artifact.id))
        if artifact.kind != "chain":
            self._toast(
                t("artifacts.toast.inspectChainOnly", kind=artifact.kind),
                severity="warning",
            )
            return
        if not artifact.saved_to_memory or not artifact.memory_entity_id:
            self._toast(
                t("artifacts.toast.inspectNeedsSaved"),
                severity="warning",
            )
            return
        try:
            from care.screens.inspection import InspectionScreen

            self.app.push_screen(
                InspectionScreen(artifact.memory_entity_id),
            )
        except Exception as exc:  # noqa: BLE001
            self._toast(
                t("artifacts.toast.inspectFailed", error=exc),
                severity="error",
            )

    def action_save(self) -> None:
        artifact = self.current_artifact
        if artifact is None:
            return
        self.action_log.append(("save", artifact.id))
        if artifact.kind != "chain":
            self._toast(
                t("artifacts.toast.saveChainOnly", kind=artifact.kind),
                severity="warning",
            )
            return
        if artifact.saved_to_memory:
            self._toast(
                t("artifacts.toast.alreadySaved", id=artifact.memory_entity_id),
                severity="info",
            )
            return
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._toast(
                t("artifacts.toast.saveNeedsMemory"),
                severity="error",
            )
            return
        # §3 P2 — push the TagEditorModal before saving so the
        # user can attach tags on persistence. Modal Cancel
        # aborts the save (no tags ≠ "save without tags");
        # Apply with empty add_tags = save without tags.
        try:
            from care.screens.tag_editor import (
                TagEditorModal,
                TagEditorResult,
            )
        except Exception:
            # If TagEditorModal can't be imported for any
            # reason, fall back to the legacy direct-save path
            # so the user isn't blocked.
            self.run_worker(
                self._save_worker(artifact, memory),
                name="artifacts_save",
                group="artifacts",
                exclusive=False,
                exit_on_error=False,
            )
            return

        # §3 P3 — pre-compute the LLM-suggested chain name
        # before pushing the modal so the user can accept /
        # tweak / clear it inside the save flow. Falls back to
        # the artifact's bootstrap title when the LLM is
        # unavailable. Synchronous helper; failures degrade
        # silently to the existing artifact title.
        suggested_title = self._suggest_chain_name(
            artifact, fallback=artifact.title or "",
        )

        def _on_tags(result: TagEditorResult | None) -> None:
            if result is None or not getattr(result, "submitted", False):
                self._toast(
                    t("artifacts.toast.saveCancelled"), severity="info",
                )
                return
            tags = tuple(getattr(result, "add_tags", ()) or ())
            # §3 P3 — when the modal carries a non-empty
            # `title`, the user accepted or edited the
            # suggestion; we forward it to `_save_worker` as
            # the explicit save name. Empty `title` means the
            # user cleared the field — fall back to the
            # artifact's bootstrap title so the save never
            # winds up nameless.
            picked_title = (
                str(getattr(result, "title", "") or "").strip()
                or suggested_title
                or artifact.title
            )
            self.run_worker(
                self._save_worker(
                    artifact, memory,
                    tags=tags, name=picked_title,
                ),
                name="artifacts_save",
                group="artifacts",
                exclusive=False,
                exit_on_error=False,
            )

        self.app.push_screen(
            TagEditorModal(
                initial_tags=(),
                target_count=1,
                initial_title=suggested_title,
            ),
            _on_tags,
        )

    async def _save_worker(
        self,
        artifact: SessionArtifact,
        memory: Any,
        *,
        tags: tuple[str, ...] = (),
        name: str = "",
    ) -> None:
        try:
            entity_id = await self._run_save(
                artifact, memory, tags=tags, name=name,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "ArtifactsScreen save failed for id=%s: %s",
                artifact.id, exc, exc_info=False,
            )
            self._toast(
                t("artifacts.toast.saveFailed", error=exc),
                severity="error",
            )
            return
        if not entity_id:
            self._toast(
                t("artifacts.toast.saveNoEntityId"), severity="warning",
            )
            return
        try:
            self.store.mark_saved(
                artifact.id, memory_entity_id=str(entity_id),
            )
        except Exception as exc:  # noqa: BLE001
            self._toast(
                t("artifacts.toast.savedMarkFailed", error=exc),
                severity="warning",
            )
            return
        self._toast(
            t("artifacts.toast.saved", id=entity_id), severity="success",
        )
        # §3 P0 — post-save "Use it now" reveal. Push a
        # modal showing the stable chain_id + integration
        # snippets so the user reaches an external-service
        # call recipe in ≤ 2 keystrokes from /artifacts.
        # Pure presentation; the dismiss callback hands the
        # evolve-request off to the host's existing
        # EvolutionLaunchModal opener.
        self._push_use_it_now(
            entity_id=str(entity_id),
            display_name=artifact.title,
        )

    def _push_use_it_now(
        self, *, entity_id: str, display_name: str = "",
    ) -> None:
        try:
            from care.screens.use_it_now import (
                UseItNowModal,
                UseItNowResult,
            )
        except Exception:
            return

        memory = getattr(self.app, "memory", None)
        base_url = ""
        if memory is not None:
            # Try the most common shapes for the base URL —
            # production CareMemory exposes `.client.base_url`
            # but tests / older SDKs may put it elsewhere.
            client = getattr(memory, "client", None)
            base_url = (
                str(getattr(client, "base_url", "") or "")
                or str(getattr(memory, "base_url", "") or "")
            )

        def _on_dismiss(result: UseItNowResult | None) -> None:
            if result is None:
                return
            if result.evolve_requested:
                self._open_evolution_launch(entity_id)

        self.app.push_screen(
            UseItNowModal(
                entity_id=entity_id,
                display_name=display_name,
                memory_base_url=base_url,
            ),
            _on_dismiss,
        )

    def _open_evolution_launch(self, entity_id: str) -> None:
        """Open the §4 EvolutionLaunchModal pre-filled with
        ``entity_id``. Routes through the app's existing
        ``_push_evolution_for`` opener when available, else
        falls back to a friendly toast pointing at the
        slash command."""
        opener = getattr(self.app, "_push_evolution_for", None)
        if callable(opener):
            try:
                opener(entity_id)
                return
            except Exception as exc:  # noqa: BLE001
                self._toast(
                    t("artifacts.toast.openEvolutionFailed", error=exc),
                    severity="error",
                )
                return
        self._toast(
            t("artifacts.toast.openEvolutionHint", id=entity_id),
            severity="info",
        )

    def _suggest_chain_name(
        self, artifact: SessionArtifact, *, fallback: str,
    ) -> str:
        """§3 P2 — return a one-line LLM-suggested title for
        the artifact's chain payload, or ``fallback`` on any
        failure / non-chain artifact / unconfigured LLM.

        Synchronous because :class:`care.runtime.chain_title.suggest_chain_title`
        wraps a sync `OpenAI` client. Called from inside the
        sync portion of `_run_save` (just before the
        `asyncio.to_thread(save_chain, …)` dispatch) so the
        suggestion lands as the saved-entity name without
        forking a worker.

        Best-effort throughout — failures degrade silently
        + return the fallback so the save never blocks on a
        bad model / network / config.
        """
        if artifact.kind != "chain":
            return fallback
        try:
            cfg = getattr(self.app, "config", None)
            mage_cfg = getattr(cfg, "mage", None) if cfg else None
            if mage_cfg is None:
                return fallback
            from care.runtime.chain_title import suggest_chain_title
            from care.runtime.llm_client import build_llm_client
        except Exception:
            return fallback
        try:
            client = build_llm_client(mage_cfg)
        except Exception:
            return fallback
        model = str(getattr(mage_cfg, "model", "") or "")
        if not model:
            return fallback
        return suggest_chain_title(
            artifact.payload,
            client=client,
            model=model,
            fallback=fallback,
        )

    async def _run_save(
        self,
        artifact: SessionArtifact,
        memory: Any,
        *,
        tags: tuple[str, ...] = (),
        name: str = "",
    ) -> str | None:
        """`memory.save_chain` is sync in the current facade
        but the test scaffold sometimes exposes an async stub.
        Marshal both shapes via `asyncio.to_thread` for sync +
        `await` for async. Optional ``tags`` are forwarded when
        non-empty (§3 P2 tag-editor-on-save flow).

        ``name`` (§3 P3) — when supplied, used verbatim as the
        ``save_chain`` ``name=`` kwarg. The save-flow path in
        `_run_save_called_from_run_save_button` pre-computes the
        LLM suggestion and surfaces it through the
        TagEditorModal so the user can accept / tweak / clear
        before persistence; the result string lands here. Empty
        ``name`` falls back to the LLM helper + artifact title
        (legacy behaviour from §3 P2) so the bulk save-all path
        and any non-modal call site stay unchanged.
        """
        import asyncio
        import inspect as _inspect

        fn = getattr(memory, "save_chain", None)
        if fn is None:
            raise RuntimeError(
                "memory facade has no save_chain method",
            )
        if name:
            resolved_name = name
        else:
            # Legacy path (bulk save-all, missing-modal): try
            # the LLM suggestion + fall back to the artifact's
            # bootstrap title. Best-effort throughout.
            resolved_name = (
                self._suggest_chain_name(
                    artifact, fallback=artifact.title or "",
                )
                or artifact.title
                or None
            )
        kwargs: dict[str, Any] = {"name": resolved_name}
        if tags:
            kwargs["tags"] = list(tags)
        if _inspect.iscoroutinefunction(fn):
            entity_id = await fn(artifact.payload, **kwargs)
        else:
            entity_id = await asyncio.to_thread(
                fn, artifact.payload, **kwargs,
            )
        return entity_id

    def action_save_all_unsaved(self) -> None:
        """§3 P0 Save-all flow — persist every unsaved chain
        artifact in the store via the configured Memory
        facade. Per-row toast on success / failure so the
        user knows what happened to each one. Skips silently
        when nothing's unsaved (the button hides itself in
        that state).

        §3 P2 — opens a single :class:`TagEditorModal` at the
        head of the flow so a batch tag (e.g.
        ``domain:weather``) can be applied to every
        persistence in one gesture. Cancel aborts the
        entire save-all; Apply-with-empty-tags falls through
        to the legacy no-tags path."""
        self.action_log.append(("save_all_unsaved", ""))
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._toast(
                t("artifacts.toast.saveAllNeedsMemory"),
                severity="error",
            )
            return
        # Snapshot now — the store may mutate during the
        # worker run (new artifacts arriving from the chat
        # screen, manual `mark_saved` from another path).
        # Saving against the snapshot keeps the behaviour
        # predictable: the user clicked at a moment when N
        # rows were unsaved, we attempt those N.
        unsaved = [
            a for a in self.store.list_artifacts()
            if a.kind == "chain" and not a.saved_to_memory
        ]
        if not unsaved:
            self._toast(t("artifacts.toast.nothingToSave"), severity="info")
            return

        try:
            from care.screens.tag_editor import (
                TagEditorModal,
                TagEditorResult,
            )
        except Exception:
            # Modal unavailable → fall back to the legacy
            # direct save so users aren't blocked by a
            # broken module.
            self.run_worker(
                self._save_all_worker(unsaved, memory),
                name="artifacts_save_all",
                group="artifacts",
                exclusive=False,
                exit_on_error=False,
            )
            return

        def _on_tags(result: TagEditorResult | None) -> None:
            if result is None or not getattr(
                result, "submitted", False,
            ):
                self._toast(
                    t("artifacts.toast.saveAllCancelled"), severity="info",
                )
                return
            tags = tuple(getattr(result, "add_tags", ()) or ())
            self.run_worker(
                self._save_all_worker(
                    unsaved, memory, tags=tags,
                ),
                name="artifacts_save_all",
                group="artifacts",
                exclusive=False,
                exit_on_error=False,
            )

        # §3 P3 — seed the modal with the UNION of tags
        # already attached to the unsaved chains so the user
        # can see and tweak the existing labels instead of
        # re-typing them. Insertion order preserved + deduped.
        seed_tags: list[str] = []
        for art in unsaved:
            for tag in getattr(art, "tags", ()) or ():
                cleaned = str(tag).strip()
                if cleaned and cleaned not in seed_tags:
                    seed_tags.append(cleaned)
        self.app.push_screen(
            TagEditorModal(
                initial_tags=tuple(seed_tags),
                target_count=len(unsaved),
            ),
            _on_tags,
        )

    async def _save_all_worker(
        self,
        unsaved: list[SessionArtifact],
        memory: Any,
        *,
        tags: tuple[str, ...] = (),
    ) -> None:
        """Iterate the snapshot + save each artifact through
        the same `_run_save` / `mark_saved` chain the per-row
        `action_save` uses. Per-artifact failures are toasted
        + counted but don't abort the loop — the user gets a
        summary toast at the end (`"saved N of M"`).

        §3 P1 — For large batches (≥ 5 artifacts) or any
        failure, also pushes the `SaveReport` modal so the
        user gets a scannable post-mortem. Small all-success
        batches keep the toast-only flow."""
        from care.screens.save_report import SaveReportRow

        saved_count = 0
        failed_count = 0
        report_rows: list[SaveReportRow] = []
        for artifact in unsaved:
            try:
                entity_id = await self._run_save(
                    artifact, memory, tags=tags,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "save_all: artifact %s failed: %s",
                    artifact.id, exc, exc_info=False,
                )
                self._toast(
                    t(
                        "artifacts.toast.saveFailedFor",
                        title=repr(artifact.title), error=exc,
                    ),
                    severity="warning",
                )
                failed_count += 1
                report_rows.append(SaveReportRow(
                    artifact_id=artifact.id,
                    title=artifact.title,
                    status="failure",
                    error=str(exc)[:280],
                ))
                continue
            if not entity_id:
                failed_count += 1
                self._toast(
                    t(
                        "artifacts.toast.saveNoEntityIdFor",
                        title=repr(artifact.title),
                    ),
                    severity="warning",
                )
                report_rows.append(SaveReportRow(
                    artifact_id=artifact.id,
                    title=artifact.title,
                    status="failure",
                    error="save returned no entity_id",
                ))
                continue
            try:
                self.store.mark_saved(
                    artifact.id, memory_entity_id=str(entity_id),
                )
            except Exception as exc:  # noqa: BLE001
                self._toast(
                    t(
                        "artifacts.toast.savedMarkFailedFor",
                        title=repr(artifact.title), error=exc,
                    ),
                    severity="warning",
                )
                # Still counts as saved on Memory's side.
                saved_count += 1
                report_rows.append(SaveReportRow(
                    artifact_id=artifact.id,
                    title=artifact.title,
                    status="success",
                    entity_id=str(entity_id),
                    error=(
                        f"saved but local store update "
                        f"failed: {exc}"[:280]
                    ),
                ))
                continue
            saved_count += 1
            self._toast(
                t(
                    "artifacts.toast.savedFor",
                    title=repr(artifact.title), id=entity_id,
                ),
                severity="success",
            )
            report_rows.append(SaveReportRow(
                artifact_id=artifact.id,
                title=artifact.title,
                status="success",
                entity_id=str(entity_id),
            ))
        # Final summary toast — surfaces partial-success
        # cases that a per-row toast stream would bury.
        if failed_count == 0:
            key = (
                "artifacts.toast.savedAllOne"
                if saved_count == 1
                else "artifacts.toast.savedAllMany"
            )
            self._toast(
                t(key, count=saved_count),
                severity="success",
            )
        else:
            self._toast(
                t(
                    "artifacts.toast.savedPartial",
                    saved=saved_count,
                    total=len(unsaved),
                    failed=failed_count,
                ),
                severity="warning",
            )
        # §3 P1 — push SaveReport modal for large batches
        # OR on any failure. Small clean batches keep the
        # existing toast-only flow.
        if len(unsaved) >= 5 or failed_count > 0:
            self._push_save_report(tuple(report_rows))

    def _push_save_report(
        self, rows: tuple["SaveReportRow", ...],
    ) -> None:
        """Push the §3 P1 SaveReport modal. On dismiss with
        ``show_id`` populated, route the artifact to its
        UseItNowModal via the existing `_push_use_it_now`
        helper."""
        try:
            from care.screens.save_report import (
                SaveReport,
                SaveReportResult,
            )
        except Exception:
            return

        def _on_dismiss(result: SaveReportResult | None) -> None:
            if result is None or not getattr(
                result, "show_id", "",
            ):
                return
            try:
                artifact = self.store.get(result.show_id)
            except Exception:
                return
            entity_id = artifact.memory_entity_id or ""
            if not entity_id:
                return
            self._push_use_it_now(
                entity_id=str(entity_id),
                display_name=artifact.title,
            )

        try:
            self.app.push_screen(SaveReport(rows), _on_dismiss)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "couldn't push SaveReport modal: %s",
                exc, exc_info=False,
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Wire the detail-pane buttons to their keyboard
        equivalents."""
        if event.button.id == "artifacts-save-all-btn":
            self.action_save_all_unsaved()
        elif event.button.id == "artifacts-detail-toggle":
            self.action_toggle_view()
        elif event.button.id == "artifacts-detail-save":
            self.action_save()

    def action_toggle_view(self) -> None:
        """`v` / detail-pane button — flip the chain detail pane
        between the raw JSON dump and the ASCII DAG view. No-op
        with a hint for non-chain artifacts (a DAG only makes
        sense for chains)."""
        artifact = self.current_artifact
        if artifact is None:
            return
        if artifact.kind != "chain":
            self._toast(
                t("artifacts.toast.dagChainOnly", kind=artifact.kind),
                severity="info",
            )
            return
        self._detail_view = "dag" if self._detail_view == "json" else "json"
        self.action_log.append(("toggle_view", self._detail_view))
        self._refresh_detail()

    def action_toggle_select(self) -> None:
        """`Space` — toggle the highlighted row's bulk-select
        mark. Only chain artifacts can be marked (diff is
        chain-only). Surfaces a hint toast when the user
        marks a third row so the cap is discoverable."""
        artifact = self.current_artifact
        if artifact is None:
            return
        if artifact.kind != "chain":
            self._toast(
                t("artifacts.toast.selectChainOnly", kind=artifact.kind),
                severity="warning",
            )
            return
        if artifact.id in self._selected_ids:
            self._selected_ids.discard(artifact.id)
        else:
            self._selected_ids.add(artifact.id)
        self.action_log.append(("toggle_select", artifact.id))
        if len(self._selected_ids) > 2:
            self._toast(
                t(
                    "artifacts.toast.selectCapHint",
                    count=len(self._selected_ids),
                ),
                severity="info",
            )

    def action_diff_selected(self) -> None:
        """`D` — open a DiffModal against the two selected
        chain artifacts. Refuses on selection sizes < 2 (with
        a hint) and quietly truncates to the first two when
        more are selected so the user doesn't have to clear +
        retry."""
        self.action_log.append(("diff_selected", ""))
        ids = [
            i for i in self._selected_ids
            if any(a.id == i for a in self._rows)
        ]
        if len(ids) < 2:
            self._toast(
                t("artifacts.toast.diffNeedsTwo"),
                severity="warning",
            )
            return
        # Stable ordering — pick the two oldest-marked ids so
        # repeated `D` presses on the same selection produce
        # identical headers / panes.
        ids = sorted(ids)[:2]
        left = next(a for a in self._rows if a.id == ids[0])
        right = next(a for a in self._rows if a.id == ids[1])
        try:
            from care.screens.diff import DiffModal

            self.app.push_screen(
                DiffModal(
                    left_entity_id=left.id,
                    right_entity_id=right.id,
                    left_label=left.title or left.id,
                    right_label=right.title or right.id,
                    left_payload=left.payload,
                    right_payload=right.payload,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._toast(
                t("artifacts.toast.diffOpenFailed", error=exc),
                severity="error",
            )

    def action_promote_stable(self) -> None:
        """§3 P1 — Pin the highlighted saved chain to the
        stable Memory channel. Refuses on unsaved chains
        (nothing to promote yet) + non-chain artifact kinds
        (stage payloads / tool outputs aren't chain entities).
        Dispatches a worker so a slow network call doesn't
        block the UI."""
        artifact = self.current_artifact
        if artifact is None:
            return
        self.action_log.append(("promote_stable", artifact.id))
        if artifact.kind != "chain":
            self._toast(
                t("artifacts.toast.promoteChainOnly", kind=artifact.kind),
                severity="warning",
            )
            return
        if not artifact.saved_to_memory or not artifact.memory_entity_id:
            self._toast(
                t("artifacts.toast.promoteNeedsSaved"),
                severity="warning",
            )
            return
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._toast(
                t("artifacts.toast.promoteNeedsMemory"),
                severity="error",
            )
            return
        self.run_worker(
            self._promote_worker(artifact, memory),
            name="artifacts_promote",
            group="artifacts",
            exclusive=False,
            exit_on_error=False,
        )

    async def _promote_worker(
        self, artifact: SessionArtifact, memory: Any,
    ) -> None:
        import asyncio

        entity_id = artifact.memory_entity_id or ""
        fn = getattr(memory, "promote_to_stable", None)
        if not callable(fn):
            self._toast(
                t("artifacts.toast.promoteUnsupported"),
                severity="warning",
            )
            return
        try:
            await asyncio.to_thread(fn, entity_id)
        except NotImplementedError as exc:
            self._toast(
                t("artifacts.toast.promoteNotSupported", error=exc),
                severity="warning",
            )
            return
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "ArtifactsScreen promote failed for entity=%s: %s",
                entity_id, exc, exc_info=False,
            )
            self._toast(
                t("artifacts.toast.promoteFailed", error=exc),
                severity="error",
            )
            return
        self._toast(
            t("artifacts.toast.promoted", id=entity_id),
            severity="success",
        )

    def action_delete_from_session(self) -> None:
        artifact = self.current_artifact
        if artifact is None:
            return
        self.action_log.append(("delete_from_session", artifact.id))
        try:
            self.store.forget(artifact.id)
        except Exception as exc:  # noqa: BLE001
            self._toast(
                t("artifacts.toast.dropFailed", error=exc),
                severity="warning",
            )
            return
        self._toast(
            t("artifacts.toast.dropped", title=repr(artifact.title)),
            severity="info",
        )
        self.refresh_rows()

    def action_copy_payload(self) -> None:
        artifact = self.current_artifact
        if artifact is None:
            return
        self.action_log.append(("copy_payload", artifact.id))
        if artifact.kind == "chain":
            try:
                text = json.dumps(
                    artifact.payload, indent=2, default=str,
                    ensure_ascii=False,
                )
            except Exception:
                text = repr(artifact.payload)
        else:
            text = str(artifact.payload)
        try:
            from care.runtime.clipboard import copy_text

            copy_text(text)
        except Exception as exc:  # noqa: BLE001
            self._toast(
                t("artifacts.toast.copyFailed", error=exc),
                severity="warning",
            )
            return
        self._toast(t("artifacts.toast.copied"), severity="info")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _toast(self, message: str, *, severity: str = "info") -> None:
        """Thin wrapper over `app.push_toast` that no-ops on a
        bare test host (no toast facade)."""
        push = getattr(self.app, "push_toast", None)
        if callable(push):
            try:
                push(message, severity=severity)
                return
            except Exception:
                pass
        # Fallback: log so the message isn't silently lost.
        _log.info(
            "ArtifactsScreen toast [%s]: %s", severity, message,
        )


__all__ = ["ArtifactsScreen"]
