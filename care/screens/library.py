"""LibraryScreen — saved-agent DataTable (TODO §1.1 P0.7).

The user's home screen — a full-screen DataTable of every
saved agent in their namespace. This sub-task ships the
**scaffold** + **table population**: Header / sidebar
placeholder / DataTable / Footer composition + a fetch worker
that populates rows from `fetch_library_view(...)`.

Per-row actions (P0.11), bulk-select (P0.13), sort (P0.10),
empty-state rendering (P0.9), and the sidebar's filter chips
(P0.8 / P0.14) land in subsequent sub-tasks — each consumes a
shipped data layer (`row_actions`, `bulk_ops`, `library_view`
persistence, `empty_state`, `collections`) and turns a
`[DONE — data layer]` bullet into fully DONE.

The screen reads `app.memory`; an unconfigured (`None`) facade
leaves the table empty rather than crashing — the SettingsScreen
(P0.32) is responsible for resolving credentials before the user
lands here.
"""

from __future__ import annotations

import inspect
from collections import Counter
from datetime import datetime
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Static, TabbedContent, TabPane

from care.runtime.bulk_ops import (
    BulkOperationResult,
    BulkSelection,
    BulkTarget,
    apply_delete,
    apply_favourite,
    apply_tag_edits,
)
from care.runtime.collections import (
    Collection,
    filter_by_collection,
    list_collections,
)
from care.runtime.empty_state import EmptyState, classify_empty_state
from care.runtime.i18n import t
from care.runtime.library_view import (
    LibraryFilters,
    LibraryRow,
    LibrarySort,
    LibraryView,
    LibraryViewError,
    LibraryViewState,
    clear_filters,
    fetch_library_view,
    load_view_state,
    save_view_state,
)
from care.runtime.row_actions import (
    RowAction,
    RowActionKind,
    RowMutationOutcome,
    actions_for_row,
    delete_row,
    duplicate_chain,
    toggle_favourite_row,
)
from care.screens.confirm import ConfirmModal
from care.widgets.context_menu import ContextMenu
from care.widgets.empty_state import EmptyStateView
from care.runtime.global_bindings import (
    GlobalBinding,
    default_global_bindings,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader
from care.widgets.library_sidebar import LibrarySidebar


def _rank_tags_by_frequency(
    rows: tuple[LibraryRow, ...] | list[LibraryRow],
) -> tuple[str, ...]:
    """Frequency-rank tags harvested from ``rows[*].tags``.

    Order: count desc, then alphabetical as a deterministic
    tiebreaker so back-to-back refreshes don't shuffle chip
    positions when several tags tie at the same count.
    """
    counts: Counter[str] = Counter()
    for row in rows:
        for tag in row.tags:
            cleaned = tag.strip()
            if cleaned:
                counts[cleaned] += 1
    if not counts:
        return ()
    return tuple(
        tag for tag, _ in sorted(
            counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    )


class LibraryScreen(Screen):
    """Full-screen DataTable of saved agents.

    The screen owns the `filters` + `sort` state (mutated by
    future P0.8 / P0.10 sub-tasks) and re-runs the fetch
    worker on every change. The DataTable is the canonical
    list-of-agents render; the sidebar to the left hosts
    collection chips + filter toggles (P0.8 / P0.14).
    """

    DEFAULT_CSS = """
    LibraryScreen {
        layout: vertical;
    }
    LibraryScreen #library-tabs {
        height: 1fr;
    }
    LibraryScreen #library-main {
        height: 1fr;
    }
    LibraryScreen #library-sessions {
        height: 1fr;
    }
    LibraryScreen #library-sessions-table {
        width: 1fr;
        height: 1fr;
    }
    LibraryScreen #library-sessions-empty {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        text-align: center;
        color: $text-muted;
    }
    LibraryScreen LibrarySidebar {
        width: 28;
    }
    LibraryScreen #library-content {
        width: 1fr;
        height: 1fr;
    }
    LibraryScreen #library-table {
        width: 1fr;
        height: 1fr;
    }
    LibraryScreen EmptyStateView {
        width: 1fr;
        height: 1fr;
    }
    LibraryScreen #library-actions {
        height: 3;
        padding: 0 1;
        align-horizontal: left;
    }
    """

    BINDINGS = [
        Binding("ctrl+f", "focus_search", "Search", show=False),
        # `/` mirrors the vim / chat-tool convention. A
        # distinct action routes through the sidebar's
        # one-shot absorber so the activating `/` keystroke
        # (which Textual redelivers to the focused widget on
        # the next dispatch tick) is dropped + the prompt
        # opens empty.
        Binding(
            "slash", "focus_search_absorb", "Search",
            show=False,
        ),
        # P0.11 per-row keyboard actions. Keys mirror
        # `default_actions()` `key_binding` fields. `Enter` is
        # not listed because Textual's DataTable consumes it
        # for row selection — we react to RowSelected instead.
        Binding("r", "row_run", "Run", show=False),
        Binding("e", "row_edit", "Edit", show=False),
        # NL-driven AI edit — hands off to chat's `/revise <id>` flow.
        Binding("R", "row_revise", "Revise (AI)", show=False),
        Binding("d", "row_duplicate", "Duplicate", show=False),
        Binding("v", "row_evolve", "Evolve", show=False),
        # ``Z`` stops every running evolution of the focused
        # chain and adds those runs to the dashboard's archive.
        # Mirrors the ``z`` "Stop + Archive" binding on the
        # EvolutionScreen — handy when the user wants to clean
        # up multiple stale runs without opening each one.
        Binding(
            "Z", "row_archive_evolutions", "Archive runs", show=False,
        ),
        # §4 P0 — `E` (uppercase) is the spec'd binding for
        # "Evolve with my data", routing through the same
        # `_push_evolution_for` chain as `v`. Kept distinct
        # from lowercase `e` (Edit) so keyboard users have
        # both the canonical mnemonic and the legacy `v`
        # binding available.
        Binding("E", "row_evolve", "Evolve with my data", show=False),
        Binding("l", "row_show_lineage", "Lineage", show=False),
        Binding("f", "row_toggle_favourite", "Favourite", show=False),
        Binding("delete", "row_delete", "Delete", show=True),
        # P0.12 context menu — `Menu` key (rarely on laptops,
        # so we surface `Ctrl+M` as the canonical chord) and
        # right-click open the same menu via on_click.
        Binding("ctrl+m", "row_context_menu", "Menu", show=False),
        # P0.13 bulk-select mode. `Space` toggles the focused
        # row into / out of the selection; `T` opens the
        # tag-edit modal (P0.22 — currently logged). `F` and
        # `Delete` route to bulk variants when the selection
        # is non-empty; otherwise they fall through to the
        # P0.11 single-row dispatch above.
        Binding("space", "row_toggle_select", "Select", show=False),
        Binding("t", "row_tag_edit", "Tags", show=False),
        # §4 P1 — multi-select two chains + `D` opens a
        # `DiffModal` against the saved entity ids. Symmetric
        # with the §3 P1 in-session artifact diff but uses
        # the Memory-backed fetch path (chains here are
        # already persisted).
        Binding("D", "diff_selected", "Diff", show=False),
        # §4 P1 — explicit bulk shortcuts. Distinct from
        # the lowercase `t` / `delete` keys (which switch
        # on `is_bulk_active`); these always route to the
        # bulk worker so users with a fresh selection can
        # commit without the mode-detect surprise.
        Binding(
            "T", "bulk_tag_edit", "Bulk-tag", show=False,
        ),
        Binding(
            "shift+delete", "bulk_delete", "Bulk-delete",
            show=False,
        ),
        # §4 P1 — Import / Export. `i` pushes ImportModal +
        # `x` pushes ExportModal seeded with the current
        # selection (or the focused row when nothing's
        # multi-selected).
        Binding("i", "import_bundle", "Import", show=False),
        Binding("x", "export_bundle", "Export", show=False),
        Binding("escape", "clear_selection", "Clear", show=False),
        # Jump back to the ChatScreen (the home surface). Routes
        # through the app's `action_palette_open_chat` so the
        # key, palette entry, and empty-state CTA all land on
        # the same pop-to-or-push behaviour.
        Binding("c", "back_to_chat", "Chat", show=True),
    ]

    COLUMNS = (
        "★",
        "Name",
        "Domain",
        "Steps",
        "Last Run",
        "Runs",
        "Fitness",
        "Cost",
        "Tags",
    )
    """Columns the DataTable renders, in render order. The
    leading favourite indicator is a single character so the
    column stays narrow; future sub-tasks may swap it for an
    icon."""

    @staticmethod
    def _localized_columns() -> tuple[str, ...]:
        """The :data:`COLUMNS` order rendered through the i18n catalog.
        ``COLUMNS`` stays the canonical English contract (tests +
        :data:`COLUMN_SORT_FIELDS` index map key off it); this is what
        the DataTable actually shows. The leading ★ favourite glyph is
        language-neutral and stays literal."""
        return (
            "★",
            t("library.columns.name"),
            t("library.columns.domain"),
            t("library.columns.steps"),
            t("library.columns.lastRun"),
            t("library.columns.runs"),
            t("library.columns.fitness"),
            t("library.columns.cost"),
            t("library.columns.tags"),
        )

    COLUMN_SORT_FIELDS: dict[int, str] = {
        1: "display_name",
        4: "last_run_at",
        5: "run_count",
    }
    """Maps DataTable column indexes to the `LibrarySort.field`
    values the user can sort by. Columns not in this map are
    not sortable (header clicks no-op). The server-side data
    layer (`LibrarySort` constructor) validates the field
    name against the supported set."""

    def __init__(
        self,
        *,
        sort: LibrarySort | None = None,
        filters: LibraryFilters | None = None,
        restore_state: bool = True,
    ) -> None:
        super().__init__()
        # Restore persisted sort when both kwargs default and
        # `restore_state` is True. Explicit kwargs always win —
        # tests pass them to pin behaviour without touching the
        # user's real preference file.
        #
        # Content filters (domain / tags / status / favourites /
        # search) are deliberately NOT restored: a freshly-saved
        # chain that doesn't match a stale filter would be hidden
        # until the user hit "Clear filters", so the Library opens
        # showing everything. Only the operator-level namespace +
        # channel ride along (via `clear_filters`).
        restored = (
            load_view_state()
            if restore_state and sort is None and filters is None
            else None
        )
        if sort is not None:
            self.sort: LibrarySort = sort
        elif restored is not None:
            self.sort = restored.sort
        else:
            self.sort = LibrarySort()
        if filters is not None:
            self.filters: LibraryFilters = filters
        elif restored is not None:
            self.filters = clear_filters(restored.filters)
        else:
            self.filters = LibraryFilters()
        # Last fetched view — exposed for tests + the empty-
        # state classifier.
        self.view: LibraryView | None = None
        # Saved/Sessions partition of the fetched Memory rows — parallel
        # to each tab's DataTable rows so a cursor index resolves back to
        # its LibraryRow.
        self._saved_rows: list[LibraryRow] = []
        self._session_rows: list[LibraryRow] = []
        # Last error message (if any) — surfaced by the
        # empty-state widget.
        self.last_error: str | None = None
        # `True` while the fetch worker is in flight. Used by
        # the empty-state classifier to render the loading
        # template on first paint.
        self.is_loading: bool = False
        # CTA-click telemetry — tests + future TaskList drawer
        # read this; production callers rarely need it.
        self._cta_log: list[str] = []
        # Per-row action log — tests + future Inspection /
        # Edit / Evolution screens (P0.16+) read this until
        # the destination screens land.
        self._row_action_log: list[tuple[str, str]] = []
        # Last mutator outcome — surfaced to tests so they can
        # assert success / error without scraping the toast
        # area.
        self.last_row_outcome: RowMutationOutcome | None = None
        # P0.13 bulk-selection state. Frozen :class:`BulkSelection`;
        # mutated functionally so each toggle returns a fresh
        # instance — gives the future undo stack a free history.
        self.bulk_selection: BulkSelection = BulkSelection()
        # Last bulk operation outcome — tests + the future
        # toast renderer read this.
        self.last_bulk_result: BulkOperationResult | None = None
        # P0.14 collections sidebar. Aggregated from the active
        # library view by the fetch worker; rendered by the
        # sidebar's collections section.
        self.collections: tuple[Collection, ...] = ()
        # §4 P2 — per-chain run aggregate keyed by
        # `chain_id`. Populated from `~/.cache/care/runs/` on
        # every `_refresh`; absent keys mean "no local runs
        # yet" and the row cells fall back to the Memory-side
        # metadata.
        self._run_stats: dict[str, Any] = {}

    @staticmethod
    def _footer_registry() -> tuple[GlobalBinding, ...]:
        """Footer hints for the Library = the five app-wide
        globals plus a screen-scoped ``Del Delete`` hint so the
        destructive row action is discoverable. The chord itself
        is the ``delete`` :class:`Binding` above; this only
        surfaces it in the footer."""
        return default_global_bindings() + (
            GlobalBinding(
                action_id="back_to_chat",
                key="C",
                label=t("library.action.chat.label"),
                scope="screen",
                description=t("library.action.chat.description"),
            ),
            GlobalBinding(
                action_id="delete_row",
                key="Del",
                label=t("library.action.delete.label"),
                scope="screen",
                description=t("library.action.delete.description"),
            ),
        )

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with TabbedContent(id="library-tabs"):
            with TabPane(t("library.tabSaved"), id="library-tab-saved"):
                with Horizontal(id="library-main"):
                    yield LibrarySidebar(filters=self.filters)
                    with Horizontal(id="library-content"):
                        yield DataTable(id="library-table")
                        yield EmptyStateView()
            with TabPane(t("library.tabSessions"), id="library-tab-sessions"):
                with Vertical(id="library-sessions"):
                    yield DataTable(id="library-sessions-table")
                    yield Static("", id="library-sessions-empty")
        with Horizontal(id="library-actions"):
            yield Button(t("library.action.back"), id="library-btn-chat")
        yield CareFooter()

    def on_mount(self) -> None:
        # Header / footer reflect the current screen.
        self.query_one(CareHeader).refresh_from_app(
            active_screen="LibraryScreen",
            breadcrumb=(t("library.title"),),
        )
        self.query_one(CareFooter).refresh_from_app(
            active_screen="LibraryScreen",
            scope="screen",
            registry=self._footer_registry(),
        )
        # DataTable scaffolding: columns + cursor row mode.
        table = self.query_one("#library-table", DataTable)
        for col in self._localized_columns():
            table.add_column(col)
        table.cursor_type = "row"
        table.zebra_stripes = True
        # Sessions tab table — same columns as Saved (both hold Memory
        # rows; the split is by name, not data shape).
        try:
            sess = self.query_one("#library-sessions-table", DataTable)
            for col in self._localized_columns():
                sess.add_column(col)
            sess.cursor_type = "row"
            sess.zebra_stripes = True
        except Exception:
            pass
        self._update_sessions_empty()
        # Show loading state until the first fetch lands.
        self.is_loading = True
        self._update_empty_state_view()
        # Kick the fetch worker. `exclusive=True` so a fresh
        # refresh while a previous fetch is still in flight
        # cancels the old worker first.
        self.run_worker(
            self._refresh(),
            name="library_fetch",
            group="library",
            exclusive=True,
            exit_on_error=False,
        )

    # ------------------------------------------------------------------
    # Fetch worker
    # ------------------------------------------------------------------

    async def _refresh(self) -> None:
        """Fetch the active view from Memory + populate the
        DataTable. Stays cancellable — Textual's worker
        manager raises on the awaited fetch when the worker
        is cancelled."""
        memory = getattr(self.app, "memory", None)
        try:
            if memory is None:
                # No facade wired yet — empty-state widget
                # renders the "no_library" template.
                self.view = None
                self.last_error = None
                return
            try:
                view = await fetch_library_view(
                    memory, filters=self.filters, sort=self.sort,
                )
            except LibraryViewError as exc:
                self.last_error = str(exc)
                self.view = None
                return
            self.last_error = None
            self.view = view
            # §4 P2 — pull local run history once per refresh so
            # the row cells can render last-run-age + success
            # rate + mean cost off-line. Load before populating
            # the table so the first paint already carries the
            # enriched cells.
            self._refresh_run_stats()
            self._populate_table(view)
            await self._refresh_collections(memory)
            await self._refresh_tag_pool(memory, view)
        finally:
            self.is_loading = False
            self._update_empty_state_view()

    def _refresh_run_stats(self) -> None:
        """§4 P2 — reload the per-chain run aggregate from
        ``~/.cache/care/runs/``. Best-effort: load failures
        leave `_run_stats` empty so the row cells fall back to
        the Memory-side metadata."""
        try:
            from care.runtime.local_run_history import (
                load_local_runs,
                summarise_runs_by_chain,
            )
        except Exception:
            self._run_stats = {}
            return
        try:
            runs = load_local_runs()
        except Exception:
            self._run_stats = {}
            return
        try:
            self._run_stats = dict(summarise_runs_by_chain(runs))
        except Exception:
            self._run_stats = {}

    async def _refresh_tag_pool(
        self, memory: object, view: LibraryView,
    ) -> None:
        """Harvest tags + push them to the sidebar's chip pool
        (§4 P1).

        Two-source resolution:

        1. If the Memory facade exposes ``list_tags()`` (sync or
           async), prefer it — the server already ranks tags by
           ``namespace`` + ``channel`` scope.
        2. Otherwise, aggregate tags from ``view.rows[*].tags``
           and rank by frequency desc, then alphabetical as a
           tiebreaker so the chip order is deterministic.

        Best-effort throughout: a sidebar query miss or a
        ``list_tags`` exception just leaves the previous chip
        set in place. The sidebar's own ``set_tag_pool``
        short-circuits when the pool is unchanged so this is
        safe to call on every refresh.
        """
        tags: tuple[str, ...] = ()
        lister = getattr(memory, "list_tags", None)
        if callable(lister):
            try:
                raw = lister()
                if inspect.isawaitable(raw):
                    raw = await raw
            except Exception:
                raw = None
            if raw:
                deduped: list[str] = []
                for item in raw:
                    tag = str(item).strip()
                    if tag and tag not in deduped:
                        deduped.append(tag)
                tags = tuple(deduped)
        if not tags:
            tags = _rank_tags_by_frequency(view.rows)
        if not self.is_mounted:
            return
        try:
            sidebar = self.query_one(LibrarySidebar)
        except Exception:
            return
        try:
            sidebar.set_tag_pool(tags)
        except Exception:
            pass

    async def _refresh_collections(self, memory: object) -> None:
        """Aggregate the collections visible in the library +
        feed them to the sidebar. Best-effort — a failure here
        leaves the previous list intact rather than blanking
        the sidebar."""
        try:
            collections = await list_collections(memory)
        except Exception:
            return
        self.collections = collections
        if not self.is_mounted:
            return
        try:
            sidebar = self.query_one(LibrarySidebar)
        except Exception:
            return
        sidebar.set_collections(collections)

    # ------------------------------------------------------------------
    # Saved vs Sessions partition
    # ------------------------------------------------------------------
    # The fetched Memory chains are split across two tabs by name:
    # deliberately-named saved DAGs land on "Saved"; auto-named
    # "General …" chains (quick / ad-hoc saves) land on "Sessions" so the
    # Saved tab stays focused on the agents the user named on purpose.

    _SESSION_NAME_PREFIX = "general"

    @classmethod
    def _is_session_row(cls, row: LibraryRow) -> bool:
        name = (getattr(row, "label", "") or "").strip().lower()
        return name.startswith(cls._SESSION_NAME_PREFIX)

    def _active_table_id(self) -> str:
        """Id of the DataTable in the active tab (Saved vs Sessions)."""
        try:
            tabs = self.query_one("#library-tabs", TabbedContent)
        except Exception:  # noqa: BLE001
            return "library-table"
        if tabs.active == "library-tab-sessions":
            return "library-sessions-table"
        return "library-table"

    def _active_rows(self) -> list[LibraryRow]:
        if self._active_table_id() == "library-sessions-table":
            return self._session_rows
        return self._saved_rows

    def _update_sessions_empty(self) -> None:
        try:
            empty = self.query_one("#library-sessions-empty", Static)
            table = self.query_one("#library-sessions-table", DataTable)
        except Exception:  # noqa: BLE001
            return
        has_rows = bool(self._session_rows)
        empty.update("" if has_rows else t("library.sessions.empty"))
        empty.display = not has_rows
        table.display = has_rows

    # ------------------------------------------------------------------
    # DataTable population
    # ------------------------------------------------------------------

    def _populate_table(self, view: LibraryView) -> None:
        # Split the fetched rows into the Saved (named) + Sessions
        # ("General …") partitions, then fill each tab's table.
        # Order each partition by the active sort CLIENT-SIDE so the result
        # is deterministic regardless of what order the server returns —
        # the default (`created_at` desc) puts the latest saved item first.
        self._saved_rows = self._order_for_display(
            [r for r in view.rows if not self._is_session_row(r)]
        )
        self._session_rows = self._order_for_display(
            [r for r in view.rows if self._is_session_row(r)]
        )
        self._fill_table("#library-table", self._saved_rows)
        self._fill_table("#library-sessions-table", self._session_rows)
        self._update_sessions_empty()

    # Sort-field → row key. Used for the deterministic client-side ordering
    # in :meth:`_order_for_display`. Datetime fields go through `_epoch` so a
    # missing timestamp sorts last under the (default) descending order.
    _DISPLAY_SORT_KEYS: dict[str, Any] = {
        "created_at": lambda r: _epoch(r.created_at),
        "last_run_at": lambda r: _epoch(r.last_run_at),
        "run_count": lambda r: r.run_count,
        "display_name": lambda r: (r.display_name or r.name or "").lower(),
    }

    def _order_for_display(
        self, rows: list[LibraryRow],
    ) -> list[LibraryRow]:
        """Order ``rows`` by the active sort (default: ``created_at`` desc,
        newest first), pinning favourites to the top when the sort asks for
        it. Stable so ties keep their relative order."""
        keyfn = self._DISPLAY_SORT_KEYS.get(self.sort.field)
        if keyfn is not None:
            rows = sorted(
                rows, key=keyfn, reverse=self.sort.direction == "desc",
            )
        if self.sort.favourites_first:
            # Stable secondary pass: ⭐ rows bubble to the top, the
            # creation-date order preserved within each partition.
            rows = sorted(rows, key=lambda r: not r.favourite)
        return rows

    def _fill_table(self, table_id: str, rows: list[LibraryRow]) -> None:
        try:
            table = self.query_one(table_id, DataTable)
        except Exception:  # noqa: BLE001
            return
        # `clear()` keeps the column definitions; just wipes rows.
        table.clear()
        for row in rows:
            stats = self._run_stats.get(row.entity_id)
            table.add_row(
                *self._row_cells(row, stats),
                key=row.entity_id,
            )

    @classmethod
    def _row_cells(
        cls,
        row: LibraryRow,
        stats: Any = None,
    ) -> tuple[str, ...]:
        """Project a :class:`LibraryRow` into the cells the
        DataTable renders. Exposed at class scope so tests
        can pin formatting without driving the full widget.

        ``stats`` is an optional :class:`ChainRunStats` (from
        the §4 P2 local-run-history aggregate). When present
        and non-empty:

        * The "Last Run" cell carries an additional
          "· <rate>/<count>" annotation strip rendered by
          :func:`format_recency` — e.g.
          ``"2h ago · 0.84/12"``. Falls back to the Memory-
          side `_format_datetime(row.last_run_at)` when stats
          are absent or report zero runs.
        * The "Runs" cell picks the higher of the Memory-side
          count and the local count, so a row recently
          executed locally surfaces the right number even when
          Memory hasn't caught up yet.
        * The new "Cost" cell renders the mean USD cost from
          local runs (`"$0.42"` / `"<$0.01"` / `"—"`).
        """
        # Local stats win when present; otherwise fall back to
        # Memory-side metadata (`last_run_at`, `run_count`).
        recency = cls._format_recency_cell(row, stats)
        runs_count = row.run_count
        if stats is not None:
            run_count_local = int(getattr(stats, "run_count", 0) or 0)
            if run_count_local > runs_count:
                runs_count = run_count_local
        cost_cell = cls._format_cost_cell(stats)
        return (
            "★" if row.favourite else "",
            row.label,
            row.domain,
            str(row.step_count) if row.step_count is not None else "—",
            recency,
            str(runs_count),
            cls._format_fitness(row.fitness),
            cost_cell,
            ", ".join(row.tags),
        )

    @staticmethod
    def _format_recency_cell(
        row: LibraryRow, stats: Any,
    ) -> str:
        """Pick the §4 P2 "last run · rate/N" strip when local
        runs are available; otherwise render the Memory-side
        timestamp via the existing formatter."""
        from care.runtime.local_run_history import format_recency

        strip = format_recency(stats) if stats is not None else ""
        if strip:
            return strip
        return _format_datetime(row.last_run_at)

    @staticmethod
    def _format_cost_cell(stats: Any) -> str:
        """§4 P2 — mean USD cost from local runs, ``"—"`` when
        no cost data is in scope."""
        from care.runtime.local_run_history import format_mean_cost

        return format_mean_cost(stats)

    @staticmethod
    def _format_fitness(fitness: float | None) -> str:
        if fitness is None:
            return "—"
        return f"{fitness:.3f}"

    # ------------------------------------------------------------------
    # Sidebar wiring
    # ------------------------------------------------------------------

    def on_library_sidebar_filters_changed(
        self, event: LibrarySidebar.FiltersChanged,
    ) -> None:
        """Listen for sidebar filter changes; update the
        screen's `filters` state, persist, and re-run the
        worker."""
        self.filters = event.filters
        self._save_view_state()
        self.refresh_library()

    def on_library_sidebar_collection_selected(
        self, event: LibrarySidebar.CollectionSelected,
    ) -> None:
        """Listen for sidebar collection picks. Routes through
        `filter_by_collection` so the existing tag chips stay
        intact and the collection tag rides as an AND-filter."""
        new_filters = filter_by_collection(self.filters, event.name)
        self.filters = new_filters
        sidebar = self.query_one(LibrarySidebar)
        sidebar.set_filters(self.filters)
        sidebar._sync_collection_highlight()
        self._save_view_state()
        self.refresh_library()

    def action_focus_search(self) -> None:
        """`Ctrl+F` → focus the sidebar's search input. The
        activating chord doesn't redeliver to the focused
        widget so no absorber is needed."""
        sidebar = self.query_one(LibrarySidebar)
        sidebar.focus_search()

    def action_focus_search_absorb(self) -> None:
        """`/` → focus the sidebar's search input + drop the
        activating `/` keystroke that Textual redelivers to
        the now-focused widget on the next dispatch tick."""
        sidebar = self.query_one(LibrarySidebar)
        sidebar.focus_search(absorb_next_keystroke=True)

    # ------------------------------------------------------------------
    # Sort + persistence (P0.10)
    # ------------------------------------------------------------------

    def on_data_table_header_selected(
        self, event: DataTable.HeaderSelected,
    ) -> None:
        """DataTable header click → flip the active sort.

        Clicking the same column toggles direction (asc ↔
        desc); clicking a different sortable column switches
        the field and resets direction to `desc` (the default
        "recently used first" gesture). Clicks on
        non-sortable columns (favourite indicator, domain,
        steps, fitness, tags) are no-ops.
        """
        if getattr(event.data_table, "id", None) != "library-table":
            return
        if event.column_index is None:
            return
        field = self.COLUMN_SORT_FIELDS.get(event.column_index)
        if field is None:
            return
        if self.sort.field == field:
            new_direction = "asc" if self.sort.direction == "desc" else "desc"
        else:
            new_direction = "desc"
        self.sort = LibrarySort(
            field=field,
            direction=new_direction,
            favourites_first=self.sort.favourites_first,
        )
        self._save_view_state()
        self.refresh_library()

    def _save_view_state(self) -> None:
        """Persist the current sort + filter snapshot.
        Degrades silently on read-only filesystem so a write
        failure can't crash the worker."""
        try:
            save_view_state(
                LibraryViewState(sort=self.sort, filters=self.filters),
            )
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Empty-state wiring
    # ------------------------------------------------------------------

    def _classify_empty_state(self) -> EmptyState | None:
        """Pure projection — runs the shipped classifier
        against the screen's current state. Exposed at
        instance scope so tests can pin classification
        without driving the full mount lifecycle.

        The Saved tab only shows non-"General" rows, so the classifier
        sees a view filtered to that subset — an all-"General" Memory
        renders the Saved tab's empty state rather than a blank table."""
        view = self.view
        if view is not None:
            from dataclasses import replace

            saved = tuple(
                r for r in view.rows if not self._is_session_row(r)
            )
            view = replace(view, rows=saved)
        return classify_empty_state(
            view,
            filters=self.filters,
            is_loading=self.is_loading,
            error=self.last_error,
        )

    def _update_empty_state_view(self) -> None:
        """Sync the EmptyStateView + DataTable display states
        against the current classification. Idempotent — safe
        to call from worker callbacks + filter changes."""
        if not self.is_mounted:
            return
        empty = self._classify_empty_state()
        table = self.query_one("#library-table", DataTable)
        view = self.query_one(EmptyStateView)
        # The fetch worker's `finally` calls this on resolve — drop any
        # in-flight loading overlay armed in `_arm_table_loading`. The
        # first-paint loading is owned by the EmptyStateView's "loading"
        # template; `.loading` only covers re-fetches over a populated table.
        table.loading = False
        if empty is None:
            table.display = True
            view.display = False
            view.set_state(None)
        else:
            table.display = False
            view.display = True
            view.set_state(empty)

    def _arm_table_loading(self) -> None:
        """Show Textual's animated loading overlay on the table while a
        re-fetch is in flight — but only when rows are already showing, so
        the empty first-paint keeps the EmptyStateView "loading" template
        (the overlay needs a sized, visible table to render over)."""
        if not self.is_mounted:
            return
        try:
            table = self.query_one("#library-table", DataTable)
        except Exception:
            return
        if table.row_count and table.display:
            table.loading = True

    def on_empty_state_view_action_fired(
        self, event: EmptyStateView.ActionFired,
    ) -> None:
        """Listen for empty-state CTA clicks and dispatch."""
        kind = event.action_kind
        if kind == "clear_filters":
            self.filters = clear_filters(self.filters)
            sidebar = self.query_one(LibrarySidebar)
            sidebar.set_filters(self.filters)
            self.refresh_library()
        elif kind == "retry":
            self.refresh_library()
        elif kind == "create_first_agent":
            # P0.15 — push the QueryScreen on top of the
            # library stack. Telemetry log still appended so
            # legacy tests + future analytics can read the
            # gesture without scraping the screen stack.
            self._cta_log.append("create_first_agent")
            from care.screens.query import QueryScreen

            self.app.push_screen(QueryScreen())
        elif kind == "back_to_chat":
            # §4 P0 — the no-library card surfaces this as a
            # secondary CTA so users with an unsaved chain in
            # the current chat session can jump back, type
            # `/artifacts`, and persist it. Routes through the
            # same `action_palette_open_chat` helper so the
            # popup + palette + slash gesture stay in lockstep.
            self._cta_log.append("back_to_chat")
            opener = getattr(self.app, "action_palette_open_chat", None)
            if callable(opener):
                opener()
        # noop kinds don't get fired (the loading state has
        # no button), but defend against future templates.

    # ------------------------------------------------------------------
    # Per-row actions (P0.11)
    # ------------------------------------------------------------------

    @property
    def current_row(self) -> LibraryRow | None:
        """Return the :class:`LibraryRow` under the active tab's
        DataTable cursor (Saved or Sessions), or ``None`` when that
        tab is empty / the cursor is out of range. Exposed as a
        property so tests + action handlers share one resolution path."""
        rows = self._active_rows()
        if not rows:
            return None
        try:
            table = self.query_one(f"#{self._active_table_id()}", DataTable)
        except Exception:
            return None
        idx = table.cursor_row
        if idx is None or idx < 0 or idx >= len(rows):
            return None
        return rows[idx]

    def _row_action_available(self, kind: RowActionKind) -> RowAction | None:
        """Return the registered :class:`RowAction` for ``kind``
        if it's enabled on the current row's status. Returns
        ``None`` when no row is focused or the action is gated
        out (e.g. `evolve` on a `draft` row)."""
        row = self.current_row
        if row is None:
            return None
        for action in actions_for_row(row):
            if action.kind == kind:
                return action
        return None

    def _record_row_action(self, kind: RowActionKind, row: LibraryRow) -> None:
        """Append ``(kind, entity_id)`` to the action log so
        tests + future destination screens can read the dispatch
        without driving the (not-yet-shipped) screens."""
        self._row_action_log.append((kind, row.entity_id))

    async def _confirm_destructive(self, action: RowAction, row: LibraryRow) -> bool:
        """Push :class:`ConfirmModal` and await the user's
        choice. Returns the modal's dismiss value (`True` =
        confirmed, `False` = cancelled)."""
        modal = ConfirmModal(
            title=f"{action.label}?",
            body=f"{row.label} ({row.entity_id[:12]})",
            confirm_label=action.label,
        )
        # NOTE: `action.label` is already localized via the row-action
        # registry, so the "{label}?" / body format needs no catalog key.
        result = await self.app.push_screen_wait(modal)
        return bool(result)

    async def _run_toggle_favourite(self, row: LibraryRow) -> None:
        outcome = await toggle_favourite_row(self.app.memory, row)
        self.last_row_outcome = outcome
        if outcome.success:
            self.refresh_library()

    # ---- P0.13 bulk-select ----

    @property
    def is_bulk_active(self) -> bool:
        """``True`` when the user has multi-selected at least
        one row. F / Delete bindings switch to bulk-apply when
        this is set; Space stays the toggle either way."""
        return not self.bulk_selection.is_empty

    @staticmethod
    def _row_to_bulk_target(row: LibraryRow) -> BulkTarget:
        return BulkTarget(
            entity_id=row.entity_id,
            entity_type=row.entity_type,
            current_tags=tuple(row.tags),
            display_name=row.label,
        )

    def action_row_toggle_select(self) -> None:
        """`Space` → add / remove the focused row from the
        bulk selection. No-op when no row is focused."""
        row = self.current_row
        if row is None:
            return
        self.bulk_selection = self.bulk_selection.toggle(
            self._row_to_bulk_target(row),
        )

    def action_clear_selection(self) -> None:
        """`Escape` → drop the bulk selection if one is active; otherwise
        go back to the chat screen (so Esc always has an obvious effect and
        matches the ``[Esc] Back`` footer hint)."""
        if self.bulk_selection.is_empty:
            self.action_back_to_chat()
            return
        self.bulk_selection = self.bulk_selection.clear()

    def action_back_to_chat(self) -> None:
        """`c` / "Back to Chat" button → return to the
        ChatScreen. Delegates to the app's
        `action_palette_open_chat`, which pops to an existing
        ChatScreen or pushes a fresh one. Shares the
        `back_to_chat` CTA log entry with the empty-state path
        so both gestures stay observable."""
        self._cta_log.append("back_to_chat")
        opener = getattr(self.app, "action_palette_open_chat", None)
        if callable(opener):
            opener()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "library-btn-chat":
            self.action_back_to_chat()

    async def _run_bulk_favourite(self) -> None:
        # Flip to the opposite of "all currently favourites" —
        # mirrors the per-row toggle's gesture. Memory's bulk
        # endpoint takes one explicit value, so we honour the
        # majority intent: if any selected row is unfavourited,
        # promote the whole batch to favourite; if every row is
        # already favourite, unfavourite them all.
        selection = self.bulk_selection
        if selection.is_empty:
            return
        target_value = self._bulk_favourite_target(selection)
        result = await apply_favourite(
            self.app.memory, selection, favourite=target_value,
        )
        self.last_bulk_result = result
        if result.succeeded:
            self.refresh_library()

    def _bulk_favourite_target(
        self, selection: BulkSelection,
    ) -> bool:
        """Resolve the bool to pass to `apply_favourite`.

        Reads each target's current state off the loaded view
        — if every row is already favourited, unfavourite;
        otherwise favourite the whole batch."""
        if self.view is None:
            return True
        index = {r.entity_id: r for r in self.view.rows}
        rows_in_sel = [
            index[t.entity_id]
            for t in selection.targets
            if t.entity_id in index
        ]
        if not rows_in_sel:
            return True
        return not all(r.favourite for r in rows_in_sel)

    async def _run_bulk_delete(self) -> None:
        selection = self.bulk_selection
        if selection.is_empty:
            return
        confirmed = await self._confirm_bulk_destructive(
            selection, t("library.rowAction.delete.label"),
        )
        if not confirmed:
            return
        result = await apply_delete(self.app.memory, selection)
        self.last_bulk_result = result
        if result.succeeded:
            self.bulk_selection = self.bulk_selection.clear()
            self.refresh_library()

    async def _confirm_bulk_destructive(
        self, selection: BulkSelection, label: str,
    ) -> bool:
        count = len(selection)
        item = (
            t("library.confirm.itemOne") if count == 1
            else t("library.confirm.itemMany")
        )
        modal = ConfirmModal(
            title=t(
                "library.confirm.bulkTitle",
                label=label, count=count, item=item,
            ),
            body=", ".join(
                (t.display_name or t.entity_id)
                for t in selection.targets[:5]
            ) + ("…" if count > 5 else ""),
            confirm_label=label,
        )
        result = await self.app.push_screen_wait(modal)
        return bool(result)

    async def _run_delete(self, action: RowAction, row: LibraryRow) -> None:
        confirmed = await self._confirm_destructive(action, row)
        if not confirmed:
            return
        outcome = await delete_row(self.app.memory, row)
        self.last_row_outcome = outcome
        if outcome.success:
            self.refresh_library()

    async def _run_duplicate(self, row: LibraryRow) -> None:
        outcome = await duplicate_chain(self.app.memory, row)
        self.last_row_outcome = outcome
        if outcome.success:
            self.refresh_library()

    async def _run_archive_evolutions(self, row: LibraryRow) -> None:
        """Stop every Platform run whose ``base_chain_id`` equals
        the row's entity id and add those runs to the dashboard's
        archive set.

        ``platform.list_evolutions()`` is the source of truth for
        the run inbox; we filter client-side and fire
        ``platform.cancel(...)`` on each active row. Cancellations
        run sequentially because the runner pool is typically
        size 1 and parallel stops just queue up. Failures are
        non-fatal — best-effort archival should still proceed for
        the runs we could identify.
        """
        import asyncio as _asyncio

        from care.screens.evolution_dashboard import (
            _load_archive,
            _save_archive,
        )

        platform = getattr(self.app, "platform", None)
        if platform is None:
            try:
                self.app.notify(
                    t("library.archive.platformNotConfigured"),
                    title=t("library.archive.title"),
                    severity="warning",
                    timeout=8,
                )
            except Exception:
                pass
            return

        try:
            envelope = await _asyncio.to_thread(platform.list_evolutions)
        except Exception as exc:  # noqa: BLE001
            try:
                self.app.notify(
                    t(
                        "library.archive.listFailed",
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    title=t("library.archive.title"),
                    severity="error",
                    timeout=10,
                )
            except Exception:
                pass
            return

        all_items = envelope.get("items") if isinstance(envelope, dict) else []
        chain_id = row.entity_id
        # Match on the normalised ``base_chain_id`` field CARE
        # writes into chain-experiment rows via
        # ``_experiment_to_evolution_row``.
        active_states = {
            "running", "preparing", "initializing",
            "dispatching", "queued", "prepared",
        }
        targets = []
        for raw in all_items or []:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("base_chain_id") or "") != chain_id:
                continue
            targets.append(raw)
        if not targets:
            try:
                self.app.notify(
                    t("library.archive.noRuns", id=chain_id[:18]),
                    title=t("library.archive.title"),
                    severity="information",
                    timeout=6,
                )
            except Exception:
                pass
            return

        stopped = 0
        failures = 0
        archived = _load_archive()
        for raw in targets:
            evo_id = str(raw.get("evolution_id") or raw.get("id") or "")
            status = str(raw.get("status") or "").lower()
            if not evo_id:
                continue
            if status in active_states:
                try:
                    await _asyncio.to_thread(platform.cancel, evo_id)
                    stopped += 1
                except Exception:
                    failures += 1
            archived.add(evo_id)
        _save_archive(archived)

        msg_parts = [t("library.archive.archived", count=len(targets))]
        if stopped:
            msg_parts.append(t("library.archive.stopped", count=stopped))
        if failures:
            msg_parts.append(t("library.archive.failed", count=failures))
        msg_parts.append(t("library.archive.seeTab"))
        try:
            self.app.notify(
                " — ".join(msg_parts[:1]) + " (" + ", ".join(msg_parts[1:-1]) + ") "
                + msg_parts[-1] if len(msg_parts) > 2 else " ".join(msg_parts),
                title=t("library.archive.title"),
                severity="warning" if failures else "information",
                timeout=8,
            )
        except Exception:
            pass

    def _dispatch_row_action(self, kind: RowActionKind) -> None:
        """Resolve the action against the focused row + status
        registry, then dispatch.

        Three dispatch paths:

        * **Mutators** (``toggle_favourite`` / ``delete`` /
          ``duplicate``) run on a worker that hits Memory.
        * **Navigation-only** (``open`` / ``edit`` / ``evolve``
          / ``show_lineage``) record the click on the action
          log and push the matching destination screen via
          :meth:`_push_screen_for_row_action`.
        * **Runtime-pending** (``run``) records the click +
          surfaces an info toast pointing at the CLI
          (``care run <id> --execute``); the in-TUI executor
          push is gated on the same runtime wiring the CLI's
          ``--execute`` path uses.
        """
        action = self._row_action_available(kind)
        if action is None:
            return
        row = self.current_row
        if row is None:
            return
        if action.kind == "toggle_favourite":
            self.run_worker(
                self._run_toggle_favourite(row),
                name="library_row_action",
                group="library_row",
                exclusive=False,
                exit_on_error=False,
            )
        elif action.kind == "delete":
            self.run_worker(
                self._run_delete(action, row),
                name="library_row_action",
                group="library_row",
                exclusive=False,
                exit_on_error=False,
            )
        elif action.kind == "duplicate":
            self.run_worker(
                self._run_duplicate(row),
                name="library_row_action",
                group="library_row",
                exclusive=False,
                exit_on_error=False,
            )
        elif action.kind == "archive_evolutions":
            # Stop + archive every Platform run whose
            # ``base_chain_id`` matches this library row's
            # entity id. Runs as a worker so the slow
            # ``cancel`` HTTP hops don't freeze the table.
            self._record_row_action(action.kind, row)
            self.run_worker(
                self._run_archive_evolutions(row),
                name="library_row_action",
                group="library_row",
                exclusive=False,
                exit_on_error=False,
            )
        else:
            # `run` / `open` / `edit` / `evolve` / `show_lineage`
            # — all destination screens have shipped. Record the
            # dispatch first (tests + future telemetry rely on
            # the action log staying populated) then push the
            # matching screen. Push failures fall through to
            # the toast surface; the action log entry is the
            # canonical "gesture fired" signal.
            self._record_row_action(action.kind, row)
            self._push_screen_for_row_action(action.kind, row)

    def _push_screen_for_row_action(
        self, kind: "RowActionKind", row: object,
    ) -> None:
        """Push the destination screen for a row action.

        Kinds handled here are the navigation-only ones the
        else-branch in :meth:`_dispatch_row_action` dispatches
        — i.e. everything except ``toggle_favourite`` / ``delete``
        / ``duplicate`` (those run mutating workers above).

        Each push is best-effort: import / construction
        failures fall through to a toast rather than crashing
        the LibraryScreen.
        """
        entity_id = getattr(row, "entity_id", None) or ""
        if not entity_id:
            return
        try:
            if kind == "open":
                from care.screens.inspection import InspectionScreen

                self.app.push_screen(InspectionScreen(entity_id))
                return
            if kind == "edit":
                # Delegate to the App helper that fetches the
                # chain + handles the missing-memory case.
                pusher = getattr(self.app, "_push_edit_agent_for", None)
                if callable(pusher):
                    pusher(entity_id)
                return
            if kind == "evolve":
                pusher = getattr(self.app, "_push_evolution_for", None)
                if callable(pusher):
                    pusher(entity_id)
                return
            if kind == "show_lineage":
                from care.screens.lineage import LineageModal

                self.app.push_screen(
                    LineageModal(entity_id, memory=self.app.memory),
                )
                return
            if kind == "run":
                # Run the saved chain in-TUI: load it, open the
                # RunContextModal (task + context-file attach), then
                # execute on an ExecutionScreen — same pipeline the
                # InspectionScreen `run` action uses. Delegates to the
                # App helper that handles the missing-Memory / missing-
                # LLM-key cases with clear toasts.
                runner = getattr(self.app, "_push_run_for", None)
                if callable(runner):
                    runner(entity_id)
                return
        except Exception:
            # Any push failure (constructor raised, screen
            # module missing, app not fully mounted) → silent
            # no-op. The action-log entry already recorded the
            # gesture, so tests + telemetry stay accurate.
            return

    # Individual action handlers — bound to keys via BINDINGS.
    def action_row_run(self) -> None:
        self._dispatch_row_action("run")

    def action_row_open(self) -> None:
        self._dispatch_row_action("open")

    def action_row_edit(self) -> None:
        self._dispatch_row_action("edit")

    def action_row_revise(self) -> None:
        """``R`` — hand off to chat's ``/revise`` for an AI edit of this chain.

        Standalone (not a typed ``RowActionKind``) — it just resolves the
        focused row's ``entity_id`` and asks the app to drop to chat with
        ``/revise <id> `` seeded.
        """
        row = self.current_row
        if row is None:
            return
        entity_id = getattr(row, "entity_id", "") or ""
        if not entity_id:
            return
        reviser = getattr(self.app, "_revise_chain_for", None)
        if callable(reviser):
            reviser(entity_id)

    def action_row_duplicate(self) -> None:
        self._dispatch_row_action("duplicate")

    def action_row_evolve(self) -> None:
        self._dispatch_row_action("evolve")

    def action_row_archive_evolutions(self) -> None:
        self._dispatch_row_action("archive_evolutions")

    def action_row_show_lineage(self) -> None:
        self._dispatch_row_action("show_lineage")

    def action_diff_selected(self) -> None:
        """`D` — open a DiffModal against the two selected
        chain rows. Symmetric with the §3 P1 in-session
        artifact diff but Memory-backed: passes saved
        ``entity_id`` strings + the configured Memory
        facade so the modal fetches both chains via
        `client.get_chain_dict` on mount.

        Refuses on selection sizes ≠ 2 with a friendly
        toast. Refuses on a non-chain entity type in
        either slot (lineage / agent_skill / etc. can't
        be diffed by this modal). Refuses when Memory
        isn't configured — the modal needs the facade
        to load the payloads."""
        selection = self.bulk_selection
        if len(selection) != 2:
            self._diff_toast(
                t("library.diff.needTwo", count=len(selection)),
                severity="warning",
            )
            return
        non_chain = [
            target for target in selection.targets
            if target.entity_type != "chain"
        ]
        if non_chain:
            kinds = ", ".join(target.entity_type for target in non_chain)
            self._diff_toast(
                t("library.diff.chainOnly", kinds=kinds),
                severity="warning",
            )
            return
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._diff_toast(
                t("library.diff.needMemory"),
                severity="error",
            )
            return
        targets = list(selection.targets)
        try:
            from care.screens.diff import DiffModal

            self.app.push_screen(
                DiffModal(
                    left_entity_id=targets[0].entity_id,
                    right_entity_id=targets[1].entity_id,
                    memory=memory,
                    left_label=(
                        targets[0].display_name
                        or targets[0].entity_id
                    ),
                    right_label=(
                        targets[1].display_name
                        or targets[1].entity_id
                    ),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._diff_toast(
                t("library.diff.openFailed", error=str(exc)),
                severity="error",
            )

    def _diff_toast(
        self, message: str, *, severity: str = "info",
    ) -> None:
        push_toast = getattr(self.app, "push_toast", None)
        if callable(push_toast):
            push_toast(message, severity=severity)

    def action_import_bundle(self) -> None:
        """`i` — open the ImportModal pre-pointed at the
        default tarball path. Refuses without a Memory facade
        (the modal needs it to write rows on Import).

        After a successful import the LibraryScreen reloads
        from Memory so the new rows surface immediately."""
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._diff_toast(
                t("library.import.needMemory"),
                severity="error",
            )
            return
        try:
            from care.screens.import_bundle import ImportModal

            def _on_dismiss(result):
                # `result` is a `BundleImportResult` when the
                # import succeeded, `None` on cancel.
                if result is not None:
                    self.refresh_library()
                    self._diff_toast(
                        t(
                            "library.import.done",
                            count=getattr(result, "chains_imported", "?"),
                        ),
                        severity="success",
                    )

            self.app.push_screen(
                ImportModal(memory=memory), _on_dismiss,
            )
        except Exception as exc:  # noqa: BLE001
            self._diff_toast(
                t("library.import.openFailed", error=str(exc)),
                severity="error",
            )

    def action_export_bundle(self) -> None:
        """`x` — open the ExportModal seeded with the user's
        bulk-selected chain rows, or the focused row when
        nothing is multi-selected. Refuses on non-chain
        selections + an empty stack.

        Selection precedence: multi-select wins over focused
        row; agent_skill / lineage rows in a multi-selection
        are dropped (with a hint toast) since the bundle
        exporter is chain-centric. AgentSkill IDs ride into
        the modal's `skill_entity_ids` slot when the user's
        selection happens to include any."""
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._diff_toast(
                t("library.export.needMemory"),
                severity="error",
            )
            return

        selection = self.bulk_selection
        if selection.is_empty:
            row = self.current_row
            if row is None:
                self._diff_toast(
                    t("library.export.needSelection"),
                    severity="warning",
                )
                return
            if row.entity_type != "chain":
                self._diff_toast(
                    t("library.export.chainCentric", kind=row.entity_type),
                    severity="warning",
                )
                return
            entity_ids = (row.entity_id,)
            skill_ids: tuple[str, ...] = ()
        else:
            entity_ids = tuple(
                target.entity_id for target in selection.targets
                if target.entity_type == "chain"
            )
            skill_ids = tuple(
                target.entity_id for target in selection.targets
                if target.entity_type == "agent_skill"
            )
            if not entity_ids:
                self._diff_toast(
                    t("library.export.needChain"),
                    severity="warning",
                )
                return
            dropped = (
                len(selection.targets) - len(entity_ids) - len(skill_ids)
            )
            if dropped:
                self._diff_toast(
                    t("library.export.dropped", count=dropped),
                    severity="info",
                )

        try:
            from care.screens.export import ExportModal

            def _on_dismiss(result):
                if result is not None:
                    self._diff_toast(
                        t(
                            "library.export.done",
                            chains=len(entity_ids),
                            skills=len(skill_ids),
                        ),
                        severity="success",
                    )

            self.app.push_screen(
                ExportModal(
                    memory=memory,
                    entity_ids=entity_ids,
                    skill_entity_ids=skill_ids,
                ),
                _on_dismiss,
            )
        except Exception as exc:  # noqa: BLE001
            self._diff_toast(
                t("library.export.openFailed", error=str(exc)),
                severity="error",
            )

    def action_row_toggle_favourite(self) -> None:
        if self.is_bulk_active:
            self.run_worker(
                self._run_bulk_favourite(),
                name="library_bulk",
                group="library_row",
                exclusive=False,
                exit_on_error=False,
            )
            return
        self._dispatch_row_action("toggle_favourite")

    def action_row_delete(self) -> None:
        if self.is_bulk_active:
            self.run_worker(
                self._run_bulk_delete(),
                name="library_bulk",
                group="library_row",
                exclusive=False,
                exit_on_error=False,
            )
            return
        self._dispatch_row_action("delete")

    def action_bulk_delete(self) -> None:
        """§4 P1 — explicit bulk-delete (`Shift+Del`).
        Always routes to the bulk worker; no-op when the
        selection is empty (no single-row fallback)."""
        if self.bulk_selection.is_empty:
            self._diff_toast(
                t("library.bulk.deleteNeedSelection"),
                severity="info",
            )
            return
        self.run_worker(
            self._run_bulk_delete(),
            name="library_bulk",
            group="library_row",
            exclusive=False,
            exit_on_error=False,
        )

    def action_bulk_tag_edit(self) -> None:
        """§4 P1 — explicit bulk-tag (`Shift+T`). Opens the
        TagEditorModal against the current selection; no-op
        when the selection is empty (no single-row
        fallback)."""
        if self.bulk_selection.is_empty:
            self._diff_toast(
                t("library.bulk.tagNeedSelection"),
                severity="info",
            )
            return
        # Route through the existing bulk-aware
        # `action_row_tag_edit` — it already does the
        # right thing when `is_bulk_active` is True.
        self.action_row_tag_edit()

    def action_row_tag_edit(self) -> None:
        """`T` → open the :class:`TagEditorModal` (P0.28). In
        bulk-select mode the modal seeds from the union of
        currently-applied tags across the selection and the
        resulting `add_tags` / `remove_tags` feed
        `apply_tag_edits` on the worker. For a single focused
        row the modal seeds from that row's tag set."""
        from care.screens.tag_editor import TagEditorModal

        if self.is_bulk_active:
            selection = self.bulk_selection
            initial_tags = self._union_tags(selection)
            count = len(selection)
        else:
            row = self.current_row
            if row is None:
                return
            selection = BulkSelection(
                targets=(self._row_to_bulk_target(row),),
            )
            initial_tags = tuple(row.tags)
            count = 1

        modal = TagEditorModal(
            initial_tags=initial_tags,
            target_count=count,
        )

        def _on_dismiss(result):
            if result is None or not getattr(result, "submitted", False):
                return
            self.run_worker(
                self._run_bulk_tag_edit(
                    selection,
                    add=tuple(result.add_tags),
                    remove=tuple(result.remove_tags),
                ),
                name="library_tag_edit",
                group="library_row",
                exclusive=False,
                exit_on_error=False,
            )

        self.app.push_screen(modal, _on_dismiss)

    @staticmethod
    def _union_tags(selection: BulkSelection) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for target in selection.targets:
            for tag in target.current_tags:
                if tag not in seen:
                    out.append(tag)
                    seen.add(tag)
        return tuple(out)

    async def _run_bulk_tag_edit(
        self,
        selection: BulkSelection,
        *,
        add: tuple[str, ...],
        remove: tuple[str, ...],
    ) -> None:
        if not add and not remove:
            return
        result = await apply_tag_edits(
            self.app.memory,
            selection,
            add_tags=add,
            remove_tags=remove,
        )
        self.last_bulk_result = result
        if result.succeeded:
            self.refresh_library()

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected,
    ) -> None:
        """Enter / click on a row → `open` action.

        Dispatches through :meth:`_dispatch_row_action`, which
        records the gesture on `_row_action_log` and pushes
        :class:`care.screens.inspection.InspectionScreen` via
        :meth:`_push_screen_for_row_action`. The `open` branch
        is best-effort — a constructor failure (e.g. a stale
        ``entity_id``) falls through to an empty no-op rather
        than crashing the LibraryScreen.

        Works for both tabs: ``current_row`` is tab-aware, so a row
        selected in either table opens the right chain.
        """
        self._dispatch_row_action("open")

    # ------------------------------------------------------------------
    # Context menu (P0.12)
    # ------------------------------------------------------------------

    def action_row_context_menu(self) -> None:
        """`Menu` / Ctrl+M → open the per-row :class:`ContextMenu`
        for the focused row. Dispatch reuses the P0.11
        `_dispatch_row_action` path so keyboard + pointer
        paths stay symmetrical."""
        self._open_context_menu()

    def _open_context_menu(self) -> None:
        row = self.current_row
        if row is None:
            return
        actions = actions_for_row(row)
        if not actions:
            return
        menu = ContextMenu(actions=actions)

        def _on_pick(kind: RowActionKind | None) -> None:
            if kind is None:
                return
            self._dispatch_row_action(kind)

        self.app.push_screen(menu, _on_pick)

    def on_click(self, event) -> None:
        """Right-click on the DataTable → open the context
        menu. Textual's `Click.button == 3` is the
        right-mouse-button convention."""
        if getattr(event, "button", None) != 3:
            return
        # Only trigger when the click lands on the table.
        try:
            table = self.query_one("#library-table", DataTable)
        except Exception:
            return
        widget = getattr(event, "widget", None)
        if widget is not None and widget is not table and widget not in table.walk_children():
            return
        self._open_context_menu()

    # ------------------------------------------------------------------
    # Public refresh hook
    # ------------------------------------------------------------------

    def refresh_library(self) -> None:
        """Restart the fetch worker against the current
        `filters` + `sort` state. Called by future P0.8 /
        P0.10 / P0.13 / P0.14 sub-tasks after they mutate
        the screen's state."""
        self._arm_table_loading()
        self.run_worker(
            self._refresh(),
            name="library_fetch",
            group="library",
            exclusive=True,
            exit_on_error=False,
        )


def _epoch(dt: datetime | None) -> float:
    """Sort key for a datetime: epoch seconds, or ``-inf`` when missing so
    timestamp-less rows sort LAST under the default descending order.
    Tolerant of naive/aware mixes (``.timestamp()`` never raises here)."""
    if dt is None:
        return float("-inf")
    try:
        return dt.timestamp()
    except Exception:  # noqa: BLE001 — exotic/over-/under-flow datetimes
        return float("-inf")


def _format_datetime(dt: datetime | None) -> str:
    """``"2026-05-19 12:00"`` — short enough to fit a narrow
    column without truncation."""
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M")


__all__ = ["LibraryScreen"]
