"""Pilot tests for `LibraryScreen` scaffold + DataTable (TODO §1.1 P0.7).

Mounts the screen with a stub memory facade, asserts the
DataTable populates from `fetch_library_view`, and pins the
cell projection format.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

from care.runtime.library_view import LibraryRow, LibrarySort
from care.screens.library import LibraryScreen
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubClient:
    """Mimics `memory.client` exposing `list_chains` (which
    `fetch_library_view` calls under the hood)."""

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def list_chains(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self._rows


class _StubMemory:
    def __init__(self, client):
        self.client = client


def _row(
    *,
    entity_id="ent-1",
    display_name="Weather forecaster",
    description="Hourly forecast",
    favourite=False,
    run_count=3,
    last_run_at="2026-05-19T12:00:00+00:00",
    tags=("domain:weather", "favourite"),
    fitness=None,
):
    return {
        "entity_type": "chain",
        "entity_id": entity_id,
        "version_id": "v-1",
        "channel": "latest",
        "etag": "etag",
        "favourite": favourite,
        "run_count": run_count,
        "last_run_at": last_run_at,
        "display_name": display_name,
        "description": description,
        "meta": {"tags": list(tags), "name": "internal"},
        "content": {
            "steps": [{}, {}],
            "metadata": {
                "care": (
                    {"fitness_score": fitness} if fitness is not None else {}
                ),
            },
        },
        "evolution_meta": None,
    }


class _HostApp(App):
    """Mounts LibraryScreen on boot. Tests pre-populate
    `app.memory` before `on_mount` fires."""

    def __init__(self, *, memory=None):
        super().__init__()
        self.memory = memory

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(LibraryScreen())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_state(self):
        screen = LibraryScreen()
        assert screen.view is None
        assert screen.last_error is None
        assert isinstance(screen.sort, LibrarySort)

    def test_restore_drops_content_filters_keeps_sort_and_channel(
        self, monkeypatch, tmp_path,
    ):
        # A persisted content filter (domain / tags / status / …) must
        # NOT come back on open — otherwise a freshly-saved chain that
        # doesn't match it stays hidden until the user hits "Clear
        # filters". Sort + operator-level namespace/channel still ride.
        from care.runtime.library_view import (
            LibraryFilters,
            LibrarySort,
            LibraryViewState,
            save_view_state,
        )

        path = tmp_path / "library_view.json"
        monkeypatch.setenv("CARE_VIEW_STATE_PATH", str(path))
        save_view_state(
            LibraryViewState(
                sort=LibrarySort(
                    field="run_count", direction="asc",
                    favourites_first=False,
                ),
                filters=LibraryFilters(
                    domain="weather",
                    tags=("favourite",),
                    status="evolved",
                    favourites_only=True,
                    search="storm",
                    namespace="team-a",
                    channel="stable",
                ),
            ),
            path=path,
        )

        screen = LibraryScreen()  # restore_state=True by default

        # Content filters are wiped so the list opens unfiltered.
        assert screen.filters.domain is None
        assert screen.filters.tags == ()
        assert screen.filters.status is None
        assert screen.filters.favourites_only is False
        assert screen.filters.search == ""
        assert screen.filters.is_filtering is False
        # Operator-level namespace + channel survive.
        assert screen.filters.namespace == "team-a"
        assert screen.filters.channel == "stable"
        # Sort preference is still restored.
        assert screen.sort.field == "run_count"
        assert screen.sort.direction == "asc"
        assert screen.sort.favourites_first is False

    def test_columns_constant(self):
        # Pins the column order — subsequent UI work must
        # match this when filling P0.10 sort affordances.
        # §4 P2 inserted "Cost" before "Tags".
        assert LibraryScreen.COLUMNS == (
            "★", "Name", "Domain", "Steps", "Last Run",
            "Runs", "Fitness", "Cost", "Tags",
        )


# ---------------------------------------------------------------------------
# Mount + composition
# ---------------------------------------------------------------------------


class TestMount:
    @pytest.mark.asyncio
    async def test_mounts_header_table_footer(self):
        app = _HostApp(memory=_StubMemory(_StubClient(rows=[])))
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert screen.query_one(CareHeader) is not None
            assert screen.query_one("#library-table", DataTable) is not None
            assert screen.query_one(CareFooter) is not None

    @pytest.mark.asyncio
    async def test_footer_surfaces_delete_hint(self):
        app = _HostApp(memory=_StubMemory(_StubClient(rows=[])))
        async with app.run_test() as pilot:
            await pilot.pause()
            footer = app.screen.query_one(CareFooter)
            action_ids = {h.action_id for h in footer.model.hints}
            assert "delete_row" in action_ids
            delete = next(
                h for h in footer.model.hints if h.action_id == "delete_row"
            )
            assert delete.label == "Delete"

    @pytest.mark.asyncio
    async def test_footer_surfaces_chat_hint(self):
        app = _HostApp(memory=_StubMemory(_StubClient(rows=[])))
        async with app.run_test() as pilot:
            await pilot.pause()
            footer = app.screen.query_one(CareFooter)
            chat = next(
                (h for h in footer.model.hints if h.action_id == "back_to_chat"),
                None,
            )
            assert chat is not None
            assert chat.label == "Chat"

    @pytest.mark.asyncio
    async def test_back_to_chat_invokes_app_opener(self):
        calls: list[str] = []

        class _ChatHost(_HostApp):
            def action_palette_open_chat(self) -> None:
                calls.append("open_chat")

        app = _ChatHost(memory=_StubMemory(_StubClient(rows=[])))
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            screen.action_back_to_chat()
            assert calls == ["open_chat"]
            assert "back_to_chat" in screen._cta_log

    @pytest.mark.asyncio
    async def test_back_to_chat_button_present_and_routes(self):
        from textual.widgets import Button

        calls: list[str] = []

        class _ChatHost(_HostApp):
            def action_palette_open_chat(self) -> None:
                calls.append("open_chat")

        app = _ChatHost(memory=_StubMemory(_StubClient(rows=[])))
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            btn = screen.query_one("#library-btn-chat", Button)
            assert str(btn.label) == "← Back"  # localized (EN pinned in tests)
            btn.press()
            await pilot.pause()
            assert calls == ["open_chat"]

    @pytest.mark.asyncio
    async def test_escape_with_no_selection_returns_to_chat(self):
        """B4 — with no bulk selection, Esc goes back to chat (matching the
        [Esc] Back footer hint) instead of being a no-op."""
        calls: list[str] = []

        class _ChatHost(_HostApp):
            def action_palette_open_chat(self) -> None:
                calls.append("open_chat")

        app = _ChatHost(memory=_StubMemory(_StubClient(rows=[])))
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            assert screen.bulk_selection.is_empty
            screen.action_clear_selection()  # Esc with no active selection
            await pilot.pause()
            assert calls == ["open_chat"]

    @pytest.mark.asyncio
    async def test_sidebar_present(self):
        from care.widgets.library_sidebar import LibrarySidebar

        app = _HostApp(memory=_StubMemory(_StubClient(rows=[])))
        async with app.run_test() as pilot:
            await pilot.pause()
            sidebar = app.screen.query_one(LibrarySidebar)
            assert sidebar is not None

    @pytest.mark.asyncio
    async def test_columns_added_to_table(self):
        app = _HostApp(memory=_StubMemory(_StubClient(rows=[])))
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.screen.query_one("#library-table", DataTable)
            # §4 P2 inserted "Cost" before "Tags" — 9 columns now.
            assert len(table.columns) == 9

    @pytest.mark.asyncio
    async def test_header_breadcrumb_set(self):
        app = _HostApp(memory=_StubMemory(_StubClient(rows=[])))
        async with app.run_test() as pilot:
            await pilot.pause()
            header = app.screen.query_one(CareHeader)
            assert header.model.active_screen == "LibraryScreen"
            assert header.model.breadcrumb == ("Library",)


# ---------------------------------------------------------------------------
# DataTable population
# ---------------------------------------------------------------------------


class TestDataTablePopulation:
    @pytest.mark.asyncio
    async def test_rows_populate_from_fetch(self):
        rows = [
            _row(entity_id="a", display_name="Alpha"),
            _row(entity_id="b", display_name="Beta", favourite=True),
        ]
        app = _HostApp(memory=_StubMemory(_StubClient(rows=rows)))
        async with app.run_test() as pilot:
            await pilot.pause()
            # The worker is async — give it a beat.
            await pilot.pause()
            await pilot.pause()
            table = app.screen.query_one("#library-table", DataTable)
            assert table.row_count == 2

    @pytest.mark.asyncio
    async def test_view_attribute_populated(self):
        rows = [_row(entity_id="a", display_name="Alpha")]
        app = _HostApp(memory=_StubMemory(_StubClient(rows=rows)))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.screen.view is not None
            assert len(app.screen.view.rows) == 1

    @pytest.mark.asyncio
    async def test_favourites_pinned_to_top(self):
        # Default sort pins favourites first.
        rows = [
            _row(entity_id="a", display_name="Alpha", favourite=False),
            _row(entity_id="b", display_name="Beta", favourite=True),
        ]
        app = _HostApp(memory=_StubMemory(_StubClient(rows=rows)))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # First row in the view should be the favourite.
            assert app.screen.view.rows[0].entity_id == "b"

    @pytest.mark.asyncio
    async def test_no_memory_leaves_table_empty(self):
        # `app.memory is None` → no fetch attempt → table stays
        # empty but no crash.
        app = _HostApp(memory=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            table = app.screen.query_one("#library-table", DataTable)
            assert table.row_count == 0
            assert app.screen.view is None

    @pytest.mark.asyncio
    async def test_fetch_error_recorded_on_last_error(self):
        class _BoomClient:
            def list_chains(self, **kw):
                raise RuntimeError("503 service unavailable")

        app = _HostApp(memory=_StubMemory(_BoomClient()))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.screen.view is None
            assert "503" in (app.screen.last_error or "")


# ---------------------------------------------------------------------------
# Cell projection (pure — testable without mount)
# ---------------------------------------------------------------------------


class TestCellProjection:
    def test_unfavourite_no_star(self):
        row = LibraryRow(entity_id="x", display_name="X")
        cells = LibraryScreen._row_cells(row)
        assert cells[0] == ""

    def test_favourite_star(self):
        row = LibraryRow(entity_id="x", display_name="X", favourite=True)
        cells = LibraryScreen._row_cells(row)
        assert cells[0] == "★"

    def test_label_falls_back_to_entity_id(self):
        # `LibraryRow.label` reads display_name → name → first
        # 12 of entity_id. The cell projection uses .label.
        row = LibraryRow(entity_id="long-entity-id-7777")
        cells = LibraryScreen._row_cells(row)
        # display_name is empty, name is empty → label = first 12.
        assert cells[1] == "long-entity-"

    def test_step_count_em_dash_when_none(self):
        row = LibraryRow(entity_id="x", display_name="X", step_count=None)
        cells = LibraryScreen._row_cells(row)
        assert cells[3] == "—"

    def test_step_count_numeric(self):
        row = LibraryRow(entity_id="x", display_name="X", step_count=7)
        cells = LibraryScreen._row_cells(row)
        assert cells[3] == "7"

    def test_last_run_em_dash_when_none(self):
        row = LibraryRow(entity_id="x", display_name="X", last_run_at=None)
        cells = LibraryScreen._row_cells(row)
        assert cells[4] == "—"

    def test_last_run_formatted(self):
        when = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
        row = LibraryRow(entity_id="x", display_name="X", last_run_at=when)
        cells = LibraryScreen._row_cells(row)
        assert cells[4] == "2026-05-19 12:00"

    def test_fitness_em_dash_when_none(self):
        row = LibraryRow(entity_id="x", display_name="X", fitness=None)
        cells = LibraryScreen._row_cells(row)
        assert cells[6] == "—"

    def test_fitness_three_decimals(self):
        row = LibraryRow(entity_id="x", display_name="X", fitness=0.87)
        cells = LibraryScreen._row_cells(row)
        assert cells[6] == "0.870"

    def test_tags_joined(self):
        row = LibraryRow(
            entity_id="x", display_name="X",
            tags=("domain:weather", "favourite"),
        )
        cells = LibraryScreen._row_cells(row)
        # §4 P2 — "Cost" slot now sits at index 7;
        # "Tags" moved to index 8.
        assert cells[8] == "domain:weather, favourite"

    def test_cost_cell_default_em_dash(self):
        # No stats → mean-cost cell renders "—".
        row = LibraryRow(entity_id="x", display_name="X")
        cells = LibraryScreen._row_cells(row)
        assert cells[7] == "—"

    def test_row_enrichment_with_stats(self):
        """§4 P2 — when stats are passed, the Last Run cell
        carries the recency strip, Runs picks the local
        count, and the Cost cell renders the mean USD."""
        import time

        from care.runtime.local_run_history import ChainRunStats

        row = LibraryRow(
            entity_id="x", display_name="X", run_count=2,
        )
        stats = ChainRunStats(
            chain_id="x",
            run_count=5,
            success_count=4,
            last_run_at=time.time() - 60 * 60 * 2,  # 2h ago
            mean_cost_usd=0.42,
        )
        cells = LibraryScreen._row_cells(row, stats)
        # Last Run cell carries the recency strip with rate.
        assert "0.80" in cells[4]
        assert "/5" in cells[4]
        # Runs picks the higher (local) count.
        assert cells[5] == "5"
        # Cost cell renders $0.42.
        assert cells[7] == "$0.42"

    def test_row_enrichment_keeps_memory_run_count_when_local_lower(
        self,
    ):
        # Stats present but local count is lower than Memory's
        # — Memory's count wins so we don't show a regression.
        from care.runtime.local_run_history import ChainRunStats

        row = LibraryRow(
            entity_id="x", display_name="X", run_count=20,
        )
        stats = ChainRunStats(
            chain_id="x", run_count=3, success_count=3,
        )
        cells = LibraryScreen._row_cells(row, stats)
        assert cells[5] == "20"


# ---------------------------------------------------------------------------
# Refresh hook
# ---------------------------------------------------------------------------


class TestRefreshLibrary:
    @pytest.mark.asyncio
    async def test_refresh_library_reruns_worker(self):
        client = _StubClient(rows=[_row(entity_id="a", display_name="Alpha")])
        app = _HostApp(memory=_StubMemory(client))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            initial_calls = len(client.calls)
            app.screen.refresh_library()
            await pilot.pause()
            await pilot.pause()
            assert len(client.calls) > initial_calls


# ---------------------------------------------------------------------------
# Search binding (TODO §4 P0)
# ---------------------------------------------------------------------------


class TestSearchBinding:
    """Pin the `/` + `Ctrl+F` keybindings to
    `action_focus_search` so the chat-tool-convention vim
    gesture lands the user on the sidebar's search input.
    The server-side search scope (name / description / tags)
    is owned by Memory's `list_chains(q=...)` — verified by
    the existing sidebar tests; here we only validate the
    binding-to-focus contract.
    """

    def test_slash_and_ctrl_f_bound_to_focus_search(self):
        # §4 P2 — `/` routes through a distinct action
        # (`focus_search_absorb`) that drops the activating
        # `/` keystroke; `Ctrl+F` stays on the plain
        # `focus_search` action.
        action_by_key = {
            b.key: getattr(b, "action", None)
            for b in LibraryScreen.BINDINGS
        }
        assert action_by_key.get("slash") == "focus_search_absorb", (
            f"`/` should be bound to focus_search_absorb; got "
            f"{action_by_key.get('slash')!r}"
        )
        assert action_by_key.get("ctrl+f") == "focus_search", (
            f"`Ctrl+F` should stay on focus_search; got "
            f"{action_by_key.get('ctrl+f')!r}"
        )

    @pytest.mark.asyncio
    async def test_slash_focuses_sidebar_search_input(self):
        from textual.widgets import Input

        client = _StubClient(rows=[_row(entity_id="a")])
        app = _HostApp(memory=_StubMemory(client))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # Action call mirrors the binding's resolution; the
            # screen tests already pilot key chords via
            # `pilot.press` in row-action tests but the focus
            # state is easier to assert from the action than
            # to chase through the keyboard router.
            app.screen.action_focus_search()
            await pilot.pause()
            search = app.screen.query_one(
                "#library-sidebar-search", Input,
            )
            assert app.focused is search, (
                f"expected focus on #library-sidebar-search, "
                f"got {type(app.focused).__name__ if app.focused else None}"
            )

    @pytest.mark.asyncio
    async def test_slash_keypress_routes_through_binding(self):
        """End-to-end check: pressing `/` while the
        LibraryScreen is active focuses the search input via
        Textual's binding dispatch, not just the action call
        path."""
        from textual.widgets import Input

        client = _StubClient(rows=[_row(entity_id="a")])
        app = _HostApp(memory=_StubMemory(client))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            search = app.screen.query_one(
                "#library-sidebar-search", Input,
            )
            assert app.focused is search

    def test_absorber_clears_input_on_changed_with_slash(
        self,
    ):
        """Unit test for the sidebar's one-shot absorber: when
        ``_absorb_next_search_keystroke`` is True and an
        ``Input.Changed`` arrives with value ``"/"``, the
        handler resets the input + clears the flag + drops
        the FiltersChanged broadcast."""
        from unittest.mock import MagicMock

        from care.widgets.library_sidebar import LibrarySidebar

        sidebar = LibrarySidebar()
        sidebar._absorb_next_search_keystroke = True
        # Fake an Input.Changed against the search input.
        fake_input = MagicMock()
        fake_input.id = "library-sidebar-search"
        fake_input.value = "/"
        event = MagicMock()
        event.input = fake_input
        event.value = "/"

        # No FiltersChanged should land — capture posts.
        posted: list = []
        sidebar.post_message = lambda msg: posted.append(msg)

        sidebar.on_input_changed(event)
        assert fake_input.value == "", (
            "absorber must reset input value to empty"
        )
        assert sidebar._absorb_next_search_keystroke is False, (
            "one-shot flag must clear after firing"
        )
        assert sidebar._suppress_count == 1, (
            "secondary Changed from the reset must be queued "
            "for suppression"
        )
        assert posted == [], (
            "absorber must not broadcast FiltersChanged"
        )

    def test_absorber_passes_through_non_slash_keystroke(
        self,
    ):
        """When the absorber is armed but the first Changed
        arrives with a value that ISN'T ``"/"`` (user pressed
        `/` then typed something fast), the absorber clears
        without swallowing the value."""
        from unittest.mock import MagicMock

        from care.widgets.library_sidebar import LibrarySidebar

        sidebar = LibrarySidebar()
        sidebar._absorb_next_search_keystroke = True

        fake_input = MagicMock()
        fake_input.id = "library-sidebar-search"
        fake_input.value = "x"
        event = MagicMock()
        event.input = fake_input
        event.value = "x"

        posted: list = []
        sidebar.post_message = lambda msg: posted.append(msg)

        sidebar.on_input_changed(event)
        # Flag cleared; value retained; FiltersChanged posted.
        assert sidebar._absorb_next_search_keystroke is False
        assert fake_input.value == "x"
        assert len(posted) == 1

    @pytest.mark.asyncio
    async def test_ctrl_f_does_not_set_absorber(self):
        """Ctrl+F preserves any text already in the prompt —
        the absorber flag only arms via the `/` path."""
        from care.runtime.library_view import LibraryFilters
        from care.widgets.library_sidebar import LibrarySidebar
        from textual.widgets import Input

        client = _StubClient(rows=[_row(entity_id="a")])
        app = _HostApp(memory=_StubMemory(client))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            sidebar = app.screen.query_one(LibrarySidebar)
            sidebar.set_filters(
                LibraryFilters(search="existing-query"),
            )
            await pilot.pause()
            app.screen.action_focus_search()
            await pilot.pause()
            search = app.screen.query_one(
                "#library-sidebar-search", Input,
            )
            assert search.value == "existing-query", (
                "Ctrl+F path must not clear the prompt"
            )
            assert sidebar._absorb_next_search_keystroke is False, (
                "Ctrl+F must NOT arm the absorber"
            )

    @pytest.mark.asyncio
    async def test_slash_action_arms_absorber(self):
        """`action_focus_search_absorb` sets the one-shot
        flag + moves focus. (Pilot key dispatch ordering makes
        the end-to-end keypress flaky to drive in pilot — the
        action-level contract is the durable one.)"""
        from care.widgets.library_sidebar import LibrarySidebar
        from textual.widgets import Input

        client = _StubClient(rows=[_row(entity_id="a")])
        app = _HostApp(memory=_StubMemory(client))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            sidebar = app.screen.query_one(LibrarySidebar)
            assert sidebar._absorb_next_search_keystroke is False
            app.screen.action_focus_search_absorb()
            await pilot.pause()
            search = app.screen.query_one(
                "#library-sidebar-search", Input,
            )
            assert app.focused is search
            assert sidebar._absorb_next_search_keystroke is True, (
                "`/` action must arm the absorber"
            )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports_library_screen(self):
        from care.screens import LibraryScreen as ReExported

        assert ReExported is LibraryScreen


# ---------------------------------------------------------------------------
# Tag-pool refresh (§4 P1)
# ---------------------------------------------------------------------------


class TestTagPoolRefresh:
    """`_refresh` pushes the harvested tag pool to the sidebar
    so chips populate automatically.  Two-source resolution:
    `memory.list_tags()` wins; row-tag frequency-rank falls
    back when it doesn't exist or fails."""

    def test_rank_tags_by_frequency_orders_desc_then_alpha(self):
        from care.screens.library import _rank_tags_by_frequency

        rows = (
            LibraryRow(
                entity_id="a", display_name="A",
                tags=("ml", "data"),
            ),
            LibraryRow(
                entity_id="b", display_name="B",
                tags=("ml", "ai"),
            ),
            LibraryRow(
                entity_id="c", display_name="C",
                tags=("ml", "data"),
            ),
            LibraryRow(
                entity_id="d", display_name="D",
                tags=("ai",),
            ),
        )
        # ml=3, data=2, ai=2, then alpha → ml, ai, data
        out = _rank_tags_by_frequency(rows)
        assert out == ("ml", "ai", "data")

    def test_rank_tags_by_frequency_skips_blanks_and_dedupes(
        self,
    ):
        from care.screens.library import _rank_tags_by_frequency

        rows = (
            LibraryRow(
                entity_id="a", display_name="A",
                tags=("ml", "", "  ", "ml"),
            ),
        )
        out = _rank_tags_by_frequency(rows)
        assert out == ("ml",)

    def test_rank_tags_by_frequency_empty(self):
        from care.screens.library import _rank_tags_by_frequency

        assert _rank_tags_by_frequency(()) == ()

    @pytest.mark.asyncio
    async def test_refresh_populates_tag_pool_from_rows(self):
        from care.widgets.library_sidebar import LibrarySidebar

        rows = [
            _row(entity_id="a", display_name="A",
                 tags=("ml", "data")),
            _row(entity_id="b", display_name="B",
                 tags=("ml", "ai")),
            _row(entity_id="c", display_name="C",
                 tags=("ml", "data")),
        ]
        client = _StubClient(rows=rows)
        app = _HostApp(memory=_StubMemory(client))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            sidebar = app.screen.query_one(LibrarySidebar)
            # ml=3, data=2, ai=1.
            assert sidebar.tag_pool == ("ml", "data", "ai")

    @pytest.mark.asyncio
    async def test_refresh_prefers_memory_list_tags(self):
        from care.widgets.library_sidebar import LibrarySidebar

        class _TagListingMemory(_StubMemory):
            def __init__(self, client, *, tags):
                super().__init__(client)
                self._tags = tags
                self.list_tags_calls = 0

            def list_tags(self):
                self.list_tags_calls += 1
                return self._tags

        client = _StubClient(rows=[
            _row(entity_id="a", tags=("row-tag-1", "row-tag-2")),
        ])
        memory = _TagListingMemory(
            client, tags=("server-tag-1", "server-tag-2"),
        )
        app = _HostApp(memory=memory)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            sidebar = app.screen.query_one(LibrarySidebar)
            # `list_tags()` wins over row harvest.
            assert sidebar.tag_pool == (
                "server-tag-1", "server-tag-2",
            )
            assert memory.list_tags_calls >= 1

    @pytest.mark.asyncio
    async def test_refresh_falls_back_when_list_tags_raises(
        self,
    ):
        from care.widgets.library_sidebar import LibrarySidebar

        class _BrokenTagMemory(_StubMemory):
            def list_tags(self):
                raise RuntimeError("upstream offline")

        client = _StubClient(rows=[
            _row(entity_id="a", tags=("ml",)),
            _row(entity_id="b", tags=("ml", "ai")),
        ])
        app = _HostApp(memory=_BrokenTagMemory(client))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            sidebar = app.screen.query_one(LibrarySidebar)
            # `list_tags()` raised → row-harvest fallback.
            assert sidebar.tag_pool == ("ml", "ai")

    @pytest.mark.asyncio
    async def test_refresh_accepts_async_list_tags(self):
        from care.widgets.library_sidebar import LibrarySidebar

        class _AsyncTagMemory(_StubMemory):
            async def list_tags(self):
                return ("alpha", "beta")

        client = _StubClient(rows=[_row(entity_id="a")])
        app = _HostApp(memory=_AsyncTagMemory(client))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            sidebar = app.screen.query_one(LibrarySidebar)
            assert sidebar.tag_pool == ("alpha", "beta")


# ---------------------------------------------------------------------------
# Diff-two-chains binding (§4 P1)
# ---------------------------------------------------------------------------


class TestDiffSelected:
    """`D` opens a DiffModal against the two selected chain rows."""

    def _make_selection(self, *ids: str):
        from care.runtime.bulk_ops import BulkSelection, BulkTarget

        return BulkSelection(
            targets=tuple(
                BulkTarget(entity_id=i, entity_type="chain")
                for i in ids
            ),
        )

    def _make_host(self, *, memory=None):
        class _HostAppWithToasts(App):
            def __init__(self):
                super().__init__()
                self.memory = memory
                self.toasts: list[tuple[str, str]] = []

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(LibraryScreen())

            def push_toast(
                self, message, *, severity="info", ttl=None,
            ) -> None:  # type: ignore[override]
                self.toasts.append((message, severity))

        return _HostAppWithToasts()

    @pytest.mark.asyncio
    async def test_diff_with_no_selection_warns(self):
        app = self._make_host(
            memory=_StubMemory(_StubClient(rows=[_row(entity_id="a")])),
        )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            screen.action_diff_selected()
            await pilot.pause()
            assert any(
                "needs exactly two" in m for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_diff_with_three_selected_warns(self):
        app = self._make_host(
            memory=_StubMemory(_StubClient(rows=[_row(entity_id="a")])),
        )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            screen.bulk_selection = self._make_selection(
                "a", "b", "c",
            )
            screen.action_diff_selected()
            await pilot.pause()
            assert any(
                "needs exactly two" in m for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_diff_without_memory_facade_warns(self):
        app = self._make_host(memory=None)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            screen.bulk_selection = self._make_selection("a", "b")
            screen.action_diff_selected()
            await pilot.pause()
            assert any(
                "configured Memory facade" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_diff_with_non_chain_kind_warns(self):
        from care.runtime.bulk_ops import BulkSelection, BulkTarget

        app = self._make_host(
            memory=_StubMemory(_StubClient(rows=[_row(entity_id="a")])),
        )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            screen.bulk_selection = BulkSelection(targets=(
                BulkTarget(entity_id="a", entity_type="chain"),
                BulkTarget(entity_id="b", entity_type="agent_skill"),
            ))
            screen.action_diff_selected()
            await pilot.pause()
            assert any(
                "chain-only" in m for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_diff_with_two_chains_pushes_modal(self):
        from care.screens.diff import DiffModal

        class _MemWithGetChain(_StubMemory):
            pass

        class _ClientWithGetChain(_StubClient):
            def __init__(self, rows):
                super().__init__(rows)

            def get_chain_dict(self, entity_id, channel="latest"):
                return {
                    "metadata": {"display_name": entity_id},
                    "steps": [
                        {
                            "id": f"s-{entity_id}",
                            "tool_id": "fetch",
                            "config": {},
                        },
                    ],
                }

        client = _ClientWithGetChain(rows=[_row(entity_id="a")])
        app = self._make_host(memory=_MemWithGetChain(client))
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            screen.bulk_selection = self._make_selection("alpha", "beta")
            screen.action_diff_selected()
            for _ in range(6):
                await pilot.pause()
            assert any(
                isinstance(s, DiffModal)
                for s in app.screen_stack
            )

    def test_D_binding_routes_to_diff_selected(self):
        action_keys = {
            b.key for b in LibraryScreen.BINDINGS
            if getattr(b, "action", None) == "diff_selected"
        }
        assert "D" in action_keys, (
            f"`D` should be bound to diff_selected; "
            f"got {sorted(action_keys)}"
        )


# ---------------------------------------------------------------------------
# Import / Export bindings (§4 P1)
# ---------------------------------------------------------------------------


class TestImportExportBindings:
    def _make_host(self, *, memory=None, rows=None):
        if rows is None:
            rows = [_row(entity_id="chain-a")]

        class _Host(App):
            def __init__(self):
                super().__init__()
                self.memory = memory or _StubMemory(
                    _StubClient(rows=rows),
                )
                self.toasts: list[tuple[str, str]] = []

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(LibraryScreen())

            def push_toast(
                self, message, *, severity="info", ttl=None,
            ) -> None:  # type: ignore[override]
                self.toasts.append((message, severity))

        return _Host()

    def test_i_and_x_bindings_registered(self):
        bindings = {
            getattr(b, "action", None): b.key
            for b in LibraryScreen.BINDINGS
        }
        assert bindings.get("import_bundle") == "i"
        assert bindings.get("export_bundle") == "x"

    @pytest.mark.asyncio
    async def test_import_without_memory_warns(self):
        host = self._make_host()
        async with host.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            host.memory = None
            screen = host.screen
            assert isinstance(screen, LibraryScreen)
            screen.action_import_bundle()
            await pilot.pause()
            assert any(
                "configured Memory facade" in m
                for m, _ in host.toasts
            )

    @pytest.mark.asyncio
    async def test_import_pushes_modal(self):
        from care.screens.import_bundle import ImportModal

        host = self._make_host()
        async with host.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = host.screen
            assert isinstance(screen, LibraryScreen)
            screen.action_import_bundle()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, ImportModal)
                for s in host.screen_stack
            )

    @pytest.mark.asyncio
    async def test_export_without_memory_warns(self):
        host = self._make_host()
        async with host.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            host.memory = None
            screen = host.screen
            assert isinstance(screen, LibraryScreen)
            screen.action_export_bundle()
            await pilot.pause()
            assert any(
                "configured Memory facade" in m
                for m, _ in host.toasts
            )

    @pytest.mark.asyncio
    async def test_export_with_empty_table_and_no_selection_warns(
        self,
    ):
        host = self._make_host(rows=[])
        async with host.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = host.screen
            assert isinstance(screen, LibraryScreen)
            screen.action_export_bundle()
            await pilot.pause()
            assert any(
                "focused row or a multi-selection" in m
                for m, _ in host.toasts
            )

    @pytest.mark.asyncio
    async def test_export_with_focused_chain_pushes_modal(self):
        from care.screens.export import ExportModal

        host = self._make_host(rows=[_row(entity_id="chain-z")])
        async with host.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            screen = host.screen
            assert isinstance(screen, LibraryScreen)
            screen.action_export_bundle()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, ExportModal)
                for s in host.screen_stack
            )
            modal = next(
                s for s in host.screen_stack
                if isinstance(s, ExportModal)
            )
            assert modal._entity_ids == ("chain-z",)

    @pytest.mark.asyncio
    async def test_export_with_bulk_selection_uses_chains_only(self):
        from care.runtime.bulk_ops import BulkSelection, BulkTarget
        from care.screens.export import ExportModal

        host = self._make_host(rows=[_row(entity_id="chain-a")])
        async with host.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = host.screen
            assert isinstance(screen, LibraryScreen)
            screen.bulk_selection = BulkSelection(targets=(
                BulkTarget(entity_id="chain-a", entity_type="chain"),
                BulkTarget(entity_id="chain-b", entity_type="chain"),
                BulkTarget(
                    entity_id="skill-x",
                    entity_type="agent_skill",
                ),
            ))
            screen.action_export_bundle()
            for _ in range(4):
                await pilot.pause()
            modal = next(
                s for s in host.screen_stack
                if isinstance(s, ExportModal)
            )
            assert modal._entity_ids == ("chain-a", "chain-b")
            assert modal._skill_ids == ("skill-x",)


# ---------------------------------------------------------------------------
# WelcomeScreen routing now lands on LibraryScreen
# ---------------------------------------------------------------------------


class TestWelcomeRouting:
    def _isolate_config(self, monkeypatch, tmp_path):
        # Point CareConfig.load() at a non-existent file so the
        # developer's real `~/.config/care/config.toml` (and
        # any project-level `./care.toml`) doesn't leak creds
        # into the routing decision.
        from care import config as config_module

        fake_path = tmp_path / "care.toml"
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", fake_path)
        # Wipe every CARE_* env var the dev shell might have set.
        for name in list(__import__("os").environ.keys()):
            if name.startswith("CARE_"):
                monkeypatch.delenv(name, raising=False)

    def test_default_next_screen_returning_with_creds_routes_to_chat(
        self, monkeypatch, tmp_path,
    ):
        from care.screens.chat import ChatScreen
        from care.screens.welcome import default_next_screen

        self._isolate_config(monkeypatch, tmp_path)
        # MAGE key set + Memory pointed at a non-default URL
        # (anonymous-mode local deployment) → ChatScreen.
        # LibraryScreen remains reachable via the `/library`
        # slash command from within ChatScreen.
        monkeypatch.setenv("CARE_MAGE__API_KEY", "sk-test")
        monkeypatch.setenv("CARE_MEMORY__BASE_URL", "http://localhost:8002")
        result = default_next_screen("returning")
        assert isinstance(result, ChatScreen)

    def test_default_next_screen_returning_missing_mage_routes_to_settings(
        self, monkeypatch, tmp_path,
    ):
        from care.screens.settings import SettingsScreen
        from care.screens.welcome import default_next_screen

        self._isolate_config(monkeypatch, tmp_path)
        # No MAGE api_key (true hard gate) → Settings prompt so
        # the user can fill it in before reaching the Library.
        monkeypatch.setenv("CARE_MEMORY__BASE_URL", "http://localhost:8002")
        result = default_next_screen("returning")
        assert isinstance(result, SettingsScreen)

    def test_default_next_screen_first_run_routes_to_settings(self):
        from care.screens.settings import SettingsScreen
        from care.screens.welcome import default_next_screen

        # SettingsScreen (P0.32) has shipped — first_run users
        # land on it so they can configure Memory / MAGE
        # credentials before generating.
        result = default_next_screen("first_run")
        assert isinstance(result, SettingsScreen)


class TestSessionsTab:
    """The Library splits Memory chains across two tabs by name:
    deliberately-named saved DAGs → Saved; auto-named "General …"
    chains → Sessions."""

    @pytest.mark.asyncio
    async def test_both_tabs_present(self):
        from textual.widgets import TabPane

        app = _HostApp(memory=_StubMemory(_StubClient(rows=[])))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            assert screen.query_one("#library-tab-saved", TabPane) is not None
            assert (
                screen.query_one("#library-tab-sessions", TabPane) is not None
            )

    @pytest.mark.asyncio
    async def test_general_named_chains_go_to_sessions_tab(self):
        rows = [
            _row(entity_id="a", display_name="Weather forecaster"),
            _row(entity_id="b", display_name="General - quick task"),
            _row(entity_id="c", display_name="general helper"),
        ]
        app = _HostApp(memory=_StubMemory(_StubClient(rows=rows)))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen
            assert [r.entity_id for r in screen._saved_rows] == ["a"]
            assert {r.entity_id for r in screen._session_rows} == {"b", "c"}
            saved_tbl = screen.query_one("#library-table", DataTable)
            sess_tbl = screen.query_one("#library-sessions-table", DataTable)
            assert saved_tbl.row_count == 1
            assert sess_tbl.row_count == 2

    @pytest.mark.asyncio
    async def test_sessions_empty_when_no_general(self):
        rows = [_row(entity_id="a", display_name="Real DAG")]
        app = _HostApp(memory=_StubMemory(_StubClient(rows=rows)))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen
            assert screen._session_rows == []
            sess_tbl = screen.query_one("#library-sessions-table", DataTable)
            assert sess_tbl.row_count == 0

    @pytest.mark.asyncio
    async def test_current_row_is_active_tab_aware(self):
        from textual.widgets import TabbedContent

        rows = [
            _row(entity_id="a", display_name="Real DAG"),
            _row(entity_id="b", display_name="General - quick task"),
        ]
        app = _HostApp(memory=_StubMemory(_StubClient(rows=rows)))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = app.screen
            # Saved tab active → current_row resolves to the saved row.
            assert screen.current_row.entity_id == "a"
            # Switch to Sessions → current_row resolves to the General row.
            screen.query_one(
                "#library-tabs", TabbedContent,
            ).active = "library-tab-sessions"
            await pilot.pause()
            assert screen.current_row.entity_id == "b"
