"""Pilot tests for LibraryScreen sort + persistence (TODO §1.1 P0.10).

Exercises:
* `load_view_state()` restore on mount (when no kwargs passed).
* Column-header clicks flip the sort field / direction.
* Saved sort persists across screen instances; content filters are
  intentionally NOT restored on open.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

from care.runtime.library_view import (
    LibraryFilters,
    LibrarySort,
    LibraryViewState,
    save_view_state,
)
from care.screens.library import LibraryScreen


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_view_state(tmp_path, monkeypatch):
    """Point the view-state store at a tmp path so tests
    don't touch the user's real preference file."""
    from care.runtime import library_view as lv_module

    fake_path = tmp_path / "library_view.json"
    monkeypatch.setattr(lv_module, "DEFAULT_VIEW_STATE_PATH", fake_path)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(self):
        self.calls = []

    def list_chains(self, **kw):
        self.calls.append(kw)
        return []


class _StubMemory:
    def __init__(self):
        self.client = _StubClient()


class _LibHost(App):
    def __init__(self, *, screen=None, memory=None):
        super().__init__()
        self.memory = memory if memory is not None else _StubMemory()
        self._initial_screen = screen

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self._initial_screen or LibraryScreen())


# ---------------------------------------------------------------------------
# Constructor restore from view state
# ---------------------------------------------------------------------------


class TestConstructorRestore:
    def test_default_no_state_uses_defaults(self):
        screen = LibraryScreen()
        # Default sort + filters when no file persisted yet.
        assert screen.sort == LibrarySort()
        assert screen.filters == LibraryFilters()

    def test_restore_persists_sort_but_not_content_filters(
        self, tmp_path, monkeypatch,
    ):
        # Pre-write a state file carrying both a sort + a content filter.
        save_view_state(
            LibraryViewState(
                sort=LibrarySort(field="run_count", direction="asc"),
                filters=LibraryFilters(search="storm"),
            )
        )
        screen = LibraryScreen()
        # Sort preference is restored…
        assert screen.sort.field == "run_count"
        assert screen.sort.direction == "asc"
        # …but content filters are intentionally dropped so a freshly
        # saved chain isn't hidden behind a stale filter on open.
        assert screen.filters.search == ""
        assert screen.filters.is_filtering is False

    def test_explicit_sort_kwarg_wins(self):
        save_view_state(
            LibraryViewState(
                sort=LibrarySort(field="run_count", direction="asc"),
            )
        )
        explicit = LibrarySort(field="display_name", direction="desc")
        screen = LibraryScreen(sort=explicit)
        assert screen.sort is explicit

    def test_explicit_filters_kwarg_wins(self):
        save_view_state(
            LibraryViewState(filters=LibraryFilters(search="from-state"))
        )
        explicit = LibraryFilters(search="from-kwarg")
        screen = LibraryScreen(filters=explicit)
        assert screen.filters is explicit

    def test_restore_state_false_uses_defaults(self):
        save_view_state(
            LibraryViewState(
                sort=LibrarySort(field="run_count", direction="asc"),
            )
        )
        screen = LibraryScreen(restore_state=False)
        assert screen.sort == LibrarySort()


# ---------------------------------------------------------------------------
# Column-header click → sort flip
# ---------------------------------------------------------------------------


class TestCreationDateOrdering:
    """The Library lists saved chains + sessions newest-first by default
    (`created_at` descending), regardless of the order the server returns."""

    @staticmethod
    def _view(rows):
        from care.runtime.library_view import LibraryView

        return LibraryView(
            rows=tuple(rows),
            filters=LibraryFilters(),
            sort=LibrarySort(),
            total_returned=len(rows),
            has_more=False,
            next_cursor=None,
        )

    @staticmethod
    def _row(name, day, *, fav=False):
        from datetime import datetime, timezone

        from care.runtime.library_view import LibraryRow

        return LibraryRow(
            entity_id=name,
            display_name=name,
            created_at=datetime(2026, 1, day, tzinfo=timezone.utc),
            favourite=fav,
        )

    @pytest.mark.asyncio
    async def test_saved_and_sessions_newest_first(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            scr = app.screen
            scr.sort = LibrarySort()  # default created_at desc
            # Deliberately scrambled creation order from the "server".
            view = self._view([
                self._row("Alpha", 3),
                self._row("Bravo", 1),
                self._row("Charlie", 5),       # newest saved
                self._row("general-x", 2),
                self._row("general-y", 9),      # newest session
            ])
            scr.view = view
            scr._populate_table(view)
            await pilot.pause()
            assert [r.display_name for r in scr._saved_rows] == [
                "Charlie", "Alpha", "Bravo",
            ]
            assert [r.display_name for r in scr._session_rows] == [
                "general-y", "general-x",
            ]

    @pytest.mark.asyncio
    async def test_rows_without_created_at_sort_last(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            scr = app.screen
            scr.sort = LibrarySort()
            from care.runtime.library_view import LibraryRow

            no_ts = LibraryRow(entity_id="NoTs", display_name="NoTs")
            view = self._view([no_ts, self._row("New", 9), self._row("Old", 2)])
            scr.view = view
            scr._populate_table(view)
            await pilot.pause()
            assert [r.display_name for r in scr._saved_rows] == [
                "New", "Old", "NoTs",
            ]


class TestColumnHeaderClick:
    @pytest.mark.asyncio
    async def test_click_name_column_sorts_by_display_name(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            table = app.screen.query_one("#library-table", DataTable)
            table.post_message(DataTable.HeaderSelected(
                data_table=table,
                column_key=table.ordered_columns[1].key,
                column_index=1,
                label=None,
            ))
            await pilot.pause()
            assert app.screen.sort.field == "display_name"

    @pytest.mark.asyncio
    async def test_click_runs_column_sorts_by_run_count(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            table = app.screen.query_one("#library-table", DataTable)
            table.post_message(DataTable.HeaderSelected(
                data_table=table,
                column_key=table.ordered_columns[5].key,
                column_index=5,
                label=None,
            ))
            await pilot.pause()
            assert app.screen.sort.field == "run_count"

    @pytest.mark.asyncio
    async def test_click_same_column_toggles_direction(self):
        # Default sort is `created_at desc` (no clickable column header),
        # so first click "Last Run" to land on that field, then clicking
        # the SAME column toggles direction.
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.screen.sort.field == "created_at"
            assert app.screen.sort.direction == "desc"
            table = app.screen.query_one("#library-table", DataTable)

            def _click_last_run() -> None:
                table.post_message(DataTable.HeaderSelected(
                    data_table=table,
                    column_key=table.ordered_columns[4].key,
                    column_index=4,
                    label=None,
                ))

            # First click switches the field → last_run_at desc.
            _click_last_run()
            await pilot.pause()
            assert app.screen.sort.field == "last_run_at"
            assert app.screen.sort.direction == "desc"
            # Same column again → toggle to asc.
            _click_last_run()
            await pilot.pause()
            assert app.screen.sort.direction == "asc"
            # Again → back to desc.
            _click_last_run()
            await pilot.pause()
            assert app.screen.sort.direction == "desc"

    @pytest.mark.asyncio
    async def test_click_non_sortable_column_noop(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            initial_sort = app.screen.sort
            # Click "Domain" (index 2 — not sortable).
            table = app.screen.query_one("#library-table", DataTable)
            table.post_message(DataTable.HeaderSelected(
                data_table=table,
                column_key=table.ordered_columns[2].key,
                column_index=2,
                label=None,
            ))
            await pilot.pause()
            assert app.screen.sort == initial_sort

    @pytest.mark.asyncio
    async def test_click_switches_field_resets_direction(self):
        # Start: created_at desc (default). Click Name column → switch
        # to display_name desc (not toggle to asc).
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            table = app.screen.query_one("#library-table", DataTable)
            table.post_message(DataTable.HeaderSelected(
                data_table=table,
                column_key=table.ordered_columns[1].key,
                column_index=1,
                label=None,
            ))
            await pilot.pause()
            assert app.screen.sort.field == "display_name"
            assert app.screen.sort.direction == "desc"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    @pytest.mark.asyncio
    async def test_sort_change_persists(self):
        from care.runtime.library_view import load_view_state

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            table = app.screen.query_one("#library-table", DataTable)
            table.post_message(DataTable.HeaderSelected(
                data_table=table,
                column_key=table.ordered_columns[1].key,
                column_index=1,
                label=None,
            ))
            await pilot.pause()
            persisted = load_view_state()
            assert persisted is not None
            assert persisted.sort.field == "display_name"

    @pytest.mark.asyncio
    async def test_filter_change_persists(self):
        from care.runtime.library_view import load_view_state
        from care.widgets.library_sidebar import LibrarySidebar
        from textual.widgets import Input

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            sidebar = app.screen.query_one(LibrarySidebar)
            search = sidebar.query_one("#library-sidebar-search", Input)
            search.value = "weather"
            await pilot.pause()
            await pilot.pause()
            persisted = load_view_state()
            assert persisted is not None
            assert persisted.filters.search == "weather"


# ---------------------------------------------------------------------------
# Save degrades on OSError
# ---------------------------------------------------------------------------


class TestSaveResilience:
    def test_save_view_state_swallows_os_error(self, monkeypatch):
        from care.runtime import library_view as lv_module

        def _boom(state, **kw):
            raise OSError("read-only fs")

        monkeypatch.setattr(lv_module, "save_view_state", _boom)
        # Also patch the symbol imported into the screen module.
        from care.screens import library as lib_module

        monkeypatch.setattr(lib_module, "save_view_state", _boom)

        screen = LibraryScreen(restore_state=False)
        # Calling _save_view_state when the persist fails
        # should NOT raise.
        screen._save_view_state()
