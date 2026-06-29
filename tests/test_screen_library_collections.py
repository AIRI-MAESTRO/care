"""Pilot tests for LibraryScreen collections sidebar
(TODO §1.1 P0.14).

Exercises:
* `_refresh` worker aggregates collections via
  :func:`list_collections` and feeds the sidebar.
* Sidebar emits :class:`LibrarySidebar.CollectionSelected`
  when the user picks a node.
* The screen routes that through :func:`filter_by_collection`
  so the existing tag chips stay intact and the worker re-
  runs against the new filter.
* The "All" entry clears the active collection tag.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import OptionList

from care.runtime.collections import collection_tag_for
from care.runtime.library_view import LibraryFilters
from care.screens.library import LibraryScreen
from care.widgets.library_sidebar import LibrarySidebar


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _row(entity_id, *, tags=()) -> dict:
    return {
        "entity_type": "chain",
        "entity_id": entity_id,
        "version_id": "v1",
        "channel": "latest",
        "etag": "e",
        "favourite": False,
        "run_count": 0,
        "last_run_at": None,
        "display_name": entity_id.title(),
        "description": "",
        "meta": {"tags": list(tags), "name": entity_id},
        "content": {"steps": []},
        "evolution_meta": None,
    }


class _StubClient:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls: list[dict] = []

    def list_chains(self, **kw):
        self.calls.append(dict(kw))
        # Filter by tag AND-match to mimic Memory's contract.
        wanted_tags = kw.get("tags") or []
        rows = []
        for r in self.rows:
            row_tags = set(r["meta"].get("tags") or [])
            if all(t in row_tags for t in wanted_tags):
                rows.append(dict(r))
        return rows


class _StubMemory:
    def __init__(self, rows):
        self.client = _StubClient(rows)


class _LibHost(App):
    def __init__(self, rows):
        super().__init__()
        self.memory = _StubMemory(rows)

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(LibraryScreen(restore_state=False))


def _rows_with_collections():
    return [
        _row("alpha", tags=["collection:Marketing", "urgent"]),
        _row("beta", tags=["collection:Marketing"]),
        _row("gamma", tags=["collection:Research"]),
        _row("delta", tags=[]),
    ]


def _library(app: App) -> LibraryScreen:
    screen = app.screen_stack[-1]
    assert isinstance(screen, LibraryScreen)
    return screen


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    @pytest.mark.asyncio
    async def test_refresh_populates_sidebar_collections(self):
        app = _LibHost(_rows_with_collections())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            names = {c.name for c in lib.collections}
            assert names == {"Marketing", "Research"}
            sidebar = lib.query_one(LibrarySidebar)
            # Options include the synthetic "All collections" +
            # one per collection.
            option_list = sidebar.query_one(
                "#library-sidebar-collections", OptionList,
            )
            assert option_list.option_count == 3

    @pytest.mark.asyncio
    async def test_empty_library_has_no_collections(self):
        app = _LibHost([_row("alpha", tags=["urgent"])])
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            assert lib.collections == ()


# ---------------------------------------------------------------------------
# Click-to-filter
# ---------------------------------------------------------------------------


class TestClickToFilter:
    @pytest.mark.asyncio
    async def test_selecting_collection_pins_tag(self):
        app = _LibHost(_rows_with_collections())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            sidebar = lib.query_one(LibrarySidebar)
            sidebar.post_message(
                LibrarySidebar.CollectionSelected("Marketing"),
            )
            await pilot.pause()
            await pilot.pause()
            assert collection_tag_for("Marketing") in lib.filters.tags

    @pytest.mark.asyncio
    async def test_all_clears_collection_tag(self):
        app = _LibHost(_rows_with_collections())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            sidebar = lib.query_one(LibrarySidebar)
            sidebar.post_message(
                LibrarySidebar.CollectionSelected("Marketing"),
            )
            await pilot.pause()
            await pilot.pause()
            assert collection_tag_for("Marketing") in lib.filters.tags
            sidebar.post_message(LibrarySidebar.CollectionSelected(None))
            await pilot.pause()
            await pilot.pause()
            assert collection_tag_for("Marketing") not in lib.filters.tags

    @pytest.mark.asyncio
    async def test_collection_filter_passes_through_to_fetch(self):
        app = _LibHost(_rows_with_collections())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            sidebar = lib.query_one(LibrarySidebar)
            sidebar.post_message(
                LibrarySidebar.CollectionSelected("Research"),
            )
            await pilot.pause()
            await pilot.pause()
            # The fetch worker should have called list_chains
            # with the collection tag (the refresh-collections
            # pass that follows lists every chain).
            tagged_calls = [
                c for c in app.memory.client.calls
                if collection_tag_for("Research") in (c.get("tags") or [])
            ]
            assert tagged_calls != []

    @pytest.mark.asyncio
    async def test_existing_tags_preserved_when_collection_picked(self):
        # Construct screen with a pre-existing tag filter.
        class _App(App):
            def __init__(self, rows):
                super().__init__()
                self.memory = _StubMemory(rows)

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(
                    LibraryScreen(
                        restore_state=False,
                        filters=LibraryFilters(tags=("urgent",)),
                    ),
                )

        app = _App(_rows_with_collections())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            sidebar = lib.query_one(LibrarySidebar)
            sidebar.post_message(
                LibrarySidebar.CollectionSelected("Marketing"),
            )
            await pilot.pause()
            await pilot.pause()
            assert "urgent" in lib.filters.tags
            assert collection_tag_for("Marketing") in lib.filters.tags


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_collection_selected_is_in_sidebar(self):
        assert hasattr(LibrarySidebar, "CollectionSelected")
        assert hasattr(LibrarySidebar, "CollectionActionRequested")
