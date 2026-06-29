"""Pilot tests for WelcomeScreen recents sidebar (TODO §1.1 P0.34).

Exercises:
* `on_mount` fires `fetch_library_view(sort=LibrarySort(
  "last_run_at"))` when `app.memory` is wired.
* Recents pane renders up to `recents_limit` rows.
* Selecting a row posts `WelcomeScreen.RecentSelected`.
* No-memory hosts skip the fetch but compose cleanly.
* Empty library shows `(no runs yet)` placeholder.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ListItem, ListView

from care.runtime.library_view import LibraryRow
from care.screens.welcome import WelcomeScreen


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _row(entity_id: str, **overrides) -> dict:
    base = {
        "entity_type": "chain",
        "entity_id": entity_id,
        "version_id": "v1",
        "channel": "latest",
        "etag": "e",
        "favourite": False,
        "run_count": 1,
        "last_run_at": datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
        "display_name": entity_id.title(),
        "description": "",
        "meta": {"tags": [], "name": entity_id},
        "content": {"steps": []},
        "evolution_meta": None,
    }
    base.update(overrides)
    return base


class _StubClient:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls: list[dict] = []

    def list_chains(self, **kw):
        self.calls.append(dict(kw))
        return [dict(r) for r in self.rows]


class _StubMemory:
    def __init__(self, rows):
        self.client = _StubClient(rows)


class _Host(App):
    def __init__(
        self,
        *,
        memory=None,
        rows=None,
        splash_seconds: float = 5.0,
    ) -> None:
        super().__init__()
        if memory is None and rows is not None:
            memory = _StubMemory(rows)
        self.memory = memory
        self._splash_seconds = splash_seconds
        self.selected: list[LibraryRow] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        # Long splash so the auto-route doesn't fire before the
        # tests' first pause.
        self.push_screen(
            WelcomeScreen(splash_seconds=self._splash_seconds),
        )

    def on_welcome_screen_recent_selected(
        self, event: WelcomeScreen.RecentSelected,
    ) -> None:
        self.selected.append(event.row)


def _welcome(app: App) -> WelcomeScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, WelcomeScreen)
    return s


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


class TestLoad:
    @pytest.mark.asyncio
    async def test_recents_populates_pane(self):
        app = _Host(
            rows=[
                _row("alpha"),
                _row("beta"),
                _row("gamma"),
            ],
        )
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            screen = _welcome(app)
            assert len(screen.recents) == 3
            listview = screen.query_one(
                "#welcome-recents-list", ListView,
            )
            # 3 rows (placeholder swapped out).
            items = list(listview.query(ListItem))
            assert len(items) == 3

    @pytest.mark.asyncio
    async def test_recents_respects_limit(self):
        rows = [_row(f"agent-{i}") for i in range(10)]
        app = _Host(rows=rows)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            screen = _welcome(app)
            assert len(screen.recents) == screen.DEFAULT_RECENTS_LIMIT

    @pytest.mark.asyncio
    async def test_no_memory_skips_fetch_but_composes(self):
        app = _Host(memory=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _welcome(app)
            # Pane mounts even without a memory facade.
            assert screen.query_one(
                "#welcome-recents-list", ListView,
            ) is not None
            assert screen.recents == ()

    @pytest.mark.asyncio
    async def test_sort_passed_to_fetch(self):
        rows = [_row("alpha")]
        memory = _StubMemory(rows)
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            # list_chains should have been called with the
            # last_run_at sort.
            assert memory.client.calls != []
            kw = memory.client.calls[0]
            assert kw.get("sort_by") == "last_run_at"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class TestSelection:
    @pytest.mark.asyncio
    async def test_selecting_row_posts_message(self):
        app = _Host(
            rows=[
                _row("alpha"),
                _row("beta"),
            ],
        )
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            screen = _welcome(app)
            assert len(screen.recents) == 2
            listview = screen.query_one(
                "#welcome-recents-list", ListView,
            )
            # Simulate clicking the first item.
            listview.index = 0
            await pilot.pause()
            listview.post_message(
                ListView.Selected(
                    listview, listview.children[0], index=0,
                ),
            )
            for _ in range(3):
                await pilot.pause()
            assert len(app.selected) == 1
            assert app.selected[0].entity_id in {"alpha", "beta"}


# ---------------------------------------------------------------------------
# Empty library
# ---------------------------------------------------------------------------


class TestEmpty:
    @pytest.mark.asyncio
    async def test_empty_library_shows_placeholder(self):
        app = _Host(rows=[])
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            screen = _welcome(app)
            assert screen.recents == ()
            # Placeholder ListItem mounted.
            items = list(
                screen.query("#welcome-recents-list ListItem"),
            )
            assert len(items) >= 1
