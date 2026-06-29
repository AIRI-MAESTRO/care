"""Pilot tests for MarketplaceScreen (§8 P2 [DONE — data half] → DONE).

Wires :func:`care.search_marketplace` into a Textual ``Screen``
that lets the user browse + install shared `agent_skill`
listings. Tests exercise:

* Compose — search bar / tag sidebar / results table all mount.
* Initial query seeds the search.
* New queries fire `search_marketplace` (debounce honoured for
  typed input, immediate for `Enter`).
* Results render in score order.
* Tag-chip selection narrows the visible listings.
* Install action calls `memory.client.save_agent_skill` and
  posts a `MarketplaceScreen.Installed` message.
* Backend failure surfaces on `last_error` + the status line.
* Re-exports.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input, Static

from care.screens.marketplace import (
    MarketplaceInstalled,
    MarketplaceScreen,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _hit(
    *,
    entity_id: str,
    name: str,
    score: float = 0.5,
    description: str = "",
    tags=(),
    matched_via: str = "skill_description",
) -> dict:
    """Build a CapabilityHit-shaped dict the data layer accepts."""
    return {
        "entity_id": entity_id,
        "name": name,
        "description": description,
        "score": score,
        "tags": list(tags),
        "matched_via": matched_via,
        "snippet": None,
    }


class _StubSaveClient:
    def __init__(self, *, raise_save: bool = False):
        self._raise = raise_save
        self.save_calls: list[dict] = []

    def save_agent_skill(self, **kw):
        self.save_calls.append(dict(kw))
        if self._raise:
            raise RuntimeError("save-down")
        return {"entity_id": "saved-1", "name": kw.get("name", "")}


class _StubMarketMemory:
    """Test double for the memory facade. Exposes both
    `find_capability_matches` (search) + `client.save_agent_skill`
    (install) for end-to-end coverage."""

    def __init__(
        self,
        *,
        hits: list[dict] | None = None,
        raise_search: bool = False,
        raise_save: bool = False,
    ):
        self._hits = list(hits or [])
        self._raise_search = raise_search
        self.search_calls: list[dict] = []
        self.client = _StubSaveClient(raise_save=raise_save)

    def find_capability_matches(
        self, query, *, top_k=10, namespace=None, deep=False,
    ):
        self.search_calls.append({
            "query": query,
            "top_k": top_k,
            "namespace": namespace,
            "deep": deep,
        })
        if self._raise_search:
            raise RuntimeError("search-down")
        return list(self._hits)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(
        self,
        *,
        memory=None,
        initial_query: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self._marketplace_memory = memory
        self._initial_query = initial_query
        self._extra_kwargs = kwargs
        self.installed: list[MarketplaceInstalled] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(MarketplaceScreen(
            memory=self._marketplace_memory,
            initial_query=self._initial_query,
            **self._extra_kwargs,
        ))

    def on_marketplace_screen_installed(
        self, event: MarketplaceScreen.Installed,
    ) -> None:
        self.installed.append(
            MarketplaceInstalled(
                listing=event.listing,
                saved_entity_id=event.saved_entity_id,
            ),
        )


def _screen(app: App) -> MarketplaceScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, MarketplaceScreen)
    return s


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_panes_mount(self):
        app = _Host(memory=_StubMarketMemory())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.query_one(
                "#marketplace-search-input", Input,
            ) is not None
            assert screen.query_one(
                "#marketplace-table", DataTable,
            ) is not None
            assert screen.query_one(
                "#marketplace-tag-list",
            ) is not None
            assert screen.query_one(
                "#marketplace-status", Static,
            ) is not None


# ---------------------------------------------------------------------------
# Initial query + search worker
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.mark.asyncio
    async def test_initial_query_fires_search(self):
        memory = _StubMarketMemory(hits=[
            _hit(entity_id="sk-1", name="pdf-extract", score=0.91),
            _hit(entity_id="sk-2", name="csv-parser", score=0.7),
        ])
        app = _Host(memory=memory, initial_query="extract pdf")
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            assert memory.search_calls
            assert memory.search_calls[0]["query"] == "extract pdf"
            screen = _screen(app)
            assert screen.result.listings
            assert screen.result.listings[0].score == pytest.approx(0.91)

    @pytest.mark.asyncio
    async def test_enter_submits_query(self):
        memory = _StubMarketMemory(hits=[
            _hit(entity_id="sk-1", name="pdf-extract"),
        ])
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.query_text ="pdf"
            screen.action_submit_search()
            for _ in range(6):
                await pilot.pause()
            assert memory.search_calls
            assert memory.search_calls[-1]["query"] == "pdf"
            assert len(screen.result.listings) == 1

    @pytest.mark.asyncio
    async def test_empty_query_short_circuits_search(self):
        memory = _StubMarketMemory(hits=[])
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.query_text =""
            screen.action_submit_search()
            for _ in range(4):
                await pilot.pause()
            # The data layer short-circuits empty queries — no
            # backend call should have happened.
            assert memory.search_calls == []
            assert screen.result.is_empty


# ---------------------------------------------------------------------------
# Tag-chip filter
# ---------------------------------------------------------------------------


class TestTagFilter:
    @pytest.mark.asyncio
    async def test_collect_tags_distinct_in_order(self):
        memory = _StubMarketMemory(hits=[
            _hit(
                entity_id="sk-1", name="a",
                tags=["pdf", "finance"],
            ),
            _hit(
                entity_id="sk-2", name="b",
                tags=["finance", "ocr"],
            ),
        ])
        app = _Host(memory=memory, initial_query="any")
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            screen = _screen(app)
            assert screen.collect_tags() == ("pdf", "finance", "ocr")

    @pytest.mark.asyncio
    async def test_select_tag_narrows_results(self):
        memory = _StubMarketMemory(hits=[
            _hit(
                entity_id="sk-1", name="a",
                tags=["pdf", "finance"],
            ),
            _hit(
                entity_id="sk-2", name="b",
                tags=["ocr"],
            ),
        ])
        app = _Host(memory=memory, initial_query="any")
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            screen = _screen(app)
            assert len(screen.visible_listings()) == 2
            screen._select_tag("pdf")
            assert len(screen.visible_listings()) == 1
            assert screen.visible_listings()[0].entity_id == "sk-1"
            # Re-selecting the active tag clears the filter.
            screen._select_tag("pdf")
            assert screen.selected_tag is None
            assert len(screen.visible_listings()) == 2


# ---------------------------------------------------------------------------
# Install action
# ---------------------------------------------------------------------------


class TestInstall:
    @pytest.mark.asyncio
    async def test_install_calls_save_and_posts_message(self):
        memory = _StubMarketMemory(hits=[
            _hit(entity_id="sk-1", name="pdf-extract", score=0.91),
        ])
        app = _Host(memory=memory, initial_query="pdf")
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            screen = _screen(app)
            assert screen.result.listings
            screen.selected_listing = screen.result.listings[0]
            screen.action_install_selected()
            for _ in range(6):
                await pilot.pause()
            assert memory.client.save_calls
            call = memory.client.save_calls[0]
            assert call["entity_id"] == "sk-1"
            assert call["name"] == "pdf-extract"
            # Host received the Installed message.
            assert app.installed
            assert app.installed[0].listing.entity_id == "sk-1"
            assert app.installed[0].saved_entity_id == "saved-1"

    @pytest.mark.asyncio
    async def test_install_no_selection_is_noop(self):
        memory = _StubMarketMemory(hits=[])
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.selected_listing is None
            screen.action_install_selected()
            for _ in range(4):
                await pilot.pause()
            assert memory.client.save_calls == []
            assert app.installed == []

    @pytest.mark.asyncio
    async def test_install_failure_records_error(self):
        memory = _StubMarketMemory(
            hits=[_hit(entity_id="sk-1", name="x")],
            raise_save=True,
        )
        app = _Host(memory=memory, initial_query="any")
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            screen = _screen(app)
            screen.selected_listing = screen.result.listings[0]
            screen.action_install_selected()
            for _ in range(15):
                await pilot.pause()
            # The stub recorded a save call (so the worker ran),
            # the screen's last_error captured the failure, and
            # no Installed message was posted.
            assert memory.client.save_calls
            assert screen.last_error is not None
            assert "install failed" in screen.last_error
            assert "save-down" in screen.last_error
            assert app.installed == []


# ---------------------------------------------------------------------------
# Backend failure paths
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.asyncio
    async def test_search_backend_failure_lands_on_status(self):
        memory = _StubMarketMemory(raise_search=True)
        app = _Host(memory=memory, initial_query="boom")
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            screen = _screen(app)
            assert screen.last_error is not None
            assert screen.result.is_empty

    @pytest.mark.asyncio
    async def test_no_memory_lands_on_error(self):
        app = _Host(memory=None, initial_query="x")
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            screen = _screen(app)
            assert screen.last_error is not None
            assert "no memory facade" in screen.last_error


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


class TestRender:
    @pytest.mark.asyncio
    async def test_high_signal_match_gets_star(self):
        memory = _StubMarketMemory(hits=[
            _hit(
                entity_id="sk-1", name="a",
                matched_via="skill_description",
            ),
            _hit(
                entity_id="sk-2", name="b",
                matched_via="skill_instructions",
            ),
        ])
        app = _Host(memory=memory, initial_query="any")
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            screen = _screen(app)
            table = screen.query_one(
                "#marketplace-table", DataTable,
            )
            badges = {}
            for row_key in table.rows:
                cell = table.get_cell(
                    row_key, table.ordered_columns[0].key,
                )
                badges[str(row_key.value)] = cell
            assert badges["sk-1"] == "★"
            assert badges["sk-2"] == ""

    @pytest.mark.asyncio
    async def test_tag_buttons_render_per_unique_tag(self):
        memory = _StubMarketMemory(hits=[
            _hit(
                entity_id="sk-1", name="a",
                tags=["pdf", "finance"],
            ),
            _hit(
                entity_id="sk-2", name="b",
                tags=["pdf"],
            ),
        ])
        app = _Host(memory=memory, initial_query="any")
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            screen = _screen(app)
            tag_buttons = list(
                screen.query("#marketplace-tag-list Button"),
            )
            ids = {str(b.id) for b in tag_buttons}
            assert "marketplace-tag-pdf" in ids
            assert "marketplace-tag-finance" in ids


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import MarketplaceScreen as M

        assert M is MarketplaceScreen


# ---------------------------------------------------------------------------
# /marketplace slash command (§6 P1)
# ---------------------------------------------------------------------------


class TestSlashIntegration:
    @pytest.mark.asyncio
    async def test_bare_marketplace_pushes_screen(self):
        from care.screens.chat import ChatScreen
        from care.widgets.chat_input import ChatInput

        class _ChatHost(App):
            def compose(self):
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(ChatScreen())

        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = next(
                s for s in app.screen_stack if isinstance(s, ChatScreen)
            )
            inp = chat.query_one("#chat-input", ChatInput)
            inp.value = "/marketplace"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, MarketplaceScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_marketplace_with_query_arg_prefills_input(self):
        from textual.widgets import Input
        from care.screens.chat import ChatScreen
        from care.widgets.chat_input import ChatInput

        class _ChatHost(App):
            def compose(self):
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(ChatScreen())

        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = next(
                s for s in app.screen_stack if isinstance(s, ChatScreen)
            )
            inp = chat.query_one("#chat-input", ChatInput)
            inp.value = "/marketplace summarise pdf"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            screen = next(
                s for s in app.screen_stack
                if isinstance(s, MarketplaceScreen)
            )
            search = screen.query_one(
                "#marketplace-search-input", Input,
            )
            assert search.value == "summarise pdf"
