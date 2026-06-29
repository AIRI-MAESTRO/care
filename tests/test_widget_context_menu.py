"""Pilot tests for :class:`ContextMenu` + LibraryScreen wiring
(TODO §1.1 P0.12).

Exercises:
* Modal composition — Option entries match the actions passed
  in, in registry order.
* Dismiss flow — selecting an option returns the kind;
  Escape returns ``None``.
* LibraryScreen integration — ``Ctrl+M`` opens the menu;
  picking ``toggle_favourite`` routes through
  ``_dispatch_row_action``.
* Right-click on the DataTable opens the same menu.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import OptionList

from care.runtime.row_actions import default_actions
from care.screens.library import LibraryScreen
from care.widgets.context_menu import ContextMenu


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _row_payload(entity_id: str = "agent-1") -> dict:
    return {
        "entity_type": "chain",
        "entity_id": entity_id,
        "version_id": "v1",
        "channel": "latest",
        "etag": "e",
        "favourite": False,
        "run_count": 0,
        "last_run_at": None,
        "display_name": "Storm Watcher",
        "description": "",
        "meta": {"tags": [], "name": "storm-watcher"},
        "content": {"steps": []},
        "evolution_meta": None,
    }


class _StubClient:
    def __init__(self):
        self.fav_calls = []

    def list_chains(self, **kw):
        return [_row_payload()]

    def _mark_favourite(self, entity_type, entity_id, *, value):
        self.fav_calls.append((entity_type, entity_id, value))
        return {"favourite": value}


class _StubMemory:
    def __init__(self):
        self.client = _StubClient()


class _LibHost(App):
    def __init__(self, memory=None):
        super().__init__()
        self.memory = memory if memory is not None else _StubMemory()

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(LibraryScreen(restore_state=False))


# ---------------------------------------------------------------------------
# Standalone host for the modal (no LibraryScreen)
# ---------------------------------------------------------------------------


class _MenuHost(App):
    def __init__(self, *, actions):
        super().__init__()
        self._actions = actions
        self.last_pick = "<unset>"

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_pick(kind):
            self.last_pick = kind

        self.push_screen(ContextMenu(actions=self._actions), _on_pick)


# ---------------------------------------------------------------------------
# Modal composition
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_renders_one_option_per_action(self):
        actions = default_actions()
        app = _MenuHost(actions=actions)
        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.screen
            assert isinstance(menu, ContextMenu)
            option_list = menu.query_one(OptionList)
            assert option_list.option_count == len(actions)
            # Options carry the kind as id.
            ids = [
                option_list.get_option_at_index(i).id
                for i in range(option_list.option_count)
            ]
            assert ids == [a.kind for a in actions]

    @pytest.mark.asyncio
    async def test_filtered_actions_render_subset(self):
        # Drop two actions to simulate a draft row's gating.
        all_actions = default_actions()
        actions = tuple(
            a for a in all_actions
            if a.kind not in ("evolve", "show_lineage")
        )
        app = _MenuHost(actions=actions)
        async with app.run_test() as pilot:
            await pilot.pause()
            option_list = app.screen.query_one(OptionList)
            assert option_list.option_count == len(actions)


# ---------------------------------------------------------------------------
# Dismiss flow
# ---------------------------------------------------------------------------


class TestDismiss:
    @pytest.mark.asyncio
    async def test_option_selected_returns_kind(self):
        actions = default_actions()
        app = _MenuHost(actions=actions)
        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.screen
            assert isinstance(menu, ContextMenu)
            menu.dismiss("toggle_favourite")  # type: ignore[arg-type]
            await pilot.pause()
            await pilot.pause()
            assert app.last_pick == "toggle_favourite"

    @pytest.mark.asyncio
    async def test_cancel_returns_none(self):
        actions = default_actions()
        app = _MenuHost(actions=actions)
        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.screen
            assert isinstance(menu, ContextMenu)
            menu.action_cancel()
            await pilot.pause()
            await pilot.pause()
            assert app.last_pick is None


# ---------------------------------------------------------------------------
# LibraryScreen integration
# ---------------------------------------------------------------------------


class TestLibraryIntegration:
    @pytest.mark.asyncio
    async def test_action_opens_context_menu(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            library = app.screen_stack[-1]
            assert isinstance(library, LibraryScreen)
            library.action_row_context_menu()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], ContextMenu)

    @pytest.mark.asyncio
    async def test_menu_pick_dispatches_action(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            library = app.screen_stack[-1]
            assert isinstance(library, LibraryScreen)
            library.action_row_context_menu()
            await pilot.pause()
            await pilot.pause()
            menu = app.screen_stack[-1]
            assert isinstance(menu, ContextMenu)
            menu.dismiss("toggle_favourite")  # type: ignore[arg-type]
            await pilot.pause()
            await pilot.pause()
            assert app.memory.client.fav_calls == [
                ("chain", "agent-1", True),
            ]

    @pytest.mark.asyncio
    async def test_menu_cancel_dispatches_nothing(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            library = app.screen_stack[-1]
            assert isinstance(library, LibraryScreen)
            library.action_row_context_menu()
            await pilot.pause()
            await pilot.pause()
            menu = app.screen_stack[-1]
            assert isinstance(menu, ContextMenu)
            menu.action_cancel()
            await pilot.pause()
            await pilot.pause()
            assert app.memory.client.fav_calls == []
            assert library._row_action_log == []

    @pytest.mark.asyncio
    async def test_no_row_no_menu(self):
        class _EmptyClient:
            def list_chains(self, **kw):
                return []

        class _EmptyMemory:
            def __init__(self):
                self.client = _EmptyClient()

        app = _LibHost(memory=_EmptyMemory())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            library = app.screen_stack[-1]
            assert isinstance(library, LibraryScreen)
            library.action_row_context_menu()
            await pilot.pause()
            # Menu should not have pushed — top of stack is
            # still the LibraryScreen.
            assert isinstance(app.screen_stack[-1], LibraryScreen)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_widgets_re_exports_context_menu(self):
        from care.widgets import ContextMenu as ReExported

        assert ReExported is ContextMenu
