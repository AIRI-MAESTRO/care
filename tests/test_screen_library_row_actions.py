"""Pilot tests for LibraryScreen per-row keyboard actions (TODO §1.1 P0.11).

Exercises:
* `current_row` resolution against the focused DataTable row.
* `BINDINGS` keys → action dispatch (`F` toggles favourite,
  `D` duplicates, `Delete` pushes ConfirmModal then deletes,
  `R` / `E` / `V` / `L` / `Enter` log a navigation request).
* Status gating — `evolve` / `show_lineage` no-op on `draft`
  rows because `actions_for_row` drops them.
* Destructive confirm flow — cancelling the modal aborts the
  delete; confirming runs it.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from care.screens.library import LibraryScreen


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _runnable_row(entity_id: str = "agent-1", **overrides) -> dict:
    base = {
        "entity_type": "chain",
        "entity_id": entity_id,
        "version_id": "v1",
        "channel": "latest",
        "etag": "e",
        "favourite": False,
        "run_count": 3,
        "last_run_at": None,
        "display_name": "Storm Watcher",
        "description": "agent that watches storms",
        "meta": {"tags": ["weather"], "name": "storm-watcher"},
        "content": {"steps": [{"type": "llm"}]},
        "evolution_meta": None,
    }
    base.update(overrides)
    return base


def _draft_row(entity_id: str = "draft-1") -> dict:
    return _runnable_row(
        entity_id=entity_id,
        channel="draft",
        meta={"tags": ["draft"], "name": "draft-agent"},
        display_name="Draft Agent",
    )


class _StubClient:
    def __init__(self, *, rows: list[dict] | None = None):
        self._rows = rows if rows is not None else [_runnable_row()]
        self.fav_calls: list[tuple[str, str, bool]] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.get_calls: list[tuple[str, str]] = []
        self.save_calls: list[dict] = []

    def list_chains(self, **kw):
        return list(self._rows)

    def _mark_favourite(self, entity_type, entity_id, *, value):
        self.fav_calls.append((entity_type, entity_id, value))
        for r in self._rows:
            if r["entity_id"] == entity_id:
                r["favourite"] = value
        return {"favourite": value}

    def _delete_entity(self, entity_type, entity_id):
        self.delete_calls.append((entity_type, entity_id))
        self._rows = [r for r in self._rows if r["entity_id"] != entity_id]
        return {"deleted": True}

    def get_chain_dict(self, entity_id, channel):
        self.get_calls.append((entity_id, channel))
        for r in self._rows:
            if r["entity_id"] == entity_id:
                return r["content"]
        return None


class _StubMemory:
    def __init__(self, *, rows: list[dict] | None = None):
        self.client = _StubClient(rows=rows)
        self.save_chain_calls: list[dict] = []

    def save_chain(self, chain, *, name=None, tags=None, entity_id=None, channel="latest"):
        rec = {
            "chain": chain,
            "name": name,
            "tags": tags,
            "entity_id": entity_id,
            "channel": channel,
        }
        self.save_chain_calls.append(rec)
        return "new-entity-id"


class _LibHost(App):
    def __init__(self, *, memory: _StubMemory | None = None, rows=None):
        super().__init__()
        if memory is None:
            memory = _StubMemory(rows=rows)
        self.memory = memory

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(LibraryScreen(restore_state=False))


# ---------------------------------------------------------------------------
# current_row
# ---------------------------------------------------------------------------


class TestCurrentRow:
    @pytest.mark.asyncio
    async def test_current_row_returns_first_row(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            row = app.screen.current_row
            assert row is not None
            assert row.entity_id == "agent-1"

    @pytest.mark.asyncio
    async def test_current_row_none_when_empty(self):
        app = _LibHost(memory=_StubMemory(rows=[]))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.screen.current_row is None


# ---------------------------------------------------------------------------
# Favourite + delete + duplicate (mutators)
# ---------------------------------------------------------------------------


class TestToggleFavourite:
    @pytest.mark.asyncio
    async def test_action_toggles_favourite(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.screen.action_row_toggle_favourite()
            await pilot.pause()
            await pilot.pause()
            assert app.memory.client.fav_calls == [("chain", "agent-1", True)]
            assert app.screen.last_row_outcome is not None
            assert app.screen.last_row_outcome.success is True


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_pushes_confirm_modal(self):
        from care.screens.confirm import ConfirmModal

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            library = app.screen_stack[-1]
            assert isinstance(library, LibraryScreen)
            library.action_row_delete()
            # Yield to let the worker push the modal.
            await pilot.pause()
            await pilot.pause()
            # ConfirmModal should be on top of the screen stack.
            assert isinstance(app.screen_stack[-1], ConfirmModal)

    @pytest.mark.asyncio
    async def test_delete_cancel_does_nothing(self):
        from care.screens.confirm import ConfirmModal

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            library = app.screen_stack[-1]
            assert isinstance(library, LibraryScreen)
            library.action_row_delete()
            await pilot.pause()
            await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ConfirmModal)
            modal.dismiss(False)
            await pilot.pause()
            await pilot.pause()
            assert app.memory.client.delete_calls == []

    @pytest.mark.asyncio
    async def test_delete_confirm_runs_delete(self):
        from care.screens.confirm import ConfirmModal

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            library = app.screen_stack[-1]
            assert isinstance(library, LibraryScreen)
            library.action_row_delete()
            await pilot.pause()
            await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ConfirmModal)
            modal.dismiss(True)
            await pilot.pause()
            await pilot.pause()
            assert app.memory.client.delete_calls == [("chain", "agent-1")]


class TestDuplicate:
    @pytest.mark.asyncio
    async def test_duplicate_chain_writes_copy(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.screen.action_row_duplicate()
            await pilot.pause()
            await pilot.pause()
            assert app.memory.save_chain_calls != []
            rec = app.memory.save_chain_calls[0]
            assert rec["entity_id"] is None
            assert rec["name"] == "Storm Watcher (copy)"
            assert app.screen.last_row_outcome.success is True


# ---------------------------------------------------------------------------
# Navigation-only actions (run/open/edit/evolve/show_lineage)
# ---------------------------------------------------------------------------


class TestNavigationActions:
    @pytest.mark.asyncio
    async def test_action_run_logs_dispatch(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.screen.action_row_run()
            assert ("run", "agent-1") in app.screen._row_action_log

    @pytest.mark.asyncio
    async def test_action_edit_logs_dispatch(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.screen.action_row_edit()
            assert ("edit", "agent-1") in app.screen._row_action_log

    @pytest.mark.asyncio
    async def test_action_open_logs_dispatch(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # `action_row_open` pushes InspectionScreen so
            # `app.screen` may now be that screen — find the
            # LibraryScreen on the stack to read its log.
            library = next(
                s for s in app.screen_stack
                if isinstance(s, LibraryScreen)
            )
            library.action_row_open()
            assert ("open", "agent-1") in library._row_action_log

    @pytest.mark.asyncio
    async def test_action_open_pushes_inspection_screen(self):
        # The else-branch of `_dispatch_row_action` now pushes
        # the matching destination screen in addition to
        # recording the action log entry.
        from care.screens.inspection import InspectionScreen

        app = _LibHost()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.screen.action_row_open()
            for _ in range(6):
                await pilot.pause()
            assert any(
                isinstance(s, InspectionScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_row_selected_event_pushes_inspection_screen(self):
        """TODO §4 P0 — Enter / click on a row triggers
        `on_data_table_row_selected`, which must push the
        InspectionScreen via the same dispatch chain as
        `action_row_open`. Locks the gesture-→-screen wiring
        the chat-banner already advertises (`/library` → pick
        a row → inspect).
        """
        from textual.coordinate import Coordinate
        from textual.widgets import DataTable

        from care.screens.inspection import InspectionScreen

        app = _LibHost()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            library = next(
                s for s in app.screen_stack
                if isinstance(s, LibraryScreen)
            )
            table = library.query_one("#library-table", DataTable)
            # Move the cursor to the seeded row + fire the
            # `RowSelected` event the way Textual would on
            # Enter. We bypass keyboard simulation so the test
            # asserts the message handler itself, not the
            # binding plumbing.
            assert table.row_count >= 1
            table.cursor_coordinate = Coordinate(0, 0)
            await pilot.pause()
            row_key = table.coordinate_to_cell_key(
                Coordinate(0, 0),
            ).row_key
            library.post_message(
                DataTable.RowSelected(
                    table, cursor_row=0, row_key=row_key,
                ),
            )
            for _ in range(6):
                await pilot.pause()
            assert any(
                isinstance(s, InspectionScreen)
                for s in app.screen_stack
            ), "row_selected should push InspectionScreen"
            # The action log records the gesture symmetrically
            # with the keyboard-driven path.
            assert ("open", "agent-1") in library._row_action_log

    @pytest.mark.asyncio
    async def test_action_show_lineage_pushes_lineage_modal(self):
        from care.screens.lineage import LineageModal

        app = _LibHost()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.screen.action_row_show_lineage()
            for _ in range(6):
                await pilot.pause()
            assert any(
                isinstance(s, LineageModal)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_action_run_delegates_to_push_run_for(self):
        # `run` now drives the in-TUI run pipeline via the App helper
        # (load chain → RunContextModal → ExecutionScreen), the same
        # entry the InspectionScreen `run` action uses.
        app = _LibHost()
        seen: list[str] = []
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            library = next(
                s for s in app.screen_stack
                if isinstance(s, LibraryScreen)
            )
            app._push_run_for = lambda eid: seen.append(eid)  # type: ignore[attr-defined]
            library.action_row_run()
            for _ in range(2):
                await pilot.pause()
            assert seen == ["agent-1"]
            assert ("run", "agent-1") in library._row_action_log


# ---------------------------------------------------------------------------
# Status gating
# ---------------------------------------------------------------------------


class TestStatusGating:
    @pytest.mark.asyncio
    async def test_evolve_noop_on_draft(self):
        app = _LibHost(memory=_StubMemory(rows=[_draft_row()]))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.screen.current_row is not None
            assert app.screen.current_row.status == "draft"
            app.screen.action_row_evolve()
            # Draft rows can't be evolved → log stays empty for
            # this kind.
            assert all(
                k != "evolve" for k, _ in app.screen._row_action_log
            )

    @pytest.mark.asyncio
    async def test_show_lineage_noop_on_draft(self):
        app = _LibHost(memory=_StubMemory(rows=[_draft_row()]))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.screen.action_row_show_lineage()
            assert all(
                k != "show_lineage" for k, _ in app.screen._row_action_log
            )


# ---------------------------------------------------------------------------
# Empty view: no row → no dispatch
# ---------------------------------------------------------------------------


class TestEmptyView:
    @pytest.mark.asyncio
    async def test_no_row_no_op(self):
        app = _LibHost(memory=_StubMemory(rows=[]))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.screen.action_row_toggle_favourite()
            app.screen.action_row_delete()
            app.screen.action_row_duplicate()
            await pilot.pause()
            assert app.memory.client.fav_calls == []
            assert app.memory.client.delete_calls == []
            assert app.memory.save_chain_calls == []
            assert app.screen._row_action_log == []
