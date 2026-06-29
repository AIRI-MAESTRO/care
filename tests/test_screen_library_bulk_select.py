"""Pilot tests for LibraryScreen bulk-select mode (TODO §1.1 P0.13).

Exercises:
* `Space` toggles row selection (state on `bulk_selection`).
* `F` runs `apply_favourite` over the selection when active.
* `Delete` opens ConfirmModal then runs `apply_delete`.
* `T` logs a tag-edit request per selected target (until
  P0.22 TagEditorModal lands).
* `Escape` clears the selection.
* Bulk mode falls back to per-row dispatch when selection is
  empty.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from care.screens.confirm import ConfirmModal
from care.screens.library import LibraryScreen


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _row_payload(entity_id: str, *, favourite: bool = False) -> dict:
    return {
        "entity_type": "chain",
        "entity_id": entity_id,
        "version_id": "v1",
        "channel": "latest",
        "etag": "e",
        "favourite": favourite,
        "run_count": 0,
        "last_run_at": None,
        "display_name": entity_id.title(),
        "description": "",
        "meta": {"tags": [], "name": entity_id},
        "content": {"steps": []},
        "evolution_meta": None,
    }


class _StubClient:
    def __init__(self, rows):
        self._rows = list(rows)
        self.fav_calls = []
        self.delete_calls = []

    def list_chains(self, **kw):
        return [dict(r) for r in self._rows]

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


class _StubMemory:
    def __init__(self, rows):
        self.client = _StubClient(rows)


class _LibHost(App):
    def __init__(self, rows=None):
        super().__init__()
        if rows is None:
            rows = [
                _row_payload("alpha"),
                _row_payload("beta"),
                _row_payload("gamma"),
            ]
        self.memory = _StubMemory(rows)
        self.toasts: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(LibraryScreen(restore_state=False))

    def push_toast(
        self, message, *, severity="info", ttl=None,
    ) -> None:  # type: ignore[override]
        self.toasts.append((message, severity))


def _library(app: App) -> LibraryScreen:
    screen = app.screen_stack[-1]
    assert isinstance(screen, LibraryScreen)
    return screen


# ---------------------------------------------------------------------------
# Space toggles selection
# ---------------------------------------------------------------------------


class TestToggleSelection:
    @pytest.mark.asyncio
    async def test_space_adds_focused_row(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            assert lib.bulk_selection.is_empty
            lib.action_row_toggle_select()
            assert len(lib.bulk_selection) == 1
            assert lib.bulk_selection.entity_ids == ("alpha",)
            assert lib.is_bulk_active is True

    @pytest.mark.asyncio
    async def test_space_again_removes_row(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            lib.action_row_toggle_select()
            lib.action_row_toggle_select()
            assert lib.bulk_selection.is_empty

    @pytest.mark.asyncio
    async def test_escape_clears_selection(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            lib.action_row_toggle_select()
            assert lib.is_bulk_active
            lib.action_clear_selection()
            assert lib.bulk_selection.is_empty


# ---------------------------------------------------------------------------
# Bulk favourite
# ---------------------------------------------------------------------------


class TestBulkFavourite:
    @pytest.mark.asyncio
    async def test_f_favourites_selection(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            # Manually populate the selection (cursor only
            # ever sits on row 0).
            from care.runtime.bulk_ops import BulkSelection, BulkTarget

            lib.bulk_selection = BulkSelection(
                targets=(
                    BulkTarget(entity_id="alpha", entity_type="chain"),
                    BulkTarget(entity_id="beta", entity_type="chain"),
                ),
            )
            lib.action_row_toggle_favourite()
            await pilot.pause()
            await pilot.pause()
            ids = {c[1] for c in app.memory.client.fav_calls}
            assert ids == {"alpha", "beta"}
            # Per spec: when none are favourited, promote all.
            for c in app.memory.client.fav_calls:
                assert c[2] is True
            assert lib.last_bulk_result is not None
            assert lib.last_bulk_result.succeeded == 2

    @pytest.mark.asyncio
    async def test_f_unfavourites_when_all_already_favourite(self):
        app = _LibHost(rows=[
            _row_payload("alpha", favourite=True),
            _row_payload("beta", favourite=True),
        ])
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            from care.runtime.bulk_ops import BulkSelection, BulkTarget

            lib.bulk_selection = BulkSelection(
                targets=(
                    BulkTarget(entity_id="alpha", entity_type="chain"),
                    BulkTarget(entity_id="beta", entity_type="chain"),
                ),
            )
            lib.action_row_toggle_favourite()
            await pilot.pause()
            await pilot.pause()
            for c in app.memory.client.fav_calls:
                assert c[2] is False


# ---------------------------------------------------------------------------
# Bulk delete
# ---------------------------------------------------------------------------


class TestBulkDelete:
    @pytest.mark.asyncio
    async def test_delete_pushes_confirm_modal(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            from care.runtime.bulk_ops import BulkSelection, BulkTarget

            lib.bulk_selection = BulkSelection(
                targets=(
                    BulkTarget(entity_id="alpha", entity_type="chain"),
                    BulkTarget(entity_id="beta", entity_type="chain"),
                ),
            )
            lib.action_row_delete()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], ConfirmModal)

    @pytest.mark.asyncio
    async def test_delete_confirm_runs_apply_delete(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            from care.runtime.bulk_ops import BulkSelection, BulkTarget

            lib.bulk_selection = BulkSelection(
                targets=(
                    BulkTarget(entity_id="alpha", entity_type="chain"),
                    BulkTarget(entity_id="beta", entity_type="chain"),
                ),
            )
            lib.action_row_delete()
            await pilot.pause()
            await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ConfirmModal)
            modal.dismiss(True)
            await pilot.pause()
            await pilot.pause()
            ids = {c[1] for c in app.memory.client.delete_calls}
            assert ids == {"alpha", "beta"}
            assert lib.bulk_selection.is_empty

    @pytest.mark.asyncio
    async def test_delete_cancel_keeps_selection(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            from care.runtime.bulk_ops import BulkSelection, BulkTarget

            lib.bulk_selection = BulkSelection(
                targets=(
                    BulkTarget(entity_id="alpha", entity_type="chain"),
                ),
            )
            lib.action_row_delete()
            await pilot.pause()
            await pilot.pause()
            modal = app.screen_stack[-1]
            modal.dismiss(False)
            await pilot.pause()
            await pilot.pause()
            assert app.memory.client.delete_calls == []
            assert lib.is_bulk_active


# ---------------------------------------------------------------------------
# Tag edit (P0.28 — modal pushes onto the screen stack)
# ---------------------------------------------------------------------------


class TestTagEdit:
    @pytest.mark.asyncio
    async def test_t_in_bulk_mode_opens_tag_editor_modal(self):
        from care.runtime.bulk_ops import BulkSelection, BulkTarget
        from care.screens.tag_editor import TagEditorModal

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            lib.bulk_selection = BulkSelection(
                targets=(
                    BulkTarget(
                        entity_id="alpha",
                        entity_type="chain",
                        current_tags=("weather",),
                    ),
                    BulkTarget(
                        entity_id="beta",
                        entity_type="chain",
                        current_tags=("urgent",),
                    ),
                ),
            )
            lib.action_row_tag_edit()
            for _ in range(4):
                await pilot.pause()
            assert isinstance(app.screen_stack[-1], TagEditorModal)
            modal = app.screen_stack[-1]
            assert modal.target_count == 2
            assert set(modal.initial_tags) == {"weather", "urgent"}

    @pytest.mark.asyncio
    async def test_t_with_no_selection_uses_focused_row(self):
        from care.screens.tag_editor import TagEditorModal

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            assert lib.is_bulk_active is False
            lib.action_row_tag_edit()
            for _ in range(4):
                await pilot.pause()
            assert isinstance(app.screen_stack[-1], TagEditorModal)
            modal = app.screen_stack[-1]
            assert modal.target_count == 1


# ---------------------------------------------------------------------------
# Fallback to per-row when no selection
# ---------------------------------------------------------------------------


class TestFallback:
    @pytest.mark.asyncio
    async def test_f_falls_through_when_no_selection(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            assert not lib.is_bulk_active
            lib.action_row_toggle_favourite()
            await pilot.pause()
            await pilot.pause()
            # Per-row dispatch fired against the focused row only.
            assert app.memory.client.fav_calls == [
                ("chain", "alpha", True),
            ]


# ---------------------------------------------------------------------------
# Explicit bulk bindings (§4 P1)
# ---------------------------------------------------------------------------


class TestExplicitBulkBindings:
    """§4 P1 — `Shift+T` and `Shift+Del` route to the bulk
    workers regardless of `is_bulk_active`, with friendly
    info toasts when the selection is empty (no single-row
    fallback)."""

    @pytest.mark.asyncio
    async def test_bulk_delete_no_selection_warns(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            assert lib.bulk_selection.is_empty
            lib.action_bulk_delete()
            await pilot.pause()
            assert any(
                "Bulk-delete needs a selection" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_bulk_delete_pushes_confirm_modal(self):
        from care.runtime.bulk_ops import (
            BulkSelection, BulkTarget,
        )

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            lib.bulk_selection = BulkSelection(targets=(
                BulkTarget(entity_id="alpha", entity_type="chain"),
                BulkTarget(entity_id="beta", entity_type="chain"),
            ))
            lib.action_bulk_delete()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], ConfirmModal)

    @pytest.mark.asyncio
    async def test_bulk_tag_edit_no_selection_warns(self):
        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            assert lib.bulk_selection.is_empty
            lib.action_bulk_tag_edit()
            await pilot.pause()
            assert any(
                "Bulk-tag needs a selection" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_bulk_tag_edit_opens_tag_editor(self):
        from care.runtime.bulk_ops import (
            BulkSelection, BulkTarget,
        )
        from care.screens.tag_editor import TagEditorModal

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            lib = _library(app)
            lib.bulk_selection = BulkSelection(targets=(
                BulkTarget(entity_id="alpha", entity_type="chain"),
            ))
            lib.action_bulk_tag_edit()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(
                app.screen_stack[-1], TagEditorModal,
            )

    def test_shift_T_and_shift_del_bindings_registered(self):
        # Drift guard: the new keys stay bound to the
        # bulk-only actions.
        bindings = {
            getattr(b, "action", None): b.key
            for b in LibraryScreen.BINDINGS
        }
        assert bindings.get("bulk_delete") == "shift+delete"
        assert bindings.get("bulk_tag_edit") == "T"
