"""Pilot tests for TagEditorModal (TODO §1.1 P0.28).

Exercises:
* Seed renders current tags + title text.
* Typing into add / remove Inputs renders a `merge_tags` preview.
* Apply dismisses with `submitted=True` carrying the cleaned
  lists.
* Apply with empty lists falls through to a cancel-style
  dismiss.
* Cancel / Escape dismiss with `submitted=False`.
* LibraryScreen `T` (bulk mode) opens the modal + fires
  `apply_tag_edits` on submit.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input

from care.screens.library import LibraryScreen
from care.screens.tag_editor import TagEditorModal, TagEditorResult


# ---------------------------------------------------------------------------
# Modal-only host
# ---------------------------------------------------------------------------


class _ModalHost(App):
    def __init__(self, **modal_kwargs) -> None:
        super().__init__()
        self._kwargs = modal_kwargs
        self.dismissed: list[TagEditorResult] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(result):
            self.dismissed.append(result)

        self.push_screen(TagEditorModal(**self._kwargs), _on_dismiss)


def _modal(app: App) -> TagEditorModal:
    s = app.screen_stack[-1]
    assert isinstance(s, TagEditorModal)
    return s


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


class TestSeed:
    @pytest.mark.asyncio
    async def test_initial_tags_render(self):
        app = _ModalHost(
            initial_tags=("weather", "urgent"),
            target_count=3,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.initial_tags == ("weather", "urgent")
            assert modal.target_count == 3

    @pytest.mark.asyncio
    async def test_empty_initial_tags_show_placeholder(self):
        app = _ModalHost(initial_tags=())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal._current_text() == "(no tags currently applied)"


# ---------------------------------------------------------------------------
# Inputs + preview
# ---------------------------------------------------------------------------


class TestInputs:
    @pytest.mark.asyncio
    async def test_add_tags_parsed(self):
        app = _ModalHost(initial_tags=("weather",))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#tag-editor-add", Input).value = "foo, bar"
            await pilot.pause()
            assert modal.add_tags == ("foo", "bar")

    @pytest.mark.asyncio
    async def test_remove_tags_parsed(self):
        app = _ModalHost(initial_tags=("weather",))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#tag-editor-remove", Input).value = "weather"
            await pilot.pause()
            assert modal.remove_tags == ("weather",)

    @pytest.mark.asyncio
    async def test_preview_reflects_merge(self):
        app = _ModalHost(initial_tags=("weather",))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#tag-editor-add", Input).value = "urgent"
            modal.query_one("#tag-editor-remove", Input).value = "weather"
            await pilot.pause()
            preview = modal._preview_tags()
            assert preview == ("urgent",)


# ---------------------------------------------------------------------------
# Dismiss
# ---------------------------------------------------------------------------


class TestDismiss:
    @pytest.mark.asyncio
    async def test_apply_dismisses_with_lists(self):
        app = _ModalHost(initial_tags=("weather",))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#tag-editor-add", Input).value = "foo"
            modal.query_one("#tag-editor-remove", Input).value = "weather"
            await pilot.pause()
            modal.query_one("#tag-editor-apply", Button).press()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].submitted is True
            assert app.dismissed[0].add_tags == ("foo",)
            assert app.dismissed[0].remove_tags == ("weather",)

    @pytest.mark.asyncio
    async def test_apply_with_empty_dismisses_as_cancel(self):
        app = _ModalHost(initial_tags=("weather",))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_apply()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].submitted is False

    @pytest.mark.asyncio
    async def test_escape_dismisses_as_cancel(self):
        app = _ModalHost(initial_tags=("weather",))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_cancel()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].submitted is False


class TestEditableTitle:
    """§3 P3 — when `initial_title` is non-empty, the modal
    grows a Name Input above the tag rows, and the dismiss
    envelope carries the (possibly user-edited) title."""

    @pytest.mark.asyncio
    async def test_name_field_hidden_without_initial_title(
        self,
    ):
        from textual.css.query import NoMatches

        app = _ModalHost(initial_tags=("weather",))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.initial_title == ""
            with pytest.raises(NoMatches):
                modal.query_one("#tag-editor-name", Input)

    @pytest.mark.asyncio
    async def test_name_field_renders_with_seed(self):
        app = _ModalHost(
            initial_tags=(),
            initial_title="Weather Forecaster",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.initial_title == "Weather Forecaster"
            name_input = modal.query_one("#tag-editor-name", Input)
            assert name_input.value == "Weather Forecaster"

    @pytest.mark.asyncio
    async def test_apply_with_only_title_change_dismisses(
        self,
    ):
        """Editing only the title (no tag changes) MUST count
        as a meaningful apply — the save flow needs the user's
        accepted name."""
        app = _ModalHost(
            initial_tags=(),
            initial_title="Suggested Title",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#tag-editor-name", Input,
            ).value = "User Edited"
            await pilot.pause()
            modal.action_apply()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            result = app.dismissed[0]
            assert result.submitted is True
            assert result.title == "User Edited"
            assert result.add_tags == ()
            assert result.remove_tags == ()

    @pytest.mark.asyncio
    async def test_apply_accepts_unchanged_suggestion(self):
        """User pressed Apply without touching the field →
        result.title carries the original suggestion."""
        app = _ModalHost(
            initial_tags=(),
            initial_title="Auto Generated Name",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_apply()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].title == "Auto Generated Name"

    @pytest.mark.asyncio
    async def test_apply_with_cleared_title_returns_empty(
        self,
    ):
        """User cleared the field → result.title is empty
        (caller can decide whether to fall back to the
        suggestion or persist nameless)."""
        app = _ModalHost(
            initial_tags=(),
            initial_title="Suggested",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#tag-editor-name", Input,
            ).value = ""
            await pilot.pause()
            modal.action_apply()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].title == ""

    def test_result_dataclass_title_default_empty(self):
        from care.screens.tag_editor import TagEditorResult

        r = TagEditorResult()
        assert r.title == ""

    def test_initial_title_strips_whitespace(self):
        from care.screens.tag_editor import TagEditorModal

        modal = TagEditorModal(initial_title="  padded  ")
        assert modal.initial_title == "padded"


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestResultDataclass:
    def test_default_envelope(self):
        result = TagEditorResult()
        assert result.submitted is False
        assert result.add_tags == ()
        assert result.remove_tags == ()


# ---------------------------------------------------------------------------
# LibraryScreen integration
# ---------------------------------------------------------------------------


def _row_payload(entity_id: str, *, tags=()) -> dict:
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
        self.metadata_calls: list[dict] = []
        self.get_entity_calls: list[tuple[str, str]] = []

    def list_chains(self, **kw):
        return [dict(r) for r in self.rows]

    def _update_metadata(self, entity_type, entity_id, **kw):
        self.metadata_calls.append(
            {"entity_type": entity_type, "entity_id": entity_id, **kw}
        )
        return {"updated": True}

    def _get_entity(self, entity_type, entity_id):
        self.get_entity_calls.append((entity_type, entity_id))
        return {"meta": {"tags": []}}


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


def _library(app: App) -> LibraryScreen:
    s = app.screen_stack[-1]
    while not isinstance(s, LibraryScreen):
        # Walk up the stack in case a modal pushed.
        idx = app.screen_stack.index(s) - 1
        if idx < 0:
            raise AssertionError("LibraryScreen not in stack")
        s = app.screen_stack[idx]
    return s


class TestLibraryIntegration:
    @pytest.mark.asyncio
    async def test_t_in_bulk_mode_opens_modal_and_submits(self):
        from care.runtime.bulk_ops import BulkSelection, BulkTarget

        app = _LibHost([
            _row_payload("alpha", tags=["x"]),
            _row_payload("beta", tags=["x", "y"]),
        ])
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            lib = _library(app)
            lib.bulk_selection = BulkSelection(
                targets=(
                    BulkTarget(
                        entity_id="alpha",
                        entity_type="chain",
                        current_tags=("x",),
                    ),
                    BulkTarget(
                        entity_id="beta",
                        entity_type="chain",
                        current_tags=("x", "y"),
                    ),
                ),
            )
            lib.action_row_tag_edit()
            for _ in range(4):
                await pilot.pause()
            # Modal is now on top.
            assert isinstance(
                app.screen_stack[-1], TagEditorModal,
            )
            modal = app.screen_stack[-1]
            assert set(modal.initial_tags) == {"x", "y"}
            assert modal.target_count == 2
            # Simulate user adding "urgent".
            modal.query_one("#tag-editor-add", Input).value = "urgent"
            await pilot.pause()
            modal.action_apply()
            # Worker fires apply_tag_edits → metadata calls.
            for _ in range(8):
                await pilot.pause()
            ids = {c["entity_id"] for c in app.memory.client.metadata_calls}
            assert ids == {"alpha", "beta"}
            for call in app.memory.client.metadata_calls:
                assert "urgent" in (call.get("tags") or [])

    @pytest.mark.asyncio
    async def test_t_with_no_selection_uses_focused_row(self):
        app = _LibHost([_row_payload("alpha", tags=["weather"])])
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            lib = _library(app)
            assert lib.is_bulk_active is False
            lib.action_row_tag_edit()
            for _ in range(4):
                await pilot.pause()
            assert isinstance(app.screen_stack[-1], TagEditorModal)
            modal = app.screen_stack[-1]
            assert modal.initial_tags == ("weather",)
            assert modal.target_count == 1


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import TagEditorModal as M
        from care.screens import TagEditorResult as R

        assert M is TagEditorModal
        assert R is TagEditorResult
