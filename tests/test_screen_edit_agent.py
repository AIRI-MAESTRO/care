"""Pilot tests for EditAgentScreen (TODO §1.1 P0.23).

Exercises:
* `extract_edit_draft` seeds the form on construction.
* Field edits route through the data-layer `set_*` helpers.
* `validate_edit_draft` issues render inline.
* Save fires `save_edit_as_new_version` on the worker.
* Promote fires `promote_to_stable`.
* The screen posts `Submitted` envelopes after each terminal
  action.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static, TextArea

from care.runtime.edit_draft import EditAgentDraft
from care.screens.edit_agent import EditAgentEvent, EditAgentScreen


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _saved_chain(*, entity_id: str = "agent-1") -> dict:
    return {
        "entity_id": entity_id,
        "version_id": "v1",
        "channel": "latest",
        "content": {
            "steps": [{"name": "fetch", "type": "llm"}],
        },
        "metadata": {
            "care": {
                "display_name": "Storm Watcher",
                "description": "Watches storms",
                "tags": ["weather"],
                "task_description": "Run forecast",
            },
        },
    }


@dataclass
class _SaveResult:
    entity_id: str = "agent-1"
    success: bool = True
    error: str | None = None
    fields_written: tuple = ()


@dataclass
class _PromoteResult:
    entity_id: str = "agent-1"
    from_channel: str = "latest"
    to_channel: str = "stable"
    success: bool = True
    error: str | None = None
    response: dict = None


class _StubClient:
    def __init__(self):
        self.promote_calls: list[dict] = []

    def promote(self, entity_id, *, from_channel, to_channel, entity_type):
        self.promote_calls.append(
            {
                "entity_id": entity_id,
                "from_channel": from_channel,
                "to_channel": to_channel,
                "entity_type": entity_type,
            }
        )
        return {"promoted": True}


class _StubMemory:
    def __init__(self):
        self.client = _StubClient()
        self.save_calls: list[dict] = []

    def save_chain(self, chain, **kw):
        self.save_calls.append({"chain": chain, **kw})
        return "agent-1"


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _EditHost(App):
    def __init__(self, *, chain=None, memory=None) -> None:
        super().__init__()
        self.chain = chain if chain is not None else _saved_chain()
        self.memory = memory if memory is not None else _StubMemory()
        self.submitted: list[EditAgentEvent] = []
        self.toasts: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(
            EditAgentScreen(self.chain, memory=self.memory),
        )

    def on_edit_agent_screen_submitted(
        self, event: EditAgentScreen.Submitted,
    ) -> None:
        self.submitted.append(event.payload)

    def push_toast(self, message, *, severity="info", ttl=None):  # type: ignore[override]
        self.toasts.append((message, severity))


def _screen(app: App) -> EditAgentScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, EditAgentScreen)
    return s


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


class TestSeed:
    @pytest.mark.asyncio
    async def test_form_pre_filled(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.draft.display_name == "Storm Watcher"
            assert screen.draft.task_description == "Run forecast"
            assert "weather" in screen.draft.tags
            name_input = screen.query_one("#edit-display-name", Input)
            assert name_input.value == "Storm Watcher"

    @pytest.mark.asyncio
    async def test_draft_not_dirty_on_seed(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.draft.is_dirty() is False


# ---------------------------------------------------------------------------
# Field edits
# ---------------------------------------------------------------------------


class TestFieldEdits:
    @pytest.mark.asyncio
    async def test_display_name_edit_makes_dirty(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one(
                "#edit-display-name", Input,
            ).value = "Renamed"
            await pilot.pause()
            assert screen.draft.display_name == "Renamed"
            assert screen.draft.is_dirty() is True

    @pytest.mark.asyncio
    async def test_description_edit(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one(
                "#edit-description", TextArea,
            ).load_text("New description")
            await pilot.pause()
            assert "New description" in screen.draft.description

    @pytest.mark.asyncio
    async def test_tags_input_splits_on_comma(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one("#edit-tags", Input).value = "a, b, c"
            await pilot.pause()
            assert set(screen.draft.tags) >= {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_save_label_flips_on_dirty(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            save = screen.query_one("#edit-btn-save", Button)
            assert str(save.label) == "Save"
            screen.query_one(
                "#edit-display-name", Input,
            ).value = "Renamed"
            await pilot.pause()
            await pilot.pause()
            assert "(" in str(save.label)


# ---------------------------------------------------------------------------
# Content tab (manual chain-content editing)
# ---------------------------------------------------------------------------


class TestContentTab:
    @pytest.mark.asyncio
    async def test_content_editor_seeds_with_chain_json(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            editor = screen.query_one("#edit-content-json", TextArea)
            assert editor.language == "json"
            # Line numbers help navigate larger chains.
            assert editor.show_line_numbers is True
            # The saved chain's steps round-trip into the editor.
            assert '"steps"' in editor.text
            assert '"fetch"' in editor.text
            # Seeding doesn't mark the draft dirty.
            assert screen.draft.is_dirty() is False

    @pytest.mark.asyncio
    async def test_editing_content_marks_dirty_and_updates_draft(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            editor = screen.query_one("#edit-content-json", TextArea)
            editor.load_text(
                '{"content": {"steps": [{"name": "renamed", "type": "llm"}]}}'
            )
            await pilot.pause()
            assert screen.draft.chain_content_dirty is True
            assert screen.draft.is_dirty() is True
            assert (
                screen.draft.chain_content["content"]["steps"][0]["name"]
                == "renamed"
            )

    @pytest.mark.asyncio
    async def test_invalid_json_blocks_save(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            editor = screen.query_one("#edit-content-json", TextArea)
            editor.load_text('{"steps": [')  # truncated → invalid
            await pilot.pause()
            assert screen.has_blocking_issues is True
            err = screen.query_one("#edit-content-error", Static)
            assert "Invalid JSON" in str(err.render())
            # Fixing the JSON clears the block.
            editor.load_text('{"steps": []}')
            await pilot.pause()
            assert screen._content_error == ""
            assert "Invalid JSON" not in str(err.render())

    @pytest.mark.asyncio
    async def test_reformat_only_is_not_dirty(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            editor = screen.query_one("#edit-content-json", TextArea)
            # Re-serialise the same object with different whitespace.
            import json as _json

            compact = _json.dumps(screen._original_content_obj)
            editor.load_text(compact)
            await pilot.pause()
            assert screen.draft.chain_content_dirty is False
            assert screen.draft.is_dirty() is False

    @pytest.mark.asyncio
    async def test_content_edit_save_passes_edited_chain(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            editor = screen.query_one("#edit-content-json", TextArea)
            editor.load_text(
                '{"content": {"steps": [{"name": "renamed", "type": "llm"}]}}'
            )
            await pilot.pause()
            # Structural edits require a change summary to save.
            assert screen.has_blocking_issues is True
            screen.query_one("#edit-summary", TextArea).load_text(
                "Renamed the fetch step",
            )
            await pilot.pause()
            assert screen.has_blocking_issues is False
            screen.query_one("#edit-btn-save", Button).press()
            for _ in range(6):
                await pilot.pause()
            assert app.memory.save_calls
            saved_chain = app.memory.save_calls[-1]["chain"]
            assert (
                saved_chain["content"]["steps"][0]["name"] == "renamed"
            )

    @pytest.mark.asyncio
    async def test_content_save_autofills_change_summary(self):
        """Best-UX: editing the JSON and clicking Save just works —
        the change-summary requirement is auto-satisfied rather than
        silently blocking the save."""
        from textual.widgets import TabbedContent

        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            editor = screen.query_one("#edit-content-json", TextArea)
            editor.load_text(
                '{"content": {"steps": [{"name": "x", "type": "llm"}]}}'
            )
            await pilot.pause()
            # User is on the Content tab and never touches the summary.
            screen.query_one("#edit-tabs", TabbedContent).active = (
                "edit-tab-content"
            )
            await pilot.pause()
            assert screen.draft.change_summary == ""
            screen.query_one("#edit-btn-save", Button).press()
            for _ in range(8):
                await pilot.pause()
            # Saved, with an auto-filled change summary.
            assert app.memory.save_calls
            assert screen.draft.change_summary != ""

    @pytest.mark.asyncio
    async def test_real_blocker_gives_feedback_not_silence(self):
        """A genuinely unfixable blocker (empty display name) must NOT
        fail silently — it surfaces a warning toast and switches to the
        offending field's tab instead of doing nothing."""
        from textual.widgets import Input, TabbedContent

        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            # Clear the (required) display name from the Metadata tab.
            screen.query_one("#edit-display-name", Input).value = ""
            await pilot.pause()
            # Move to the Content tab as the user would have been.
            screen.query_one("#edit-tabs", TabbedContent).active = (
                "edit-tab-content"
            )
            await pilot.pause()
            screen.query_one("#edit-btn-save", Button).press()
            for _ in range(4):
                await pilot.pause()
            # Nothing saved...
            assert app.memory.save_calls == []
            # ...but the user got a clear warning toast,
            assert any(
                sev == "warning" and "save" in msg.lower()
                for msg, sev in app.toasts
            )
            # ...and was sent back to the tab hosting the missing field.
            assert (
                screen.query_one("#edit-tabs", TabbedContent).active
                == "edit-tab-metadata"
            )

    @pytest.mark.asyncio
    async def test_save_rebaselines_to_clean_state(self):
        """After a successful save the form reports clean (no lingering
        '(N changes)') so the user sees the save took effect."""
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one("#edit-content-json", TextArea).load_text(
                '{"content": {"steps": [{"name": "y", "type": "llm"}]}}'
            )
            await pilot.pause()
            screen.query_one("#edit-summary", TextArea).load_text("note")
            await pilot.pause()
            assert screen.draft.is_dirty() is True
            screen.query_one("#edit-btn-save", Button).press()
            for _ in range(8):
                await pilot.pause()
            assert app.memory.save_calls
            # Re-baselined → clean.
            assert screen.draft.is_dirty() is False
            assert str(
                screen.query_one("#edit-btn-save", Button).label
            ) == "Save"


# ---------------------------------------------------------------------------
# Save action
# ---------------------------------------------------------------------------


class TestSave:
    @pytest.mark.asyncio
    async def test_save_button_fires_worker(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            # Make a change so save isn't a no-op.
            screen.query_one(
                "#edit-display-name", Input,
            ).value = "Renamed"
            await pilot.pause()
            screen.query_one("#edit-btn-save", Button).press()
            for _ in range(6):
                await pilot.pause()
            # Save submitted envelope.
            actions = [e.action for e in app.submitted]
            assert "save" in actions
            # Real save_chain stub was invoked.
            assert app.memory.save_calls != []

    @pytest.mark.asyncio
    async def test_save_without_facade_dismisses_anyway(self):
        class _NoMemHost(App):
            chain = _saved_chain()
            submitted = []

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(EditAgentScreen(self.chain, memory=None))

            def on_edit_agent_screen_submitted(self, event):
                self.submitted.append(event.payload)

        app = _NoMemHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one(
                "#edit-display-name", Input,
            ).value = "Renamed"
            await pilot.pause()
            screen.action_save_edit()
            for _ in range(6):
                await pilot.pause()
            assert any(
                e.action == "save" for e in app.submitted
            )


# ---------------------------------------------------------------------------
# Promote action
# ---------------------------------------------------------------------------


class TestPromote:
    @pytest.mark.asyncio
    async def test_promote_button_fires_worker(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.query_one("#edit-btn-promote", Button).press()
            for _ in range(6):
                await pilot.pause()
            actions = [e.action for e in app.submitted]
            assert "promote" in actions
            # Real client.promote stub was invoked.
            assert app.memory.client.promote_calls != []


# ---------------------------------------------------------------------------
# Back
# ---------------------------------------------------------------------------


class TestBack:
    @pytest.mark.asyncio
    async def test_escape_posts_back_and_pops(self):
        app = _EditHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            initial_depth = len(app.screen_stack)
            screen.action_back()
            for _ in range(3):
                await pilot.pause()
            assert any(
                e.action == "back" for e in app.submitted
            )
            assert len(app.screen_stack) < initial_depth


# ---------------------------------------------------------------------------
# Envelope dataclass
# ---------------------------------------------------------------------------


class TestEventDataclass:
    def test_envelope_default_fields(self):
        draft = EditAgentDraft(entity_id="x")
        e = EditAgentEvent(action="back", draft=draft)
        assert e.action == "back"
        assert e.save_result is None
        assert e.promote_result is None


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import EditAgentEvent as E
        from care.screens import EditAgentScreen as S

        assert E is EditAgentEvent
        assert S is EditAgentScreen
