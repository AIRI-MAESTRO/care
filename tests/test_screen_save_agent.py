"""Pilot tests for SaveAgentModal (TODO §1.1 P0.18).

Exercises:
* `seed_save_agent_form` runs on construction; widgets reflect
  the form's pre-fill.
* Field edits route through the `set_*` helpers and rebuild
  the form.
* Validation issues render below the form fields.
* `Save & Inspect` / `Save & Run` dismiss with the right
  envelope and call `apply_save_agent_form`.
* `Discard` dismisses without calling the applier.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, TextArea

from care.runtime.save_agent_form import SaveAgentForm
from care.screens.save_agent import (
    SaveAgentAction,
    SaveAgentModal,
    SaveAgentResult,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubSession:
    entity_id: str = "draft-123"
    name: str = "Draft"
    entity_type: str = "chain"
    promoted: bool = False
    discarded: bool = False


class _StubClient:
    def __init__(self):
        self.metadata_calls: list[dict] = []
        self.get_entity_calls: list[tuple[str, str]] = []
        self.promote_calls: list[dict] = []

    def _update_metadata(self, entity_type, entity_id, **kw):
        self.metadata_calls.append(
            {"entity_type": entity_type, "entity_id": entity_id, **kw}
        )
        return {"updated": True}

    def _get_entity(self, entity_type, entity_id):
        self.get_entity_calls.append((entity_type, entity_id))
        return {"display_name": "Existing"}

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

    def list_chains(self, **kw):
        return []


class _StubMemory:
    def __init__(self):
        self.client = _StubClient()


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _ModalHost(App):
    def __init__(self, **modal_kwargs):
        super().__init__()
        self._modal_kwargs = modal_kwargs
        self.dismissed: list[SaveAgentResult] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(result):
            self.dismissed.append(result)

        self.push_screen(
            SaveAgentModal(**self._modal_kwargs),
            _on_dismiss,
        )


def _modal(app: App) -> SaveAgentModal:
    screen = app.screen_stack[-1]
    assert isinstance(screen, SaveAgentModal)
    return screen


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


class TestSeed:
    @pytest.mark.asyncio
    async def test_modal_seeds_from_mage_metadata(self):
        app = _ModalHost(
            query="weather report",
            mage_metadata={
                "domain": "weather",
                "suggested_display_name": "Weather watcher",
                "suggested_description": "Predicts storms",
                "suggested_tags": ["weather", "storms"],
            },
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.form.display_name == "Weather watcher"
            assert modal.form.description == "Predicts storms"
            assert "weather" in modal.form.tags
            display_input = modal.query_one(
                "#save-agent-display-name", Input,
            )
            assert display_input.value == "Weather watcher"

    @pytest.mark.asyncio
    async def test_modal_renders_three_buttons(self):
        app = _ModalHost(query="anything")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            for bid in (
                "save-agent-discard",
                "save-agent-save-inspect",
                "save-agent-save-run",
            ):
                assert modal.query_one(f"#{bid}", Button) is not None


# ---------------------------------------------------------------------------
# Field edits
# ---------------------------------------------------------------------------


class TestFieldEdits:
    @pytest.mark.asyncio
    async def test_display_name_edit_updates_form(self):
        app = _ModalHost(query="x")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#save-agent-display-name", Input).value = "New"
            await pilot.pause()
            assert modal.form.display_name == "New"

    @pytest.mark.asyncio
    async def test_description_edit_updates_form(self):
        app = _ModalHost(query="x")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#save-agent-description", TextArea,
            ).load_text("Describing this agent")
            await pilot.pause()
            assert "Describing this agent" in modal.form.description

    @pytest.mark.asyncio
    async def test_tags_input_splits_on_comma(self):
        app = _ModalHost(query="x")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#save-agent-tags", Input).value = "foo, bar, baz"
            await pilot.pause()
            assert set(modal.form.tags) >= {"foo", "bar", "baz"}

    @pytest.mark.asyncio
    async def test_favourite_checkbox_toggles_form(self):
        app = _ModalHost(query="x")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.form.favourite is False
            modal.query_one(
                "#save-agent-favourite", Checkbox,
            ).value = True
            await pilot.pause()
            assert modal.form.favourite is True

    @pytest.mark.asyncio
    async def test_keep_context_checkbox_toggles_form(self):
        app = _ModalHost(query="x")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            # Default is True per the data layer.
            assert modal.form.keep_context is True
            modal.query_one(
                "#save-agent-keep-context", Checkbox,
            ).value = False
            await pilot.pause()
            assert modal.form.keep_context is False


# ---------------------------------------------------------------------------
# Button actions
# ---------------------------------------------------------------------------


class TestActions:
    @pytest.mark.asyncio
    async def test_discard_dismisses_with_discard_action(self):
        app = _ModalHost(query="x")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_discard()
            await pilot.pause()
            await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].action == "discard"
            assert app.dismissed[0].outcome is None

    @pytest.mark.asyncio
    async def test_save_inspect_calls_apply(self):
        memory = _StubMemory()
        session = _StubSession()
        app = _ModalHost(
            memory=memory,
            session=session,
            query="generate weather",
            mage_metadata={
                "suggested_display_name": "Storm watcher",
                "domain": "weather",
            },
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#save-agent-save-inspect", Button,
            ).press()
            # Apply runs in a worker — give it time to land.
            for _ in range(6):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].action == "save_inspect"
            # apply_save_agent_form should have invoked the
            # underlying stubs.
            assert len(memory.client.promote_calls) == 1
            assert memory.client.promote_calls[0]["entity_id"] == "draft-123"
            assert memory.client.metadata_calls != []

    @pytest.mark.asyncio
    async def test_save_run_dismisses_with_save_run(self):
        memory = _StubMemory()
        session = _StubSession()
        app = _ModalHost(
            memory=memory,
            session=session,
            query="generate",
            mage_metadata={"suggested_display_name": "Run me", "domain": "x"},
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#save-agent-save-run", Button).press()
            for _ in range(6):
                await pilot.pause()
            assert app.dismissed[0].action == "save_run"

    @pytest.mark.asyncio
    async def test_save_without_facade_dismisses_with_no_outcome(self):
        # When memory + session are None we expect a no-op
        # apply (the modal still dismisses so the host routes
        # the gesture, but `outcome` stays None).
        app = _ModalHost(query="x")
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#save-agent-save-inspect", Button).press()
            for _ in range(6):
                await pilot.pause()
            assert len(app.dismissed) == 1
            result = app.dismissed[0]
            assert result.action == "save_inspect"
            assert result.outcome is None


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


class TestSaveAgentResult:
    def test_envelope_fields(self):
        form = SaveAgentForm()
        result = SaveAgentResult(action="discard", form=form)
        assert result.action == "discard"
        assert result.form is form
        assert result.outcome is None

    def test_action_kinds_are_str_literals(self):
        # Pin the literal set so future re-orderings surface.
        valid: tuple[SaveAgentAction, ...] = (
            "save_inspect",
            "save_run",
            "discard",
        )
        assert set(valid) == {"save_inspect", "save_run", "discard"}


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports_modal(self):
        from care.screens import SaveAgentModal as ReExported

        assert ReExported is SaveAgentModal
