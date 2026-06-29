"""Pilot tests for CommandPaletteModal (TODO §1.1 P0.25).

Exercises:
* `on_mount` populates the OptionList (commands first when
  empty index, then fetched entries when memory is wired).
* Input keystrokes re-run `search_palette` and update results.
* Selecting an entry dismisses with `PaletteSelection`.
* `Enter` on the input picks the top result.
* `Escape` dismisses with `entry=None`.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList

from care.runtime.command_palette import (
    PaletteEntry,
    PaletteIndex,
    default_commands,
)
from care.screens.command_palette import (
    CommandPaletteModal,
    PaletteSelection,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubClient:
    def list_chains(self, **kw):
        return []

    def list_agent_skills(self, **kw):
        return []


class _StubMemory:
    def __init__(self):
        self.client = _StubClient()


def _index_with_chain(name: str = "Storm Watcher") -> PaletteIndex:
    entries = []
    # Add commands.
    for cmd in default_commands():
        entries.append(
            PaletteEntry(
                entry_id=f"command:{cmd.action_id}",
                kind="command",
                label=cmd.label,
                description=cmd.description,
                command_action=cmd.action_id,
            )
        )
    # Add one chain entry.
    entries.append(
        PaletteEntry(
            entry_id="agent-1",
            kind="chain",
            label=name,
            description="weather watcher",
        )
    )
    return PaletteIndex(entries=tuple(entries))


class _Host(App):
    def __init__(self, *, index=None, memory=None) -> None:
        super().__init__()
        self._initial_index = index
        self._initial_memory = memory
        self.dismissed: list[PaletteSelection] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(selection):
            self.dismissed.append(selection)

        self.push_screen(
            CommandPaletteModal(
                index=self._initial_index,
                memory=self._initial_memory,
            ),
            _on_dismiss,
        )


def _modal(app: App) -> CommandPaletteModal:
    s = app.screen_stack[-1]
    assert isinstance(s, CommandPaletteModal)
    return s


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_input_and_list_mount(self):
        app = _Host(index=_index_with_chain())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.query_one("#palette-input", Input) is not None
            ol = modal.query_one("#palette-list", OptionList)
            assert ol is not None
            # Empty query → commands first.
            assert ol.option_count >= len(default_commands())


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.mark.asyncio
    async def test_typing_narrows_results(self):
        app = _Host(index=_index_with_chain("Storm Watcher"))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#palette-input", Input).value = "storm"
            await pilot.pause()
            await pilot.pause()
            # At least one match for "storm".
            assert modal.results != ()
            labels = [r.label for r in modal.results]
            assert any("Storm" in label for label in labels)

    @pytest.mark.asyncio
    async def test_empty_query_returns_commands_first(self):
        app = _Host(index=_index_with_chain())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.results != ()
            assert modal.results[0].is_command is True


# ---------------------------------------------------------------------------
# Selection / dismiss
# ---------------------------------------------------------------------------


class TestSelection:
    @pytest.mark.asyncio
    async def test_enter_picks_top_result(self):
        app = _Host(index=_index_with_chain())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            input_widget = modal.query_one("#palette-input", Input)
            input_widget.post_message(
                Input.Submitted(input_widget, value=""),
            )
            await pilot.pause()
            await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].entry is not None

    @pytest.mark.asyncio
    async def test_escape_dismisses_with_none(self):
        app = _Host(index=_index_with_chain())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_cancel()
            await pilot.pause()
            await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].entry is None

    @pytest.mark.asyncio
    async def test_clicking_option_dispatches(self):
        app = _Host(index=_index_with_chain())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            ol = modal.query_one("#palette-list", OptionList)
            # Simulate option selected for the first option.
            ol.post_message(
                OptionList.OptionSelected(ol, ol.get_option_at_index(0), 0),
            )
            await pilot.pause()
            await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].entry is not None
            # First entry in results is the dispatched one.
            assert (
                app.dismissed[0].entry.entry_id
                == modal.results[0].entry_id
            )


# ---------------------------------------------------------------------------
# Memory aggregator
# ---------------------------------------------------------------------------


class TestMemoryAggregator:
    @pytest.mark.asyncio
    async def test_memory_load_runs_aggregator(self):
        memory = _StubMemory()
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            assert modal._loaded is True
            # No chains in the stub → only commands in the
            # index.
            assert len(modal.index) >= len(default_commands())


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import CommandPaletteModal as M
        from care.screens import PaletteSelection as S

        assert M is CommandPaletteModal
        assert S is PaletteSelection
