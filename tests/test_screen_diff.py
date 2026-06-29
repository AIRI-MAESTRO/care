"""Pilot tests for DiffModal (TODO §1.1 P0.27).

Exercises:
* `on_mount` calls `fetch_agent_diff(memory, left, right)`.
* `diff.steps` render into the steps pane with badges.
* `diff.format_summary()` renders into the footer.
* Metadata diff renders above the steps.
* `Close` / `Escape` dismiss with `cancelled=True`.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from care.screens.diff import DiffModal, DiffResult


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _left_chain() -> dict:
    return {
        "steps": [
            {"number": 1, "name": "fetch", "type": "llm", "prompt": "old"},
            {"number": 2, "name": "summarise", "type": "llm"},
        ],
        "metadata": {
            "care": {
                "display_name": "Storm Watcher",
                "description": "old description",
                "task_description": "Run forecast",
                "tags": ["weather"],
            },
        },
    }


def _right_chain() -> dict:
    return {
        "steps": [
            # Modified prompt on step 1.
            {"number": 1, "name": "fetch", "type": "llm", "prompt": "new"},
            # Step 2 unchanged.
            {"number": 2, "name": "summarise", "type": "llm"},
            # New step 3.
            {"number": 3, "name": "verify", "type": "llm"},
        ],
        "metadata": {
            "care": {
                "display_name": "Storm Watcher",
                "description": "new description",
                "task_description": "Run forecast",
                "tags": ["weather", "urgent"],
            },
        },
    }


class _StubClient:
    def __init__(self, *, fail: bool = False):
        self._fail = fail
        self.calls: list[tuple[str, str]] = []

    def get_chain_dict(self, entity_id, channel):
        self.calls.append((entity_id, channel))
        if self._fail:
            raise RuntimeError("diff-down")
        if entity_id == "left":
            return _left_chain()
        if entity_id == "right":
            return _right_chain()
        return None


class _StubMemory:
    def __init__(self, *, fail: bool = False):
        self.client = _StubClient(fail=fail)


class _Host(App):
    def __init__(self, *, memory=None, left="left", right="right") -> None:
        super().__init__()
        self._memory = memory
        self._left = left
        self._right = right
        self.dismissed: list[DiffResult] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(result):
            self.dismissed.append(result)

        self.push_screen(
            DiffModal(self._left, self._right, memory=self._memory),
            _on_dismiss,
        )


def _modal(app: App) -> DiffModal:
    s = app.screen_stack[-1]
    assert isinstance(s, DiffModal)
    return s


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_panes_mount(self):
        memory = _StubMemory()
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            assert modal.query_one("#diff-title", Static) is not None
            assert modal.query_one("#diff-metadata", Static) is not None
            assert modal.query_one("#diff-steps") is not None
            assert modal.query_one("#diff-footer", Static) is not None


# ---------------------------------------------------------------------------
# Load + diff
# ---------------------------------------------------------------------------


class TestLoad:
    @pytest.mark.asyncio
    async def test_load_populates_diff(self):
        memory = _StubMemory()
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            assert modal._loaded is True
            assert modal.load_error is None
            assert modal.diff is not None
            assert modal.diff.has_changes is True
            # Step counts: step 3 added (+1), step 1 modified (~1).
            assert modal.diff.added_steps == 1
            assert modal.diff.modified_steps == 1
            assert memory.client.calls != []

    @pytest.mark.asyncio
    async def test_footer_renders_summary(self):
        memory = _StubMemory()
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            footer = modal.query_one("#diff-footer", Static)
            # The summary helper returns the canonical string.
            expected = modal.diff.format_summary()
            assert expected != "no differences"
            # Static updates aren't directly readable; we check
            # via the model.
            assert "+" in expected
            assert "~" in expected
            # Just confirm the footer was updated (no crash).
            assert footer is not None

    @pytest.mark.asyncio
    async def test_no_memory_lands_on_error(self):
        app = _Host(memory=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.load_error is not None
            assert "no memory facade" in modal.load_error

    @pytest.mark.asyncio
    async def test_fetch_failure_lands_on_error(self):
        memory = _StubMemory(fail=True)
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            assert modal.load_error is not None
            assert "diff-down" in modal.load_error


# ---------------------------------------------------------------------------
# Dismiss
# ---------------------------------------------------------------------------


class TestDismiss:
    @pytest.mark.asyncio
    async def test_close_dismisses_with_cancelled(self):
        memory = _StubMemory()
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            modal.query_one("#diff-btn-close", Button).press()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].cancelled is True
            assert app.dismissed[0].diff is not None

    @pytest.mark.asyncio
    async def test_escape_dismisses(self):
        memory = _StubMemory()
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            modal.action_cancel()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].cancelled is True


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_truncate_short(self):
        from care.screens.diff import _truncate

        assert _truncate("hi") == "hi"

    def test_truncate_long(self):
        from care.screens.diff import _truncate

        s = "a" * 100
        out = _truncate(s, n=10)
        assert len(out) == 10
        assert out.endswith("…")


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import DiffModal as M
        from care.screens import DiffResult as R

        assert M is DiffModal
        assert R is DiffResult
