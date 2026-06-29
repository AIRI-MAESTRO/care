"""Pilot tests for TaskListDrawer (TODO §1.1 P0.36).

Exercises:
* `on_mount` populates the table from
  `registry.list_tasks(active_only=True)`.
* `on_change` subscription drives live refreshes.
* `Cancel` button calls `registry.cancel(id)`.
* `Switch to` posts `SwitchRequested` and dismisses.
* `Ctrl+B` on CareApp pushes the drawer.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable

from care.runtime.task_registry import TaskRegistry
from care.screens.task_list import TaskListDrawer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(registry: TaskRegistry, *, count: int = 2) -> list:
    records = []
    for i in range(count):
        records.append(
            registry.register(
                kind="mage_generation",
                label=f"task-{i}",
            )
        )
    return records


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, registry: TaskRegistry) -> None:
        super().__init__()
        self._task_registry = registry
        self.switched: list = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(TaskListDrawer(self._task_registry))

    def on_task_list_drawer_switch_requested(
        self, event: TaskListDrawer.SwitchRequested,
    ) -> None:
        self.switched.append(event.record)


def _drawer(app: App) -> TaskListDrawer:
    s = app.screen_stack[-1]
    assert isinstance(s, TaskListDrawer)
    return s


# ---------------------------------------------------------------------------
# Mount + populate
# ---------------------------------------------------------------------------


class TestPopulate:
    @pytest.mark.asyncio
    async def test_table_populates_on_mount(self):
        registry = TaskRegistry()
        _seed(registry, count=3)
        app = _Host(registry)
        async with app.run_test() as pilot:
            await pilot.pause()
            drawer = _drawer(app)
            assert len(drawer.records) == 3
            table = drawer.query_one("#task-list-table", DataTable)
            assert table.row_count == 3

    @pytest.mark.asyncio
    async def test_terminal_tasks_excluded_with_active_only(self):
        registry = TaskRegistry()
        recs = _seed(registry, count=2)
        registry.mark_running(recs[0].id)
        registry.mark_completed(recs[0].id)
        # recs[1] still pending.
        app = _Host(registry)
        async with app.run_test() as pilot:
            await pilot.pause()
            drawer = _drawer(app)
            ids = [r.id for r in drawer.records]
            assert recs[0].id not in ids
            assert recs[1].id in ids


# ---------------------------------------------------------------------------
# Live refresh
# ---------------------------------------------------------------------------


class TestLiveRefresh:
    @pytest.mark.asyncio
    async def test_new_task_refreshes_table(self):
        registry = TaskRegistry()
        _seed(registry, count=1)
        app = _Host(registry)
        async with app.run_test() as pilot:
            await pilot.pause()
            drawer = _drawer(app)
            assert len(drawer.records) == 1
            # Register a new task post-mount.
            registry.register(kind="carl_execution", label="new")
            for _ in range(3):
                await pilot.pause()
            assert len(drawer.records) == 2


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_button_cancels_first_row(self):
        registry = TaskRegistry()
        recs = _seed(registry, count=1)
        app = _Host(registry)
        async with app.run_test() as pilot:
            await pilot.pause()
            drawer = _drawer(app)
            drawer.query_one(
                "#task-list-btn-cancel", Button,
            ).press()
            for _ in range(3):
                await pilot.pause()
            assert drawer.last_cancelled_id == recs[0].id
            assert registry.get(recs[0].id).status == "cancelled"


# ---------------------------------------------------------------------------
# Switch to
# ---------------------------------------------------------------------------


class TestSwitch:
    @pytest.mark.asyncio
    async def test_switch_posts_message_and_dismisses(self):
        registry = TaskRegistry()
        recs = _seed(registry, count=1)
        app = _Host(registry)
        async with app.run_test() as pilot:
            await pilot.pause()
            drawer = _drawer(app)
            initial_depth = len(app.screen_stack)
            drawer.query_one(
                "#task-list-btn-switch", Button,
            ).press()
            for _ in range(3):
                await pilot.pause()
            assert len(app.switched) == 1
            assert app.switched[0].id == recs[0].id
            assert len(app.screen_stack) < initial_depth


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_button_dismisses(self):
        registry = TaskRegistry()
        app = _Host(registry)
        async with app.run_test() as pilot:
            await pilot.pause()
            drawer = _drawer(app)
            initial_depth = len(app.screen_stack)
            drawer.query_one(
                "#task-list-btn-close", Button,
            ).press()
            for _ in range(3):
                await pilot.pause()
            assert len(app.screen_stack) < initial_depth

    @pytest.mark.asyncio
    async def test_escape_dismisses(self):
        registry = TaskRegistry()
        app = _Host(registry)
        async with app.run_test() as pilot:
            await pilot.pause()
            initial_depth = len(app.screen_stack)
            await pilot.press("escape")
            for _ in range(3):
                await pilot.pause()
            assert len(app.screen_stack) < initial_depth


# ---------------------------------------------------------------------------
# CareApp integration
# ---------------------------------------------------------------------------


class TestAppIntegration:
    @pytest.mark.asyncio
    async def test_ctrl_b_pushes_drawer(self):
        from care.app import CareApp

        app = CareApp()
        async with app.run_test() as pilot:
            # Wait for Welcome's splash timer + auto-route to
            # settle on the production screen.
            await pilot.pause(0.5)
            for _ in range(4):
                await pilot.pause()
            app.action_open_task_list()
            for _ in range(4):
                await pilot.pause()
            assert isinstance(
                app.screen_stack[-1], TaskListDrawer,
            )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import TaskListDrawer as D

        assert D is TaskListDrawer
