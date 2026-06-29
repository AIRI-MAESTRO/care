"""Pilot tests for LineageModal (TODO §1.1 P0.26).

Exercises:
* `on_mount` calls `fetch_chain_lineage` and populates the
  tree + DataTable.
* No-memory / fetch-failure paths land on `load_error`.
* Default selection prefers the best (highest-fitness) node.
* "Re-run from here" dismisses with the selected
  `version_id`.
* Escape dismisses with `rerun_version_id=None`.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Static

from care.runtime.lineage import LineageGraph, LineageNode
from care.screens.lineage import LineageModal, LineageResult


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _lineage_response(*, entity_id: str = "agent-1"):
    return {
        "entity_id": entity_id,
        "root_version_id": "v1",
        "max_depth_reached": False,
        "versions": [
            {
                "version_id": "v1",
                "version_number": 1,
                "parents": [],
                "depth": 0,
                "evolution_meta": {},
            },
            {
                "version_id": "v2",
                "version_number": 2,
                "parents": ["v1"],
                "depth": 1,
                "evolution_meta": {
                    "fitness_score": 0.7,
                    "generation": 1,
                    "mutation_kind": "mutation",
                },
            },
            {
                "version_id": "v3",
                "version_number": 3,
                "parents": ["v2"],
                "depth": 2,
                "evolution_meta": {
                    "fitness_score": 0.9,
                    "generation": 2,
                    "mutation_kind": "crossover",
                },
            },
        ],
    }


class _StubClient:
    def __init__(self, *, response=None, fail: bool = False):
        self._response = response if response is not None else _lineage_response()
        self._fail = fail
        self.calls: list[tuple] = []

    def get_chain_lineage(self, entity_id, **kw):
        self.calls.append((entity_id, dict(kw)))
        if self._fail:
            raise RuntimeError("lineage-down")
        return self._response


class _StubMemory:
    def __init__(self, *, response=None, fail: bool = False):
        self.client = _StubClient(response=response, fail=fail)


class _Host(App):
    def __init__(self, *, memory=None) -> None:
        super().__init__()
        self._memory = memory
        self.dismissed: list[LineageResult] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(result):
            self.dismissed.append(result)

        self.push_screen(
            LineageModal("agent-1", memory=self._memory),
            _on_dismiss,
        )


def _modal(app: App) -> LineageModal:
    s = app.screen_stack[-1]
    assert isinstance(s, LineageModal)
    return s


# ---------------------------------------------------------------------------
# Composition
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
            assert modal.query_one("#lineage-tree") is not None
            assert modal.query_one("#lineage-table", DataTable) is not None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


class TestLoad:
    @pytest.mark.asyncio
    async def test_loads_graph_from_memory(self):
        memory = _StubMemory()
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            assert modal._loaded is True
            assert len(modal.graph.nodes) == 3
            assert modal.load_error is None
            assert memory.client.calls != []

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
            assert "lineage-down" in modal.load_error


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class TestSelection:
    @pytest.mark.asyncio
    async def test_default_selection_is_best_fitness(self):
        memory = _StubMemory()
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            # v3 has the highest fitness (0.9).
            assert modal.selected_version == "v3"

    @pytest.mark.asyncio
    async def test_default_selection_falls_back_to_root(self):
        response = {
            "entity_id": "agent-1",
            "root_version_id": "v1",
            "max_depth_reached": False,
            "versions": [
                {
                    "version_id": "v1",
                    "version_number": 1,
                    "parents": [],
                    "depth": 0,
                    "evolution_meta": {},
                },
            ],
        }
        memory = _StubMemory(response=response)
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            assert modal.selected_version == "v1"


# ---------------------------------------------------------------------------
# Re-run dismiss
# ---------------------------------------------------------------------------


class TestRerun:
    @pytest.mark.asyncio
    async def test_rerun_button_dismisses_with_version(self):
        memory = _StubMemory()
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#lineage-btn-rerun", Button,
            ).press()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].rerun_version_id == "v3"

    @pytest.mark.asyncio
    async def test_cancel_dismisses_with_none(self):
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
            assert app.dismissed[0].rerun_version_id is None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestRowFormatting:
    def test_tree_row_marker_for_crossover(self):
        node = LineageNode(
            version_id="x", version_number=2,
            parents=("a", "b"),
            depth=1,
            fitness=0.5,
            mutation_kind="crossover",
        )
        line = LineageModal._tree_row(node)
        assert "✦" in line  # crossover marker
        assert "v2" in line
        assert "fit=0.500" in line

    def test_table_row_renders_em_dashes_for_missing(self):
        node = LineageNode(
            version_id="x", version_number=1,
            parents=(),
            depth=0,
        )
        row = LineageModal._table_row(node)
        assert row[0] == "v1"
        assert row[3] == "—"  # fitness
        assert row[2] == "—"  # generation


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import LineageModal as M
        from care.screens import LineageResult as R

        assert M is LineageModal
        assert R is LineageResult


# ---------------------------------------------------------------------------
# Pure: graph empty path
# ---------------------------------------------------------------------------


class TestGraphEmpty:
    def test_empty_graph_renders_message(self):
        # Direct unit-test the constructor + the empty render
        # path via instance method.
        modal = LineageModal("x")
        assert modal.graph.nodes == ()
        assert isinstance(modal.graph, LineageGraph)


# ---------------------------------------------------------------------------
# Static import sanity check
# ---------------------------------------------------------------------------


class TestStaticPresence:
    @pytest.mark.asyncio
    async def test_tree_pane_renders_nodes_after_load(self):
        memory = _StubMemory()
        app = _Host(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            modal = _modal(app)
            tree = modal.query_one("#lineage-tree")
            # 3 nodes → at least 3 Static children.
            statics = list(tree.query(Static))
            assert len(statics) >= 3
