"""Pilot tests for InspectionScreen RunHistory tab (TODO §1.1 P0.20).

Exercises:
* Composition — the RunHistory `TabPane` mounts with a
  DataTable + summary Static.
* Activating the tab triggers the lazy-load worker.
* `fetch_run_history` results populate the table.
* `summarize_run_history` output lands in the summary Static
  (via `_format_history_summary`).
* Empty + error paths render gracefully.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static, TabbedContent

from care.runtime.run_history import RunHistorySummary
from care.screens.inspection import InspectionScreen


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _chain_response(*, entity_id="agent-1"):
    return {
        "entity_id": entity_id,
        "entity_type": "chain",
        "version_id": "v1",
        "channel": "latest",
        "etag": "e",
        "favourite": False,
        "meta": {
            "display_name": "Storm Watcher",
            "domain": "weather",
            "tags": ["weather"],
            "name": "storm",
        },
        "content": {
            "steps": [
                {
                    "name": "fetch",
                    "type": "llm",
                    "deps": [],
                },
            ],
            "description": "Watches storms",
        },
    }


def _run_card(
    *,
    run_id: str,
    agent: str = "agent-1",
    success: bool = True,
    finished: datetime | None = None,
    duration: float = 1.5,
    tokens: int = 100,
    error: str | None = None,
):
    finished_str = (finished or datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)).isoformat()
    status_tag = "status:success" if success else "status:failure"
    tags = ["agent_run", f"agent:{agent}", status_tag]
    metrics: dict = {
        "duration_seconds": duration,
        "total_tokens": tokens,
        "step_count": 2,
    }
    if error:
        metrics["error_message"] = error
    return {
        "entity_id": f"card-{run_id}",
        "entity_type": "memory_card",
        "version_id": "v",
        "channel": "latest",
        "etag": "e",
        "meta": {"tags": tags, "name": f"card-{run_id}"},
        "content": {
            "category": "agent_run",
            "usage": {
                "agent_entity_id": agent,
                "run_id": run_id,
                "finished_at": finished_str,
                "metrics": metrics,
            },
        },
    }


class _StubClient:
    def __init__(self, *, cards=None, fail_history: bool = False):
        self._cards = list(cards or [])
        self._fail_history = fail_history
        self.list_entities_calls: list[dict] = []

    def get_chain(self, entity_id, channel):
        return _chain_response(entity_id=entity_id)

    def _list_entities(self, entity_type, **kw):
        self.list_entities_calls.append(
            {"entity_type": entity_type, **kw}
        )
        if self._fail_history:
            raise RuntimeError("history-down")
        # Filter by tag AND-match like Memory does.
        wanted = set(kw.get("tags") or [])
        return [
            c for c in self._cards
            if wanted.issubset(set(c["meta"].get("tags") or []))
        ]


class _StubMemory:
    def __init__(self, *, cards=None, fail_history: bool = False):
        self.client = _StubClient(cards=cards, fail_history=fail_history)


class _InspHost(App):
    def __init__(self, *, cards=None, fail_history: bool = False):
        super().__init__()
        self.memory = _StubMemory(
            cards=cards,
            fail_history=fail_history,
        )

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(InspectionScreen("agent-1"))


def _screen(app: App) -> InspectionScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, InspectionScreen)
    return s


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_tab_pane_mounts(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            assert screen.query_one(
                "#inspection-tab-history",
            ) is not None
            assert screen.query_one(
                "#inspection-history-table", DataTable,
            ) is not None
            assert screen.query_one(
                "#inspection-history-summary", Static,
            ) is not None

    @pytest.mark.asyncio
    async def test_history_columns_added_on_mount(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            table = screen.query_one(
                "#inspection-history-table", DataTable,
            )
            assert len(table.columns) == 6


# ---------------------------------------------------------------------------
# Lazy-load on tab activation
# ---------------------------------------------------------------------------


class TestLazyLoad:
    @pytest.mark.asyncio
    async def test_history_not_loaded_until_tab_activated(self):
        app = _InspHost(cards=[_run_card(run_id="r1")])
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            assert screen._history_loaded is False
            assert app.memory.client.list_entities_calls == []

    @pytest.mark.asyncio
    async def test_tab_activation_loads_history(self):
        cards = [
            _run_card(run_id="r-1", success=True, tokens=100),
            _run_card(run_id="r-2", success=False, error="boom"),
        ]
        app = _InspHost(cards=cards)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            tabs = screen.query_one(TabbedContent)
            tabs.active = "inspection-tab-history"
            for _ in range(8):
                await pilot.pause()
            assert screen._history_loaded is True
            assert len(screen.run_history) == 2
            assert screen.run_history_summary.total_runs == 2
            assert screen.run_history_summary.failure_count == 1
            table = screen.query_one(
                "#inspection-history-table", DataTable,
            )
            assert table.row_count == 2

    @pytest.mark.asyncio
    async def test_tab_activation_renders_summary(self):
        cards = [
            _run_card(run_id="r-1", success=True, tokens=10, duration=2.0),
            _run_card(run_id="r-2", success=True, tokens=20, duration=4.0),
        ]
        app = _InspHost(cards=cards)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            tabs = screen.query_one(TabbedContent)
            tabs.active = "inspection-tab-history"
            for _ in range(8):
                await pilot.pause()
            text = screen._format_history_summary()
            assert "runs: 2" in text
            assert "ok: 2" in text
            assert "tokens: 30" in text


# ---------------------------------------------------------------------------
# Error + empty paths
# ---------------------------------------------------------------------------


class TestResilience:
    @pytest.mark.asyncio
    async def test_empty_history_renders_no_runs(self):
        app = _InspHost(cards=[])
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            tabs = screen.query_one(TabbedContent)
            tabs.active = "inspection-tab-history"
            for _ in range(8):
                await pilot.pause()
            assert screen.run_history == ()
            assert screen._format_history_summary() == "No runs yet"

    @pytest.mark.asyncio
    async def test_fetch_failure_lands_on_error(self):
        app = _InspHost(fail_history=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            tabs = screen.query_one(TabbedContent)
            tabs.active = "inspection-tab-history"
            for _ in range(8):
                await pilot.pause()
            assert screen.run_history_error is not None
            assert "history-down" in screen.run_history_error
            text = screen._format_history_summary()
            assert text.startswith("⚠")

    @pytest.mark.asyncio
    async def test_no_memory_lands_on_error(self):
        class _NoMemHost(App):
            memory = None

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(InspectionScreen("x"))

        app = _NoMemHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            tabs = screen.query_one(TabbedContent)
            tabs.active = "inspection-tab-history"
            for _ in range(8):
                await pilot.pause()
            assert screen.run_history_error is not None


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_reactivating_tab_doesnt_reload(self):
        app = _InspHost(cards=[_run_card(run_id="r-1")])
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            tabs = screen.query_one(TabbedContent)
            tabs.active = "inspection-tab-history"
            for _ in range(8):
                await pilot.pause()
            first_calls = len(app.memory.client.list_entities_calls)
            # Switch back to chain then to history again.
            tabs.active = "inspection-tab-chain"
            await pilot.pause()
            tabs.active = "inspection-tab-history"
            for _ in range(4):
                await pilot.pause()
            assert (
                len(app.memory.client.list_entities_calls) == first_calls
            )


# ---------------------------------------------------------------------------
# Row formatting helper (pure)
# ---------------------------------------------------------------------------


class TestRowFormatting:
    def test_row_cells_render_columns(self):
        from care.runtime.run_history import RunHistoryEntry

        entry = RunHistoryEntry(
            card_id="c", agent_entity_id="a", run_id="r-abcdefghij",
            finished_at=datetime(2026, 5, 19, 12, 0),
            status="success",
            duration_seconds=2.5,
            total_tokens=42,
        )
        cells = InspectionScreen._history_row_cells(entry)
        assert cells[0] == "2026-05-19 12:00"
        assert cells[1] == "✓"
        assert cells[3] == "2.5s"
        assert cells[4] == "42"

    def test_row_cells_for_failure(self):
        from care.runtime.run_history import RunHistoryEntry

        entry = RunHistoryEntry(
            card_id="c", agent_entity_id="a", run_id="r-x",
            finished_at=None,
            status="failure",
            error_message="boom",
            duration_seconds=None,
            total_tokens=None,
        )
        cells = InspectionScreen._history_row_cells(entry)
        assert cells[0] == "—"
        assert cells[1] == "✗"
        assert cells[3] == "—"
        assert cells[5] == "boom"


# ---------------------------------------------------------------------------
# Summary helper (pure)
# ---------------------------------------------------------------------------


class TestSummary:
    @pytest.mark.asyncio
    async def test_summary_empty(self):
        app = _InspHost(cards=[])
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert isinstance(
                screen.run_history_summary, RunHistorySummary,
            )
            assert screen.run_history_summary.total_runs == 0
