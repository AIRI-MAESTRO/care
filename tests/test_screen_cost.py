"""Pilot tests for `CostDashboardScreen` (TODO §6 P2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from care.runtime.cost_rollups import OverallTotals
from care.runtime.local_run_history import (
    LocalRunEntry,
    record_local_run,
)
from care.screens.cost import (
    CostDashboardScreen,
    format_cost,
    format_duration,
    format_overall,
)


# ---------------------------------------------------------------------------
# Pure formatters
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_cost_zero(self):
        assert format_cost(0) == "$0.0000"
        assert format_cost(None) == "$0.0000"

    def test_format_cost_value(self):
        assert format_cost(0.0042) == "$0.0042"
        assert format_cost(1.23456) == "$1.2346"

    def test_format_duration_zero(self):
        assert format_duration(0) == "0s"
        assert format_duration(None) == "0s"

    def test_format_duration_seconds(self):
        assert format_duration(4.5) == "4.5s"

    def test_format_duration_minutes(self):
        assert format_duration(125) == "2m 5s"

    def test_format_duration_hours(self):
        assert format_duration(3725) == "1h 2m"

    def test_format_overall_empty(self):
        assert format_overall(OverallTotals()) == "no runs yet"

    def test_format_overall_populated(self):
        out = format_overall(OverallTotals(
            runs=10,
            successful_runs=8,
            failed_runs=2,
            tokens_in=1000,
            tokens_out=500,
            cost_usd=1.25,
            total_duration_seconds=620.0,
        ))
        assert "runs: 10" in out
        assert "success: 80%" in out
        assert "tokens: 1500" in out
        assert "$1.2500" in out
        assert "10m 20s" in out


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, *, cache_dir: Path | None = None):
        super().__init__()
        self._cache_dir = cache_dir
        self.toasts: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(
            CostDashboardScreen(cache_dir=self._cache_dir),
        )

    def push_toast(
        self, message, *, severity="info", ttl=None,
    ) -> None:  # type: ignore[override]
        self.toasts.append((message, severity))


def _screen(app: _Host) -> CostDashboardScreen:
    for s in app.screen_stack:
        if isinstance(s, CostDashboardScreen):
            return s
    raise AssertionError("CostDashboardScreen not on stack")


class TestEmptyState:
    @pytest.mark.asyncio
    async def test_empty_cache_shows_hint_and_zero_overall(
        self, tmp_path: Path,
    ) -> None:
        app = _Host(cache_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.overall.runs == 0
            assert screen.by_provider == []
            empty = screen.query_one("#cost-empty", Static)
            assert empty.display is True
            assert "No local run history yet" in str(empty.render())


class TestPopulated:
    @pytest.mark.asyncio
    async def test_rollups_populate_from_cache(
        self, tmp_path: Path,
    ) -> None:
        for entry in [
            LocalRunEntry(
                run_id="r1",
                chain_id="c1",
                chain_name="Forecaster",
                provider="openai",
                mode="ad_hoc",
                started_at=100.0,
                tokens_in=100,
                tokens_out=50,
                cost_usd=0.5,
                status="success",
            ),
            LocalRunEntry(
                run_id="r2",
                chain_id="c2",
                provider="anthropic",
                mode="production",
                started_at=200.0,
                tokens_in=200,
                cost_usd=1.25,
                status="failure",
                error="boom",
            ),
        ]:
            record_local_run(entry, cache_dir=tmp_path)

        app = _Host(cache_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.overall.runs == 2
            assert screen.overall.cost_usd == 1.75
            # Anthropic (cost 1.25) > openai (0.5).
            assert [
                row.key for row in screen.by_provider
            ] == ["anthropic", "openai"]
            # Header text reflects totals.
            overall = screen.query_one(
                "#cost-overall", Static,
            )
            text = str(overall.render())
            assert "runs: 2" in text
            assert "$1.7500" in text
            # Empty hint stays hidden.
            empty = screen.query_one("#cost-empty", Static)
            assert empty.display is False

    @pytest.mark.asyncio
    async def test_table_columns_populate(self, tmp_path: Path):
        record_local_run(
            LocalRunEntry(
                run_id="r",
                chain_id="c",
                chain_name="Demo",
                provider="openai",
                mode="ad_hoc",
                tokens_in=10,
                tokens_out=20,
                cost_usd=0.5,
                started_at=10.0,
            ),
            cache_dir=tmp_path,
        )
        app = _Host(cache_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            ptable = screen.query_one(
                "#cost-provider-table", DataTable,
            )
            ctable = screen.query_one(
                "#cost-chain-table", DataTable,
            )
            mtable = screen.query_one(
                "#cost-mode-table", DataTable,
            )
            assert ptable.row_count == 1
            assert ctable.row_count == 1
            assert mtable.row_count == 1

    @pytest.mark.asyncio
    async def test_refresh_picks_up_new_entries(
        self, tmp_path: Path,
    ) -> None:
        app = _Host(cache_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.overall.runs == 0
            record_local_run(
                LocalRunEntry(
                    run_id="r-fresh",
                    provider="openai",
                    cost_usd=0.1,
                    started_at=10.0,
                ),
                cache_dir=tmp_path,
            )
            screen.action_refresh()
            await pilot.pause()
            assert screen.overall.runs == 1
            assert ("refresh", "") in screen.action_log


# ---------------------------------------------------------------------------
# /cost slash command integration
# ---------------------------------------------------------------------------


class TestSlashIntegration:
    @pytest.mark.asyncio
    async def test_bare_cost_command_pushes_screen(
        self,
    ) -> None:
        from care.screens.chat import ChatScreen
        from care.widgets.chat_input import ChatInput

        class _ChatHost(App):
            def compose(self):
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(ChatScreen())

        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = next(
                s for s in app.screen_stack if isinstance(s, ChatScreen)
            )
            inp = chat.query_one("#chat-input", ChatInput)
            inp.value = "/cost"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, CostDashboardScreen)
                for s in app.screen_stack
            )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports_cost_dashboard(self) -> None:
        from care.screens import CostDashboardScreen as C

        assert C is CostDashboardScreen
