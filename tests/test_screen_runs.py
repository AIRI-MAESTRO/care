"""Pilot tests for `RunsScreen` (TODO §6 P1)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from care.runtime.local_run_history import (
    LocalRunEntry,
    record_local_run,
)
from care.screens.runs import (
    RunsScreen,
    _format_chain,
    _format_cost,
    _format_duration,
    _format_status,
    _format_tokens,
    _format_when,
)


# ---------------------------------------------------------------------------
# Pure formatters
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_when_empty(self) -> None:
        assert _format_when(0) == "—"

    def test_format_when_known(self) -> None:
        ts = time.mktime(
            time.strptime("2026-05-12 14:08:00", "%Y-%m-%d %H:%M:%S")
        )
        text = _format_when(ts)
        assert "2026-05-12" in text
        assert "14:08:00" in text

    def test_format_chain_name_and_id(self) -> None:
        row = LocalRunEntry(
            run_id="r",
            chain_id="chain-abcdefghij1234",
            chain_name="Forecaster",
        )
        assert _format_chain(row) == "Forecaster (chain-abcd)"

    def test_format_chain_id_only(self) -> None:
        row = LocalRunEntry(run_id="r", chain_id="chain-x")
        assert _format_chain(row) == "chain-x"

    def test_format_chain_neither(self) -> None:
        assert _format_chain(LocalRunEntry(run_id="r")) == "—"

    def test_format_status_success(self) -> None:
        row = LocalRunEntry(run_id="r", status="success")
        assert _format_status(row) == "✓ success"

    def test_format_status_failure(self) -> None:
        row = LocalRunEntry(run_id="r", status="failure")
        assert _format_status(row) == "✗ failure"

    def test_format_duration_seconds(self) -> None:
        assert _format_duration(4.5) == "4.5s"

    def test_format_duration_minutes(self) -> None:
        assert _format_duration(125.0) == "2m 5s"

    def test_format_duration_none(self) -> None:
        assert _format_duration(None) == "—"

    def test_format_tokens_sum(self) -> None:
        row = LocalRunEntry(
            run_id="r", tokens_in=200, tokens_out=300,
        )
        assert _format_tokens(row) == "500"

    def test_format_tokens_none(self) -> None:
        assert _format_tokens(LocalRunEntry(run_id="r")) == "—"

    def test_format_cost(self) -> None:
        assert _format_cost(0.0042) == "$0.0042"

    def test_format_cost_none(self) -> None:
        assert _format_cost(None) == "—"


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
        self.push_screen(RunsScreen(cache_dir=self._cache_dir))

    def push_toast(
        self, message, *, severity="info", ttl=None,
    ) -> None:  # type: ignore[override]
        self.toasts.append((message, severity))


def _runs(app: _Host) -> RunsScreen:
    for s in app.screen_stack:
        if isinstance(s, RunsScreen):
            return s
    raise AssertionError("RunsScreen not on stack")


class TestEmptyState:
    @pytest.mark.asyncio
    async def test_empty_cache_shows_friendly_hint(
        self, tmp_path: Path,
    ) -> None:
        app = _Host(cache_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _runs(app)
            empty = screen.query_one("#runs-empty", Static)
            assert empty.display is True
            assert "No local run history yet" in str(empty.render())


class TestPopulated:
    @pytest.mark.asyncio
    async def test_rows_populate_newest_first(
        self, tmp_path: Path,
    ) -> None:
        # Three runs with distinct timestamps.
        record_local_run(
            LocalRunEntry(
                run_id="r-old",
                started_at=100.0,
                chain_id="c-1",
                status="success",
            ),
            cache_dir=tmp_path,
        )
        record_local_run(
            LocalRunEntry(
                run_id="r-mid",
                started_at=200.0,
                chain_id="c-2",
                status="failure",
                error="boom",
            ),
            cache_dir=tmp_path,
        )
        record_local_run(
            LocalRunEntry(
                run_id="r-new",
                started_at=300.0,
                chain_id="c-3",
                status="success",
                duration_seconds=2.5,
            ),
            cache_dir=tmp_path,
        )

        app = _Host(cache_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _runs(app)
            table = screen.query_one("#runs-table", DataTable)
            assert table.row_count == 3
            # Rows are sorted newest-first in the screen.
            keys = [r.value for r in table.rows.keys()]
            assert keys == ["r-new", "r-mid", "r-old"]
            empty = screen.query_one("#runs-empty", Static)
            assert empty.display is False

    @pytest.mark.asyncio
    async def test_refresh_action_reloads(self, tmp_path: Path):
        app = _Host(cache_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _runs(app)
            # Initially empty.
            assert screen.rows == []
            # Drop a fresh row + refresh.
            record_local_run(
                LocalRunEntry(
                    run_id="r-fresh",
                    chain_id="c",
                    started_at=time.time(),
                ),
                cache_dir=tmp_path,
            )
            screen.action_refresh()
            await pilot.pause()
            assert any(
                r.run_id == "r-fresh" for r in screen.rows
            )
            assert ("refresh", "") in screen.action_log


class TestActionOpenRun:
    @pytest.mark.asyncio
    async def test_open_run_toasts_when_no_replay_path(
        self, tmp_path: Path,
    ) -> None:
        record_local_run(
            LocalRunEntry(
                run_id="r-open",
                chain_id="chain-here",
                started_at=time.time(),
                # no replay_path → falls back to toast
            ),
            cache_dir=tmp_path,
        )
        app = _Host(cache_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _runs(app)
            screen.action_open_run()
            await pilot.pause()
            assert any(
                "r-open" in m and "chain-here" in m
                for m, _ in app.toasts
            )
            assert ("open_run", "r-open") in screen.action_log

    @pytest.mark.asyncio
    async def test_open_run_pushes_replay_when_sidecar_exists(
        self, tmp_path: Path,
    ) -> None:
        import json
        from care.screens.replay import ReplayScreen

        # Write a real JSON sidecar with the shape
        # `load_replay` accepts.
        sidecar = tmp_path / "replays" / "r-replay.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps({
            "step_results": [
                {
                    "step_id": "s1",
                    "step_name": "fetch",
                    "success": True,
                },
            ],
            "final_answer": "ok",
            "total_execution_time": 0.5,
        }))
        record_local_run(
            LocalRunEntry(
                run_id="r-replay",
                chain_id="chain-x",
                chain_name="Demo",
                started_at=time.time(),
                replay_path=str(sidecar),
            ),
            cache_dir=tmp_path,
        )
        app = _Host(cache_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _runs(app)
            screen.action_open_run()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, ReplayScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_open_run_missing_sidecar_warns(
        self, tmp_path: Path,
    ) -> None:
        record_local_run(
            LocalRunEntry(
                run_id="r-missing",
                started_at=time.time(),
                replay_path=str(tmp_path / "nope.json"),
            ),
            cache_dir=tmp_path,
        )
        app = _Host(cache_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _runs(app)
            screen.action_open_run()
            await pilot.pause()
            assert any(
                "missing" in m for m, _ in app.toasts
            )


# ---------------------------------------------------------------------------
# /runs slash command integration
# ---------------------------------------------------------------------------


class TestSlashIntegration:
    @pytest.mark.asyncio
    async def test_bare_runs_command_pushes_screen(self) -> None:
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
            inp.value = "/runs"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, RunsScreen)
                for s in app.screen_stack
            )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports_runs_screen(self) -> None:
        from care.screens import RunsScreen as R

        assert R is RunsScreen
