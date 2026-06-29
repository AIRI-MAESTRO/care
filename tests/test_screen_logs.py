"""Pilot tests for `LogsScreen` (TODO §6 P2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from care.screens.logs import LogsScreen


# ---------------------------------------------------------------------------
# Host scaffold
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, *, log_path: Path | None = None):
        super().__init__()
        self._log_path = log_path
        self.toasts: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(LogsScreen(log_path=self._log_path))

    def push_toast(
        self, message, *, severity="info", ttl=None,
    ) -> None:  # type: ignore[override]
        self.toasts.append((message, severity))


def _screen(app: _Host) -> LogsScreen:
    for s in app.screen_stack:
        if isinstance(s, LogsScreen):
            return s
    raise AssertionError("LogsScreen not on stack")


def _write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


class TestEmptyState:
    @pytest.mark.asyncio
    async def test_no_log_file_shows_friendly_hint(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Make sure neither env nor handler resolves to a file.
        monkeypatch.delenv("CARE_LOG_FILE", raising=False)
        app = _Host(log_path=None)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.resolved_path is None
            assert screen.lines == []
            content = screen.query_one("#logs-content", Static)
            assert "No log file found" in str(content.render())

    @pytest.mark.asyncio
    async def test_empty_file_renders_empty_hint(
        self, tmp_path: Path,
    ) -> None:
        empty = tmp_path / "care-app-empty.log"
        empty.touch()
        app = _Host(log_path=empty)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            content = screen.query_one("#logs-content", Static)
            assert "is empty" in str(content.render())


# ---------------------------------------------------------------------------
# Populated viewer
# ---------------------------------------------------------------------------


class TestPopulated:
    @pytest.mark.asyncio
    async def test_lines_render_into_content_pane(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "care-app-pop.log"
        _write_log(path, [
            "2026-06-04T10:00:00 [INFO] care.app: boot",
            "2026-06-04T10:00:01 [WARNING] care.x: slow",
            "2026-06-04T10:00:02 [ERROR] care.x: boom",
        ])
        app = _Host(log_path=path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert len(screen.lines) == 3
            content = screen.query_one("#logs-content", Static)
            text = str(content.render())
            assert "boot" in text
            assert "slow" in text
            assert "boom" in text

    @pytest.mark.asyncio
    async def test_meta_shows_resolved_path(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "care-app-x.log"
        _write_log(path, ["[INFO] care.x: 1"])
        app = _Host(log_path=path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            meta = screen.query_one("#logs-meta", Static)
            text = str(meta.render())
            assert str(path) in text


class TestBindings:
    @pytest.mark.asyncio
    async def test_cycle_level_walks_None_DEBUG_INFO_WARNING_ERROR(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "care-app.log"
        _write_log(path, ["[INFO] care: hi"])
        app = _Host(log_path=path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.level_floor is None
            screen.action_cycle_level()
            assert screen.level_floor == "DEBUG"
            screen.action_cycle_level()
            assert screen.level_floor == "INFO"
            screen.action_cycle_level()
            assert screen.level_floor == "WARNING"
            screen.action_cycle_level()
            assert screen.level_floor == "ERROR"
            screen.action_cycle_level()
            assert screen.level_floor is None
            log = screen.action_log
            assert any(t == "cycle_level" for t, _ in log)

    @pytest.mark.asyncio
    async def test_level_filter_drops_lines(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "care-app.log"
        _write_log(path, [
            "[DEBUG] care.x: d",
            "[INFO] care.x: i",
            "[WARNING] care.x: w",
        ])
        app = _Host(log_path=path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert len(screen.lines) == 3
            # Cycle to DEBUG, INFO, WARNING — at WARNING we
            # should keep only the warning line.
            screen.action_cycle_level()  # DEBUG
            screen.action_cycle_level()  # INFO
            screen.action_cycle_level()  # WARNING
            assert screen.level_floor == "WARNING"
            assert len(screen.lines) == 1
            assert "[WARNING]" in screen.lines[0]

    @pytest.mark.asyncio
    async def test_toggle_module_filter_shows_input(
        self, tmp_path: Path,
    ) -> None:
        from textual.widgets import Input

        path = tmp_path / "care-app.log"
        _write_log(path, ["[INFO] care.x: 1"])
        app = _Host(log_path=path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            inp = screen.query_one(
                "#logs-filter-input", Input,
            )
            assert not inp.has_class("-visible")
            screen.action_toggle_module_filter()
            await pilot.pause()
            assert inp.has_class("-visible")
            # Toggle again hides + clears (no filter set).
            screen.action_toggle_module_filter()
            await pilot.pause()
            assert not inp.has_class("-visible")
            assert screen.module_substr == ""

    @pytest.mark.asyncio
    async def test_input_submit_applies_module_filter(
        self, tmp_path: Path,
    ) -> None:
        from textual.widgets import Input

        path = tmp_path / "care-app.log"
        _write_log(path, [
            "[INFO] care.chat: chat-line",
            "[INFO] httpx.client: http-line",
            "[INFO] care.app: app-line",
        ])
        app = _Host(log_path=path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert len(screen.lines) == 3
            screen.action_toggle_module_filter()
            await pilot.pause()
            inp = screen.query_one(
                "#logs-filter-input", Input,
            )
            inp.value = "care.chat"
            await inp.action_submit()
            for _ in range(2):
                await pilot.pause()
            assert screen.module_substr == "care.chat"
            assert len(screen.lines) == 1
            assert "chat-line" in screen.lines[0]
            # Input hides itself after submit.
            assert not inp.has_class("-visible")

    @pytest.mark.asyncio
    async def test_status_text_mentions_active_module_filter(
        self, tmp_path: Path,
    ) -> None:
        from textual.widgets import Static

        path = tmp_path / "care-app.log"
        _write_log(path, ["[INFO] care.x: 1"])
        app = _Host(log_path=path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.module_substr = "httpx"
            screen.refresh_log()
            await pilot.pause()
            status = screen.query_one("#logs-status", Static)
            text = str(status.render())
            assert "module ~ 'httpx'" in text

    @pytest.mark.asyncio
    async def test_clearing_via_empty_submit(
        self, tmp_path: Path,
    ) -> None:
        from textual.widgets import Input

        path = tmp_path / "care-app.log"
        _write_log(path, [
            "[INFO] care.chat: kept",
            "[INFO] other: other-line",
        ])
        app = _Host(log_path=path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            # Apply a filter first.
            screen.module_substr = "care.chat"
            screen.refresh_log()
            await pilot.pause()
            assert len(screen.lines) == 1
            # Now open + clear via empty submit.
            screen.action_toggle_module_filter()
            await pilot.pause()
            inp = screen.query_one(
                "#logs-filter-input", Input,
            )
            inp.value = ""
            await inp.action_submit()
            for _ in range(2):
                await pilot.pause()
            assert screen.module_substr == ""
            assert len(screen.lines) == 2

    @pytest.mark.asyncio
    async def test_refresh_action_rereads_file(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "care-app.log"
        _write_log(path, ["[INFO] care.x: pre"])
        app = _Host(log_path=path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert len(screen.lines) == 1
            # Append a new line + refresh.
            with path.open("a") as fp:
                fp.write("[INFO] care.x: post\n")
            screen.action_refresh()
            await pilot.pause()
            assert len(screen.lines) == 2
            assert ("refresh", "") in screen.action_log


# ---------------------------------------------------------------------------
# /logs slash command
# ---------------------------------------------------------------------------


class TestSlashIntegration:
    @pytest.mark.asyncio
    async def test_bare_logs_command_pushes_screen(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from care.screens.chat import ChatScreen
        from care.widgets.chat_input import ChatInput

        monkeypatch.delenv("CARE_LOG_FILE", raising=False)

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
            inp.value = "/logs"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, LogsScreen)
                for s in app.screen_stack
            )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports_logs_screen(self) -> None:
        from care.screens import LogsScreen as L

        assert L is LogsScreen
