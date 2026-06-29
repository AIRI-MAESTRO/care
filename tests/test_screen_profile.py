"""Pilot tests for `ProfileScreen` (TODO §6 P2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from care.runtime.profiles import profiles_dir
from care.screens.profile import (
    ProfileScreen,
    _format_path,
    _format_size,
    _format_when,
)


# ---------------------------------------------------------------------------
# Pure formatters
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_path_short(self) -> None:
        p = Path("/home/me/profile.toml")
        assert _format_path(p) == "/home/me/profile.toml"

    def test_format_path_long_truncates(self) -> None:
        p = Path("/" + "x" * 80 + "/profile.toml")
        text = _format_path(p)
        assert text.startswith("…")
        assert len(text) == 56

    def test_format_size_bytes(self) -> None:
        assert _format_size(900) == "900 B"

    def test_format_size_kb(self) -> None:
        assert _format_size(2048) == "2.0 KB"

    def test_format_size_mb(self) -> None:
        assert _format_size(2 * 1024 * 1024) == "2.0 MB"

    def test_format_when_zero(self) -> None:
        assert _format_when(0) == "—"

    def test_format_when_value(self) -> None:
        import time
        ts = time.mktime(time.strptime(
            "2026-06-04 14:30", "%Y-%m-%d %H:%M",
        ))
        assert "2026-06-04" in _format_when(ts)
        assert "14:30" in _format_when(ts)


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, *, config_dir: Path):
        super().__init__()
        self._config_dir = config_dir
        self.toasts: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(
            ProfileScreen(config_dir=self._config_dir),
        )

    def push_toast(
        self, message, *, severity="info", ttl=None,
    ) -> None:  # type: ignore[override]
        self.toasts.append((message, severity))


def _screen(app: _Host) -> ProfileScreen:
    for s in app.screen_stack:
        if isinstance(s, ProfileScreen):
            return s
    raise AssertionError("ProfileScreen not on stack")


class TestEmptyState:
    @pytest.mark.asyncio
    async def test_empty_dir_shows_hint(
        self, tmp_path: Path,
    ) -> None:
        app = _Host(config_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            empty = screen.query_one("#profile-empty", Static)
            assert empty.display is True
            assert "No profiles under" in str(empty.render())
            assert "CARE_PROFILE" in str(empty.render())


class TestPopulated:
    @pytest.mark.asyncio
    async def test_rows_populate_alphabetically(
        self, tmp_path: Path,
    ) -> None:
        pdir = profiles_dir(config_dir=tmp_path)
        pdir.mkdir(parents=True)
        for name in ("zeus", "athena", "ares"):
            (pdir / f"{name}.toml").write_text("[mage]\n")
        app = _Host(config_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            table = screen.query_one("#profile-table", DataTable)
            assert table.row_count == 3
            keys = [r.value for r in table.rows.keys()]
            assert keys == ["ares", "athena", "zeus"]

    @pytest.mark.asyncio
    async def test_active_profile_highlighted(
        self, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdir = profiles_dir(config_dir=tmp_path)
        pdir.mkdir(parents=True)
        (pdir / "dev.toml").write_text("[mage]\n")
        (pdir / "prod.toml").write_text("[mage]\n")
        monkeypatch.setenv("CARE_PROFILE", "prod")
        app = _Host(config_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.active_name == "prod"
            status = screen.query_one("#profile-status", Static)
            assert "active: prod" in str(status.render())

    @pytest.mark.asyncio
    async def test_status_when_no_active_env(
        self, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdir = profiles_dir(config_dir=tmp_path)
        pdir.mkdir(parents=True)
        (pdir / "dev.toml").write_text("[mage]\n")
        monkeypatch.delenv("CARE_PROFILE", raising=False)
        app = _Host(config_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            status = screen.query_one("#profile-status", Static)
            text = str(status.render())
            assert "no CARE_PROFILE set" in text

    @pytest.mark.asyncio
    async def test_refresh_picks_up_new_profile(
        self, tmp_path: Path,
    ) -> None:
        pdir = profiles_dir(config_dir=tmp_path)
        pdir.mkdir(parents=True)
        (pdir / "dev.toml").write_text("[mage]\n")
        app = _Host(config_dir=tmp_path)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert len(screen.rows) == 1
            (pdir / "prod.toml").write_text("[mage]\n")
            screen.action_refresh()
            await pilot.pause()
            assert len(screen.rows) == 2
            assert ("refresh", "") in screen.action_log


# ---------------------------------------------------------------------------
# /profile slash command
# ---------------------------------------------------------------------------


class TestSlashIntegration:
    @pytest.mark.asyncio
    async def test_bare_profile_command_pushes_screen(self):
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
            inp.value = "/profile"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, ProfileScreen)
                for s in app.screen_stack
            )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports_profile_screen(self):
        from care.screens import ProfileScreen as P

        assert P is ProfileScreen
