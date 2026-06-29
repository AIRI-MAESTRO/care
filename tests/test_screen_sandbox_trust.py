"""Pilot tests for `SandboxTrustScreen` (TODO §6 P1)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from care.sandbox.trust import SkillTrustStore, TrustRecord
from care.screens.sandbox_trust import (
    SandboxTrustScreen,
    _format_sha,
    _format_tools,
    _format_uri,
    _format_when,
)


# ---------------------------------------------------------------------------
# Pure formatters
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_sha_empty(self) -> None:
        assert _format_sha("") == "—"

    def test_format_sha_short(self) -> None:
        assert _format_sha("abcd1234") == "abcd1234"

    def test_format_sha_long(self) -> None:
        sha = "0123456789abcdef" * 4
        assert _format_sha(sha) == "01234567…"

    def test_format_uri_empty(self) -> None:
        assert _format_uri("") == "—"

    def test_format_uri_long(self) -> None:
        uri = "https://" + "x" * 100 + ".example.com/skill"
        assert _format_uri(uri).endswith("…")
        assert len(_format_uri(uri)) <= 48

    def test_format_when_handles_datetime(self) -> None:
        dt = datetime(2026, 5, 12, 14, 8, tzinfo=timezone.utc)
        text = _format_when(dt)
        assert "2026-05-12" in text
        assert "14:08" in text

    def test_format_when_handles_none(self) -> None:
        assert _format_when(None) == "—"

    def test_format_tools_empty(self) -> None:
        assert _format_tools(()) == "—"

    def test_format_tools_few(self) -> None:
        assert _format_tools(("a", "b")) == "a, b"

    def test_format_tools_truncates_overflow(self) -> None:
        text = _format_tools(("a", "b", "c", "d", "e"))
        assert "a, b, c" in text
        assert "+2" in text


# ---------------------------------------------------------------------------
# Pilot scaffolding
# ---------------------------------------------------------------------------


def _record(
    sha: str = "abc123def456",
    *,
    name: str = "Skill One",
    uri: str = "skill://uri-1",
    policy: str = "sha_pinned",
    tools: tuple[str, ...] = (),
    approved_at: datetime | None = None,
) -> TrustRecord:
    return TrustRecord(
        sha256=sha,
        uri=uri,
        name=name,
        approved_at=approved_at or datetime(
            2026, 1, 1, tzinfo=timezone.utc,
        ),
        trust_policy=policy,  # type: ignore[arg-type]
        allowed_tools=tools,
    )


class _Host(App):
    def __init__(self, *, store: SkillTrustStore | None = None):
        super().__init__()
        self._store = store
        self.toasts: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(
            SandboxTrustScreen(store=self._store),
        )

    def push_toast(
        self, message, *, severity="info", ttl=None,
    ) -> None:  # type: ignore[override]
        self.toasts.append((message, severity))


def _screen(app: _Host) -> SandboxTrustScreen:
    for s in app.screen_stack:
        if isinstance(s, SandboxTrustScreen):
            return s
    raise AssertionError("SandboxTrustScreen not on stack")


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


class TestEmptyState:
    @pytest.mark.asyncio
    async def test_empty_store_renders_friendly_hint(self) -> None:
        store = SkillTrustStore(records={})
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            empty = screen.query_one(
                "#sandbox-trust-empty", Static,
            )
            assert empty.display is True
            assert "No trusted skills yet" in str(empty.render())


# ---------------------------------------------------------------------------
# Populated table
# ---------------------------------------------------------------------------


class TestPopulated:
    @pytest.mark.asyncio
    async def test_rows_match_store(self) -> None:
        records = {
            "sha-1": _record(
                sha="sha-1", name="A",
                approved_at=datetime(
                    2026, 5, 12, tzinfo=timezone.utc,
                ),
            ),
            "sha-2": _record(
                sha="sha-2", name="B",
                approved_at=datetime(
                    2026, 5, 13, tzinfo=timezone.utc,
                ),
            ),
        }
        store = SkillTrustStore(records=records)
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            table = screen.query_one(
                "#sandbox-trust-table", DataTable,
            )
            assert table.row_count == 2
            keys = [r.value for r in table.rows.keys()]
            # store.list_trusted sorts newest first.
            assert keys == ["sha-2", "sha-1"]

    @pytest.mark.asyncio
    async def test_refresh_action_repaints(self) -> None:
        store = SkillTrustStore(records={})
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.rows == ()
            # Mutate the injected store + refresh.
            store.trust(
                sha256="sha-late",
                uri="skill://late",
                name="Late arrival",
                allowed_tools=("read",),
            )
            screen.action_refresh()
            await pilot.pause()
            assert len(screen.rows) == 1
            assert ("refresh", "") in screen.action_log


# ---------------------------------------------------------------------------
# Revoke binding
# ---------------------------------------------------------------------------


class TestRevoke:
    @pytest.mark.asyncio
    async def test_revoke_no_row_warns(self) -> None:
        store = SkillTrustStore(records={})
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.action_revoke()
            await pilot.pause()
            assert any(
                "Highlight a row first" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_revoke_pushes_confirm_modal(self) -> None:
        from care.screens.confirm import ConfirmModal

        record = _record(sha="to-revoke")
        store = SkillTrustStore(records={"to-revoke": record})
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.action_revoke()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, ConfirmModal)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_revoke_after_confirm_removes_from_store(
        self,
    ) -> None:
        from care.screens.confirm import ConfirmModal

        record = _record(sha="to-revoke")
        store = SkillTrustStore(records={"to-revoke": record})
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.action_revoke()
            for _ in range(4):
                await pilot.pause()
            modal = next(
                s for s in app.screen_stack
                if isinstance(s, ConfirmModal)
            )
            modal.dismiss(True)  # confirm
            for _ in range(6):
                await pilot.pause()
            assert "to-revoke" not in store
            assert ("revoke", "to-revoke") in screen.action_log
            assert any(
                "Revoked" in m for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_revoke_cancelled_leaves_store_intact(
        self,
    ) -> None:
        from care.screens.confirm import ConfirmModal

        record = _record(sha="keeper")
        store = SkillTrustStore(records={"keeper": record})
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.action_revoke()
            for _ in range(4):
                await pilot.pause()
            modal = next(
                s for s in app.screen_stack
                if isinstance(s, ConfirmModal)
            )
            modal.dismiss(False)  # cancel
            for _ in range(4):
                await pilot.pause()
            assert "keeper" in store


# ---------------------------------------------------------------------------
# /sandbox slash command
# ---------------------------------------------------------------------------


class TestSlashIntegration:
    @pytest.mark.asyncio
    async def test_bare_sandbox_command_pushes_screen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from care.screens.chat import ChatScreen
        from care.widgets.chat_input import ChatInput
        from care.sandbox import trust as trust_mod

        # Redirect the default trust-store path so loading
        # doesn't read the user's real ~/.local/state file.
        monkeypatch.setattr(
            trust_mod, "DEFAULT_TRUST_PATH",
            tmp_path / "trust.json",
        )

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
            inp.value = "/sandbox"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, SandboxTrustScreen)
                for s in app.screen_stack
            )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports_sandbox_trust(self) -> None:
        from care.screens import SandboxTrustScreen as S

        assert S is SandboxTrustScreen
