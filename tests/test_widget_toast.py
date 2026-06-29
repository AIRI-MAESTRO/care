"""Pilot tests for ToastHost + CareApp.push_toast (TODO §1.1 P0.35).

Exercises:
* `push` mounts a row in the host.
* Severity badge + class land on the row.
* `app.push_toast(...)` routes through to the mounted host.
* History captures every pushed toast.
* `ttl` auto-dismisses (test with a short TTL + pilot.pause).
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from care.widgets.toast import Toast, ToastHost


# ---------------------------------------------------------------------------
# Plain host (no CareApp dependency)
# ---------------------------------------------------------------------------


class _Host(App):
    def compose(self) -> ComposeResult:
        yield ToastHost(default_ttl=0.05)


def _host(app: App) -> ToastHost:
    return app.query_one(ToastHost)


# ---------------------------------------------------------------------------
# Compose + push
# ---------------------------------------------------------------------------


class TestPush:
    @pytest.mark.asyncio
    async def test_push_mounts_row(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            host = _host(app)
            host.push("hello world")
            await pilot.pause()
            rows = list(host.query("Static.toast"))
            assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_severity_classes_on_row(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            host = _host(app)
            host.push("oops", severity="error", ttl=0)
            await pilot.pause()
            rows = list(host.query("Static.toast"))
            assert len(rows) == 1
            assert "severity-error" in rows[0].classes

    @pytest.mark.asyncio
    async def test_history_captures_pushes(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            host = _host(app)
            host.push("first")
            host.push("second", severity="warning")
            await pilot.pause()
            assert len(host.history) == 2
            assert host.history[0].message == "first"
            assert host.history[1].severity == "warning"

    @pytest.mark.asyncio
    async def test_history_bounded(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            host = _host(app)
            for i in range(40):
                host.push(f"toast-{i}", ttl=0)
            await pilot.pause()
            # Bound at 32.
            assert len(host.history) == 32
            assert host.history[-1].message == "toast-39"


# ---------------------------------------------------------------------------
# Auto-dismiss
# ---------------------------------------------------------------------------


class TestDismiss:
    @pytest.mark.asyncio
    async def test_ttl_auto_dismisses(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            host = _host(app)
            host.push("flash", ttl=0.05)
            await pilot.pause()
            # Wait long enough for the timer to fire.
            await pilot.pause(0.2)
            for _ in range(4):
                await pilot.pause()
            rows = list(host.query("Static.toast"))
            assert rows == []

    @pytest.mark.asyncio
    async def test_ttl_zero_persists(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            host = _host(app)
            host.push("sticky", ttl=0)
            await pilot.pause()
            await pilot.pause(0.1)
            await pilot.pause()
            rows = list(host.query("Static.toast"))
            assert len(rows) == 1


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestFormat:
    def test_format_row_with_message(self):
        toast = Toast(message="hi", severity="warning")
        assert ToastHost._format_row(toast) == "⚠ hi"

    def test_format_row_blank_falls_back_to_badge(self):
        toast = Toast(message="", severity="success")
        assert ToastHost._format_row(toast) == "✓"


# ---------------------------------------------------------------------------
# CareApp integration
# ---------------------------------------------------------------------------


class TestAppIntegration:
    @pytest.mark.asyncio
    async def test_care_app_mounts_host_and_routes_push(self):
        from care.app import CareApp

        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.push_toast("from app", severity="success", ttl=0)
            await pilot.pause()
            host = app.query_one(ToastHost)
            assert host.history[-1].message == "from app"
            assert host.history[-1].severity == "success"


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_widgets_re_exports(self):
        from care.widgets import Toast as T
        from care.widgets import ToastHost as H

        assert T is Toast
        assert H is ToastHost
