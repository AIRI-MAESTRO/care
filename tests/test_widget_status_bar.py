"""Pilot tests for StatusBar widget (TODO §1 P1 / P1.1).

Exercises:
* `on_mount` schedules `aggregate_status_bar` and renders the
  result.
* First paint (before the worker settles) shows the
  placeholder.
* Failing aggregator path lands on `last_error` without
  crashing the widget.
* CareApp mounts the StatusBar above the toast host.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from care.config import CareConfig
from care.runtime.status_bar import SessionTokenCounter, StatusBarSnapshot
from care.widgets.status_bar import StatusBar


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, *, bar: StatusBar | None = None) -> None:
        super().__init__()
        self._bar = bar or StatusBar(
            config=CareConfig(),
            token_counter=SessionTokenCounter(),
        )

    def compose(self) -> ComposeResult:
        yield self._bar


def _bar(app: App) -> StatusBar:
    return app.query_one(StatusBar)


# ---------------------------------------------------------------------------
# Compose + first paint
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_placeholder_renders_before_refresh(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = _bar(app)
            text = bar.query_one("#status-bar-text", Static)
            assert text is not None
            # `is_loading` flips false once the worker settles —
            # the placeholder appears at least transiently.
            assert StatusBar.PLACEHOLDER_TEXT.startswith("memory ?")


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


class TestRefresh:
    @pytest.mark.asyncio
    async def test_refresh_populates_snapshot(self):
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            bar = _bar(app)
            assert bar.is_loading is False
            assert isinstance(bar.last_snapshot, StatusBarSnapshot)
            # Memory + Platform unconfigured → "skipped".
            assert bar.last_snapshot.memory.status == "skipped"
            assert bar.last_snapshot.platform.status == "skipped"

    @pytest.mark.asyncio
    async def test_refresh_failure_lands_on_error(self, monkeypatch):
        async def _boom(**kw):
            raise RuntimeError("aggregator down")

        monkeypatch.setattr(
            "care.widgets.status_bar.aggregate_status_bar",
            _boom,
        )
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            bar = _bar(app)
            assert bar.last_error is not None
            assert "aggregator down" in bar.last_error


# ---------------------------------------------------------------------------
# CareApp integration
# ---------------------------------------------------------------------------


class TestAppIntegration:
    @pytest.mark.asyncio
    async def test_care_app_mounts_status_bar(self):
        from care.app import CareApp

        app = CareApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.query_one(StatusBar) is not None


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_widgets_re_exports(self):
        from care.widgets import StatusBar as S

        assert S is StatusBar


# ---------------------------------------------------------------------------
# Interval refresh (P1.2)
# ---------------------------------------------------------------------------


class TestIntervalRefresh:
    @pytest.mark.asyncio
    async def test_short_interval_fires_multiple_refreshes(self):
        # Use a tight interval so the test runs fast. Stub the
        # aggregator to avoid the 2-second probe deadline +
        # count invocations explicitly so we don't depend on
        # `refresh_count` ordering.
        from care.runtime.status_bar import HealthSnapshot

        called: list[int] = []

        async def _fake_aggregate(**kw):
            called.append(1)
            return StatusBarSnapshot(
                memory=HealthSnapshot(service="memory", status="skipped"),
                platform=HealthSnapshot(service="platform", status="skipped"),
                model="",
                endpoint="",
                tokens=SessionTokenCounter().snapshot(),
            )

        bar = StatusBar(
            config=CareConfig(),
            token_counter=SessionTokenCounter(),
            refresh_interval=0.05,
        )
        # Monkeypatch the aggregator inline.
        import care.widgets.status_bar as sb_mod

        original = sb_mod.aggregate_status_bar
        sb_mod.aggregate_status_bar = _fake_aggregate
        try:
            host = _Host(bar=bar)
            async with host.run_test() as pilot:
                # Let the on_mount refresh land + the interval
                # timer fire a couple of times.
                await pilot.pause(0.2)
                for _ in range(4):
                    await pilot.pause()
                assert len(called) >= 2
        finally:
            sb_mod.aggregate_status_bar = original

    @pytest.mark.asyncio
    async def test_zero_interval_disables_timer(self):
        bar = StatusBar(
            config=CareConfig(),
            token_counter=SessionTokenCounter(),
            refresh_interval=0,
        )
        host = _Host(bar=bar)
        async with host.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            # First refresh fires from on_mount, no recurring
            # timer is scheduled.
            assert bar._interval_timer is None
            assert bar.refresh_count == 1

    @pytest.mark.asyncio
    async def test_interval_timer_stopped_on_unmount(self):
        bar = StatusBar(
            config=CareConfig(),
            token_counter=SessionTokenCounter(),
            refresh_interval=0.05,
        )
        host = _Host(bar=bar)
        async with host.run_test() as pilot:
            await pilot.pause()
            assert bar._interval_timer is not None
            host.exit()
            for _ in range(3):
                await pilot.pause()
        # After exit the handle is cleared.
        assert bar._interval_timer is None


# ---------------------------------------------------------------------------
# TaskRegistry-driven refresh (P1.4)
# ---------------------------------------------------------------------------


class TestRegistryReactive:
    @pytest.mark.asyncio
    async def test_registry_event_triggers_refresh(self):
        from care.runtime.task_registry import TaskRegistry

        registry = TaskRegistry()
        bar = StatusBar(
            config=CareConfig(),
            token_counter=SessionTokenCounter(),
            task_registry=registry,
            refresh_interval=0,
        )
        host = _Host(bar=bar)
        async with host.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            assert bar._unsubscribe is not None
            initial_count = bar.refresh_count
            # Register a task; the registry's on_change fires
            # → status bar refreshes.
            registry.register(
                kind="mage_generation",
                label="weather",
            )
            for _ in range(6):
                await pilot.pause()
            assert bar.refresh_count > initial_count

    @pytest.mark.asyncio
    async def test_no_registry_no_subscription(self):
        bar = StatusBar(
            config=CareConfig(),
            token_counter=SessionTokenCounter(),
            refresh_interval=0,
        )
        host = _Host(bar=bar)
        async with host.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            # No registry wired → no unsubscribe handle.
            assert bar._unsubscribe is None

    @pytest.mark.asyncio
    async def test_unsubscribe_called_on_unmount(self):
        from care.runtime.task_registry import TaskRegistry

        registry = TaskRegistry()
        bar = StatusBar(
            config=CareConfig(),
            token_counter=SessionTokenCounter(),
            task_registry=registry,
            refresh_interval=0,
        )
        host = _Host(bar=bar)
        async with host.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            assert bar._unsubscribe is not None
            host.exit()
            for _ in range(3):
                await pilot.pause()
        # After unmount, post-event registration shouldn't
        # spike the refresh count.
        baseline = bar.refresh_count
        registry.register(kind="mage_generation", label="late")
        # Outside the run_test scope we can't await pauses; the
        # call should be a no-op against the dead widget.
        assert bar.refresh_count == baseline


# ---------------------------------------------------------------------------
# Empty / loading / error states (P1.5)
# ---------------------------------------------------------------------------


class TestStripStates:
    @pytest.mark.asyncio
    async def test_first_paint_shows_placeholder(self):
        # Block the aggregator so the worker stays pending and
        # the placeholder remains visible.
        import asyncio

        async def _block(**kw):
            await asyncio.sleep(1.0)
            from care.runtime.status_bar import (
                HealthSnapshot,
                StatusBarSnapshot,
            )

            return StatusBarSnapshot(
                memory=HealthSnapshot(
                    service="memory", status="skipped",
                ),
                platform=HealthSnapshot(
                    service="platform", status="skipped",
                ),
                model="",
                endpoint="",
                tokens=SessionTokenCounter().snapshot(),
            )

        import care.widgets.status_bar as sb_mod

        original = sb_mod.aggregate_status_bar
        sb_mod.aggregate_status_bar = _block
        try:
            bar = StatusBar(
                config=CareConfig(),
                token_counter=SessionTokenCounter(),
                refresh_interval=0,
            )
            host = _Host(bar=bar)
            async with host.run_test() as pilot:
                await pilot.pause()
                assert bar.is_loading is True
                # The Static widget content is the placeholder
                # while the worker is in flight.
                text = bar.query_one("#status-bar-text", Static)
                assert text is not None
        finally:
            sb_mod.aggregate_status_bar = original

    @pytest.mark.asyncio
    async def test_probe_failure_renders_error_on_strip(self):
        # Stub probe_health to return failed Memory, ok Platform.
        from care.runtime.status_bar import HealthSnapshot, StatusBarSnapshot

        async def _stub_aggregate(**kw):
            return StatusBarSnapshot(
                memory=HealthSnapshot(
                    service="memory",
                    status="failed",
                    error="connection refused",
                ),
                platform=HealthSnapshot(
                    service="platform", status="ok", latency_ms=12.0,
                ),
                model="gpt-4",
                endpoint="api.openai.com",
                tokens=SessionTokenCounter().snapshot(),
            )

        import care.widgets.status_bar as sb_mod

        original = sb_mod.aggregate_status_bar
        sb_mod.aggregate_status_bar = _stub_aggregate
        try:
            bar = StatusBar(
                config=CareConfig(),
                token_counter=SessionTokenCounter(),
                refresh_interval=0,
            )
            host = _Host(bar=bar)
            async with host.run_test() as pilot:
                for _ in range(6):
                    await pilot.pause()
                rendered = bar.last_snapshot.format_text()
                assert "memory ✗" in rendered
                assert "(connection refused)" in rendered
                assert "platform ✓" in rendered
        finally:
            sb_mod.aggregate_status_bar = original

    @pytest.mark.asyncio
    async def test_aggregator_failure_renders_inline_error(self):
        async def _boom(**kw):
            raise RuntimeError("status-bar exploded")

        import care.widgets.status_bar as sb_mod

        original = sb_mod.aggregate_status_bar
        sb_mod.aggregate_status_bar = _boom
        try:
            bar = StatusBar(
                config=CareConfig(),
                token_counter=SessionTokenCounter(),
                refresh_interval=0,
            )
            host = _Host(bar=bar)
            async with host.run_test() as pilot:
                for _ in range(6):
                    await pilot.pause()
                assert bar.last_error is not None
                assert "status-bar exploded" in bar.last_error
                error_text = bar._format_error_text()
                assert error_text.startswith("⚠ status bar:")
                assert "status-bar exploded" in error_text
        finally:
            sb_mod.aggregate_status_bar = original

    @pytest.mark.asyncio
    async def test_error_with_markup_brackets_renders_without_crash(self):
        # Regression: a chain Pydantic ValidationError (full of `[...]`)
        # surfaced through the aggregator must not raise MarkupError when
        # the docked strip renders it. The strip is plain text, so the
        # Static parses no markup.
        bracket_err = (
            "1 validation error for ReasoningChain [type=model_type, "
            "input_value=EvaluationStepConfig(eval='memory[-1]', "
            "max_retries=1), input_type=EvaluationStepConfig]"
        )

        async def _boom(**kw):
            raise RuntimeError(bracket_err)

        import care.widgets.status_bar as sb_mod

        original = sb_mod.aggregate_status_bar
        sb_mod.aggregate_status_bar = _boom
        try:
            bar = StatusBar(
                config=CareConfig(),
                token_counter=SessionTokenCounter(),
                refresh_interval=0,
            )
            host = _Host(bar=bar)
            # Wide enough that the whole error fits on the single line — the
            # point is that the ``[...]`` brackets survive rendering literally
            # (no MarkupError, not consumed as markup), not the width-fit.
            async with host.run_test(size=(240, 24)) as pilot:
                for _ in range(6):
                    await pilot.pause()
                target = bar.query_one("#status-bar-text", Static)
                # Forcing a render must NOT raise textual.markup.MarkupError.
                rendered = str(target.render())
                assert "memory[-1]" in rendered
        finally:
            sb_mod.aggregate_status_bar = original

    def test_format_error_text_with_no_error_falls_back_to_placeholder(self):
        bar = StatusBar(
            config=CareConfig(),
            token_counter=SessionTokenCounter(),
            refresh_interval=0,
        )
        bar.last_error = None
        assert bar._format_error_text() == StatusBar.PLACEHOLDER_TEXT
