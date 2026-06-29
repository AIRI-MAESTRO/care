"""Tests for the status-bar data layer (TODO §1 P1).

The Textual widget is gated on TODO §1 P0 multi-screen workflow,
but the data layer is independent and well-bounded. These tests
pin the contract the future widget will rely on.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from care.config import CareConfig, MageConfig
from care.runtime.status_bar import (
    HealthSnapshot,
    SessionTokenCounter,
    SessionTokenTotals,
    StatusBarSnapshot,
    aggregate_status_bar,
    derive_from_task_registry,
    probe_health,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubFacade:
    """Mimics ``CareMemory`` / ``CarePlatform`` — just the
    ``health_check`` method the probes call."""

    def __init__(self, *, response=None, exc=None, delay=0.0):
        self._response = response if response is not None else {"status": "ok"}
        self._exc = exc
        self._delay = delay
        self.calls = 0

    def health_check(self):
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        if self._exc:
            raise self._exc
        return self._response


def _config(**mage_overrides) -> CareConfig:
    mage = MageConfig(
        api_key=mage_overrides.pop("api_key", "sk-test"),
        base_url=mage_overrides.pop(
            "base_url", "https://api.openai.com/v1",
        ),
        model=mage_overrides.pop("model", "gpt-4o-mini"),
        **mage_overrides,
    )
    return CareConfig(mage=mage)


# ---------------------------------------------------------------------------
# SessionTokenCounter
# ---------------------------------------------------------------------------


class TestSessionTokenCounter:
    def test_starts_at_zero(self):
        counter = SessionTokenCounter()
        snap = counter.snapshot()
        assert snap == SessionTokenTotals()
        assert snap.total == 0
        assert snap.calls == 0

    def test_add_basic_usage(self):
        counter = SessionTokenCounter()
        counter.add({"prompt": 100, "completion": 50, "total": 150})
        snap = counter.snapshot()
        assert snap.prompt == 100
        assert snap.completion == 50
        assert snap.total == 150
        assert snap.calls == 1

    def test_add_accumulates(self):
        counter = SessionTokenCounter()
        counter.add({"prompt": 10, "completion": 5, "total": 15})
        counter.add({"prompt": 20, "completion": 8, "total": 28})
        snap = counter.snapshot()
        assert snap.prompt == 30
        assert snap.completion == 13
        assert snap.total == 43
        assert snap.calls == 2

    def test_add_none_is_noop(self):
        counter = SessionTokenCounter()
        counter.add(None)
        counter.add({})
        assert counter.snapshot() == SessionTokenTotals()

    def test_missing_total_derived_from_split(self):
        # Some providers send prompt+completion without total.
        counter = SessionTokenCounter()
        counter.add({"prompt": 7, "completion": 3})
        snap = counter.snapshot()
        assert snap.total == 10
        assert snap.calls == 1

    def test_non_int_values_coerced(self):
        counter = SessionTokenCounter()
        counter.add({"prompt": "12", "completion": "8", "total": "20"})
        assert counter.snapshot().total == 20

    def test_unparseable_values_count_as_zero_but_bump_calls(self):
        counter = SessionTokenCounter()
        counter.add({"prompt": "n/a", "completion": None, "total": "?"})
        snap = counter.snapshot()
        assert snap.total == 0
        # Call still recorded so the user can see the empty round-trip.
        assert snap.calls == 1

    def test_reset(self):
        counter = SessionTokenCounter()
        counter.add({"prompt": 1, "completion": 1, "total": 2})
        counter.reset()
        assert counter.snapshot() == SessionTokenTotals()

    def test_snapshot_is_frozen(self):
        snap = SessionTokenCounter().snapshot()
        with pytest.raises(Exception):
            snap.total = 99  # type: ignore[misc]

    def test_concurrent_add_no_loss(self):
        counter = SessionTokenCounter()

        def hammer():
            for _ in range(100):
                counter.add({"prompt": 1, "completion": 1, "total": 2})

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = counter.snapshot()
        assert snap.calls == 8 * 100
        assert snap.total == 8 * 100 * 2


# ---------------------------------------------------------------------------
# HealthSnapshot
# ---------------------------------------------------------------------------


class TestHealthSnapshot:
    def test_age_seconds_when_never_probed(self):
        snap = HealthSnapshot(service="memory")
        assert snap.age_seconds() is None

    def test_age_seconds_computed_against_now(self):
        snap = HealthSnapshot(
            service="memory", status="ok", checked_at=1000.0
        )
        assert snap.age_seconds(now=1005.0) == 5.0
        # Clock skew (now < checked_at) collapses to 0 rather
        # than going negative.
        assert snap.age_seconds(now=999.0) == 0.0


# ---------------------------------------------------------------------------
# probe_health
# ---------------------------------------------------------------------------


class TestProbeHealth:
    def test_skipped_when_facade_none(self):
        snap = asyncio.run(probe_health(service="memory", facade=None))
        assert snap.status == "skipped"
        assert snap.service == "memory"
        assert snap.checked_at is not None
        assert "not configured" in snap.error

    def test_ok_status_carries_latency_and_detail(self):
        facade = _StubFacade(response={"status": "healthy", "db": "ok"})
        snap = asyncio.run(probe_health(service="memory", facade=facade))
        assert snap.status == "ok"
        assert snap.latency_ms is not None and snap.latency_ms >= 0
        assert snap.detail == {"status": "healthy", "db": "ok"}
        assert snap.checked_at is not None
        assert facade.calls == 1

    def test_non_dict_response_wrapped_as_raw(self):
        facade = _StubFacade(response="alive")
        snap = asyncio.run(probe_health(service="platform", facade=facade))
        assert snap.status == "ok"
        assert snap.detail == {"raw": "alive"}

    def test_exception_wrapped_as_failed(self):
        facade = _StubFacade(exc=ConnectionError("refused"))
        snap = asyncio.run(probe_health(service="memory", facade=facade))
        assert snap.status == "failed"
        assert "ConnectionError" in snap.error
        assert "refused" in snap.error
        assert snap.latency_ms is not None

    def test_timeout_flips_to_failed(self):
        facade = _StubFacade(response={"ok": True}, delay=0.5)
        snap = asyncio.run(
            probe_health(service="memory", facade=facade, timeout=0.05)
        )
        assert snap.status == "failed"
        assert "timed out" in snap.error


# ---------------------------------------------------------------------------
# aggregate_status_bar
# ---------------------------------------------------------------------------


class TestProbeMageHealth:
    """§1 P0 — MAGE health joins memory + platform as the
    third status-bar dot. The probe is config-only (no LLM
    call) so it's safe to run on every 5s refresh."""

    def test_ok_when_api_key_and_base_url_set(self):
        from care.runtime.status_bar import probe_mage_health

        snap = asyncio.run(probe_mage_health(config=_config()))
        assert snap.service == "mage"
        assert snap.status == "ok"
        assert snap.detail.get("model") == "gpt-4o-mini"
        assert "openai.com" in snap.detail.get("base_url", "")

    def test_skipped_when_api_key_missing(self):
        from care.runtime.status_bar import probe_mage_health

        snap = asyncio.run(
            probe_mage_health(config=_config(api_key="")),
        )
        assert snap.status == "skipped"
        assert "api_key" in snap.error

    def test_skipped_when_base_url_missing(self):
        from care.runtime.status_bar import probe_mage_health

        snap = asyncio.run(
            probe_mage_health(config=_config(base_url="")),
        )
        assert snap.status == "skipped"
        assert "base_url" in snap.error

    def test_skipped_when_config_is_none(self):
        from care.runtime.status_bar import probe_mage_health

        snap = asyncio.run(probe_mage_health(config=None))
        assert snap.status == "skipped"
        assert "config not loaded" in snap.error

    def test_aggregator_includes_mage_snapshot(self):
        snapshot = asyncio.run(aggregate_status_bar(config=_config()))
        assert snapshot.mage.service == "mage"
        assert snapshot.mage.status == "ok"


class TestAggregateStatusBar:
    def test_full_snapshot_assembles_every_field(self):
        memory = _StubFacade(response={"status": "healthy"})
        platform = _StubFacade(response={"status": "healthy"})
        counter = SessionTokenCounter()
        counter.add({"prompt": 100, "completion": 50, "total": 150})

        class _Task:
            id = "run-abc-1234567890"
            label = "weather report"

        snapshot = asyncio.run(
            aggregate_status_bar(
                config=_config(),
                memory=memory,
                platform=platform,
                token_counter=counter,
                active_task=_Task(),
            )
        )
        assert isinstance(snapshot, StatusBarSnapshot)
        assert snapshot.memory.status == "ok"
        assert snapshot.platform.status == "ok"
        assert snapshot.model == "gpt-4o-mini"
        assert snapshot.endpoint == "api.openai.com"
        assert snapshot.tokens.total == 150
        assert snapshot.tokens.calls == 1
        assert snapshot.active_run_id == "run-abc-1234567890"
        assert snapshot.active_run_label == "weather report"
        assert snapshot.has_active_run is True

    def test_skipped_when_facades_missing(self):
        snapshot = asyncio.run(aggregate_status_bar(config=_config()))
        assert snapshot.memory.status == "skipped"
        assert snapshot.platform.status == "skipped"
        assert snapshot.tokens == SessionTokenTotals()
        assert snapshot.has_active_run is False
        assert snapshot.active_run_id is None

    def test_dict_active_task_accepted(self):
        # The aggregator duck-types on the active-task arg.
        snapshot = asyncio.run(
            aggregate_status_bar(
                config=_config(),
                active_task={"id": "run-7", "label": "load library"},
            )
        )
        assert snapshot.active_run_id == "run-7"
        assert snapshot.active_run_label == "load library"

    def test_partial_failure(self):
        # Memory works, Platform fails — both rows still surface.
        memory = _StubFacade(response={"status": "healthy"})
        platform = _StubFacade(exc=RuntimeError("503"))
        snapshot = asyncio.run(
            aggregate_status_bar(
                config=_config(),
                memory=memory,
                platform=platform,
            )
        )
        assert snapshot.memory.status == "ok"
        assert snapshot.platform.status == "failed"
        assert "503" in snapshot.platform.error

    def test_probes_run_concurrently(self):
        # If probes were serial the wall-clock would be ~0.2s
        # (2 × 0.1). Run with a 2s timeout and verify both
        # complete in well under the serial floor.
        memory = _StubFacade(response={"ok": True}, delay=0.1)
        platform = _StubFacade(response={"ok": True}, delay=0.1)
        start = time.monotonic()
        snapshot = asyncio.run(
            aggregate_status_bar(
                config=_config(),
                memory=memory,
                platform=platform,
                timeout=2.0,
            )
        )
        elapsed = time.monotonic() - start
        # Both probes ran (sanity check).
        assert snapshot.memory.status == "ok"
        assert snapshot.platform.status == "ok"
        # Concurrent execution should be well under the serial
        # floor (0.2s); leave headroom for slow CI.
        assert elapsed < 0.18, f"probes appear to be serial: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def _snap(self, **overrides):
        defaults = dict(
            memory=HealthSnapshot(
                service="memory",
                status="ok",
                latency_ms=12.3,
                checked_at=1000.0,
            ),
            platform=HealthSnapshot(
                service="platform",
                status="ok",
                latency_ms=8.7,
                checked_at=1000.0,
            ),
            model="gpt-4o-mini",
            endpoint="api.openai.com",
            tokens=SessionTokenTotals(prompt=100, completion=50, total=150, calls=2),
            active_run_id=None,
            active_run_label=None,
            captured_at=1005.0,
        )
        defaults.update(overrides)
        return StatusBarSnapshot(**defaults)

    def test_happy_path(self):
        text = self._snap().format_text(now=1005.0)
        assert "memory ✓" in text
        assert "platform ✓" in text
        assert "5s ago" in text  # captured at 1000, now 1005
        assert "gpt-4o-mini @ api.openai.com" in text
        assert "150 tok" in text

    def test_failed_service_shows_error(self):
        snap = self._snap(
            memory=HealthSnapshot(
                service="memory",
                status="failed",
                error="ConnectionError: refused",
                checked_at=1000.0,
            )
        )
        text = snap.format_text(now=1005.0)
        assert "memory ✗" in text
        assert "ConnectionError" in text

    def test_active_run_shows_short_id(self):
        snap = self._snap(
            active_run_id="run-abc-1234567890",
            active_run_label="weather",
        )
        text = snap.format_text(now=1005.0)
        # First 8 chars of the id.
        assert "run run-abc-" in text
        assert "weather" in text

    def test_tokens_formatted_for_thousands(self):
        snap = self._snap(
            tokens=SessionTokenTotals(total=12_500, calls=10)
        )
        text = snap.format_text(now=1005.0)
        assert "12.5k tok" in text

    def test_tokens_formatted_for_millions(self):
        snap = self._snap(
            tokens=SessionTokenTotals(total=2_300_000, calls=300)
        )
        text = snap.format_text(now=1005.0)
        assert "2.3M tok" in text

    def test_zero_tokens_omitted(self):
        snap = self._snap(tokens=SessionTokenTotals())
        assert "tok" not in snap.format_text(now=1005.0)

    def test_skipped_service_no_age_suffix(self):
        snap = self._snap(
            platform=HealthSnapshot(
                service="platform",
                status="skipped",
                error="platform.base_url is empty",
                checked_at=1000.0,
            )
        )
        text = snap.format_text(now=1005.0)
        assert "platform ·" in text
        # No age suffix on non-ok rows.
        assert "platform · (5s ago)" not in text


# ---------------------------------------------------------------------------
# derive_from_task_registry
# ---------------------------------------------------------------------------


class _FakeTask:
    def __init__(self, task_id, label, status, started_at=None):
        self.id = task_id
        self.label = label
        self.status = status
        self.started_at = started_at


class _FakeRegistry:
    def __init__(self, tasks):
        self._tasks = tasks

    def list_tasks(self, *, status=None, **_):
        return [t for t in self._tasks if status is None or t.status == status]


class TestDeriveFromTaskRegistry:
    def test_none_registry_returns_none(self):
        assert derive_from_task_registry(None) is None

    def test_no_active_returns_none(self):
        registry = _FakeRegistry([])
        assert derive_from_task_registry(registry) is None

    def test_running_preferred_over_pending(self):
        pending = _FakeTask("p", "pending one", "pending")
        running = _FakeTask("r", "running one", "running", started_at=10.0)
        result = derive_from_task_registry(_FakeRegistry([pending, running]))
        assert result is running

    def test_most_recently_started_running_wins(self):
        older = _FakeTask("a", "old", "running", started_at=10.0)
        newer = _FakeTask("b", "new", "running", started_at=20.0)
        result = derive_from_task_registry(_FakeRegistry([older, newer]))
        assert result is newer

    def test_pending_when_no_running(self):
        pending = _FakeTask("p", "pending one", "pending")
        result = derive_from_task_registry(_FakeRegistry([pending]))
        assert result is pending

    def test_registry_without_list_tasks(self):
        assert derive_from_task_registry(object()) is None


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports_module(self):
        # Importing through the top-level entry point matches how
        # the future widget will pull this in.
        from care.runtime import (
            SessionTokenCounter as ReExported,
            StatusBarSnapshot as Snap,
            aggregate_status_bar as agg,
        )

        assert ReExported is SessionTokenCounter
        assert Snap is StatusBarSnapshot
        assert agg is aggregate_status_bar
