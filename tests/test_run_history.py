"""Tests for the run-history data layer (TODO §3 P1 — Run history tab).

The Textual tab is gated on §1 P0 multi-screen workflow; this
suite pins the projection contract the tab will rely on.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from care.runtime.run_history import (
    RunHistoryEntry,
    RunHistoryError,
    RunHistorySummary,
    fetch_run_history,
    parse_run_history_entry,
    summarize_run_history,
)


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------


def _card(
    *,
    card_id: str = "card-1",
    agent_entity_id: str = "agent-1",
    run_id: str = "run-20260519T120000000000",
    status_label: str = "success",
    finished_at: str = "2026-05-19T12:00:00+00:00",
    duration_seconds: float = 12.5,
    step_count: int = 4,
    total_tokens: int | None = 1234,
    error_message: str | None = None,
    task: str = "Summarise the PDF",
    description: str = "Run of agent demo — success",
    extra_tags: list[str] | None = None,
    category: str = "agent_run",
) -> dict:
    metrics: dict = {
        "duration_seconds": duration_seconds,
        "step_count": step_count,
        "exit_status": status_label,
    }
    if total_tokens is not None:
        metrics["total_tokens"] = total_tokens
    if error_message is not None:
        metrics["error_message"] = error_message
    tags = ["agent_run", f"agent:{agent_entity_id}", f"status:{status_label}"]
    if extra_tags:
        tags.extend(extra_tags)
    return {
        "entity_type": "memory_card",
        "entity_id": card_id,
        "version_id": "v-1",
        "channel": "latest",
        "etag": "etag",
        "meta": {"tags": tags},
        "content": {
            "category": category,
            "task_description": task,
            "description": description,
            "keywords": ["agent_run", f"agent:{agent_entity_id}", status_label],
            "usage": {
                "run_id": run_id,
                "agent_entity_id": agent_entity_id,
                "agent_name": "demo",
                "finished_at": finished_at,
                "metrics": metrics,
            },
        },
    }


# ---------------------------------------------------------------------------
# parse_run_history_entry
# ---------------------------------------------------------------------------


class TestParseRunHistoryEntry:
    def test_happy_path(self):
        entry = parse_run_history_entry(_card())
        assert entry is not None
        assert entry.card_id == "card-1"
        assert entry.agent_entity_id == "agent-1"
        assert entry.run_id == "run-20260519T120000000000"
        assert entry.status == "success"
        assert entry.success is True
        assert entry.duration_seconds == 12.5
        assert entry.step_count == 4
        assert entry.total_tokens == 1234
        assert entry.error_message is None
        assert entry.task_description == "Summarise the PDF"
        assert "agent_run" in entry.tags
        assert "agent:agent-1" in entry.tags

    def test_finished_at_parsed_as_datetime(self):
        entry = parse_run_history_entry(_card())
        assert isinstance(entry.finished_at, datetime)
        assert entry.finished_at.year == 2026
        assert entry.finished_at.tzinfo is not None

    def test_failed_run_status(self):
        card = _card(status_label="failure", error_message="step 3 crashed")
        entry = parse_run_history_entry(card)
        assert entry.status == "failure"
        assert entry.success is False
        assert entry.error_message == "step 3 crashed"

    def test_status_from_tag_wins_over_metrics(self):
        # Tag says failure, metrics say success — tag wins
        # because the tag is the authoritative declaration the
        # writer stamps.
        card = _card(status_label="failure")
        # Tamper the metrics to ensure tag-first precedence:
        card["content"]["usage"]["metrics"]["exit_status"] = "success"
        entry = parse_run_history_entry(card)
        assert entry.status == "failure"

    def test_wrong_category_returns_none(self):
        card = _card(category="lesson_learned")
        assert parse_run_history_entry(card) is None

    def test_missing_run_id_returns_none(self):
        card = _card()
        card["content"]["usage"].pop("run_id")
        assert parse_run_history_entry(card) is None

    def test_filter_by_agent_id(self):
        # Card is for agent-1; filter for agent-2 → None.
        card = _card(agent_entity_id="agent-1")
        assert parse_run_history_entry(card, agent_entity_id="agent-2") is None
        # Matching id passes through.
        assert parse_run_history_entry(card, agent_entity_id="agent-1") is not None

    def test_missing_metrics_collapses_to_none_not_crash(self):
        card = _card()
        card["content"]["usage"]["metrics"] = {}
        entry = parse_run_history_entry(card)
        assert entry.duration_seconds is None
        assert entry.step_count is None
        assert entry.total_tokens is None

    def test_invalid_finished_at_collapses_to_none(self):
        card = _card(finished_at="not a date")
        entry = parse_run_history_entry(card)
        assert entry.finished_at is None

    def test_non_dict_content_returns_none(self):
        card = _card()
        card["content"] = "not a dict"
        assert parse_run_history_entry(card) is None

    def test_format_one_line(self):
        entry = parse_run_history_entry(_card())
        line = entry.format_one_line()
        assert "✓" in line
        assert "2026-05-19" in line
        assert "12.5s" in line
        assert "1234 tok" in line

    def test_format_one_line_failure_carries_error(self):
        card = _card(status_label="failure", error_message="boom!")
        entry = parse_run_history_entry(card)
        line = entry.format_one_line()
        assert "✗" in line
        assert "boom!" in line


# ---------------------------------------------------------------------------
# RunHistoryEntry / RunHistorySummary shape
# ---------------------------------------------------------------------------


class TestModelShape:
    def test_entry_is_frozen(self):
        entry = RunHistoryEntry(
            card_id="c", agent_entity_id="a", run_id="r"
        )
        with pytest.raises(FrozenInstanceError):
            entry.card_id = "x"  # type: ignore[misc]

    def test_summary_is_frozen(self):
        summary = RunHistorySummary()
        with pytest.raises(FrozenInstanceError):
            summary.total_runs = 1  # type: ignore[misc]

    def test_empty_summary_predicates(self):
        s = RunHistorySummary()
        assert s.success_rate is None
        assert s.avg_duration_seconds is None


# ---------------------------------------------------------------------------
# summarize_run_history
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_empty_iterable(self):
        s = summarize_run_history([])
        assert s.total_runs == 0
        assert s.success_count == 0
        assert s.failure_count == 0
        assert s.total_tokens == 0
        assert s.success_rate is None

    def test_aggregates(self):
        entries = [
            parse_run_history_entry(
                _card(
                    card_id="c1",
                    run_id="r1",
                    finished_at="2026-05-19T12:00:00+00:00",
                    duration_seconds=10.0,
                    total_tokens=100,
                )
            ),
            parse_run_history_entry(
                _card(
                    card_id="c2",
                    run_id="r2",
                    status_label="failure",
                    error_message="boom",
                    finished_at="2026-05-19T13:00:00+00:00",
                    duration_seconds=5.0,
                    total_tokens=50,
                )
            ),
            parse_run_history_entry(
                _card(
                    card_id="c3",
                    run_id="r3",
                    finished_at="2026-05-19T14:00:00+00:00",
                    duration_seconds=20.0,
                    total_tokens=200,
                )
            ),
        ]
        summary = summarize_run_history(entries)
        assert summary.total_runs == 3
        assert summary.success_count == 2
        assert summary.failure_count == 1
        assert summary.total_tokens == 350
        assert summary.total_duration_seconds == 35.0
        assert summary.success_rate == pytest.approx(2 / 3)
        assert summary.avg_duration_seconds == pytest.approx(35.0 / 3)
        # last_success_at picks the most-recent success (c3 @ 14:00).
        assert summary.last_success_at.hour == 14
        # last_failure_at points at c2 @ 13:00.
        assert summary.last_failure_at.hour == 13

    def test_ignores_missing_duration_and_tokens(self):
        e = parse_run_history_entry(_card(total_tokens=None))
        e_no_dur = RunHistoryEntry(
            card_id="x", agent_entity_id="a", run_id="r",
            status="success",
        )
        summary = summarize_run_history([e, e_no_dur])
        assert summary.total_runs == 2
        # First entry has 12.5s; second has None.
        assert summary.total_duration_seconds == 12.5
        # First has no tokens; second has no tokens.
        assert summary.total_tokens == 0


# ---------------------------------------------------------------------------
# fetch_run_history
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(self, *, rows=None, exc=None, delay=0.0):
        self._rows = rows if rows is not None else []
        self._exc = exc
        self._delay = delay
        self.calls: list[dict] = []

    def _list_entities(self, entity_type, *, limit, channel, tags, namespace):
        self.calls.append(
            {
                "entity_type": entity_type,
                "limit": limit,
                "channel": channel,
                "tags": tags,
                "namespace": namespace,
            }
        )
        if self._delay:
            time.sleep(self._delay)
        if self._exc:
            raise self._exc
        return self._rows


class _StubMemory:
    def __init__(self, client):
        self.client = client


class TestFetchRunHistory:
    def test_happy_path(self):
        rows = [
            _card(card_id="c1", run_id="r1", finished_at="2026-05-19T12:00:00+00:00"),
            _card(
                card_id="c2",
                run_id="r2",
                status_label="failure",
                finished_at="2026-05-19T13:00:00+00:00",
            ),
        ]
        memory = _StubMemory(_StubClient(rows=rows))
        entries = asyncio.run(fetch_run_history(memory, "agent-1"))
        assert len(entries) == 2
        # Sorted desc by finished_at — c2 (13:00) is first.
        assert entries[0].run_id == "r2"
        assert entries[1].run_id == "r1"
        # Verify the SDK call shape.
        call = memory.client.calls[0]
        assert call["entity_type"] == "memory_card"
        assert call["tags"] == ["agent_run", "agent:agent-1"]
        assert call["channel"] == "latest"

    def test_forwards_namespace_channel_limit(self):
        memory = _StubMemory(_StubClient(rows=[]))
        asyncio.run(
            fetch_run_history(
                memory,
                "agent-1",
                namespace="alice",
                channel="stable",
                limit=50,
            )
        )
        call = memory.client.calls[0]
        assert call["namespace"] == "alice"
        assert call["channel"] == "stable"
        assert call["limit"] == 50

    def test_limit_clamped(self):
        memory = _StubMemory(_StubClient(rows=[]))
        # Memory's API caps at 200; passing 9999 should clamp.
        asyncio.run(fetch_run_history(memory, "agent-1", limit=9999))
        assert memory.client.calls[0]["limit"] == 200
        # Passing 0 should clamp UP to 1.
        memory.client.calls.clear()
        asyncio.run(fetch_run_history(memory, "agent-1", limit=0))
        assert memory.client.calls[0]["limit"] == 1

    def test_drops_non_agent_run_cards(self):
        # Memory might return a card with a tag collision but
        # the wrong category — projection filters it out.
        wrong = _card(category="lesson_learned")
        good = _card(card_id="c-good", run_id="r-good")
        memory = _StubMemory(_StubClient(rows=[wrong, good]))
        entries = asyncio.run(fetch_run_history(memory, "agent-1"))
        assert len(entries) == 1
        assert entries[0].run_id == "r-good"

    def test_drops_cards_for_other_agents(self):
        other = _card(card_id="c-other", agent_entity_id="agent-2", run_id="r-other")
        ours = _card(card_id="c-ours", run_id="r-ours")
        memory = _StubMemory(_StubClient(rows=[other, ours]))
        entries = asyncio.run(fetch_run_history(memory, "agent-1"))
        assert len(entries) == 1
        assert entries[0].run_id == "r-ours"

    def test_empty_response(self):
        memory = _StubMemory(_StubClient(rows=[]))
        entries = asyncio.run(fetch_run_history(memory, "agent-1"))
        assert entries == ()

    def test_no_agent_id_raises(self):
        memory = _StubMemory(_StubClient(rows=[]))
        with pytest.raises(RunHistoryError, match="agent_entity_id"):
            asyncio.run(fetch_run_history(memory, ""))

    def test_missing_client_raises(self):
        with pytest.raises(RunHistoryError, match="_list_entities"):
            asyncio.run(fetch_run_history(object(), "agent-1"))

    def test_client_without_list_method_raises(self):
        class _BadClient:
            pass

        memory = _StubMemory(_BadClient())
        with pytest.raises(RunHistoryError, match="_list_entities"):
            asyncio.run(fetch_run_history(memory, "agent-1"))

    def test_sdk_exception_wraps(self):
        memory = _StubMemory(_StubClient(exc=RuntimeError("503")))
        with pytest.raises(RunHistoryError, match="fetch failed"):
            asyncio.run(fetch_run_history(memory, "agent-1"))

    def test_timeout_raises(self):
        memory = _StubMemory(_StubClient(rows=[], delay=0.5))
        with pytest.raises(RunHistoryError, match="timed out"):
            asyncio.run(fetch_run_history(memory, "agent-1", timeout=0.05))

    def test_attribute_access_facade(self):
        # _client (underscored) is the legacy accessor — also supported.
        class _Memory:
            def __init__(self, client):
                self._client = client

        client = _StubClient(rows=[_card()])
        entries = asyncio.run(fetch_run_history(_Memory(client), "agent-1"))
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            RunHistoryEntry as E,
            RunHistoryError as Err,
            RunHistorySummary as S,
            fetch_run_history as fetch,
            parse_run_history_entry as parse,
            summarize_run_history as summarize,
        )

        assert E is RunHistoryEntry
        assert Err is RunHistoryError
        assert S is RunHistorySummary
        assert fetch is fetch_run_history
        assert parse is parse_run_history_entry
        assert summarize is summarize_run_history


# ---------------------------------------------------------------------------
# Attribute-access SDK shape
# ---------------------------------------------------------------------------


class _FakeEntityResponse:
    """Mimics ``gigaevo_client.models.EntityResponse`` —
    attribute access, not dict."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestSDKShape:
    def test_attribute_access_objects_work(self):
        finished = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
        response = _FakeEntityResponse(
            entity_id="card-1",
            meta={"tags": ["agent_run", "agent:agent-1", "status:success"]},
            content={
                "category": "agent_run",
                "task_description": "do it",
                "description": "Run of demo — success",
                "usage": {
                    "run_id": "r-1",
                    "agent_entity_id": "agent-1",
                    "finished_at": finished,
                    "metrics": {
                        "duration_seconds": 1.0,
                        "step_count": 2,
                        "total_tokens": 99,
                        "exit_status": "success",
                    },
                },
            },
        )
        entry = parse_run_history_entry(response)
        assert entry is not None
        assert entry.card_id == "card-1"
        assert entry.run_id == "r-1"
        assert entry.finished_at is finished
        assert entry.total_tokens == 99
