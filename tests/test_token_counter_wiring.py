"""Tests for the P1.3 token-counter wiring.

`MagePoster.handle_stage_completed` and
`CarlStreamer.on_chain_complete` both fold an LLM-usage dict
into the wired `SessionTokenCounter` so the next StatusBar
refresh reflects the new totals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from care.runtime.carl_streamer import CarlStreamer
from care.runtime.mage_poster import MagePoster
from care.runtime.status_bar import SessionTokenCounter


# ---------------------------------------------------------------------------
# Stub sinks
# ---------------------------------------------------------------------------


class _Sink:
    def __init__(self) -> None:
        self.messages: list = []

    def post_message(self, message: Any) -> bool:
        self.messages.append(message)
        return True


@dataclass
class _UsageBearingResult:
    name: str = "stage-result"
    usage: dict | None = None


# ---------------------------------------------------------------------------
# MagePoster
# ---------------------------------------------------------------------------


class TestMagePosterTokenWiring:
    def test_no_counter_is_a_noop(self):
        sink = _Sink()
        poster = MagePoster(sink)
        # No exception even with usage-bearing result.
        poster.on_stage_complete(
            "plan", _UsageBearingResult(usage={"total": 100}),
        )
        assert len(sink.messages) == 1

    def test_handle_extracts_usage_from_attr(self):
        counter = SessionTokenCounter()
        poster = MagePoster(_Sink(), token_counter=counter)
        poster.on_stage_complete(
            "plan",
            _UsageBearingResult(
                usage={"prompt": 100, "completion": 50, "total": 150},
            ),
        )
        snap = counter.snapshot()
        assert snap.total == 150
        assert snap.prompt == 100
        assert snap.calls == 1

    def test_handle_extracts_usage_from_dict(self):
        counter = SessionTokenCounter()
        poster = MagePoster(_Sink(), token_counter=counter)
        poster.on_stage_complete(
            "plan",
            {"usage": {"prompt": 10, "completion": 5}},
        )
        snap = counter.snapshot()
        assert snap.prompt == 10
        assert snap.completion == 5

    def test_handle_extracts_from_metrics_inner(self):
        counter = SessionTokenCounter()
        poster = MagePoster(_Sink(), token_counter=counter)
        poster.on_stage_complete(
            "plan",
            {"metrics": {"usage": {"total": 77}}},
        )
        snap = counter.snapshot()
        assert snap.total == 77

    def test_missing_usage_is_a_noop(self):
        counter = SessionTokenCounter()
        poster = MagePoster(_Sink(), token_counter=counter)
        poster.on_stage_complete("plan", {"name": "no usage here"})
        snap = counter.snapshot()
        assert snap.calls == 0
        assert snap.total == 0

    def test_counter_failure_doesnt_propagate(self):
        class _BadCounter:
            def add(self, usage):
                raise RuntimeError("counter exploded")

        poster = MagePoster(_Sink(), token_counter=_BadCounter())
        # No exception.
        poster.on_stage_complete(
            "plan", {"usage": {"total": 10}},
        )


# ---------------------------------------------------------------------------
# CarlStreamer
# ---------------------------------------------------------------------------


@dataclass
class _ReasoningResult:
    total_usage: dict | None = None
    usage: dict | None = None


class TestCarlStreamerTokenWiring:
    def test_no_counter_is_a_noop(self):
        sink = _Sink()
        streamer = CarlStreamer(sink)
        streamer.on_chain_complete(
            _ReasoningResult(total_usage={"total": 100}),
        )
        assert len(sink.messages) == 1

    def test_total_usage_attr_lands_on_counter(self):
        counter = SessionTokenCounter()
        streamer = CarlStreamer(_Sink(), token_counter=counter)
        streamer.on_chain_complete(
            _ReasoningResult(
                total_usage={"prompt": 200, "completion": 50, "total": 250},
            ),
        )
        snap = counter.snapshot()
        assert snap.total == 250
        assert snap.prompt == 200

    def test_dict_usage_lands_on_counter(self):
        counter = SessionTokenCounter()
        streamer = CarlStreamer(_Sink(), token_counter=counter)
        streamer.on_chain_complete(
            {"usage": {"prompt": 5, "completion": 5}},
        )
        snap = counter.snapshot()
        assert snap.prompt == 5
        assert snap.completion == 5

    def test_falls_back_to_usage_when_total_usage_missing(self):
        counter = SessionTokenCounter()
        streamer = CarlStreamer(_Sink(), token_counter=counter)
        streamer.on_chain_complete(
            _ReasoningResult(usage={"total": 42}),
        )
        snap = counter.snapshot()
        assert snap.total == 42

    def test_missing_usage_is_a_noop(self):
        counter = SessionTokenCounter()
        streamer = CarlStreamer(_Sink(), token_counter=counter)
        streamer.on_chain_complete(_ReasoningResult())
        snap = counter.snapshot()
        assert snap.calls == 0


# ---------------------------------------------------------------------------
# Pure projection
# ---------------------------------------------------------------------------


class TestProjection:
    def test_mage_extract_none_result(self):
        from care.runtime.mage_poster import _extract_usage

        assert _extract_usage(None) is None

    def test_carl_extract_none_result(self):
        from care.runtime.carl_streamer import _extract_usage_from_result

        assert _extract_usage_from_result(None) is None
