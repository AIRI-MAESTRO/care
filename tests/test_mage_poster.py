"""Tests for ``care.runtime.MagePoster`` (TODO §1.2 P0).

The adapter has no UI side effects — its only job is to construct
the right Textual message and call ``target.post_message``. Tests
use a tiny list-collector stub for the target so we never need a
running ``App``.

Coverage layers:
1. **Message construction** — each Message subclass round-trips its
   fields and remains a Textual ``Message`` instance (so the
   ``on_<snake_case>`` dispatch path works).
2. **Adapter dispatch** — every MAGE callback method posts exactly
   one message of the right type with the right payload.
3. **Protocol conformance** — the adapter is duck-typed and exposes
   every method the upstream ``MAGEProgressCallback`` requires.
"""

from __future__ import annotations

import pytest
from textual.message import Message

from care.runtime import (
    CostEstimate,
    LLMChunk,
    MagePoster,
    StageCompleted,
    StageError,
    StageProgress,
    StageRetry,
    StageStarted,
)


class _Collector:
    """Stand-in for a Textual ``App``/``Screen``.

    Captures every message in declaration order so tests can assert
    on type + payload without spinning up a real app loop.
    """

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def post_message(self, message: Message) -> bool:
        self.messages.append(message)
        return True


@pytest.fixture
def collector() -> _Collector:
    return _Collector()


@pytest.fixture
def poster(collector: _Collector) -> MagePoster:
    return MagePoster(collector)


# ---------------------------------------------------------------------------
# Message class shape
# ---------------------------------------------------------------------------


class TestMessageClasses:
    @pytest.mark.parametrize(
        "cls,kwargs,expected",
        [
            (StageStarted, {"stage": "domain_analysis"}, {"stage": "domain_analysis"}),
            (
                StageCompleted,
                {"stage": "plan_steps", "result": {"steps": 4}},
                {"stage": "plan_steps", "result": {"steps": 4}},
            ),
            (LLMChunk, {"stage": "describe", "delta": "Step 1 "}, {"stage": "describe", "delta": "Step 1 "}),
            (
                StageProgress,
                {"stage": "describe", "artifact": {"step_number": 1}},
                {"stage": "describe", "artifact": {"step_number": 1}},
            ),
        ],
    )
    def test_fields_round_trip(self, cls, kwargs, expected):
        msg = cls(**kwargs)
        for k, v in expected.items():
            assert getattr(msg, k) == v
        # Critically: every adapter message is still a Textual Message
        # so screens can use the on_<snake_case> dispatch pattern.
        assert isinstance(msg, Message)

    def test_stage_error_carries_exception(self):
        exc = RuntimeError("oops")
        msg = StageError("verify_chain", exc)
        assert msg.stage == "verify_chain"
        assert msg.error is exc
        assert isinstance(msg, Message)

    def test_stage_retry_carries_attempt_and_exception(self):
        exc = TimeoutError("LLM timed out")
        msg = StageRetry("describe", 2, exc)
        assert msg.stage == "describe"
        assert msg.attempt == 2
        assert msg.error is exc

    def test_cost_estimate_carries_estimate(self):
        estimate = {"usd": 0.42, "tokens": 12_500}
        msg = CostEstimate(estimate)
        assert msg.estimate == estimate


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------


class TestPosterDispatch:
    def test_on_stage_start_posts_StageStarted(self, poster, collector):
        poster.on_stage_start("domain_analysis")
        assert len(collector.messages) == 1
        msg = collector.messages[0]
        assert isinstance(msg, StageStarted)
        assert msg.stage == "domain_analysis"

    def test_on_stage_complete_posts_StageCompleted(self, poster, collector):
        result = {"steps": 4, "domain": "weather"}
        poster.on_stage_complete("plan_steps", result)
        msg = collector.messages[0]
        assert isinstance(msg, StageCompleted)
        assert msg.stage == "plan_steps"
        assert msg.result is result

    def test_on_error_posts_StageError(self, poster, collector):
        exc = ValueError("bad plan")
        poster.on_error("plan_steps", exc)
        msg = collector.messages[0]
        assert isinstance(msg, StageError)
        assert msg.error is exc

    def test_on_llm_chunk_posts_LLMChunk(self, poster, collector):
        poster.on_llm_chunk("describe", "hello ")
        poster.on_llm_chunk("describe", "world")
        assert len(collector.messages) == 2
        assert all(isinstance(m, LLMChunk) for m in collector.messages)
        assert collector.messages[0].delta == "hello "
        assert collector.messages[1].delta == "world"

    def test_on_stage_progress_posts_StageProgress(self, poster, collector):
        artifact = {"step_number": 3, "title": "Risk Analysis"}
        poster.on_stage_progress("describe", artifact)
        msg = collector.messages[0]
        assert isinstance(msg, StageProgress)
        assert msg.artifact is artifact

    def test_on_retry_posts_StageRetry(self, poster, collector):
        exc = TimeoutError("LLM 504")
        poster.on_retry("verify_chain", 1, exc)
        msg = collector.messages[0]
        assert isinstance(msg, StageRetry)
        assert msg.attempt == 1
        assert msg.error is exc

    def test_on_cost_estimate_posts_CostEstimate(self, poster, collector):
        poster.on_cost_estimate({"usd": 0.10})
        msg = collector.messages[0]
        assert isinstance(msg, CostEstimate)
        assert msg.estimate == {"usd": 0.10}

    def test_message_order_preserved(self, poster, collector):
        """Adapter is sync — message order must match call order so
        screens can render a deterministic stage timeline."""
        poster.on_stage_start("domain_analysis")
        poster.on_stage_complete("domain_analysis", {"d": 1})
        poster.on_stage_start("plan_steps")
        poster.on_stage_progress("plan_steps", {"i": 1})
        poster.on_stage_progress("plan_steps", {"i": 2})
        poster.on_stage_complete("plan_steps", {"steps": 4})
        types = [type(m).__name__ for m in collector.messages]
        assert types == [
            "StageStarted",
            "StageCompleted",
            "StageStarted",
            "StageProgress",
            "StageProgress",
            "StageCompleted",
        ]


class TestTargetExposure:
    def test_target_property_returns_constructed_sink(self, collector):
        poster = MagePoster(collector)
        assert poster.target is collector


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """``MagePoster`` is duck-typed against MAGE's ``MAGEProgressCallback``.

    These tests pin the exact method names + arities so a rename
    upstream would surface here loudly. We don't import the protocol
    from ``mmar_mage`` (keeping CARE startup independent of the
    MAGE install) — instead we maintain the contract here.
    """

    REQUIRED_METHODS = {
        "on_stage_start": ("stage",),
        "on_stage_complete": ("stage", "result"),
        "on_error": ("stage", "error"),
        "on_llm_chunk": ("stage", "delta"),
        "on_stage_progress": ("stage", "artifact"),
        "on_retry": ("stage", "attempt", "error"),
        "on_cost_estimate": ("estimate",),
    }

    @pytest.mark.parametrize("method_name", REQUIRED_METHODS.keys())
    def test_method_exists(self, poster, method_name):
        assert hasattr(poster, method_name)
        assert callable(getattr(poster, method_name))

    @pytest.mark.parametrize(
        "method_name,positional_args",
        [
            (m, args) for m, args in REQUIRED_METHODS.items()
        ],
    )
    def test_method_accepts_positional_args(
        self, poster, method_name, positional_args
    ):
        """Smoke test: every required method accepts its declared
        positional args without raising."""
        sentinels = tuple(object() for _ in positional_args)
        getattr(poster, method_name)(*sentinels)
