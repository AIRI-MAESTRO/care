"""Tests for ``care.runtime.CarlStreamer`` (TODO §1.2 P0).

Coverage layers mirror ``test_mage_poster.py``:
1. Message subclasses construct correctly and remain Textual
   ``Message`` instances.
2. Adapter dispatch — each CARL callback method posts exactly one
   message of the right type with the right payload.
3. ``attach`` populates every supported callback on a CARL-like
   context object.
4. Protocol conformance: every callback CARL's ``ReasoningContext``
   declares is callable on the streamer with the right shape.
"""

from __future__ import annotations

import pytest
from textual.message import Message

from care.runtime import (
    CarlStreamer,
    ChainCompleted,
    HumanInputRequested,
    LlmChunk,
    Progress,
    StepCompleted,
    StepEvent,
    StepStarted,
)


class _Collector:
    """Tiny Textual sink stub."""

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def post_message(self, message: Message) -> bool:
        self.messages.append(message)
        return True


@pytest.fixture
def collector() -> _Collector:
    return _Collector()


@pytest.fixture
def streamer(collector: _Collector) -> CarlStreamer:
    return CarlStreamer(collector)


# ---------------------------------------------------------------------------
# Message class shape
# ---------------------------------------------------------------------------


class TestMessageClasses:
    @pytest.mark.parametrize(
        "cls,args,kwargs,expected",
        [
            (StepStarted, (3, "Risk Assessment"), {}, {"step_number": 3, "step_title": "Risk Assessment"}),
            (StepCompleted, (object(),), {}, None),  # special: object identity asserted below
            (ChainCompleted, (object(),), {}, None),
            (Progress, (2, 4), {}, {"completed": 2, "total": 4}),
            (
                LlmChunk,
                ("hello",),
                {"step_number": 1, "stage": "fast"},
                {"chunk": "hello", "step_number": 1, "stage": "fast"},
            ),
            (
                LlmChunk,
                ("legacy chunk",),
                {},
                {"chunk": "legacy chunk", "step_number": None, "stage": None},
            ),
            (
                StepEvent,
                (2, "debate.round_started", {"round": 1, "role": "advocate"}),
                {},
                {
                    "step_number": 2,
                    "event_type": "debate.round_started",
                    "payload": {"round": 1, "role": "advocate"},
                },
            ),
        ],
    )
    def test_fields_round_trip(self, cls, args, kwargs, expected):
        msg = cls(*args, **kwargs)
        assert isinstance(msg, Message)
        if expected is not None:
            for k, v in expected.items():
                assert getattr(msg, k) == v

    def test_step_completed_preserves_result_identity(self):
        sentinel = object()
        msg = StepCompleted(sentinel)
        assert msg.result is sentinel

    def test_chain_completed_preserves_result_identity(self):
        sentinel = object()
        msg = ChainCompleted(sentinel)
        assert msg.result is sentinel

    def test_human_input_requested_carries_future_handle(self):
        future = object()  # opaque future handle
        msg = HumanInputRequested("Choose an option:", future)
        assert msg.prompt == "Choose an option:"
        assert msg.future is future


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------


class TestStreamerDispatch:
    def test_on_step_start_posts_StepStarted(self, streamer, collector):
        streamer.on_step_start(1, "Extract data")
        msg = collector.messages[0]
        assert isinstance(msg, StepStarted)
        assert msg.step_number == 1
        assert msg.step_title == "Extract data"

    def test_on_step_complete_posts_StepCompleted(self, streamer, collector):
        result = object()
        streamer.on_step_complete(result)
        msg = collector.messages[0]
        assert isinstance(msg, StepCompleted)
        assert msg.result is result

    def test_on_chain_complete_posts_ChainCompleted(self, streamer, collector):
        result = object()
        streamer.on_chain_complete(result)
        msg = collector.messages[0]
        assert isinstance(msg, ChainCompleted)
        assert msg.result is result

    def test_on_progress_posts_Progress(self, streamer, collector):
        streamer.on_progress(2, 5)
        msg = collector.messages[0]
        assert isinstance(msg, Progress)
        assert (msg.completed, msg.total) == (2, 5)

    def test_on_llm_chunk_extended_signature(self, streamer, collector):
        """Streamer always exposes the extended signature; CARL's
        signature introspection routes both shapes through it."""
        streamer.on_llm_chunk("tok", step_number=4, stage="critic")
        msg = collector.messages[0]
        assert isinstance(msg, LlmChunk)
        assert msg.chunk == "tok"
        assert msg.step_number == 4
        assert msg.stage == "critic"

    def test_on_llm_chunk_without_kwargs(self, streamer, collector):
        """Calling with only the chunk (legacy shape) leaves the
        per-step metadata as ``None`` rather than failing."""
        streamer.on_llm_chunk("tok")
        msg = collector.messages[0]
        assert msg.step_number is None
        assert msg.stage is None

    def test_on_human_input_requested_posts_HumanInputRequested(
        self, streamer, collector
    ):
        future = object()
        streamer.on_human_input_requested("Pick:", future)
        msg = collector.messages[0]
        assert isinstance(msg, HumanInputRequested)
        assert msg.prompt == "Pick:"
        assert msg.future is future

    def test_on_step_event_posts_StepEvent(self, streamer, collector):
        payload = {"tool": "WebFetch", "args": {"url": "https://example.com"}}
        streamer.on_step_event(2, "llm_agent.tool_call", payload)
        msg = collector.messages[0]
        assert isinstance(msg, StepEvent)
        assert msg.step_number == 2
        assert msg.event_type == "llm_agent.tool_call"
        assert msg.payload is payload

    def test_message_order_preserved(self, streamer, collector):
        streamer.on_step_start(1, "a")
        streamer.on_llm_chunk("hello ", step_number=1, stage="fast")
        streamer.on_llm_chunk("world", step_number=1, stage="fast")
        streamer.on_step_complete(object())
        streamer.on_progress(1, 3)
        types = [type(m).__name__ for m in collector.messages]
        assert types == [
            "StepStarted",
            "LlmChunk",
            "LlmChunk",
            "StepCompleted",
            "Progress",
        ]


# ---------------------------------------------------------------------------
# attach()
# ---------------------------------------------------------------------------


class _FakeContext:
    """Duck-typed stand-in for ``mmar_carl.ReasoningContext``.

    Has the same ``on_*`` attribute surface so :meth:`CarlStreamer.attach`
    can populate them, but doesn't require pulling in the real CARL
    install during tests.
    """

    on_step_start = None
    on_step_complete = None
    on_chain_complete = None
    on_progress = None
    on_llm_chunk = None
    on_human_input_requested = None
    on_step_event = None


class TestAttach:
    def test_attach_returns_same_context(self, streamer):
        ctx = _FakeContext()
        assert streamer.attach(ctx) is ctx

    def test_attach_populates_every_callback(self, streamer):
        """Each callback must be assigned to a non-None callable.
        Identity comparisons would fail on bound methods (Python
        creates a fresh wrapper on each attribute access), so we
        check ``is not None`` + ``callable``."""
        ctx = _FakeContext()
        streamer.attach(ctx)
        for attr in (
            "on_step_start",
            "on_step_complete",
            "on_chain_complete",
            "on_progress",
            "on_llm_chunk",
            "on_human_input_requested",
            "on_step_event",
        ):
            assigned = getattr(ctx, attr)
            assert assigned is not None
            assert callable(assigned)

    def test_attach_overwrites_existing_callbacks(self, streamer, collector):
        ctx = _FakeContext()
        prior_calls: list[tuple] = []

        def prior(*args, **kwargs):
            prior_calls.append((args, kwargs))

        ctx.on_step_start = prior
        streamer.attach(ctx)
        # After attach, calling the slot must hit the streamer (which
        # posts a message), not the prior callback.
        ctx.on_step_start(1, "first")
        assert prior_calls == []
        assert len(collector.messages) == 1
        assert isinstance(collector.messages[0], StepStarted)

    def test_attached_callbacks_post_messages(self, streamer, collector):
        ctx = _FakeContext()
        streamer.attach(ctx)
        ctx.on_step_start(1, "first")
        ctx.on_progress(1, 1)
        assert len(collector.messages) == 2
        assert isinstance(collector.messages[0], StepStarted)
        assert isinstance(collector.messages[1], Progress)


class TestAttachDefensiveAgainstAPIDrift:
    """The real-world ReasoningContext is a Pydantic model with
    ``extra="forbid"`` (or `validate_assignment=True`). Upstream
    CARL has historically dropped + re-added callback fields
    across releases, and assigning a name the model doesn't
    declare raises ``ValueError`` — which previously killed the
    whole `_run_generation` worker silently. The fix walks the
    callback list and only assigns ones the model actually
    declares, so partial-support is OK."""

    def test_attach_skips_pydantic_unknown_field(self, streamer, collector):
        """Pydantic model exposing only a SUBSET of callbacks
        (e.g. CARL dropped ``on_human_input_requested``).
        ``attach`` must NOT raise, and must wire the
        supported callbacks correctly."""
        from pydantic import BaseModel, ConfigDict

        class _PartialContext(BaseModel):
            model_config = ConfigDict(
                arbitrary_types_allowed=True,
                validate_assignment=True,
                extra="forbid",
            )
            # Drops ``on_human_input_requested`` + ``on_step_event``
            # to simulate the real upstream drift that caused the
            # ValueError in the wild.
            on_step_start: object | None = None
            on_step_complete: object | None = None
            on_chain_complete: object | None = None
            on_progress: object | None = None
            on_llm_chunk: object | None = None

        ctx = _PartialContext()
        # Should not raise even though two callback fields are
        # missing from the model schema.
        returned = streamer.attach(ctx)
        assert returned is ctx
        # Supported callbacks got wired.
        assert callable(ctx.on_step_start)
        assert callable(ctx.on_progress)
        # Driving a wired callback still posts to the collector.
        ctx.on_step_start(1, "demo")
        assert len(collector.messages) == 1
        assert isinstance(collector.messages[0], StepStarted)

    def test_attach_skips_setattr_raise(self, streamer):
        """Even when a field shows up in `model_fields` but the
        setter raises (custom validator rejecting our callable),
        ``attach`` keeps going for the other fields rather than
        propagating the error."""
        from pydantic import BaseModel, ConfigDict, field_validator

        class _StrictContext(BaseModel):
            model_config = ConfigDict(
                arbitrary_types_allowed=True,
                validate_assignment=True,
                extra="forbid",
            )
            on_step_start: object | None = None
            on_step_complete: object | None = None
            on_chain_complete: object | None = None
            on_progress: object | None = None
            on_llm_chunk: object | None = None
            on_human_input_requested: object | None = None
            on_step_event: object | None = None

            @field_validator("on_chain_complete")
            @classmethod
            def _reject(cls, v):  # noqa: ANN001
                if v is not None:
                    raise ValueError("custom rejector")
                return v

        ctx = _StrictContext()
        # Doesn't raise even though `on_chain_complete` rejects.
        streamer.attach(ctx)
        # Every OTHER callback is wired.
        assert callable(ctx.on_step_start)
        assert callable(ctx.on_progress)
        assert callable(ctx.on_human_input_requested)

    def test_attach_falls_back_to_hasattr_for_non_pydantic(
        self, streamer,
    ):
        """Non-Pydantic shapes (test stubs, future plain-object
        CARL contexts) keep using the legacy `hasattr` check so
        the existing fake-context pattern stays valid."""

        class _PlainStub:
            on_step_start = None
            on_progress = None
            # No other callback fields.

        ctx = _PlainStub()
        streamer.attach(ctx)
        assert callable(ctx.on_step_start)
        assert callable(ctx.on_progress)
        # Unsupported callbacks are NOT set on the plain stub
        # (hasattr returned False).
        assert not hasattr(ctx, "on_step_complete")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Pins exact callback names + arities so an upstream rename
    surfaces here loudly. Does NOT import ``mmar_carl`` — CARE
    maintains its own mirror of the contract."""

    REQUIRED_METHODS = {
        "on_step_start": ("step_number", "step_title"),
        "on_step_complete": ("result",),
        "on_chain_complete": ("result",),
        "on_progress": ("completed", "total"),
        "on_llm_chunk": ("chunk",),  # *step_number, stage as kwargs
        "on_human_input_requested": ("prompt", "future"),
        "on_step_event": ("step_number", "event_type", "payload"),
    }

    @pytest.mark.parametrize("method_name", REQUIRED_METHODS.keys())
    def test_method_exists(self, streamer, method_name):
        assert hasattr(streamer, method_name)
        assert callable(getattr(streamer, method_name))

    @pytest.mark.parametrize(
        "method_name,args",
        [
            ("on_step_start", (1, "title")),
            ("on_step_complete", (object(),)),
            ("on_chain_complete", (object(),)),
            ("on_progress", (0, 5)),
            ("on_llm_chunk", ("chunk",)),
            ("on_human_input_requested", ("prompt", object())),
            ("on_step_event", (1, "evt", {})),
        ],
    )
    def test_method_accepts_positional_args(self, streamer, method_name, args):
        getattr(streamer, method_name)(*args)

    def test_on_llm_chunk_accepts_extended_signature(self, streamer):
        streamer.on_llm_chunk("chunk", step_number=1, stage="fast")
