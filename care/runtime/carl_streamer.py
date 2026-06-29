"""CARL → Textual message adapter (TODO §1.2 P0).

`CarlStreamer` is the execution-side sibling of `MagePoster`: it
adapts CARL's ``ReasoningContext`` callback contract into Textual
``Message`` instances posted on a target screen / app.

Wiring at runtime::

    streamer = CarlStreamer(self.app)  # `self` is an ExecutionScreen
    streamer.attach(context)           # populates ``context.on_*``
    result = await chain.execute_async(context)

Or directly::

    context = ReasoningContext(
        on_step_start=streamer.on_step_start,
        on_step_complete=streamer.on_step_complete,
        ...
    )

The streamer intentionally **uses the extended ``on_llm_chunk``
signature** ``(chunk, *, step_number, stage)`` so CARL's signature-
introspection in :mod:`mmar_carl.step_executors` routes per-step
metadata through. The legacy ``(chunk)``-only shape is still supported
in CARL, but CARE wants the extra fields to render chunks into the
right step pane.

Like ``MagePoster``, this adapter avoids importing ``mmar_carl`` so
a broken CARL install can't break CARE startup — the contract is
duck-typed against ``ReasoningContext.on_*`` callable fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from textual.message import Message

_log = logging.getLogger("care.runtime.carl_streamer")


class _PostTarget(Protocol):
    """Minimal duck type for a Textual message sink."""

    def post_message(self, message: Message) -> bool: ...


# ---------------------------------------------------------------------------
# Message classes
# ---------------------------------------------------------------------------


@dataclass
class StepStarted(Message):
    """A CARL chain step just began executing."""

    step_number: int
    step_title: str

    def __init__(self, step_number: int, step_title: str) -> None:
        self.step_number = step_number
        self.step_title = step_title
        super().__init__()


@dataclass
class StepCompleted(Message):
    """A CARL chain step finished.

    ``result`` is a CARL ``StepExecutionResult`` — kept as ``Any``
    here so we don't import ``mmar_carl`` at module-load time. Screens
    that render the result depend on the CARL install themselves.
    """

    result: Any

    def __init__(self, result: Any) -> None:
        self.result = result
        super().__init__()


@dataclass
class ChainCompleted(Message):
    """The whole chain finished (success or partial). ``result`` is a
    CARL ``ReasoningResult``."""

    result: Any

    def __init__(self, result: Any) -> None:
        self.result = result
        super().__init__()


@dataclass
class Progress(Message):
    """Chain-level progress update: ``completed`` of ``total`` steps."""

    completed: int
    total: int

    def __init__(self, completed: int, total: int) -> None:
        self.completed = completed
        self.total = total
        super().__init__()


@dataclass
class LlmChunk(Message):
    """A streaming LLM chunk landed mid-step.

    ``step_number`` and ``stage`` come from CARL's extended callback
    signature (CARE-M3 §2.3). ``stage`` is one of
    ``"fast"`` / ``"critic"`` / ``"regenerate"`` etc.; both fields
    are optional because CARL falls back to the legacy
    ``on_llm_chunk(chunk)`` shape when the caller's signature
    doesn't accept the kwargs.
    """

    chunk: str
    step_number: int | None
    stage: str | None

    def __init__(
        self,
        chunk: str,
        *,
        step_number: int | None = None,
        stage: str | None = None,
    ) -> None:
        self.chunk = chunk
        self.step_number = step_number
        self.stage = stage
        super().__init__()


@dataclass
class HumanInputRequested(Message):
    """A ``HumanInputStep`` is asking for the user's input.

    Screens should pop a modal that collects ``prompt`` and resolve
    ``future`` via ``context.provide_human_input(value)``. ``future``
    is kept as ``Any`` because we don't import ``asyncio.Future`` at
    module load time — the consumer already has it.
    """

    prompt: str
    future: Any

    def __init__(self, prompt: str, future: Any) -> None:
        self.prompt = prompt
        self.future = future
        super().__init__()


@dataclass
class StepEvent(Message):
    """Fine-grained intra-step event (CARL ``on_step_event``).

    Covers the canonical events documented in CARL's
    ``ReasoningContext.on_step_event`` field: ``'llm_agent.tool_call'``,
    ``'debate.round_started'``, ``'parallel_sampling.sample'``,
    ``'supervisor.route_selected'``, etc. ``payload`` is the
    callback's third positional argument — passed through verbatim.
    """

    step_number: int
    event_type: str
    payload: dict[str, Any]

    def __init__(
        self,
        step_number: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.step_number = step_number
        self.event_type = event_type
        self.payload = payload
        super().__init__()


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class CarlStreamer:
    """Adapter: implement CARL's ``ReasoningContext`` callback shape
    by posting Textual messages to a target.

    Construct with the Textual sink (an ``App`` or ``Screen``), then
    either pass individual ``streamer.on_step_*`` callables to
    :class:`mmar_carl.ReasoningContext`, or call
    :meth:`attach` to populate every supported callback on a context
    in one go.
    """

    def __init__(
        self,
        target: _PostTarget,
        *,
        token_counter: Any = None,
    ) -> None:
        self._target = target
        # P1.3: optional sink for chain-level LLM usage. When
        # set, :meth:`on_chain_complete` extracts a `usage`
        # dict from the `ReasoningResult` and folds it into
        # the counter so the StatusBar's next refresh tick
        # reflects the new total.
        self._token_counter = token_counter

    @property
    def target(self) -> _PostTarget:
        """The message sink this streamer forwards to."""
        return self._target

    # Every ``on_*`` callback the streamer can drive. Some are
    # optional on the upstream :class:`ReasoningContext` (CARL
    # has dropped + re-added these hooks across versions), so
    # :meth:`attach` checks for each before assigning rather
    # than hard-coding the set. A missing field is logged at
    # DEBUG and skipped — the streamer keeps working for every
    # OTHER callback that IS supported.
    _SUPPORTED_CALLBACKS: tuple[str, ...] = (
        "on_step_start",
        "on_step_complete",
        "on_chain_complete",
        "on_progress",
        "on_llm_chunk",
        "on_human_input_requested",
        "on_step_event",
    )

    def attach(self, context: Any) -> Any:
        """Populate every supported ``on_*`` callback on ``context``.

        Returns ``context`` so call-sites can chain
        ``streamer.attach(ReasoningContext(...))``. Existing
        callbacks on the context are overwritten — the screen owns
        the streaming surface during execution.

        Defensive against upstream API drift: ``ReasoningContext``
        is a Pydantic model with strict field validation, and CARL
        evolves which callbacks it exposes across releases. We
        check for each callback field on the model before assigning
        so a removed hook surfaces a DEBUG log instead of crashing
        the whole execution path.
        """
        for name in self._SUPPORTED_CALLBACKS:
            if not self._context_supports(context, name):
                _log.debug(
                    "CarlStreamer: skipping unsupported callback %r "
                    "(upstream ReasoningContext doesn't expose it)",
                    name,
                )
                continue
            try:
                setattr(context, name, getattr(self, name))
            except (AttributeError, ValueError, TypeError) as exc:
                _log.debug(
                    "CarlStreamer: couldn't wire %r: %s", name, exc,
                )
        return context

    @staticmethod
    def _context_supports(context: Any, name: str) -> bool:
        """Return True when ``context`` accepts attribute ``name``.

        Prefers Pydantic's ``model_fields`` (definitive list of
        accepted fields on v2 models) and falls back to
        :func:`hasattr` for non-Pydantic shapes (test stubs,
        future CARL non-model contexts).
        """
        fields = getattr(type(context), "model_fields", None)
        if isinstance(fields, dict):
            return name in fields
        return hasattr(context, name)

    # ----- step lifecycle ------------------------------------------

    def on_step_start(self, step_number: int, step_title: str) -> None:
        self._target.post_message(StepStarted(step_number, step_title))

    def on_step_complete(self, result: Any) -> None:
        self._target.post_message(StepCompleted(result))

    def on_chain_complete(self, result: Any) -> None:
        self._update_token_counter(result)
        self._target.post_message(ChainCompleted(result))

    def _update_token_counter(self, result: Any) -> None:
        """P1.3 hook — extract chain-level LLM usage from a
        :class:`mmar_carl.ReasoningResult`-like object and fold
        it into the wired counter. ``None`` counter is a
        no-op; non-dict / missing ``usage`` is a no-op."""
        counter = self._token_counter
        if counter is None:
            return
        usage = _extract_usage_from_result(result)
        if usage:
            try:
                counter.add(usage)
            except Exception:
                pass

    # ----- progress / streaming ------------------------------------

    def on_progress(self, completed: int, total: int) -> None:
        self._target.post_message(Progress(completed, total))

    def on_llm_chunk(
        self,
        chunk: str,
        *,
        step_number: int | None = None,
        stage: str | None = None,
    ) -> None:
        self._target.post_message(
            LlmChunk(chunk, step_number=step_number, stage=stage)
        )

    # ----- human-in-the-loop --------------------------------------

    def on_human_input_requested(self, prompt: str, future: Any) -> None:
        self._target.post_message(HumanInputRequested(prompt, future))

    # ----- fine-grained sub-step events ----------------------------

    def on_step_event(
        self,
        step_number: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self._target.post_message(StepEvent(step_number, event_type, payload))


def _extract_usage_from_result(result: Any) -> dict[str, Any] | None:
    """Project a CARL ``ReasoningResult``-like object's LLM
    usage into the ``{prompt, completion, total}`` dict the
    :class:`SessionTokenCounter` consumes.

    CARL's `ReasoningResult` exposes ``total_usage`` /
    ``usage`` depending on version; we probe both. Step
    results carry per-step `usage` too — but the
    chain-complete callback feeds the aggregate, not the
    per-step rollup."""
    if result is None:
        return None
    if isinstance(result, dict):
        for key in ("total_usage", "usage", "token_usage"):
            value = result.get(key)
            if isinstance(value, dict):
                return dict(value)
        metrics = result.get("metrics")
        if isinstance(metrics, dict):
            inner = metrics.get("usage")
            if isinstance(inner, dict):
                return dict(inner)
        return None
    for attr in ("total_usage", "usage", "token_usage"):
        candidate = getattr(result, attr, None)
        if isinstance(candidate, dict):
            return dict(candidate)
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, dict):
        inner = metrics.get("usage")
        if isinstance(inner, dict):
            return dict(inner)
    return None


__all__ = [
    "CarlStreamer",
    "ChainCompleted",
    "HumanInputRequested",
    "LlmChunk",
    "Progress",
    "StepCompleted",
    "StepEvent",
    "StepStarted",
]
