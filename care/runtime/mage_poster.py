"""MAGE → Textual message adapter (TODO §1.2 P0).

`MagePoster` implements the `MAGEProgressCallback` contract (defined
in ``mmar_mage.generator``) by posting Textual ``Message`` instances
to a target — typically the running ``CareApp`` or a specific
``GenerationScreen``. This lets the TUI receive MAGE progress
reactively (via ``on_<message>`` handlers) without coupling its
screens to the MAGE callback signatures.

Design notes:

- **Duck-typed target.** ``MagePoster`` accepts any object exposing
  ``post_message(Message) -> bool`` — both ``App`` and ``Screen``
  fit. Tests can pass a tiny stub collector.
- **No upstream imports.** Adapter doesn't import ``mmar_mage`` so a
  broken MAGE install can't break CARE startup. The ``MAGEProgressCallback``
  protocol is duck-typed: MAGE calls our methods by name.
- **Sync callback path.** MAGE's callbacks are synchronous so we call
  ``target.post_message`` directly. Textual marshals the message
  onto the app's event loop internally — safe to call from any
  thread MAGE happens to be running on.
- **One Message subclass per event.** Screens dispatch via the
  Textual ``on_<snake_case>`` convention (``on_stage_started`` etc.)
  so isolating each event into its own class keeps handlers narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from textual.message import Message


class _PostTarget(Protocol):
    """Minimal duck type for a Textual message sink."""

    def post_message(self, message: Message) -> bool: ...


# ---------------------------------------------------------------------------
# Message classes
# ---------------------------------------------------------------------------


@dataclass
class StageStarted(Message):
    """A MAGE pipeline stage just began."""

    stage: str

    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__()


@dataclass
class StageCompleted(Message):
    """A MAGE pipeline stage finished successfully.

    ``result`` is whatever the stage returned — a ``DomainAnalysis``,
    ``StepPlan``, ``DAGStructure``, ``CritiqueResult``, etc. CARE's
    Inspection screen can drill into it directly.
    """

    stage: str
    result: Any

    def __init__(self, stage: str, result: Any) -> None:
        self.stage = stage
        self.result = result
        super().__init__()


@dataclass
class StageError(Message):
    """A MAGE pipeline stage raised."""

    stage: str
    error: BaseException

    def __init__(self, stage: str, error: BaseException) -> None:
        self.stage = stage
        self.error = error
        super().__init__()


@dataclass
class LLMChunk(Message):
    """A streaming LLM chunk landed mid-stage.

    Only emitted when the underlying client + MAGE config enable
    streaming (``MAGEConfig.enable_streaming=True``).
    """

    stage: str
    delta: str

    def __init__(self, stage: str, delta: str) -> None:
        self.stage = stage
        self.delta = delta
        super().__init__()


@dataclass
class StageProgress(Message):
    """A stage emitted an incremental artifact (per-step describe,
    per-branch ToT, per-sim MCTS, etc.).
    """

    stage: str
    artifact: Any

    def __init__(self, stage: str, artifact: Any) -> None:
        self.stage = stage
        self.artifact = artifact
        super().__init__()


@dataclass
class StageRetry(Message):
    """MAGE retried a stage after a transient LLM-client failure.

    ``attempt`` is 1-indexed: the first failed attempt fires
    ``attempt=1`` with the next attempt about to begin.
    """

    stage: str
    attempt: int
    error: BaseException

    def __init__(self, stage: str, attempt: int, error: BaseException) -> None:
        self.stage = stage
        self.attempt = attempt
        self.error = error
        super().__init__()


@dataclass
class CostEstimate(Message):
    """MAGE pre-flight cost estimate.

    Fires once before any LLM calls when
    ``MAGEConfig.enable_preflight_cost_estimate=True``. CARE can show
    a "this run will cost ~$X" confirmation panel.
    """

    estimate: Any

    def __init__(self, estimate: Any) -> None:
        self.estimate = estimate
        super().__init__()


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class MagePoster:
    """Adapter: implement MAGE's ``MAGEProgressCallback`` by posting
    Textual messages to a target.

    Construct with the target object whose ``post_message`` should
    receive events (typically the ``CareApp`` instance or the active
    ``GenerationScreen``). Pass the resulting poster as
    ``MAGEGenerator(progress=poster)``.

    Example::

        poster = MagePoster(self.app)  # `self` is a Screen
        generator = MAGEGenerator(config=mage_cfg, progress=poster)
        result = await generator.generate(query)
    """

    def __init__(
        self,
        target: _PostTarget,
        *,
        token_counter: Any = None,
    ) -> None:
        self._target = target
        # P1.3: optional sink for per-stage LLM usage. When set,
        # :meth:`on_stage_complete` extracts a `usage` dict from
        # the stage result and folds it into the counter so the
        # StatusBar's next refresh tick reflects the new total.
        self._token_counter = token_counter

    @property
    def target(self) -> _PostTarget:
        """The message sink this poster forwards to. Exposed for
        tests + debugging; do not swap it at runtime mid-generation
        — Textual's event loop assumes a stable target."""
        return self._target

    # ----- required callbacks --------------------------------------

    def on_stage_start(self, stage: str) -> None:
        self._target.post_message(StageStarted(stage))

    def on_stage_complete(self, stage: str, result: Any) -> None:
        self.handle_stage_completed(stage, result)
        self._target.post_message(StageCompleted(stage, result))

    def handle_stage_completed(self, stage: str, result: Any) -> None:
        """P1.3 hook — extract per-stage LLM usage from the
        ``result`` and fold it into the token counter.

        Exposed as a public method so callers can drive the
        token-update side-effect from a test without having to
        also build a Textual sink. The default
        :meth:`on_stage_complete` calls this helper before
        posting the message, so a wired
        :class:`SessionTokenCounter` updates on every stage
        completion automatically.
        """
        counter = self._token_counter
        if counter is None:
            return
        usage = _extract_usage(result)
        if usage:
            try:
                counter.add(usage)
            except Exception:
                pass

    def on_error(self, stage: str, error: BaseException) -> None:
        self._target.post_message(StageError(stage, error))

    # ----- optional streaming / fine-grained hooks -----------------

    def on_llm_chunk(self, stage: str, delta: str) -> None:
        self._target.post_message(LLMChunk(stage, delta))

    def on_stage_progress(self, stage: str, artifact: Any) -> None:
        self._target.post_message(StageProgress(stage, artifact))

    def on_retry(self, stage: str, attempt: int, error: BaseException) -> None:
        self._target.post_message(StageRetry(stage, attempt, error))

    def on_cost_estimate(self, estimate: Any) -> None:
        self._target.post_message(CostEstimate(estimate))


def _extract_usage(result: Any) -> dict[str, Any] | None:
    """Best-effort projection: pull a `{prompt, completion,
    total}`-shaped dict out of a stage result.

    MAGE's per-stage result objects vary by stage type — some
    carry ``.usage``, some ``.metrics["usage"]``, some are
    plain dicts. We probe the documented locations and return
    the first dict we find; non-dict / missing values yield
    ``None`` so the counter side-effect short-circuits."""
    if result is None:
        return None
    if isinstance(result, dict):
        usage = result.get("usage") or result.get("token_usage")
        if isinstance(usage, dict):
            return dict(usage)
        metrics = result.get("metrics")
        if isinstance(metrics, dict):
            inner = metrics.get("usage")
            if isinstance(inner, dict):
                return dict(inner)
        return None
    for attr in ("usage", "token_usage"):
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
    "CostEstimate",
    "LLMChunk",
    "MagePoster",
    "StageCompleted",
    "StageError",
    "StageProgress",
    "StageRetry",
    "StageStarted",
]
