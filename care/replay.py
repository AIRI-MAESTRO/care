"""Replay mode — step-through inspection of a past run (TODO §5 P2).

CARE saves every chain run as a `memory_card` carrying the
serialised :class:`ReasoningResult` (and optionally a
:class:`RunRecord` wrapper). The future "Replay" screen
loads that blob and lets the user step through it like a
debugger — see each step's prompt, output, history snapshot,
profiling, success/skipped flag.

This module owns the **data + navigation** layer behind the
replay UI:

* :class:`ReplayStep` — one step in the replay, frozen so the
  view doesn't have to defensively copy. Carries the
  user-visible fields (number, title, type, result, error,
  history snapshot, profiling, success/skipped).
* :class:`ReplaySession` — ordered collection + navigation
  cursor. ``current()`` returns the current step;
  ``next()`` / ``previous()`` / ``seek(idx)`` move the
  cursor; bounds are checked.
* :func:`load_replay(source)` — duck-typed entry point.
  Accepts a `ReasoningResult`-like, a `RunRecord`-like, the
  result/record dict directly (e.g. fresh out of Memory), or
  a JSON string.

The session is **mutable on the cursor only**. Steps + chain-
level metadata stay frozen so the UI can read them safely
across coroutines. The UI calls ``session.next()`` etc. to
update the cursor, then re-reads ``session.current()`` for the
display.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


_MAX_RESULT_PREVIEW_CHARS = 4000
"""Cap the per-step ``result`` projection at ~4KB so a runaway
LLM output doesn't blow the replay session into RAM. The full
content stays on the raw step dict; the preview is what the UI
defaults to."""


@dataclass(frozen=True)
class ReplayStep:
    """One step in a replay session.

    Frozen so the screen can pass it across messages without
    defensive copies. ``history_snapshot`` is a list of strings
    matching CARL's ``StepExecutionResult.updated_history``;
    the future replay UI renders it as a collapsible "history
    after this step" panel.

    Fields:
        step_number: 1-indexed step number.
        step_title: Human-readable name from the chain.
        step_type: CARL step-type string (``"llm"``, ``"tool"``,
            ``"mcp"``, ...).
        result_preview: First N chars of the step's `result`
            string (capped at ~4KB so a big LLM dump doesn't
            balloon the session).
        result_truncated: ``True`` when `result_preview` is a
            truncated view of the original.
        result_data: Structured `result_data` (for non-LLM
            steps). Pass-through from CARL.
        success: Did the step succeed?
        skipped: Did the step get skipped (conditional routing)?
        error_message: Present when ``success=False``.
        execution_time_s: Wall-clock duration.
        history_snapshot: ``updated_history`` after this step
            (full list).
        profiling: Free-form profiling dict —
            ``history_bytes_added`` / ``memory_bytes_after`` /
            ``batch_index`` / etc. Same shape `StepProfile`
            (§5 P1) reads.
        model: LLM model the step used (None for non-LLM steps).
        token_usage: Per-step token dict.
    """

    step_number: int
    step_title: str
    step_type: str
    result_preview: str = ""
    result_truncated: bool = False
    result_data: Any = None
    success: bool = True
    skipped: bool = False
    error_message: str | None = None
    execution_time_s: float | None = None
    history_snapshot: tuple[str, ...] = field(default_factory=tuple)
    profiling: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    token_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplaySession:
    """Ordered :class:`ReplayStep` collection with a cursor.

    Not frozen because the cursor moves — but every other field
    is treated as read-only. Use :meth:`next` / :meth:`previous`
    / :meth:`seek` to navigate; :meth:`current` reads the step
    under the cursor.

    Fields:
        steps: Ordered tuple of every step in the run.
        chain_id: When the source ``RunRecord`` carries a
            chain entity id, stored here for "open in library"
            actions.
        chain_title: When the source carries the chain's
            display name, stored here for screen headers.
        total_execution_time_s: Chain-level wall-clock total.
        token_usage: Chain-level token aggregate.
        final_answer: The chain's final answer string when
            present on the source (`ReasoningResult.final_answer`).
        cursor: 0-indexed position. Starts at ``0`` when there
            are steps, ``-1`` otherwise.
    """

    steps: tuple[ReplayStep, ...] = field(default_factory=tuple)
    chain_id: str | None = None
    chain_title: str = ""
    total_execution_time_s: float = 0.0
    token_usage: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    cursor: int = -1

    def __post_init__(self) -> None:
        if self.steps and self.cursor < 0:
            self.cursor = 0
        elif not self.steps:
            self.cursor = -1

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def is_empty(self) -> bool:
        return not self.steps

    def current(self) -> ReplayStep | None:
        """Step at the cursor, or ``None`` when empty."""
        if not self.steps or self.cursor < 0 or self.cursor >= len(self.steps):
            return None
        return self.steps[self.cursor]

    def at(self, index: int) -> ReplayStep:
        """Step at an explicit 0-indexed position. Raises
        ``IndexError`` on out-of-bounds — use :meth:`seek` for
        clamped navigation."""
        return self.steps[index]

    def step_titles(self) -> tuple[str, ...]:
        """Display-friendly titles in order, useful for the
        future stepper sidebar."""
        return tuple(s.step_title for s in self.steps)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def next(self) -> ReplayStep | None:
        """Advance the cursor by one. Returns the new current
        step (or ``None`` when the cursor is at the end). Stays
        clamped at the last step — calling :meth:`next` past
        the end is a no-op."""
        if not self.steps:
            return None
        if self.cursor < len(self.steps) - 1:
            self.cursor += 1
        return self.current()

    def previous(self) -> ReplayStep | None:
        """Move the cursor back by one. Clamped at ``0``."""
        if not self.steps:
            return None
        if self.cursor > 0:
            self.cursor -= 1
        return self.current()

    def seek(self, index: int) -> ReplayStep | None:
        """Move the cursor to ``index``. Negative indices count
        from the end (Python convention). Clamped to
        ``[0, step_count - 1]`` — out-of-bounds inputs land on
        the nearest endpoint rather than raising."""
        if not self.steps:
            return None
        if index < 0:
            index = max(0, len(self.steps) + index)
        if index >= len(self.steps):
            index = len(self.steps) - 1
        self.cursor = index
        return self.current()

    def restart(self) -> ReplayStep | None:
        """Reset to step 0."""
        return self.seek(0)

    @property
    def at_end(self) -> bool:
        return self.cursor >= len(self.steps) - 1

    @property
    def at_start(self) -> bool:
        return self.cursor <= 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def format_text(self) -> str:
        """Header + current-step block. The future replay
        screen renders something richer; this stays
        text-friendly for CLI / `care replay` output."""
        if self.is_empty:
            return "no steps to replay"
        header_bits: list[str] = []
        if self.chain_title:
            header_bits.append(f"chain: {self.chain_title}")
        if self.chain_id:
            header_bits.append(f"id: {self.chain_id}")
        header_bits.append(
            f"step {self.cursor + 1}/{self.step_count}"
        )
        if self.total_execution_time_s:
            header_bits.append(f"total {self.total_execution_time_s:.2f}s")
        lines = [" · ".join(header_bits), ""]
        cur = self.current()
        if cur is None:
            return "\n".join(lines)
        badge = ""
        if cur.skipped:
            badge = " [SKIPPED]"
        elif not cur.success:
            badge = " [FAILED]"
        lines.append(
            f"#{cur.step_number} {cur.step_title} ({cur.step_type}){badge}"
        )
        if cur.execution_time_s is not None:
            lines.append(f"  time: {cur.execution_time_s:.3f}s")
        if cur.error_message:
            lines.append(f"  error: {cur.error_message}")
        if cur.model:
            lines.append(f"  model: {cur.model}")
        if cur.result_preview:
            preview = cur.result_preview
            if cur.result_truncated:
                preview = preview + f" […truncated at {_MAX_RESULT_PREVIEW_CHARS} chars]"
            lines.append("  result:")
            for line in preview.splitlines():
                lines.append(f"    {line}")
        if cur.history_snapshot:
            lines.append(
                f"  history snapshot: {len(cur.history_snapshot)} entr"
                f"{'y' if len(cur.history_snapshot) == 1 else 'ies'}"
            )
        return "\n".join(lines)


class ReplayError(RuntimeError):
    """Raised when the replay loader can't parse the input —
    malformed JSON, missing ``step_results`` field, etc."""


def load_replay(source: Any) -> ReplaySession:
    """Build a :class:`ReplaySession` from a saved run.

    Args:
        source: One of five accepted shapes:

            - A :class:`mmar_carl.ReasoningResult`-like (has
              ``step_results`` attribute).
            - A :class:`mmar_carl.RunRecord`-like (has
              ``result`` attribute pointing at a
              ReasoningResult-like, plus chain metadata).
            - The result/record dict directly (e.g.
              ``ReasoningResult.to_dict()`` output).
            - A JSON string (decoded via :func:`json.loads`).
            - ``None`` — returns an empty session.

    Returns:
        :class:`ReplaySession`. Empty session for ``None`` /
        unrecognised inputs. Raises :class:`ReplayError` only
        when a JSON string fails to parse.
    """
    if source is None:
        return ReplaySession()

    # Decode JSON if given as a string.
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ReplayError(
                f"replay JSON failed to parse at line {exc.lineno}, "
                f"col {exc.colno}: {exc.msg}"
            ) from exc

    # `RunRecord`-shaped — has `.result` carrying the
    # `ReasoningResult`. Pull the chain metadata off the outer
    # record + the steps off the inner result.
    chain_id: str | None = None
    chain_title = ""
    inner = source

    if isinstance(source, dict):
        if "result" in source and isinstance(source.get("result"), dict):
            inner = source["result"]
            chain_id = _opt_str(source.get("chain_id"))
            chain_title = str(source.get("chain_title") or "")
        elif "step_results" not in source:
            # Bare dict but doesn't carry steps — empty session
            # rather than raising (caller may have passed an
            # adjacent metadata blob by mistake).
            return ReplaySession()
    else:
        # Attribute path — RunRecord-like.
        nested_result = getattr(source, "result", None)
        if nested_result is not None and not _looks_like_reasoning_result(source):
            chain_id = _opt_str(getattr(source, "chain_id", None))
            chain_title = str(getattr(source, "chain_title", "") or "")
            inner = nested_result

    step_results = _get(inner, "step_results", None) or ()
    steps_out: list[ReplayStep] = []
    for sr in step_results:
        step = _project_step(sr)
        if step is not None:
            steps_out.append(step)

    return ReplaySession(
        steps=tuple(steps_out),
        chain_id=chain_id,
        chain_title=chain_title,
        total_execution_time_s=_opt_float(
            _get(inner, "total_execution_time", 0.0), 0.0
        ),
        token_usage=dict(_get(inner, "token_usage", {}) or {})
        if isinstance(_get(inner, "token_usage", {}), dict)
        else {},
        final_answer=str(_get(inner, "final_answer", "") or ""),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_reasoning_result(source: Any) -> bool:
    """Disambiguate `RunRecord` (has `.result`) from
    `ReasoningResult` (has `.step_results` directly)."""
    return hasattr(source, "step_results")


def _project_step(sr: Any) -> ReplayStep | None:
    """Project one StepExecutionResult-like into a
    :class:`ReplayStep`. Returns ``None`` when the input is
    too malformed to make sense of."""
    if sr is None:
        return None
    step_number = _opt_int(_get(sr, "step_number", None), -1)
    step_title = str(_get(sr, "step_title", "") or "")
    if step_number < 0 and not step_title:
        # Couldn't identify the step at all — drop it.
        return None

    step_type = _step_type_to_str(_get(sr, "step_type", None))
    result_raw = _get(sr, "result", "") or ""
    if not isinstance(result_raw, str):
        result_raw = str(result_raw)
    truncated = len(result_raw) > _MAX_RESULT_PREVIEW_CHARS
    result_preview = (
        result_raw[:_MAX_RESULT_PREVIEW_CHARS] if truncated else result_raw
    )
    history = _get(sr, "updated_history", ()) or ()
    if not isinstance(history, (list, tuple)):
        history = ()
    profiling = _get(sr, "profiling", {}) or {}
    if not isinstance(profiling, dict):
        profiling = {}
    token_usage = _get(sr, "token_usage", {}) or {}
    if not isinstance(token_usage, dict):
        token_usage = {}

    return ReplayStep(
        step_number=max(step_number, 0),
        step_title=step_title,
        step_type=step_type,
        result_preview=result_preview,
        result_truncated=truncated,
        result_data=_get(sr, "result_data", None),
        success=bool(_get(sr, "success", True)),
        skipped=bool(_get(sr, "skipped", False)),
        error_message=_opt_str(_get(sr, "error_message", None)),
        execution_time_s=_opt_float_or_none(_get(sr, "execution_time", None)),
        history_snapshot=tuple(str(h) for h in history),
        profiling=dict(profiling),
        model=_opt_str(_get(sr, "model", None)),
        token_usage=dict(token_usage),
    )


def _get(source: Any, name: str, default: Any) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _step_type_to_str(step_type: Any) -> str:
    if step_type is None:
        return "unknown"
    if hasattr(step_type, "value"):
        return str(step_type.value)
    return str(step_type)


def _opt_int(value: Any, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _opt_float(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _opt_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


__all__ = [
    "ReplayError",
    "ReplaySession",
    "ReplayStep",
    "load_replay",
]
