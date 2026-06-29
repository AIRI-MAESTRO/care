"""Project a CARL ``ReasoningResult`` into a single user-
facing answer string.

Sibling of :mod:`care.mage_summary`. Where ``summarise_mage_result``
flattens MAGE generation *metadata* into a multi-line block, this
module flattens CARL *execution output* into the single string the
ChatScreen renders as the assistant's reply.

The contract is pragmatic — CARL's ``ReasoningResult`` shape varies
across step types (LLM, tool, MCP, transform, …) and across CARL
versions. We read a list of well-known fields in priority order,
fall back to the last entry in ``outputs`` when present, and finally
``str(result)`` so the user always sees *something* rather than a
blank assistant line.

The function is duck-typed so tests can pass plain dicts /
``SimpleNamespace`` stubs without importing ``mmar_carl``.
"""

from __future__ import annotations

from typing import Any


# Field-name priority chain. Order matters — earlier wins.
# Keep the list in sync with the duck-type chat-side fallback
# in :meth:`care.screens.chat.ChatScreen._format_carl_result`.
_KNOWN_STRING_FIELDS: tuple[str, ...] = (
    "final_output",
    "final_answer",
    "output",
    "answer",
    "summary",
    "text",
)


_DEFAULT_FALLBACK = "Chain executed (no textual output)."


def summarise_carl_result(result: Any) -> str:
    """Project ``result`` into a single answer string.

    Args:
        result: A CARL ``ReasoningResult`` (or a duck-typed
            equivalent — any object with one of the
            :data:`_KNOWN_STRING_FIELDS`, or a dict with one of
            those keys, or anything whose ``str()`` carries
            something useful).

    Returns:
        A trimmed string — never ``None``, never empty. Falls
        back to :data:`_DEFAULT_FALLBACK` only when every other
        avenue produced an empty string.
    """
    if result is None:
        return _DEFAULT_FALLBACK

    # 1. Attribute lookup on the well-known fields.
    for attr in _KNOWN_STRING_FIELDS:
        value = getattr(result, attr, None)
        text = _coerce_to_string(value)
        if text:
            return text

    # 2. Dict-style lookup (same key chain).
    if isinstance(result, dict):
        for key in _KNOWN_STRING_FIELDS:
            text = _coerce_to_string(result.get(key))
            if text:
                return text

    # 3. CARL ``ReasoningResult`` carries an ``outputs`` mapping
    #    keyed by step id. The terminal step's output is what the
    #    user actually wants — pull the last entry.
    outputs = _read_outputs(result)
    if outputs:
        last_value = list(outputs.values())[-1]
        text = _coerce_to_string(last_value)
        if text:
            return text
        # ``last_value`` itself may have a known string field
        # (e.g. an LLMStepResult with ``.text``). Recurse one
        # level to surface it.
        text = _coerce_to_string(_try_inner_fields(last_value))
        if text:
            return text

    # 4. Current CARL versions (mmar_carl >= 0.2) drop ``outputs``
    #    in favour of a flat ``step_results`` list. The terminal
    #    successful step's ``result`` field is the user's answer.
    #    Walk the list backwards so we skip any trailing
    #    failed-but-success=True row and surface the last
    #    meaningful payload.
    step_text = _read_terminal_step_result(result)
    if step_text:
        return step_text

    # 5. Last resort — ``str(result)`` so the user sees the raw
    #    repr rather than an empty line.
    text = str(result).strip()
    return text or _DEFAULT_FALLBACK


def _coerce_to_string(value: Any) -> str:
    """Return a non-empty string for ``value`` or an empty
    string. Only str-typed values pass through; non-strings
    return ``""`` so the priority chain keeps searching."""
    if isinstance(value, str):
        return value.strip()
    return ""


def _read_outputs(result: Any) -> dict[str, Any] | None:
    """Locate the step-outputs mapping on a CARL result. Tries
    attribute then dict-style. Returns ``None`` when missing /
    malformed."""
    outputs = getattr(result, "outputs", None)
    if isinstance(outputs, dict) and outputs:
        return outputs
    if isinstance(result, dict):
        outputs = result.get("outputs")
        if isinstance(outputs, dict) and outputs:
            return outputs
    return None


def _read_terminal_step_result(result: Any) -> str:
    """Return the terminal step's textual ``result`` from a CARL
    ``ReasoningResult.step_results`` list.

    Walks backwards so a trailing failed-but-success=True step
    doesn't bury the user's actual answer (we surface the last
    step whose ``success`` flag is truthy and whose ``result``
    coerces to a non-empty string). Falls back to JSON-stringifying
    ``result_data`` when ``result`` itself is empty — STRUCTURED_OUTPUT
    steps land their payload in the structured field with an
    empty string in ``result``.
    """
    step_results = getattr(result, "step_results", None)
    if not isinstance(step_results, list) or not step_results:
        if isinstance(result, dict):
            step_results = result.get("step_results")
            if not isinstance(step_results, list) or not step_results:
                return ""
        else:
            return ""
    for step in reversed(step_results):
        if _read_step_field(step, "success", default=True) is False:
            continue
        text = _coerce_to_string(_read_step_field(step, "result"))
        if text:
            return text
        # STRUCTURED_OUTPUT steps stash the payload in
        # `result_data` (dict / pydantic model) and leave
        # `result` empty. Fall back to a compact JSON dump so
        # the user at least sees the structured payload.
        text = _coerce_data_to_string(
            _read_step_field(step, "result_data"),
        )
        if text:
            return text
    return ""


def _read_step_field(step: Any, name: str, *, default: Any = None) -> Any:
    """Read ``name`` off ``step`` whether it's a dataclass-like
    object (attribute access) or a dict (key lookup). Tests pass
    dicts; the production CARL `StepExecutionResult` is a
    pydantic model — same call site supports both."""
    if isinstance(step, dict):
        if name in step:
            return step[name]
        return default
    return getattr(step, name, default)


def _coerce_data_to_string(value: Any) -> str:
    """Best-effort projection of ``result_data`` (typically a
    dict or pydantic model) into a readable string. Returns ""
    when there's nothing useful — the caller keeps searching."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        import json

        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return ""


def _try_inner_fields(value: Any) -> str:
    """One-level recursion into a step result so wrappers like
    ``LLMStepResult(text=...)`` surface their inner string."""
    for attr in _KNOWN_STRING_FIELDS:
        inner = getattr(value, attr, None)
        if isinstance(inner, str) and inner.strip():
            return inner
    if isinstance(value, dict):
        for key in _KNOWN_STRING_FIELDS:
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner
    return ""


__all__ = ["summarise_carl_result"]
