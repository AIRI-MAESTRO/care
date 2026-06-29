"""LLM-suggested chain titles (TODO §3 P2).

When the user saves a chain to Memory, the modal seeds the
title field from the artifact's existing ``title`` slot — which
is usually the first 60 chars of the prompt that produced the
chain. That's fine for personal notebooks; for a library
shared across teammates / services, a one-line summary of what
the chain DOES reads much better.

This module ships :func:`suggest_chain_title`: a tiny
``chat.completions`` wrapper that takes a chain dict + an
OpenAI-compatible client and returns a single-line summary
suitable for the ``name`` field on a saved chain entity.
"""

from __future__ import annotations

import logging
import re
from typing import Any

_log = logging.getLogger("care.runtime.chain_title")


# Cap the suggestion so it fits in Memory's name field + the
# Library DataTable's name column without truncation surprises.
MAX_TITLE_CHARS: int = 80

_SYSTEM_PROMPT = (
    "You are a concise assistant who names CARL reasoning "
    "chains. Given the chain's JSON, reply with a single line "
    "of plain text — at most 80 characters — that names the "
    "chain's purpose. No leading verbs like 'A chain that…'; "
    "use a noun phrase. No quotes, no markdown, no trailing "
    "punctuation."
)


def _build_user_prompt(chain: Any) -> str:
    """Render the chain dict in a compact form the model can
    skim. Keeps step names / titles / aims but drops the full
    `stage_action` / `example_reasoning` payloads so the
    prompt stays cheap."""
    steps = []
    raw_steps = (
        chain.get("steps")
        if isinstance(chain, dict) else None
    )
    if isinstance(raw_steps, list):
        for step in raw_steps:
            if not isinstance(step, dict):
                continue
            title = (
                step.get("title")
                or step.get("name")
                or step.get("step_title")
                or ""
            )
            aim = (
                step.get("aim")
                or step.get("description")
                or step.get("stage_action")
                or ""
            )
            steps.append({
                "title": str(title)[:80],
                "aim": str(aim)[:160],
                "type": str(
                    step.get("step_type") or step.get("type") or "",
                ),
            })
    payload = {
        "name": str(
            chain.get("name") if isinstance(chain, dict) else "",
        )[:80],
        "steps": steps[:12],
    }
    import json

    return (
        "Chain payload (truncated):\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nReply with the chain's name on a single line."
    )


def _clean_suggestion(raw: str) -> str:
    """Strip surrounding whitespace + quotes, collapse runs of
    whitespace, drop trailing punctuation, cap at
    :data:`MAX_TITLE_CHARS`. Returns an empty string when the
    cleaned text would be empty."""
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    # Some providers wrap in ``\`code\``  / `"quoted"` — peel
    # symmetric wraps once.
    while len(text) >= 2 and text[0] == text[-1] and text[0] in '"\'`':
        text = text[1:-1].strip()
    # Collapse internal newlines / repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    # Drop trailing punctuation that doesn't add meaning.
    text = re.sub(r"[\s.;,:!?]+$", "", text).strip()
    if not text:
        return ""
    return text[:MAX_TITLE_CHARS].strip()


def suggest_chain_title(
    chain: Any,
    *,
    client: Any,
    model: str,
    fallback: str = "",
) -> str:
    """Return a one-line LLM-suggested name for ``chain``.

    Args:
        chain: Chain dict (the same shape ``ReasoningChain.to_dict()``
            produces). Non-dict input → ``fallback``.
        client: OpenAI-compatible sync client with
            ``chat.completions.create``. The function calls it
            with ``temperature=0.2`` (deterministic-ish) +
            ``max_tokens=40`` (the spec caps suggestions at
            80 chars; 40 tokens is comfortably above that).
        model: Model id to pass to ``chat.completions.create``.
            Required (empty string raises ValueError).
        fallback: Returned verbatim when the call can't produce
            a useful suggestion (missing client, exception,
            empty response, model returned only whitespace).
            Defaults to empty.

    Returns:
        Cleaned one-line title, or ``fallback`` on any failure.
        NEVER raises — callers can wrap-and-use without try/except.
    """
    if not isinstance(chain, dict):
        return fallback
    if client is None or not model:
        return fallback
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(chain),
                },
            ],
            temperature=0.2,
            max_tokens=40,
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug(
            "suggest_chain_title LLM call failed: %s", exc,
        )
        return fallback
    try:
        raw = response.choices[0].message.content or ""
    except Exception:
        return fallback
    cleaned = _clean_suggestion(raw)
    return cleaned or fallback


__all__ = [
    "MAX_TITLE_CHARS",
    "suggest_chain_title",
    "_clean_suggestion",
    "_build_user_prompt",
]
