"""CARL long-term memory (LTM) — always-on injection + conservative post-turn
save-decision.

CARE attaches CARL's native :class:`mmar_carl.JsonFileLTM` to every run and
uses it two ways:

1. **Always inject** — before answering the user AND before planning a chain,
   a recalled digest of what's remembered (``recall_digest``) is prepended to
   the prompt, so every reply/plan is personalised by durable context.
2. **Post-turn save-decision** — after the reply, one cheap LLM call decides
   whether anything *durable about the user* is worth remembering
   (``save_from_turn``). It is deliberately CONSERVATIVE: only stable
   user-level facts (role, preferences, recurring constraints, ongoing
   projects), never one-off task data; identical facts are skipped (dedup) and
   a changed fact supersedes the old value under the same key.

The store is keyed by ``session_id`` (``CARE_CONTEXT__LTM_SESSION_ID``), so
contexts sharing the id share memory. Everything here is best-effort — a
missing/broken CARL or LLM never raises into the turn; memory simply no-ops.

Cross-platform: the store dir comes from ``ContextConfig.ltm_dir`` (default
``~/.config/care/ltm``) resolved via :meth:`pathlib.Path.expanduser`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

_log = logging.getLogger("care.memory_ltm")

__all__ = [
    "MEMORY_MERGE_SYSTEM_PROMPT",
    "SAVE_DECISION_SYSTEM_PROMPT",
    "apply_facts",
    "build_ltm",
    "decide_facts",
    "format_saved",
    "ltm_session_id",
    "merge_into_memory",
    "recall_digest",
    "remember_text",
    "save_from_turn",
]

# A sync ``(system_prompt, user_prompt) -> raw_response`` callable. CARE builds
# it from its OpenAI client; tests pass a stub. Decoupling the LLM here keeps
# the engine deterministically testable.
Complete = Callable[[str, str], str]

_DEFAULT_DIGEST_HEADER = "## What I remember about you (long-term memory)"


def ltm_session_id(care_config: Any) -> str:
    """The LTM scope key from ``config.context.ltm_session_id`` (``"default"``
    fallback). Contexts sharing it share memory."""
    ctx = getattr(care_config, "context", None)
    return (getattr(ctx, "ltm_session_id", None) or "default") if ctx else "default"


def build_ltm(care_config: Any) -> Any | None:
    """Construct the configured ``JsonFileLTM``, or ``None`` when LTM is
    disabled / CARL is unavailable. Never raises — memory is never load-bearing.
    """
    ctx = getattr(care_config, "context", None)
    if ctx is None or not getattr(ctx, "ltm_enabled", False):
        return None
    try:
        from mmar_carl import JsonFileLTM

        # ``CARE_CONTEXT__LTM_DIR`` overrides the configured dir (env is CARE's
        # highest-precedence layer; also lets tests redirect the store).
        ltm_dir_raw = os.environ.get("CARE_CONTEXT__LTM_DIR") or str(
            getattr(ctx, "ltm_dir", "~/.config/care/ltm"),
        )
        ltm_dir = Path(ltm_dir_raw).expanduser()
        return JsonFileLTM(ltm_dir)
    except Exception as exc:  # noqa: BLE001
        _log.info("LTM unavailable (%s) — continuing without long-term memory", exc)
        return None


def recall_digest(
    ltm: Any,
    session_id: str,
    *,
    query: str = "",
    max_chars: int = 2000,
    header: str = _DEFAULT_DIGEST_HEADER,
) -> str:
    """Render a compact digest of everything in LTM for prompt injection.

    Enumerates stored facts (``keys()`` + ``retrieve()``) into a bullet block,
    capped at ``max_chars``. ``""`` when LTM is empty/absent or ``max_chars``
    is 0. ``query`` is accepted for future relevance-ranking; today the store
    is user-scoped + small, so we inject all of it.
    """
    if ltm is None or max_chars <= 0:
        return ""
    try:
        keys = list(ltm.keys(session_id=session_id))
    except Exception:  # noqa: BLE001
        return ""
    lines: list[str] = []
    for key in keys:
        try:
            value = ltm.retrieve(key, session_id=session_id)
        except Exception:  # noqa: BLE001
            continue
        if value not in (None, ""):
            lines.append(f"- {key}: {value}")
    if not lines:
        return ""
    block = header + "\n" + "\n".join(lines)
    if len(block) > max_chars:
        block = block[: max_chars].rstrip() + " …"
    return block


SAVE_DECISION_SYSTEM_PROMPT = """You maintain CARE's durable long-term memory ABOUT THE USER — facts worth recalling in EVERY future session.

SAVE only DURABLE, USER-LEVEL facts that the user stated or clearly implied:
- the user's role / identity / domain of work
- stable preferences (language, tone, output formats, preferred tools/models)
- recurring constraints or requirements they keep asking for
- ongoing projects / goals / key standing context

NEVER save:
- one-off task details, transient inputs, or the specific answer just produced
- anything already in CURRENT MEMORY with the same meaning
- speculation — only what the user actually expressed

Be CONSERVATIVE: when in doubt, save nothing. Most turns save nothing.

If a fact CHANGED, reuse its existing key so the new value supersedes the old.

Return STRICT JSON only:
{"save": true|false, "facts": [{"key": "snake_case_id", "value": "one concise sentence"}]}
- key: short stable snake_case id (e.g. "role", "preferred_language", "project_acme").
- value: one concise sentence.
- Nothing durable this turn -> {"save": false, "facts": []}."""


def decide_facts(
    complete: Complete,
    *,
    query: str,
    answer: str = "",
    existing_digest: str = "",
) -> dict[str, Any]:
    """Run the save-decision LLM call → ``{"save": bool, "facts": [...]}``.

    Best-effort: any failure or unparseable response → ``{"save": False,
    "facts": []}`` (the turn is never blocked by memory).
    """
    user = (
        f"CURRENT MEMORY:\n{existing_digest or '(empty)'}\n\n"
        f"USER MESSAGE:\n{query}\n\n"
        f"CARE'S ANSWER (context only — do NOT store the answer itself):\n"
        f"{(answer or '')[:1500]}"
    )
    try:
        raw = complete(SAVE_DECISION_SYSTEM_PROMPT, user)
        data = json.loads(_extract_json(raw))
    except Exception as exc:  # noqa: BLE001
        _log.info("save-decision failed (%s) — saving nothing", exc)
        return {"save": False, "facts": []}
    if not isinstance(data, dict) or not data.get("save"):
        return {"save": False, "facts": []}
    facts = data.get("facts")
    return {"save": True, "facts": facts if isinstance(facts, list) else []}


def apply_facts(ltm: Any, session_id: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist decided facts to LTM. Dedups (identical key+value → skip) and
    supersedes (existing key, new value → overwrite). Returns the facts
    actually written (each tagged ``superseded``). Best-effort per fact."""
    if ltm is None or not facts:
        return []
    saved: list[dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        key = str(fact.get("key") or "").strip()
        value = str(fact.get("value") or "").strip()
        if not key or not value:
            continue
        try:
            existing = ltm.retrieve(key, session_id=session_id)
        except Exception:  # noqa: BLE001
            existing = None
        if existing == value:
            continue  # already remembered, identically — dedup
        try:
            ltm.store(key, value, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            _log.info("LTM store failed for %r: %s", key, exc)
            continue
        saved.append({"key": key, "value": value, "superseded": existing not in (None, "")})
    return saved


def save_from_turn(
    ltm: Any,
    session_id: str,
    *,
    query: str,
    answer: str = "",
    complete: Complete,
    existing_digest: str = "",
) -> list[dict[str, Any]]:
    """End-to-end post-turn save: decide (LLM) → apply (dedup/supersede/store).
    Returns the facts written (empty when nothing durable). Never raises."""
    if ltm is None:
        return []
    decision = decide_facts(
        complete, query=query, answer=answer, existing_digest=existing_digest,
    )
    if not decision.get("save"):
        return []
    return apply_facts(ltm, session_id, decision.get("facts") or [])


def format_saved(saved: list[dict[str, Any]]) -> str:
    """A one-line ``🧠 remembered: …`` summary for the chat, or ``""``."""
    if not saved:
        return ""
    return "🧠 remembered: " + ", ".join(str(f.get("key", "?")) for f in saved)


# --------------------------------------------------------------------------- #
#  Explicit "remember this" — `#…` / `/remember` (P6.8 hashtag-to-memory)       #
# --------------------------------------------------------------------------- #

MEMORY_MERGE_SYSTEM_PROMPT = """The user EXPLICITLY asked to remember the note below in CARE's durable long-term memory about them. Save it — but intelligently:

- Extract the durable fact(s) from the note and adapt them into concise memory entries. Do NOT lose information the user gave.
- Reconcile with CURRENT MEMORY: if the note UPDATES or CONTRADICTS an existing fact, REUSE that fact's exact key so the new value SUPERSEDES the old — do not create a duplicate.
- Use a NEW snake_case key only for genuinely new facts.

Return STRICT JSON only:
{"facts": [{"key": "snake_case_id", "value": "one concise fact"}]}
Always return at least one fact (the user asked to remember this)."""


def merge_into_memory(
    complete: Complete,
    *,
    content: str,
    existing_digest: str = "",
) -> list[dict[str, Any]]:
    """One LLM call that adapts an explicit "remember this" note into durable
    fact(s), reconciling with existing memory (reuse a key to supersede a
    stale/contradictory fact). Returns the facts to upsert — empty ONLY on
    failure (the caller then falls back so nothing is lost). Best-effort."""
    user = (
        f"CURRENT MEMORY:\n{existing_digest or '(empty)'}\n\n"
        f"NOTE TO REMEMBER:\n{content}"
    )
    try:
        raw = complete(MEMORY_MERGE_SYSTEM_PROMPT, user)
        data = json.loads(_extract_json(raw))
    except Exception as exc:  # noqa: BLE001
        _log.info("memory merge failed (%s)", exc)
        return []
    facts = data.get("facts") if isinstance(data, dict) else None
    return facts if isinstance(facts, list) else []


def remember_text(
    ltm: Any,
    session_id: str,
    *,
    content: str,
    complete: Complete,
    existing_digest: str = "",
) -> list[dict[str, Any]]:
    """Persist an explicit "remember this" note: LLM-merge → apply (dedup /
    supersede). If the merge yields nothing (LLM down), FALL BACK to storing the
    raw note under a derived key so an explicit request is never silently lost."""
    if ltm is None or not (content or "").strip():
        return []
    facts = merge_into_memory(complete, content=content, existing_digest=existing_digest)
    if not facts:
        facts = [{"key": _note_key(content), "value": content.strip()[:500]}]
    return apply_facts(ltm, session_id, facts)


def _note_key(content: str) -> str:
    """A stable ``note_<slug>`` key from a note's first words (fallback store)."""
    words = re.findall(r"[a-zA-Zа-яёА-ЯЁ0-9]+", content.lower())[:4]
    slug = "_".join(words)[:40].strip("_")
    return f"note_{slug}" if slug else "note"


def _extract_json(raw: str) -> str:
    """Pull the outermost ``{…}`` object out of a possibly fenced/prefixed LLM
    response so ``json.loads`` succeeds on chatty models."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
    i, j = s.find("{"), s.rfind("}")
    return s[i : j + 1] if i != -1 and j > i else s
