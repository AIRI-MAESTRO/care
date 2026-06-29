"""Deterministic skill selection — guarantee a chain actually uses a packaged
skill when the user asks for a file/document (or explicitly requests one),
instead of relying on a weak planner LLM to choose an ``agent_skill`` step.

Two triggers:

* **Explicit** — the user says ``/skill pptx`` / «используй скилл pptx» /
  ``use the pptx skill``. Always honoured.
* **Implicit** — the task asks to PRODUCE a file (a produce-verb + a file-type
  keyword like *pptx / презентация / docx / xlsx / pdf*). Gated on the verb so
  "explain the pdf format" doesn't trigger.

When a skill is required but the generated chain has no matching ``agent_skill``
step, :func:`ensure_skill_step` rewrites the chain's final step into one (using
the canonical registry URI), so the chain produces the real artifact. CARL then
runs the skill and P6.5's sink saves the output file.

Everything is best-effort + deterministic — no LLM, no network.
"""

from __future__ import annotations

import logging
import re
from typing import Any

_log = logging.getLogger("care.skill_enforcement")

__all__ = ["detect_requested_skill", "ensure_skill_step"]

# Alias → canonical registry skill name.
_SKILL_ALIASES: dict[str, str] = {
    "pptx": "pptx", "powerpoint": "pptx", "presentation": "pptx", "keynote": "pptx",
    "docx": "docx", "word": "docx", "msword": "docx",
    "pdf": "pdf",
    "xlsx": "xlsx", "excel": "xlsx", "spreadsheet": "xlsx",
}

# Implicit file-type keywords (incl. RU) → canonical registry skill name.
_FILE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pptx": ("pptx", "powerpoint", "презентац", "слайд", "deck"),
    "docx": ("docx", "ворд", "word document", "вордовск", ".docx"),
    "xlsx": ("xlsx", "excel", "эксель", "spreadsheet", "таблиц", ".xlsx"),
    "pdf": ("pdf", ".pdf"),
}

# Produce intent (EN + RU stems) — gates the implicit trigger.
_PRODUCE_RE = re.compile(
    r"\b(make|create|generate|build|produce|prepare|export|render|write|"
    r"сдела|созда|сгенер|постро|подготов|сформир|свёрст|сверст|оформ|выгруз)",
    re.IGNORECASE,
)

# Explicit request, name AFTER the marker: "/skill X", «используй скилл X»,
# "use [the] skill X".
_EXPLICIT_AFTER_RE = re.compile(
    r"(?:/skill\s+|использ\w*\s+скилл\w*\s+|use\s+(?:the\s+)?skill\s+)"
    r"([a-zA-Zа-яёА-ЯЁ][\w-]*)",
    re.IGNORECASE,
)
# Explicit request, name BEFORE "skill": "use the powerpoint skill" — restricted
# to known skill words so "use this skill" doesn't falsely trigger.
_EXPLICIT_BEFORE_RE = re.compile(
    r"\buse\s+(?:the\s+)?"
    r"(pptx|powerpoint|presentation|docx|word|pdf|xlsx|excel|spreadsheet)\s+skill\b",
    re.IGNORECASE,
)


def detect_requested_skill(query: str) -> str | None:
    """Canonical registry skill name the user wants, or ``None``.

    Explicit ``/skill`` / «используй скилл» / ``use the … skill`` wins; otherwise
    a file-type keyword PAIRED with a produce verb (so a passing mention of a
    format doesn't force a skill).
    """
    if not query:
        return None
    explicit = _EXPLICIT_AFTER_RE.search(query) or _EXPLICIT_BEFORE_RE.search(query)
    if explicit:
        name = explicit.group(1).lower()
        return _SKILL_ALIASES.get(name, name)
    low = query.lower()
    if not _PRODUCE_RE.search(low):
        return None
    for skill, keywords in _FILE_KEYWORDS.items():
        if any(k in low for k in keywords):
            return skill
    return None


def _skill_uri(skill_name: str) -> str | None:
    """Canonical URI for a registered skill, or ``None`` when unavailable."""
    try:
        from mmar_mage.skills import default_registry
    except Exception:  # noqa: BLE001
        return None
    known = default_registry.get(skill_name)
    return known.uri if known is not None else None


# Skills whose deliverable is a binary file (not text). For these we force the
# CARL LLM_AGENT loop to actually write a file instead of accepting prose.
_FILE_SKILLS = frozenset({"pptx", "docx", "xlsx", "pdf"})


def _apply_file_production(cfg: dict[str, Any], skill_name: str) -> None:
    """Make a file-producing skill actually emit its file.

    Sets the CARL knobs so a weak model can't "describe" the artifact instead
    of creating it:

    * ``execution_mode=llm_agent`` — the iterative tool-calling mode that can
      run the skill's scripts and write to ``/workspace/out/``.
    * ``require_output_file=True`` — the loop pushes back (once) if it finishes
      with no file written, and flags ``no_output_file`` if it still fails.
    * ``persist_workspace=True`` — the produced file survives cleanup so CARE
      can copy it out and show the path.

    Also rewrites the task so the deliverable is unambiguously the file. No-op
    for non-file skills. Mutates ``cfg`` in place.
    """
    if skill_name not in _FILE_SKILLS:
        return
    cfg["execution_mode"] = "llm_agent"
    cfg["require_output_file"] = True
    cfg["persist_workspace"] = True
    base = str(cfg.get("task") or "").strip()
    cfg["task"] = (
        f"{base}\n\n"
        f"Produce an ACTUAL .{skill_name} file: use run_script to execute the "
        f"skill's bundled scripts and write the finished file to /workspace/out/. "
        f"Do NOT return the content as text — the deliverable is the file itself."
    ).strip()


def _find_skill_step(
    steps: list[dict[str, Any]], skill_name: str,
) -> dict[str, Any] | None:
    """The first ``agent_skill`` step already referencing the skill, or ``None``."""
    for step in steps:
        if not isinstance(step, dict) or step.get("step_type") != "agent_skill":
            continue
        cfg = step.get("step_config") if isinstance(step.get("step_config"), dict) else {}
        ref = f"{cfg.get('skill', '')} {step.get('skill', '')}".lower()
        if skill_name in ref:
            return step
    return None


def ensure_skill_step(
    chain_dict: dict[str, Any], skill_name: str | None,
) -> dict[str, Any]:
    """Guarantee the chain uses ``skill_name``: if no ``agent_skill`` step
    already does, rewrite the FINAL step into one (CARL-nested ``step_config``
    with the canonical registry URI), feeding the prior steps' output in.

    Returns the (possibly mutated) ``chain_dict``. No-op when ``skill_name`` is
    falsy, the skill isn't registered, the chain already uses it, or the shape
    is unexpected. Never raises.
    """
    if not skill_name:
        return chain_dict
    try:
        steps = chain_dict.get("steps") if isinstance(chain_dict, dict) else None
        if not isinstance(steps, list) or not steps:
            return chain_dict
        uri = _skill_uri(skill_name)
        if not uri:
            _log.info("skill %r not in registry — leaving chain unchanged", skill_name)
            return chain_dict
        existing = _find_skill_step(steps, skill_name)
        if existing is not None:
            # The planner already chose the skill — but it may have named it by
            # the BARE name (e.g. "pptx"), which CARL can't resolve. Replace any
            # non-URI ref with the canonical registry URI so it loads.
            cfg = existing.get("step_config")
            cfg = cfg if isinstance(cfg, dict) else {}
            ref = str(cfg.get("skill") or existing.get("skill") or "")
            if "://" not in ref:
                cfg["skill"] = uri
                existing.pop("skill", None)
                _log.info("normalised bare skill ref → %s", uri)
            # Force the skill to actually emit its file (nudge + persist).
            _apply_file_production(cfg, skill_name)
            existing["step_config"] = cfg
            return chain_dict
        last = steps[-1]
        if not isinstance(last, dict):
            return chain_dict
        task = (
            last.get("aim")
            or last.get("title")
            or f"Create the requested {skill_name} file from the provided content."
        )
        last["step_type"] = "agent_skill"
        cfg = {
            "skill": uri,
            "task": task,
            "execution_mode": "llm_agent",
            # feed the immediately-preceding step's output to the skill
            "input_mapping": {"content": "$history[-1]"},
        }
        # force the skill to actually write the file, not just describe it
        _apply_file_production(cfg, skill_name)
        last["step_config"] = cfg
        # drop a stale llm-config so the agent_skill step is clean
        last.pop("llm_config", None)
        _log.info(
            "enforced agent_skill(%s) on final step %s",
            skill_name, last.get("number", "?"),
        )
    except Exception as exc:  # noqa: BLE001
        _log.info("skill enforcement skipped: %s", exc)
    return chain_dict
