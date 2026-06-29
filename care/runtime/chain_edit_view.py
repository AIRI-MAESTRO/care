"""Render an NL-edit plan / preview as transcript lines.

The edit flow (ChatScreen ``/revise``, library "Revise (AI)") needs to show the
user WHAT changed before they confirm a save: a plain-language summary, the list
of edits with rationales, and a compact before/after step delta. This module
turns a ``MAGEEditResult`` (duck-typed) into a list of strings so screens can
post them via ``_post_line`` without embedding formatting logic.

Pure + dependency-light (operates on dicts / duck-typed results) so it is
trivially unit-testable and reusable across screens.
"""

from __future__ import annotations

from typing import Any


def _steps(chain: Any) -> list[dict[str, Any]]:
    if not isinstance(chain, dict):
        return []
    return [s for s in chain.get("steps", []) if isinstance(s, dict)]


def _step_label(step: dict[str, Any]) -> str:
    title = step.get("title") or step.get("aim") or "(untitled)"
    return f"{title} [{step.get('step_type', 'llm')}]"


def render_step_delta(before: Any, after: Any) -> list[str]:
    """Compact before/after delta: name change, step-count change, after-steps.

    Step numbers are reassigned after structural edits, so this is an honest
    *summary* (name + count + final step list), not a line-level diff.
    """
    lines: list[str] = []
    b_name = before.get("name") if isinstance(before, dict) else None
    a_name = after.get("name") if isinstance(after, dict) else None
    if b_name != a_name:
        lines.append(f"name: {b_name!r} → {a_name!r}")
    bsteps, asteps = _steps(before), _steps(after)
    if len(bsteps) != len(asteps):
        lines.append(f"steps: {len(bsteps)} → {len(asteps)}")
    lines.extend(f"{s.get('number')}. {_step_label(s)}" for s in asteps)
    return lines


def revise_result_has_changes(result: Any) -> bool:
    """True when MAGE reports edits or the chain payload actually changed."""
    edits = getattr(result, "edits", None) or []
    if edits:
        return True
    before = getattr(result, "before_chain_dict", None) or {}
    after = getattr(result, "chain_dict", None) or {}
    if isinstance(before, dict) and isinstance(after, dict):
        return before != after
    return bool(after)


def render_edit_plan_lines(result: Any) -> list[str]:
    """Transcript lines for an applied/proposed edit: summary + edits + delta."""
    lines: list[str] = []
    summary = (getattr(result, "summary", "") or "").strip()
    lines.append(summary or "No change.")

    edits = list(getattr(result, "edits", []) or [])
    if edits:
        for e in edits:
            op = getattr(e, "op", "?")
            tgt = getattr(e, "target_step_number", None)
            rationale = (getattr(e, "rationale", "") or "").strip()
            target = f" step {tgt}" if tgt is not None else ""
            lines.append(f"• {op}{target}" + (f" — {rationale}" if rationale else ""))
    else:
        lines.append("• (no edits)")

    lines.extend(
        render_step_delta(
            getattr(result, "before_chain_dict", {}),
            getattr(result, "chain_dict", {}),
        )
    )
    return lines


def format_revise_confirm_body(
    result: Any,
    *,
    intro: str = "",
    empty_preview: str = "(no edit summary — see the diff above)",
    max_chars: int = 3500,
) -> str:
    """Multiline confirm-modal body: intro + edit preview."""
    preview = "\n".join(render_edit_plan_lines(result)).strip()
    if not preview or preview == "No change.":
        preview = empty_preview
    parts = [p for p in (intro.strip(), preview) if p]
    body = "\n\n".join(parts)
    if len(body) <= max_chars:
        return body
    return body[: max_chars - 1].rstrip() + "…"


def render_disambiguation_lines(result: Any) -> list[str]:
    """Transcript lines listing candidate chains when an edit target is ambiguous."""
    lines = ["Multiple chains match — re-run `/revise <id> <instruction>` with the one you mean:"]
    for c in getattr(result, "candidates", []) or []:
        eid = getattr(c, "entity_id", "")
        name = getattr(c, "name", "") or "(unnamed)"
        score = getattr(c, "score", 0.0)
        lines.append(f"• {eid}  {name}  (score {score:.2f})")
    return lines


__all__ = [
    "format_revise_confirm_body",
    "render_step_delta",
    "render_edit_plan_lines",
    "render_disambiguation_lines",
    "revise_result_has_changes",
]
