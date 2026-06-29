"""Route hallucination-prone "live data" steps to real tools.

Even a capable planner sometimes makes a plain ``llm`` step for a task
that fundamentally needs live/external data — "what's today's date",
"current weather", "latest price". The LLM step then answers from the
model's stale memory and is simply wrong (e.g. "today is 5 Dec 2024").

This module adds the decision step the user asked for: *before* an
all-LLM chain runs, ask the model whether any step needs a tool, and if
so **rewrite that step into a ``tool`` call**. The tool is reused from
the registered set when one fits (e.g. ``current_datetime``); otherwise
it's left for :mod:`care.tool_synthesis` to generate. The chain's
dependency structure is untouched — only the step's *type* changes — so
a downstream "format the answer" step now receives real data.

Gated by ``CareConfig.tools.route_live_data_to_tools``. Only runs when
the chain has **no** tool step already (if MAGE picked a tool, synthesis
handles it) and there's at least one LLM step to convert. Never raises.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Callable

_log = logging.getLogger("care.tool_planning")

#: Correct ``input_mapping`` for known builtins — the classifier's
#: ``params`` are unreliable for these (it tends to return the search
#: text, not the parameter name), so we hard-map them to the right
#: signature against the user's task (``$outer_context``).
_KNOWN_TOOL_INPUTS: dict[str, dict[str, str]] = {
    "web_search": {"query": "$outer_context"},
    "fetch_url": {"url": "$outer_context"},
    "http_request": {"url": "$outer_context"},
    "current_datetime": {},
}

_CLASSIFIER_PROMPT = """A planner produced steps for a user task. Some may be plain LLM steps that will HALLUCINATE because they need LIVE / EXTERNAL data a language model cannot know from memory: the current date/time/day/year, today's weather, latest news, live prices, real-time facts, or anything requiring a web lookup.

User task:
{task}

Planned steps:
{steps}

Available tools (reuse an EXACT name when one fits):
{tools}

Decide whether ONE step must become a TOOL call instead of an LLM step.
Return ONLY a JSON object:
{{"needs_tool": true, "step_number": <n>, "tool_name": "<exact available name if one fits, else a new snake_case name>", "params": ["p1", ...], "reason": "<short>"}}
or {{"needs_tool": false}}.

Rules:
- Current date / time / day / month / year  -> tool_name "current_datetime", params [].
- Live web facts / news / weather / prices   -> reuse "web_search" (params ["query"]) or "http_request", or propose a specific new tool.
- Pick the data-GATHERING step, never a pure formatting/reasoning step.
- If every step is fine as pure reasoning/formatting, return {{"needs_tool": false}}.
"""


async def augment_chain_for_live_data(
    chain_dict: dict[str, Any],
    *,
    task: str,
    api: Any,
    config: Any,
) -> dict[str, Any]:
    """Maybe rewrite one LLM step into a tool call.

    Returns ``{"rewrote": bool, ...}``. When ``rewrote`` is true the
    payload also carries ``chain_dict`` (a rewritten copy), ``tool_name``,
    ``step_number``, ``step_title`` and ``reason``. Always safe to call.
    """
    none = {"rewrote": False}
    tools_cfg = getattr(config, "tools", None)
    if tools_cfg is None or not getattr(tools_cfg, "route_live_data_to_tools", True):
        return none

    steps = chain_dict.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return none
    # If the planner already used a tool, synthesis covers it — don't meddle.
    if any(str(s.get("step_type", "")).lower() == "tool" for s in steps if isinstance(s, dict)):
        return none
    llm_steps = [
        s for s in steps
        if isinstance(s, dict) and str(s.get("step_type", "")).lower() == "llm"
    ]
    if not llm_steps:
        return none

    generate = _resolve_llm(api)
    if generate is None:
        return none

    available = _available_tools(config)
    try:
        decision = await _classify(generate, task, steps, available)
    except Exception as exc:  # noqa: BLE001
        _log.warning("live-data classifier failed: %s", exc)
        return none
    if not decision.get("needs_tool"):
        return none

    target = _find_step(steps, decision.get("step_number")) or llm_steps[0]
    if str(target.get("step_type", "")).lower() != "llm":
        return none

    tool_name = str(decision.get("tool_name") or "").strip()
    if not tool_name:
        return none
    params = decision.get("params") or []
    if not isinstance(params, list):
        params = []

    new_chain = copy.deepcopy(chain_dict)
    if not _rewrite_step_to_tool(new_chain, target.get("number"), tool_name, params):
        return none

    _log.info(
        "routed step %s (%r) to tool %r: %s",
        target.get("number"), target.get("title"), tool_name, decision.get("reason", ""),
    )
    return {
        "rewrote": True,
        "chain_dict": new_chain,
        "tool_name": tool_name,
        "step_number": target.get("number"),
        "step_title": target.get("title") or "",
        "reason": str(decision.get("reason") or ""),
    }


# CARL step types the *installed* mmar_carl can execute, used as a fallback
# when MAGE's introspecting helper isn't importable.
_FALLBACK_LOADABLE_STEP_TYPES = frozenset(
    {"llm", "tool", "mcp", "memory", "transform", "conditional", "structured_output"}
)


def _carl_loadable_step_types() -> frozenset[str]:
    """Step types the installed CARL can load. Prefer MAGE's introspecting
    helper (it reads the installed ``mmar_carl``); fall back to the seven base
    types when it's unavailable."""
    try:
        from mmar_mage.carl_export import carl_loadable_step_types

        types = carl_loadable_step_types()
        if types:
            return frozenset(types)
    except Exception:  # noqa: BLE001
        pass
    return _FALLBACK_LOADABLE_STEP_TYPES


def downgrade_unsupported_step_types(chain_dict: dict[str, Any]) -> dict[str, Any]:
    """Rewrite steps whose ``step_type`` the installed CARL can't load → ``llm``.

    Richer topologies (``map_reduce`` / ``debate`` / ``critique_revise_loop`` /
    …) let MAGE author agent-features step types. ``carl_export`` translates
    some to ``llm`` (aggregator / parallel_sampling / supervisor) but keeps
    ``debate`` / ``evaluation`` because MAGE's own type set is forward-looking
    — and the installed ``mmar_carl`` (e.g. the 0.2.0 wheel) then rejects them
    at ``ReasoningChain.from_dict``. This pass rewrites any unsupported step to
    a plain ``llm`` step IN PLACE, keeping its number / title / aim /
    dependencies — so the topology SHAPE survives — and dropping the exotic
    nested ``step_config``. Idempotent, never raises; returns the same dict.
    """
    steps = chain_dict.get("steps")
    if not isinstance(steps, list):
        return chain_dict
    loadable = _carl_loadable_step_types()
    converted: list[str] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        st = str(s.get("step_type", "")).lower()
        if st and st not in loadable:
            s["step_type"] = "llm"
            s.pop("step_config", None)
            converted.append(st)
    if converted:
        _log.info(
            "downgraded %d step(s) to llm (installed CARL can't load: %s)",
            len(converted),
            ", ".join(sorted(set(converted))),
        )
    return chain_dict


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _available_tools(config: Any) -> list[dict[str, Any]]:
    """Builtin + cached-synthesised tool descriptors for the classifier."""
    out: list[dict[str, Any]] = []
    tools_cfg = getattr(config, "tools", None)
    try:
        from care.builtin_tools import builtin_tool_specs

        if tools_cfg is None or getattr(tools_cfg, "enable_builtins", True):
            out.extend(builtin_tool_specs(tools_cfg))
    except Exception:  # noqa: BLE001
        pass
    try:
        from care.tool_synthesis import cached_tool_specs

        out.extend(cached_tool_specs(config))
    except Exception:  # noqa: BLE001
        pass
    return out


async def _classify(
    generate: Callable[[str], Any],
    task: str,
    steps: list[dict[str, Any]],
    available: list[dict[str, Any]],
) -> dict[str, Any]:
    steps_txt = "\n".join(
        f"  {s.get('number')}. [{s.get('step_type')}] {s.get('title', '')}"
        f" — {s.get('aim', '')}"
        for s in steps if isinstance(s, dict)
    )
    tools_txt = "\n".join(
        f"  - {t.get('name')}: {t.get('description', '')}" for t in available
    ) or "  (none)"
    prompt = _CLASSIFIER_PROMPT.format(task=task, steps=steps_txt, tools=tools_txt)
    raw = await generate(prompt)
    return _parse_json(str(raw or "")) or {"needs_tool": False}


def _rewrite_step_to_tool(
    chain_dict: dict[str, Any],
    number: Any,
    tool_name: str,
    params: list[str],
) -> bool:
    """Flip the step with ``number`` from ``llm`` to a ``tool`` call.

    Keeps number / title / aim / dependencies / retry_max; drops
    LLM-only fields that a tool step doesn't take; builds ``step_config``
    with ``tool_name`` + an ``input_mapping`` (empty for known zero-arg
    builtins so we never pass them unexpected kwargs)."""
    for step in chain_dict.get("steps") or []:
        if not isinstance(step, dict) or step.get("number") != number:
            continue
        if tool_name in _KNOWN_TOOL_INPUTS:
            input_mapping: dict[str, str] = dict(_KNOWN_TOOL_INPUTS[tool_name])
        else:
            input_mapping = {p: "$outer_context" for p in params if isinstance(p, str)}
        cfg = step.get("step_config") if isinstance(step.get("step_config"), dict) else {}
        cfg.update({"tool_name": tool_name, "input_mapping": input_mapping})
        cfg.setdefault("timeout", 30.0)
        step["step_type"] = "tool"
        step["step_config"] = cfg
        for k in (
            "reasoning_questions", "example_reasoning",
            "stage_action", "step_context_queries", "llm_config",
        ):
            step.pop(k, None)
        return True
    return False


def _find_step(steps: list[dict[str, Any]], number: Any) -> dict[str, Any] | None:
    if number is None:
        return None
    for s in steps:
        if isinstance(s, dict) and s.get("number") == number:
            return s
    return None


def _parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort: parse the first JSON object in the model's reply."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    start, depth = text.find("{"), 0
    if start < 0:
        return None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:  # noqa: BLE001
                    return None
    return None


def _resolve_llm(api: Any) -> Callable[[str], Any] | None:
    for attr in ("get_response_with_retries", "get_response"):
        fn = getattr(api, attr, None)
        if callable(fn):
            async def _call(prompt: str, _fn: Any = fn) -> str:
                return await _fn(prompt)

            return _call
    return None


__all__ = ["augment_chain_for_live_data"]
