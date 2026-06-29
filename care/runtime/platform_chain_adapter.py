"""Adapt CARE / CARL chat chains for GigaEvo Platform evolution runner.

Platform executes only ``llm`` and ``tool`` steps with ``$-reference`` mappings.
Chat-built chains often include ``structured_output``, ``run_python``, MCP,
host paths, ``$inputs.*``, or hardcoded sample sentences. This module rewrites
those chains *before* submit so evolution can score them on the uploaded CSV.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from care.runtime.platform_chain_gate import (
    _HARDCODED_CYRILLIC,
    _HOST_PATH_MARKERS,
    _looks_hardcoded_sample,
)

_PLATFORM_STEP_TYPES = frozenset({"llm", "tool"})
_STEP_TYPE_TO_LLM = frozenset(
    {
        "structured_output",
        "transform",
        "conditional",
        "memory",
        "mcp",
        "agent_skill",
        "debate",
        "map_reduce",
        "critique_revise_loop",
        "evaluation",
        "eval",
        "judge",
        "reasoning",
    }
)
_STEP_TYPE_ALIASES = {
    "evaluation": "llm",
    "eval": "llm",
    "judge": "llm",
    "reasoning": "llm",
}
_PLATFORM_TOOLS = frozenset({"retrieve"})
_DROP_TOOLS = frozenset({"run_python", "pdf_extractor", "web_search"})

_REF_REWRITE = (
    (re.compile(r"\$inputs\."), "$sample."),
    (re.compile(r"\$input\."), "$sample."),
    (re.compile(r"\$inputs\b"), "$outer_context"),
    (re.compile(r"\$input\b"), "$sample.input"),
)


@dataclass(frozen=True)
class PlatformChainPrepareResult:
    """Output of :func:`prepare_chain_for_platform_evolution`."""

    chain: dict[str, Any]
    notes: tuple[str, ...] = ()
    adapted: bool = False


def prepare_chain_for_platform_evolution(
    chain_dict: dict[str, Any],
    *,
    target_column: str = "expected",
    bundled_tool_names: frozenset[str] | set[str] | None = None,
) -> PlatformChainPrepareResult:
    """Deep-copy *chain_dict*, apply Platform compatibility rewrites, return result."""
    if not isinstance(chain_dict, dict):
        return PlatformChainPrepareResult(chain={}, notes=("chain is not a dict",))

    allowed_tools = _PLATFORM_TOOLS | frozenset(bundled_tool_names or ())
    original = copy.deepcopy(chain_dict)
    notes: list[str] = []
    data = copy.deepcopy(chain_dict)
    steps = data.get("steps")
    if not isinstance(steps, list):
        return PlatformChainPrepareResult(chain=data, notes=("chain has no steps list",))

    _rewrite_refs_in_place(data)
    _strip_host_paths_from_metadata(data, notes)

    kept: list[dict[str, Any]] = []
    dropped_numbers: set[int] = set()

    for step in steps:
        if not isinstance(step, dict):
            notes.append("skipped non-object step entry")
            continue
        num = int(step.get("number") or 0)
        st = str(step.get("step_type", "llm")).strip().lower()
        st = _STEP_TYPE_ALIASES.get(st, st)

        if st == "tool":
            adapted_step, drop, note = _adapt_tool_step(step, allowed_tools=allowed_tools)
            if drop:
                dropped_numbers.add(num)
                if note:
                    notes.append(note)
                continue
            kept.append(adapted_step)
            continue

        if st in _STEP_TYPE_TO_LLM or st not in _PLATFORM_STEP_TYPES:
            kept.append(_exotic_step_to_llm(step, notes))
            continue

        if st == "llm":
            kept.append(_adapt_llm_step(step, target_column=target_column, notes=notes))
            continue

        dropped_numbers.add(num)
        notes.append(f"dropped step {num}: unsupported type {st!r}")

    if dropped_numbers:
        kept = _prune_dependencies(kept, dropped_numbers)

    if _should_collapse_few_shot_pipeline(original, kept):
        task = str(data.get("task_description") or data.get("metadata", {}).get("description") or "")
        kept = [_default_row_llm_step(task, target_column=target_column)]
        notes.append(
            "collapsed file-based few-shot pipeline to a single llm step — "
            "Platform supplies each CSV row as outer_context"
        )

    kept = _renumber_steps(kept)
    data["steps"] = kept

    if not kept:
        task = str(data.get("task_description") or "")
        data["steps"] = [_default_row_llm_step(task, target_column=target_column)]
        notes.append("synthesized default llm step (nothing executable remained)")

    _scrub_host_paths_deep(data, notes)
    _finalize_llm_steps(data, target_column=target_column, notes=notes)

    adapted = notes != [] or json.dumps(original, sort_keys=True) != json.dumps(
        data, sort_keys=True
    )
    return PlatformChainPrepareResult(
        chain=data,
        notes=tuple(notes),
        adapted=adapted,
    )


def _rewrite_refs_in_place(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, str):
                obj[key] = _rewrite_ref_string(val)
            else:
                _rewrite_refs_in_place(val)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                obj[i] = _rewrite_ref_string(item)
            else:
                _rewrite_refs_in_place(item)


def _rewrite_ref_string(text: str) -> str:
    out = text
    for pattern, repl in _REF_REWRITE:
        out = pattern.sub(repl, out)
    return out


def _strip_host_paths_from_metadata(data: dict[str, Any], notes: list[str]) -> None:
    meta = data.get("metadata")
    if isinstance(meta, dict):
        desc = meta.get("description")
        if isinstance(desc, str) and any(m in desc for m in _HOST_PATH_MARKERS):
            meta["description"] = _strip_path_mentions(desc)
            notes.append("trimmed host paths from metadata.description")
    task = data.get("task_description")
    if isinstance(task, str) and any(m in task for m in _HOST_PATH_MARKERS):
        data["task_description"] = _strip_path_mentions(task)
        notes.append("trimmed host paths from task_description")


def _strip_path_mentions(text: str) -> str:
    out = text
    for marker in _HOST_PATH_MARKERS:
        while marker in out:
            start = out.index(marker)
            end = start + len(marker)
            while end < len(out) and out[end] not in " \n\t\"'<>":
                end += 1
            out = (out[:start] + "[dataset row]" + out[end:]).strip()
    return re.sub(r"\s+", " ", out).strip()


def _scrub_host_paths_deep(obj: Any, notes: list[str]) -> None:
    """Remove host path literals from every string field in the chain dict."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, str) and any(m in val for m in _HOST_PATH_MARKERS):
                cleaned = _strip_path_mentions(val)
                if cleaned != val:
                    obj[key] = cleaned
                    notes.append(f"scrubbed host paths from {key!r}")
            else:
                _scrub_host_paths_deep(val, notes)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and any(m in item for m in _HOST_PATH_MARKERS):
                cleaned = _strip_path_mentions(item)
                if cleaned != item:
                    obj[i] = cleaned
                    notes.append("scrubbed host paths from list entry")
            else:
                _scrub_host_paths_deep(item, notes)


def _adapt_tool_step(
    step: dict[str, Any],
    *,
    allowed_tools: frozenset[str] | None = None,
) -> tuple[dict[str, Any], bool, str]:
    num = step.get("number", "?")
    sc = step.get("step_config") if isinstance(step.get("step_config"), dict) else {}
    tool_name = str(sc.get("tool_name") or "").strip().lower()
    platform_tools = allowed_tools if allowed_tools is not None else _PLATFORM_TOOLS

    if tool_name in _DROP_TOOLS or _mapping_has_host_path(sc):
        return step, True, (
            f"dropped step {num}: tool {tool_name!r} needs CARE sandbox / host files"
        )

    if tool_name and tool_name not in platform_tools:
        return step, True, (
            f"dropped step {num}: tool {tool_name!r} is not registered on Platform runner"
        )

    mapping = sc.get("input_mapping") if isinstance(sc.get("input_mapping"), dict) else {}
    fixed_mapping: dict[str, str] = {}
    for param, ref in mapping.items():
        ref_s = _rewrite_ref_string(str(ref).strip())
        if not ref_s.startswith("$"):
            return step, True, (
                f"dropped step {num}: tool input_mapping[{param!r}] is not a $-reference"
            )
        fixed_mapping[str(param)] = ref_s
    sc = {**sc, "input_mapping": fixed_mapping}
    return {**step, "step_type": "tool", "step_config": sc}, False, ""


def _mapping_has_host_path(sc: dict[str, Any]) -> bool:
    blob = json.dumps(sc, ensure_ascii=False)
    return any(m in blob for m in _HOST_PATH_MARKERS)


def _exotic_step_to_llm(step: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    num = step.get("number", "?")
    st = str(step.get("step_type", "")).lower()
    sc = step.get("step_config") if isinstance(step.get("step_config"), dict) else {}
    instruction = str(sc.get("instruction") or step.get("aim") or "").strip()
    input_source = _rewrite_ref_string(str(sc.get("input_source") or "$history[-1]"))
    schema_hint = ""
    schema = sc.get("output_schema")
    if isinstance(schema, dict):
        schema_hint = f" Output JSON matching schema: {json.dumps(schema, ensure_ascii=False)[:400]}"

    notes.append(f"step {num}: converted {st!r} → llm")
    aim = instruction or str(step.get("aim") or "Complete the task from context")
    stage = (
        f"Use prior context ({input_source}). "
        f"{instruction or 'Produce the required output.'}"
        f"{schema_hint}"
    ).strip()
    return {
        "number": step.get("number"),
        "title": step.get("title") or f"Step {num}",
        "step_type": "llm",
        "dependencies": list(step.get("dependencies") or []),
        "aim": aim[:2000],
        "stage_action": stage[:2000],
        "reasoning_questions": str(step.get("reasoning_questions") or ""),
        "example_reasoning": str(step.get("example_reasoning") or ""),
    }


def _adapt_llm_step(
    step: dict[str, Any],
    *,
    target_column: str,
    notes: list[str],
) -> dict[str, Any]:
    out = copy.deepcopy(step)
    out["step_type"] = "llm"
    for field_name in ("aim", "stage_action", "example_reasoning"):
        text = str(out.get(field_name) or "")
        if any(m in text for m in _HOST_PATH_MARKERS):
            out[field_name] = _strip_path_mentions(text)
            notes.append(
                f"step {out.get('number')}: scrubbed host paths from {field_name!r}"
            )
        elif _looks_hardcoded_sample(text):
            out[field_name] = _dehardcode_llm_field(
                field_name, target_column=target_column
            )
            notes.append(
                f"step {out.get('number')}: replaced hardcoded sample in {field_name!r} "
                f"with outer_context / $sample.input guidance"
            )
        elif _HARDCODED_CYRILLIC.search(text) and "$" not in text:
            out[field_name] = _dehardcode_llm_field(
                field_name, target_column=target_column
            )
            notes.append(
                f"step {out.get('number')}: replaced Cyrillic blob in {field_name!r}"
            )
    if not str(out.get("aim") or "").strip():
        out["aim"] = "Solve the task using the provided context"
    if not str(out.get("stage_action") or "").strip():
        out["stage_action"] = (
            f"Read the input from context (fields input / task). "
            f"Reply in the same language. Target column for scoring: {target_column!r}."
        )
    if not str(out.get("reasoning_questions") or "").strip():
        out["reasoning_questions"] = "What is the input? What output format is required?"
    return out


def _dehardcode_llm_field(field: str, *, target_column: str) -> str:
    if field == "stage_action":
        return (
            "Read the source text from context (field 'input' or 'task'). "
            "Reply with a single concise sentence in the same language as the input. "
            "No markdown, no quotes, no preamble."
        )
    if field == "example_reasoning":
        return (
            "Match the style of concise one-sentence summaries implied by the task; "
            "use only the current row's input, not a fixed example string."
        )
    return (
        "Transform the input text from context into one concise sentence summary "
        f"in the same language (scored against {target_column!r})."
    )


def _should_collapse_few_shot_pipeline(
    original: dict[str, Any],
    kept: list[dict[str, Any]],
) -> bool:
    orig_steps = original.get("steps") if isinstance(original.get("steps"), list) else []
    had_run_python = False
    for step in orig_steps:
        if not isinstance(step, dict):
            continue
        st = str(step.get("step_type", "")).lower()
        sc = step.get("step_config") if isinstance(step.get("step_config"), dict) else {}
        if st == "tool" and str(sc.get("tool_name") or "").lower() == "run_python":
            had_run_python = True
            break
    if not had_run_python:
        return False
    # Any chain that read examples from disk must use Platform CSV rows instead.
    return True


def _default_row_llm_step(task: str, *, target_column: str) -> dict[str, Any]:
    """Platform row step — never embed chat task blobs (JSONL / few-shot) in aim."""
    aim = _safe_task_aim(task, target_column=target_column)
    return {
        "number": 1,
        "title": "Answer from dataset row",
        "step_type": "llm",
        "dependencies": [],
        "aim": aim,
        "stage_action": (
            "The context contains one dataset row (fields such as input, task). "
            f"Produce the answer scored against column {target_column!r}. "
            "Use the same language as the input unless the task says otherwise."
        ),
        "reasoning_questions": "What is being asked? What format should the answer take?",
    }


def _safe_task_aim(task: str, *, target_column: str) -> str:
    """Use a short clean task line as aim, else generic Platform-safe wording."""
    cleaned = task.strip()
    if cleaned and len(cleaned) <= 120 and not _looks_hardcoded_sample(cleaned):
        return cleaned[:500]
    first_line = cleaned.split("\n", 1)[0].strip() if cleaned else ""
    if first_line and len(first_line) <= 120 and not _looks_hardcoded_sample(first_line):
        return first_line
    return _dehardcode_llm_field("aim", target_column=target_column)


def _finalize_llm_steps(
    data: dict[str, Any],
    *,
    target_column: str,
    notes: list[str],
) -> None:
    """Last pass: ensure every llm step passes Platform hardcoded-sample checks."""
    steps = data.get("steps")
    if not isinstance(steps, list):
        return
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if str(step.get("step_type", "llm")).lower() != "llm":
            continue
        steps[i] = _adapt_llm_step(step, target_column=target_column, notes=notes)


def _prune_dependencies(steps: list[dict[str, Any]], dropped: set[int]) -> list[dict[str, Any]]:
    kept_numbers = {int(s.get("number") or 0) for s in steps}
    for step in steps:
        deps = [int(d) for d in (step.get("dependencies") or []) if int(d) not in dropped]
        step["dependencies"] = [d for d in deps if d in kept_numbers]
    return steps


def _renumber_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(steps, key=lambda s: int(s.get("number") or 0))
    old_to_new: dict[int, int] = {}
    for i, step in enumerate(ordered, start=1):
        old_num = int(step.get("number") or i)
        old_to_new[old_num] = i
        step["number"] = i
    for step in ordered:
        step["dependencies"] = [
            old_to_new[int(d)]
            for d in (step.get("dependencies") or [])
            if int(d) in old_to_new and old_to_new[int(d)] != step["number"]
        ]
    return ordered


__all__ = ["PlatformChainPrepareResult", "prepare_chain_for_platform_evolution"]
