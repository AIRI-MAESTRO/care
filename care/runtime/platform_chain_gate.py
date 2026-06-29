"""Preflight checks: is a CARL/Memory chain safe to run on Platform runner?

Platform's chain helper executes only ``llm`` and ``tool`` steps with
``$-reference`` input mappings. Chat-built chains often embed host paths,
``run_python`` code, ``structured_output`` parsers, or hardcoded sample
text — they run in CARE but score fitness 0 on Platform evolution.
"""

from __future__ import annotations

import json
import re
from typing import Any

_PLATFORM_STEP_TYPES = frozenset({"llm", "tool"})
_PLATFORM_TOOL_NAMES = frozenset({"retrieve", "calculator", "current_datetime"})

_REF_OK_PREFIXES = ("$outer_context", "$history", "$sample", "$inputs", "$input")

_HOST_PATH_MARKERS = ("/home/", "/Users/", "C:\\", "\\\\")
_CODE_MARKERS = ("open(", "import ", "with open", "eval(", "exec(")

_HARDCODED_CYRILLIC = re.compile(r"[\u0400-\u04FF][\u0400-\u04FF\s,.-]{60,}")
_EMBEDDED_DATASET_MARKERS = ('{"input"', '"expected"', "<file path", "```", "eval.jsonl")


def gate_chain_for_platform_evolution(
    chain_dict: dict[str, Any],
    *,
    target_column: str = "expected",
    bundled_tool_names: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Return human-readable blockers; empty list means OK to submit."""
    platform_tools = _PLATFORM_TOOL_NAMES | frozenset(bundled_tool_names or ())
    if not isinstance(chain_dict, dict):
        return ["chain content is not a JSON object"]
    steps = chain_dict.get("steps")
    if not isinstance(steps, list) or not steps:
        return ["chain has no steps — nothing to evolve"]

    issues: list[str] = []
    blob = json.dumps(chain_dict, ensure_ascii=False)

    for marker in _HOST_PATH_MARKERS:
        if marker in blob:
            issues.append(
                "chain contains host filesystem paths — Platform runner is "
                "Docker-isolated; use Platform-uploaded dataset (dataset/data.csv) "
                "via $outer_context / $sample.* instead"
            )
            break

    llm_count = 0
    for step in steps:
        if not isinstance(step, dict):
            issues.append("chain contains a non-object step entry")
            continue
        num = step.get("number", "?")
        st = str(step.get("step_type", "llm")).strip().lower()
        title = str(step.get("title") or f"step {num}")

        if st not in _PLATFORM_STEP_TYPES:
            issues.append(
                f"step {num} ({title}): step_type '{st}' is ignored by Platform "
                f"runner — only llm and tool execute (use a single llm step for "
                f"summarization; dataset rows come from CSV, not run_python)"
            )
            continue

        if st == "llm":
            llm_count += 1
            for field in ("aim", "stage_action"):
                if not str(step.get(field) or "").strip():
                    issues.append(
                        f"step {num}: missing required Platform field '{field}'"
                    )
            for field in ("aim", "stage_action", "example_reasoning"):
                text = str(step.get(field) or "")
                if _looks_hardcoded_sample(text):
                    issues.append(
                        f"step {num}: '{field}' embeds a fixed sample sentence — "
                        f"validation runs many CSV rows; summarize $outer_context "
                        f"instead (target_column={target_column!r})"
                    )

        if st == "tool":
            sc = step.get("step_config") if isinstance(step.get("step_config"), dict) else {}
            tool_name = str(sc.get("tool_name") or "").strip().lower()
            if tool_name == "run_python":
                issues.append(
                    f"step {num}: run_python is unavailable in Platform runner "
                    "(no CARE sandbox) — load data from Platform dataset/CSV instead"
                )
            elif tool_name and tool_name not in platform_tools:
                issues.append(
                    f"step {num}: tool '{tool_name}' is not registered in Platform "
                    f"runner (available: {', '.join(sorted(platform_tools))})"
                )
            mapping = sc.get("input_mapping") if isinstance(sc.get("input_mapping"), dict) else {}
            for param, ref in mapping.items():
                ref_s = str(ref).strip()
                if not ref_s.startswith("$"):
                    hint = ref_s[:80] + ("…" if len(ref_s) > 80 else "")
                    issues.append(
                        f"step {num}: input_mapping[{param!r}] must be a $-reference "
                        f"(got literal {hint!r}) — causes 'Unknown reference syntax'"
                    )
                elif not ref_s.startswith(_REF_OK_PREFIXES):
                    issues.append(
                        f"step {num}: input_mapping[{param!r}]={ref_s!r} is not a "
                        f"supported Platform reference"
                    )
                for code_marker in _CODE_MARKERS:
                    if code_marker in ref_s:
                        issues.append(
                            f"step {num}: input_mapping[{param!r}] looks like inline "
                            f"code, not a $-reference"
                        )
                        break

    if llm_count == 0:
        issues.append(
            "chain has no llm steps after Platform filtering — fitness will stay 0"
        )

    return issues


def _looks_hardcoded_sample(text: str) -> bool:
    if "$sample" in text or "$outer_context" in text:
        return False
    if len(text) < 80:
        return False
    if any(m in text for m in _EMBEDDED_DATASET_MARKERS):
        return True
    return bool(_HARDCODED_CYRILLIC.search(text))


__all__ = ["gate_chain_for_platform_evolution", "_looks_hardcoded_sample"]
