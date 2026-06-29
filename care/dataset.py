"""Reusable dataset-entry helpers (memory-card backed).

A "dataset" is the set of ``memory_card`` entities tagged
``dataset-entry:<chain_id>`` — each a ``{task, expected, rubric}`` test
case for a chain. This module factors the CRUD + substring scorer + JSONL
export out of the TUI's ``/dataset`` worker so the headless
``care dataset`` subcommand shares the same storage convention.

The CARE-internal scored replay's LLM-judge (rubric) path stays in the
TUI; the CLI uses the deterministic substring scorer (the same fallback
the TUI uses when a judge can't be reached).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASET_ENTRY_PREFIX = "dataset-entry:"


def collect_dataset_entries(memory: Any, chain_id: str) -> list[dict[str, Any]]:
    """Return every dataset entry for ``chain_id`` as flat dicts.

    Lists ``memory_card`` entities and filters on the
    ``dataset-entry:<chain_id>`` tag (list-by-tag isn't on the SDK
    surface yet). Fine for v1 dataset sizes (<500 entries/chain)."""
    tag = f"{DATASET_ENTRY_PREFIX}{chain_id}"
    rows = memory.list_entities(entity_type="memory_card", limit=500) or []
    out: list[dict[str, Any]] = []
    for row in rows:
        tags = row.get("tags") or (row.get("meta") or {}).get("tags") or []
        if tag not in tags:
            continue
        content = row.get("content") or row.get("content_json") or {}
        if not isinstance(content, dict):
            content = {}
        out.append(
            {
                "entity_id": row.get("entity_id") or row.get("id"),
                "task": content.get("task"),
                "expected": content.get("expected"),
                "rubric": content.get("rubric") or "",
                "actual": content.get("actual"),
                "status": content.get("status") or "pending",
            }
        )
    return out


def add_dataset_entry(
    memory: Any,
    chain_id: str,
    task: str,
    expected: str,
    *,
    rubric: str = "",
) -> str:
    """Save a new dataset entry as a tagged ``memory_card``; return its id.

    Mirrors the card shape the TUI's ``/dataset add`` writes so entries
    are interchangeable between the CLI and the TUI."""
    content = {
        "kind": "dataset-entry",
        "chain_id": chain_id,
        "task": task,
        "expected": expected,
        "rubric": rubric,
        "actual": None,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tags = [
        f"{DATASET_ENTRY_PREFIX}{chain_id}",
        "agent_run",
        f"agent:{chain_id}",
        "status:pending",
    ]
    if rubric:
        tags.append("scorer:rubric")
    short = task if len(task) <= 40 else task[:37] + "…"
    name = f"dataset · {chain_id} · {short}"
    return memory.save_memory_card(
        content,
        name=name,
        tags=tags,
        when_to_use=f"Dataset test case for chain {chain_id}.",
    )


def entry_passes(actual: str, expected: str) -> bool:
    """Case-insensitive substring scorer (expected ⊆ actual).

    Matches the TUI's deterministic default/fallback scorer."""
    if not expected:
        return False
    return expected.strip().lower() in (actual or "").strip().lower()


def export_entries_jsonl(entries: list[dict[str, Any]], path: Path | str) -> int:
    """Write entries to ``path`` as JSONL (one object per line); return count.

    The JSONL shape (``task``/``expected``/``rubric``/``actual``/``status``)
    is what external eval frameworks (Inspect-AI / promptfoo / OpenAI
    evals) accept natively."""
    out = Path(str(path)).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(
                json.dumps(
                    {
                        "task": entry.get("task"),
                        "expected": entry.get("expected"),
                        "rubric": entry.get("rubric") or "",
                        "actual": entry.get("actual"),
                        "status": entry.get("status"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


__all__ = [
    "DATASET_ENTRY_PREFIX",
    "add_dataset_entry",
    "collect_dataset_entries",
    "entry_passes",
    "export_entries_jsonl",
]
