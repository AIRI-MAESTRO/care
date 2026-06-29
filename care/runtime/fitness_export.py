"""Serialise an evolution fitness curve to CSV / JSON for export.

Pure helpers (no Textual import) so they're unit-testable and reusable by
both the EvolutionScreen's "export curve" action and any headless caller.
Input is the ``GenerationStat``-shaped records
``EvolutionProgressTracker.fitness_curve()`` returns. Placeholder ``-inf``
records (generations started but with no winner yet) are dropped.
"""

from __future__ import annotations

import json
from typing import Any, Iterable


def fitness_curve_rows(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Project records into ``{generation, best_fitness, mean_fitness}``
    dicts, dropping placeholder / non-finite best-fitness rows."""
    out: list[dict[str, Any]] = []
    for rec in records:
        gen = getattr(rec, "generation", None)
        best = getattr(rec, "best_fitness", None)
        if not isinstance(gen, int):
            continue
        if not isinstance(best, (int, float)) or best == float("-inf"):
            continue
        mean = getattr(rec, "mean_fitness", None)
        out.append(
            {
                "generation": int(gen),
                "best_fitness": float(best),
                "mean_fitness": (
                    float(mean) if isinstance(mean, (int, float)) else None
                ),
            }
        )
    out.sort(key=lambda r: r["generation"])
    return out


def fitness_curve_csv(records: Iterable[Any]) -> str:
    """Render the curve as CSV (header + one row per generation)."""
    rows = fitness_curve_rows(records)
    lines = ["generation,best_fitness,mean_fitness"]
    for row in rows:
        mean = "" if row["mean_fitness"] is None else repr(row["mean_fitness"])
        lines.append(f"{row['generation']},{row['best_fitness']!r},{mean}")
    return "\n".join(lines) + "\n"


def fitness_curve_json(records: Iterable[Any]) -> str:
    """Render the curve as a pretty-printed JSON array."""
    return json.dumps(fitness_curve_rows(records), indent=2) + "\n"


__all__ = ["fitness_curve_csv", "fitness_curve_json", "fitness_curve_rows"]
