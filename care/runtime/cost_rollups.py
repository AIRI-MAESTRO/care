"""Cost / token rollups for the `/cost` dashboard (TODO §6 P2).

Pure aggregation layer over :class:`LocalRunEntry` records —
no Textual, no I/O. The screen reads a list of entries via
:func:`load_local_runs` then projects it through these
helpers to populate the dashboard tables.

Four kinds of rollup:

* :func:`compute_overall` — single aggregate carrying total
  tokens (in / out / sum), total cost, total wall-clock
  duration, run count, and success rate. The dashboard
  shows this as a one-line header.
* :func:`compute_per_provider` / :func:`compute_per_chain` /
  :func:`compute_per_mode` — list of :class:`RollupRow`
  sorted by ``cost_usd`` descending so the user sees the
  biggest spenders first. Missing keys collapse into the
  literal ``"(unknown)"`` so the rollup never silently
  drops rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from care.runtime.local_run_history import LocalRunEntry


_UNKNOWN_KEY = "(unknown)"
"""Bucket label for rows whose grouping key is empty."""


@dataclass(frozen=True)
class OverallTotals:
    """Aggregate of every run currently visible."""

    runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    total_duration_seconds: float = 0.0

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def success_rate(self) -> float | None:
        if self.runs == 0:
            return None
        return self.successful_runs / self.runs


@dataclass(frozen=True)
class RollupRow:
    """One row in a per-key dashboard table."""

    key: str
    label: str
    runs: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


def compute_overall(
    entries: Iterable[LocalRunEntry],
) -> OverallTotals:
    """Single-aggregate roll-up for the dashboard header."""
    runs = 0
    successes = 0
    failures = 0
    tokens_in = 0
    tokens_out = 0
    cost = 0.0
    duration = 0.0
    for entry in entries:
        runs += 1
        if entry.status == "success":
            successes += 1
        elif entry.status == "failure":
            failures += 1
        tokens_in += entry.tokens_in or 0
        tokens_out += entry.tokens_out or 0
        cost += entry.cost_usd or 0.0
        duration += entry.duration_seconds or 0.0
    return OverallTotals(
        runs=runs,
        successful_runs=successes,
        failed_runs=failures,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        total_duration_seconds=duration,
    )


def _group_by(
    entries: Iterable[LocalRunEntry],
    *,
    key_fn,
    label_fn,
) -> list[RollupRow]:
    """Shared loop for per-X rollups."""
    buckets: dict[str, dict[str, float | int | str]] = {}
    for entry in entries:
        raw_key = key_fn(entry) or ""
        key = raw_key or _UNKNOWN_KEY
        b = buckets.setdefault(key, {
            "label": label_fn(entry, raw_key),
            "runs": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        })
        b["runs"] = int(b["runs"]) + 1
        b["tokens_in"] = (
            int(b["tokens_in"]) + (entry.tokens_in or 0)
        )
        b["tokens_out"] = (
            int(b["tokens_out"]) + (entry.tokens_out or 0)
        )
        b["cost_usd"] = (
            float(b["cost_usd"]) + (entry.cost_usd or 0.0)
        )
    rows = [
        RollupRow(
            key=key,
            label=str(values["label"]),
            runs=int(values["runs"]),
            tokens_in=int(values["tokens_in"]),
            tokens_out=int(values["tokens_out"]),
            cost_usd=float(values["cost_usd"]),
        )
        for key, values in buckets.items()
    ]
    # Highest spenders first; ties broken by token count then
    # by key for deterministic ordering across rebuilds.
    rows.sort(
        key=lambda r: (-r.cost_usd, -r.tokens_total, r.key),
    )
    return rows


def compute_per_provider(
    entries: Iterable[LocalRunEntry],
) -> list[RollupRow]:
    """Group by LLM provider (e.g. `openai`, `anthropic`)."""
    return _group_by(
        entries,
        key_fn=lambda e: e.provider,
        label_fn=lambda e, k: k or _UNKNOWN_KEY,
    )


def compute_per_chain(
    entries: Iterable[LocalRunEntry],
) -> list[RollupRow]:
    """Group by `chain_id`; label uses `chain_name` when
    present (falls back to the id)."""
    return _group_by(
        entries,
        key_fn=lambda e: e.chain_id,
        label_fn=lambda e, k: (
            e.chain_name or k or _UNKNOWN_KEY
        ),
    )


def compute_per_mode(
    entries: Iterable[LocalRunEntry],
) -> list[RollupRow]:
    """Group by execution mode (`ad_hoc` / `production` /
    `evolution`)."""
    return _group_by(
        entries,
        key_fn=lambda e: e.mode,
        label_fn=lambda e, k: k or _UNKNOWN_KEY,
    )


__all__ = [
    "OverallTotals",
    "RollupRow",
    "compute_overall",
    "compute_per_chain",
    "compute_per_mode",
    "compute_per_provider",
]
