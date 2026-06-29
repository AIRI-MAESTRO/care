"""Run-history data layer (TODO §3 P1 — Run history tab).

The InspectionScreen's "Run history" tab lists every prior run of
a saved agent: timestamp, success/failure, duration, total tokens,
and a link back to the underlying ``memory_card`` with full step
traces. The Textual tab is gated on §1 P0 multi-screen workflow,
but the projection + Memory query land now as the data layer.

Contents:

* :class:`RunHistoryEntry` — frozen per-row projection of a
  CARE-saved ``memory_card``. Reads the conventions
  :func:`care.runtime.record_run_completion` writes (content
  shape + ``meta.tags``).
* :class:`RunHistorySummary` — frozen aggregate the future tab
  renders as a header (total runs, success / failure counts,
  total tokens, etc.).
* :func:`parse_run_history_entry` — pure projection from a
  ``memory_card`` EntityResponse-shaped dict. Returns ``None``
  when the card doesn't follow the ``agent_run`` convention
  (e.g. a lesson-learned card stamped with a different
  category) so the caller can mix card kinds in the same fetch.
* :func:`fetch_run_history` — async helper that asks Memory
  for every ``memory_card`` tagged ``agent_run`` +
  ``agent:{entity_id}`` and projects them in descending
  finished-at order.
* :func:`summarize_run_history` — pure aggregator.

Duck-typed boundaries: the fetch helper reaches into
``memory.client._list_entities`` (the SDK's generic typed
lister) with a tag filter — this is the SDK's existing supported
path for tag-filtered listings; the public
``MemoryCardsMixin.list_memory_cards`` doesn't yet expose
``tags=``. CARE's facade owns the call. The projection accepts
plain dicts OR `EntityResponse` model objects (attribute access
via `_read`).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Optional


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RunHistoryError(RuntimeError):
    """Raised when run-history retrieval fails — unreachable
    Memory, timeout, or a malformed response. The future
    InspectionScreen tab catches this and shows a friendly
    "couldn't load runs" toast."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


RunStatus = Literal["success", "failure"]


@dataclass(frozen=True)
class RunHistoryEntry:
    """One row in the InspectionScreen's run-history tab.

    Frozen so snapshots flow through Textual messages without
    defensive copies.

    Fields mirror the shape :func:`care.runtime.record_run_completion`
    persists, with light projection so the tab doesn't re-read
    nested ``usage.metrics`` on every render.
    """

    card_id: str
    agent_entity_id: str
    run_id: str
    finished_at: Optional[datetime] = None
    status: RunStatus = "success"
    duration_seconds: Optional[float] = None
    step_count: Optional[int] = None
    total_tokens: Optional[int] = None
    error_message: Optional[str] = None
    task_description: Optional[str] = None
    description: Optional[str] = None
    tags: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """``True`` when the run finished without an error.
        Drives the success / failure badge."""
        return self.status == "success"

    def format_one_line(self) -> str:
        """Compact rendering for log output / debug commands.
        The future Textual tab does something prettier."""
        ts = (
            self.finished_at.strftime("%Y-%m-%d %H:%M:%S")
            if self.finished_at
            else "—"
        )
        badge = "✓" if self.success else "✗"
        bits = [f"{badge} {ts}", f"run {self.run_id[:18]}"]
        if self.duration_seconds is not None:
            bits.append(f"{self.duration_seconds:.1f}s")
        if self.total_tokens is not None:
            bits.append(f"{self.total_tokens} tok")
        if self.error_message and not self.success:
            bits.append(self.error_message[:60])
        return " · ".join(bits)


@dataclass(frozen=True)
class RunHistorySummary:
    """Aggregate header for the run-history tab."""

    total_runs: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_tokens: int = 0
    total_duration_seconds: float = 0.0
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None

    @property
    def success_rate(self) -> Optional[float]:
        """Fraction in ``[0, 1]``. ``None`` when no runs."""
        if self.total_runs == 0:
            return None
        return self.success_count / self.total_runs

    @property
    def avg_duration_seconds(self) -> Optional[float]:
        if self.total_runs == 0:
            return None
        return self.total_duration_seconds / self.total_runs


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def parse_run_history_entry(
    card: Any,
    *,
    agent_entity_id: Optional[str] = None,
) -> Optional[RunHistoryEntry]:
    """Project a ``memory_card`` EntityResponse (or dict) into a
    :class:`RunHistoryEntry`.

    Returns ``None`` when the card doesn't look like an agent-run
    record (wrong category, missing tag, or unparseable). The
    caller can sweep a mixed-kind list and filter Nones out.

    Args:
        card: ``memory_card`` row from Memory.
        agent_entity_id: If supplied, project only when the
            card's ``agent:{id}`` tag matches this id; cards
            for other agents return ``None``. ``None`` accepts
            cards from any agent (useful for activity feeds).
    """
    content = _read(card, "content") or {}
    if not isinstance(content, dict):
        return None
    if content.get("category") != "agent_run":
        return None

    usage = content.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    metrics = usage.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    card_agent_id = str(usage.get("agent_entity_id") or "")
    if agent_entity_id is not None and card_agent_id != agent_entity_id:
        # Tag-based filtering happens on Memory side, but defend
        # against namespace cross-talk by re-checking here.
        return None

    run_id = str(usage.get("run_id") or "")
    if not run_id:
        return None

    # Card-level metadata.
    meta = _read(card, "meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    raw_tags = meta.get("tags") or []
    if not isinstance(raw_tags, list):
        raw_tags = []

    status = _extract_status(meta_tags=raw_tags, metrics=metrics)
    finished_at = _coerce_datetime(usage.get("finished_at"))

    duration = metrics.get("duration_seconds")
    duration_seconds = float(duration) if isinstance(duration, (int, float)) else None

    step_count = metrics.get("step_count")
    step_count_int = int(step_count) if isinstance(step_count, (int, float)) else None

    tokens = metrics.get("total_tokens")
    total_tokens = int(tokens) if isinstance(tokens, (int, float)) else None

    return RunHistoryEntry(
        card_id=str(_read(card, "entity_id") or ""),
        agent_entity_id=card_agent_id,
        run_id=run_id,
        finished_at=finished_at,
        status=status,
        duration_seconds=duration_seconds,
        step_count=step_count_int,
        total_tokens=total_tokens,
        error_message=_extract_error_message(metrics),
        task_description=_read_str(content, "task_description"),
        description=_read_str(content, "description"),
        tags=tuple(str(t) for t in raw_tags),
        metrics=dict(metrics),
    )


def _extract_status(
    *,
    meta_tags: list,
    metrics: dict,
) -> RunStatus:
    """Read the run's status from the card's tags first, then
    fall back to the metrics ``exit_status`` field."""
    for tag in meta_tags:
        if not isinstance(tag, str):
            continue
        if tag.startswith("status:"):
            label = tag.split(":", 1)[1].strip().lower()
            return "success" if label == "success" else "failure"
    exit_status = metrics.get("exit_status")
    if isinstance(exit_status, str):
        return "success" if exit_status.lower() == "success" else "failure"
    return "success"


def _extract_error_message(metrics: dict) -> Optional[str]:
    """Pull the original error message out of the card's metrics.

    ``record_run_completion`` writes the description as ``"…: <msg>"``
    when a run failed; we don't try to re-parse that. If
    ``metrics`` carries an explicit ``error_message`` / ``error``
    field, use it. Otherwise ``None``.
    """
    direct = metrics.get("error_message") or metrics.get("error")
    if isinstance(direct, str) and direct:
        return direct
    return None


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------


async def fetch_run_history(
    memory: Any,
    agent_entity_id: str,
    *,
    limit: int = 100,
    namespace: Optional[str] = None,
    channel: str = "latest",
    timeout: float = 10.0,
) -> tuple[RunHistoryEntry, ...]:
    """Fetch the ordered run history for one agent.

    Asks Memory for every ``memory_card`` carrying both the
    ``"agent_run"`` and ``"agent:{entity_id}"`` tags and projects
    each row via :func:`parse_run_history_entry`. Results are
    sorted by ``finished_at`` descending (most-recent first) — the
    natural reading order for the tab.

    Wraps the sync SDK call in :func:`asyncio.to_thread` with a
    deadline so the modal doesn't freeze on a hung server. Any
    error — timeout, HTTP failure, malformed response — surfaces
    as :class:`RunHistoryError` so the tab's toast handler stays
    single-branch.

    Args:
        memory: A `CareMemory` facade (or any object exposing
            ``.client._list_entities(...)``). Tests pass a stub.
        agent_entity_id: Agent / chain entity id to filter on.
        limit: Max number of cards to fetch. Memory's API caps
            at 200; we clamp here so callers can pass
            sane defaults without surprising the server.
        namespace: Restrict to a single CARE namespace. ``None``
            inherits the caller's auth-scope.
        channel: Memory channel to read (default ``"latest"`` —
            agent_run cards are always written there).
        timeout: Per-call deadline in seconds.

    Returns:
        Tuple of :class:`RunHistoryEntry`, most-recent first.

    Raises:
        RunHistoryError: Memory was unreachable, timed out, or
            returned a malformed response.
    """
    if not agent_entity_id:
        raise RunHistoryError("agent_entity_id is required")

    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    lister = getattr(client, "_list_entities", None) if client else None
    if lister is None or not callable(lister):
        raise RunHistoryError(
            "memory facade does not expose client._list_entities()"
        )

    clamped_limit = max(1, min(limit, 200))
    tags = ["agent_run", f"agent:{agent_entity_id}"]

    start = time.monotonic()
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(
                lister,
                "memory_card",
                limit=clamped_limit,
                channel=channel,
                tags=tags,
                namespace=namespace,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        latency = (time.monotonic() - start) * 1000
        raise RunHistoryError(
            f"run history fetch timed out after {timeout:.1f}s ({latency:.0f}ms elapsed)"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RunHistoryError(
            f"run history fetch failed: {type(exc).__name__}: {exc}"
        ) from exc

    entries: list[RunHistoryEntry] = []
    iterable = rows if isinstance(rows, (list, tuple)) else ()
    for row in iterable:
        entry = parse_run_history_entry(row, agent_entity_id=agent_entity_id)
        if entry is not None:
            entries.append(entry)

    entries.sort(
        key=lambda e: e.finished_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return tuple(entries)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_run_history(
    entries: Iterable[RunHistoryEntry],
) -> RunHistorySummary:
    """Aggregate :class:`RunHistoryEntry` rows for the tab header."""
    total = 0
    successes = 0
    failures = 0
    total_tokens = 0
    total_duration = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None

    for e in entries:
        total += 1
        if e.success:
            successes += 1
            if e.finished_at is not None and (
                last_success is None or e.finished_at > last_success
            ):
                last_success = e.finished_at
        else:
            failures += 1
            if e.finished_at is not None and (
                last_failure is None or e.finished_at > last_failure
            ):
                last_failure = e.finished_at
        if e.total_tokens is not None:
            total_tokens += e.total_tokens
        if e.duration_seconds is not None:
            total_duration += e.duration_seconds

    return RunHistorySummary(
        total_runs=total,
        success_count=successes,
        failure_count=failures,
        total_tokens=total_tokens,
        total_duration_seconds=total_duration,
        last_success_at=last_success,
        last_failure_at=last_failure,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _read_str(obj: Any, name: str) -> Optional[str]:
    value = _read(obj, name)
    if isinstance(value, str):
        return value
    return None


def _coerce_datetime(value: Any) -> Optional[datetime]:
    """Accept already-parsed ``datetime``, ISO-8601 string, or
    None. Anything else collapses to ``None``."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


__all__ = [
    "RunHistoryEntry",
    "RunHistoryError",
    "RunHistorySummary",
    "RunStatus",
    "fetch_run_history",
    "parse_run_history_entry",
    "summarize_run_history",
]
