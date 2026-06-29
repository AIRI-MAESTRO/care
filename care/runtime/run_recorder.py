"""Post-run persistence helper (TODO §3 P0).

After every successful (or failed) chain execution, CARE writes:

1. A ``memory_card`` capturing the run outcome (success / failure /
   key metrics / link back to the source agent) so the library's
   "Run history" tab can show it.
2. A ``run-recorded`` ping against the source agent entity that
   bumps ``run_count`` and sets ``last_run_at = now()`` —
   library sorting depends on these (Memory TODO §1.4 shipped).

This module is a thin coordinator: callers (CARE's
``ExecutionScreen`` + the headless CLI's ``care run`` command) hand
in a duck-typed run-result object and the already-constructed
:class:`care.CareMemory`. The helper:

* normalises whatever shape the result actually has into a flat
  :class:`RunSummary`,
* builds a memory-card content dict + ``meta.tags`` that link
  back to the source agent,
* fires both writes in order, returning a typed
  :class:`RunCompletion` so call-sites can show "saved as
  card-7c9…" + "run #4" in one place.

Duck typing on the result keeps CARE startup free of a hard
``mmar_carl`` import — anything matching the documented attribute
shape works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from care.memory import CareMemory


@dataclass(frozen=True)
class RunSummary:
    """Flattened view of a chain execution outcome.

    Mirrors the subset of ``mmar_carl.ReasoningResult`` CARE actually
    persists. Frozen so it can be hashed + safely stored on Textual
    messages without defensive copies.
    """

    success: bool
    step_count: int = 0
    duration_seconds: float = 0.0
    total_tokens: int | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def status_label(self) -> str:
        """Human-readable status: ``"success"`` / ``"failed"``."""
        return "success" if self.success else "failed"


@dataclass(frozen=True)
class RunCompletion:
    """Return shape of :func:`record_run_completion`.

    ``memory_card_entity_id`` is the persisted ``memory_card``'s id;
    ``agent_recorded`` is ``True`` when the ``run-recorded`` ping
    succeeded against the source agent. ``run_id`` is whatever the
    caller passed (or a generated one) so the UI can correlate a
    later ``stream_events`` frame with this run.
    """

    memory_card_entity_id: str
    agent_entity_id: str
    run_id: str
    summary: RunSummary
    agent_recorded: bool = True
    final_output: str | None = None
    """The chain's final answer text (``result.get_final_output()``) —
    surfaced for headless callers (``care run`` prints it as ``answer``)."""


def summarise_reasoning_result(result: Any) -> RunSummary:
    """Duck-type ``result`` into a :class:`RunSummary`.

    Recognised fields (all optional — missing attributes default
    to a sensible empty value):

    - ``success: bool`` — overall outcome.
    - ``steps: list`` / ``step_results: list`` — for ``step_count``
      (whichever exists).
    - ``duration_seconds: float`` / ``elapsed_seconds: float``.
    - ``total_tokens: int`` / ``tokens: int``.
    - ``error_message: str`` / ``error: str``.
    - ``metrics: dict`` — passed through verbatim.

    A plain ``dict`` is accepted too — the helper reads by ``[]``
    in that case so callers can hand in a JSON blob without
    wrapping it in a stub object.
    """
    def _get(name: str, default: Any = None) -> Any:
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)

    success_raw = _get("success", None)
    if success_raw is None:
        # Fall back: if there's no error_message AND step_count > 0,
        # assume success. This matches CARL where ``ReasoningResult``
        # always carries an explicit success flag — the fallback is
        # for ad-hoc dict input.
        error = _get("error_message") or _get("error")
        metrics = _get("metrics")
        exit_status = ""
        if isinstance(metrics, dict):
            exit_status = str(metrics.get("exit_status") or "").lower()
        # Honour an explicit failure signalled only via metrics.exit_status,
        # so a failed run isn't recorded (and tagged) as a success.
        success = error is None and exit_status not in {"failed", "error"}
    else:
        success = bool(success_raw)

    steps = _get("step_results") or _get("steps") or []
    step_count = len(steps) if hasattr(steps, "__len__") else 0
    duration = float(_get("duration_seconds") or _get("elapsed_seconds") or 0.0)
    tokens_raw = _get("total_tokens", _get("tokens"))
    total_tokens = int(tokens_raw) if tokens_raw is not None else None
    error_message = _get("error_message") or _get("error")
    metrics = _get("metrics") or {}

    return RunSummary(
        success=success,
        step_count=step_count,
        duration_seconds=duration,
        total_tokens=total_tokens,
        error_message=error_message,
        metrics=dict(metrics),
    )


def _build_memory_card_content(
    *,
    agent_entity_id: str,
    agent_name: str,
    query: str | None,
    summary: RunSummary,
    run_id: str,
    finished_at: datetime,
) -> dict[str, Any]:
    """Build the ``MemoryCardSpec``-shaped dict that CARE persists."""
    metrics: dict[str, Any] = {
        "duration_seconds": summary.duration_seconds,
        "step_count": summary.step_count,
        "exit_status": summary.status_label,
    }
    if summary.total_tokens is not None:
        metrics["total_tokens"] = summary.total_tokens
    if summary.metrics:
        metrics.update(summary.metrics)

    description = f"Run of agent {agent_name} — {summary.status_label}"
    if summary.error_message:
        description = f"{description}: {summary.error_message[:200]}"

    return {
        "category": "agent_run",
        "task_description": query,
        "description": description,
        "keywords": [
            "agent_run",
            f"agent:{agent_entity_id}",
            summary.status_label,
        ],
        "usage": {
            "run_id": run_id,
            "agent_entity_id": agent_entity_id,
            "agent_name": agent_name,
            "finished_at": finished_at.isoformat(),
            "metrics": metrics,
        },
    }


def _build_run_id(now: datetime) -> str:
    """Time-ordered, human-readable, collision-resistant-enough.

    Format: ``run-YYYYMMDDTHHMMSSffffff``. CARE never relies on
    this being unique across machines — the caller can pass their
    own (e.g. ULID) if needed.
    """
    return f"run-{now.strftime('%Y%m%dT%H%M%S%f')}"


def record_run_completion(
    memory: CareMemory,
    *,
    agent_entity_id: str,
    agent_name: str,
    result: Any,
    query: str | None = None,
    run_id: str | None = None,
    agent_entity_type: str = "chain",
    extra_tags: list[str] | None = None,
    author: str | None = None,
    finished_at: datetime | None = None,
) -> RunCompletion:
    """Persist a memory_card + bump run counters; return a typed
    :class:`RunCompletion`.

    Args:
        memory: A constructed :class:`care.CareMemory`.
        agent_entity_id: The source agent / chain entity id.
        agent_name: Display name (for the card description).
        result: A CARL ``ReasoningResult``, a ``RunSummary``, or
            any duck-typed object / dict :func:`summarise_reasoning_result`
            recognises.
        query: Original user query, persisted on the card's
            ``task_description``.
        run_id: Caller-supplied correlation id. ``None`` synthesises
            one from the current UTC time.
        agent_entity_type: One of ``"chain" | "agent" | "agent_skill"``.
            Tells the SDK which typed router to hit when bumping
            ``run_count`` / ``last_run_at``.
        extra_tags: Additional tags merged into ``meta.tags`` on
            the saved memory card (on top of the always-present
            ``"agent_run"`` and ``"agent:{id}"``).
        author: Forwarded to ``save_memory_card``.
        finished_at: Override the ``finished_at`` timestamp written
            into the card. Defaults to ``datetime.now(timezone.utc)``.

    Returns:
        :class:`RunCompletion`. ``agent_recorded`` is ``False`` if
        the SDK's ``record_run`` call raised (the card still got
        saved — we don't roll back persistence on a counter-bump
        failure).
    """
    finished_at = finished_at or datetime.now(timezone.utc)
    run_id = run_id or _build_run_id(finished_at)

    if isinstance(result, RunSummary):
        summary = result
    else:
        summary = summarise_reasoning_result(result)

    card_content = _build_memory_card_content(
        agent_entity_id=agent_entity_id,
        agent_name=agent_name,
        query=query,
        summary=summary,
        run_id=run_id,
        finished_at=finished_at,
    )
    card_tags = [
        "agent_run",
        f"agent:{agent_entity_id}",
        f"status:{summary.status_label}",
    ]
    if extra_tags:
        for tag in extra_tags:
            if tag and tag not in card_tags:
                card_tags.append(tag)

    card_entity_id = memory.save_memory_card(
        card_content,
        name=f"{agent_name} · {summary.status_label} · {run_id}",
        tags=card_tags,
        when_to_use=(
            f"Replay context / debug for run {run_id} of agent {agent_name}."
        ),
        author=author,
    )

    agent_recorded = True
    try:
        memory.client._record_run(  # type: ignore[attr-defined]
            agent_entity_type, agent_entity_id, run_id=run_id
        )
    except Exception:  # noqa: BLE001
        # Memory might be momentarily unreachable; the card has
        # already landed so we leave the failure as a soft signal
        # on the return value. Callers can retry.
        agent_recorded = False

    return RunCompletion(
        memory_card_entity_id=card_entity_id,
        agent_entity_id=agent_entity_id,
        run_id=run_id,
        summary=summary,
        agent_recorded=agent_recorded,
        final_output=extract_final_output(result),
    )


def extract_final_output(result: Any) -> str | None:
    """Best-effort final answer text from a duck-typed run result."""
    try:
        getter = getattr(result, "get_final_output", None)
        value = getter() if callable(getter) else getattr(result, "final_output", None)
        return str(value) if value is not None else None
    except Exception:  # noqa: BLE001 — cosmetic surface only
        return None


__all__ = [
    "RunCompletion",
    "RunSummary",
    "record_run_completion",
    "summarise_reasoning_result",
]
