"""Per-step profiling projection (TODO §5 P1).

After a CARL chain finishes, CARE's ExecutionScreen wants to
show a per-step profiling table: wall-clock time, history bytes
added, working-memory size, batch index, plus chain-level
totals (peak memory, total tokens, total time). CARL exposes
the raw data on every :class:`StepExecutionResult.profiling`
dict — this module projects it into a CARE-stable shape ready
for the screen.

Duck-typed against ``mmar_carl``: the projector accepts three
shapes so it works against future CARL versions, older CARL
versions that don't ship ``get_profiling_summary()`` yet, and
serialised replay artefacts:

1. A `ReasoningResult`-like with `.get_profiling_summary()` —
   preferred, lets CARL own the shape.
2. A `ReasoningResult`-like *without* that method (older
   CARL) — we walk `.step_results` + `.history` ourselves to
   build the same dict shape.
3. The summary dict directly — useful when callers replay a
   saved run.

The screen reads :attr:`ProfilingSummary.steps` for the table
rows and :meth:`format_text` for a footer / CLI summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StepProfile:
    """Profiling data for one step.

    Frozen so the row can be passed into a Textual DataTable
    cell without defensive copies.

    Fields mirror CARL's ``StepExecutionResult.profiling`` dict
    plus the basics from the step result itself (number, title,
    type). Missing source values default to ``0`` / ``""`` so
    the row always renders cleanly.
    """

    step_number: int
    step_title: str
    step_type: str
    execution_time_s: float
    history_bytes_added: int
    memory_bytes_after: int
    history_bytes_after: int
    batch_index: int | None
    skipped: bool = False
    success: bool = True


@dataclass(frozen=True)
class ProfilingSummary:
    """Per-step + chain-level profiling.

    Returned by :func:`project_profiling`. Frozen so the
    summary flows through screens / log handlers without
    defensive copies.

    Fields:
        steps: One :class:`StepProfile` per step, in execution
            order.
        total_execution_time_s: Sum of every step's wall-clock
            time. Source: CARL's chain-level total when
            available, otherwise computed from the steps.
        total_history_bytes: Cumulative history size at chain
            end (sum of entry lengths).
        peak_memory_bytes: Largest ``memory_bytes_after`` seen
            across all steps.
        token_usage: Token-usage dict from CARL — typically
            ``{"total_tokens": N, "prompt_tokens": ...,
            "completion_tokens": ...}``. Free-form so future
            CARL token additions land without a schema bump.
    """

    steps: tuple[StepProfile, ...] = field(default_factory=tuple)
    total_execution_time_s: float = 0.0
    total_history_bytes: int = 0
    peak_memory_bytes: int = 0
    token_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def format_text(self) -> str:
        """Multi-line human-readable summary.

        Renders into the ExecutionScreen footer + ``care run``
        CLI output once that subcommand lands. Empty when no
        steps were recorded.
        """
        if self.is_empty:
            return "no profiling data"
        lines = [
            f"steps: {self.step_count}",
            f"total time: {self.total_execution_time_s:.3f}s",
            f"peak memory: {_format_bytes(self.peak_memory_bytes)}",
            f"total history: {_format_bytes(self.total_history_bytes)}",
        ]
        if self.token_usage:
            total_tokens = self.token_usage.get("total_tokens")
            if isinstance(total_tokens, int):
                lines.append(f"total tokens: {total_tokens}")
        # Per-step rows.
        lines.append("")
        lines.append("per-step:")
        for s in self.steps:
            badge = ""
            if s.skipped:
                badge = " [SKIPPED]"
            elif not s.success:
                badge = " [FAILED]"
            batch = (
                f"batch {s.batch_index}"
                if s.batch_index is not None
                else "batch -"
            )
            lines.append(
                f"  #{s.step_number} {s.step_title} ({s.step_type})"
                f"{badge} — {s.execution_time_s:.3f}s, "
                f"+{_format_bytes(s.history_bytes_added)} hist, "
                f"{_format_bytes(s.memory_bytes_after)} mem, "
                f"{batch}"
            )
        return "\n".join(lines)


def project_profiling(source: Any) -> ProfilingSummary:
    """Project CARL profiling data into a :class:`ProfilingSummary`.

    Args:
        source: One of three accepted shapes (duck-typed):

            - A :class:`mmar_carl.ReasoningResult` with a
              ``get_profiling_summary()`` method — preferred
              when available. Result of that call lands here.
            - A :class:`mmar_carl.ReasoningResult` without
              ``get_profiling_summary()`` (older CARL) — the
              projector walks ``.step_results`` + ``.history``
              + ``.token_usage`` itself.
            - The summary dict directly — useful for replay
              of a saved run.

    Returns:
        Populated :class:`ProfilingSummary`. Empty result for
        ``None`` / unrecognised inputs (never raises).
    """
    if source is None:
        return ProfilingSummary()

    # 1. Already the summary dict shape.
    if isinstance(source, dict):
        return _from_summary_dict(source)

    # 2. Has the helper method — let CARL build the dict.
    helper = getattr(source, "get_profiling_summary", None)
    if callable(helper):
        try:
            summary: Any = helper()
        except Exception:  # noqa: BLE001
            summary = None
        if isinstance(summary, dict):
            return _from_summary_dict(summary)

    # 3. Manual fallback: read `.step_results` / `.history`.
    step_results = getattr(source, "step_results", None) or []
    history = getattr(source, "history", None) or []
    total_time = getattr(source, "total_execution_time", None) or 0.0
    token_usage = getattr(source, "token_usage", None) or {}

    step_rows: list[StepProfile] = []
    peak_memory = 0
    for sr in step_results:
        prof = getattr(sr, "profiling", None) or {}
        mem_after = _opt_int(prof.get("memory_bytes_after"), 0)
        if mem_after > peak_memory:
            peak_memory = mem_after
        step_type = getattr(sr, "step_type", None)
        step_type_str = _step_type_to_str(step_type)
        step_rows.append(
            StepProfile(
                step_number=_opt_int(getattr(sr, "step_number", 0), 0),
                step_title=str(getattr(sr, "step_title", "") or ""),
                step_type=step_type_str,
                execution_time_s=_opt_float(
                    getattr(sr, "execution_time", 0.0), 0.0
                ),
                history_bytes_added=_opt_int(
                    prof.get("history_bytes_added"), 0
                ),
                memory_bytes_after=mem_after,
                history_bytes_after=_opt_int(
                    prof.get("history_bytes_after"), 0
                ),
                batch_index=_opt_int_or_none(prof.get("batch_index")),
                skipped=bool(getattr(sr, "skipped", False)),
                success=bool(getattr(sr, "success", True)),
            )
        )
    total_history = sum(len(h) for h in history if h is not None)

    return ProfilingSummary(
        steps=tuple(step_rows),
        total_execution_time_s=float(total_time),
        total_history_bytes=total_history,
        peak_memory_bytes=peak_memory,
        token_usage=dict(token_usage) if isinstance(token_usage, dict) else {},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _from_summary_dict(data: dict[str, Any]) -> ProfilingSummary:
    """Build a :class:`ProfilingSummary` from CARL's
    ``get_profiling_summary()`` output (or a replay dict that
    matches the same shape)."""
    raw_steps = data.get("steps") or []
    if not isinstance(raw_steps, list):
        raw_steps = []
    step_rows: list[StepProfile] = []
    for row in raw_steps:
        if not isinstance(row, dict):
            continue
        step_rows.append(
            StepProfile(
                step_number=_opt_int(row.get("step_number"), 0),
                step_title=str(row.get("step_title", "") or ""),
                step_type=str(row.get("step_type", "unknown") or "unknown"),
                execution_time_s=_opt_float(row.get("execution_time_s"), 0.0),
                history_bytes_added=_opt_int(
                    row.get("history_bytes_added"), 0
                ),
                memory_bytes_after=_opt_int(
                    row.get("memory_bytes_after"), 0
                ),
                history_bytes_after=_opt_int(
                    row.get("history_bytes_after"), 0
                ),
                batch_index=_opt_int_or_none(row.get("batch_index")),
                skipped=bool(row.get("skipped", False)),
                success=bool(row.get("success", True)),
            )
        )
    return ProfilingSummary(
        steps=tuple(step_rows),
        total_execution_time_s=_opt_float(
            data.get("total_execution_time_s"), 0.0
        ),
        total_history_bytes=_opt_int(data.get("total_history_bytes"), 0),
        peak_memory_bytes=_opt_int(data.get("peak_memory_bytes"), 0),
        token_usage=dict(data.get("token_usage") or {})
        if isinstance(data.get("token_usage"), dict)
        else {},
    )


_METRIC_FIELD = {
    "time": "execution_time_s",
    "history": "history_bytes_added",
    "memory": "memory_bytes_after",
}


def profiling_metric(
    summary: ProfilingSummary, kind: str = "time",
) -> dict[str, float]:
    """Project a :class:`ProfilingSummary` into a ``{step_ref: value}`` map
    for the DAG heat overlay (``render_dag_styled(metric_by_ref=…)``).

    ``kind`` selects the metric: ``"time"`` (wall-clock seconds, the
    default), ``"history"`` (history bytes added) or ``"memory"`` (working-
    memory bytes after). Keys are the step number as a string — the same
    ref the DAG renderer derives — so the overlay lines up with the boxes.
    """
    field = _METRIC_FIELD.get(kind, "execution_time_s")
    return {
        str(step.step_number): float(getattr(step, field, 0.0) or 0.0)
        for step in summary.steps
    }


def _step_type_to_str(step_type: Any) -> str:
    """Coerce CARL's StepType enum / string / None into a
    display string."""
    if step_type is None:
        return "unknown"
    if hasattr(step_type, "value"):
        return str(step_type.value)
    return str(step_type)


def _opt_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _opt_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _format_bytes(n: int) -> str:
    """Format a byte count with K/M/G suffixes — kept tight so
    DataTable rows stay scannable."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}K"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}M"
    return f"{n / (1024 * 1024 * 1024):.2f}G"


__all__ = [
    "ProfilingSummary",
    "StepProfile",
    "project_profiling",
    "profiling_metric",
]
