"""Local JSONL run-history (TODO §6 P1).

CARE's existing `run_history.py` reads run completions from
Memory cards (one card per run, queried per chain). The §6 P1
`/runs` screen wants something different: a *global* history
of every chain execution this user has ever fired, surfacing
without a Memory round-trip.

This module owns the local persistence layer:

* JSONL files under ``~/.cache/care/runs/<YYYY-MM-DD>.jsonl``.
  One row per recorded run. Daily file rotation caps any
  single file's growth even on a heavy day.
* :func:`record_local_run(entry)` appends one entry to the
  current day's file. Atomic-append via ``open(..., "a")``;
  cheap enough that the executor can call it inline after
  every chain.
* :func:`load_local_runs(*, limit=N)` reads back the most
  recent ``limit`` entries (newest first) across all daily
  files. The screen reads this on mount + on the `r`
  refresh binding.

The cache directory itself is guaranteed by
:func:`care.runtime.user_paths.ensure_user_dirs` which
:class:`CareApp.__init__` runs at boot.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from care.runtime.user_paths import CARE_CACHE_DIR

_log = logging.getLogger("care.runtime.local_run_history")


RUNS_SUBDIR = "runs"
"""Subdirectory under the cache root that holds per-day
JSONL files."""


REPLAYS_SUBDIR = "runs/replays"
"""Subdirectory under the cache root that holds one
``<run_id>.json`` per recorded run (§6 P1 ReplayScreen
sidecar). The /runs screen reads these via
:func:`care.load_replay` to drill into a run's per-step
detail."""


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
"""Validates filename stems so a stray ``hello.jsonl`` can't
mask the real runs."""


@dataclass(frozen=True)
class LocalRunEntry:
    """One row in the /runs screen.

    Frozen so snapshots flow through workers without
    defensive copies. Fields match the §6 P1 spec
    (`chain_id`, `started_at`, `duration`, `status`,
    `tokens_in`, `tokens_out`, `cost`, `error`) plus a
    `run_id` for stable navigation.
    """

    run_id: str
    chain_id: str = ""
    chain_name: str = ""
    started_at: float = 0.0
    duration_seconds: float | None = None
    status: str = "success"
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str = ""
    mode: str = ""
    provider: str = ""
    replay_path: str = ""
    """§6 P1 — Path to a sidecar JSON file (under
    ``~/.cache/care/runs/replays/<run_id>.json``) that the
    `/runs` screen can deserialise via
    :func:`care.load_replay` to push the existing
    `ReplayScreen`. Empty string when the recorder didn't
    have a replay-capable result (e.g. CARL crashed before
    finishing) or the sidecar write failed."""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_total(self) -> int | None:
        if self.tokens_in is None and self.tokens_out is None:
            return None
        return (self.tokens_in or 0) + (self.tokens_out or 0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def runs_dir(*, cache_dir: Path | None = None) -> Path:
    root = cache_dir if cache_dir is not None else CARE_CACHE_DIR
    return root / RUNS_SUBDIR


def replays_dir(*, cache_dir: Path | None = None) -> Path:
    root = cache_dir if cache_dir is not None else CARE_CACHE_DIR
    return root / REPLAYS_SUBDIR


def write_replay_sidecar(
    result: Any,
    *,
    run_id: str,
    cache_dir: Path | None = None,
) -> str:
    """Serialise ``result`` to
    ``<cache_dir>/runs/replays/<run_id>.json`` for the
    `/runs → Enter` drill.

    Returns the resolved path string on success or ``""`` on
    any failure (no serialisable shape, disk error). The
    failure cases collapse to empty so the run-history row's
    `replay_path` reads as "no replay available".

    Accepts both `ReasoningResult`-like objects (with a
    ``to_dict`` method) and raw dicts (which CARL-side
    helpers emit). Strings pass through verbatim.
    """
    payload: Any = None
    if result is None:
        return ""
    if isinstance(result, str):
        payload = result
    elif isinstance(result, dict):
        payload = result
    elif hasattr(result, "to_dict"):
        try:
            payload = result.to_dict()
        except Exception:  # noqa: BLE001
            return ""
    else:
        # Last-ditch attempt — dump dataclass-like via
        # `dataclasses.asdict` when available.
        try:
            from dataclasses import asdict, is_dataclass

            if is_dataclass(result):
                payload = asdict(result)
        except Exception:  # noqa: BLE001
            return ""
    if payload is None:
        return ""
    target = replays_dir(cache_dir=cache_dir) / f"{run_id}.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fp:
            if isinstance(payload, str):
                fp.write(payload)
            else:
                json.dump(payload, fp, default=str)
    except OSError:
        return ""
    return str(target)


def _day_file(
    *, when: float | None = None, cache_dir: Path | None = None,
) -> Path:
    """Resolve the JSONL path for the given timestamp's day."""
    ts = when if when is not None else time.time()
    stem = time.strftime("%Y-%m-%d", time.localtime(ts))
    return runs_dir(cache_dir=cache_dir) / f"{stem}.jsonl"


def record_local_run(
    entry: LocalRunEntry,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Append ``entry`` to the day's JSONL file.

    Creates the runs directory on demand (defensive: even if
    `ensure_user_dirs` ran the file might be missing if the
    user wiped the cache). Returns the resolved file path so
    the caller can log it.

    Failures (permission denied / full disk) raise — the
    executor's `try` block catches them so a recording
    failure doesn't kill the run.
    """
    target = _day_file(
        when=entry.started_at or None, cache_dir=cache_dir,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fp:
        fp.write(
            json.dumps(entry.to_dict(), default=str) + "\n",
        )
    return target


def load_local_runs(
    *,
    cache_dir: Path | None = None,
    limit: int | None = None,
) -> list[LocalRunEntry]:
    """Load runs from every daily JSONL, newest first.

    Args:
        cache_dir: Override the cache root (tests).
        limit: Cap the returned list. ``None`` returns
            everything. The /runs screen passes 200 to keep
            the table responsive on heavy users.

    Returns:
        List of :class:`LocalRunEntry` sorted by
        ``started_at`` descending. Malformed JSON lines are
        skipped + logged at WARNING — a corrupted row should
        never block the read.
    """
    out: list[LocalRunEntry] = []
    target_dir = runs_dir(cache_dir=cache_dir)
    if not target_dir.is_dir():
        return out
    files = []
    for entry in target_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".jsonl":
            continue
        if not _DATE_RE.match(entry.stem):
            continue
        files.append(entry)
    # Newest day first — date-ordered descending.
    files.sort(key=lambda p: p.stem, reverse=True)
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as fp:
                for lineno, line in enumerate(fp, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        _log.warning(
                            "load_local_runs: bad json at "
                            "%s:%d (%s)", path, lineno, exc,
                        )
                        continue
                    try:
                        entry = _from_raw(raw)
                    except (KeyError, ValueError, TypeError) as exc:
                        _log.warning(
                            "load_local_runs: bad row at "
                            "%s:%d (%s)", path, lineno, exc,
                        )
                        continue
                    out.append(entry)
        except OSError as exc:
            _log.warning(
                "load_local_runs: skip %s — %s", path, exc,
            )
            continue
    out.sort(key=lambda e: e.started_at, reverse=True)
    if limit is not None:
        out = out[:limit]
    return out


def _from_raw(raw: dict[str, Any]) -> LocalRunEntry:
    """Project a raw JSON dict into :class:`LocalRunEntry`.

    Tolerates missing optional fields; ``run_id`` is
    required (no stable handle = unsavable row)."""
    if not raw.get("run_id"):
        raise ValueError("missing run_id")
    return LocalRunEntry(
        run_id=str(raw["run_id"]),
        chain_id=str(raw.get("chain_id") or ""),
        chain_name=str(raw.get("chain_name") or ""),
        started_at=_coerce_float(raw.get("started_at"), 0.0),
        duration_seconds=_coerce_optional_float(
            raw.get("duration_seconds"),
        ),
        status=str(raw.get("status") or "success"),
        tokens_in=_coerce_optional_int(raw.get("tokens_in")),
        tokens_out=_coerce_optional_int(raw.get("tokens_out")),
        cost_usd=_coerce_optional_float(raw.get("cost_usd")),
        error=str(raw.get("error") or ""),
        mode=_normalise_mode_label(str(raw.get("mode") or "")),
        provider=str(raw.get("provider") or ""),
        replay_path=str(raw.get("replay_path") or ""),
        extra=dict(raw.get("extra") or {}),
    )


def _normalise_mode_label(value: str) -> str:
    """Map the legacy ``ad_hoc`` mode label onto ``interactive`` so
    historical run rows group consistently with current ones in the
    cost/usage rollups. Self-contained (no ``screens`` import) to keep
    the runtime layer free of UI dependencies."""
    return "interactive" if value in ("ad_hoc", "ad-hoc", "adhoc") else value


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_run_entry(
    *,
    run_id: str,
    chain: Any,
    task: str = "",
    result: Any = None,
    started_at: float,
    duration: float,
    status: str,
    error: str = "",
    mode: str = "",
    provider: str = "",
    extra: dict[str, Any] | None = None,
    write_replay: bool = False,
    cache_dir: Path | None = None,
) -> LocalRunEntry:
    """Project the executor-side ``chain`` + ``result`` into
    a :class:`LocalRunEntry` (TODO §6 P1 shared helper).

    The three execution call sites (ad-hoc CARL in
    `ChatScreen._execute_chain_interactive`, the `care run` CLI,
    and the Production dataset runner) all need the same
    projection: defensive reads for `chain.entity_id` /
    `chain.name`, dual-shape token extraction
    (CARL `{prompt, completion}` vs OpenAI
    `{prompt_tokens, completion_tokens}`), and an
    `extra["task"]` slot truncated to 200 chars for triage.

    Centralising the projection here lets the three call
    sites stay short + identical in field shape, so future
    `/runs` columns can be wired in one place rather than
    three.
    """
    chain_id = ""
    chain_name = ""
    if chain is not None:
        chain_id = str(getattr(chain, "entity_id", "") or "")
        chain_name = str(getattr(chain, "name", "") or "")
    tokens_in: int | None = None
    tokens_out: int | None = None
    if result is not None:
        usage = (
            getattr(result, "usage", None)
            or getattr(result, "token_usage", None)
            or {}
        )
        if isinstance(usage, dict):
            # `.get(key, default)` keys on presence, not truthiness, so a real
            # 0-token count is preserved instead of falling through to None.
            raw_in = usage.get("prompt", usage.get("prompt_tokens"))
            raw_out = usage.get("completion", usage.get("completion_tokens"))
            if raw_in is not None:
                try:
                    tokens_in = int(raw_in)
                except (TypeError, ValueError):
                    tokens_in = None
            if raw_out is not None:
                try:
                    tokens_out = int(raw_out)
                except (TypeError, ValueError):
                    tokens_out = None
    merged_extra: dict[str, Any] = dict(extra or {})
    if task:
        merged_extra["task"] = task[:200]
    replay_path = ""
    if write_replay and result is not None:
        replay_path = write_replay_sidecar(
            result, run_id=run_id, cache_dir=cache_dir,
        )
    return LocalRunEntry(
        run_id=run_id,
        chain_id=chain_id,
        chain_name=chain_name,
        started_at=started_at,
        duration_seconds=duration,
        status=status,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        error=error,
        mode=mode,
        provider=provider,
        replay_path=replay_path,
        extra=merged_extra,
    )


@dataclass(frozen=True)
class ChainRunStats:
    """Per-chain aggregate over local run history (§4 P2).

    Computed by :func:`summarise_runs_by_chain` from the
    full set of :class:`LocalRunEntry` rows the user has
    accumulated under ``~/.cache/care/runs/``. Library row
    cells consume these to render a "last run · success rate
    · mean cost" annotation strip.
    """

    chain_id: str
    run_count: int = 0
    success_count: int = 0
    last_run_at: float | None = None
    mean_duration_seconds: float | None = None
    mean_tokens: float | None = None
    mean_cost_usd: float | None = None

    @property
    def success_rate(self) -> float | None:
        """0.0–1.0; ``None`` when no runs are recorded."""
        if self.run_count == 0:
            return None
        return self.success_count / self.run_count


def summarise_runs_by_chain(
    runs: Iterable[LocalRunEntry],
) -> dict[str, ChainRunStats]:
    """Group local run entries by ``chain_id`` and compute
    per-chain aggregates.

    Entries with an empty ``chain_id`` are skipped — they
    don't surface in the Library which keys off the saved
    chain entity. Means are computed only over the rows that
    actually carried the metric (so missing token/cost data
    on some runs doesn't pull the average toward zero).

    Returns a mapping keyed by ``chain_id``; absent keys mean
    "no runs yet" and the row falls back to its existing
    Memory-side metadata (`LibraryRow.run_count` /
    `last_run_at`).
    """
    buckets: dict[str, list[LocalRunEntry]] = {}
    for run in runs:
        cid = (run.chain_id or "").strip()
        if not cid:
            continue
        buckets.setdefault(cid, []).append(run)

    out: dict[str, ChainRunStats] = {}
    for cid, group in buckets.items():
        success_count = sum(
            1 for r in group if r.status == "success"
        )
        last_run_at = max(
            (r.started_at for r in group if r.started_at),
            default=None,
        )
        durations = [
            r.duration_seconds for r in group
            if r.duration_seconds is not None
        ]
        tokens = [
            r.tokens_total for r in group
            if r.tokens_total is not None
        ]
        costs = [
            r.cost_usd for r in group
            if r.cost_usd is not None
        ]
        out[cid] = ChainRunStats(
            chain_id=cid,
            run_count=len(group),
            success_count=success_count,
            last_run_at=last_run_at,
            mean_duration_seconds=(
                sum(durations) / len(durations)
                if durations else None
            ),
            mean_tokens=(
                sum(tokens) / len(tokens) if tokens else None
            ),
            mean_cost_usd=(
                sum(costs) / len(costs) if costs else None
            ),
        )
    return out


def format_recency(stats: ChainRunStats | None) -> str:
    """One-line "last run + success rate" strip for the
    Library "Last Run" cell.

    Returns the empty string when ``stats`` is None or carries
    no runs — the caller falls back to the Memory-side
    timestamp formatter.
    """
    if stats is None or stats.run_count == 0:
        return ""
    age = _format_relative_age(stats.last_run_at)
    rate = stats.success_rate
    rate_str = f"{rate:.2f}" if rate is not None else "—"
    return f"{age} · {rate_str}/{stats.run_count}"


def format_mean_cost(stats: ChainRunStats | None) -> str:
    """Compact mean-cost rendering for the Library "Cost"
    cell. Falls back to ``"—"`` when no cost data is in scope.

    Shape: ``"$0.42"`` for non-trivial costs, ``"<$0.01"`` for
    sub-cent means, ``"$0.00"`` for explicit zero (free
    providers), ``"—"`` when no run reported a cost field.
    """
    if stats is None or stats.mean_cost_usd is None:
        return "—"
    cost = stats.mean_cost_usd
    if cost <= 0:
        return "$0.00"
    if cost < 0.01:
        return "<$0.01"
    return f"${cost:.2f}"


def _format_relative_age(at: float | None) -> str:
    """Render a unix timestamp as a tight relative age:
    ``2h ago``, ``3d ago``, ``45m ago``, ``just now``.

    ``None`` returns ``"—"`` so the caller's f-string stays
    safe."""
    if at is None or at <= 0:
        return "—"
    import time as _time
    delta = max(0.0, _time.time() - at)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


__all__ = [
    "ChainRunStats",
    "LocalRunEntry",
    "REPLAYS_SUBDIR",
    "RUNS_SUBDIR",
    "build_run_entry",
    "format_mean_cost",
    "format_recency",
    "load_local_runs",
    "record_local_run",
    "replays_dir",
    "runs_dir",
    "summarise_runs_by_chain",
    "write_replay_sidecar",
]
