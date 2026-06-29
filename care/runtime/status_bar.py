"""Status-bar data layer (TODO §1 P1).

CARE's TUI footer pins a single-line status strip showing the
liveness of every external dependency plus the current
session-level telemetry — the kind of thing a developer glances
at to confirm "yes, the wiring is live" before submitting a
generation.

The TODO §1 P1 specifies five facts in the strip:

* Memory API health (last successful ping + age).
* Platform API health (same).
* Current LLM model the active MAGE/CARL run will use.
* Total LLM tokens consumed in the session.
* Current run-id (when something is in flight).

The Textual widget itself is gated on the TODO §1 P0 multi-screen
workflow, but the data layer is independent and well-bounded —
this module ships it so the widget is a thin formatting + refresh
shell when the screens land.

Contents:

* :class:`SessionTokenCounter` — thread-safe accumulator the CARL
  streamer + MAGE poster push into. Tracks prompt / completion /
  total separately so the bar can render "12.3k in / 4.7k out"
  when there's room.
* :class:`HealthSnapshot` — frozen per-service health row
  (status / latency / age / detail). Reuses the
  :class:`care.first_run.ProbeResult` shape so the SettingsScreen
  + status bar can share probe results when the same call already
  ran.
* :class:`StatusBarSnapshot` — full bar payload (the three
  service rows + model name + token counter + active-run id +
  captured-at timestamp). Frozen.
* :func:`aggregate_status_bar` — async helper that fans out the
  health probes (with a per-service timeout so a hung Memory
  doesn't freeze the bar) and assembles a snapshot.

Duck-typed at the boundaries: the function accepts anything that
exposes ``health_check()`` for memory + platform, and any
:class:`SessionTokenCounter`-like object for the token tally.
Tests inject lightweight stubs.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional

from care.config import CareConfig


HealthStatus = Literal["ok", "skipped", "failed", "unknown"]
"""Per-service liveness states for the status bar.

* ``ok`` — service responded within the probe timeout.
* ``skipped`` — config field for the service is empty (e.g.
  ``platform.base_url`` not set) so the probe never tried.
* ``failed`` — probe ran but errored / timed out.
* ``unknown`` — no probe has run yet this session (initial
  snapshot before the first refresh tick).
"""


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionTokenTotals:
    """Immutable snapshot of the session-wide token counter.

    Three fields so the bar can render either a single combined
    number (``total``) or the input/output split when there's
    room. ``calls`` counts the number of LLM responses that
    contributed — useful for "avg tokens / call" debugging.
    """

    prompt: int = 0
    completion: int = 0
    total: int = 0
    calls: int = 0


class SessionTokenCounter:
    """Thread-safe accumulator the LLM streamers push into.

    The MAGE poster's :class:`StageCompleted` carries usage on
    every stage; the CARL streamer's :class:`ChainCompleted`
    carries chain-level usage. Both can call :meth:`add` with the
    dict shape CARL emits (``{"prompt", "completion", "total"}``)
    and the counter does the arithmetic.

    Tests grab a snapshot via :meth:`snapshot` to assert state
    without touching internals.
    """

    def __init__(self) -> None:
        self._prompt = 0
        self._completion = 0
        self._total = 0
        self._calls = 0
        self._lock = threading.Lock()

    def add(self, usage: dict[str, Any] | None) -> None:
        """Fold one LLM-response usage dict into the running total.

        Accepts the standard CARL shape ``{"prompt", "completion",
        "total"}``; missing keys default to 0. ``None`` / empty
        dicts are no-ops (CARL emits ``{}`` when the provider
        didn't surface usage). Non-int values are coerced via
        ``int()`` so a string from a flaky provider doesn't
        crash the counter — unparseable values just count as 0
        (the call still bumps the call counter so the avg
        calculation reflects the missing data).
        """
        if not usage:
            return
        prompt = _coerce_int(usage.get("prompt"))
        completion = _coerce_int(usage.get("completion"))
        total = _coerce_int(usage.get("total"))
        if total == 0 and (prompt or completion):
            total = prompt + completion
        with self._lock:
            self._prompt += prompt
            self._completion += completion
            self._total += total
            self._calls += 1

    def reset(self) -> None:
        """Zero every running counter (call between sessions)."""
        with self._lock:
            self._prompt = 0
            self._completion = 0
            self._total = 0
            self._calls = 0

    def snapshot(self) -> SessionTokenTotals:
        """Frozen-snapshot view — safe to pass through Textual
        messages and other persistence layers."""
        with self._lock:
            return SessionTokenTotals(
                prompt=self._prompt,
                completion=self._completion,
                total=self._total,
                calls=self._calls,
            )


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Health snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthSnapshot:
    """One service's status-bar row.

    Mirrors :class:`care.first_run.ProbeResult` but adds
    ``checked_at`` so the bar can render "memory ✓ (3s ago)" — a
    stale snapshot is still surfaced rather than silently masked
    as unknown.
    """

    service: Literal["memory", "platform", "mage"]
    status: HealthStatus = "unknown"
    latency_ms: float | None = None
    error: str = ""
    checked_at: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def age_seconds(self, *, now: float | None = None) -> float | None:
        """Seconds since the probe ran. ``None`` when never
        probed."""
        if self.checked_at is None:
            return None
        return max(0.0, (now if now is not None else time.time()) - self.checked_at)


# ---------------------------------------------------------------------------
# Aggregate snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatusBarSnapshot:
    """Complete payload the future ``StatusBar`` widget renders.

    Frozen so it ships through Textual messages without defensive
    copies. The widget is the only consumer that knows how to
    pixel-budget the strip; this layer just gives it every fact
    in one immutable bundle.
    """

    memory: HealthSnapshot
    platform: HealthSnapshot
    model: str
    # Short host label derived from ``MageConfig.base_url`` —
    # e.g. ``"openrouter.ai"``. Empty when base_url isn't
    # configured. Renders next to the model name in the status
    # strip.
    endpoint: str
    tokens: SessionTokenTotals
    # §1 P0 — MAGE health joins memory + platform as the
    # third dot in the strip. Defaults to a `unknown` snapshot
    # so callers (eg. tests constructing StatusBarSnapshot
    # by hand pre-iter-17) keep working; the aggregator
    # always fills it.
    mage: HealthSnapshot = field(
        default_factory=lambda: HealthSnapshot(service="mage"),
    )
    active_run_id: Optional[str] = None
    active_run_label: Optional[str] = None
    captured_at: float = field(default_factory=time.time)

    @property
    def has_active_run(self) -> bool:
        return self.active_run_id is not None

    def format_text(self, *, now: float | None = None) -> str:
        """Single-line human-readable rendering. The widget will
        do something prettier with markup, but this matches the
        shape callers can dump into logs or a debug command."""
        now_ts = now if now is not None else time.time()
        parts: list[str] = []
        # §1 P0 — mage joins memory + platform as the third
        # health dot so the boot strip shows the user every
        # connection that matters before the first prompt.
        for snap in (self.mage, self.memory, self.platform):
            badge = {
                "ok": "✓",
                "skipped": "·",
                "failed": "✗",
                "unknown": "?",
            }.get(snap.status, "?")
            chunk = f"{snap.service} {badge}"
            age = snap.age_seconds(now=now_ts)
            if age is not None and snap.status == "ok":
                chunk += f" ({_format_age(age)})"
            elif snap.error and snap.status == "failed":
                chunk += f" ({snap.error})"
            parts.append(chunk)
        model_part = (
            f"{self.model} @ {self.endpoint}"
            if self.model and self.endpoint
            else (self.model or self.endpoint)
        )
        if model_part:
            parts.append(model_part)
        if self.tokens.total or self.tokens.calls:
            parts.append(f"{_format_tokens(self.tokens.total)} tok")
        if self.active_run_id:
            run_chunk = f"run {self.active_run_id[:8]}"
            if self.active_run_label:
                run_chunk += f" ({self.active_run_label})"
            parts.append(run_chunk)
        return " · ".join(parts)


def _format_age(seconds: float) -> str:
    if seconds < 1:
        return "now"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    return f"{int(seconds / 3600)}h ago"


def _format_tokens(total: int) -> str:
    if total < 1000:
        return str(total)
    if total < 1_000_000:
        return f"{total / 1000:.1f}k"
    return f"{total / 1_000_000:.1f}M"


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


async def probe_health(
    *,
    service: Literal["memory", "platform"],
    facade: Any,
    timeout: float = 2.0,
) -> HealthSnapshot:
    """Hit ``facade.health_check()`` with a deadline.

    The status bar refreshes on a tick (e.g. every 5s); a hung
    Memory shouldn't freeze the bar. A timeout flips the snapshot
    to ``failed`` with an explanatory error rather than blocking
    forever.

    Args:
        service: Which row the snapshot is for.
        facade: Anything with a sync ``health_check() -> dict``
            method. Production passes :class:`CareMemory` /
            :class:`CarePlatform`; tests pass stubs.
        timeout: Per-probe deadline in seconds.

    Returns:
        :class:`HealthSnapshot` with ``checked_at`` stamped.
    """
    if facade is None:
        return HealthSnapshot(
            service=service,
            status="skipped",
            error=f"{service} facade not configured",
            checked_at=time.time(),
        )
    start = time.monotonic()
    try:
        detail = await asyncio.wait_for(
            asyncio.to_thread(facade.health_check),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        latency = (time.monotonic() - start) * 1000
        return HealthSnapshot(
            service=service,
            status="failed",
            latency_ms=latency,
            error=f"timed out after {timeout:.1f}s",
            checked_at=time.time(),
        )
    except Exception as exc:  # noqa: BLE001
        latency = (time.monotonic() - start) * 1000
        return HealthSnapshot(
            service=service,
            status="failed",
            latency_ms=latency,
            error=f"{type(exc).__name__}: {exc}",
            checked_at=time.time(),
        )
    latency = (time.monotonic() - start) * 1000
    detail_dict = detail if isinstance(detail, dict) else {"raw": detail}
    return HealthSnapshot(
        service=service,
        status="ok",
        latency_ms=latency,
        checked_at=time.time(),
        detail=detail_dict,
    )


async def probe_mage_health(
    *,
    config: Any,
    timeout: float = 2.0,
) -> HealthSnapshot:
    """Cheap "is MAGE reachable?" check.

    Unlike memory/platform, MAGE doesn't expose a `health_check`
    method on the facade (the upstream `mmar_mage.generate` is
    the only entry point). We score health off the three knobs
    the user can actually configure:

    * `config.mage.api_key` and `config.mage.base_url` both set
      → status ``ok``. The widget renders this as a green dot
      meaning "MAGE is configured + ready". Note that we don't
      actually call the LLM here — a real probe would burn
      tokens on every status refresh.
    * Either knob missing → status ``skipped`` with an
      explanatory error so the user knows what to fix.
    * Anything else (config-load failure, attribute error) →
      ``failed`` with the exception class name.

    Future enhancement: when the wizard ships, add a one-token
    "echo" call gated behind an env var so users who want a
    live probe can opt in.
    """
    if config is None:
        return HealthSnapshot(
            service="mage",
            status="skipped",
            error="config not loaded",
            checked_at=time.time(),
        )
    try:
        mage = getattr(config, "mage", None)
        if mage is None:
            return HealthSnapshot(
                service="mage",
                status="skipped",
                error="mage config missing",
                checked_at=time.time(),
            )
        api_key = getattr(mage, "api_key", None) or ""
        base_url = getattr(mage, "base_url", None) or ""
        model = getattr(mage, "model", None) or ""
        if not api_key:
            return HealthSnapshot(
                service="mage",
                status="skipped",
                error="api_key not set",
                checked_at=time.time(),
                detail={
                    "base_url": base_url, "model": model,
                },
            )
        if not base_url:
            return HealthSnapshot(
                service="mage",
                status="skipped",
                error="base_url not set",
                checked_at=time.time(),
                detail={"model": model},
            )
        # `timeout` accepted for API parity with `probe_health`;
        # the no-op probe doesn't await anything.
        _ = timeout
        return HealthSnapshot(
            service="mage",
            status="ok",
            checked_at=time.time(),
            detail={"base_url": base_url, "model": model},
        )
    except Exception as exc:  # noqa: BLE001
        return HealthSnapshot(
            service="mage",
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            checked_at=time.time(),
        )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


async def aggregate_status_bar(
    *,
    config: CareConfig,
    memory: Any = None,
    platform: Any = None,
    token_counter: SessionTokenCounter | None = None,
    active_task: Any = None,
    timeout: float = 2.0,
) -> StatusBarSnapshot:
    """Build a fresh :class:`StatusBarSnapshot`.

    Args:
        config: CARE's runtime config; supplies the model + provider
            for the bar.
        memory: A :class:`CareMemory`-like facade. ``None`` produces
            a ``skipped`` memory row (e.g. before the first-run
            wizard ran).
        platform: A :class:`CarePlatform`-like facade. ``None`` →
            ``skipped``.
        token_counter: The session's :class:`SessionTokenCounter`.
            ``None`` produces a zeroed :class:`SessionTokenTotals`.
        active_task: Anything with ``id`` and ``label`` attributes
            (a :class:`TaskRecord`, a frozen dict-shaped object,
            etc.). ``None`` means "no run in flight".
        timeout: Per-probe deadline. The two health probes run
            concurrently so the wall-clock floor is one timeout,
            not two.

    Returns:
        Frozen :class:`StatusBarSnapshot` ready for the widget.
    """
    memory_task = probe_health(
        service="memory", facade=memory, timeout=timeout
    )
    platform_task = probe_health(
        service="platform", facade=platform, timeout=timeout
    )
    mage_task = probe_mage_health(config=config, timeout=timeout)
    memory_snap, platform_snap, mage_snap = await asyncio.gather(
        memory_task, platform_task, mage_task,
    )

    tokens = (
        token_counter.snapshot()
        if token_counter is not None
        else SessionTokenTotals()
    )
    run_id: Optional[str] = None
    run_label: Optional[str] = None
    if active_task is not None:
        run_id = _read_attr(active_task, "id")
        run_label = _read_attr(active_task, "label")

    return StatusBarSnapshot(
        memory=memory_snap,
        platform=platform_snap,
        mage=mage_snap,
        model=config.mage.model or "",
        endpoint=_endpoint_label(config.mage.base_url),
        tokens=tokens,
        active_run_id=run_id,
        active_run_label=run_label,
    )


def _endpoint_label(base_url: str | None) -> str:
    """Reduce a base URL to a short host label for the strip.

    ``https://openrouter.ai/api/v1`` → ``openrouter.ai``;
    ``http://localhost:11434`` → ``localhost:11434``; empty
    when ``base_url`` is blank. Falls back to the raw string
    when the URL doesn't parse.
    """
    if not base_url:
        return ""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(base_url)
    except Exception:
        return base_url
    return parsed.netloc or base_url


def derive_from_task_registry(registry: Any) -> Any:
    """Pick a sensible "current run" out of a :class:`TaskRegistry`.

    The status bar shows ONE active run at a time. The choice rule
    is the same one users mentally apply: prefer the most recently
    started task that's still running, falling back to the most
    recently started pending task, then to ``None``. This is a
    pure projection — no I/O, no side effects.

    Returns the :class:`TaskRecord` directly (preserving the
    record's identity so callers can chain :meth:`cancel`) or
    ``None``.
    """
    if registry is None:
        return None
    list_tasks = getattr(registry, "list_tasks", None)
    if list_tasks is None:
        return None
    running = list_tasks(status="running")
    if running:
        # `list_tasks` already sorts pending-first then by
        # started_at asc; for "most recent" we want the last
        # element with a started_at stamp.
        timed = [t for t in running if getattr(t, "started_at", None)]
        if timed:
            return max(timed, key=lambda t: t.started_at)
        return running[-1]
    pending = list_tasks(status="pending")
    if pending:
        return pending[-1]
    return None


def _read_attr(obj: Any, name: str) -> Optional[str]:
    """Read ``name`` off a frozen object OR a dict; coerce to str
    so the snapshot never carries a UUID / other non-string. The
    status bar formats with string ops only."""
    if isinstance(obj, dict):
        value = obj.get(name)
    else:
        value = getattr(obj, name, None)
    return str(value) if value is not None else None


__all__ = [
    "HealthSnapshot",
    "HealthStatus",
    "SessionTokenCounter",
    "SessionTokenTotals",
    "StatusBarSnapshot",
    "aggregate_status_bar",
    "derive_from_task_registry",
    "probe_health",
    "probe_mage_health",
]


# Mirror `dataclasses.replace` for callers building diff'd
# snapshots (e.g. flipping `active_run_id` between refresh ticks
# without re-running every probe). Re-exported here so consumers
# don't have to import dataclasses themselves.
update_snapshot = replace
