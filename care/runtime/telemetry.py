"""Opt-in telemetry sink (TODO §9 P3).

CARE drives three long-running pipelines that benefit from
out-of-process inspection — MAGE generations, CARL chain runs,
Platform evolutions. With telemetry **off** (the default), the
recording surface is a no-op so production users pay nothing.
With it **on**, every generation/run/evolution event is
forwarded to a sink — Langfuse out of the box, or any custom
backend registered via :func:`register_telemetry_backend`.

Design parallels :mod:`care.runtime.llm_client`:

* :class:`TelemetryEvent` frozen value object — kind,
  timestamp, trace_id, attributes.
* :class:`TelemetrySink` Protocol — the shape every backend
  satisfies. ``record(event)`` is the hot path; ``flush()`` /
  ``close()`` are lifecycle hooks the TUI calls on shutdown.
* :class:`NullTelemetrySink` — the no-op default. Drops
  everything; exists so callers can unconditionally call
  ``sink.record(...)`` without checking ``config.enabled``.
* :func:`build_telemetry_sink` — factory dispatching on
  ``config.telemetry.backend``. Returns ``NullTelemetrySink``
  when ``enabled=False``, regardless of backend name.
* :func:`register_telemetry_backend` / :func:`unregister_*` /
  :func:`list_known_backends` — same registry pattern the LLM
  client factory uses, so test harnesses can inject sinks.

The Langfuse backend is lazy-imported only when
``config.telemetry.backend == "langfuse"`` and ``enabled=True``
so a vanilla CARE install doesn't pay for the optional dep.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from care.config import CareConfig, TelemetryConfig

TelemetryBackendFactory = Callable[[TelemetryConfig], "TelemetrySink"]
"""Callable that takes a :class:`TelemetryConfig` and returns a
:class:`TelemetrySink` — same shape custom LLM provider
factories use."""


class TelemetrySinkError(RuntimeError):
    """Raised when the telemetry factory can't construct a sink —
    unknown backend, missing optional dependency, malformed config."""


# ---------------------------------------------------------------------------
# Event value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelemetryEvent:
    """One telemetry record.

    Frozen so callers can pass events to async sinks / loggers
    without worrying about mutation. ``trace_id`` is auto-stamped
    when omitted so every event has a stable identity; ``kind``
    is a free-form string (no ``Literal``) so new event kinds
    don't force a schema bump.

    Fields:
        kind: Free-form category. CARE uses ``mage.generation``,
            ``carl.chain_run``, ``platform.evolution``, plus
            sub-events like ``mage.stage_started``.
        attributes: Arbitrary payload — chain id, query, stage
            name, duration, cost, etc. Each backend serialises
            this dict however it shapes its own protocol.
        trace_id: Stable identifier linking related events
            (e.g. all events for one chain run). Auto-stamped
            with a fresh ``uuid4().hex`` when omitted.
        timestamp: Wall-clock seconds. Default factory yields
            the current time; callers stamping past events
            (e.g. replaying logs) supply it explicitly.
    """

    kind: str
    attributes: dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Sink protocol + null implementation
# ---------------------------------------------------------------------------


class TelemetrySink(Protocol):
    """Minimal contract every backend implements.

    All three methods are best-effort: telemetry failures must
    never break the surrounding chain run. Backends should catch
    their own exceptions internally; the protocol is documented
    not to raise from any method.
    """

    def record(self, event: TelemetryEvent) -> None:
        """Forward one event to the backend. Hot path — must
        return quickly (backends batch / fire-and-forget under
        the hood)."""

    def flush(self) -> None:
        """Force any buffered events to ship. CARE's TUI calls
        this on background-task transition events so users get
        live dashboard updates."""

    def close(self) -> None:
        """Release backend resources. Called on app shutdown."""


class NullTelemetrySink:
    """The default no-op sink.

    Implements :class:`TelemetrySink` so every caller can
    unconditionally call ``sink.record(event)`` without
    branching on ``config.enabled``. All methods return
    immediately; the sink also tracks how many events it
    "received" so tests can assert "nothing was recorded".
    """

    def __init__(self) -> None:
        self._dropped: int = 0

    def record(self, event: TelemetryEvent) -> None:
        self._dropped += 1

    def flush(self) -> None:
        return

    def close(self) -> None:
        return

    @property
    def dropped(self) -> int:
        """Number of events dropped since construction. Useful
        only in tests — production code shouldn't introspect."""
        return self._dropped


# ---------------------------------------------------------------------------
# Factory + registry
# ---------------------------------------------------------------------------


def build_telemetry_sink(config: CareConfig | TelemetryConfig) -> TelemetrySink:
    """Construct the sink described by ``config``.

    When ``config.telemetry.enabled`` is ``False`` (the default),
    returns a fresh :class:`NullTelemetrySink` regardless of
    which backend is configured. This is the zero-overhead
    branch the production majority takes — the backend factory
    is only consulted when the user explicitly opted in.

    Args:
        config: Either a full :class:`CareConfig` or just its
            ``telemetry`` section. CARE's TUI passes the full
            config; library callers grabbing this from
            :mod:`care.runtime` typically hand-roll a
            :class:`TelemetryConfig`.

    Raises:
        TelemetrySinkError: When ``enabled=True`` and the
            requested backend is unknown, missing its optional
            dependency, or refuses to construct.
    """
    cfg = config.telemetry if isinstance(config, CareConfig) else config
    if not cfg.enabled:
        return NullTelemetrySink()

    backend = (cfg.backend or "null").lower().strip()
    if backend in ("", "null"):
        # Explicit `enabled=True, backend="null"` — caller may want
        # a sink that counts dropped events for debugging.
        return NullTelemetrySink()

    if backend in _REGISTERED_BACKENDS:
        try:
            return _REGISTERED_BACKENDS[backend](cfg)
        except TelemetrySinkError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TelemetrySinkError(
                f"custom telemetry backend {backend!r} raised: {exc}"
            ) from exc

    if backend == "langfuse":
        return _build_langfuse(cfg)

    raise TelemetrySinkError(
        f"unknown telemetry backend {backend!r}; supported: "
        f"{', '.join(sorted(list_known_backends()))}. Register a custom "
        "backend via care.runtime.register_telemetry_backend()."
    )


def register_telemetry_backend(
    name: str,
    factory: TelemetryBackendFactory,
) -> None:
    """Plug in a custom telemetry backend.

    Lets users wire CARE up to Honeycomb, Phoenix, or an
    in-house tracing system. The factory is called with the
    full :class:`TelemetryConfig` and should return whatever
    object satisfies :class:`TelemetrySink`.

    Re-registering the same name overwrites silently — matches
    the convention :func:`care.runtime.register_provider_factory`
    uses.
    """
    if not name:
        raise TelemetrySinkError("telemetry backend name must be non-empty")
    _REGISTERED_BACKENDS[name.lower().strip()] = factory


def unregister_telemetry_backend(name: str) -> bool:
    """Drop a custom backend. Returns whether anything was removed.
    Mostly for tests."""
    return _REGISTERED_BACKENDS.pop(name.lower().strip(), None) is not None


def list_known_backends() -> list[str]:
    """Sorted list of backend names CARE can build sinks for —
    ``"null"`` + ``"langfuse"`` plus any registered custom backends."""
    return sorted({"null", "langfuse", *_REGISTERED_BACKENDS})


# ---------------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------------


def _build_langfuse(cfg: TelemetryConfig) -> TelemetrySink:
    """Lazy-imported Langfuse backend."""
    try:
        from langfuse import Langfuse
    except ImportError as exc:
        raise TelemetrySinkError(
            "langfuse SDK is not installed; install with "
            "`pip install langfuse` to use the 'langfuse' telemetry backend"
        ) from exc

    if not cfg.public_key or not cfg.secret_key:
        raise TelemetrySinkError(
            "TelemetryConfig.public_key and secret_key must be set for "
            "the 'langfuse' backend"
        )

    try:
        client = Langfuse(
            public_key=cfg.public_key,
            secret_key=cfg.secret_key,
            host=cfg.host,
        )
    except Exception as exc:  # noqa: BLE001
        raise TelemetrySinkError(
            f"failed to construct Langfuse client: {exc}"
        ) from exc
    return _LangfuseTelemetrySink(client)


class _LangfuseTelemetrySink:
    """Thin wrapper turning :class:`TelemetryEvent` instances into
    Langfuse ``event`` calls.

    Hidden from ``__all__`` because callers shouldn't construct
    these directly — use :func:`build_telemetry_sink` so the
    factory registry stays the single source of truth.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def record(self, event: TelemetryEvent) -> None:
        # Langfuse swallows its own auth errors; we still
        # belt-and-braces wrap so a telemetry hiccup never
        # breaks the surrounding chain run.
        try:
            self._client.event(
                name=event.kind,
                trace_id=event.trace_id,
                start_time=event.timestamp,
                metadata=dict(event.attributes),
            )
        except Exception:  # noqa: BLE001
            return

    def flush(self) -> None:
        flush = getattr(self._client, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:  # noqa: BLE001
                return

    def close(self) -> None:
        shutdown = getattr(self._client, "shutdown", None) or getattr(
            self._client, "close", None
        )
        if callable(shutdown):
            try:
                shutdown()
            except Exception:  # noqa: BLE001
                return


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_REGISTERED_BACKENDS: dict[str, TelemetryBackendFactory] = {}


__all__ = [
    "NullTelemetrySink",
    "TelemetryBackendFactory",
    "TelemetryEvent",
    "TelemetrySink",
    "TelemetrySinkError",
    "build_telemetry_sink",
    "list_known_backends",
    "register_telemetry_backend",
    "unregister_telemetry_backend",
]
