"""Library hot-reload watcher (TODO §3 P1).

Wraps the SDK's shipped `GigaEvoClient.watch_entities` (PREPARE.md
§2.8) so CARE's LibraryScreen — and any other surface that wants
live updates — gets typed events instead of raw dicts. The Memory
server publishes events through ``/v1/events`` (Memory §1.8) for
every entity mutation:

- ``created`` / ``updated`` / ``deleted``
- ``favourite_toggled`` / ``run_recorded`` / ``metadata_updated``
- ``pinned`` / ``promoted``

The LibraryScreen's hot-reload semantics are simple: on **any**
event in the user's namespace, refresh the affected row (or remove
it on ``deleted``). This module provides:

* :class:`LibraryEvent` — frozen typed view of the SDK's raw event
  dict. Stable field set so screens / tests don't drift over
  upstream key naming.
* :class:`LibrarySubscription` — opaque wrapper around the SDK
  ``Subscription`` with ``.stop()`` and context-manager support.
* :meth:`care.CareMemory.watch_library` — facade-side entrypoint
  (added separately to keep this module tiny).

The watcher imports the SDK lazily so test code can swap it out
via the ``_watch_via=`` injection seam, mirroring the pattern used
in the executor and skill-runtime adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

EVENT_KINDS = (
    "created",
    "updated",
    "deleted",
    "favourite_toggled",
    "run_recorded",
    "metadata_updated",
    "pinned",
    "promoted",
)
"""The canonical event-type set Memory publishes on ``/v1/events``.

Pinned here so test fixtures + dispatch tables on the screen side
have a single source of truth. The watcher does NOT enforce this
list — it passes whatever the server sent through verbatim — but
the docstring on :class:`LibraryEvent.event_type` recommends
filtering on this set."""


@dataclass(frozen=True)
class LibraryEvent:
    """One mutation event the LibraryScreen renders.

    Mirrors the shape Memory publishes on its SSE channel (per the
    Memory §1.8 spec): every event carries the entity it touched,
    the kind of mutation, and a timestamp. The screen dispatches
    on ``event_type`` to decide whether to insert / update / remove
    the row.

    Fields:
        event_type: One of :data:`EVENT_KINDS`. Unknown values
            still pass through so the LibraryScreen sees them and
            can log "ignoring unknown event_type" — better than
            silently dropping new server-side events.
        entity_id: The entity that changed.
        entity_type: ``"chain" | "agent" | "agent_skill" |
            "memory_card" | "step"``.
        version_id: Set on every event that pinned a specific
            version (``updated``, ``created``, ``pinned``,
            ``promoted``). ``None`` for ``deleted`` /
            ``favourite_toggled`` / ``run_recorded`` /
            ``metadata_updated`` (those operate at the entity
            level).
        channel: ``"latest" | "stable" | "draft" | ...`` — only
            populated on version-pinned events.
        namespace: The CARE namespace the entity lives in.
        tags: Tag set at event time (snapshot — may differ from
            current entity state).
        timestamp: Server-side timestamp the event was published.
        raw: The full event dict the SDK forwarded, kept so screens
            can read fields this view hasn't typed yet (forward-
            compat).
    """

    event_type: str
    entity_id: str
    entity_type: str
    version_id: str | None = None
    channel: str | None = None
    namespace: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    timestamp: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "LibraryEvent":
        """Promote a raw SDK event dict to a typed :class:`LibraryEvent`.

        Missing required fields fall back to empty strings (the
        screen handles "blank entity_id" as ignore-and-log; we
        don't raise on the watcher path because dropping events
        is worse than passing through suspicious ones).
        """
        ts_raw = raw.get("timestamp")
        ts: datetime | None
        if isinstance(ts_raw, datetime):
            ts = ts_raw
        elif isinstance(ts_raw, str) and ts_raw:
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                ts = None
        else:
            ts = None

        tags_raw = raw.get("tags") or ()
        tags: tuple[str, ...]
        if isinstance(tags_raw, (list, tuple, set)):
            tags = tuple(str(t) for t in tags_raw)
        else:
            tags = ()

        return cls(
            event_type=str(raw.get("event_type") or ""),
            entity_id=str(raw.get("entity_id") or ""),
            entity_type=str(raw.get("entity_type") or ""),
            version_id=_optional_str(raw.get("version_id")),
            channel=_optional_str(raw.get("channel")),
            namespace=_optional_str(raw.get("namespace")),
            tags=tags,
            timestamp=ts,
            raw=dict(raw),
        )

    @property
    def is_known_kind(self) -> bool:
        """``True`` when ``event_type`` is in the documented set.

        Screens can log a warning when an unknown kind shows up
        (likely a new server-side event added without CARE bumping)."""
        return self.event_type in EVENT_KINDS

    @property
    def is_terminal(self) -> bool:
        """``True`` for events that remove the entity — the
        LibraryScreen should drop the row rather than refresh it."""
        return self.event_type == "deleted"


class LibrarySubscription:
    """Opaque handle for a running library subscription.

    Wraps the SDK's `Subscription` so screens don't import SDK
    internals. Supports context-manager use::

        with memory.watch_library(on_event, namespace="glazkov") as sub:
            await long_running_screen()
        # sub.stop() called automatically on __exit__.
    """

    def __init__(self, sdk_subscription: Any) -> None:
        self._sdk = sdk_subscription

    def stop(self) -> None:
        """Cancel the subscription. Idempotent: calling twice is
        a no-op on the second call (the SDK's `Subscription.stop`
        is itself idempotent)."""
        stopper = getattr(self._sdk, "stop", None)
        if callable(stopper):
            stopper()

    @property
    def underlying(self) -> Any:
        """Escape hatch for callers that need the raw SDK
        subscription (e.g. introspecting filter state in tests)."""
        return self._sdk

    def __enter__(self) -> "LibrarySubscription":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def watch_library(
    client: Any,
    callback: Callable[[LibraryEvent], None],
    *,
    namespace: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    tags: list[str] | None = None,
    event_type: str | None = None,
) -> LibrarySubscription:
    """Subscribe to library mutations via the SDK's
    ``watch_entities`` and wrap the result.

    ``callback`` is called with a typed :class:`LibraryEvent` per
    server-side event. The conversion is done in the wrapper
    closure so the SDK never sees CARE types.

    Args:
        client: Anything exposing ``watch_entities`` — typically
            :class:`gigaevo_client.GigaEvoClient` (CARE's
            :class:`CareMemory.client`).
        callback: Receives one :class:`LibraryEvent` per event.
        namespace, entity_type, entity_id, tags, event_type:
            Forwarded as filters to the SDK. ``namespace`` is the
            usual library hot-reload pivot.

    Returns:
        :class:`LibrarySubscription`. Call ``.stop()`` (or use as
        a context manager) when the screen is done.
    """

    def _adapter(raw: dict[str, Any]) -> None:
        try:
            event = LibraryEvent.from_raw(raw or {})
        except Exception:  # noqa: BLE001
            # Defensive: a single malformed event shouldn't kill
            # the whole subscription. Drop it silently — the SDK's
            # next event will arrive normally.
            return
        callback(event)

    sdk_sub = client.watch_entities(
        _adapter,
        entity_type=entity_type,
        entity_id=entity_id,
        namespace=namespace,
        tags=tags,
        event_type=event_type,
    )
    return LibrarySubscription(sdk_sub)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _optional_str(value: Any) -> str | None:
    """``None`` / empty-string → ``None``; otherwise ``str(value)``."""
    if value is None:
        return None
    out = str(value)
    return out if out else None


__all__ = [
    "EVENT_KINDS",
    "LibraryEvent",
    "LibrarySubscription",
    "watch_library",
]
