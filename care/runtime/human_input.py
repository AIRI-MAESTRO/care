"""Human-in-the-loop request broker (TODO §5 P1).

When a CARL chain hits a ``HumanInputStep``, the
:meth:`ReasoningContext.on_human_input_requested(prompt,
future)` callback fires and the chain blocks on ``future`` until
someone sets a result via ``context.provide_human_input(value)``.

CARE's existing :class:`care.runtime.CarlStreamer` adapter turns
each callback into a :class:`HumanInputRequested` Textual
``Message`` (already shipped) — the future-modal screen pops a
dialog and resolves the future when the user submits. But two
gaps remain:

1. **Headless consumers.** The Textual ``Message`` route only
   works inside a running app. CLI subcommands like a future
   ``care run --interactive`` need to render the prompt in the
   terminal + collect input there. Same for replay /
   scripting use.
2. **Decoupled resolution.** Tests, MCP integrations, and
   automation can't post a Textual message — they need a
   plain Python handle.

This module ships the bridge: :class:`HumanInputBroker` is a
thread-safe queue + resolver that any consumer can plug into.
The CARE Textual app subscribes a listener that turns each new
request into a Textual ``Message``; the CLI subscribes a
listener that prompts on stdin; tests subscribe a listener that
resolves immediately. Same broker, three consumers.

Decoupled from CARL: the broker doesn't import CARL. The
``future`` argument is duck-typed against
``set_result`` / ``set_exception`` (matches ``asyncio.Future``,
``concurrent.futures.Future``, and any custom completion-style
object). When CARL hands the broker a future, the broker
fulfils it on the consumer's behalf.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class HumanInputRequest:
    """One pending question for the user.

    Frozen so listeners can hold the snapshot safely. The
    underlying future stays on the broker's private side; the
    UI / CLI only sees the request's ``id`` + display fields.

    Fields:
        id: Stable identifier — what the consumer passes back
            to :meth:`HumanInputBroker.resolve`.
        prompt: The question text MAGE's HumanInputStep
            supplied. Render this in whatever surface the
            consumer prefers.
        default: Suggested value to pre-fill the input box
            with (when the screen / CLI supports it). Empty
            string when CARL didn't supply one.
        options: When the question has a discrete answer set
            (e.g. "yes" / "no" / "skip"), the broker
            forwards it here so the consumer can render a
            picker. Empty tuple = free-form text input.
        created_at: Wall-clock seconds — useful for the UI
            to show "asked 4s ago" indicators.
        metadata: Free-form per-request extras (step_number,
            stage, etc.) so consumers don't have to thread
            them separately.
    """

    id: str
    prompt: str
    default: str = ""
    options: tuple[str, ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PendingEntry:
    """Internal record holding the future alongside the request."""

    request: HumanInputRequest
    future: Any
    """Anything with ``set_result(value)`` /
    ``set_exception(exc)``. The broker doesn't constrain the
    type."""


class HumanInputCancelled(RuntimeError):
    """Raised on the future when :meth:`HumanInputBroker.cancel`
    completes it — distinguishes user cancellation from any
    other exception path."""


class HumanInputBroker:
    """Thread-safe broker between CARL's HumanInput requests +
    any consumer (UI / CLI / scripting / tests).

    Lifecycle::

        broker = HumanInputBroker()
        broker.on_request(lambda req: pop_modal(req))

        # CARL side (wired in `attach_to_context` below):
        broker.submit(prompt, future=carl_future)

        # Consumer side (UI button click handler / CLI input):
        broker.resolve(request_id, value)
        # → future.set_result(value); request removed from pending.

    Thread-safe — all mutators take a lock so the Textual
    thread + worker threads can both touch the broker without
    races.
    """

    def __init__(self) -> None:
        self._pending: dict[str, _PendingEntry] = {}
        self._listeners: list[Callable[[HumanInputRequest], None]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # CARL → broker
    # ------------------------------------------------------------------

    def submit(
        self,
        prompt: str,
        *,
        future: Any,
        request_id: str | None = None,
        default: str = "",
        options: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HumanInputRequest:
        """Queue a new request and notify listeners.

        Args:
            prompt: The HumanInputStep's question text.
            future: Anything with ``set_result(value)`` /
                ``set_exception(exc)``. CARL passes its own
                future here; tests pass a
                :class:`concurrent.futures.Future`.
            request_id: Optional pre-allocated id. Most
                callers omit this — the broker generates a
                fresh ``uuid4().hex``.
            default: Suggested input the screen pre-fills
                with. Defaults to empty string.
            options: Discrete answer set when applicable.
                Empty / ``None`` means free-form text.
            metadata: Free-form per-request extras (step
                number, stage, etc.).

        Returns:
            The :class:`HumanInputRequest` that was queued.
            Useful so the caller can correlate later.
        """
        req_id = request_id or uuid.uuid4().hex
        request = HumanInputRequest(
            id=req_id,
            prompt=prompt,
            default=default,
            options=tuple(options or ()),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._pending[req_id] = _PendingEntry(request=request, future=future)
            listeners = list(self._listeners)
        # Fire listeners OUTSIDE the lock so a slow listener
        # doesn't block other submitters.
        for listener in listeners:
            try:
                listener(request)
            except Exception:  # noqa: BLE001
                # Telemetry sinks / log listeners are
                # best-effort — a misbehaving subscriber must
                # not kill the broker.
                continue
        return request

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def pending(self) -> tuple[HumanInputRequest, ...]:
        """Snapshot of unresolved requests in submission
        order."""
        with self._lock:
            return tuple(entry.request for entry in self._pending.values())

    def get(self, request_id: str) -> HumanInputRequest | None:
        """Look up a pending request by id. Returns ``None``
        when the request was already resolved or never
        existed."""
        with self._lock:
            entry = self._pending.get(request_id)
            return entry.request if entry is not None else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)

    def __contains__(self, request_id: object) -> bool:
        if not isinstance(request_id, str):
            return False
        with self._lock:
            return request_id in self._pending

    # ------------------------------------------------------------------
    # Consumer → broker
    # ------------------------------------------------------------------

    def resolve(self, request_id: str, value: str) -> bool:
        """Provide the user's input. Returns ``True`` when the
        request was pending (and got resolved), ``False`` when
        the id was unknown or already resolved.

        ``value`` lands on the future via ``set_result(value)``.
        Future objects that have already been completed (e.g.
        a stale CARL cancel got there first) are tolerated —
        the broker logs nothing and returns ``False``.
        """
        with self._lock:
            entry = self._pending.pop(request_id, None)
        if entry is None:
            return False
        try:
            entry.future.set_result(value)
        except Exception:  # noqa: BLE001
            # Already done / cancelled — fine, the awaiter has
            # moved on.
            return False
        return True

    def cancel(
        self,
        request_id: str,
        reason: str = "user cancelled",
    ) -> bool:
        """Reject the request — the future completes with
        :class:`HumanInputCancelled`. Returns ``True`` when a
        pending request was found."""
        with self._lock:
            entry = self._pending.pop(request_id, None)
        if entry is None:
            return False
        try:
            entry.future.set_exception(HumanInputCancelled(reason))
        except Exception:  # noqa: BLE001
            return False
        return True

    def cancel_all(self, reason: str = "broker shutdown") -> int:
        """Cancel every pending request. Returns the number
        cancelled — useful when shutting CARE down so chains
        blocked on input don't hang."""
        with self._lock:
            entries = list(self._pending.values())
            self._pending.clear()
        count = 0
        for entry in entries:
            try:
                entry.future.set_exception(HumanInputCancelled(reason))
                count += 1
            except Exception:  # noqa: BLE001
                continue
        return count

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def on_request(
        self,
        listener: Callable[[HumanInputRequest], None],
    ) -> Callable[[], None]:
        """Subscribe to new-request events.

        Returns an unsubscribe callable. The listener is invoked
        synchronously after a successful :meth:`submit` — it
        should kick off whatever asynchronous work the
        consumer needs (post a Textual message, push to
        stdin, schedule a task) and return promptly.

        Listener exceptions are swallowed so a misbehaving
        subscriber can't poison subsequent listeners.
        """
        with self._lock:
            self._listeners.append(listener)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(listener)
                except ValueError:
                    pass

        return _unsubscribe


def attach_to_context(broker: HumanInputBroker, context: Any) -> None:
    """Wire ``broker`` as ``context``'s human-input handler.

    Sets ``context.on_human_input_requested = ...`` to a
    callback that calls :meth:`HumanInputBroker.submit` with
    CARL's future. CARL then blocks on the future until any
    consumer calls :meth:`HumanInputBroker.resolve` /
    :meth:`HumanInputBroker.cancel`.

    Duck-typed against `mmar_carl.ReasoningContext` — anything
    with a writable ``on_human_input_requested`` attribute
    works. Used by :class:`care.runtime.CarlStreamer` when the
    caller wants headless / CLI / scripted resolution instead
    of (or alongside) the modal route.
    """

    def _handler(prompt: str, future: Any) -> None:
        broker.submit(prompt, future=future)

    context.on_human_input_requested = _handler


__all__ = [
    "HumanInputBroker",
    "HumanInputCancelled",
    "HumanInputRequest",
    "attach_to_context",
]
