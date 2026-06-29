"""Cancellation token primitive (TODO §1.2 P1).

CARE's long-running tasks (MAGE generation, CARL execution,
Platform evolution polling) need to respect a user cancel signal —
typically `Esc` on the screen that owns the task. The token lives
here so every layer (Textual screens, the executor, the
`MagePoster` adapter, future task-registry entries) shares the
same primitive and the same exception type.

Design:

- :class:`CancellationToken` — a tiny mutable carrier with
  `is_cancelled` + `cancel()` + the standard `raise_if_cancelled()`
  poll helper. Internal flag is set atomically (Python's GIL
  guarantees this for single-attribute writes); for async paths
  the token also fires an :class:`asyncio.Event` so consumers can
  `await wait_cancelled()` instead of polling.
- :class:`CancellationGroup` — a parent/child relationship so a
  screen can cancel everything it owns with one call. Children
  inherit cancellation from their parent immediately on creation;
  later cancels propagate downward through the tree.
- :class:`CancelledByUserError` — distinct exception so screens
  can show "Cancelled" instead of "Failed: <traceback>". Subclass
  of :class:`asyncio.CancelledError` so `try/except` blocks that
  already handle the asyncio version catch it for free.

The module is pure-Python with only standard-library imports —
no Textual / no CARL coupling. Adapter code (e.g. the executor's
`_check_cancel`) takes a token and polls; the screen owns the
token + maps `Esc` → `token.cancel()`.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable


class CancelledByUserError(asyncio.CancelledError):
    """Raised by :meth:`CancellationToken.raise_if_cancelled` /
    `await token.raise_if_cancelled_async()` when the token has
    been flipped.

    Subclasses :class:`asyncio.CancelledError` so async code that
    already handles asyncio cancellation catches it transparently
    — but it's a distinct class so screens can detect
    "user-initiated" vs "task-died" and show a friendlier message.

    The exception carries an optional `reason` string that the UI
    can render ("Cancelled by user", "Cancelled because parent
    task finished", etc.).
    """

    def __init__(self, reason: str = "cancelled by user") -> None:
        super().__init__(reason)
        self.reason = reason


class CancellationToken:
    """Async-safe cancel signal.

    Lifecycle: created un-cancelled; flipped exactly once via
    :meth:`cancel`. Subsequent ``cancel()`` calls are no-ops
    (idempotent). Once cancelled, the token stays cancelled —
    there's no `uncancel`. Make a fresh token when the screen
    starts a new task.

    Two consumption patterns:

    * **Polling** — `if token.is_cancelled: ...` or
      `token.raise_if_cancelled()` between work items.
    * **Awaiting** — `await token.wait_cancelled()` blocks until
      cancel fires.

    The token is safe to share across threads; a `threading.Lock`
    guards the atomic flip + the asyncio.Event notification.
    """

    def __init__(self, *, reason: str = "cancelled by user") -> None:
        self._reason = reason
        self._cancelled = False
        self._cancelled_at: float | None = None
        self._lock = threading.Lock()
        # Event lazily created on first `wait_cancelled` call so we
        # don't need a running event loop at construction time.
        self._event: asyncio.Event | None = None
        self._callbacks: list[Callable[["CancellationToken"], None]] = []

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def is_cancelled(self) -> bool:
        """Has :meth:`cancel` been called?"""
        return self._cancelled

    @property
    def reason(self) -> str:
        """Human-readable string the UI shows when the token fires.
        Set at construction; carry-on-cancel pattern."""
        return self._reason

    @property
    def cancelled_at(self) -> float | None:
        """``time.monotonic()`` timestamp of the cancel, or
        ``None`` when still pending. Useful for the audit log."""
        return self._cancelled_at

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def cancel(self) -> bool:
        """Flip the token. Returns whether anything actually
        changed (``False`` on idempotent re-cancel).

        Notifies every callback registered via :meth:`on_cancel`
        and wakes any tasks awaiting :meth:`wait_cancelled`.
        Callback exceptions are swallowed — one misbehaving
        listener can't block the others.
        """
        with self._lock:
            if self._cancelled:
                return False
            self._cancelled = True
            self._cancelled_at = time.monotonic()
            event = self._event
            callbacks = list(self._callbacks)

        # Notify outside the lock to avoid awaiting a coroutine
        # while holding it.
        if event is not None:
            event.set()
        for cb in callbacks:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                # A bad listener must not break the cancel path.
                pass
        return True

    # ------------------------------------------------------------------
    # Polling helpers
    # ------------------------------------------------------------------

    def raise_if_cancelled(self) -> None:
        """Raise :class:`CancelledByUserError` when cancelled, no-op
        otherwise. Sync-friendly: call between work items inside a
        synchronous loop."""
        if self._cancelled:
            raise CancelledByUserError(self._reason)

    async def raise_if_cancelled_async(self) -> None:
        """Async-friendly alias of :meth:`raise_if_cancelled`.

        Exists for consistency with the rest of the async surface —
        callers can `await token.raise_if_cancelled_async()` inside
        an `async def` without remembering which version is sync.
        """
        if self._cancelled:
            raise CancelledByUserError(self._reason)

    # ------------------------------------------------------------------
    # Async-await surface
    # ------------------------------------------------------------------

    async def wait_cancelled(self) -> None:
        """Block until the token is cancelled. Returns immediately
        if already cancelled.

        Lazily binds the underlying :class:`asyncio.Event` to the
        running loop on first call — works when the token is
        constructed before the loop spins up.
        """
        event = self._ensure_event()
        await event.wait()

    def on_cancel(
        self, callback: Callable[["CancellationToken"], None]
    ) -> None:
        """Register a callable that fires synchronously when the
        token flips. Registration after cancel runs the callback
        immediately. Multiple callbacks are supported; order is
        registration-order.

        Callback exceptions are swallowed. Use this for cleanup
        hooks that must run regardless of which path triggered
        the cancel (background task, modal Esc handler, parent
        token in a group)."""
        with self._lock:
            if not self._cancelled:
                self._callbacks.append(callback)
                return
        # Already cancelled — fire immediately.
        try:
            callback(self)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_event(self) -> asyncio.Event:
        with self._lock:
            if self._event is None:
                self._event = asyncio.Event()
                if self._cancelled:
                    self._event.set()
            return self._event

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        state = "cancelled" if self._cancelled else "pending"
        return f"CancellationToken({state!r}, reason={self._reason!r})"


class CancellationGroup:
    """A tree of tokens that cancel together.

    Use this when a screen owns multiple long-running tasks (MAGE
    + CARL streaming + Platform polling, say) and a single `Esc`
    must stop all of them.

    Children created via :meth:`token` inherit the parent's
    cancelled state at creation time. Cancelling the parent
    propagates to every child; cancelling one child does NOT
    propagate upward (the screen might want to abort one stage
    without killing the others)."""

    def __init__(self, *, reason: str = "cancelled by user") -> None:
        self._root = CancellationToken(reason=reason)
        self._children: list[CancellationToken] = []
        self._lock = threading.Lock()

    @property
    def root(self) -> CancellationToken:
        """The parent token. Cancel it to cancel the whole group."""
        return self._root

    def token(self, *, reason: str | None = None) -> CancellationToken:
        """Create a child token. ``reason`` defaults to the parent's
        reason — override for a more specific message ("MAGE
        cancelled because pre-flight failed", etc.).

        If the parent is already cancelled, the child is returned
        in cancelled state immediately."""
        child = CancellationToken(reason=reason or self._root.reason)

        # Hook the parent → child propagation. `on_cancel` expects
        # a callback that returns ``None``; `child.cancel()` returns
        # ``bool``, so wrap it.
        def _propagate(_token: "CancellationToken", *, c: "CancellationToken" = child) -> None:
            c.cancel()

        self._root.on_cancel(_propagate)
        with self._lock:
            self._children.append(child)
        return child

    def cancel_all(self) -> None:
        """Cancel the parent (and therefore every child)."""
        self._root.cancel()

    @property
    def is_cancelled(self) -> bool:
        return self._root.is_cancelled

    @property
    def children(self) -> tuple[CancellationToken, ...]:
        """Snapshot of currently registered child tokens."""
        with self._lock:
            return tuple(self._children)


def join_tokens(*tokens: CancellationToken) -> CancellationToken:
    """Return a derived token that fires when any of ``tokens``
    fires. Useful when a task should cancel on either its own
    deadline OR a user `Esc`.

    The derived token starts un-cancelled (unless any input is
    already cancelled, in which case it fires immediately). A
    one-way fan-in — cancelling the derived token does NOT
    propagate back to the inputs.
    """
    if not tokens:
        raise ValueError("join_tokens requires at least one token")
    derived = CancellationToken(reason="cancelled by joined token")

    def _fire(_token: "CancellationToken", *, d: "CancellationToken" = derived) -> None:
        d.cancel()

    for src in tokens:
        if src.is_cancelled:
            derived.cancel()
            break
        src.on_cancel(_fire)
    return derived


__all__ = [
    "CancellationGroup",
    "CancellationToken",
    "CancelledByUserError",
    "join_tokens",
]


# `Any` is imported above for forward-compat with future signatures
# (e.g. on_cancel returning the registered handle).
_ = Any
