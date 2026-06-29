"""Tests for ``care.runtime.cancellation`` (TODO §1.2 P1).

Pure-Python primitive — no upstream deps. Coverage layers:

1. ``CancellationToken`` polling surface: `is_cancelled`,
   `cancel` (idempotent + returns bool), `raise_if_cancelled` +
   async variant.
2. Async surface: `wait_cancelled` returns immediately when
   already cancelled, blocks until cancel fires when pending.
3. Callbacks: `on_cancel` fires once per token, in registration
   order; late registration after cancel fires immediately;
   listener exceptions are swallowed.
4. ``CancelledByUserError`` is a subclass of
   ``asyncio.CancelledError`` and carries the `reason` field.
5. ``CancellationGroup``: child created after parent already
   cancelled is itself cancelled; cancel-parent propagates to
   children; cancel-one-child does NOT propagate up.
6. ``join_tokens`` fan-in: derived token fires on first input
   cancel; cancelling derived doesn't propagate back to inputs;
   empty input raises ValueError.
7. Concurrency: simultaneous cancels from multiple threads still
   produce exactly one transition.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from care.runtime import (
    CancellationGroup,
    CancellationToken,
    CancelledByUserError,
    join_tokens,
)


# ---------------------------------------------------------------------------
# Basic token surface
# ---------------------------------------------------------------------------


class TestBasicToken:
    def test_new_token_not_cancelled(self):
        tok = CancellationToken()
        assert tok.is_cancelled is False
        assert tok.cancelled_at is None

    def test_default_reason(self):
        tok = CancellationToken()
        assert tok.reason == "cancelled by user"

    def test_custom_reason(self):
        tok = CancellationToken(reason="deadline exceeded")
        assert tok.reason == "deadline exceeded"

    def test_cancel_flips_state(self):
        tok = CancellationToken()
        assert tok.cancel() is True
        assert tok.is_cancelled is True
        assert tok.cancelled_at is not None

    def test_cancel_idempotent(self):
        tok = CancellationToken()
        first = tok.cancel()
        second = tok.cancel()
        assert first is True
        assert second is False  # already cancelled
        # cancelled_at doesn't move on the second call.
        first_ts = tok.cancelled_at
        time.sleep(0.001)
        tok.cancel()
        assert tok.cancelled_at == first_ts


class TestRaiseIfCancelled:
    def test_no_op_when_pending(self):
        tok = CancellationToken()
        tok.raise_if_cancelled()  # no exception

    def test_raises_when_cancelled(self):
        tok = CancellationToken(reason="bye")
        tok.cancel()
        with pytest.raises(CancelledByUserError, match="bye"):
            tok.raise_if_cancelled()

    @pytest.mark.asyncio
    async def test_async_variant_raises(self):
        tok = CancellationToken()
        tok.cancel()
        with pytest.raises(CancelledByUserError):
            await tok.raise_if_cancelled_async()

    @pytest.mark.asyncio
    async def test_async_variant_no_op_when_pending(self):
        tok = CancellationToken()
        await tok.raise_if_cancelled_async()


# ---------------------------------------------------------------------------
# Exception class
# ---------------------------------------------------------------------------


class TestCancelledByUserError:
    def test_is_subclass_of_asyncio_cancelled_error(self):
        """Distinct enough for screens to identify user-cancel, but
        compatible with existing `except asyncio.CancelledError`
        blocks so async code doesn't accidentally swallow it as
        a programming error."""
        assert issubclass(CancelledByUserError, asyncio.CancelledError)

    def test_carries_reason(self):
        exc = CancelledByUserError("deadline")
        assert exc.reason == "deadline"

    def test_default_reason(self):
        exc = CancelledByUserError()
        assert exc.reason == "cancelled by user"


# ---------------------------------------------------------------------------
# Async wait_cancelled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_cancelled_returns_immediately_if_already_cancelled():
    tok = CancellationToken()
    tok.cancel()
    await asyncio.wait_for(tok.wait_cancelled(), timeout=0.5)


@pytest.mark.asyncio
async def test_wait_cancelled_blocks_until_cancel():
    tok = CancellationToken()
    waiter = asyncio.create_task(tok.wait_cancelled())
    await asyncio.sleep(0.05)
    assert not waiter.done()
    tok.cancel()
    await asyncio.wait_for(waiter, timeout=0.5)


@pytest.mark.asyncio
async def test_wait_cancelled_supports_multiple_waiters():
    tok = CancellationToken()
    a = asyncio.create_task(tok.wait_cancelled())
    b = asyncio.create_task(tok.wait_cancelled())
    tok.cancel()
    await asyncio.gather(a, b)


# ---------------------------------------------------------------------------
# on_cancel callbacks
# ---------------------------------------------------------------------------


class TestOnCancel:
    def test_fires_in_registration_order(self):
        tok = CancellationToken()
        seen: list[str] = []
        tok.on_cancel(lambda t: seen.append("a"))
        tok.on_cancel(lambda t: seen.append("b"))
        tok.on_cancel(lambda t: seen.append("c"))
        tok.cancel()
        assert seen == ["a", "b", "c"]

    def test_late_registration_fires_immediately(self):
        tok = CancellationToken()
        tok.cancel()
        seen: list[CancellationToken] = []
        tok.on_cancel(seen.append)
        assert seen == [tok]

    def test_listener_exception_does_not_block_others(self):
        tok = CancellationToken()
        seen: list[str] = []

        def boom(t):
            raise RuntimeError("bad listener")

        tok.on_cancel(boom)
        tok.on_cancel(lambda t: seen.append("after"))
        tok.cancel()  # no exception escapes
        assert seen == ["after"]

    def test_callbacks_fire_only_once(self):
        """Second cancel() is a no-op — callbacks already ran."""
        tok = CancellationToken()
        counter: list[int] = []
        tok.on_cancel(lambda t: counter.append(1))
        tok.cancel()
        tok.cancel()  # idempotent
        assert sum(counter) == 1


# ---------------------------------------------------------------------------
# CancellationGroup
# ---------------------------------------------------------------------------


class TestCancellationGroup:
    def test_new_group_not_cancelled(self):
        g = CancellationGroup()
        assert g.is_cancelled is False

    def test_cancel_all_cancels_root(self):
        g = CancellationGroup()
        g.cancel_all()
        assert g.is_cancelled is True
        assert g.root.is_cancelled is True

    def test_parent_cancel_propagates_to_children(self):
        g = CancellationGroup()
        a = g.token()
        b = g.token()
        assert not (a.is_cancelled or b.is_cancelled)
        g.cancel_all()
        assert a.is_cancelled and b.is_cancelled

    def test_child_created_after_parent_cancel_is_cancelled(self):
        g = CancellationGroup()
        g.cancel_all()
        c = g.token()
        # Parent already cancelled → child cancelled immediately
        # via the synchronous on_cancel call inside `token`.
        assert c.is_cancelled

    def test_one_child_cancel_does_not_propagate_up(self):
        g = CancellationGroup()
        a = g.token()
        b = g.token()
        a.cancel()
        assert a.is_cancelled
        assert not g.is_cancelled
        assert not b.is_cancelled

    def test_children_snapshot_is_copy(self):
        g = CancellationGroup()
        g.token()
        kids = g.children
        assert len(kids) == 1
        # Mutating the snapshot doesn't affect the group.
        kids = (*kids, "ghost")  # type: ignore[assignment]
        assert len(g.children) == 1

    def test_custom_child_reason_overrides(self):
        g = CancellationGroup(reason="user esc")
        c = g.token(reason="preflight failed")
        assert g.root.reason == "user esc"
        assert c.reason == "preflight failed"


# ---------------------------------------------------------------------------
# join_tokens fan-in
# ---------------------------------------------------------------------------


class TestJoinTokens:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            join_tokens()

    def test_single_token(self):
        a = CancellationToken()
        joined = join_tokens(a)
        assert joined.is_cancelled is False
        a.cancel()
        assert joined.is_cancelled is True

    def test_fires_on_first_input_cancel(self):
        a = CancellationToken()
        b = CancellationToken()
        c = CancellationToken()
        joined = join_tokens(a, b, c)
        b.cancel()
        assert joined.is_cancelled
        # Other inputs stay untouched.
        assert not a.is_cancelled
        assert not c.is_cancelled

    def test_already_cancelled_input_fires_immediately(self):
        a = CancellationToken()
        b = CancellationToken()
        b.cancel()
        joined = join_tokens(a, b)
        assert joined.is_cancelled

    def test_cancel_derived_does_not_propagate_back(self):
        a = CancellationToken()
        joined = join_tokens(a)
        joined.cancel()
        # Cancelling the join does NOT cancel its inputs.
        assert not a.is_cancelled


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_cancels_produce_exactly_one_transition():
    """Race multiple threads to call cancel() — only one wins
    (returns True); the rest return False. Callback fires once."""
    tok = CancellationToken()
    transitions: list[bool] = []
    callback_count: list[int] = []
    tok.on_cancel(lambda t: callback_count.append(1))

    barrier = threading.Barrier(8)

    def race():
        barrier.wait()  # synchronise start
        transitions.append(tok.cancel())

    threads = [threading.Thread(target=race) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one cancel returned True, the rest returned False.
    assert transitions.count(True) == 1
    assert transitions.count(False) == 7
    # Callback fired exactly once.
    assert sum(callback_count) == 1
