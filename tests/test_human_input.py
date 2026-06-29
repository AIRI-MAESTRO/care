"""Tests for ``care.runtime.human_input.HumanInputBroker``
(TODO §5 P1).

Six coverage layers:

1. **submit() shape** — id auto-stamped, default fields,
   metadata + options pass-through.
2. **Lookup** — `pending()`, `get()`, `__len__`, `__contains__`.
3. **resolve()** — happy path sets the future result;
   unknown id returns False; already-resolved future returns
   False without raising.
4. **cancel()** + `cancel_all()` — future receives
   `HumanInputCancelled`; unknown id returns False; bulk
   cancel returns count.
5. **Listeners** — fire on submit; unsubscribe returns void;
   exception in listener doesn't break the broker.
6. **`attach_to_context()`** — wires the broker as CARL's
   `on_human_input_requested` callback; subsequent invocation
   queues a request.
7. **Thread-safety smoke** — concurrent submits + resolves
   don't lose entries or crash.
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest

from care.runtime.human_input import (
    HumanInputBroker,
    HumanInputCancelled,
    HumanInputRequest,
    attach_to_context,
)


def _future() -> concurrent.futures.Future:
    """Make a fresh future for each test."""
    return concurrent.futures.Future()


# ---------------------------------------------------------------------------
# submit()
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_submit_returns_request_with_auto_id(self):
        broker = HumanInputBroker()
        fut = _future()
        request = broker.submit("Continue?", future=fut)
        assert isinstance(request, HumanInputRequest)
        assert request.id
        assert len(request.id) == 32  # uuid4().hex
        assert request.prompt == "Continue?"
        # Defaults.
        assert request.default == ""
        assert request.options == ()
        assert request.metadata == {}

    def test_submit_accepts_explicit_id(self):
        broker = HumanInputBroker()
        fut = _future()
        broker.submit("Q", future=fut, request_id="my-id")
        assert "my-id" in broker

    def test_submit_options_pass_through(self):
        broker = HumanInputBroker()
        req = broker.submit(
            "Pick one:",
            future=_future(),
            options=["yes", "no"],
        )
        assert req.options == ("yes", "no")

    def test_submit_metadata_pass_through(self):
        broker = HumanInputBroker()
        req = broker.submit(
            "Q",
            future=_future(),
            metadata={"step_number": 3},
        )
        assert req.metadata == {"step_number": 3}

    def test_submit_default_value(self):
        broker = HumanInputBroker()
        req = broker.submit("Q", future=_future(), default="yes")
        assert req.default == "yes"


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_pending_returns_in_submission_order(self):
        broker = HumanInputBroker()
        a = broker.submit("Q1", future=_future())
        b = broker.submit("Q2", future=_future())
        pending = broker.pending()
        assert [r.id for r in pending] == [a.id, b.id]

    def test_get_returns_request(self):
        broker = HumanInputBroker()
        req = broker.submit("Q", future=_future())
        assert broker.get(req.id) is req

    def test_get_returns_none_for_unknown(self):
        broker = HumanInputBroker()
        assert broker.get("not-a-real-id") is None

    def test_len_tracks_pending(self):
        broker = HumanInputBroker()
        assert len(broker) == 0
        broker.submit("Q", future=_future())
        assert len(broker) == 1
        broker.submit("Q2", future=_future())
        assert len(broker) == 2

    def test_contains(self):
        broker = HumanInputBroker()
        req = broker.submit("Q", future=_future())
        assert req.id in broker
        assert "definitely-not" not in broker
        # Non-string contains returns False.
        assert 42 not in broker


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------


class TestResolve:
    def test_resolve_sets_future_result(self):
        broker = HumanInputBroker()
        fut = _future()
        req = broker.submit("Q", future=fut)
        assert broker.resolve(req.id, "the answer") is True
        assert fut.result() == "the answer"
        # Removed from pending.
        assert req.id not in broker

    def test_resolve_unknown_id_returns_false(self):
        broker = HumanInputBroker()
        assert broker.resolve("no-such-id", "value") is False

    def test_resolve_double_returns_false(self):
        broker = HumanInputBroker()
        fut = _future()
        req = broker.submit("Q", future=fut)
        assert broker.resolve(req.id, "first") is True
        # Second call sees nothing pending.
        assert broker.resolve(req.id, "second") is False
        assert fut.result() == "first"

    def test_resolve_with_already_completed_future_returns_false(self):
        broker = HumanInputBroker()
        fut = _future()
        # Caller (CARL) cancelled the future before resolve fires.
        fut.cancel()
        req = broker.submit("Q", future=fut)
        # The broker tries `set_result`, future raises, broker
        # swallows and returns False. Pending entry was removed
        # so a subsequent resolve also returns False.
        assert broker.resolve(req.id, "value") is False
        assert req.id not in broker


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_sets_exception_on_future(self):
        broker = HumanInputBroker()
        fut = _future()
        req = broker.submit("Q", future=fut)
        assert broker.cancel(req.id) is True
        with pytest.raises(HumanInputCancelled):
            fut.result()
        assert req.id not in broker

    def test_cancel_unknown_id_returns_false(self):
        broker = HumanInputBroker()
        assert broker.cancel("no-such-id") is False

    def test_cancel_custom_reason(self):
        broker = HumanInputBroker()
        fut = _future()
        req = broker.submit("Q", future=fut)
        broker.cancel(req.id, reason="timeout")
        try:
            fut.result()
        except HumanInputCancelled as exc:
            assert "timeout" in str(exc)
        else:
            pytest.fail("expected HumanInputCancelled")

    def test_cancel_all_drains_pending(self):
        broker = HumanInputBroker()
        futs = [_future() for _ in range(3)]
        for i, fut in enumerate(futs):
            broker.submit(f"Q{i}", future=fut)
        count = broker.cancel_all()
        assert count == 3
        assert len(broker) == 0
        for fut in futs:
            with pytest.raises(HumanInputCancelled):
                fut.result()

    def test_cancel_all_empty_returns_zero(self):
        broker = HumanInputBroker()
        assert broker.cancel_all() == 0


# ---------------------------------------------------------------------------
# Listeners
# ---------------------------------------------------------------------------


class TestListeners:
    def test_listener_fires_on_submit(self):
        broker = HumanInputBroker()
        seen: list[HumanInputRequest] = []
        broker.on_request(seen.append)
        req = broker.submit("Q", future=_future())
        assert len(seen) == 1
        assert seen[0] is req

    def test_multiple_listeners_all_fire(self):
        broker = HumanInputBroker()
        seen_a: list[str] = []
        seen_b: list[str] = []
        broker.on_request(lambda r: seen_a.append(r.id))
        broker.on_request(lambda r: seen_b.append(r.id))
        req = broker.submit("Q", future=_future())
        assert seen_a == [req.id]
        assert seen_b == [req.id]

    def test_unsubscribe_stops_notifications(self):
        broker = HumanInputBroker()
        seen: list[HumanInputRequest] = []
        unsub = broker.on_request(seen.append)
        broker.submit("first", future=_future())
        unsub()
        broker.submit("second", future=_future())
        assert len(seen) == 1
        assert seen[0].prompt == "first"

    def test_listener_exception_does_not_break_broker(self):
        broker = HumanInputBroker()
        broker.on_request(lambda r: (_ for _ in ()).throw(RuntimeError("bad listener")))
        # Subsequent listener still gets the event.
        good_calls: list[str] = []
        broker.on_request(lambda r: good_calls.append(r.id))
        req = broker.submit("Q", future=_future())
        assert good_calls == [req.id]
        # Broker still functional — request is pending.
        assert req.id in broker

    def test_unsubscribe_idempotent(self):
        broker = HumanInputBroker()
        unsub = broker.on_request(lambda r: None)
        unsub()
        # Second call is a no-op; doesn't raise.
        unsub()


# ---------------------------------------------------------------------------
# attach_to_context
# ---------------------------------------------------------------------------


class TestAttachToContext:
    def test_handler_assignment(self):
        broker = HumanInputBroker()

        class _Ctx:
            on_human_input_requested = None

        ctx = _Ctx()
        attach_to_context(broker, ctx)
        assert callable(ctx.on_human_input_requested)

    def test_handler_invocation_submits_to_broker(self):
        broker = HumanInputBroker()

        class _Ctx:
            on_human_input_requested = None

        ctx = _Ctx()
        attach_to_context(broker, ctx)
        fut = _future()
        ctx.on_human_input_requested("Continue?", fut)
        assert len(broker) == 1
        req = broker.pending()[0]
        assert req.prompt == "Continue?"
        # Resolve via broker → future gets the value.
        broker.resolve(req.id, "yes")
        assert fut.result() == "yes"


# ---------------------------------------------------------------------------
# Thread-safety smoke
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_submits_dont_lose_entries(self):
        broker = HumanInputBroker()
        N = 32
        futs = [_future() for _ in range(N)]

        def submit(i: int) -> None:
            broker.submit(f"Q{i}", future=futs[i])

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(broker) == N

    def test_concurrent_submit_resolve_round_trip(self):
        broker = HumanInputBroker()
        N = 16
        results: dict[int, str] = {}

        def submit_then_resolve(i: int) -> None:
            fut = _future()
            req = broker.submit(f"Q{i}", future=fut)
            broker.resolve(req.id, f"value-{i}")
            results[i] = fut.result()

        threads = [
            threading.Thread(target=submit_then_resolve, args=(i,))
            for i in range(N)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == N
        for i, v in results.items():
            assert v == f"value-{i}"
        # All resolved → none pending.
        assert len(broker) == 0


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequestShape:
    def test_request_is_frozen(self):
        broker = HumanInputBroker()
        req = broker.submit("Q", future=_future())
        with pytest.raises(Exception):
            req.prompt = "other"  # type: ignore[misc]

    def test_request_default_factories_independent(self):
        broker = HumanInputBroker()
        a = broker.submit("Q1", future=_future())
        b = broker.submit("Q2", future=_future())
        # Different ids → different requests.
        assert a.id != b.id
