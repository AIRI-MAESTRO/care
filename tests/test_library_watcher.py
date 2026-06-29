"""Tests for ``care.runtime.library_watcher`` (TODO §3 P1).

Strategy: the watcher delegates to the SDK's ``watch_entities``,
so tests stub that method with a tiny fake that captures the
callback and lets us drive it manually. End-to-end behaviour
(closure conversion, stop/idempotency, context-manager wrapping,
typed-event field promotion) is verified without touching real
SSE infrastructure.

Coverage layers:
1. ``LibraryEvent.from_raw`` — every field shape promoted
   correctly; missing fields default safely; bad timestamps
   degrade gracefully; tags accept str / list / tuple / set.
2. ``LibraryEvent`` predicates — ``is_known_kind`` flags unknown
   kinds; ``is_terminal`` only true for ``deleted``.
3. ``LibrarySubscription`` — `.stop()` delegates + is idempotent;
   context-manager exits stop the subscription.
4. ``watch_library`` — forwards filter kwargs to the SDK, wraps
   the callback so the consumer sees typed events, swallows
   conversion errors per-event without killing the subscription.
5. ``CareMemory.watch_library`` — facade wrapper passes through
   the same filters + returns a LibrarySubscription.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from gigaevo_client import GigaEvoClient

from care.memory import CareMemory
from care.runtime import (
    EVENT_KINDS,
    LibraryEvent,
    LibrarySubscription,
    watch_library,
)


# ---------------------------------------------------------------------------
# Fake SDK client with a controllable watch_entities
# ---------------------------------------------------------------------------


class _FakeSubscription:
    def __init__(self) -> None:
        self.stopped = False
        self.stop_calls = 0

    def stop(self) -> None:
        self.stopped = True
        self.stop_calls += 1


class _FakeClient:
    """Captures the wrapped callback so tests can drive events
    through it without spinning up real SSE."""

    def __init__(self) -> None:
        self.captured_kwargs: dict[str, Any] | None = None
        self.captured_callback: Any = None
        self.last_subscription: _FakeSubscription | None = None

    def watch_entities(self, callback, *, entity_type=None, entity_id=None,
                       namespace=None, tags=None, event_type=None):
        self.captured_kwargs = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "namespace": namespace,
            "tags": tags,
            "event_type": event_type,
        }
        self.captured_callback = callback
        self.last_subscription = _FakeSubscription()
        return self.last_subscription


# ---------------------------------------------------------------------------
# LibraryEvent.from_raw
# ---------------------------------------------------------------------------


class TestLibraryEventFromRaw:
    def test_full_payload(self):
        raw = {
            "event_type": "updated",
            "entity_id": "e-1",
            "entity_type": "chain",
            "version_id": "v-9",
            "channel": "latest",
            "namespace": "glazkov",
            "tags": ["finance", "demo"],
            "timestamp": "2026-05-19T12:00:00+00:00",
        }
        evt = LibraryEvent.from_raw(raw)
        assert evt.event_type == "updated"
        assert evt.entity_id == "e-1"
        assert evt.entity_type == "chain"
        assert evt.version_id == "v-9"
        assert evt.channel == "latest"
        assert evt.namespace == "glazkov"
        assert evt.tags == ("finance", "demo")
        assert evt.timestamp == datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
        # raw preserved verbatim for forward-compat.
        assert evt.raw == raw

    def test_minimal_payload_defaults_blank(self):
        evt = LibraryEvent.from_raw({})
        assert evt.event_type == ""
        assert evt.entity_id == ""
        assert evt.entity_type == ""
        assert evt.version_id is None
        assert evt.channel is None
        assert evt.namespace is None
        assert evt.tags == ()
        assert evt.timestamp is None

    @pytest.mark.parametrize(
        "tags_in",
        [
            ["a", "b"],
            ("a", "b"),
            {"a", "b"},
        ],
    )
    def test_tags_accept_str_list_tuple_set(self, tags_in):
        evt = LibraryEvent.from_raw({"tags": tags_in})
        assert set(evt.tags) == {"a", "b"}
        # Always materialised as a tuple — hashable for frozen dataclass.
        assert isinstance(evt.tags, tuple)

    def test_tags_non_iterable_degrades_to_empty(self):
        evt = LibraryEvent.from_raw({"tags": 42})
        assert evt.tags == ()

    def test_bad_timestamp_string_degrades_to_none(self):
        evt = LibraryEvent.from_raw({"timestamp": "not-a-date"})
        assert evt.timestamp is None

    def test_already_datetime_timestamp_kept(self):
        ts = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
        evt = LibraryEvent.from_raw({"timestamp": ts})
        assert evt.timestamp is ts

    def test_empty_string_fields_become_none(self):
        evt = LibraryEvent.from_raw(
            {"version_id": "", "channel": "", "namespace": ""}
        )
        assert evt.version_id is None
        assert evt.channel is None
        assert evt.namespace is None


# ---------------------------------------------------------------------------
# LibraryEvent predicates
# ---------------------------------------------------------------------------


class TestLibraryEventPredicates:
    @pytest.mark.parametrize("kind", EVENT_KINDS)
    def test_is_known_kind_true_for_canonical(self, kind):
        evt = LibraryEvent(event_type=kind, entity_id="x", entity_type="chain")
        assert evt.is_known_kind is True

    def test_is_known_kind_false_for_unknown(self):
        evt = LibraryEvent(
            event_type="future_event", entity_id="x", entity_type="chain"
        )
        assert evt.is_known_kind is False

    def test_is_terminal_only_for_deleted(self):
        for kind in EVENT_KINDS:
            evt = LibraryEvent(event_type=kind, entity_id="x", entity_type="chain")
            assert evt.is_terminal is (kind == "deleted")


# ---------------------------------------------------------------------------
# LibrarySubscription
# ---------------------------------------------------------------------------


class TestLibrarySubscription:
    def test_stop_delegates_to_underlying(self):
        sub = _FakeSubscription()
        wrapper = LibrarySubscription(sub)
        wrapper.stop()
        assert sub.stopped is True

    def test_stop_idempotent_at_wrapper_level(self):
        """Calling .stop() twice on the wrapper hits the SDK twice
        too; the SDK's own Subscription is responsible for being
        a no-op the second time (it is). Our test just verifies we
        don't add an extra layer of "you already called this" that
        would mask SDK behaviour."""
        sub = _FakeSubscription()
        wrapper = LibrarySubscription(sub)
        wrapper.stop()
        wrapper.stop()
        assert sub.stop_calls == 2  # passed through both times

    def test_context_manager_stops_on_exit(self):
        sub = _FakeSubscription()
        with LibrarySubscription(sub) as wrapper:
            assert wrapper.underlying is sub
            assert sub.stopped is False
        assert sub.stopped is True

    def test_context_manager_stops_on_exception(self):
        sub = _FakeSubscription()
        with pytest.raises(RuntimeError, match="boom"):
            with LibrarySubscription(sub):
                raise RuntimeError("boom")
        assert sub.stopped is True

    def test_stop_with_no_stop_method_is_silent(self):
        """If a future SDK Subscription drops the stop method
        we shouldn't crash."""

        class NoStop:
            pass

        wrapper = LibrarySubscription(NoStop())
        wrapper.stop()  # No exception.


# ---------------------------------------------------------------------------
# watch_library
# ---------------------------------------------------------------------------


class TestWatchLibrary:
    def test_forwards_filters_to_sdk(self):
        client = _FakeClient()
        watch_library(
            client,
            lambda evt: None,
            namespace="glazkov",
            entity_type="agent",
            entity_id="e-1",
            tags=["finance"],
            event_type="run_recorded",
        )
        assert client.captured_kwargs == {
            "entity_type": "agent",
            "entity_id": "e-1",
            "namespace": "glazkov",
            "tags": ["finance"],
            "event_type": "run_recorded",
        }

    def test_callback_receives_typed_events(self):
        client = _FakeClient()
        events: list[LibraryEvent] = []
        watch_library(client, events.append, namespace="glazkov")

        # Drive an event through the captured wrapper callback.
        client.captured_callback(
            {
                "event_type": "favourite_toggled",
                "entity_id": "e-1",
                "entity_type": "agent",
                "namespace": "glazkov",
                "tags": ["favourite"],
            }
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, LibraryEvent)
        assert evt.event_type == "favourite_toggled"
        assert evt.entity_id == "e-1"
        assert evt.tags == ("favourite",)

    def test_conversion_error_is_swallowed_per_event(self, monkeypatch):
        """A single malformed event must NOT kill the
        subscription. We force `from_raw` to raise and verify
        the wrapper closure swallows it without re-raising — that's
        the property that keeps the SSE stream alive past one
        bad message."""
        client = _FakeClient()
        events: list[LibraryEvent] = []
        watch_library(client, events.append, namespace="glazkov")

        def boom(_raw):
            raise RuntimeError("simulated parse failure")

        monkeypatch.setattr(LibraryEvent, "from_raw", staticmethod(boom))

        # The whole point of the test: this call should NOT raise.
        client.captured_callback({"event_type": "updated", "entity_id": "x"})
        assert events == []

    def test_returns_library_subscription_wrapper(self):
        client = _FakeClient()
        sub = watch_library(client, lambda evt: None)
        assert isinstance(sub, LibrarySubscription)
        # Stopping the wrapper stops the SDK subscription.
        sub.stop()
        assert client.last_subscription.stopped is True


# ---------------------------------------------------------------------------
# CareMemory.watch_library
# ---------------------------------------------------------------------------


class TestCareMemoryWatchLibrary:
    """Verify the facade method forwards to the shared helper.

    Uses a real `GigaEvoClient` (cheap to construct) wrapped in a
    `CareMemory`, then monkey-patches `watch_entities` on the client
    so we can assert the forwarding without touching real SSE."""

    def test_forwards_filters(self, monkeypatch):
        client = GigaEvoClient(base_url="http://test", api_key="sk-x", timeout=1.0)
        captured: dict = {}
        sentinel_sub = _FakeSubscription()

        def fake_watch(callback, **kwargs):
            captured["callback"] = callback
            captured["kwargs"] = kwargs
            return sentinel_sub

        monkeypatch.setattr(client, "watch_entities", fake_watch)
        mem = CareMemory(client)

        sub = mem.watch_library(
            lambda evt: None,
            namespace="glazkov",
            entity_type="chain",
            event_type="run_recorded",
        )
        assert isinstance(sub, LibrarySubscription)
        assert captured["kwargs"]["namespace"] == "glazkov"
        assert captured["kwargs"]["entity_type"] == "chain"
        assert captured["kwargs"]["event_type"] == "run_recorded"
        assert sub.underlying is sentinel_sub
