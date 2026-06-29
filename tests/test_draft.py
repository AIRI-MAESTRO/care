"""Tests for ``care.runtime.draft`` (TODO §3 P0).

End-to-end against a ``respx``-mocked Memory:

1. ``auto_save_draft`` posts to ``/v1/chains`` with
   ``meta.channel="draft"`` and the ``"draft"`` tag stamped onto
   the request body.
2. ``promote_draft`` posts to ``/v1/chains/{id}/promote`` with
   ``from_channel="draft"`` / ``to_channel="latest"`` and flips
   ``session.promoted=True``.
3. ``discard_draft`` posts (DELETE) to ``/v1/chains/{id}`` and
   flips ``session.discarded=True``.
4. Lifecycle guards: can't promote-after-discard, can't
   discard-after-promote, second-discard is a no-op.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest
import respx
from gigaevo_client import GigaEvoClient

from care.memory import CareMemory
from care.runtime import (
    DRAFT_CHANNEL,
    DRAFT_TAG,
    DraftError,
    DraftSession,
    auto_save_draft,
    discard_draft,
    promote_draft,
)

BASE = "http://test-memory:8000"


@pytest.fixture
def memory():
    return CareMemory(GigaEvoClient(base_url=BASE, api_key="sk-test", timeout=5.0))


def _save_handler(entity_id: str = "draft-1"):
    captured: dict = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "entity_type": "chain",
                "entity_id": entity_id,
                "version_id": "v-1",
                "channel": DRAFT_CHANNEL,
            },
        )

    return handler, captured


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_draft_channel_is_draft(self):
        assert DRAFT_CHANNEL == "draft"

    def test_draft_tag_matches_channel(self):
        """The tag and channel both being "draft" is intentional —
        a single string the LibraryScreen filter recognises in both
        contexts."""
        assert DRAFT_TAG == "draft"


# ---------------------------------------------------------------------------
# DraftSession shape
# ---------------------------------------------------------------------------


class TestDraftSession:
    def test_defaults(self):
        s = DraftSession(entity_id="x", name="n")
        assert s.entity_type == "chain"
        assert s.promoted is False
        assert s.discarded is False
        assert s.terminal is False

    def test_terminal_true_after_promote(self):
        s = DraftSession(entity_id="x", name="n", promoted=True)
        assert s.terminal is True

    def test_terminal_true_after_discard(self):
        s = DraftSession(entity_id="x", name="n", discarded=True)
        assert s.terminal is True


# ---------------------------------------------------------------------------
# auto_save_draft
# ---------------------------------------------------------------------------


class TestAutoSaveDraft:
    @respx.mock
    def test_writes_to_draft_channel_with_draft_tag(self, memory):
        handler, captured = _save_handler("draft-7")
        respx.post(f"{BASE}/v1/chains").mock(side_effect=handler)

        session = auto_save_draft(
            memory,
            {"version": "1.1", "steps": []},
            name="weather-bot",
            query="weather report",
            domain="weather",
        )

        assert isinstance(session, DraftSession)
        assert session.entity_id == "draft-7"
        assert session.name == "weather-bot"
        assert session.terminal is False

        body = captured["body"]
        assert body["channel"] == DRAFT_CHANNEL
        # DRAFT_TAG always present (regardless of position — the
        # CareMemory facade also prepends a domain tag).
        assert DRAFT_TAG in body["meta"]["tags"]
        # User-supplied tags still present + domain stamp.
        assert "domain:weather" in body["meta"]["tags"]
        # Original query persisted via CareChainMetadata.
        assert body["content"]["metadata"]["task_description"] == "weather report"

    @respx.mock
    def test_user_tags_merged_without_duplicating_draft(self, memory):
        handler, captured = _save_handler()
        respx.post(f"{BASE}/v1/chains").mock(side_effect=handler)

        auto_save_draft(
            memory,
            {"steps": []},
            name="x",
            tags=["finance", DRAFT_TAG, "demo"],
        )
        tags = captured["body"]["meta"]["tags"]
        assert tags.count(DRAFT_TAG) == 1
        assert "finance" in tags
        assert "demo" in tags


# ---------------------------------------------------------------------------
# promote_draft
# ---------------------------------------------------------------------------


class TestPromoteDraft:
    @respx.mock
    def test_promotes_and_marks_session(self, memory):
        captured: dict = {}

        def promote(request):
            captured["body"] = json.loads(request.content)
            captured["url"] = str(request.url)
            return httpx.Response(
                200, json={"from_channel": "draft", "to_channel": "latest"}
            )

        respx.post(f"{BASE}/v1/chains/draft-1/promote").mock(side_effect=promote)
        session = DraftSession(entity_id="draft-1", name="x")
        out = promote_draft(memory, session)

        assert out is session  # in-place mutation
        assert session.promoted is True
        assert session.terminal is True
        assert captured["body"]["from_channel"] == "draft"
        assert captured["body"]["to_channel"] == "latest"
        assert "/v1/chains/draft-1/promote" in captured["url"]

    @respx.mock
    def test_custom_target_channel(self, memory):
        respx.post(f"{BASE}/v1/chains/draft-1/promote").mock(
            return_value=httpx.Response(200, json={})
        )
        session = DraftSession(entity_id="draft-1", name="x")
        promote_draft(memory, session, to_channel="stable")
        # Just smoke: it didn't raise.
        assert session.promoted is True

    def test_promote_after_discard_raises(self, memory):
        session = DraftSession(entity_id="draft-1", name="x", discarded=True)
        with pytest.raises(DraftError, match="already discarded"):
            promote_draft(memory, session)

    def test_promote_after_promote_raises(self, memory):
        session = DraftSession(entity_id="draft-1", name="x", promoted=True)
        with pytest.raises(DraftError, match="already promoted"):
            promote_draft(memory, session)

    @respx.mock
    def test_server_error_wraps_in_draft_error(self, memory):
        respx.post(f"{BASE}/v1/chains/draft-1/promote").mock(
            return_value=httpx.Response(500, json={"detail": "boom"})
        )
        session = DraftSession(entity_id="draft-1", name="x")
        with pytest.raises(DraftError, match="failed to promote"):
            promote_draft(memory, session)
        # On failure, session must NOT be marked promoted.
        assert session.promoted is False


# ---------------------------------------------------------------------------
# discard_draft
# ---------------------------------------------------------------------------


class TestDiscardDraft:
    @respx.mock
    def test_deletes_and_marks_session(self, memory):
        captured: dict = {}

        def delete(request):
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"deleted": True})

        respx.delete(
            re.compile(rf"{BASE}/v1/chains/[^/]+")
        ).mock(side_effect=delete)

        session = DraftSession(entity_id="draft-1", name="x")
        out = discard_draft(memory, session)

        assert out is session
        assert session.discarded is True
        assert session.terminal is True
        assert "/v1/chains/draft-1" in captured["url"]

    @respx.mock
    def test_second_discard_is_noop(self, memory):
        """No HTTP request the second time — already gone."""
        delete_route = respx.delete(re.compile(rf"{BASE}/v1/chains/[^/]+")).mock(
            return_value=httpx.Response(200, json={"deleted": True})
        )
        session = DraftSession(entity_id="draft-1", name="x")
        discard_draft(memory, session)
        first_call_count = delete_route.call_count
        discard_draft(memory, session)
        assert delete_route.call_count == first_call_count
        assert session.discarded is True

    def test_discard_after_promote_raises(self, memory):
        session = DraftSession(entity_id="draft-1", name="x", promoted=True)
        with pytest.raises(DraftError, match="refusing to discard promoted"):
            discard_draft(memory, session)

    @respx.mock
    def test_server_error_wraps_in_draft_error(self, memory):
        respx.delete(re.compile(rf"{BASE}/v1/chains/[^/]+")).mock(
            return_value=httpx.Response(500, json={"detail": "boom"})
        )
        session = DraftSession(entity_id="draft-1", name="x")
        with pytest.raises(DraftError, match="failed to discard"):
            discard_draft(memory, session)
        assert session.discarded is False


# ---------------------------------------------------------------------------
# Lifecycle: save → promote / discard happy paths
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    @respx.mock
    def test_save_then_promote(self, memory):
        save_h, _ = _save_handler("draft-99")
        respx.post(f"{BASE}/v1/chains").mock(side_effect=save_h)
        respx.post(f"{BASE}/v1/chains/draft-99/promote").mock(
            return_value=httpx.Response(200, json={})
        )

        session = auto_save_draft(memory, {"steps": []}, name="x")
        promote_draft(memory, session)
        assert session.promoted is True
        assert session.discarded is False

    @respx.mock
    def test_save_then_discard(self, memory):
        save_h, _ = _save_handler("draft-99")
        respx.post(f"{BASE}/v1/chains").mock(side_effect=save_h)
        respx.delete(
            re.compile(rf"{BASE}/v1/chains/[^/]+")
        ).mock(return_value=httpx.Response(200, json={"deleted": True}))

        session = auto_save_draft(memory, {"steps": []}, name="x")
        discard_draft(memory, session)
        assert session.discarded is True
        assert session.promoted is False
