"""Tests for the LibraryScreen per-row actions data layer (TODO §1.3 P0).

The Textual key handler + context menu are gated on §1 P0;
this suite pins the contract those screens will bind to.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError

import pytest

from care.runtime.library_view import LibraryRow
from care.runtime.row_actions import (
    RowAction,
    RowActionError,
    RowMutationOutcome,
    actions_for_row,
    default_actions,
    delete_row,
    duplicate_chain,
    find_action_by_key,
    find_action_by_kind,
    is_destructive,
    toggle_favourite_row,
)


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    entity_id: str = "ent-1",
    entity_type: str = "chain",
    favourite: bool = False,
    is_draft: bool = False,
    is_evolved: bool = False,
    channel: str = "latest",
    tags: tuple[str, ...] = (),
) -> LibraryRow:
    return LibraryRow(
        entity_id=entity_id,
        entity_type=entity_type,
        display_name="row",
        favourite=favourite,
        is_draft=is_draft,
        is_evolved=is_evolved,
        channel=channel,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_default_actions_order_matches_spec(self):
        kinds = [a.kind for a in default_actions()]
        assert kinds == [
            "run", "open", "edit", "duplicate", "evolve",
            "archive_evolutions", "show_lineage",
            "toggle_favourite", "delete",
        ]

    def test_action_is_frozen(self):
        action = default_actions()[0]
        with pytest.raises(FrozenInstanceError):
            action.label = "x"  # type: ignore[misc]

    def test_key_bindings_match_spec(self):
        # TODO bullet pins specific keys.
        registry = {a.kind: a for a in default_actions()}
        assert registry["run"].key_binding == "R"
        assert registry["open"].key_binding == "Enter"
        assert registry["edit"].key_binding == "E"
        assert registry["toggle_favourite"].key_binding == "F"
        assert registry["delete"].key_binding == "Delete"

    def test_delete_requires_confirm(self):
        registry = {a.kind: a for a in default_actions()}
        assert registry["delete"].requires_confirm is True
        assert registry["delete"].is_destructive is True

    def test_non_destructive_actions(self):
        for action in default_actions():
            if action.kind != "delete":
                assert action.requires_confirm is False
                assert action.is_destructive is False


class TestActionsForRow:
    def test_runnable_includes_all_actions(self):
        row = _row()
        kinds = {a.kind for a in actions_for_row(row)}
        # Every default action applies.
        assert kinds == {a.kind for a in default_actions()}

    def test_draft_excludes_evolve_and_lineage(self):
        row = _row(is_draft=True, tags=("draft",))
        kinds = {a.kind for a in actions_for_row(row)}
        assert "evolve" not in kinds
        assert "show_lineage" not in kinds
        # Run/Edit/Delete etc. still apply.
        assert "run" in kinds
        assert "delete" in kinds

    def test_evolved_includes_evolve_and_lineage(self):
        row = _row(is_evolved=True)
        kinds = {a.kind for a in actions_for_row(row)}
        assert "evolve" in kinds
        assert "show_lineage" in kinds

    def test_custom_registry_respected(self):
        custom = (
            RowAction(kind="run", label="Custom Run", key_binding="X"),
        )
        result = actions_for_row(_row(), registry=custom)
        assert len(result) == 1
        assert result[0].label == "Custom Run"

    def test_order_preserved(self):
        # Filtering shouldn't reorder.
        actions = actions_for_row(_row(is_draft=True, tags=("draft",)))
        assert [a.kind for a in actions] == [
            "run", "open", "edit", "duplicate",
            "toggle_favourite", "delete",
        ]


class TestFindAction:
    def test_find_by_key(self):
        action = find_action_by_key("R")
        assert action is not None
        assert action.kind == "run"

    def test_find_by_key_case_insensitive(self):
        assert find_action_by_key("r").kind == "run"
        assert find_action_by_key("DELETE").kind == "delete"
        assert find_action_by_key("delete").kind == "delete"

    def test_find_by_key_unknown_returns_none(self):
        assert find_action_by_key("Q") is None
        assert find_action_by_key("") is None

    def test_find_by_kind(self):
        action = find_action_by_kind("toggle_favourite")
        assert action is not None
        assert action.key_binding == "F"

    def test_find_by_kind_unknown_returns_none(self):
        # "unknown" isn't in RowActionKind — passes through as no-match.
        assert find_action_by_kind("not-a-kind") is None  # type: ignore[arg-type]


class TestIsDestructive:
    def test_with_action(self):
        registry = {a.kind: a for a in default_actions()}
        assert is_destructive(registry["delete"]) is True
        assert is_destructive(registry["run"]) is False

    def test_with_kind(self):
        assert is_destructive("delete") is True
        assert is_destructive("run") is False


# ---------------------------------------------------------------------------
# Stubs for mutators
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(
        self,
        *,
        fav_exc: Exception | None = None,
        delete_exc: Exception | None = None,
        get_response=None,
        get_exc: Exception | None = None,
        get_delay: float = 0.0,
        fav_delay: float = 0.0,
        delete_delay: float = 0.0,
    ):
        self.calls: list[dict] = []
        self._fav_exc = fav_exc
        self._delete_exc = delete_exc
        self._get_response = get_response
        self._get_exc = get_exc
        self._get_delay = get_delay
        self._fav_delay = fav_delay
        self._delete_delay = delete_delay

    def _mark_favourite(self, entity_type, entity_id, *, value=True):
        self.calls.append(
            {
                "op": "fav", "entity_type": entity_type,
                "entity_id": entity_id, "value": value,
            }
        )
        if self._fav_delay:
            time.sleep(self._fav_delay)
        if self._fav_exc:
            raise self._fav_exc
        return {"ok": True}

    def _delete_entity(self, entity_type, entity_id):
        self.calls.append(
            {
                "op": "delete", "entity_type": entity_type,
                "entity_id": entity_id,
            }
        )
        if self._delete_delay:
            time.sleep(self._delete_delay)
        if self._delete_exc:
            raise self._delete_exc
        return True

    def get_chain_dict(self, entity_id, channel="latest", **_):
        self.calls.append(
            {"op": "get", "entity_id": entity_id, "channel": channel}
        )
        if self._get_delay:
            time.sleep(self._get_delay)
        if self._get_exc:
            raise self._get_exc
        return self._get_response


class _StubMemory:
    def __init__(self, client, *, save_exc=None, save_delay=0.0, save_result="new-id"):
        self.client = client
        self.save_calls: list[dict] = []
        self._save_exc = save_exc
        self._save_delay = save_delay
        self._save_result = save_result

    def save_chain(self, chain, *, name, tags=None, entity_id=None,
                   channel="latest", **_):
        self.save_calls.append(
            {
                "chain": chain, "name": name, "tags": list(tags) if tags else None,
                "entity_id": entity_id, "channel": channel,
            }
        )
        if self._save_delay:
            time.sleep(self._save_delay)
        if self._save_exc:
            raise self._save_exc
        return self._save_result


# ---------------------------------------------------------------------------
# toggle_favourite_row
# ---------------------------------------------------------------------------


class TestToggleFavouriteRow:
    def test_flips_current_state(self):
        client = _StubClient()
        memory = _StubMemory(client)
        row = _row(favourite=False)
        outcome = asyncio.run(toggle_favourite_row(memory, row))
        assert outcome.success
        assert outcome.detail["current"] is True
        assert client.calls[0]["value"] is True

    def test_flips_when_already_favourite(self):
        client = _StubClient()
        memory = _StubMemory(client)
        row = _row(favourite=True)
        asyncio.run(toggle_favourite_row(memory, row))
        assert client.calls[0]["value"] is False

    def test_explicit_value_overrides_flip(self):
        client = _StubClient()
        memory = _StubMemory(client)
        row = _row(favourite=False)
        outcome = asyncio.run(
            toggle_favourite_row(memory, row, value=False)
        )
        # Despite the row already being unfavourited, explicit
        # value=False still fires (idempotent server-side).
        assert outcome.success
        assert client.calls[0]["value"] is False

    def test_routes_to_typed_endpoint(self):
        client = _StubClient()
        memory = _StubMemory(client)
        row = _row(entity_type="agent_skill")
        asyncio.run(toggle_favourite_row(memory, row))
        assert client.calls[0]["entity_type"] == "agent_skill"

    def test_failure_surfaces_on_outcome(self):
        client = _StubClient(fav_exc=RuntimeError("503"))
        memory = _StubMemory(client)
        outcome = asyncio.run(toggle_favourite_row(memory, _row()))
        assert outcome.success is False
        assert "503" in outcome.error

    def test_timeout(self):
        client = _StubClient(fav_delay=0.5)
        memory = _StubMemory(client)
        outcome = asyncio.run(
            toggle_favourite_row(memory, _row(), timeout=0.05)
        )
        assert outcome.success is False
        assert "timed out" in outcome.error

    def test_missing_method_returns_error_outcome(self):
        class _Empty:
            pass

        memory = _StubMemory(_Empty())
        outcome = asyncio.run(toggle_favourite_row(memory, _row()))
        assert outcome.success is False
        assert "_mark_favourite" in outcome.error


# ---------------------------------------------------------------------------
# delete_row
# ---------------------------------------------------------------------------


class TestDeleteRow:
    def test_happy_path(self):
        client = _StubClient()
        memory = _StubMemory(client)
        outcome = asyncio.run(delete_row(memory, _row()))
        assert outcome.success
        assert client.calls[0]["op"] == "delete"
        assert client.calls[0]["entity_type"] == "chain"

    def test_routes_to_typed_endpoint(self):
        client = _StubClient()
        memory = _StubMemory(client)
        row = _row(entity_type="agent")
        asyncio.run(delete_row(memory, row))
        assert client.calls[0]["entity_type"] == "agent"

    def test_failure_surfaces(self):
        client = _StubClient(delete_exc=RuntimeError("503"))
        memory = _StubMemory(client)
        outcome = asyncio.run(delete_row(memory, _row()))
        assert outcome.success is False

    def test_timeout(self):
        client = _StubClient(delete_delay=0.5)
        memory = _StubMemory(client)
        outcome = asyncio.run(
            delete_row(memory, _row(), timeout=0.05)
        )
        assert outcome.success is False
        assert "timed out" in outcome.error

    def test_missing_method(self):
        class _Empty:
            pass

        memory = _StubMemory(_Empty())
        outcome = asyncio.run(delete_row(memory, _row()))
        assert outcome.success is False
        assert "_delete_entity" in outcome.error


# ---------------------------------------------------------------------------
# duplicate_chain
# ---------------------------------------------------------------------------


class TestDuplicateChain:
    def test_round_trip(self):
        client = _StubClient(get_response={"steps": [{"number": 1}]})
        memory = _StubMemory(client, save_result="dup-1")
        outcome = asyncio.run(duplicate_chain(memory, _row()))
        assert outcome.success
        assert outcome.detail["new_entity_id"] == "dup-1"
        # GET fired first, then save_chain with entity_id=None.
        assert client.calls[0]["op"] == "get"
        assert memory.save_calls[0]["entity_id"] is None
        # Default name has " (copy)" suffix.
        assert "(copy)" in memory.save_calls[0]["name"]

    def test_custom_name(self):
        client = _StubClient(get_response={"steps": []})
        memory = _StubMemory(client)
        asyncio.run(
            duplicate_chain(memory, _row(), new_name="My Clone")
        )
        assert memory.save_calls[0]["name"] == "My Clone"

    def test_unwraps_entity_response_content(self):
        # `get_chain_dict` may return either the raw chain OR a
        # full EntityResponse wrapper; the duplicator unwraps to
        # the content dict before save_chain.
        client = _StubClient(
            get_response={
                "entity_id": "ent-1",
                "content": {"steps": [{"number": 1, "title": "Step"}]},
                "meta": {"tags": []},
            }
        )
        memory = _StubMemory(client)
        asyncio.run(duplicate_chain(memory, _row()))
        # save_chain saw just the content dict.
        saved_chain = memory.save_calls[0]["chain"]
        assert saved_chain == {"steps": [{"number": 1, "title": "Step"}]}

    def test_get_failure_surfaces(self):
        client = _StubClient(get_exc=RuntimeError("503"))
        memory = _StubMemory(client)
        outcome = asyncio.run(duplicate_chain(memory, _row()))
        assert outcome.success is False
        assert "503" in outcome.error
        # save_chain never called.
        assert memory.save_calls == []

    def test_not_found_surfaces(self):
        client = _StubClient(get_response=None)
        memory = _StubMemory(client)
        outcome = asyncio.run(duplicate_chain(memory, _row()))
        assert outcome.success is False
        assert "not found" in outcome.error

    def test_save_failure_surfaces(self):
        client = _StubClient(get_response={"steps": []})
        memory = _StubMemory(client, save_exc=RuntimeError("422"))
        outcome = asyncio.run(duplicate_chain(memory, _row()))
        assert outcome.success is False
        assert "422" in outcome.error

    def test_get_timeout(self):
        client = _StubClient(
            get_response={"steps": []}, get_delay=0.5,
        )
        memory = _StubMemory(client)
        outcome = asyncio.run(
            duplicate_chain(memory, _row(), timeout=0.05)
        )
        assert outcome.success is False
        assert "timed out" in outcome.error

    def test_save_timeout(self):
        client = _StubClient(get_response={"steps": []})
        memory = _StubMemory(client, save_delay=0.5)
        outcome = asyncio.run(
            duplicate_chain(memory, _row(), timeout=0.05)
        )
        assert outcome.success is False
        assert "timed out" in outcome.error

    def test_tags_carried_over(self):
        client = _StubClient(get_response={"steps": []})
        memory = _StubMemory(client)
        row = _row(tags=("domain:weather", "favourite"))
        asyncio.run(duplicate_chain(memory, row))
        saved = memory.save_calls[0]
        assert "domain:weather" in (saved["tags"] or [])

    def test_missing_get_method(self):
        class _Empty:
            pass

        memory = _StubMemory(_Empty())
        outcome = asyncio.run(duplicate_chain(memory, _row()))
        assert outcome.success is False
        assert "get_chain_dict" in outcome.error

    def test_missing_save_method(self):
        class _NoSave:
            def __init__(self, client):
                self.client = client

        client = _StubClient(get_response={"steps": []})
        memory = _NoSave(client)
        outcome = asyncio.run(duplicate_chain(memory, _row()))
        assert outcome.success is False
        assert "save_chain" in outcome.error


# ---------------------------------------------------------------------------
# RowMutationOutcome shape
# ---------------------------------------------------------------------------


class TestOutcome:
    def test_outcome_is_frozen(self):
        outcome = RowMutationOutcome(entity_id="x")
        with pytest.raises(FrozenInstanceError):
            outcome.success = False  # type: ignore[misc]

    def test_default_success_true(self):
        outcome = RowMutationOutcome(entity_id="x")
        assert outcome.success is True
        assert outcome.error is None


# ---------------------------------------------------------------------------
# Missing-client error path on the synchronous resolve helper
# ---------------------------------------------------------------------------


class TestClientResolution:
    def test_no_client_attribute_raises_sync(self):
        from care.runtime.row_actions import _resolve_client

        with pytest.raises(RowActionError, match="`.client`"):
            _resolve_client(object())

    def test_underscored_client_accepted(self):
        from care.runtime.row_actions import _resolve_client

        class _M:
            def __init__(self, client):
                self._client = client

        sentinel = object()
        assert _resolve_client(_M(sentinel)) is sentinel


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            RowAction as A,
            RowActionError as Err,
            RowMutationOutcome as O,
            actions_for_row as af,
            default_actions as defaults,
            delete_row as dr,
            duplicate_chain as dc,
            find_action_by_key as fak,
            find_action_by_kind as fakd,
            is_destructive as isd,
            toggle_favourite_row as tfr,
        )

        assert A is RowAction
        assert Err is RowActionError
        assert O is RowMutationOutcome
        assert af is actions_for_row
        assert defaults is default_actions
        assert dr is delete_row
        assert dc is duplicate_chain
        assert fak is find_action_by_key
        assert fakd is find_action_by_kind
        assert isd is is_destructive
        assert tfr is toggle_favourite_row
