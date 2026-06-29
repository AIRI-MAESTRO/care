"""Tests for the library collections data layer (TODO §1.3 P2).

The Textual sidebar + context menu are gated on §1 P0; this
suite pins the contract the sidebar will bind to.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import FrozenInstanceError

import pytest

from care.runtime.bulk_ops import BulkSelection, BulkTarget
from care.runtime.collections import (
    COLLECTION_PREFIX,
    Collection,
    CollectionError,
    active_collection_name,
    apply_add_to_collection,
    apply_delete_collection,
    apply_remove_from_collection,
    apply_rename_collection,
    collection_name_from_tag,
    collection_tag_for,
    extract_collections_from_tags,
    filter_by_collection,
    is_collection_tag,
    list_collections,
)
from care.runtime.library_view import LibraryFilters


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_collection_tag_for(self):
        assert collection_tag_for("Marketing") == "collection:Marketing"

    def test_collection_tag_strips_whitespace(self):
        assert collection_tag_for("  Sales  ") == "collection:Sales"

    def test_collection_tag_empty_raises(self):
        with pytest.raises(CollectionError, match="empty"):
            collection_tag_for("")
        with pytest.raises(CollectionError, match="empty"):
            collection_tag_for("   ")

    def test_collection_tag_double_prefix_raises(self):
        # Refuse to silently normalise; callers should pass the bare name.
        with pytest.raises(CollectionError, match="already prefixed"):
            collection_tag_for("collection:Marketing")

    def test_collection_name_from_tag(self):
        assert collection_name_from_tag("collection:Sales") == "Sales"
        assert collection_name_from_tag("collection:  Sales  ") == "Sales"
        assert collection_name_from_tag("domain:weather") is None
        assert collection_name_from_tag("plain-tag") is None
        assert collection_name_from_tag("") is None
        assert collection_name_from_tag(None) is None  # type: ignore[arg-type]

    def test_collection_name_empty_after_prefix(self):
        # "collection:" alone isn't a valid collection.
        assert collection_name_from_tag("collection:") is None
        assert collection_name_from_tag("collection:   ") is None

    def test_is_collection_tag(self):
        assert is_collection_tag("collection:Marketing") is True
        assert is_collection_tag("domain:weather") is False
        assert is_collection_tag("collection:") is False
        assert is_collection_tag("") is False

    def test_extract_collections_from_tags(self):
        tags = [
            "domain:weather",
            "collection:Marketing",
            "favourite",
            "collection:Q1",
            "collection:Marketing",  # duplicate
        ]
        names = extract_collections_from_tags(tags)
        assert names == ("Marketing", "Q1")  # dedupe + insertion order

    def test_extract_collections_empty(self):
        assert extract_collections_from_tags([]) == ()
        assert extract_collections_from_tags(None) == ()

    def test_collection_prefix_constant(self):
        assert COLLECTION_PREFIX == "collection:"


# ---------------------------------------------------------------------------
# Collection model
# ---------------------------------------------------------------------------


class TestCollectionModel:
    def test_frozen(self):
        c = Collection(name="x", member_count=1)
        with pytest.raises(FrozenInstanceError):
            c.name = "y"  # type: ignore[misc]

    def test_tag_property(self):
        assert Collection(name="Sales").tag == "collection:Sales"

    def test_is_empty(self):
        assert Collection(name="x", member_count=0).is_empty is True
        assert Collection(name="x", member_count=3).is_empty is False


# ---------------------------------------------------------------------------
# Filter integration
# ---------------------------------------------------------------------------


class TestFilterIntegration:
    def test_filter_by_collection_adds_tag(self):
        filters = LibraryFilters(tags=("favourite",))
        new = filter_by_collection(filters, "Marketing")
        assert "favourite" in new.tags
        assert "collection:Marketing" in new.tags

    def test_filter_by_collection_replaces_existing_collection(self):
        # Switching from Marketing → Sales strips the old prefix.
        filters = LibraryFilters(
            tags=("favourite", "collection:Marketing")
        )
        new = filter_by_collection(filters, "Sales")
        assert "collection:Marketing" not in new.tags
        assert "collection:Sales" in new.tags
        assert "favourite" in new.tags

    def test_filter_by_collection_none_strips_collection_tags(self):
        filters = LibraryFilters(
            tags=("favourite", "collection:Marketing")
        )
        new = filter_by_collection(filters, None)
        assert new.tags == ("favourite",)

    def test_active_collection_name(self):
        # Empty filter → None.
        assert active_collection_name(LibraryFilters()) is None
        # Pinned → the bare name.
        filters = LibraryFilters(
            tags=("favourite", "collection:Q1"),
        )
        assert active_collection_name(filters) == "Q1"

    def test_active_collection_name_first_wins(self):
        # Defensive: malformed filters with two collection tags
        # return the first one rather than crashing.
        filters = LibraryFilters(
            tags=("collection:Marketing", "collection:Sales"),
        )
        assert active_collection_name(filters) == "Marketing"


# ---------------------------------------------------------------------------
# list_collections (async enumeration)
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(self, *, rows=None, exc=None, delay=0.0, bad_type=False):
        self.calls: list[dict] = []
        self._rows = rows or []
        self._exc = exc
        self._delay = delay
        self._bad_type = bad_type

    def list_chains(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self._delay:
            time.sleep(self._delay)
        if self._exc:
            raise self._exc
        if self._bad_type:
            return 42
        return self._rows


class _StubMemory:
    def __init__(self, client):
        self.client = client


def _row(entity_id: str, tags: list[str]) -> dict:
    return {
        "entity_id": entity_id,
        "meta": {"tags": tags},
    }


class TestListCollections:
    def test_empty_library_yields_empty(self):
        memory = _StubMemory(_StubClient(rows=[]))
        result = asyncio.run(list_collections(memory))
        assert result == ()

    def test_aggregates_counts_and_samples(self):
        memory = _StubMemory(_StubClient(rows=[
            _row("a", ["collection:Marketing", "favourite"]),
            _row("b", ["collection:Marketing", "collection:Q1"]),
            _row("c", ["collection:Q1"]),
            _row("d", []),  # no collections
        ]))
        result = asyncio.run(list_collections(memory))
        # Sorted alphabetically.
        assert [c.name for c in result] == ["Marketing", "Q1"]
        marketing = result[0]
        q1 = result[1]
        assert marketing.member_count == 2
        assert q1.member_count == 2
        assert "a" in marketing.sample_entity_ids
        assert "b" in marketing.sample_entity_ids
        assert "b" in q1.sample_entity_ids
        assert "c" in q1.sample_entity_ids

    def test_sample_capped(self):
        rows = [
            _row(f"e-{i}", ["collection:Big"]) for i in range(10)
        ]
        memory = _StubMemory(_StubClient(rows=rows))
        result = asyncio.run(
            list_collections(memory, sample_cap=3)
        )
        assert result[0].member_count == 10
        assert len(result[0].sample_entity_ids) == 3

    def test_zero_sample_cap(self):
        rows = [_row("e-1", ["collection:X"])]
        memory = _StubMemory(_StubClient(rows=rows))
        result = asyncio.run(list_collections(memory, sample_cap=0))
        assert result[0].member_count == 1
        assert result[0].sample_entity_ids == ()

    def test_namespace_channel_forwarded(self):
        memory = _StubMemory(_StubClient(rows=[]))
        asyncio.run(
            list_collections(
                memory, namespace="alice", channel="stable",
            )
        )
        call = memory.client.calls[0]
        assert call["namespace"] == "alice"
        assert call["channel"] == "stable"

    def test_fetch_limit_clamped(self):
        memory = _StubMemory(_StubClient(rows=[]))
        asyncio.run(list_collections(memory, fetch_limit=9999))
        assert memory.client.calls[0]["limit"] == 200
        memory.client.calls.clear()
        asyncio.run(list_collections(memory, fetch_limit=0))
        assert memory.client.calls[0]["limit"] == 1

    def test_missing_client_raises(self):
        with pytest.raises(CollectionError, match="list_chains"):
            asyncio.run(list_collections(object()))

    def test_sdk_exception_wraps(self):
        memory = _StubMemory(_StubClient(exc=RuntimeError("503")))
        with pytest.raises(CollectionError, match="enumeration failed"):
            asyncio.run(list_collections(memory))

    def test_timeout_wraps(self):
        memory = _StubMemory(_StubClient(rows=[], delay=0.5))
        with pytest.raises(CollectionError, match="timed out"):
            asyncio.run(list_collections(memory, timeout=0.05))

    def test_unexpected_type_wraps(self):
        memory = _StubMemory(_StubClient(bad_type=True))
        with pytest.raises(CollectionError, match="unexpected type"):
            asyncio.run(list_collections(memory))

    def test_underscored_client_works(self):
        class _M:
            def __init__(self, client):
                self._client = client

        memory = _M(_StubClient(rows=[_row("a", ["collection:X"])]))
        result = asyncio.run(list_collections(memory))
        assert result[0].name == "X"

    def test_attribute_access_rows_supported(self):
        # SDK shape — attribute access instead of dict.
        class _R:
            def __init__(self, entity_id, meta):
                self.entity_id = entity_id
                self.meta = meta

        memory = _StubMemory(_StubClient(rows=[
            _R("a", {"tags": ["collection:M"]}),
        ]))
        result = asyncio.run(list_collections(memory))
        assert result[0].name == "M"
        assert "a" in result[0].sample_entity_ids

    def test_malformed_meta_skipped(self):
        # Row with non-dict meta or non-list tags survives without
        # crashing.
        rows = [
            {"entity_id": "a", "meta": "not-a-dict"},
            {"entity_id": "b", "meta": {"tags": "not-a-list"}},
            _row("c", ["collection:Good"]),
        ]
        memory = _StubMemory(_StubClient(rows=rows))
        result = asyncio.run(list_collections(memory))
        assert len(result) == 1
        assert result[0].name == "Good"


# ---------------------------------------------------------------------------
# Bulk mutators
# ---------------------------------------------------------------------------


class _BulkStubClient:
    """Records `_update_metadata` calls so we can assert the
    rename/add/remove tag arguments."""

    def __init__(self):
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def _update_metadata(self, entity_type, entity_id, *,
                         display_name=None, description=None,
                         tags=None, favourite=None):
        with self._lock:
            self.calls.append(
                {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "tags": list(tags) if tags is not None else None,
                }
            )
        return {"ok": True}


class _BulkStubMemory:
    def __init__(self, client):
        self.client = client


def _target(entity_id="x", *, tags=()):
    return BulkTarget(
        entity_id=entity_id, entity_type="chain", current_tags=tuple(tags),
    )


class TestApplyMutators:
    def test_add_to_collection(self):
        client = _BulkStubClient()
        memory = _BulkStubMemory(client)
        sel = BulkSelection().add(_target("a", tags=("existing",)))
        result = asyncio.run(
            apply_add_to_collection(memory, sel, "Marketing")
        )
        assert result.all_succeeded
        patch = client.calls[0]
        assert "collection:Marketing" in (patch["tags"] or [])
        assert "existing" in (patch["tags"] or [])

    def test_remove_from_collection(self):
        client = _BulkStubClient()
        memory = _BulkStubMemory(client)
        sel = BulkSelection().add(
            _target("a", tags=("collection:Marketing", "other"))
        )
        asyncio.run(
            apply_remove_from_collection(memory, sel, "Marketing")
        )
        patch = client.calls[0]
        assert "collection:Marketing" not in (patch["tags"] or [])
        assert "other" in (patch["tags"] or [])

    def test_rename_collection(self):
        client = _BulkStubClient()
        memory = _BulkStubMemory(client)
        sel = BulkSelection().add(
            _target("a", tags=("collection:Old", "other"))
        )
        asyncio.run(
            apply_rename_collection(
                memory, sel, old_name="Old", new_name="New",
            )
        )
        patch = client.calls[0]
        # Old tag stripped, new tag added.
        assert "collection:Old" not in (patch["tags"] or [])
        assert "collection:New" in (patch["tags"] or [])
        assert "other" in (patch["tags"] or [])

    def test_rename_collection_same_name_raises(self):
        memory = _BulkStubMemory(_BulkStubClient())
        sel = BulkSelection().add(_target("a"))
        with pytest.raises(CollectionError, match="nothing to do"):
            asyncio.run(
                apply_rename_collection(
                    memory, sel, old_name="X", new_name="X",
                )
            )

    def test_rename_collection_same_name_after_strip_raises(self):
        memory = _BulkStubMemory(_BulkStubClient())
        sel = BulkSelection().add(_target("a"))
        with pytest.raises(CollectionError, match="nothing to do"):
            asyncio.run(
                apply_rename_collection(
                    memory, sel, old_name="X", new_name="  X  ",
                )
            )

    def test_delete_collection_alias_of_remove(self):
        client = _BulkStubClient()
        memory = _BulkStubMemory(client)
        sel = BulkSelection().add(
            _target("a", tags=("collection:Doomed", "keep"))
        )
        result = asyncio.run(
            apply_delete_collection(memory, sel, "Doomed")
        )
        assert result.all_succeeded
        patch = client.calls[0]
        assert "collection:Doomed" not in (patch["tags"] or [])
        assert "keep" in (patch["tags"] or [])

    def test_add_invalid_name_raises(self):
        memory = _BulkStubMemory(_BulkStubClient())
        sel = BulkSelection().add(_target("a"))
        with pytest.raises(CollectionError):
            asyncio.run(apply_add_to_collection(memory, sel, ""))

    def test_empty_selection_returns_zero_result(self):
        memory = _BulkStubMemory(_BulkStubClient())
        result = asyncio.run(
            apply_add_to_collection(memory, BulkSelection(), "X")
        )
        assert result.total == 0


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            COLLECTION_PREFIX as PFX,
            Collection as C,
            CollectionError as Err,
            apply_add_to_collection as add,
            apply_remove_from_collection as remove,
            apply_rename_collection as rename,
            apply_delete_collection as delete,
            collection_name_from_tag as name_from,
            collection_tag_for as tag_for,
            extract_collections_from_tags as extract,
            filter_by_collection as filt,
            list_collections as listc,
        )

        assert PFX == COLLECTION_PREFIX
        assert C is Collection
        assert Err is CollectionError
        assert add is apply_add_to_collection
        assert remove is apply_remove_from_collection
        assert rename is apply_rename_collection
        assert delete is apply_delete_collection
        assert name_from is collection_name_from_tag
        assert tag_for is collection_tag_for
        assert extract is extract_collections_from_tags
        assert filt is filter_by_collection
        assert listc is list_collections
