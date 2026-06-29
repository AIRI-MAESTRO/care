"""Library collections data layer (TODO §1.3 P2).

CARE lets users group library entries into named collections via
a reserved tag prefix: ``collection:{name}``. The LibraryScreen's
sidebar surfaces collections above the free-form tag chips, and
right-click affordances on a collection node offer rename /
delete / "filter the table by this collection".

The Textual sidebar + context menu are gated on TODO §1 P0
multi-screen workflow, but the projection + async enumeration +
bulk-tag mutation orchestrators land now.

What this module provides:

* :data:`COLLECTION_PREFIX` — the canonical tag prefix CARE
  reserves for collection membership.
* :class:`Collection` — frozen per-collection projection
  (name + member count + sample entity ids).
* :func:`collection_tag_for` / :func:`collection_name_from_tag`
  — pure tag ↔ name helpers.
* :func:`extract_collections_from_tags` — pure list-of-tags →
  list-of-collection-names.
* :func:`list_collections` — async aggregator that fetches all
  chains via `client.list_chains(...)`, mines their
  ``collection:`` tags, and returns sorted
  :class:`Collection` rows ready for the sidebar.
* :func:`filter_by_collection` — pure projection from a name
  into a :class:`LibraryFilters` tag filter.
* :func:`apply_add_to_collection` / :func:`apply_remove_from_collection`
  / :func:`apply_rename_collection` — async bulk-tag orchestrators
  that piggy-back on the shipped :func:`apply_tag_edits`
  helper from `care.runtime.bulk_ops`.

Duck-typed boundaries: the aggregator accepts any
`CareMemory`-like facade exposing `.client.list_chains(...)`;
the bulk mutators accept a `BulkSelection`. Tests use
lightweight stubs.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional

from care.runtime.bulk_ops import (
    BulkOperationResult,
    BulkSelection,
    apply_tag_edits,
)
from care.runtime.library_view import LibraryFilters


COLLECTION_PREFIX = "collection:"
"""The canonical tag prefix CARE reserves for collection
membership. Every tag starting with this prefix names a
collection; everything else is a plain user tag.

Convention chosen because:

* Memory's tag column is already a JSONB string array — no
  schema migration needed.
* The prefix sorts alphabetically near other namespaced tags
  (e.g. ``domain:weather``), keeping the sidebar tidy.
* Plain string matching means the entire collection feature
  works without any new endpoints.
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CollectionError(RuntimeError):
    """Raised for caller-mistake or unrecoverable enumeration
    failures (empty name, missing SDK method, etc.). Per-target
    HTTP errors from the bulk mutators land on
    :class:`BulkOperationResult.failures` rather than raising —
    same contract as :mod:`care.runtime.bulk_ops`."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Collection:
    """One sidebar node.

    Frozen so a snapshot flows through Textual messages without
    defensive copies. ``member_count`` drives the
    "Marketing (5)" badge; ``sample_entity_ids`` is a small
    list (up to 5) the modal can hover-preview without fetching
    every member.
    """

    name: str
    member_count: int = 0
    sample_entity_ids: tuple[str, ...] = ()

    @property
    def tag(self) -> str:
        """The full tag this collection corresponds to —
        ``"collection:<name>"``."""
        return collection_tag_for(self.name)

    @property
    def is_empty(self) -> bool:
        """``True`` for a collection name the sidebar still
        renders (e.g. just-created via rename-from-empty) but
        with no members yet."""
        return self.member_count == 0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def collection_tag_for(name: str) -> str:
    """Map a collection name to its canonical tag.

    Raises:
        CollectionError: ``name`` is empty / whitespace-only or
            already starts with ``collection:`` (callers
            occasionally double-prefix; we refuse rather than
            silently normalising).
    """
    cleaned = (name or "").strip()
    if not cleaned:
        raise CollectionError("collection name cannot be empty")
    if cleaned.startswith(COLLECTION_PREFIX):
        raise CollectionError(
            f"collection name {name!r} is already prefixed; "
            f"pass the bare name instead"
        )
    return f"{COLLECTION_PREFIX}{cleaned}"


def collection_name_from_tag(tag: str) -> Optional[str]:
    """Inverse of :func:`collection_tag_for`. Returns the bare
    collection name when ``tag`` carries the prefix, else
    ``None``."""
    if not isinstance(tag, str):
        return None
    if not tag.startswith(COLLECTION_PREFIX):
        return None
    name = tag[len(COLLECTION_PREFIX) :].strip()
    return name or None


def extract_collections_from_tags(tags: Iterable[str]) -> tuple[str, ...]:
    """Return every collection name a tag list carries, in
    declaration order, dedupe."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags or ():
        name = collection_name_from_tag(tag)
        if name is None or name in seen:
            continue
        out.append(name)
        seen.add(name)
    return tuple(out)


def is_collection_tag(tag: str) -> bool:
    """``True`` when ``tag`` is shaped like ``collection:<name>``
    (with a non-empty name)."""
    return collection_name_from_tag(tag) is not None


# ---------------------------------------------------------------------------
# Async enumeration
# ---------------------------------------------------------------------------


_DEFAULT_SAMPLE_CAP = 5
"""Maximum number of entity_ids to retain per collection for the
sidebar's hover preview. Keeps the projection bounded even when
a collection has hundreds of members."""


async def list_collections(
    memory: Any,
    *,
    namespace: Optional[str] = None,
    channel: str = "latest",
    sample_cap: int = _DEFAULT_SAMPLE_CAP,
    fetch_limit: int = 200,
    timeout: float = 10.0,
) -> tuple[Collection, ...]:
    """Enumerate every collection visible in the library.

    Asks Memory for every chain in the namespace, mines the
    ``collection:`` tags, and aggregates per-name member counts
    and a small sample of entity ids.

    Args:
        memory: A `CareMemory`-like facade exposing
            ``.client.list_chains(...)``.
        namespace: Optional namespace filter — ``None`` inherits
            the auth scope.
        channel: Memory channel (default ``"latest"``).
        sample_cap: Max number of entity_ids per collection
            kept on the projection.
        fetch_limit: Server-side `list_chains` cap (max 200).
            Future enhancement: paginate when the namespace
            grows past one page; for now CARE namespaces stay
            small enough that one page is the common case.
        timeout: Per-call deadline.

    Returns:
        Tuple of :class:`Collection` rows, sorted by name
        ascending (stable for the sidebar's render order).

    Raises:
        CollectionError: Memory unreachable / timed out / missing
            facade method.
    """
    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    fn = getattr(client, "list_chains", None) if client else None
    if not callable(fn):
        raise CollectionError(
            "memory facade does not expose client.list_chains()"
        )

    capped_limit = max(1, min(fetch_limit, 200))
    start = time.monotonic()
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(
                fn,
                limit=capped_limit,
                channel=channel,
                namespace=namespace,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        latency = (time.monotonic() - start) * 1000
        raise CollectionError(
            f"collection enumeration timed out after {timeout:.1f}s "
            f"({latency:.0f}ms elapsed)"
        ) from exc
    except CollectionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CollectionError(
            f"collection enumeration failed: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(rows, (list, tuple)):
        raise CollectionError(
            f"list_chains returned unexpected type {type(rows).__name__}"
        )

    return _aggregate_collections(rows, sample_cap=sample_cap)


def _aggregate_collections(
    rows: Iterable[Any], *, sample_cap: int,
) -> tuple[Collection, ...]:
    """Pure aggregation step. Walks the row iterable, builds
    per-name counters + sample lists, and returns sorted
    :class:`Collection` rows."""
    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for row in rows:
        entity_id = str(_read(row, "entity_id") or "")
        meta = _read(row, "meta") or {}
        if not isinstance(meta, dict):
            continue
        tags = meta.get("tags") or []
        if not isinstance(tags, (list, tuple)):
            continue
        for name in extract_collections_from_tags(tags):
            counts[name] = counts.get(name, 0) + 1
            bucket = samples.setdefault(name, [])
            if len(bucket) < max(0, sample_cap) and entity_id:
                bucket.append(entity_id)

    sorted_names = sorted(counts.keys())
    return tuple(
        Collection(
            name=name,
            member_count=counts[name],
            sample_entity_ids=tuple(samples.get(name, [])),
        )
        for name in sorted_names
    )


def _read(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# ---------------------------------------------------------------------------
# Filter integration
# ---------------------------------------------------------------------------


def filter_by_collection(
    filters: LibraryFilters,
    name: Optional[str],
) -> LibraryFilters:
    """Return a new :class:`LibraryFilters` pinned to the given
    collection.

    Implementation: appends ``collection:{name}`` to the
    ``tags`` AND-filter (Memory's `list_chains` returns chains
    whose tag set contains every listed token). Passing
    ``name=None`` strips any existing collection tags from the
    filter — the "Show all" sidebar action.

    The free-form tag chips already in the filter are preserved
    so a user can combine "Marketing collection ∩ #urgent".
    """
    new_tags: list[str] = []
    for tag in filters.tags:
        if not is_collection_tag(tag):
            new_tags.append(tag)
    if name:
        new_tags.append(collection_tag_for(name))
    return replace(filters, tags=tuple(new_tags))


def active_collection_name(filters: LibraryFilters) -> Optional[str]:
    """The collection currently pinning the sidebar selection, or
    ``None`` when no collection filter is active. Walks the
    filter's tag list looking for the canonical prefix."""
    for tag in filters.tags:
        name = collection_name_from_tag(tag)
        if name:
            return name
    return None


# ---------------------------------------------------------------------------
# Bulk mutation orchestrators
# ---------------------------------------------------------------------------


async def apply_add_to_collection(
    memory: Any,
    selection: BulkSelection,
    name: str,
    *,
    concurrency: int = 5,
    timeout: float = 10.0,
) -> BulkOperationResult:
    """Add every selected row to the named collection.

    Thin wrapper over :func:`care.runtime.apply_tag_edits`:
    builds the canonical tag via :func:`collection_tag_for`,
    forwards `concurrency` + `timeout` semantics, and returns
    the same :class:`BulkOperationResult` shape the modal
    already renders. Partial failures land on `outcomes` so
    the toast can render "added 3/5 to Marketing".
    """
    tag = collection_tag_for(name)
    return await apply_tag_edits(
        memory,
        selection,
        add_tags=[tag],
        concurrency=concurrency,
        timeout=timeout,
    )


async def apply_remove_from_collection(
    memory: Any,
    selection: BulkSelection,
    name: str,
    *,
    concurrency: int = 5,
    timeout: float = 10.0,
) -> BulkOperationResult:
    """Remove every selected row from the named collection."""
    tag = collection_tag_for(name)
    return await apply_tag_edits(
        memory,
        selection,
        remove_tags=[tag],
        concurrency=concurrency,
        timeout=timeout,
    )


async def apply_rename_collection(
    memory: Any,
    selection: BulkSelection,
    *,
    old_name: str,
    new_name: str,
    concurrency: int = 5,
    timeout: float = 10.0,
) -> BulkOperationResult:
    """Rename a collection across every member in ``selection``.

    Sidebar workflow: collection rename pulls every member of
    the collection into a `BulkSelection` upstream (via
    `list_collections().sample_entity_ids` for the small case,
    or a dedicated fetch for the large case), then calls this
    helper. The atomic per-row PATCH replaces
    ``collection:<old>`` with ``collection:<new>``.

    Raises:
        CollectionError: ``old_name == new_name`` (no-op caller
            misuse) or either name is invalid (delegated to
            :func:`collection_tag_for`).
    """
    if old_name.strip() == new_name.strip():
        raise CollectionError(
            "rename: new_name matches old_name; nothing to do"
        )
    old_tag = collection_tag_for(old_name)
    new_tag = collection_tag_for(new_name)
    return await apply_tag_edits(
        memory,
        selection,
        add_tags=[new_tag],
        remove_tags=[old_tag],
        concurrency=concurrency,
        timeout=timeout,
    )


async def apply_delete_collection(
    memory: Any,
    selection: BulkSelection,
    name: str,
    *,
    concurrency: int = 5,
    timeout: float = 10.0,
) -> BulkOperationResult:
    """Delete a collection (remove the ``collection:`` tag from
    every member). The library entries themselves stay — only
    the grouping disappears.

    Equivalent to :func:`apply_remove_from_collection`; exposed
    separately so the sidebar's "Delete collection" affordance
    has a self-documenting call site (matches the verb the
    user sees)."""
    return await apply_remove_from_collection(
        memory, selection, name, concurrency=concurrency, timeout=timeout,
    )


# Re-export the unused ``field`` import so future dataclass
# extensions (e.g. ``Collection`` gaining a `tags` field) don't
# need a separate import. Keeps the public namespace tidy.
_ = field


__all__ = [
    "COLLECTION_PREFIX",
    "Collection",
    "CollectionError",
    "active_collection_name",
    "apply_add_to_collection",
    "apply_delete_collection",
    "apply_remove_from_collection",
    "apply_rename_collection",
    "collection_name_from_tag",
    "collection_tag_for",
    "extract_collections_from_tags",
    "filter_by_collection",
    "is_collection_tag",
    "list_collections",
]
