"""Bulk library-operations data layer (TODO §1.3 P1).

The LibraryScreen's bulk-action affordance lets the user
multi-select rows with `Space`, then press `F` to favourite,
`T` to tag-edit, or `Del` to delete. The Textual key bindings +
tag-editor modal are gated on TODO §1 P0 multi-screen workflow,
but the selection-state model + concurrent SDK mutation drivers
+ per-target result aggregation ship now as the data layer.

What this module provides:

* :class:`BulkTarget` — frozen row-snapshot the modal hands in
  per selected entity (id + type + currently-applied tags).
* :class:`BulkSelection` — frozen ordered set of targets with
  membership predicates + functional mutators (`add`, `remove`,
  `toggle`, `clear`).
* :class:`BulkOperationOutcome` — per-target frozen result row.
* :class:`BulkOperationResult` — aggregate with predicates the
  modal renders into a toast (`"3/5 updated"`).
* :func:`merge_tags` — pure helper that resolves
  ``(existing ∪ add) \\ remove`` with stable order + dedup.
* :func:`apply_favourite` — bulk favourite / unfavourite.
* :func:`apply_tag_edits` — bulk tag mutation.
* :func:`apply_delete` — bulk soft-delete.

Concurrency: each helper fans out via :func:`asyncio.gather`
with a bounded semaphore so 50 selected rows fire at most N
in-flight requests (default 5). Per-target timeouts mean one
stuck server doesn't freeze the whole batch.

Duck-typed boundaries: the async helpers reach into
``memory.client`` for the SDK's typed mutation methods
(`_mark_favourite`, `_update_metadata`, `_delete_entity` — the
base helpers that every per-type mixin delegates to). Tests
inject stub clients.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional


EntityType = Literal["chain", "agent", "agent_skill"]
"""Library entity kinds bulk operations work on. ``memory_card``
is excluded — run-records aren't user-mutable in the library."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BulkOperationError(RuntimeError):
    """Raised for caller-mistake failures — empty selection where
    the operation requires one, malformed target, etc. Per-target
    HTTP failures don't raise; they land on
    :class:`BulkOperationOutcome.error` so the modal can render
    a per-row badge."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BulkTarget:
    """One row the user selected.

    Carries the metadata the bulk operations need:

    * ``entity_id`` / ``entity_type`` — routing to the right
      SDK endpoint.
    * ``current_tags`` — pre-loaded from the LibraryScreen's
      DataTable so the tag-edit helper doesn't have to re-fetch
      every row. Empty when unknown (the helper falls back to
      a per-target GET in that case — see :func:`apply_tag_edits`).
    * ``display_name`` — purely for outcome messages; the
      modal can render "Updated 'Weather report'" instead of
      "Updated 7f3a-…".
    """

    entity_id: str
    entity_type: EntityType
    current_tags: tuple[str, ...] = ()
    display_name: Optional[str] = None


@dataclass(frozen=True)
class BulkSelection:
    """Ordered, dedup'd set of :class:`BulkTarget`.

    Frozen — every mutator returns a new selection so the modal's
    undo stack is a list of past instances.
    """

    targets: tuple[BulkTarget, ...] = ()

    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self):
        return iter(self.targets)

    def __contains__(self, entity_id: object) -> bool:
        if not isinstance(entity_id, str):
            return False
        return any(t.entity_id == entity_id for t in self.targets)

    @property
    def is_empty(self) -> bool:
        return not self.targets

    @property
    def entity_ids(self) -> tuple[str, ...]:
        return tuple(t.entity_id for t in self.targets)

    def find(self, entity_id: str) -> Optional[BulkTarget]:
        for t in self.targets:
            if t.entity_id == entity_id:
                return t
        return None

    # ---- mutators ----

    def add(self, target: BulkTarget) -> "BulkSelection":
        """Return a new selection with ``target`` appended.
        No-op when ``target.entity_id`` is already present."""
        if target.entity_id in self:
            return self
        return BulkSelection(targets=self.targets + (target,))

    def remove(self, entity_id: str) -> "BulkSelection":
        """Return a new selection without ``entity_id``. No-op
        when not selected."""
        if entity_id not in self:
            return self
        return BulkSelection(
            targets=tuple(t for t in self.targets if t.entity_id != entity_id)
        )

    def toggle(self, target: BulkTarget) -> "BulkSelection":
        """Convenience: add when absent, remove when present.
        Mirrors `Space`-bar behaviour."""
        if target.entity_id in self:
            return self.remove(target.entity_id)
        return self.add(target)

    def clear(self) -> "BulkSelection":
        return BulkSelection()


@dataclass(frozen=True)
class BulkOperationOutcome:
    """Result for one target inside a batch."""

    target: BulkTarget
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def entity_id(self) -> str:
        return self.target.entity_id


@dataclass(frozen=True)
class BulkOperationResult:
    """Aggregate over a batch.

    Used by the modal to render the post-action toast:
    "Updated 3 / 5 (2 failed)".
    """

    outcomes: tuple[BulkOperationOutcome, ...] = ()
    operation: str = ""

    def __len__(self) -> int:
        return len(self.outcomes)

    def __iter__(self):
        return iter(self.outcomes)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def succeeded(self) -> int:
        return sum(1 for o in self.outcomes if o.success)

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if not o.success)

    @property
    def all_succeeded(self) -> bool:
        return all(o.success for o in self.outcomes) and bool(self.outcomes)

    @property
    def any_failed(self) -> bool:
        return any(not o.success for o in self.outcomes)

    @property
    def failures(self) -> tuple[BulkOperationOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.success)

    def format_summary(self) -> str:
        """One-line summary the modal pipes into a toast."""
        op = self.operation or "operation"
        if self.total == 0:
            return f"{op}: nothing to do"
        if self.all_succeeded:
            return f"{op}: {self.succeeded}/{self.total} ok"
        return f"{op}: {self.succeeded}/{self.total} ok, {self.failed} failed"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def merge_tags(
    current: Iterable[str],
    *,
    add: Iterable[str] = (),
    remove: Iterable[str] = (),
) -> list[str]:
    """Compute ``(current ∪ add) \\ remove`` with stable order +
    dedup.

    Stable order rule: tags appear in the result in the order
    they were first seen across ``current`` then ``add``. Removed
    tags drop entirely (case-sensitive match — Memory's tag set
    is case-sensitive).

    Whitespace-only entries in ``add`` are skipped so the modal's
    free-form chip input doesn't write empty tags.
    """
    remove_set = {t for t in remove if t}
    result: list[str] = []
    seen: set[str] = set()

    def _push(tag: str) -> None:
        clean = tag.strip()
        if not clean or clean in remove_set or clean in seen:
            return
        result.append(clean)
        seen.add(clean)

    for tag in current:
        _push(tag)
    for tag in add:
        _push(tag)
    return result


# ---------------------------------------------------------------------------
# Async drivers
# ---------------------------------------------------------------------------


_DEFAULT_CONCURRENCY = 5
_DEFAULT_TIMEOUT = 10.0


async def apply_favourite(
    memory: Any,
    selection: BulkSelection,
    *,
    favourite: bool = True,
    concurrency: int = _DEFAULT_CONCURRENCY,
    timeout: float = _DEFAULT_TIMEOUT,
) -> BulkOperationResult:
    """Bulk-set the favourite flag.

    Returns a :class:`BulkOperationResult` with one outcome per
    target — per-target HTTP failures don't propagate so the
    modal can render partial success without a single bad row
    cratering the batch.
    """
    op_name = "favourite" if favourite else "unfavourite"
    if selection.is_empty:
        return BulkOperationResult(outcomes=(), operation=op_name)

    client = _resolve_client(memory)
    fn = getattr(client, "_mark_favourite", None)
    if fn is None:
        raise BulkOperationError(
            "memory facade does not expose client._mark_favourite()"
        )

    async def _do(target: BulkTarget) -> BulkOperationOutcome:
        return await _run_one(
            target=target,
            timeout=timeout,
            call=lambda: fn(target.entity_type, target.entity_id, value=favourite),
        )

    outcomes = await _gather_bounded(_do, selection.targets, concurrency)
    return BulkOperationResult(outcomes=outcomes, operation=op_name)


async def apply_tag_edits(
    memory: Any,
    selection: BulkSelection,
    *,
    add_tags: Iterable[str] = (),
    remove_tags: Iterable[str] = (),
    concurrency: int = _DEFAULT_CONCURRENCY,
    timeout: float = _DEFAULT_TIMEOUT,
) -> BulkOperationResult:
    """Bulk add / remove tags across the selection.

    For each target, the new tag set is
    ``merge_tags(target.current_tags, add=add_tags, remove=remove_tags)``.
    When ``target.current_tags`` is empty AND ``remove_tags`` is
    non-empty, the helper falls back to a per-target GET via
    ``client._get_entity`` so the remove operation doesn't
    silently no-op against a freshly-loaded row.

    No-ops when neither ``add_tags`` nor ``remove_tags`` carries
    a non-empty value — returns an empty result with the
    descriptive operation name.
    """
    add_clean = [t.strip() for t in add_tags if t and t.strip()]
    remove_clean = [t.strip() for t in remove_tags if t and t.strip()]
    op_name = _tag_op_name(add_clean, remove_clean)

    if not add_clean and not remove_clean:
        return BulkOperationResult(outcomes=(), operation=op_name)
    if selection.is_empty:
        return BulkOperationResult(outcomes=(), operation=op_name)

    client = _resolve_client(memory)
    patch_fn = getattr(client, "_update_metadata", None)
    if patch_fn is None:
        raise BulkOperationError(
            "memory facade does not expose client._update_metadata()"
        )
    get_fn = getattr(client, "_get_entity", None)

    async def _do(target: BulkTarget) -> BulkOperationOutcome:
        async def _call() -> Any:
            current = target.current_tags
            if not current and remove_clean and get_fn is not None:
                # Read the entity so the remove pass has something to bite on.
                raw = await asyncio.to_thread(
                    get_fn, target.entity_type, target.entity_id
                )
                current = tuple(_extract_tags(raw))
            new_tags = merge_tags(current, add=add_clean, remove=remove_clean)
            return await asyncio.to_thread(
                patch_fn,
                target.entity_type,
                target.entity_id,
                tags=new_tags,
            )

        return await _run_one_async(
            target=target, timeout=timeout, call=_call
        )

    outcomes = await _gather_bounded(_do, selection.targets, concurrency)
    return BulkOperationResult(outcomes=outcomes, operation=op_name)


async def apply_delete(
    memory: Any,
    selection: BulkSelection,
    *,
    concurrency: int = _DEFAULT_CONCURRENCY,
    timeout: float = _DEFAULT_TIMEOUT,
) -> BulkOperationResult:
    """Bulk soft-delete.

    Memory's delete is a soft delete (sets ``deleted_at``) so the
    row can be restored from the trash — this helper doesn't try
    to confirm; the modal's "Are you sure?" guard runs upstream.
    """
    op_name = "delete"
    if selection.is_empty:
        return BulkOperationResult(outcomes=(), operation=op_name)

    client = _resolve_client(memory)
    fn = getattr(client, "_delete_entity", None)
    if fn is None:
        raise BulkOperationError(
            "memory facade does not expose client._delete_entity()"
        )

    async def _do(target: BulkTarget) -> BulkOperationOutcome:
        return await _run_one(
            target=target,
            timeout=timeout,
            call=lambda: fn(target.entity_type, target.entity_id),
        )

    outcomes = await _gather_bounded(_do, selection.targets, concurrency)
    return BulkOperationResult(outcomes=outcomes, operation=op_name)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_client(memory: Any) -> Any:
    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    if client is None:
        raise BulkOperationError(
            "memory facade does not expose a `.client` attribute"
        )
    return client


async def _run_one(
    *,
    target: BulkTarget,
    timeout: float,
    call,
) -> BulkOperationOutcome:
    """Run a synchronous SDK call with a per-target deadline."""
    start = time.monotonic()
    try:
        await asyncio.wait_for(asyncio.to_thread(call), timeout=timeout)
    except asyncio.TimeoutError:
        return BulkOperationOutcome(
            target=target,
            success=False,
            error=f"timed out after {timeout:.1f}s",
            duration_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as exc:  # noqa: BLE001
        return BulkOperationOutcome(
            target=target,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=(time.monotonic() - start) * 1000,
        )
    return BulkOperationOutcome(
        target=target,
        success=True,
        duration_ms=(time.monotonic() - start) * 1000,
    )


async def _run_one_async(
    *,
    target: BulkTarget,
    timeout: float,
    call,
) -> BulkOperationOutcome:
    """Same shape as :func:`_run_one` but the call is an
    ``async def`` (used when the operation chains multiple SDK
    calls, e.g. GET-then-PATCH)."""
    start = time.monotonic()
    try:
        await asyncio.wait_for(call(), timeout=timeout)
    except asyncio.TimeoutError:
        return BulkOperationOutcome(
            target=target,
            success=False,
            error=f"timed out after {timeout:.1f}s",
            duration_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as exc:  # noqa: BLE001
        return BulkOperationOutcome(
            target=target,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=(time.monotonic() - start) * 1000,
        )
    return BulkOperationOutcome(
        target=target,
        success=True,
        duration_ms=(time.monotonic() - start) * 1000,
    )


async def _gather_bounded(
    fn,
    targets: tuple[BulkTarget, ...],
    concurrency: int,
) -> tuple[BulkOperationOutcome, ...]:
    """Fan out ``fn(target)`` with a bounded semaphore. Output
    order matches input order so the modal can render outcomes
    next to the table rows."""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _wrapped(t: BulkTarget) -> BulkOperationOutcome:
        async with sem:
            return await fn(t)

    return tuple(await asyncio.gather(*(_wrapped(t) for t in targets)))


def _extract_tags(raw: Any) -> list[str]:
    """Read ``meta.tags`` off a `_get_entity` response (dict or
    `EntityResponse`-shaped object). Defensive: missing / non-list
    values collapse to an empty list rather than raising."""
    if isinstance(raw, dict):
        meta = raw.get("meta") or {}
    else:
        meta = getattr(raw, "meta", {}) or {}
    if not isinstance(meta, dict):
        return []
    tags = meta.get("tags") or []
    if not isinstance(tags, list):
        return []
    return [str(t) for t in tags if isinstance(t, str)]


def _tag_op_name(add: list[str], remove: list[str]) -> str:
    parts: list[str] = []
    if add:
        parts.append(f"+{len(add)} tag" if len(add) == 1 else f"+{len(add)} tags")
    if remove:
        parts.append(
            f"-{len(remove)} tag" if len(remove) == 1 else f"-{len(remove)} tags"
        )
    if not parts:
        return "tag edits"
    return f"tag {', '.join(parts)}"


__all__ = [
    "BulkOperationError",
    "BulkOperationOutcome",
    "BulkOperationResult",
    "BulkSelection",
    "BulkTarget",
    "EntityType",
    "apply_delete",
    "apply_favourite",
    "apply_tag_edits",
    "merge_tags",
]


