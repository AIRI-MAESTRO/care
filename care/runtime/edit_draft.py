"""Edit-agent-screen data layer (TODO §3 P1).

``EditAgentScreen`` lets the user edit a saved agent's editable
metadata (name, description, tags, task description) plus per-step
prompts / configs, and submit the result as a NEW version of the
same ``entity_id``. Memory's version semantics preserve the prior
version for rollback; channel ``latest`` always points at the most
recent edit, and ``stable`` is set manually via a "Promote to
stable" action.

The Textual screen itself is gated on TODO §1 P0 multi-screen
workflow, but the form-binding model + dirty tracking + validation
+ projection back into the SDK ship now as the data layer.

What this module provides:

* :class:`EditAgentDraft` — frozen form state with original-vs-
  edited snapshots for dirty-tracking. `is_dirty()` / `dirty_fields()`
  drive the "Save" / "Save (n changes)" button label.
* :class:`EditDraftIssue` — frozen inline validation finding the
  screen renders next to the offending field.
* :func:`extract_edit_draft(chain, entity_id, *, entity_type)` —
  pure projection from a saved CARL ``ReasoningChain`` (or any
  duck-typed object exposing CARE metadata) into the initial
  draft.
* Mutators — `set_display_name`, `set_description`, `set_tags`,
  `set_task_description`, `set_change_summary`, `update_chain` —
  every returns a new frozen draft.
* :func:`validate_edit_draft` — surfaces field-level issues
  (empty name, blank change-summary on a structural edit, etc.).
* :func:`save_edit_as_new_version` — async wrapper around the
  CARE / SDK ``save_chain`` path so the screen has one call site.
  Memory keys off ``entity_id`` so a fresh version lands without
  losing the prior one.
* :func:`promote_to_stable` — async wrapper around the SDK's
  ``promote`` endpoint.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Optional


EditFieldName = Literal[
    "display_name",
    "description",
    "tags",
    "task_description",
    "chain_content",
]
"""Editable fields the screen renders + the draft tracks for
dirty state. ``chain_content`` is the catch-all for per-step
prompt / config edits — the screen mutates a `ReasoningChain`
in-place and stamps the draft via :func:`update_chain`."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EditDraftError(RuntimeError):
    """Raised for unrecoverable edit-flow failures — missing
    entity_id, save-time SDK error, promote on non-saved agent.
    Per-field validation issues land on
    :class:`EditDraftIssue` rather than this exception."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EditDraftIssue:
    """One inline validation finding the screen renders."""

    severity: Literal["error", "warning"]
    field: EditFieldName
    message: str
    detail: Optional[str] = None


@dataclass(frozen=True)
class EditAgentDraft:
    """Frozen form state for the EditAgentScreen.

    Carries both the user-editable values AND the original
    snapshots so :meth:`is_dirty` works without re-reading
    Memory. Every mutator returns a new instance via
    :func:`dataclasses.replace` so the screen's undo stack is
    trivial (a list of past drafts).
    """

    entity_id: str
    entity_type: Literal["chain", "agent", "agent_skill"] = "chain"
    parent_version_id: Optional[str] = None
    channel: str = "latest"

    # Editable fields + original snapshots
    display_name: str = ""
    original_display_name: str = ""
    description: str = ""
    original_description: str = ""
    tags: tuple[str, ...] = ()
    original_tags: tuple[str, ...] = ()
    task_description: str = ""
    original_task_description: str = ""

    # Catch-all carrier for per-step prompt / config edits — opaque
    # to this layer; the modal mutates a CARL `ReasoningChain` (or a
    # content dict) and stamps it here. ``chain_content_dirty``
    # tracks whether it was touched so dirty-tracking doesn't have
    # to deep-compare nested step dicts.
    chain_content: Optional[Any] = None
    chain_content_dirty: bool = False

    # Author-supplied note saved alongside the new version.
    change_summary: str = ""

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    @property
    def display_name_dirty(self) -> bool:
        return self.display_name.strip() != self.original_display_name.strip()

    @property
    def description_dirty(self) -> bool:
        return self.description.strip() != self.original_description.strip()

    @property
    def tags_dirty(self) -> bool:
        # Compare as sets so re-ordering chips in the modal doesn't
        # flag a dirty state when the underlying tag set is the
        # same.
        return set(self.tags) != set(self.original_tags)

    @property
    def task_description_dirty(self) -> bool:
        return (
            self.task_description.strip()
            != self.original_task_description.strip()
        )

    def is_dirty(self) -> bool:
        """``True`` when any tracked field changed vs. the saved
        version. Drives the "Save" button label."""
        return any(
            (
                self.display_name_dirty,
                self.description_dirty,
                self.tags_dirty,
                self.task_description_dirty,
                self.chain_content_dirty,
            )
        )

    def dirty_fields(self) -> tuple[EditFieldName, ...]:
        """Tuple of dirty field names — used by the screen to
        render "(n changes)" badges + by
        :func:`save_edit_as_new_version` to skip writes that
        wouldn't change anything."""
        dirty: list[EditFieldName] = []
        if self.display_name_dirty:
            dirty.append("display_name")
        if self.description_dirty:
            dirty.append("description")
        if self.tags_dirty:
            dirty.append("tags")
        if self.task_description_dirty:
            dirty.append("task_description")
        if self.chain_content_dirty:
            dirty.append("chain_content")
        return tuple(dirty)

    @property
    def is_structural_edit(self) -> bool:
        """``True`` when the user touched chain content
        (prompts / dependencies / step configs). Drives the
        "change summary required" rule — pure metadata renames
        don't require a summary, but structural edits do."""
        return self.chain_content_dirty


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def extract_edit_draft(
    chain: Any,
    entity_id: str,
    *,
    entity_type: Literal["chain", "agent", "agent_skill"] = "chain",
    parent_version_id: Optional[str] = None,
    channel: str = "latest",
) -> EditAgentDraft:
    """Project a saved chain (CARL `ReasoningChain`, dict, or
    duck-typed object) into the initial draft state.

    Reads ``display_name`` / ``description`` / ``tags`` /
    ``task_description`` from the chain's CARE metadata block
    (set by :class:`care.memory.CareMemory.save_chain`). Missing
    fields default to empty strings / tuples — the modal renders
    placeholder hints next to blank fields.

    Args:
        chain: Saved chain object.
        entity_id: Memory entity id (load-bearing — without it,
            :func:`save_edit_as_new_version` can't write the new
            version under the same id).
        entity_type: One of ``"chain" | "agent" | "agent_skill"``.
        parent_version_id: Optional; when supplied, the saved
            version chains its `parent_version_id` to this — gives
            CARE explicit lineage when the user is editing a
            non-tip version.
        channel: Target channel for the save (default ``"latest"``).

    Returns:
        Initial :class:`EditAgentDraft` with ``original_*`` fields
        mirroring the editable ones.
    """
    if not entity_id:
        raise EditDraftError("entity_id is required to build an edit draft")

    metadata = _extract_care_metadata(chain)

    display_name = _read_str(metadata, "display_name")
    description = _read_str(metadata, "description")
    tags_raw = metadata.get("tags") if isinstance(metadata, dict) else None
    tags: tuple[str, ...] = (
        tuple(str(t) for t in tags_raw)
        if isinstance(tags_raw, (list, tuple))
        else ()
    )
    task_description = _read_str(metadata, "task_description")

    return EditAgentDraft(
        entity_id=entity_id,
        entity_type=entity_type,
        parent_version_id=parent_version_id,
        channel=channel,
        display_name=display_name,
        original_display_name=display_name,
        description=description,
        original_description=description,
        tags=tags,
        original_tags=tags,
        task_description=task_description,
        original_task_description=task_description,
        chain_content=chain,
        chain_content_dirty=False,
    )


def _extract_care_metadata(chain: Any) -> dict[str, Any]:
    """Read the CARE metadata block whichever way the chain
    exposes it."""
    getter = getattr(chain, "get_care_metadata", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # noqa: BLE001
            value = None
        if value is None:
            return {}
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(exclude_none=False)
            except TypeError:
                dumped = model_dump()
            if isinstance(dumped, dict):
                return dict(dumped)
        if isinstance(value, dict):
            return dict(value)
    raw = _read(chain, "metadata")
    if isinstance(raw, dict):
        care = raw.get("care") or raw.get("metadata") or raw
        if isinstance(care, dict):
            return dict(care)
    return {}


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


def set_display_name(draft: EditAgentDraft, value: str) -> EditAgentDraft:
    return replace(draft, display_name=value)


def set_description(draft: EditAgentDraft, value: str) -> EditAgentDraft:
    return replace(draft, description=value)


def set_tags(draft: EditAgentDraft, value: Iterable[str]) -> EditAgentDraft:
    cleaned = tuple(t.strip() for t in value if t and t.strip())
    return replace(draft, tags=cleaned)


def set_task_description(draft: EditAgentDraft, value: str) -> EditAgentDraft:
    return replace(draft, task_description=value)


def set_change_summary(draft: EditAgentDraft, value: str) -> EditAgentDraft:
    return replace(draft, change_summary=value)


def update_chain(
    draft: EditAgentDraft,
    new_chain: Any,
    *,
    dirty: bool = True,
) -> EditAgentDraft:
    """Replace the draft's ``chain_content`` with a mutated
    chain. ``dirty=True`` (default) flags the structural-edit
    state so the screen knows to require a change summary. Pass
    ``dirty=False`` for refresh-from-disk operations that
    shouldn't count as user edits."""
    return replace(draft, chain_content=new_chain, chain_content_dirty=dirty)


def reset(draft: EditAgentDraft) -> EditAgentDraft:
    """Discard every edit and return the draft to its post-extract
    state. Used by the screen's "Reset changes" affordance."""
    return replace(
        draft,
        display_name=draft.original_display_name,
        description=draft.original_description,
        tags=draft.original_tags,
        task_description=draft.original_task_description,
        chain_content_dirty=False,
        change_summary="",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_edit_draft(draft: EditAgentDraft) -> tuple[EditDraftIssue, ...]:
    """Check the draft and return any issues the screen should
    render inline. Empty tuple when the draft is ready to save."""
    issues: list[EditDraftIssue] = []

    if not draft.display_name.strip():
        issues.append(
            EditDraftIssue(
                severity="error",
                field="display_name",
                message="Display name is required",
            )
        )

    if draft.is_structural_edit and not draft.change_summary.strip():
        # CARE convention: pure metadata edits (rename / re-tag)
        # don't need a summary; structural edits (changing step
        # prompts / configs) do, so the version history reads
        # well when someone walks the lineage.
        issues.append(
            EditDraftIssue(
                severity="error",
                field="chain_content",
                message="Change summary is required when editing chain content",
            )
        )

    # Duplicate-tag detection — set semantics on the draft means
    # the modal might let users type the same chip twice; flag
    # it so they can clean up before save.
    seen: set[str] = set()
    duplicates: list[str] = []
    for tag in draft.tags:
        if tag in seen and tag not in duplicates:
            duplicates.append(tag)
        seen.add(tag)
    if duplicates:
        issues.append(
            EditDraftIssue(
                severity="warning",
                field="tags",
                message=f"Duplicate tags ignored on save: {', '.join(duplicates)}",
            )
        )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Save / promote drivers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SaveEditResult:
    """Outcome of :func:`save_edit_as_new_version`."""

    entity_id: str
    success: bool = True
    error: Optional[str] = None
    fields_written: tuple[EditFieldName, ...] = ()


@dataclass(frozen=True)
class PromoteResult:
    """Outcome of :func:`promote_to_stable`."""

    entity_id: str
    from_channel: str
    to_channel: str
    success: bool = True
    error: Optional[str] = None
    response: dict[str, Any] = field(default_factory=dict)


async def save_edit_as_new_version(
    memory: Any,
    draft: EditAgentDraft,
    *,
    author: Optional[str] = None,
    timeout: float = 30.0,
) -> SaveEditResult:
    """Write the draft as a new version of ``draft.entity_id``.

    No-op when ``draft.is_dirty() is False`` — returns a
    :class:`SaveEditResult` with ``fields_written=()`` so the
    screen can render "Nothing to save". Otherwise wraps the
    SDK's sync ``save_chain`` call (which creates a new version
    when ``entity_id`` is supplied) in :func:`asyncio.to_thread`
    with a deadline.

    Args:
        memory: A `CareMemory` facade (or any object with
            ``.save_chain(...)``).
        draft: Current form state.
        author: Optional author tag forwarded to the SDK.
        timeout: Per-call deadline in seconds.

    Returns:
        :class:`SaveEditResult`. Per-field write failures land
        on ``error`` so the screen renders a toast — the function
        never raises for HTTP errors. Pre-flight errors (missing
        entity_id, missing facade method) DO raise
        :class:`EditDraftError`.
    """
    if not draft.entity_id:
        raise EditDraftError("draft has no entity_id; nothing to save")

    if not draft.is_dirty():
        return SaveEditResult(
            entity_id=draft.entity_id,
            success=True,
            fields_written=(),
        )

    save_fn = getattr(memory, "save_chain", None)
    if save_fn is None:
        raise EditDraftError(
            "memory facade does not expose save_chain(...)"
        )

    fields = draft.dirty_fields()
    chain_payload = draft.chain_content if draft.chain_content is not None else {}

    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                save_fn,
                chain_payload,
                name=draft.display_name,
                query=draft.task_description or None,
                tags=list(draft.tags) or None,
                author=author,
                entity_id=draft.entity_id,
                channel=draft.channel,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return SaveEditResult(
            entity_id=draft.entity_id,
            success=False,
            error=f"save timed out after {timeout:.1f}s",
            fields_written=(),
        )
    except Exception as exc:  # noqa: BLE001
        return SaveEditResult(
            entity_id=draft.entity_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            fields_written=(),
        )

    return SaveEditResult(
        entity_id=draft.entity_id,
        success=True,
        fields_written=fields,
    )


async def promote_to_stable(
    memory: Any,
    draft: EditAgentDraft,
    *,
    from_channel: Optional[str] = None,
    to_channel: str = "stable",
    timeout: float = 10.0,
) -> PromoteResult:
    """Copy the draft's saved channel pointer to ``stable``.

    The SDK's ``promote`` endpoint flips a channel pointer
    server-side WITHOUT creating a new version; the user is
    saying "the version currently pinned to ``latest`` is the
    one I want to be the stable one". Idempotent.

    Args:
        memory: A `CareMemory` facade (or any object with
            ``.client.promote(...)``).
        draft: Current form state — the helper uses
            ``draft.entity_id`` + ``draft.entity_type``.
        from_channel: Source channel (default: the draft's
            current channel, typically ``"latest"``).
        to_channel: Target channel (default ``"stable"``).
        timeout: Per-call deadline in seconds.

    Returns:
        :class:`PromoteResult`. Per-call failures land on
        ``error``; pre-flight errors raise
        :class:`EditDraftError`.
    """
    if not draft.entity_id:
        raise EditDraftError(
            "draft has no entity_id; nothing to promote"
        )

    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    if client is None or not hasattr(client, "promote"):
        raise EditDraftError(
            "memory facade does not expose client.promote()"
        )

    src = from_channel or draft.channel

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.promote,
                draft.entity_id,
                from_channel=src,
                to_channel=to_channel,
                entity_type=draft.entity_type,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return PromoteResult(
            entity_id=draft.entity_id,
            from_channel=src,
            to_channel=to_channel,
            success=False,
            error=f"promote timed out after {timeout:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        return PromoteResult(
            entity_id=draft.entity_id,
            from_channel=src,
            to_channel=to_channel,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    return PromoteResult(
        entity_id=draft.entity_id,
        from_channel=src,
        to_channel=to_channel,
        success=True,
        response=response if isinstance(response, dict) else {"raw": response},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _read_str(obj: Any, name: str) -> str:
    value = _read(obj, name)
    return value if isinstance(value, str) else ""


__all__ = [
    "EditAgentDraft",
    "EditDraftError",
    "EditDraftIssue",
    "EditFieldName",
    "PromoteResult",
    "SaveEditResult",
    "extract_edit_draft",
    "promote_to_stable",
    "reset",
    "save_edit_as_new_version",
    "set_change_summary",
    "set_description",
    "set_display_name",
    "set_tags",
    "set_task_description",
    "update_chain",
    "validate_edit_draft",
]
