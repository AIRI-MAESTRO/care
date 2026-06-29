"""SaveAgentModal data layer (TODO §3 P0).

The SaveAgentModal opens at the end of every successful MAGE
generation. It prompts the user to confirm/edit:

* ``name`` (pre-filled with MAGE's suggested display name, falling
  back to ``domain`` + first 60 chars of the query; must be
  non-empty and unique in the user's namespace).
* ``description`` (pre-filled with MAGE's suggested description,
  falling back to the original query verbatim).
* ``tags`` (multi-select against existing tags + free-form, plus
  a "Favourite ⭐" toggle that adds the `favourite` tag).
* ``keep_context`` checkbox — when on, the task description +
  the list of context file paths used at generation time are
  saved alongside the agent so "Re-run" can reuse them.
* Buttons: ``Save & Inspect``, ``Save & Run``, ``Discard``.

The Textual modal itself is gated on TODO §1 P0 multi-screen
workflow. This module ships the form-state model + validation +
the promotion path on top of the pre-shipped
:func:`care.runtime.promote_draft`.

What this module provides:

* :data:`FAVOURITE_TAG` — constant tag name; toggling the
  favourite checkbox just adds/removes this tag from
  :attr:`SaveAgentForm.tags`.
* :class:`SaveAgentForm` — frozen form state with
  ``original_*`` snapshots for dirty tracking.
* :class:`SaveAgentIssue` — frozen inline validation finding.
* :class:`SaveAgentOutcome` — frozen result the modal renders
  into a toast.
* :func:`seed_save_agent_form` — pure projection that builds
  the initial form state from a generation's query +
  MAGE metadata + context files.
* Mutators (``set_display_name``, ``set_description``,
  ``set_tags``, ``add_tag``, ``remove_tag``, ``toggle_favourite``,
  ``set_keep_context``) returning new frozen drafts.
* :func:`validate_save_agent_form` — sync + async-aware
  validator. Returns inline issues; uniqueness check can be
  skipped when offline (``check_unique=False``).
* :func:`apply_save_agent_form` — async helper that promotes
  the auto-saved draft (via the shipped
  :func:`care.runtime.promote_draft`) and applies the
  user-finalised metadata via
  ``memory.client._update_metadata(...)``.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Optional


FAVOURITE_TAG = "favourite"
"""The canonical tag name CARE stamps for the favourite flag.

The modal's "Favourite ⭐" checkbox is just sugar around adding /
removing this tag from :attr:`SaveAgentForm.tags`. Memory's
``favourite`` boolean column is bumped separately via
:func:`apply_save_agent_form` when the tag is present at promote
time.
"""


_QUERY_TRUNCATE = 60
"""Maximum number of query characters used in the fallback
display-name heuristic. Matches the TODO §3 P0 spec."""


# CARE wraps the task before generation — `_with_today_preamble` prepends a
# "(Today is …)" date block and `_prepend_user_context` adds a "--- TASK:"
# separator. MAGE derives its suggested name from the leading chars of that
# wrapped prompt, so the wrapper can leak into the name (e.g.
# "Finance — --- TASK: (Today is 2026-…"). These strip it back out.
_NAME_JUNK_RES = (
    # Full date preamble ends at "current date/time.)"; a truncated name may
    # cut it off, so also match an unterminated "(Today is …" to end-of-string.
    re.compile(
        r"\(Today is\b.*?(?:current date/time\.\)|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"-{2,}\s*TASK:?", re.IGNORECASE),
    re.compile(r"^\s*TASK:\s*", re.IGNORECASE),
)
_NAME_STRIP_CHARS = " —–-:·|\t"


def sanitize_chain_name(name: str, *, max_len: int = 80) -> str:
    """Strip the generation-prompt preamble MAGE can leak into a suggested
    chain name (the ``(Today is …)`` date block + the ``--- TASK:`` user-context
    separator), collapse whitespace, and cap the length. Returns ``""`` when
    nothing meaningful survives so callers fall back to a query-derived name."""
    cleaned = name or ""
    for rx in _NAME_JUNK_RES:
        cleaned = rx.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(_NAME_STRIP_CHARS).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip(_NAME_STRIP_CHARS) + "…"
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SaveAgentError(RuntimeError):
    """Raised for save-flow failures the modal can't handle
    inline — missing draft session, network outage during
    promote, etc. Per-field validation issues land on
    :class:`SaveAgentIssue`."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


SaveAgentField = Literal[
    "display_name",
    "description",
    "tags",
    "keep_context",
]


@dataclass(frozen=True)
class SaveAgentIssue:
    """One inline validation finding."""

    severity: Literal["error", "warning"]
    field: SaveAgentField
    message: str
    detail: Optional[str] = None


@dataclass(frozen=True)
class SaveAgentForm:
    """Frozen form state for the SaveAgentModal.

    Carries the editable fields plus the read-only context (the
    original query + MAGE metadata + context files) that the
    promote helper needs when ``keep_context`` is on.

    Every mutator returns a new frozen instance so undo is a
    list of past forms.
    """

    # User-editable
    display_name: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    keep_context: bool = True
    # Snapshots for dirty tracking
    suggested_display_name: str = ""
    suggested_description: str = ""
    suggested_tags: tuple[str, ...] = ()
    # Read-only generation context
    query: str = ""
    domain: str = ""
    mage_metadata: dict[str, Any] = field(default_factory=dict)
    context_files: tuple[dict[str, Any], ...] = ()

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    @property
    def favourite(self) -> bool:
        """``True`` when the favourite tag is currently in
        ``tags``. Matches the checkbox state."""
        return FAVOURITE_TAG in self.tags

    @property
    def display_name_dirty(self) -> bool:
        return (
            self.display_name.strip() != self.suggested_display_name.strip()
        )

    @property
    def description_dirty(self) -> bool:
        return self.description.strip() != self.suggested_description.strip()

    @property
    def tags_dirty(self) -> bool:
        return set(self.tags) != set(self.suggested_tags)

    def is_dirty(self) -> bool:
        return (
            self.display_name_dirty
            or self.description_dirty
            or self.tags_dirty
        )


@dataclass(frozen=True)
class SaveAgentOutcome:
    """Frozen outcome the modal renders into a toast after the
    user clicks Save (either button)."""

    entity_id: str
    success: bool = True
    promoted: bool = False
    metadata_written: bool = False
    error: Optional[str] = None
    issues: tuple[SaveAgentIssue, ...] = ()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _heuristic_display_name(query: str, domain: str) -> str:
    """Fallback name when MAGE's suggested name is empty.

    Format: ``"<domain> · <first 60 chars of query>"``. When
    either piece is missing the helper degrades gracefully (no
    leading separator on an empty domain, full empty when both
    are blank).
    """
    truncated = (query or "").strip()
    if len(truncated) > _QUERY_TRUNCATE:
        truncated = truncated[:_QUERY_TRUNCATE].rstrip() + "…"
    if domain and truncated:
        return f"{domain} · {truncated}"
    return domain or truncated


def _project_files(files: Any) -> tuple[dict[str, Any], ...]:
    """Project a list of `ContextFileRef`-shaped objects / dicts
    into the form's tuple."""
    if not files:
        return ()
    out: list[dict[str, Any]] = []
    for raw in files:
        if isinstance(raw, dict):
            out.append(dict(raw))
        elif hasattr(raw, "model_dump"):
            try:
                dumped = raw.model_dump(exclude_none=False)
            except TypeError:
                dumped = raw.model_dump()
            if isinstance(dumped, dict):
                out.append(dict(dumped))
        else:
            # Last-ditch fallback: read known attributes.
            out.append(
                {
                    "path": str(getattr(raw, "path", "") or ""),
                    "sha256": str(getattr(raw, "sha256", "") or ""),
                    "size_bytes": int(getattr(raw, "size_bytes", 0) or 0),
                    "mime_type": getattr(raw, "mime_type", None),
                }
            )
    return tuple(out)


def _read_meta_attr(meta: Any, name: str, default: Any = "") -> Any:
    """Read ``name`` off a `MAGEMetadata`-shaped object OR a dict."""
    if isinstance(meta, dict):
        return meta.get(name, default)
    return getattr(meta, name, default)


def _seed_tags(
    suggested: Iterable[str],
    domain: str,
) -> tuple[str, ...]:
    """Pre-fill the tag list, stamping a ``domain:{value}`` tag
    when ``domain`` is supplied and not already present. The
    `CareMemory.save_chain` facade adds this tag automatically
    at write time too; we surface it in the form so the modal
    can render it as a removable chip."""
    out: list[str] = []
    seen: set[str] = set()
    for t in suggested or ():
        clean = str(t).strip()
        if clean and clean not in seen:
            out.append(clean)
            seen.add(clean)
    if domain:
        tag = f"domain:{domain}"
        if tag not in seen:
            out.insert(0, tag)
    return tuple(out)


# ---------------------------------------------------------------------------
# Seeder
# ---------------------------------------------------------------------------


def seed_save_agent_form(
    *,
    query: str = "",
    mage_metadata: Any = None,
    context_files: Any = None,
    suggested_name_override: Optional[str] = None,
) -> SaveAgentForm:
    """Build the initial form state from a finished MAGE generation.

    Pre-fill rules:

    * **Display name** — :attr:`MAGEMetadata.suggested_display_name`
      wins. Falls back to a heuristic
      ``"<domain> · <first 60 chars of query>"``. The
      ``suggested_name_override`` kwarg wins over both for
      programmatic callers (e.g. CLI ``care save`` with a
      ``--name`` flag).
    * **Description** — :attr:`MAGEMetadata.suggested_description`
      wins. Falls back to the original ``query`` verbatim.
    * **Tags** — :attr:`MAGEMetadata.suggested_tags` with a
      leading ``"domain:{value}"`` tag (deduped).
    * **keep_context** — defaults to ``True`` so re-run works
      out of the box; the modal flips it off when the user
      doesn't want files saved.

    The same MAGE metadata + query + files are stored as
    read-only fields on the form so :func:`apply_save_agent_form`
    can route the right pieces through CARE's facades without
    requiring the modal to re-thread them.

    Args:
        query: Original user query.
        mage_metadata: A `MAGEMetadata` instance OR a dict with
            ``suggested_display_name`` / ``suggested_description``
            / ``suggested_tags`` / ``domain``. ``None`` is
            equivalent to an empty metadata dict.
        context_files: Iterable of file refs (
            `gigaevo_client.ContextFileRef` objects, plain dicts,
            or duck-typed objects with ``path`` / ``sha256``
            attributes).
        suggested_name_override: Optional programmatic override.

    Returns:
        Initial :class:`SaveAgentForm`.
    """
    domain = str(_read_meta_attr(mage_metadata, "domain", "") or "")
    sugg_name = sanitize_chain_name(
        str(_read_meta_attr(mage_metadata, "suggested_display_name", "") or "")
    )
    sugg_desc = str(
        _read_meta_attr(mage_metadata, "suggested_description", "") or ""
    ).strip()
    sugg_tags_raw = _read_meta_attr(mage_metadata, "suggested_tags", []) or []

    if suggested_name_override is not None:
        display_name = suggested_name_override
    elif sugg_name:
        display_name = sugg_name
    else:
        display_name = _heuristic_display_name(query, domain)

    description = sugg_desc or query

    seeded_tags = _seed_tags(sugg_tags_raw, domain)

    meta_dict: dict[str, Any] = {}
    if isinstance(mage_metadata, dict):
        meta_dict = dict(mage_metadata)
    elif mage_metadata is not None:
        dump = getattr(mage_metadata, "model_dump", None)
        if callable(dump):
            try:
                payload = dump(exclude_none=False)
            except TypeError:
                payload = dump()
            if isinstance(payload, dict):
                meta_dict = dict(payload)

    return SaveAgentForm(
        display_name=display_name,
        description=description,
        tags=seeded_tags,
        keep_context=True,
        suggested_display_name=display_name,
        suggested_description=description,
        suggested_tags=seeded_tags,
        query=query or "",
        domain=domain,
        mage_metadata=meta_dict,
        context_files=_project_files(context_files),
    )


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


def set_display_name(form: SaveAgentForm, value: str) -> SaveAgentForm:
    return replace(form, display_name=value)


def set_description(form: SaveAgentForm, value: str) -> SaveAgentForm:
    return replace(form, description=value)


def set_tags(form: SaveAgentForm, value: Iterable[str]) -> SaveAgentForm:
    cleaned = tuple(t.strip() for t in value if t and t.strip())
    return replace(form, tags=cleaned)


def add_tag(form: SaveAgentForm, tag: str) -> SaveAgentForm:
    """Append a tag if it isn't already present (stable order)."""
    clean = tag.strip()
    if not clean or clean in form.tags:
        return form
    return replace(form, tags=form.tags + (clean,))


def remove_tag(form: SaveAgentForm, tag: str) -> SaveAgentForm:
    """Drop a tag if present; no-op otherwise."""
    if tag not in form.tags:
        return form
    return replace(form, tags=tuple(t for t in form.tags if t != tag))


def toggle_favourite(form: SaveAgentForm) -> SaveAgentForm:
    """Flip the favourite tag (the "⭐" checkbox)."""
    if form.favourite:
        return remove_tag(form, FAVOURITE_TAG)
    return add_tag(form, FAVOURITE_TAG)


def set_keep_context(form: SaveAgentForm, value: bool) -> SaveAgentForm:
    return replace(form, keep_context=bool(value))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


async def validate_save_agent_form(
    form: SaveAgentForm,
    memory: Any = None,
    *,
    namespace: Optional[str] = None,
    check_unique: bool = True,
    timeout: float = 5.0,
) -> tuple[SaveAgentIssue, ...]:
    """Validate the form.

    Sync checks (always run):
        * ``display_name`` must be non-empty.
        * Duplicate tags warning.

    Async check (when ``check_unique=True`` AND ``memory`` is
    supplied): queries ``memory.client.list_chains(q=name,
    namespace=namespace)`` and flags an error if any returned
    row has a case-insensitive exact `display_name` match. The
    server-side ``q=`` filter does substring matching, so we
    re-filter client-side for exact equality.

    Network failures during the uniqueness check are NOT
    surfaced as issues — the modal would block on a flaky
    Memory. Instead they degrade silently (returns no issue for
    that branch) so the user can still save; collisions surface
    on the actual POST.

    Returns:
        Tuple of issues; empty when the form is ready to save.
    """
    issues: list[SaveAgentIssue] = []

    if not form.display_name.strip():
        issues.append(
            SaveAgentIssue(
                severity="error",
                field="display_name",
                message="Name is required",
            )
        )

    # Duplicate-tag warning (matches the EditAgentDraft behaviour
    # — duplicate chips ignored on save).
    seen: set[str] = set()
    duplicates: list[str] = []
    for tag in form.tags:
        if tag in seen and tag not in duplicates:
            duplicates.append(tag)
        seen.add(tag)
    if duplicates:
        issues.append(
            SaveAgentIssue(
                severity="warning",
                field="tags",
                message=f"Duplicate tags ignored on save: {', '.join(duplicates)}",
            )
        )

    if (
        check_unique
        and memory is not None
        and form.display_name.strip()
        and not any(i.field == "display_name" and i.severity == "error" for i in issues)
    ):
        clash = await _check_name_unique(
            memory,
            name=form.display_name.strip(),
            namespace=namespace,
            timeout=timeout,
        )
        if clash:
            issues.append(
                SaveAgentIssue(
                    severity="error",
                    field="display_name",
                    message=f"Name '{form.display_name.strip()}' already exists in your library",
                    detail=clash,
                )
            )

    return tuple(issues)


async def _check_name_unique(
    memory: Any,
    *,
    name: str,
    namespace: Optional[str],
    timeout: float,
) -> Optional[str]:
    """Run the unique-name probe. Returns the colliding
    ``entity_id`` or ``None`` when no clash. Silently returns
    ``None`` on network failure — modal should rely on the POST
    error path for those cases."""
    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    list_fn = getattr(client, "list_chains", None) if client else None
    if not callable(list_fn):
        return None
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(
                list_fn,
                limit=10,
                channel="latest",
                q=name,
                namespace=namespace,
            ),
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001
        return None
    needle = name.casefold()
    rows_iter = rows if isinstance(rows, (list, tuple)) else ()
    for row in rows_iter:
        candidate = (
            row.get("display_name")
            if isinstance(row, dict)
            else getattr(row, "display_name", None)
        )
        if isinstance(candidate, str) and candidate.casefold() == needle:
            entity_id = (
                row.get("entity_id")
                if isinstance(row, dict)
                else getattr(row, "entity_id", None)
            )
            return str(entity_id) if entity_id else "unknown"
    return None


# ---------------------------------------------------------------------------
# Apply (promote + metadata update)
# ---------------------------------------------------------------------------


async def apply_save_agent_form(
    memory: Any,
    session: Any,
    form: SaveAgentForm,
    *,
    to_channel: str = "latest",
    timeout: float = 15.0,
) -> SaveAgentOutcome:
    """Promote the draft + apply the form's edited metadata.

    Workflow:

    1. Call :func:`care.runtime.promote_draft` to move the
       entity from the ``draft`` channel to ``to_channel``
       (default ``"latest"``).
    2. Build the final tag set (deduped, ``draft`` removed since
       the entity is no longer a draft, ``favourite`` flag
       routed to both the tag list AND the dedicated
       ``favourite`` column).
    3. Call ``memory.client._update_metadata(entity_type,
       entity_id, display_name=..., description=...,
       tags=..., favourite=...)`` to apply the user's edits in
       a single PATCH.

    The ``keep_context`` flag is NOT touched here — it was
    already baked into the draft's CARE metadata block when
    :func:`care.runtime.auto_save_draft` ran. When the user
    turned off ``keep_context``, the modal should re-call
    ``CareMemory.save_chain`` (with ``entity_id=`` for a new
    version, no ``context_files``) BEFORE the promote so the
    promoted version drops the file pins.

    Args:
        memory: A `CareMemory` facade.
        session: The :class:`DraftSession` returned by
            :func:`care.runtime.auto_save_draft`.
        form: The user-finalised :class:`SaveAgentForm`.
        to_channel: Destination channel (default ``"latest"``).
        timeout: Per-call deadline.

    Returns:
        :class:`SaveAgentOutcome`. Per-call failures land on
        ``error`` so the modal renders a toast — the function
        never raises for HTTP errors.

    Raises:
        SaveAgentError: ``session`` is missing required fields
            (no ``entity_id``).
    """
    if not getattr(session, "entity_id", None):
        raise SaveAgentError("draft session has no entity_id; nothing to promote")

    outcome_entity_id = session.entity_id

    # Step 1: promote.
    promote_fn = _import_promote_draft()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(promote_fn, memory, session, to_channel=to_channel),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return SaveAgentOutcome(
            entity_id=outcome_entity_id,
            success=False,
            error=f"promote timed out after {timeout:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        return SaveAgentOutcome(
            entity_id=outcome_entity_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    # Step 2: build the final tag set.
    final_tags = _finalise_tags(form.tags)
    is_favourite = FAVOURITE_TAG in final_tags
    tags_without_favourite = tuple(t for t in final_tags if t != FAVOURITE_TAG)

    # Step 3: apply metadata edits via PATCH.
    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    update_fn = getattr(client, "_update_metadata", None) if client else None
    if update_fn is None:
        # Promoted but couldn't update metadata. Surface as a
        # warning-shaped outcome (promoted=True, written=False).
        return SaveAgentOutcome(
            entity_id=outcome_entity_id,
            success=True,
            promoted=True,
            metadata_written=False,
            error="memory facade does not expose client._update_metadata()",
        )

    update_kwargs: dict[str, Any] = {
        "display_name": form.display_name.strip() or None,
        "description": form.description.strip() or None,
        "tags": list(tags_without_favourite),
        "favourite": is_favourite,
    }
    # Strip None values so the PATCH only carries fields we
    # actually want to mutate.
    update_kwargs = {k: v for k, v in update_kwargs.items() if v is not None}

    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                update_fn,
                session.entity_type,
                outcome_entity_id,
                **update_kwargs,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return SaveAgentOutcome(
            entity_id=outcome_entity_id,
            success=False,
            promoted=True,
            error=f"metadata update timed out after {timeout:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        return SaveAgentOutcome(
            entity_id=outcome_entity_id,
            success=False,
            promoted=True,
            error=f"{type(exc).__name__}: {exc}",
        )

    return SaveAgentOutcome(
        entity_id=outcome_entity_id,
        success=True,
        promoted=True,
        metadata_written=True,
    )


def _finalise_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    """Strip the per-draft sentinel tag, dedupe, and preserve
    stable order. The ``draft`` tag is added by
    :func:`care.runtime.auto_save_draft` to prevent the draft
    leaking into normal library views; once promoted the tag is
    no longer informative."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        clean = tag.strip()
        if not clean or clean == "draft" or clean in seen:
            continue
        out.append(clean)
        seen.add(clean)
    return tuple(out)


def _import_promote_draft() -> Any:
    """Lazy import — keeps `save_agent_form` testable without
    pulling the full draft module at import time. (`draft.py`
    imports `care.memory` which pulls the SDK in.)"""
    from care.runtime.draft import promote_draft

    return promote_draft


__all__ = [
    "FAVOURITE_TAG",
    "SaveAgentError",
    "SaveAgentField",
    "SaveAgentForm",
    "SaveAgentIssue",
    "SaveAgentOutcome",
    "add_tag",
    "apply_save_agent_form",
    "remove_tag",
    "seed_save_agent_form",
    "set_description",
    "set_display_name",
    "set_keep_context",
    "set_tags",
    "toggle_favourite",
    "validate_save_agent_form",
]
