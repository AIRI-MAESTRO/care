"""Draft-channel auto-save for generated chains (TODO §3 P0).

CARE writes every freshly-generated chain to Memory's ``draft``
channel **before** showing the ``SaveAgentModal`` so a crash
between "MAGE finished" and "user clicked Save" can't lose the
work. The modal then either:

* **Saves** — the draft is promoted to channel ``latest`` (a
  cheap server-side ``POST /promote`` that copies the channel
  pointer; no second upload), or
* **Discards** — the draft entity is deleted entirely.

This module owns the helper trio (`auto_save_draft`,
`promote_draft`, `discard_draft`) plus the small
:class:`DraftSession` state carrier the modal mutates. The TUI
modal stays a thin shell on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from care.memory import CareMemory

DRAFT_CHANNEL = "draft"
"""Channel name CARE writes to during the draft window.
Server-side `channel` is a free-form string; we use this constant
so the value is consistent across save / promote / discard."""

LATEST_CHANNEL = "latest"
"""Channel the draft promotes to once the user clicks Save."""

DRAFT_TAG = "draft"
"""Tag stamped onto every draft so the LibraryScreen can filter
stale drafts and let the user clean up explicitly. The tag is
removed when the draft is promoted (the SDK's ``update_metadata``
PATCH handles tag mutation without a new version)."""


class DraftError(RuntimeError):
    """Raised when a draft operation fails in a way the caller can
    actually do something about (e.g. promoting an already-deleted
    draft). Distinguishes from raw ``httpx.HTTPStatusError`` so
    UIs can show a friendly toast without parsing HTTP details."""


@dataclass
class DraftSession:
    """State the SaveAgentModal threads through the draft lifecycle.

    Created by :func:`auto_save_draft` and consumed by
    :func:`promote_draft` / :func:`discard_draft`. Mutable on
    purpose: the modal flips ``promoted`` / ``discarded`` so the
    surrounding screen knows the terminal state without inspecting
    HTTP responses.
    """

    entity_id: str
    name: str
    domain: str | None = None
    entity_type: str = "chain"
    promoted: bool = False
    discarded: bool = False

    @property
    def terminal(self) -> bool:
        """``True`` once the draft has been promoted or discarded —
        the surrounding screen can stop polling state."""
        return self.promoted or self.discarded


def auto_save_draft(
    memory: CareMemory,
    chain: Any,
    *,
    name: str,
    query: str | None = None,
    domain: str | None = None,
    context_files: list | None = None,
    mage_metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    author: str | None = None,
    entity_type: str = "chain",
) -> DraftSession:
    """Persist ``chain`` to Memory's ``draft`` channel immediately
    after generation. Returns a :class:`DraftSession` the modal
    threads through promote/discard.

    The ``DRAFT_TAG`` is auto-stamped so the LibraryScreen can
    surface (and let the user clean up) stale drafts. The chain's
    CARE metadata (``CareChainMetadata``) is built normally — when
    the user accepts the modal there's no second upload, just a
    channel-pointer copy.

    Args:
        memory: A constructed :class:`care.CareMemory`.
        chain: A CARL ``ReasoningChain`` or raw content dict.
        name: Draft name. The modal can override it before
            promote — the underlying entity is renamed via
            ``update_metadata`` in that case.
        query: Original user query (saved on the chain content
            via ``CareChainMetadata.task_description``).
        domain, context_files, mage_metadata, tags, author:
            Forwarded to :meth:`CareMemory.save_chain`.
        entity_type: ``"chain"`` (default), ``"agent"``, or
            ``"agent_skill"`` — picks the typed router on Memory.

    Returns:
        :class:`DraftSession` with the new ``entity_id``.
    """
    merged_tags = _merge_draft_tag(tags)
    entity_id = memory.save_chain(
        chain,
        name=name,
        query=query,
        domain=domain,
        context_files=context_files,
        mage_metadata=mage_metadata,
        tags=merged_tags,
        author=author,
        channel=DRAFT_CHANNEL,
    )
    return DraftSession(
        entity_id=entity_id,
        name=name,
        domain=domain,
        entity_type=entity_type,
    )


def promote_draft(
    memory: CareMemory,
    session: DraftSession,
    *,
    from_channel: str = DRAFT_CHANNEL,
    to_channel: str = LATEST_CHANNEL,
) -> DraftSession:
    """Promote a draft to ``latest`` (or another channel).

    Server-side this is a single ``POST /promote`` — no new
    version uploaded, just the channel pointer copied. The
    ``DRAFT_TAG`` is left on the entity for now; CARE's
    SaveAgentModal removes it via ``update_metadata`` when it
    finalises the agent's tag set.

    Args:
        memory: The same facade used to save the draft.
        session: The :class:`DraftSession` returned by
            :func:`auto_save_draft`.
        from_channel: Source channel. Defaults to ``DRAFT_CHANNEL``;
            override only for tests or unusual flows.
        to_channel: Destination channel. Defaults to
            ``LATEST_CHANNEL``.

    Returns:
        The mutated ``session`` (``promoted=True``).

    Raises:
        DraftError: When the session has already been promoted /
            discarded — the caller's UI is in an inconsistent
            state and should surface a friendly message.
    """
    _check_not_terminal(session, action="promote")
    try:
        memory.client.promote(
            session.entity_id,
            from_channel=from_channel,
            to_channel=to_channel,
            entity_type=session.entity_type,
        )
    except Exception as exc:  # noqa: BLE001
        raise DraftError(
            f"failed to promote draft {session.entity_id}: {exc}"
        ) from exc
    session.promoted = True
    return session


def discard_draft(
    memory: CareMemory,
    session: DraftSession,
) -> DraftSession:
    """Delete the draft entity entirely.

    Currently chain-typed deletion only (the SDK exposes
    ``delete_chain`` via :class:`ChainsMixin`). When
    ``session.entity_type`` is something else we delegate to the
    generic ``_delete_entity`` so the helper supports agent +
    agent_skill drafts as soon as MAGE starts generating those
    too.

    Idempotent: discarding an already-discarded session is a no-op
    that returns the session unchanged. Discarding an already-
    promoted session raises :class:`DraftError` — the entity now
    lives on ``latest`` and shouldn't be silently nuked.
    """
    if session.discarded:
        return session
    if session.promoted:
        raise DraftError(
            f"refusing to discard promoted draft {session.entity_id} — "
            "use the standard delete action on the LibraryScreen instead."
        )
    try:
        memory.client._delete_entity(  # type: ignore[attr-defined]
            session.entity_type, session.entity_id
        )
    except Exception as exc:  # noqa: BLE001
        raise DraftError(
            f"failed to discard draft {session.entity_id}: {exc}"
        ) from exc
    session.discarded = True
    return session


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _merge_draft_tag(tags: list[str] | None) -> list[str]:
    """Insert ``DRAFT_TAG`` at the front; dedupe in place."""
    out: list[str] = [DRAFT_TAG]
    for tag in tags or []:
        if tag and tag not in out:
            out.append(tag)
    return out


def _check_not_terminal(session: DraftSession, *, action: str) -> None:
    if session.promoted:
        raise DraftError(
            f"cannot {action}: draft {session.entity_id} already promoted"
        )
    if session.discarded:
        raise DraftError(
            f"cannot {action}: draft {session.entity_id} already discarded"
        )


__all__ = [
    "DRAFT_CHANNEL",
    "DRAFT_TAG",
    "DraftError",
    "DraftSession",
    "LATEST_CHANNEL",
    "auto_save_draft",
    "discard_draft",
    "promote_draft",
]
