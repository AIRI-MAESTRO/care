"""LibraryScreen per-row actions data layer (TODO §1.3 P0).

The LibraryScreen exposes per-row affordances via Enter or
right-click context menu plus single-key bindings (`Enter` opens,
`R` runs, `E` edits, `F` favourites, `Del` deletes). The
Textual key handler + context-menu widget are gated on TODO §1
P0 multi-screen workflow, but the action registry + dispatch +
single-row mutation helpers are bounded and land now.

This layer ships:

* :class:`RowActionKind` literal pinning the canonical action
  names per the TODO spec.
* :class:`RowAction` — frozen action descriptor (kind, label,
  key binding, confirm requirement, enabled-for-status set).
* :func:`default_actions` — canonical set straight from the
  TODO bullet (`Run` / `Open` / `Edit` / `Duplicate` / `Evolve`
  / `Show lineage` / `Toggle favourite` / `Delete`).
* :func:`actions_for_row` — filter the registry by a
  :class:`LibraryRow` status (`draft` agents can't be evolved
  yet; the function pins the per-status visibility rules).
* :func:`find_action_by_key` — key → action dispatch.
* :func:`is_destructive` — predicate for confirmation dialogs.
* Single-row mutators (async) the screen calls into when an
  action fires:
  - :func:`toggle_favourite_row` — flips the favourite column.
  - :func:`delete_row` — soft-delete via the typed router.
  - :func:`duplicate_chain` — load + re-save with a new name +
    fresh entity_id (no upstream `clone` endpoint, so CARE
    composes from `get_chain_dict` + `save_chain`).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

from care.runtime.i18n import t
from care.runtime.library_view import LibraryRow


RowActionKind = Literal[
    "run",
    "open",
    "edit",
    "duplicate",
    "evolve",
    "archive_evolutions",
    "show_lineage",
    "toggle_favourite",
    "delete",
]
"""Canonical per-row action names per the TODO §1.3 P0 spec."""


_DESTRUCTIVE_KINDS: frozenset[RowActionKind] = frozenset({"delete"})
"""Actions that should fire a "Are you sure?" confirmation
dialog before running. Currently just `delete` (soft-delete is
recoverable via Memory's trash, but the gesture is still
worth pinning behind a confirm)."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RowActionError(RuntimeError):
    """Raised for caller-mistake failures — unknown key in
    :func:`find_action_by_key`, missing client method on a
    mutator call. Per-call HTTP failures don't raise; they land
    on the mutator's return value (mirrors the bulk-ops
    contract)."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowAction:
    """One row-level affordance.

    Frozen so the registry flows through Textual messages
    without defensive copies. ``key_binding`` is a single-key
    label per the TODO ("Enter", "R", "E", ...); the Textual
    screen wires this to `Binding` definitions.

    ``enabled_for_status`` is a frozenset of status values
    (`draft`, `runnable`, `evolved`) the action is valid on.
    The empty frozenset is sugar for "every status". Drives the
    sidebar's right-click menu (hide entries that don't apply
    to the selected row).
    """

    kind: RowActionKind
    label: str
    key_binding: str = ""
    requires_confirm: bool = False
    description: str = ""
    enabled_for_status: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_destructive(self) -> bool:
        """``True`` when the action should fire a confirmation
        prompt before running."""
        return self.kind in _DESTRUCTIVE_KINDS


@dataclass(frozen=True)
class RowMutationOutcome:
    """Result of a single-row mutator.

    Per-call HTTP failures land here rather than raising so
    the screen's catch clause stays single-branch.
    """

    entity_id: str
    success: bool = True
    error: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def default_actions() -> tuple[RowAction, ...]:
    """Return the canonical action registry per the TODO §1.3
    P0 spec.

    Ordering matches the menu's render order (matches the TODO
    bullet exactly: Run / Open / Edit / Duplicate / Evolve /
    Show lineage / Toggle favourite / Delete). The screen
    binds key shortcuts off ``key_binding`` and the menu
    sources labels from ``label``.

    Status gating:
      * `run`, `open`, `edit`, `duplicate`, `toggle_favourite`,
        `delete` apply to every status.
      * `evolve` requires a runnable or evolved chain — draft
        chains can't be evolved yet (the §3 P0 SaveAgentModal
        is the promote-to-runnable step).
      * `show_lineage` requires either an `evolved` chain (has
        a parent chain) or a runnable one (might have evolved
        descendants); draft chains have no ancestry.
    """
    return (
        RowAction(
            kind="run",
            label=t("library.rowAction.run.label"),
            key_binding="R",
            description=t("library.rowAction.run.description"),
        ),
        RowAction(
            kind="open",
            label=t("library.rowAction.open.label"),
            key_binding="Enter",
            description=t("library.rowAction.open.description"),
        ),
        RowAction(
            kind="edit",
            label=t("library.rowAction.edit.label"),
            key_binding="E",
            description=t("library.rowAction.edit.description"),
        ),
        RowAction(
            kind="duplicate",
            label=t("library.rowAction.duplicate.label"),
            key_binding="D",
            description=t("library.rowAction.duplicate.description"),
        ),
        RowAction(
            kind="evolve",
            label=t("library.rowAction.evolve.label"),
            key_binding="V",
            description=t("library.rowAction.evolve.description"),
            enabled_for_status=frozenset({"runnable", "evolved"}),
        ),
        RowAction(
            kind="archive_evolutions",
            label=t("library.rowAction.archiveEvolutions.label"),
            key_binding="Z",
            description=t("library.rowAction.archiveEvolutions.description"),
            enabled_for_status=frozenset({"runnable", "evolved"}),
        ),
        RowAction(
            kind="show_lineage",
            label=t("library.rowAction.showLineage.label"),
            key_binding="L",
            description=t("library.rowAction.showLineage.description"),
            enabled_for_status=frozenset({"runnable", "evolved"}),
        ),
        RowAction(
            kind="toggle_favourite",
            label=t("library.rowAction.toggleFavourite.label"),
            key_binding="F",
            description=t("library.rowAction.toggleFavourite.description"),
        ),
        RowAction(
            kind="delete",
            label=t("library.rowAction.delete.label"),
            key_binding="Delete",
            requires_confirm=True,
            description=t("library.rowAction.delete.description"),
        ),
    )


def actions_for_row(
    row: LibraryRow,
    *,
    registry: Optional[Iterable[RowAction]] = None,
) -> tuple[RowAction, ...]:
    """Filter the registry by the row's status.

    Returns actions in their canonical order, dropping any
    whose ``enabled_for_status`` doesn't include
    :attr:`row.status`. An action with empty
    ``enabled_for_status`` applies to every status.
    """
    actions = tuple(registry) if registry is not None else default_actions()
    status = row.status
    return tuple(
        a for a in actions
        if not a.enabled_for_status or status in a.enabled_for_status
    )


def find_action_by_key(
    key: str,
    *,
    registry: Optional[Iterable[RowAction]] = None,
) -> Optional[RowAction]:
    """Look up a registered action by its ``key_binding``.

    Case-insensitive match so the screen can pass through
    whatever the keymap emits without normalising. Returns
    ``None`` when no action matches — the screen falls through
    to default Textual behaviour.
    """
    if not key:
        return None
    actions = registry if registry is not None else default_actions()
    needle = key.casefold()
    for action in actions:
        if not action.key_binding:
            continue
        if action.key_binding.casefold() == needle:
            return action
    return None


def find_action_by_kind(
    kind: RowActionKind,
    *,
    registry: Optional[Iterable[RowAction]] = None,
) -> Optional[RowAction]:
    """Look up an action by its canonical kind."""
    actions = registry if registry is not None else default_actions()
    for action in actions:
        if action.kind == kind:
            return action
    return None


def is_destructive(action: RowAction | RowActionKind) -> bool:
    """Predicate: should the screen prompt before running?

    Accepts either a :class:`RowAction` or just its kind.
    """
    if isinstance(action, RowAction):
        return action.is_destructive
    return action in _DESTRUCTIVE_KINDS


# ---------------------------------------------------------------------------
# Single-row mutators
# ---------------------------------------------------------------------------


_DEFAULT_TIMEOUT = 10.0


async def toggle_favourite_row(
    memory: Any,
    row: LibraryRow,
    *,
    value: Optional[bool] = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> RowMutationOutcome:
    """Toggle (or set) the favourite flag on one row.

    ``value=None`` flips the current state; pass ``True`` / ``False``
    explicitly to set without reading the row first. Returns a
    :class:`RowMutationOutcome` — HTTP failures land on
    ``error`` so the screen renders a toast.
    """
    client = _resolve_client(memory)
    fn = getattr(client, "_mark_favourite", None)
    if not callable(fn):
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error="memory facade does not expose client._mark_favourite()",
        )
    target = (not row.favourite) if value is None else bool(value)
    start = time.monotonic()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                fn, row.entity_type, row.entity_id, value=target,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error=f"favourite toggle timed out after {timeout:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return RowMutationOutcome(
        entity_id=row.entity_id,
        success=True,
        detail={
            "previous": row.favourite,
            "current": target,
            "duration_ms": (time.monotonic() - start) * 1000,
        },
    )


async def delete_row(
    memory: Any,
    row: LibraryRow,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> RowMutationOutcome:
    """Soft-delete one row.

    Memory's delete is a soft-delete (sets ``deleted_at``) so
    the row stays recoverable via Memory's trash. The screen's
    confirmation prompt is the user-facing safeguard; this
    layer just fires the typed router call.
    """
    client = _resolve_client(memory)
    fn = getattr(client, "_delete_entity", None)
    if not callable(fn):
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error="memory facade does not expose client._delete_entity()",
        )
    start = time.monotonic()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(fn, row.entity_type, row.entity_id),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error=f"delete timed out after {timeout:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return RowMutationOutcome(
        entity_id=row.entity_id,
        success=True,
        detail={"duration_ms": (time.monotonic() - start) * 1000},
    )


async def duplicate_chain(
    memory: Any,
    row: LibraryRow,
    *,
    new_name: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> RowMutationOutcome:
    """Save a copy of ``row`` under a new entity_id.

    Memory doesn't expose a `clone` endpoint, so CARE composes
    from `get_chain_dict` (read) + `save_chain` with
    ``entity_id=None`` (creates fresh entity).

    Args:
        memory: A `CareMemory`-like facade exposing
            ``.client.get_chain_dict(...)`` AND ``.save_chain(...)``.
        row: The :class:`LibraryRow` to duplicate.
        new_name: Display name for the copy. ``None`` synthesises
            ``"{original} (copy)"``.
        timeout: Per-call deadline.

    Returns:
        :class:`RowMutationOutcome` carrying the new entity_id
        in ``detail["new_entity_id"]`` on success.
    """
    client = _resolve_client(memory)
    get_fn = getattr(client, "get_chain_dict", None) or getattr(
        client, "get_chain_raw", None
    )
    if not callable(get_fn):
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error="memory facade does not expose client.get_chain_dict()",
        )
    save_fn = getattr(memory, "save_chain", None)
    if not callable(save_fn):
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error="memory facade does not expose save_chain()",
        )

    target_name = new_name or f"{row.label} (copy)"
    start = time.monotonic()
    try:
        original = await asyncio.wait_for(
            asyncio.to_thread(get_fn, row.entity_id, row.channel),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error=f"duplicate read timed out after {timeout:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    if original is None:
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error=f"chain {row.entity_id!r} not found on channel {row.channel!r}",
        )

    chain_payload = _extract_chain_payload(original)
    try:
        new_entity_id = await asyncio.wait_for(
            asyncio.to_thread(
                save_fn,
                chain_payload,
                name=target_name,
                tags=list(row.tags) or None,
                entity_id=None,  # force create
                channel="latest",
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error=f"duplicate write timed out after {timeout:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        return RowMutationOutcome(
            entity_id=row.entity_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    return RowMutationOutcome(
        entity_id=row.entity_id,
        success=True,
        detail={
            "new_entity_id": str(new_entity_id),
            "new_name": target_name,
            "duration_ms": (time.monotonic() - start) * 1000,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_client(memory: Any) -> Any:
    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    if client is None:
        raise RowActionError(
            "memory facade does not expose a `.client` attribute"
        )
    return client


def _extract_chain_payload(original: Any) -> Any:
    """`get_chain_dict` may return the raw chain dict OR an
    EntityResponse-shaped wrapper with `content`. Normalise to
    the chain dict that `save_chain` expects."""
    if original is None:
        return {}
    if isinstance(original, dict):
        # If the dict is an EntityResponse-shape, unwrap content.
        if "content" in original and isinstance(original["content"], dict):
            return original["content"]
        return original
    content = getattr(original, "content", None)
    if isinstance(content, dict):
        return content
    # Last-ditch: model_dump.
    dump = getattr(original, "model_dump", None)
    if callable(dump):
        try:
            payload = dump()
        except TypeError:
            payload = dump()
        if isinstance(payload, dict):
            return payload.get("content") or payload
    return original


__all__ = [
    "RowAction",
    "RowActionError",
    "RowActionKind",
    "RowMutationOutcome",
    "actions_for_row",
    "default_actions",
    "delete_row",
    "duplicate_chain",
    "find_action_by_key",
    "find_action_by_kind",
    "is_destructive",
    "toggle_favourite_row",
]
