"""Save-time conflict detection + resolution (TODO §3 P1).

Two CARE users can edit the same chain in parallel — one
re-runs MAGE on the original task with a tweak, the other
modifies a step prompt in EditAgentScreen — and both try to
save under the same library name. The current behaviour
silently overwrites whichever lands second. The future
`SaveAgentModal` should:

1. Detect the conflict before the write hits Memory.
2. Show the user a unified diff between the existing version
   and the one they're about to save.
3. Offer three resolutions:
   - **accept incoming** — replace the existing entity outright.
   - **keep existing** — abort the save, return the existing
     entity_id.
   - **new version** — write the incoming content as a fresh
     version of the same entity_id, preserving history.

This module is the data layer behind that modal:

* :func:`compute_content_sha256` — stable digest of a chain /
  skill / memory_card content dict so equality checks work
  across processes.
* :func:`detect_conflict` — queries Memory for an existing
  entity with the same display name + entity_type, compares
  SHAs, and (on mismatch) projects a :class:`ConflictReport`
  with the unified diff lines the modal renders.
* :func:`apply_resolution` — dispatches the user's chosen
  :class:`ConflictResolution` to the right Memory mutation
  (overwrite / no-op / new-version save).

Duck-typed against CARE's :class:`CareMemory` facade — anything
exposing a small list-or-find surface + the appropriate save
methods works. Tests inject a `_StubMemory` so no SDK / HTTP /
real Memory server is touched.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal


ConflictResolution = Literal[
    "accept_incoming",
    "keep_existing",
    "new_version",
]
"""How the user chose to resolve a conflict.

- ``accept_incoming``: overwrite the existing entity outright.
  Memory loses the previous content (only the version row
  survives if Memory enables versioning by default — CARE
  uses ``new_version`` when version history matters).
- ``keep_existing``: abort the save; return the existing
  entity_id unchanged. The user's local edits stay local.
- ``new_version``: save the incoming content as a new
  version of the same entity_id. Channel `latest` moves to
  the new version; the previous one stays reachable by
  version_id. This is the default the modal recommends.
"""


@dataclass(frozen=True)
class ConflictReport:
    """The data the future SaveAgentModal renders.

    Frozen so the report flows through messages / persisted
    drafts without defensive copies.

    Fields:
        existing_entity_id: Memory id of the entity already on
            the server.
        existing_sha256: SHA-256 of the existing content dict
            (computed via :func:`compute_content_sha256`).
        incoming_sha256: SHA-256 of what the user is trying to
            save. ``existing_sha256 == incoming_sha256`` means
            no conflict (callers check :attr:`is_conflict`
            before reading the diff).
        is_conflict: ``True`` when the SHAs differ. Convenience
            so callers don't have to compare manually.
        existing_content: The current content dict in Memory.
        incoming_content: The content the user is about to save.
        diff_lines: Unified-diff lines comparing the
            JSON-pretty-printed representations of the two
            content dicts. Pre-rendered so the modal doesn't
            need to drive `difflib`.
        name: Display name CARE was about to save under.
        entity_type: ``"chain"`` / ``"agent_skill"`` /
            ``"memory_card"`` — drives which save method
            ``apply_resolution`` dispatches to.
    """

    existing_entity_id: str
    existing_sha256: str
    incoming_sha256: str
    is_conflict: bool
    existing_content: dict[str, Any]
    incoming_content: dict[str, Any]
    diff_lines: tuple[str, ...] = field(default_factory=tuple)
    name: str = ""
    entity_type: str = ""


class ConflictResolutionError(RuntimeError):
    """Raised when the resolution step fails — unknown
    resolution literal, missing memory method, downstream save
    raised."""


def compute_content_sha256(content: dict[str, Any]) -> str:
    """Stable SHA-256 of a content dict.

    Uses ``json.dumps(..., sort_keys=True, separators=(",", ":"))``
    so two semantically-equal dicts hash identically regardless
    of key ordering. Non-JSON-serialisable values (datetimes,
    Pydantic models) get coerced via ``default=str`` rather
    than raising — better to hash a stringified value than to
    error out on a corner case.
    """
    raw = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def detect_conflict(
    memory: Any,
    *,
    name: str,
    entity_type: Literal["chain", "agent_skill", "memory_card"],
    incoming_content: dict[str, Any],
    namespace: str | None = None,
) -> ConflictReport | None:
    """Look up an existing entity with the same name + type and
    compare its content SHA against the incoming content.

    Args:
        memory: Anything exposing ``find_entity_by_name(name,
            entity_type=, namespace=) -> {"entity_id": str,
            "content": dict} | None``. CARE's :class:`CareMemory`
            provides this surface; tests inject a stub.
        name: Display name CARE was going to save under.
        entity_type: ``"chain"``, ``"agent_skill"``, or
            ``"memory_card"``.
        incoming_content: The full content dict the caller
            intends to write.
        namespace: Optional namespace scope. Forwarded to the
            lookup helper.

    Returns:
        :class:`ConflictReport` when a same-name entity exists,
        regardless of whether the SHAs match — caller can use
        ``report.is_conflict`` to decide whether to prompt.
        ``None`` when no existing entity was found (the save
        will create a fresh one, no conflict possible).
    """
    finder = getattr(memory, "find_entity_by_name", None)
    if not callable(finder):
        raise ConflictResolutionError(
            "memory facade is missing `find_entity_by_name(...)` — "
            "wire CareMemory or supply a compatible duck-typed object"
        )

    try:
        existing = finder(
            name=name,
            entity_type=entity_type,
            namespace=namespace,
        )
    except Exception as exc:  # noqa: BLE001
        raise ConflictResolutionError(
            f"conflict lookup failed: {exc}"
        ) from exc

    if not existing:
        return None
    if not isinstance(existing, dict):
        raise ConflictResolutionError(
            f"find_entity_by_name returned {type(existing).__name__}; "
            "expected dict with `entity_id` + `content`"
        )

    existing_id = str(existing.get("entity_id") or "")
    existing_content = existing.get("content") or {}
    if not isinstance(existing_content, dict):
        raise ConflictResolutionError(
            f"existing content is {type(existing_content).__name__}; "
            "expected dict"
        )

    existing_sha = compute_content_sha256(existing_content)
    incoming_sha = compute_content_sha256(incoming_content)
    is_conflict = existing_sha != incoming_sha

    if is_conflict:
        diff_lines = _unified_diff_lines(
            existing_content, incoming_content, name=name
        )
    else:
        diff_lines = ()

    return ConflictReport(
        existing_entity_id=existing_id,
        existing_sha256=existing_sha,
        incoming_sha256=incoming_sha,
        is_conflict=is_conflict,
        existing_content=existing_content,
        incoming_content=incoming_content,
        diff_lines=diff_lines,
        name=name,
        entity_type=entity_type,
    )


def apply_resolution(
    memory: Any,
    report: ConflictReport,
    resolution: ConflictResolution,
    *,
    save_kwargs: dict[str, Any] | None = None,
) -> str:
    """Dispatch the user's chosen resolution.

    Args:
        memory: Same duck-typed memory facade
            :func:`detect_conflict` consumed.
        report: A :class:`ConflictReport` from
            :func:`detect_conflict`. Carries the
            entity-type-specific save shape.
        resolution: One of the :class:`ConflictResolution`
            literal values.
        save_kwargs: Extra kwargs forwarded to the underlying
            save method (e.g. ``tags``, ``when_to_use``,
            ``author``). The conflict module doesn't constrain
            these — the modal collects them from the user.

    Returns:
        The resulting ``entity_id`` (existing or freshly
        created).

    Raises:
        ConflictResolutionError: Unknown resolution literal,
            missing save method on the memory facade, or any
            downstream save raised.
    """
    if resolution not in ("accept_incoming", "keep_existing", "new_version"):
        raise ConflictResolutionError(
            f"unknown resolution {resolution!r}; expected "
            "'accept_incoming', 'keep_existing', or 'new_version'"
        )

    if resolution == "keep_existing":
        return report.existing_entity_id

    method_name = _SAVE_METHOD_FOR_KIND.get(report.entity_type)
    if method_name is None:
        raise ConflictResolutionError(
            f"unknown entity_type {report.entity_type!r}; "
            "expected 'chain' / 'agent_skill' / 'memory_card'"
        )
    save = getattr(memory, method_name, None)
    if not callable(save):
        raise ConflictResolutionError(
            f"memory has no {method_name!r} method"
        )

    kwargs: dict[str, Any] = dict(save_kwargs or {})
    kwargs.setdefault("name", report.name)
    if resolution == "accept_incoming":
        # Overwrite the existing entity in place.
        kwargs["entity_id"] = report.existing_entity_id
    elif resolution == "new_version":
        # Same entity_id, but Memory's save layer treats this as
        # a versioned write because the SHA changed.
        kwargs["entity_id"] = report.existing_entity_id

    try:
        result = save(report.incoming_content, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise ConflictResolutionError(
            f"{method_name}() failed: {exc}"
        ) from exc

    # `CareMemory.save_*` returns bare entity_id strings; some
    # facades may return an EntityRef-like with `.entity_id`.
    if isinstance(result, str):
        return result
    return str(getattr(result, "entity_id", "") or "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SAVE_METHOD_FOR_KIND: dict[str, str] = {
    "chain": "save_chain",
    "agent_skill": "save_agent_skill",
    "memory_card": "save_memory_card",
}


def _unified_diff_lines(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    name: str,
) -> tuple[str, ...]:
    """Render a unified diff between two content dicts.

    Both sides go through ``json.dumps(..., indent=2,
    sort_keys=True)`` so the diff is stable across runs. The
    `difflib.unified_diff` output is what the future
    SaveAgentModal renders — typically inside a Textual
    ``RichLog`` or ``TextArea``.
    """
    a = json.dumps(
        existing,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).splitlines(keepends=False)
    b = json.dumps(
        incoming,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).splitlines(keepends=False)
    return tuple(
        difflib.unified_diff(
            a,
            b,
            fromfile=f"{name} (existing)",
            tofile=f"{name} (incoming)",
            lineterm="",
        )
    )


__all__ = [
    "ConflictReport",
    "ConflictResolution",
    "ConflictResolutionError",
    "apply_resolution",
    "compute_content_sha256",
    "detect_conflict",
]
