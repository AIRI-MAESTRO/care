"""Agent side-by-side comparison data layer (TODO §1.3 P2).

The LibraryScreen's "diff modal" (`D` on a 2-row multi-select)
shows prompt / step / metadata differences between two saved
agents. The Memory SDK's `client.diff_versions(entity_id, from,
to)` works when both versions share an `entity_id` — i.e. a
lineage walk. For two arbitrary library entries (different
`entity_id`s), Memory can't compute the diff server-side; this
module ships the pure client-side projection that handles both
cases uniformly.

The Textual modal is gated on TODO §1 P0 multi-screen workflow.
This layer ships the projection + async fetch / projection
orchestrator so the modal lands as a thin renderer.

What this module provides:

* :class:`FieldDiff` — one mutated leaf-value row the modal
  renders ("prompt_template", left snippet, right snippet).
* :class:`StepDiff` — one step's diff (added / removed /
  modified / unchanged) carrying the field-level breakdown.
* :class:`MetadataDiff` — top-level chain-metadata diff
  (display_name, description, tags, task_description).
* :class:`AgentDiff` — frozen aggregate (sides + steps +
  metadata + summary counters).
* :func:`diff_chains` — pure comparison. Accepts CARL
  `ReasoningChain` objects, dicts (the SDK's
  `get_chain_dict()` shape), or any duck-typed object exposing
  `steps` + metadata.
* :func:`fetch_agent_diff` — async helper that loads both
  chains via `memory.client.get_chain_dict(...)` and projects.

Step identity rule: steps are paired by ``number``. A step that
exists on only one side surfaces as ``added`` or ``removed``.
Re-numbered steps (semantically the same step but moved in
order) currently appear as one removal + one addition — the
modal can render those side-by-side, but matching them is
left to a future "fuzzy step pairing" enhancement.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AgentDiffError(RuntimeError):
    """Raised when the diff helpers fail — unreachable Memory,
    missing chain on one side, or a malformed chain payload.
    The modal catches this and shows a toast."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


DiffKind = Literal["added", "removed", "modified", "unchanged"]
"""Per-step diff kind. ``added`` = only on the right side;
``removed`` = only on the left; ``modified`` = on both with
field-level differences; ``unchanged`` = byte-equal on both."""


@dataclass(frozen=True)
class FieldDiff:
    """One leaf-value difference inside a step.

    The modal renders these as side-by-side rows. ``field`` is
    the dotted path inside the step's content (e.g.
    ``"config.prompt_template"``); ``left_value`` and
    ``right_value`` are the two values. ``None`` on a side means
    "key absent" — distinct from a present-but-falsy value
    (the modal can colour-code "missing" vs. "empty string").
    """

    field: str
    left_value: Any = None
    right_value: Any = None
    left_present: bool = True
    right_present: bool = True


@dataclass(frozen=True)
class StepDiff:
    """One step's diff entry.

    Frozen so it flows through Textual messages without
    defensive copies. ``number`` is the canonical pairing key;
    ``title_*`` / ``step_type_*`` are surfaced for header
    rendering even when the step is otherwise unchanged.
    """

    number: int
    kind: DiffKind
    title_left: Optional[str] = None
    title_right: Optional[str] = None
    step_type_left: Optional[str] = None
    step_type_right: Optional[str] = None
    fields: tuple[FieldDiff, ...] = ()

    @property
    def is_change(self) -> bool:
        """``True`` when the modal should render an indicator —
        any kind besides ``unchanged``."""
        return self.kind != "unchanged"

    @property
    def label(self) -> str:
        """Best-effort title for the modal header. Right wins on
        modified rows so the "new state" is what's shown."""
        return self.title_right or self.title_left or f"Step {self.number}"


@dataclass(frozen=True)
class MetadataDiff:
    """Top-level chain-metadata diff (display_name, description,
    tags, task_description). Surfaces as a header section above
    the per-step rows."""

    fields: tuple[FieldDiff, ...] = ()
    added_tags: tuple[str, ...] = ()
    removed_tags: tuple[str, ...] = ()

    @property
    def has_changes(self) -> bool:
        return bool(self.fields) or bool(self.added_tags) or bool(self.removed_tags)


@dataclass(frozen=True)
class AgentDiff:
    """Frozen aggregate for the diff modal.

    `left_*` / `right_*` describe the two sides for the modal
    header; `steps` carries the per-step diff rows in canonical
    order (numbers ascending). `metadata` is the chain-level
    diff. `summary` is convenience counters for the footer
    ("3 added · 1 removed · 2 modified").
    """

    left_entity_id: str = ""
    right_entity_id: str = ""
    left_label: str = ""
    right_label: str = ""
    steps: tuple[StepDiff, ...] = ()
    metadata: MetadataDiff = field(default_factory=MetadataDiff)

    @property
    def added_steps(self) -> int:
        return sum(1 for s in self.steps if s.kind == "added")

    @property
    def removed_steps(self) -> int:
        return sum(1 for s in self.steps if s.kind == "removed")

    @property
    def modified_steps(self) -> int:
        return sum(1 for s in self.steps if s.kind == "modified")

    @property
    def unchanged_steps(self) -> int:
        return sum(1 for s in self.steps if s.kind == "unchanged")

    @property
    def has_changes(self) -> bool:
        """``True`` when the modal should render anything other
        than a "no differences" empty state."""
        return self.metadata.has_changes or any(
            s.is_change for s in self.steps
        )

    def format_summary(self) -> str:
        """One-line summary the modal can pipe into the footer."""
        parts: list[str] = []
        if self.added_steps:
            parts.append(f"+{self.added_steps}")
        if self.removed_steps:
            parts.append(f"-{self.removed_steps}")
        if self.modified_steps:
            parts.append(f"~{self.modified_steps}")
        if not parts:
            return "no differences"
        return " · ".join(parts) + f" of {len(self.steps)} steps"


# ---------------------------------------------------------------------------
# Pure projection
# ---------------------------------------------------------------------------


_METADATA_FIELDS: tuple[str, ...] = (
    "display_name",
    "description",
    "task_description",
)
"""Top-level CARE metadata fields the diff includes. ``tags``
gets its own dedicated set-diff via ``added_tags`` /
``removed_tags`` so the modal can render add/remove badges
explicitly."""


def diff_chains(
    left: Any,
    right: Any,
    *,
    left_entity_id: str = "",
    right_entity_id: str = "",
    left_label: str = "",
    right_label: str = "",
) -> AgentDiff:
    """Compute the side-by-side diff of two chains.

    The two arguments can be any of:

    * A CARL ``ReasoningChain`` object (uses ``.to_dict()``).
    * A plain dict matching the SDK's ``get_chain_dict()``
      shape: ``{steps: [...], metadata: {care: {...}}}`` (or
      a flatter ``{steps, display_name, ...}`` shape — both
      handled).
    * Any duck-typed object exposing ``steps`` and the CARE
      metadata accessors.

    Args:
        left: "Before" side (typically the older / first-
            selected agent).
        right: "After" side.
        left_entity_id / right_entity_id: Stamped onto the
            result for modal-header rendering. Optional.
        left_label / right_label: Display labels (typically
            display_name from the library row). Optional —
            falls back to the projected metadata.

    Returns:
        :class:`AgentDiff` ready for the modal to render.
    """
    left_payload = _coerce_chain_dict(left)
    right_payload = _coerce_chain_dict(right)

    left_meta = _read_metadata(left_payload)
    right_meta = _read_metadata(right_payload)

    label_left = left_label or _read_str(left_meta, "display_name") or "left"
    label_right = right_label or _read_str(right_meta, "display_name") or "right"

    metadata_diff = _diff_metadata(left_meta, right_meta)
    step_diffs = _diff_steps(
        _read_steps(left_payload), _read_steps(right_payload)
    )

    return AgentDiff(
        left_entity_id=left_entity_id,
        right_entity_id=right_entity_id,
        left_label=label_left,
        right_label=label_right,
        steps=tuple(step_diffs),
        metadata=metadata_diff,
    )


# ---------------------------------------------------------------------------
# Step diff
# ---------------------------------------------------------------------------


def _diff_steps(
    left_steps: Iterable[dict[str, Any]],
    right_steps: Iterable[dict[str, Any]],
) -> list[StepDiff]:
    """Pair steps by ``number`` and produce one
    :class:`StepDiff` per union member."""
    left_by_num: dict[int, dict[str, Any]] = {
        _step_number(s): _normalise_step(s) for s in left_steps
    }
    right_by_num: dict[int, dict[str, Any]] = {
        _step_number(s): _normalise_step(s) for s in right_steps
    }
    numbers = sorted(set(left_by_num) | set(right_by_num))

    diffs: list[StepDiff] = []
    for n in numbers:
        left_step = left_by_num.get(n)
        right_step = right_by_num.get(n)
        if left_step is None and right_step is not None:
            diffs.append(
                StepDiff(
                    number=n,
                    kind="added",
                    title_right=_read_str(right_step, "title"),
                    step_type_right=(
                        _read_str(right_step, "step_type")
                        or _read_str(right_step, "type")
                    ),
                    fields=_field_diffs_for_added(right_step),
                )
            )
        elif right_step is None and left_step is not None:
            diffs.append(
                StepDiff(
                    number=n,
                    kind="removed",
                    title_left=_read_str(left_step, "title"),
                    step_type_left=(
                        _read_str(left_step, "step_type")
                        or _read_str(left_step, "type")
                    ),
                    fields=_field_diffs_for_removed(left_step),
                )
            )
        else:
            assert left_step is not None and right_step is not None
            field_diffs = _field_diffs(left_step, right_step)
            kind: DiffKind = "modified" if field_diffs else "unchanged"
            diffs.append(
                StepDiff(
                    number=n,
                    kind=kind,
                    title_left=_read_str(left_step, "title"),
                    title_right=_read_str(right_step, "title"),
                    step_type_left=(
                        _read_str(left_step, "step_type")
                        or _read_str(left_step, "type")
                    ),
                    step_type_right=(
                        _read_str(right_step, "step_type")
                        or _read_str(right_step, "type")
                    ),
                    fields=field_diffs,
                )
            )
    return diffs


def _field_diffs(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[FieldDiff, ...]:
    """Walk both step dicts and emit a :class:`FieldDiff` for
    every leaf-value mismatch. Nested dicts surface as dotted
    paths (``config.prompt_template``). Lists are compared by
    serialised equality — full per-element list diffing is
    overkill for the modal's current scope."""
    diffs: list[FieldDiff] = []
    _walk(left, right, prefix="", out=diffs)
    return tuple(diffs)


def _walk(
    left: Any, right: Any, *, prefix: str, out: list[FieldDiff]
) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        for k in keys:
            path = f"{prefix}.{k}" if prefix else k
            if k not in left:
                out.append(
                    FieldDiff(
                        field=path,
                        right_value=right[k],
                        left_present=False,
                    )
                )
            elif k not in right:
                out.append(
                    FieldDiff(
                        field=path,
                        left_value=left[k],
                        right_present=False,
                    )
                )
            else:
                _walk(left[k], right[k], prefix=path, out=out)
    else:
        if _values_equal(left, right):
            return
        out.append(
            FieldDiff(field=prefix or "value", left_value=left, right_value=right)
        )


def _values_equal(a: Any, b: Any) -> bool:
    """Equality with list-of-dicts canonicalisation: two lists of
    step-shaped dicts compare equal when their sorted-by-keys
    JSON projection matches. Falls back to native ``==``."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for av, bv in zip(a, b):
            if not _values_equal(av, bv):
                return False
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_values_equal(a[k], b[k]) for k in a)
    return a == b


def _field_diffs_for_added(step: dict[str, Any]) -> tuple[FieldDiff, ...]:
    """Project every leaf of an "added" step as a `FieldDiff`
    with `left_present=False`. Lets the modal render the
    "new fields" panel uniformly with the modified case."""
    out: list[FieldDiff] = []
    _walk({}, step, prefix="", out=out)
    return tuple(out)


def _field_diffs_for_removed(step: dict[str, Any]) -> tuple[FieldDiff, ...]:
    out: list[FieldDiff] = []
    _walk(step, {}, prefix="", out=out)
    return tuple(out)


# ---------------------------------------------------------------------------
# Metadata diff
# ---------------------------------------------------------------------------


def _diff_metadata(
    left: dict[str, Any], right: dict[str, Any]
) -> MetadataDiff:
    fields: list[FieldDiff] = []
    for name in _METADATA_FIELDS:
        left_value = left.get(name)
        right_value = right.get(name)
        if not _values_equal(left_value, right_value):
            fields.append(
                FieldDiff(
                    field=name,
                    left_value=left_value,
                    right_value=right_value,
                    left_present=name in left,
                    right_present=name in right,
                )
            )

    left_tags = _read_tag_set(left)
    right_tags = _read_tag_set(right)
    added = tuple(sorted(right_tags - left_tags))
    removed = tuple(sorted(left_tags - right_tags))

    return MetadataDiff(
        fields=tuple(fields),
        added_tags=added,
        removed_tags=removed,
    )


def _read_tag_set(meta: dict[str, Any]) -> frozenset[str]:
    tags = meta.get("tags")
    if not isinstance(tags, (list, tuple)):
        return frozenset()
    return frozenset(str(t) for t in tags if isinstance(t, str))


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------


async def fetch_agent_diff(
    memory: Any,
    left_entity_id: str,
    right_entity_id: str,
    *,
    left_label: str = "",
    right_label: str = "",
    channel: str = "latest",
    timeout: float = 10.0,
) -> AgentDiff:
    """Load both chains via Memory + project into a frozen diff.

    Uses ``memory.client.get_chain_dict(entity_id, channel)`` —
    the raw-content accessor — so the diff doesn't depend on
    CARL chain reconstruction. Both fetches run concurrently
    via :func:`asyncio.gather` (per-call timeout shared via
    the same `wait_for` deadline so a hung server hits one
    timeout, not two).

    Args:
        memory: A `CareMemory`-like facade exposing
            `.client.get_chain_dict(entity_id, channel)`.
        left_entity_id / right_entity_id: Two saved chains to
            compare. Must be non-empty; passing the same id
            twice is a no-op equivalent to "identity diff"
            (every step `unchanged`).
        left_label / right_label: Optional display labels.
        channel: Memory channel to read (default ``"latest"``).
        timeout: Per-call deadline.

    Returns:
        :class:`AgentDiff`.

    Raises:
        AgentDiffError: Empty entity_id on either side, missing
            SDK method, timeout, HTTP failure, or one of the
            chains comes back as None ("not found").
    """
    if not left_entity_id:
        raise AgentDiffError("left_entity_id is required")
    if not right_entity_id:
        raise AgentDiffError("right_entity_id is required")

    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    fn = getattr(client, "get_chain_dict", None) if client else None
    if not callable(fn):
        raise AgentDiffError(
            "memory facade does not expose client.get_chain_dict()"
        )

    start = time.monotonic()
    try:
        left_payload, right_payload = await asyncio.wait_for(
            asyncio.gather(
                asyncio.to_thread(fn, left_entity_id, channel),
                asyncio.to_thread(fn, right_entity_id, channel),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        latency = (time.monotonic() - start) * 1000
        raise AgentDiffError(
            f"chain fetch timed out after {timeout:.1f}s ({latency:.0f}ms elapsed)"
        ) from exc
    except AgentDiffError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AgentDiffError(
            f"chain fetch failed: {type(exc).__name__}: {exc}"
        ) from exc

    if left_payload is None:
        raise AgentDiffError(
            f"chain {left_entity_id!r} not found on channel {channel!r}"
        )
    if right_payload is None:
        raise AgentDiffError(
            f"chain {right_entity_id!r} not found on channel {channel!r}"
        )

    return diff_chains(
        left_payload,
        right_payload,
        left_entity_id=left_entity_id,
        right_entity_id=right_entity_id,
        left_label=left_label,
        right_label=right_label,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_chain_dict(chain: Any) -> dict[str, Any]:
    """Project a CARL `ReasoningChain` / dict / duck-typed
    object into a uniform dict shape."""
    if chain is None:
        return {}
    if isinstance(chain, dict):
        return dict(chain)
    to_dict = getattr(chain, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except TypeError:
            payload = to_dict()
        if isinstance(payload, dict):
            return dict(payload)
    model_dump = getattr(chain, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump(exclude_none=False)
        except TypeError:
            payload = model_dump()
        if isinstance(payload, dict):
            return dict(payload)
    # Last-ditch: build a dict from documented attributes.
    out: dict[str, Any] = {}
    for name in ("steps", "metadata", "display_name", "description", "tags"):
        value = getattr(chain, name, None)
        if value is not None:
            out[name] = value
    return out


def _read_steps(chain: dict[str, Any]) -> list[dict[str, Any]]:
    steps = chain.get("steps")
    if not isinstance(steps, list):
        return []
    return [s for s in steps if isinstance(s, dict)]


def _read_metadata(chain: dict[str, Any]) -> dict[str, Any]:
    """Read the CARE metadata block. Tries (in order):
    ``content.metadata.care``, ``metadata.care``,
    ``metadata`` (flat), and finally a flat-on-chain fallback."""
    if not chain:
        return {}
    content = chain.get("content")
    if isinstance(content, dict):
        meta = content.get("metadata")
        if isinstance(meta, dict):
            care = meta.get("care")
            if isinstance(care, dict):
                return dict(care)
            return dict(meta)
    meta_block = chain.get("metadata")
    if isinstance(meta_block, dict):
        care = meta_block.get("care")
        if isinstance(care, dict):
            return dict(care)
        return dict(meta_block)
    # Flat fallback: read fields directly off the chain dict.
    return {
        k: chain[k]
        for k in (*_METADATA_FIELDS, "tags")
        if k in chain
    }


def _normalise_step(step: dict[str, Any]) -> dict[str, Any]:
    """Drop transient/runtime-only fields that shouldn't show up
    as diff churn (matches the documented rebuild contract in
    PREPARE.md §5.8)."""
    if not isinstance(step, dict):
        return {}
    return {
        k: v
        for k, v in step.items()
        if k not in {"sub_chain", "agents", "base_step", "metrics", "cache"}
    }


def _step_number(step: dict[str, Any]) -> int:
    if not isinstance(step, dict):
        return 0
    raw = step.get("number")
    if isinstance(raw, int):
        return raw
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _read_str(obj: dict[str, Any] | Any, name: str) -> str:
    if isinstance(obj, dict):
        value = obj.get(name)
    else:
        value = getattr(obj, name, None)
    return value if isinstance(value, str) else ""


__all__ = [
    "AgentDiff",
    "AgentDiffError",
    "DiffKind",
    "FieldDiff",
    "MetadataDiff",
    "StepDiff",
    "diff_chains",
    "fetch_agent_diff",
]
