"""Bulk chain import (TODO §3 P2).

CARE generates chains one at a time through MAGE in the normal
flow, but power users want to bring in batches at once — exported
chain JSON from a colleague, archived runs from CI, the
``generated_chains/*.json`` directory the ``care generate``
CLI scripts write into. This module implements the contract
behind the future ``care import ./generated_chains/*.json``
sub-command:

* Glob each input pattern.
* For each file: parse JSON, pull out the chain payload, validate
  it via :func:`care.validate_chain`, then save it to Memory.
* **Continue on per-file errors.** One broken file shouldn't
  block the rest of the batch — the report lists every failure
  by path so the user can fix and re-run just the failures.
* Optional ``dry_run=True`` does the parse + validation but
  skips the save, so CI can pre-flight a directory before
  letting a destructive `care import` proceed.

Accepted JSON shapes:

1. **Bare chain dict** — the raw `ReasoningChain.to_dict()`
   output. The file name (minus extension) becomes the library
   name; no domain / query / tags propagate.
2. **Wrapper dict** — ``{"chain": <chain-dict>, "name": "...",
   "query": "...", "tags": [...], "domain": "...",
   "when_to_use": "...", "author": "...", "channel": "..."}``.
   Every key except ``chain`` is optional and forwards verbatim
   to :meth:`CareMemory.save_chain`. Matches the shape
   :meth:`MAGEResult.to_care_dict` writes when CARE exports a
   freshly-generated chain to disk (MAGE §3.10).

The function returns a structured report rather than raising —
the TUI / CLI uses the per-entry status to render a summary
table at the end of the run.
"""

from __future__ import annotations

import glob as _glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from care.preflight import validate_chain

ImportStatus = Literal["imported", "validated", "failed"]
"""Per-entry outcome:

* ``imported`` — chain saved to Memory; ``entity_id`` populated.
* ``validated`` — chain parsed cleanly in ``dry_run`` mode; no
  save was attempted.
* ``failed`` — chain didn't make it past one of the steps;
  ``errors`` carries the reason(s).
"""


@dataclass(frozen=True)
class BulkImportEntry:
    """Per-file outcome from :func:`import_chains`.

    Frozen so the report can be passed around without defensive
    copies. ``errors`` is a list because a single file can fail
    at multiple steps (JSON parse → chain validate → save) and
    the user benefits from seeing the full chain of cause.
    """

    path: Path
    status: ImportStatus
    entity_id: str | None = None
    name: str | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BulkImportReport:
    """Aggregate report from :func:`import_chains`."""

    entries: tuple[BulkImportEntry, ...] = field(default_factory=tuple)

    @property
    def imported(self) -> tuple[BulkImportEntry, ...]:
        return tuple(e for e in self.entries if e.status == "imported")

    @property
    def validated(self) -> tuple[BulkImportEntry, ...]:
        return tuple(e for e in self.entries if e.status == "validated")

    @property
    def failed(self) -> tuple[BulkImportEntry, ...]:
        return tuple(e for e in self.entries if e.status == "failed")

    @property
    def all_ok(self) -> bool:
        """``True`` when every entry succeeded (imported OR
        validated in dry-run mode)."""
        return not self.failed

    def format_text(self) -> str:
        """Human-readable summary suitable for CLI output."""
        n_imp = len(self.imported)
        n_val = len(self.validated)
        n_fail = len(self.failed)
        lines = [
            f"bulk import: {n_imp} imported, {n_val} validated, "
            f"{n_fail} failed"
        ]
        for entry in self.failed:
            lines.append(f"  FAIL {entry.path}")
            for err in entry.errors:
                lines.append(f"    {err}")
        return "\n".join(lines)


def import_chains(
    patterns: list[str | Path],
    memory: Any = None,
    *,
    dry_run: bool = False,
    channel: str = "latest",
) -> BulkImportReport:
    """Import every chain matching ``patterns`` into Memory.

    Args:
        patterns: Glob patterns or literal paths. Each pattern
            is expanded via :func:`glob.glob` with
            ``recursive=True`` so ``"**/*.json"`` walks
            directories. Tilde-expanded.
        memory: Anything exposing
            ``save_chain(chain, *, name, query=..., domain=...,
            tags=..., when_to_use=..., author=..., channel=...)``.
            Required unless ``dry_run=True``.
        dry_run: When ``True``, validate every file without
            calling ``memory.save_chain``. ``memory`` may be
            ``None``. Useful for ``care import --dry-run`` to
            preview what would happen.
        channel: Default channel passed to ``save_chain`` for
            entries that don't carry their own ``channel`` field.
            Wrapper-dict ``channel`` always wins over this
            default.

    Returns:
        A :class:`BulkImportReport` with one entry per matched
        file. Never raises — every per-file failure surfaces as
        a ``failed`` entry on the report.

    Raises:
        ValueError: When ``dry_run=False`` and ``memory`` is
            ``None`` — the only configuration that genuinely
            can't proceed.
    """
    if not dry_run and memory is None:
        raise ValueError(
            "import_chains: memory is required when dry_run=False"
        )

    matched_paths = _expand_patterns(patterns)
    entries: list[BulkImportEntry] = []
    for path in matched_paths:
        entries.append(
            _import_one(path, memory=memory, dry_run=dry_run, channel=channel)
        )
    return BulkImportReport(entries=tuple(entries))


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------


def _import_one(
    path: Path,
    *,
    memory: Any,
    dry_run: bool,
    channel: str,
) -> BulkImportEntry:
    # 1. Read file.
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return BulkImportEntry(
            path=path,
            status="failed",
            errors=(f"read failed: {exc}",),
        )

    # 2. Parse JSON.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return BulkImportEntry(
            path=path,
            status="failed",
            errors=(
                f"invalid JSON at line {exc.lineno}, col {exc.colno}: {exc.msg}",
            ),
        )

    # 3. Pull out chain + save kwargs.
    try:
        chain_payload, save_kwargs = _split_wrapper(data, path=path)
    except _WrapperError as exc:
        return BulkImportEntry(
            path=path,
            status="failed",
            errors=(str(exc),),
        )

    # 4. Validate the chain.
    result = validate_chain(chain_payload, use_typed_steps=True)
    if not result.is_valid:
        return BulkImportEntry(
            path=path,
            status="failed",
            name=save_kwargs.get("name"),
            errors=tuple(result.parse_errors),
        )

    # 5. Save (or skip on dry run).
    if dry_run:
        return BulkImportEntry(
            path=path,
            status="validated",
            name=save_kwargs.get("name"),
        )

    try:
        entity_id = memory.save_chain(
            result.chain,
            channel=save_kwargs.pop("channel", channel),
            **save_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        return BulkImportEntry(
            path=path,
            status="failed",
            name=save_kwargs.get("name"),
            errors=(f"save_chain failed: {exc}",),
        )

    return BulkImportEntry(
        path=path,
        status="imported",
        entity_id=entity_id,
        name=save_kwargs.get("name"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _WrapperError(RuntimeError):
    """Raised when the wrapper-form payload is malformed."""


_SAVE_KWARG_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "query",
        "domain",
        "context_files",
        "mage_metadata",
        "tags",
        "when_to_use",
        "author",
        "entity_id",
        "channel",
    }
)
"""Keys we forward from the wrapper dict into ``save_chain``.
Any other top-level keys on the wrapper are ignored — keeps the
contract forwards-compatible (callers can stash extra metadata
without the import refusing the file)."""


def _split_wrapper(
    data: Any,
    *,
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(chain_dict, save_kwargs)`` from raw JSON.

    Accepts both the bare-chain and wrapper shapes; raises
    :class:`_WrapperError` for anything else.
    """
    if not isinstance(data, dict):
        raise _WrapperError(
            f"top-level JSON must be a dict; got {type(data).__name__}"
        )

    chain_payload = data.get("chain")
    if isinstance(chain_payload, dict):
        # Wrapper form.
        save_kwargs: dict[str, Any] = {
            k: v for k, v in data.items() if k in _SAVE_KWARG_KEYS
        }
        save_kwargs.setdefault("name", path.stem)
        return chain_payload, save_kwargs

    # Bare-chain form — `data` itself is the chain dict.
    if "steps" not in data:
        raise _WrapperError(
            "missing both `chain` (wrapper) and `steps` (bare-chain) keys"
        )
    return data, {"name": path.stem}


def _expand_patterns(patterns: list[str | Path]) -> list[Path]:
    """Glob-expand every pattern; tilde-expansion + recursive."""
    found: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        expanded = str(Path(str(pattern)).expanduser())
        for match in sorted(_glob.glob(expanded, recursive=True)):
            if match in seen:
                continue
            seen.add(match)
            mp = Path(match)
            if mp.is_file():
                found.append(mp)
    return found


__all__ = [
    "BulkImportEntry",
    "BulkImportReport",
    "ImportStatus",
    "import_chains",
]
