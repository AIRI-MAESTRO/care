"""Resumable runs — persisted in-flight job snapshot (TODO §1.2 P2).

CARE's TUI runs long jobs (MAGE generation, CARL execution,
Platform evolution) that can take many minutes. If the process
crashes mid-run, we want the next startup to notice "you had a
job in flight" and let the user resume it. This module owns the
persistence layer:

* :class:`RunState` — frozen snapshot of one in-flight job. The
  payload dict is per-kind so a chain run carries
  ``{"chain_entity_id": "...", "context_files": [...]}`` while a
  MAGE generation carries ``{"query": "...", "mode": "deep"}``.
  CARE's job-specific resume code reads what it needs out of
  ``payload``.
* :class:`RunStateStore` — atomic file-backed JSON store. One
  ``run_state.json`` per CARE installation (default
  ``~/.local/state/care/run_state.json``). Reads tolerate a
  missing / malformed / version-stale file by returning ``None``
  so a corrupted state file never blocks startup.

The store is intentionally **single-slot**: only one in-flight
job at a time. Background work like memory syncs doesn't write
here — only user-initiated runs that the user might want to
resume. The TaskRegistry already covers session-local
multi-task tracking (TODO §1.2 P1).

Atomic-write contract: every save goes through a tempfile in the
same directory followed by ``os.replace``, so a crash mid-write
leaves the prior state intact rather than producing a truncated
file. Matches the convention used by
:class:`care.sandbox.trust.SkillTrustStore` and
:class:`care.runtime.draft.DraftSession`.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
"""Bumped when :class:`RunState` gains/loses a field. Older
files are dropped on load (treated as "no state") so a CARE
upgrade doesn't choke on a snapshot from the prior version."""

DEFAULT_RUN_STATE_PATH = Path(
    "~/.local/state/care/run_state.json"
).expanduser()
"""User-global default. XDG-style location keeps state separate
from config (``~/.config/care``) and from caches (``~/.cache``)."""


@dataclass(frozen=True)
class RunState:
    """Snapshot of one in-flight CARE job.

    Frozen so the same instance can be passed across screens /
    log handlers without defensive copies. JSON round-trips
    through :meth:`to_dict` / :meth:`from_dict`.

    Fields:
        run_id: Stable identifier — typically the TaskRegistry
            id so resume can reattach to the in-memory record
            once the task is re-registered.
        kind: Job kind. Free-form string (no Literal here so
            future job kinds don't need a schema bump). CARE's
            resume dispatcher matches on this.
        label: Human-readable summary the resume prompt shows
            ("Generate weather agent" / "Run forecaster v3").
        started_at: Wall-clock seconds when the job started.
            Default-factory'd to ``time.time()`` so callers
            who don't supply it get a sensible value.
        payload: Per-kind resume data. CARE's chain-execution
            resume reads ``payload["chain_entity_id"]`` and
            re-primes a :class:`ReasoningContext` from it;
            MAGE resume reads ``payload["query"]`` and re-runs
            the generator. The store doesn't constrain the
            shape — each job kind owns its own contract.
        schema_version: Lets future CARE versions detect old
            snapshots. The store loads matching versions and
            drops mismatches.
    """

    run_id: str
    kind: str
    label: str
    started_at: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation. Tuples / sets become
        lists; everything else round-trips."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        """Inverse of :meth:`to_dict`. Raises ``KeyError`` /
        ``TypeError`` for malformed input — the store wraps
        these into a structured "drop the file" decision so
        callers never see raw decoder errors."""
        return cls(
            run_id=str(data["run_id"]),
            kind=str(data["kind"]),
            label=str(data["label"]),
            started_at=float(data.get("started_at", time.time())),
            payload=dict(data.get("payload") or {}),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


class RunStateStore:
    """File-backed single-slot run-state persistence.

    Thread-safe via an internal `threading.Lock` so the TUI
    thread + worker threads can both touch it without racing.
    All public methods are best-effort: malformed files surface
    as ``load() == None`` rather than an exception, matching
    CARE's "loud config errors, quiet state errors" stance
    (state can be regenerated; config can't).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        """Construct a store rooted at ``path``.

        Args:
            path: Where the JSON lives. ``None`` uses
                :data:`DEFAULT_RUN_STATE_PATH`. Tilde-expanded.
                Parent directories are created lazily on first
                save.
        """
        if path is None:
            self._path = DEFAULT_RUN_STATE_PATH
        else:
            self._path = Path(path).expanduser()
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Absolute path the store reads/writes."""
        return self._path

    def save(self, state: RunState) -> None:
        """Atomically persist ``state``.

        Writes go through a tempfile in the same directory + an
        ``os.replace`` so a crash mid-write leaves the prior
        snapshot intact instead of producing a truncated file.
        Creates parent directories as needed.
        """
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(state.to_dict(), indent=2, sort_keys=True)
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=".run_state-",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fp:
                    fp.write(payload)
                os.replace(tmp_name, self._path)
            except Exception:
                # Best-effort cleanup of the tempfile if replace
                # didn't get there.
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

    def load(self) -> RunState | None:
        """Return the persisted state, or ``None`` when there
        isn't a valid one.

        ``None`` covers four cases that all mean "nothing to
        resume": file missing, file unreadable, JSON malformed,
        schema_version mismatch. Distinguishing them in the
        signature would force every caller to handle four error
        types; CARE's resume flow only ever needs the binary
        "do I have something to offer the user?" answer.
        """
        with self._lock:
            if not self._path.exists():
                return None
            try:
                raw = self._path.read_text(encoding="utf-8")
            except OSError:
                return None
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if not isinstance(data, dict):
                return None
            version = data.get("schema_version")
            if version != SCHEMA_VERSION:
                return None
            try:
                return RunState.from_dict(data)
            except (KeyError, TypeError, ValueError):
                return None

    def clear(self) -> bool:
        """Remove the state file. Returns ``True`` when a file
        was removed, ``False`` when there was nothing to remove.

        Called on graceful run completion so a successful run
        doesn't leave a stale "resume?" prompt for next startup.
        Errors during unlink propagate — the caller wants to
        know if cleanup failed (e.g. filesystem read-only)
        even though that's rare.
        """
        with self._lock:
            try:
                self._path.unlink()
                return True
            except FileNotFoundError:
                return False


__all__ = [
    "DEFAULT_RUN_STATE_PATH",
    "RunState",
    "RunStateStore",
    "SCHEMA_VERSION",
]
