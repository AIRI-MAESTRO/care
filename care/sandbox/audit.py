"""Sandbox audit log (TODO §6.2 P1).

Every call to :meth:`SandboxBackend.run` should leave a trace at
``~/.local/state/care/sandbox.log`` so the user (and CARE's
SettingsScreen viewer, later) can see exactly what each AgentSkill
did: which command, when, how long, exit code, content hashes of
stdout/stderr (so post-hoc diffs can detect "did this run produce
the same thing twice?"), and the relative paths the run wrote into
``workspace/out/``.

Format: one JSON object per line (``.jsonl``) — easy to ``tail -f``
in dev and easy for the future TUI viewer to parse incrementally.
Schema-versioned so a future shape change can refuse old files
without silently dropping fields.

The logger is **append-only and best-effort**: a write failure
(disk full, permission denied) logs to stderr and returns ``False``
rather than raising — the skill run itself shouldn't fail just
because the audit log is broken.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from care.sandbox.backend import RunResult, SandboxHandle

DEFAULT_AUDIT_PATH = Path("~/.local/state/care/sandbox.log").expanduser()
"""Conventional XDG-aligned location for the audit log."""

AUDIT_FORMAT_VERSION = 1
"""On-disk schema version. Bumped only when an old reader would
misinterpret new fields; additive fields don't bump."""


@dataclass(frozen=True)
class SandboxAuditEntry:
    """One line in the audit log.

    Frozen so the test fixtures + the future SettingsScreen viewer
    can pass these around without defensive copies. The on-disk
    line is exactly ``json.dumps(self.to_dict())`` — round-trips
    losslessly through :meth:`from_dict`.
    """

    timestamp: datetime
    backend_name: str
    skill_sha256: str
    cmd: tuple[str, ...]
    exit_code: int
    duration_seconds: float
    timed_out: bool
    stdout_sha256: str
    stderr_sha256: str
    network_enforced: bool
    files_written: tuple[str, ...] = field(default_factory=tuple)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": AUDIT_FORMAT_VERSION,
            "timestamp": self.timestamp.isoformat(),
            "backend_name": self.backend_name,
            "skill_sha256": self.skill_sha256,
            "cmd": list(self.cmd),
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "network_enforced": self.network_enforced,
            "files_written": list(self.files_written),
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SandboxAuditEntry":
        version = data.get("version")
        if version != AUDIT_FORMAT_VERSION:
            raise SandboxAuditError(
                f"unknown audit-log version {version!r}; expected "
                f"{AUDIT_FORMAT_VERSION}"
            )
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            backend_name=data["backend_name"],
            skill_sha256=data["skill_sha256"],
            cmd=tuple(data["cmd"]),
            exit_code=int(data["exit_code"]),
            duration_seconds=float(data["duration_seconds"]),
            timed_out=bool(data["timed_out"]),
            stdout_sha256=data["stdout_sha256"],
            stderr_sha256=data["stderr_sha256"],
            network_enforced=bool(data["network_enforced"]),
            files_written=tuple(data.get("files_written") or []),
            extras=dict(data.get("extras") or {}),
        )


class SandboxAuditError(RuntimeError):
    """Schema mismatch or corrupt entry when parsing the audit
    log. :meth:`SandboxAuditLogger.log_run` never raises — this is
    only raised on the read path."""


class SandboxAuditLogger:
    """Append-only JSON-line writer for sandbox run events.

    Construct with a path (default ``~/.local/state/care/sandbox.log``).
    Parent dirs are created on first write. Writes are best-effort:
    IO errors print to stderr and return ``False`` instead of
    raising, so a broken disk doesn't fail otherwise-successful
    skill runs.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        clock: Any = None,
    ) -> None:
        """Args:
        path: Override the log file location.
        clock: Override ``datetime.now`` for deterministic tests.
            Must be a zero-arg callable returning a timezone-aware
            :class:`datetime`.
        """
        self._path = path or DEFAULT_AUDIT_PATH
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_run(
        self,
        handle: SandboxHandle,
        cmd: list[str] | tuple[str, ...],
        result: RunResult,
        *,
        extras: dict[str, Any] | None = None,
    ) -> bool:
        """Persist one entry. Returns ``True`` on success, ``False``
        when the write failed (already reported to stderr).

        Auto-discovers files written under ``handle.workspace/out/``
        relative to the workspace root — the same convention CARL's
        skill runtimes use for skill outputs.
        """
        entry = self.build_entry(handle, cmd, result, extras=extras)
        try:
            self._append(entry)
            return True
        except OSError as exc:  # disk full, permission denied, etc.
            print(
                f"[care.sandbox.audit] failed to write {self._path}: {exc}",
                file=sys.stderr,
            )
            return False

    def build_entry(
        self,
        handle: SandboxHandle,
        cmd: list[str] | tuple[str, ...],
        result: RunResult,
        *,
        extras: dict[str, Any] | None = None,
    ) -> SandboxAuditEntry:
        """Build the entry without writing. Exposed for callers that
        want to log to a different sink (Langfuse, a test buffer)
        without reinventing the field set."""
        return SandboxAuditEntry(
            timestamp=self._clock(),
            backend_name=handle.backend_name,
            skill_sha256=handle.skill_sha256,
            cmd=tuple(cmd),
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
            timed_out=result.timed_out,
            stdout_sha256=_hash_bytes(result.stdout),
            stderr_sha256=_hash_bytes(result.stderr),
            network_enforced=result.network_enforced,
            files_written=_list_output_files(handle.workspace),
            extras=dict(extras or {}),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def tail(self, n: int = 50) -> list[SandboxAuditEntry]:
        """Return the last ``n`` entries (oldest-first within the
        slice). Useful for a SettingsScreen "Recent sandbox runs"
        panel. Returns ``[]`` when the log doesn't exist.

        Raises :class:`SandboxAuditError` if a line has an unknown
        schema version or is malformed JSON — the read path is
        strict because silently dropping corrupt entries hides
        problems."""
        if not self._path.exists():
            return []
        if n <= 0:
            return []
        with self._path.open() as fp:
            lines = fp.readlines()
        # Take the last n non-empty lines (file might end with
        # trailing newlines).
        tail_lines = [line for line in lines if line.strip()][-n:]
        return [self._parse_line(raw) for raw in tail_lines]

    def all_entries(self) -> list[SandboxAuditEntry]:
        """Parse every entry. ``[]`` when the log doesn't exist."""
        if not self._path.exists():
            return []
        with self._path.open() as fp:
            return [
                self._parse_line(raw)
                for raw in fp
                if raw.strip()
            ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(self, entry: SandboxAuditEntry) -> None:
        """Atomic-ish single-line append.

        ``open(..., "a")`` + a single ``write`` of one line is
        POSIX-atomic up to ``PIPE_BUF`` for non-readers; since each
        entry is a single JSON object and well under that limit on
        every modern OS, this is good enough for an audit log that
        a single CARE process owns.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Build the line in memory first so a serialisation error
        # (unlikely — we control the field set) doesn't leave a
        # half-written line in the file.
        payload = json.dumps(entry.to_dict(), sort_keys=True)
        line = payload + "\n"
        with self._path.open("a") as fp:
            fp.write(line)

    @staticmethod
    def _parse_line(raw: str) -> SandboxAuditEntry:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SandboxAuditError(
                f"could not parse audit-log line: {exc}"
            ) from exc
        return SandboxAuditEntry.from_dict(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_bytes(data: bytes) -> str:
    """SHA-256 hex of ``data``. Empty bytes hash to the canonical
    SHA-256-of-empty value — let the reader decide if that's
    meaningful (it often isn't, for runs with no stderr)."""
    return hashlib.sha256(data or b"").hexdigest()


def _list_output_files(workspace: Path) -> tuple[str, ...]:
    """Return relative paths under ``workspace/out/`` sorted
    deterministically. Missing ``out/`` → empty tuple. Symlinks
    are reported as their relative path; we don't resolve targets
    (a skill that wrote a symlink pointing outside ``out/`` is
    something the audit log should record verbatim)."""
    out_dir = workspace / "out"
    if not out_dir.is_dir():
        return ()
    paths: list[str] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() or path.is_symlink():
            paths.append(str(path.relative_to(workspace)))
    return tuple(paths)


def _async_time_ns() -> int:
    """Internal: monotonic-ish clock helper kept for future
    correlation IDs. Not used by the writer today; lives here so a
    future audit-log evolution doesn't need a new import."""
    return time.monotonic_ns()


__all__ = [
    "AUDIT_FORMAT_VERSION",
    "DEFAULT_AUDIT_PATH",
    "SandboxAuditEntry",
    "SandboxAuditError",
    "SandboxAuditLogger",
]
