"""Sandbox output mediation (TODO §6.2 P2).

Skills declare their workspace contract by writing artifacts into
``workspace/out/``. Before CARE merges those files back into the
host (copying to the user's chosen destination, attaching to a
memory_card, etc.) it scans the tree for content that's surprising
for what a skill is "supposed" to produce:

- **Executable binaries** — ELF / Mach-O / PE / shebang files
  flagged at ``severity="blocked"``. A data-extraction skill
  has no business writing a binary; surface it before the user
  trusts it.
- **Suspicious scripts** — text files referencing network egress
  primitives (`curl`, `wget`, `nc`, hardcoded URLs) at
  ``severity="warning"``. Could be a legitimate result file
  (a report containing URLs) — but the user should see them
  before they're forwarded anywhere.
- **Symlinks pointing outside the workspace** — a sandbox-escape
  attempt; always ``severity="blocked"``.
- **Empty `out/`** — informational; the skill wrote nothing.

The scanner returns a structured :class:`MediationReport` the
caller (CARE's ExecutionScreen, eventually) renders as a list with
per-finding actions: ``"accept anyway"`` / ``"discard"`` / ``"view
diff"``. Nothing here decides for the user — we surface signal,
not policy.

Pure file-IO module: no upstream dependencies, no async. Tests
use ``tmp_path`` to drive every code path against real files.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Severity = Literal["info", "warning", "blocked"]
"""Three-level severity ladder.

* ``info``: noteworthy but expected (empty output dir, very small
  text files).
* ``warning``: surface to the user before merging (script with
  URLs, large binary).
* ``blocked``: refuse to merge by default (raw executable,
  workspace-escape symlink). Caller can still override via the UI
  but the default is "drop".
"""

FindingKind = Literal[
    "executable_bit",
    "binary_content",
    "shebang_script",
    "network_token",
    "symlink_escape",
    "empty_out",
    "large_file",
]
"""Canonical kinds the scanner emits. Add to this list when adding
new heuristics — the LibraryScreen / ExecutionScreen pattern-match
on it for icons / colors."""


@dataclass(frozen=True)
class OutputFinding:
    """One heuristic match in ``workspace/out/``.

    Frozen so multiple findings can be dropped into reports without
    defensive copies.
    """

    severity: Severity
    kind: FindingKind
    path: str
    """Workspace-relative path of the offending file
    (``"out/report.bin"``)."""
    detail: str = ""
    """Short human-readable explanation. Renders directly in the
    TUI; keep one line, no markup."""
    bytes_inspected: int = 0
    """How much of the file we actually read. ``0`` for findings
    that didn't need content inspection (executable bit, symlink
    escape)."""


@dataclass(frozen=True)
class MediationReport:
    """Aggregate of a single ``out/`` directory scan."""

    findings: tuple[OutputFinding, ...] = field(default_factory=tuple)
    total_files: int = 0
    scanned_bytes: int = 0

    @property
    def is_clean(self) -> bool:
        """``True`` when no finding is at warning or blocked
        severity. ``info``-only reports are clean for merge-back."""
        return not any(f.severity in ("warning", "blocked") for f in self.findings)

    @property
    def has_blockers(self) -> bool:
        return any(f.severity == "blocked" for f in self.findings)

    def findings_by_severity(self, severity: Severity) -> tuple[OutputFinding, ...]:
        return tuple(f for f in self.findings if f.severity == severity)


# ---------------------------------------------------------------------------
# Magic bytes / regexes
# ---------------------------------------------------------------------------

_BINARY_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x7fELF", "ELF binary"),
    (b"MZ", "Windows PE/DOS executable"),
    (b"\xca\xfe\xba\xbe", "Mach-O fat binary"),
    (b"\xfe\xed\xfa\xce", "Mach-O 32-bit (little-endian)"),
    (b"\xfe\xed\xfa\xcf", "Mach-O 64-bit (little-endian)"),
    (b"\xce\xfa\xed\xfe", "Mach-O 32-bit (big-endian)"),
    (b"\xcf\xfa\xed\xfe", "Mach-O 64-bit (big-endian)"),
)
"""Recognised executable magic numbers. First-N-byte match."""

_SHEBANG_BYTES = b"#!"

_NETWORK_TOKEN_RE = re.compile(
    rb"(?:^|[^a-zA-Z0-9_])"
    rb"(curl|wget|nc|netcat|socat|ssh|telnet|"
    rb"https?://|ftp://|s3://|file:///etc/)",
    re.IGNORECASE | re.MULTILINE,
)
"""Heuristic regex for network egress primitives + suspicious
URI schemes in text files. False positives are expected on
documentation files; the scanner surfaces as ``warning``, never
``blocked``."""

DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024
"""Cap how much of any single file we read for content
inspection — defaults to 5 MiB. A binary's magic bytes are in
the first 16 bytes; a script's risky content is usually near the
top. Larger files get the read-cap warning logged but aren't
blocked outright."""


def scan_output_dir(
    workspace: Path,
    *,
    out_subdir: str = "out",
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> MediationReport:
    """Scan ``workspace/out_subdir`` recursively.

    Walks the tree, classifies every file, emits one or more
    :class:`OutputFinding` per file. Missing / empty output
    directory yields ``info`` finding(s) — not warnings — so the
    UI distinguishes "no output" from "output looks suspicious".

    Args:
        workspace: The sandbox workspace root. Must exist (the
            caller's :class:`SandboxHandle` always points at one).
        out_subdir: Subdir name to scan. Defaults to ``"out"``
            matching CARL's skill-runtime convention.
        max_file_bytes: Per-file read cap. Files larger than this
            emit a ``large_file`` warning AND get scanned within
            the cap (so a 20 MiB malicious binary doesn't escape
            magic-byte detection).
    """
    out_dir = workspace / out_subdir
    findings: list[OutputFinding] = []
    total_files = 0
    scanned_bytes = 0

    if not out_dir.exists():
        findings.append(
            OutputFinding(
                severity="info",
                kind="empty_out",
                path=out_subdir,
                detail=f"{out_subdir}/ directory does not exist",
            )
        )
        return MediationReport(tuple(findings), 0, 0)

    if not out_dir.is_dir():
        findings.append(
            OutputFinding(
                severity="warning",
                kind="empty_out",
                path=out_subdir,
                detail=f"{out_subdir} exists but is not a directory",
            )
        )
        return MediationReport(tuple(findings), 0, 0)

    workspace_resolved = workspace.resolve()
    saw_anything = False

    for entry in sorted(out_dir.rglob("*")):
        try:
            relative = entry.relative_to(workspace).as_posix()
        except ValueError:
            relative = str(entry)

        if entry.is_symlink():
            saw_anything = True
            findings.extend(
                _check_symlink(entry, relative, workspace_resolved)
            )
            continue

        if not entry.is_file():
            continue

        saw_anything = True
        total_files += 1
        size = entry.stat().st_size
        read_cap = min(size, max_file_bytes)
        with entry.open("rb") as fp:
            head = fp.read(read_cap)
        scanned_bytes += len(head)

        findings.extend(_classify_file(entry, relative, head, size, max_file_bytes))

    if not saw_anything:
        findings.append(
            OutputFinding(
                severity="info",
                kind="empty_out",
                path=out_subdir,
                detail=f"{out_subdir}/ is empty",
            )
        )

    return MediationReport(tuple(findings), total_files, scanned_bytes)


# ---------------------------------------------------------------------------
# Per-file classifiers
# ---------------------------------------------------------------------------


def _classify_file(
    path: Path,
    relative: str,
    head: bytes,
    full_size: int,
    max_file_bytes: int,
) -> list[OutputFinding]:
    out: list[OutputFinding] = []

    # Cap exceeded — warn so the user knows we didn't read the
    # whole thing. Always emit this BEFORE the content checks so a
    # 20 MiB malicious file shows both the cap warning + the
    # content blocker.
    if full_size > max_file_bytes:
        out.append(
            OutputFinding(
                severity="warning",
                kind="large_file",
                path=relative,
                detail=(
                    f"file is {full_size} bytes, scanner cap is "
                    f"{max_file_bytes}; only first {max_file_bytes} "
                    "bytes inspected"
                ),
                bytes_inspected=max_file_bytes,
            )
        )

    # POSIX executable bit — always blocked. A skill that wanted
    # to write a script for the user to run can drop the bit and
    # ask CARE for explicit chmod later.
    mode = path.stat().st_mode
    if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        out.append(
            OutputFinding(
                severity="blocked",
                kind="executable_bit",
                path=relative,
                detail=f"file mode {oct(mode & 0o777)} has executable bit set",
            )
        )

    # Binary magic bytes (ELF / PE / Mach-O) — blocked.
    matched_magic = _match_binary_magic(head)
    if matched_magic is not None:
        out.append(
            OutputFinding(
                severity="blocked",
                kind="binary_content",
                path=relative,
                detail=f"detected {matched_magic} magic bytes",
                bytes_inspected=len(head),
            )
        )

    # Shebang scripts — warn (could be a perfectly fine helper
    # script the skill wrote on purpose, but the user should know).
    if head.startswith(_SHEBANG_BYTES):
        first_line = head.split(b"\n", 1)[0][:200].decode(
            "utf-8", errors="replace"
        )
        out.append(
            OutputFinding(
                severity="warning",
                kind="shebang_script",
                path=relative,
                detail=f"shebang: {first_line!r}",
                bytes_inspected=len(head),
            )
        )

    # Network egress tokens — only meaningful for text-ish files.
    if _looks_textual(head):
        matches = _NETWORK_TOKEN_RE.findall(head[: min(len(head), 64 * 1024)])
        if matches:
            unique = sorted({m.decode("utf-8", errors="replace").lower() for m in matches})
            out.append(
                OutputFinding(
                    severity="warning",
                    kind="network_token",
                    path=relative,
                    detail="contains network-related tokens: " + ", ".join(unique[:5]),
                    bytes_inspected=len(head),
                )
            )

    return out


def _check_symlink(
    entry: Path, relative: str, workspace_resolved: Path
) -> list[OutputFinding]:
    """Symlinks must point inside the workspace. Anything else is
    a sandbox-escape attempt."""
    try:
        target = entry.resolve(strict=False)
    except OSError as exc:
        return [
            OutputFinding(
                severity="warning",
                kind="symlink_escape",
                path=relative,
                detail=f"could not resolve symlink: {exc}",
            )
        ]
    try:
        target.relative_to(workspace_resolved)
    except ValueError:
        return [
            OutputFinding(
                severity="blocked",
                kind="symlink_escape",
                path=relative,
                detail=f"symlink targets path outside workspace: {target}",
            )
        ]
    return []


def _match_binary_magic(head: bytes) -> str | None:
    for magic, label in _BINARY_MAGIC:
        if head.startswith(magic):
            return label
    return None


def _looks_textual(data: bytes) -> bool:
    """Cheap textual-vs-binary heuristic.

    Walks the first 512 bytes — if more than 30% are non-printable
    non-whitespace bytes, treat as binary. Adapted from the
    ``file(1)``-style heuristics used in many text-detection
    libraries; deliberately permissive so utf-8 + non-ASCII docs
    register as text."""
    if not data:
        return True
    sample = data[:512]
    text_chars = bytearray(
        {7, 8, 9, 10, 11, 12, 13, 27} | set(range(0x20, 0x7F)) | set(range(0x80, 0x100))
    )
    nontext = sum(1 for b in sample if b not in text_chars)
    return nontext / len(sample) < 0.30


# `os` is imported but currently unused at module scope — keep it as
# the future audit-log integration will need it for stat() info we
# don't yet surface.
_ = os


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "FindingKind",
    "MediationReport",
    "OutputFinding",
    "Severity",
    "scan_output_dir",
]
