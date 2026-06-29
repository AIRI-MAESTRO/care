"""Log-file discovery for the in-app `/logs` viewer (§6 P2).

The app-side file logger
(:mod:`care.logging_setup.configure_from_env`) writes to
whatever path ``CARE_LOG_FILE`` points at. The `/logs`
screen needs a robust way to find that file:

1. ``CARE_LOG_FILE`` env — the canonical source. Set when the
   user runs ``make run LOG=1`` or any wrapper that opts into
   file logging.
2. The currently-attached `care-app-file` handler on the root
   logger — populated by `configure_from_env`. Catches the
   case where the env var was unset later but the handler
   was already wired.
3. A heuristic glob of common locations
   (``logs/care-app-*.log`` next to CWD, then
   ``~/.local/state/care/logs/care-app-*.log``) — used to
   surface historical logs even when the current session
   isn't writing one.

The screen also wants a level-filtered tail; this module
ships a pure helper :func:`tail_log_lines` that reads at
most ``max_lines`` lines off the end of a file with an
optional level filter (case-insensitive `LEVEL` token in
the line, matching the canonical
``%(asctime)s [%(levelname)s] %(name)s: …`` format from
:mod:`care.logging_setup`).
"""

from __future__ import annotations

import logging
import os
import re
from collections import deque
from pathlib import Path

_log = logging.getLogger("care.runtime.log_discovery")


LOG_LEVELS: tuple[str, ...] = (
    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
)
"""Order: ascending severity. Filtering by `INFO` shows
INFO + WARNING + ERROR + CRITICAL; `DEBUG` shows
everything."""


_LEVEL_TOKEN_RE = re.compile(r"\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]")
"""Matches the bracketed level token in the canonical
formatter (`%(asctime)s [%(levelname)s] %(name)s: …`)."""


def active_log_path() -> Path | None:
    """Best-effort resolution of the *currently active* app
    log file.

    Order:
    1. ``CARE_LOG_FILE`` env var (canonical).
    2. Attached `care-app-file` handler on the root logger
       (set by :func:`configure_from_env`).

    Returns ``None`` when nothing's writing a log.
    """
    raw = os.environ.get("CARE_LOG_FILE", "").strip()
    if raw:
        return Path(raw).expanduser()
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "name", "") == "care-app-file":
            base = getattr(handler, "baseFilename", "")
            if base:
                return Path(base)
    return None


def find_log_files(
    *, search_dirs: list[Path] | None = None,
) -> list[Path]:
    """Heuristic glob for historical `care-app-*.log` files.

    Searches the CWD ``logs/`` subdir + the XDG state
    directory by default. Returns a newest-first list (by
    file mtime).
    """
    if search_dirs is None:
        candidates = [
            Path.cwd() / "logs",
            Path("~/.local/state/care/logs").expanduser(),
        ]
    else:
        candidates = list(search_dirs)
    found: list[Path] = []
    for dir_ in candidates:
        if not dir_.is_dir():
            continue
        try:
            for entry in dir_.glob("care-app-*.log"):
                if entry.is_file():
                    found.append(entry)
        except OSError as exc:
            _log.warning(
                "find_log_files: skip %s — %s", dir_, exc,
            )
            continue
    # Deduplicate by resolved path (the env-pointed file may
    # also live in the CWD glob).
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    unique.sort(
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return unique


_MODULE_TOKEN_RE = re.compile(
    r"\[(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\]\s+([^:]+):",
)
"""Matches the logger-name segment between `]` and `:` in
the canonical formatter (`%(asctime)s [%(levelname)s]
%(name)s: %(message)s`)."""


def tail_log_lines(
    path: Path,
    *,
    max_lines: int = 500,
    level_floor: str | None = None,
    module_substr: str = "",
) -> list[str]:
    """Read the last ``max_lines`` lines of ``path``,
    optionally filtered to records at or above
    ``level_floor`` (e.g. ``"INFO"`` keeps INFO/WARNING/
    ERROR/CRITICAL) and/or to records whose logger name
    contains ``module_substr`` (e.g. ``"care.chat"``).

    Lines that don't carry a bracketed `[LEVEL]` token (e.g.
    multi-line tracebacks where only the first line carries
    the metadata) are kept verbatim when either filter is
    set — dropping them would orphan the trace from its
    header. The screen's render still shows them under the
    matched parent line. Once a level-bearing parent line is
    dropped, every continuation up to the next level-bearing
    line is dropped too.

    ``module_substr`` is case-insensitive substring match
    against the captured logger name. Empty / whitespace-only
    string disables the filter.

    Returns an empty list when the file doesn't exist.
    """
    if not path.exists():
        return []
    floor_idx: int | None = None
    if level_floor is not None:
        upper = level_floor.upper()
        if upper not in LOG_LEVELS:
            raise ValueError(
                f"unknown level: {level_floor!r}; expected "
                f"one of {LOG_LEVELS}"
            )
        floor_idx = LOG_LEVELS.index(upper)
    needle = (module_substr or "").strip().lower()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fp:
            buffer: deque[str] = deque(fp, maxlen=max_lines)
    except OSError:
        return []
    lines = [line.rstrip("\n") for line in buffer]
    if floor_idx is None and not needle:
        return lines
    out: list[str] = []
    keep = False
    for line in lines:
        match = _LEVEL_TOKEN_RE.search(line)
        if match is not None:
            level_ok = (
                floor_idx is None
                or LOG_LEVELS.index(match.group(1)) >= floor_idx
            )
            module_ok = True
            if needle:
                mod_match = _MODULE_TOKEN_RE.search(line)
                module_name = (
                    mod_match.group(1).strip().lower()
                    if mod_match else ""
                )
                module_ok = needle in module_name
            keep = level_ok and module_ok
        if keep:
            out.append(line)
    return out


__all__ = [
    "LOG_LEVELS",
    "active_log_path",
    "find_log_files",
    "tail_log_lines",
]
