"""First-run user-data directory setup (TODO §7 P0).

CARE writes to three XDG-style locations on the user's machine:

* ``~/.config/care/`` — `config.toml`, `theme.json`,
  `secrets.json`, `mcp_servers.toml`, `tools/`,
  `synthesized_tools/`. Editable hand-curated user state.
* ``~/.cache/care/`` — derived caches (LLM responses,
  Memory bundle thumbnails, etc.). Safe to wipe; CARE
  rebuilds on demand.
* ``~/.local/state/care/`` — resumable runs (run_state.json),
  draft sessions, skill trust ledger. Not user-edited, not
  safe to wipe mid-flight.

This module owns the first-boot guarantee that every one of
those directories exists. The TUI calls
:func:`ensure_user_dirs` once during ``CareApp.__init__`` so
later code (config writers, RunState, theme persistence, …)
can `mkdir(parents=True, exist_ok=True)` against a known-good
parent and only worry about its own file.

The contract is intentionally non-fatal: a permission
failure on (e.g.) ``~/.cache/`` shouldn't refuse boot
— the directory creator returns a structured
:class:`UserPathReport` that the wizard / SettingsScreen
surfaces as a friendly diagnostic. CARE keeps running; the
specific feature that needs the missing directory will
re-raise (or skip with a friendly toast) when invoked.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_log = logging.getLogger("care.runtime.user_paths")


CARE_CONFIG_DIR: Path = Path("~/.config/care").expanduser()
"""User-editable config root. See :mod:`care.config` for the
files that live here."""

CARE_CACHE_DIR: Path = Path("~/.cache/care").expanduser()
"""Derived-caches root. Safe to wipe between sessions."""

CARE_STATE_DIR: Path = Path("~/.local/state/care").expanduser()
"""In-flight run state. See :mod:`care.runtime.run_state`."""


@dataclass(frozen=True)
class UserPathResult:
    """One directory's setup outcome.

    Fields:
        kind: ``"config"`` / ``"cache"`` / ``"state"``.
        path: The directory we tried to create.
        ok: ``True`` iff the directory exists + is writable
            after the call.
        error: Failure reason, empty when ``ok=True``.
        existed: ``True`` when the directory already existed
            on entry (idempotent — no creation happened).
    """

    kind: str
    path: Path
    ok: bool
    error: str = ""
    existed: bool = False


@dataclass(frozen=True)
class UserPathReport:
    """Aggregate of all three per-directory results."""

    results: tuple[UserPathResult, ...] = field(default_factory=tuple)

    @property
    def all_ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def failures(self) -> tuple[UserPathResult, ...]:
        return tuple(r for r in self.results if not r.ok)

    def by_kind(self, kind: str) -> UserPathResult | None:
        for r in self.results:
            if r.kind == kind:
                return r
        return None

    def format_text(self) -> str:
        """Human-readable summary, one line per directory.

        Identical layout to :meth:`FirstRunReport.format_text`
        so the SettingsScreen / `care doctor` can stack both
        without restyling.
        """
        lines: list[str] = []
        for r in self.results:
            badge = "✓" if r.ok else "✗"
            line = f"{badge} {r.kind} ({r.path})"
            if r.error:
                line += f" — {r.error}"
            elif r.existed:
                line += " — already present"
            else:
                line += " — created"
            lines.append(line)
        return "\n".join(lines)


def ensure_user_dirs(
    *,
    config_dir: Path | None = None,
    cache_dir: Path | None = None,
    state_dir: Path | None = None,
) -> UserPathReport:
    """Create every CARE user-data directory if missing.

    Idempotent: re-runs are cheap (each directory's
    ``mkdir(parents=True, exist_ok=True)`` is a no-op when
    the path already exists), and the report flags whether
    the directory was created or already present.

    Each directory is created independently — one failure
    doesn't short-circuit the others, so a missing
    ``~/.cache/`` parent won't block ``~/.config/`` setup.

    Args:
        config_dir: Override for the config root. ``None``
            uses :data:`CARE_CONFIG_DIR`. Tests pass a
            ``tmp_path``-rooted directory.
        cache_dir: Override for the cache root.
        state_dir: Override for the state root.

    Returns:
        :class:`UserPathReport` aggregating per-directory
        outcomes. Caller decides how to surface failures —
        the TUI logs at WARNING and shows the report on
        SettingsScreen; the headless CLI prints it inline.
    """
    targets: list[tuple[str, Path]] = [
        ("config", config_dir if config_dir is not None else CARE_CONFIG_DIR),
        ("cache", cache_dir if cache_dir is not None else CARE_CACHE_DIR),
        ("state", state_dir if state_dir is not None else CARE_STATE_DIR),
    ]
    results: list[UserPathResult] = []
    for kind, path in targets:
        results.append(_ensure_one(kind, path))
    report = UserPathReport(results=tuple(results))
    if not report.all_ok:
        _log.warning(
            "user-paths setup partially failed: %s",
            "; ".join(
                f"{r.kind}={r.error}" for r in report.failures
            ),
        )
    return report


def _ensure_one(kind: str, path: Path) -> UserPathResult:
    """Create one directory + verify it's writable.

    Pulled out so the per-kind error path can also exercise
    the "already exists but not writable" branch — that's
    rare in practice but easy to test against a chmod'd
    tmp dir.
    """
    existed = path.exists()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return UserPathResult(
            kind=kind,
            path=path,
            ok=False,
            error=f"could not create {path}: {exc}",
            existed=existed,
        )
    if not path.is_dir():
        return UserPathResult(
            kind=kind,
            path=path,
            ok=False,
            error=f"{path} exists but is not a directory",
            existed=existed,
        )
    if not os.access(path, os.W_OK):
        return UserPathResult(
            kind=kind,
            path=path,
            ok=False,
            error=f"{path} is not writable by the current user",
            existed=existed,
        )
    return UserPathResult(
        kind=kind, path=path, ok=True, existed=existed,
    )


def collect_user_paths(
    *,
    config_dir: Path | None = None,
    cache_dir: Path | None = None,
    state_dir: Path | None = None,
) -> Iterable[Path]:
    """Iterate the resolved CARE directory paths in order.

    Lightweight helper for callers (e.g. `care doctor`) that
    only need to display the resolved paths without doing the
    full setup roundtrip.
    """
    yield config_dir if config_dir is not None else CARE_CONFIG_DIR
    yield cache_dir if cache_dir is not None else CARE_CACHE_DIR
    yield state_dir if state_dir is not None else CARE_STATE_DIR


__all__ = [
    "CARE_CACHE_DIR",
    "CARE_CONFIG_DIR",
    "CARE_STATE_DIR",
    "UserPathReport",
    "UserPathResult",
    "collect_user_paths",
    "ensure_user_dirs",
]
