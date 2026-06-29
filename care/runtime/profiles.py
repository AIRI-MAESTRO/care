"""Multi-account profile discovery (TODO §6 P2).

CARE supports multiple credential profiles via TOML files
under ``~/.config/care/profiles/<name>.toml``. The
:env:`CARE_PROFILE` env variable selects which file gets
loaded; absent → uses the default ``~/.config/care/config.toml``.

This module is the data layer for the `/profile` screen:

* :func:`profiles_dir(*, config_dir=None)` resolves the
  profiles subdirectory (under :data:`CARE_CONFIG_DIR`).
* :func:`active_profile_name()` reads `CARE_PROFILE` —
  empty string when unset.
* :func:`list_profiles(*, config_dir=None)` discovers every
  ``*.toml`` in the profiles directory, returning
  ``ProfileInfo`` rows sorted by name.
* :func:`profile_path(name, *, config_dir=None)` resolves
  the TOML path for a named profile, with a strict
  whitelist that prevents path traversal.

The actual *loading* of a profile (rebuilding the
:class:`CareConfig` from the selected TOML) is intentionally
left for a follow-up — that needs careful surgery in the
config-precedence stack, and the §6 P2 deliverable is the
audit view.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from care.runtime.user_paths import CARE_CONFIG_DIR

_log = logging.getLogger("care.runtime.profiles")


PROFILES_SUBDIR = "profiles"


_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
"""Conservative whitelist. Letters, digits, underscores,
hyphens; max 64 chars. Rejects ``..``, ``/``, and other
filesystem-traversal vectors."""


@dataclass(frozen=True)
class ProfileInfo:
    """One profile entry the screen renders."""

    name: str
    path: Path
    size_bytes: int
    mtime: float


def profiles_dir(
    *, config_dir: Path | None = None,
) -> Path:
    root = (
        config_dir if config_dir is not None
        else CARE_CONFIG_DIR
    )
    return root / PROFILES_SUBDIR


def active_profile_name() -> str:
    """Read ``CARE_PROFILE`` and return the trimmed name.

    Empty string (or unset) means "use the default
    ``config.toml``". The screen treats both states the
    same — "no explicit selection".
    """
    return os.environ.get("CARE_PROFILE", "").strip()


def profile_path(
    name: str, *, config_dir: Path | None = None,
) -> Path:
    """Resolve the TOML path for ``name``.

    Raises:
        ValueError: When the name contains characters
            outside the safe whitelist.
    """
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid profile name {name!r}: must match "
            f"[A-Za-z0-9_-]{{1,64}}"
        )
    return profiles_dir(config_dir=config_dir) / f"{name}.toml"


def list_profiles(
    *, config_dir: Path | None = None,
) -> list[ProfileInfo]:
    """Discover every ``*.toml`` in the profiles directory.

    Sorted alphabetically by name. Files whose name doesn't
    match the safe whitelist are skipped + logged at
    WARNING — we don't want a malicious file to surface in
    the picker.
    """
    pdir = profiles_dir(config_dir=config_dir)
    if not pdir.is_dir():
        return []
    rows: list[ProfileInfo] = []
    for entry in pdir.iterdir():
        if not entry.is_file() or entry.suffix != ".toml":
            continue
        name = entry.stem
        if not _NAME_RE.match(name):
            _log.warning(
                "list_profiles: skipping %s — name fails "
                "whitelist",
                entry,
            )
            continue
        try:
            stat = entry.stat()
        except OSError as exc:
            _log.warning(
                "list_profiles: skip %s — stat failed: %s",
                entry, exc,
            )
            continue
        rows.append(
            ProfileInfo(
                name=name,
                path=entry,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
            )
        )
    rows.sort(key=lambda r: r.name.lower())
    return rows


__all__ = [
    "PROFILES_SUBDIR",
    "ProfileInfo",
    "active_profile_name",
    "list_profiles",
    "profile_path",
    "profiles_dir",
]
