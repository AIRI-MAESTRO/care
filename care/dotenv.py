"""Tiny `.env` loader.

CARE doesn't depend on `python-dotenv` — for the small set of
``KEY=VALUE`` lines we accept here, a hand-rolled parser keeps
the dependency surface tight. Loaded early in
``care.cli.main`` so the resulting env vars are visible to
:class:`care.config.CareConfig` and the rest of the app.

Real shell env vars always win — :func:`load_env_file` only
populates keys that aren't already set in ``os.environ``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_log = logging.getLogger("care.dotenv")


def load_env_file(
    path: str | Path | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from ``path`` into ``os.environ``.

    Args:
        path: File to read. ``None`` searches ``./.env`` at the
            current working directory.
        override: When ``True``, replace existing env vars with
            the file's values. Default ``False`` — real shell
            env takes precedence over the file.

    Returns:
        Mapping of keys that were actually written. Empty when
        the file is absent or unreadable.
    """
    target = Path(path) if path is not None else Path.cwd() / ".env"
    if not target.exists() or not target.is_file():
        return {}
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning("could not read %s: %s", target, exc)
        return {}

    applied: dict[str, str] = {}
    for line_no, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        eq = line.find("=")
        if eq <= 0:
            _log.debug("ignoring malformed line %d in %s", line_no, target)
            continue
        key = line[:eq].strip()
        value = line[eq + 1:].strip()
        # Strip a single matching pair of surrounding quotes.
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in ('"', "'")
        ):
            value = value[1:-1]
        if not key:
            continue
        if not override and key in os.environ and os.environ[key] != "":
            continue
        os.environ[key] = value
        applied[key] = value
    if applied:
        _log.info(
            "loaded %d entries from %s (override=%s)",
            len(applied), target, override,
        )
    return applied


def _format_env_value(value: str) -> str:
    """Quote a ``.env`` value only when it carries characters the
    loader would otherwise mangle (spaces, ``#``, quotes)."""
    if value == "":
        return ""
    if any(c in value for c in (" ", "#", '"', "'", "\t")):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def update_env_file(
    updates: dict[str, str],
    path: str | Path | None = None,
    *,
    apply_to_environ: bool = True,
) -> Path:
    """Persist ``KEY=VALUE`` pairs into ``path`` (default ``./.env``).

    Existing keys are rewritten in place — comments, ordering, and
    untouched keys are preserved. New keys are appended under a
    generated section header. When ``apply_to_environ`` is ``True``
    (default) each pair is also written into :data:`os.environ`
    (overriding any prior value) so a same-process config reload
    reflects the edit immediately — without this the original
    ``.env`` values loaded at startup would keep masking the save,
    since ``CARE_*`` env vars outrank the on-disk ``config.toml``.

    Args:
        updates: Mapping of env keys to string values. Empty-string
            values are written as ``KEY=`` (an explicit blank).
        path: Destination file. ``None`` targets ``./.env`` in the
            current working directory; the file is created if absent.
        apply_to_environ: Mirror each pair into ``os.environ``.

    Returns:
        The resolved path the bytes landed at.

    Raises:
        OSError: When the file can't be read or written. The caller
            surfaces this so the save toast shows the real reason.
    """
    target = Path(path) if path is not None else Path.cwd() / ".env"

    existing_lines: list[str] = []
    if target.exists() and target.is_file():
        existing_lines = target.read_text(encoding="utf-8").splitlines()

    remaining = dict(updates)
    out_lines: list[str] = []
    for raw_line in existing_lines:
        stripped = raw_line.strip()
        body = stripped
        if body.startswith("export "):
            body = body[len("export "):].lstrip()
        eq = body.find("=")
        matched_key: str | None = None
        if eq > 0 and not stripped.startswith("#"):
            candidate = body[:eq].strip()
            if candidate in remaining:
                matched_key = candidate
        if matched_key is not None:
            value = remaining.pop(matched_key)
            out_lines.append(f"{matched_key}={_format_env_value(value)}")
        else:
            out_lines.append(raw_line)

    if remaining:
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")
        out_lines.append("# -- Updated from CARE settings ----------------")
        for key, value in remaining.items():
            out_lines.append(f"{key}={_format_env_value(value)}")

    target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    if apply_to_environ:
        for key, value in updates.items():
            os.environ[key] = value

    return target


__all__ = ["load_env_file", "update_env_file"]
