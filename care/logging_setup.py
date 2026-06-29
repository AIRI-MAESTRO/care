"""Optional Python-side file logging for CARE.

Driven by env vars so the harness (Makefile, launcher scripts,
CI) can opt CARE into a structured app-side log without
touching code:

* ``CARE_LOG_FILE`` — destination path. Unset means no file log
  (callers can still wire their own handlers).
* ``CARE_LOG_LEVEL`` — root level for the attached handler
  (``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``).
  Default ``INFO``.

UI events stay in ``TEXTUAL_LOG`` (Textual's own internal log);
this module captures the parallel Python stream — ``care.*``
modules, ``httpx`` requests to Memory / Platform, MAGE / CARL
workers, asyncio warnings, etc. The two files together give a
post-mortem view of both halves of a run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# Loggers we deliberately quiet so the app log isn't drowned in
# framework noise. Textual's own events ride on TEXTUAL_LOG; we
# don't want to mirror them here.
_QUIET = {
    "textual": logging.WARNING,
    "markdown_it": logging.WARNING,
    "asyncio": logging.WARNING,
}

# Loggers we want at the configured level (or louder), since
# they carry the technical info the user is debugging.
_VERBOSE = (
    "care",
    "mmar_mage",
    "mmar_carl",
    "gigaevo",
    "httpx",
    "httpcore",
)


def configure_from_env() -> Path | None:
    """Attach a file handler to the root logger when
    ``CARE_LOG_FILE`` is set. Idempotent: a second call replaces
    the previous handler so the launcher can re-configure
    between runs without leaking file descriptors.

    Returns the resolved log path on success, ``None`` when the
    env var is unset or the file can't be opened.
    """
    raw = os.environ.get("CARE_LOG_FILE", "").strip()
    if not raw:
        return None

    level_name = os.environ.get("CARE_LOG_LEVEL", "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)

    path = Path(raw).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    except OSError:
        return None

    handler.setFormatter(_FORMATTER)
    handler.setLevel(level)
    handler.set_name("care-app-file")

    root = logging.getLogger()
    # Drop any previous file handler we attached so repeated
    # `configure_from_env()` calls don't accumulate.
    for existing in list(root.handlers):
        if getattr(existing, "name", "") == "care-app-file":
            root.removeHandler(existing)
            try:
                existing.close()
            except Exception:
                pass

    root.addHandler(handler)
    # Open the gate at the root so library loggers (httpx,
    # mmar_*) actually reach the handler. Without this the root
    # default of WARNING would drop INFO records on the floor.
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    for name in _VERBOSE:
        logging.getLogger(name).setLevel(level)
    for name, ceiling in _QUIET.items():
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET or logger.level < ceiling:
            logger.setLevel(ceiling)

    logging.getLogger("care").info(
        "app logging enabled: path=%s level=%s", path, logging.getLevelName(level),
    )
    return path


__all__ = ["configure_from_env"]
