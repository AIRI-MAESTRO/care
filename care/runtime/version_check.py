"""Daily check for a newer ``maestro-care`` release on PyPI.

CARE ships to PyPI as ``maestro-care`` (import package ``care``, CLI command
``maestro``). The chat screen surfaces an available update in the input-hint
strip below the prompt. Everything here is best-effort: any failure (offline,
malformed JSON, missing metadata) yields ``None`` and the UI silently keeps
its normal hints.

The PyPI request is cached for one calendar day under
``~/.cache/care/version_check.json`` so repeat launches add no latency and
don't hammer the index (only *successful* fetches are cached — a transient
failure is retried on the next launch rather than suppressed for the day).

Set ``CARE_UPDATE_CHECK=0`` (or ``false`` / ``no`` / ``off``) to disable the
check entirely — useful on air-gapped machines or in CI.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import date
from importlib.metadata import PackageNotFoundError, version as _dist_version

from care.runtime.user_paths import CARE_CACHE_DIR

#: PyPI distribution name (the import package is ``care``).
PYPI_PACKAGE = "maestro-care"
_CACHE_PATH = CARE_CACHE_DIR / "version_check.json"
_PYPI_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE}/json"


def _enabled() -> bool:
    """False when the user opted out via ``CARE_UPDATE_CHECK``."""
    raw = (os.environ.get("CARE_UPDATE_CHECK") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def installed_version() -> str | None:
    """The running ``maestro-care`` version, or ``None`` when uninstalled
    (e.g. running straight from a source tree without metadata)."""
    try:
        return _dist_version(PYPI_PACKAGE)
    except PackageNotFoundError:
        return None


def _parse_version(value: str) -> tuple[int, ...]:
    """Split ``0.1.12`` into ``(0, 1, 12)`` for numeric (not lexicographic)
    comparison; stops at the first non-numeric chunk so a pre-release suffix
    degrades to its release part."""
    parts: list[int] = []
    for chunk in re.split(r"[.+\-]", (value or "").strip()):
        m = re.match(r"\d+", chunk)
        if not m:
            break
        parts.append(int(m.group()))
    return tuple(parts) or (0,)


def is_newer(candidate: str, current: str) -> bool:
    """True when version ``candidate`` is strictly newer than ``current``."""
    a, b = _parse_version(candidate), _parse_version(current)
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


def _today() -> str:
    return date.today().isoformat()


def _read_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_cache(latest: str) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(
            json.dumps({"date": _today(), "latest": latest}), encoding="utf-8"
        )
    except OSError:
        pass


def latest_version(*, timeout: float = 3.0) -> str | None:
    """Latest ``maestro-care`` on PyPI, cached for one calendar day.

    Network happens at most once per day on success; same-day calls read the
    cache. Returns ``None`` on any failure (offline, timeout, malformed JSON),
    and a failure is *not* cached so the next launch retries."""
    cache = _read_cache()
    if cache.get("date") == _today() and cache.get("latest"):
        return cache["latest"]
    try:
        req = urllib.request.Request(_PYPI_URL, headers={"User-Agent": "maestro-care"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        latest = ((data.get("info") or {}).get("version") or "").strip() or None
    except Exception:  # noqa: BLE001 — a version check must never raise
        return None
    if latest:
        _write_cache(latest)
    return latest


def available_update(*, timeout: float = 3.0) -> str | None:
    """Return the latest version string when a newer ``maestro-care`` is on
    PyPI than the running one, else ``None``.

    Safe to call from a worker thread; never raises. Honours the
    ``CARE_UPDATE_CHECK`` opt-out."""
    if not _enabled():
        return None
    current = installed_version()
    if not current:
        return None
    latest = latest_version(timeout=timeout)
    if latest and is_newer(latest, current):
        return latest
    return None
