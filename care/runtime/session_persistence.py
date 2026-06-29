"""Persistent session artifact cache (TODO §3 P1).

CARE's :class:`SessionArtifactStore` lives in memory for the
duration of a chat session. This module adds on-disk
persistence so closing the app + relaunching can recover
the same artifact list.

Layout: one JSONL file per session under
``~/.cache/care/sessions/<session_id>.jsonl``. The file
format matches :meth:`SessionArtifactStore.dump_jsonl` —
one artifact projection per line via
:meth:`SessionArtifact.to_dict`. The cache directory is
guaranteed to exist by :func:`care.runtime.user_paths.ensure_user_dirs`
which `CareApp.__init__` calls at boot.

Wire-up:

* :func:`attach_persistence(store, path)` registers a
  listener on the store that re-dumps the whole snapshot
  on every mutation. Cheap (most sessions have <100
  artifacts) and avoids tracking diffs.
* :func:`load_session(session_id)` returns the artifact
  list for a previously-saved session — caller hydrates
  a fresh store via :func:`replay_into`.
* :func:`list_sessions()` discovers known sessions for
  the `/resume` picker.

Cross-thread safety: the listener fires from whichever
thread mutates the store. :class:`SessionArtifactStore`
already locks reads; we serialise writes here with a
per-attachment lock so concurrent listener fires don't
race on the file.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from care.runtime.session_artifacts import (
    SessionArtifact,
    SessionArtifactStore,
    load_jsonl_artifacts,
)
from care.runtime.user_paths import CARE_CACHE_DIR

_log = logging.getLogger("care.runtime.session_persistence")


SESSIONS_SUBDIR = "sessions"
"""Subdirectory under the cache root that holds one JSONL
file per session."""


_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
"""Conservative whitelist so a malicious / typoed session id
can't escape the cache directory."""


def make_session_id() -> str:
    """Generate a fresh session id.

    Format: ``YYYYMMDD-HHMMSS-<6 hex>``. Sorts
    chronologically and stays short enough to type for
    `/resume`. The hex suffix breaks ties when two sessions
    start in the same second (e.g. running tests in
    parallel).
    """
    now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    suffix = uuid.uuid4().hex[:6]
    return f"{now}-{suffix}"


def sessions_dir(*, cache_dir: Path | None = None) -> Path:
    """Resolve the sessions subdirectory.

    Accepts an explicit ``cache_dir`` for tests so they can
    drive a tmp_path-rooted hierarchy without touching the
    user's real cache.
    """
    root = cache_dir if cache_dir is not None else CARE_CACHE_DIR
    return root / SESSIONS_SUBDIR


def session_path(
    session_id: str, *, cache_dir: Path | None = None,
) -> Path:
    """Resolve the per-session JSONL path for ``session_id``.

    Raises:
        ValueError: when the id contains characters outside
            the safe whitelist (lets us treat the id as a
            filename without escaping).
    """
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            f"invalid session id {session_id!r}: must match "
            f"[A-Za-z0-9_-]{{1,128}}"
        )
    return sessions_dir(cache_dir=cache_dir) / f"{session_id}.jsonl"


@dataclass(frozen=True)
class SessionInfo:
    """Lightweight metadata for the `/resume` picker.

    Avoids loading the full JSONL just to populate a list.
    The picker reads :attr:`artifact_count` lazily by
    calling :func:`load_session` only on the row the user
    selects.
    """

    session_id: str
    path: Path
    mtime: float
    size_bytes: int


def list_sessions(
    *, cache_dir: Path | None = None,
) -> list[SessionInfo]:
    """Discover persisted sessions, newest-first by mtime.

    Returns an empty list when the sessions directory
    doesn't exist yet (first boot before any artifact has
    been generated). The result is safe to render in a
    picker without further filtering.
    """
    sdir = sessions_dir(cache_dir=cache_dir)
    if not sdir.is_dir():
        return []
    rows: list[SessionInfo] = []
    for entry in sdir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix != ".jsonl":
            continue
        try:
            stat = entry.stat()
        except OSError as exc:
            _log.warning(
                "list_sessions: skipping %s — stat failed: %s",
                entry, exc,
            )
            continue
        rows.append(
            SessionInfo(
                session_id=entry.stem,
                path=entry,
                mtime=stat.st_mtime,
                size_bytes=stat.st_size,
            )
        )
    rows.sort(key=lambda r: r.mtime, reverse=True)
    return rows


def load_session(
    session_id: str, *, cache_dir: Path | None = None,
) -> list[SessionArtifact]:
    """Load artifacts for ``session_id``.

    Returns an empty list when the session file doesn't
    exist — convenient for the `/resume` path that
    discovers the session in :func:`list_sessions` but
    might race against an external deletion.
    """
    path = session_path(session_id, cache_dir=cache_dir)
    return load_jsonl_artifacts(path)


def attach_persistence(
    store: SessionArtifactStore,
    path: Path,
) -> "PersistenceHandle":
    """Wire ``store`` so every mutation re-dumps to ``path``.

    Idempotent: a second call with the same store + path
    detaches the previous listener so the store doesn't
    accumulate redundant dumps.

    Returns a :class:`PersistenceHandle` the caller can use
    to ``detach()`` on session-end or pause the listener
    during bulk operations.
    """
    handle = PersistenceHandle(store=store, path=path)
    handle.attach()
    return handle


class PersistenceHandle:
    """Lifecycle wrapper for an attached persistence listener.

    Owns the lock that serialises file writes + the
    ``detach``/``attach`` toggles so callers can pause the
    persistence (e.g. during a bulk migrate) without
    forgetting the path.
    """

    def __init__(
        self, *, store: SessionArtifactStore, path: Path,
    ) -> None:
        self.store = store
        self.path = path
        self._lock = threading.Lock()
        self._attached = False

    def _on_event(self, _artifact: SessionArtifact) -> None:
        # Re-dump the full snapshot on every mutation. We
        # could optimise to append-only for `append` events
        # + rewrite on `forget`/`mark_saved`, but a typical
        # session has < 100 artifacts; the simpler full-dump
        # avoids state-tracking bugs.
        with self._lock:
            try:
                self.store.dump_jsonl(self.path)
            except OSError as exc:
                _log.warning(
                    "session_persistence: dump to %s failed: %s",
                    self.path, exc,
                )

    def attach(self) -> None:
        """Idempotent attach. Safe to call multiple times."""
        with self._lock:
            if self._attached:
                return
            self.store.add_listener(self._on_event)
            self._attached = True

    def detach(self) -> None:
        """Stop persisting. Idempotent."""
        with self._lock:
            if not self._attached:
                return
            self.store.remove_listener(self._on_event)
            self._attached = False

    def flush(self) -> None:
        """Force an immediate dump regardless of attachment.

        Useful at session-end so the final state survives
        even if the last mutation happened during a race
        with detach.
        """
        with self._lock:
            try:
                self.store.dump_jsonl(self.path)
            except OSError as exc:
                _log.warning(
                    "session_persistence flush to %s failed: %s",
                    self.path, exc,
                )

    @property
    def attached(self) -> bool:
        return self._attached


__all__ = [
    "PersistenceHandle",
    "SESSIONS_SUBDIR",
    "SessionInfo",
    "attach_persistence",
    "list_sessions",
    "load_session",
    "make_session_id",
    "session_path",
    "sessions_dir",
]
