"""Session artifact store (TODO §3 P0).

A **session artifact** is anything CARE produces during a chat
session that the user might want to inspect, persist, or export
later: generated CARL chains (primary), intermediate MAGE stage
payloads, tool / skill outputs from Ad-Hoc execution, dataset
eval rows (Production mode), LLM follow-up answers worth
keeping.

This module owns the in-session store + the projection helpers
the future :class:`care.screens.artifacts.ArtifactsScreen` will
read. It is deliberately data-layer only:

* No Textual / widget imports.
* No Memory / Platform calls — the screen / chat hook decides
  when to persist; this module just tracks ``saved_to_memory``
  + the resulting ``memory_entity_id`` once the save returns.
* No I/O at construction time. The optional JSON-Lines cache
  (TODO §3 P1) plugs in later via :meth:`SessionArtifactStore.dump_jsonl`
  / :func:`load_jsonl_artifacts`.

The store is append-only within a session and reset whenever
``/new`` or ``/clear`` runs. ChatScreen owns one instance; the
ArtifactsScreen (P0 later in §3) reads it via
``app.screen.artifact_store``.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal

_log = logging.getLogger("care.session_artifacts")


SessionArtifactKind = Literal[
    "chain",
    "stage_payload",
    "tool_output",
    "dataset_row",
    "synthesised_answer",
]
"""Discriminator for the artifact's payload shape.

* ``chain`` — a CARL chain dict (MAGE output, primary case).
* ``stage_payload`` — one MAGE stage projection (domain /
  step_plan / dag / critique / verification / refine).
* ``tool_output`` — a tool / AgentSkill execution result captured
  during an Ad-Hoc run.
* ``dataset_row`` — a Production-mode dataset entry (input +
  expected + score).
* ``synthesised_answer`` — the LLM-synthesised final reply built
  from multiple step outputs.
"""


_VALID_KINDS: frozenset[str] = frozenset(
    [
        "chain",
        "stage_payload",
        "tool_output",
        "dataset_row",
        "synthesised_answer",
    ]
)


def _utcnow() -> datetime:
    """Module-level seam so tests can monkeypatch
    ``care.runtime.session_artifacts._utcnow``."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """Short uuid4-hex token. 12 hex chars is plenty for a
    session-scoped collection — collision risk is negligible
    and the shorter id renders cleaner in the artifact list."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class SessionArtifact:
    """One in-session artifact.

    Frozen so listeners can cache instances without worrying
    about mutation. The store hands out fresh instances on
    every ``mark_saved`` update via :func:`dataclasses.replace`.
    """

    id: str
    kind: SessionArtifactKind
    created_at: datetime
    title: str
    summary: str
    payload: Any
    origin: dict[str, Any] = field(default_factory=dict)
    saved_to_memory: bool = False
    memory_entity_id: str | None = None
    # §3 P3 — tags the user (or generator) already attached
    # to the chain in-session. Surfaced via the artifact
    # store so `action_save_all_unsaved` can seed its
    # TagEditorModal with the UNION across unsaved chains
    # instead of starting from an empty chip set.
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable projection used by ``dump_jsonl`` +
        future export flows. ``created_at`` becomes an ISO-8601
        string; ``payload`` and ``origin`` are passed through
        with ``json.dumps(default=str)`` so non-serialisable
        nested objects (e.g. ``Path`` in ``origin``) degrade to
        their ``str()``."""
        return {
            "id": self.id,
            "kind": self.kind,
            "created_at": self.created_at.isoformat(),
            "title": self.title,
            "summary": self.summary,
            "payload": self.payload,
            "origin": dict(self.origin),
            "saved_to_memory": self.saved_to_memory,
            "memory_entity_id": self.memory_entity_id,
            "tags": list(self.tags),
        }


class SessionArtifactStoreError(RuntimeError):
    """Raised when the store rejects an operation — unknown
    artifact id on :meth:`SessionArtifactStore.mark_saved`,
    invalid ``kind``, etc. Caller surfaces as a toast / warning
    line rather than crashing the chat screen."""


# ---------------------------------------------------------------------------
# Listener contract
# ---------------------------------------------------------------------------


Listener = Callable[["SessionArtifact"], None]
"""Called with the newly-appended / updated artifact. Listeners
run in the thread that mutated the store; if a listener needs
the Textual event loop it should marshal via ``app.call_from_thread``.

The listener receives the fresh snapshot — never the previous
one — so handlers that only care about state transitions can
diff against their own cache."""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SessionArtifactStore:
    """Append-only collection of :class:`SessionArtifact`.

    Thread-safe (a :class:`threading.RLock` guards mutations) so
    background workers — MAGE poster, CARL streamer, dataset
    runner — can append from any thread. The chat screen reads
    on the UI thread; the lock keeps the snapshot consistent.

    The store deliberately exposes a small surface:

    * :meth:`append_*` — one helper per kind so callers don't
      have to remember the discriminator strings.
    * :meth:`mark_saved` — flip ``saved_to_memory`` + record
      the Memory entity_id once persistence returns.
    * :meth:`forget` — remove a single artifact (in-memory
      drop; does NOT touch Memory).
    * :meth:`clear` — reset the whole store. Fired on
      ``/new`` / ``/clear`` and when the session ends.
    * :meth:`list_artifacts` / :meth:`unsaved` — projections
      the screen / save-all flow read.
    * :meth:`add_listener` / :meth:`remove_listener` —
      pub/sub for the eventual screen badge ("3 artifacts •
      1 unsaved").
    """

    def __init__(self) -> None:
        self._artifacts: list[SessionArtifact] = []
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._artifacts)

    def __iter__(self) -> Iterator[SessionArtifact]:
        # Snapshot so iteration is lock-free after construction.
        with self._lock:
            return iter(list(self._artifacts))

    def __contains__(self, artifact_id: object) -> bool:
        if not isinstance(artifact_id, str):
            return False
        with self._lock:
            return any(a.id == artifact_id for a in self._artifacts)

    # ------------------------------------------------------------------
    # Append API
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        kind: SessionArtifactKind,
        title: str,
        summary: str,
        payload: Any,
        origin: dict[str, Any] | None = None,
        tags: Iterable[str] = (),
    ) -> SessionArtifact:
        """Append a new artifact and notify listeners.

        Returns the materialised :class:`SessionArtifact` so the
        caller can keep its id for a later :meth:`mark_saved` /
        :meth:`forget`. Raises :class:`SessionArtifactStoreError`
        on an invalid ``kind`` so a bug in the caller doesn't
        silently leak unknown discriminators into the screen.

        ``tags`` (§3 P3) lets a caller seed the
        :class:`SessionArtifact.tags` field so the save-all
        flow can compute the union of pre-attached tags across
        unsaved chains. Whitespace is stripped + empty entries
        dropped + duplicates removed (insertion order
        preserved).
        """
        if kind not in _VALID_KINDS:
            raise SessionArtifactStoreError(
                f"unknown artifact kind: {kind!r} "
                f"(expected one of {sorted(_VALID_KINDS)})"
            )
        cleaned_tags: list[str] = []
        for tag in tags:
            cleaned = str(tag).strip()
            if cleaned and cleaned not in cleaned_tags:
                cleaned_tags.append(cleaned)
        artifact = SessionArtifact(
            id=_new_id(),
            kind=kind,
            created_at=_utcnow(),
            title=title,
            summary=summary,
            payload=payload,
            origin=dict(origin or {}),
            tags=tuple(cleaned_tags),
        )
        with self._lock:
            self._artifacts.append(artifact)
        _log.debug(
            "appended artifact: id=%s kind=%s title=%r",
            artifact.id, artifact.kind, artifact.title,
        )
        self._notify(artifact)
        return artifact

    def append_chain(
        self,
        *,
        chain: Any,
        title: str,
        summary: str,
        origin: dict[str, Any] | None = None,
        tags: Iterable[str] = (),
    ) -> SessionArtifact:
        """Convenience wrapper for the most common case — a CARL
        chain dict produced by a MAGE generation. ``tags``
        (§3 P3) carries the pre-attached chain tags so the
        save-all flow can seed its TagEditorModal with the
        union across unsaved chains."""
        return self.append(
            kind="chain", title=title, summary=summary,
            payload=chain, origin=origin, tags=tags,
        )

    def append_stage_payload(
        self,
        *,
        stage: str,
        payload: Any,
        title: str | None = None,
        summary: str | None = None,
        origin: dict[str, Any] | None = None,
    ) -> SessionArtifact:
        """Convenience wrapper for a MAGE stage projection. The
        ``stage`` name lands in ``origin["stage"]`` so the
        screen can filter / group by stage without re-parsing
        the title."""
        merged_origin = dict(origin or {})
        merged_origin.setdefault("stage", stage)
        return self.append(
            kind="stage_payload",
            title=title or f"Agent chain generator {stage}",
            summary=summary or f"Stage payload from {stage}.",
            payload=payload,
            origin=merged_origin,
        )

    def append_synthesised_answer(
        self,
        *,
        answer: str,
        title: str | None = None,
        origin: dict[str, Any] | None = None,
    ) -> SessionArtifact:
        """Convenience wrapper for the Ad-Hoc final-answer
        synthesis output."""
        summary = answer.strip().splitlines()[0][:120] if answer.strip() else ""
        return self.append(
            kind="synthesised_answer",
            title=title or "Synthesised answer",
            summary=summary,
            payload=answer,
            origin=origin,
        )

    def append_tool_output(
        self,
        *,
        tool: str,
        output: Any,
        title: str | None = None,
        summary: str | None = None,
        origin: dict[str, Any] | None = None,
    ) -> SessionArtifact:
        """Convenience wrapper for a tool / AgentSkill output."""
        merged_origin = dict(origin or {})
        merged_origin.setdefault("tool", tool)
        return self.append(
            kind="tool_output",
            title=title or f"Tool: {tool}",
            summary=summary or f"Output from {tool}.",
            payload=output,
            origin=merged_origin,
        )

    def append_dataset_row(
        self,
        *,
        row: Any,
        title: str | None = None,
        summary: str | None = None,
        origin: dict[str, Any] | None = None,
    ) -> SessionArtifact:
        """Convenience wrapper for a Production-mode dataset row
        captured during ``/dataset add`` / ``/dataset run``."""
        return self.append(
            kind="dataset_row",
            title=title or "Dataset row",
            summary=summary or "",
            payload=row,
            origin=origin,
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mark_saved(
        self, artifact_id: str, *, memory_entity_id: str,
    ) -> SessionArtifact:
        """Flip ``saved_to_memory=True`` and record the Memory
        entity_id. Returns the updated snapshot. Raises
        :class:`SessionArtifactStoreError` when the id is
        unknown — that points at a bug in the caller (a save
        worker referencing a forgotten artifact)."""
        if not memory_entity_id:
            raise SessionArtifactStoreError(
                "mark_saved requires a non-empty memory_entity_id"
            )
        with self._lock:
            for idx, current in enumerate(self._artifacts):
                if current.id == artifact_id:
                    updated = replace(
                        current,
                        saved_to_memory=True,
                        memory_entity_id=memory_entity_id,
                    )
                    self._artifacts[idx] = updated
                    break
            else:
                raise SessionArtifactStoreError(
                    f"unknown artifact id: {artifact_id!r}"
                )
        self._notify(updated)
        return updated

    def forget(self, artifact_id: str) -> SessionArtifact:
        """Drop a single artifact from the store. Returns the
        removed snapshot. Raises
        :class:`SessionArtifactStoreError` when unknown.

        Note: deletion is in-memory only — the caller must
        separately decide whether to soft-delete the
        corresponding Memory entity (see
        :meth:`care.memory.CareMemory.delete_chain`)."""
        with self._lock:
            for idx, current in enumerate(self._artifacts):
                if current.id == artifact_id:
                    del self._artifacts[idx]
                    self._notify(current)
                    return current
        raise SessionArtifactStoreError(
            f"unknown artifact id: {artifact_id!r}"
        )

    def clear(self) -> int:
        """Reset the store. Returns the count dropped so the
        caller can log / toast ``"cleared N artifacts"``."""
        with self._lock:
            count = len(self._artifacts)
            self._artifacts.clear()
        if count:
            _log.info("session artifact store cleared (n=%d)", count)
        return count

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_artifacts(
        self,
        *,
        kind: SessionArtifactKind | None = None,
        saved: bool | None = None,
        newest_first: bool = True,
    ) -> list[SessionArtifact]:
        """Snapshot of artifacts with optional filters.

        Args:
            kind: filter by discriminator (``None`` returns all
                kinds).
            saved: filter by ``saved_to_memory`` (``None``
                returns both).
            newest_first: order. ``True`` (default) puts the
                most-recently-appended row first — the layout
                the future DataTable expects.
        """
        with self._lock:
            rows = list(self._artifacts)
        if kind is not None:
            rows = [a for a in rows if a.kind == kind]
        if saved is not None:
            rows = [a for a in rows if a.saved_to_memory is saved]
        if newest_first:
            rows.reverse()
        return rows

    def get(self, artifact_id: str) -> SessionArtifact:
        """Return one artifact by id. Raises
        :class:`SessionArtifactStoreError` when unknown."""
        with self._lock:
            for current in self._artifacts:
                if current.id == artifact_id:
                    return current
        raise SessionArtifactStoreError(
            f"unknown artifact id: {artifact_id!r}"
        )

    def unsaved(
        self, *, kind: SessionArtifactKind | None = None,
    ) -> list[SessionArtifact]:
        """Shortcut for the save-all footer: every artifact
        that hasn't been persisted yet. Newest first so the
        most recent generation is the first save attempt."""
        return self.list_artifacts(kind=kind, saved=False)

    def counts(self) -> dict[str, int]:
        """Header-pill projection — total + unsaved + per-kind
        counts. The future chat header reads this for the
        ``[ N artifacts • M unsaved ]`` indicator."""
        with self._lock:
            rows = list(self._artifacts)
        per_kind: dict[str, int] = {}
        for a in rows:
            per_kind[a.kind] = per_kind.get(a.kind, 0) + 1
        return {
            "total": len(rows),
            "unsaved": sum(1 for a in rows if not a.saved_to_memory),
            "saved": sum(1 for a in rows if a.saved_to_memory),
            **{f"kind:{k}": v for k, v in per_kind.items()},
        }

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def add_listener(self, fn: Listener) -> None:
        """Subscribe to append + mutation events. The same fn
        only registers once even on repeated calls — keeps the
        ChatScreen's mount lifecycle idempotent against
        re-mounts during tests."""
        with self._lock:
            if fn not in self._listeners:
                self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        """Unsubscribe. Silent no-op when ``fn`` wasn't
        registered — keeps screen-unmount paths simple."""
        with self._lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def _notify(self, artifact: SessionArtifact) -> None:
        """Fan out to listeners. Captures + swallows listener
        exceptions so one bad subscriber doesn't break the
        chain for the rest. Logs at WARNING so issues are
        still visible in ``logs/care-app-*.log``."""
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(artifact)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "artifact listener %r raised: %s",
                    getattr(fn, "__qualname__", repr(fn)), exc,
                )

    # ------------------------------------------------------------------
    # Optional persistence (§3 P1 hook)
    # ------------------------------------------------------------------

    def dump_jsonl(self, path: Path) -> int:
        """Write every artifact as JSON-Lines to ``path``.

        Returns the count written. Parent directory is created
        if missing. Uses ``default=str`` so non-serialisable
        payload pieces degrade to their ``str()`` rather than
        crashing the export — the projection is meant to be
        recovery-friendly, not round-trippable for arbitrary
        payloads."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.list_artifacts(newest_first=False)
        with path.open("w", encoding="utf-8") as fp:
            for artifact in snapshot:
                fp.write(json.dumps(artifact.to_dict(), default=str))
                fp.write("\n")
        return len(snapshot)


def load_jsonl_artifacts(path: Path) -> list[SessionArtifact]:
    """Inverse of :meth:`SessionArtifactStore.dump_jsonl`.

    Returns the artifacts in stored order (so a future
    ``/resume`` flow can rebuild a store via
    ``store.append(...)`` per row). Skips rows that don't
    project cleanly + logs them at WARNING — a corrupted line
    shouldn't kill the resume.
    """
    out: list[SessionArtifact] = []
    if not Path(path).exists():
        return out
    with Path(path).open("r", encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                created_at = datetime.fromisoformat(raw["created_at"])
                out.append(
                    SessionArtifact(
                        id=raw["id"],
                        kind=raw["kind"],
                        created_at=created_at,
                        title=raw.get("title", ""),
                        summary=raw.get("summary", ""),
                        payload=raw.get("payload"),
                        origin=dict(raw.get("origin") or {}),
                        saved_to_memory=bool(raw.get("saved_to_memory")),
                        memory_entity_id=raw.get("memory_entity_id"),
                        tags=tuple(raw.get("tags") or ()),
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                _log.warning(
                    "session_artifacts: dropped malformed row %d in %s: %s",
                    lineno, path, exc,
                )
    return out


def replay_into(
    store: SessionArtifactStore, artifacts: Iterable[SessionArtifact],
) -> None:
    """Hydrate an empty store with previously-dumped artifacts.

    Note: this preserves ``id`` / ``created_at`` / ``saved_to_memory``
    — the resumed store is a faithful copy of the dumped one.
    Use only against a freshly-constructed store; calling on a
    non-empty store mixes old + new artifacts which usually
    isn't what the caller wants.
    """
    if len(store):
        raise SessionArtifactStoreError(
            "replay_into requires an empty store; "
            f"got one with {len(store)} entries"
        )
    with store._lock:  # noqa: SLF001 — internal helper
        store._artifacts.extend(artifacts)


def dump_jsonl(store: SessionArtifactStore, path: Path) -> int:
    """Functional alias for :meth:`SessionArtifactStore.dump_jsonl`.

    Mirrors :func:`load_jsonl_artifacts` for callers that prefer
    the symmetric functional form over the bound method.
    """
    return store.dump_jsonl(path)


__all__ = [
    "Listener",
    "SessionArtifact",
    "SessionArtifactKind",
    "SessionArtifactStore",
    "SessionArtifactStoreError",
    "dump_jsonl",
    "load_jsonl_artifacts",
    "replay_into",
]
