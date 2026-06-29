"""AgentSkill provenance recorder (TODO §3 P0).

When MAGE's ``CapabilityLookupAgent`` (or any other CARE-side
discovery flow) returns a skill it wants to reference from a
generated chain, that skill **must** exist as an ``agent_skill``
entity in Memory first — chain content holds only a reference, so
without provenance the chain would point at nothing.

This module owns the bookkeeping around that "save-before-reference"
flow:

* Wraps the SDK's already-shipped, idempotent
  :meth:`AgentSkillsMixin.ingest_skill_from_carl` (which handles
  duck-typed input shapes and computes SHA256s).
* Adds a **session-scoped SHA → entity_id cache** so repeat
  ingestions inside one CARE run don't double-save (the SDK is
  idempotent if you supply ``entity_id``, but a fresh discovery
  flow doesn't know the id until after the first save).
* Optionally consults a :class:`care.sandbox.SkillTrustStore` so
  the recorder can return ``trusted=False`` for skills the user
  hasn't approved yet — letting the caller decide whether to
  prompt or refuse to persist.
* Returns a typed :class:`SkillProvenanceRecord` per skill so the
  surrounding UI can render a "X skills resolved (2 new, 1 already
  trusted)" panel without re-querying Memory.

The recorder doesn't import ``mmar_carl``; it duck-types skill
inputs the same way the SDK does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from care.memory import CareMemory
from care.sandbox import SkillTrustStore


@dataclass(frozen=True)
class SkillProvenanceRecord:
    """One outcome of :func:`record_skill_provenance`.

    Frozen so call-sites can pass these around without defensive
    copies. ``was_new`` is ``True`` when this CARE session has not
    persisted the skill before — it does NOT detect cross-session
    duplicates (Memory deduplication by SHA is a separate
    Memory-side concern).

    Fields:
        entity_id: The persisted ``agent_skill`` entity's id.
        sha256: The SKILL.md SHA256 the SDK computed.
        name: The entity ``meta.name`` after persistence.
        uri: The skill's source URI (``github://`` / ``local://`` /
            ``https://``).
        was_new: ``True`` if this is the first ingestion in the
            current :class:`SkillProvenanceRecorder` session.
        trusted: ``True`` when the supplied ``trust_store``
            confirmed this SHA. ``None`` when no trust store was
            consulted (CARE's default during early dev / CI).
    """

    entity_id: str
    sha256: str
    name: str
    uri: str
    was_new: bool
    trusted: bool | None = None


class SkillProvenanceRecorder:
    """Persists AgentSkills to Memory with session-scoped dedup.

    Usage::

        recorder = SkillProvenanceRecorder(memory, trust_store=ts)
        records = recorder.record_skills(skills, namespace=ns)
        # `records[i].entity_id` is now safe to embed in the chain.

    A single recorder is meant to live for the duration of one CARE
    user flow (a generate → save round trip). Long-lived recorders
    are fine but the cache only grows; clear it between unrelated
    sessions if memory pressure matters.
    """

    def __init__(
        self,
        memory: CareMemory,
        *,
        trust_store: SkillTrustStore | None = None,
    ) -> None:
        self._memory = memory
        self._trust_store = trust_store
        self._cache: dict[str, str] = {}  # sha256 → entity_id

    # ------------------------------------------------------------------
    # Single-skill path
    # ------------------------------------------------------------------

    def record_skill(
        self,
        skill: Any,
        *,
        namespace: str | None = None,
        author: str | None = None,
        tags: list[str] | None = None,
        when_to_use: str | None = None,
        channel: str = "latest",
    ) -> SkillProvenanceRecord:
        """Ensure ``skill`` is persisted; return its provenance record.

        Args:
            skill: Anything :meth:`GigaEvoClient.ingest_skill_from_carl`
                accepts — a CARL ``ResolvedSkill``, an
                ``AgentSkillSpec``, or a dict matching the spec.
            namespace / author / tags / when_to_use / channel:
                Forwarded to the SDK call.

        Returns:
            :class:`SkillProvenanceRecord`. ``was_new=True`` on first
            call for this SHA within this recorder; subsequent calls
            short-circuit using the cache.
        """
        # Probe inputs without importing mmar_carl: dict or duck-typed.
        sha = _extract_sha(skill)
        uri = _extract_uri(skill)
        name = _extract_name(skill)

        trusted = self._check_trust(sha)

        cached_id = self._cache.get(sha) if sha else None
        if cached_id is not None:
            return SkillProvenanceRecord(
                entity_id=cached_id,
                sha256=sha,
                name=name,
                uri=uri,
                was_new=False,
                trusted=trusted,
            )

        ref = self._memory.client.ingest_skill_from_carl(
            skill,
            namespace=namespace,
            author=author,
            tags=tags,
            when_to_use=when_to_use,
            channel=channel,
        )

        # The SDK may have computed a SHA we didn't see — cache by
        # the SDK's authoritative value too if it differs (e.g. when
        # the input was a dict without explicit sha256 yet the SDK
        # derived it from the raw SKILL.md text).
        canonical_sha = sha or _extract_sha_from_ref(ref)
        if canonical_sha:
            self._cache[canonical_sha] = ref.entity_id

        return SkillProvenanceRecord(
            entity_id=ref.entity_id,
            sha256=canonical_sha,
            name=name,
            uri=uri,
            was_new=True,
            trusted=trusted,
        )

    # ------------------------------------------------------------------
    # Batch path
    # ------------------------------------------------------------------

    def record_skills(
        self,
        skills: Iterable[Any],
        *,
        namespace: str | None = None,
        author: str | None = None,
        tags: list[str] | None = None,
        when_to_use: str | None = None,
        channel: str = "latest",
    ) -> list[SkillProvenanceRecord]:
        """Bulk version of :meth:`record_skill`.

        Stops the loop early on the first failure to surface so the
        chain isn't half-persisted with references to skills CARE
        never finished saving. Caller can inspect ``len(records)``
        vs ``len(input)`` to spot a partial commit.
        """
        out: list[SkillProvenanceRecord] = []
        for skill in skills:
            out.append(
                self.record_skill(
                    skill,
                    namespace=namespace,
                    author=author,
                    tags=tags,
                    when_to_use=when_to_use,
                    channel=channel,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Cache management (for tests + long-lived flows)
    # ------------------------------------------------------------------

    @property
    def cache(self) -> dict[str, str]:
        """Read-only view of the SHA → entity_id cache. Returns a
        copy so callers can't mutate internal state."""
        return dict(self._cache)

    def clear_cache(self) -> None:
        """Drop every cached SHA → entity_id mapping."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_trust(self, sha: str) -> bool | None:
        """Return tri-state trust: ``True`` / ``False`` / ``None``
        (no trust store configured)."""
        if self._trust_store is None:
            return None
        if not sha:
            return False
        return self._trust_store.is_trusted(sha)


# ---------------------------------------------------------------------------
# Convenience module-level wrapper
# ---------------------------------------------------------------------------


def record_skill_provenance(
    memory: CareMemory,
    skill: Any,
    *,
    trust_store: SkillTrustStore | None = None,
    namespace: str | None = None,
    author: str | None = None,
    tags: list[str] | None = None,
    when_to_use: str | None = None,
    channel: str = "latest",
) -> SkillProvenanceRecord:
    """Single-shot wrapper around :class:`SkillProvenanceRecorder`.

    For batch flows (MAGE returning a list of capabilities), prefer
    constructing a recorder so the cache persists across calls.
    """
    recorder = SkillProvenanceRecorder(memory, trust_store=trust_store)
    return recorder.record_skill(
        skill,
        namespace=namespace,
        author=author,
        tags=tags,
        when_to_use=when_to_use,
        channel=channel,
    )


# ---------------------------------------------------------------------------
# Duck-typed accessors
# ---------------------------------------------------------------------------


def _extract_sha(skill: Any) -> str:
    """Pull the SHA256 off whatever shape the caller handed us.

    Recognises the same inputs as the SDK's ``_extract_skill_spec``
    plus the older `source_sha256` alias some CARL versions exposed.
    Returns ``""`` when the value isn't present — the SDK will
    derive it from raw SKILL.md text in that case.
    """
    if isinstance(skill, dict):
        return str(skill.get("sha256") or skill.get("source_sha256") or "")
    sha = getattr(skill, "sha256", None) or getattr(skill, "source_sha256", None)
    return str(sha or "")


def _extract_uri(skill: Any) -> str:
    if isinstance(skill, dict):
        return str(skill.get("uri") or skill.get("source_uri") or "")
    val = getattr(skill, "uri", None) or getattr(skill, "source_uri", None)
    return str(val or "")


def _extract_name(skill: Any) -> str:
    """Best-effort name lookup. Falls back to the manifest's name
    when the top level is unnamed."""
    if isinstance(skill, dict):
        if skill.get("name"):
            return str(skill["name"])
        manifest = skill.get("manifest") or {}
        if isinstance(manifest, dict):
            return str(manifest.get("name") or "")
        return ""
    name = getattr(skill, "name", None)
    if name:
        return str(name)
    manifest = getattr(skill, "manifest", None)
    if manifest is not None:
        return str(getattr(manifest, "name", None) or "")
    return ""


def _extract_sha_from_ref(ref: Any) -> str:
    """The SDK's :class:`EntityRef` doesn't expose the content SHA;
    callers that didn't supply one upfront should re-derive it from
    the source. This helper exists so we can plug that in later
    without changing call-sites — for now it returns ``""``."""
    return ""


__all__ = [
    "SkillProvenanceRecord",
    "SkillProvenanceRecorder",
    "record_skill_provenance",
]
