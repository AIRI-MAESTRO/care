"""SHA-pinned skill trust store (TODO §6.3 P0).

Before CARE runs an AgentSkill that came from anywhere outside the
local checkout it must confirm with the user that the skill's
SKILL.md contents (identified by SHA256) are trusted. This module
implements the **persistent store + decision API** behind that
flow. The UI prompt itself is a separate piece — it calls
:meth:`SkillTrustStore.is_trusted` to decide whether to interrupt
the user, and :meth:`SkillTrustStore.trust` once the user clicks
"Approve once + remember".

Storage layout — JSON at ``~/.local/state/care/trusted_skills.json``::

    {
      "version": 1,
      "trusted": {
        "<sha256>": {
          "sha256": "<sha256>",
          "uri": "github://anthropics/skills/pdf",
          "name": "pdf-extract",
          "approved_at": "2026-05-19T12:34:56+00:00",
          "trust_policy": "sha_pinned",
          "allowed_tools": ["Bash(pdftotext:*)", "Read", "Write"]
        }
      }
    }

``trust_policy="sha_pinned"`` is the only mode shipped today: any
change to SKILL.md flips the SHA and the user has to re-approve.
Future modes (``"name_pinned"``, ``"uri_pinned"``) can be added
without changing the file format because the SHA stays the
top-level key.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

DEFAULT_TRUST_PATH = Path("~/.local/state/care/trusted_skills.json").expanduser()
"""Where :class:`SkillTrustStore` persists by default."""

STORE_FORMAT_VERSION = 1
"""On-disk schema version. Bump when a load-incompatible field
shape change lands; readers should refuse unknown versions rather
than silently dropping fields."""

TrustPolicy = Literal["sha_pinned"]
"""How strictly the trust record binds. Only ``"sha_pinned"`` ships
today: any SKILL.md byte change → SHA flip → re-approval. The
Literal stays open so future policies layer on without breaking
callers that pin the value."""


@dataclass(frozen=True)
class TrustRecord:
    """One entry in the trust store.

    Frozen so callers (including UI code) can pass records around
    without worrying about mutation. ``approved_at`` is always
    timezone-aware UTC.
    """

    sha256: str
    uri: str
    name: str
    approved_at: datetime
    trust_policy: TrustPolicy = "sha_pinned"
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "uri": self.uri,
            "name": self.name,
            "approved_at": self.approved_at.isoformat(),
            "trust_policy": self.trust_policy,
            "allowed_tools": list(self.allowed_tools),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrustRecord":
        return cls(
            sha256=data["sha256"],
            uri=data["uri"],
            name=data["name"],
            approved_at=datetime.fromisoformat(data["approved_at"]),
            trust_policy=data.get("trust_policy", "sha_pinned"),
            allowed_tools=tuple(data.get("allowed_tools") or []),
        )


class SkillTrustStoreError(RuntimeError):
    """Trust-store load / save failure (corrupt JSON, schema
    version mismatch, etc.). Raised loudly so a corrupt store can't
    silently downgrade CARE's security posture."""


class SkillTrustStore:
    """Persistent SHA-pinned approval store.

    Use :meth:`load` for the normal CARE startup path. The bare
    constructor is for tests and advanced flows that already
    assembled the records dict.
    """

    def __init__(
        self,
        records: dict[str, TrustRecord] | None = None,
        *,
        path: Path | None = None,
    ) -> None:
        self._records: dict[str, TrustRecord] = dict(records or {})
        self._path = path

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, *, path: Path | None = None) -> "SkillTrustStore":
        """Load the store from disk.

        Missing files are returned as empty stores (first-run
        normal case). Corrupt JSON or an unknown ``version`` raises
        :class:`SkillTrustStoreError` — better to fail loudly than
        silently treat untrusted skills as trusted.
        """
        target = path or DEFAULT_TRUST_PATH
        if not target.exists():
            return cls({}, path=target)
        try:
            raw = json.loads(target.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise SkillTrustStoreError(
                f"could not read trust store {target}: {exc}"
            ) from exc
        version = raw.get("version")
        if version != STORE_FORMAT_VERSION:
            raise SkillTrustStoreError(
                f"unknown trust-store version {version!r} at {target}; "
                f"expected {STORE_FORMAT_VERSION}"
            )
        trusted = raw.get("trusted") or {}
        records: dict[str, TrustRecord] = {}
        for key, entry in trusted.items():
            try:
                records[key] = TrustRecord.from_dict(entry)
            except (KeyError, ValueError) as exc:
                raise SkillTrustStoreError(
                    f"corrupt entry for sha {key!r}: {exc}"
                ) from exc
        return cls(records, path=target)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_trusted(self, sha256: str) -> bool:
        """Has the user approved this exact SKILL.md SHA?

        Empty/None SHA is never trusted — defends against an
        upstream bug that hands the store an unhashed manifest.
        """
        if not sha256:
            return False
        return sha256 in self._records

    def get(self, sha256: str) -> TrustRecord | None:
        return self._records.get(sha256)

    def list_trusted(self) -> list[TrustRecord]:
        """Snapshot of every approved record, sorted newest first."""
        return sorted(
            self._records.values(),
            key=lambda r: r.approved_at,
            reverse=True,
        )

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, sha256: object) -> bool:
        return isinstance(sha256, str) and self.is_trusted(sha256)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def trust(
        self,
        *,
        sha256: str,
        uri: str,
        name: str,
        allowed_tools: list[str] | tuple[str, ...] | None = None,
        approved_at: datetime | None = None,
        trust_policy: TrustPolicy = "sha_pinned",
    ) -> TrustRecord:
        """Approve a skill by SHA and persist.

        Returns the stored :class:`TrustRecord`. Calling twice with
        the same SHA refreshes ``approved_at`` and ``allowed_tools``
        — the user re-confirmed; we re-anchor the record.
        """
        if not sha256:
            raise SkillTrustStoreError("sha256 must be non-empty")
        if not uri:
            raise SkillTrustStoreError("uri must be non-empty")
        record = TrustRecord(
            sha256=sha256,
            uri=uri,
            name=name,
            approved_at=approved_at or datetime.now(timezone.utc),
            trust_policy=trust_policy,
            allowed_tools=tuple(allowed_tools or []),
        )
        self._records[sha256] = record
        self._save()
        return record

    def revoke(self, sha256: str) -> bool:
        """Forget approval for ``sha256``. Returns whether anything
        was removed (idempotent — revoking an unknown SHA is fine
        and returns False)."""
        if sha256 not in self._records:
            return False
        del self._records[sha256]
        self._save()
        return True

    def clear(self) -> None:
        """Wipe every approval. Mainly for tests + a future "trust
        nothing" action in the settings UI."""
        if not self._records:
            return
        self._records.clear()
        self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Write the store atomically (tempfile + rename)."""
        target = self._path or DEFAULT_TRUST_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STORE_FORMAT_VERSION,
            "trusted": {
                sha: rec.to_dict() for sha, rec in self._records.items()
            },
        }
        # Atomic write: tempfile in the same dir, then os.replace.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".trusted_skills-", suffix=".json", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w") as fp:
                json.dump(payload, fp, indent=2, sort_keys=True)
                fp.write("\n")
            os.replace(tmp_path, target)
        except Exception:
            # Clean up the tempfile on failure so we don't leak.
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise


__all__ = [
    "DEFAULT_TRUST_PATH",
    "STORE_FORMAT_VERSION",
    "SkillTrustStore",
    "SkillTrustStoreError",
    "TrustPolicy",
    "TrustRecord",
]
