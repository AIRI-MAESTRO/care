"""Tests for ``care.sandbox.SkillTrustStore`` (TODO §6.3 P0).

Real file IO throughout — tests use ``tmp_path`` to avoid touching
the user's actual ``~/.local/state/care/trusted_skills.json``.

Coverage layers:
1. ``TrustRecord`` round-trips through ``to_dict`` / ``from_dict``.
2. Empty load (missing file), populated load, and corrupt-file
   failure modes.
3. ``trust`` + ``revoke`` + ``clear`` mutate and persist correctly.
4. Atomic-save behaviour (parent dir auto-created, no leftover
   tempfile after a successful save).
5. ``is_trusted`` predicate covers the edge cases (empty SHA,
   non-string key, missing SHA).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from care.sandbox.trust import (
    DEFAULT_TRUST_PATH,
    STORE_FORMAT_VERSION,
    SkillTrustStore,
    SkillTrustStoreError,
    TrustRecord,
)

# Reusable SHAs (deterministic dummies — must be 64 hex chars to look real).
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


# ---------------------------------------------------------------------------
# TrustRecord shape
# ---------------------------------------------------------------------------


class TestTrustRecord:
    def test_round_trip_via_dict(self):
        approved = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
        rec = TrustRecord(
            sha256=SHA_A,
            uri="github://anthropics/skills/pdf",
            name="pdf-extract",
            approved_at=approved,
            trust_policy="sha_pinned",
            allowed_tools=("Bash", "Read"),
        )
        restored = TrustRecord.from_dict(rec.to_dict())
        assert restored == rec

    def test_from_dict_accepts_legacy_records_missing_policy(self):
        """Old records (before ``trust_policy`` shipped) should
        default to ``sha_pinned`` rather than raise."""
        rec = TrustRecord.from_dict(
            {
                "sha256": SHA_A,
                "uri": "local:///x",
                "name": "x",
                "approved_at": "2026-05-19T12:00:00+00:00",
            }
        )
        assert rec.trust_policy == "sha_pinned"
        assert rec.allowed_tools == ()

    def test_records_are_frozen(self):
        rec = TrustRecord(
            sha256=SHA_A,
            uri="local:///x",
            name="x",
            approved_at=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            rec.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_path_under_local_state(self):
        """``~/.local/state/care/trusted_skills.json`` per XDG."""
        assert DEFAULT_TRUST_PATH.name == "trusted_skills.json"
        assert DEFAULT_TRUST_PATH.parent.name == "care"
        assert ".local" in DEFAULT_TRUST_PATH.parts
        assert "state" in DEFAULT_TRUST_PATH.parts

    def test_store_format_version_is_one(self):
        """Pin the version; bumping it must be deliberate."""
        assert STORE_FORMAT_VERSION == 1


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_missing_file_yields_empty_store(self, tmp_path: Path):
        store = SkillTrustStore.load(path=tmp_path / "absent.json")
        assert len(store) == 0
        assert store.list_trusted() == []

    def test_load_populated_file(self, tmp_path: Path):
        target = tmp_path / "trusted.json"
        target.write_text(
            json.dumps(
                {
                    "version": STORE_FORMAT_VERSION,
                    "trusted": {
                        SHA_A: {
                            "sha256": SHA_A,
                            "uri": "github://x/y",
                            "name": "y",
                            "approved_at": "2026-05-19T12:00:00+00:00",
                            "trust_policy": "sha_pinned",
                            "allowed_tools": ["Bash"],
                        }
                    },
                }
            )
        )
        store = SkillTrustStore.load(path=target)
        assert len(store) == 1
        rec = store.get(SHA_A)
        assert rec is not None
        assert rec.name == "y"
        assert rec.allowed_tools == ("Bash",)

    def test_load_rejects_corrupt_json(self, tmp_path: Path):
        target = tmp_path / "trusted.json"
        target.write_text("{not json")
        with pytest.raises(SkillTrustStoreError, match="could not read"):
            SkillTrustStore.load(path=target)

    def test_load_rejects_unknown_version(self, tmp_path: Path):
        target = tmp_path / "trusted.json"
        target.write_text(json.dumps({"version": 99, "trusted": {}}))
        with pytest.raises(SkillTrustStoreError, match="unknown trust-store version"):
            SkillTrustStore.load(path=target)

    def test_load_rejects_corrupt_record(self, tmp_path: Path):
        """Per-record corruption (missing required field) must
        fail loudly — silently dropping the bad row would lower the
        security posture."""
        target = tmp_path / "trusted.json"
        target.write_text(
            json.dumps(
                {
                    "version": STORE_FORMAT_VERSION,
                    "trusted": {SHA_A: {"sha256": SHA_A}},  # missing uri / name / approved_at
                }
            )
        )
        with pytest.raises(SkillTrustStoreError, match="corrupt entry"):
            SkillTrustStore.load(path=target)


# ---------------------------------------------------------------------------
# Trust / query / mutation
# ---------------------------------------------------------------------------


class TestQuery:
    def test_is_trusted_returns_false_for_empty_store(self, tmp_path):
        store = SkillTrustStore.load(path=tmp_path / "missing.json")
        assert store.is_trusted(SHA_A) is False

    def test_is_trusted_rejects_empty_sha(self, tmp_path):
        store = SkillTrustStore.load(path=tmp_path / "missing.json")
        assert store.is_trusted("") is False

    def test_contains_operator(self, tmp_path):
        store = SkillTrustStore.load(path=tmp_path / "missing.json")
        store.trust(sha256=SHA_A, uri="local:///x", name="x")
        assert SHA_A in store
        assert SHA_B not in store
        # Non-string keys don't crash.
        assert 12345 not in store  # type: ignore[operator]


class TestTrustAndRevoke:
    def test_trust_persists_and_round_trips(self, tmp_path: Path):
        path = tmp_path / "trusted.json"
        store = SkillTrustStore.load(path=path)
        rec = store.trust(
            sha256=SHA_A,
            uri="github://anthropics/skills/pdf",
            name="pdf-extract",
            allowed_tools=["Bash(pdftotext:*)", "Read"],
        )
        assert rec.sha256 == SHA_A
        assert rec.trust_policy == "sha_pinned"

        # Round-trip through disk.
        reloaded = SkillTrustStore.load(path=path)
        assert reloaded.is_trusted(SHA_A)
        round_tripped = reloaded.get(SHA_A)
        assert round_tripped is not None
        assert round_tripped.uri == "github://anthropics/skills/pdf"
        assert round_tripped.allowed_tools == ("Bash(pdftotext:*)", "Read")

    def test_trust_rejects_empty_sha_or_uri(self, tmp_path: Path):
        store = SkillTrustStore.load(path=tmp_path / "trusted.json")
        with pytest.raises(SkillTrustStoreError, match="sha256"):
            store.trust(sha256="", uri="local:///x", name="x")
        with pytest.raises(SkillTrustStoreError, match="uri"):
            store.trust(sha256=SHA_A, uri="", name="x")

    def test_trust_twice_refreshes_record(self, tmp_path: Path):
        store = SkillTrustStore.load(path=tmp_path / "trusted.json")
        early = datetime(2026, 1, 1, tzinfo=timezone.utc)
        late = datetime(2026, 6, 1, tzinfo=timezone.utc)
        store.trust(
            sha256=SHA_A,
            uri="local:///x",
            name="x",
            approved_at=early,
            allowed_tools=["Bash"],
        )
        store.trust(
            sha256=SHA_A,
            uri="local:///x",
            name="x",
            approved_at=late,
            allowed_tools=["Bash", "Read"],
        )
        rec = store.get(SHA_A)
        assert rec is not None
        assert rec.approved_at == late
        assert rec.allowed_tools == ("Bash", "Read")

    def test_revoke_removes_and_persists(self, tmp_path: Path):
        path = tmp_path / "trusted.json"
        store = SkillTrustStore.load(path=path)
        store.trust(sha256=SHA_A, uri="local:///x", name="x")
        store.trust(sha256=SHA_B, uri="local:///y", name="y")
        assert store.revoke(SHA_A) is True
        assert store.is_trusted(SHA_A) is False
        assert store.is_trusted(SHA_B) is True

        reloaded = SkillTrustStore.load(path=path)
        assert reloaded.is_trusted(SHA_A) is False
        assert reloaded.is_trusted(SHA_B) is True

    def test_revoke_unknown_sha_is_no_op(self, tmp_path: Path):
        store = SkillTrustStore.load(path=tmp_path / "trusted.json")
        assert store.revoke(SHA_A) is False

    def test_clear_empties_store(self, tmp_path: Path):
        path = tmp_path / "trusted.json"
        store = SkillTrustStore.load(path=path)
        store.trust(sha256=SHA_A, uri="local:///x", name="x")
        store.trust(sha256=SHA_B, uri="local:///y", name="y")
        store.clear()
        assert len(store) == 0
        assert SkillTrustStore.load(path=path).list_trusted() == []

    def test_clear_on_empty_store_is_no_op(self, tmp_path: Path):
        """Empty + clear() should NOT touch disk; verifies the
        early-return doesn't accidentally create an empty file."""
        path = tmp_path / "trusted.json"
        store = SkillTrustStore.load(path=path)
        store.clear()
        assert not path.exists()


class TestListing:
    def test_list_trusted_sorted_newest_first(self, tmp_path: Path):
        store = SkillTrustStore.load(path=tmp_path / "trusted.json")
        oldest = datetime(2026, 1, 1, tzinfo=timezone.utc)
        middle = datetime(2026, 3, 1, tzinfo=timezone.utc)
        newest = datetime(2026, 5, 1, tzinfo=timezone.utc)
        store.trust(sha256=SHA_A, uri="local:///a", name="a", approved_at=oldest)
        store.trust(sha256=SHA_C, uri="local:///c", name="c", approved_at=newest)
        store.trust(sha256=SHA_B, uri="local:///b", name="b", approved_at=middle)
        names = [r.name for r in store.list_trusted()]
        assert names == ["c", "b", "a"]


# ---------------------------------------------------------------------------
# Persistence behaviour
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_parent_directory(self, tmp_path: Path):
        nested = tmp_path / "deep" / "deeper" / "trusted.json"
        store = SkillTrustStore.load(path=nested)
        store.trust(sha256=SHA_A, uri="local:///x", name="x")
        assert nested.exists()
        assert nested.parent.is_dir()

    def test_save_is_valid_json(self, tmp_path: Path):
        path = tmp_path / "trusted.json"
        store = SkillTrustStore.load(path=path)
        store.trust(sha256=SHA_A, uri="local:///x", name="x")
        # File must be readable + parse cleanly.
        raw = json.loads(path.read_text())
        assert raw["version"] == STORE_FORMAT_VERSION
        assert SHA_A in raw["trusted"]
        assert raw["trusted"][SHA_A]["uri"] == "local:///x"

    def test_no_leftover_tempfiles_after_save(self, tmp_path: Path):
        path = tmp_path / "trusted.json"
        store = SkillTrustStore.load(path=path)
        store.trust(sha256=SHA_A, uri="local:///x", name="x")
        # Only the canonical file should remain — atomic-write
        # tempfiles must be renamed away.
        leftover = list(tmp_path.glob(".trusted_skills-*.json"))
        assert leftover == []
        assert path.exists()
