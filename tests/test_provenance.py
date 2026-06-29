"""Tests for ``care.runtime.provenance`` (TODO §3 P0).

Coverage layers:
1. Duck-typed accessors handle every documented input shape (dict
   / spec / CARL-like).
2. ``record_skill`` posts to ``/v1/agent-skills`` on first SHA,
   short-circuits on subsequent calls (cache hit).
3. ``record_skills`` (batch) preserves order + caches across the loop.
4. Trust-store integration returns the right tri-state on
   ``SkillProvenanceRecord.trusted``.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from gigaevo_client import GigaEvoClient

from care.memory import CareMemory
from care.runtime import (
    SkillProvenanceRecord,
    SkillProvenanceRecorder,
    record_skill_provenance,
)
from care.sandbox import SkillTrustStore

BASE = "http://test-memory:8000"

# Realistic-looking SKILL.md SHAs (must be 64 hex chars).
SHA_A = "a" * 64
SHA_B = "b" * 64


@pytest.fixture
def memory():
    return CareMemory(GigaEvoClient(base_url=BASE, api_key="sk-test", timeout=5.0))


def _spec_dict(*, sha: str = SHA_A, name: str = "pdf-extract") -> dict:
    """Build a minimal `AgentSkillSpec`-shaped dict the SDK accepts."""
    return {
        "name": name,
        "description": "Extract text from PDF files",
        "uri": f"github://anthropics/skills/{name}",
        "sha256": sha,
        "manifest": {
            "name": name,
            "description": "Extract text from PDF files",
            "allowed-tools": ["Bash(pdftotext:*)", "Read"],
            "tags": ["pdf"],
        },
        "instructions": "Run pdftotext on the input.",
        "allowed_tools": ["Bash(pdftotext:*)", "Read"],
        "tags": ["pdf"],
    }


def _save_handler(entity_id: str = "skill-1"):
    captured: dict = {"calls": 0, "bodies": []}

    def handler(request):
        captured["calls"] += 1
        captured["bodies"].append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "entity_type": "agent_skill",
                "entity_id": entity_id,
                "version_id": "v-1",
                "channel": "latest",
            },
        )

    return handler, captured


# ---------------------------------------------------------------------------
# SkillProvenanceRecord shape
# ---------------------------------------------------------------------------


class TestSkillProvenanceRecord:
    def test_defaults_trusted_none(self):
        rec = SkillProvenanceRecord(
            entity_id="e", sha256=SHA_A, name="n", uri="u", was_new=True
        )
        assert rec.trusted is None

    def test_frozen(self):
        rec = SkillProvenanceRecord(
            entity_id="e", sha256=SHA_A, name="n", uri="u", was_new=True
        )
        with pytest.raises(AttributeError):
            rec.entity_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# record_skill: dict input + cache
# ---------------------------------------------------------------------------


class TestRecordSkillDict:
    @respx.mock
    def test_first_call_persists_and_returns_was_new(self, memory):
        handler, captured = _save_handler("skill-7")
        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=handler)

        recorder = SkillProvenanceRecorder(memory)
        record = recorder.record_skill(_spec_dict())

        assert isinstance(record, SkillProvenanceRecord)
        assert record.entity_id == "skill-7"
        assert record.sha256 == SHA_A
        assert record.name == "pdf-extract"
        assert record.uri == "github://anthropics/skills/pdf-extract"
        assert record.was_new is True
        assert record.trusted is None  # no trust store configured

        assert captured["calls"] == 1
        body = captured["bodies"][0]
        assert body["content"]["sha256"] == SHA_A
        assert body["content"]["uri"] == "github://anthropics/skills/pdf-extract"

    @respx.mock
    def test_second_call_same_sha_short_circuits(self, memory):
        handler, captured = _save_handler("skill-7")
        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=handler)

        recorder = SkillProvenanceRecorder(memory)
        recorder.record_skill(_spec_dict())
        record = recorder.record_skill(_spec_dict())

        # Only one HTTP call total.
        assert captured["calls"] == 1
        assert record.entity_id == "skill-7"
        assert record.was_new is False

    @respx.mock
    def test_distinct_shas_both_persist(self, memory):
        captured: dict = {"calls": 0, "bodies": []}
        counter = {"i": 0}

        def handler(request):
            captured["calls"] += 1
            captured["bodies"].append(json.loads(request.content))
            counter["i"] += 1
            return httpx.Response(
                200,
                json={
                    "entity_type": "agent_skill",
                    "entity_id": f"skill-{counter['i']}",
                    "version_id": "v-1",
                    "channel": "latest",
                },
            )

        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=handler)

        recorder = SkillProvenanceRecorder(memory)
        r1 = recorder.record_skill(_spec_dict(sha=SHA_A, name="a"))
        r2 = recorder.record_skill(_spec_dict(sha=SHA_B, name="b"))

        assert captured["calls"] == 2
        assert r1.entity_id == "skill-1"
        assert r2.entity_id == "skill-2"
        assert r1.was_new is True
        assert r2.was_new is True


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------


class TestRecordSkillsBatch:
    @respx.mock
    def test_preserves_order_and_caches_dups(self, memory):
        captured: dict = {"calls": 0}
        counter = {"i": 0}

        def handler(request):
            captured["calls"] += 1
            counter["i"] += 1
            return httpx.Response(
                200,
                json={
                    "entity_type": "agent_skill",
                    "entity_id": f"skill-{counter['i']}",
                    "version_id": "v-1",
                    "channel": "latest",
                },
            )

        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=handler)

        recorder = SkillProvenanceRecorder(memory)
        skills = [
            _spec_dict(sha=SHA_A, name="a"),
            _spec_dict(sha=SHA_B, name="b"),
            _spec_dict(sha=SHA_A, name="a"),  # dup of the first
        ]
        records = recorder.record_skills(skills)

        assert [r.name for r in records] == ["a", "b", "a"]
        assert [r.was_new for r in records] == [True, True, False]
        assert captured["calls"] == 2  # third was cached
        assert records[0].entity_id == records[2].entity_id

    @respx.mock
    def test_module_level_wrapper_works(self, memory):
        handler, captured = _save_handler("skill-once")
        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=handler)

        record = record_skill_provenance(memory, _spec_dict())

        assert record.entity_id == "skill-once"
        assert record.was_new is True
        assert captured["calls"] == 1


# ---------------------------------------------------------------------------
# Trust-store integration
# ---------------------------------------------------------------------------


class TestTrustStoreIntegration:
    @respx.mock
    def test_trusted_true_when_sha_in_store(self, memory, tmp_path):
        handler, _ = _save_handler()
        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=handler)

        ts = SkillTrustStore.load(path=tmp_path / "trust.json")
        ts.trust(sha256=SHA_A, uri="github://x", name="pdf-extract")

        recorder = SkillProvenanceRecorder(memory, trust_store=ts)
        record = recorder.record_skill(_spec_dict())
        assert record.trusted is True

    @respx.mock
    def test_trusted_false_when_sha_not_in_store(self, memory, tmp_path):
        handler, _ = _save_handler()
        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=handler)

        ts = SkillTrustStore.load(path=tmp_path / "trust.json")
        # Store empty → SHA not trusted, but recorder still persists.
        recorder = SkillProvenanceRecorder(memory, trust_store=ts)
        record = recorder.record_skill(_spec_dict())
        assert record.trusted is False
        assert record.entity_id  # still saved

    def test_trusted_none_when_no_store(self, memory):
        recorder = SkillProvenanceRecorder(memory)
        assert recorder._check_trust(SHA_A) is None

    def test_trusted_false_on_empty_sha_even_with_store(
        self, memory, tmp_path
    ):
        ts = SkillTrustStore.load(path=tmp_path / "trust.json")
        recorder = SkillProvenanceRecorder(memory, trust_store=ts)
        assert recorder._check_trust("") is False


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


class TestCache:
    @respx.mock
    def test_cache_property_returns_copy(self, memory):
        handler, _ = _save_handler("skill-1")
        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=handler)

        recorder = SkillProvenanceRecorder(memory)
        recorder.record_skill(_spec_dict())
        snapshot = recorder.cache
        assert snapshot == {SHA_A: "skill-1"}
        # Mutating the snapshot must NOT affect the recorder.
        snapshot["hacked"] = "X"
        assert recorder.cache == {SHA_A: "skill-1"}

    @respx.mock
    def test_clear_cache_drops_entries(self, memory):
        handler, captured = _save_handler("skill-1")
        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=handler)

        recorder = SkillProvenanceRecorder(memory)
        recorder.record_skill(_spec_dict())
        recorder.clear_cache()
        # After clear, second ingestion of same SHA hits the wire again.
        recorder.record_skill(_spec_dict())
        assert captured["calls"] == 2


# ---------------------------------------------------------------------------
# Duck-typed inputs (object-shaped, not dict)
# ---------------------------------------------------------------------------


class _FakeManifest:
    def __init__(self, name: str = "pdf-extract"):
        self.name = name
        self.description = "Extract text from PDF files"
        self.instructions = "Run pdftotext"
        self.allowed_tools = ["Bash"]
        self.tags = ["pdf"]
        self.metadata = {}
        self.compatibility = {}

    def get_allowed_tools(self):
        return self.allowed_tools


class _FakeResolvedSkill:
    """Mirror the attribute surface ``_extract_skill_spec`` recognises."""

    def __init__(self, sha: str = SHA_A, uri: str = "github://x/y"):
        self.manifest = _FakeManifest()
        self.sha256 = sha
        self.source_uri = uri
        self.tarball_url = None
        self.tarball_sha256 = None


class TestDuckTypedInputs:
    @respx.mock
    def test_object_with_manifest_is_accepted(self, memory):
        handler, captured = _save_handler("skill-duck")
        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=handler)

        recorder = SkillProvenanceRecorder(memory)
        record = recorder.record_skill(_FakeResolvedSkill())

        assert record.entity_id == "skill-duck"
        assert record.sha256 == SHA_A
        assert record.name == "pdf-extract"
        # source_uri attribute is recognised as the URI source.
        assert record.uri == "github://x/y"

    def test_extract_name_dict_falls_back_to_manifest(self):
        """Pure helper: when the top-level ``name`` is empty, the
        manifest's name wins. The SDK's strict ``AgentSkillSpec``
        validation would reject the dict in practice — this just
        pins the CARE-side accessor's fallback semantics."""
        from care.runtime.provenance import _extract_name

        skill_dict = {"name": "", "manifest": {"name": "from-manifest"}}
        assert _extract_name(skill_dict) == "from-manifest"

    def test_extract_name_object_falls_back_to_manifest(self):
        from care.runtime.provenance import _extract_name

        skill_obj = _FakeResolvedSkill()
        # _FakeResolvedSkill has no top-level `name`; manifest.name wins.
        assert _extract_name(skill_obj) == "pdf-extract"
