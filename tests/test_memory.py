"""Tests for ``care.memory.CareMemory`` (TODO §3 P0).

Two layers of coverage:

1. **Pure helpers** — ``_build_care_metadata`` /
   ``_apply_chain_metadata`` are exercised directly with no HTTP,
   verifying that CARE-shaped inputs (query, context files,
   MAGEMetadata) turn into the right ``CareChainMetadata`` shape and
   that the merge lands under ``content.metadata`` correctly.
2. **End-to-end save/read** — every public method that talks to
   Memory is run against a ``respx``-mocked HTTP layer. We assert:
   - the method hits the right URL with the right verb,
   - the request body carries the expected CARE-side data,
   - the return value is the bare ``entity_id`` string.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from gigaevo_client import (
    CareChainMetadata,
    ContextFileRef,
    GigaEvoClient,
)

from care.config import CareConfig
from care.memory import CareMemory

BASE = "http://test-memory:8000"


@pytest.fixture
def client():
    return GigaEvoClient(base_url=BASE, api_key="sk-test", timeout=5.0)


@pytest.fixture
def memory(client):
    return CareMemory(client)


# ---------------------------------------------------------------------------
# Pure helper behaviour
# ---------------------------------------------------------------------------


class TestBuildCareMetadata:
    def test_simple_query_only(self):
        meta = CareMemory._build_care_metadata(
            query="weather report for SF",
            context_files=None,
            mage_metadata=None,
            display_name="weather-agent",
            description="weather report for SF",
            tags=None,
        )
        assert isinstance(meta, CareChainMetadata)
        assert meta.task_description == "weather report for SF"
        assert meta.context_files == []
        assert meta.display_name == "weather-agent"
        assert meta.description == "weather report for SF"

    def test_context_files_accept_dicts(self):
        meta = CareMemory._build_care_metadata(
            query="analyse Q3",
            context_files=[
                {"path": "report.pdf", "sha256": "a" * 64, "size_bytes": 1024},
            ],
            mage_metadata=None,
            display_name="fin-q3",
            description="analyse Q3",
            tags=None,
        )
        assert len(meta.context_files) == 1
        assert isinstance(meta.context_files[0], ContextFileRef)
        assert meta.context_files[0].path == "report.pdf"
        assert meta.context_files[0].sha256 == "a" * 64

    def test_context_files_accept_typed_refs(self):
        ref = ContextFileRef(path="x.txt", sha256="b" * 64, size_bytes=10)
        meta = CareMemory._build_care_metadata(
            query=None,
            context_files=[ref],
            mage_metadata=None,
            display_name=None,
            description=None,
            tags=None,
        )
        assert meta.context_files == [ref]

    def test_context_files_reject_unknown_types(self):
        with pytest.raises(TypeError, match="ContextFileRef"):
            CareMemory._build_care_metadata(
                query=None,
                context_files=["not-a-dict"],  # type: ignore[list-item]
                mage_metadata=None,
                display_name=None,
                description=None,
                tags=None,
            )

    def test_mage_metadata_round_trips(self):
        mage = {"mode": "deep", "domain": "finance", "stages_completed": 7}
        meta = CareMemory._build_care_metadata(
            query="q",
            context_files=None,
            mage_metadata=mage,
            display_name=None,
            description=None,
            tags=None,
        )
        assert meta.mage_metadata == mage


class TestApplyChainMetadata:
    def test_dict_chain_merges_into_metadata_block(self):
        chain = {"version": "1.1", "steps": []}
        meta = CareChainMetadata(
            task_description="run it",
            tags=["finance"],
            display_name="X",
        )
        out = CareMemory._apply_chain_metadata(chain, meta)
        assert out["steps"] == []  # original keys preserved
        assert out["metadata"]["task_description"] == "run it"
        assert out["metadata"]["tags"] == ["finance"]
        assert out["metadata"]["display_name"] == "X"

    def test_dict_chain_preserves_existing_metadata(self):
        chain = {"metadata": {"name": "old", "domain": "finance"}, "steps": []}
        meta = CareChainMetadata(task_description="run")
        out = CareMemory._apply_chain_metadata(chain, meta)
        # SDK's merge_into_content keeps non-overridden keys
        assert out["metadata"]["name"] == "old"
        assert out["metadata"]["domain"] == "finance"
        assert out["metadata"]["task_description"] == "run"

    def test_duck_typed_chain_object(self):
        class FakeChain:
            def __init__(self):
                self.metadata = {"name": "carl-fake"}

        chain = FakeChain()
        meta = CareChainMetadata(task_description="hi", display_name="d")
        out = CareMemory._apply_chain_metadata(chain, meta)
        assert out is chain
        assert chain.metadata["name"] == "carl-fake"
        assert chain.metadata["task_description"] == "hi"
        assert chain.metadata["display_name"] == "d"


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_pulls_url_and_key_from_care_config(self, tmp_path: Path):
        cfg = CareConfig.load(
            path=tmp_path / "missing.toml",
            env={
                "CARE_MEMORY__BASE_URL": "https://prod.example.com",
                "CARE_MEMORY__API_KEY": "sk-from-cfg",
                "CARE_MEMORY__TIMEOUT": "12",
            },
        )
        mem = CareMemory.from_config(cfg)
        assert mem.client._base_url == "https://prod.example.com"
        assert mem.client._http.headers.get("X-API-Key") == "sk-from-cfg"


# ---------------------------------------------------------------------------
# Save methods against a mocked Memory server
# ---------------------------------------------------------------------------


class TestSaveChain:
    @respx.mock
    def test_returns_entity_id_string(self, memory):
        captured: dict = {}

        def _handler(request):
            import json

            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "entity_type": "chain",
                    "entity_id": "e-1",
                    "version_id": "v-1",
                    "channel": "latest",
                },
            )

        respx.post(f"{BASE}/v1/chains").mock(side_effect=_handler)
        eid = memory.save_chain(
            {"version": "1.1", "steps": []},
            name="weather",
            query="weather report",
            domain="weather",
        )
        assert eid == "e-1"
        body = captured["body"]
        assert body["meta"]["name"] == "weather"
        # domain tag stamped automatically
        assert "domain:weather" in body["meta"]["tags"]
        # CARE metadata merged into content
        assert body["content"]["metadata"]["task_description"] == "weather report"

    @respx.mock
    def test_dict_chain_without_version_gets_default(self, memory):
        """Regression: Memory's validator 400s on a chain with no top-level
        ``version``; MAGE production chains (dicts) omit it. We must default it."""
        captured: dict = {}

        def _handler(request):
            import json

            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"entity_type": "chain", "entity_id": "e-2", "version_id": "v", "channel": "latest"},
            )

        respx.post(f"{BASE}/v1/chains").mock(side_effect=_handler)
        memory.save_chain(
            {"name": "News Agent", "steps": []},  # NO version (as MAGE emits)
            name="News Agent",
            query="topic",
        )
        assert captured["body"]["content"]["version"] == "1.0"

    @respx.mock
    def test_reasoningchain_object_without_version_gets_default(self, memory):
        """The SDK's chain_to_content (ReasoningChain.to_dict) doesn't emit a
        ``version`` either — the object path must be normalised + stamped too."""
        captured: dict = {}

        def _handler(request):
            import json

            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"entity_type": "chain", "entity_id": "e-3", "version_id": "v", "channel": "latest"},
            )

        class _FakeChain:  # duck-typed CARL chain: settable .metadata + .to_dict()
            def __init__(self):
                self.metadata = {}

            def to_dict(self):
                return {"name": "obj-chain", "steps": [], "metadata": self.metadata}

        respx.post(f"{BASE}/v1/chains").mock(side_effect=_handler)
        memory.save_chain(_FakeChain(), name="obj-chain", query="t")
        assert captured["body"]["content"]["version"] == "1.0"

    @respx.mock
    def test_parent_version_id_omitted_when_sdk_lacks_param(self, memory, monkeypatch):
        """Older gigaevo-client wheels reject unknown save_chain kwargs."""
        from care import memory as memory_mod

        monkeypatch.setattr(
            memory_mod,
            "_gigaevo_save_chain_param_names",
            lambda: frozenset({
                "self", "chain", "name", "tags", "when_to_use",
                "author", "entity_id", "channel", "evolution_meta",
            }),
        )

        def _handler(request):
            return httpx.Response(
                200,
                json={
                    "entity_type": "chain",
                    "entity_id": "e-p",
                    "version_id": "v",
                    "channel": "latest",
                },
            )

        respx.put(f"{BASE}/v1/chains/e-p").mock(side_effect=_handler)
        eid = memory.save_chain(
            {"version": "1.0", "steps": []},
            name="branch",
            entity_id="e-p",
            parent_version_id="vid-old",
            change_summary="tweak",
        )
        assert eid == "e-p"

    @respx.mock
    def test_existing_domain_tag_not_duplicated(self, memory):
        captured: dict = {}

        def _handler(request):
            import json

            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "entity_type": "chain",
                    "entity_id": "e-2",
                    "version_id": "v-2",
                    "channel": "latest",
                },
            )

        respx.post(f"{BASE}/v1/chains").mock(side_effect=_handler)
        memory.save_chain(
            {"version": "1.1", "steps": []},
            name="finance",
            query="q",
            domain="finance",
            tags=["domain:finance", "shared"],
        )
        tags = captured["body"]["meta"]["tags"]
        assert tags.count("domain:finance") == 1
        assert "shared" in tags

    @respx.mock
    def test_passes_context_files_into_metadata(self, memory):
        captured: dict = {}

        def _handler(request):
            import json

            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "entity_type": "chain",
                    "entity_id": "e-3",
                    "version_id": "v-3",
                    "channel": "latest",
                },
            )

        respx.post(f"{BASE}/v1/chains").mock(side_effect=_handler)
        memory.save_chain(
            {"version": "1.1", "steps": []},
            name="report",
            query="analyse report",
            context_files=[
                {"path": "in.pdf", "sha256": "a" * 64, "size_bytes": 1024},
            ],
        )
        files = captured["body"]["content"]["metadata"]["context_files"]
        assert len(files) == 1
        assert files[0]["path"] == "in.pdf"


class TestSaveAgentSkill:
    @respx.mock
    def test_builds_spec_from_manifest(self, memory):
        captured: dict = {}

        def _handler(request):
            import json

            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "entity_type": "agent_skill",
                    "entity_id": "skill-1",
                    "version_id": "v-1",
                    "channel": "latest",
                },
            )

        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=_handler)
        eid = memory.save_agent_skill(
            skill_uri="github://anthropics/skills/pdf",
            manifest={
                "name": "pdf-extract",
                "description": "Extract text from PDFs",
                "allowed-tools": ["Bash", "Read"],
                "tags": ["pdf", "extraction"],
            },
            sha256="c" * 64,
            instructions="Run pdftotext on the input file.",
        )
        assert eid == "skill-1"
        content = captured["body"]["content"]
        assert content["name"] == "pdf-extract"
        assert content["uri"] == "github://anthropics/skills/pdf"
        assert content["sha256"] == "c" * 64
        assert content["allowed_tools"] == ["Bash", "Read"]
        assert content["tags"] == ["pdf", "extraction"]
        assert content["instructions"] == "Run pdftotext on the input file."

    @respx.mock
    def test_name_override_wins_over_manifest(self, memory):
        captured: dict = {}

        def _handler(request):
            import json

            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "entity_type": "agent_skill",
                    "entity_id": "skill-2",
                    "version_id": "v-2",
                    "channel": "latest",
                },
            )

        respx.post(f"{BASE}/v1/agent-skills").mock(side_effect=_handler)
        memory.save_agent_skill(
            skill_uri="local:///tmp/skills/x",
            manifest={"name": "manifest-name", "description": "d"},
            sha256="d" * 64,
            name="caller-name",
        )
        assert captured["body"]["meta"]["name"] == "caller-name"
        assert captured["body"]["content"]["name"] == "caller-name"


class TestSearch:
    @respx.mock
    def test_forwards_query_and_returns_hits(self, memory):
        respx.post(f"{BASE}/v1/search/unified").mock(
            return_value=httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "entity_id": "e-1",
                            "entity_type": "chain",
                            "version_id": "v-1",
                            "score": 0.91,
                            "name": "weather",
                        }
                    ]
                },
            )
        )
        hits = memory.search("weather", entity_type="chain", top_k=5)
        assert len(hits) == 1
        assert hits[0]["entity_id"] == "e-1"


class TestHealthCheck:
    @respx.mock
    def test_returns_health_payload(self, memory):
        respx.get(f"{BASE}/health").mock(
            return_value=httpx.Response(
                200,
                json={"status": "ok", "postgres": "ok", "redis": "ok"},
            )
        )
        out = memory.health_check()
        assert out["status"] == "ok"
        assert out["postgres"] == "ok"


class TestGetChain:
    @respx.mock
    def test_returns_content_dict(self, memory):
        respx.get(f"{BASE}/v1/chains/ent-1").mock(
            return_value=httpx.Response(
                200,
                json={
                    "entity_type": "chain",
                    "entity_id": "ent-1",
                    "version_id": "v-1",
                    "channel": "latest",
                    "etag": "e-1",
                    "meta": {"name": "X"},
                    "content": {"steps": [{"prompt": "hi"}]},
                },
            )
        )
        chain = memory.get_chain("ent-1")
        assert chain == {"steps": [{"prompt": "hi"}]}

    @respx.mock
    def test_channel_forwarded(self, memory):
        captured: dict = {}

        def _handler(request):
            captured["channel"] = request.url.params.get("channel")
            return httpx.Response(
                200,
                json={
                    "entity_type": "chain",
                    "entity_id": "ent-1",
                    "version_id": "v-1",
                    "channel": "stable",
                    "etag": "e-1",
                    "meta": {},
                    "content": {"steps": []},
                },
            )

        respx.get(f"{BASE}/v1/chains/ent-1").mock(side_effect=_handler)
        memory.get_chain("ent-1", channel="stable")
        assert captured["channel"] == "stable"


class TestListEntities:
    @respx.mock
    def test_lists_chains(self, memory):
        respx.get(f"{BASE}/v1/chains").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "entity_type": "chain",
                        "entity_id": "ch-1",
                        "version_id": "v-1",
                        "channel": "latest",
                        "etag": "e-1",
                        "meta": {"name": "First"},
                        "content": {},
                        "display_name": "First",
                        "favourite": False,
                        "run_count": 3,
                    },
                ],
            )
        )
        rows = memory.list_entities(entity_type="chain")
        assert len(rows) == 1
        assert rows[0]["entity_id"] == "ch-1"
        assert rows[0]["display_name"] == "First"
        assert rows[0]["run_count"] == 3

    @respx.mock
    def test_lists_agent_skills(self, memory):
        respx.get(f"{BASE}/v1/agent-skills").mock(
            return_value=httpx.Response(200, json=[]),
        )
        rows = memory.list_entities(entity_type="agent_skill")
        assert rows == []

    @respx.mock
    def test_lists_memory_cards(self, memory):
        respx.get(f"{BASE}/v1/memory-cards").mock(
            return_value=httpx.Response(200, json=[]),
        )
        rows = memory.list_entities(entity_type="memory_card")
        assert rows == []

    @respx.mock
    def test_forwards_filter_params(self, memory):
        captured: dict = {}

        def _handler(request):
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json=[])

        respx.get(f"{BASE}/v1/chains").mock(side_effect=_handler)
        memory.list_entities(
            entity_type="chain",
            limit=7,
            offset=2,
            channel="stable",
            namespace="team-a",
            tags=["weather"],
            q="storm",
            favourites_only=True,
            sort_by="last_run_at",
            sort_dir="asc",
        )
        params = captured["params"]
        assert params["limit"] == "7"
        assert params["offset"] == "2"
        assert params["channel"] == "stable"
        assert params["namespace"] == "team-a"
        assert params["tags"] == "weather"
        assert params["q"] == "storm"
        assert params["favourites_only"] == "true"
        assert params["sort_by"] == "last_run_at"
        assert params["sort_dir"] == "asc"

    def test_unknown_entity_type_raises(self, memory):
        with pytest.raises(ValueError, match="unsupported entity_type"):
            memory.list_entities(entity_type="bogus")


def _entity_row(
    *,
    entity_id: str,
    name: str,
    content: dict | None = None,
    display_name: str | None = None,
    entity_type: str = "chain",
) -> dict:
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "version_id": "v-1",
        "channel": "latest",
        "etag": "etag-1",
        "meta": {"name": name},
        "content": content or {},
        "display_name": display_name if display_name is not None else name,
        "description": None,
        "favourite": False,
        "run_count": 0,
        "last_run_at": None,
    }


class TestFindEntityByName:
    @respx.mock
    def test_chain_match_returns_dict(self, memory):
        respx.get(f"{BASE}/v1/chains").mock(
            return_value=httpx.Response(
                200,
                json=[
                    _entity_row(
                        entity_id="ch-1",
                        name="Storm Watcher",
                        content={"steps": [{"prompt": "go"}]},
                    ),
                    _entity_row(
                        entity_id="ch-2",
                        name="Other",
                    ),
                ],
            )
        )
        found = memory.find_entity_by_name(
            name="Storm Watcher",
            entity_type="chain",
        )
        assert found is not None
        assert found["entity_id"] == "ch-1"
        assert found["content"] == {"steps": [{"prompt": "go"}]}

    @respx.mock
    def test_chain_no_match_returns_none(self, memory):
        respx.get(f"{BASE}/v1/chains").mock(
            return_value=httpx.Response(200, json=[]),
        )
        found = memory.find_entity_by_name(
            name="Nothing",
            entity_type="chain",
        )
        assert found is None

    @respx.mock
    def test_chain_forwards_q_param(self, memory):
        captured: dict = {}

        def _handler(request):
            captured["q"] = request.url.params.get("q")
            captured["namespace"] = request.url.params.get("namespace")
            return httpx.Response(200, json=[])

        respx.get(f"{BASE}/v1/chains").mock(side_effect=_handler)
        memory.find_entity_by_name(
            name="hello world",
            entity_type="chain",
            namespace="team-a",
        )
        assert captured["q"] == "hello world"
        assert captured["namespace"] == "team-a"

    @respx.mock
    def test_agent_skill_match(self, memory):
        respx.get(f"{BASE}/v1/agent-skills").mock(
            return_value=httpx.Response(
                200,
                json=[
                    _entity_row(
                        entity_id="sk-1",
                        name="pdf-extract",
                        entity_type="agent_skill",
                        content={"sha256": "a" * 64},
                    ),
                ],
            )
        )
        found = memory.find_entity_by_name(
            name="pdf-extract",
            entity_type="agent_skill",
        )
        assert found is not None
        assert found["entity_id"] == "sk-1"

    @respx.mock
    def test_memory_card_match(self, memory):
        respx.get(f"{BASE}/v1/memory-cards").mock(
            return_value=httpx.Response(
                200,
                json=[
                    _entity_row(
                        entity_id="mc-1",
                        name="lesson-1",
                        entity_type="memory_card",
                        content={"body": "x"},
                    ),
                ],
            )
        )
        found = memory.find_entity_by_name(
            name="lesson-1",
            entity_type="memory_card",
        )
        assert found is not None
        assert found["entity_id"] == "mc-1"

    @respx.mock
    def test_falls_back_to_meta_name(self, memory):
        # display_name absent — fall back to meta.name.
        row = _entity_row(
            entity_id="ch-3",
            name="Storm Watcher",
        )
        row["display_name"] = None
        respx.get(f"{BASE}/v1/chains").mock(
            return_value=httpx.Response(200, json=[row]),
        )
        found = memory.find_entity_by_name(
            name="Storm Watcher",
            entity_type="chain",
        )
        assert found is not None
        assert found["entity_id"] == "ch-3"

    @respx.mock
    def test_falls_back_to_care_metadata_display_name(self, memory):
        row = _entity_row(
            entity_id="ch-4",
            name="server-name",
            content={
                "metadata": {
                    "care": {"display_name": "Care Display"},
                },
            },
        )
        row["display_name"] = None
        row["meta"] = {}
        respx.get(f"{BASE}/v1/chains").mock(
            return_value=httpx.Response(200, json=[row]),
        )
        found = memory.find_entity_by_name(
            name="Care Display",
            entity_type="chain",
        )
        assert found is not None
        assert found["entity_id"] == "ch-4"

    def test_unknown_entity_type_raises(self, memory):
        with pytest.raises(ValueError, match="unsupported entity_type"):
            memory.find_entity_by_name(
                name="x",
                entity_type="something_else",
            )

    def test_duck_typed_lookup_contract(self):
        # Verify the docstring's contract — what conflict.py expects.
        # Pure stub, no SDK; ensures the return shape matches the
        # `find_entity_by_name` contract that detect_conflict checks.
        from care.conflict import detect_conflict

        class _Stub:
            def find_entity_by_name(self, *, name, entity_type, namespace=None):
                if name == "Match":
                    return {
                        "entity_id": "stub-1",
                        "content": {"steps": []},
                    }
                return None

        report = detect_conflict(
            _Stub(),
            name="Match",
            entity_type="chain",
            incoming_content={"steps": [{"prompt": "different"}]},
        )
        assert report is not None
        assert report.existing_entity_id == "stub-1"
        assert report.is_conflict is True
