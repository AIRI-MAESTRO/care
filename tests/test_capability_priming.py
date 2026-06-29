"""Tests for ``care.capability_priming`` (TODO §4 P2).

The priming bridge has two halves:

1. **Shape correctness** — given a :class:`CapabilityCatalog`
   (populated by going through real ``build_catalog`` on tmp_path),
   :func:`build_capability_payload` produces dicts that match the
   field set MAGE's ``CapabilityContext`` / ``AgentSkillEntry``
   expect.
2. **Lazy import** — :meth:`CapabilityPayload.to_mage_context`
   raises a friendly :class:`CapabilityPrimingError` when
   ``mmar_mage`` isn't on the install — which is the case in CARE's
   dev env. Perfect for exercising the missing-dep path for real
   instead of mocking.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from care.capability_priming import (
    CapabilityPayload,
    CapabilityPrimingError,
    build_capability_payload,
)
from care.catalog import (
    CapabilityCatalog,
    CapabilityCatalogEntry,
    build_catalog,
)


def _write_skill(folder: Path, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    skill_md = folder / "SKILL.md"
    skill_md.write_text(body, encoding="utf-8")
    return skill_md


# ---------------------------------------------------------------------------
# CapabilityPayload shape
# ---------------------------------------------------------------------------


class TestPayloadShape:
    def test_empty_payload(self):
        payload = CapabilityPayload()
        assert payload.is_empty
        assert payload.tools == ()
        assert payload.mcp_servers == ()
        assert payload.agent_skills == ()
        assert payload.environment_id == "default"

    def test_to_dict_roundtrips(self):
        payload = CapabilityPayload(
            tools=({"name": "t", "source": "s"},),
            mcp_servers=({"name": "m"},),
            agent_skills=({"name": "skill", "uri": "local://x"},),
            environment_id="custom",
        )
        d = payload.to_dict()
        assert d["tools"] == [{"name": "t", "source": "s"}]
        assert d["mcp_servers"] == [{"name": "m"}]
        assert d["agent_skills"] == [{"name": "skill", "uri": "local://x"}]
        assert d["environment_id"] == "custom"
        # to_dict returns deep copies — mutating doesn't bleed back.
        d["tools"][0]["name"] = "MUTATED"
        assert payload.tools[0]["name"] == "t"

    def test_payload_is_frozen(self):
        payload = CapabilityPayload()
        with pytest.raises(Exception):
            payload.environment_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_capability_payload — tools
# ---------------------------------------------------------------------------


class TestToolPriming:
    def test_tool_entry_maps_to_planner_dict(self, tmp_path: Path):
        (tmp_path / "weather.py").write_text(
            '"""Fetch weather"""\nimport sys\nprint(sys.argv)\n'
        )
        catalog = build_catalog(tools_path=tmp_path)
        payload = build_capability_payload(catalog)
        assert len(payload.tools) == 1
        tool = payload.tools[0]
        assert tool["name"] == "weather"
        assert tool["source"].endswith("weather.py")
        assert tool["description"] == "Fetch weather"
        assert tool["tags"] == []

    def test_tools_drop_memory_cards(self, tmp_path: Path):
        # Memory cards never appear in the planner-context tools list.
        entries = (
            CapabilityCatalogEntry(
                kind="memory_card", name="card", source="memory://x"
            ),
            CapabilityCatalogEntry(
                kind="tool", name="real_tool", source=str(tmp_path / "t.py")
            ),
        )
        catalog = CapabilityCatalog(entries=entries)
        payload = build_capability_payload(catalog)
        assert [t["name"] for t in payload.tools] == ["real_tool"]
        assert payload.mcp_servers == ()
        assert payload.agent_skills == ()


# ---------------------------------------------------------------------------
# build_capability_payload — MCP servers
# ---------------------------------------------------------------------------


class TestMcpPriming:
    def test_mcp_entry_preserves_command_args_and_config(self, tmp_path: Path):
        mcp = tmp_path / "mcp.toml"
        mcp.write_text(
            '[servers.weather]\n'
            'command = "node"\n'
            'args = ["/opt/weather.js"]\n'
            'description = "Forecast server"\n'
            'tags = ["weather"]\n'
        )
        catalog = build_catalog(mcp_config_path=mcp)
        payload = build_capability_payload(catalog)
        assert len(payload.mcp_servers) == 1
        server = payload.mcp_servers[0]
        assert server["name"] == "weather"
        assert server["description"] == "Forecast server"
        assert server["tags"] == ["weather"]
        # The top-level convenience fields the planner uses…
        assert server["command"] == "node"
        assert server["args"] == ["/opt/weather.js"]
        # …plus the full TOML body under config for round-trip.
        assert server["config"]["command"] == "node"
        assert server["config"]["description"] == "Forecast server"

    def test_mcp_without_args_field_defaults_to_empty_list(self, tmp_path: Path):
        mcp = tmp_path / "mcp.toml"
        mcp.write_text('[servers.bare]\ncommand = "x"\n')
        catalog = build_catalog(mcp_config_path=mcp)
        payload = build_capability_payload(catalog)
        assert payload.mcp_servers[0]["args"] == []


# ---------------------------------------------------------------------------
# build_capability_payload — agent_skill
# ---------------------------------------------------------------------------


class TestAgentSkillPriming:
    def test_skill_entry_carries_uri_sha_and_manifest_summary(
        self, tmp_path: Path
    ):
        body = (
            "---\n"
            "name: weather\n"
            'description: "Look up forecasts"\n'
            "tags:\n  - weather\n"
            "allowed-tools:\n  - WebFetch\n"
            "---\n"
            "# instructions\n"
        )
        skill_md = _write_skill(tmp_path / "weather", body)
        catalog = build_catalog(skills_paths=[tmp_path])
        payload = build_capability_payload(catalog)
        assert len(payload.agent_skills) == 1
        skill = payload.agent_skills[0]
        assert skill["name"] == "weather"
        assert skill["description"] == "Look up forecasts"
        # URI is local:// + the absolute SKILL.md path.
        assert skill["uri"].startswith("local://")
        assert skill["uri"].endswith("/SKILL.md")
        # SHA matches the actual file bytes.
        expected = hashlib.sha256(skill_md.read_bytes()).hexdigest()
        assert skill["sha256"] == expected
        assert skill["manifest_summary"] == "Look up forecasts"
        assert skill["tags"] == ["weather"]
        assert skill["allowed_tools"] == ["WebFetch"]
        # Provenance label — locally-discovered skills are tagged
        # "local" so the planner can show the user where they came
        # from in the prompt rationale.
        assert skill["source"] == "local"
        # No relevance / why on a discovery-only entry.
        assert skill["relevance"] == 0.0
        assert skill["why"] == ""

    def test_compute_sha_false_skips_hashing(self, tmp_path: Path):
        _write_skill(tmp_path / "x", "---\nname: x\n---\n")
        catalog = build_catalog(skills_paths=[tmp_path])
        payload = build_capability_payload(catalog, compute_skill_sha=False)
        assert payload.agent_skills[0]["sha256"] == ""

    def test_missing_skill_md_yields_empty_sha(self, tmp_path: Path):
        # Build a catalog entry pointing at a path that doesn't exist
        # any more — simulates a delete between scan and prime.
        entry = CapabilityCatalogEntry(
            kind="agent_skill",
            name="ghost",
            source=str(tmp_path / "ghost" / "SKILL.md"),
            summary="vanished",
            metadata={"manifest": {"name": "ghost", "description": "vanished"}},
        )
        catalog = CapabilityCatalog(entries=(entry,))
        payload = build_capability_payload(catalog)
        # No raise; SHA is empty so the planner falls back to URI.
        assert payload.agent_skills[0]["sha256"] == ""

    def test_environment_id_forwarded(self, tmp_path: Path):
        _write_skill(tmp_path / "x", "---\nname: x\n---\n")
        catalog = build_catalog(skills_paths=[tmp_path])
        payload = build_capability_payload(catalog, environment_id="prod")
        assert payload.environment_id == "prod"


# ---------------------------------------------------------------------------
# Integration — all four sources together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_combined_catalog_yields_all_fields(self, tmp_path: Path):
        _write_skill(tmp_path / "sk", "---\nname: sk\n---\n")
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "t.py").write_text('"""tool body"""\n')
        mcp = tmp_path / "mcp.toml"
        mcp.write_text('[servers.m]\ncommand = "x"\n')
        catalog = build_catalog(
            skills_paths=[tmp_path],
            tools_path=tmp_path / "tools",
            mcp_config_path=mcp,
        )
        payload = build_capability_payload(catalog)
        assert [t["name"] for t in payload.tools] == ["t"]
        assert [s["name"] for s in payload.mcp_servers] == ["m"]
        assert [s["name"] for s in payload.agent_skills] == ["sk"]
        assert not payload.is_empty


# ---------------------------------------------------------------------------
# Lazy import — exercises whichever branch the install actually has
# ---------------------------------------------------------------------------


def _mage_installed() -> bool:
    try:
        import mmar_mage  # noqa: F401
    except ImportError:
        return False
    return True


class TestMageImport:
    @pytest.mark.skipif(
        _mage_installed(),
        reason="mmar_mage is installed; exercise the success branch instead",
    )
    def test_to_mage_context_raises_when_mage_missing(self):
        """Default CARE install has no ``mmar_mage`` — this asserts
        the friendly install-hint error."""
        payload = CapabilityPayload(
            agent_skills=(
                {
                    "name": "x",
                    "uri": "local://x",
                    "description": "y",
                },
            ),
        )
        with pytest.raises(
            CapabilityPrimingError, match="mmar_mage is not installed"
        ):
            payload.to_mage_context()

    @pytest.mark.skipif(
        not _mage_installed(),
        reason="mmar_mage isn't installed; skip the success branch",
    )
    def test_to_mage_context_returns_real_capability_context(self):
        """With the ``mage`` extra installed, the lazy import works
        and we get back a real ``mmar_mage.CapabilityContext``."""
        from mmar_mage.agents.capability_lookup_agent import CapabilityContext

        payload = CapabilityPayload(
            tools=({"name": "t", "source": "/x.py"},),
            mcp_servers=({"name": "m", "command": "x"},),
            agent_skills=(
                {
                    "name": "skill",
                    "uri": "local:///x/SKILL.md",
                    "description": "y",
                },
            ),
            environment_id="prod",
        )
        ctx = payload.to_mage_context()
        assert isinstance(ctx, CapabilityContext)
        assert ctx.environment_id == "prod"
        assert [t["name"] for t in ctx.tools] == ["t"]
        assert [s["name"] for s in ctx.mcp_servers] == ["m"]
        assert [s.name for s in ctx.agent_skills] == ["skill"]
