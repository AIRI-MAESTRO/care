"""Tests for ``care.catalog`` (TODO §8 P1).

The catalog module is best-effort: a broken file in any source
shouldn't kill the whole scan. So each scanner gets:

1. **Happy-path coverage** — feed it a realistic input and assert
   the resulting :class:`CapabilityCatalogEntry` is shaped right.
2. **Failure-mode coverage** — feed it something malformed and
   assert the error lands on ``CapabilityCatalog.errors`` rather
   than raising.

The memory-card path uses an in-memory stub that quacks like
``CareMemory.search`` — keeps the test isolated from gigaevo-client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from care.catalog import (
    CapabilityCatalog,
    CapabilityCatalogEntry,
    build_catalog,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_skill(root: Path, name: str, body: str) -> Path:
    """Create ``root/<name>/SKILL.md`` and return the path."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(body, encoding="utf-8")
    return skill_file


class _StubMemory:
    """Stand-in for ``CareMemory`` exposing the ``search`` shape the
    catalog uses."""

    def __init__(self, hits: list[Any] | None = None, raise_exc: Exception | None = None):
        self._hits = hits or []
        self._raise = raise_exc
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, *, entity_type: str, top_k: int) -> list[Any]:
        self.calls.append(
            {"query": query, "entity_type": entity_type, "top_k": top_k}
        )
        if self._raise is not None:
            raise self._raise
        return self._hits


# ---------------------------------------------------------------------------
# CapabilityCatalogEntry / CapabilityCatalog basics
# ---------------------------------------------------------------------------


class TestCatalogShape:
    def test_entry_is_frozen(self):
        entry = CapabilityCatalogEntry(
            kind="tool", name="x", source="/tmp/x.py"
        )
        with pytest.raises(Exception):  # FrozenInstanceError subclass of AttributeError
            entry.name = "y"  # type: ignore[misc]

    def test_default_factory_independence(self):
        a = CapabilityCatalogEntry(kind="tool", name="a", source="a")
        b = CapabilityCatalogEntry(kind="tool", name="b", source="b")
        # `metadata={}` shouldn't be shared across instances.
        assert a.metadata is not b.metadata
        assert a.tags == ()
        assert b.tags == ()

    def test_empty_catalog(self):
        cat = CapabilityCatalog()
        assert cat.is_empty
        assert cat.entries == ()
        assert cat.errors == ()
        assert cat.by_kind("agent_skill") == ()
        assert cat.by_tag("anything") == ()

    def test_by_kind_and_by_tag(self):
        entries = (
            CapabilityCatalogEntry(kind="tool", name="a", source="a", tags=("io",)),
            CapabilityCatalogEntry(kind="agent_skill", name="b", source="b", tags=("io", "fs")),
            CapabilityCatalogEntry(kind="mcp_server", name="c", source="c", tags=("net",)),
        )
        cat = CapabilityCatalog(entries=entries)
        assert not cat.is_empty
        assert [e.name for e in cat.by_kind("agent_skill")] == ["b"]
        assert [e.name for e in cat.by_kind("tool")] == ["a"]
        assert {e.name for e in cat.by_tag("io")} == {"a", "b"}
        assert [e.name for e in cat.by_tag("net")] == ["c"]
        assert cat.by_tag("missing") == ()


# ---------------------------------------------------------------------------
# Skill discovery
# ---------------------------------------------------------------------------


class TestSkillDiscovery:
    def test_parses_frontmatter_with_scalars_and_list(self, tmp_path: Path):
        body = (
            "---\n"
            "name: weather\n"
            'description: "Fetch a forecast"\n'
            "tags:\n"
            "  - weather\n"
            "  - external\n"
            "allowed-tools:\n"
            "  - WebFetch\n"
            "  - Bash\n"
            "---\n"
            "# Weather skill body\n"
        )
        skill_path = _write_skill(tmp_path, "weather", body)
        cat = build_catalog(skills_paths=[tmp_path])
        assert len(cat.entries) == 1
        entry = cat.entries[0]
        assert entry.kind == "agent_skill"
        assert entry.name == "weather"
        assert entry.source == str(skill_path)
        assert entry.summary == "Fetch a forecast"
        assert entry.tags == ("weather", "external")
        assert entry.metadata["allowed_tools"] == ["WebFetch", "Bash"]
        assert "Weather skill body" in entry.metadata["instructions_preview"]
        assert cat.errors == ()

    def test_falls_back_to_directory_name(self, tmp_path: Path):
        # No frontmatter at all — name comes from the directory.
        _write_skill(tmp_path, "fallback", "# just markdown, no frontmatter\n")
        cat = build_catalog(skills_paths=[tmp_path])
        assert len(cat.entries) == 1
        assert cat.entries[0].name == "fallback"
        assert cat.entries[0].summary == ""
        assert cat.entries[0].tags == ()

    def test_skips_nonexistent_path(self):
        cat = build_catalog(skills_paths=[Path("/definitely/does/not/exist")])
        assert cat.entries == ()
        assert cat.errors == ()

    def test_records_error_when_path_is_a_file(self, tmp_path: Path):
        # User pointed at a file instead of a dir.
        f = tmp_path / "file.md"
        f.write_text("not a dir\n")
        cat = build_catalog(skills_paths=[f])
        assert cat.entries == ()
        assert any("not a directory" in e for e in cat.errors)

    def test_finds_nested_skills(self, tmp_path: Path):
        # Catalog walks subdirectories.
        _write_skill(tmp_path / "vendor", "alpha", "---\nname: alpha\n---\n")
        _write_skill(tmp_path / "vendor" / "deep", "beta", "---\nname: beta\n---\n")
        cat = build_catalog(skills_paths=[tmp_path])
        names = {e.name for e in cat.entries}
        assert names == {"alpha", "beta"}

    def test_summary_first_line_only(self, tmp_path: Path):
        body = (
            "---\n"
            "name: multiline\n"
            'description: "First line only"\n'
            "---\n"
            "body\n"
        )
        _write_skill(tmp_path, "multi", body)
        cat = build_catalog(skills_paths=[tmp_path])
        assert cat.entries[0].summary == "First line only"

    def test_single_quoted_value(self, tmp_path: Path):
        body = (
            "---\n"
            "name: quoted\n"
            "description: 'single-quoted value'\n"
            "---\n"
        )
        _write_skill(tmp_path, "q", body)
        cat = build_catalog(skills_paths=[tmp_path])
        assert cat.entries[0].summary == "single-quoted value"
        assert cat.entries[0].name == "quoted"

    def test_tilde_path_is_expanded(self, tmp_path: Path, monkeypatch):
        # Use HOME to verify expansion happens through ``_expand``.
        monkeypatch.setenv("HOME", str(tmp_path))
        _write_skill(tmp_path / "skills", "hello", "---\nname: hello\n---\n")
        cat = build_catalog(skills_paths=["~/skills"])
        assert {e.name for e in cat.entries} == {"hello"}

    def test_string_paths_accepted(self, tmp_path: Path):
        _write_skill(tmp_path, "strpath", "---\nname: strpath\n---\n")
        cat = build_catalog(skills_paths=[str(tmp_path)])
        assert {e.name for e in cat.entries} == {"strpath"}


# ---------------------------------------------------------------------------
# MCP server discovery
# ---------------------------------------------------------------------------


class TestMcpDiscovery:
    def test_happy_path(self, tmp_path: Path):
        cfg = tmp_path / "mcp_servers.toml"
        cfg.write_text(
            '[servers.weather]\n'
            'command = "node"\n'
            'args = ["/opt/mcp/weather.js"]\n'
            'description = "Fetches forecasts"\n'
            'tags = ["weather", "net"]\n'
            '\n'
            '[servers.search]\n'
            'command = "uvx"\n'
            'args = ["mcp-search"]\n'
        )
        cat = build_catalog(mcp_config_path=cfg)
        servers = cat.by_kind("mcp_server")
        names = [s.name for s in servers]
        # Sorted alphabetically by name (catalog deterministic sort).
        assert names == ["search", "weather"]
        weather = next(s for s in servers if s.name == "weather")
        assert weather.source == "node /opt/mcp/weather.js"
        assert weather.summary == "Fetches forecasts"
        assert weather.tags == ("weather", "net")
        assert weather.metadata["command"] == "node"

    def test_malformed_toml_records_error(self, tmp_path: Path):
        cfg = tmp_path / "broken.toml"
        cfg.write_text("this is = not [valid toml\n")
        cat = build_catalog(mcp_config_path=cfg)
        assert cat.entries == ()
        assert any("could not parse" in e for e in cat.errors)

    def test_skips_when_path_missing(self, tmp_path: Path):
        cat = build_catalog(mcp_config_path=tmp_path / "missing.toml")
        assert cat.entries == ()
        assert cat.errors == ()

    def test_servers_not_a_table(self, tmp_path: Path):
        cfg = tmp_path / "weird.toml"
        cfg.write_text('servers = "not a table"\n')
        cat = build_catalog(mcp_config_path=cfg)
        assert cat.entries == ()
        assert any("must be a table" in e for e in cat.errors)

    def test_individual_server_not_a_table(self, tmp_path: Path):
        cfg = tmp_path / "mix.toml"
        cfg.write_text(
            'servers = { broken = "stringy", good = { command = "x" } }\n'
        )
        cat = build_catalog(mcp_config_path=cfg)
        # The "good" one still lands; the "broken" one becomes an error.
        names = [e.name for e in cat.by_kind("mcp_server")]
        assert names == ["good"]
        assert any("broken" in e for e in cat.errors)

    def test_empty_servers_table(self, tmp_path: Path):
        cfg = tmp_path / "empty.toml"
        cfg.write_text("[servers]\n")
        cat = build_catalog(mcp_config_path=cfg)
        assert cat.entries == ()
        assert cat.errors == ()


# ---------------------------------------------------------------------------
# Tools dir discovery
# ---------------------------------------------------------------------------


class TestToolsDiscovery:
    def test_lists_py_files_and_skips_underscored(self, tmp_path: Path):
        (tmp_path / "alpha.py").write_text('"""Alpha tool."""\nprint(1)\n')
        (tmp_path / "beta.py").write_text("# Description: beta does things\nprint(2)\n")
        (tmp_path / "__init__.py").write_text("# package marker\n")
        (tmp_path / "_private.py").write_text("# hidden\n")
        (tmp_path / "readme.md").write_text("# not a tool\n")
        cat = build_catalog(tools_path=tmp_path)
        names = sorted(e.name for e in cat.by_kind("tool"))
        assert names == ["alpha", "beta"]
        # Summary picks up docstring / description-comment.
        summaries = {e.name: e.summary for e in cat.by_kind("tool")}
        assert summaries["alpha"] == "Alpha tool."
        assert summaries["beta"] == "beta does things"

    def test_missing_dir(self, tmp_path: Path):
        cat = build_catalog(tools_path=tmp_path / "nope")
        assert cat.entries == ()
        assert cat.errors == ()

    def test_path_is_a_file(self, tmp_path: Path):
        f = tmp_path / "file.py"
        f.write_text("print(1)\n")
        cat = build_catalog(tools_path=f)
        assert cat.entries == ()
        assert any("not a directory" in e for e in cat.errors)

    def test_no_summary_when_no_docstring(self, tmp_path: Path):
        (tmp_path / "bare.py").write_text("x = 1\ny = 2\n")
        cat = build_catalog(tools_path=tmp_path)
        bare = cat.by_kind("tool")[0]
        assert bare.summary == ""
        assert bare.metadata["line_count"] == 2


# ---------------------------------------------------------------------------
# Memory cards
# ---------------------------------------------------------------------------


class TestMemoryCards:
    def test_happy_path(self):
        memory = _StubMemory(
            hits=[
                {
                    "entity_id": "ent-1",
                    "name": "Useful card",
                    "description": "Helpful tip",
                    "meta": {"tags": ["capability", "tip"]},
                },
                {
                    "entity_id": "ent-2",
                    "name": "Another",
                    "description": "Second",
                    "meta": {"tags": []},
                },
            ]
        )
        cat = build_catalog(memory=memory)
        cards = cat.by_kind("memory_card")
        assert [c.name for c in cards] == ["Another", "Useful card"]
        first = next(c for c in cards if c.name == "Useful card")
        assert first.source == "memory://ent-1"
        assert first.summary == "Helpful tip"
        assert "capability" in first.tags
        # Stub was called with our defaults.
        assert memory.calls == [
            {"query": "capability", "entity_type": "memory_card", "top_k": 50}
        ]

    def test_custom_tag_and_top_k(self):
        memory = _StubMemory(hits=[])
        build_catalog(memory=memory, memory_card_tag="customcap", memory_top_k=3)
        assert memory.calls[0]["query"] == "customcap"
        assert memory.calls[0]["top_k"] == 3

    def test_search_error_recorded(self):
        memory = _StubMemory(raise_exc=RuntimeError("memory down"))
        cat = build_catalog(memory=memory)
        assert cat.entries == ()
        assert any("memory_card search failed" in e for e in cat.errors)

    def test_non_dict_hits_skipped(self):
        memory = _StubMemory(hits=[{"entity_id": "ok", "name": "ok"}, "bad"])
        cat = build_catalog(memory=memory)
        # Only the dict hit lands.
        assert [e.name for e in cat.by_kind("memory_card")] == ["ok"]

    def test_missing_entity_id_falls_back(self):
        memory = _StubMemory(hits=[{"name": "no-id"}])
        cat = build_catalog(memory=memory)
        e = cat.by_kind("memory_card")[0]
        assert e.source == "memory://"

    def test_memory_none_skips(self):
        cat = build_catalog(memory=None)
        assert cat.entries == ()


# ---------------------------------------------------------------------------
# Integration: combined sources
# ---------------------------------------------------------------------------


class TestCombined:
    def test_all_four_sources_combined_and_sorted(self, tmp_path: Path):
        # 1. Skill
        skills_root = tmp_path / "skills"
        _write_skill(skills_root, "skill1", "---\nname: skillA\n---\n")
        # 2. MCP
        mcp = tmp_path / "mcp.toml"
        mcp.write_text('[servers.mcpA]\ncommand = "x"\n')
        # 3. Tool
        tools = tmp_path / "tools"
        tools.mkdir()
        (tools / "toolA.py").write_text('"""docline"""\n')
        # 4. Memory card
        memory = _StubMemory(hits=[{"entity_id": "e1", "name": "cardA"}])
        cat = build_catalog(
            skills_paths=[skills_root],
            mcp_config_path=mcp,
            tools_path=tools,
            memory=memory,
        )
        # Deterministic ordering: kind first, then name (case-insensitive).
        kinds = [e.kind for e in cat.entries]
        assert kinds == ["agent_skill", "mcp_server", "memory_card", "tool"]

    def test_each_source_independent_on_error(self, tmp_path: Path):
        # Skill OK; MCP broken; tools dir is a file (error); memory raises.
        skills_root = tmp_path / "skills"
        _write_skill(skills_root, "ok", "---\nname: ok\n---\n")
        mcp = tmp_path / "broken.toml"
        mcp.write_text("invalid = = ")
        tools_file = tmp_path / "file.py"
        tools_file.write_text("# noop\n")
        memory = _StubMemory(raise_exc=RuntimeError("boom"))
        cat = build_catalog(
            skills_paths=[skills_root],
            mcp_config_path=mcp,
            tools_path=tools_file,
            memory=memory,
        )
        # The good source still produced an entry.
        names = [e.name for e in cat.entries]
        assert names == ["ok"]
        # And every broken source contributed an error.
        joined = " | ".join(cat.errors)
        assert "could not parse" in joined
        assert "not a directory" in joined
        assert "memory_card search failed" in joined
