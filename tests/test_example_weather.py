"""Structural pin for ``examples/weather/`` (TODO §10 P2).

The weather example is the canonical "read this when getting started"
artefact CARE ships. The README walks through three CLI surfaces
(`care validate`, `care catalog`, `care import`); each surface uses
a different CARE primitive. These tests pin the example against
every primitive it claims to use, so drift in either direction
(an example update that breaks parse / a primitive change that
breaks the example) fails CI immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

from care.bulk_import import import_chains
from care.catalog import build_catalog
from care.preflight import validate_chain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = PROJECT_ROOT / "examples" / "weather"


# ---------------------------------------------------------------------------
# Files exist
# ---------------------------------------------------------------------------


class TestFilesPresent:
    def test_directory_exists(self):
        assert EXAMPLE_DIR.is_dir(), (
            f"{EXAMPLE_DIR} is missing — the weather example must "
            "live in `examples/weather/`."
        )

    def test_required_files_present(self):
        for name in ("chain.json", "mcp_servers.toml", "README.md"):
            assert (EXAMPLE_DIR / name).is_file(), (
                f"`examples/weather/{name}` is missing."
            )


# ---------------------------------------------------------------------------
# chain.json
# ---------------------------------------------------------------------------


class TestChainJson:
    def test_chain_is_valid_json(self):
        data = json.loads((EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data["task_description"]
        assert isinstance(data["steps"], list) and len(data["steps"]) == 2

    def test_chain_parses_via_validate_chain(self):
        raw = (EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8")
        result = validate_chain(raw)
        assert result.is_valid, (
            "examples/weather/chain.json failed to parse: "
            + " | ".join(result.parse_errors)
        )

    def test_steps_have_expected_kinds(self):
        data = json.loads((EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8"))
        kinds = [s["step_type"] for s in data["steps"]]
        # The example walks "MCP fetch → LLM summarise" — the README
        # asserts this layout, so we pin it.
        assert kinds == ["mcp", "llm"]

    def test_mcp_step_references_weather_server(self):
        data = json.loads((EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8"))
        mcp_step = data["steps"][0]
        assert mcp_step["step_config"]["server"]["server_name"] == "weather"
        # The MCP TOML file must register a server under the same name —
        # tested separately below.

    def test_llm_step_depends_on_mcp_step(self):
        data = json.loads((EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8"))
        llm_step = data["steps"][1]
        assert llm_step["dependencies"] == [1]


# ---------------------------------------------------------------------------
# mcp_servers.toml
# ---------------------------------------------------------------------------


class TestMcpServersToml:
    def test_catalog_discovers_weather_server(self):
        catalog = build_catalog(
            mcp_config_path=EXAMPLE_DIR / "mcp_servers.toml",
        )
        servers = catalog.by_kind("mcp_server")
        names = [s.name for s in servers]
        assert names == ["weather"]
        assert catalog.errors == ()

    def test_weather_server_carries_tags_and_summary(self):
        catalog = build_catalog(
            mcp_config_path=EXAMPLE_DIR / "mcp_servers.toml",
        )
        weather = catalog.by_kind("mcp_server")[0]
        assert "weather" in weather.tags
        assert weather.summary  # non-empty


# ---------------------------------------------------------------------------
# Cross-file consistency
# ---------------------------------------------------------------------------


class TestCrossFileConsistency:
    def test_chain_mcp_server_name_matches_toml(self):
        chain = json.loads(
            (EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8")
        )
        chain_server = chain["steps"][0]["step_config"]["server"]["server_name"]

        catalog = build_catalog(
            mcp_config_path=EXAMPLE_DIR / "mcp_servers.toml",
        )
        toml_server_names = {s.name for s in catalog.by_kind("mcp_server")}
        assert chain_server in toml_server_names, (
            f"chain.json references MCP server {chain_server!r} but "
            f"mcp_servers.toml only registers: {sorted(toml_server_names)}"
        )


# ---------------------------------------------------------------------------
# import_chains dry-run
# ---------------------------------------------------------------------------


class TestImportDryRun:
    def test_dry_run_validates_chain(self):
        report = import_chains(
            [EXAMPLE_DIR / "chain.json"],
            dry_run=True,
        )
        assert report.all_ok
        assert len(report.validated) == 1
        assert report.validated[0].path.name == "chain.json"


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


class TestReadme:
    def test_readme_mentions_every_bundled_file(self):
        readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
        for name in ("chain.json", "mcp_servers.toml"):
            assert name in readme, (
                f"`examples/weather/README.md` doesn't mention {name!r}."
            )

    def test_readme_shows_documented_cli_commands(self):
        readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
        # All three documented surfaces must be in the walkthrough.
        for cmd in (
            "care validate",
            "care catalog",
            "care import",
        ):
            assert cmd in readme, (
                f"`examples/weather/README.md` doesn't show `{cmd}` — "
                "the README walkthrough must cover every CLI surface "
                "the example exercises."
            )
