"""Structural pin for ``examples/financier/`` (TODO §10 P2).

Same shape as ``tests/test_example_weather.py`` — pin the bundled
example against every primitive its README claims to use, so
drift in either direction (example update breaks parse / a
primitive change breaks the example) fails CI immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

from care.bulk_import import import_chains
from care.catalog import build_catalog
from care.preflight import validate_chain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = PROJECT_ROOT / "examples" / "financier"


# ---------------------------------------------------------------------------
# Files exist
# ---------------------------------------------------------------------------


class TestFilesPresent:
    def test_directory_exists(self):
        assert EXAMPLE_DIR.is_dir(), (
            f"{EXAMPLE_DIR} is missing — the financier example must "
            "live in `examples/financier/`."
        )

    def test_required_files_present(self):
        for rel in (
            "chain.json",
            "README.md",
            "skills/pdf-extractor/SKILL.md",
        ):
            assert (EXAMPLE_DIR / rel).is_file(), (
                f"`examples/financier/{rel}` is missing."
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
            "examples/financier/chain.json failed to parse: "
            + " | ".join(result.parse_errors)
        )

    def test_steps_have_expected_kinds(self):
        data = json.loads((EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8"))
        kinds = [s["step_type"] for s in data["steps"]]
        # `tool` is a placeholder for the future `agent_skill` step
        # type — see the README note. `structured_output` is real.
        assert kinds == ["tool", "structured_output"]

    def test_tool_step_references_pdf_extractor(self):
        data = json.loads((EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8"))
        tool_step = data["steps"][0]
        assert tool_step["step_config"]["tool_name"] == "pdf_extractor"

    def test_structured_output_schema_required_fields(self):
        data = json.loads((EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8"))
        so_step = data["steps"][1]
        schema = so_step["step_config"]["output_schema"]
        # The README's QuarterlyFinancials shape — pin so the
        # documentation stays accurate.
        for field in (
            "period",
            "currency",
            "revenue",
            "net_income",
            "total_expenses",
        ):
            assert field in schema["properties"], (
                f"output_schema missing `{field}` — README documents it."
            )
            assert field in schema["required"], (
                f"`{field}` must be required per README contract."
            )
        # `notable_alerts` is optional per the README — exists but
        # not in `required`.
        assert "notable_alerts" in schema["properties"]
        assert "notable_alerts" not in schema["required"]

    def test_structured_output_step_depends_on_extract(self):
        data = json.loads((EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8"))
        so_step = data["steps"][1]
        assert so_step["dependencies"] == [1]


# ---------------------------------------------------------------------------
# SKILL.md catalog discovery
# ---------------------------------------------------------------------------


class TestSkillCatalog:
    def test_catalog_discovers_pdf_extractor(self):
        catalog = build_catalog(
            skills_paths=[EXAMPLE_DIR / "skills"],
        )
        skills = catalog.by_kind("agent_skill")
        names = [s.name for s in skills]
        assert names == ["pdf-extractor"]
        assert catalog.errors == ()

    def test_skill_carries_documented_tags_and_tools(self):
        catalog = build_catalog(
            skills_paths=[EXAMPLE_DIR / "skills"],
        )
        skill = catalog.by_kind("agent_skill")[0]
        assert "pdf" in skill.tags
        assert "finance" in skill.tags
        # `allowed-tools` lands on metadata.
        assert skill.metadata.get("allowed_tools") == ["Read", "Bash"]


# ---------------------------------------------------------------------------
# Cross-file consistency
# ---------------------------------------------------------------------------


class TestCrossFileConsistency:
    def test_tool_name_referenced_in_readme(self):
        """The chain's `tool_name` must appear in the README so a
        user reading the docs can match it to the chain spec."""
        chain = json.loads(
            (EXAMPLE_DIR / "chain.json").read_text(encoding="utf-8")
        )
        tool_name = chain["steps"][0]["step_config"]["tool_name"]
        readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
        assert tool_name in readme, (
            f"README doesn't reference tool_name={tool_name!r} — "
            "the chain ↔ README pairing must stay in lock-step."
        )

    def test_skill_name_referenced_in_readme(self):
        """The SKILL.md's `name` (`pdf-extractor`) should appear in
        the README, since the README describes the AgentSkill the
        chain consumes."""
        readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
        assert "pdf-extractor" in readme


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


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


class TestReadme:
    def test_readme_shows_all_documented_cli_commands(self):
        readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
        for cmd in (
            "care validate",
            "care catalog",
            "care import",
        ):
            assert cmd in readme

    def test_readme_describes_memory_card_persistence(self):
        readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
        # The unique aspect of this example (vs weather) is the
        # memory_card persistence at the end of the run.
        assert "memory_card" in readme
        assert "save_memory_card" in readme

    def test_readme_describes_structured_output(self):
        readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
        assert "structured_output" in readme
        assert "QuarterlyFinancials" in readme
