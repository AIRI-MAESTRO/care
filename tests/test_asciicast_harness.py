"""Tests for ``examples/asciicast/`` + ``scripts/record_demo.sh``
(TODO §10 P3).

The asciicast is a manual artefact — the `.cast` file lands
the day someone runs the recording wrapper. The bounded
data-layer ship is the **reproducible harness**: seed
fixtures + recording script + wrapper. These tests pin every
piece + verify the expected CLI output the script promises
still matches reality, so a future refactor that breaks the
recording fails CI immediately instead of producing a stale
demo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from care.bulk_import import import_chains
from care.catalog import build_catalog
from care.preflight import validate_chain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASCIICAST_DIR = PROJECT_ROOT / "examples" / "asciicast"
SEED_DIR = ASCIICAST_DIR / "seed"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "record_demo.sh"


# ---------------------------------------------------------------------------
# Harness files exist
# ---------------------------------------------------------------------------


class TestHarnessFiles:
    def test_asciicast_dir_exists(self):
        assert ASCIICAST_DIR.is_dir()

    def test_recording_script_exists(self):
        assert (ASCIICAST_DIR / "recording_script.md").is_file()

    def test_seed_files_present(self):
        for rel in (
            "chain.json",
            "mcp_servers.toml",
            "skills/pdf-helper/SKILL.md",
            "tools/demo_tool.py",
        ):
            assert (SEED_DIR / rel).is_file(), f"missing seed/{rel}"

    def test_wrapper_script_exists(self):
        assert SCRIPT_PATH.is_file()

    def test_wrapper_script_executable(self):
        # Without the executable bit the user has to `bash
        # scripts/record_demo.sh` — the README documents the
        # direct invocation, so this guards against an
        # accidental `chmod -x`.
        st = os.stat(SCRIPT_PATH)
        assert st.st_mode & 0o111, "scripts/record_demo.sh is not executable"


# ---------------------------------------------------------------------------
# Seed chain integrity
# ---------------------------------------------------------------------------


class TestSeedChain:
    def test_chain_json_parses(self):
        result = validate_chain(
            (SEED_DIR / "chain.json").read_text(encoding="utf-8")
        )
        assert result.is_valid, (
            "asciicast seed chain failed to parse: "
            + " | ".join(result.parse_errors)
        )

    def test_chain_steps_match_script(self):
        data = json.loads((SEED_DIR / "chain.json").read_text(encoding="utf-8"))
        # Recording script narrates "MCP fetch → LLM summarise".
        kinds = [s["step_type"] for s in data["steps"]]
        assert kinds == ["mcp", "llm"]

    def test_chain_references_weather_server(self):
        data = json.loads((SEED_DIR / "chain.json").read_text(encoding="utf-8"))
        server_name = data["steps"][0]["step_config"]["server"]["server_name"]
        assert server_name == "weather"


# ---------------------------------------------------------------------------
# Catalog discovery against the seed
# ---------------------------------------------------------------------------


class TestCatalogAgainstSeed:
    def test_all_three_kinds_discovered(self):
        catalog = build_catalog(
            skills_paths=[SEED_DIR / "skills"],
            mcp_config_path=SEED_DIR / "mcp_servers.toml",
            tools_path=SEED_DIR / "tools",
        )
        kinds = {e.kind for e in catalog.entries}
        # The recording script promises one of each.
        assert kinds == {"agent_skill", "mcp_server", "tool"}

    def test_pdf_helper_skill_discovered(self):
        catalog = build_catalog(skills_paths=[SEED_DIR / "skills"])
        skills = catalog.by_kind("agent_skill")
        assert [s.name for s in skills] == ["pdf-helper"]

    def test_weather_mcp_server_discovered(self):
        catalog = build_catalog(mcp_config_path=SEED_DIR / "mcp_servers.toml")
        servers = catalog.by_kind("mcp_server")
        assert [s.name for s in servers] == ["weather"]

    def test_demo_tool_discovered(self):
        catalog = build_catalog(tools_path=SEED_DIR / "tools")
        tools = catalog.by_kind("tool")
        assert [t.name for t in tools] == ["demo_tool"]


# ---------------------------------------------------------------------------
# Validate / import CLI outputs match the script
# ---------------------------------------------------------------------------


class TestExpectedCliOutputs:
    def test_validate_output_matches_script(self):
        """The recording script promises this exact line. If
        `validate_chain.format_text()` changes, the recording
        becomes stale."""
        raw = (SEED_DIR / "chain.json").read_text(encoding="utf-8")
        result = validate_chain(raw)
        assert result.is_valid
        text = result.format_text()
        # Pinned in `recording_script.md` Act 2. The verdict line depends on
        # the installed CARL: base 0.2.0 SKIPS preflight ("chain parsed;
        # …skipped…"); the agent-features build (this branch's rule) actually
        # RUNS it ("preflight: ok …"). Accept either so the pin survives both.
        assert ("preflight: ok" in text) or ("preflight skipped" in text)

    def test_import_dry_run_matches_script(self):
        """Recording script promises the dry-run line."""
        report = import_chains([SEED_DIR / "chain.json"], dry_run=True)
        text = report.format_text()
        # Pinned in `recording_script.md` Act 3.
        assert "0 imported, 1 validated, 0 failed" in text


# ---------------------------------------------------------------------------
# Recording script content
# ---------------------------------------------------------------------------


class TestRecordingScriptText:
    def test_mentions_every_seed_file(self):
        script = (ASCIICAST_DIR / "recording_script.md").read_text(
            encoding="utf-8"
        )
        # Every seed path the wrapper depends on should appear
        # somewhere in the script so a reader knows what to
        # touch.
        for rel in (
            "examples/asciicast/seed/chain.json",
            "examples/asciicast/seed/mcp_servers.toml",
            "examples/asciicast/seed/skills",
            "examples/asciicast/seed/tools",
        ):
            assert rel in script, f"recording script doesn't mention {rel}"

    def test_documents_each_cli_surface(self):
        script = (ASCIICAST_DIR / "recording_script.md").read_text(
            encoding="utf-8"
        )
        for surface in ("care catalog", "care validate", "care import"):
            assert surface in script

    def test_wrapper_invocation_mentioned(self):
        script = (ASCIICAST_DIR / "recording_script.md").read_text(
            encoding="utf-8"
        )
        assert "scripts/record_demo.sh" in script


# ---------------------------------------------------------------------------
# Wrapper script content
# ---------------------------------------------------------------------------


class TestWrapperScript:
    def test_uses_asciinema(self):
        body = SCRIPT_PATH.read_text(encoding="utf-8")
        # Wrapper has to call `asciinema rec` for the recording
        # to actually capture anything.
        assert "asciinema rec" in body

    def test_default_output_path(self):
        body = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "docs/asciicasts/care-tour.cast" in body

    def test_sets_demo_friendly_env(self):
        body = SCRIPT_PATH.read_text(encoding="utf-8")
        # Sets `CARE_MAGE__API_KEY` so first-run probes don't
        # short-circuit on missing creds during the recording.
        assert "CARE_MAGE__API_KEY" in body

    def test_refuses_to_overwrite_silently(self):
        body = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "Overwrite?" in body

    def test_validates_prerequisites(self):
        body = SCRIPT_PATH.read_text(encoding="utf-8")
        # Friendly error messages for missing tools.
        assert "asciinema not on PATH" in body
        assert "uv not on PATH" in body


# ---------------------------------------------------------------------------
# README link
# ---------------------------------------------------------------------------


class TestReadmeLink:
    def test_readme_mentions_demo_script(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "scripts/record_demo.sh" in readme
        assert "recording_script.md" in readme
