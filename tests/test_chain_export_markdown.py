"""Tests for the Markdown export format of `care.chain_export.export_chain`.

The Markdown body is a human walkthrough (`## Step N` / `### Aim`) that ends
with a fenced ``python`` block — a valid CARL build script.
"""

from __future__ import annotations

import re

from care.chain_export import ChainExportError, export_chain

CHAIN = {
    "name": "Weather + News",
    "domain": "reporting",
    "description": "Fetch weather and news, then synthesize a brief.",
    "steps": [
        {
            "number": 1, "type": "tool", "title": "Fetch weather",
            "config": {"tool_name": "web_search"}, "dependencies": [],
        },
        {
            "number": 2, "type": "llm", "title": "Synthesize report",
            "aim": "Merge the inputs into a concise brief.",
            "dependencies": [1], "llm_config": {"model": "gpt-4o-mini"},
        },
    ],
}


def _python_block(md: str) -> str:
    m = re.search(r"```python\n(.*?)```", md, re.S)
    assert m, "no fenced python block in the markdown export"
    return m.group(1)


class TestMarkdownExport:
    def test_writes_md_file(self, tmp_path):
        dest = tmp_path / "chain.md"
        result = export_chain(CHAIN, dest, format="markdown")
        assert result.format == "markdown"
        assert result.path == dest
        assert dest.read_text(encoding="utf-8")

    def test_md_extension_infers_markdown(self, tmp_path):
        dest = tmp_path / "chain.md"
        result = export_chain(CHAIN, dest)  # no explicit format
        assert result.format == "markdown"

    def test_human_walkthrough_structure(self, tmp_path):
        dest = tmp_path / "chain.md"
        export_chain(CHAIN, dest, format="markdown", title="Weather + News")
        md = dest.read_text(encoding="utf-8")
        assert md.startswith("# Weather + News")
        # Per-step headings + Aim subheading (the requested shape).
        assert "## Step 1. Fetch weather" in md
        assert "## Step 2. Synthesize report" in md
        assert "### Aim" in md
        assert "Merge the inputs into a concise brief." in md
        # Type + dependency + tool/model surfaced.
        assert "Tool" in md and "AI" in md
        assert "**Depends on:** 1" in md
        assert "web_search" in md
        assert "gpt-4o-mini" in md

    def test_ends_with_valid_python_block(self, tmp_path):
        dest = tmp_path / "chain.md"
        export_chain(CHAIN, dest, format="markdown")
        md = dest.read_text(encoding="utf-8")
        assert "## Python (CARL)" in md
        block = _python_block(md)
        # The embedded script must be valid, compilable Python.
        compile(block, "<chain-export>", "exec")
        assert "mmar_carl" in block  # proper CARL imports

    def test_fallback_python_when_codegen_unavailable(self, tmp_path, monkeypatch):
        """When `mmar_mage`'s CodeGenerator is unavailable, the python block
        falls back to a `ReasoningChain.from_dict` reconstruction — still
        valid, compilable Python with CARL imports."""
        import care.chain_export as ce

        def _boom(chain_dict, *, query, mage_config):
            raise ChainExportError("mmar_mage not installed")

        monkeypatch.setattr(ce, "_render_python", _boom)
        dest = tmp_path / "chain.md"
        export_chain(CHAIN, dest, format="markdown")
        block = _python_block(dest.read_text(encoding="utf-8"))
        compile(block, "<chain-export-fallback>", "exec")
        assert "from mmar_carl import ReasoningChain" in block
        assert "ReasoningChain.from_dict" in block

    def test_empty_steps_still_valid(self, tmp_path):
        dest = tmp_path / "empty.md"
        export_chain({"name": "Empty", "steps": []}, dest, format="markdown")
        md = dest.read_text(encoding="utf-8")
        assert "# Empty" in md
        compile(_python_block(md), "<empty>", "exec")


# Saved chains are in CARL-compat form: tool config is nested under
# ``step_config`` with CARL field names (``input_mapping`` not
# ``tool_input_mapping``). The export must lift those back to the flat MAGE
# names so the codegen doesn't emit ``tool_name='unknown'`` / ``{}``.
CARL_FORM_CHAIN = {
    "name": "Weather",
    "domain": "weather",
    "version": "1.0",
    "steps": [
        {"number": 1, "title": "Analyse", "step_type": "llm", "aim": "Understand"},
        {
            "number": 2, "title": "Fetch weather", "step_type": "tool",
            "dependencies": [1], "aim": "Get the forecast",
            "step_config": {
                "tool_name": "weather_api",
                "input_mapping": {"query": "$history[1]"},
                "output_key": "forecast",
                "timeout": 30,
            },
        },
    ],
}


class TestToolStepFlattening:
    def test_markdown_narrative_shows_real_tool_name(self, tmp_path):
        dest = tmp_path / "chain.md"
        export_chain(CARL_FORM_CHAIN, dest, format="markdown")
        md = dest.read_text(encoding="utf-8")
        assert "**Tool:** weather_api" in md
        assert "tool_name='unknown'" not in md
        assert "input_mapping={}" not in md

    def test_markdown_python_block_has_real_tool_args(self, tmp_path):
        dest = tmp_path / "chain.md"
        export_chain(CARL_FORM_CHAIN, dest, format="markdown")
        block = _python_block(dest.read_text(encoding="utf-8"))
        compile(block, "<carl-tool>", "exec")
        assert "tool_name='weather_api'" in block
        assert "'query': '$history[1]'" in block

    def test_python_export_flattens_nested_tool_config(self, tmp_path):
        dest = tmp_path / "chain.py"
        export_chain(CARL_FORM_CHAIN, dest, format="python")
        py = dest.read_text(encoding="utf-8")
        assert "tool_name='weather_api'" in py
        assert "tool_name='unknown'" not in py
        assert "input_mapping={}" not in py
