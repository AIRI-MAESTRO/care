"""Tests for ``care.chain_export.export_chain`` (TODO §9 P3).

Coverage layers:

1. **Format resolution** — explicit kwarg wins; extension-based
   inference for ``.json`` / ``.py``; explicit unknown format
   raises; missing-extension + no kwarg raises.
2. **JSON output** (always available) — dict input writes
   sorted-keys pretty-printed JSON; JSON-string input is
   accepted + re-emitted; object with ``.to_dict()`` (a
   ReasoningChain shape) is unwrapped; invalid input raises
   :class:`ChainExportError`.
3. **Python output** dispatches on whether MAGE is installed:
   - With the ``mage`` extra → real ``CodeGenerator`` output is
     written; the script starts with the auto-generated header.
   - Without the ``mage`` extra → friendly install-hint error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from care.chain_export import (
    ChainExportError,
    ExportResult,
    export_chain,
)


def _mage_installed() -> bool:
    try:
        import mmar_mage  # noqa: F401
    except ImportError:
        return False
    return True


def _valid_chain_dict() -> dict:
    return {
        "task_description": "demo",
        "steps": [
            {"number": 1, "title": "first", "step_type": "llm", "aim": "hi"},
        ],
    }


# ---------------------------------------------------------------------------
# Format resolution
# ---------------------------------------------------------------------------


class TestFormatResolution:
    def test_extension_json(self, tmp_path: Path):
        result = export_chain(_valid_chain_dict(), tmp_path / "x.json")
        assert result.format == "json"

    def test_extension_python_with_mage(self, tmp_path: Path):
        if not _mage_installed():
            pytest.skip("mmar_mage not installed")
        result = export_chain(_valid_chain_dict(), tmp_path / "x.py")
        assert result.format == "python"

    def test_explicit_format_wins_over_extension(self, tmp_path: Path):
        # File named .py but format='json' → JSON output.
        result = export_chain(
            _valid_chain_dict(), tmp_path / "weird.py", format="json"
        )
        assert result.format == "json"
        body = (tmp_path / "weird.py").read_text(encoding="utf-8")
        # JSON body, not Python.
        assert body.startswith("{")

    def test_unknown_format_raises(self, tmp_path: Path):
        with pytest.raises(ChainExportError, match="unknown export format"):
            export_chain(
                _valid_chain_dict(),
                tmp_path / "x.json",
                format="yaml",  # type: ignore[arg-type]
            )

    def test_missing_extension_no_kwarg_raises(self, tmp_path: Path):
        with pytest.raises(ChainExportError, match="cannot infer format"):
            export_chain(_valid_chain_dict(), tmp_path / "noext")

    def test_uppercase_extension_recognised(self, tmp_path: Path):
        # `.JSON` works the same as `.json`.
        result = export_chain(_valid_chain_dict(), tmp_path / "X.JSON")
        assert result.format == "json"


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_dict_input_writes_pretty_json(self, tmp_path: Path):
        dest = tmp_path / "x.json"
        result = export_chain(_valid_chain_dict(), dest)
        assert isinstance(result, ExportResult)
        assert result.path == dest
        assert result.format == "json"
        body = dest.read_text(encoding="utf-8")
        # Pretty-printed + sorted keys.
        assert "\n" in body
        parsed = json.loads(body)
        assert parsed["task_description"] == "demo"
        # Sorted keys: `steps` precedes `task_description` alphabetically.
        assert body.index('"steps"') < body.index('"task_description"')
        assert result.bytes_written == len(body)

    def test_json_string_input(self, tmp_path: Path):
        raw = json.dumps(_valid_chain_dict())
        dest = tmp_path / "from-str.json"
        export_chain(raw, dest)
        # Round-trips through json.loads.
        parsed = json.loads(dest.read_text(encoding="utf-8"))
        assert parsed["task_description"] == "demo"

    def test_bytes_input(self, tmp_path: Path):
        raw = json.dumps(_valid_chain_dict()).encode("utf-8")
        dest = tmp_path / "bytes.json"
        export_chain(raw, dest)
        assert json.loads(dest.read_text(encoding="utf-8"))["task_description"] == "demo"

    def test_object_with_to_dict(self, tmp_path: Path):
        class _ChainLike:
            def to_dict(self):
                return _valid_chain_dict()

        dest = tmp_path / "obj.json"
        export_chain(_ChainLike(), dest)
        assert json.loads(dest.read_text(encoding="utf-8"))["task_description"] == "demo"

    def test_invalid_input_type_raises(self, tmp_path: Path):
        with pytest.raises(ChainExportError, match="expected dict"):
            export_chain(42, tmp_path / "x.json")

    def test_malformed_json_string_raises(self, tmp_path: Path):
        with pytest.raises(ChainExportError, match="failed to parse"):
            export_chain("not json", tmp_path / "x.json")

    def test_json_string_not_dict_raises(self, tmp_path: Path):
        with pytest.raises(ChainExportError, match="must decode to a dict"):
            export_chain("[1, 2, 3]", tmp_path / "x.json")

    def test_to_dict_returning_non_dict_raises(self, tmp_path: Path):
        class _BadChain:
            def to_dict(self):
                return [1, 2, 3]

        with pytest.raises(ChainExportError, match="must return a dict"):
            export_chain(_BadChain(), tmp_path / "x.json")

    def test_invalid_utf8_bytes_raise(self, tmp_path: Path):
        with pytest.raises(ChainExportError, match="not valid utf-8"):
            export_chain(b"\xff\xfe\xfd", tmp_path / "x.json")

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        result = export_chain(_valid_chain_dict(), "~/exported.json")
        assert result.path == tmp_path / "exported.json"
        assert result.path.exists()


# ---------------------------------------------------------------------------
# Python output (dispatch on installed MAGE)
# ---------------------------------------------------------------------------


class TestPythonOutputWithMage:
    pytestmark = pytest.mark.skipif(
        not _mage_installed(),
        reason="mmar_mage not installed; skipping success path",
    )

    def test_python_output_runs_real_code_generator(self, tmp_path: Path):
        dest = tmp_path / "script.py"
        result = export_chain(
            _valid_chain_dict(), dest, query="forecast the weather"
        )
        assert result.format == "python"
        body = dest.read_text(encoding="utf-8")
        # MAGE's CodeGenerator header.
        assert body.startswith('"""Auto-generated CARL chain script."""')
        # Query lands in the build_chain docstring.
        assert "forecast the weather" in body
        # ChainBuilder fluent API references.
        assert "from mmar_carl import ChainBuilder" in body

    def test_query_is_optional(self, tmp_path: Path):
        # Default empty query — script still renders.
        dest = tmp_path / "no-query.py"
        export_chain(_valid_chain_dict(), dest)
        body = dest.read_text(encoding="utf-8")
        assert "Auto-generated CARL chain script" in body


class TestPythonOutputWithoutMage:
    pytestmark = pytest.mark.skipif(
        _mage_installed(),
        reason="mmar_mage IS installed; skipping missing-dep path",
    )

    def test_python_output_raises_friendly_error(self, tmp_path: Path):
        with pytest.raises(ChainExportError, match="mmar_mage is not installed"):
            export_chain(_valid_chain_dict(), tmp_path / "x.py")
