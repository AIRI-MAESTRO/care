"""Tests for ``care.bulk_import.import_chains`` (TODO §3 P2).

Each test crafts files on disk + a `_StubMemory` that records
calls, then asserts the per-file outcome on
:class:`BulkImportReport`. Validation flows through the same
real CARL parse the rest of the suite uses (no chain-side mocks)
so the import respects the actual `from_dict` shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from care.bulk_import import (
    BulkImportEntry,
    BulkImportReport,
    import_chains,
)


class _StubMemory:
    """Records every ``save_chain`` call so tests can assert on it."""

    def __init__(self, returned_id: str = "ent-saved", raise_exc: Exception | None = None):
        self.calls: list[dict[str, Any]] = []
        self.returned_id = returned_id
        self._raise = raise_exc

    def save_chain(self, chain: Any, **kwargs: Any) -> str:
        self.calls.append({"chain": chain, **kwargs})
        if self._raise:
            raise self._raise
        return self.returned_id


def _valid_chain_dict() -> dict[str, Any]:
    """Minimal typed-step chain dict the installed CARL parses cleanly."""
    return {
        "task_description": "demo",
        "steps": [
            {
                "number": 1,
                "title": "first",
                "step_type": "llm",
                "aim": "say hi",
            },
        ],
    }


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


class TestReportShape:
    def test_empty_report_is_all_ok(self):
        r = BulkImportReport()
        assert r.entries == ()
        assert r.all_ok
        assert r.imported == ()
        assert r.failed == ()

    def test_partitioning(self):
        e1 = BulkImportEntry(path=Path("a"), status="imported", entity_id="e1")
        e2 = BulkImportEntry(path=Path("b"), status="validated")
        e3 = BulkImportEntry(path=Path("c"), status="failed", errors=("nope",))
        r = BulkImportReport(entries=(e1, e2, e3))
        assert r.imported == (e1,)
        assert r.validated == (e2,)
        assert r.failed == (e3,)
        assert not r.all_ok

    def test_format_text_summary(self):
        r = BulkImportReport(
            entries=(
                BulkImportEntry(path=Path("a"), status="imported"),
                BulkImportEntry(path=Path("b"), status="failed", errors=("boom",)),
            ),
        )
        text = r.format_text()
        assert "1 imported" in text
        assert "1 failed" in text
        assert "FAIL b" in text
        assert "boom" in text


# ---------------------------------------------------------------------------
# Bare-chain form
# ---------------------------------------------------------------------------


class TestBareChainForm:
    def test_imports_a_single_file(self, tmp_path: Path):
        _write(tmp_path / "weather.json", _valid_chain_dict())
        memory = _StubMemory(returned_id="ent-1")
        report = import_chains([tmp_path / "weather.json"], memory)
        assert report.all_ok
        assert len(report.imported) == 1
        e = report.imported[0]
        assert e.entity_id == "ent-1"
        assert e.name == "weather"  # file stem
        assert memory.calls[0]["name"] == "weather"

    def test_filename_stem_becomes_default_name(self, tmp_path: Path):
        _write(tmp_path / "forecaster.json", _valid_chain_dict())
        memory = _StubMemory()
        import_chains([tmp_path / "forecaster.json"], memory)
        assert memory.calls[0]["name"] == "forecaster"


# ---------------------------------------------------------------------------
# Wrapper form
# ---------------------------------------------------------------------------


class TestWrapperForm:
    def test_forwards_save_kwargs(self, tmp_path: Path):
        wrapped = {
            "chain": _valid_chain_dict(),
            "name": "weather-agent",
            "query": "what's the weather",
            "domain": "weather",
            "tags": ["external"],
            "when_to_use": "for weather questions",
            "author": "carl",
            "channel": "stable",
        }
        _write(tmp_path / "x.json", wrapped)
        memory = _StubMemory()
        import_chains([tmp_path / "x.json"], memory)
        call = memory.calls[0]
        assert call["name"] == "weather-agent"
        assert call["query"] == "what's the weather"
        assert call["domain"] == "weather"
        assert call["tags"] == ["external"]
        assert call["when_to_use"] == "for weather questions"
        assert call["author"] == "carl"
        assert call["channel"] == "stable"

    def test_extra_unknown_keys_are_ignored(self, tmp_path: Path):
        wrapped = {
            "chain": _valid_chain_dict(),
            "name": "z",
            "irrelevant_extension_field": {"foo": "bar"},
        }
        _write(tmp_path / "z.json", wrapped)
        memory = _StubMemory()
        report = import_chains([tmp_path / "z.json"], memory)
        assert report.all_ok
        # Unknown key isn't passed through.
        assert "irrelevant_extension_field" not in memory.calls[0]

    def test_default_channel_when_wrapper_omits_it(self, tmp_path: Path):
        wrapped = {"chain": _valid_chain_dict(), "name": "n"}
        _write(tmp_path / "n.json", wrapped)
        memory = _StubMemory()
        import_chains(
            [tmp_path / "n.json"], memory, channel="experimental"
        )
        assert memory.calls[0]["channel"] == "experimental"


# ---------------------------------------------------------------------------
# Glob expansion
# ---------------------------------------------------------------------------


class TestGlobExpansion:
    def test_glob_matches_multiple_files(self, tmp_path: Path):
        _write(tmp_path / "a.json", _valid_chain_dict())
        _write(tmp_path / "b.json", _valid_chain_dict())
        _write(tmp_path / "c.txt", _valid_chain_dict())  # filtered out by glob
        memory = _StubMemory()
        report = import_chains([str(tmp_path / "*.json")], memory)
        assert len(report.imported) == 2
        names = sorted(e.name for e in report.imported)
        assert names == ["a", "b"]

    def test_recursive_glob(self, tmp_path: Path):
        nested = tmp_path / "x" / "y"
        nested.mkdir(parents=True)
        _write(tmp_path / "top.json", _valid_chain_dict())
        _write(nested / "deep.json", _valid_chain_dict())
        memory = _StubMemory()
        report = import_chains([str(tmp_path / "**" / "*.json")], memory)
        names = sorted(e.name for e in report.imported)
        assert names == ["deep", "top"]

    def test_duplicate_patterns_deduplicate(self, tmp_path: Path):
        f = _write(tmp_path / "once.json", _valid_chain_dict())
        memory = _StubMemory()
        report = import_chains([f, str(f), str(tmp_path / "*.json")], memory)
        # Three patterns all matching the same file → one import.
        assert len(report.imported) == 1

    def test_empty_match_yields_empty_report(self, tmp_path: Path):
        memory = _StubMemory()
        report = import_chains([str(tmp_path / "no-match-*.json")], memory)
        assert report.entries == ()
        assert report.all_ok

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        _write(tmp_path / "h.json", _valid_chain_dict())
        memory = _StubMemory()
        report = import_chains(["~/h.json"], memory)
        assert len(report.imported) == 1


# ---------------------------------------------------------------------------
# Per-file failure modes
# ---------------------------------------------------------------------------


class TestPerFileFailures:
    def test_invalid_json_recorded_as_failure(self, tmp_path: Path):
        path = tmp_path / "broken.json"
        path.write_text("{ invalid", encoding="utf-8")
        memory = _StubMemory()
        report = import_chains([path], memory)
        assert len(report.failed) == 1
        assert "invalid JSON" in report.failed[0].errors[0]

    def test_chain_validation_failure_recorded(self, tmp_path: Path):
        # Wrapper form with empty steps — CARL parse rejects this.
        broken = {"chain": {"task_description": "d", "steps": [{}]}}
        _write(tmp_path / "missing.json", broken)
        memory = _StubMemory()
        report = import_chains([tmp_path / "missing.json"], memory)
        assert len(report.failed) == 1
        # Memory was never touched because parse failed.
        assert memory.calls == []

    def test_top_level_not_a_dict(self, tmp_path: Path):
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        memory = _StubMemory()
        report = import_chains([path], memory)
        assert len(report.failed) == 1
        assert "top-level JSON" in report.failed[0].errors[0]

    def test_missing_chain_and_steps_keys(self, tmp_path: Path):
        path = tmp_path / "x.json"
        path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        memory = _StubMemory()
        report = import_chains([path], memory)
        assert len(report.failed) == 1
        assert "missing both" in report.failed[0].errors[0]

    def test_save_chain_exception_recorded(self, tmp_path: Path):
        _write(tmp_path / "save-boom.json", _valid_chain_dict())
        memory = _StubMemory(raise_exc=RuntimeError("memory down"))
        report = import_chains([tmp_path / "save-boom.json"], memory)
        assert len(report.failed) == 1
        assert "memory down" in report.failed[0].errors[0]

    def test_continues_after_single_failure(self, tmp_path: Path):
        # Two files: one broken JSON, one good chain. The good one
        # must still import.
        broken = tmp_path / "broken.json"
        broken.write_text("{ invalid", encoding="utf-8")
        _write(tmp_path / "good.json", _valid_chain_dict())
        memory = _StubMemory(returned_id="ent-good")
        report = import_chains([broken, tmp_path / "good.json"], memory)
        assert len(report.imported) == 1
        assert len(report.failed) == 1
        assert report.imported[0].entity_id == "ent-good"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_skips_save(self, tmp_path: Path):
        _write(tmp_path / "x.json", _valid_chain_dict())
        memory = _StubMemory()
        report = import_chains([tmp_path / "x.json"], memory, dry_run=True)
        assert len(report.validated) == 1
        assert report.imported == ()
        assert memory.calls == []

    def test_dry_run_without_memory(self, tmp_path: Path):
        _write(tmp_path / "x.json", _valid_chain_dict())
        report = import_chains([tmp_path / "x.json"], dry_run=True)
        assert len(report.validated) == 1

    def test_dry_run_still_reports_failures(self, tmp_path: Path):
        # Validation still runs in dry-run mode.
        path = tmp_path / "broken.json"
        path.write_text("{ invalid", encoding="utf-8")
        report = import_chains([path], dry_run=True)
        assert len(report.failed) == 1

    def test_no_memory_outside_dry_run_raises(self, tmp_path: Path):
        _write(tmp_path / "x.json", _valid_chain_dict())
        with pytest.raises(ValueError, match="memory is required"):
            import_chains([tmp_path / "x.json"])
