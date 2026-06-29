"""P6.5 — artifact sink (:mod:`care.runtime.artifacts`).

A chain/skill that writes files surfaces them on each step's
``output_files``; the sink copies them OUT of the throwaway sandbox into a
stable, cross-platform directory and returns the saved paths. These tests
pin: the home-based default root, the ``CARE_ARTIFACTS__DIR`` override, the
per-request ``dest`` override, sub-dir/foreign-separator flattening (so
nothing escapes the target), and no-overwrite on a name clash. All
deterministic — no live OpenRouter, no real CARL.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from care.runtime.artifacts import (
    collect_output_files,
    default_artifacts_root,
    missing_required_output,
    resolve_artifacts_root,
    save_run_artifacts,
)


def _result_with_file(src: Path, *, name: str | None = None) -> SimpleNamespace:
    """A duck-typed ``ReasoningResult``: one step whose ``result_data``
    carries an ``output_files`` entry pointing at ``src`` (CARL's shape)."""
    entry = {
        "path": str(src),
        "name": name or src.name,
        "size": str(src.stat().st_size),
    }
    step = SimpleNamespace(result_data={"output_files": [entry]})
    return SimpleNamespace(step_results=[step])


class TestArtifactsRoot:
    def test_default_root_is_under_home(self):
        root = default_artifacts_root()
        # Path.home()-based + built with pathlib (no hardcoded separators).
        assert root == Path.home() / ".care" / "artifacts"
        assert root.parent.parent == Path.home()

    def test_resolve_uses_config_dir_when_set(self, tmp_path):
        cfg = SimpleNamespace(artifacts=SimpleNamespace(dir=tmp_path / "arts"))
        assert resolve_artifacts_root(cfg) == tmp_path / "arts"

    def test_resolve_expands_user_tilde(self):
        cfg = SimpleNamespace(artifacts=SimpleNamespace(dir=Path("~/care-arts")))
        assert resolve_artifacts_root(cfg) == Path.home() / "care-arts"

    def test_resolve_falls_back_to_home_default(self):
        cfg = SimpleNamespace(artifacts=SimpleNamespace(dir=None))
        assert resolve_artifacts_root(cfg) == default_artifacts_root()
        assert resolve_artifacts_root(None) == default_artifacts_root()


class TestMissingRequiredOutput:
    def test_true_when_step_flags_no_output_file(self):
        step = SimpleNamespace(result_data={"no_output_file": True})
        result = SimpleNamespace(step_results=[step])
        assert missing_required_output(result) is True

    def test_false_when_file_was_produced(self, tmp_path):
        src = tmp_path / "deck.pptx"
        src.write_text("x")
        assert missing_required_output(_result_with_file(src)) is False

    def test_false_when_no_flag_present(self):
        result = SimpleNamespace(step_results=[SimpleNamespace(result_data={})])
        assert missing_required_output(result) is False

    def test_handles_empty_and_nondict(self):
        assert missing_required_output(SimpleNamespace(step_results=[])) is False
        assert missing_required_output(SimpleNamespace()) is False
        bad = SimpleNamespace(step_results=[SimpleNamespace(result_data=None)])
        assert missing_required_output(bad) is False


class TestCollect:
    def test_reads_raw_result_data(self, tmp_path):
        src = tmp_path / "deck.pptx"
        src.write_bytes(b"PK\x03\x04 fake pptx")
        entries = collect_output_files(_result_with_file(src))
        assert [e["name"] for e in entries] == ["deck.pptx"]

    def test_reads_typed_skill_output(self, tmp_path):
        src = tmp_path / "deck.pptx"
        src.write_bytes(b"x")
        skill_view = SimpleNamespace(
            output_files=[{"path": str(src), "name": "deck.pptx"}],
        )
        step = SimpleNamespace(as_skill_output=lambda: skill_view, result_data={})
        result = SimpleNamespace(step_results=[step])
        assert [e["name"] for e in collect_output_files(result)] == ["deck.pptx"]

    def test_dedupes_by_path(self, tmp_path):
        src = tmp_path / "f.txt"
        src.write_text("hi")
        entry = {"path": str(src), "name": "f.txt"}
        step = SimpleNamespace(result_data={"output_files": [entry, entry]})
        result = SimpleNamespace(step_results=[step])
        assert len(collect_output_files(result)) == 1


class TestSaveRunArtifacts:
    def test_saves_under_configured_root(self, tmp_path):
        src = tmp_path / "src" / "report.xlsx"
        src.parent.mkdir()
        src.write_bytes(b"xlsx-bytes")
        cfg = SimpleNamespace(artifacts=SimpleNamespace(dir=tmp_path / "store"))
        saved = save_run_artifacts(
            _result_with_file(src), care_config=cfg, slug="Build a Q3 report",
        )
        assert len(saved) == 1
        out = saved[0]
        assert out.exists()
        assert out.read_bytes() == b"xlsx-bytes"
        # landed under the configured root, in a deterministic per-run subdir.
        assert (tmp_path / "store") in out.parents
        assert out.parent.name == "run-build-a-q3-report"
        assert out.name == "report.xlsx"

    def test_per_request_dest_override_honored(self, tmp_path):
        src = tmp_path / "a.docx"
        src.write_bytes(b"docx")
        dest = tmp_path / "explicit" / "here"  # nested + missing → created
        saved = save_run_artifacts(_result_with_file(src), dest=dest)
        assert saved == [dest / "a.docx"]
        assert (dest / "a.docx").read_bytes() == b"docx"

    def test_no_files_returns_empty(self):
        result = SimpleNamespace(step_results=[SimpleNamespace(result_data={})])
        assert save_run_artifacts(result) == []

    def test_subdir_name_is_flattened_to_leaf(self, tmp_path):
        # CARL's `name` is an rglob rel path like "sub/deck.pptx"; only the
        # leaf is used, so nothing escapes the target dir (cross-platform).
        src = tmp_path / "deck.pptx"
        src.write_bytes(b"x")
        dest = tmp_path / "dst"
        saved = save_run_artifacts(
            _result_with_file(src, name="sub/deck.pptx"), dest=dest,
        )
        assert saved == [dest / "deck.pptx"]

    def test_name_clash_does_not_overwrite(self, tmp_path):
        src1 = tmp_path / "one" / "f.txt"
        src1.parent.mkdir()
        src1.write_text("one")
        src2 = tmp_path / "two" / "f.txt"
        src2.parent.mkdir()
        src2.write_text("two")
        dest = tmp_path / "dst"
        r1 = save_run_artifacts(_result_with_file(src1), dest=dest)
        r2 = save_run_artifacts(_result_with_file(src2), dest=dest)
        assert r1 == [dest / "f.txt"]
        assert r2 == [dest / "f-1.txt"]  # suffixed — both preserved
        assert (dest / "f.txt").read_text() == "one"
        assert (dest / "f-1.txt").read_text() == "two"

    def test_missing_source_is_skipped_not_raised(self, tmp_path):
        (tmp_path / "exists.txt").write_text("ok")
        result = _result_with_file(tmp_path / "exists.txt")
        # add a second entry pointing at a vanished file
        result.step_results[0].result_data["output_files"].append(
            {"path": str(tmp_path / "gone.txt"), "name": "gone.txt"},
        )
        dest = tmp_path / "dst"
        saved = save_run_artifacts(result, dest=dest)
        assert saved == [dest / "exists.txt"]  # good file kept, bad one skipped
