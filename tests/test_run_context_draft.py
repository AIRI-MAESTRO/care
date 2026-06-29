"""Tests for the run-context-modal data layer (TODO §1.3 P1).

The Textual modal is gated on §1 P0 multi-screen workflow; this
suite pins the contract the modal binds to.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from care.runtime.run_context_draft import (
    ContextFile,
    RunContextDraft,
    RunContextIssue,
    add_file,
    apply_overrides,
    attach_path,
    build_extra_kwargs,
    compute_file_stat,
    drop_file,
    extract_run_context_draft,
    missing_active_files,
    replace_file,
    resolve_file_arg,
    restore_file,
    set_model_override,
    set_task,
    validate_run_context_draft,
)


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------


_SAMPLE_SHA = "a" * 64


def _care_meta_dict(
    *,
    task: str = "Summarise the PDF",
    files: list[dict] | None = None,
    display_name: str = "PDF summariser",
) -> dict:
    return {
        "task_description": task,
        "context_files": files
        if files is not None
        else [
            {
                "path": "/tmp/example.pdf",
                "sha256": _SAMPLE_SHA,
                "size_bytes": 1024,
                "mime_type": "application/pdf",
            }
        ],
        "display_name": display_name,
        "tags": ["pdf"],
    }


class _StubChain:
    """Mimics a CARL ``ReasoningChain`` that CARE saved earlier —
    exposes ``get_care_metadata()`` returning a dict."""

    def __init__(self, *, entity_id: str = "chain-1", meta: dict | None = None):
        self.entity_id = entity_id
        self._meta = meta if meta is not None else _care_meta_dict()

    def get_care_metadata(self):
        return self._meta


# ---------------------------------------------------------------------------
# extract_run_context_draft
# ---------------------------------------------------------------------------


class TestExtractRunContextDraft:
    def test_pulls_task_and_files_from_metadata(self):
        chain = _StubChain()
        draft = extract_run_context_draft(chain)
        assert draft.source_entity_id == "chain-1"
        assert draft.source_name == "PDF summariser"
        assert draft.original_task == "Summarise the PDF"
        assert draft.task_description == "Summarise the PDF"
        assert len(draft.files) == 1
        f = draft.files[0]
        assert f.path == "/tmp/example.pdf"
        assert f.sha256 == _SAMPLE_SHA
        assert f.size_bytes == 1024
        assert f.mime_type == "application/pdf"
        assert f.status == "saved"

    def test_no_metadata_yields_empty_draft(self):
        class _Bare:
            entity_id = "c"

            def get_care_metadata(self):
                return None

        draft = extract_run_context_draft(_Bare())
        assert draft.source_entity_id == "c"
        assert draft.task_description == ""
        assert draft.files == ()

    def test_none_getter_falls_back_to_flat_metadata(self):
        """Regression: CARL's get_care_metadata() returns None when there is
        no metadata["care"] namespace — but CARE's facade stamps the care
        block FLAT under metadata. The extractor must fall back to the raw
        dict (per CARL's own docstring) instead of returning {} early; the
        C1 promotion gate's baseline run reads the task from here."""

        class _FacadeSaved:
            entity_id = "chain-flat"
            metadata = {
                "task_description": "собери новости по теме",
                "display_name": "News Agent",
                "context_files": [],
            }

            def get_care_metadata(self):
                return None  # no nested `care` namespace

        draft = extract_run_context_draft(_FacadeSaved())
        assert draft.task_description == "собери новости по теме"
        assert draft.source_name == "News Agent"

    def test_dict_chain_accepted(self):
        # Pure-dict chain (test fixture shape).
        chain = {
            "entity_id": "c-2",
            "metadata": {
                "care": _care_meta_dict(task="run twice", display_name="Loop"),
            },
        }
        draft = extract_run_context_draft(chain)
        assert draft.source_entity_id == "c-2"
        assert draft.task_description == "run twice"
        assert draft.source_name == "Loop"

    def test_source_name_param_wins_over_metadata(self):
        chain = _StubChain()
        draft = extract_run_context_draft(chain, source_name="My label")
        assert draft.source_name == "My label"

    def test_pydantic_model_dump_accepted(self):
        class _Meta:
            def model_dump(self, **_kw):
                return _care_meta_dict(task="from model")

        class _ChainWithModel:
            entity_id = "c"

            def get_care_metadata(self):
                return _Meta()

        draft = extract_run_context_draft(_ChainWithModel())
        assert draft.task_description == "from model"

    def test_raw_files_dict_with_missing_keys_tolerated(self):
        # ContextFile defaults absorb the missing fields.
        meta = _care_meta_dict(files=[{"path": "/tmp/x.txt"}])
        chain = _StubChain(meta=meta)
        draft = extract_run_context_draft(chain)
        assert draft.files[0].path == "/tmp/x.txt"
        assert draft.files[0].sha256 == ""
        assert draft.files[0].size_bytes == 0


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


class TestPredicates:
    def test_no_edits_initially(self):
        draft = extract_run_context_draft(_StubChain())
        assert draft.task_edited is False
        assert draft.has_overrides is False
        assert draft.has_file_edits is False
        assert draft.has_edits is False

    def test_edited_task(self):
        draft = extract_run_context_draft(_StubChain())
        edited = set_task(draft, "Different task")
        assert edited.task_edited is True
        assert edited.has_edits is True

    def test_edited_task_whitespace_only_does_not_count(self):
        # Strip-equality means trailing whitespace alone isn't an edit.
        draft = extract_run_context_draft(_StubChain())
        edited = set_task(draft, "Summarise the PDF  ")
        assert edited.task_edited is False

    def test_overrides_set(self):
        draft = extract_run_context_draft(_StubChain())
        with_override = set_model_override(
            draft, model="gpt-5", base_url="https://api.openai.com/v1",
        )
        assert with_override.has_overrides is True
        assert with_override.has_edits is True

    def test_drop_marks_file_edit(self):
        draft = extract_run_context_draft(_StubChain())
        edited = drop_file(draft, "/tmp/example.pdf")
        assert edited.has_file_edits is True
        # Original file kept on the tuple for undo.
        assert len(edited.files) == 1
        assert edited.files[0].status == "dropped"
        # Excluded from active list.
        assert edited.active_files == ()

    def test_active_files_skips_dropped_and_includes_added(self):
        draft = extract_run_context_draft(_StubChain())
        with_added = add_file(draft, "/tmp/extra.csv")
        dropped = drop_file(with_added, "/tmp/example.pdf")
        actives = dropped.active_files
        assert {f.path for f in actives} == {"/tmp/extra.csv"}


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


class TestMutators:
    def test_add_file_appends_with_added_status(self):
        draft = extract_run_context_draft(_StubChain())
        edited = add_file(draft, "/tmp/extra.csv", sha256="b" * 64, size_bytes=99)
        assert len(edited.files) == 2
        new = edited.files[-1]
        assert new.path == "/tmp/extra.csv"
        assert new.status == "added"
        assert new.sha256 == "b" * 64
        assert new.size_bytes == 99

    def test_drop_user_added_file_removes_outright(self):
        draft = extract_run_context_draft(_StubChain())
        with_added = add_file(draft, "/tmp/extra.csv")
        cleared = drop_file(with_added, "/tmp/extra.csv")
        assert len(cleared.files) == 1
        assert cleared.files[0].path == "/tmp/example.pdf"

    def test_restore_dropped_saved_file(self):
        draft = extract_run_context_draft(_StubChain())
        dropped = drop_file(draft, "/tmp/example.pdf")
        restored = restore_file(dropped, "/tmp/example.pdf")
        assert restored.files[0].status == "saved"
        assert restored.active_files[0].path == "/tmp/example.pdf"

    def test_restore_non_dropped_is_noop(self):
        draft = extract_run_context_draft(_StubChain())
        same = restore_file(draft, "/tmp/example.pdf")
        assert same == draft

    def test_replace_file_flips_to_replaced(self):
        draft = extract_run_context_draft(_StubChain())
        edited = replace_file(
            draft,
            "/tmp/example.pdf",
            sha256="c" * 64,
            size_bytes=2048,
            mime_type="application/pdf",
        )
        f = edited.files[0]
        assert f.status == "replaced"
        assert f.sha256 == "c" * 64
        assert f.size_bytes == 2048
        assert edited.has_file_edits is True

    def test_replace_unknown_path_is_noop(self):
        draft = extract_run_context_draft(_StubChain())
        same = replace_file(
            draft,
            "/tmp/nonexistent.pdf",
            sha256="d" * 64,
            size_bytes=10,
        )
        assert same == draft

    def test_set_model_override_clear_both(self):
        draft = extract_run_context_draft(_StubChain())
        with_set = set_model_override(draft, model="x", base_url="y")
        assert with_set.has_overrides is True
        cleared = set_model_override(with_set, model=None, base_url=None)
        assert cleared.has_overrides is False

    def test_draft_is_frozen(self):
        draft = extract_run_context_draft(_StubChain())
        with pytest.raises(FrozenInstanceError):
            draft.task_description = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidate:
    def test_clean_draft_no_issues(self, tmp_path: Path):
        # File must exist on disk for check_files=True to pass.
        file_path = tmp_path / "x.txt"
        file_path.write_text("data")
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="do it",
            files=(ContextFile(path=str(file_path), status="saved"),),
        )
        assert validate_run_context_draft(draft) == ()

    def test_empty_task_flagged(self):
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="   ",
        )
        issues = validate_run_context_draft(draft, check_files=False)
        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == "error"
        assert issue.field == "task_description"

    def test_missing_file_warning(self, tmp_path: Path):
        ghost = tmp_path / "ghost.txt"  # never created
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            files=(ContextFile(path=str(ghost), status="saved"),),
        )
        issues = validate_run_context_draft(draft)
        assert any(
            i.field == "files" and i.severity == "warning" for i in issues
        )

    def test_dropped_file_not_validated(self, tmp_path: Path):
        # Dropped files are excluded from active_files so they
        # shouldn't trigger missing-file warnings.
        ghost = tmp_path / "ghost.txt"
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            files=(ContextFile(path=str(ghost), status="dropped"),),
        )
        assert validate_run_context_draft(draft) == ()

    def test_blank_model_override_rejected(self):
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            model_override="   ",
        )
        issues = validate_run_context_draft(draft, check_files=False)
        assert any(
            i.field == "model_override" and i.severity == "error" for i in issues
        )

    def test_blank_base_url_override_rejected(self):
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            base_url_override="",
        )
        issues = validate_run_context_draft(draft, check_files=False)
        assert any(
            i.field == "base_url_override" and i.severity == "error" for i in issues
        )

    def test_issue_is_frozen(self):
        issue = RunContextIssue(severity="error", field="task_description", message="m")
        with pytest.raises(FrozenInstanceError):
            issue.severity = "warning"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def _config(self):
        from care.config import CareConfig, MageConfig

        return CareConfig(
            mage=MageConfig(api_key="k", model="m1", base_url="u1"),
        )

    def test_no_overrides_returns_same_config(self):
        cfg = self._config()
        draft = RunContextDraft(source_entity_id="c", task_description="ok")
        assert apply_overrides(cfg, draft) is cfg

    def test_model_override_applied(self):
        cfg = self._config()
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            model_override="m2",
        )
        new = apply_overrides(cfg, draft)
        assert new is not cfg
        assert new.mage.model == "m2"
        # Original untouched.
        assert cfg.mage.model == "m1"

    def test_base_url_override_applied(self):
        cfg = self._config()
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            base_url_override="u2",
        )
        new = apply_overrides(cfg, draft)
        assert new.mage.base_url == "u2"
        assert new.mage.model == "m1"  # untouched

    def test_both_overrides_applied(self):
        cfg = self._config()
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            model_override="m2",
            base_url_override="u2",
        )
        new = apply_overrides(cfg, draft)
        assert new.mage.model == "m2"
        assert new.mage.base_url == "u2"

    def test_api_key_override_applied(self):
        cfg = self._config()
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            api_key_override="sk-run",
        )
        new = apply_overrides(cfg, draft)
        assert new.mage.api_key == "sk-run"
        assert new.mage.model == "m1"  # untouched
        assert cfg.mage.api_key == "k"  # original untouched

    def test_api_key_override_triggers_has_overrides(self):
        draft = RunContextDraft(
            source_entity_id="c", task_description="ok",
            api_key_override="sk-x",
        )
        assert draft.has_overrides is True

    def test_set_model_override_carries_api_key(self):
        draft = RunContextDraft(source_entity_id="c", task_description="ok")
        out = set_model_override(draft, model="m", base_url="u", api_key="sk")
        assert out.model_override == "m"
        assert out.base_url_override == "u"
        assert out.api_key_override == "sk"

    def test_object_without_model_copy_returned_as_is(self):
        sentinel = object()
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            model_override="m",
        )
        assert apply_overrides(sentinel, draft) is sentinel


# ---------------------------------------------------------------------------
# build_extra_kwargs
# ---------------------------------------------------------------------------


class TestBuildExtraKwargs:
    def test_no_edits_returns_empty(self):
        draft = extract_run_context_draft(_StubChain())
        assert build_extra_kwargs(draft) == {}

    def test_task_edit_routed_to_outer_context(self):
        draft = extract_run_context_draft(_StubChain())
        edited = set_task(draft, "new task")
        kwargs = build_extra_kwargs(edited)
        assert kwargs["outer_context"] == "new task"

    def test_dropped_file_disables_metadata_auto_load(self, tmp_path: Path):
        # Create the saved file so the read attempt doesn't crash.
        saved = tmp_path / "saved.txt"
        saved.write_text("content")
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            files=(
                ContextFile(path=str(saved), status="saved"),
                ContextFile(path="/tmp/ghost.txt", status="dropped"),
            ),
        )
        # Force ``has_file_edits`` via the dropped row.
        kwargs = build_extra_kwargs(draft)
        assert "files" in kwargs
        assert kwargs["load_files_from_metadata"] is False
        # The dropped file is NOT in the files payload.
        assert "/tmp/ghost.txt" not in kwargs["files"]
        assert str(saved) in kwargs["files"]

    def test_added_file_in_payload(self, tmp_path: Path):
        added = tmp_path / "added.txt"
        added.write_text("hello")
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            files=(ContextFile(path=str(added), status="added"),),
        )
        kwargs = build_extra_kwargs(draft)
        assert kwargs["files"][str(added)] == "hello"
        assert kwargs["load_files_from_metadata"] is False

    def test_missing_file_collapses_to_empty_string(self):
        # The helper doesn't crash on a non-existent path — the
        # executor surfaces the missing file via step-level
        # error rather than this layer.
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            files=(ContextFile(path="/tmp/nope.txt", status="added"),),
        )
        kwargs = build_extra_kwargs(draft)
        assert kwargs["files"]["/tmp/nope.txt"] == ""


# ---------------------------------------------------------------------------
# resolve_file_arg
# ---------------------------------------------------------------------------


class TestResolveFileArg:
    def test_blank_returns_empty(self):
        assert resolve_file_arg("") == ""
        assert resolve_file_arg("   ") == ""
        assert resolve_file_arg("@") == ""

    def test_strips_leading_at(self):
        assert resolve_file_arg("@/tmp/x.txt") == "/tmp/x.txt"

    def test_strips_surrounding_quotes(self):
        assert resolve_file_arg('@"/tmp/a b.txt"') == "/tmp/a b.txt"
        assert resolve_file_arg("'/tmp/c.txt'") == "/tmp/c.txt"

    def test_relative_resolved_against_cwd(self):
        out = resolve_file_arg("notes.md")
        assert out == str(Path.cwd() / "notes.md")
        assert Path(out).is_absolute()

    def test_user_home_expanded(self):
        out = resolve_file_arg("@~/x.txt")
        assert out == str(Path.home() / "x.txt")


# ---------------------------------------------------------------------------
# compute_file_stat + attach_path
# ---------------------------------------------------------------------------


class TestComputeFileStat:
    def test_real_file(self, tmp_path: Path):
        import hashlib

        f = tmp_path / "doc.txt"
        f.write_text("hello world")
        sha, size, mime = compute_file_stat(str(f))
        assert sha == hashlib.sha256(b"hello world").hexdigest()
        assert size == len("hello world")
        assert mime == "text/plain"

    def test_missing_file_is_best_effort(self):
        sha, size, mime = compute_file_stat("/tmp/does-not-exist-xyz.txt")
        assert sha == ""
        assert size == 0
        assert mime is None


class TestAttachPath:
    def test_attaches_with_computed_stat(self, tmp_path: Path):
        f = tmp_path / "extra.csv"
        f.write_text("a,b,c")
        draft = extract_run_context_draft(_StubChain())
        out = attach_path(draft, str(f))
        assert len(out.files) == 2
        new = out.files[-1]
        assert new.path == str(f)
        assert new.status == "added"
        assert new.size_bytes == len("a,b,c")
        assert new.sha256  # computed, non-empty
        assert out.has_file_edits is True

    def test_honours_at_prefix(self, tmp_path: Path):
        f = tmp_path / "extra.csv"
        f.write_text("x")
        draft = RunContextDraft(source_entity_id="c", task_description="t")
        out = attach_path(draft, f"@{f}")
        assert out.files[-1].path == str(f)

    def test_duplicate_active_path_is_noop(self, tmp_path: Path):
        f = tmp_path / "extra.csv"
        f.write_text("x")
        draft = RunContextDraft(source_entity_id="c", task_description="t")
        once = attach_path(draft, str(f))
        twice = attach_path(once, str(f))
        assert len(twice.files) == 1

    def test_reattaching_dropped_path_restores_it(self, tmp_path: Path):
        f = tmp_path / "saved.txt"
        f.write_text("x")
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="t",
            files=(ContextFile(path=str(f), status="saved"),),
        )
        dropped = drop_file(draft, str(f))
        assert dropped.active_files == ()
        restored = attach_path(dropped, str(f))
        # Restored in place — not duplicated.
        assert len(restored.files) == 1
        assert restored.files[0].status == "saved"
        assert restored.active_files[0].path == str(f)

    def test_blank_is_noop(self):
        draft = RunContextDraft(source_entity_id="c", task_description="t")
        assert attach_path(draft, "   ") == draft


# ---------------------------------------------------------------------------
# missing_active_files
# ---------------------------------------------------------------------------


class TestMissingActiveFiles:
    def test_flags_missing_active_file(self):
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="t",
            files=(ContextFile(path="/tmp/ghost-xyz.txt", status="saved"),),
        )
        missing = missing_active_files(draft)
        assert len(missing) == 1
        assert missing[0].path == "/tmp/ghost-xyz.txt"

    def test_present_file_not_flagged(self, tmp_path: Path):
        f = tmp_path / "here.txt"
        f.write_text("x")
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="t",
            files=(ContextFile(path=str(f), status="saved"),),
        )
        assert missing_active_files(draft) == ()

    def test_dropped_missing_file_not_flagged(self):
        # A dropped row is excluded from the active set, so a missing
        # dropped file isn't surfaced — the user already excluded it.
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="t",
            files=(ContextFile(path="/tmp/ghost-xyz.txt", status="dropped"),),
        )
        assert missing_active_files(draft) == ()


# ---------------------------------------------------------------------------
# build_extra_kwargs — basename aliasing
# ---------------------------------------------------------------------------


class TestBuildExtraKwargsBasename:
    def test_files_exposed_under_basename_and_path(self, tmp_path: Path):
        f = tmp_path / "report.txt"
        f.write_text("payload")
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            files=(ContextFile(path=str(f), status="added"),),
        )
        files = build_extra_kwargs(draft)["files"]
        # Both keys present, same content — chains reference the basename
        # (${input.report.txt}) while callers may key by full path.
        assert files[str(f)] == "payload"
        assert files["report.txt"] == "payload"

    def test_added_file_wins_basename_over_saved(self, tmp_path: Path):
        old = tmp_path / "old" / "report.txt"
        old.parent.mkdir()
        old.write_text("OLD")
        new = tmp_path / "new" / "report.txt"
        new.parent.mkdir()
        new.write_text("NEW")
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            files=(
                ContextFile(path=str(old), status="saved"),
                ContextFile(path=str(new), status="added"),
            ),
        )
        files = build_extra_kwargs(draft)["files"]
        # Both full paths resolve; the attached (added) file wins the
        # shared basename slot so the replacement actually takes effect.
        assert files[str(old)] == "OLD"
        assert files[str(new)] == "NEW"
        assert files["report.txt"] == "NEW"

    def test_added_replacement_wins_over_missing_saved(self, tmp_path: Path):
        # The core fix: a saved file that's now MISSING (empty content)
        # must NOT win the basename slot over a freshly-attached
        # replacement of the same name — else the run silently uses "".
        missing = tmp_path / "report.txt"  # never created → empty content
        repl = tmp_path / "sub" / "report.txt"
        repl.parent.mkdir()
        repl.write_text("REAL")
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="ok",
            files=(
                ContextFile(path=str(missing), status="saved"),
                ContextFile(path=str(repl), status="added"),
            ),
        )
        files = build_extra_kwargs(draft)["files"]
        assert files["report.txt"] == "REAL"


# ---------------------------------------------------------------------------
# Context-file content injection + large-file handling
# ---------------------------------------------------------------------------


class TestContextFileInjection:
    def test_attached_file_injected_into_outer_context(self, tmp_path: Path):
        # A generic chain never reads $memory.input.<file>; the attached
        # content must land in outer_context so the chain actually sees it.
        f = tmp_path / "data.txt"
        f.write_text("SECRET REVENUE FIGURE 42")
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="study the file",
            files=(ContextFile(path=str(f), status="added"),),
        )
        oc = build_extra_kwargs(draft)["outer_context"]
        assert "study the file" in oc
        assert "SECRET REVENUE FIGURE 42" in oc
        assert "data.txt" in oc

    def test_large_file_truncated(self, tmp_path: Path):
        from care.runtime.run_context_draft import MAX_CONTEXT_FILE_CHARS

        big = tmp_path / "big.txt"
        big.write_text("A" * (MAX_CONTEXT_FILE_CHARS + 5_000))
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="t",
            files=(ContextFile(path=str(big), status="added"),),
        )
        kwargs = build_extra_kwargs(draft)
        assert "truncated" in kwargs["outer_context"]
        # The files= payload is capped too.
        assert len(kwargs["files"][str(big)]) == MAX_CONTEXT_FILE_CHARS

    def test_image_injected_as_image_block(self, tmp_path: Path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="describe the screenshot",
            files=(ContextFile(path=str(img), status="added"),),
        )
        kwargs = build_extra_kwargs(draft)
        oc = kwargs["outer_context"]
        assert '<image path="shot.png"' in oc
        assert "data:image/png;base64," in oc
        # memory payload carries the data URI (not empty text)
        assert kwargs["files"]["shot.png"].startswith("data:image/png;base64,")

    def test_read_context_file_caps(self, tmp_path: Path):
        from care.runtime.run_context_draft import _read_context_file

        f = tmp_path / "x.txt"
        f.write_text("hello world")
        assert _read_context_file(str(f), max_chars=5) == ("hello", True)
        assert _read_context_file(str(f), max_chars=100) == ("hello world", False)

    def test_read_context_file_docx(self, tmp_path: Path):
        from docx import Document

        from care.runtime.run_context_draft import _read_context_file

        f = tmp_path / "r.docx"
        d = Document()
        d.add_paragraph("Hello from DOCX")
        d.save(str(f))
        text, _ = _read_context_file(str(f))
        assert "Hello from DOCX" in text

    def test_read_context_file_binary_safe(self, tmp_path: Path):
        from care.runtime.run_context_draft import _read_context_file

        f = tmp_path / "x.bin"
        f.write_bytes(b"\x00\x01\x02\xff\xfe\x80")
        text, _ = _read_context_file(str(f))  # must not raise
        assert isinstance(text, str)

    def test_read_context_file_missing(self):
        from care.runtime.run_context_draft import _read_context_file

        assert _read_context_file("/no/such/file-xyz.txt") == ("", False)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            ContextFile as Cf,
            RunContextDraft as Draft,
            RunContextIssue as Issue,
            apply_overrides as ao,
            build_extra_kwargs as bek,
            extract_run_context_draft as extract,
            validate_run_context_draft as val,
        )

        assert Cf is ContextFile
        assert Draft is RunContextDraft
        assert Issue is RunContextIssue
        assert ao is apply_overrides
        assert bek is build_extra_kwargs
        assert extract is extract_run_context_draft
        assert val is validate_run_context_draft
