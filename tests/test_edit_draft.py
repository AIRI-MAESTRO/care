"""Tests for the edit-agent-screen data layer (TODO §3 P1).

EditAgentScreen is gated on §1 P0; this suite pins the contract
the screen will rely on.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError

import pytest

from care.runtime.edit_draft import (
    EditAgentDraft,
    EditDraftError,
    EditDraftIssue,
    PromoteResult,
    SaveEditResult,
    extract_edit_draft,
    promote_to_stable,
    reset,
    save_edit_as_new_version,
    set_change_summary,
    set_description,
    set_display_name,
    set_tags,
    set_task_description,
    update_chain,
    validate_edit_draft,
)


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------


def _care_meta(
    *,
    display_name: str = "Weather report",
    description: str = "Hourly weather agent",
    tags: list[str] | None = None,
    task: str = "Build a weather summary",
) -> dict:
    return {
        "display_name": display_name,
        "description": description,
        "tags": tags if tags is not None else ["domain:weather"],
        "task_description": task,
    }


class _StubChain:
    """Mimics a CARL ReasoningChain exposing `get_care_metadata`."""

    def __init__(self, *, meta: dict | None = None):
        self._meta = meta if meta is not None else _care_meta()

    def get_care_metadata(self):
        return self._meta


class _StubMemoryWithSave:
    """Mimics a CareMemory facade. Records every `save_chain` /
    promote call so tests can assert the SDK contract."""

    def __init__(
        self,
        *,
        save_exc: Exception | None = None,
        save_delay: float = 0.0,
        promote_exc: Exception | None = None,
        promote_delay: float = 0.0,
        promote_response: dict | None = None,
    ):
        self.save_calls: list[dict] = []
        self.promote_calls: list[dict] = []
        self._save_exc = save_exc
        self._save_delay = save_delay
        self._promote_exc = promote_exc
        self._promote_delay = promote_delay
        self._promote_response = promote_response or {"ok": True}

        memory = self
        class _Client:
            def promote(self_inner, entity_id, from_channel="latest",
                        to_channel="stable", entity_type="chain"):
                memory.promote_calls.append(
                    {
                        "entity_id": entity_id,
                        "from_channel": from_channel,
                        "to_channel": to_channel,
                        "entity_type": entity_type,
                    }
                )
                if memory._promote_delay:
                    time.sleep(memory._promote_delay)
                if memory._promote_exc:
                    raise memory._promote_exc
                return memory._promote_response

        self.client = _Client()

    def save_chain(self, chain, *, name, query=None, tags=None,
                   author=None, entity_id=None, channel="latest", **_):
        self.save_calls.append(
            {
                "chain": chain,
                "name": name,
                "query": query,
                "tags": list(tags) if tags is not None else None,
                "author": author,
                "entity_id": entity_id,
                "channel": channel,
            }
        )
        if self._save_delay:
            time.sleep(self._save_delay)
        if self._save_exc:
            raise self._save_exc
        return entity_id or "new-id"


# ---------------------------------------------------------------------------
# extract_edit_draft
# ---------------------------------------------------------------------------


class TestExtractEditDraft:
    def test_extracts_every_editable_field(self):
        chain = _StubChain()
        draft = extract_edit_draft(chain, "ent-1")
        assert draft.entity_id == "ent-1"
        assert draft.entity_type == "chain"
        assert draft.display_name == "Weather report"
        assert draft.original_display_name == "Weather report"
        assert draft.description == "Hourly weather agent"
        assert draft.tags == ("domain:weather",)
        assert draft.original_tags == ("domain:weather",)
        assert draft.task_description == "Build a weather summary"

    def test_empty_metadata_yields_blank_originals(self):
        class _Bare:
            def get_care_metadata(self):
                return None

        draft = extract_edit_draft(_Bare(), "ent-1")
        assert draft.display_name == ""
        assert draft.tags == ()
        assert draft.task_description == ""

    def test_dict_chain_accepted(self):
        chain = {"metadata": {"care": _care_meta(display_name="From dict")}}
        draft = extract_edit_draft(chain, "ent-1")
        assert draft.display_name == "From dict"

    def test_pydantic_model_dump_accepted(self):
        class _Meta:
            def model_dump(self, **_kw):
                return _care_meta(display_name="From model")

        class _Chain:
            def get_care_metadata(self):
                return _Meta()

        assert extract_edit_draft(_Chain(), "ent-1").display_name == "From model"

    def test_missing_entity_id_raises(self):
        with pytest.raises(EditDraftError, match="entity_id"):
            extract_edit_draft(_StubChain(), "")

    def test_parent_version_id_and_channel_carry_through(self):
        draft = extract_edit_draft(
            _StubChain(),
            "ent-1",
            parent_version_id="v-7",
            channel="stable",
            entity_type="agent",
        )
        assert draft.parent_version_id == "v-7"
        assert draft.channel == "stable"
        assert draft.entity_type == "agent"


# ---------------------------------------------------------------------------
# Dirty tracking
# ---------------------------------------------------------------------------


class TestDirtyTracking:
    def test_fresh_extract_is_clean(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        assert draft.is_dirty() is False
        assert draft.dirty_fields() == ()
        assert draft.is_structural_edit is False

    def test_display_name_edit_flags_dirty(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_display_name(draft, "Better name")
        assert edited.is_dirty()
        assert edited.dirty_fields() == ("display_name",)

    def test_whitespace_only_change_is_clean(self):
        # Strip-equality means trailing whitespace alone isn't an edit.
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_display_name(draft, "Weather report  ")
        assert edited.is_dirty() is False

    def test_tag_reorder_is_clean(self):
        # Tags compare as sets so re-ordering chips doesn't trigger
        # the "modified" badge.
        draft = extract_edit_draft(
            _StubChain(meta=_care_meta(tags=["a", "b", "c"])), "ent-1"
        )
        edited = set_tags(draft, ["c", "a", "b"])
        assert edited.is_dirty() is False

    def test_tag_added_is_dirty(self):
        draft = extract_edit_draft(
            _StubChain(meta=_care_meta(tags=["a"])), "ent-1"
        )
        edited = set_tags(draft, ["a", "b"])
        assert edited.is_dirty()
        assert "tags" in edited.dirty_fields()

    def test_update_chain_flags_structural_edit(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = update_chain(draft, {"steps": [{"prompt": "v2"}]})
        assert edited.is_structural_edit
        assert "chain_content" in edited.dirty_fields()

    def test_update_chain_dirty_false_skips_flag(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        # Refresh from disk: dirty=False so it doesn't count.
        edited = update_chain(draft, "refreshed", dirty=False)
        assert edited.is_structural_edit is False
        assert edited.is_dirty() is False

    def test_set_description_dirty(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_description(draft, "Brand new pitch")
        assert "description" in edited.dirty_fields()

    def test_set_task_description_dirty(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_task_description(draft, "Different task")
        assert "task_description" in edited.dirty_fields()

    def test_change_summary_does_not_affect_dirty(self):
        # change_summary IS a write-time slot; setting it without
        # touching other fields shouldn't flag the draft dirty.
        draft = extract_edit_draft(_StubChain(), "ent-1")
        with_summary = set_change_summary(draft, "Tweaked it")
        assert with_summary.is_dirty() is False
        assert with_summary.change_summary == "Tweaked it"

    def test_set_tags_strips_whitespace(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_tags(draft, ["  a  ", "", "b "])
        assert edited.tags == ("a", "b")


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_restores_originals(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = (
            set_display_name(
                set_tags(
                    set_description(draft, "New desc"),
                    ["new-tag"],
                ),
                "New name",
            )
        )
        cleared = reset(edited)
        assert cleared.is_dirty() is False
        assert cleared.display_name == "Weather report"
        assert cleared.description == "Hourly weather agent"
        assert cleared.tags == ("domain:weather",)

    def test_reset_clears_change_summary(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        with_summary = set_change_summary(draft, "Some note")
        cleared = reset(with_summary)
        assert cleared.change_summary == ""

    def test_reset_clears_chain_content_dirty(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = update_chain(draft, {"steps": []})
        cleared = reset(edited)
        assert cleared.chain_content_dirty is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_clean_draft_no_issues(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        assert validate_edit_draft(draft) == ()

    def test_empty_display_name_error(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_display_name(draft, "   ")
        issues = validate_edit_draft(edited)
        assert any(
            i.field == "display_name" and i.severity == "error"
            for i in issues
        )

    def test_structural_edit_requires_change_summary(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = update_chain(draft, {"steps": [{"prompt": "v2"}]})
        issues = validate_edit_draft(edited)
        assert any(
            i.field == "chain_content" and i.severity == "error"
            for i in issues
        )
        # Add a summary → no more error.
        with_summary = set_change_summary(edited, "Tweaked step 1")
        issues = validate_edit_draft(with_summary)
        assert not any(
            i.field == "chain_content" and i.severity == "error"
            for i in issues
        )

    def test_metadata_edit_does_not_require_summary(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        renamed = set_display_name(draft, "Renamed")
        assert validate_edit_draft(renamed) == ()

    def test_duplicate_tags_warning(self):
        # Bypass set_tags by constructing directly to retain
        # duplicates (set_tags strips whitespace but doesn't dedupe).
        edited = EditAgentDraft(
            entity_id="ent-1",
            display_name="ok",
            original_display_name="ok",
            tags=("foo", "foo", "bar"),
            original_tags=("foo",),
        )
        issues = validate_edit_draft(edited)
        assert any(
            i.field == "tags" and i.severity == "warning" for i in issues
        )

    def test_issue_is_frozen(self):
        issue = EditDraftIssue(severity="error", field="display_name", message="x")
        with pytest.raises(FrozenInstanceError):
            issue.severity = "warning"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# save_edit_as_new_version
# ---------------------------------------------------------------------------


class TestSaveEditAsNewVersion:
    def test_no_dirty_short_circuits(self):
        memory = _StubMemoryWithSave()
        draft = extract_edit_draft(_StubChain(), "ent-1")
        result = asyncio.run(save_edit_as_new_version(memory, draft))
        assert result.success is True
        assert result.fields_written == ()
        assert memory.save_calls == []

    def test_writes_with_entity_id_for_new_version(self):
        memory = _StubMemoryWithSave()
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_display_name(draft, "Renamed agent")
        result = asyncio.run(save_edit_as_new_version(memory, edited))
        assert result.success
        assert "display_name" in result.fields_written
        # The SDK call carries the entity_id so a new version
        # lands under the same chain.
        call = memory.save_calls[0]
        assert call["entity_id"] == "ent-1"
        assert call["channel"] == "latest"
        assert call["name"] == "Renamed agent"

    def test_tags_forwarded(self):
        memory = _StubMemoryWithSave()
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_tags(draft, ["domain:weather", "favourite"])
        asyncio.run(save_edit_as_new_version(memory, edited))
        assert memory.save_calls[0]["tags"] == ["domain:weather", "favourite"]

    def test_save_failure_surfaces_on_result(self):
        memory = _StubMemoryWithSave(save_exc=RuntimeError("503 Service Unavailable"))
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_display_name(draft, "Renamed")
        result = asyncio.run(save_edit_as_new_version(memory, edited))
        assert result.success is False
        assert "503" in result.error
        assert result.fields_written == ()

    def test_timeout(self):
        memory = _StubMemoryWithSave(save_delay=0.5)
        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_display_name(draft, "Renamed")
        result = asyncio.run(
            save_edit_as_new_version(memory, edited, timeout=0.05)
        )
        assert result.success is False
        assert "timed out" in result.error

    def test_missing_entity_id_raises(self):
        memory = _StubMemoryWithSave()
        draft = EditAgentDraft(entity_id="")
        with pytest.raises(EditDraftError, match="entity_id"):
            asyncio.run(save_edit_as_new_version(memory, draft))

    def test_missing_save_chain_raises(self):
        class _BadMemory:
            pass

        draft = extract_edit_draft(_StubChain(), "ent-1")
        edited = set_display_name(draft, "Renamed")
        with pytest.raises(EditDraftError, match="save_chain"):
            asyncio.run(save_edit_as_new_version(_BadMemory(), edited))

    def test_chain_content_dirty_is_written(self):
        memory = _StubMemoryWithSave()
        draft = extract_edit_draft(_StubChain(), "ent-1")
        new_chain_content = {"steps": [{"prompt": "edited"}]}
        edited = update_chain(draft, new_chain_content)
        edited = set_change_summary(edited, "Step 1 prompt v2")
        result = asyncio.run(save_edit_as_new_version(memory, edited))
        assert result.success
        assert "chain_content" in result.fields_written
        # The chain content carrier was forwarded as the chain payload.
        assert memory.save_calls[0]["chain"] == new_chain_content


# ---------------------------------------------------------------------------
# promote_to_stable
# ---------------------------------------------------------------------------


class TestPromoteToStable:
    def test_calls_sdk_promote(self):
        memory = _StubMemoryWithSave()
        draft = extract_edit_draft(_StubChain(), "ent-1")
        result = asyncio.run(promote_to_stable(memory, draft))
        assert result.success
        assert result.from_channel == "latest"
        assert result.to_channel == "stable"
        call = memory.promote_calls[0]
        assert call["entity_id"] == "ent-1"
        assert call["entity_type"] == "chain"
        assert call["from_channel"] == "latest"
        assert call["to_channel"] == "stable"

    def test_custom_source_channel(self):
        memory = _StubMemoryWithSave()
        draft = extract_edit_draft(_StubChain(), "ent-1", channel="evolved")
        asyncio.run(promote_to_stable(memory, draft, from_channel="evolved"))
        assert memory.promote_calls[0]["from_channel"] == "evolved"

    def test_failure_surfaces_on_result(self):
        memory = _StubMemoryWithSave(promote_exc=RuntimeError("403 Forbidden"))
        draft = extract_edit_draft(_StubChain(), "ent-1")
        result = asyncio.run(promote_to_stable(memory, draft))
        assert result.success is False
        assert "403" in result.error

    def test_timeout(self):
        memory = _StubMemoryWithSave(promote_delay=0.5)
        draft = extract_edit_draft(_StubChain(), "ent-1")
        result = asyncio.run(
            promote_to_stable(memory, draft, timeout=0.05)
        )
        assert result.success is False
        assert "timed out" in result.error

    def test_missing_entity_id_raises(self):
        memory = _StubMemoryWithSave()
        draft = EditAgentDraft(entity_id="")
        with pytest.raises(EditDraftError, match="entity_id"):
            asyncio.run(promote_to_stable(memory, draft))

    def test_missing_promote_raises(self):
        class _BadMemory:
            class client:
                pass

        draft = extract_edit_draft(_StubChain(), "ent-1")
        with pytest.raises(EditDraftError, match="promote"):
            asyncio.run(promote_to_stable(_BadMemory(), draft))


# ---------------------------------------------------------------------------
# Frozen models
# ---------------------------------------------------------------------------


class TestFrozenModels:
    def test_draft_is_frozen(self):
        draft = extract_edit_draft(_StubChain(), "ent-1")
        with pytest.raises(FrozenInstanceError):
            draft.display_name = "x"  # type: ignore[misc]

    def test_save_result_is_frozen(self):
        result = SaveEditResult(entity_id="x")
        with pytest.raises(FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    def test_promote_result_is_frozen(self):
        result = PromoteResult(entity_id="x", from_channel="a", to_channel="b")
        with pytest.raises(FrozenInstanceError):
            result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            EditAgentDraft as Draft,
            EditDraftError as Err,
            EditDraftIssue as Issue,
            extract_edit_draft as extract,
            save_edit_as_new_version as save,
            promote_to_stable as promote,
            validate_edit_draft as validate,
        )

        assert Draft is EditAgentDraft
        assert Err is EditDraftError
        assert Issue is EditDraftIssue
        assert extract is extract_edit_draft
        assert save is save_edit_as_new_version
        assert promote is promote_to_stable
        assert validate is validate_edit_draft
