"""Tests for the SaveAgentModal data layer (TODO §3 P0).

The Textual modal is gated on §1 P0; this suite pins the
contract the modal binds to.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError

import pytest

from care.runtime.save_agent_form import (
    FAVOURITE_TAG,
    SaveAgentError,
    SaveAgentForm,
    SaveAgentIssue,
    SaveAgentOutcome,
    add_tag,
    apply_save_agent_form,
    remove_tag,
    seed_save_agent_form,
    set_description,
    set_display_name,
    set_keep_context,
    set_tags,
    toggle_favourite,
    validate_save_agent_form,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeMageMetadata:
    """Mimics a `MAGEMetadata` instance."""

    def __init__(
        self,
        *,
        domain: str = "weather",
        suggested_display_name: str = "Weather forecaster",
        suggested_description: str = "Generates 24h weather summaries",
        suggested_tags=("forecast", "weather"),
    ):
        self.domain = domain
        self.suggested_display_name = suggested_display_name
        self.suggested_description = suggested_description
        self.suggested_tags = list(suggested_tags)

    def model_dump(self, **_kw):
        return {
            "domain": self.domain,
            "suggested_display_name": self.suggested_display_name,
            "suggested_description": self.suggested_description,
            "suggested_tags": list(self.suggested_tags),
        }


class _FakeContextFile:
    """Mimics a `ContextFileRef`."""

    def __init__(self, path, sha256="a" * 64, size_bytes=128, mime_type="text/plain"):
        self.path = path
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self.mime_type = mime_type

    def model_dump(self, **_kw):
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
        }


class _FakeDraftSession:
    """Mimics a `DraftSession`."""

    def __init__(self, entity_id="ent-1", entity_type="chain"):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.promoted = False
        self.discarded = False


class _StubClient:
    def __init__(
        self,
        *,
        list_response=None,
        list_exc=None,
        list_delay=0.0,
        promote_exc=None,
        update_exc=None,
        update_delay=0.0,
    ):
        self.calls: list[dict] = []
        self._list_response = list_response or []
        self._list_exc = list_exc
        self._list_delay = list_delay
        self._promote_exc = promote_exc
        self._update_exc = update_exc
        self._update_delay = update_delay

    def list_chains(self, *, limit, channel, q=None, namespace=None, **_):
        self.calls.append(
            {"op": "list", "q": q, "namespace": namespace, "limit": limit}
        )
        if self._list_delay:
            time.sleep(self._list_delay)
        if self._list_exc:
            raise self._list_exc
        return self._list_response

    def promote(self, entity_id, from_channel="draft", to_channel="latest",
                entity_type="chain"):
        self.calls.append(
            {
                "op": "promote", "entity_id": entity_id,
                "from_channel": from_channel, "to_channel": to_channel,
                "entity_type": entity_type,
            }
        )
        if self._promote_exc:
            raise self._promote_exc
        return {"ok": True}

    def _update_metadata(
        self, entity_type, entity_id, *, display_name=None, description=None,
        tags=None, favourite=None,
    ):
        self.calls.append(
            {
                "op": "patch", "entity_type": entity_type, "entity_id": entity_id,
                "display_name": display_name,
                "description": description,
                "tags": list(tags) if tags is not None else None,
                "favourite": favourite,
            }
        )
        if self._update_delay:
            time.sleep(self._update_delay)
        if self._update_exc:
            raise self._update_exc
        return {"ok": True}


class _StubMemory:
    def __init__(self, client):
        self.client = client


# ---------------------------------------------------------------------------
# seed_save_agent_form
# ---------------------------------------------------------------------------


class TestSeed:
    def test_uses_suggested_fields(self):
        form = seed_save_agent_form(
            query="What's the weather?",
            mage_metadata=_FakeMageMetadata(),
            context_files=[],
        )
        assert form.display_name == "Weather forecaster"
        assert form.description == "Generates 24h weather summaries"
        # `domain:weather` prepended to the seeded tags.
        assert form.tags[0] == "domain:weather"
        assert "forecast" in form.tags
        assert "weather" in form.tags
        assert form.keep_context is True
        # Snapshots match the seed (form starts clean).
        assert form.suggested_display_name == "Weather forecaster"
        assert form.is_dirty() is False

    def test_heuristic_name_when_no_suggested(self):
        # Empty suggested_display_name → heuristic builds from
        # domain + first 60 chars of query.
        meta = _FakeMageMetadata(suggested_display_name="")
        long_query = "Describe weather conditions across the Pacific Northwest"
        form = seed_save_agent_form(query=long_query, mage_metadata=meta)
        assert form.display_name.startswith("weather · ")
        # 60-char truncation includes the suffix as-is when short.
        assert long_query in form.display_name

    def test_heuristic_truncates_long_query(self):
        meta = _FakeMageMetadata(suggested_display_name="", domain="x")
        query = "A" * 100
        form = seed_save_agent_form(query=query, mage_metadata=meta)
        # Truncated to 60 chars + ellipsis.
        assert "…" in form.display_name
        assert form.display_name.count("A") == 60

    def test_no_metadata_yields_empty_form(self):
        form = seed_save_agent_form(query="x")
        # No suggested name, no domain → name is just the truncated query.
        assert form.display_name == "x"
        assert form.description == "x"
        assert form.tags == ()
        assert form.domain == ""

    def test_dict_metadata_accepted(self):
        meta = {
            "domain": "finance",
            "suggested_display_name": "Stock summariser",
            "suggested_description": "Daily stock report",
            "suggested_tags": ["stocks", "summary"],
        }
        form = seed_save_agent_form(query="summarise stocks", mage_metadata=meta)
        assert form.display_name == "Stock summariser"
        assert form.description == "Daily stock report"
        assert "domain:finance" in form.tags
        assert "stocks" in form.tags

    def test_context_files_preserved(self):
        files = [
            _FakeContextFile("/tmp/a.pdf"),
            {"path": "/tmp/b.txt", "sha256": "b" * 64, "size_bytes": 8},
        ]
        form = seed_save_agent_form(
            query="x", mage_metadata=_FakeMageMetadata(), context_files=files,
        )
        assert len(form.context_files) == 2
        assert form.context_files[0]["path"] == "/tmp/a.pdf"
        assert form.context_files[1]["path"] == "/tmp/b.txt"

    def test_suggested_name_override_wins(self):
        form = seed_save_agent_form(
            query="x",
            mage_metadata=_FakeMageMetadata(),
            suggested_name_override="My custom name",
        )
        assert form.display_name == "My custom name"
        assert form.suggested_display_name == "My custom name"

    def test_form_is_frozen(self):
        form = seed_save_agent_form(query="x")
        with pytest.raises(FrozenInstanceError):
            form.display_name = "y"  # type: ignore[misc]

    def test_duplicate_seeded_tags_deduped(self):
        meta = _FakeMageMetadata(
            domain="weather",
            suggested_tags=["weather", "weather", "forecast"],
        )
        form = seed_save_agent_form(query="x", mage_metadata=meta)
        # `weather` appears twice in suggested + `domain:weather` —
        # only unique values remain.
        assert list(form.tags).count("weather") == 1
        assert list(form.tags).count("domain:weather") == 1


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


class TestMutators:
    def _form(self):
        return seed_save_agent_form(
            query="What's the weather?", mage_metadata=_FakeMageMetadata(),
        )

    def test_set_display_name(self):
        f = self._form()
        edited = set_display_name(f, "Renamed")
        assert edited.display_name == "Renamed"
        assert edited.display_name_dirty
        assert edited.is_dirty()

    def test_whitespace_only_change_isnt_dirty(self):
        f = self._form()
        edited = set_display_name(f, f.display_name + "  ")
        assert edited.display_name_dirty is False

    def test_set_description(self):
        f = self._form()
        edited = set_description(f, "Brand new pitch")
        assert edited.description == "Brand new pitch"
        assert edited.description_dirty

    def test_set_tags_strips_whitespace(self):
        f = self._form()
        edited = set_tags(f, ["  alpha", "beta  ", "", "  "])
        assert edited.tags == ("alpha", "beta")

    def test_add_tag_dedupe(self):
        f = self._form()
        with_added = add_tag(f, "new-tag")
        assert "new-tag" in with_added.tags
        # Adding the same again → no-op (same instance).
        same = add_tag(with_added, "new-tag")
        assert same is with_added

    def test_remove_tag_present(self):
        f = self._form()
        with_added = add_tag(f, "drop-me")
        cleared = remove_tag(with_added, "drop-me")
        assert "drop-me" not in cleared.tags

    def test_remove_tag_absent_is_noop(self):
        f = self._form()
        same = remove_tag(f, "not-there")
        assert same is f

    def test_toggle_favourite(self):
        f = self._form()
        assert f.favourite is False
        on = toggle_favourite(f)
        assert on.favourite is True
        assert FAVOURITE_TAG in on.tags
        off = toggle_favourite(on)
        assert off.favourite is False
        assert FAVOURITE_TAG not in off.tags

    def test_set_keep_context(self):
        f = self._form()
        off = set_keep_context(f, False)
        assert off.keep_context is False
        on = set_keep_context(off, True)
        assert on.keep_context is True

    def test_add_tag_strips_whitespace(self):
        f = self._form()
        added = add_tag(f, "  spaced  ")
        assert "spaced" in added.tags
        assert "  spaced  " not in added.tags

    def test_add_tag_empty_is_noop(self):
        f = self._form()
        same = add_tag(f, "   ")
        assert same is f


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidate:
    def test_clean_form_no_issues(self):
        form = seed_save_agent_form(
            query="x", mage_metadata=_FakeMageMetadata(),
        )
        issues = asyncio.run(
            validate_save_agent_form(form, memory=None, check_unique=False)
        )
        assert issues == ()

    def test_empty_name_error(self):
        form = seed_save_agent_form(
            query="x", mage_metadata=_FakeMageMetadata(),
            suggested_name_override="   ",
        )
        issues = asyncio.run(
            validate_save_agent_form(form, check_unique=False)
        )
        assert any(
            i.field == "display_name" and i.severity == "error" for i in issues
        )

    def test_duplicate_tags_warning(self):
        # Bypass the dedupe in set_tags to retain duplicates.
        form = SaveAgentForm(
            display_name="ok",
            description="ok",
            tags=("a", "a", "b"),
        )
        issues = asyncio.run(
            validate_save_agent_form(form, check_unique=False)
        )
        assert any(
            i.field == "tags" and i.severity == "warning" for i in issues
        )

    def test_unique_check_flags_collision(self):
        client = _StubClient(
            list_response=[
                {"display_name": "Weather forecaster", "entity_id": "other-1"},
            ]
        )
        memory = _StubMemory(client)
        form = seed_save_agent_form(
            query="x", mage_metadata=_FakeMageMetadata(),
        )
        issues = asyncio.run(
            validate_save_agent_form(form, memory=memory, namespace="alice")
        )
        assert any(
            i.field == "display_name" and i.severity == "error" for i in issues
        )
        # Detail carries the colliding entity id.
        clash_issue = next(
            i for i in issues if i.field == "display_name" and i.severity == "error"
        )
        assert clash_issue.detail == "other-1"

    def test_unique_check_case_insensitive(self):
        client = _StubClient(
            list_response=[
                {"display_name": "WEATHER forecaster", "entity_id": "x"},
            ]
        )
        memory = _StubMemory(client)
        form = seed_save_agent_form(
            query="x", mage_metadata=_FakeMageMetadata(),
        )
        issues = asyncio.run(
            validate_save_agent_form(form, memory=memory)
        )
        assert any(
            i.field == "display_name" and i.severity == "error" for i in issues
        )

    def test_unique_check_no_collision_passes(self):
        # `q=` substring matches return rows where the substring
        # appears but no exact match — should pass.
        client = _StubClient(
            list_response=[
                {"display_name": "Weather forecaster v2", "entity_id": "v2"},
            ]
        )
        memory = _StubMemory(client)
        form = seed_save_agent_form(
            query="x", mage_metadata=_FakeMageMetadata(),
        )
        issues = asyncio.run(
            validate_save_agent_form(form, memory=memory)
        )
        assert not any(
            i.field == "display_name" and i.severity == "error" for i in issues
        )

    def test_unique_check_network_failure_silent(self):
        # Network outage → degrades silently (no issue), the
        # POST surfaces the collision if any.
        client = _StubClient(list_exc=RuntimeError("connection refused"))
        memory = _StubMemory(client)
        form = seed_save_agent_form(
            query="x", mage_metadata=_FakeMageMetadata(),
        )
        issues = asyncio.run(
            validate_save_agent_form(form, memory=memory)
        )
        # No display_name error from network failure.
        assert not any(
            i.field == "display_name" and i.severity == "error"
            for i in issues
        )

    def test_unique_check_skipped_when_disabled(self):
        client = _StubClient(
            list_response=[
                {"display_name": "Weather forecaster", "entity_id": "x"},
            ]
        )
        memory = _StubMemory(client)
        form = seed_save_agent_form(
            query="x", mage_metadata=_FakeMageMetadata(),
        )
        issues = asyncio.run(
            validate_save_agent_form(form, memory=memory, check_unique=False)
        )
        # Even though a clash exists, check_unique=False skips the probe.
        assert not any(
            i.field == "display_name" and i.severity == "error" for i in issues
        )
        # The probe wasn't even attempted.
        assert not any(c["op"] == "list" for c in client.calls)

    def test_unique_check_skipped_when_name_already_empty(self):
        # When the name is empty we already flagged an error; the
        # uniqueness probe is wasted bandwidth.
        client = _StubClient(
            list_response=[
                {"display_name": "ignored", "entity_id": "x"},
            ]
        )
        memory = _StubMemory(client)
        form = seed_save_agent_form(
            query="x", suggested_name_override="",
        )
        asyncio.run(
            validate_save_agent_form(form, memory=memory)
        )
        # Empty name → no list call (probe was skipped).
        assert not any(c["op"] == "list" for c in client.calls)


# ---------------------------------------------------------------------------
# apply_save_agent_form
# ---------------------------------------------------------------------------


class TestApplySaveAgentForm:
    def _setup(self, **client_kwargs):
        client = _StubClient(**client_kwargs)
        memory = _StubMemory(client)
        session = _FakeDraftSession()
        form = seed_save_agent_form(
            query="x", mage_metadata=_FakeMageMetadata(),
        )
        return memory, session, form

    def test_promote_then_patch(self):
        memory, session, form = self._setup()
        result = asyncio.run(apply_save_agent_form(memory, session, form))
        assert result.success
        assert result.promoted
        assert result.metadata_written
        assert result.entity_id == "ent-1"
        # Promote ran first, then PATCH.
        ops = [c["op"] for c in memory.client.calls]
        assert ops == ["promote", "patch"]
        # Promote args.
        promote = memory.client.calls[0]
        assert promote["from_channel"] == "draft"
        assert promote["to_channel"] == "latest"
        # PATCH args carry the form's edits.
        patch = memory.client.calls[1]
        assert patch["display_name"] == "Weather forecaster"
        assert patch["favourite"] is False
        # `draft` tag stripped automatically.
        assert "draft" not in (patch["tags"] or [])

    def test_favourite_routed_to_dedicated_column(self):
        memory, session, form = self._setup()
        favourited = toggle_favourite(form)
        result = asyncio.run(
            apply_save_agent_form(memory, session, favourited)
        )
        assert result.success
        patch = next(c for c in memory.client.calls if c["op"] == "patch")
        # `favourite` flips to True on the column.
        assert patch["favourite"] is True
        # `favourite` tag removed from the tag list (it lives on
        # the column instead).
        assert FAVOURITE_TAG not in (patch["tags"] or [])

    def test_missing_entity_id_raises(self):
        memory, _, form = self._setup()
        session = _FakeDraftSession(entity_id="")
        with pytest.raises(SaveAgentError, match="entity_id"):
            asyncio.run(apply_save_agent_form(memory, session, form))

    def test_promote_failure_surfaces_on_result(self):
        memory, session, form = self._setup(
            promote_exc=RuntimeError("503 backend dead")
        )
        result = asyncio.run(apply_save_agent_form(memory, session, form))
        assert result.success is False
        assert result.promoted is False
        assert "503" in result.error

    def test_patch_failure_surfaces_with_promoted_true(self):
        memory, session, form = self._setup(
            update_exc=RuntimeError("422 invalid tag")
        )
        result = asyncio.run(apply_save_agent_form(memory, session, form))
        # Promote landed, but PATCH failed — outcome reflects
        # the partial-success state.
        assert result.success is False
        assert result.promoted is True
        assert result.metadata_written is False
        assert "422" in result.error

    def test_promote_timeout(self):
        memory, session, form = self._setup()

        def slow_promote(entity_id, from_channel="draft", to_channel="latest",
                         entity_type="chain"):
            time.sleep(0.5)
            return {"ok": True}

        memory.client.promote = slow_promote  # type: ignore[method-assign]
        result = asyncio.run(
            apply_save_agent_form(memory, session, form, timeout=0.05)
        )
        assert result.success is False
        assert "timed out" in result.error

    def test_patch_timeout_keeps_promoted_flag(self):
        memory, session, form = self._setup(update_delay=0.5)
        result = asyncio.run(
            apply_save_agent_form(memory, session, form, timeout=0.05)
        )
        # Promote succeeded fast; PATCH timed out.
        assert result.promoted is True
        assert result.metadata_written is False
        assert "timed out" in result.error

    def test_missing_update_metadata_returns_warning_outcome(self):
        memory, session, form = self._setup()
        # Drop `_update_metadata` from the instance (it lives on
        # the class, so override with a property-less attribute
        # to mask it).
        memory.client._update_metadata = None  # type: ignore[assignment]
        result = asyncio.run(apply_save_agent_form(memory, session, form))
        # Promote landed, PATCH skipped (no method) — surfaced as
        # promoted=True, metadata_written=False with an error
        # describing the missing method.
        assert result.success is True
        assert result.promoted is True
        assert result.metadata_written is False
        assert "_update_metadata" in (result.error or "")

    def test_to_channel_override(self):
        memory, session, form = self._setup()
        asyncio.run(
            apply_save_agent_form(
                memory, session, form, to_channel="stable",
            )
        )
        promote = memory.client.calls[0]
        assert promote["to_channel"] == "stable"


# ---------------------------------------------------------------------------
# Outcome shape
# ---------------------------------------------------------------------------


class TestOutcomeShape:
    def test_outcome_is_frozen(self):
        outcome = SaveAgentOutcome(entity_id="x")
        with pytest.raises(FrozenInstanceError):
            outcome.success = False  # type: ignore[misc]

    def test_issue_is_frozen(self):
        issue = SaveAgentIssue(severity="error", field="display_name", message="x")
        with pytest.raises(FrozenInstanceError):
            issue.severity = "warning"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            FAVOURITE_TAG as TAG,
            SaveAgentForm as F,
            SaveAgentIssue as Iss,
            SaveAgentOutcome as O,
            apply_save_agent_form as apply,
            seed_save_agent_form as seed,
            toggle_favourite as toggle,
            validate_save_agent_form as validate,
        )

        assert TAG == FAVOURITE_TAG
        assert F is SaveAgentForm
        assert Iss is SaveAgentIssue
        assert O is SaveAgentOutcome
        assert apply is apply_save_agent_form
        assert seed is seed_save_agent_form
        assert toggle is toggle_favourite
        assert validate is validate_save_agent_form
