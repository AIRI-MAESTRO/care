"""Tests for ``care.conflict`` (TODO §3 P1).

Six coverage layers:

1. **`compute_content_sha256`** — stable across key orderings,
   deterministic across runs, copes with non-JSON values.
2. **`detect_conflict` no-existing path** — returns ``None``
   when the lookup finds nothing.
3. **`detect_conflict` no-conflict path** — same content
   yields a report with `is_conflict=False` and empty diff.
4. **`detect_conflict` conflict path** — different content
   produces unified-diff lines + populated SHAs.
5. **`apply_resolution`** — every literal value dispatches to
   the right save method; unknown resolution / unknown
   entity_type / missing save method raise; downstream save
   exceptions wrap.
6. **Error wrapping** — memory.find_entity_by_name raising
   wraps in :class:`ConflictResolutionError`; non-dict
   responses raise too.
"""

from __future__ import annotations

from typing import Any

import pytest

from care.conflict import (
    ConflictReport,
    ConflictResolutionError,
    apply_resolution,
    compute_content_sha256,
    detect_conflict,
)


# ---------------------------------------------------------------------------
# Stub memory
# ---------------------------------------------------------------------------


class _StubMemory:
    """Records every method call so tests can assert."""

    def __init__(
        self,
        *,
        existing: dict[str, Any] | None = None,
        raise_on_lookup: Exception | None = None,
        raise_on_save: Exception | None = None,
        return_lookup_type: Any = None,
    ):
        self.existing = existing
        self.raise_on_lookup = raise_on_lookup
        self.raise_on_save = raise_on_save
        self.return_lookup_type = return_lookup_type
        self.lookup_calls: list[dict[str, Any]] = []
        self.save_calls: list[dict[str, Any]] = []

    def find_entity_by_name(self, **kwargs: Any) -> Any:
        self.lookup_calls.append(kwargs)
        if self.raise_on_lookup is not None:
            raise self.raise_on_lookup
        if self.return_lookup_type is not None:
            return self.return_lookup_type
        return self.existing

    def save_chain(self, content: dict[str, Any], **kwargs: Any) -> str:
        self.save_calls.append({"method": "save_chain", "content": content, **kwargs})
        if self.raise_on_save is not None:
            raise self.raise_on_save
        return kwargs.get("entity_id") or "new-entity-id"

    def save_agent_skill(self, content: dict[str, Any], **kwargs: Any) -> str:
        self.save_calls.append({"method": "save_agent_skill", "content": content, **kwargs})
        return kwargs.get("entity_id") or "new-skill-id"

    def save_memory_card(self, content: dict[str, Any], **kwargs: Any) -> str:
        self.save_calls.append({"method": "save_memory_card", "content": content, **kwargs})
        return kwargs.get("entity_id") or "new-card-id"


# ---------------------------------------------------------------------------
# compute_content_sha256
# ---------------------------------------------------------------------------


class TestSha:
    def test_deterministic(self):
        a = {"steps": [{"number": 1, "title": "x"}]}
        b = {"steps": [{"number": 1, "title": "x"}]}
        assert compute_content_sha256(a) == compute_content_sha256(b)

    def test_key_order_independent(self):
        a = {"a": 1, "b": 2, "c": 3}
        b = {"c": 3, "a": 1, "b": 2}
        assert compute_content_sha256(a) == compute_content_sha256(b)

    def test_nested_key_order_independent(self):
        a = {"x": {"a": 1, "b": 2}}
        b = {"x": {"b": 2, "a": 1}}
        assert compute_content_sha256(a) == compute_content_sha256(b)

    def test_different_content_different_sha(self):
        a = {"task": "weather"}
        b = {"task": "stocks"}
        assert compute_content_sha256(a) != compute_content_sha256(b)

    def test_handles_non_json_values(self):
        # Datetimes coerce via str(); doesn't raise.
        from datetime import datetime
        a = {"ts": datetime(2026, 1, 1)}
        b = {"ts": datetime(2026, 1, 1)}
        # Same value → same hash.
        assert compute_content_sha256(a) == compute_content_sha256(b)

    def test_returns_64_char_hex(self):
        digest = compute_content_sha256({"x": 1})
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# detect_conflict
# ---------------------------------------------------------------------------


class TestDetectConflict:
    def test_no_existing_entity_returns_none(self):
        memory = _StubMemory(existing=None)
        report = detect_conflict(
            memory,
            name="Weather",
            entity_type="chain",
            incoming_content={"steps": []},
        )
        assert report is None
        # Lookup was attempted with the documented kwargs.
        assert memory.lookup_calls[0]["name"] == "Weather"
        assert memory.lookup_calls[0]["entity_type"] == "chain"

    def test_no_conflict_when_sha_matches(self):
        content = {"task": "weather", "steps": []}
        memory = _StubMemory(
            existing={"entity_id": "ent-1", "content": content}
        )
        report = detect_conflict(
            memory,
            name="Weather",
            entity_type="chain",
            incoming_content=content,
        )
        assert report is not None
        assert report.is_conflict is False
        assert report.existing_entity_id == "ent-1"
        assert report.existing_sha256 == report.incoming_sha256
        # No diff when no conflict.
        assert report.diff_lines == ()

    def test_conflict_produces_diff(self):
        existing = {"task": "weather", "steps": [{"title": "old"}]}
        incoming = {"task": "weather", "steps": [{"title": "new"}]}
        memory = _StubMemory(
            existing={"entity_id": "ent-1", "content": existing}
        )
        report = detect_conflict(
            memory,
            name="Weather",
            entity_type="chain",
            incoming_content=incoming,
        )
        assert report is not None
        assert report.is_conflict is True
        assert report.existing_sha256 != report.incoming_sha256
        # Unified-diff lines reference both labels.
        joined = "\n".join(report.diff_lines)
        assert "Weather (existing)" in joined
        assert "Weather (incoming)" in joined
        # The actual diff content includes the changed fields.
        assert "old" in joined
        assert "new" in joined

    def test_conflict_preserves_full_contents(self):
        existing = {"a": 1}
        incoming = {"a": 2}
        memory = _StubMemory(
            existing={"entity_id": "ent-2", "content": existing}
        )
        report = detect_conflict(
            memory,
            name="X",
            entity_type="chain",
            incoming_content=incoming,
        )
        assert report.existing_content == existing
        assert report.incoming_content == incoming

    def test_namespace_forwarded(self):
        memory = _StubMemory(existing=None)
        detect_conflict(
            memory,
            name="X",
            entity_type="chain",
            incoming_content={},
            namespace="org1",
        )
        assert memory.lookup_calls[0]["namespace"] == "org1"

    def test_lookup_raising_wraps(self):
        memory = _StubMemory(raise_on_lookup=RuntimeError("DB down"))
        with pytest.raises(ConflictResolutionError, match="lookup failed.*DB down"):
            detect_conflict(
                memory,
                name="X",
                entity_type="chain",
                incoming_content={},
            )

    def test_memory_without_finder_raises(self):
        class _Bare:
            pass

        with pytest.raises(ConflictResolutionError, match="find_entity_by_name"):
            detect_conflict(
                _Bare(),
                name="X",
                entity_type="chain",
                incoming_content={},
            )

    def test_non_dict_response_raises(self):
        memory = _StubMemory(return_lookup_type=["not a dict"])
        with pytest.raises(ConflictResolutionError, match="expected dict"):
            detect_conflict(
                memory,
                name="X",
                entity_type="chain",
                incoming_content={},
            )

    def test_non_dict_existing_content_raises(self):
        memory = _StubMemory(
            existing={"entity_id": "x", "content": ["not", "a", "dict"]}
        )
        with pytest.raises(ConflictResolutionError, match="existing content"):
            detect_conflict(
                memory,
                name="X",
                entity_type="chain",
                incoming_content={},
            )


# ---------------------------------------------------------------------------
# apply_resolution
# ---------------------------------------------------------------------------


def _report(
    *,
    entity_type: str = "chain",
    name: str = "X",
    existing_content: dict[str, Any] | None = None,
    incoming_content: dict[str, Any] | None = None,
) -> ConflictReport:
    existing_content = existing_content or {"v": "old"}
    incoming_content = incoming_content or {"v": "new"}
    return ConflictReport(
        existing_entity_id="ent-1",
        existing_sha256=compute_content_sha256(existing_content),
        incoming_sha256=compute_content_sha256(incoming_content),
        is_conflict=True,
        existing_content=existing_content,
        incoming_content=incoming_content,
        diff_lines=(),
        name=name,
        entity_type=entity_type,
    )


class TestApplyResolution:
    def test_keep_existing_returns_existing_id(self):
        memory = _StubMemory()
        result = apply_resolution(memory, _report(), "keep_existing")
        assert result == "ent-1"
        # No save happened.
        assert memory.save_calls == []

    def test_accept_incoming_overwrites_in_place(self):
        memory = _StubMemory()
        result = apply_resolution(memory, _report(), "accept_incoming")
        assert result == "ent-1"
        # save_chain called with entity_id pinned.
        assert len(memory.save_calls) == 1
        call = memory.save_calls[0]
        assert call["method"] == "save_chain"
        assert call["entity_id"] == "ent-1"
        assert call["name"] == "X"

    def test_new_version_saves_under_same_entity_id(self):
        memory = _StubMemory()
        result = apply_resolution(memory, _report(), "new_version")
        assert result == "ent-1"
        call = memory.save_calls[0]
        assert call["entity_id"] == "ent-1"

    def test_agent_skill_dispatches_to_save_agent_skill(self):
        memory = _StubMemory()
        report = _report(entity_type="agent_skill")
        apply_resolution(memory, report, "accept_incoming")
        assert memory.save_calls[0]["method"] == "save_agent_skill"

    def test_memory_card_dispatches_to_save_memory_card(self):
        memory = _StubMemory()
        report = _report(entity_type="memory_card")
        apply_resolution(memory, report, "accept_incoming")
        assert memory.save_calls[0]["method"] == "save_memory_card"

    def test_save_kwargs_forwarded(self):
        memory = _StubMemory()
        apply_resolution(
            memory,
            _report(),
            "accept_incoming",
            save_kwargs={"tags": ["t1"], "author": "carl"},
        )
        call = memory.save_calls[0]
        assert call["tags"] == ["t1"]
        assert call["author"] == "carl"

    def test_explicit_name_kwarg_wins_over_report_name(self):
        memory = _StubMemory()
        apply_resolution(
            memory,
            _report(name="ReportName"),
            "accept_incoming",
            save_kwargs={"name": "Override"},
        )
        assert memory.save_calls[0]["name"] == "Override"

    def test_unknown_resolution_raises(self):
        memory = _StubMemory()
        with pytest.raises(ConflictResolutionError, match="unknown resolution"):
            apply_resolution(memory, _report(), "bogus")  # type: ignore[arg-type]

    def test_unknown_entity_type_raises(self):
        memory = _StubMemory()
        with pytest.raises(ConflictResolutionError, match="unknown entity_type"):
            apply_resolution(
                memory,
                _report(entity_type="weird_kind"),
                "accept_incoming",
            )

    def test_missing_save_method_raises(self):
        class _Bare:
            pass

        with pytest.raises(ConflictResolutionError, match="no 'save_chain'"):
            apply_resolution(_Bare(), _report(), "accept_incoming")

    def test_save_exception_wraps(self):
        memory = _StubMemory(raise_on_save=RuntimeError("save broken"))
        with pytest.raises(ConflictResolutionError, match="save_chain\\(\\) failed.*save broken"):
            apply_resolution(memory, _report(), "accept_incoming")

    def test_keep_existing_skips_unknown_entity_type(self):
        # keep_existing doesn't touch memory, so an unknown
        # entity_type still works.
        memory = _StubMemory()
        result = apply_resolution(
            memory,
            _report(entity_type="anything"),
            "keep_existing",
        )
        assert result == "ent-1"


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


class TestReportShape:
    def test_report_is_frozen(self):
        report = _report()
        with pytest.raises(Exception):
            report.is_conflict = False  # type: ignore[misc]

    def test_diff_lines_is_tuple(self):
        memory = _StubMemory(
            existing={
                "entity_id": "ent-1",
                "content": {"a": 1},
            }
        )
        report = detect_conflict(
            memory,
            name="X",
            entity_type="chain",
            incoming_content={"a": 2},
        )
        assert isinstance(report.diff_lines, tuple)
