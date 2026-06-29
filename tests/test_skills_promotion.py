"""Tests for ``care.skills.promote_skill_to_memory`` (TODO §8 P1).

The promote action is orchestration glue: read SKILL.md from disk,
parse the frontmatter, hash it, and call the existing
:meth:`CareMemory.save_agent_skill`. So the tests exercise:

1. **Locator behaviour** — accepts both a SKILL.md path and a
   folder containing one; tilde expansion; pointing at the wrong
   file type raises a clear error.
2. **Manifest extraction** — name / description / tags /
   allowed_tools propagate from frontmatter; explicit overrides
   win; sensible fallbacks when the frontmatter is sparse.
3. **SHA pinning** — the value passed to memory is a real
   SHA-256 of the SKILL.md bytes (matches what CARE's sandbox
   trust store already pins against).
4. **URI handling** — defaults to ``local://<absolute path>``,
   takes an explicit override for git-checkout sources.
5. **Memory passthrough** — when_to_use / author / entity_id /
   channel forward verbatim so users can re-version an existing
   skill instead of inserting a duplicate.

The memory side uses a tiny ``_StubMemory`` that quacks like
``CareMemory.save_agent_skill`` — no SDK / no HTTP / no fixtures
around either.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from care.skills import SkillPromotionError, promote_skill_to_memory


class _StubMemory:
    """Capture every ``save_agent_skill`` call for assertion."""

    def __init__(self, returned_id: str = "skill-ent-1"):
        self.calls: list[dict[str, Any]] = []
        self.returned_id = returned_id

    def save_agent_skill(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.returned_id


def _write_skill_md(folder: Path, body: str) -> Path:
    """Convenience: create folder + SKILL.md and return file path."""
    folder.mkdir(parents=True, exist_ok=True)
    skill_md = folder / "SKILL.md"
    skill_md.write_text(body, encoding="utf-8")
    return skill_md


# ---------------------------------------------------------------------------
# Locator
# ---------------------------------------------------------------------------


class TestLocator:
    def test_accepts_skill_md_path_directly(self, tmp_path: Path):
        skill_md = _write_skill_md(
            tmp_path / "weather", "---\nname: weather\n---\nbody\n"
        )
        memory = _StubMemory()
        entity_id = promote_skill_to_memory(skill_md, memory)
        assert entity_id == "skill-ent-1"
        assert len(memory.calls) == 1

    def test_accepts_folder_containing_skill_md(self, tmp_path: Path):
        _write_skill_md(tmp_path / "weather", "---\nname: weather\n---\n")
        memory = _StubMemory()
        promote_skill_to_memory(tmp_path / "weather", memory)
        assert memory.calls[0]["name"] == "weather"

    def test_accepts_string_paths(self, tmp_path: Path):
        skill_md = _write_skill_md(tmp_path / "p", "---\nname: p\n---\n")
        memory = _StubMemory()
        promote_skill_to_memory(str(skill_md), memory)
        assert memory.calls[0]["name"] == "p"

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        _write_skill_md(tmp_path / "tilde", "---\nname: tilde\n---\n")
        memory = _StubMemory()
        promote_skill_to_memory("~/tilde", memory)
        assert memory.calls[0]["name"] == "tilde"

    def test_missing_path_raises(self, tmp_path: Path):
        memory = _StubMemory()
        with pytest.raises(SkillPromotionError, match="does not exist"):
            promote_skill_to_memory(tmp_path / "nope", memory)

    def test_wrong_filename_raises(self, tmp_path: Path):
        wrong = tmp_path / "README.md"
        wrong.write_text("---\nname: x\n---\n")
        with pytest.raises(SkillPromotionError, match="expected a SKILL.md"):
            promote_skill_to_memory(wrong, _StubMemory())

    def test_folder_without_skill_md_raises(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(SkillPromotionError, match="no SKILL.md"):
            promote_skill_to_memory(tmp_path / "empty", _StubMemory())


# ---------------------------------------------------------------------------
# Manifest extraction
# ---------------------------------------------------------------------------


class TestManifestExtraction:
    def test_name_description_tags_from_frontmatter(self, tmp_path: Path):
        body = (
            "---\n"
            "name: forecaster\n"
            'description: "Forecasts weather"\n'
            "tags:\n"
            "  - weather\n"
            "  - external\n"
            "allowed-tools:\n"
            "  - WebFetch\n"
            "---\n"
            "# Body\n"
            "Instructions follow.\n"
        )
        _write_skill_md(tmp_path / "forecaster", body)
        memory = _StubMemory()
        promote_skill_to_memory(tmp_path / "forecaster", memory)
        call = memory.calls[0]
        assert call["name"] == "forecaster"
        assert call["description"] == "Forecasts weather"
        assert call["tags"] == ["weather", "external"]
        assert call["allowed_tools"] == ["WebFetch"]
        # Body is forwarded as ``instructions``.
        assert "# Body" in call["instructions"]

    def test_explicit_name_override_wins(self, tmp_path: Path):
        _write_skill_md(
            tmp_path / "x", "---\nname: from_frontmatter\n---\n"
        )
        memory = _StubMemory()
        promote_skill_to_memory(tmp_path / "x", memory, name="OVERRIDDEN")
        assert memory.calls[0]["name"] == "OVERRIDDEN"

    def test_explicit_description_override_wins(self, tmp_path: Path):
        _write_skill_md(
            tmp_path / "x",
            "---\nname: x\ndescription: from-frontmatter\n---\n",
        )
        memory = _StubMemory()
        promote_skill_to_memory(
            tmp_path / "x", memory, description="custom desc"
        )
        assert memory.calls[0]["description"] == "custom desc"

    def test_explicit_tags_override_wins(self, tmp_path: Path):
        _write_skill_md(
            tmp_path / "x",
            "---\nname: x\ntags:\n  - manifest_tag\n---\n",
        )
        memory = _StubMemory()
        promote_skill_to_memory(tmp_path / "x", memory, tags=["a", "b"])
        assert memory.calls[0]["tags"] == ["a", "b"]

    def test_falls_back_to_folder_name(self, tmp_path: Path):
        # No name in frontmatter — folder name "fallback" is used.
        _write_skill_md(tmp_path / "fallback", "---\ndescription: hi\n---\n")
        memory = _StubMemory()
        promote_skill_to_memory(tmp_path / "fallback", memory)
        assert memory.calls[0]["name"] == "fallback"

    def test_description_falls_back_to_first_paragraph(self, tmp_path: Path):
        body = (
            "---\nname: bodydesc\n---\n"
            "# Heading\n"
            "\n"
            "This is the first paragraph.\n"
            "Continues on a second line.\n"
            "\n"
            "Another paragraph here.\n"
        )
        _write_skill_md(tmp_path / "bodydesc", body)
        memory = _StubMemory()
        promote_skill_to_memory(tmp_path / "bodydesc", memory)
        assert memory.calls[0]["description"] == "This is the first paragraph."

    def test_description_falls_back_to_empty(self, tmp_path: Path):
        # No frontmatter description, no body paragraphs (only heading).
        _write_skill_md(tmp_path / "empty", "---\nname: empty\n---\n# Heading only\n")
        memory = _StubMemory()
        promote_skill_to_memory(tmp_path / "empty", memory)
        assert memory.calls[0]["description"] == ""

    def test_no_allowed_tools_yields_empty_list(self, tmp_path: Path):
        _write_skill_md(tmp_path / "x", "---\nname: x\n---\n")
        memory = _StubMemory()
        promote_skill_to_memory(tmp_path / "x", memory)
        assert memory.calls[0]["allowed_tools"] == []


# ---------------------------------------------------------------------------
# SHA-256 pinning
# ---------------------------------------------------------------------------


class TestShaPinning:
    def test_sha_matches_file_contents(self, tmp_path: Path):
        body = "---\nname: hashme\n---\nbody bytes\n"
        skill_md = _write_skill_md(tmp_path / "hash", body)
        memory = _StubMemory()
        promote_skill_to_memory(skill_md, memory)
        expected = hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert memory.calls[0]["sha256"] == expected
        assert len(memory.calls[0]["sha256"]) == 64

    def test_sha_changes_when_body_changes(self, tmp_path: Path):
        skill_md = _write_skill_md(
            tmp_path / "h", "---\nname: h\n---\nv1\n"
        )
        memory = _StubMemory()
        promote_skill_to_memory(skill_md, memory)
        first = memory.calls[0]["sha256"]

        skill_md.write_text("---\nname: h\n---\nv2\n", encoding="utf-8")
        promote_skill_to_memory(skill_md, memory)
        second = memory.calls[1]["sha256"]
        assert first != second


# ---------------------------------------------------------------------------
# URI handling
# ---------------------------------------------------------------------------


class TestUriHandling:
    def test_default_uri_is_local_with_absolute_path(self, tmp_path: Path):
        skill_md = _write_skill_md(tmp_path / "u", "---\nname: u\n---\n")
        memory = _StubMemory()
        promote_skill_to_memory(skill_md, memory)
        uri = memory.calls[0]["skill_uri"]
        assert uri.startswith("local://")
        # Absolute path, ends with SKILL.md.
        assert uri.endswith("/SKILL.md")

    def test_explicit_source_uri_wins(self, tmp_path: Path):
        skill_md = _write_skill_md(tmp_path / "u", "---\nname: u\n---\n")
        memory = _StubMemory()
        promote_skill_to_memory(
            skill_md,
            memory,
            source_uri="github://acme/skills/u@v1",
        )
        assert memory.calls[0]["skill_uri"] == "github://acme/skills/u@v1"


# ---------------------------------------------------------------------------
# Memory passthrough
# ---------------------------------------------------------------------------


class TestMemoryPassthrough:
    def test_when_to_use_author_entity_id_channel_forwarded(
        self, tmp_path: Path
    ):
        skill_md = _write_skill_md(tmp_path / "p", "---\nname: p\n---\n")
        memory = _StubMemory(returned_id="existing-id")
        result = promote_skill_to_memory(
            skill_md,
            memory,
            when_to_use="forecasting tasks",
            author="weather-team",
            entity_id="existing-id",
            channel="stable",
        )
        assert result == "existing-id"
        call = memory.calls[0]
        assert call["when_to_use"] == "forecasting tasks"
        assert call["author"] == "weather-team"
        assert call["entity_id"] == "existing-id"
        assert call["channel"] == "stable"

    def test_returns_memory_assigned_id(self, tmp_path: Path):
        _write_skill_md(tmp_path / "r", "---\nname: r\n---\n")
        memory = _StubMemory(returned_id="fresh-entity-42")
        assert (
            promote_skill_to_memory(tmp_path / "r", memory)
            == "fresh-entity-42"
        )

    def test_re_promote_with_entity_id_creates_new_version(
        self, tmp_path: Path
    ):
        """Idempotent re-ingestion: same entity_id is forwarded so
        Memory creates a new version of the existing skill rather
        than a duplicate row."""
        skill_md = _write_skill_md(
            tmp_path / "v", "---\nname: v\n---\nv1\n"
        )
        memory = _StubMemory()
        first = promote_skill_to_memory(skill_md, memory)
        # Author updates the skill body — promote again pinning the
        # same entity_id.
        skill_md.write_text("---\nname: v\n---\nv2\n", encoding="utf-8")
        second = promote_skill_to_memory(
            skill_md, memory, entity_id=first
        )
        assert second == "skill-ent-1"  # stub echoes regardless
        assert memory.calls[0]["entity_id"] is None
        assert memory.calls[1]["entity_id"] == first
        # And the SHA actually changed across versions.
        assert memory.calls[0]["sha256"] != memory.calls[1]["sha256"]
