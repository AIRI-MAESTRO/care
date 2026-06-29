"""Skill-promotion helpers (TODO §8 P1).

CARE's `CapabilityCatalog` already discovers locally-installed
AgentSkills under ``~/.agents/skills/`` and friends. The next step
on the M3 capability story is letting the user **promote** one of
those local skills into GigaEvo Memory so other people (or other
machines) can rediscover it via ``find_capability_matches``.

The promote action is:

1. Find the SKILL.md file (the user can hand us either the file
   itself or the enclosing folder — the SKILL.md convention is
   ``<skill-name>/SKILL.md``).
2. Parse its YAML-ish frontmatter (reusing the same parser the
   catalog scanner uses, so what shows up in the CLI/screen and
   what lands in Memory stays consistent).
3. Compute the SHA-256 of SKILL.md — Memory needs this for the
   trust-pinning workflow CARE already uses on the sandbox side.
4. Build a ``local://`` URI (or take an explicit override when the
   skill came from a git checkout the user can point us at), and
   call :meth:`CareMemory.save_agent_skill`.

This sits in its own tiny module so the catalog stays
discovery-only and the Memory facade stays I/O-only — the action
is the orchestration glue between them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from care.catalog import _parse_skill_md


class SkillPromotionError(RuntimeError):
    """Raised when a SKILL.md can't be located, read, or normalised
    into an :class:`AgentSkillSpec`-compatible payload."""


def promote_skill_to_memory(
    skill_path: Path | str,
    memory: Any,
    *,
    source_uri: str | None = None,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    when_to_use: str | None = None,
    author: str | None = None,
    entity_id: str | None = None,
    channel: str = "latest",
) -> str:
    """Upload a locally-installed SKILL.md to GigaEvo Memory.

    Args:
        skill_path: Either the SKILL.md file directly or the
            folder containing it. Tilde-expanded.
        memory: Anything exposing ``CareMemory.save_agent_skill`` —
            typically a :class:`care.CareMemory`, but tests pass in
            a stub.
        source_uri: Canonical source URL. Defaults to
            ``local://<absolute-skill-md-path>``. Pass an explicit
            ``github://owner/repo[/subpath][@ref]`` when the skill
            was checked out from git so other users can re-fetch it.
        name: Overrides the manifest's ``name``.
        description: Overrides the manifest's ``description`` (or
            its first-line summary).
        tags: Overrides the manifest's ``tags``.
        when_to_use, author, entity_id, channel: Passed through to
            :meth:`CareMemory.save_agent_skill`. Use ``entity_id``
            to overwrite (create a new version of) an existing
            skill entity instead of inserting a fresh one.

    Returns:
        The ``entity_id`` Memory assigned (or echoed back).

    Raises:
        SkillPromotionError: If the SKILL.md can't be found / read,
            or the manifest is missing the fields Memory requires
            (``name``).
    """
    skill_md = _locate_skill_md(skill_path)
    try:
        raw_text = skill_md.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillPromotionError(
            f"could not read SKILL.md at {skill_md}: {exc}"
        ) from exc

    manifest, body = _parse_skill_md(raw_text)
    resolved_name = name or str(manifest.get("name") or "").strip()
    if not resolved_name:
        # Fall back to the parent directory — handy when a vendor
        # ships SKILL.md without a `name:` line and just relies on
        # the folder name as the identifier.
        resolved_name = skill_md.parent.name
    if not resolved_name:
        raise SkillPromotionError(
            f"SKILL.md at {skill_md} has no `name` and no parent "
            "directory to fall back on — pass `name=` explicitly."
        )

    sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    uri = source_uri or f"local://{skill_md.resolve()}"

    resolved_description = description
    if resolved_description is None:
        manifest_desc = str(manifest.get("description") or "").strip()
        resolved_description = manifest_desc or _first_paragraph(body)

    allowed_tools = _coerce_string_list(manifest.get("allowed-tools"))
    manifest_tags = _coerce_string_list(manifest.get("tags"))

    return memory.save_agent_skill(
        skill_uri=uri,
        manifest=manifest,
        sha256=sha,
        instructions=body,
        allowed_tools=allowed_tools,
        name=resolved_name,
        description=resolved_description,
        tags=list(tags) if tags is not None else manifest_tags,
        when_to_use=when_to_use,
        author=author,
        entity_id=entity_id,
        channel=channel,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _locate_skill_md(raw: Path | str) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        if path.name != "SKILL.md":
            raise SkillPromotionError(
                f"expected a SKILL.md file, got {path}"
            )
        return path
    if path.is_dir():
        candidate = path / "SKILL.md"
        if not candidate.is_file():
            raise SkillPromotionError(
                f"no SKILL.md found in {path}"
            )
        return candidate
    raise SkillPromotionError(f"skill path does not exist: {path}")


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []


def _first_paragraph(body: str) -> str:
    """Pick the first non-heading paragraph from the SKILL.md body
    as a fallback description. Keeps the Memory entity readable
    when the author forgot the frontmatter."""
    for paragraph in body.split("\n\n"):
        stripped = paragraph.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        return stripped.splitlines()[0].strip()
    return ""


__all__ = [
    "SkillPromotionError",
    "promote_skill_to_memory",
]
