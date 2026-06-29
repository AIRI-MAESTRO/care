"""Capability catalog data model (TODO §8 P1).

CARE's `care catalog` CLI subcommand (and the future
`CatalogScreen`) needs one place that knows what capabilities are
available to a chain author:

* **AgentSkills** installed on disk — typically
  ``~/.agents/skills/*/SKILL.md`` and
  ``./.claude/skills/*/SKILL.md``.
* **MCP servers** configured via
  ``~/.config/care/mcp_servers.toml``.
* **Tools** — Python callables under
  ``~/.config/care/tools/*.py``.
* **Capability memory cards** — entries in GigaEvo Memory tagged
  ``capability`` (CARE writes one of these for every promoted
  skill, MCP server, or tool template, TODO §8.2).

This module owns the data shape + the file/TOML discovery. The
CLI and TUI screens render :class:`CapabilityCatalog.entries`
without knowing how they were sourced. Memory-side lookup takes
a :class:`care.CareMemory` so we don't reimplement search; the
helper falls back to "Memory unreachable → skip memory_card
entries" so the catalog still renders without a running server.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

EntryKind = Literal["agent_skill", "mcp_server", "tool", "memory_card"]
"""The four kinds the catalog renders. New kinds bump this Literal."""


@dataclass(frozen=True)
class CapabilityCatalogEntry:
    """One row in the catalog.

    Frozen so screens + the CLI's `--json` output can pass entries
    around without defensive copies.

    Fields:
        kind: Which collection this came from.
        name: Display name. For skills/cards: the manifest's `name`.
            For MCP servers: the TOML section key. For tools: the
            file stem.
        source: Where CARE found it — file path string, MCP server
            command, or `"memory://<entity_id>"` for memory cards.
        summary: One-liner the catalog screen renders next to the
            name. Falls back to ``""`` when no description is
            available.
        tags: Tags / categories the entry self-declares. Empty
            tuple when none.
        metadata: Free-form per-kind extras (manifest dict for
            skills, transport for MCP servers, etc.). Kept so the
            future "Promote to Memory" action has the data it
            needs without a re-scan.
    """

    kind: EntryKind
    name: str
    source: str
    summary: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityCatalog:
    """Aggregate of every catalog entry plus discovery diagnostics."""

    entries: tuple[CapabilityCatalogEntry, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    """Non-fatal discovery errors (a malformed SKILL.md, an
    unreadable mcp_servers.toml, a Memory call that 503'd). The CLI
    prints these under a "warnings" section; the TUI renders them
    in a footer panel. Catalog is best-effort: one broken file
    shouldn't make the whole catalog unavailable."""

    @property
    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def by_kind(self, kind: EntryKind) -> tuple[CapabilityCatalogEntry, ...]:
        return tuple(e for e in self.entries if e.kind == kind)

    def by_tag(self, tag: str) -> tuple[CapabilityCatalogEntry, ...]:
        return tuple(e for e in self.entries if tag in e.tags)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def build_catalog(
    *,
    skills_paths: Iterable[Path | str] | None = None,
    mcp_config_path: Path | str | None = None,
    tools_path: Path | str | None = None,
    memory: Any = None,
    memory_card_tag: str = "capability",
    memory_top_k: int = 50,
) -> CapabilityCatalog:
    """Scan every configured source and return a catalog.

    Args:
        skills_paths: Directories to scan for ``*/SKILL.md`` files.
            Defaults to the empty list; CARE's CLI typically passes
            ``[~/.agents/skills, ./.claude/skills]``.
        mcp_config_path: Path to an ``mcp_servers.toml`` file.
            ``None`` skips MCP discovery.
        tools_path: Directory of Python tool files (e.g.
            ``~/.config/care/tools``). Files are listed by name —
            the catalog doesn't import them.
        memory: Optional :class:`care.CareMemory` (or anything
            exposing ``.search(query, *, entity_type, top_k)``).
            When supplied, memory cards tagged
            ``memory_card_tag`` are fetched and added.
        memory_card_tag: The tag we filter capability cards by.
        memory_top_k: Hard cap on the number of cards we fetch.

    Returns:
        :class:`CapabilityCatalog`. Discovery errors land on
        ``errors`` and never raise — a broken SKILL.md doesn't
        kill the rest of the scan.
    """
    entries: list[CapabilityCatalogEntry] = []
    errors: list[str] = []

    for raw in skills_paths or ():
        path = _expand(raw)
        if not path.exists():
            continue
        skill_entries, skill_errors = _scan_skill_dir(path)
        entries.extend(skill_entries)
        errors.extend(skill_errors)

    if mcp_config_path is not None:
        mcp_path = _expand(mcp_config_path)
        if mcp_path.exists():
            mcp_entries, mcp_errors = _scan_mcp_config(mcp_path)
            entries.extend(mcp_entries)
            errors.extend(mcp_errors)

    if tools_path is not None:
        tp = _expand(tools_path)
        if tp.exists():
            tool_entries, tool_errors = _scan_tools_dir(tp)
            entries.extend(tool_entries)
            errors.extend(tool_errors)

    if memory is not None:
        card_entries, card_errors = _fetch_memory_cards(
            memory, tag=memory_card_tag, top_k=memory_top_k
        )
        entries.extend(card_entries)
        errors.extend(card_errors)

    # Deterministic ordering — kind first (alphabetical), then
    # name. Keeps the CLI's --json output stable across runs.
    entries.sort(key=lambda e: (e.kind, e.name.lower()))

    return CapabilityCatalog(
        entries=tuple(entries),
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# Per-source scanners
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL
)
"""Match a SKILL.md / Markdown frontmatter block. Group 1 is the
YAML-ish body; group 2 is everything after."""


def _scan_skill_dir(root: Path) -> tuple[list[CapabilityCatalogEntry], list[str]]:
    """Walk ``root`` looking for ``*/SKILL.md`` files. Tolerates
    nested layouts — the SKILL.md can sit one or more levels deep."""
    entries: list[CapabilityCatalogEntry] = []
    errors: list[str] = []
    if not root.is_dir():
        return entries, [f"skills path is not a directory: {root}"]

    for skill_md in sorted(root.rglob("SKILL.md")):
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"could not read {skill_md}: {exc}")
            continue
        manifest, body = _parse_skill_md(text)
        name = str(manifest.get("name") or skill_md.parent.name)
        summary = str(manifest.get("description") or "").strip().splitlines()[:1]
        tags_raw = manifest.get("tags") or ()
        tags = tuple(str(t) for t in tags_raw) if isinstance(tags_raw, (list, tuple, set)) else ()
        allowed_tools = manifest.get("allowed-tools") or ()
        if isinstance(allowed_tools, (list, tuple, set)):
            allowed_tools_norm = list(allowed_tools)
        elif isinstance(allowed_tools, str):
            allowed_tools_norm = [allowed_tools]
        else:
            allowed_tools_norm = []
        entries.append(
            CapabilityCatalogEntry(
                kind="agent_skill",
                name=name,
                source=str(skill_md),
                summary=(summary[0] if summary else ""),
                tags=tags,
                metadata={
                    "manifest": manifest,
                    "instructions_preview": body[:200],
                    "allowed_tools": allowed_tools_norm,
                },
            )
        )
    return entries, errors


def _parse_skill_md(text: str) -> tuple[dict[str, Any], str]:
    """Tiny YAML-ish frontmatter parser.

    Handles the SKILL.md shapes CARE writes today: top-level
    ``key: value`` lines + ``key:`` followed by a ``- item`` list.
    Anything more complex (nested dicts, multi-line strings) is
    a TODO — the catalog flags those as ``errors`` rather than
    crashing.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text
    body = match.group(2)
    fm_text = match.group(1)
    manifest: dict[str, Any] = {}
    current_list_key: str | None = None
    for line in fm_text.splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        if line.startswith(("  -", "\t-")) and current_list_key is not None:
            item = stripped.lstrip().lstrip("-").strip()
            if item:
                manifest.setdefault(current_list_key, []).append(item)
            continue
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if not value:
                manifest[key] = []
                current_list_key = key
            else:
                # Strip simple surrounding quotes.
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                manifest[key] = value
                current_list_key = None
    return manifest, body


def _scan_mcp_config(
    path: Path,
) -> tuple[list[CapabilityCatalogEntry], list[str]]:
    """Parse a ``mcp_servers.toml`` file.

    Expected shape::

        [servers.weather]
        command = "node"
        args = ["/opt/mcp/weather.js"]
        description = "Fetches forecasts"
        tags = ["weather"]
    """
    entries: list[CapabilityCatalogEntry] = []
    errors: list[str] = []
    try:
        with path.open("rb") as fp:
            data = tomllib.load(fp)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"could not parse {path}: {exc}")
        return entries, errors

    servers = data.get("servers", {})
    if not isinstance(servers, dict):
        errors.append(f"{path}: [servers] must be a table")
        return entries, errors

    for name, spec in sorted(servers.items()):
        if not isinstance(spec, dict):
            errors.append(f"{path}: servers.{name} must be a table")
            continue
        tags_raw = spec.get("tags") or ()
        tags = tuple(str(t) for t in tags_raw) if isinstance(tags_raw, (list, tuple, set)) else ()
        command = spec.get("command", "")
        args_value = spec.get("args", [])
        args_str = " ".join(str(a) for a in args_value) if isinstance(args_value, list) else ""
        source = f"{command} {args_str}".strip() if command else str(path)
        entries.append(
            CapabilityCatalogEntry(
                kind="mcp_server",
                name=str(name),
                source=source,
                summary=str(spec.get("description") or "").strip(),
                tags=tags,
                metadata=dict(spec),
            )
        )
    return entries, errors


def _scan_tools_dir(
    root: Path,
) -> tuple[list[CapabilityCatalogEntry], list[str]]:
    """List ``*.py`` files. Doesn't import them — that's the
    `context.register_tools_from_path` job at runtime; here we
    just enumerate so the CatalogScreen can show "you have 7
    tool files installed"."""
    entries: list[CapabilityCatalogEntry] = []
    errors: list[str] = []
    if not root.is_dir():
        return entries, [f"tools path is not a directory: {root}"]

    for tool_file in sorted(root.glob("*.py")):
        if tool_file.name.startswith("_"):
            # Skip __init__.py / __pycache__ / private helpers.
            continue
        try:
            head = tool_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            errors.append(f"could not read {tool_file}: {exc}")
            continue
        # Best-effort description: first docstring or `# Description: ...`
        summary = ""
        for line in head[:30]:
            stripped = line.strip()
            if stripped.startswith('"""') and stripped != '"""':
                summary = stripped.strip('"').strip()
                break
            if stripped.startswith("# Description:"):
                summary = stripped[len("# Description:"):].strip()
                break
        entries.append(
            CapabilityCatalogEntry(
                kind="tool",
                name=tool_file.stem,
                source=str(tool_file),
                summary=summary,
                tags=(),
                metadata={"line_count": len(head)},
            )
        )
    return entries, errors


def _fetch_memory_cards(
    memory: Any,
    *,
    tag: str,
    top_k: int,
) -> tuple[list[CapabilityCatalogEntry], list[str]]:
    entries: list[CapabilityCatalogEntry] = []
    errors: list[str] = []
    try:
        hits = memory.search(
            tag, entity_type="memory_card", top_k=top_k
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"memory_card search failed: {exc}")
        return entries, errors

    for hit in hits:
        if not isinstance(hit, dict):
            continue
        entity_id = str(hit.get("entity_id") or "")
        name = str(hit.get("name") or hit.get("meta", {}).get("name") or "card")
        meta = hit.get("meta") or {}
        tags_raw = meta.get("tags") if isinstance(meta, dict) else None
        tags = tuple(str(t) for t in tags_raw) if isinstance(tags_raw, (list, tuple, set)) else ()
        summary = str(hit.get("description") or "").strip()
        entries.append(
            CapabilityCatalogEntry(
                kind="memory_card",
                name=name,
                source=f"memory://{entity_id}" if entity_id else "memory://",
                summary=summary,
                tags=tags,
                metadata=dict(hit),
            )
        )
    return entries, errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand(raw: Path | str) -> Path:
    """Tilde + env-var expansion for user-supplied paths."""
    return Path(raw).expanduser()


__all__ = [
    "CapabilityCatalog",
    "CapabilityCatalogEntry",
    "EntryKind",
    "build_catalog",
]
