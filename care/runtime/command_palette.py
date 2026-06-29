"""Command-palette data layer (TODO §1 P3).

CARE binds ``Ctrl+P`` to a fzf-style palette that searches across
chains, agent_skills, and built-in commands. The palette is the
keyboard-first navigation primitive: type a few characters, see
ranked candidates, press Enter to execute.

The Textual modal is gated on TODO §1 P0 multi-screen workflow,
but the index + scorer + async loader land now so the modal is
a thin renderer + key handler.

What this module provides:

* :class:`PaletteEntryKind` literal pinning what the palette
  indexes (chains / agent_skills / built-in commands).
* :class:`PaletteEntry` — frozen indexed row (id, kind, label,
  description, tags, score, optional command action).
* :class:`PaletteIndex` — frozen aggregate over every entry.
* :class:`Command` — frozen built-in command descriptor
  (label, action_id, optional shortcut hint, kinds it applies to).
* :func:`default_commands` — canonical built-in command list
  (Create new agent / Open settings / Quit / …).
* :func:`fuzzy_score` — pure scorer. Subsequence match with
  bonuses for prefix / word-start / consecutive-character /
  case-match positions. Returns 0.0 for non-matches.
* :func:`search_palette` — pure ranker. Filters by kind,
  scores every entry against the query, returns the top-k
  sorted by score descending then label ascending.
* :func:`fetch_palette_index` — async aggregator that hits
  Memory for chains + agent_skills and bundles in the
  registered commands.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Optional

from care.runtime.i18n import t


PaletteEntryKind = Literal["chain", "agent_skill", "command"]
"""What the palette indexes. ``chain`` + ``agent_skill`` come
from Memory; ``command`` is a CARE-side built-in action (e.g.
"Create new agent"). Step + memory_card kinds are excluded by
design — the palette is a navigation primitive over user-
facing entities, not internal building blocks."""


CommandActionId = Literal[
    "create_new_agent",
    "open_chat",
    "open_artifacts",
    "open_settings",
    "show_library",
    "open_evolution",
    "show_help",
    "open_catalog",
    "open_marketplace",
    "import_bundle",
    "export_library",
    "quit",
]
"""Canonical built-in command identifiers. The screen's
keyboard handler dispatches off these — adding a new command
requires extending this literal.

The ``open_*`` family (``open_chat`` / ``open_artifacts`` /
``open_evolution``) maps to the four destination screens the
chat banner advertises — surfacing them in the palette gives
keyboard users a one-keystroke jump that mirrors the slash-
command discovery surface."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PaletteError(RuntimeError):
    """Raised when palette aggregation fails — unreachable
    Memory, timeout, missing SDK method. The modal catches this
    and shows a friendly toast."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Command:
    """One built-in command descriptor.

    Frozen so the registry flows through Textual messages without
    defensive copies. ``shortcut_hint`` is a human-readable
    label the modal renders alongside the entry (e.g. ``"Ctrl+N"``);
    the actual key binding lives on the wrapping screen.
    """

    action_id: CommandActionId
    label: str
    description: str = ""
    shortcut_hint: str = ""
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class PaletteEntry:
    """One indexed palette row.

    Frozen so the modal can hold snapshots safely. ``score``
    is populated by :func:`search_palette` per query; the
    indexed instance has ``score=0.0`` and the search returns
    a new frozen copy with the score set (matches the
    `dataclasses.replace` pattern other CARE modules use).
    """

    entry_id: str
    kind: PaletteEntryKind
    label: str
    description: str = ""
    tags: tuple[str, ...] = ()
    command_action: Optional[CommandActionId] = None
    score: float = 0.0

    @property
    def is_command(self) -> bool:
        return self.kind == "command"

    @property
    def search_haystack(self) -> str:
        """Concatenated lowercase haystack the scorer matches
        against. Computed lazily by `fuzzy_score`."""
        parts = [self.label]
        if self.description:
            parts.append(self.description)
        if self.tags:
            parts.extend(self.tags)
        return " · ".join(parts).casefold()


@dataclass(frozen=True)
class PaletteIndex:
    """Frozen aggregate of every entry visible to the palette.

    The modal holds one of these and re-runs :func:`search_palette`
    on every keystroke. Cheap because the index is small and the
    scorer is linear in haystack length.
    """

    entries: tuple[PaletteEntry, ...] = ()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def filter_kinds(
        self,
        kinds: Iterable[PaletteEntryKind],
    ) -> "PaletteIndex":
        """Return a new index containing only entries whose
        ``kind`` is in ``kinds``. The modal uses this for the
        "type @ to restrict to chains, # for skills" style
        prefix shortcuts."""
        kinds_set = frozenset(kinds)
        return PaletteIndex(
            entries=tuple(e for e in self.entries if e.kind in kinds_set),
        )


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------


def default_commands() -> tuple[Command, ...]:
    """Canonical built-in command list. Returned as a tuple so
    callers can extend it without mutating the module state.

    Built fresh on every call (rather than a module-level constant)
    so the labels + descriptions resolve :func:`t` in the active UI
    language at access time — a language change shows up the next time
    the palette opens.
    """
    return (
        # ----- Screens group -----------------------------------------------
        # The five destinations the welcome banner names — kept at the top of
        # the palette so an empty query surfaces the app's map of itself.
        # Each label uses the "Open <Screen>" pattern so a `op` typeahead
        # lists them as a group; the screen-name suffix means `lib` / `evo` /
        # `art` / `set` / `chat` typeaheads each land their target row.
        Command(
            action_id="open_chat",
            label=t("commandPalette.cmd.openChat.label"),
            description=t("commandPalette.cmd.openChat.description"),
            shortcut_hint="Esc Esc",
            keywords=("chat", "home", "prompt", "back"),
        ),
        Command(
            action_id="open_artifacts",
            label=t("commandPalette.cmd.openArtifacts.label"),
            description=t("commandPalette.cmd.openArtifacts.description"),
            keywords=("artifacts", "session", "chains", "current"),
        ),
        Command(
            action_id="show_library",
            label=t("commandPalette.cmd.openLibrary.label"),
            description=t("commandPalette.cmd.openLibrary.description"),
            shortcut_hint="Ctrl+L",
            keywords=("library", "home", "agents", "list", "saved"),
        ),
        Command(
            action_id="open_evolution",
            label=t("commandPalette.cmd.openEvolution.label"),
            description=t("commandPalette.cmd.openEvolution.description"),
            keywords=("evolution", "runs", "dashboard", "pareto", "platform"),
        ),
        Command(
            action_id="open_settings",
            label=t("commandPalette.cmd.openSettings.label"),
            description=t("commandPalette.cmd.openSettings.description"),
            shortcut_hint="Ctrl+,",
            keywords=("settings", "config", "preferences"),
        ),
        # ----- Build / share group -----------------------------------------
        Command(
            action_id="create_new_agent",
            label=t("commandPalette.cmd.createNewAgent.label"),
            description=t("commandPalette.cmd.createNewAgent.description"),
            shortcut_hint="Ctrl+N",
            keywords=("new", "create", "generate", "agent"),
        ),
        Command(
            action_id="import_bundle",
            label=t("commandPalette.cmd.importBundle.label"),
            description=t("commandPalette.cmd.importBundle.description"),
            keywords=("import", "bundle", "tarball", "restore"),
        ),
        Command(
            action_id="export_library",
            label=t("commandPalette.cmd.exportLibrary.label"),
            description=t("commandPalette.cmd.exportLibrary.description"),
            keywords=("export", "bundle", "tarball", "share"),
        ),
        Command(
            action_id="show_help",
            label=t("commandPalette.cmd.help.label"),
            description=t("commandPalette.cmd.help.description"),
            shortcut_hint="?",
            keywords=("help", "docs", "manual", "keys"),
        ),
        Command(
            action_id="open_catalog",
            label=t("commandPalette.cmd.catalog.label"),
            description=t("commandPalette.cmd.catalog.description"),
            keywords=(
                "catalog", "capabilities", "skills", "mcp", "tools",
                "browse",
            ),
        ),
        Command(
            action_id="open_marketplace",
            label=t("commandPalette.cmd.marketplace.label"),
            description=t("commandPalette.cmd.marketplace.description"),
            keywords=(
                "marketplace", "search", "skills", "install", "share",
            ),
        ),
        Command(
            action_id="quit",
            label=t("commandPalette.cmd.quit.label"),
            description=t("commandPalette.cmd.quit.description"),
            shortcut_hint="Ctrl+Q",
            keywords=("quit", "exit", "close"),
        ),
    )


def commands_to_entries(commands: Iterable[Command]) -> tuple[PaletteEntry, ...]:
    """Project a list of :class:`Command` into palette entries.

    Commands sort to the top when the user types nothing (the
    scorer ties broken by label ascending), giving the empty-
    query case a useful default — first-time users see the
    built-in actions immediately.
    """
    return tuple(
        PaletteEntry(
            entry_id=f"command:{c.action_id}",
            kind="command",
            label=c.label,
            description=c.description,
            tags=c.keywords,
            command_action=c.action_id,
        )
        for c in commands
    )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def fuzzy_score(query: str, candidate: str) -> float:
    """Subsequence-match score between ``query`` and ``candidate``.

    Returns 0.0 when ``query`` characters don't appear in order
    inside ``candidate``. Higher scores mean a better match.
    Scoring rules (all bonuses additive, scaled to fit roughly
    in [0, 1] for typical inputs):

    * **Subsequence match** — every query character must appear
      in ``candidate`` in order. Missing chars → 0.0.
    * **Prefix bonus** — extra 0.5 when ``candidate`` starts
      with ``query`` (case-insensitive). Strong signal.
    * **Exact substring bonus** — extra 0.3 when ``query``
      appears verbatim as a substring anywhere.
    * **Word-start bonus** — small bonus (0.1) when a match
      lands right after a separator (space, dot, slash, hyphen,
      underscore).
    * **Consecutive bonus** — 0.05 per consecutive-character
      run beyond the first character of a run.
    * **Length penalty** — divide by ``len(candidate)`` so
      shorter candidates score slightly higher for the same
      match.

    Empty query → 0.0 (the caller can treat 0.0 as "no
    keystrokes yet" and surface the default ordering).
    """
    if not query:
        return 0.0
    if not candidate:
        return 0.0
    q = query.casefold()
    c = candidate.casefold()

    # 1. Subsequence walk.
    qi = 0
    last_pos = -2
    consecutive_bonus = 0.0
    word_start_bonus = 0.0
    separators = " ·.-_/\n\t,:"
    for ci, ch in enumerate(c):
        if qi >= len(q):
            break
        if ch == q[qi]:
            qi += 1
            if ci == last_pos + 1:
                consecutive_bonus += 0.05
            if ci == 0 or c[ci - 1] in separators:
                word_start_bonus += 0.1
            last_pos = ci
    if qi < len(q):
        return 0.0  # Not a subsequence.

    base = 1.0 / max(1, len(c))

    prefix_bonus = 0.5 if c.startswith(q) else 0.0
    substring_bonus = 0.3 if q in c else 0.0

    return base + prefix_bonus + substring_bonus + word_start_bonus + consecutive_bonus


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_palette(
    index: PaletteIndex,
    query: str,
    *,
    top_k: int = 20,
    kinds: Optional[Iterable[PaletteEntryKind]] = None,
) -> tuple[PaletteEntry, ...]:
    """Rank entries against ``query``.

    Empty query short-circuits to a kind-filtered prefix of
    the index (commands first, then the rest in insertion
    order). Non-empty query runs :func:`fuzzy_score` over every
    entry's haystack and returns the top-k by descending
    score (ties broken by label ascending).

    Args:
        index: Pre-loaded :class:`PaletteIndex`.
        query: Search input.
        top_k: Maximum results to return.
        kinds: Restrict to these palette kinds. ``None`` means
            "every kind".

    Returns:
        Tuple of :class:`PaletteEntry` (each with a populated
        ``score`` field on non-empty queries).
    """
    if kinds is not None:
        kinds_set = frozenset(kinds)
        candidates = tuple(e for e in index.entries if e.kind in kinds_set)
    else:
        candidates = index.entries

    if not query.strip():
        # Default ordering: commands first, then entities in
        # insertion order. Capped at top_k.
        commands = tuple(e for e in candidates if e.is_command)
        rest = tuple(e for e in candidates if not e.is_command)
        return (commands + rest)[: max(0, top_k)]

    scored: list[PaletteEntry] = []
    for entry in candidates:
        score = fuzzy_score(query, entry.search_haystack)
        if score > 0:
            scored.append(replace(entry, score=score))

    scored.sort(key=lambda e: (-e.score, e.label.casefold()))
    return tuple(scored[: max(0, top_k)])


# ---------------------------------------------------------------------------
# Async aggregator
# ---------------------------------------------------------------------------


_DEFAULT_LIMIT = 200


async def fetch_palette_index(
    memory: Any,
    *,
    namespace: Optional[str] = None,
    channel: str = "latest",
    limit: int = _DEFAULT_LIMIT,
    commands: Optional[Iterable[Command]] = None,
    timeout: float = 10.0,
) -> PaletteIndex:
    """Build a :class:`PaletteIndex` from Memory + the built-in
    command list.

    Fans out two concurrent SDK calls
    (``client.list_chains`` + ``client.list_agent_skills``)
    so the wall-clock floor is one timeout, not two. The
    built-in commands are projected synchronously and prepended
    so the empty-query case has useful defaults.

    Args:
        memory: A `CareMemory`-like facade with
            ``client.list_chains`` and ``client.list_agent_skills``.
            Missing methods are tolerated — the palette just
            skips that kind's entries.
        namespace: Optional namespace filter.
        channel: Memory channel.
        limit: Per-kind fetch cap (Memory caps at 200).
        commands: Override the built-in command list.
            ``None`` uses :func:`default_commands`.
        timeout: Wall-clock deadline for the fan-out.

    Returns:
        :class:`PaletteIndex` ready for :func:`search_palette`.

    Raises:
        PaletteError: Memory unreachable or timed out.
    """
    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    if client is None:
        raise PaletteError(
            "memory facade does not expose a `.client` attribute"
        )

    list_chains = getattr(client, "list_chains", None)
    list_skills = getattr(client, "list_agent_skills", None)
    capped = max(1, min(limit, 200))

    async def _fetch(fn: Any) -> list[Any]:
        if not callable(fn):
            return []
        try:
            rows = await asyncio.to_thread(
                fn,
                limit=capped,
                channel=channel,
                namespace=namespace,
            )
        except Exception:  # noqa: BLE001
            return []
        if isinstance(rows, list):
            return rows
        if isinstance(rows, tuple):
            return list(rows)
        return []

    start = time.monotonic()
    try:
        chain_rows, skill_rows = await asyncio.wait_for(
            asyncio.gather(_fetch(list_chains), _fetch(list_skills)),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        latency = (time.monotonic() - start) * 1000
        raise PaletteError(
            f"palette fetch timed out after {timeout:.1f}s ({latency:.0f}ms elapsed)"
        ) from exc

    entries: list[PaletteEntry] = list(
        commands_to_entries(commands if commands is not None else default_commands())
    )
    entries.extend(_project_chains(chain_rows))
    entries.extend(_project_skills(skill_rows))
    return PaletteIndex(entries=tuple(entries))


def _project_chains(rows: Iterable[Any]) -> list[PaletteEntry]:
    out: list[PaletteEntry] = []
    for row in rows:
        entity_id = str(_read(row, "entity_id") or "")
        if not entity_id:
            continue
        label = (
            _read_str(row, "display_name")
            or _read_meta_str(row, "name")
            or entity_id[:12]
        )
        out.append(
            PaletteEntry(
                entry_id=entity_id,
                kind="chain",
                label=label,
                description=_read_str(row, "description"),
                tags=_read_tags(row),
            )
        )
    return out


def _project_skills(rows: Iterable[Any]) -> list[PaletteEntry]:
    out: list[PaletteEntry] = []
    for row in rows:
        entity_id = str(_read(row, "entity_id") or "")
        if not entity_id:
            continue
        content = _read(row, "content") or {}
        name = ""
        description = ""
        if isinstance(content, dict):
            name = str(content.get("name") or "")
            description = str(content.get("description") or "")
        label = name or _read_str(row, "display_name") or entity_id[:12]
        out.append(
            PaletteEntry(
                entry_id=entity_id,
                kind="agent_skill",
                label=label,
                description=description,
                tags=_read_tags(row),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _read_str(obj: Any, name: str) -> str:
    value = _read(obj, name)
    return value if isinstance(value, str) else ""


def _read_meta_str(obj: Any, name: str) -> str:
    meta = _read(obj, "meta")
    if isinstance(meta, dict):
        value = meta.get(name)
        return value if isinstance(value, str) else ""
    return ""


def _read_tags(obj: Any) -> tuple[str, ...]:
    meta = _read(obj, "meta")
    if isinstance(meta, dict):
        tags = meta.get("tags") or []
        if isinstance(tags, (list, tuple)):
            return tuple(str(t) for t in tags if isinstance(t, str))
    return ()


# Re-export the unused field marker for future dataclass
# extensions.
_ = field


__all__ = [
    "Command",
    "CommandActionId",
    "PaletteEntry",
    "PaletteEntryKind",
    "PaletteError",
    "PaletteIndex",
    "commands_to_entries",
    "default_commands",
    "fetch_palette_index",
    "fuzzy_score",
    "search_palette",
]
