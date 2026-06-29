"""Tests for the command-palette data layer (TODO §1 P3).

The Textual modal is gated on §1 P0; this suite pins the
contract the modal binds to.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError

import pytest

from care.runtime.command_palette import (
    Command,
    PaletteEntry,
    PaletteError,
    PaletteIndex,
    commands_to_entries,
    default_commands,
    fetch_palette_index,
    fuzzy_score,
    search_palette,
)


# ---------------------------------------------------------------------------
# fuzzy_score
# ---------------------------------------------------------------------------


class TestFuzzyScore:
    def test_empty_query_returns_zero(self):
        assert fuzzy_score("", "anything") == 0.0

    def test_empty_candidate_returns_zero(self):
        assert fuzzy_score("x", "") == 0.0

    def test_non_subsequence_returns_zero(self):
        # `xyz` not in `hello` → 0.
        assert fuzzy_score("xyz", "hello") == 0.0

    def test_subsequence_match_scores_positive(self):
        assert fuzzy_score("abc", "aXbXc") > 0
        assert fuzzy_score("abc", "abc") > 0

    def test_prefix_match_scores_higher_than_substring(self):
        prefix = fuzzy_score("weather", "weather report")
        substr = fuzzy_score("weather", "the weather report")
        assert prefix > substr

    def test_substring_bonus_beats_loose_subsequence(self):
        substring = fuzzy_score("abc", "xabcx")
        loose = fuzzy_score("abc", "xayzbqrcz")
        assert substring > loose

    def test_case_insensitive(self):
        assert fuzzy_score("WEATHER", "weather") > 0
        assert fuzzy_score("Weather", "WEATHER") > 0

    def test_word_start_bonus(self):
        # `pdf` after "PDF parser" beats `pdf` at "alphaPDFomega".
        word_start = fuzzy_score("pdf", "PDF parser")
        mid_word = fuzzy_score("pdf", "alphaPDFomega")
        assert word_start > mid_word

    def test_consecutive_bonus(self):
        consecutive = fuzzy_score("abcd", "abcd")
        scattered = fuzzy_score("abcd", "a-b-c-d")
        assert consecutive > scattered

    def test_shorter_candidate_scores_higher(self):
        short = fuzzy_score("a", "a")
        long_match = fuzzy_score("a", "a" + "x" * 100)
        assert short > long_match


# ---------------------------------------------------------------------------
# Commands + entry projection
# ---------------------------------------------------------------------------


class TestDefaultCommands:
    def test_default_command_set(self):
        commands = default_commands()
        action_ids = {c.action_id for c in commands}
        # Pin the spec'd actions exist. The Screens-group entries
        # added by TODO §2 P0 land here too — every destination the
        # welcome banner names should also surface in the palette.
        expected = {
            "open_chat",
            "open_artifacts",
            "show_library",
            "open_evolution",
            "open_settings",
            "create_new_agent",
            "import_bundle",
            "export_library",
            "show_help",
            "open_catalog",
            "open_marketplace",
            "quit",
        }
        assert expected.issubset(action_ids)

    def test_screens_group_leads_palette(self):
        """The five Screens-group commands should appear at the
        top of the default order so an empty palette query
        surfaces the app's map of itself first."""
        commands = default_commands()
        leading_ids = [c.action_id for c in commands[:5]]
        assert leading_ids == [
            "open_chat",
            "open_artifacts",
            "show_library",
            "open_evolution",
            "open_settings",
        ]

    def test_screens_group_labels_follow_open_x_pattern(self):
        screens_ids = {
            "open_chat", "open_artifacts", "show_library",
            "open_evolution", "open_settings",
        }
        by_id = {c.action_id: c for c in default_commands()}
        for action_id in screens_ids:
            label = by_id[action_id].label
            assert label.startswith("Open "), (
                f"{action_id} → {label!r} should start with 'Open '"
            )

    def test_command_is_frozen(self):
        c = default_commands()[0]
        with pytest.raises(FrozenInstanceError):
            c.label = "x"  # type: ignore[misc]

    def test_create_new_agent_has_shortcut_hint(self):
        commands = {c.action_id: c for c in default_commands()}
        assert commands["create_new_agent"].shortcut_hint == "Ctrl+N"

    def test_commands_to_entries_projection(self):
        commands = default_commands()
        entries = commands_to_entries(commands)
        assert len(entries) == len(commands)
        first = entries[0]
        assert first.kind == "command"
        assert first.is_command
        # entry_id namespaced by the kind so we can dispatch off it.
        assert first.entry_id.startswith("command:")
        assert first.command_action is not None


# ---------------------------------------------------------------------------
# PaletteIndex shape
# ---------------------------------------------------------------------------


class TestPaletteIndex:
    def _entries(self) -> tuple[PaletteEntry, ...]:
        return (
            PaletteEntry(entry_id="c-1", kind="chain", label="Weather"),
            PaletteEntry(entry_id="s-1", kind="agent_skill", label="PDF extract"),
            *commands_to_entries(default_commands()),
        )

    def test_len_and_iter(self):
        index = PaletteIndex(entries=self._entries())
        assert len(index) > 0
        assert len(list(index)) == len(index)

    def test_is_empty(self):
        assert PaletteIndex().is_empty
        assert not PaletteIndex(entries=self._entries()).is_empty

    def test_filter_kinds(self):
        index = PaletteIndex(entries=self._entries())
        chains_only = index.filter_kinds(["chain"])
        assert all(e.kind == "chain" for e in chains_only)
        assert len(chains_only) == 1

    def test_index_is_frozen(self):
        index = PaletteIndex()
        with pytest.raises(FrozenInstanceError):
            index.entries = ()  # type: ignore[misc]

    def test_entry_is_frozen(self):
        entry = PaletteEntry(entry_id="x", kind="chain", label="x")
        with pytest.raises(FrozenInstanceError):
            entry.label = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# search_palette
# ---------------------------------------------------------------------------


class TestSearchPalette:
    def _index(self) -> PaletteIndex:
        return PaletteIndex(
            entries=(
                PaletteEntry(
                    entry_id="c-1", kind="chain", label="Weather forecaster",
                    description="Daily forecast",
                ),
                PaletteEntry(
                    entry_id="c-2", kind="chain", label="Stock summariser",
                    description="Daily report",
                ),
                PaletteEntry(
                    entry_id="s-1", kind="agent_skill", label="pdf-extract",
                    description="Extract text from PDFs",
                ),
                PaletteEntry(
                    entry_id="s-2", kind="agent_skill", label="excel-report",
                ),
                *commands_to_entries(default_commands()),
            )
        )

    def test_empty_query_returns_commands_first(self):
        results = search_palette(self._index(), "")
        # First N results are commands.
        assert results[0].is_command

    def test_empty_query_respects_top_k(self):
        results = search_palette(self._index(), "", top_k=3)
        assert len(results) == 3

    def test_query_ranks_by_score(self):
        results = search_palette(self._index(), "weather")
        # Top result should be the weather chain.
        assert results[0].entry_id == "c-1"

    def test_kind_filter(self):
        results = search_palette(
            self._index(), "report", kinds=["chain"],
        )
        assert all(r.kind == "chain" for r in results)
        # "Daily report" on the stock chain.
        assert any(r.entry_id == "c-2" for r in results)

    def test_non_match_returns_empty(self):
        # `xyzqwerty` matches nothing.
        results = search_palette(self._index(), "xyzqwerty")
        assert results == ()

    def test_top_k_caps_results(self):
        results = search_palette(self._index(), "a", top_k=2)
        assert len(results) <= 2

    def test_score_populated_on_query_results(self):
        results = search_palette(self._index(), "weather")
        assert results[0].score > 0

    def test_score_zero_on_empty_query(self):
        results = search_palette(self._index(), "")
        # Empty query → default ordering, no score computation.
        assert results[0].score == 0.0

    def test_command_searched_by_keywords(self):
        # `create_new_agent` command has "new" as a keyword.
        # Searching "new" should surface it.
        results = search_palette(self._index(), "new", kinds=["command"])
        action_ids = [r.command_action for r in results]
        assert "create_new_agent" in action_ids

    def test_ties_broken_by_label_ascending(self):
        # Two entries with identical scores → label alphabetical.
        index = PaletteIndex(
            entries=(
                PaletteEntry(entry_id="z", kind="chain", label="zeta thing"),
                PaletteEntry(entry_id="a", kind="chain", label="alpha thing"),
            )
        )
        results = search_palette(index, "thing")
        # Both match `thing`; alpha sorts before zeta.
        assert results[0].entry_id == "a"
        assert results[1].entry_id == "z"


# ---------------------------------------------------------------------------
# fetch_palette_index
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(
        self,
        *,
        chain_rows=None,
        skill_rows=None,
        chain_exc=None,
        skill_exc=None,
        delay=0.0,
    ):
        self.calls: list[dict] = []
        self._chain_rows = chain_rows or []
        self._skill_rows = skill_rows or []
        self._chain_exc = chain_exc
        self._skill_exc = skill_exc
        self._delay = delay

    def list_chains(self, **kwargs):
        self.calls.append({"op": "chains", **kwargs})
        if self._delay:
            time.sleep(self._delay)
        if self._chain_exc:
            raise self._chain_exc
        return self._chain_rows

    def list_agent_skills(self, **kwargs):
        self.calls.append({"op": "skills", **kwargs})
        if self._delay:
            time.sleep(self._delay)
        if self._skill_exc:
            raise self._skill_exc
        return self._skill_rows


class _StubMemory:
    def __init__(self, client):
        self.client = client


def _chain_row(entity_id: str, name: str = "Agent") -> dict:
    return {
        "entity_id": entity_id,
        "display_name": name,
        "description": "desc",
        "meta": {"tags": ["domain:weather"]},
    }


def _skill_row(entity_id: str, name: str = "skill") -> dict:
    return {
        "entity_id": entity_id,
        "content": {"name": name, "description": "skill desc"},
        "meta": {"tags": ["pdf"]},
    }


class TestFetchPaletteIndex:
    def test_happy_path(self):
        client = _StubClient(
            chain_rows=[_chain_row("c-1", "Weather")],
            skill_rows=[_skill_row("s-1", "pdf-extract")],
        )
        memory = _StubMemory(client)
        index = asyncio.run(fetch_palette_index(memory))
        # Commands + 1 chain + 1 skill.
        kinds = [e.kind for e in index]
        assert "chain" in kinds
        assert "agent_skill" in kinds
        assert "command" in kinds

    def test_concurrent_fetch(self):
        client = _StubClient(
            chain_rows=[_chain_row("c-1")],
            skill_rows=[_skill_row("s-1")],
            delay=0.05,
        )
        memory = _StubMemory(client)
        start = time.monotonic()
        asyncio.run(fetch_palette_index(memory, timeout=2.0))
        elapsed = time.monotonic() - start
        # Two 0.05s fetches in parallel → well under 0.09s.
        assert elapsed < 0.09, (
            f"expected concurrent fetch, got {elapsed:.3f}s"
        )

    def test_chain_exception_skipped(self):
        # Chains raise → only skills + commands in the index.
        client = _StubClient(
            chain_exc=RuntimeError("503"),
            skill_rows=[_skill_row("s-1")],
        )
        memory = _StubMemory(client)
        index = asyncio.run(fetch_palette_index(memory))
        kinds = [e.kind for e in index]
        assert "chain" not in kinds
        assert "agent_skill" in kinds

    def test_skill_exception_skipped(self):
        client = _StubClient(
            chain_rows=[_chain_row("c-1")],
            skill_exc=RuntimeError("503"),
        )
        memory = _StubMemory(client)
        index = asyncio.run(fetch_palette_index(memory))
        kinds = [e.kind for e in index]
        assert "chain" in kinds
        assert "agent_skill" not in kinds

    def test_missing_client_raises(self):
        with pytest.raises(PaletteError, match="`.client`"):
            asyncio.run(fetch_palette_index(object()))

    def test_missing_list_methods_skipped(self):
        # No list_chains or list_agent_skills → only commands.
        class _Empty:
            pass

        memory = _StubMemory(_Empty())
        index = asyncio.run(fetch_palette_index(memory))
        kinds = {e.kind for e in index}
        assert kinds == {"command"}

    def test_timeout_wraps(self):
        client = _StubClient(
            chain_rows=[], skill_rows=[], delay=0.5,
        )
        memory = _StubMemory(client)
        with pytest.raises(PaletteError, match="timed out"):
            asyncio.run(fetch_palette_index(memory, timeout=0.05))

    def test_custom_commands(self):
        custom = (
            Command(action_id="quit", label="Custom Quit"),
        )
        memory = _StubMemory(_StubClient())
        index = asyncio.run(
            fetch_palette_index(memory, commands=custom)
        )
        command_entries = [e for e in index if e.is_command]
        # Only one command (the custom one).
        assert len(command_entries) == 1
        assert command_entries[0].label == "Custom Quit"

    def test_namespace_channel_forwarded(self):
        client = _StubClient()
        memory = _StubMemory(client)
        asyncio.run(
            fetch_palette_index(
                memory, namespace="alice", channel="stable",
            )
        )
        # Both list calls carry the namespace + channel.
        for call in client.calls:
            assert call["namespace"] == "alice"
            assert call["channel"] == "stable"

    def test_limit_clamped(self):
        client = _StubClient()
        memory = _StubMemory(client)
        asyncio.run(fetch_palette_index(memory, limit=9999))
        for call in client.calls:
            assert call["limit"] == 200
        client.calls.clear()
        asyncio.run(fetch_palette_index(memory, limit=0))
        for call in client.calls:
            assert call["limit"] == 1

    def test_underscored_client_works(self):
        class _M:
            def __init__(self, client):
                self._client = client

        memory = _M(_StubClient(chain_rows=[_chain_row("c-1")]))
        index = asyncio.run(fetch_palette_index(memory))
        assert any(e.kind == "chain" for e in index)

    def test_skipped_rows_with_missing_id(self):
        # Row without entity_id is skipped quietly.
        client = _StubClient(
            chain_rows=[{"display_name": "no id"}, _chain_row("c-1", "ok")],
        )
        memory = _StubMemory(client)
        index = asyncio.run(fetch_palette_index(memory))
        chain_entries = [e for e in index if e.kind == "chain"]
        assert len(chain_entries) == 1
        assert chain_entries[0].entry_id == "c-1"

    def test_attribute_access_rows_supported(self):
        # SDK shape — attribute access not dict.
        class _Row:
            def __init__(self, entity_id, display_name="Attr", description="d", meta=None):
                self.entity_id = entity_id
                self.display_name = display_name
                self.description = description
                self.meta = meta or {"tags": []}

        client = _StubClient(chain_rows=[_Row("c-1")])
        memory = _StubMemory(client)
        index = asyncio.run(fetch_palette_index(memory))
        chain_entries = [e for e in index if e.kind == "chain"]
        assert chain_entries[0].label == "Attr"


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            Command as Cmd,
            PaletteEntry as PE,
            PaletteIndex as PI,
            PaletteError as Err,
            default_commands as defaults,
            commands_to_entries as cte,
            fetch_palette_index as fetch,
            fuzzy_score as score,
            search_palette as search,
        )

        assert Cmd is Command
        assert PE is PaletteEntry
        assert PI is PaletteIndex
        assert Err is PaletteError
        assert defaults is default_commands
        assert cte is commands_to_entries
        assert fetch is fetch_palette_index
        assert score is fuzzy_score
        assert search is search_palette
