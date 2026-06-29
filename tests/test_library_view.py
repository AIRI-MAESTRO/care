"""Tests for the LibraryScreen data layer (TODO §1.3 P0).

The Textual DataTable + sidebar are gated on §1 P0; this suite
pins the contract those screens will bind to.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from care.runtime.library_view import (
    DEFAULT_VIEW_STATE_PATH,
    LibraryFilters,
    LibraryRow,
    LibrarySort,
    LibraryView,
    LibraryViewError,
    LibraryViewState,
    LibraryViewStateStore,
    clear_filters,
    fetch_library_view,
    load_view_state,
    parse_library_row,
    save_view_state,
    with_domain,
    with_favourites_only,
    with_search,
    with_status,
    with_tags,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


def _entity(
    *,
    entity_id: str = "ent-1",
    display_name: str = "Weather forecaster",
    description: str = "Daily forecast",
    favourite: bool = False,
    run_count: int = 0,
    last_run_at: str | None = "2026-05-19T12:00:00+00:00",
    channel: str = "latest",
    tags: list[str] | None = None,
    steps: list | None = None,
    fitness: float | None = None,
    care_domain: str = "",
    is_draft: bool = False,
    evolution_meta: dict | None = None,
) -> dict:
    tags = list(tags) if tags else ["domain:weather"]
    if is_draft and "draft" not in tags:
        tags.append("draft")
    content: dict = {"steps": steps if steps is not None else [{}, {}]}
    care_block: dict = {}
    if care_domain:
        care_block["domain"] = care_domain
    if fitness is not None:
        care_block["fitness_score"] = fitness
    if care_block:
        content["metadata"] = {"care": care_block}
    return {
        "entity_type": "chain",
        "entity_id": entity_id,
        "version_id": "v-1",
        "channel": channel,
        "etag": "etag",
        "favourite": favourite,
        "run_count": run_count,
        "last_run_at": last_run_at,
        "display_name": display_name,
        "description": description,
        "meta": {"tags": tags, "name": "internal-name"},
        "content": content,
        "evolution_meta": evolution_meta,
    }


# ---------------------------------------------------------------------------
# parse_library_row
# ---------------------------------------------------------------------------


class TestParseLibraryRow:
    def test_happy_path(self):
        row = parse_library_row(_entity())
        assert row.entity_id == "ent-1"
        assert row.display_name == "Weather forecaster"
        assert row.description == "Daily forecast"
        assert row.favourite is False
        assert row.run_count == 0
        assert isinstance(row.last_run_at, datetime)
        assert row.tags == ("domain:weather",)
        assert row.step_count == 2
        assert row.domain == "weather"
        assert row.status == "runnable"
        assert row.is_draft is False
        assert row.is_evolved is False

    def test_extracts_domain_from_tag_when_absent(self):
        row = parse_library_row(_entity(tags=["domain:finance"]))
        assert row.domain == "finance"

    def test_extracts_domain_from_care_block_first(self):
        # CARE-block `domain` wins over the tag if both are present.
        row = parse_library_row(
            _entity(
                tags=["domain:tagged"],
                care_domain="block-domain",
            )
        )
        assert row.domain == "block-domain"

    def test_draft_status(self):
        row = parse_library_row(_entity(is_draft=True))
        assert row.is_draft
        assert row.status == "draft"

    def test_draft_via_channel(self):
        row = parse_library_row(_entity(channel="draft", tags=[]))
        assert row.is_draft
        assert row.status == "draft"

    def test_evolved_status_from_fitness(self):
        row = parse_library_row(_entity(fitness=0.87))
        assert row.is_evolved
        assert row.fitness == 0.87
        assert row.status == "evolved"

    def test_evolved_from_evolution_meta_column(self):
        row = parse_library_row(
            _entity(evolution_meta={"fitness_score": 0.55})
        )
        assert row.fitness == 0.55
        assert row.is_evolved

    def test_label_falls_back_to_internal_name(self):
        ent = _entity(display_name="")
        ent["display_name"] = None
        row = parse_library_row(ent)
        assert row.label == "internal-name"

    def test_label_truncates_to_entity_id(self):
        ent = _entity(display_name="")
        ent["display_name"] = None
        ent["meta"]["name"] = ""
        row = parse_library_row(ent)
        assert row.label == "ent-1"[:12]

    def test_none_entity_raises(self):
        with pytest.raises(LibraryViewError):
            parse_library_row(None)

    def test_handles_missing_optional_fields(self):
        ent = _entity()
        # Strip every optional column.
        del ent["last_run_at"]
        del ent["favourite"]
        del ent["run_count"]
        del ent["evolution_meta"]
        ent["content"] = {}
        row = parse_library_row(ent)
        assert row.last_run_at is None
        assert row.favourite is False
        assert row.run_count == 0
        assert row.step_count is None
        assert row.fitness is None

    def test_iso_datetime_with_z(self):
        row = parse_library_row(
            _entity(last_run_at="2026-05-19T12:00:00Z")
        )
        assert isinstance(row.last_run_at, datetime)
        assert row.last_run_at.year == 2026

    def test_invalid_datetime_collapses_to_none(self):
        row = parse_library_row(_entity(last_run_at="not a date"))
        assert row.last_run_at is None

    def test_attribute_access_objects_work(self):
        # SDK shape — attribute access not dict.
        class _Resp:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        ent = _Resp(
            entity_id="x",
            entity_type="chain",
            version_id="v",
            channel="latest",
            etag="e",
            favourite=True,
            run_count=5,
            last_run_at=datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
            display_name="Attr-access",
            description="d",
            meta={"tags": ["domain:x"]},
            content={"steps": [{}]},
            evolution_meta=None,
        )
        row = parse_library_row(ent)
        assert row.display_name == "Attr-access"
        assert row.favourite is True
        assert row.run_count == 5


# ---------------------------------------------------------------------------
# Frozen models
# ---------------------------------------------------------------------------


class TestModelShape:
    def test_row_is_frozen(self):
        row = LibraryRow(entity_id="x")
        with pytest.raises(FrozenInstanceError):
            row.entity_id = "y"  # type: ignore[misc]

    def test_filters_is_frozen(self):
        f = LibraryFilters()
        with pytest.raises(FrozenInstanceError):
            f.search = "x"  # type: ignore[misc]

    def test_sort_validates(self):
        with pytest.raises(LibraryViewError):
            LibrarySort(field="not-a-field")
        with pytest.raises(LibraryViewError):
            LibrarySort(direction="sideways")  # type: ignore[arg-type]

    def test_filters_predicates(self):
        empty = LibraryFilters()
        assert empty.is_filtering is False
        assert empty.tag_set == frozenset()

        non_empty = LibraryFilters(tags=("a", "b"), favourites_only=True)
        assert non_empty.is_filtering is True
        assert non_empty.tag_set == {"a", "b"}

    def test_view_iter_and_len(self):
        rows = (LibraryRow(entity_id="a"), LibraryRow(entity_id="b"))
        view = LibraryView(rows=rows)
        assert len(view) == 2
        assert [r.entity_id for r in view] == ["a", "b"]
        assert view.is_empty is False
        assert LibraryView().is_empty


# ---------------------------------------------------------------------------
# Filter mutators
# ---------------------------------------------------------------------------


class TestFilterMutators:
    def test_with_domain(self):
        f = with_domain(LibraryFilters(), "weather")
        assert f.domain == "weather"

    def test_with_domain_clear(self):
        f = with_domain(LibraryFilters(domain="weather"), None)
        assert f.domain is None

    def test_with_tags_strips_whitespace(self):
        f = with_tags(LibraryFilters(), ["  a  ", "b", ""])
        assert f.tags == ("a", "b")

    def test_with_status_known(self):
        f = with_status(LibraryFilters(), "draft")
        assert f.status == "draft"

    def test_with_status_unknown_raises(self):
        with pytest.raises(LibraryViewError):
            with_status(LibraryFilters(), "nonexistent")

    def test_with_favourites_only(self):
        f = with_favourites_only(LibraryFilters(), True)
        assert f.favourites_only is True

    def test_with_search(self):
        f = with_search(LibraryFilters(), "weather")
        assert f.search == "weather"

    def test_clear_filters_preserves_operator_settings(self):
        f = LibraryFilters(
            domain="weather", search="hi", namespace="alice", channel="stable",
        )
        cleared = clear_filters(f)
        assert cleared.domain is None
        assert cleared.search == ""
        assert cleared.namespace == "alice"
        assert cleared.channel == "stable"


# ---------------------------------------------------------------------------
# fetch_library_view
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(
        self,
        *,
        rows: list | None = None,
        exc: Exception | None = None,
        delay: float = 0.0,
        bad_type: bool = False,
    ):
        self._rows = rows if rows is not None else []
        self._exc = exc
        self._delay = delay
        self._bad_type = bad_type
        self.calls: list[dict] = []

    def list_chains(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self._delay:
            time.sleep(self._delay)
        if self._exc:
            raise self._exc
        if self._bad_type:
            return 42  # not a list
        return self._rows


class _StubMemory:
    def __init__(self, client):
        self.client = client


class TestFetchLibraryView:
    def test_malformed_row_surfaces_library_view_error(self):
        # A non-numeric run_count makes parse_library_row's int() raise; the
        # parse loop must surface a friendly LibraryViewError, not a raw
        # traceback (parsing was outside the fetch guard — Library-500 class).
        bad = _entity(entity_id="x", run_count="not-a-number")  # type: ignore[arg-type]
        memory = _StubMemory(_StubClient(rows=[bad]))
        with pytest.raises(LibraryViewError):
            asyncio.run(fetch_library_view(memory))

    def test_happy_path(self):
        rows = [
            _entity(entity_id="a", display_name="alpha"),
            _entity(entity_id="b", display_name="beta", favourite=True),
        ]
        memory = _StubMemory(_StubClient(rows=rows))
        view = asyncio.run(fetch_library_view(memory))
        assert isinstance(view, LibraryView)
        assert len(view) == 2
        # Favourites first (default sort).
        assert view.rows[0].entity_id == "b"
        assert view.rows[1].entity_id == "a"
        # Call shape sane.
        call = memory.client.calls[0]
        assert call["limit"] == 50
        assert call["sort_by"] == "created_at"
        assert call["sort_dir"] == "desc"
        assert call["channel"] == "latest"

    def test_favourites_first_disabled(self):
        rows = [
            _entity(entity_id="a", display_name="alpha"),
            _entity(entity_id="b", display_name="beta", favourite=True),
        ]
        memory = _StubMemory(_StubClient(rows=rows))
        view = asyncio.run(
            fetch_library_view(
                memory, sort=LibrarySort(favourites_first=False),
            )
        )
        # Server order preserved.
        assert view.rows[0].entity_id == "a"
        assert view.rows[1].entity_id == "b"

    def test_filter_knobs_forwarded(self):
        memory = _StubMemory(_StubClient(rows=[]))
        asyncio.run(
            fetch_library_view(
                memory,
                filters=LibraryFilters(
                    domain="weather",
                    tags=("favourite",),
                    favourites_only=True,
                    search="storm",
                    namespace="alice",
                ),
                sort=LibrarySort(field="run_count", direction="asc"),
                limit=25,
            )
        )
        call = memory.client.calls[0]
        # Domain prepended to tags list.
        assert "domain:weather" in call["tags"]
        assert "favourite" in call["tags"]
        assert call["favourites_only"] is True
        assert call["q"] == "storm"
        assert call["namespace"] == "alice"
        assert call["sort_by"] == "run_count"
        assert call["sort_dir"] == "asc"
        assert call["limit"] == 25

    def test_status_draft_narrows_channel(self):
        memory = _StubMemory(_StubClient(rows=[]))
        asyncio.run(
            fetch_library_view(
                memory, filters=LibraryFilters(status="draft"),
            )
        )
        call = memory.client.calls[0]
        assert call["channel"] == "draft"
        assert "draft" in call["tags"]

    def test_status_evolved_narrows_channel(self):
        memory = _StubMemory(_StubClient(rows=[]))
        asyncio.run(
            fetch_library_view(
                memory, filters=LibraryFilters(status="evolved"),
            )
        )
        call = memory.client.calls[0]
        assert call["channel"] == "evolved"

    def test_limit_clamped(self):
        memory = _StubMemory(_StubClient(rows=[]))
        asyncio.run(fetch_library_view(memory, limit=9999))
        assert memory.client.calls[0]["limit"] == 200
        memory.client.calls.clear()
        asyncio.run(fetch_library_view(memory, limit=0))
        assert memory.client.calls[0]["limit"] == 1

    def test_has_more_when_limit_hit(self):
        # Page is full → has_more=True.
        rows = [_entity(entity_id=f"e-{i}") for i in range(50)]
        memory = _StubMemory(_StubClient(rows=rows))
        view = asyncio.run(fetch_library_view(memory, limit=50))
        assert view.has_more is True

    def test_no_more_when_below_limit(self):
        rows = [_entity(entity_id=f"e-{i}") for i in range(3)]
        memory = _StubMemory(_StubClient(rows=rows))
        view = asyncio.run(fetch_library_view(memory, limit=50))
        assert view.has_more is False

    def test_missing_client_raises(self):
        with pytest.raises(LibraryViewError, match="list_chains"):
            asyncio.run(fetch_library_view(object()))

    def test_sdk_exception_wraps(self):
        memory = _StubMemory(_StubClient(exc=RuntimeError("503")))
        with pytest.raises(LibraryViewError, match="fetch failed"):
            asyncio.run(fetch_library_view(memory))

    def test_timeout_wraps(self):
        memory = _StubMemory(_StubClient(delay=0.5))
        with pytest.raises(LibraryViewError, match="timed out"):
            asyncio.run(fetch_library_view(memory, timeout=0.05))

    def test_unexpected_type_wraps(self):
        memory = _StubMemory(_StubClient(bad_type=True))
        with pytest.raises(LibraryViewError, match="unexpected type"):
            asyncio.run(fetch_library_view(memory))

    def test_underscored_client_works(self):
        class _M:
            def __init__(self, client):
                self._client = client

        memory = _M(_StubClient(rows=[_entity()]))
        view = asyncio.run(fetch_library_view(memory))
        assert len(view) == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestViewStateStore:
    def test_save_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "view.json"
        state = LibraryViewState(
            sort=LibrarySort(field="run_count", direction="asc",
                            favourites_first=False),
            filters=LibraryFilters(
                domain="weather", tags=("favourite",),
                status="evolved", search="storm",
            ),
        )
        save_view_state(state, path=path)
        loaded = load_view_state(path)
        assert loaded is not None
        assert loaded.sort.field == "run_count"
        assert loaded.sort.direction == "asc"
        assert loaded.sort.favourites_first is False
        assert loaded.filters.domain == "weather"
        assert loaded.filters.tags == ("favourite",)
        assert loaded.filters.status == "evolved"
        assert loaded.filters.search == "storm"

    def test_load_missing_returns_none(self, tmp_path: Path):
        assert load_view_state(tmp_path / "nope.json") is None

    def test_load_malformed_returns_none(self, tmp_path: Path):
        path = tmp_path / "view.json"
        path.write_text("not json")
        assert load_view_state(path) is None

    def test_load_non_dict_returns_none(self, tmp_path: Path):
        path = tmp_path / "view.json"
        path.write_text("[]")
        assert load_view_state(path) is None

    def test_load_schema_mismatch_returns_none(self, tmp_path: Path):
        path = tmp_path / "view.json"
        path.write_text('{"schema_version": 99, "sort": {}, "filters": {}}')
        assert load_view_state(path) is None

    def test_load_invalid_status_collapses(self, tmp_path: Path):
        # Unknown status in the JSON → load returns None (refuses
        # to materialise a partially-corrupt state).
        path = tmp_path / "view.json"
        import json as _json

        _json_str = _json.dumps(
            {
                "schema_version": 1,
                "sort": {
                    "field": "last_run_at", "direction": "desc",
                    "favourites_first": True,
                },
                "filters": {
                    "status": "not-a-status",
                    "tags": [],
                },
            }
        )
        path.write_text(_json_str)
        # _state_from_dict drops the bad status to None and returns
        # state — load doesn't return None for this case.
        loaded = load_view_state(path)
        assert loaded is not None
        assert loaded.filters.status is None

    def test_clear_idempotent(self, tmp_path: Path):
        path = tmp_path / "view.json"
        save_view_state(LibraryViewState(), path=path)
        store = LibraryViewStateStore(path)
        assert store.clear() is True
        assert store.clear() is False  # already gone

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "dir" / "view.json"
        save_view_state(LibraryViewState(), path=path)
        assert path.exists()

    def test_atomic_save_no_tempfile_leftovers(self, tmp_path: Path):
        path = tmp_path / "view.json"
        for _ in range(5):
            save_view_state(LibraryViewState(), path=path)
        leftovers = list(tmp_path.glob(".library_view-*"))
        assert leftovers == []

    def test_concurrent_save_no_corruption(self, tmp_path: Path):
        path = tmp_path / "view.json"
        store = LibraryViewStateStore(path)

        def hammer(idx):
            state = LibraryViewState(
                filters=LibraryFilters(search=f"thread-{idx}"),
            )
            store.save(state)

        threads = [
            threading.Thread(target=hammer, args=(i,)) for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # File ends up well-formed.
        loaded = store.load()
        assert loaded is not None
        assert loaded.filters.search.startswith("thread-")

    def test_default_path_constant(self):
        # The constant is documented + ready for callers.
        assert str(DEFAULT_VIEW_STATE_PATH).endswith(
            "/care/library_view.json"
        )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestResolveDefaultViewStatePath:
    """§8 P3 — `CARE_VIEW_STATE_PATH` env override
    redirects the persisted view-state sidecar so tests can
    isolate from the user's real home dir."""

    def test_env_unset_returns_default(self, monkeypatch):
        from care.runtime.library_view import (
            DEFAULT_VIEW_STATE_PATH,
            resolve_default_view_state_path,
        )

        # Remove the env var if any prior fixture set it
        # (autouse `_isolate_library_view_state` in conftest
        # is the suspect — the autouse fixture sets it to
        # tmp_path; we temporarily drop it here).
        monkeypatch.delenv(
            "CARE_VIEW_STATE_PATH", raising=False,
        )
        assert (
            resolve_default_view_state_path()
            == DEFAULT_VIEW_STATE_PATH
        )

    def test_env_set_returns_override(
        self, monkeypatch, tmp_path,
    ):
        from care.runtime.library_view import (
            resolve_default_view_state_path,
        )

        target = tmp_path / "custom-library_view.json"
        monkeypatch.setenv(
            "CARE_VIEW_STATE_PATH", str(target),
        )
        assert (
            resolve_default_view_state_path() == target
        )

    def test_store_default_constructor_honours_env(
        self, monkeypatch, tmp_path,
    ):
        from care.runtime.library_view import (
            LibraryViewStateStore,
        )

        target = tmp_path / "ctor-library_view.json"
        monkeypatch.setenv(
            "CARE_VIEW_STATE_PATH", str(target),
        )
        store = LibraryViewStateStore()  # no explicit path
        assert store.path == target

    def test_env_expands_tilde(self, monkeypatch):
        from care.runtime.library_view import (
            resolve_default_view_state_path,
        )

        monkeypatch.setenv(
            "CARE_VIEW_STATE_PATH",
            "~/custom/library_view.json",
        )
        resolved = resolve_default_view_state_path()
        assert not str(resolved).startswith("~")
        assert resolved.name == "library_view.json"

    def test_autouse_isolation_keeps_user_home_clean(
        self, tmp_path,
    ):
        """Sanity: the conftest autouse fixture redirects the
        default path into the per-test tmp_path. Touching the
        store with no kwargs should NOT land on the user's
        real home dir."""
        from care.runtime.library_view import (
            LibraryFilters,
            LibrarySort,
            LibraryViewState,
            LibraryViewStateStore,
        )

        store = LibraryViewStateStore()
        path = store.save(LibraryViewState(
            sort=LibrarySort(),
            filters=LibraryFilters(),
        ))
        # The fixture pointed us at the per-test tmp_path so
        # the resolved file lives under it, NOT under
        # ~/.local/state/care/.
        assert str(path).startswith(str(tmp_path))


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            LibraryFilters as F,
            LibraryRow as R,
            LibrarySort as S,
            LibraryView as V,
            fetch_library_view as fetch,
            parse_library_row as parse,
            load_view_state as load,
            save_view_state as save,
        )

        assert F is LibraryFilters
        assert R is LibraryRow
        assert S is LibrarySort
        assert V is LibraryView
        assert fetch is fetch_library_view
        assert parse is parse_library_row
        assert load is load_view_state
        assert save is save_view_state
