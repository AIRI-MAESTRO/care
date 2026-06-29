"""LibraryScreen data layer (TODO §1.3 P0).

The LibraryScreen's full-screen DataTable shows the user's saved
agents with columns: ⭐ favourite, name, domain, #steps,
last_run_at, run_count, fitness (if evolved), tags. A sidebar
filters by domain / tag chips / status / favourites; a search
bar does substring match on name + description.

The Textual DataTable + sidebar are gated on TODO §1 P0
multi-screen workflow, but the row projection + fetch driver +
filter / sort state + sort-preference persistence land now.

What this module provides:

* :class:`LibraryRow` — frozen per-row projection of a
  `gigaevo_client.EntityResponse` carrying every column the
  table renders. Pure-data; no presentation choices baked in.
* :class:`LibraryFilters` — frozen sidebar state with predicates
  (`is_filtering`, `tag_set`).
* :class:`LibrarySort` — frozen sort spec; ``favourites_first``
  pins ⭐ rows above the rest.
* :class:`LibraryView` — frozen aggregate (rows + applied
  filters + sort + total + has_more + next_cursor).
* :func:`parse_library_row` — pure projection.
* :func:`fetch_library_view` — async helper hitting
  ``memory.client.list_chains(...)`` with the filter knobs.
* :class:`LibraryViewState` + :func:`save_view_state` /
  :func:`load_view_state` — persistence for the user's last
  sort + filter selection at
  ``~/.local/state/care/library_view.json``.

Duck-typed: the fetcher accepts any `CareMemory`-like facade
exposing ``client.list_chains(...)``. Projection accepts dicts
OR `EntityResponse` model objects (attribute reads).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_VIEW_STATE_PATH = Path(
    "~/.local/state/care/library_view.json"
).expanduser()
"""XDG-style location for the persisted library sort/filter
preferences. Matches the convention `run_state.json` uses so the
two state files sit side by side."""


_VIEW_STATE_PATH_ENV = "CARE_VIEW_STATE_PATH"
"""Env var override for :data:`DEFAULT_VIEW_STATE_PATH`. Tests
set this to a tmp_path so pilot runs don't read / write the
user's real preferences (§8 P3 — caught in iter 63 when a
persisted ``search: "/"`` from interactive debugging bled into
test assertions about the default state).

NOTE: deliberately uses a single ``_`` between ``CARE`` and the
rest — the ``__`` delimiter pydantic-settings uses to nest
keys would route ``CARE_LIBRARY__VIEW_STATE_PATH`` into a
non-existent `CareConfig.library.view_state_path` field and
fail validation. Single-underscore form sits outside
CareConfig's prefix scan."""


def resolve_default_view_state_path() -> Path:
    """Resolve the persistent view-state path, honouring
    :data:`_VIEW_STATE_PATH_ENV` when set.

    Resolved lazily on every call so a test that sets the env
    var AFTER importing this module still picks the override
    up — the original `DEFAULT_VIEW_STATE_PATH` constant was
    captured at module-import time and survived as a stable
    reference for code that explicitly opts into the user's
    real path."""
    override = os.environ.get(_VIEW_STATE_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_VIEW_STATE_PATH


_VIEW_STATE_SCHEMA_VERSION = 1
"""Bump when the on-disk shape changes incompatibly; older
snapshots are dropped on load (matches `run_state.py`)."""


_VALID_SORT_FIELDS: frozenset[str] = frozenset(
    {"last_run_at", "run_count", "display_name", "created_at"}
)
"""Sort fields the server's ``GET /v1/chains`` endpoint accepts."""


_VALID_STATUSES: frozenset[str] = frozenset(
    {"draft", "runnable", "evolved"}
)
"""Status filter values the LibraryScreen's sidebar exposes.

* ``draft`` — chains still pinned to the draft channel (not
  promoted via SaveAgentModal). Picked up by the `draft` tag.
* ``runnable`` — chains pinned to `latest` (the default user
  view).
* ``evolved`` — chains carrying any `evolution_meta.fitness_score`.
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LibraryViewError(RuntimeError):
    """Raised when library view retrieval fails — unreachable
    Memory, timeout, malformed response. The LibraryScreen
    catches this and shows a friendly toast."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryRow:
    """One row in the LibraryScreen's DataTable.

    Mirrors the columns the modal renders. ``step_count`` and
    ``fitness`` are surfaced when available; ``None`` means
    "render an em-dash" (no data) without conflating with
    integer 0 / float 0.0.
    """

    entity_id: str
    entity_type: Literal["chain", "agent", "agent_skill"] = "chain"
    display_name: str = ""
    name: str = ""  # underlying (non-renamed) name
    description: str = ""
    domain: str = ""
    favourite: bool = False
    tags: tuple[str, ...] = ()
    run_count: int = 0
    last_run_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    step_count: Optional[int] = None
    fitness: Optional[float] = None
    is_draft: bool = False
    is_evolved: bool = False
    channel: str = "latest"

    @property
    def status(self) -> str:
        """Status badge the sidebar groups by: ``draft`` /
        ``runnable`` / ``evolved``. ``evolved`` wins over
        ``runnable`` when a chain carries fitness AND is pinned
        to ``latest``."""
        if self.is_draft:
            return "draft"
        if self.is_evolved:
            return "evolved"
        return "runnable"

    @property
    def label(self) -> str:
        """The string to render in the name column. Display name
        wins; otherwise fall back to the underlying name; finally
        to the first 12 chars of the entity_id."""
        if self.display_name:
            return self.display_name
        if self.name:
            return self.name
        return self.entity_id[:12]


@dataclass(frozen=True)
class LibraryFilters:
    """Sidebar filter state.

    Frozen — sidebar emits a new instance on every checkbox /
    chip toggle. ``is_filtering`` predicate drives the "Clear
    filters" button visibility.
    """

    domain: Optional[str] = None
    tags: tuple[str, ...] = ()
    status: Optional[str] = None
    favourites_only: bool = False
    search: str = ""
    namespace: Optional[str] = None
    channel: str = "latest"

    @property
    def is_filtering(self) -> bool:
        return bool(
            self.domain
            or self.tags
            or self.status
            or self.favourites_only
            or self.search.strip()
        )

    @property
    def tag_set(self) -> frozenset[str]:
        return frozenset(self.tags)


@dataclass(frozen=True)
class LibrarySort:
    """Sort spec.

    The server enforces direction + field validity. We pin a
    `favourites_first` toggle so the LibraryScreen's default
    view (favourites pinned above the rest) is honoured client-
    side — Memory's API doesn't sort by `favourite DESC` as a
    primary key.

    Default ordering is newest-first: ``created_at`` descending, so a
    freshly-saved chain lands at the top of the Library on open.
    """

    field: str = "created_at"
    direction: Literal["asc", "desc"] = "desc"
    favourites_first: bool = True

    def __post_init__(self) -> None:
        if self.field not in _VALID_SORT_FIELDS:
            raise LibraryViewError(
                f"unknown sort field {self.field!r}; valid: "
                f"{sorted(_VALID_SORT_FIELDS)}"
            )
        if self.direction not in ("asc", "desc"):
            raise LibraryViewError(
                f"unknown sort direction {self.direction!r}"
            )


@dataclass(frozen=True)
class LibraryView:
    """Frozen aggregate for the DataTable + footer."""

    rows: tuple[LibraryRow, ...] = ()
    filters: LibraryFilters = field(default_factory=LibraryFilters)
    sort: LibrarySort = field(default_factory=LibrarySort)
    total_returned: int = 0
    has_more: bool = False
    next_cursor: Optional[str] = None

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    @property
    def is_empty(self) -> bool:
        return not self.rows


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def parse_library_row(
    entity: Any,
    *,
    entity_type: Literal["chain", "agent", "agent_skill"] = "chain",
) -> LibraryRow:
    """Project a `gigaevo_client.EntityResponse` (or dict) into
    a :class:`LibraryRow`.

    Reads:
    * `display_name` / `description` / `favourite` / `run_count`
      / `last_run_at` from the top-level library columns.
    * `meta.tags` for the tag chips.
    * `content` for step count + CARE metadata
      (`care.domain`, evolution fitness).

    Tolerant: missing fields default; the row still renders.
    """
    if entity is None:
        raise LibraryViewError("cannot project a None entity row")

    entity_id = str(_read(entity, "entity_id") or "")
    display_name = _read_str(entity, "display_name")
    description = _read_str(entity, "description")
    favourite = bool(_read(entity, "favourite") or False)
    run_count = int(_read(entity, "run_count") or 0)
    last_run_at = _read(entity, "last_run_at")
    if not isinstance(last_run_at, datetime):
        last_run_at = _coerce_datetime(last_run_at)
    created_at = _read(entity, "created_at")
    if not isinstance(created_at, datetime):
        created_at = _coerce_datetime(created_at)
    channel = str(_read(entity, "channel") or "latest")

    meta = _read(entity, "meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    raw_tags = meta.get("tags") or []
    tags = tuple(str(t) for t in raw_tags if isinstance(t, str))
    name = str(meta.get("name") or "")

    content = _read(entity, "content") or {}
    if not isinstance(content, dict):
        content = {}

    steps_payload = content.get("steps")
    step_count: Optional[int] = None
    if isinstance(steps_payload, list):
        step_count = len(steps_payload)

    care_meta = content.get("metadata", {})
    if isinstance(care_meta, dict):
        care_block = care_meta.get("care") or care_meta
    else:
        care_block = {}
    domain = ""
    if isinstance(care_block, dict):
        # Prefer explicit `domain` field; fall back to mining a
        # `domain:{value}` tag.
        domain = _read_str(care_block, "domain")
    if not domain:
        for tag in tags:
            if tag.startswith("domain:"):
                domain = tag[len("domain:") :]
                break

    is_draft = "draft" in tags or channel == "draft"
    fitness = _extract_fitness(care_block, _read(entity, "evolution_meta"))
    is_evolved = fitness is not None or channel == "evolved"

    return LibraryRow(
        entity_id=entity_id,
        entity_type=entity_type,
        display_name=display_name,
        name=name,
        description=description,
        domain=domain,
        favourite=favourite,
        tags=tags,
        run_count=run_count,
        last_run_at=last_run_at,
        created_at=created_at,
        step_count=step_count,
        fitness=fitness,
        is_draft=is_draft,
        is_evolved=is_evolved,
        channel=channel,
    )


def _read(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _read_str(obj: Any, name: str) -> str:
    value = _read(obj, name)
    return value if isinstance(value, str) else ""


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _extract_fitness(care_block: Any, evolution_meta: Any) -> Optional[float]:
    """Mine fitness from either the CARE metadata block (legacy)
    or the dedicated `evolution_meta` column (PREPARE.md §1.6
    standard). `evolution_meta.fitness_score` wins."""
    candidates: list[Any] = []
    if isinstance(evolution_meta, dict):
        candidates.append(evolution_meta.get("fitness_score"))
        candidates.append(evolution_meta.get("fitness"))
    if isinstance(care_block, dict):
        candidates.append(care_block.get("fitness_score"))
        candidates.append(care_block.get("fitness"))
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------


async def fetch_library_view(
    memory: Any,
    *,
    filters: LibraryFilters | None = None,
    sort: LibrarySort | None = None,
    limit: int = 50,
    offset: int = 0,
    timeout: float = 10.0,
) -> LibraryView:
    """Hit Memory's `list_chains` endpoint with the filter + sort
    knobs and return a projected :class:`LibraryView`.

    Args:
        memory: A `CareMemory` facade (or any object exposing
            `.client.list_chains(...)`).
        filters: Sidebar state. ``None`` uses defaults (no
            filtering).
        sort: Sort spec. ``None`` uses defaults
            (`last_run_at desc` + favourites pinned).
        limit: Rows per page (server caps at 200).
        offset: Pagination offset.
        timeout: Per-call deadline.

    Returns:
        :class:`LibraryView` with the projected rows + the
        effective filters/sort echoed back so the table footer
        can render "showing 23 of 100 — filtered by ⭐, tags".

    Raises:
        LibraryViewError: Memory unreachable, timed out, or
            returned a malformed shape.
    """
    filters = filters or LibraryFilters()
    sort = sort or LibrarySort()

    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    fn = getattr(client, "list_chains", None) if client else None
    if not callable(fn):
        raise LibraryViewError(
            "memory facade does not expose client.list_chains()"
        )

    list_tags = list(filters.tags) if filters.tags else None
    if filters.domain:
        domain_tag = f"domain:{filters.domain}"
        if not list_tags or domain_tag not in list_tags:
            list_tags = (list_tags or []) + [domain_tag]
    if filters.status == "draft":
        list_tags = (list_tags or []) + ["draft"]

    # When the sidebar selects status=evolved, we narrow the read
    # to the `evolved` channel; this matches the server-side
    # auto-promotion convention (PREPARE.md §5).
    channel = filters.channel
    if filters.status == "evolved":
        channel = "evolved"
    elif filters.status == "draft":
        channel = "draft"

    capped_limit = max(1, min(limit, 200))

    start = time.monotonic()
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(
                fn,
                limit=capped_limit,
                offset=offset,
                channel=channel,
                sort_by=sort.field,
                sort_dir=sort.direction,
                favourites_only=filters.favourites_only or None,
                tags=list_tags,
                q=filters.search.strip() or None,
                namespace=filters.namespace,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        latency = (time.monotonic() - start) * 1000
        raise LibraryViewError(
            f"library fetch timed out after {timeout:.1f}s ({latency:.0f}ms elapsed)"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise LibraryViewError(
            f"library fetch failed: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(rows, (list, tuple)):
        raise LibraryViewError(
            f"list_chains returned unexpected type {type(rows).__name__}"
        )

    try:
        parsed = tuple(parse_library_row(row) for row in rows)
        sorted_rows = _apply_favourites_first(parsed, sort)
    except LibraryViewError:
        raise
    except Exception as exc:  # noqa: BLE001
        # A malformed row (e.g. a non-numeric run_count → int()) must surface
        # as the friendly LibraryViewError the empty-state renders, not a raw
        # traceback — same contract as the fetch guard above. Parsing was
        # outside any try, so this is the Library-500 bug class.
        raise LibraryViewError(
            f"library parse failed: {type(exc).__name__}: {exc}"
        ) from exc

    has_more = len(rows) >= capped_limit
    return LibraryView(
        rows=sorted_rows,
        filters=filters,
        sort=sort,
        total_returned=len(sorted_rows),
        has_more=has_more,
        next_cursor=None,
    )


def _apply_favourites_first(
    rows: tuple[LibraryRow, ...],
    sort: LibrarySort,
) -> tuple[LibraryRow, ...]:
    """Stable-partition rows so ⭐ rows come first, preserving
    the server's sort order within each partition."""
    if not sort.favourites_first or not rows:
        return rows
    favourites = tuple(r for r in rows if r.favourite)
    rest = tuple(r for r in rows if not r.favourite)
    return favourites + rest


# ---------------------------------------------------------------------------
# Sort/filter persistence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryViewState:
    """The persisted slice of LibraryScreen state.

    Frozen so the screen can hold snapshots without defensive
    copies; `save_view_state` / `load_view_state` round-trip
    this through atomic JSON writes against
    :data:`DEFAULT_VIEW_STATE_PATH`.
    """

    sort: LibrarySort = field(default_factory=LibrarySort)
    filters: LibraryFilters = field(default_factory=LibraryFilters)
    schema_version: int = _VIEW_STATE_SCHEMA_VERSION


class LibraryViewStateStore:
    """Atomic on-disk store for :class:`LibraryViewState`.

    Mirrors the contract of :class:`care.runtime.RunStateStore`:

    * ``DEFAULT_VIEW_STATE_PATH`` default.
    * Atomic writes via tempfile + `os.replace`.
    * Tolerant `load()` — returns ``None`` for every failure
      mode (file missing, malformed JSON, schema mismatch).
    * `clear()` returns `bool` (idempotent).
    * Thread-safe via an internal lock so the LibraryScreen +
      Settings worker can both touch it without races.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            self._path = resolve_default_view_state_path()
        else:
            self._path = Path(str(path)).expanduser()
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def save(self, state: LibraryViewState) -> Path:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(_state_to_dict(state), sort_keys=True)
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=".library_view-",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fp:
                    fp.write(payload)
                os.replace(tmp_name, self._path)
            except OSError:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
            return self._path

    def load(self) -> Optional[LibraryViewState]:
        with self._lock:
            try:
                raw = self._path.read_text(encoding="utf-8")
            except OSError:
                return None
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if not isinstance(data, dict):
                return None
            if data.get("schema_version") != _VIEW_STATE_SCHEMA_VERSION:
                return None
            try:
                return _state_from_dict(data)
            except (LibraryViewError, KeyError, TypeError, ValueError):
                return None

    def clear(self) -> bool:
        with self._lock:
            try:
                self._path.unlink()
                return True
            except FileNotFoundError:
                return False


def save_view_state(
    state: LibraryViewState,
    *,
    path: Path | str | None = None,
) -> Path:
    """Persist ``state`` to ``path`` (default
    :data:`DEFAULT_VIEW_STATE_PATH`). Returns the resolved path."""
    return LibraryViewStateStore(path).save(state)


def load_view_state(
    path: Path | str | None = None,
) -> Optional[LibraryViewState]:
    """Load the persisted state from ``path``. Returns ``None``
    on every failure mode (file missing / malformed JSON /
    schema mismatch / unknown sort field) — the LibraryScreen
    just falls back to defaults."""
    return LibraryViewStateStore(path).load()


def _state_to_dict(state: LibraryViewState) -> dict[str, Any]:
    return {
        "schema_version": _VIEW_STATE_SCHEMA_VERSION,
        "sort": {
            "field": state.sort.field,
            "direction": state.sort.direction,
            "favourites_first": state.sort.favourites_first,
        },
        "filters": {
            "domain": state.filters.domain,
            "tags": list(state.filters.tags),
            "status": state.filters.status,
            "favourites_only": state.filters.favourites_only,
            "search": state.filters.search,
            "namespace": state.filters.namespace,
            "channel": state.filters.channel,
        },
    }


def _state_from_dict(data: dict[str, Any]) -> LibraryViewState:
    sort_block = data.get("sort") or {}
    if not isinstance(sort_block, dict):
        sort_block = {}
    filt_block = data.get("filters") or {}
    if not isinstance(filt_block, dict):
        filt_block = {}

    sort = LibrarySort(
        field=str(sort_block.get("field") or "last_run_at"),
        direction=("asc" if sort_block.get("direction") == "asc" else "desc"),
        favourites_first=bool(sort_block.get("favourites_first", True)),
    )

    tags_raw = filt_block.get("tags") or []
    tags = (
        tuple(str(t) for t in tags_raw)
        if isinstance(tags_raw, (list, tuple))
        else ()
    )
    status = filt_block.get("status")
    if isinstance(status, str) and status not in _VALID_STATUSES:
        status = None

    filters = LibraryFilters(
        domain=_optional_str(filt_block.get("domain")),
        tags=tags,
        status=status if isinstance(status, str) else None,
        favourites_only=bool(filt_block.get("favourites_only", False)),
        search=str(filt_block.get("search") or ""),
        namespace=_optional_str(filt_block.get("namespace")),
        channel=str(filt_block.get("channel") or "latest"),
    )

    return LibraryViewState(sort=sort, filters=filters)


def _optional_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


# ---------------------------------------------------------------------------
# Client-side filter mutators
# ---------------------------------------------------------------------------


def with_domain(filters: LibraryFilters, domain: Optional[str]) -> LibraryFilters:
    return replace(filters, domain=domain or None)


def with_tags(
    filters: LibraryFilters, tags: Iterable[str]
) -> LibraryFilters:
    cleaned = tuple(t.strip() for t in tags if t and t.strip())
    return replace(filters, tags=cleaned)


def with_status(
    filters: LibraryFilters, status: Optional[str]
) -> LibraryFilters:
    if status and status not in _VALID_STATUSES:
        raise LibraryViewError(
            f"unknown status {status!r}; valid: {sorted(_VALID_STATUSES)}"
        )
    return replace(filters, status=status or None)


def with_favourites_only(
    filters: LibraryFilters, value: bool
) -> LibraryFilters:
    return replace(filters, favourites_only=bool(value))


def with_search(filters: LibraryFilters, query: str) -> LibraryFilters:
    return replace(filters, search=query)


def clear_filters(filters: LibraryFilters) -> LibraryFilters:
    """Reset the filtering knobs while preserving namespace +
    channel (operator-level settings the user didn't choose)."""
    return LibraryFilters(
        namespace=filters.namespace,
        channel=filters.channel,
    )


__all__ = [
    "DEFAULT_VIEW_STATE_PATH",
    "LibraryFilters",
    "LibraryRow",
    "LibrarySort",
    "LibraryView",
    "LibraryViewError",
    "LibraryViewState",
    "LibraryViewStateStore",
    "clear_filters",
    "fetch_library_view",
    "load_view_state",
    "resolve_default_view_state_path",
    "parse_library_row",
    "save_view_state",
    "with_domain",
    "with_favourites_only",
    "with_search",
    "with_status",
    "with_tags",
]
