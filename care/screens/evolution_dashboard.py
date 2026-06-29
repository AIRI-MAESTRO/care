"""EvolutionDashboard — list of recent + active evolution runs
(TODO §5 P0).

The home screen for evolution: one DataTable showing every run
the user's account has visible on the platform, sorted by
recency. From a row the user can:

* ``Enter`` — push :class:`care.screens.evolution.EvolutionScreen`
  for the highlighted run (resumes the SSE stream when the run
  is still active, or shows the final Pareto front for a
  finished one).
* ``s`` — stop a running evolution. Duck-types
  `CarePlatform.cancel(evolution_id)` against the configured
  facade; falls back to an info toast when the SDK doesn't
  expose the method yet (cross-module ask filed in §10).

Auto-refresh fires every ``DEFAULT_REFRESH_INTERVAL_SECONDS``
so a long-running watch surface stays current. Manual ``r``
binding re-runs the fetch on demand.

Construction is duck-typed against the platform facade:

* `platform.list_evolutions()` is called when present and
  iterated for the rows. When absent (older SDK), the screen
  renders a friendly placeholder + the user can still navigate
  to a specific run via `/evolution <run_id>` from the chat.
* Per-row data is read off a `dict | object` projection
  identical to the SSE event shape — we tolerate both so the
  cross-module ask doesn't block this screen.

Pushed by:

* `/evolution` slash command (no args) — handled by
  `care.screens.chat._cmd_evolution` once the dashboard ships
  (today the command requires a `<run_id>`; that path stays
  + the no-arg path lands here).
* `CareApp.action_palette_open_evolution` — palette entry
  toggle that until iter 7 toasted a "not yet implemented"
  warning. Now pushes the dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.text import Text

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Static, Tab, Tabs

from care.runtime.i18n import t
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader

_log = logging.getLogger("care.screen.evolution_dashboard")


_COLUMNS: tuple[str, ...] = (
    "Status", "Run", "Base chain", "Gen", "Best", "Current",
    "Valid", "Invalid", "Started", "Elapsed",
)


def _localized_columns() -> tuple[str, ...]:
    """The :data:`_COLUMNS` headers resolved through the i18n catalog
    (same order). ``_COLUMNS`` remains the stable English ``key=`` so
    DataTable row lookups don't shift when the UI language changes."""
    return (
        t("evolutionDashboard.col.status"),
        t("evolutionDashboard.col.run"),
        t("evolutionDashboard.col.baseChain"),
        t("evolutionDashboard.col.gen"),
        t("evolutionDashboard.col.best"),
        t("evolutionDashboard.col.current"),
        t("evolutionDashboard.col.valid"),
        t("evolutionDashboard.col.invalid"),
        t("evolutionDashboard.col.started"),
        t("evolutionDashboard.col.elapsed"),
    )

# Status → (colour token, glyph). Colour tokens are Rich-style
# names so terminals without truecolour still pick a sensible
# named ANSI hue. The glyph stays ``●`` everywhere so the column
# width is stable regardless of status length.
_STATUS_COLOURS: dict[str, str] = {
    # green — actively executing on the runner pool.
    "running": "green",
    "dispatching": "green",
    "initializing": "green",
    "preparing": "green",
    # yellow — waiting to start (queued / prepared / paused).
    "queued": "yellow",
    "pending": "yellow",
    "prepared": "yellow",
    "paused": "yellow",
    # blue — finished successfully.
    "completed": "blue",
    "accepted": "blue",
    "finished": "blue",
    # gray — user-stopped or never ran.
    "cancelled": "grey50",
    "stopped": "grey50",
    "skipped": "grey50",
    # red — terminal failure.
    "failed": "red",
    "error": "red",
    "preparation_failed": "red",
    "submit_failed": "red",
    "stream_failed": "red",
}
_STATUS_DEFAULT_COLOUR = "grey50"


def _format_status_cell(status: str | None) -> "Text":
    """Render a status cell with a coloured ``●`` prefix.

    Returns a ``rich.text.Text`` so DataTable picks up the
    colour markup; falls back to a plain ``—`` em-dash for
    rows that arrived without a status."""
    from rich.text import Text

    label = (status or "").strip()
    if not label:
        cell = Text("—", style=_STATUS_DEFAULT_COLOUR)
        return cell
    colour = _STATUS_COLOURS.get(label.lower(), _STATUS_DEFAULT_COLOUR)
    cell = Text()
    cell.append("● ", style=colour)
    cell.append(label, style="")
    return cell


@dataclass(frozen=True)
class EvolutionRunRow:
    """One row in the dashboard table. Frozen so the screen
    can hold snapshots without defensive copies.

    `run_id` is mandatory — every row must be addressable;
    the rest is best-effort projection from the platform's
    response (different SDK versions surface different keys
    so the screen reads each field defensively).
    """

    run_id: str
    base_chain_id: str = ""
    status: str = ""
    generation: int | None = None
    best_fitness: float | None = None
    current_fitness: float | None = None
    programs_valid: int | None = None
    programs_invalid: int | None = None
    started_at: float | None = None
    finished_at: float | None = None

    def elapsed_seconds(self, *, now: float | None = None) -> float | None:
        """Wall-clock since start. Stops the clock when the
        run finished. Returns `None` when `started_at` is
        absent."""
        if self.started_at is None:
            return None
        end = (
            self.finished_at if self.finished_at is not None
            else (now if now is not None else time.time())
        )
        return max(0.0, end - self.started_at)


def _iter_evolution_payloads(raw_rows: Any) -> list[Any]:
    """Normalise whatever ``platform.list_evolutions()`` returned
    into a flat list of per-run payloads.

    The Platform's ``GET /api/v1/evolutions`` returns a paginated
    envelope (``{"items": [...], "next_cursor": ...}``); older /
    stubbed surfaces hand back a bare list. Accept both — pull the
    list out of the common envelope keys, else treat the value as a
    list already. Anything else (``None``, scalar) collapses to ``[]``.
    """
    if raw_rows is None:
        return []
    if isinstance(raw_rows, dict):
        for key in ("items", "evolutions", "results", "data"):
            value = raw_rows.get(key)
            if isinstance(value, list):
                return value
        return []
    if isinstance(raw_rows, list):
        return raw_rows
    return []


def parse_evolution_run_row(raw: Any) -> EvolutionRunRow | None:
    """Project the platform's per-run payload into a typed
    row. Returns ``None`` when the payload doesn't carry
    enough info to be addressable (no ``id`` / ``run_id`` /
    ``evolution_id``).

    Accepts dicts or attribute-shaped objects so the screen
    can consume both `list_evolutions()`-returns-dicts and
    `list_evolutions()`-returns-typed-objects SDK variants.
    """

    def _read(name: str) -> Any:
        if isinstance(raw, dict):
            return raw.get(name)
        return getattr(raw, name, None)

    run_id = (
        _read("evolution_id")
        or _read("run_id")
        or _read("id")
    )
    if not run_id:
        return None
    started_raw = _read("started_at") or _read("created_at")
    finished_raw = _read("finished_at") or _read("completed_at")
    # Use explicit None checks instead of ``or`` chains so legit
    # falsy values (``0`` for the very first generation, ``0.0``
    # for an honest fitness floor) survive the projection
    # instead of getting collapsed to ``None``.
    fitness_raw = _read("best_fitness")
    if fitness_raw is None:
        fitness_raw = _read("fitness")
    generation_raw = _read("generation")
    if generation_raw is None:
        generation_raw = _read("gen")

    # Per-run live metrics: read top-level first, then the nested
    # ``metrics`` blob some list payloads carry. Usually absent on the
    # bare list endpoint → rendered as "—".
    metrics = _read("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}

    def _read_metric(name: str) -> Any:
        top = _read(name)
        return top if top is not None else metrics.get(name)

    return EvolutionRunRow(
        run_id=str(run_id),
        base_chain_id=str(
            _read("base_chain_id") or _read("chain_id") or ""
        ),
        status=str(_read("status") or ""),
        generation=_coerce_int(generation_raw),
        best_fitness=_coerce_float(fitness_raw),
        current_fitness=_coerce_float(_read_metric("current_fitness")),
        programs_valid=_coerce_int(_read_metric("programs_valid")),
        programs_invalid=_coerce_int(_read_metric("programs_invalid")),
        started_at=_coerce_float(started_raw),
        finished_at=_coerce_float(finished_raw),
    )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    """Coerce ``value`` into a float epoch seconds.

    Accepts:

    * Real numbers (``int`` / ``float``) — passed through.
    * Numeric strings (``"1234.5"``) — parsed.
    * ISO-8601 timestamps (``"2026-06-11T07:20:57Z"`` and
      Python's ``+00:00`` variant) — converted to epoch seconds
      so the dashboard's ``Started`` + ``Elapsed`` columns can
      render values the Platform's chain-experiment route
      reports as strings.

    Returns ``None`` for anything unparseable so the caller's
    fall-through (empty cell) still works.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # ``fromisoformat`` requires Python 3.11+ to accept the
        # trailing ``Z``. The Platform writes both ``Z`` and
        # ``+00:00`` shapes; normalise so we accept either on
        # older builds too.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            from datetime import datetime

            return datetime.fromisoformat(text).timestamp()
        except (TypeError, ValueError):
            return None
    return None


@dataclass
class _DashboardState:
    """Mutable per-mount state. Kept off the screen body so
    tests can construct + drive it without a Textual
    instance."""

    rows: list[EvolutionRunRow] = field(default_factory=list)
    last_error: str | None = None
    last_fetched_at: float | None = None
    is_loading: bool = False
    # §5 P1 — multi-select for the compare-runs flow. Ordered
    # so the user sees the same ordering they marked them in
    # (the compare modal uses the first as "left", second as
    # "right"). Capped at 2 by the action handler.
    selected_run_ids: list[str] = field(default_factory=list)
    # Set of archived run ids; persisted to disk via
    # ``_load_archive`` / ``_save_archive``. Filtered out of
    # the visible table unless ``show_archived`` is on.
    archived_run_ids: set[str] = field(default_factory=set)
    show_archived: bool = False


def _archive_path() -> Path:
    """Where the archived-run-id list lives across sessions.

    Lives next to the rest of CARE's user config so wipes are
    discoverable via ``~/.config/care``. Each launch reads it
    on mount; ``archive_run`` / unarchive writes back."""
    import os

    home = Path(os.environ.get("HOME") or "~").expanduser()
    return home / ".config" / "care" / "evolution_archive.json"


def _load_archive() -> set[str]:
    """Read the persisted archived-run-id set. Missing /
    malformed files return an empty set so the dashboard
    always opens clean."""
    path = _archive_path()
    if not path.is_file():
        return set()
    try:
        import json as _json

        data = _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(data, dict):
        return set()
    archived = data.get("archived") or []
    if not isinstance(archived, list):
        return set()
    return {str(x) for x in archived if x}


def _save_archive(archived: set[str]) -> None:
    """Persist the archived-run-id set. Failures are silently
    logged at WARN — losing the archive list is not worth
    surfacing a modal to the user."""
    path = _archive_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json

        path.write_text(
            _json.dumps({"archived": sorted(archived)}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to persist evolution archive: %s", exc)


def _format_started(started: float | None) -> str:
    if started is None:
        return "—"
    return time.strftime("%H:%M:%S", time.localtime(started))


def _format_elapsed(elapsed: float | None) -> str:
    if elapsed is None:
        return "—"
    secs = int(elapsed)
    if secs < 60:
        return f"{secs}s"
    mins, s = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {s}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m"


def _format_fitness(fitness: float | None) -> str:
    if fitness is None:
        return "—"
    return f"{fitness:.3f}"


def _format_gen(gen: int | None) -> str:
    if gen is None:
        return "—"
    return str(gen)


def _format_count(value: int | None) -> str:
    """Render an optional non-negative count; "—" when unknown."""
    if value is None or value < 0:
        return "—"
    return str(value)


def _reconcile_table(
    table: DataTable,
    desired: list[tuple[str, tuple[Any, ...]]],
    column_keys: tuple[str, ...],
) -> None:
    """Diff ``table`` toward ``desired`` instead of clear()+re-add.

    ``desired`` is the ordered list of ``(row_key, cells)`` the table
    should show after this render; ``column_keys`` are the stable
    ``add_column(key=…)`` keys in the same order as each ``cells`` tuple.

    Rows are keyed (``add_row(key=…)``) so we can:

    * ``update_cell`` only the cells whose value actually changed on a row
      that already exists (no churn → the cursor + scroll position survive);
    * ``add_row`` brand-new keys;
    * ``remove_row`` keys that dropped out of ``desired``.

    The newly-added rows land at the bottom (Textual appends), so a render
    whose ordering changed can leave new rows out of their sorted slot until
    the next full reconcile — acceptable for the dashboard, where new runs
    are rare relative to the per-tick metric updates this protects.

    Cursor preservation: Textual keeps the cursor on the same *coordinate*,
    so removing a row above the cursor would shift the highlight. We snapshot
    the cursor's row key before mutating and move the cursor back onto it
    afterwards when that row still exists.
    """
    from textual.widgets.data_table import RowKey

    existing_keys = {str(k.value) for k in table.rows}
    desired_keys = {key for key, _ in desired}

    # Snapshot the cursor's row so we can restore it after add/remove churn.
    cursor_row_key: str | None = None
    if table.row_count and table.cursor_type == "row":
        try:
            cell_key = table.coordinate_to_cell_key(table.cursor_coordinate)
            cursor_row_key = str(cell_key.row_key.value)
        except Exception:
            cursor_row_key = None

    # Drop rows that vanished.
    for key in existing_keys - desired_keys:
        try:
            table.remove_row(key)
        except Exception:
            pass

    # Update changed cells in place; append genuinely new rows.
    for key, cells in desired:
        if key in existing_keys:
            for col_key, value in zip(column_keys, cells):
                try:
                    current = table.get_cell(key, col_key)
                except Exception:
                    current = None
                # ``Text`` cells (status badge) aren't cheaply comparable —
                # only skip the write when both sides are equal *and* the
                # same plain-string type; otherwise always write so colour
                # markup never goes stale.
                if isinstance(current, str) and isinstance(value, str):
                    if current == value:
                        continue
                try:
                    table.update_cell(key, col_key, value)
                except Exception:
                    pass
        else:
            try:
                table.add_row(*cells, key=key)
            except Exception:
                pass

    # Restore the cursor onto its original row when it's still present.
    if cursor_row_key is not None and RowKey(cursor_row_key) in table.rows:
        try:
            table.move_cursor(
                row=table.get_row_index(cursor_row_key), scroll=False,
            )
        except Exception:
            pass


class EvolutionDashboard(Screen):
    """DataTable of recent + active evolution runs.

    Construct without args — the screen reads `app.platform`
    + auto-refreshes. Tests pass `auto_refresh_interval=0`
    to disable the timer so they can drive refreshes
    explicitly.
    """

    DEFAULT_REFRESH_INTERVAL_SECONDS = 5.0
    """How often the dashboard re-runs the fetch. 5s matches
    the StatusBar cadence (§1 P0)."""

    DEFAULT_CSS = """
    EvolutionDashboard #dashboard-error {
        padding: 0 1;
        color: $error;
    }
    EvolutionDashboard #dashboard-actions {
        height: 3;
        align: left middle;
        padding: 0 1;
    }
    EvolutionDashboard #dashboard-actions Button {
        margin: 0 1 0 0;
    }
    """

    BINDINGS = [
        Binding("enter", "open_run", "Open", show=True),
        Binding("s", "stop_run", "Stop", show=True),
        # §5 P1 — multi-select 2 runs for the compare flow.
        # `space` toggles the highlighted row; `c` opens the
        # compare modal when exactly 2 are picked.
        Binding(
            "space", "toggle_select", "Select for compare",
            show=True,
        ),
        Binding(
            "c", "compare_runs", "Compare", show=True,
        ),
        # Archive lifecycle — keeps cancelled / failed runs from
        # cluttering the inbox without losing them entirely.
        # ``a`` toggles archive on the highlighted row; the
        # ``Active`` / ``Archived`` tabs at the top of the body
        # flip between the two views (drives them via arrow
        # keys when the tab strip has focus, or click).
        Binding("a", "archive_run", "Archive", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        # ``escape`` is the conventional pop-screen — kept for
        # muscle memory. ``b`` exists too so the footer surfaces
        # a labelled "Back" hint alongside the cryptic ``Esc``.
        Binding("b", "back", "Back to chat", show=True),
        Binding("escape", "back", "Back", show=False),
    ]

    def __init__(
        self,
        *,
        auto_refresh_interval: float | None = None,
    ) -> None:
        super().__init__()
        self.state = _DashboardState()
        self.auto_refresh_interval = (
            self.DEFAULT_REFRESH_INTERVAL_SECONDS
            if auto_refresh_interval is None
            else auto_refresh_interval
        )
        # Per-binding action log — tests + telemetry read
        # this rather than scraping screen internals.
        self.action_log: list[tuple[str, str]] = []
        self._interval_timer: Any = None
        # Whether the screen is currently backgrounded (another screen on
        # top). Drives the suspend/resume poll-pause + the resume re-fetch
        # guard so the initial post-mount ``ScreenResume`` is a no-op.
        self._suspended: bool = False

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Vertical(id="dashboard-body"):
            yield Tabs(
                Tab(t("evolutionDashboard.tabActive"), id="tab-active"),
                Tab(t("evolutionDashboard.tabArchived"), id="tab-archived"),
                id="dashboard-tabs",
            )
            yield DataTable(id="dashboard-table")
            yield Static(" ", id="dashboard-empty")
            yield Static(" ", id="dashboard-error")
            # Visible action buttons — same handlers as the
            # keyboard bindings (``b`` / ``a`` / ``s`` / ``r``).
            # Click affordances make the screen self-describing
            # for users who haven't memorised the bindings yet.
            with Horizontal(id="dashboard-actions"):
                yield Button(
                    t("evolutionDashboard.backToChat"),
                    id="dashboard-btn-back",
                )
                yield Button(
                    t("evolutionDashboard.archive"),
                    id="dashboard-btn-archive",
                )
                yield Button(
                    t("evolutionDashboard.stopRun"),
                    id="dashboard-btn-stop",
                    variant="warning",
                )
                yield Button(
                    t("evolutionDashboard.refresh"),
                    id="dashboard-btn-refresh",
                    variant="primary",
                )
        yield CareFooter()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "dashboard-btn-back":
            self.action_back()
        elif bid == "dashboard-btn-archive":
            self.action_archive_run()
        elif bid == "dashboard-btn-stop":
            self.action_stop_run()
        elif bid == "dashboard-btn-refresh":
            self.action_refresh()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="EvolutionDashboard",
                breadcrumb=(
                    t("evolutionDashboard.breadcrumbRoot"),
                    t("evolutionDashboard.breadcrumb"),
                ),
            )
        except Exception:
            pass
        try:
            table = self.query_one("#dashboard-table", DataTable)
            # `_COLUMNS` stays the stable English `key=`; the visible
            # header is localized so row-key lookups don't shift with the
            # UI language.
            for key, label in zip(_COLUMNS, _localized_columns()):
                table.add_column(label, key=key)
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        # Hydrate the archived-id set so the first paint
        # already filters runs the user previously hid.
        self.state.archived_run_ids = _load_archive()
        # Defer the first refresh until after mount so
        # `query_one` finds the body.
        self.app.call_after_refresh(self.refresh_rows)
        if self.auto_refresh_interval > 0:
            self._interval_timer = self.set_interval(
                self.auto_refresh_interval, self.refresh_rows,
            )

    def on_unmount(self) -> None:
        timer = self._interval_timer
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._interval_timer = None

    def on_screen_suspend(self) -> None:
        """Pause the auto-refresh poll while another screen is on top —
        no point hitting the Platform every 10s for a table the user
        can't see. Resumed in :meth:`on_screen_resume`."""
        self._suspended = True
        timer = self._interval_timer
        if timer is not None:
            try:
                timer.pause()
            except Exception:
                pass

    def on_screen_resume(self) -> None:
        """Resume the auto-refresh poll when the dashboard is back on
        top, and fire one immediate refresh so the table isn't stale
        after the pause. Guarded on a prior suspend so the initial
        ``ScreenResume`` (which Textual fires right after mount) doesn't
        double up on the ``on_mount`` first fetch."""
        if not self._suspended:
            return
        self._suspended = False
        timer = self._interval_timer
        if timer is not None:
            try:
                timer.resume()
            except Exception:
                pass
        self.refresh_rows()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_rows(self) -> None:
        """Spawn a fetch worker. Idempotent — `exclusive=True`
        cancels any in-flight worker before starting fresh."""
        # Show Textual's native animated loading overlay only on the FIRST
        # fetch (empty table) — gating on "no rows yet" keeps the 5s
        # auto-refresh from flickering the overlay over a populated table
        # every tick. Cleared in ``_apply_view`` once the fetch resolves.
        if not self.state.rows:
            try:
                self.query_one("#dashboard-table", DataTable).loading = True
            except Exception:
                pass
        self.run_worker(
            self._refresh(),
            name="dashboard_fetch",
            group="dashboard",
            exclusive=True,
            exit_on_error=False,
        )

    async def _refresh(self) -> None:
        platform = getattr(self.app, "platform", None)
        if platform is None:
            self.state.last_error = (
                "Platform facade not configured — "
                "set CARE_PLATFORM__BASE_URL first."
            )
            self.state.rows = []
            self._apply_view()
            return
        list_fn = getattr(platform, "list_evolutions", None)
        if not callable(list_fn):
            self.state.last_error = (
                "Platform SDK doesn't expose list_evolutions() "
                "yet — open /evolution <run_id> from chat to "
                "watch a known run while the platform follow-up "
                "lands."
            )
            self.state.rows = []
            self._apply_view()
            return
        self.state.is_loading = True
        try:
            raw_rows: Any = await asyncio.to_thread(list_fn)
        except Exception as exc:  # noqa: BLE001
            self.state.last_error = (
                f"list_evolutions failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self.state.rows = []
            self.state.is_loading = False
            self._apply_view()
            return
        parsed: list[EvolutionRunRow] = []
        for raw in _iter_evolution_payloads(raw_rows):
            row = parse_evolution_run_row(raw)
            if row is not None:
                parsed.append(row)
        # Newest-first by started_at (rows without a start
        # land at the bottom).
        parsed.sort(
            key=lambda r: (
                -(r.started_at if r.started_at is not None else 0),
                r.run_id,
            ),
        )
        parsed = await self._hydrate_archived_rows(platform, parsed)
        self.state.rows = parsed
        self.state.last_error = None
        self.state.last_fetched_at = time.time()
        self.state.is_loading = False
        self._apply_view()

    async def _hydrate_archived_rows(
        self,
        platform: Any,
        parsed: list[EvolutionRunRow],
    ) -> list[EvolutionRunRow]:
        """Ensure locally archived run ids appear in the inbox.

        Archive is persisted client-side; when ``list_evolutions``
        fails or returns a partial list the Archived tab would
        otherwise look empty even though ids were archived."""
        archived = self.state.archived_run_ids
        if not archived:
            return parsed
        by_id = {r.run_id: r for r in parsed}
        get_fn = getattr(platform, "get_evolution", None)
        if not callable(get_fn):
            return parsed
        for run_id in archived:
            if run_id in by_id:
                continue
            try:
                raw = await asyncio.to_thread(get_fn, run_id)
                row = parse_evolution_run_row(raw)
                if row is not None:
                    by_id[row.run_id] = row
                    continue
            except Exception as exc:  # noqa: BLE001
                _log.debug(
                    "hydrate archived run %s failed: %s",
                    run_id,
                    exc,
                )
            by_id[run_id] = EvolutionRunRow(
                run_id=run_id,
                status="archived",
            )
        merged = list(by_id.values())
        merged.sort(
            key=lambda r: (
                -(r.started_at if r.started_at is not None else 0),
                r.run_id,
            ),
        )
        return merged

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _apply_view(self) -> None:
        try:
            table = self.query_one("#dashboard-table", DataTable)
            empty = self.query_one("#dashboard-empty", Static)
            error = self.query_one("#dashboard-error", Static)
        except Exception:
            return
        # The fetch has resolved (this is its sink) — drop the loading
        # overlay armed in ``refresh_rows``.
        table.loading = False
        now = time.time()
        archived = self.state.archived_run_ids
        show_archived = self.state.show_archived
        # Two-tab semantics: ``Active`` (default) hides archived
        # rows; ``Archived`` shows ONLY archived rows. Each tab
        # gets a clean inbox instead of mixing both views.
        if show_archived:
            visible_rows = [r for r in self.state.rows if r.run_id in archived]
        else:
            visible_rows = [r for r in self.state.rows if r.run_id not in archived]
        hidden_count = len(self.state.rows) - len(visible_rows)
        # Build the desired row set and reconcile in place rather than
        # clear()+re-add — the 10s auto-refresh would otherwise wipe the
        # user's selection + scroll position on every tick (rows already
        # carry a stable ``key=run_id``).
        desired: list[tuple[str, tuple[Any, ...]]] = [
            (
                row.run_id,
                (
                    _format_status_cell(row.status),
                    row.run_id,
                    row.base_chain_id or "—",
                    _format_gen(row.generation),
                    _format_fitness(row.best_fitness),
                    _format_fitness(row.current_fitness),
                    _format_count(row.programs_valid),
                    _format_count(row.programs_invalid),
                    _format_started(row.started_at),
                    _format_elapsed(row.elapsed_seconds(now=now)),
                ),
            )
            for row in visible_rows
        ]
        _reconcile_table(table, desired, _COLUMNS)
        is_empty = not visible_rows
        empty.display = is_empty and not self.state.last_error
        if is_empty and not self.state.last_error:
            if show_archived:
                empty.update(t("evolutionDashboard.emptyArchived"))
            elif hidden_count > 0:
                empty.update(
                    t("evolutionDashboard.emptyAllArchived", count=hidden_count),
                )
            else:
                empty.update(t("evolutionDashboard.emptyNoRuns"))
        else:
            empty.update(" ")
        if self.state.last_error:
            error.update(f"⚠ {self.state.last_error}")
            error.display = True
        else:
            error.update(" ")
            error.display = False

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @property
    def current_run(self) -> EvolutionRunRow | None:
        if not self.state.rows:
            return None
        try:
            table = self.query_one("#dashboard-table", DataTable)
        except Exception:
            return None
        idx = table.cursor_row
        if idx is None or idx < 0 or idx >= len(self.state.rows):
            return None
        return self.state.rows[idx]

    def action_open_run(self) -> None:
        run = self.current_run
        if run is None:
            return
        self.action_log.append(("open_run", run.run_id))
        try:
            from care.screens.evolution import EvolutionScreen

            # Push in observe-only mode so we WATCH the existing
            # run instead of re-submitting a fresh evolution.
            # Before this guard, opening a row from the dashboard
            # would call ``EvolutionScreen.on_mount`` →
            # ``start_evolution`` and create a duplicate run.
            self.app.push_screen(
                EvolutionScreen(
                    base_chain_id=run.base_chain_id or run.run_id,
                    observe_evolution_id=run.run_id,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._toast(
                t("evolutionDashboard.openFailed", error=exc),
                severity="error",
            )

    def action_stop_run(self) -> None:
        run = self.current_run
        if run is None:
            return
        self.action_log.append(("stop_run", run.run_id))
        platform = getattr(self.app, "platform", None)
        cancel_fn = getattr(platform, "cancel", None) if platform else None
        if not callable(cancel_fn):
            self._toast(
                t("evolutionDashboard.cancelUnavailable"),
                severity="warning",
            )
            return
        self.run_worker(
            self._stop_worker(run.run_id, cancel_fn),
            name="dashboard_stop",
            group="dashboard",
            exclusive=False,
            exit_on_error=False,
        )

    async def _stop_worker(self, run_id: str, cancel_fn: Any) -> None:
        try:
            await asyncio.to_thread(cancel_fn, run_id)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "dashboard stop run=%s failed: %s", run_id, exc,
                exc_info=False,
            )
            self._toast(
                t("evolutionDashboard.stopFailed", error=exc),
                severity="warning",
            )
            return
        self._toast(
            t("evolutionDashboard.stopRequested", runId=run_id),
            severity="info",
        )
        self.refresh_rows()

    def action_toggle_select(self) -> None:
        """§5 P1 — toggle the highlighted row's membership in
        the compare-selection set. Capped at 2; the third
        attempt toasts a hint instead of silently bouncing
        the user's earliest pick."""
        run = self.current_run
        if run is None:
            return
        selected = self.state.selected_run_ids
        if run.run_id in selected:
            selected.remove(run.run_id)
            self.action_log.append(("deselect", run.run_id))
        else:
            if len(selected) >= 2:
                self._toast(
                    t("evolutionDashboard.compareLimit"),
                    severity="info",
                )
                return
            selected.append(run.run_id)
            self.action_log.append(("select", run.run_id))

    def action_compare_runs(self) -> None:
        """§5 P1 — open the compare modal with the 2 selected
        runs. Toasts a hint when fewer than 2 are picked."""
        selected = list(self.state.selected_run_ids)
        if len(selected) < 2:
            need = 2 - len(selected)
            key = (
                "evolutionDashboard.compareNeed.one" if need == 1
                else "evolutionDashboard.compareNeed.many"
            )
            self._toast(
                t(key, count=need),
                severity="info",
            )
            return
        if len(selected) > 2:
            # Defensive — the toggle path caps at 2 already.
            selected = selected[:2]
        self.action_log.append(
            ("compare_runs", ",".join(selected)),
        )
        try:
            from care.screens.evolution_compare import (
                EvolutionCompareModal,
            )
        except Exception as exc:  # noqa: BLE001
            self._toast(
                t("evolutionDashboard.compareOpenFailed", error=exc),
                severity="error",
            )
            return
        platform = getattr(self.app, "platform", None)
        try:
            self.app.push_screen(EvolutionCompareModal(
                left_run_id=selected[0],
                right_run_id=selected[1],
                platform=platform,
            ))
        except Exception as exc:  # noqa: BLE001
            self._toast(
                t("evolutionDashboard.comparePushFailed", error=exc),
                severity="error",
            )

    def action_refresh(self) -> None:
        self.action_log.append(("refresh", ""))
        self.refresh_rows()

    def action_back(self) -> None:
        """Pop the dashboard so the user lands back on the chat
        (or whatever screen pushed this). The binding shows as
        "Back to chat" in the footer (``b``); ``escape`` does
        the same."""
        self.action_log.append(("back", ""))
        try:
            self.app.pop_screen()
        except Exception:
            pass

    def action_archive_run(self) -> None:
        """Toggle archive state for the highlighted row.

        Archived runs stay on the Platform but disappear from
        the dashboard's default view; press ``t`` to bring
        them back into sight. Persisted to
        ``~/.config/care/evolution_archive.json`` so the
        archive list survives restarts."""
        run = self.current_run
        if run is None:
            return
        archived = self.state.archived_run_ids
        if run.run_id in archived:
            archived.discard(run.run_id)
            self.action_log.append(("unarchive_run", run.run_id))
            toast_key = "evolutionDashboard.runUnarchived"
        else:
            archived.add(run.run_id)
            self.action_log.append(("archive_run", run.run_id))
            toast_key = "evolutionDashboard.runArchived"
        _save_archive(archived)
        self._apply_view()
        self._toast(t(toast_key, runId=run.run_id[:18]), severity="info")

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Tab switch is the canonical "show archived?" toggle.

        ``Active`` (default) filters out archived runs so the
        inbox stays focused on live + recent work; ``Archived``
        shows only archived runs so the user can audit them or
        un-archive (``a`` again on the row)."""
        tab_id = getattr(event.tab, "id", None)
        new_state = tab_id == "tab-archived"
        if new_state == self.state.show_archived:
            return
        self.state.show_archived = new_state
        self.action_log.append(("tab_changed", tab_id or ""))
        self._apply_view()

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected,
    ) -> None:
        if event.data_table.id != "dashboard-table":
            return
        self.action_open_run()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _toast(self, message: str, *, severity: str = "info") -> None:
        push = getattr(self.app, "push_toast", None)
        if callable(push):
            try:
                push(message, severity=severity)
                return
            except Exception:
                pass
        _log.info(
            "EvolutionDashboard toast [%s]: %s", severity, message,
        )


__all__ = [
    "EvolutionDashboard",
    "EvolutionRunRow",
    "parse_evolution_run_row",
]
