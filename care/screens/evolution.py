"""EvolutionScreen — submit + watch a GigaEvo Platform run
(TODO §1.1 P0.24).

Pushed when the user invokes `Evolve` on a saved agent. Calls
:meth:`CarePlatform.start_evolution(...)` on mount and
subscribes to :meth:`CarePlatform.stream_events(...)` SSE
events. Renders three panes:

* **Status** — `evolution_id`, current `status`, current
  generation, elapsed time.
* **Pareto front** — table of individuals ordered by primary
  objective.
* **Event log** — chronological stream of `generation_started`,
  `individual_evaluated`, `best_updated`, `completed`,
  `accepted`, etc.

The "Accept winner" action calls
:meth:`CarePlatform.accept_individual(...)` which promotes
the chosen individual to Memory's `stable` channel.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from care.evolution_session import (
    EvolutionConfig,
    EvolutionPlan,
    EvolutionProgressTracker,
    build_evolution_request,
    evolution_diff,
)
from care.micro_evolution import (
    Individual as _ParetoIndividual,
)
from care.micro_evolution import (
    compute_pareto_front,
)
from care.runtime.i18n import t
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader

_log = logging.getLogger("care.screens.evolution")

# Run statuses that mean "the runner hasn't produced metrics yet" — used
# to pick a context-aware empty-pane placeholder so the user sees
# "waiting for the runner…" instead of a blank rectangle.
_PRE_RUN_STATUSES = frozenset(
    {"", "submitting", "prepared", "preparing", "queued", "pending", "dispatching"}
)
_TERMINAL_DISPLAY_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "terminated", "error"}
)


def _format_objectives_inline(
    objectives: Any, *, cap: int = 3
) -> str:
    """Compact ``key=val`` projection of an individual's objectives for a
    table cell (capped, with an ellipsis when truncated). Empty when the
    individual carries no objectives (single-objective runs)."""
    try:
        pairs = list(objectives or ())
    except TypeError:
        return ""
    if not pairs:
        return ""
    out: list[str] = []
    for item in pairs[:cap]:
        try:
            key, value = item
            out.append(f"{key}={float(value):.3f}")
        except (TypeError, ValueError):
            continue
    if len(pairs) > cap:
        out.append("…")
    return " ".join(out)


@dataclass(frozen=True)
class EvolutionIndividual:
    """One Pareto-front row the screen renders."""

    individual_id: str
    generation: int = 0
    fitness: float | None = None
    objectives: tuple[tuple[str, float], ...] = ()
    summary: str = ""
    # §5 P0 — chain payload extracted from the SSE event when
    # the platform ships it. `None` means we haven't seen one
    # yet; the preview pane renders "(no chain content)" in
    # that case rather than blanking out.
    chain_dict: dict | None = None


@dataclass
class EvolutionRunState:
    """Mutable bookkeeping the screen exposes to tests + the
    future SaveAgentModal handoff."""

    base_chain_id: str = ""
    evolution_id: str = ""
    status: str = "submitting"
    generation: int = 0
    started_at: float | None = None
    # Wall-clock epoch seconds the Platform reports as the
    # experiment's actual start (parsed from the ISO ``started_at``
    # field on status / list payloads). Distinguished from
    # ``started_at`` above which tracks when the screen mounted
    # — needed because observe-mode opens existing runs whose
    # real start is hours in the past, so monotonic deltas from
    # mount would mislead the user.
    wall_started_at: float | None = None
    finished: bool = False
    individuals: list[EvolutionIndividual] = field(default_factory=list)
    accepted_id: str | None = None
    last_error: str | None = None
    events: list[tuple[str, dict]] = field(default_factory=list)
    # §5 P1 — running cost aggregator. Fed from `cost_tick` SSE
    # events (canonical) plus any `individual_evaluated` /
    # `best_updated` payload that carries token/cost data as a
    # fallback (some platforms attach usage to per-individual
    # events instead of dedicated cost ticks).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    # Latest valid / invalid program counters from the runner
    # (gigavolve Redis ``programs_valid_count`` /
    # ``programs_invalid_count`` lists). Drive the Programs
    # tab's pie chart. ``-1`` means "not reported yet" so the
    # pane can distinguish "no data" from a real ``0``.
    programs_valid: int = -1
    programs_invalid: int = -1
    # Latest current-iteration mean fitness (vs ``best_fitness``
    # which is the hall-of-fame). Shown alongside ``Best`` on
    # the Statistics card so the user can compare "what's the
    # population doing now" with "best so far".
    current_fitness: float | None = None
    # P0.4 — liveness. ``last_event_monotonic`` is bumped every time an
    # event is processed so the status pane can show "updated Ns ago"
    # (proving the screen is live, or revealing a stall). ``data_source``
    # records where the latest metrics came from ("platform" via
    # ``/results``, or "redis_probe" via the local gigavolve fallback) so
    # the user understands why charts may be sparse.
    last_event_monotonic: float | None = None
    data_source: str = ""
    # Platform runner that executed this run (shown in the metadata
    # card so observe-mode users can tell which runner produced it).
    runner_id: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def elapsed(self) -> float | None:
        """Seconds since the experiment started.

        Prefers the Platform-reported wall-clock start
        (``wall_started_at``) so observing an existing run
        shows the real run-time, not the time since the screen
        mounted. Falls back to monotonic-since-mount for fresh
        submits before the first status arrives."""
        if self.wall_started_at is not None:
            return max(0.0, time.time() - self.wall_started_at)
        if self.started_at is None:
            return None
        return max(0.0, time.monotonic() - self.started_at)


_TERMINAL_RUN_STATUSES = frozenset({
    "completed", "failed", "cancelled", "error",
    "preparation_failed", "submit_failed", "stream_failed",
})


def _coerce_status_from_payload(payload: dict) -> str | None:
    """Best-effort status string from a poll / snapshot payload."""
    for key in ("status",):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    nested = payload.get("data")
    if isinstance(nested, dict):
        inner = nested.get("status")
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
    return None


def _coerce_generation_from_payload(payload: dict) -> int | None:
    """Best-effort generation counter from heterogeneous payloads."""
    metrics = payload.get("metrics")
    sources: list[dict] = [payload]
    if isinstance(metrics, dict):
        sources.append(metrics)
    nested_results = payload.get("results")
    if isinstance(nested_results, dict):
        sources.append(nested_results)
        nested_metrics = nested_results.get("metrics")
        if isinstance(nested_metrics, dict):
            sources.append(nested_metrics)
    for source in sources:
        for key in (
            "generation",
            "gen",
            "current_generation",
        ):
            raw = source.get(key)
            if isinstance(raw, (int, float)) and raw >= 0:
                return int(raw)
    return None


def _accumulate_cost(
    state: EvolutionRunState, payload: dict,
) -> None:
    """§5 P1 — fold token + USD cost fields from a payload
    into the running aggregator.

    The platform SSE schema isn't finalized (§10 ask), so this
    helper accepts a handful of plausible field shapes:

    * Tokens — looks at top-level `prompt_tokens` /
      `completion_tokens` / `total_tokens`, or a nested
      `usage` mapping with the same keys. `total_tokens`
      without the prompt/completion split is folded into
      `completion_tokens` so the running total stays accurate
      even when the breakdown isn't shipped.
    * USD — top-level `cost_usd` / `cost` / `usd` /
      `total_cost`, or `usage.cost_usd`. Strings that parse
      as floats are accepted (Memory's JSON serialiser
      sometimes drops trailing zeros as strings).

    Each field is best-effort: malformed values are silently
    skipped so a single bad tick doesn't poison the running
    total. Empty payload is a no-op.
    """
    if not isinstance(payload, dict):
        return
    usage = payload.get("usage") if isinstance(
        payload.get("usage"), dict,
    ) else {}
    prompt = (
        payload.get("prompt_tokens")
        if payload.get("prompt_tokens") is not None
        else usage.get("prompt_tokens")
    )
    completion = (
        payload.get("completion_tokens")
        if payload.get("completion_tokens") is not None
        else usage.get("completion_tokens")
    )
    total_tokens = (
        payload.get("total_tokens")
        if payload.get("total_tokens") is not None
        else usage.get("total_tokens")
    )

    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    p = _to_int(prompt)
    c = _to_int(completion)
    t = _to_int(total_tokens)
    state.prompt_tokens += p
    state.completion_tokens += c
    # When only `total_tokens` is shipped, fold it into the
    # completion side so the aggregate is still right.
    if t and not (p or c):
        state.completion_tokens += t

    cost = None
    for key in ("cost_usd", "cost", "usd", "total_cost"):
        candidate = payload.get(key)
        if candidate is None and key in usage:
            candidate = usage.get(key)
        if candidate is not None:
            cost = candidate
            break
    if cost is None:
        return
    try:
        state.total_cost_usd += float(cost)
    except (TypeError, ValueError):
        pass


def _coerce_int(value: Any) -> int | None:
    """Best-effort `int` coercion. Returns `None` for `None`
    inputs + anything that can't be parsed. Used by the
    accept-winner flow to read `previous_version` /
    `new_version` off the platform response without
    crashing when the field is absent or non-numeric."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_platform_version(health: Any) -> str:
    """Pull a version string out of a platform ``/health`` payload.

    Accepts the three keys the Platform has shipped under (``version`` /
    ``platform_version`` / ``api_version``) and returns ``""`` when the
    payload isn't a dict or carries no usable version — so the metadata
    card simply omits the line instead of crashing."""
    if not isinstance(health, dict):
        return ""
    version = (
        health.get("version")
        or health.get("platform_version")
        or health.get("api_version")
    )
    return str(version) if version else ""


def _frontier_entry_to_individual(entry: Any) -> dict[str, Any] | None:
    """Map a ``frontier_programs`` record onto an ``_upsert_individual``
    payload so the hall-of-fame frontier also fills the Pareto / individuals
    table — not just the Versions tab.

    The Platform's ``/results`` metrics ship the only per-individual data a
    chain experiment has under ``frontier_programs`` (one improving record
    per generation): ``{generation, program_id, fitness, chain_config,
    mutation_summary}``. Returns ``None`` when there's no id to key the row
    on (the row can't be deduped/updated without one)."""
    if not isinstance(entry, dict):
        return None
    ind_id = (
        entry.get("program_id")
        or entry.get("id")
        or entry.get("individual_id")
    )
    if not ind_id:
        return None
    payload: dict[str, Any] = {"individual_id": str(ind_id)}
    gen = entry.get("generation")
    if isinstance(gen, int):
        payload["generation"] = gen
    fitness = entry.get("fitness")
    if isinstance(fitness, (int, float)):
        payload["fitness"] = fitness
    # Real runners emit ``chain_config``; the snapshot/hydrate fixtures use
    # ``chain_content`` — accept either so the preview pane has content.
    chain = entry.get("chain_config")
    if not isinstance(chain, dict):
        chain = entry.get("chain_content")
    if isinstance(chain, dict):
        payload["chain_content"] = chain
    summary = entry.get("mutation_summary") or entry.get("summary")
    if isinstance(summary, str) and summary:
        payload["summary"] = summary
    return payload


class EvolutionScreen(Screen):
    """Live evolution viewer + accept-winner control.

    Construct with `base_chain_id` (required) and optional
    spec fields. The host app provides `app.platform` (a
    :class:`care.platform.CarePlatform` facade); a `None`
    facade lands an error in :attr:`run.last_error` rather
    than crashing.
    """

    DEFAULT_CSS = """
    EvolutionScreen {
        layout: vertical;
    }
    EvolutionScreen #evolution-body {
        height: 1fr;
    }
    EvolutionScreen #evolution-status {
        width: 1fr;
        padding: 1 2;
    }
    EvolutionScreen #evolution-pareto {
        width: 2fr;
        padding: 1 2;
    }
    EvolutionScreen #evolution-events {
        width: 1fr;
        padding: 1 2;
    }
    EvolutionScreen #evolution-individual {
        height: auto;
        max-height: 14;
        padding: 0 2;
        background: $panel;
    }
    EvolutionScreen #evolution-individual-text {
        padding: 0 1;
    }
    EvolutionScreen #evolution-fitness {
        height: auto;
        max-height: 14;
        padding: 0 2;
        background: $panel;
    }
    EvolutionScreen #evolution-fitness-text {
        padding: 0 1;
    }
    EvolutionScreen #evolution-pareto-plot {
        height: auto;
        max-height: 16;
        padding: 0 2;
        background: $panel;
    }
    EvolutionScreen #evolution-pareto-plot-text {
        padding: 0 1;
    }
    /* TabbedContent + PlotWidget sizing — without ``height: 1fr``
       on both, the PlotWidget's internal canvas reports
       ``_canvas_size: None`` and ``_render_plot`` bails before
       drawing anything (this was why the Fitness tab stayed
       blank). The plot widget needs a sized container plus its
       own height to lay out the canvas. */
    EvolutionScreen #evolution-tabs {
        height: 1fr;
    }
    EvolutionScreen TabPane {
        height: 1fr;
    }
    EvolutionScreen #evolution-fitness-plot {
        height: 1fr;
        min-height: 12;
    }
    EvolutionScreen #evolution-fitness-plot-fallback {
        padding: 1 2;
    }
    EvolutionScreen #evolution-versions-body {
        height: 1fr;
    }
    EvolutionScreen #evolution-versions-list-pane {
        width: 30;
        padding: 0 1;
    }
    EvolutionScreen #evolution-versions-preview-pane {
        width: 1fr;
        padding: 0 1;
    }
    EvolutionScreen #evolution-versions-toolbar {
        height: 3;
        align: left middle;
    }
    EvolutionScreen #evolution-versions-toolbar Button {
        margin: 0 1 0 0;
    }
    EvolutionScreen #evolution-versions-preview {
        height: 1fr;
        background: $panel;
        padding: 1 2;
    }
    EvolutionScreen .pane-title {
        text-style: bold;
        color: $accent;
    }
    EvolutionScreen #evolution-actions {
        height: 3;
        padding: 0 1;
    }
    EvolutionScreen #evolution-actions Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("a", "accept_winner", "Accept", show=True),
        # §5 P1 — `x` exports the highlighted Pareto-front
        # individual's chain payload to disk via the
        # ExportChainModal (JSON / Python switch).
        Binding("x", "export_individual", "Export", show=True),
        Binding("c", "export_curve", "Export curve", show=True),
        # §5 P1 — `D` (uppercase) opens a DiffModal between
        # the seed/parent chain and the highlighted individual.
        Binding("D", "compare_to_parent", "Diff vs. parent", show=True),
        # ``z`` stops the run server-side AND archives it so it
        # drops out of the dashboard's Active tab in one
        # action — useful when the user knows a run is going
        # nowhere (e.g. LLM rate-limited) and just wants it
        # out of the way without leaving the live view.
        Binding("z", "archive_run", "Stop + Archive", show=True),
        Binding("escape", "cancel_evolution", "Cancel", show=True),
    ]

    class AcceptanceComplete(Message):
        """Posted after a successful accept-winner call so the
        host app can pop + refresh the LibraryScreen.

        §5 P0 — `chain_id`, `previous_version`, `new_version`
        are populated from the platform's accept response
        when the SDK ships them; absent fields read as
        ``None`` / ``""`` so older platforms keep working +
        the app handler degrades to the legacy toast format.
        """

        def __init__(
            self,
            evolution_id: str,
            individual_id: str,
            *,
            chain_id: str = "",
            previous_version: int | None = None,
            new_version: int | None = None,
        ) -> None:
            super().__init__()
            self.evolution_id = evolution_id
            self.chain_id = chain_id
            self.previous_version = previous_version
            self.new_version = new_version
            self.individual_id = individual_id

    def __init__(
        self,
        *,
        base_chain_id: str,
        max_iterations: int = 10,
        population_size: int = 8,
        evolution_mode: str = "full_chain",
        objectives: list[str] | None = None,
        validation_criteria: str = "",
        test_data_path: Any = None,
        validation_threshold: float | None = None,
        validation_type: str = "Continuous (0..1)",
        continuous_metric: str = "ROUGE-L",
        binary_method: str = "equality",
        target_column: str = "expected",
        base_chain_content: dict | None = None,
        base_chain_name: str = "",
        directions: dict[str, str] | None = None,
        observe_evolution_id: str | None = None,
        mutation_max_tokens: int | None = None,
    ) -> None:
        """Open the evolution run UI.

        Two modes:

        * Launch (default) — submit a fresh evolution on
          ``on_mount`` using the plan kwargs (``base_chain_id``,
          ``max_iterations``, …) and stream events for the run
          the Platform creates.
        * Observe — set ``observe_evolution_id`` to an existing
          run id (legacy ``evo_*`` or chain experiment
          ``exp_*``) to skip submission and only stream/poll
          its events. Used by the EvolutionDashboard's "open
          row" path so revisiting a run doesn't accidentally
          fire a duplicate submission.
        """
        super().__init__()
        self._observe_evolution_id = observe_evolution_id
        # `directions` maps objective key → "maximize" | "minimize".
        # Forwarded to `compute_pareto_front` so latency / cost
        # objectives use the right comparison direction. Missing
        # keys default to "maximize" inside the data layer.
        self.directions: dict[str, str] = dict(directions or {})
        self.run = EvolutionRunState(base_chain_id=base_chain_id)
        # Build the canonical data-layer plan up-front so the
        # request body + UI stay in sync. EvolutionConfig.objectives
        # is a tuple — coerce list-or-None.
        self.config = EvolutionConfig(
            evolution_mode=evolution_mode,  # type: ignore[arg-type]
            max_iterations=max_iterations,
            population_size=population_size,
            validation_criteria=validation_criteria,
            test_data_path=test_data_path,
            validation_threshold=validation_threshold,
            validation_type=validation_type,
            continuous_metric=continuous_metric,
            binary_method=binary_method,
            target_column=target_column,
            objectives=tuple(objectives or ()),
            mutation_max_tokens=mutation_max_tokens,
        )
        self.plan = EvolutionPlan(
            config=self.config,
            base_chain_entity_id=base_chain_id,
            base_chain_content=base_chain_content,
            base_chain_name=base_chain_name,
        )
        # Per-generation fitness aggregator. Fed from `_handle_event`
        # so the fitness-curve line in the status pane stays in
        # sync with the SSE stream.
        self.tracker = EvolutionProgressTracker()
        # Selected individual on the Pareto table (drives
        # "Accept" + future "Inspect" actions).
        self.selected_individual: str | None = None
        # §5 P0 — live cursor position on the table (drives
        # the per-individual chain preview pane). Distinct
        # from `selected_individual` so the pane updates on
        # cursor move while Accept stays bound to the
        # committed selection.
        self._highlighted_individual: str | None = None
        # §5 P0 — raw platform response from the last
        # `accept_individual` call. Tests + downstream
        # toast-rendering read `chain_id` / `new_version`
        # / `previous_version` off this; older platforms
        # may return `None` (worker doesn't post on
        # exception so callers always see the dict).
        self.accept_result: dict | None = None
        self.cancelled: bool = False
        # Versions tab state — list of growth points (gen,
        # fitness, optional chain dict) and the currently
        # highlighted index. Filled by ``_refresh_versions_tab``
        # from the tracker's fitness curve + the run's
        # individuals; the diff view consumes the same list.
        self._versions: list[dict[str, Any]] = []
        self._selected_version_idx: int | None = None
        self._versions_mode: str = "chain"  # "chain" | "diff"
        # Per-generation frontier programs from the Platform —
        # keyed by generation so ``_chain_for_version`` can look
        # up real chain content + mutation rationale. Populated
        # by ``frontier_programs_snapshot`` events.
        self._frontier_by_gen: dict[int, dict[str, Any]] = {}
        # Rubric recovered from a run's description when observing (the
        # launch path already has it on ``self.config``).
        self._observed_rubric: str | None = None
        self._observed_max_iterations: int | None = None
        # Platform version (fetched once on mount) for the metadata card.
        self._platform_version: str = ""

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        # Top row: STATUS + EVENTS. The wide PARETO FRONT table
        # used to ride here too, but it moved into the
        # "Pareto Front" tab below so the Statistics tab has
        # space for the fitness curve.
        with Horizontal(id="evolution-body"):
            with Vertical(id="evolution-status"):
                yield Label(t("evolution.status"), classes="pane-title")
                yield Static(t("evolution.submitting"), id="evolution-status-text")
            with Vertical(id="evolution-events"):
                yield Label(t("evolution.events"), classes="pane-title")
                yield VerticalScroll(id="evolution-events-log")
        # Three-tab visualization deck — mirrors the
        # gigaevo-platform web UI's "Statistics / Pareto /
        # Programs" layout so users moving between the two
        # surfaces see the same information architecture.
        with TabbedContent(id="evolution-tabs", initial="tab-fitness"):
            with TabPane(t("evolution.tabFitness"), id="tab-fitness"):
                # Full-sized two-axis line chart (best fitness +
                # current iteration mean) via ``textual-plot``'s
                # ``PlotWidget`` — high-res braille rendering,
                # repaints incrementally as ``_render_fitness_pane``
                # pushes new points.
                try:
                    from textual_plot import PlotWidget as _PlotWidget

                    yield _PlotWidget(id="evolution-fitness-plot")
                except Exception:
                    # Defensive: if the optional widget can't
                    # import (e.g. test envs without the dep),
                    # keep the legacy sparkline so the screen
                    # still composes.
                    yield Static(
                        "(textual-plot widget unavailable — "
                        "falling back to sparkline)",
                        id="evolution-fitness-plot-fallback",
                    )
            with TabPane(t("evolution.tabStatistics"), id="tab-statistics"):
                yield Static(
                    t("evolution.noFitnessData"),
                    id="evolution-stats-cards",
                    classes="evolution-stats-cards",
                )
                yield Label(
                    t("evolution.bestFitness"),
                    classes="pane-title",
                )
                yield Static(
                    t("evolution.noFitnessData"),
                    id="evolution-fitness-text",
                )
            with TabPane(t("evolution.tabPareto"), id="tab-pareto"):
                yield Label(t("evolution.paretoFront"), classes="pane-title")
                yield DataTable(id="evolution-pareto-table")
                # Per-row detail (full summary + all objectives) for the
                # highlighted individual — the table truncates; this shows
                # the whole thing.
                yield Static(
                    t("evolution.paretoDetailEmpty"),
                    id="evolution-pareto-detail",
                )
                # §5 P1 — 2D scatter beneath the table for
                # runs that carry ≥ 2 objectives.
                yield Label(t("evolution.scatter"), classes="pane-title")
                yield Static(
                    t("evolution.needsObjectives"),
                    id="evolution-pareto-plot-text",
                )
            with TabPane(t("evolution.tabPrograms"), id="tab-programs"):
                yield Label(
                    t("evolution.validInvalid"),
                    classes="pane-title",
                )
                yield Static(
                    t("evolution.noProgramData"),
                    id="evolution-programs-text",
                )
            with TabPane(t("evolution.tabVersions"), id="tab-versions"):
                # Left side: list of growth points (every
                # generation where best fitness strictly
                # increased). User picks one to inspect; the
                # preview / diff lands on the right.
                with Horizontal(id="evolution-versions-body"):
                    with Vertical(id="evolution-versions-list-pane"):
                        yield Label(
                            t("evolution.fitnessGrowth"),
                            classes="pane-title",
                        )
                        yield DataTable(
                            id="evolution-versions-table",
                        )
                    with Vertical(id="evolution-versions-preview-pane"):
                        with Horizontal(id="evolution-versions-toolbar"):
                            yield Button(
                                t("evolution.viewDiff"),
                                id="evolution-btn-versions-mode",
                                variant="primary",
                            )
                        yield Static(
                            t("evolution.selectVersion"),
                            id="evolution-versions-preview",
                        )
        with Horizontal(id="evolution-actions"):
            yield Button(t("common.back"), id="evolution-btn-back")
            yield Button(
                t("evolution.stopArchive"),
                id="evolution-btn-archive",
                variant="warning",
            )
            yield Button(
                t("evolution.acceptWinner"),
                id="evolution-btn-accept",
                variant="success",
            )
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="EvolutionScreen",
                breadcrumb=(t("header.breadcrumb.library"), t("header.breadcrumb.evolve")),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="EvolutionScreen",
                scope="screen",
            )
        except Exception:
            pass
        try:
            table = self.query_one(
                "#evolution-pareto-table", DataTable,
            )
            # Leading badge column flags rows on the Pareto front
            # (★) — the user picks one of those for "Accept".
            table.add_columns(
                t("evolution.col.star"),
                t("evolution.col.id"),
                t("evolution.col.gen"),
                t("evolution.col.fitness"),
                t("evolution.col.objectives"),
                t("evolution.col.summary"),
            )
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        try:
            v_table = self.query_one(
                "#evolution-versions-table", DataTable,
            )
            v_table.add_columns(
                t("evolution.col.gen"),
                t("evolution.col.fitness"),
                t("evolution.col.delta"),
            )
            v_table.cursor_type = "row"
            v_table.zebra_stripes = True
        except Exception:
            pass
        self._sync_versions_mode_button()
        self.run.started_at = time.monotonic()
        self.refresh_status()
        # Native animated loading overlay while we submit the run / fetch the
        # first observe snapshot. Cleared in `_handle_event` (first platform
        # event) or by the submit worker's terminal/error early-returns.
        # Set *after* the pre-worker `refresh_status()` so that paint doesn't
        # have to know about the overlay. Reduced-motion-safe via Textual.
        self.loading = True
        self.run_worker(
            self._submit_and_stream(),
            name="evolution_run",
            group="evolution",
            exclusive=True,
            exit_on_error=False,
        )
        self.run_worker(
            self._fetch_platform_version(),
            name="evolution_platform_version",
            group="evolution_meta",
            exclusive=True,
            exit_on_error=False,
        )

    async def _fetch_platform_version(self) -> None:
        """One-shot platform-version fetch (off-thread) for the metadata
        card. Best-effort — never raises into the UI."""
        platform = getattr(self.app, "platform", None)
        if platform is None:
            return
        try:
            health = await asyncio.to_thread(platform.health_check)
        except Exception:  # noqa: BLE001
            return
        version = _extract_platform_version(health)
        if version:
            self._platform_version = version
            self._render_stats_cards()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    async def _resolve_base_chain_content(self) -> dict[str, Any] | None:
        """Return seed chain JSON, fetching from Memory when needed.

        The live Platform scheduler only drains chain-experiment
        submissions (``POST /api/v1/experiments/chains``), which
        require inlined ``base_chain_content``. Without it CARE
        falls back to legacy ``POST /api/v1/evolutions`` — that
        route 404s on current Platform builds.
        """
        existing = self.plan.base_chain_content
        if isinstance(existing, dict) and existing.get("steps"):
            return existing
        memory = getattr(self.app, "memory", None)
        chain_id = self.plan.base_chain_entity_id or self.run.base_chain_id
        if memory is None or not chain_id:
            return None
        try:
            fetched = await asyncio.to_thread(memory.get_chain, chain_id)
        except Exception:
            return None
        if not isinstance(fetched, dict) or not fetched.get("steps"):
            return None
        self.plan = EvolutionPlan(
            config=self.config,
            base_chain_entity_id=self.plan.base_chain_entity_id,
            base_chain_content=fetched,
            base_chain_name=self.plan.base_chain_name,
        )
        return fetched

    async def _submit_and_stream(self) -> None:
        platform = getattr(self.app, "platform", None)
        if platform is None:
            self.run.status = "no_platform"
            self.run.last_error = "no platform facade configured"
            self.refresh_status()
            return
        # Observe-only mode: skip submission and stream events
        # for an existing run. Used by the EvolutionDashboard's
        # "open row" path so revisiting a run doesn't fire a
        # duplicate submission.
        if self._observe_evolution_id:
            self.run.evolution_id = self._observe_evolution_id
            self.run.status = "watching"
            # Fetch the full experiment record once so we know
            # when the run actually started — neither
            # ``/status`` nor ``/results`` carries that field,
            # but ``get_evolution`` (which routes to
            # ``GET /api/v1/experiments/{id}`` for ``exp_*`` ids)
            # does. Without this the elapsed clock would start
            # at 0 when the user opens the screen.
            try:
                snapshot = await asyncio.to_thread(
                    platform.get_evolution, self._observe_evolution_id,
                )
                if isinstance(snapshot, dict):
                    self._handle_event({"event": "snapshot", "data": snapshot})
                    self._hydrate_from_snapshot(snapshot)
            except Exception:
                pass
            self.refresh_status()
            try:
                await self._drain_events(platform, self._observe_evolution_id)
            except Exception as exc:  # noqa: BLE001
                self.run.status = "stream_failed"
                self.run.last_error = f"{type(exc).__name__}: {exc}"
                self.refresh_status()
            return
        try:
            body = build_evolution_request(self.plan)
        except Exception as exc:  # noqa: BLE001
            self.run.status = "submit_failed"
            self.run.last_error = f"{type(exc).__name__}: {exc}"
            self.refresh_status()
            return
        # `build_evolution_request` returns the server-shape body;
        # CarePlatform.start_evolution takes the same fields as
        # keyword args (it merges in CARE-source tags and forwards
        # via the SDK). Map `seed_chain_id` → `base_chain_id` for
        # the facade's signature.
        spec = dict(body)
        spec["base_chain_id"] = spec.pop("seed_chain_id", self.run.base_chain_id)
        if self.config.mutation_max_tokens is not None:
            spec["mutation_max_tokens"] = self.config.mutation_max_tokens
        base_chain_content = await self._resolve_base_chain_content()
        if base_chain_content is None:
            self.run.status = "submit_failed"
            self.run.last_error = (
                "chain content unavailable — fetch the seed chain from "
                "Memory before submitting (Platform requires "
                "experiments/chains, not /api/v1/evolutions)"
            )
            self.refresh_status()
            return
        spec["base_chain_content"] = base_chain_content
        try:
            ref = await asyncio.to_thread(
                platform.start_evolution, **spec,
            )
        except Exception as exc:  # noqa: BLE001
            self.run.status = "submit_failed"
            self.run.last_error = f"{type(exc).__name__}: {exc}"
            self.refresh_status()
            return
        self.run.evolution_id = ref.evolution_id
        await self._sync_run_snapshot(platform, ref.evolution_id)
        if self.run.status == "submitting":
            self.run.status = getattr(ref, "status", None) or "running"
        self._refresh_status_header()
        try:
            await self._drain_events(platform, ref.evolution_id)
        except Exception as exc:  # noqa: BLE001
            self.run.status = "stream_failed"
            self.run.last_error = f"{type(exc).__name__}: {exc}"
            self.refresh_status()

    def _hydrate_from_snapshot(self, snapshot: dict) -> None:
        """Eagerly populate the Fitness / Programs / Versions tabs from a
        one-shot ``get_evolution`` snapshot when opening a run in
        observe mode.

        Without this, observing a **completed** run shows empty charts:
        the poll loop returns the terminal event before it ever emits a
        ``fitness_history_snapshot`` (that branch only runs while the
        status is non-terminal), so the curve/programs/frontier never
        load. We read the same ``/results`` metrics the poll loop would
        have and replay them as the usual synthetic events. As a final
        fallback for local stacks whose Platform ``/results`` is empty,
        probe gigavolve Redis directly."""
        max_iters = snapshot.get("max_iterations")
        if isinstance(max_iters, (int, float)) and max_iters >= 1:
            self._observed_max_iterations = int(max_iters)

        raw = snapshot.get("_raw") if isinstance(snapshot, dict) else None
        if self._observed_max_iterations is None and isinstance(raw, dict):
            experiment = raw.get("experiment")
            if isinstance(experiment, dict):
                cfg = experiment.get("config")
                if isinstance(cfg, dict):
                    cfg_max = cfg.get("max_iterations")
                    if isinstance(cfg_max, (int, float)) and cfg_max >= 1:
                        self._observed_max_iterations = int(cfg_max)

        results = raw.get("results") if isinstance(raw, dict) else None
        metrics: dict[str, Any] = {}
        if isinstance(results, dict) and isinstance(results.get("metrics"), dict):
            metrics = results["metrics"]

        history = metrics.get("fitness_history")
        if isinstance(history, list) and history:
            self._handle_event(
                {
                    "event": "fitness_history_snapshot",
                    "data": {"history": history, "source": "platform"},
                }
            )
        else:
            # Local-stack fallback — read the curve straight from Redis.
            probed = self._observe_probe_fitness_history()
            if probed:
                self._handle_event(
                    {
                        "event": "fitness_history_snapshot",
                        "data": {"history": probed, "source": "redis_probe"},
                    }
                )

        frontier = metrics.get("frontier_programs")
        if isinstance(frontier, list) and frontier:
            self._handle_event(
                {
                    "event": "frontier_programs_snapshot",
                    "data": {"frontier": frontier},
                }
            )

        pv = metrics.get("programs_valid")
        pi = metrics.get("programs_invalid")
        programs_from_platform = isinstance(pv, int) or isinstance(pi, int)
        if not programs_from_platform:
            pv, pi = self._observe_probe_programs_counts()
        if isinstance(pv, int) or isinstance(pi, int):
            self._handle_event(
                {
                    "event": "programs_snapshot",
                    "data": {
                        "programs_valid": pv if isinstance(pv, int) else -1,
                        "programs_invalid": pi if isinstance(pi, int) else -1,
                        # Tag by where the counts ACTUALLY came from, not
                        # by whether the snapshot carried other metrics —
                        # otherwise the liveness line mislabels Redis-probed
                        # counts as "platform".
                        "source": "platform"
                        if programs_from_platform
                        else "redis_probe",
                    },
                }
            )

        # Guarantee at least one real curve point for runs whose only
        # reported metric is the final best fitness.
        best = snapshot.get("best_fitness")
        gen = snapshot.get("generation")
        if isinstance(best, (int, float)) and isinstance(gen, int):
            self._handle_event(
                {
                    "event": "best_updated",
                    "data": {
                        "best_fitness": best,
                        "generation": gen,
                        "source": "platform",
                    },
                }
            )

    def _observe_probe_fitness_history(self) -> list[dict[str, Any]]:
        """Best-effort gigavolve Redis read for the observed run's curve."""
        eid = self._observe_evolution_id or self.run.evolution_id
        if not isinstance(eid, str) or not eid.startswith("exp_"):
            return []
        try:
            from care.runtime.evolution_redis_probe import probe_fitness_history

            return probe_fitness_history(eid)
        except Exception as exc:  # noqa: BLE001
            _log.debug("observe fitness-history probe failed: %s", exc)
            return []

    def _observe_probe_programs_counts(self) -> tuple[int | None, int | None]:
        """Best-effort gigavolve Redis read for the observed run's counts."""
        eid = self._observe_evolution_id or self.run.evolution_id
        if not isinstance(eid, str) or not eid.startswith("exp_"):
            return (None, None)
        try:
            from care.runtime.evolution_redis_probe import probe_programs_counts

            return probe_programs_counts(eid)
        except Exception as exc:  # noqa: BLE001
            _log.debug("observe programs probe failed: %s", exc)
            return (None, None)

    async def _drain_events(self, platform: Any, evolution_id: str) -> None:
        """Iterate the platform's sync SSE generator one event at
        a time, hopping back to the event loop between events so
        rendering stays smooth. The first ``next(iter)`` call
        runs on a worker thread so a slow Platform server can't
        block the loop."""
        iterator = await asyncio.to_thread(
            platform.stream_events, evolution_id,
        )
        sentinel = object()
        while True:
            if self.cancelled:
                return
            event = await asyncio.to_thread(next, iterator, sentinel)
            if event is sentinel:
                return
            try:
                self._handle_event(event)
            except Exception:
                continue
            await asyncio.sleep(0)

    async def _sync_run_snapshot(
        self, platform: Any, evolution_id: str,
    ) -> None:
        """One-shot fetch so the status header reflects Platform
        state before the poll loop's first tick (create often
        returns ``prepared`` even after ``/start``)."""
        get_fn = getattr(platform, "get_evolution", None)
        if not callable(get_fn):
            return
        try:
            snapshot = await asyncio.to_thread(get_fn, evolution_id)
        except Exception:
            return
        if isinstance(snapshot, dict):
            self._handle_event({"event": "snapshot", "data": snapshot})

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _apply_live_progress(self, payload: dict) -> None:
        """Fold status / generation fields into ``self.run``."""
        status = _coerce_status_from_payload(payload)
        if status:
            current = (self.run.status or "").lower()
            incoming = status.lower()
            if (
                not self.run.finished
                or incoming in _TERMINAL_RUN_STATUSES
                or current in {"", "submitting", "prepared", "queued", "pending"}
            ):
                self.run.status = status
        generation = _coerce_generation_from_payload(payload)
        if generation is not None and generation >= self.run.generation:
            self.run.generation = generation
        best_raw = payload.get("best_fitness")
        if best_raw is None and isinstance(payload.get("metrics"), dict):
            best_raw = payload["metrics"].get("best_fitness")
        if isinstance(best_raw, (int, float)):
            # Tracker picks this up via downstream best_updated;
            # nothing else required here for the status header.
            pass

    def _handle_event(self, event: dict) -> None:
        if not isinstance(event, dict):
            return
        kind = str(event.get("event") or event.get("type") or "")
        if kind == "heartbeat":
            # Keepalive frame from the live SSE stream — proves the stream
            # is healthy during a quiet generation. Refresh the liveness
            # clock so the status pane doesn't drift into "stalled", but
            # don't log it as an event row or feed the trackers.
            self.run.last_event_monotonic = time.monotonic()
            self._refresh_status_header()
            return
        payload = event.get("data") or event
        if not isinstance(payload, dict):
            payload = {}
        self.run.events.append((kind, payload))
        # P0.4 — liveness bookkeeping: stamp the last-event clock + the
        # data source so the status pane can show "updated Ns ago ·
        # source: redis-probe". ``status`` heartbeats count too — they
        # prove the poller is alive even when no metrics moved.
        self.run.last_event_monotonic = time.monotonic()
        source = payload.get("source")
        if isinstance(source, str) and source:
            self.run.data_source = source
        elif kind in {"fitness_history_snapshot", "best_updated", "individual_evaluated"}:
            # These ride real metrics; default to the Platform path when
            # the event didn't tag an explicit source.
            self.run.data_source = self.run.data_source or "platform"
        # Feed the fitness-curve aggregator. `record_event` reads
        # the `event_type`/`type` key, so build a flat dict that
        # carries the kind alongside the payload values — the
        # SSE wire shape `{"event": "...", "data": {...}}` would
        # be discarded by the tracker otherwise.
        try:
            self.tracker.record_event({"event_type": kind, **payload})
        except Exception:
            pass
        # Absorb the Platform-reported wall-clock start the
        # first time it arrives so ``elapsed`` reflects the
        # experiment's real lifetime rather than time since the
        # screen mounted (critical for observe-mode opens).
        if self.run.wall_started_at is None:
            iso = (
                payload.get("started_at")
                or payload.get("created_at")
            )
            if isinstance(iso, str) and iso:
                try:
                    from datetime import datetime as _dt

                    text = iso[:-1] + "+00:00" if iso.endswith("Z") else iso
                    self.run.wall_started_at = _dt.fromisoformat(text).timestamp()
                except (TypeError, ValueError):
                    pass
            elif isinstance(iso, (int, float)):
                self.run.wall_started_at = float(iso)
        # Absorb live program counts + current-iteration mean
        # fitness whenever any event carries them so the
        # Statistics + Programs tabs stay current.
        cf = payload.get("current_fitness")
        if isinstance(cf, (int, float)):
            self.run.current_fitness = float(cf)
        pv = payload.get("programs_valid")
        if isinstance(pv, int) and pv >= 0:
            self.run.programs_valid = pv
        pi = payload.get("programs_invalid")
        if isinstance(pi, int) and pi >= 0:
            self.run.programs_invalid = pi
        rid = payload.get("runner_id")
        if isinstance(rid, str) and rid:
            self.run.runner_id = rid
        rubric = payload.get("validation_rubric")
        if isinstance(rubric, str) and rubric.strip():
            self._observed_rubric = rubric.strip()
        # Per-generation program history for the Programs-tab trend.
        # Only when the event ties counts to a specific generation
        # (the /results path does; the bare redis programs_snapshot
        # doesn't, so it only updates the latest-count bar).
        ev_gen = payload.get("generation")
        if isinstance(ev_gen, int) and (
            (isinstance(pv, int) and pv >= 0)
            or (isinstance(pi, int) and pi >= 0)
        ):
            try:
                self.tracker.record_programs(
                    ev_gen,
                    pv if isinstance(pv, int) else None,
                    pi if isinstance(pi, int) else None,
                )
            except Exception:  # noqa: BLE001
                pass
        self._apply_live_progress(payload)
        if kind == "status":
            self._render_event_row(kind, payload)
            self._refresh_status_header()
            return
        if kind == "snapshot":
            self._render_event_row(kind, payload)
            self._refresh_status_header()
            return
        if kind == "fitness_history_snapshot":
            # Bulk-load the entire fitness curve into the
            # tracker so the line plot has the full series the
            # moment results land — not just the deltas from
            # subsequent ``best_updated`` events. Each history
            # entry is ``{"generation": int, "best_fitness":
            # float?, "current_fitness": float?}``.
            history = payload.get("history") or []
            from care.evolution_session import GenerationStat as _GS
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                gen = entry.get("generation")
                bf = entry.get("best_fitness")
                cf = entry.get("current_fitness")
                if not isinstance(gen, int):
                    continue
                try:
                    self.tracker.record_generation(
                        _GS(
                            generation=int(gen),
                            best_fitness=float(bf) if isinstance(bf, (int, float)) else float("-inf"),
                            mean_fitness=float(cf) if isinstance(cf, (int, float)) else None,
                        )
                    )
                except Exception:
                    continue
            # Trigger a full refresh so Versions / Statistics
            # / Programs tabs pick up the new history points too
            # — without this the curve loads but the Versions
            # tab stays empty.
            self._render_fitness_pane()
            self._refresh_versions_tab()
            self.refresh_status()
            return
        if kind == "frontier_programs_snapshot":
            # Cache real per-generation chain content + mutation
            # rationale so the Versions tab can show them
            # instead of the "(not exposed)" placeholder. The same
            # records are the only per-individual data a chain run
            # has, so feed them into the Pareto/individuals table too
            # (otherwise it stays on its "waiting for individuals"
            # placeholder for the whole run).
            frontier = payload.get("frontier") or []
            updated = dict(self._frontier_by_gen)
            upserted = False
            for entry in frontier:
                if not isinstance(entry, dict):
                    continue
                gen = entry.get("generation")
                if isinstance(gen, int):
                    updated[gen] = entry
                ind_payload = _frontier_entry_to_individual(entry)
                if ind_payload is not None:
                    self._upsert_individual(ind_payload, render=False)
                    upserted = True
            self._frontier_by_gen = updated
            if upserted:
                self._render_pareto_table()
            self._refresh_versions_tab()
            return
        if kind == "programs_snapshot":
            # Redis-probe fallback for the Programs chart. The
            # valid/invalid counts were already folded into
            # ``self.run`` by the generic absorption above; just
            # repaint the affected panes. Returns early (no event
            # row) because this re-fires every poll tick and would
            # otherwise spam the log.
            self._render_programs_pane()
            self._render_stats_cards()
            self._refresh_status_header()
            return
        if kind == "cost_tick":
            # §5 P1 — canonical cost telemetry. Payload carries one delta
            # tick (the platform emits cost_tick from cumulative token
            # deltas and does NOT also put tokens on individual events, so
            # there's no double-count). Returns early without an event row
            # because it re-fires every poll tick and would otherwise spam
            # the log — just refresh the cost line.
            _accumulate_cost(self.run, payload)
            self._refresh_status_header()
            return
        if kind == "generation_started":
            gen = payload.get("generation")
            if isinstance(gen, int):
                self.run.generation = gen
        elif kind in {"individual_evaluated", "best_updated"}:
            self._upsert_individual(payload)
            # §5 P1 — fold any token/cost data that rode with
            # this individual event into the running aggregator.
            # Platforms that ship dedicated `cost_tick` events
            # also land above; double-counting is avoided by
            # the canonical `cost_tick` branch operating on
            # explicit delta payloads.
            _accumulate_cost(self.run, payload)
        elif kind == "completed":
            self.run.finished = True
            self.run.status = str(payload.get("status") or "completed")
            # Some platforms ship a final cost summary on the
            # completion event — fold it in just in case.
            _accumulate_cost(self.run, payload)
        elif kind == "accepted":
            ind = payload.get("individual_id")
            if isinstance(ind, str):
                self.run.accepted_id = ind
        elif kind in {"failed", "cancelled"}:
            self.run.finished = True
            self.run.status = kind
            self.run.last_error = (
                str(payload.get("error") or "") or None
            )
        self._render_event_row(kind, payload)
        self.refresh_status()

    def _upsert_individual(self, payload: dict, *, render: bool = True) -> None:
        ind_id = str(
            payload.get("individual_id")
            or payload.get("id")
            or ""
        )
        if not ind_id:
            return
        fitness_raw = payload.get("fitness")
        fitness: float | None
        try:
            fitness = float(fitness_raw) if fitness_raw is not None else None
        except (TypeError, ValueError):
            fitness = None
        objectives_raw = payload.get("objectives") or {}
        objectives: tuple[tuple[str, float], ...] = ()
        if isinstance(objectives_raw, dict):
            try:
                objectives = tuple(
                    (str(k), float(v))
                    for k, v in objectives_raw.items()
                )
            except (TypeError, ValueError):
                objectives = ()
        # §5 P0 — capture the chain payload when the SSE event
        # ships one so the preview pane has something to render.
        # Platforms use varying keys (`chain`, `chain_content`,
        # `content`); accept the first dict-shaped match.
        chain_dict: dict | None = None
        for key in ("chain", "chain_content", "content"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                chain_dict = candidate
                break
        ind = EvolutionIndividual(
            individual_id=ind_id,
            generation=int(payload.get("generation") or 0),
            fitness=fitness,
            objectives=objectives,
            summary=str(payload.get("summary") or "")[:80],
            chain_dict=chain_dict,
        )
        # Replace existing entry for the same id.
        existing_idx = next(
            (
                i for i, e in enumerate(self.run.individuals)
                if e.individual_id == ind_id
            ),
            None,
        )
        if existing_idx is None:
            self.run.individuals.append(ind)
        else:
            self.run.individuals[existing_idx] = ind
        self.run.individuals.sort(
            key=lambda e: (
                -(e.fitness if e.fitness is not None else float("-inf")),
                e.individual_id,
            ),
        )
        if render:
            self._render_pareto_table()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh_status_header(self) -> None:
        """Repaint only the STATUS pane — safe during high-frequency
        poll ticks (avoids touching ``PlotWidget`` every 2s)."""
        if not self.is_mounted:
            return
        # Any status repaint driven from the submit/stream worker (first
        # snapshot, first event, or a terminal/error early-return) means the
        # mount-time loading overlay has served its purpose — drop it. The
        # pre-worker paint in `on_mount` runs before `self.loading` is armed,
        # so clearing here is a no-op for that call.
        if self.loading:
            self.loading = False
        try:
            target = self.query_one("#evolution-status-text", Static)
        except Exception:
            return
        parts = [
            f"evolution: {self.run.evolution_id or '?'}",
            f"status: {self.run.status}",
        ]
        gen_line = f"gen: {self.run.generation}"
        max_iters = self._observed_max_iterations
        if max_iters is None:
            max_iters = getattr(self.config, "max_iterations", None)
        if isinstance(max_iters, int) and max_iters > 0:
            gen_line += f" / {max_iters}"
        parts.append(gen_line)
        elapsed = self.run.elapsed()
        if elapsed is not None:
            parts.append(f"elapsed: {elapsed:0.1f}s")
        if self.run.last_error:
            parts.append(f"error: {self.run.last_error}")
        if self.run.accepted_id:
            parts.append(f"accepted: {self.run.accepted_id}")
        cost_line = self.format_cost_meter()
        if cost_line:
            parts.append(cost_line)
        start_fitness, best_fitness = self._fitness_endpoints()
        if start_fitness is not None or best_fitness is not None:
            sf = (
                f"{start_fitness:.4f}" if start_fitness is not None else "—"
            )
            bf = (
                f"{best_fitness:.4f}" if best_fitness is not None else "—"
            )
            parts.append(f"start fitness: {sf}")
            parts.append(f"best fitness:  {bf}")
        liveness = self._liveness_line()
        if liveness:
            parts.append(liveness)
        target.update("\n".join(parts))

    def _liveness_line(self) -> str:
        """One-line "is anything happening?" indicator for the status
        pane: how long since the last event + where the data came from.

        Empty until the first event arrives (the status header already
        shows ``status: submitting`` then). Once events flow it reads
        e.g. ``live · updated 2s ago · source: redis-probe`` so the user
        can tell the difference between "still working" and "stalled"."""
        ts = self.run.last_event_monotonic
        if ts is None:
            return ""
        age = max(0.0, time.monotonic() - ts)
        if self.run.finished:
            tag = "done"
        elif age <= 15.0:
            tag = "[green]live[/green]"
        else:
            # No event for a while — surface it rather than looking idle.
            tag = "[yellow]stalled?[/yellow]"
        bits = [tag, f"updated {age:0.0f}s ago"]
        source = self.run.data_source
        if source:
            pretty = {"redis_probe": "redis-probe", "platform": "platform"}.get(
                source, source
            )
            bits.append(f"source: {pretty}")
        return " · ".join(bits)

    def refresh_status(self) -> None:
        if not self.is_mounted:
            return
        self._refresh_status_header()
        self._render_stats_cards()
        self._render_fitness_pane()
        self._render_pareto_pane()
        self._render_programs_pane()
        self._refresh_versions_tab()

    def _fitness_endpoints(self) -> tuple[float | None, float | None]:
        """Return ``(start_fitness, best_fitness)`` from the tracker.

        ``start_fitness`` is the best fitness recorded for the
        lowest generation we know about — typically generation
        0, i.e. the seed chain's score before any mutation.
        ``best_fitness`` is the maximum across the whole curve
        (hall-of-fame). Both are ``None`` before the runner
        reports any real value (``-inf`` placeholder rows are
        skipped so we don't display ``inf`` to the user)."""
        curve = self.tracker.fitness_curve()
        real = [
            r for r in curve
            if isinstance(getattr(r, "best_fitness", None), (int, float))
            and r.best_fitness != float("-inf")
        ]
        if not real:
            return None, None
        # ``fitness_curve`` returns records ordered by generation
        # so ``[0]`` is the earliest real point.
        start = real[0].best_fitness
        best = max(r.best_fitness for r in real)
        return float(start), float(best)

    def _render_stats_cards(self) -> None:
        """Update the Statistics tab's summary cards with the
        latest run state — generation, best fitness, current
        iteration mean fitness, valid/invalid program counts.
        Silently no-ops when the pane isn't mounted (older
        tests / subclasses that compose only the status row)."""
        if not self.is_mounted:
            return
        try:
            pane = self.query_one(
                "#evolution-stats-cards", Static,
            )
        except Exception:
            return
        best = self.run.best_fitness if hasattr(self.run, "best_fitness") else None
        # ``best_fitness`` is computed from the tracker because
        # the dataclass doesn't carry it directly.
        best_curve = self.tracker.best_overall
        if best_curve is not None:
            best = best_curve.best_fitness
        gen_str = str(self.run.generation) if self.run.generation else "—"
        best_str = (
            f"{best:.6f}" if isinstance(best, (int, float)) else "—"
        )
        cur = self.run.current_fitness
        cur_str = (
            f"{cur:.6f}" if isinstance(cur, (int, float)) else "—"
        )
        v = self.run.programs_valid
        i = self.run.programs_invalid
        valid_str = str(v) if v >= 0 else "—"
        invalid_str = str(i) if i >= 0 else "—"
        # Compact card rows so the pane reads at a glance.
        # Brackets so dynamic colour markup doesn't fight the
        # Static's default style.
        body = (
            f"[b]Generation:[/b]      {gen_str}\n"
            f"[b]Best fitness:[/b]    [green]{best_str}[/green]\n"
            f"[b]Current fitness:[/b] {cur_str}\n"
            f"[b]Valid programs:[/b]   [green]{valid_str}[/green]\n"
            f"[b]Invalid programs:[/b] [grey50]{invalid_str}[/grey50]"
        )
        meta = self._run_metadata_lines()
        if meta:
            body += "\n\n" + "\n".join(meta)
        pane.update(body)

    def _run_metadata_lines(self, *, rubric_cap: int = 240) -> list[str]:
        """"What is this run optimising?" metadata for the Statistics
        card: the validation rubric (launch config or recovered from an
        observed run's description), the evolution mode, and the runner.

        Directly answers the user's "show details on what is going on"
        — without it the screen never says what fitness is measuring."""
        lines: list[str] = []
        rubric = (
            getattr(self.config, "validation_criteria", "") or ""
        ).strip() or (self._observed_rubric or "").strip()
        if rubric:
            if len(rubric) > rubric_cap:
                rubric = rubric[: rubric_cap - 1].rstrip() + "…"
            lines.append(f"[b]Optimising for:[/b] {rubric}")
        mode = getattr(self.config, "evolution_mode", "") or ""
        if mode:
            lines.append(f"[b]Mode:[/b] {mode}")
        if self.run.runner_id:
            lines.append(f"[b]Runner:[/b] [grey50]{self.run.runner_id}[/grey50]")
        if self._platform_version:
            lines.append(
                f"[b]Platform:[/b] [grey50]v{self._platform_version}[/grey50]"
            )
        return lines

    def _refresh_versions_tab(self) -> None:
        """Rebuild the Versions tab from the latest fitness curve.

        Each "version" is a generation where the **best** fitness
        strictly improved over the previous record — i.e. an
        evolutionary checkpoint worth inspecting. Seeds the
        table with rows ``Gen | Fitness | Δ`` and keeps the
        previously selected version highlighted across refreshes
        when its generation still belongs to the new list."""
        if not self.is_mounted:
            return
        try:
            table = self.query_one(
                "#evolution-versions-table", DataTable,
            )
        except Exception:
            return
        curve = self.tracker.fitness_curve()
        records = [
            r for r in curve
            if isinstance(getattr(r, "best_fitness", None), (int, float))
            and r.best_fitness != float("-inf")
        ]
        growth: list[dict[str, Any]] = []
        last_best: float | None = None
        for r in records:
            bf = float(r.best_fitness)
            if last_best is None or bf > last_best + 1e-9:
                growth.append(
                    {
                        "generation": int(r.generation),
                        "best_fitness": bf,
                        "delta": (bf - last_best) if last_best is not None else None,
                        "best_individual_id": r.best_individual_id,
                    }
                )
                last_best = bf
        # Preserve current selection across re-renders.
        previously_selected_gen: int | None = None
        if (
            self._selected_version_idx is not None
            and 0 <= self._selected_version_idx < len(self._versions)
        ):
            previously_selected_gen = self._versions[
                self._selected_version_idx
            ].get("generation")
        self._versions = growth
        table.clear()
        for v in growth:
            delta = v.get("delta")
            delta_str = f"+{delta:.4f}" if isinstance(delta, float) else "—"
            table.add_row(
                str(v["generation"]),
                f"{v['best_fitness']:.4f}",
                delta_str,
            )
        # Re-pick the same generation if it survived; else
        # default to the latest (top of the list grows downward,
        # so the last index is the highest gen).
        new_idx: int | None = None
        if previously_selected_gen is not None:
            for i, v in enumerate(growth):
                if v["generation"] == previously_selected_gen:
                    new_idx = i
                    break
        if new_idx is None and growth:
            new_idx = len(growth) - 1
        self._selected_version_idx = new_idx
        if new_idx is not None:
            try:
                table.move_cursor(row=new_idx)
            except Exception:
                pass
        self._render_versions_preview()

    def _chain_for_version(self, version: dict[str, Any]) -> dict | None:
        """Best-effort lookup of the chain content for a growth
        point.

        Priority:

        1. The Platform's per-generation ``frontier_programs``
           snapshot (read from gigavolve Redis). Each entry
           carries the actual chain config the LLM produced.
        2. Gigavolve Redis ``program:<id>`` when the Platform
           left ``chain_config`` null (common when
           ``BASE_CHAIN_CONFIG`` literals are malformed but the
           evaluated ``CallProgramFunction`` stage still holds
           clean ``chain_config_json``).
        3. The seed chain (``plan.base_chain_content``) for the
           first growth point when no frontier data is in yet.
        4. ``run.individuals`` legacy SSE fallback.
        5. The Platform's ``best_chain_config`` for completed
           runs (MinIO-served).
        """
        if not version:
            return None
        gen = version.get("generation")
        frontier_entry: dict[str, Any] | None = None
        if isinstance(gen, int):
            frontier_entry = self._frontier_by_gen.get(gen)
            if frontier_entry:
                for key in ("chain_config", "chain_content"):
                    chain = frontier_entry.get(key)
                    if isinstance(chain, dict):
                        return chain
        program_id = self._program_id_for_version(version, frontier_entry)
        experiment_id = self._experiment_id_for_probe()
        if (
            isinstance(experiment_id, str)
            and experiment_id.startswith("exp_")
            and isinstance(program_id, str)
            and program_id
        ):
            try:
                from care.runtime.evolution_redis_probe import (
                    probe_program_chain_config,
                )

                probed = probe_program_chain_config(experiment_id, program_id)
                if isinstance(probed, dict):
                    return probed
            except Exception as exc:  # noqa: BLE001
                _log.debug("versions redis chain probe failed: %s", exc)
        # Seed for the earliest growth point — used when the
        # Platform's program scan hasn't populated yet.
        if (
            self._versions
            and version is self._versions[0]
            and isinstance(self.plan.base_chain_content, dict)
        ):
            return self.plan.base_chain_content
        if isinstance(program_id, str) and program_id:
            for e in self.run.individuals:
                if e.individual_id == program_id and e.chain_dict:
                    return e.chain_dict
        ind_id = version.get("best_individual_id")
        if isinstance(ind_id, str) and ind_id:
            for e in self.run.individuals:
                if e.individual_id == ind_id and e.chain_dict:
                    return e.chain_dict
        raw = self.accept_result or {}
        if isinstance(raw.get("best_chain_config"), dict):
            return raw["best_chain_config"]
        return None

    def _experiment_id_for_probe(self) -> str | None:
        """Evolution id used for gigavolve Redis probes."""
        eid = self._observe_evolution_id or self.run.evolution_id
        return eid if isinstance(eid, str) and eid else None

    def _program_id_for_version(
        self,
        version: dict[str, Any],
        frontier_entry: dict[str, Any] | None = None,
    ) -> str | None:
        """Best program id for a growth-point version."""
        entry = frontier_entry
        if entry is None and isinstance(version.get("generation"), int):
            entry = self._frontier_by_gen.get(version["generation"])
        if isinstance(entry, dict):
            program_id = entry.get("program_id")
            if isinstance(program_id, str) and program_id:
                return program_id
        ind_id = version.get("best_individual_id")
        if isinstance(ind_id, str) and ind_id:
            return ind_id
        return None

    def _versions_view_mode_label(self) -> str:
        """Localized toolbar label for the Chain/Diff toggle."""
        if self._versions_mode == "diff":
            return t("evolution.viewChain")
        return t("evolution.viewDiff")

    def _sync_versions_mode_button(self) -> None:
        """Keep the Versions-tab mode button label in sync with state."""
        if not self.is_mounted:
            return
        try:
            button = self.query_one(
                "#evolution-btn-versions-mode", Button,
            )
        except Exception:
            return
        try:
            button.label = self._versions_view_mode_label()
        except Exception:
            pass

    def _frontier_for_version(self, version: dict[str, Any]) -> dict[str, Any] | None:
        """Return the Platform's frontier-program record for
        ``version`` (program_id, mutation_summary, changes)."""
        gen = version.get("generation") if version else None
        if isinstance(gen, int):
            return self._frontier_by_gen.get(gen)
        return None

    def _render_versions_preview(self) -> None:
        """Repaint the preview pane based on
        ``_versions_mode`` (chain vs diff) and the currently
        selected version."""
        if not self.is_mounted:
            return
        try:
            pane = self.query_one(
                "#evolution-versions-preview", Static,
            )
        except Exception:
            return
        if (
            self._selected_version_idx is None
            or not self._versions
        ):
            pane.update(t("evolution.versionsNoGrowthPoints"))
            return
        idx = self._selected_version_idx
        version = self._versions[idx]
        chain = self._chain_for_version(version)
        frontier = self._frontier_for_version(version)
        header_lines = [
            f"[bold]{t('evolution.versionsGenLabel')}[/bold] {version['generation']}",
            f"[bold]{t('evolution.versionsFitnessLabel')}[/bold]    [green]{version['best_fitness']:.4f}[/green]"
            + (
                f"  ([green]+{version['delta']:.4f}[/green])"
                if isinstance(version.get("delta"), float) else ""
            ),
        ]
        if frontier and frontier.get("program_id"):
            header_lines.append(
                f"[bold]{t('evolution.versionsProgramLabel')}[/bold]    "
                f"{str(frontier['program_id'])[:8]}…"
            )
        header = "\n".join(header_lines)
        # Mutation rationale block — surfaces the LLM's
        # justification + the specific structural changes it
        # applied to bump fitness. Shown above the chain so
        # the user sees the "why" before the "what".
        rationale_block = ""
        if frontier and self._versions_mode == "chain":
            summary = (frontier.get("mutation_summary") or "").strip()
            changes = frontier.get("mutation_changes") or []
            if summary or changes:
                # Escape user-provided text so Redis content with
                # literal ``[...]`` brackets (e.g. ``[invalid_streak: 8]``)
                # doesn't get interpreted as Textual markup tags.
                from rich.markup import escape as _esc

                rationale_lines = [
                    f"[bold cyan]{t('evolution.versionsMutationRationale')}[/bold cyan]",
                ]
                if summary:
                    rationale_lines.append(_esc(summary))
                if isinstance(changes, list) and changes:
                    rationale_lines.append("")
                    rationale_lines.append(f"[bold]{t('evolution.versionsChanges')}[/bold]")
                    for c in changes:
                        if not isinstance(c, dict):
                            continue
                        desc = (c.get("description") or "").strip()
                        if desc:
                            rationale_lines.append(f"  • {_esc(desc)}")
                rationale_block = "\n".join(rationale_lines) + "\n\n"
        if self._versions_mode == "diff":
            prev = self._versions[idx - 1] if idx > 0 else None
            prev_chain = self._chain_for_version(prev) if prev else None
            body = self._render_chain_diff(prev_chain, chain)
        else:
            body = self._render_chain_text(chain)
        content = header + "\n\n" + rationale_block + body
        try:
            pane.update(content)
        except Exception:
            # Markup parse failed for some malformed segment
            # (typically a stray unmatched bracket leaked through
            # the diff). Fall back to plain text so the user
            # still sees the chain / diff content.
            from rich.text import Text as _Text

            pane.update(_Text(content))

    def _render_chain_text(self, chain: dict | None) -> str:
        if chain is None:
            return t("evolution.versionsChainUnavailable")
        from rich.markup import escape as _esc

        try:
            from care.screens.inspection import render_chain_dag

            steps = chain.get("steps") or []
            return _esc(render_chain_dag(steps))
        except Exception:
            import json as _json

            return _esc(_json.dumps(chain, indent=2, default=str))

    def _render_chain_diff(self, prev: dict | None, curr: dict | None) -> str:
        """Unified git-style diff between two chain JSON
        snapshots. Falls back to a friendly note when either
        side is unavailable."""
        if prev is None and curr is None:
            return t("evolution.versionsDiffNeither")
        if prev is None:
            return (
                t("evolution.versionsDiffFirstPoint")
                + "\n\n"
                + self._render_chain_text(curr)
            )
        if curr is None:
            return t("evolution.versionsDiffCurrUnavailable")
        import difflib
        import json as _json

        from rich.markup import escape as _esc

        a = _json.dumps(prev, indent=2, sort_keys=True, default=str).splitlines()
        b = _json.dumps(curr, indent=2, sort_keys=True, default=str).splitlines()
        diff_lines: list[str] = []
        for line in difflib.unified_diff(
            a, b,
            fromfile=t("evolution.versionsDiffFrom"),
            tofile=t("evolution.versionsDiffTo"),
            lineterm="",
            n=3,
        ):
            # ``line`` may carry literal ``[...]`` JSON brackets;
            # escape before wrapping in markup so Textual's
            # parser doesn't treat them as tags.
            safe = _esc(line)
            if line.startswith("+++") or line.startswith("---"):
                diff_lines.append(f"[bold]{safe}[/bold]")
            elif line.startswith("+"):
                diff_lines.append(f"[green]{safe}[/green]")
            elif line.startswith("-"):
                diff_lines.append(f"[red]{safe}[/red]")
            elif line.startswith("@@"):
                diff_lines.append(f"[cyan]{safe}[/cyan]")
            else:
                diff_lines.append(safe)
        if not diff_lines:
            return t("evolution.versionsDiffEmpty")
        return "\n".join(diff_lines)

    def _empty_pane_text(self, what: str) -> str:
        """Context-aware placeholder for a chart pane with no data yet.

        Distinguishes "the runner hasn't started" from "running but no
        ``{what}`` reported yet" from "the run finished without reporting
        ``{what}``" so the user always knows whether to keep waiting
        instead of staring at a blank rectangle (the old behaviour)."""
        status = (self.run.status or "").lower()
        if self.run.finished or status in _TERMINAL_DISPLAY_STATUSES:
            return f"(no {what} was reported for this run)"
        if status in _PRE_RUN_STATUSES:
            return f"waiting for the runner to start… (no {what} yet)"
        return f"waiting for the first generation… (no {what} reported yet)"

    def _render_programs_pane(self) -> None:
        """Update the Programs tab's pie-equivalent bar."""
        if not self.is_mounted:
            return
        try:
            from care.runtime.programs_chart import (
                render_programs_pie,
                render_programs_trend,
            )

            pane = self.query_one(
                "#evolution-programs-text", Static,
            )
        except Exception as exc:
            _log.debug("programs pane not mounted (#evolution-programs-text): %s", exc)
            return
        chart = render_programs_pie(
            self.run.programs_valid,
            self.run.programs_invalid,
        )
        if not chart:
            pane.update(self._empty_pane_text("program data"))
            return
        trend = render_programs_trend(self.tracker.programs_curve())
        pane.update(chart + (f"\n\n{trend}" if trend else ""))

    def _render_fitness_pane(self) -> None:
        """Update both fitness surfaces with the latest plot.

        Two surfaces live in parallel:

        * The full-sized **Fitness** tab's ``PlotWidget`` —
          two-series (best + current) high-res chart. Repaints
          incrementally on data updates.
        * The compact sparkline in the **Statistics** tab —
          unchanged for backwards compat + as a quick-glance
          summary for viewports where the user hasn't switched
          to the dedicated Fitness tab yet.

        Silently no-ops when either pane isn't mounted (older
        EvolutionScreen subclasses / tests that don't compose
        the full screen)."""
        if not self.is_mounted:
            return
        records = self.tracker.fitness_curve()
        # Statistics-tab sparkline.
        try:
            from care.runtime.fitness_plot import render_fitness_plot

            pane = self.query_one(
                "#evolution-fitness-text", Static,
            )
            plot = render_fitness_plot(records, width=60, height=10)
            pane.update(plot or self._empty_pane_text("fitness data"))
        except Exception as exc:
            _log.debug("fitness sparkline pane not mounted: %s", exc)
        # Fitness-tab full-sized chart.
        self._render_fitness_plot_widget(records)

    def _render_fitness_plot_widget(self, records) -> None:
        """Drive the ``textual-plot`` widget on the Fitness tab.

        ``records`` is an iterable of ``GenerationStat``-shaped
        objects (``generation``, ``best_fitness``, optional
        ``mean_fitness``). Filters out placeholder ``-inf`` best
        values the tracker uses to mark "generation started but
        no winner reported" so the plot doesn't drop to the
        floor.
        """
        if not self.is_mounted:
            return
        try:
            from textual_plot import PlotWidget

            plot_widget = self.query_one(
                "#evolution-fitness-plot", PlotWidget,
            )
        except Exception:
            return
        xs: list[int] = []
        best_ys: list[float] = []
        mean_xs: list[int] = []
        mean_ys: list[float] = []
        for r in records:
            gen = getattr(r, "generation", None)
            if not isinstance(gen, int):
                continue
            bf = getattr(r, "best_fitness", None)
            if isinstance(bf, (int, float)) and bf != float("-inf"):
                xs.append(gen)
                best_ys.append(float(bf))
            mf = getattr(r, "mean_fitness", None)
            if isinstance(mf, (int, float)):
                mean_xs.append(gen)
                mean_ys.append(float(mf))
        if not xs and not mean_xs:
            return
        # De-flicker: the status poll fires every ~2s but the curve
        # usually hasn't changed. Skip the clear+replot when the data is
        # identical to what's already on the widget.
        signature = (tuple(xs), tuple(best_ys), tuple(mean_xs), tuple(mean_ys))
        if signature == getattr(self, "_last_fitness_plot_sig", None):
            return
        try:
            plot_widget.clear()
        except Exception:
            return
        self._last_fitness_plot_sig = signature
        # ``set_xlabel`` / ``set_ylabel`` exist on the widget;
        # call them defensively in case the dep drops them in a
        # future version (older textual-plot variants use
        # ``xlabel=`` kwarg on ``plot`` instead).
        for attr, value in (
            ("set_xlabel", "Generation"),
            ("set_ylabel", "Fitness"),
        ):
            fn = getattr(plot_widget, attr, None)
            if callable(fn):
                try:
                    fn(value)
                except Exception:
                    pass
        # Mean / current-fitness series — softer colour so the
        # best-fitness frontier reads first.
        if mean_xs and mean_ys:
            try:
                plot_widget.plot(
                    x=mean_xs,
                    y=mean_ys,
                    line_style="grey50",
                )
            except Exception:
                pass
        # Best-fitness frontier — bright accent so it stands
        # out as the headline metric.
        if xs and best_ys:
            try:
                plot_widget.plot(
                    x=xs,
                    y=best_ys,
                    line_style="green",
                )
            except Exception:
                pass

    def _render_pareto_pane(self) -> None:
        """Update the `#evolution-pareto-plot` pane with the
        latest 2D scatter of objective_0 × objective_1.

        Pares down when the run carries < 2 objectives — the
        placeholder text stays so single-objective runs don't
        get an empty rectangle. Uses plotext when
        `care[plots]` is installed; falls back to a textual
        summary otherwise (§5 P1). Silently no-ops when the
        pane isn't mounted."""
        if not self.is_mounted:
            return
        try:
            from care.runtime.pareto_plot import (
                render_pareto_scatter,
            )

            pane = self.query_one(
                "#evolution-pareto-plot-text", Static,
            )
        except Exception as exc:
            _log.debug("pareto pane not mounted: %s", exc)
            return
        front_ids = tuple(
            ind.individual_id
            for ind in self.pareto_front_individuals()
        )
        plot = render_pareto_scatter(
            self.run.individuals,
            front_ids=front_ids,
            width=60, height=10,
        )
        if plot:
            pane.update(plot)
            return
        # No scatter: either nothing evaluated yet, or — the common
        # case — a single-objective run where fitness IS the only
        # objective. Give a meaningful message instead of the dead
        # "(needs ≥ 2 objectives)" so the pane isn't a mystery.
        pane.update(self._pareto_placeholder())

    def _pareto_placeholder(self) -> str:
        """Explain why the 2D Pareto scatter is empty for this run."""
        inds = self.run.individuals
        if not inds:
            return self._empty_pane_text("Pareto data")
        multi = any(len(getattr(i, "objectives", ()) or ()) >= 2 for i in inds)
        if multi:
            return self._empty_pane_text("Pareto data")
        best = self.tracker.best_overall
        best_line = (
            f"\nBest fitness so far: [green]{best.best_fitness:.4f}[/green]"
            if best is not None
            else ""
        )
        return (
            "Single-objective run — a 2D Pareto front needs ≥ 2 objectives.\n"
            "Fitness is the only objective here, so see the [b]Fitness[/b] "
            "tab for the trajectory and the ranked table below for the top "
            f"candidates.{best_line}"
        )

    def format_cost_meter(self) -> str:
        """§5 P1 — one-line cost summary for the status pane.

        Shape: ``"cost: $0.42  (1,234 tokens — 700 in / 534 out)"``.
        Falls back to ``"cost: 1,234 tokens"`` when no USD spend
        was reported (some platforms ship token counts but not
        cost). Returns an empty string when nothing was
        reported at all — the caller drops the line entirely so
        runs that don't surface cost telemetry don't show a
        zero-everything line.
        """
        tokens = self.run.total_tokens
        cost = self.run.total_cost_usd
        if not tokens and not cost:
            return ""
        parts: list[str] = ["cost:"]
        if cost > 0:
            parts.append(f"${cost:,.2f}")
        if tokens:
            breakdown_bits: list[str] = []
            if self.run.prompt_tokens:
                breakdown_bits.append(
                    f"{self.run.prompt_tokens:,} in"
                )
            if self.run.completion_tokens:
                breakdown_bits.append(
                    f"{self.run.completion_tokens:,} out"
                )
            token_str = f"{tokens:,} tokens"
            if breakdown_bits and len(breakdown_bits) == 2:
                # Only render the breakdown when both sides
                # have a non-zero split — otherwise it's just
                # noise.
                token_str += f" ({' / '.join(breakdown_bits)})"
            if cost > 0:
                parts.append(f" ({token_str})")
            else:
                parts.append(token_str)
        return " ".join(parts)

    def format_fitness_curve(self, *, width: int = 6) -> str:
        """Render the tracker's fitness curve as a tiny text strip
        the status pane shows beneath the run summary.

        Format: ``gen 0: 0.420   gen 1: 0.553   ...`` for the last
        ``width`` generations. Placeholder ``-inf`` records (from
        ``generation_started`` events with no winners yet) are
        skipped. Empty tracker → empty string so the caller can
        decide whether to render a header.
        """
        records = [
            stat for stat in self.tracker.fitness_curve()
            if stat.best_fitness != float("-inf")
        ]
        if not records:
            return ""
        tail = records[-width:]
        return "   ".join(
            f"gen {s.generation}: {s.best_fitness:.3f}" for s in tail
        )

    def evolved_diff_lines(
        self,
        evolved_content: dict | None = None,
    ) -> tuple[str, ...]:
        """Public helper — return the unified-diff lines between
        the base chain content the screen was seeded with and the
        evolved chain content the caller supplies.

        Wraps :func:`care.evolution_diff` with sensible labels
        derived from :attr:`plan.base_chain_name`. The future
        "show diff" pane drives this; tests exercise it directly.
        Returns an empty tuple when either side is missing —
        callers render "no diff available" without a special
        branch.
        """
        base_label = (
            f"{self.plan.base_chain_name} (base)"
            if self.plan.base_chain_name else "base"
        )
        evolved_label = (
            f"{self.plan.base_chain_name} (evolved)"
            if self.plan.base_chain_name else "evolved"
        )
        return evolution_diff(
            self.plan.base_chain_content,
            evolved_content,
            base_label=base_label,
            evolved_label=evolved_label,
        )

    def _render_pareto_table(self) -> None:
        if not self.is_mounted:
            return
        try:
            table = self.query_one(
                "#evolution-pareto-table", DataTable,
            )
        except Exception:
            return
        try:
            table.clear()
        except Exception:
            pass
        # Compute the Pareto front once per render so each row's
        # badge column matches the screen's reported front.
        front_ids = {
            ind.individual_id for ind in self.pareto_front_individuals()
        }
        best_id = self._best_individual_id()
        for ind in self.run.individuals:
            fitness = (
                f"{ind.fitness:.3f}" if ind.fitness is not None else "—"
            )
            # Badge the overall best distinctly (★★) from other
            # front members (★) so the winner stands out.
            if ind.individual_id == best_id:
                badge = "★★"
            elif ind.individual_id in front_ids:
                badge = "★"
            else:
                badge = ""
            table.add_row(
                badge,
                ind.individual_id[:18],
                str(ind.generation),
                fitness,
                _format_objectives_inline(ind.objectives),
                ind.summary,
                key=ind.individual_id,
            )

    def _render_pareto_detail(self) -> None:
        """Show the highlighted individual's full (untruncated) summary +
        all objectives + fitness/gen in the detail card beneath the table."""
        if not self.is_mounted:
            return
        try:
            pane = self.query_one("#evolution-pareto-detail", Static)
        except Exception:
            return
        ind_id = getattr(self, "_highlighted_individual", None)
        ind = next(
            (i for i in self.run.individuals if i.individual_id == ind_id),
            None,
        )
        if ind is None:
            pane.update(t("evolution.paretoDetailEmpty"))
            return
        from rich.markup import escape as _esc

        lines: list[str] = [f"[b]{_esc(ind.individual_id)}[/b]  ·  gen {ind.generation}"]
        markers = []
        if ind.individual_id == self._best_individual_id():
            markers.append("[green]★★ best[/green]")
        elif ind.individual_id in {
            i.individual_id for i in self.pareto_front_individuals()
        }:
            markers.append("★ on front")
        if ind.individual_id == self.run.accepted_id:
            markers.append("[green]accepted[/green]")
        if markers:
            lines.append("  ·  ".join(markers))
        if ind.fitness is not None:
            lines.append(f"[b]fitness:[/b] {ind.fitness:.6f}")
        if ind.objectives:
            lines.append("[b]objectives:[/b]")
            for key, value in ind.objectives:
                lines.append(f"  • {_esc(str(key))} = {float(value):.4f}")
        if ind.summary:
            lines.append("[b]summary:[/b]")
            lines.append(_esc(ind.summary))
        pane.update("\n".join(lines))

    # ------------------------------------------------------------------
    # Pareto-front projection (§7 P2 UI half)
    # ------------------------------------------------------------------

    def pareto_front_individuals(
        self,
    ) -> tuple[EvolutionIndividual, ...]:
        """Return the non-dominated subset of `self.run.individuals`.

        Projects each :class:`EvolutionIndividual` (the screen's
        SSE-fed row shape) into a
        :class:`care.micro_evolution.Individual` so the shipped
        :func:`care.compute_pareto_front` data layer can compute
        the front. The screen's directions kwarg drives per-
        objective maximise / minimise direction.

        Returns:
            Tuple of :class:`EvolutionIndividual` instances —
            the original screen-shape rows, NOT the projection
            adapter, so callers can render them directly. Empty
            input → empty tuple. When no individual carries any
            objectives, the function falls back to the scalar
            ``fitness`` comparison (highest wins, ties included).
        """
        rows = list(self.run.individuals)
        if not rows:
            return ()
        # Build a parallel `_ParetoIndividual` list for the data
        # layer, indexing by position so we can map results back
        # to the original `EvolutionIndividual` rows.
        adapters: list[_ParetoIndividual] = []
        for ind in rows:
            breakdown = {k: float(v) for k, v in ind.objectives}
            adapters.append(_ParetoIndividual(
                chain=ind.individual_id,
                score=float(ind.fitness) if ind.fitness is not None else 0.0,
                breakdown=breakdown,
                generation=ind.generation,
            ))
        front_adapters = compute_pareto_front(
            adapters, directions=self.directions or None,
        )
        # Map adapter chain (= individual_id) back to the original
        # row. Use a dict for O(N) lookup since adapters may be
        # returned out of input order in edge cases.
        by_id = {ind.individual_id: ind for ind in rows}
        return tuple(
            by_id[a.chain]
            for a in front_adapters
            if a.chain in by_id
        )

    def format_pareto_front(self) -> str:
        """One-line summary of the Pareto front for tests / logs.

        Empty front → ``"no non-dominated individuals"``. Otherwise
        ``"N on front: id1, id2, ..."`` (up to 5 ids; ellipsis after).
        """
        front = self.pareto_front_individuals()
        if not front:
            return "no non-dominated individuals"
        ids = [ind.individual_id for ind in front[:5]]
        suffix = "" if len(front) <= 5 else "…"
        return f"{len(front)} on front: " + ", ".join(ids) + suffix

    def _render_event_row(self, kind: str, payload: dict) -> None:
        """Mount one human-readable line per Platform event.

        Each entry follows ``HH:MM:SS <kind> · <detail>`` so the
        EVENTS pane reads like a chat transcript instead of a
        cryptic ``gen 39`` repeat. ``detail`` is built from the
        most useful payload fields for the given ``kind``;
        anything we don't recognise falls back to ``gen N`` so
        the screen still shows something for new event types.
        """
        if not self.is_mounted:
            return
        try:
            container = self.query_one(
                "#evolution-events-log", VerticalScroll,
            )
        except Exception:
            return
        import time as _time

        clock = _time.strftime("%H:%M:%S")
        detail = self._format_event_detail(kind, payload)
        # Style cue per event family so the user can scan the
        # log quickly: green = progress / completion, yellow =
        # status transitions, red = errors, grey = chatter.
        style = {
            "best_updated": "[green]",
            "individual_evaluated": "[green]",
            "completed": "[green b]",
            "accepted": "[green b]",
            "generation_started": "",
            "status": "[yellow]",
            "fitness_history_snapshot": "[grey50]",
            "cost_tick": "[grey50]",
            "snapshot": "[grey50]",
            "failed": "[red b]",
            "cancelled": "[red]",
            "error": "[red]",
        }.get(kind, "")
        close = "[/]" if style else ""
        line = f"[grey50]{clock}[/] {style}{kind}{close}"
        if detail:
            line += f" [grey50]·[/] {detail}"
        container.mount(Static(line, markup=True))
        # Auto-scroll so the latest line stays visible.
        try:
            container.scroll_end(animate=False)
        except Exception:
            pass

    def _format_event_detail(self, kind: str, payload: dict) -> str:
        """Project ``payload`` into a one-liner describing the
        most user-relevant fields for ``kind``.

        Kept on the screen (not the SDK) because the SSE schema
        is best-effort: fields that mean "this generation's mean
        fitness" can ride under different names depending on
        platform version. Falls back to ``gen N`` so unknown
        kinds still surface something."""
        def _f(value, fmt: str = ".3f") -> str | None:
            if isinstance(value, (int, float)):
                try:
                    return format(float(value), fmt)
                except (TypeError, ValueError):
                    return None
            return None

        gen = payload.get("generation")
        parts: list[str] = []
        if kind == "status":
            status = payload.get("status") or payload.get("data", {}).get("status")
            if status:
                parts.append(f"→ {status}")
            msg = payload.get("status_message") or payload.get("message")
            if msg:
                parts.append(f"({msg})")
        elif kind == "generation_started":
            if isinstance(gen, int):
                parts.append(f"gen {gen}")
            cf = _f(payload.get("current_fitness"))
            if cf:
                parts.append(f"current={cf}")
            pv = payload.get("programs_valid")
            pi = payload.get("programs_invalid")
            if isinstance(pv, int) and isinstance(pi, int):
                parts.append(f"valid {pv}/{pv+pi}")
        elif kind == "best_updated":
            if isinstance(gen, int):
                parts.append(f"gen {gen}")
            bf = _f(payload.get("best_fitness"))
            if bf:
                parts.append(f"best={bf}")
            cf = _f(payload.get("current_fitness"))
            if cf:
                parts.append(f"current={cf}")
        elif kind == "individual_evaluated":
            ind = payload.get("individual_id") or payload.get("id")
            if ind:
                parts.append(f"{str(ind)[:12]}…")
            f = _f(payload.get("fitness"))
            if f:
                parts.append(f"fitness={f}")
        elif kind == "fitness_history_snapshot":
            history = payload.get("history") or []
            if history:
                parts.append(f"+{len(history)} pts")
                first_gen = history[0].get("generation")
                last_gen = history[-1].get("generation")
                if first_gen is not None and last_gen is not None:
                    parts.append(f"gen {first_gen}..{last_gen}")
        elif kind == "snapshot":
            status = payload.get("status")
            if status:
                parts.append(str(status))
            sa = payload.get("started_at")
            if isinstance(sa, str) and sa:
                parts.append(f"started {sa[11:19]}")
        elif kind == "cost_tick":
            tokens = payload.get("total_tokens") or payload.get("tokens")
            cost = _f(
                payload.get("cost_usd") or payload.get("cost"),
                fmt=".4f",
            )
            if tokens:
                parts.append(f"{tokens} tok")
            if cost:
                parts.append(f"${cost}")
        elif kind in {"failed", "cancelled", "error"}:
            err = payload.get("error") or payload.get("error_message")
            if err:
                parts.append(str(err)[:80])
        elif kind == "completed":
            bf = _f(payload.get("best_fitness"))
            if bf:
                parts.append(f"best={bf}")
            if isinstance(gen, int):
                parts.append(f"at gen {gen}")
        elif kind == "accepted":
            ind = payload.get("individual_id")
            if ind:
                parts.append(f"{str(ind)[:12]}…")
        # Fallback — show at least the gen so users see SOMETHING.
        if not parts and isinstance(gen, int):
            parts.append(f"gen {gen}")
        return " · ".join(parts)

    # ------------------------------------------------------------------
    # Selection + actions
    # ------------------------------------------------------------------

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected,
    ) -> None:
        if event.data_table.id == "evolution-versions-table":
            self._select_version_at_cursor(event)
            return
        if event.data_table.id != "evolution-pareto-table":
            return
        try:
            self.selected_individual = str(event.row_key.value or "")
        except Exception:
            self.selected_individual = None
        self._refresh_individual_preview()

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted,
    ) -> None:
        """Cursor move → live-update the preview pane.

        Distinct from row_selected (Enter / click) so the user
        sees the chain change as they navigate the table
        without having to commit a selection. The committed
        `selected_individual` tracker only updates on
        row_selected.
        """
        if event.data_table.id == "evolution-versions-table":
            self._select_version_at_cursor(event)
            return
        if event.data_table.id != "evolution-pareto-table":
            return
        try:
            highlighted_id = str(event.row_key.value or "")
        except Exception:
            highlighted_id = ""
        self._highlighted_individual = highlighted_id or None
        self._refresh_individual_preview()
        self._render_pareto_detail()

    def _select_version_at_cursor(self, event: Any) -> None:
        """Project a row event from the versions table into the
        ``_selected_version_idx`` field + repaint."""
        try:
            cursor_row = event.cursor_row
        except AttributeError:
            cursor_row = getattr(getattr(event, "data_table", None), "cursor_row", None)
        if not isinstance(cursor_row, int):
            return
        if cursor_row < 0 or cursor_row >= len(self._versions):
            return
        self._selected_version_idx = cursor_row
        self._render_versions_preview()

    def _refresh_individual_preview(self) -> None:
        """Render the highlighted (or selected) individual's
        chain into the `#evolution-individual-text` Static.
        Falls back to the placeholder when nothing's
        highlighted, a friendly note when the chain payload
        wasn't shipped in the SSE event."""
        try:
            target = self.query_one(
                "#evolution-individual-text", Static,
            )
        except Exception:
            return
        # Prefer the live-highlighted row; fall back to the
        # last committed selection so the preview survives a
        # focus shift to a button.
        ind_id = getattr(self, "_highlighted_individual", None) or (
            self.selected_individual or ""
        )
        if not ind_id:
            target.update("Select a row to preview its chain.")
            return
        ind = next(
            (
                e for e in self.run.individuals
                if e.individual_id == ind_id
            ),
            None,
        )
        if ind is None:
            target.update(f"Individual {ind_id} not found in run state.")
            return
        header_lines = [
            f"id: {ind.individual_id}",
            f"generation: {ind.generation}",
            f"fitness: {ind.fitness if ind.fitness is not None else '—'}",
        ]
        if ind.summary:
            header_lines.append(f"summary: {ind.summary}")
        if ind.chain_dict is None:
            body = "(no chain content shipped with this individual)"
        else:
            from care.screens.inspection import render_chain_dag

            steps = ind.chain_dict.get("steps") or []
            body = render_chain_dag(steps)
        target.update("\n".join(header_lines) + "\n\n" + body)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "evolution-btn-versions-mode":
            # Toggle Chain ↔ Diff view in the Versions tab and
            # relabel the button so the user can see which view
            # they'll get on next press.
            self._versions_mode = "diff" if self._versions_mode == "chain" else "chain"
            self._sync_versions_mode_button()
            self._render_versions_preview()
            return
        if bid == "evolution-btn-accept":
            self.action_accept_winner()
        elif bid == "evolution-btn-archive":
            self.action_archive_run()
        elif bid == "evolution-btn-back":
            try:
                self.app.pop_screen()
            except Exception:
                pass

    def action_accept_winner(self) -> None:
        """§5 P0 — accept-winner now pushes a confirmation
        modal so the user sees `chain_id` + the version
        transition before the irreversible `latest` pointer
        flip. Bypassed when `app` has no `push_screen` (rare
        unit-test scaffolds); the worker then fires
        directly so legacy tests keep their assertions."""
        winner = self.selected_individual or self._best_individual_id()
        if winner is None:
            return
        try:
            from care.screens.confirm import ConfirmModal
        except Exception:  # noqa: BLE001
            ConfirmModal = None  # type: ignore[assignment]
        push = getattr(self.app, "push_screen", None) if self.app else None
        if ConfirmModal is None or not callable(push):
            self._spawn_accept_worker(winner)
            return
        modal = ConfirmModal(
            title=t("evolution.acceptConfirmTitle"),
            body=(
                f"chain_id: {self.run.base_chain_id or '?'}\n"
                f"individual: {winner}\n"
                f"evolution: {self.run.evolution_id or '?'}\n"
                f"\n{t('evolution.acceptConfirmBody')}"
            ),
            confirm_label=t("evolution.acceptWinner"),
            cancel_label=t("common.cancel"),
        )

        def _on_dismiss(confirmed: bool | None) -> None:
            if confirmed:
                self._spawn_accept_worker(winner)

        try:
            self.app.push_screen(modal, _on_dismiss)
        except Exception:  # noqa: BLE001
            # Fall back to direct dispatch when the host
            # rejects the push (e.g. screen stack locked).
            self._spawn_accept_worker(winner)

    def _spawn_accept_worker(self, winner: str) -> None:
        self.run_worker(
            self._accept_winner(winner),
            name="evolution_accept",
            group="evolution",
            exclusive=False,
            exit_on_error=False,
        )

    def _best_individual_id(self) -> str | None:
        # Highest fitness first (post-sort). Falls back to the
        # first row when none carry a fitness.
        if not self.run.individuals:
            return None
        return self.run.individuals[0].individual_id

    def _selected_individual_chain(
        self, individual_id: str
    ) -> dict[str, Any] | None:
        """Chain content for the accepted individual, when known.

        Frontier-derived rows carry their per-generation ``chain_dict``;
        returns it (only when it's a real chain with ``steps``) so accept
        promotes the selected chain. ``None`` → the platform falls back to
        the run's overall ``best_chain_config``."""
        for ind in self.run.individuals:
            if ind.individual_id == individual_id:
                chain = ind.chain_dict
                if isinstance(chain, dict) and chain.get("steps"):
                    return chain
                return None
        return None

    async def _accept_winner(self, individual_id: str) -> None:
        platform = getattr(self.app, "platform", None)
        if platform is None or not self.run.evolution_id:
            return
        memory = getattr(self.app, "memory", None)
        # Promote the SELECTED individual's chain when we have its content
        # (frontier rows carry their per-generation chain) so picking a
        # non-best row in the Pareto table actually promotes that chain
        # instead of silently falling back to the overall best.
        chain_override = self._selected_individual_chain(individual_id)
        try:
            response = await asyncio.to_thread(
                platform.accept_individual,
                self.run.evolution_id,
                individual_id,
                memory=memory,
                chain_override=chain_override,
            )
        except TypeError:
            # Older facade without the ``memory``/``chain_override`` kwargs
            # (test stubs, downgraded SDK) — retry with the legacy
            # signature so we don't break in those scaffolds.
            try:
                response = await asyncio.to_thread(
                    platform.accept_individual,
                    self.run.evolution_id,
                    individual_id,
                )
            except Exception as exc:  # noqa: BLE001
                self.run.last_error = f"{type(exc).__name__}: {exc}"
                self.refresh_status()
                try:
                    self.app.notify(
                        t(
                            "evolution.acceptFailed",
                            error=f"{type(exc).__name__}: {exc}",
                        ),
                        title=t("evolution.acceptErrorTitle"),
                        severity="error",
                        timeout=10,
                    )
                except Exception:
                    pass
                return
        except ValueError as exc:
            # ``CarePlatform`` surfaces user-actionable errors
            # (missing best_chain_config, missing base_chain_id,
            # missing memory facade) as ``ValueError``. Toast +
            # log them so the user sees what to fix.
            self.run.last_error = str(exc)
            self.refresh_status()
            try:
                self.app.notify(
                    str(exc),
                    title=t("evolution.acceptBlockedTitle"),
                    severity="warning",
                    timeout=10,
                )
            except Exception:
                pass
            return
        except Exception as exc:  # noqa: BLE001
            self.run.last_error = f"{type(exc).__name__}: {exc}"
            self.refresh_status()
            try:
                self.app.notify(
                    t(
                        "evolution.acceptFailed",
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    title=t("evolution.acceptErrorTitle"),
                    severity="error",
                    timeout=10,
                )
            except Exception:
                pass
            return
        self.run.accepted_id = individual_id
        self.accept_result = response if isinstance(response, dict) else {}
        self.refresh_status()
        # §5 P0 — extract version + chain_id from the response
        # when the platform ships them. Falls back to the
        # screen's base_chain_id so the toast at least names
        # the chain even on older platforms.
        chain_id = (
            str(self.accept_result.get("chain_id") or "")
            or self.run.base_chain_id
            or ""
        )
        previous_version = _coerce_int(
            self.accept_result.get("previous_version"),
        )
        new_version = _coerce_int(
            self.accept_result.get("new_version")
            or self.accept_result.get("version"),
        )
        self.post_message(
            self.AcceptanceComplete(
                self.run.evolution_id,
                individual_id,
                chain_id=chain_id,
                previous_version=previous_version,
                new_version=new_version,
            ),
        )

    def action_cancel_evolution(self) -> None:
        self.cancelled = True
        try:
            self.workers.cancel_group(self, "evolution")
        except Exception:
            pass
        # Stop the run server-side too — without this the local
        # workers stop but the runner keeps churning, burning LLM
        # tokens / runner capacity. Best-effort: failures here
        # (e.g. platform unreachable or test stub without
        # ``cancel``) shouldn't block the UI transition.
        platform = getattr(self.app, "platform", None)
        evo_id = self.run.evolution_id
        cancel_fn = getattr(platform, "cancel", None) if platform is not None else None
        if callable(cancel_fn) and evo_id:
            try:
                self.run_worker(
                    lambda: cancel_fn(evo_id),
                    group="evolution",
                    name=f"cancel-{evo_id}",
                    exclusive=False,
                    thread=True,
                )
            except Exception:
                pass
        self.run.status = "cancelled"
        self.refresh_status()

    def action_archive_run(self) -> None:
        """Stop the run server-side, add it to the dashboard's
        archive list, and pop back to whatever pushed this
        screen. Combines what would otherwise be three
        keystrokes (``Esc`` to cancel, then ``a`` to archive
        on the dashboard, then ``b`` to go back) into one for
        the common "this run is a dud, hide it" case."""
        evo_id = self.run.evolution_id
        # Cancel server-side + locally — same path as Esc.
        self.action_cancel_evolution()
        if evo_id:
            try:
                from care.screens.evolution_dashboard import (
                    _load_archive,
                    _save_archive,
                )

                archived = _load_archive()
                archived.add(evo_id)
                _save_archive(archived)
            except Exception:
                pass
            try:
                self.app.notify(
                    t("evolution.runArchived", id=evo_id[:18]),
                    title=t("evolution.runArchivedTitle"),
                    severity="information",
                    timeout=6,
                )
            except Exception:
                pass
        try:
            self.app.pop_screen()
        except Exception:
            pass

    def action_export_curve(self) -> None:
        """Export the fitness curve to ``evolution-<id>-curve.{csv,json}``
        in the cwd (CSV for spreadsheets/eval frameworks, JSON for
        tooling). Toasts the path, or a friendly hint when there's no
        curve yet."""
        from care.runtime.fitness_export import (
            fitness_curve_csv,
            fitness_curve_json,
            fitness_curve_rows,
        )

        records = self.tracker.fitness_curve()
        if not fitness_curve_rows(records):
            self._toast(
                "No fitness curve to export yet — wait for the first "
                "generation to report a fitness.",
                severity="info",
            )
            return
        import re
        from pathlib import Path

        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.run.evolution_id or "run")[:60]
        stem = f"evolution-{slug or 'run'}-curve"
        try:
            csv_path = Path(f"{stem}.csv")
            json_path = Path(f"{stem}.json")
            csv_path.write_text(fitness_curve_csv(records), encoding="utf-8")
            json_path.write_text(fitness_curve_json(records), encoding="utf-8")
        except OSError as exc:
            self._toast(f"Couldn't write curve export: {exc}", severity="error")
            return
        self._toast(
            f"Exported fitness curve → {csv_path} + {json_path.name}",
            severity="success",
        )

    def action_export_individual(self) -> None:
        """§5 P1 — export the highlighted Pareto-front
        individual's chain payload to disk via the
        ExportChainModal (JSON / Python switch).

        Routes through `self._highlighted_individual` first
        (cursor-driven) then falls back to
        `selected_individual`. Toasts a friendly hint when no
        row is highlighted or the individual hasn't shipped a
        chain payload yet (older platform SSE events).
        """
        target_id = (
            self._highlighted_individual
            or self.selected_individual
        )
        if target_id is None:
            self._toast(
                "Highlight a Pareto-front row first "
                "(arrow keys), then press `x` to export.",
                severity="info",
            )
            return
        individual = next(
            (ind for ind in self.run.individuals
             if ind.individual_id == target_id),
            None,
        )
        if individual is None or individual.chain_dict is None:
            self._toast(
                f"Individual `{target_id}` has no chain payload "
                "to export yet — wait for the next "
                "`individual_evaluated` event with a chain "
                "shipped.",
                severity="warning",
            )
            return
        try:
            from care.screens.export_chain import (
                ExportChainModal,
                ExportChainResult,
            )
        except Exception as exc:  # noqa: BLE001
            self._toast(
                f"Couldn't open chain export modal: {exc}",
                severity="error",
            )
            return

        display_name = (
            individual.summary
            or f"{self.plan.base_chain_name or 'evolved'}"
            f"-{target_id}"
        )

        def _on_dismiss(result: ExportChainResult | None) -> None:
            if result is None or not result.ok:
                return
            self._toast(
                f"Exported chain to {result.path} "
                f"({result.bytes_written:,} bytes, "
                f"{result.format}).",
                severity="success",
            )

        push = getattr(self.app, "push_screen", None)
        if not callable(push):
            self._toast(
                "Host doesn't expose `push_screen` — "
                "can't open the export modal.",
                severity="error",
            )
            return
        try:
            push(
                ExportChainModal(
                    chain=individual.chain_dict,
                    display_name=display_name,
                ),
                _on_dismiss,
            )
        except Exception as exc:  # noqa: BLE001
            self._toast(
                f"Couldn't push chain export modal: {exc}",
                severity="error",
            )

    def action_compare_to_parent(self) -> None:
        """§5 P1 — open a DiffModal between the seed/parent
        chain and the highlighted Pareto-front individual.

        Resolves the target via the cursor-driven
        ``_highlighted_individual`` first, falling back to the
        committed ``selected_individual``. The parent is the
        seed chain the screen was constructed with
        (``self.plan.base_chain_content``).

        Friendly-toast failure modes:

        * No row highlighted → "Highlight a Pareto-front row
          first…"
        * Individual hasn't shipped a chain payload yet (older
          platform SSE events) → wait-for-next-event hint
        * Seed chain content wasn't available at screen
          construction → can't diff against a missing seed

        Older platforms don't ship ``parent_id`` on
        ``individual_evaluated`` events, so individuals whose
        true parent is another evolved individual still diff
        against the original seed — filed as a platform
        follow-up under `[→ gigaevo-platform]`.
        """
        target_id = (
            self._highlighted_individual
            or self.selected_individual
        )
        if target_id is None:
            self._toast(
                "Highlight a Pareto-front row first "
                "(arrow keys), then press `D` to diff against "
                "the parent chain.",
                severity="info",
            )
            return
        individual = next(
            (ind for ind in self.run.individuals
             if ind.individual_id == target_id),
            None,
        )
        if individual is None or individual.chain_dict is None:
            self._toast(
                f"Individual `{target_id}` has no chain payload "
                "to diff yet — wait for the next "
                "`individual_evaluated` event with a chain "
                "shipped.",
                severity="warning",
            )
            return
        parent = self.plan.base_chain_content
        if not parent:
            self._toast(
                "No seed chain content available — the screen "
                "was constructed without `base_chain_content`, "
                "so there's nothing to diff against.",
                severity="warning",
            )
            return
        try:
            from care.screens.diff import DiffModal
        except Exception as exc:  # noqa: BLE001
            self._toast(
                f"Couldn't open diff modal: {exc}",
                severity="error",
            )
            return
        push = getattr(self.app, "push_screen", None)
        if not callable(push):
            self._toast(
                "Host doesn't expose `push_screen` — "
                "can't open the diff modal.",
                severity="error",
            )
            return
        parent_label = (
            f"seed ({self.plan.base_chain_name})"
            if self.plan.base_chain_name
            else "seed"
        )
        try:
            push(DiffModal(
                left_payload=parent,
                right_payload=individual.chain_dict,
                left_label=parent_label,
                right_label=target_id,
            ))
        except Exception as exc:  # noqa: BLE001
            self._toast(
                f"Couldn't push diff modal: {exc}",
                severity="error",
            )

    def _toast(self, message: str, *, severity: str = "info") -> None:
        push = getattr(self.app, "push_toast", None)
        if callable(push):
            try:
                push(message, severity=severity)
                return
            except Exception:
                pass


__all__ = [
    "EvolutionIndividual",
    "EvolutionRunState",
    "EvolutionScreen",
]
