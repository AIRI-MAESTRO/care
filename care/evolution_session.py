"""Evolution session data layer (TODO §7 P1).

The future ``EvolutionScreen`` drives the user through a
5-step flow:

1. Pick a seed chain.
2. Configure mode / iterations / population / criteria.
3. Submit + poll for status.
4. Render fitness curve + best-individual diff.
5. Accept the winner back to Memory.

CARE's :class:`CarePlatform` facade already owns step 1
(``start_evolution``) + step 5 (``accept_individual``). This
module ships the data layer for steps 2–4:

* :class:`EvolutionConfig` — frozen form-field bundle the
  screen renders as a side panel.
* :class:`EvolutionPlan` — config + base chain, ready to
  serialise into a Platform request.
* :func:`build_evolution_request` — projects a plan into the
  ``POST /api/v1/evolutions`` request body the SDK expects.
* :class:`EvolutionProgressTracker` — mutable per-generation
  aggregator that feeds the Rich fitness-curve plot.
* :func:`evolution_diff` — unified diff between the seed
  chain and the best individual, ready for the diff pane.

No upstream import: the Platform SDK / chain shapes stay
duck-typed. Tests drive the helpers directly.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_log = logging.getLogger("care.evolution_session")


EvolutionMode = Literal["full_chain", "per_step"]
"""Which slice of the chain GA can mutate.

- ``"full_chain"`` — every step is fair game; the GA can
  rewrite the whole DAG. Slower + more disruptive but
  produces wider variation.
- ``"per_step"`` — only one step at a time gets perturbed;
  the rest stay frozen. Faster + safer; matches the
  CARE-facing default when iterating on a step-prompt fix.
"""


@dataclass(frozen=True)
class EvolutionConfig:
    """Configuration the EvolutionScreen collects from the user.

    Frozen so it flows through the screen as a single immutable
    value. Defaults match the screen's pre-filled values; the
    user overrides individual fields and re-submits.

    Fields:
        evolution_mode: See :class:`EvolutionMode`.
        max_iterations: Upper bound on GA generations.
        population_size: How many individuals per generation.
        validation_criteria: Free-form prompt the platform's
            judge LLM uses to score each individual. Empty
            string means rely on Platform defaults.
        test_data_path: Optional path to a CSV / JSONL file
            with eval cases. ``None`` runs validation against
            the seed chain's task description only.
        validation_threshold: Optional 0-1 cutoff — Platform
            stops early when a generation's best meets it.
            ``None`` lets the GA run to ``max_iterations``.
        validation_type: Platform judge mode — ``Continuous (0..1)``
            or ``Binary (0/1)``. Forwarded to chain-experiment create.
        continuous_metric: ROUGE/BERTScore/BLEU when continuous.
        binary_method: equality / substring / regexp when binary.
        target_column: Dataset column scored against chain output.
        objectives: Multi-objective hint (TODO §7 P2 multi-
            objective work used this same shape). Free-form
            list of objective names — empty defaults to
            ``["fitness"]``.
        mutation_max_tokens: Per-run override for the mutation
            LLM completion limit. ``None`` uses
            :attr:`PlatformConfig.mutation_max_tokens`.
    """

    evolution_mode: EvolutionMode = "full_chain"
    max_iterations: int = 5
    population_size: int = 8
    validation_criteria: str = ""
    test_data_path: Path | None = None
    validation_threshold: float | None = None
    validation_type: str = "Continuous (0..1)"
    continuous_metric: str = "ROUGE-L"
    binary_method: str = "equality"
    target_column: str = "expected"
    objectives: tuple[str, ...] = field(default_factory=tuple)
    mutation_max_tokens: int | None = None


class EvolutionPlanError(RuntimeError):
    """Raised when the plan can't be projected into a Platform
    request — missing seed chain, invalid config combination."""


@dataclass(frozen=True)
class EvolutionPlan:
    """Ready-to-submit evolution job: config + seed chain.

    Frozen so the screen passes a single value into
    :func:`build_evolution_request` instead of juggling
    separate args.

    Fields:
        config: :class:`EvolutionConfig` the user assembled.
        base_chain_entity_id: Memory id of the seed chain the
            GA evolves from. The Platform pulls the content
            via the SDK; CARE just refers to it by id.
        base_chain_content: Optional seed-chain content
            (``ReasoningChain.to_dict()`` output). Required
            for diff rendering against the best individual;
            the SDK doesn't need it (server already has the
            content).
        base_chain_name: Display name — surfaces in the screen
            header + the diff labels.
    """

    config: EvolutionConfig
    base_chain_entity_id: str
    base_chain_content: dict[str, Any] = field(default_factory=dict)
    base_chain_name: str = ""


def build_evolution_request(plan: EvolutionPlan) -> dict[str, Any]:
    """Project a plan into a `POST /api/v1/evolutions` body.

    The Platform's ``PlatformClient.create_evolution(...)``
    accepts a body dict directly; this helper does the
    user-facing → server-facing field mapping so the screen
    doesn't have to remember the names.

    Args:
        plan: An :class:`EvolutionPlan`.

    Returns:
        Dict ready to feed
        :meth:`gigaevo_client.PlatformClient.create_evolution`.

    Raises:
        EvolutionPlanError: When the plan is missing required
            fields (``base_chain_entity_id`` empty, illegal
            counts).
    """
    if not plan.base_chain_entity_id:
        raise EvolutionPlanError(
            "EvolutionPlan.base_chain_entity_id is empty; "
            "the GA needs a seed chain to evolve from"
        )
    if plan.config.max_iterations < 1:
        raise EvolutionPlanError(
            f"max_iterations must be >= 1; got {plan.config.max_iterations}"
        )
    if plan.config.population_size < 2:
        raise EvolutionPlanError(
            f"population_size must be >= 2; got {plan.config.population_size}"
        )
    body: dict[str, Any] = {
        "seed_chain_id": plan.base_chain_entity_id,
        "evolution_mode": plan.config.evolution_mode,
        "max_iterations": plan.config.max_iterations,
        "population_size": plan.config.population_size,
    }
    if plan.config.validation_criteria.strip():
        body["validation_criteria"] = plan.config.validation_criteria
    if plan.config.test_data_path is not None:
        # Platform takes a string path; CARE's frontend keeps it
        # as a `Path` for ergonomic file-picker integration.
        body["test_data_path"] = str(plan.config.test_data_path)
    if plan.config.validation_threshold is not None:
        body["validation_threshold"] = float(plan.config.validation_threshold)
    body["validation_type"] = plan.config.validation_type
    body["continuous_metric"] = plan.config.continuous_metric
    body["binary_method"] = plan.config.binary_method
    body["target_column"] = plan.config.target_column
    if plan.config.objectives:
        body["objectives"] = list(plan.config.objectives)
    return body


# ---------------------------------------------------------------------------
# Per-generation progress aggregator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationStat:
    """One generation's record on the fitness curve.

    Frozen so the screen's Rich plot can hold tuples without
    defensive copies.

    Fields:
        generation: 0-indexed generation number.
        best_fitness: Top fitness score this generation.
        mean_fitness: Average across the generation, when
            Platform reports it. ``None`` when only the best
            score arrives.
        individuals_evaluated: How many genomes scored this
            generation.
        best_individual_id: Platform-side identifier of the
            top individual — used by the "accept this winner"
            action.
    """

    generation: int
    best_fitness: float
    mean_fitness: float | None = None
    individuals_evaluated: int = 0
    best_individual_id: str | None = None


class EvolutionProgressTracker:
    """Mutable aggregator that turns Platform SSE events into a
    fitness curve.

    The screen's worker subscribes to
    ``PlatformClient.stream_events`` and calls
    :meth:`record_generation` for every ``individual_evaluated``
    / ``best_updated`` event. The Rich plot reads
    :meth:`fitness_curve` to render.

    Mutable on the record path; reads are thread-safe via the
    same lock pattern as `TaskRegistry`.
    """

    def __init__(self) -> None:
        self._records: dict[int, GenerationStat] = {}
        # Per-generation program counts ``gen -> (valid, invalid)``.
        # Populated from events that carry both a generation and
        # program counts so the Programs tab can show a trend, not
        # just the latest snapshot. Separate from ``_records`` because
        # program counts and fitness arrive on different events.
        self._programs: dict[int, tuple[int, int]] = {}
        # No threading.Lock — the screen's worker drives this
        # sequentially per evolution. Concurrent access can be
        # layered on later if needed; keeping it simple now.

    def record_programs(
        self, generation: int, valid: int | None, invalid: int | None
    ) -> None:
        """Record valid/invalid program counts for ``generation``.

        ``None`` for a side means "not reported"; the existing value
        (or 0) is kept so a later partial update doesn't clobber a
        real count. No-ops on a non-int generation."""
        if not isinstance(generation, int):
            return
        prev_v, prev_i = self._programs.get(generation, (0, 0))
        new_v = valid if isinstance(valid, int) and valid >= 0 else prev_v
        new_i = invalid if isinstance(invalid, int) and invalid >= 0 else prev_i
        self._programs[generation] = (new_v, new_i)

    def programs_curve(self) -> tuple[tuple[int, int, int], ...]:
        """``(generation, valid, invalid)`` rows ordered by generation."""
        return tuple(
            (gen, self._programs[gen][0], self._programs[gen][1])
            for gen in sorted(self._programs)
        )

    def record_generation(self, stat: GenerationStat) -> None:
        """Insert or update the record for ``stat.generation``.

        When the same generation arrives multiple times (e.g. a
        late ``best_updated`` after several ``individual_evaluated``
        events), the highest ``best_fitness`` wins. Other fields
        update in-place.
        """
        existing = self._records.get(stat.generation)
        if existing is None:
            self._records[stat.generation] = stat
            return
        merged_best = max(existing.best_fitness, stat.best_fitness)
        merged_mean = stat.mean_fitness if stat.mean_fitness is not None else existing.mean_fitness
        merged_evaluated = max(
            existing.individuals_evaluated, stat.individuals_evaluated
        )
        merged_best_id = (
            stat.best_individual_id
            if stat.best_fitness >= existing.best_fitness
            else existing.best_individual_id
        )
        self._records[stat.generation] = GenerationStat(
            generation=stat.generation,
            best_fitness=merged_best,
            mean_fitness=merged_mean,
            individuals_evaluated=merged_evaluated,
            best_individual_id=merged_best_id,
        )

    def record_event(self, event: dict[str, Any]) -> None:
        """Parse a Platform SSE event dict and update the
        tracker.

        Handles the documented event types:

        - ``"generation_started"`` — registers an empty record
          so the curve includes generations with no winners.
        - ``"individual_evaluated"`` — bumps the generation's
          ``individuals_evaluated`` count + tracks the best.
        - ``"best_updated"`` — pins ``best_fitness`` +
          ``best_individual_id``.

        Unknown event types are silently ignored — Platform may
        add new types and CARE shouldn't crash on them.
        """
        event_type = event.get("event_type") or event.get("type")
        if event_type not in ("generation_started", "individual_evaluated", "best_updated"):
            return
        try:
            generation = int(event.get("generation", 0))
        except (TypeError, ValueError):
            return
        if event_type == "generation_started":
            self.record_generation(
                GenerationStat(generation=generation, best_fitness=float("-inf"))
            )
            return
        fitness_raw = event.get("fitness")
        if fitness_raw is None:
            fitness_raw = event.get("best_fitness")
        fitness = _coerce_metric(fitness_raw, field="fitness", generation=generation)
        if fitness is None:
            # Present-but-malformed already warned in _coerce_metric;
            # absent → treat as 0.0 (the prior behaviour for empty events).
            fitness = 0.0
        evaluated = 1 if event_type == "individual_evaluated" else 0
        best_id = event.get("individual_id") or event.get("best_individual_id")
        self.record_generation(
            GenerationStat(
                generation=generation,
                best_fitness=fitness,
                mean_fitness=_opt_float(event.get("mean_fitness")),
                individuals_evaluated=evaluated,
                best_individual_id=str(best_id) if best_id else None,
            )
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def fitness_curve(self) -> tuple[GenerationStat, ...]:
        """Per-generation records sorted by generation. Empty
        when no events have arrived yet."""
        return tuple(
            self._records[gen] for gen in sorted(self._records)
        )

    @property
    def is_empty(self) -> bool:
        return not self._records

    @property
    def latest(self) -> GenerationStat | None:
        """Most-recent generation record (or ``None``)."""
        if not self._records:
            return None
        return self._records[max(self._records)]

    @property
    def best_overall(self) -> GenerationStat | None:
        """Highest-fitness record across every generation."""
        if not self._records:
            return None
        # Skip the placeholder `-inf` entries from
        # generation_started events.
        real_records = [
            r for r in self._records.values()
            if r.best_fitness != float("-inf")
        ]
        if not real_records:
            return None
        return max(real_records, key=lambda r: r.best_fitness)


# ---------------------------------------------------------------------------
# Diff projection
# ---------------------------------------------------------------------------


def evolution_diff(
    base_chain: dict[str, Any] | None,
    evolved_chain: dict[str, Any] | None,
    *,
    base_label: str = "base",
    evolved_label: str = "evolved",
) -> tuple[str, ...]:
    """Render a unified diff between the seed chain and the
    best individual.

    Mirrors :func:`care.conflict._unified_diff_lines` —
    `json.dumps(..., indent=2, sort_keys=True)` on both sides so
    semantically-equal dicts produce no diff regardless of key
    order. Empty / None inputs yield empty tuples; the screen
    can show "no diff available" without a special branch.

    Args:
        base_chain: Seed chain content dict.
        evolved_chain: Best-individual content dict.
        base_label: Diff label for the existing version.
        evolved_label: Diff label for the new version.

    Returns:
        Tuple of unified-diff lines ready for a Textual
        ``RichLog``.
    """
    if not base_chain or not evolved_chain:
        return ()
    a = json.dumps(
        base_chain, indent=2, sort_keys=True, ensure_ascii=False, default=str
    ).splitlines()
    b = json.dumps(
        evolved_chain, indent=2, sort_keys=True, ensure_ascii=False, default=str
    ).splitlines()
    return tuple(
        difflib.unified_diff(
            a, b,
            fromfile=base_label,
            tofile=evolved_label,
            lineterm="",
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opt_float(value: Any) -> float | None:
    """Coerce to float, returning ``None`` for absent OR malformed/NaN
    values. WARNs when a value was *present* but unusable so a data-quality
    problem (e.g. the Platform shipped ``"n/a"`` or ``NaN`` for mean
    fitness) is diagnosable instead of silently showing "—"."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        _log.warning("dropping malformed numeric value %r (expected a float)", value)
        return None
    if math.isnan(out) or math.isinf(out):
        _log.warning("dropping non-finite numeric value %r", value)
        return None
    return out


def _coerce_metric(value: Any, *, field: str, generation: int) -> float | None:
    """Like :func:`_opt_float` but tags the warning with which field /
    generation produced the bad value (used for ``fitness`` on the hot
    event path). Returns ``None`` for absent or malformed/NaN."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        _log.warning(
            "gen %s: malformed %s value %r — ignoring", generation, field, value
        )
        return None
    if math.isnan(out) or math.isinf(out):
        _log.warning(
            "gen %s: non-finite %s value %r — ignoring", generation, field, value
        )
        return None
    return out


__all__ = [
    "EvolutionConfig",
    "EvolutionMode",
    "EvolutionPlan",
    "EvolutionPlanError",
    "EvolutionProgressTracker",
    "GenerationStat",
    "build_evolution_request",
    "evolution_diff",
]
