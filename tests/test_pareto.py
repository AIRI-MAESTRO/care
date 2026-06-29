"""Tests for ``care.compute_pareto_front`` (TODO §7 P2 multi-objective).

Verifies the non-domination algorithm + every documented edge:

1. **Basic non-domination** — dominated points are filtered out;
   non-dominated are preserved in input order.
2. **Mixed directions** — accuracy maximised + latency minimised
   ⇒ Pareto front contains the {high-accuracy, low-latency}
   tradeoff set.
3. **Equality** — two identical breakdowns both survive (neither
   strictly dominates the other).
4. **Missing objective** — an individual missing one of the
   objectives is treated as the worst possible value (matches
   Platform §4.2's "missing scores as worst" convention).
5. **Scalar-only fallback** — individuals without `breakdown`
   collapse to a single-objective max-by-score comparison.
6. **`MicroEvolutionResult.pareto_front` helper** — convenience
   wrapper round-trips against the standalone function.
"""

from __future__ import annotations

from care.micro_evolution import (
    Individual,
    MicroEvolution,
    MicroEvolutionConfig,
    MicroEvolutionResult,
    compute_pareto_front,
    noop_mutator,
)


def _ind(name: str, breakdown: dict[str, float]) -> Individual:
    return Individual(
        chain={"task_description": name},
        score=sum(breakdown.values()),
        breakdown=breakdown,
    )


def _names(individuals) -> list[str]:
    """Pull `task_description` out — Individual itself isn't
    hashable (the wrapped dict makes it unhashable), so tests
    compare on identifying names instead of building sets."""
    return sorted(ind.chain.get("task_description", "?") for ind in individuals)


# ---------------------------------------------------------------------------
# Empty / single-element inputs
# ---------------------------------------------------------------------------


class TestEdgeInputs:
    def test_empty_input_returns_empty_tuple(self):
        assert compute_pareto_front([]) == ()

    def test_single_individual_returned_as_front(self):
        only = _ind("solo", {"a": 1.0, "b": 2.0})
        front = compute_pareto_front([only])
        assert front == (only,)


# ---------------------------------------------------------------------------
# Maximise + maximise
# ---------------------------------------------------------------------------


class TestMaximiseMaximise:
    def test_dominated_point_removed(self):
        # All maximised. (2, 2) dominates (1, 1).
        a = _ind("a", {"x": 2.0, "y": 2.0})
        b = _ind("b", {"x": 1.0, "y": 1.0})
        front = compute_pareto_front([a, b])
        assert front == (a,)

    def test_tradeoff_set_preserved(self):
        # All maximised. (3, 1) and (1, 3) trade off — both survive.
        a = _ind("hi-x", {"x": 3.0, "y": 1.0})
        b = _ind("hi-y", {"x": 1.0, "y": 3.0})
        c = _ind("dominated", {"x": 1.0, "y": 1.0})
        front = compute_pareto_front([a, b, c])
        assert _names(front) == _names([a, b])
        # Insertion order is preserved (a came before b in input).
        assert [ind.chain["task_description"] for ind in front] == [
            "hi-x",
            "hi-y",
        ]

    def test_three_way_tradeoff(self):
        # Classic 3-point tradeoff.
        a = _ind("a", {"acc": 0.9, "speed": 0.1, "cheap": 0.5})
        b = _ind("b", {"acc": 0.5, "speed": 0.9, "cheap": 0.5})
        c = _ind("c", {"acc": 0.5, "speed": 0.5, "cheap": 0.9})
        front = compute_pareto_front([a, b, c])
        assert _names(front) == _names([a, b, c])


# ---------------------------------------------------------------------------
# Mixed directions
# ---------------------------------------------------------------------------


class TestMixedDirections:
    def test_accuracy_max_latency_min(self):
        # Three points: high acc + high latency, mid acc + low
        # latency, low acc + low latency. The third is dominated
        # by the second (same latency, higher acc).
        a = _ind("hi-acc-slow", {"accuracy": 0.95, "latency_ms": 500.0})
        b = _ind("mid-acc-fast", {"accuracy": 0.80, "latency_ms": 100.0})
        c = _ind("low-acc-fast", {"accuracy": 0.50, "latency_ms": 100.0})
        front = compute_pareto_front(
            [a, b, c],
            directions={"accuracy": "maximize", "latency_ms": "minimize"},
        )
        # `c` is dominated by `b` (same latency, higher acc).
        assert _names(front) == _names([a, b])

    def test_all_minimise(self):
        # Both axes minimised (e.g. cost + latency). (1, 1) wins
        # over (2, 2).
        a = _ind("cheap-fast", {"cost": 1.0, "latency": 1.0})
        b = _ind("dear-slow", {"cost": 2.0, "latency": 2.0})
        front = compute_pareto_front(
            [a, b],
            directions={"cost": "minimize", "latency": "minimize"},
        )
        assert front == (a,)

    def test_missing_direction_defaults_to_maximize(self):
        # Specify only one direction; the other defaults to max.
        # (2, 1) vs (1, 2) — both survive under maximise/maximise.
        a = _ind("a", {"acc": 2.0, "cov": 1.0})
        b = _ind("b", {"acc": 1.0, "cov": 2.0})
        front = compute_pareto_front(
            [a, b],
            directions={"acc": "maximize"},  # `cov` defaults to maximise
        )
        assert _names(front) == _names([a, b])


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


class TestEquality:
    def test_identical_points_both_survive(self):
        # Neither dominates the other when every objective is
        # equal — domination requires "strictly better somewhere".
        a = _ind("twin-a", {"x": 1.0, "y": 1.0})
        b = _ind("twin-b", {"x": 1.0, "y": 1.0})
        front = compute_pareto_front([a, b])
        assert _names(front) == _names([a, b])


# ---------------------------------------------------------------------------
# Missing objectives (treated as worst possible)
# ---------------------------------------------------------------------------


class TestMissingObjective:
    def test_missing_max_objective_treated_as_minus_inf(self):
        full = _ind("full", {"acc": 0.5, "speed": 0.5})
        partial = _ind("partial", {"acc": 0.5})  # missing `speed`
        front = compute_pareto_front(
            [full, partial],
            directions={"acc": "maximize", "speed": "maximize"},
        )
        # `partial` has speed=-inf, dominated by `full`.
        assert front == (full,)

    def test_missing_min_objective_treated_as_plus_inf(self):
        full = _ind("full", {"acc": 0.5, "cost": 1.0})
        partial = _ind("partial", {"acc": 0.5})  # missing `cost`
        front = compute_pareto_front(
            [full, partial],
            directions={"acc": "maximize", "cost": "minimize"},
        )
        # `partial`'s cost=+inf (worst when minimising) → dominated.
        assert front == (full,)

    def test_partial_breakdown_can_still_be_non_dominated(self):
        # Even with a missing key, the individual survives when
        # it strictly beats everyone else on the keys it does have.
        a = _ind("partial-but-good", {"acc": 0.99})
        b = _ind("full", {"acc": 0.5, "speed": 0.5})
        # Maximise both. `b` beats `a` on `speed` (a has -inf),
        # but `a` beats `b` on `acc`. Neither dominates.
        front = compute_pareto_front(
            [a, b],
            directions={"acc": "maximize", "speed": "maximize"},
        )
        assert _names(front) == _names([a, b])


# ---------------------------------------------------------------------------
# Scalar fallback (no breakdown anywhere)
# ---------------------------------------------------------------------------


class TestScalarFallback:
    def test_all_scalar_falls_back_to_max_score(self):
        # No breakdown anywhere → single-objective comparison on
        # `.score`. The top score wins; ties all included.
        a = Individual(chain={}, score=1.0)
        b = Individual(chain={}, score=3.0)
        c = Individual(chain={}, score=3.0)
        d = Individual(chain={}, score=2.0)
        front = compute_pareto_front([a, b, c, d])
        # Both ties for the max score survive.
        assert sorted(ind.score for ind in front) == [3.0, 3.0]
        assert len(front) == 2

    def test_mixed_scalar_and_breakdown_uses_breakdown_path(self):
        # One individual has a breakdown → multi-objective mode
        # kicks in for ALL individuals. The scalar-only one
        # contributes an empty breakdown, which makes it
        # dominated as soon as any objective key exists.
        with_breakdown = _ind("has-it", {"acc": 0.5, "cost": 0.5})
        scalar = Individual(chain={}, score=999.0, breakdown={})
        front = compute_pareto_front(
            [with_breakdown, scalar],
            directions={"acc": "maximize", "cost": "maximize"},
        )
        # `scalar` has empty breakdown → both objectives -inf →
        # dominated by `with_breakdown`.
        assert front == (with_breakdown,)


# ---------------------------------------------------------------------------
# MicroEvolutionResult.pareto_front helper
# ---------------------------------------------------------------------------


class TestResultHelper:
    def test_helper_matches_standalone(self):
        # Run a tiny GA with a multi-objective evaluator; assert
        # the convenience method delegates correctly.
        def evaluator(chain):
            # Two trade-off objectives based on title hashes —
            # not meaningful but enough to produce diverse
            # breakdowns.
            title = chain["steps"][0].get("title", "")
            return {
                "accuracy": float(len(title) % 7),
                "latency_ms": float(hash(title) % 5),
            }

        cfg = MicroEvolutionConfig(
            population_size=8, generations=3, seed=42
        )
        result: MicroEvolutionResult = MicroEvolution(
            {
                "task_description": "x",
                "steps": [
                    {"number": 1, "title": "extract", "step_type": "llm"}
                ],
            },
            evaluator,
            mutator=noop_mutator,
            config=cfg,
        ).run()
        directions = {"accuracy": "maximize", "latency_ms": "minimize"}
        from_helper = result.pareto_front(directions)
        from_function = compute_pareto_front(
            result.population, directions=directions
        )
        assert from_helper == from_function

    def test_helper_default_directions(self):
        cfg = MicroEvolutionConfig(
            population_size=3, generations=1, seed=4
        )
        result = MicroEvolution(
            {"task_description": "x", "steps": []},
            lambda c: {"a": 1.0, "b": 1.0},
            mutator=noop_mutator,
            config=cfg,
        ).run()
        # All individuals have identical breakdown — neither
        # dominates → all survive.
        front = result.pareto_front()
        assert len(front) == cfg.population_size
