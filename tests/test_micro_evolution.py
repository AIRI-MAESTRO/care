"""Tests for ``care.micro_evolution`` (TODO §7 P2).

Coverage layers:

1. **Config validation** — invalid population_size / generations
   / elitism / tournament_size all raise
   :class:`MicroEvolutionError` with a clear message.
2. **Deterministic with seed** — same seed + same evaluator
   → identical result across two independent runs.
3. **Selection + elitism** — best individual is preserved
   across generations; total evaluations match the documented
   formula.
4. **Mutator drives diversity** — without a useful mutator
   evolution stalls; with one it converges toward the
   fitness target.
5. **Multi-objective evaluator** — dict-returning evaluator
   has its `breakdown` preserved on each `Individual`;
   `score_reducer` defaults to sum, overridable via config.
6. **Built-in mutators** — `perturb_step_titles`,
   `drop_optional_step`, `noop_mutator`, `compose_mutators`,
   `builtin_mutators` all work against realistic chain dicts.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from care.micro_evolution import (
    Individual,
    MicroEvolution,
    MicroEvolutionConfig,
    MicroEvolutionError,
    MicroEvolutionResult,
    builtin_mutators,
    compose_mutators,
    drop_optional_step,
    noop_mutator,
    perturb_step_titles,
)


def _seed_chain() -> dict[str, Any]:
    return {
        "task_description": "demo",
        "steps": [
            {
                "number": 1,
                "title": "extract",
                "step_type": "llm",
                "aim": "extract",
            },
            {
                "number": 2,
                "title": "summarise",
                "step_type": "llm",
                "aim": "summarise",
                "dependencies": [1],
            },
            {
                "number": 3,
                "title": "format",
                "step_type": "llm",
                "aim": "format",
                "dependencies": [2],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_seed_chain_must_be_dict(self):
        with pytest.raises(MicroEvolutionError, match="must be a dict"):
            MicroEvolution(
                "not a chain",  # type: ignore[arg-type]
                lambda c: 1.0,
                mutator=noop_mutator,
            )

    def test_population_size_too_small(self):
        with pytest.raises(MicroEvolutionError, match="population_size"):
            MicroEvolution(
                _seed_chain(),
                lambda c: 0.0,
                mutator=noop_mutator,
                config=MicroEvolutionConfig(population_size=0),
            )

    def test_generations_too_small(self):
        with pytest.raises(MicroEvolutionError, match="generations"):
            MicroEvolution(
                _seed_chain(),
                lambda c: 0.0,
                mutator=noop_mutator,
                config=MicroEvolutionConfig(generations=0),
            )

    def test_elitism_too_large(self):
        with pytest.raises(MicroEvolutionError, match="elitism"):
            MicroEvolution(
                _seed_chain(),
                lambda c: 0.0,
                mutator=noop_mutator,
                config=MicroEvolutionConfig(
                    population_size=5, elitism=99
                ),
            )

    def test_negative_elitism(self):
        with pytest.raises(MicroEvolutionError, match="elitism"):
            MicroEvolution(
                _seed_chain(),
                lambda c: 0.0,
                mutator=noop_mutator,
                config=MicroEvolutionConfig(elitism=-1),
            )

    def test_tournament_size_zero(self):
        with pytest.raises(MicroEvolutionError, match="tournament_size"):
            MicroEvolution(
                _seed_chain(),
                lambda c: 0.0,
                mutator=noop_mutator,
                config=MicroEvolutionConfig(tournament_size=0),
            )


# ---------------------------------------------------------------------------
# Deterministic
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_produces_same_result(self):
        def evaluator(chain: dict[str, Any]) -> float:
            # Fitness = total title length, so mutating titles
            # changes score.
            return sum(
                len(s.get("title", "")) for s in chain.get("steps", [])
            )

        cfg = MicroEvolutionConfig(
            population_size=4, generations=3, seed=42
        )
        run_a = MicroEvolution(
            _seed_chain(),
            evaluator,
            mutator=perturb_step_titles,
            config=cfg,
        ).run()
        run_b = MicroEvolution(
            _seed_chain(),
            evaluator,
            mutator=perturb_step_titles,
            config=cfg,
        ).run()
        assert run_a.best.score == run_b.best.score
        # And the history is byte-identical, modulo dict ordering.
        assert [h.score for h in run_a.history] == [
            h.score for h in run_b.history
        ]


# ---------------------------------------------------------------------------
# Selection + elitism
# ---------------------------------------------------------------------------


class TestSelectionAndElitism:
    def test_best_score_is_monotonic_non_decreasing(self):
        def evaluator(chain: dict[str, Any]) -> float:
            return sum(
                len(s.get("title", "")) for s in chain.get("steps", [])
            )

        cfg = MicroEvolutionConfig(
            population_size=6, generations=5, elitism=1, seed=1
        )
        result = MicroEvolution(
            _seed_chain(),
            evaluator,
            mutator=perturb_step_titles,
            config=cfg,
        ).run()
        scores = [h.score for h in result.history]
        # Elitism=1 → best score never drops across generations.
        for prev, curr in zip(scores, scores[1:]):
            assert curr >= prev, f"score dropped: {scores}"

    def test_evaluations_count_matches_formula(self):
        """Doc says:
        ``population_size + (generations - 1) * (population_size - elitism)``"""
        cfg = MicroEvolutionConfig(
            population_size=6,
            generations=4,
            elitism=2,
            seed=7,
        )

        calls: list[int] = [0]

        def evaluator(chain: dict[str, Any]) -> float:
            calls[0] += 1
            return float(calls[0])

        result = MicroEvolution(
            _seed_chain(),
            evaluator,
            mutator=perturb_step_titles,
            config=cfg,
        ).run()
        expected = 6 + (4 - 1) * (6 - 2)
        assert result.evaluations == expected
        assert calls[0] == expected

    def test_history_length_equals_generations(self):
        cfg = MicroEvolutionConfig(
            population_size=3, generations=7, seed=2
        )
        result = MicroEvolution(
            _seed_chain(),
            lambda c: 1.0,
            mutator=noop_mutator,
            config=cfg,
        ).run()
        assert len(result.history) == 7

    def test_population_sorted_descending(self):
        cfg = MicroEvolutionConfig(
            population_size=5, generations=3, seed=3
        )
        result = MicroEvolution(
            _seed_chain(),
            lambda c: sum(len(s.get("title", "")) for s in c.get("steps", [])),
            mutator=perturb_step_titles,
            config=cfg,
        ).run()
        scores = [ind.score for ind in result.population]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Mutator drives diversity
# ---------------------------------------------------------------------------


class TestMutatorDrivesEvolution:
    def test_noop_mutator_keeps_score_constant(self):
        # When mutator is no-op + evaluator is deterministic, the
        # population is N copies of the seed with identical score.
        cfg = MicroEvolutionConfig(
            population_size=4, generations=3, seed=10
        )
        result = MicroEvolution(
            _seed_chain(),
            lambda c: 42.0,
            mutator=noop_mutator,
            config=cfg,
        ).run()
        for ind in result.population:
            assert ind.score == 42.0
        assert result.best.score == 42.0

    def test_perturb_titles_does_change_score(self):
        # With a fitness function that rewards mutation, the best
        # of generation N+ should beat the seed.
        cfg = MicroEvolutionConfig(
            population_size=10, generations=4, seed=5
        )
        seed_score = sum(
            len(s["title"]) for s in _seed_chain()["steps"]
        )
        result = MicroEvolution(
            _seed_chain(),
            lambda c: sum(len(s.get("title", "")) for s in c["steps"]),
            mutator=perturb_step_titles,
            config=cfg,
        ).run()
        assert result.best.score > seed_score


# ---------------------------------------------------------------------------
# Multi-objective evaluator
# ---------------------------------------------------------------------------


class TestMultiObjective:
    def test_dict_evaluator_preserves_breakdown(self):
        def evaluator(chain: dict[str, Any]) -> dict[str, float]:
            return {"accuracy": 0.8, "latency": -0.5, "cost": -0.1}

        cfg = MicroEvolutionConfig(
            population_size=3, generations=2, seed=11
        )
        result = MicroEvolution(
            _seed_chain(),
            evaluator,
            mutator=noop_mutator,
            config=cfg,
        ).run()
        # Default reducer = sum.
        for ind in result.population:
            assert ind.breakdown == {
                "accuracy": 0.8,
                "latency": -0.5,
                "cost": -0.1,
            }
            assert ind.score == pytest.approx(0.2)

    def test_custom_score_reducer(self):
        # Weighted reducer: 2 * accuracy + cost.
        def reducer(d: dict[str, float]) -> float:
            return 2.0 * d.get("accuracy", 0.0) + d.get("cost", 0.0)

        cfg = MicroEvolutionConfig(
            population_size=3,
            generations=2,
            score_reducer=reducer,
            seed=12,
        )
        result = MicroEvolution(
            _seed_chain(),
            lambda c: {"accuracy": 0.5, "cost": -0.1},
            mutator=noop_mutator,
            config=cfg,
        ).run()
        for ind in result.population:
            assert ind.score == pytest.approx(2.0 * 0.5 + -0.1)

    def test_scalar_evaluator_keeps_breakdown_empty(self):
        cfg = MicroEvolutionConfig(
            population_size=2, generations=1, seed=13
        )
        result = MicroEvolution(
            _seed_chain(),
            lambda c: 3.14,
            mutator=noop_mutator,
            config=cfg,
        ).run()
        for ind in result.population:
            assert ind.breakdown == {}
            assert ind.score == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# Built-in mutators
# ---------------------------------------------------------------------------


class TestBuiltinMutators:
    def test_noop_returns_unchanged(self):
        chain = _seed_chain()
        result = noop_mutator(chain, random.Random(0))
        assert result is chain

    def test_perturb_step_titles_changes_one_title(self):
        chain = _seed_chain()
        original_titles = [s["title"] for s in chain["steps"]]
        perturb_step_titles(chain, random.Random(0))
        new_titles = [s["title"] for s in chain["steps"]]
        # Exactly one title changed.
        diffs = sum(a != b for a, b in zip(original_titles, new_titles))
        assert diffs == 1

    def test_perturb_skips_empty_steps_list(self):
        chain = {"task_description": "x", "steps": []}
        out = perturb_step_titles(chain, random.Random(0))
        # No raise; returns chain unchanged.
        assert out["steps"] == []

    def test_drop_optional_step_removes_leaf(self):
        chain = _seed_chain()
        # Step 3 is a leaf (nothing depends on it) — droppable.
        drop_optional_step(chain, random.Random(0))
        numbers = [s["number"] for s in chain["steps"]]
        # Step 3 is gone; steps 1 + 2 remain.
        assert 3 not in numbers

    def test_drop_optional_step_preserves_required(self):
        # Step 2 has step 3 depending on it; step 1 has step 2.
        # Only step 3 can be dropped.
        for _ in range(10):
            chain = _seed_chain()
            drop_optional_step(chain, random.Random(0))
            numbers = [s["number"] for s in chain["steps"]]
            # Step 1 + 2 must always survive.
            assert 1 in numbers
            assert 2 in numbers

    def test_drop_optional_step_no_candidates(self):
        # Single-step chain — drop_optional must leave it alone
        # (no non-first step to remove).
        chain = {"task_description": "x", "steps": [
            {"number": 1, "title": "only", "step_type": "llm"}
        ]}
        out = drop_optional_step(chain, random.Random(0))
        assert len(out["steps"]) == 1

    def test_compose_mutators_picks_one(self):
        called = []

        def a(c, rng):
            called.append("a")
            return c

        def b(c, rng):
            called.append("b")
            return c

        composed = compose_mutators(a, b)
        composed(_seed_chain(), random.Random(0))
        assert len(called) == 1
        assert called[0] in ("a", "b")

    def test_compose_mutators_empty_returns_noop(self):
        composed = compose_mutators()
        out = composed(_seed_chain(), random.Random(0))
        # Returns the chain unchanged.
        assert out["task_description"] == "demo"

    def test_builtin_mutators_callable(self):
        m = builtin_mutators()
        # Smoke: it runs without raising on a real seed chain.
        result = m(_seed_chain(), random.Random(0))
        assert isinstance(result, dict)
        assert "steps" in result


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_result_is_frozen_dataclass(self):
        cfg = MicroEvolutionConfig(
            population_size=2, generations=1, seed=4
        )
        result = MicroEvolution(
            _seed_chain(),
            lambda c: 1.0,
            mutator=noop_mutator,
            config=cfg,
        ).run()
        assert isinstance(result, MicroEvolutionResult)
        with pytest.raises(Exception):
            result.best = result.best  # type: ignore[misc]

    def test_best_is_population_zero(self):
        cfg = MicroEvolutionConfig(
            population_size=4, generations=3, seed=6
        )
        result = MicroEvolution(
            _seed_chain(),
            lambda c: sum(len(s.get("title", "")) for s in c["steps"]),
            mutator=perturb_step_titles,
            config=cfg,
        ).run()
        # Population is sorted descending; `best` equals `population[0]`.
        assert result.best.score == result.population[0].score

    def test_individual_records_generation(self):
        cfg = MicroEvolutionConfig(
            population_size=2, generations=3, elitism=0, seed=8
        )
        result = MicroEvolution(
            _seed_chain(),
            lambda c: random.random(),
            mutator=perturb_step_titles,
            config=cfg,
        ).run()
        # No elitism — final population members are all from
        # generation 2 (the last generation index).
        for ind in result.population:
            assert ind.generation == 2

    def test_individual_is_frozen(self):
        ind = Individual(chain={}, score=1.0)
        with pytest.raises(Exception):
            ind.score = 2.0  # type: ignore[misc]
