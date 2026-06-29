"""Tests for ``care.evolution_session`` (TODO §7 P1).

Coverage:

1. **EvolutionConfig** — defaults, frozen, accepts mode literals.
2. **build_evolution_request** — every config field maps to the
   right server-body key; missing seed chain raises; invalid
   counts raise; optional fields omitted when unset.
3. **EvolutionProgressTracker** — `record_generation`
   merges by max best_fitness; `record_event` dispatches by
   event_type; unknown event types ignored; `fitness_curve()`
   sorted; `latest` / `best_overall` predicates;
   placeholder `-inf` from generation_started skipped in
   `best_overall`.
4. **evolution_diff** — produces unified diff between two
   chain dicts; empty/None inputs yield empty tuple; labels
   forwarded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from care.evolution_session import (
    EvolutionConfig,
    EvolutionPlan,
    EvolutionPlanError,
    EvolutionProgressTracker,
    GenerationStat,
    _opt_float,
    build_evolution_request,
    evolution_diff,
)


class TestNumericCoercion:
    def test_opt_float_passes_finite(self):
        assert _opt_float("0.5") == 0.5
        assert _opt_float(2) == 2.0

    def test_opt_float_none_for_absent(self):
        assert _opt_float(None) is None

    def test_opt_float_warns_on_malformed(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="care.evolution_session"):
            assert _opt_float("n/a") is None
        assert any("malformed" in r.message for r in caplog.records)

    def test_opt_float_rejects_nan_inf(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="care.evolution_session"):
            assert _opt_float(float("nan")) is None
            assert _opt_float(float("inf")) is None
        assert any("non-finite" in r.message for r in caplog.records)

    def test_record_event_drops_nan_fitness_but_keeps_record(self, caplog):
        import logging

        tracker = EvolutionProgressTracker()
        with caplog.at_level(logging.WARNING, logger="care.evolution_session"):
            tracker.record_event(
                {"event_type": "best_updated", "generation": 1, "fitness": "NaN"}
            )
        # Malformed fitness → floored to 0.0 (not a crash), warned.
        latest = tracker.latest
        assert latest is not None and latest.generation == 1
        assert any("fitness" in r.message for r in caplog.records)

    def test_record_event_keeps_zero_fitness(self):
        # 0.0 is a valid fitness and must not be skipped (regression: the
        # old `or` chain dropped a real 0.0 in favour of best_fitness).
        tracker = EvolutionProgressTracker()
        tracker.record_event(
            {"event_type": "best_updated", "generation": 0, "fitness": 0.0}
        )
        assert tracker.latest is not None
        assert tracker.latest.best_fitness == 0.0


# ---------------------------------------------------------------------------
# EvolutionConfig
# ---------------------------------------------------------------------------


class TestEvolutionConfig:
    def test_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.evolution_mode == "full_chain"
        assert cfg.max_iterations == 5
        assert cfg.population_size == 8
        assert cfg.validation_criteria == ""
        assert cfg.test_data_path is None
        assert cfg.validation_threshold is None
        assert cfg.validation_type == "Continuous (0..1)"
        assert cfg.continuous_metric == "ROUGE-L"
        assert cfg.binary_method == "equality"
        assert cfg.target_column == "expected"
        assert cfg.objectives == ()

    def test_frozen(self):
        cfg = EvolutionConfig()
        with pytest.raises(Exception):
            cfg.max_iterations = 10  # type: ignore[misc]

    def test_per_step_mode(self):
        cfg = EvolutionConfig(evolution_mode="per_step")
        assert cfg.evolution_mode == "per_step"


# ---------------------------------------------------------------------------
# build_evolution_request
# ---------------------------------------------------------------------------


def _plan(**overrides) -> EvolutionPlan:
    base = {
        "config": EvolutionConfig(),
        "base_chain_entity_id": "chain-1",
        "base_chain_content": {"steps": []},
        "base_chain_name": "Weather",
    }
    base.update(overrides)
    return EvolutionPlan(**base)


class TestBuildEvolutionRequest:
    def test_defaults_produce_minimal_body(self):
        body = build_evolution_request(_plan())
        assert body["seed_chain_id"] == "chain-1"
        assert body["evolution_mode"] == "full_chain"
        assert body["max_iterations"] == 5
        assert body["population_size"] == 8
        assert body["validation_type"] == "Continuous (0..1)"
        assert body["continuous_metric"] == "ROUGE-L"
        assert body["target_column"] == "expected"
        # Optional fields absent when unset.
        assert "validation_criteria" not in body
        assert "test_data_path" not in body
        assert "validation_threshold" not in body
        assert "objectives" not in body

    def test_full_config_serialised(self):
        cfg = EvolutionConfig(
            evolution_mode="per_step",
            max_iterations=10,
            population_size=16,
            validation_criteria="prefer concise",
            test_data_path=Path("/tmp/eval.jsonl"),
            validation_threshold=0.85,
            objectives=("accuracy", "latency"),
        )
        body = build_evolution_request(_plan(config=cfg))
        assert body["evolution_mode"] == "per_step"
        assert body["max_iterations"] == 10
        assert body["population_size"] == 16
        assert body["validation_criteria"] == "prefer concise"
        assert body["test_data_path"] == "/tmp/eval.jsonl"
        assert body["validation_threshold"] == 0.85
        assert body["continuous_metric"] == "ROUGE-L"
        assert body["objectives"] == ["accuracy", "latency"]

    def test_empty_validation_criteria_omitted(self):
        body = build_evolution_request(
            _plan(config=EvolutionConfig(validation_criteria="   "))
        )
        assert "validation_criteria" not in body

    def test_missing_seed_chain_raises(self):
        with pytest.raises(EvolutionPlanError, match="base_chain_entity_id"):
            build_evolution_request(_plan(base_chain_entity_id=""))

    def test_invalid_max_iterations_raises(self):
        with pytest.raises(EvolutionPlanError, match="max_iterations"):
            build_evolution_request(
                _plan(config=EvolutionConfig(max_iterations=0))
            )

    def test_population_size_too_small_raises(self):
        with pytest.raises(EvolutionPlanError, match="population_size"):
            build_evolution_request(
                _plan(config=EvolutionConfig(population_size=1))
            )


# ---------------------------------------------------------------------------
# EvolutionProgressTracker
# ---------------------------------------------------------------------------


class TestProgressTracker:
    def test_empty_tracker(self):
        t = EvolutionProgressTracker()
        assert t.is_empty
        assert t.fitness_curve() == ()
        assert t.latest is None
        assert t.best_overall is None

    def test_record_generation_inserts(self):
        t = EvolutionProgressTracker()
        t.record_generation(GenerationStat(generation=0, best_fitness=0.5))
        assert not t.is_empty
        assert t.fitness_curve() == (
            GenerationStat(generation=0, best_fitness=0.5),
        )

    def test_record_same_generation_keeps_max_best(self):
        t = EvolutionProgressTracker()
        t.record_generation(GenerationStat(generation=0, best_fitness=0.5))
        t.record_generation(GenerationStat(generation=0, best_fitness=0.8))
        # Followed by a worse stat — still 0.8.
        t.record_generation(GenerationStat(generation=0, best_fitness=0.3))
        curve = t.fitness_curve()
        assert curve[0].best_fitness == 0.8

    def test_record_event_individual_evaluated(self):
        t = EvolutionProgressTracker()
        t.record_event(
            {
                "event_type": "individual_evaluated",
                "generation": 1,
                "fitness": 0.75,
                "individual_id": "ind-7",
            }
        )
        assert t.latest.best_fitness == 0.75
        assert t.latest.best_individual_id == "ind-7"

    def test_record_event_best_updated_overrides(self):
        t = EvolutionProgressTracker()
        t.record_event(
            {
                "event_type": "individual_evaluated",
                "generation": 1,
                "fitness": 0.5,
                "individual_id": "ind-1",
            }
        )
        t.record_event(
            {
                "event_type": "best_updated",
                "generation": 1,
                "best_fitness": 0.9,
                "best_individual_id": "ind-99",
            }
        )
        latest = t.latest
        assert latest.best_fitness == 0.9
        assert latest.best_individual_id == "ind-99"

    def test_generation_started_creates_placeholder(self):
        t = EvolutionProgressTracker()
        t.record_event({"event_type": "generation_started", "generation": 2})
        assert not t.is_empty
        # Placeholder uses -inf so it isn't the best_overall.
        assert t.best_overall is None

    def test_best_overall_skips_inf_placeholder(self):
        t = EvolutionProgressTracker()
        t.record_event({"event_type": "generation_started", "generation": 0})
        t.record_event(
            {
                "event_type": "best_updated",
                "generation": 1,
                "best_fitness": 0.6,
                "best_individual_id": "ind-x",
            }
        )
        assert t.best_overall.generation == 1
        assert t.best_overall.best_fitness == 0.6

    def test_fitness_curve_sorted_by_generation(self):
        t = EvolutionProgressTracker()
        t.record_event({"event_type": "best_updated", "generation": 2, "best_fitness": 0.8})
        t.record_event({"event_type": "best_updated", "generation": 0, "best_fitness": 0.3})
        t.record_event({"event_type": "best_updated", "generation": 1, "best_fitness": 0.5})
        curve = t.fitness_curve()
        assert [g.generation for g in curve] == [0, 1, 2]

    def test_individuals_evaluated_count(self):
        t = EvolutionProgressTracker()
        for i in range(5):
            t.record_event(
                {
                    "event_type": "individual_evaluated",
                    "generation": 3,
                    "fitness": 0.1 + i * 0.1,
                }
            )
        # Each individual_evaluated bumps the count.
        assert t.fitness_curve()[0].individuals_evaluated == 1
        # Best fitness is the max across all five.
        assert t.fitness_curve()[0].best_fitness == pytest.approx(0.5)

    def test_unknown_event_type_ignored(self):
        t = EvolutionProgressTracker()
        t.record_event({"event_type": "future_event", "generation": 0})
        assert t.is_empty

    def test_event_missing_generation_ignored(self):
        t = EvolutionProgressTracker()
        t.record_event({"event_type": "best_updated", "best_fitness": 0.5})
        # Default generation=0 lands.
        assert not t.is_empty

    def test_unparseable_fitness_falls_back_to_zero(self):
        t = EvolutionProgressTracker()
        t.record_event(
            {
                "event_type": "best_updated",
                "generation": 0,
                "best_fitness": "garbage",
            }
        )
        assert t.fitness_curve()[0].best_fitness == 0.0

    def test_latest_is_highest_generation(self):
        t = EvolutionProgressTracker()
        t.record_event({"event_type": "best_updated", "generation": 5, "best_fitness": 0.5})
        t.record_event({"event_type": "best_updated", "generation": 3, "best_fitness": 0.9})
        # `latest` is by generation number, not by arrival order.
        assert t.latest.generation == 5


# ---------------------------------------------------------------------------
# evolution_diff
# ---------------------------------------------------------------------------


class TestEvolutionDiff:
    def test_identical_chains_no_diff(self):
        chain = {"task": "weather", "steps": [{"id": 1}]}
        assert evolution_diff(chain, chain) == ()

    def test_different_chains_produce_diff(self):
        base = {"task": "weather", "steps": [{"prompt": "old"}]}
        evolved = {"task": "weather", "steps": [{"prompt": "new"}]}
        diff = evolution_diff(base, evolved)
        joined = "\n".join(diff)
        assert "old" in joined
        assert "new" in joined

    def test_labels_forwarded(self):
        diff = evolution_diff(
            {"a": 1},
            {"a": 2},
            base_label="v1",
            evolved_label="v2",
        )
        joined = "\n".join(diff)
        assert "v1" in joined
        assert "v2" in joined

    def test_none_inputs_yield_empty_tuple(self):
        assert evolution_diff(None, {"x": 1}) == ()
        assert evolution_diff({"x": 1}, None) == ()
        assert evolution_diff(None, None) == ()

    def test_empty_chains_yield_empty_tuple(self):
        # Empty dicts treated as "nothing to diff".
        assert evolution_diff({}, {"x": 1}) == ()
        assert evolution_diff({"x": 1}, {}) == ()

    def test_key_order_independent(self):
        # Same content, different key order → empty diff.
        a = {"a": 1, "b": 2, "c": 3}
        b = {"c": 3, "a": 1, "b": 2}
        assert evolution_diff(a, b) == ()
