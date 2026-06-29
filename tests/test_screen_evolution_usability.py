"""Unit tests for the evolution-screen usability helpers (P0).

These cover the pure, mount-free helpers that fix the "empty chart / no
detail" complaint: context-aware pane placeholders, the single-objective
Pareto explanation, and the status-pane liveness line.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from care.evolution_session import GenerationStat
from care.screens.evolution import (
    EvolutionIndividual,
    EvolutionScreen,
    _extract_platform_version,
    _format_objectives_inline,
    _frontier_entry_to_individual,
)


def _screen() -> EvolutionScreen:
    return EvolutionScreen(base_chain_id="chain-1")


class TestEmptyPaneText:
    def test_pre_run_status_says_waiting_for_runner(self) -> None:
        s = _screen()
        s.run.status = "queued"
        assert "waiting for the runner" in s._empty_pane_text("fitness data")
        assert "fitness data" in s._empty_pane_text("fitness data")

    def test_running_status_says_waiting_for_generation(self) -> None:
        s = _screen()
        s.run.status = "running"
        msg = s._empty_pane_text("program data")
        assert "first generation" in msg
        assert "program data" in msg

    def test_finished_run_says_no_data_reported(self) -> None:
        s = _screen()
        s.run.status = "completed"
        s.run.finished = True
        assert "no fitness data was reported" in s._empty_pane_text("fitness data")


class TestParetoPlaceholder:
    def test_no_individuals_uses_waiting_placeholder(self) -> None:
        s = _screen()
        s.run.status = "running"
        assert "first generation" in s._pareto_placeholder()

    def test_single_objective_explains_and_points_to_fitness(self) -> None:
        s = _screen()
        s.run.status = "running"
        s.run.individuals = [
            EvolutionIndividual(
                individual_id="i1",
                generation=1,
                fitness=0.5,
                objectives=(("fitness", 0.5),),  # single objective
            )
        ]
        s.tracker.record_generation(GenerationStat(generation=1, best_fitness=0.5))
        msg = s._pareto_placeholder()
        assert "Single-objective run" in msg
        assert "Fitness" in msg
        assert "0.5000" in msg  # best fitness surfaced

    def test_multi_objective_with_too_few_points_uses_waiting(self) -> None:
        s = _screen()
        s.run.status = "running"
        s.run.individuals = [
            EvolutionIndividual(
                individual_id="i1",
                generation=1,
                fitness=0.5,
                objectives=(("acc", 0.5), ("speed", 0.9)),
            )
        ]
        # multi-objective but <2 points → not the single-objective text
        assert "Single-objective" not in s._pareto_placeholder()


class TestHydrateFromSnapshot:
    def test_platform_metrics_fill_curve_programs_frontier(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        snap = {
            "best_fitness": 0.6,
            "generation": 3,
            "_raw": {
                "results": {
                    "metrics": {
                        "fitness_history": [
                            {"generation": 0, "best_fitness": 0.2},
                            {"generation": 1, "best_fitness": 0.5},
                        ],
                        "programs_valid": 7,
                        "programs_invalid": 2,
                        "frontier_programs": [
                            {"generation": 1, "chain_content": {}}
                        ],
                    }
                }
            },
        }
        s._hydrate_from_snapshot(snap)
        assert len(s.tracker.fitness_curve()) >= 2
        assert s.run.programs_valid == 7
        assert s.run.programs_invalid == 2
        assert 1 in s._frontier_by_gen
        assert s.run.data_source == "platform"

    def test_redis_fallback_when_platform_metrics_empty(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_y")
        with patch(
            "care.runtime.evolution_redis_probe.probe_fitness_history",
            return_value=[{"generation": 0, "best_fitness": 0.3}],
        ), patch(
            "care.runtime.evolution_redis_probe.probe_programs_counts",
            return_value=(4, 1),
        ):
            s._hydrate_from_snapshot({"best_fitness": None, "generation": None})
        assert s.run.programs_valid == 4
        assert s.run.programs_invalid == 1
        assert any(r.best_fitness == 0.3 for r in s.tracker.fitness_curve())
        assert s.run.data_source == "redis_probe"

    def test_probed_program_counts_tagged_redis_even_with_other_metrics(
        self,
    ) -> None:
        """When the snapshot carries fitness metrics but NOT program
        counts, probed counts must be tagged redis_probe, not platform."""
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_w")
        snap = {
            "_raw": {
                "results": {
                    "metrics": {
                        # fitness present, program counts absent
                        "fitness_history": [
                            {"generation": 0, "best_fitness": 0.4}
                        ]
                    }
                }
            }
        }
        with patch(
            "care.runtime.evolution_redis_probe.probe_programs_counts",
            return_value=(5, 2),
        ):
            s._hydrate_from_snapshot(snap)
        assert s.run.programs_valid == 5
        # last event was the probed programs_snapshot → source redis_probe
        assert s.run.data_source == "redis_probe"


class TestFrontierEntryToIndividual:
    def test_maps_real_runner_record(self) -> None:
        payload = _frontier_entry_to_individual(
            {
                "generation": 2,
                "program_id": "prog-7",
                "fitness": 0.42,
                "chain_config": {"steps": []},
                "mutation_summary": "shortened the prompt",
            }
        )
        assert payload == {
            "individual_id": "prog-7",
            "generation": 2,
            "fitness": 0.42,
            "chain_content": {"steps": []},
            "summary": "shortened the prompt",
        }

    def test_accepts_chain_content_alias(self) -> None:
        payload = _frontier_entry_to_individual(
            {"id": "p1", "chain_content": {"a": 1}}
        )
        assert payload["individual_id"] == "p1"
        assert payload["chain_content"] == {"a": 1}

    def test_none_without_usable_id(self) -> None:
        assert _frontier_entry_to_individual({"generation": 1}) is None
        assert _frontier_entry_to_individual("nope") is None

    def test_omits_absent_optional_fields(self) -> None:
        payload = _frontier_entry_to_individual({"program_id": "p"})
        assert payload == {"individual_id": "p"}


class TestFrontierPopulatesParetoTable:
    def test_frontier_snapshot_upserts_individuals(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        s._handle_event(
            {
                "event": "frontier_programs_snapshot",
                "data": {
                    "frontier": [
                        {
                            "generation": 0,
                            "program_id": "p0",
                            "fitness": 0.2,
                            "chain_config": {"v": 0},
                        },
                        {
                            "generation": 1,
                            "program_id": "p1",
                            "fitness": 0.6,
                            "chain_config": {"v": 1},
                        },
                    ]
                },
            }
        )
        ids = {i.individual_id for i in s.run.individuals}
        assert ids == {"p0", "p1"}
        # Sorted by fitness desc → best first; chain content carried through.
        assert s.run.individuals[0].individual_id == "p1"
        assert s.run.individuals[0].fitness == 0.6
        assert s.run.individuals[0].chain_dict == {"v": 1}
        # Versions-tab cache still populated as before.
        assert 0 in s._frontier_by_gen and 1 in s._frontier_by_gen

    def test_frontier_without_ids_leaves_table_empty(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        s._handle_event(
            {
                "event": "frontier_programs_snapshot",
                "data": {"frontier": [{"generation": 1, "chain_content": {}}]},
            }
        )
        assert s.run.individuals == []
        assert 1 in s._frontier_by_gen

    def test_repeated_snapshot_updates_not_duplicates(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        first = {
            "event": "frontier_programs_snapshot",
            "data": {"frontier": [{"generation": 0, "program_id": "p0", "fitness": 0.2}]},
        }
        s._handle_event(first)
        # Same id re-emitted next poll with an improved fitness.
        s._handle_event(
            {
                "event": "frontier_programs_snapshot",
                "data": {"frontier": [{"generation": 0, "program_id": "p0", "fitness": 0.5}]},
            }
        )
        assert len(s.run.individuals) == 1
        assert s.run.individuals[0].fitness == 0.5


class TestHeartbeatLiveness:
    def test_heartbeat_refreshes_liveness_clock(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        assert s.run.last_event_monotonic is None
        s._handle_event({"event": "heartbeat", "data": {"experiment_id": "exp_x"}})
        assert s.run.last_event_monotonic is not None

    def test_heartbeat_is_not_logged_as_event_or_tracked(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        s._handle_event({"event": "heartbeat", "data": {}})
        # No event row, no tracker rows, no status churn.
        assert s.run.events == []
        assert s.tracker.is_empty
        assert s.run.status == "submitting"


class TestSelectedIndividualChain:
    def test_returns_chain_for_frontier_individual(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        s._handle_event(
            {
                "event": "frontier_programs_snapshot",
                "data": {
                    "frontier": [
                        {"generation": 1, "program_id": "p1", "fitness": 0.6, "chain_config": {"steps": [1]}},
                    ]
                },
            }
        )
        assert s._selected_individual_chain("p1") == {"steps": [1]}

    def test_none_when_individual_has_no_chain(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        # best_updated individual carries an id but no chain content.
        s._handle_event(
            {"event": "best_updated", "data": {"individual_id": "b1", "fitness": 0.5}}
        )
        assert s._selected_individual_chain("b1") is None

    def test_none_for_unknown_id(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        assert s._selected_individual_chain("missing") is None


class TestVersionsChainFallback:
    def test_redis_probe_used_when_frontier_chain_config_null(self) -> None:
        s = EvolutionScreen(
            base_chain_id="c",
            observe_evolution_id="exp_abc",
            max_iterations=10,
        )
        s.run.evolution_id = "exp_abc"
        s._frontier_by_gen[2] = {
            "generation": 2,
            "program_id": "prog-2",
            "fitness": 0.42,
            "chain_config": None,
            "mutation_summary": "fixed prompts",
        }
        s._versions = [
            {
                "generation": 2,
                "best_fitness": 0.42,
                "delta": 0.1,
                "best_individual_id": None,
            },
        ]
        with patch(
            "care.runtime.evolution_redis_probe.probe_program_chain_config",
            return_value={"steps": [{"number": 1, "title": "A"}]},
        ) as probe:
            chain = s._chain_for_version(s._versions[0])
        probe.assert_called_once_with("exp_abc", "prog-2")
        assert chain == {"steps": [{"number": 1, "title": "A"}]}

    def test_versions_mode_button_label_is_localized(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        s._versions_mode = "chain"
        assert s._versions_view_mode_label() == s._versions_view_mode_label()
        s._versions_mode = "diff"
        assert s._versions_view_mode_label() != ""


class TestCostTickHandling:
    def test_cost_tick_accumulates_tokens_without_event_row(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        rows: list[str] = []
        s._render_event_row = lambda kind, payload: rows.append(kind)  # type: ignore[method-assign]
        s._handle_event({"event": "cost_tick", "data": {"total_tokens": 1000}})
        s._handle_event({"event": "cost_tick", "data": {"total_tokens": 500}})
        # Deltas accumulate (additive) → 1500 total.
        assert s.run.total_tokens == 1500
        # High-frequency tick must not render a visible event-feed row.
        assert rows == []

    def test_cost_tick_folds_usd_when_present(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_x")
        s._handle_event(
            {"event": "cost_tick", "data": {"total_tokens": 100, "cost_usd": 0.25}}
        )
        assert s.run.total_tokens == 100
        assert s.run.total_cost_usd == 0.25
        # The cost meter now renders instead of staying blank.
        assert "cost:" in s.format_cost_meter()


class TestRunMetadataCard:
    def test_launch_mode_shows_rubric_mode(self) -> None:
        s = EvolutionScreen(
            base_chain_id="c",
            validation_criteria="Answers must be concise and correct",
            evolution_mode="full_chain",
        )
        meta = "\n".join(s._run_metadata_lines())
        assert "Optimising for:" in meta
        assert "concise and correct" in meta
        assert "full_chain" in meta

    def test_platform_version_shown_when_known(self) -> None:
        s = EvolutionScreen(base_chain_id="c", evolution_mode="full_chain")
        assert all("Platform:" not in ln for ln in s._run_metadata_lines())
        s._platform_version = "0.4.1"
        meta = "\n".join(s._run_metadata_lines())
        assert "Platform:" in meta and "0.4.1" in meta


class TestExtractPlatformVersion:
    def test_reads_primary_version_key(self) -> None:
        assert _extract_platform_version({"version": "0.4.1"}) == "0.4.1"

    def test_falls_back_to_platform_version_then_api_version(self) -> None:
        assert _extract_platform_version({"platform_version": "1.2"}) == "1.2"
        assert _extract_platform_version({"api_version": "3"}) == "3"

    def test_primary_key_wins_over_fallbacks(self) -> None:
        health = {"version": "a", "platform_version": "b", "api_version": "c"}
        assert _extract_platform_version(health) == "a"

    def test_coerces_non_string_version(self) -> None:
        assert _extract_platform_version({"version": 5}) == "5"

    def test_empty_when_no_version_key(self) -> None:
        assert _extract_platform_version({"status": "ok"}) == ""

    def test_empty_when_version_is_falsy(self) -> None:
        assert _extract_platform_version({"version": ""}) == ""

    def test_empty_when_not_a_dict(self) -> None:
        assert _extract_platform_version(None) == ""
        assert _extract_platform_version("0.4.1") == ""

    def test_observe_mode_recovers_rubric_and_runner_from_snapshot(self) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_z")
        s._handle_event(
            {
                "event": "snapshot",
                "data": {
                    "validation_rubric": "Reward factual answers",
                    "runner_id": "runner-3",
                },
            }
        )
        meta = "\n".join(s._run_metadata_lines())
        assert "Reward factual answers" in meta
        assert "runner-3" in meta

    def test_long_rubric_is_truncated(self) -> None:
        s = EvolutionScreen(
            base_chain_id="c", validation_criteria="x" * 500
        )
        line = s._run_metadata_lines()[0]
        assert line.endswith("…")
        assert len(line) < 300

    def test_no_metadata_when_nothing_known(self) -> None:
        s = EvolutionScreen(base_chain_id="c", evolution_mode="")
        assert s._run_metadata_lines() == []


class TestExportCurve:
    def test_exports_csv_and_json(self, tmp_path, monkeypatch) -> None:
        s = EvolutionScreen(base_chain_id="c", observe_evolution_id="exp_abc")
        s.run.evolution_id = "exp_abc"
        s.tracker.record_generation(
            GenerationStat(generation=0, best_fitness=0.2)
        )
        s.tracker.record_generation(
            GenerationStat(generation=1, best_fitness=0.5)
        )
        toasts: list = []
        monkeypatch.setattr(
            s, "_toast", lambda msg, **kw: toasts.append((msg, kw))
        )
        monkeypatch.chdir(tmp_path)
        s.action_export_curve()
        csv_path = tmp_path / "evolution-exp_abc-curve.csv"
        json_path = tmp_path / "evolution-exp_abc-curve.json"
        assert csv_path.exists() and json_path.exists()
        assert "generation,best_fitness" in csv_path.read_text()
        assert toasts and "Exported fitness curve" in toasts[0][0]

    def test_no_curve_toasts_hint(self, tmp_path, monkeypatch) -> None:
        s = EvolutionScreen(base_chain_id="c")
        toasts: list = []
        monkeypatch.setattr(
            s, "_toast", lambda msg, **kw: toasts.append((msg, kw))
        )
        monkeypatch.chdir(tmp_path)
        s.action_export_curve()
        assert toasts and "No fitness curve" in toasts[0][0]
        assert not list(tmp_path.glob("*.csv"))


class TestFitnessPlotDeflicker:
    def test_skips_replot_when_unchanged(self) -> None:
        class _FakePlot:
            def __init__(self):
                self.clears = 0

            def clear(self):
                self.clears += 1

            def plot(self, **kw):
                pass

            def set_xlabel(self, v):
                pass

            def set_ylabel(self, v):
                pass

        fake = _FakePlot()

        class _S(EvolutionScreen):
            @property
            def is_mounted(self):
                return True

            def query_one(self, *args, **kwargs):
                return fake

        s = _S(base_chain_id="c")
        recs = [
            GenerationStat(generation=0, best_fitness=0.2),
            GenerationStat(generation=1, best_fitness=0.5),
        ]
        s._render_fitness_plot_widget(recs)
        s._render_fitness_plot_widget(recs)  # identical → must skip
        assert fake.clears == 1
        # New data → re-renders.
        recs2 = recs + [GenerationStat(generation=2, best_fitness=0.7)]
        s._render_fitness_plot_widget(recs2)
        assert fake.clears == 2


class TestParetoDetail:
    def test_format_objectives_inline(self) -> None:
        assert _format_objectives_inline(()) == ""
        assert _format_objectives_inline((("acc", 0.8),)) == "acc=0.800"
        out = _format_objectives_inline(
            (("a", 0.1), ("b", 0.2), ("c", 0.3), ("d", 0.4))
        )
        assert out.endswith("…") and "a=0.100" in out

    def test_detail_card_shows_highlighted(self) -> None:
        captured = {}

        class _FakeStatic:
            def update(self, content):
                captured["content"] = str(content)

        fake = _FakeStatic()

        class _S(EvolutionScreen):
            @property
            def is_mounted(self):
                return True

            def query_one(self, *args, **kwargs):
                return fake

        s = _S(base_chain_id="c")
        s.run.individuals = [
            EvolutionIndividual(
                individual_id="ind-1",
                generation=2,
                fitness=0.7,
                objectives=(("accuracy", 0.7), ("speed", 0.9)),
                summary="a much longer summary that the table would truncate",
            )
        ]
        s._highlighted_individual = "ind-1"
        s._render_pareto_detail()
        body = captured["content"]
        assert "ind-1" in body
        assert "accuracy = 0.7000" in body
        assert "much longer summary" in body
        assert "best" in body  # only individual → it's the best

    def test_detail_card_empty_without_highlight(self) -> None:
        captured = {}

        class _FakeStatic:
            def update(self, content):
                captured["content"] = str(content)

        fake = _FakeStatic()

        class _S(EvolutionScreen):
            @property
            def is_mounted(self):
                return True

            def query_one(self, *args, **kwargs):
                return fake

        s = _S(base_chain_id="c")
        s.run.individuals = [
            EvolutionIndividual(individual_id="ind-1", generation=1)
        ]
        s._highlighted_individual = None  # nothing highlighted
        s._render_pareto_detail()
        # Placeholder (i18n) — must NOT render an individual's detail.
        assert captured["content"]
        assert "ind-1" not in captured["content"]


class TestLivenessLine:
    def test_empty_before_first_event(self) -> None:
        s = _screen()
        assert s._liveness_line() == ""

    def test_live_when_recent(self) -> None:
        s = _screen()
        s.run.last_event_monotonic = time.monotonic()
        s.run.data_source = "redis_probe"
        line = s._liveness_line()
        assert "live" in line
        assert "source: redis-probe" in line
        assert "updated" in line

    def test_stalled_when_old(self) -> None:
        s = _screen()
        s.run.last_event_monotonic = time.monotonic() - 60.0
        assert "stalled?" in s._liveness_line()

    def test_done_when_finished(self) -> None:
        s = _screen()
        s.run.last_event_monotonic = time.monotonic()
        s.run.finished = True
        assert "done" in s._liveness_line()
