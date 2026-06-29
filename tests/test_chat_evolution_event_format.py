"""ChatScreen._format_evolution_event — the `/evolution watch` line renderer.

Locks in that it unwraps the ``{"event":..., "data":{...}}`` SSE shape (the
poll path + SDK normaliser both wrap fields under ``data``) and drops
high-frequency / chart-only frames so the transcript stays readable.
"""

from __future__ import annotations

from care.screens.chat import ChatScreen

_fmt = ChatScreen._format_evolution_event


class TestUnwrapsData:
    def test_generation_started_reads_nested_data(self) -> None:
        out = _fmt({"event": "generation_started", "data": {"generation": 3}})
        assert out == "▶ generation 3 started"

    def test_best_updated_reads_nested_fitness(self) -> None:
        out = _fmt(
            {"event": "best_updated", "data": {"generation": 2, "best_fitness": 0.75}}
        )
        assert out == "★ new best (gen 2): 0.750"

    def test_status_reads_nested_status(self) -> None:
        out = _fmt({"event": "status", "data": {"status": "running"}})
        assert out == "· status: running"

    def test_individual_evaluated_reads_nested(self) -> None:
        out = _fmt(
            {"event": "individual_evaluated", "data": {"individual_id": "p1", "fitness": 0.5}}
        )
        assert out == "  · p1: 0.500"

    def test_flat_shape_still_works(self) -> None:
        # Older/looser shape without the data wrapper.
        assert _fmt({"event": "generation_started", "generation": 5}) == "▶ generation 5 started"


class TestSkipsNoiseFrames:
    def test_high_frequency_and_chart_frames_return_empty(self) -> None:
        for kind in (
            "heartbeat",
            "cost_tick",
            "fitness_history_snapshot",
            "frontier_programs_snapshot",
            "programs_snapshot",
        ):
            assert _fmt({"event": kind, "data": {"experiment_id": "exp_x"}}) == "", kind


class TestTerminal:
    def test_completed_with_winner(self) -> None:
        out = _fmt({"event": "completed", "data": {"individual_id": "w1"}})
        assert out == "🏁 evolution finished — winner: w1"

    def test_completed_without_winner(self) -> None:
        assert _fmt({"event": "completed", "data": {}}) == "🏁 evolution finished"

    def test_failed_with_error(self) -> None:
        out = _fmt({"event": "failed", "data": {"error": "boom"}})
        assert out == "✗ evolution failed: boom"

    def test_cancelled_without_error(self) -> None:
        assert _fmt({"event": "cancelled", "data": {}}) == "✗ evolution cancelled"


class TestUnknownFallback:
    def test_unknown_kind_renders_payload_without_experiment_id(self) -> None:
        out = _fmt({"event": "mystery", "data": {"a": 1, "experiment_id": "exp_x"}})
        assert out == "· mystery: {'a': 1}"

    def test_non_dict_event(self) -> None:
        assert _fmt("nope") == "· nope"
