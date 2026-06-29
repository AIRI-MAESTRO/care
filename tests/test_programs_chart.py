"""Tests for the Programs-tab chart helpers + tracker program history."""

from __future__ import annotations

from care.evolution_session import EvolutionProgressTracker
from care.runtime.programs_chart import render_programs_pie, render_programs_trend


class TestRenderProgramsPie:
    def test_empty_when_nothing_reported(self) -> None:
        assert render_programs_pie(-1, -1) == ""

    def test_renders_counts_and_pct(self) -> None:
        out = render_programs_pie(8, 2)
        assert "Total programs: 10" in out
        assert "Valid" in out and "8" in out
        assert "Invalid" in out and "2" in out


class TestRenderProgramsTrend:
    def test_empty_for_single_point(self) -> None:
        assert render_programs_trend([(0, 5, 1)]) == ""

    def test_sparkline_for_multiple_generations(self) -> None:
        out = render_programs_trend([(0, 2, 0), (1, 5, 1), (2, 9, 2)])
        assert "valid-program trend" in out
        assert "gen 0..2" in out
        assert "min 2 → max 9" in out

    def test_flat_series_renders_without_div_by_zero(self) -> None:
        out = render_programs_trend([(0, 4, 0), (1, 4, 0)])
        assert "min 4 → max 4" in out


class TestTrackerProgramsHistory:
    def test_record_and_curve(self) -> None:
        t = EvolutionProgressTracker()
        t.record_programs(0, 3, 1)
        t.record_programs(1, 7, 2)
        assert t.programs_curve() == ((0, 3, 1), (1, 7, 2))

    def test_partial_update_keeps_prior_side(self) -> None:
        t = EvolutionProgressTracker()
        t.record_programs(0, 5, 2)
        t.record_programs(0, 6, None)  # invalid not reported → keep 2
        assert t.programs_curve() == ((0, 6, 2),)

    def test_non_int_generation_ignored(self) -> None:
        t = EvolutionProgressTracker()
        t.record_programs("x", 5, 2)  # type: ignore[arg-type]
        assert t.programs_curve() == ()
