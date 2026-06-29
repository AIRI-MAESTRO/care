"""Tests for `EvolutionCompareModal` (TODO §5 P1)."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from care.screens.evolution_compare import (
    EvolutionCompareModal,
    extract_compare_summary,
    extract_fitness_records,
    format_compare_summary_lines,
)


class TestExtractCompareSummary:
    def test_reads_top_level_and_nested_metrics(self):
        state = {
            "best_fitness": 0.7,
            "generation": 4,
            "_raw": {
                "results": {
                    "metrics": {
                        "current_fitness": 0.55,
                        "programs_valid": 9,
                        "programs_invalid": 3,
                    }
                }
            },
        }
        s = extract_compare_summary(state)
        assert s.best_fitness == 0.7
        assert s.generation == 4
        assert s.current_fitness == 0.55
        assert s.programs_valid == 9
        assert s.programs_invalid == 3
        assert s.has_any

    def test_empty_state_has_nothing(self):
        s = extract_compare_summary({})
        assert not s.has_any
        assert format_compare_summary_lines(s) == []

    def test_format_lines_include_programs_and_pct(self):
        state = {
            "generation": 2,
            "best_fitness": 0.5,
            "metrics": {"programs_valid": 8, "programs_invalid": 2},
        }
        lines = format_compare_summary_lines(extract_compare_summary(state))
        joined = "\n".join(lines)
        assert "gen 2" in joined
        assert "0.5000" in joined
        assert "8" in joined and "valid" in joined
        assert "80% valid" in joined


class TestExtractFitnessRecordsNested:
    """Chain (exp_*) runs nest the curve under _raw.results.metrics —
    the curve must render there too, not just at the top level."""

    def test_reads_curve_from_nested_chain_payload(self):
        state = {
            "best_fitness": 0.6,
            "generation": 2,
            "_raw": {
                "results": {
                    "metrics": {
                        "fitness_history": [
                            {"generation": 0, "best_fitness": 0.2},
                            {"generation": 1, "best_fitness": 0.5},
                        ]
                    }
                }
            },
        }
        out = extract_fitness_records(state)
        assert len(out) == 2
        assert out[0].generation == 0
        assert out[1].best_fitness == 0.5

    def test_top_level_still_wins_when_present(self):
        state = {
            "fitness_history": [{"generation": 0, "best_fitness": 0.9}],
            "_raw": {"results": {"metrics": {"fitness_history": []}}},
        }
        out = extract_fitness_records(state)
        assert len(out) == 1
        assert out[0].best_fitness == 0.9


# ---------------------------------------------------------------------------
# Pure projection
# ---------------------------------------------------------------------------


class TestExtractFitnessRecords:
    def test_flat_history(self):
        state = {
            "fitness_history": [
                {"generation": 0, "best_fitness": 0.2},
                {"generation": 1, "best_fitness": 0.5},
            ],
        }
        out = extract_fitness_records(state)
        assert len(out) == 2
        assert out[0].generation == 0
        assert out[0].best_fitness == 0.2
        assert out[1].best_fitness == 0.5

    def test_generations_alias(self):
        state = {
            "generations": [
                {"gen": 0, "fitness": 0.1},
                {"gen": 1, "fitness": 0.6},
            ],
        }
        out = extract_fitness_records(state)
        assert len(out) == 2
        assert out[1].best_fitness == 0.6

    def test_nested_progress(self):
        state = {
            "progress": {
                "fitness_history": [
                    {"generation": 5, "best": 0.9},
                ],
            },
        }
        out = extract_fitness_records(state)
        assert len(out) == 1
        assert out[0].best_fitness == 0.9

    def test_malformed_dropped(self):
        state = {
            "fitness_history": [
                {"generation": 0, "best_fitness": 0.2},
                "junk-not-a-dict",
                {"generation": "x", "best_fitness": 0.5},  # bad gen
                {"generation": 2},  # missing fitness
                {"generation": 3, "best_fitness": 0.8},
            ],
        }
        out = extract_fitness_records(state)
        assert len(out) == 2
        assert out[0].generation == 0
        assert out[1].generation == 3

    def test_empty_state(self):
        assert extract_fitness_records({}) == ()
        assert extract_fitness_records(None) == ()

    def test_non_mapping(self):
        assert extract_fitness_records([1, 2, 3]) == ()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_run_id_rejected(self):
        with pytest.raises(ValueError):
            EvolutionCompareModal(
                left_run_id="",
                right_run_id="evo-1",
            )
        with pytest.raises(ValueError):
            EvolutionCompareModal(
                left_run_id="evo-1",
                right_run_id="",
            )

    def test_construction_fields(self):
        modal = EvolutionCompareModal(
            left_run_id="evo-A",
            right_run_id="evo-B",
        )
        assert modal.left_run_id == "evo-A"
        assert modal.right_run_id == "evo-B"
        assert modal._left_state is None
        assert modal._right_state is None


class TestFetchTimeoutAndSpinner:
    def test_fetch_times_out(self):
        import asyncio
        import time

        class _SlowPlatform:
            def get_evolution(self, run_id):
                time.sleep(0.2)
                return {}

        modal = EvolutionCompareModal(
            left_run_id="a", right_run_id="b", platform=_SlowPlatform()
        )
        modal._FETCH_TIMEOUT_SECONDS = 0.01
        state, error = asyncio.run(modal._fetch_state("a"))
        assert state is None
        assert error is not None and "timed out" in error

    def test_spinner_lifecycle_unmounted(self):
        # On a bare (unmounted) modal set_interval no-ops; the pending-set
        # bookkeeping must still work so the spinner stops when both land.
        modal = EvolutionCompareModal(left_run_id="a", right_run_id="b")
        modal._start_spinner(("left", "right"))
        assert modal._pending_sides == {"left", "right"}
        modal._finish_side("left")
        assert modal._pending_sides == {"right"}
        modal._finish_side("right")
        assert modal._pending_sides == set()
        assert modal._spinner_timer is None


# ---------------------------------------------------------------------------
# Modal compose + render
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, modal: EvolutionCompareModal):
        super().__init__()
        self._modal = modal

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self._modal)


class TestCompose:
    @pytest.mark.asyncio
    async def test_preloaded_renders_both_sides(self):
        left = {
            "fitness_history": [
                {"generation": 0, "best_fitness": 0.2},
                {"generation": 1, "best_fitness": 0.5},
                {"generation": 2, "best_fitness": 0.7},
            ],
        }
        right = {
            "fitness_history": [
                {"generation": 0, "best_fitness": 0.3},
                {"generation": 1, "best_fitness": 0.8},
            ],
        }
        modal = EvolutionCompareModal(
            left_run_id="evo-A",
            right_run_id="evo-B",
            left_state=left,
            right_state=right,
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            left_text = str(
                modal.query_one(
                    "#evo-compare-left-text", Static,
                ).render()
            )
            right_text = str(
                modal.query_one(
                    "#evo-compare-right-text", Static,
                ).render()
            )
            # Both panes carry "best fitness:" header from
            # the sparkline renderer when records land.
            assert "best fitness" in left_text.lower()
            assert "best fitness" in right_text.lower()
            assert "0.500" in left_text or "0.700" in left_text
            assert "0.800" in right_text

    @pytest.mark.asyncio
    async def test_preloaded_no_history_shows_placeholder(self):
        modal = EvolutionCompareModal(
            left_run_id="evo-A",
            right_run_id="evo-B",
            left_state={},  # no fitness data
            right_state={"fitness_history": [
                {"generation": 0, "best_fitness": 0.5},
            ]},
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            left_text = str(
                modal.query_one(
                    "#evo-compare-left-text", Static,
                ).render()
            )
            assert "no fitness data yet" in left_text

    @pytest.mark.asyncio
    async def test_fetch_no_platform_shows_error(self):
        modal = EvolutionCompareModal(
            left_run_id="evo-A",
            right_run_id="evo-B",
            platform=None,
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            left_text = str(
                modal.query_one(
                    "#evo-compare-left-text", Static,
                ).render()
            )
            assert "no platform configured" in left_text

    @pytest.mark.asyncio
    async def test_fetch_via_platform_renders(self):
        class _Platform:
            def __init__(self):
                self.calls = []
            def get_evolution(self, run_id):
                self.calls.append(run_id)
                return {
                    "fitness_history": [
                        {"generation": 0, "best_fitness": 0.4},
                        {"generation": 1, "best_fitness": 0.6},
                    ],
                }

        platform = _Platform()
        modal = EvolutionCompareModal(
            left_run_id="evo-A",
            right_run_id="evo-B",
            platform=platform,
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            assert platform.calls == ["evo-A", "evo-B"]
            left_text = str(
                modal.query_one(
                    "#evo-compare-left-text", Static,
                ).render()
            )
            assert "best fitness" in left_text.lower()

    @pytest.mark.asyncio
    async def test_fetch_raises_shows_error_message(self):
        class _BadPlatform:
            def get_evolution(self, run_id):
                raise RuntimeError("503 backend down")

        modal = EvolutionCompareModal(
            left_run_id="evo-A",
            right_run_id="evo-B",
            platform=_BadPlatform(),
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()
            left_text = str(
                modal.query_one(
                    "#evo-compare-left-text", Static,
                ).render()
            )
            assert "503 backend down" in left_text

    @pytest.mark.asyncio
    async def test_close_action_dismisses(self):
        modal = EvolutionCompareModal(
            left_run_id="evo-A",
            right_run_id="evo-B",
            left_state={}, right_state={},
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal.action_close()
            await pilot.pause()
            assert modal.action_log[-1] == ("close", "")


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import (
            EvolutionCompareModal as Re,
        )
        assert Re is EvolutionCompareModal
