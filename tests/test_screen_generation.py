"""Pilot tests for GenerationScreen (TODO §1.1 P0.16).

Exercises:
* Composition — stage log + DAG preview both mount.
* `StageStarted` / `StageCompleted` / `StageError` update the
  left-pane row state.
* `StageProgress` appends an artifact row to the right pane.
* `StageRetry` increments the retry count.
* `MagePoster` drives the screen end-to-end (the data layer's
  primary integration point).
* The `progress` snapshot mirrors the rendered state.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from care.runtime.mage_poster import (
    MagePoster,
    StageCompleted,
    StageError,
    StageProgress,
    StageRetry,
    StageStarted,
)
from care.screens.generation import (
    GenerationProgress,
    GenerationScreen,
    StageState,
)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _GenHost(App):
    def __init__(self, *, task_preview: str = "") -> None:
        super().__init__()
        self._task_preview = task_preview

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(
            GenerationScreen(task_preview=self._task_preview),
        )


def _screen(app: App) -> GenerationScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, GenerationScreen)
    return s


# ---------------------------------------------------------------------------
# Dataclass helpers
# ---------------------------------------------------------------------------


class TestStageState:
    def test_default_pending(self):
        s = StageState(name="analyze_domain")
        assert s.status == "pending"
        assert s.elapsed() is None
        assert "·" in s.format_row()

    def test_running_format(self):
        s = StageState(name="plan", status="running", started_at=0.0)
        assert "▶" in s.format_row()

    def test_failed_format(self):
        s = StageState(name="plan", status="failed", error="boom")
        assert "✗" in s.format_row()
        assert "boom" in s.format_row()


class TestGenerationProgress:
    def test_empty_progress(self):
        p = GenerationProgress()
        assert p.is_complete is False
        assert p.has_failure is False

    def test_complete_when_every_stage_done(self):
        p = GenerationProgress(
            stages={
                "a": StageState(name="a", status="done"),
                "b": StageState(name="b", status="done"),
            },
        )
        assert p.is_complete is True

    def test_failure_detection(self):
        p = GenerationProgress(
            stages={
                "a": StageState(name="a", status="done"),
                "b": StageState(name="b", status="failed", error="x"),
            },
        )
        assert p.has_failure is True


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_mounts_stages_and_dag_panes(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.query_one("#stage-rows") is not None
            assert screen.query_one("#dag-rows") is not None


# ---------------------------------------------------------------------------
# Stage events
# ---------------------------------------------------------------------------


class TestStageEvents:
    @pytest.mark.asyncio
    async def test_stage_started_creates_running_row(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StageStarted("analyze_domain"))
            await pilot.pause()
            await pilot.pause()
            state = screen.progress.stages["analyze_domain"]
            assert state.status == "running"
            # Row mounted in stage-rows.
            row = screen.query_one(
                f"#{screen._stage_row_id('analyze_domain')}",
                Static,
            )
            assert row is not None

    @pytest.mark.asyncio
    async def test_stage_completed_marks_done(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StageStarted("plan"))
            await pilot.pause()
            screen.post_message(StageCompleted("plan", {"steps": 3}))
            await pilot.pause()
            await pilot.pause()
            assert screen.progress.stages["plan"].status == "done"

    @pytest.mark.asyncio
    async def test_stage_error_marks_failed(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StageStarted("plan"))
            await pilot.pause()
            screen.post_message(StageError("plan", RuntimeError("nope")))
            await pilot.pause()
            await pilot.pause()
            state = screen.progress.stages["plan"]
            assert state.status == "failed"
            assert "RuntimeError" in (state.error or "")
            assert screen.progress.has_failure is True

    @pytest.mark.asyncio
    async def test_stage_retry_increments_counter(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StageStarted("plan"))
            await pilot.pause()
            screen.post_message(
                StageRetry("plan", 1, RuntimeError("flaky")),
            )
            await pilot.pause()
            await pilot.pause()
            assert screen.progress.stages["plan"].retries == 1


# ---------------------------------------------------------------------------
# DAG preview
# ---------------------------------------------------------------------------


class TestDAGPreview:
    @pytest.mark.asyncio
    async def test_stage_progress_appends_row(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StageProgress("describe", {"name": "step1"}))
            screen.post_message(StageProgress("describe", {"name": "step2"}))
            await pilot.pause()
            await pilot.pause()
            assert len(screen.progress.artifacts) == 2
            rows = list(screen.query("#dag-rows Static"))
            # Two artifact rows mounted.
            assert len(rows) >= 2

    @pytest.mark.asyncio
    async def test_progress_with_string_artifact(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StageProgress("describe", "plain text"))
            await pilot.pause()
            await pilot.pause()
            # One row mounted; the model carries the string.
            assert screen.progress.artifacts == [("describe", "plain text")]
            rows = list(screen.query("#dag-rows Static"))
            assert len(rows) >= 1


# ---------------------------------------------------------------------------
# MagePoster integration
# ---------------------------------------------------------------------------


class TestMagePosterIntegration:
    @pytest.mark.asyncio
    async def test_full_run_via_poster(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            poster = MagePoster(screen)
            poster.on_stage_start("analyze_domain")
            poster.on_stage_complete("analyze_domain", {"domain": "weather"})
            poster.on_stage_start("plan_steps")
            poster.on_stage_progress("plan_steps", {"name": "step-A"})
            poster.on_stage_complete("plan_steps", {"plan": True})
            await pilot.pause()
            await pilot.pause()
            assert screen.progress.stages["analyze_domain"].status == "done"
            assert screen.progress.stages["plan_steps"].status == "done"
            assert screen.progress.is_complete is True
            assert any(
                stage == "plan_steps"
                for stage, _ in screen.progress.artifacts
            )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import GenerationProgress as G1
        from care.screens import GenerationScreen as G2

        assert G1 is GenerationProgress
        assert G2 is GenerationScreen


# ---------------------------------------------------------------------------
# Esc cancel binding (P0.17)
# ---------------------------------------------------------------------------


class TestCancelBinding:
    @pytest.mark.asyncio
    async def test_esc_sets_cancelled_flag(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.cancelled is False
            screen.action_cancel_generate()
            await pilot.pause()
            assert screen.cancelled is True

    @pytest.mark.asyncio
    async def test_esc_cancels_generate_worker_group(self):
        import asyncio

        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)

            async def _long_running():
                # Sleep long enough to be cancellable.
                await asyncio.sleep(5.0)

            worker = screen.run_worker(
                _long_running(),
                name="fake_generate",
                group="generate",
                exclusive=False,
                exit_on_error=False,
            )
            await pilot.pause()
            assert worker.is_running or worker.is_finished
            screen.action_cancel_generate()
            # Pump the event loop a few times to let the
            # cancellation propagate.
            for _ in range(5):
                await pilot.pause()
            assert not worker.is_running

    @pytest.mark.asyncio
    async def test_esc_keybinding_resolves_action(self):
        # `BINDINGS` table should resolve `escape` to the
        # cancel action — pinning the convention shared with
        # DemoScreen + future ExecutionScreen.
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            keys = {b.key for b in screen.BINDINGS}
            assert "escape" in keys
            actions = {b.action for b in screen.BINDINGS}
            assert "cancel_generate" in actions


# ---------------------------------------------------------------------------
# MAGE metadata footer (§1.2 [DONE — data layer] → fully DONE)
# ---------------------------------------------------------------------------


class TestMageMetadataFooter:
    @pytest.mark.asyncio
    async def test_record_mage_result_populates_summary(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            # Pass a duck-typed MAGEResult: anything with
            # `.metadata` + `.mode`.
            class _Result:
                mode = "deep"
                class metadata:
                    domain = "weather"
                    num_steps = 5
                    model = "gpt-4o"
                    generation_time_seconds = 3.2
                    deep_stages_completed = (
                        "analyze", "plan", "describe",
                    )
                    memory_hits_used = 2
                    web_results_used = 0
                    was_cold_start = False
                    step_critique_score = 0.92
                    verification_passed = True
                    refine_iterations = None
                    refine_quality_delta = None
                    tot_branches_explored = None
                    mcts_simulations_run = None
                    mcts_best_reward = None
                    feedback_recalled = None
                    suggested_display_name = ""
                    suggested_description = ""
                    suggested_tags = ()

            screen.record_mage_result(_Result())
            await pilot.pause()
            assert screen.metadata_summary is not None
            assert screen.metadata_summary.domain == "weather"
            assert screen.metadata_summary.num_steps == 5
            # Static rendered.
            target = screen.query_one("#generation-metadata", Static)
            assert target is not None

    @pytest.mark.asyncio
    async def test_record_failure_is_silent(self):
        app = _GenHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)

            class _Bad:
                @property
                def metadata(self):
                    raise RuntimeError("boom")

            # No exception propagates; summary stays None.
            screen.record_mage_result(_Bad())
            await pilot.pause()
            assert screen.metadata_summary is None
