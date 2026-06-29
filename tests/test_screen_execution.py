"""Pilot tests for ExecutionScreen (TODO §1.1 P0.22).

Exercises:
* Composition — three panes mount.
* `StepStarted` / `StepCompleted` / `ChainCompleted` / `Progress`
  drive the state.
* `LlmChunk` for the focused step appends to the stream.
* `HumanInputRequested` lands on `state.pending_human_prompt`.
* `Esc` cancels the `execute` worker group.
* CarlStreamer end-to-end integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from care.runtime.carl_streamer import (
    CarlStreamer,
    ChainCompleted,
    HumanInputRequested,
    LlmChunk,
    Progress,
    StepCompleted,
    StepStarted,
)
from care.screens.execution import (
    ExecutionScreen,
    ExecutionState,
    StepRecord,
    project_chain_steps,
)


CHAIN_STEPS = [
    {"number": 1, "title": "Plan", "step_type": "llm", "dependencies": []},
    {"number": 2, "title": "Fetch", "step_type": "tool", "dependencies": [1]},
    {"number": 3, "title": "Write", "step_type": "llm", "dependencies": [2]},
]


def _text_styles(renderable) -> set[str]:
    """Collect Rich style strings off a Static's rendered Text."""
    out: set[str] = set()
    if getattr(renderable, "style", None):
        out.add(str(renderable.style))
    for span in getattr(renderable, "spans", []):
        if span.style:
            out.add(str(span.style))
    return out


@dataclass
class _Result:
    step_number: int = 0
    summary: str = ""


class _ExecHost(App):
    def __init__(
        self, *, total_steps: int = 0, chain_steps: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self._total = total_steps
        self._chain_steps = chain_steps

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(
            ExecutionScreen(
                title="Test run",
                total_steps=self._total,
                chain_steps=self._chain_steps,
            ),
        )


def _screen(app: App) -> ExecutionScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, ExecutionScreen)
    return s


# ---------------------------------------------------------------------------
# Pure dataclass helpers
# ---------------------------------------------------------------------------


class TestStepRecord:
    def test_pending_format(self):
        r = StepRecord(step_number=1, title="plan")
        assert "·" in r.format_row()
        assert "step 1" in r.format_row()

    def test_running_format(self):
        r = StepRecord(
            step_number=2, title="run", status="running", started_at=0.0,
        )
        assert "▶" in r.format_row()

    def test_running_format_uses_spinner_glyph(self):
        """The running row substitutes the screen's spinner frame for the
        static badge, so the in-flight step animates."""
        r = StepRecord(
            step_number=2, title="run", status="running", started_at=0.0,
        )
        assert r.format_row(spinner="⠙").startswith("⠙ ")
        # pending / done ignore the spinner arg.
        done = StepRecord(step_number=1, status="done", started_at=0.0)
        assert "✓" in done.format_row(spinner="⠙")


class TestExecutionState:
    def test_progress_fraction_with_zero_total(self):
        st = ExecutionState()
        assert st.progress_fraction == 0.0

    def test_progress_fraction_capped_at_one(self):
        st = ExecutionState(completed=10, total=4)
        assert st.progress_fraction == 1.0


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_three_panes_mount(self):
        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.query_one("#execution-step-rows") is not None
            assert screen.query_one("#execution-stream-body") is not None
            assert screen.query_one(
                "#execution-telemetry-text", Static,
            ) is not None


# ---------------------------------------------------------------------------
# Event handling
# ---------------------------------------------------------------------------


class TestEventHandling:
    @pytest.mark.asyncio
    async def test_step_started_records_running(self):
        app = _ExecHost(total_steps=2)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StepStarted(1, "plan"))
            await pilot.pause()
            await pilot.pause()
            rec = screen.state.steps[1]
            assert rec.status == "running"
            assert rec.title == "plan"
            assert screen.focused_step == 1

    @pytest.mark.asyncio
    async def test_running_step_row_animates_spinner(self):
        """`_tick_spinner` advances the frame and re-renders the running
        step row so the in-flight step shows a live loading animation."""
        from textual.widgets import Static

        from care.screens.execution import _SPINNER_FRAMES

        app = _ExecHost(total_steps=2)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StepStarted(1, "plan"))
            await pilot.pause()
            row = screen.query_one("#execution-step-1", Static)
            # The row shows a spinner frame (not the static "▶") + tick moves it.
            before = screen._spinner_idx
            screen._tick_spinner()
            await pilot.pause()
            assert screen._spinner_idx != before
            text = str(row.render())
            assert any(f in text for f in _SPINNER_FRAMES)

    @pytest.mark.asyncio
    async def test_tick_spinner_noop_when_idle(self):
        app = _ExecHost(total_steps=2)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            idx = screen._spinner_idx
            screen._tick_spinner()  # nothing running
            assert screen._spinner_idx == idx

    @pytest.mark.asyncio
    async def test_step_completed_marks_done(self):
        app = _ExecHost(total_steps=2)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StepStarted(1, "plan"))
            await pilot.pause()
            screen.post_message(StepCompleted(_Result(step_number=1, summary="ok")))
            await pilot.pause()
            await pilot.pause()
            rec = screen.state.steps[1]
            assert rec.status == "done"
            assert screen.state.completed == 1
            assert rec.result_summary == "ok"

    @pytest.mark.asyncio
    async def test_progress_updates_total(self):
        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(Progress(2, 5))
            await pilot.pause()
            await pilot.pause()
            assert screen.state.completed == 2
            assert screen.state.total == 5

    @pytest.mark.asyncio
    async def test_chain_completed_sets_finished(self):
        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(ChainCompleted({"ok": True}))
            await pilot.pause()
            await pilot.pause()
            assert screen.state.finished is True
            assert screen.state.chain_result == {"ok": True}


# ---------------------------------------------------------------------------
# LLM stream
# ---------------------------------------------------------------------------


class TestLlmStream:
    @pytest.mark.asyncio
    async def test_chunk_for_focused_step_appends(self):
        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StepStarted(1, "plan"))
            await pilot.pause()
            screen.post_message(LlmChunk("hello ", step_number=1))
            screen.post_message(LlmChunk("world", step_number=1))
            await pilot.pause()
            await pilot.pause()
            # Chunks accumulate per step into the transcript Static.
            assert screen._step_streams.get(1) == "hello world"
            rendered = str(screen.query_one("#execution-stream-text", Static).render())
            assert "hello world" in rendered
            assert "Step 1" in rendered  # per-step header

    @pytest.mark.asyncio
    async def test_chunk_for_other_step_ignored(self):
        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StepStarted(1, "plan"))
            await pilot.pause()
            # Chunk for step 2 should not append while step 1
            # is focused.
            screen.post_message(LlmChunk("noise", step_number=2))
            await pilot.pause()
            # Non-focused step's chunk is dropped → step 1 has no output,
            # step 2 isn't registered.
            assert screen._step_streams.get(1, "") == ""
            assert 2 not in screen._step_streams

    @pytest.mark.asyncio
    async def test_chunk_without_step_appends(self):
        # No step_number means it lands on the focused step
        # by default — the streamer doesn't always have one.
        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(StepStarted(1, "plan"))
            await pilot.pause()
            screen.post_message(LlmChunk("anonymous"))
            await pilot.pause()
            # No step_number → lands on the focused step (1).
            assert "anonymous" in screen._step_streams.get(1, "")


# ---------------------------------------------------------------------------
# Human input
# ---------------------------------------------------------------------------


class TestHumanInput:
    @pytest.mark.asyncio
    async def test_request_lands_on_state(self):
        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.post_message(
                HumanInputRequested("What say you?", future=None),
            )
            await pilot.pause()
            assert screen.state.pending_human_prompt == "What say you?"


# ---------------------------------------------------------------------------
# Cancel binding
# ---------------------------------------------------------------------------


class TestCancel:
    @pytest.mark.asyncio
    async def test_esc_cancels_execute_group(self):
        import asyncio

        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)

            async def _long():
                await asyncio.sleep(5.0)

            worker = screen.run_worker(
                _long(),
                name="fake_execute",
                group="execute",
                exclusive=False,
                exit_on_error=False,
            )
            await pilot.pause()
            screen.action_cancel_execute()
            for _ in range(5):
                await pilot.pause()
            assert screen.cancelled is True
            assert not worker.is_running


# ---------------------------------------------------------------------------
# CarlStreamer end-to-end
# ---------------------------------------------------------------------------


class TestCarlStreamerIntegration:
    @pytest.mark.asyncio
    async def test_streamer_drives_screen(self):
        app = _ExecHost(total_steps=2)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            streamer = CarlStreamer(screen)
            # Use the screen's reachable callbacks directly —
            # these post the right messages.
            streamer._target.post_message(StepStarted(1, "plan"))
            streamer._target.post_message(
                StepCompleted(_Result(step_number=1, summary="planned")),
            )
            streamer._target.post_message(Progress(1, 2))
            await pilot.pause()
            await pilot.pause()
            assert screen.state.completed == 1
            assert screen.state.total == 2


# ---------------------------------------------------------------------------
# Chain → step-dict projection
# ---------------------------------------------------------------------------


class TestProjectChainSteps:
    def test_dict_chain(self):
        out = project_chain_steps({"steps": CHAIN_STEPS})
        assert [s["number"] for s in out] == [1, 2, 3]

    def test_bare_list(self):
        assert project_chain_steps(CHAIN_STEPS) == CHAIN_STEPS

    def test_object_with_model_dump(self):
        class _Step:
            def __init__(self, n: int) -> None:
                self._n = n

            def model_dump(self) -> dict:
                return {"number": self._n, "step_type": "llm",
                        "dependencies": []}

        class _Chain:
            steps = [_Step(1), _Step(2)]

        out = project_chain_steps(_Chain())
        assert [s["number"] for s in out] == [1, 2]

    def test_object_fallback_attrs(self):
        # A step object with neither model_dump nor to_dict falls back to
        # attribute scraping for the fields the DAG needs.
        class _Step:
            number = 5
            title = "x"
            step_type = "tool"
            dependencies = [1]

        class _Chain:
            steps = [_Step()]

        out = project_chain_steps(_Chain())
        assert out[0]["number"] == 5
        assert out[0]["step_type"] == "tool"
        assert out[0]["dependencies"] == [1]

    def test_serialised_chain_with_content_wrapper(self):
        class _Chain:
            def to_dict(self) -> dict:
                return {"content": {"steps": CHAIN_STEPS}}

        out = project_chain_steps(_Chain())
        assert [s["number"] for s in out] == [1, 2, 3]

    def test_unrecognised_returns_empty(self):
        assert project_chain_steps(object()) == []
        assert project_chain_steps(None) == []


# ---------------------------------------------------------------------------
# Live DAG overlay
# ---------------------------------------------------------------------------


class TestDagOverlay:
    @pytest.mark.asyncio
    async def test_dag_hidden_without_chain_steps(self):
        app = _ExecHost(total_steps=2)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.query_one("#execution-dag-section").display is False

    @pytest.mark.asyncio
    async def test_dag_renders_chain_shape(self):
        app = _ExecHost(total_steps=3, chain_steps=CHAIN_STEPS)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            assert screen.query_one("#execution-dag-section").display is True
            renderable = screen.query_one(
                "#execution-dag-text", Static,
            ).render()
            plain = getattr(renderable, "plain", "") or str(renderable)
            assert "Plan" in plain and "Fetch" in plain and "Write" in plain

    @pytest.mark.asyncio
    async def test_dag_recolours_as_steps_run(self):
        app = _ExecHost(total_steps=3, chain_steps=CHAIN_STEPS)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            # Step 1 done, step 2 running.
            screen.post_message(StepStarted(1, "Plan"))
            await pilot.pause()
            screen.post_message(StepCompleted(_Result(step_number=1, summary="ok")))
            screen.post_message(StepStarted(2, "Fetch"))
            await pilot.pause()
            await pilot.pause()
            renderable = screen.query_one(
                "#execution-dag-text", Static,
            ).render()
            styles = _text_styles(renderable)
            # Styles resolve to ansi_* / rgb() forms at render time, so
            # match by colour substring rather than the raw style string.
            assert any("green" in s for s in styles)   # step 1 → done
            assert any("yellow" in s for s in styles)  # step 2 → running

    @pytest.mark.asyncio
    async def test_overlay_toggle_is_noop_without_profiling(self):
        app = _ExecHost(total_steps=3, chain_steps=CHAIN_STEPS)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.action_cycle_dag_overlay()  # no run yet → no profiling
            assert screen._dag_overlay == "status"

    @pytest.mark.asyncio
    async def test_overlay_toggles_to_latency_heatmap(self):
        app = _ExecHost(total_steps=3, chain_steps=CHAIN_STEPS)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            # Populate profiling (step 2 is the slow one).
            screen.record_chain_result({
                "steps": [
                    {"step_number": 1, "execution_time_s": 0.1},
                    {"step_number": 2, "execution_time_s": 5.0},
                    {"step_number": 3, "execution_time_s": 1.0},
                ]
            })
            screen.action_cycle_dag_overlay()
            await pilot.pause()
            assert screen._dag_overlay == "latency"
            styles = _text_styles(
                screen.query_one("#execution-dag-text", Static).render()
            )
            assert any("red" in s for s in styles)    # slow step → hot
            assert any("green" in s for s in styles)  # fast step → cool


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import ExecutionScreen as E
        from care.screens import ExecutionState as S

        assert E is ExecutionScreen
        assert S is ExecutionState


# ---------------------------------------------------------------------------
# Profiling pane (§5 [DONE — data layer] → fully DONE)
# ---------------------------------------------------------------------------


class TestProfiling:
    @pytest.mark.asyncio
    async def test_chain_complete_populates_profiling(self):
        from care.profiling import StepProfile

        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            # Build a result with `step_results` carrying
            # profiling dicts that the projector understands.
            result = {
                "steps": [
                    {
                        "step_number": 1,
                        "step_type": "llm",
                        "step_title": "fetch",
                        "execution_time_s": 1.5,
                        "history_bytes_added": 100,
                        "memory_bytes_after": 200,
                        "batch_index": 0,
                        "success": True,
                    },
                ],
                "total_execution_time_s": 1.5,
                "total_history_bytes": 100,
                "peak_memory_bytes": 200,
            }
            screen.post_message(ChainCompleted(result))
            for _ in range(4):
                await pilot.pause()
            assert screen.profiling.step_count == 1
            assert screen.profiling.total_execution_time_s == 1.5
            assert isinstance(screen.profiling.steps[0], StepProfile)

    @pytest.mark.asyncio
    async def test_record_chain_result_silent_on_failure(self):
        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)

            class _Bad:
                @property
                def step_results(self):
                    raise RuntimeError("boom")

            # No exception propagates; profiling stays empty.
            screen.record_chain_result(_Bad())
            await pilot.pause()
            assert screen.profiling.is_empty is True

    @pytest.mark.asyncio
    async def test_empty_result_keeps_empty_profiling(self):
        app = _ExecHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.record_chain_result({})
            await pilot.pause()
            assert screen.profiling.is_empty is True
