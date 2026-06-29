"""The combined status strip — pipeline cells + the "thinking…" spinner on
one line: ``◇ Generate → ○ Run? | ● thinking…``.

Boots a minimal host straight to a ``ChatScreen`` (the real ``CareApp``
first-run routing lands on Settings/Welcome in a fresh test env) and
exercises the pure render + the show/think/hide lifecycle.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from care.config import CareConfig
from care.screens.chat import ChatScreen


class _Host(App):
    """Minimal host: default config + a ChatScreen on the stack."""

    def __init__(self):
        super().__init__()
        self.config = CareConfig()

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


def _chat(app: _Host) -> ChatScreen:
    for s in app.screen_stack:
        if isinstance(s, ChatScreen):
            return s
    raise AssertionError("ChatScreen not on stack")


class TestStatusStrip:
    @pytest.mark.asyncio
    async def test_pipeline_and_thinking_on_one_line(self):
        app = _Host()
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(8):
                await pilot.pause()
            scr = _chat(app)
            scr._show_pipeline_strip(scr._current_mode_spec())
            scr._set_spinner_visible(True)
            for _ in range(2):
                await pilot.pause()
            strip = scr.query_one("#chat-pipeline-strip", Static)
            rendered = str(strip.render())
            assert strip.display is True
            # One line carrying BOTH the pipeline cells and the thinking tail.
            assert "Generate" in rendered
            assert "thinking" in rendered
            assert "|" in rendered
            assert "\n" not in rendered.strip()

    @pytest.mark.asyncio
    async def test_thinking_tail_only_while_active(self):
        app = _Host()
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(8):
                await pilot.pause()
            scr = _chat(app)
            scr._show_pipeline_strip(scr._current_mode_spec())
            scr._set_spinner_visible(True)
            await pilot.pause()
            assert "thinking" in scr._render_status_strip()
            # Stop "thinking" → tail gone, pipeline cells remain.
            scr._set_spinner_visible(False)
            await pilot.pause()
            after = scr._render_status_strip()
            assert "thinking" not in after
            assert "Generate" in after

    @pytest.mark.asyncio
    async def test_marker_pulse_animates(self):
        app = _Host()
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(8):
                await pilot.pause()
            scr = _chat(app)
            scr._show_pipeline_strip(scr._current_mode_spec())
            scr._set_spinner_visible(True)
            await pilot.pause()
            assert scr._status_anim_timer is not None  # timer running
            # Phase 0: the diamond runs STRAIGHT (frame 0) while the dot runs
            # REVERSED (last frame) — the desync the user asked for.
            first = scr._render_status_strip()
            d0_shape, d0_colour = scr._DIAMOND_FRAMES[0]
            dot_last_shape, dot_last_colour = scr._DOT_FRAMES[-1]
            assert d0_colour in first and f"]{d0_shape}[" in first
            assert dot_last_colour in first and f"]{dot_last_shape}[" in first
            # Advancing the phase steps the diamond FORWARD (frame 1) and the
            # dot BACKWARD (frame -2) — mirrored motion, colour shifting each
            # tick for smoothness.
            scr._tick_status_anim()
            assert scr._status_phase == 1
            second = scr._render_status_strip()
            assert scr._DIAMOND_FRAMES[1][1] in second
            assert scr._DOT_FRAMES[-2][1] in second
            # Colour morphs every tick (smooth), for both markers.
            assert scr._DIAMOND_FRAMES[0][1] != scr._DIAMOND_FRAMES[1][1]
            assert scr._DOT_FRAMES[-1][1] != scr._DOT_FRAMES[-2][1]
            # Over the cycle the dot form genuinely changes shape too.
            assert len({f[0] for f in scr._DOT_FRAMES}) >= 3

    @pytest.mark.asyncio
    async def test_active_run_stage_animates_like_generate(self):
        """Regression: once Generate is done and Run is the active stage,
        the Run marker must morph (◇→◈→◆) while thinking — not sit on the
        static `○` ask glyph (which only colour-pulsed and read as frozen)."""
        from care.screens.chat import Stage, StageOutcome

        app = _Host()
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(8):
                await pilot.pause()
            scr = _chat(app)
            scr._show_pipeline_strip(scr._current_mode_spec())
            scr._set_spinner_visible(True)
            scr._update_pipeline_stage(Stage.GENERATE, StageOutcome.DONE)
            await pilot.pause()
            # While thinking, Run shows the morphing diamond, no "?".
            seen_shapes = set()
            for phase in range(len(scr._DIAMOND_FRAMES)):
                scr._status_phase = phase
                rendered = str(
                    scr.query_one("#chat-pipeline-strip", Static).render()
                )
                # Grab the marker immediately before "Run".
                assert "Run" in rendered
                marker = rendered.split("Run")[0].strip().split()[-1]
                seen_shapes.add(marker)
            assert seen_shapes & {"◇", "◈", "◆"}  # diamond morph present
            assert "○" not in seen_shapes
            # The Run "?" is dropped while actively running.
            scr._status_phase = 0
            assert "Run?" not in str(
                scr.query_one("#chat-pipeline-strip", Static).render()
            )
            # When NOT thinking (awaiting confirm), Run reverts to "○ Run?".
            scr._set_spinner_visible(False)
            await pilot.pause()
            assert "○ Run?" in str(
                scr.query_one("#chat-pipeline-strip", Static).render()
            )

    @pytest.mark.asyncio
    async def test_standalone_thinking_without_pipeline(self):
        app = _Host()
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(8):
                await pilot.pause()
            scr = _chat(app)
            # No pipeline — just a worker thinking.
            scr._set_spinner_visible(True)
            await pilot.pause()
            text = scr._render_status_strip()
            assert "thinking" in text
            assert "→" not in text  # no pipeline cells
            assert scr.query_one("#chat-pipeline-strip", Static).display is True

    @pytest.mark.asyncio
    async def test_strip_persists_after_turn_until_finish(self):
        """The strip stays on screen after an interactive turn (showing
        ``◆ Generate → ◆ Run``) until the user clicks Finish."""
        from care.screens.chat import Stage, StageOutcome

        app = _Host()
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(8):
                await pilot.pause()
            scr = _chat(app)
            scr._begin_chain_session(
                chain_dict={"steps": [{"name": "a"}]},
                display_name="X", task="do it",
            )
            scr._show_pipeline_strip(scr._current_mode_spec())
            scr._set_spinner_visible(True)
            scr._update_pipeline_stage(Stage.GENERATE, StageOutcome.DONE)
            scr._update_pipeline_stage(Stage.RUN, StageOutcome.DONE)
            # Worker finishes → thinking off, but the session is live.
            scr._set_spinner_visible(False)
            await pilot.pause()
            strip = scr.query_one("#chat-pipeline-strip", Static)
            assert strip.display is True
            rendered = str(strip.render())
            assert "Generate" in rendered and "Run" in rendered
            assert "thinking" not in rendered  # tail dropped, cells remain
            # Finish collapses it.
            scr._finish_chain_session()
            await pilot.pause()
            assert strip.display is False

    @pytest.mark.asyncio
    async def test_strip_cleared_on_reset(self):
        """A /new /clear (mode flip) reset collapses the lingering strip."""
        from care.screens.chat import Stage, StageOutcome

        app = _Host()
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(8):
                await pilot.pause()
            scr = _chat(app)
            scr._begin_chain_session(
                chain_dict={"steps": [{"name": "a"}]},
                display_name="X", task="t",
            )
            scr._show_pipeline_strip(scr._current_mode_spec())
            scr._update_pipeline_stage(Stage.GENERATE, StageOutcome.DONE)
            scr._set_spinner_visible(False)
            await pilot.pause()
            strip = scr.query_one("#chat-pipeline-strip", Static)
            assert strip.display is True
            scr._reset_interactive_history()
            await pilot.pause()
            assert strip.display is False

    @pytest.mark.asyncio
    async def test_hide_collapses_when_idle(self):
        app = _Host()
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(8):
                await pilot.pause()
            scr = _chat(app)
            scr._show_pipeline_strip(scr._current_mode_spec())
            scr._set_spinner_visible(True)
            await pilot.pause()
            scr._set_spinner_visible(False)
            scr._hide_pipeline_strip()
            await pilot.pause()
            strip = scr.query_one("#chat-pipeline-strip", Static)
            assert strip.display is False
            assert scr._status_anim_timer is None  # timer stopped
