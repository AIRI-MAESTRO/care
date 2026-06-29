"""Pilot tests for InspectionScreen intermediate-artifacts pane.

Wires `care.project_intermediate_artifacts` into the
`InspectionScreen`'s `Artifacts` tab as collapsible panes
(§1.2 [DONE — data layer] → fully DONE).
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from care.intermediate_artifacts import (
    IntermediateArtifact,
    IntermediateArtifactsView,
)
from care.screens.inspection import InspectionScreen


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _InspHost(App):
    memory = None

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(InspectionScreen("agent-x"))


def _screen(app: App) -> InspectionScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, InspectionScreen)
    return s


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


class TestRender:
    @pytest.mark.asyncio
    async def test_empty_view_renders_placeholder(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.record_intermediate_artifacts(
                IntermediateArtifactsView(),
            )
            await pilot.pause()
            assert screen.intermediate_artifacts.is_empty is True

    @pytest.mark.asyncio
    async def test_view_with_artifacts_renders_collapsibles(self):
        view = IntermediateArtifactsView(
            artifacts=(
                IntermediateArtifact(
                    stage="domain_analysis",
                    header="Domain analysis",
                    summary="domain=weather, type=research",
                    body="domain: weather\ntype: research",
                    raw={"domain": "weather"},
                ),
                IntermediateArtifact(
                    stage="step_plan",
                    header="Step plan",
                    summary="3 steps planned",
                    body="step 1: fetch",
                    raw={"steps": ["fetch"]},
                ),
            ),
        )
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.record_intermediate_artifacts(view)
            await pilot.pause()
            await pilot.pause()
            assert screen.intermediate_artifacts is view
            panes = list(
                screen.query("#inspection-artifacts-body Collapsible"),
            )
            assert len(panes) == 2

    @pytest.mark.asyncio
    async def test_record_failure_is_silent(self):
        class _Boom:
            @property
            def intermediate_artifacts(self):
                raise RuntimeError("kaboom")

        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            # No exception even with a misbehaving source.
            screen.record_intermediate_artifacts(_Boom())
            await pilot.pause()
            assert screen.intermediate_artifacts.is_empty is True

    @pytest.mark.asyncio
    async def test_record_accepts_dict_source(self):
        # `project_intermediate_artifacts` accepts dicts too —
        # the screen should pass through.
        source = {
            "intermediate_artifacts": {
                "domain_analysis": {
                    "domain": "weather",
                    "task_type": "research",
                    "complexity": "low",
                },
            },
        }
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _screen(app)
            screen.record_intermediate_artifacts(source)
            await pilot.pause()
            assert screen.intermediate_artifacts.is_empty is False
            stages = screen.intermediate_artifacts.stages()
            assert "domain_analysis" in stages


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


class TestArtifactId:
    def test_alnum_stage_passes_through(self):
        from care.screens.inspection import _artifact_id

        assert _artifact_id("step_plan") == "step_plan"

    def test_special_chars_sanitised(self):
        from care.screens.inspection import _artifact_id

        result = _artifact_id("foo:bar/baz")
        assert ":" not in result
        assert "/" not in result
