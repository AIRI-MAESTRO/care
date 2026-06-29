"""Tests for the evolution launch modal's budget preview (P4)."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from care.screens.evolution_launch import (
    EvolutionLaunchModal,
    EvolutionLaunchSpec,
    estimate_evolution_budget,
)


class TestEstimateBudget:
    def test_evaluations_and_tokens(self):
        evals, tokens = estimate_evolution_budget(10, 8)
        assert evals == 80
        assert tokens == 80 * 1500

    def test_zero_and_negative_clamp(self):
        assert estimate_evolution_budget(0, 8) == (0, 0)
        assert estimate_evolution_budget(-5, -5) == (0, 0)


class _Host(App):
    def __init__(self, modal: EvolutionLaunchModal):
        super().__init__()
        self._modal = modal

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self._modal)


class TestBudgetPreview:
    @pytest.mark.asyncio
    async def test_preview_renders_and_updates(self):
        modal = EvolutionLaunchModal(base_chain_id="chain-1")
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            pane = modal.query_one("#launch-budget", Static)
            # Defaults 10 × 8 = 80 evaluations.
            assert "80" in str(pane.render())
            # Bump iterations → 20 × 8 = 160 evaluations.
            modal.query_one("#launch-max-iter", Input).value = "20"
            await pilot.pause()
            assert "160" in str(modal.query_one("#launch-budget", Static).render())


class TestMutationMaxTokens:
    @pytest.mark.asyncio
    async def test_collect_spec_and_screen_kwargs(self):
        modal = EvolutionLaunchModal(base_chain_id="chain-1")
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal.query_one("#launch-mutation-max-tokens", Input).value = "16384"
            spec = modal.collect_spec()
            assert spec.mutation_max_tokens == 16384
            assert spec.to_screen_kwargs()["mutation_max_tokens"] == 16384

    def test_spec_to_screen_kwargs_omits_none(self):
        spec = EvolutionLaunchSpec(base_chain_id="c1", mutation_max_tokens=None)
        assert "mutation_max_tokens" not in spec.to_screen_kwargs()
