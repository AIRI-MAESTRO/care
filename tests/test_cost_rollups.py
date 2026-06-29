"""Tests for `care.runtime.cost_rollups` (TODO §6 P2)."""

from __future__ import annotations

from care.runtime.cost_rollups import (
    OverallTotals,
    compute_overall,
    compute_per_chain,
    compute_per_mode,
    compute_per_provider,
)
from care.runtime.local_run_history import LocalRunEntry


def _entry(**kwargs) -> LocalRunEntry:
    defaults = {"run_id": "r"}
    defaults.update(kwargs)
    return LocalRunEntry(**defaults)


# ---------------------------------------------------------------------------
# Overall
# ---------------------------------------------------------------------------


class TestComputeOverall:
    def test_empty_returns_zero_totals(self):
        out = compute_overall([])
        assert out == OverallTotals()
        assert out.success_rate is None

    def test_counts_runs_and_outcomes(self):
        entries = [
            _entry(status="success", tokens_in=10, tokens_out=20),
            _entry(status="success", tokens_in=5),
            _entry(status="failure", error="boom"),
        ]
        out = compute_overall(entries)
        assert out.runs == 3
        assert out.successful_runs == 2
        assert out.failed_runs == 1
        assert out.tokens_in == 15
        assert out.tokens_out == 20
        assert out.tokens_total == 35

    def test_sums_costs_and_durations(self):
        entries = [
            _entry(cost_usd=0.5, duration_seconds=10.0),
            _entry(cost_usd=0.25, duration_seconds=5.0),
            _entry(),  # missing cost / duration → contribute 0
        ]
        out = compute_overall(entries)
        assert abs(out.cost_usd - 0.75) < 1e-9
        assert out.total_duration_seconds == 15.0

    def test_success_rate(self):
        out = compute_overall([
            _entry(status="success"),
            _entry(status="success"),
            _entry(status="success"),
            _entry(status="failure"),
        ])
        assert out.success_rate == 0.75


# ---------------------------------------------------------------------------
# Per-X rollups
# ---------------------------------------------------------------------------


class TestPerProvider:
    def test_groups_by_provider(self):
        entries = [
            _entry(provider="openai", cost_usd=0.10),
            _entry(provider="anthropic", cost_usd=0.25),
            _entry(provider="openai", cost_usd=0.05),
        ]
        rows = compute_per_provider(entries)
        # Anthropic spent more → first.
        assert rows[0].key == "anthropic"
        assert rows[0].cost_usd == 0.25
        assert rows[1].key == "openai"
        assert rows[1].runs == 2
        assert abs(rows[1].cost_usd - 0.15) < 1e-9

    def test_missing_provider_collapses_to_unknown(self):
        entries = [
            _entry(provider="", cost_usd=0.10),
            _entry(provider="", cost_usd=0.20),
        ]
        rows = compute_per_provider(entries)
        assert len(rows) == 1
        assert rows[0].key == "(unknown)"
        assert rows[0].runs == 2
        assert abs(rows[0].cost_usd - 0.30) < 1e-9


class TestPerChain:
    def test_label_prefers_chain_name_falls_back_to_id(self):
        entries = [
            _entry(
                chain_id="c1",
                chain_name="Forecaster",
                cost_usd=0.5,
            ),
            _entry(chain_id="c2", cost_usd=0.25),
            _entry(chain_id="", cost_usd=0.1),
        ]
        rows = compute_per_chain(entries)
        # Sorted by cost desc → forecaster, c2, unknown
        assert rows[0].label == "Forecaster"
        assert rows[0].key == "c1"
        assert rows[1].label == "c2"
        assert rows[2].key == "(unknown)"


class TestPerMode:
    def test_groups_by_mode(self):
        entries = [
            _entry(mode="ad_hoc", tokens_in=10),
            _entry(mode="ad_hoc", tokens_in=20),
            _entry(mode="production", tokens_in=5),
            _entry(mode="evolution", cost_usd=2.0),
        ]
        rows = compute_per_mode(entries)
        # Evolution: $2 wins.
        assert rows[0].key == "evolution"
        # ad_hoc has zero cost but 30 tokens — comes after
        # evolution but before production (which has zero on
        # both axes; ties broken by key alphabetically).
        ad_hoc = next(r for r in rows if r.key == "ad_hoc")
        assert ad_hoc.tokens_in == 30
        production = next(r for r in rows if r.key == "production")
        assert production.runs == 1


class TestRollupRow:
    def test_tokens_total(self):
        entries = [
            _entry(provider="p", tokens_in=100, tokens_out=50),
        ]
        rows = compute_per_provider(entries)
        assert rows[0].tokens_total == 150
