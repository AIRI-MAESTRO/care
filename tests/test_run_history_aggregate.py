"""Tests for the §4 P2 per-chain run-history aggregate +
formatters (`care.runtime.local_run_history`)."""

from __future__ import annotations

import time

from care.runtime.local_run_history import (
    ChainRunStats,
    LocalRunEntry,
    format_mean_cost,
    format_recency,
    summarise_runs_by_chain,
)


def _run(
    *,
    run_id: str = "r-1",
    chain_id: str = "agent-A",
    started_at: float = 1_000.0,
    status: str = "success",
    duration: float | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost: float | None = None,
) -> LocalRunEntry:
    return LocalRunEntry(
        run_id=run_id,
        chain_id=chain_id,
        started_at=started_at,
        duration_seconds=duration,
        status=status,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
    )


# ---------------------------------------------------------------------------
# summarise_runs_by_chain (pure)
# ---------------------------------------------------------------------------


class TestSummariseRunsByChain:
    def test_empty_input(self):
        assert summarise_runs_by_chain([]) == {}

    def test_skips_empty_chain_id(self):
        runs = [_run(chain_id="", run_id="r1")]
        assert summarise_runs_by_chain(runs) == {}

    def test_buckets_by_chain_id(self):
        runs = [
            _run(run_id="r1", chain_id="A"),
            _run(run_id="r2", chain_id="A"),
            _run(run_id="r3", chain_id="B"),
        ]
        stats = summarise_runs_by_chain(runs)
        assert set(stats) == {"A", "B"}
        assert stats["A"].run_count == 2
        assert stats["B"].run_count == 1

    def test_success_rate(self):
        runs = [
            _run(run_id=f"r{i}", chain_id="A", status="success")
            for i in range(8)
        ] + [
            _run(run_id=f"f{i}", chain_id="A", status="failure")
            for i in range(2)
        ]
        stats = summarise_runs_by_chain(runs)["A"]
        assert stats.run_count == 10
        assert stats.success_count == 8
        assert stats.success_rate == 0.8

    def test_last_run_picks_max_started_at(self):
        runs = [
            _run(run_id="r1", chain_id="A", started_at=100.0),
            _run(run_id="r2", chain_id="A", started_at=500.0),
            _run(run_id="r3", chain_id="A", started_at=300.0),
        ]
        stats = summarise_runs_by_chain(runs)["A"]
        assert stats.last_run_at == 500.0

    def test_means_only_over_present_metrics(self):
        # Some runs have duration/tokens/cost, others don't —
        # the mean is computed only over the rows that DO.
        runs = [
            _run(run_id="r1", chain_id="A", duration=1.0,
                 tokens_in=100, tokens_out=50, cost=0.10),
            _run(run_id="r2", chain_id="A", duration=2.0,
                 tokens_in=200, tokens_out=100, cost=0.20),
            _run(run_id="r3", chain_id="A"),  # no metrics
        ]
        stats = summarise_runs_by_chain(runs)["A"]
        assert stats.mean_duration_seconds == 1.5
        # Mean total tokens: (150 + 300) / 2 = 225
        assert stats.mean_tokens == 225.0
        assert abs(stats.mean_cost_usd - 0.15) < 1e-9

    def test_no_metrics_means_are_none(self):
        runs = [_run(run_id="r1", chain_id="A")]
        stats = summarise_runs_by_chain(runs)["A"]
        assert stats.mean_duration_seconds is None
        assert stats.mean_tokens is None
        assert stats.mean_cost_usd is None


# ---------------------------------------------------------------------------
# format_recency (pure)
# ---------------------------------------------------------------------------


class TestFormatRecency:
    def test_none_stats_returns_empty(self):
        assert format_recency(None) == ""

    def test_zero_run_count_returns_empty(self):
        assert format_recency(ChainRunStats(chain_id="A")) == ""

    def test_renders_age_and_rate(self):
        now = time.time()
        stats = ChainRunStats(
            chain_id="A",
            run_count=5,
            success_count=4,
            last_run_at=now - 3700,  # ~1h ago
        )
        out = format_recency(stats)
        # Format: "<age> · <rate>/<count>"
        assert "·" in out
        assert "0.80" in out
        assert "/5" in out

    def test_just_now(self):
        stats = ChainRunStats(
            chain_id="A", run_count=1, success_count=1,
            last_run_at=time.time() - 10,
        )
        assert "just now" in format_recency(stats)

    def test_days_ago(self):
        stats = ChainRunStats(
            chain_id="A", run_count=1, success_count=1,
            last_run_at=time.time() - 86400 * 3,
        )
        assert "3d ago" in format_recency(stats)


# ---------------------------------------------------------------------------
# format_mean_cost (pure)
# ---------------------------------------------------------------------------


class TestFormatMeanCost:
    def test_none_stats(self):
        assert format_mean_cost(None) == "—"

    def test_none_cost(self):
        assert format_mean_cost(
            ChainRunStats(chain_id="A", run_count=1),
        ) == "—"

    def test_zero_cost_explicit(self):
        assert format_mean_cost(
            ChainRunStats(
                chain_id="A", run_count=1, mean_cost_usd=0.0,
            ),
        ) == "$0.00"

    def test_sub_cent(self):
        assert format_mean_cost(
            ChainRunStats(
                chain_id="A", run_count=1, mean_cost_usd=0.003,
            ),
        ) == "<$0.01"

    def test_two_decimal_cents(self):
        out = format_mean_cost(
            ChainRunStats(
                chain_id="A", run_count=1, mean_cost_usd=0.42,
            ),
        )
        assert out == "$0.42"

    def test_dollars(self):
        out = format_mean_cost(
            ChainRunStats(
                chain_id="A", run_count=1, mean_cost_usd=12.345,
            ),
        )
        assert out == "$12.35"


# ---------------------------------------------------------------------------
# ChainRunStats.success_rate
# ---------------------------------------------------------------------------


class TestSuccessRateProperty:
    def test_no_runs(self):
        assert ChainRunStats(chain_id="A").success_rate is None

    def test_perfect(self):
        stats = ChainRunStats(
            chain_id="A", run_count=5, success_count=5,
        )
        assert stats.success_rate == 1.0

    def test_partial(self):
        stats = ChainRunStats(
            chain_id="A", run_count=4, success_count=3,
        )
        assert stats.success_rate == 0.75
