"""Tests for `care.runtime.pricing` — the small in-tree pricing
table that backs the Phase 8 P1 #8 iteration footer cost segment.
"""

from __future__ import annotations

import pytest

from care.runtime.pricing import (
    ModelPricing,
    estimate_cost,
    format_cost,
)


class TestEstimateCost:
    def test_known_anthropic_model(self):
        """Claude 3.5 Sonnet pricing: $3 / $15 per 1M tokens.
        1000 prompt + 2000 completion → $0.003 + $0.030 = $0.033."""
        cost = estimate_cost("anthropic/claude-3.5-sonnet", 1000, 2000)
        assert cost == pytest.approx(0.033, rel=1e-6)

    def test_known_openai_model(self):
        """GPT-4o-mini: $0.15 / $0.60 per 1M tokens.
        10_000 prompt + 5_000 completion = $0.0015 + $0.003 = $0.0045."""
        cost = estimate_cost("openai/gpt-4o-mini", 10_000, 5_000)
        assert cost == pytest.approx(0.0045, rel=1e-6)

    def test_haiku_does_not_match_sonnet(self):
        """Pricing order matters — `claude-3.5-haiku` must NOT
        fall through to the `claude-3.5-sonnet` row. Locks the
        ordering invariant in `_PRICING_TABLE`."""
        haiku = estimate_cost("anthropic/claude-3.5-haiku", 1_000_000, 0)
        sonnet = estimate_cost("anthropic/claude-3.5-sonnet", 1_000_000, 0)
        assert haiku is not None
        assert sonnet is not None
        assert haiku < sonnet, (
            f"Haiku ({haiku}) should be cheaper than Sonnet ({sonnet})"
        )

    def test_unknown_model_returns_none(self):
        """Unknown model = None (not 0.0) so the caller can omit
        the cost segment from the UI rather than print a
        misleading '$0.00 — must be free' signal."""
        assert estimate_cost("totally-fake-model-7b", 100, 100) is None

    def test_empty_or_none_model_returns_none(self):
        assert estimate_cost(None, 100, 100) is None
        assert estimate_cost("", 100, 100) is None
        assert estimate_cost("   ", 100, 100) is None

    def test_zero_tokens_returns_zero_not_none(self):
        """Zero-token call for a *known* model is genuinely $0;
        distinct from None which means 'unknown model'."""
        assert estimate_cost("gpt-4o-mini", 0, 0) == 0.0

    def test_case_insensitive_match(self):
        """Model ids come from configs in mixed case; the
        pricing table is lowercase. Lookup must lowercase."""
        normal = estimate_cost("claude-3.5-sonnet", 1000, 1000)
        upper = estimate_cost("CLAUDE-3.5-SONNET", 1000, 1000)
        assert normal is not None
        assert upper == normal

    def test_substring_match_with_provider_prefix(self):
        """OpenRouter / Anthropic SDK both emit slugs like
        ``anthropic/claude-3.5-sonnet``. The pricing-table
        pattern is just ``claude-3.5-sonnet`` — substring
        match must find it inside the longer id."""
        assert estimate_cost("openrouter/anthropic/claude-3.5-sonnet", 1000, 0) == pytest.approx(
            3.0 * 1000 / 1_000_000, rel=1e-6,
        )

    def test_negative_tokens_coerced_to_zero(self):
        """Counter resets mid-iteration produce negative deltas;
        cost must coerce to 0, never go negative."""
        cost = estimate_cost("gpt-4o-mini", -500, -300)
        assert cost == 0.0

    def test_estimate_uses_provider_specific_rates(self):
        """1M prompt tokens at Claude 3.5 Sonnet = $3.00 exactly."""
        cost = estimate_cost("claude-3.5-sonnet", 1_000_000, 0)
        assert cost == pytest.approx(3.0, rel=1e-6)
        cost_out = estimate_cost("claude-3.5-sonnet", 0, 1_000_000)
        assert cost_out == pytest.approx(15.0, rel=1e-6)


class TestFormatCost:
    def test_none_returns_empty(self):
        """None → empty so callers can string-join without a guard."""
        assert format_cost(None) == ""

    def test_zero_renders_as_dollar_zero_two(self):
        assert format_cost(0.0) == "$0.00"
        assert format_cost(-1.0) == "$0.00"

    def test_sub_cent_uses_six_decimals(self):
        assert format_cost(0.001234) == "$0.001234"

    def test_fractional_cent_uses_four_decimals(self):
        assert format_cost(0.0432) == "$0.0432"

    def test_dollar_or_more_uses_two_decimals(self):
        assert format_cost(1.2345) == "$1.23"
        assert format_cost(42.999) == "$43.00"


class TestModelPricingDataclass:
    """The dataclass is frozen so values can't drift mid-test;
    locks the immutability invariant."""

    def test_frozen(self):
        p = ModelPricing(1.0, 2.0)
        with pytest.raises(Exception):
            p.input_per_million = 99.0  # type: ignore[misc]
