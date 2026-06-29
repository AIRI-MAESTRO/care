"""Estimate USD cost for an LLM call from a small in-tree pricing table.

Best-effort: covers the families CARE users actually pick (Anthropic
Claude, OpenAI GPT-4o + o1, OpenRouter slugs that route to either).
Unknown models return ``None`` so callers can omit the cost segment
from the UI rather than print a fake ``$0.00``.

The pricing table is intentionally hand-maintained — keeping it
in-tree avoids an extra HTTP probe at chat-iteration time, and the
table only needs to track the handful of models real users select.
Patches welcome as new models ship.

Values are per *million* tokens in USD (matching OpenAI / Anthropic
pricing pages). The match is case-insensitive substring against the
model id; the first matching entry in :data:`_PRICING_TABLE` wins, so
order by specificity (e.g. ``claude-3-5-haiku`` before
``claude-3-5-sonnet`` so a Haiku id doesn't accidentally match the
Sonnet row).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-tokens USD cost for one model family.

    ``input_per_million`` is what the provider charges for prompt
    tokens; ``output_per_million`` for completion tokens. Both are
    in USD-per-1e6-tokens — the same units OpenAI / Anthropic /
    OpenRouter all surface in their public pricing pages.
    """

    input_per_million: float
    output_per_million: float


# Ordered most-specific → least-specific so a substring like
# ``claude-3.5-haiku`` doesn't fall through and match the ``claude``
# bucket. Patterns are lowercase; lookup also lowercases the model id
# before matching.
_PRICING_TABLE: tuple[tuple[str, ModelPricing], ...] = (
    # --- Anthropic Claude family ---
    ("claude-opus-4", ModelPricing(15.0, 75.0)),
    ("claude-opus-4.1", ModelPricing(15.0, 75.0)),
    ("claude-opus-4-7", ModelPricing(15.0, 75.0)),
    ("claude-sonnet-4", ModelPricing(3.0, 15.0)),
    ("claude-sonnet-4-6", ModelPricing(3.0, 15.0)),
    ("claude-3.7-sonnet", ModelPricing(3.0, 15.0)),
    ("claude-3-7-sonnet", ModelPricing(3.0, 15.0)),
    ("claude-3.5-haiku", ModelPricing(0.80, 4.0)),
    ("claude-3-5-haiku", ModelPricing(0.80, 4.0)),
    ("claude-haiku-4.5", ModelPricing(1.0, 5.0)),
    ("claude-haiku-4-5", ModelPricing(1.0, 5.0)),
    ("claude-3.5-sonnet", ModelPricing(3.0, 15.0)),
    ("claude-3-5-sonnet", ModelPricing(3.0, 15.0)),
    ("claude-3-opus", ModelPricing(15.0, 75.0)),
    ("claude-3-haiku", ModelPricing(0.25, 1.25)),
    # --- OpenAI GPT family ---
    ("gpt-4o-mini", ModelPricing(0.15, 0.60)),
    ("gpt-4o", ModelPricing(2.50, 10.0)),
    ("gpt-4-turbo", ModelPricing(10.0, 30.0)),
    ("gpt-4.1-mini", ModelPricing(0.40, 1.60)),
    ("gpt-4.1", ModelPricing(2.0, 8.0)),
    ("o1-mini", ModelPricing(3.0, 12.0)),
    ("o1-preview", ModelPricing(15.0, 60.0)),
    ("o1", ModelPricing(15.0, 60.0)),
    ("gpt-3.5-turbo", ModelPricing(0.50, 1.50)),
    # --- Google Gemini (via OpenRouter / direct) ---
    ("gemini-2.0-flash", ModelPricing(0.10, 0.40)),
    ("gemini-1.5-pro", ModelPricing(1.25, 5.0)),
    ("gemini-1.5-flash", ModelPricing(0.075, 0.30)),
)


def estimate_cost(
    model: str | None,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Return the USD cost of an LLM call, or ``None`` when the
    model isn't in the pricing table.

    Match strategy: lowercase substring against
    :data:`_PRICING_TABLE` patterns, first hit wins. Pricing rows
    are ordered most-specific first so e.g. ``claude-3-5-haiku``
    matches its own row rather than falling through to
    ``claude-3-5-sonnet``.

    Returns ``None`` (not ``0.0``) for unknown models so callers
    can render an empty cost segment instead of a misleading
    "$0.00 — must be free!" signal.

    Negative token counts coerce to ``0`` so a counter reset
    mid-iteration never produces a negative cost. Zero-token
    inputs return ``0.0`` (the call was free *for this model*),
    distinct from ``None`` (the model is unknown).
    """
    if not model:
        return None
    needle = model.strip().lower()
    pricing: ModelPricing | None = None
    for pattern, entry in _PRICING_TABLE:
        if pattern in needle:
            pricing = entry
            break
    if pricing is None:
        return None
    prompt = max(0, int(prompt_tokens))
    completion = max(0, int(completion_tokens))
    cost = (
        (prompt * pricing.input_per_million)
        + (completion * pricing.output_per_million)
    ) / 1_000_000.0
    return round(cost, 6)


def format_cost(cost: float | None) -> str:
    """Project a USD cost into the chat-footer segment string.

    Uses six decimals when the value is below a cent (so
    sub-penny costs don't collapse to ``$0.00``), four for
    fractional-cent costs, and two for amounts ≥ $1. ``None``
    returns empty string so the caller can join footer
    segments without a guard.
    """
    if cost is None:
        return ""
    if cost <= 0:
        return "$0.00"
    if cost < 0.01:
        return f"${cost:.6f}"
    if cost < 1.0:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


__all__ = ["ModelPricing", "estimate_cost", "format_cost"]
