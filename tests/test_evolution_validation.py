"""Tests for Platform validation/metric mapping (MAESTRO → gigaevo-platform)."""

from __future__ import annotations

import pytest

from care.platform import _build_chain_validation_criteria
from care.runtime.evolution_validation import (
    build_chain_validation_criteria,
    normalize_continuous_metric,
)


class TestBuildChainValidationCriteria:
    def test_default_continuous_rouge_l(self) -> None:
        payload = build_chain_validation_criteria()
        assert payload == {
            "validation_type": "Continuous (0..1)",
            "continuous_metric": "ROUGE-L",
            "regexp_pattern": "",
        }

    def test_platform_wrapper_delegates(self) -> None:
        payload = _build_chain_validation_criteria(
            validation_type="Binary (0/1)",
            binary_method="substring",
        )
        assert payload["validation_type"] == "Binary (0/1)"
        assert payload["binary_method"] == "substring"

    def test_rouge_1_forwarded(self) -> None:
        payload = build_chain_validation_criteria(continuous_metric="ROUGE-1")
        assert payload["continuous_metric"] == "ROUGE-1"

    def test_invalid_metric_falls_back(self) -> None:
        assert normalize_continuous_metric("nope") == "ROUGE-L"

    @pytest.mark.parametrize(
        "metric",
        ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "BLEU"],
    )
    def test_all_platform_metrics(self, metric: str) -> None:
        payload = build_chain_validation_criteria(continuous_metric=metric)
        assert payload["continuous_metric"] == metric
