"""Platform-aligned validation / metric options for chain evolution.

Mirrors ``ChainValidationCriteria`` in gigaevo-platform
(``master_api/src/models/experiment.py``) and the Web UI
dropdowns in ``create_chain_experiment.py``.
"""

from __future__ import annotations

from typing import Any, Literal

ValidationType = Literal["Binary (0/1)", "Continuous (0..1)"]
ContinuousMetric = Literal["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "BLEU"]
BinaryMethod = Literal["equality", "substring", "regexp"]

VALIDATION_TYPES: tuple[ValidationType, ...] = (
    "Continuous (0..1)",
    "Binary (0/1)",
)
CONTINUOUS_METRICS: tuple[ContinuousMetric, ...] = (
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "BERTScore",
    "BLEU",
)
BINARY_METHODS: tuple[BinaryMethod, ...] = (
    "equality",
    "substring",
    "regexp",
)

DEFAULT_VALIDATION_TYPE: ValidationType = "Continuous (0..1)"
DEFAULT_CONTINUOUS_METRIC: ContinuousMetric = "ROUGE-L"
DEFAULT_BINARY_METHOD: BinaryMethod = "equality"
DEFAULT_TARGET_COLUMN = "expected"


def normalize_validation_type(value: str | None) -> ValidationType:
    if value in VALIDATION_TYPES:
        return value  # type: ignore[return-value]
    return DEFAULT_VALIDATION_TYPE


def normalize_continuous_metric(value: str | None) -> ContinuousMetric:
    if value in CONTINUOUS_METRICS:
        return value  # type: ignore[return-value]
    return DEFAULT_CONTINUOUS_METRIC


def normalize_binary_method(value: str | None) -> BinaryMethod:
    if value in BINARY_METHODS:
        return value  # type: ignore[return-value]
    return DEFAULT_BINARY_METHOD


def build_chain_validation_criteria(
    *,
    validation_type: str | None = None,
    continuous_metric: str | None = None,
    binary_method: str | None = None,
    regexp_pattern: str = "",
) -> dict[str, Any]:
    """Build the ``validation_criteria`` object for ``POST /experiments/chains``.

    The Platform's ``regexp_pattern`` field extracts the chain answer from
    raw output — it is **not** a free-form rubric. User rubric text is stored
    separately on the experiment ``description`` by :mod:`care.platform`.
    """
    vtype = normalize_validation_type(validation_type)
    body: dict[str, Any] = {
        "validation_type": vtype,
        "regexp_pattern": regexp_pattern or "",
    }
    if vtype == "Binary (0/1)":
        body["binary_method"] = normalize_binary_method(binary_method)
    else:
        body["continuous_metric"] = normalize_continuous_metric(continuous_metric)
    return body


__all__ = [
    "BINARY_METHODS",
    "CONTINUOUS_METRICS",
    "DEFAULT_BINARY_METHOD",
    "DEFAULT_CONTINUOUS_METRIC",
    "DEFAULT_TARGET_COLUMN",
    "DEFAULT_VALIDATION_TYPE",
    "VALIDATION_TYPES",
    "BinaryMethod",
    "ContinuousMetric",
    "ValidationType",
    "build_chain_validation_criteria",
    "normalize_binary_method",
    "normalize_continuous_metric",
    "normalize_validation_type",
]
