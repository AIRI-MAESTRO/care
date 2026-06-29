"""Dashboard live-metric columns (Current / Valid / Invalid)."""

from __future__ import annotations

from care.screens.evolution_dashboard import (
    _COLUMNS,
    _format_count,
    _localized_columns,
    parse_evolution_run_row,
)


def test_localized_columns_match_stable_columns():
    # Header count must match the stable `_COLUMNS` keys so DataTable
    # row tuples line up.
    assert len(_localized_columns()) == len(_COLUMNS)
    assert "Current" in _COLUMNS and "Valid" in _COLUMNS and "Invalid" in _COLUMNS


def test_parse_reads_top_level_metrics():
    row = parse_evolution_run_row(
        {
            "id": "exp_1",
            "status": "running",
            "current_fitness": 0.4,
            "programs_valid": 7,
            "programs_invalid": 2,
        }
    )
    assert row.current_fitness == 0.4
    assert row.programs_valid == 7
    assert row.programs_invalid == 2


def test_parse_reads_nested_metrics_blob():
    row = parse_evolution_run_row(
        {
            "id": "exp_2",
            "status": "running",
            "metrics": {
                "current_fitness": 0.55,
                "programs_valid": 9,
                "programs_invalid": 1,
            },
        }
    )
    assert row.current_fitness == 0.55
    assert row.programs_valid == 9
    assert row.programs_invalid == 1


def test_parse_absent_metrics_are_none():
    row = parse_evolution_run_row({"id": "exp_3", "status": "queued"})
    assert row.current_fitness is None
    assert row.programs_valid is None
    assert row.programs_invalid is None


def test_format_count():
    assert _format_count(0) == "0"
    assert _format_count(5) == "5"
    assert _format_count(None) == "—"
    assert _format_count(-1) == "—"
