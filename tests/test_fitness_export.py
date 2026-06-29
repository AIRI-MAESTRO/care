"""Tests for the fitness-curve CSV/JSON export helpers."""

from __future__ import annotations

import json

from care.evolution_session import GenerationStat
from care.runtime.fitness_export import (
    fitness_curve_csv,
    fitness_curve_json,
    fitness_curve_rows,
)


def _records():
    return [
        GenerationStat(generation=0, best_fitness=0.2, mean_fitness=0.1),
        GenerationStat(generation=1, best_fitness=float("-inf")),  # placeholder
        GenerationStat(generation=2, best_fitness=0.5, mean_fitness=None),
    ]


class TestRows:
    def test_drops_placeholder_and_sorts(self):
        rows = fitness_curve_rows(_records())
        assert [r["generation"] for r in rows] == [0, 2]
        assert rows[0]["best_fitness"] == 0.2
        assert rows[0]["mean_fitness"] == 0.1
        assert rows[1]["mean_fitness"] is None


class TestCsv:
    def test_header_and_rows(self):
        csv = fitness_curve_csv(_records())
        lines = csv.strip().splitlines()
        assert lines[0] == "generation,best_fitness,mean_fitness"
        assert len(lines) == 3  # header + 2 real rows
        assert lines[1].startswith("0,")
        # mean omitted for the None row.
        assert lines[2].endswith(",")


class TestJson:
    def test_roundtrips(self):
        data = json.loads(fitness_curve_json(_records()))
        assert [d["generation"] for d in data] == [0, 2]
        assert data[0]["best_fitness"] == 0.2
