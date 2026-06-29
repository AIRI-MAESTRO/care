"""Tests for `care.runtime.pareto_plot` (TODO §5 P1)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from care.runtime.pareto_plot import render_pareto_scatter


@dataclass
class _Ind:
    individual_id: str
    objectives: tuple[tuple[str, float], ...] = ()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_input(self):
        assert render_pareto_scatter([]) == ""

    def test_no_objectives_returns_empty(self):
        rows = [_Ind("a"), _Ind("b")]
        assert render_pareto_scatter(rows) == ""

    def test_single_objective_returns_empty(self):
        rows = [
            _Ind("a", (("fitness", 0.5),)),
            _Ind("b", (("fitness", 0.7),)),
        ]
        assert render_pareto_scatter(rows) == ""

    def test_one_point_with_two_objectives_returns_empty(
        self,
    ):
        # Need ≥ 2 well-formed points before a scatter is
        # meaningful — single-point input degenerates.
        rows = [
            _Ind("a", (("acc", 0.5), ("lat", 0.2))),
        ]
        assert render_pareto_scatter(rows) == ""

    def test_dict_objectives_accepted(self):
        rows = [
            _Ind("a", {"acc": 0.7, "lat": 0.3}),
            _Ind("b", {"acc": 0.8, "lat": 0.5}),
        ]
        out = render_pareto_scatter(rows, use_plotext=False)
        assert "acc" in out and "lat" in out

    def test_malformed_objectives_dropped(self):
        rows = [
            _Ind("a", (("acc", "not-a-number"), ("lat", 0.3))),
            _Ind("b", (("acc", 0.8), ("lat", 0.5))),
            _Ind("c", (("acc", 0.6), ("lat", 0.4))),
        ]
        out = render_pareto_scatter(rows, use_plotext=False)
        # `a` has unparseable acc; the remaining two well-formed
        # points still render.
        assert "ind=a" not in out
        assert "b" in out and "c" in out


# ---------------------------------------------------------------------------
# Fallback (no plotext)
# ---------------------------------------------------------------------------


class TestFallback:
    def test_fallback_header_lists_objective_keys(self):
        rows = [
            _Ind("a", (("acc", 0.7), ("lat", 0.3))),
            _Ind("b", (("acc", 0.8), ("lat", 0.5))),
        ]
        out = render_pareto_scatter(rows, use_plotext=False)
        assert out.startswith("pareto: acc × lat")

    def test_fallback_marks_front_individuals(self):
        rows = [
            _Ind("a", (("acc", 0.7), ("lat", 0.3))),
            _Ind("b", (("acc", 0.5), ("lat", 0.4))),
        ]
        out = render_pareto_scatter(
            rows, front_ids=("a",), use_plotext=False,
        )
        # `a` is on the front; `b` isn't.
        a_line = next(
            line for line in out.splitlines() if line.startswith("a")
        )
        b_line = next(
            line for line in out.splitlines() if line.startswith("b")
        )
        assert "★ front" in a_line
        assert "★ front" not in b_line

    def test_fallback_caps_long_population(self):
        rows = [
            _Ind(f"ind-{i:02d}", (("acc", i / 50), ("lat", i / 100)))
            for i in range(20)
        ]
        out = render_pareto_scatter(rows, use_plotext=False)
        # _FALLBACK_ROW_CAP is 12; the cap-line should appear.
        assert "showing 12/20" in out
        # Cap-line mentions the install hint.
        assert "care[plots]" in out

    def test_fallback_explicit_axes_kwargs(self):
        rows = [
            _Ind("a", {"acc": 0.7, "lat": 0.3, "cost": 0.1}),
            _Ind("b", {"acc": 0.5, "lat": 0.4, "cost": 0.2}),
        ]
        out = render_pareto_scatter(
            rows, x_obj="cost", y_obj="acc", use_plotext=False,
        )
        assert "pareto: cost × acc" in out


# ---------------------------------------------------------------------------
# plotext path (optional)
# ---------------------------------------------------------------------------


class TestPlotextPath:
    def test_force_plotext_when_missing_raises(self, monkeypatch):
        # Simulate plotext absent: force the import to fail.
        import builtins

        original_import = builtins.__import__

        def _patched_import(name, *args, **kwargs):
            if name == "plotext":
                raise ImportError("simulated missing plotext")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _patched_import)

        rows = [
            _Ind("a", (("acc", 0.7), ("lat", 0.3))),
            _Ind("b", (("acc", 0.8), ("lat", 0.5))),
        ]
        with pytest.raises(RuntimeError):
            render_pareto_scatter(rows, use_plotext=True)

    def test_plotext_path_smoke(self):
        plotext = pytest.importorskip("plotext")
        del plotext  # keep linter happy; import gates the test
        rows = [
            _Ind("a", (("acc", 0.7), ("lat", 0.3))),
            _Ind("b", (("acc", 0.8), ("lat", 0.5))),
            _Ind("c", (("acc", 0.9), ("lat", 0.6))),
        ]
        out = render_pareto_scatter(
            rows, front_ids=("a", "c"), use_plotext=True,
            width=40, height=10,
        )
        # plotext returns a multi-line ASCII chart — just
        # assert it's non-empty + has multiple rows.
        assert out
        assert "\n" in out
