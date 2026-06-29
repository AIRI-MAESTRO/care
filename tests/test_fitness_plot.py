"""Tests for `care.runtime.fitness_plot` (TODO §5 P0)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from care.runtime.fitness_plot import render_fitness_plot


@dataclass
class _Stat:
    """Lightweight stand-in for `GenerationStat`. The renderer
    duck-types on `generation` + `best_fitness`."""

    generation: int
    best_fitness: float


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_records_returns_empty_string(self) -> None:
        assert render_fitness_plot([], use_plotext=False) == ""

    def test_only_placeholder_minus_inf_returns_empty(self) -> None:
        records = [
            _Stat(0, float("-inf")),
            _Stat(1, float("-inf")),
        ]
        assert render_fitness_plot(records, use_plotext=False) == ""

    def test_skips_minus_inf_placeholders(self) -> None:
        # The real generations should drive the plot; the
        # `-inf` placeholder generation 1 is from
        # `generation_started` with no winners yet.
        records = [
            _Stat(0, 0.5),
            _Stat(1, float("-inf")),
            _Stat(2, 0.8),
        ]
        text = render_fitness_plot(records, use_plotext=False)
        assert "min 0.500" in text
        assert "max 0.800" in text
        # Two real generations contribute.
        assert "(2 generations" in text


# ---------------------------------------------------------------------------
# Sparkline fallback
# ---------------------------------------------------------------------------


class TestSparklineFallback:
    def test_renders_three_lines(self) -> None:
        records = [_Stat(i, i * 0.1) for i in range(10)]
        text = render_fitness_plot(records, use_plotext=False)
        lines = text.splitlines()
        assert len(lines) == 3
        assert lines[0] == "best fitness:"
        # Sparkline glyphs from the unicode block-fill set.
        assert any(g in lines[1] for g in "▁▂▃▄▅▆▇█")
        assert lines[2].startswith("gen 0..9")

    def test_includes_min_max_and_latest(self) -> None:
        records = [_Stat(0, 0.20), _Stat(1, 0.85), _Stat(2, 0.60)]
        text = render_fitness_plot(records, use_plotext=False)
        assert "min 0.200" in text
        assert "max 0.850" in text
        assert "latest: 0.600" in text
        assert "(3 generations" in text

    def test_handles_flat_curve_without_div_by_zero(self) -> None:
        # Every fitness identical → ymax == ymin. Renderer
        # must not divide by zero; produces a flat middle-
        # height bar instead.
        records = [_Stat(i, 0.42) for i in range(5)]
        text = render_fitness_plot(records, use_plotext=False)
        # Mid-height glyph (index 4 of 8) — but any
        # non-empty sparkline is acceptable.
        assert "0.420" in text
        spark_line = text.splitlines()[1]
        # Five generations → five glyphs.
        spark = spark_line.split()[0]
        assert len(spark) == 5

    def test_tail_window_when_width_smaller_than_records(self) -> None:
        # 20 generations but width=8 → only the tail eight
        # populate the sparkline.
        records = [_Stat(i, i * 0.05) for i in range(20)]
        text = render_fitness_plot(records, width=8, use_plotext=False)
        spark = text.splitlines()[1].split()[0]
        assert len(spark) == 8
        # But the totals line reports all 20.
        assert "(20 generations" in text

    def test_renders_when_use_plotext_default_and_plotext_missing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force the optional import to fail so the default
        # branch falls back to the sparkline.
        import builtins
        real_import = builtins.__import__

        def _no_plotext(name, *args, **kwargs):
            if name == "plotext":
                raise ImportError("forced for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_plotext)
        records = [_Stat(0, 0.5), _Stat(1, 0.7)]
        text = render_fitness_plot(records)
        assert text.startswith("best fitness:")


# ---------------------------------------------------------------------------
# Plotext branch (only when the extra is installed)
# ---------------------------------------------------------------------------


class TestPlotextBranch:
    def test_use_plotext_true_raises_when_missing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins
        real_import = builtins.__import__

        def _no_plotext(name, *args, **kwargs):
            if name == "plotext":
                raise ImportError("forced")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_plotext)
        with pytest.raises(RuntimeError, match="plotext"):
            render_fitness_plot(
                [_Stat(0, 0.5), _Stat(1, 0.7)],
                use_plotext=True,
            )

    def test_use_plotext_when_installed_renders_non_empty(self) -> None:
        pytest.importorskip("plotext")
        records = [_Stat(i, i * 0.1) for i in range(5)]
        text = render_fitness_plot(records, use_plotext=True)
        assert text  # Non-empty plotext build output.
