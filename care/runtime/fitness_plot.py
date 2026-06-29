"""Text-mode fitness-vs-generation plot (TODO §5 P0).

The EvolutionScreen's status pane shows a one-line summary
of the best-fitness per generation. This module renders a
larger plot that gives the user a feel for the trajectory
of an in-flight evolution.

Two render modes:

* ``plotext`` (extras-gated `care[plots]`). Produces a small
  ASCII line chart with axes + tick labels. The most useful
  representation when ``plotext>=5.2`` is installed.
* Unicode-block sparkline fallback. Pure-stdlib so the
  default `uvx care` install still gets a meaningful plot.

Renderer contract: input is the same
``tuple[GenerationStat, ...]`` :class:`EvolutionProgressTracker.fitness_curve`
returns. Placeholder ``-inf`` records (from
``generation_started`` events with no winners yet) are
skipped — the plot only shows generations that produced a
real fitness value. Output is a plain string the screen
can drop into a ``Static`` widget.
"""

from __future__ import annotations

import logging
from typing import Iterable

_log = logging.getLogger("care.runtime.fitness_plot")


def _real_records(records: Iterable) -> list:
    """Drop placeholder `-inf` records that come from
    `generation_started` events with no winners yet."""
    out = []
    for stat in records:
        fitness = getattr(stat, "best_fitness", None)
        if fitness is None:
            continue
        if fitness == float("-inf"):
            continue
        out.append(stat)
    return out


def render_fitness_plot(
    records,
    *,
    width: int = 60,
    height: int = 10,
    use_plotext: bool | None = None,
) -> str:
    """Render the fitness curve as a multi-line text plot.

    Args:
        records: Iterable of ``GenerationStat``-like objects
            (anything with ``generation`` + ``best_fitness``
            attributes). The :class:`EvolutionProgressTracker`'s
            ``fitness_curve()`` returns the right shape.
        width: Target plot width in characters (used by both
            backends).
        height: Target plot height in rows.
        use_plotext: When ``None`` (default), use plotext when
            it's importable, else fall back. ``True`` forces
            plotext (raises if missing — for tests). ``False``
            forces the unicode sparkline (for tests that want
            deterministic output without the optional dep).

    Returns:
        Plot text. Empty string when ``records`` produces no
        non-placeholder entries — the caller decides whether
        to render a header.
    """
    real = _real_records(records)
    if not real:
        return ""

    if use_plotext is False:
        return _render_sparkline(real, width=width)

    plot = _try_plotext(real, width=width, height=height)
    if plot is not None:
        return plot
    if use_plotext is True:
        raise RuntimeError(
            "render_fitness_plot(use_plotext=True) requested "
            "but plotext failed or isn't installed"
        )
    return _render_sparkline(real, width=width)


def _try_plotext(
    real: list, *, width: int, height: int,
) -> str | None:
    """Best-effort plotext render. Returns ``None`` when the
    optional dep isn't installed (caller falls back to the
    sparkline).

    Any other rendering exception logs at DEBUG + falls
    back too — a charting failure should NOT bring the
    EvolutionScreen down."""
    try:
        import plotext as plt  # type: ignore
    except ImportError:
        return None
    try:
        plt.clear_figure()
        xs = [s.generation for s in real]
        ys = [s.best_fitness for s in real]
        plt.theme("clear")
        plt.plot(xs, ys, marker="braille")
        plt.title("best fitness vs generation")
        plt.xlabel("gen")
        plt.ylabel("fitness")
        plt.plotsize(width, height)
        text = plt.build()
        plt.clear_figure()
        return text
    except Exception as exc:  # noqa: BLE001
        _log.debug("plotext render failed: %s", exc)
        return None


_SPARK_GLYPHS = "▁▂▃▄▅▆▇█"
"""Unicode block-fill glyphs from 1/8 to 8/8 height — the
de-facto sparkline alphabet. ASCII fallback kicks in when
the terminal can't render these (we don't auto-detect; the
default plotext path covers most terminals)."""


def _render_sparkline(real: list, *, width: int) -> str:
    """Pure-stdlib fallback when `plotext` isn't installed.

    Format::

        best fitness:
        ▁▂▃▅▇█  (min 0.42 → max 0.91)
        gen 0..7  (8 generations, latest: 0.87)

    The sparkline width is bounded by ``width`` and by the
    number of records — we sample the tail if there are
    more generations than the available columns so the
    latest progress is always visible.
    """
    if not real:
        return ""
    xs = [s.generation for s in real]
    ys = [s.best_fitness for s in real]

    # Bound the sparkline width by the available columns —
    # the right edge stays attached to the latest generation
    # so the user sees recent progress.
    max_cells = max(1, min(width, len(real)))
    tail_y = ys[-max_cells:]

    ymin = min(tail_y)
    ymax = max(tail_y)
    if ymax == ymin:
        # Avoid div-by-zero on a flat fitness curve.
        glyph = _SPARK_GLYPHS[len(_SPARK_GLYPHS) // 2]
        spark = glyph * len(tail_y)
    else:
        span = ymax - ymin
        ladder_max = len(_SPARK_GLYPHS) - 1
        spark = "".join(
            _SPARK_GLYPHS[
                min(
                    ladder_max,
                    int(((y - ymin) / span) * ladder_max),
                )
            ]
            for y in tail_y
        )

    lines = [
        "best fitness:",
        f"{spark}  (min {ymin:.3f} → max {ymax:.3f})",
        (
            f"gen {xs[0]}..{xs[-1]}  ({len(real)} generations, "
            f"latest: {ys[-1]:.3f})"
        ),
    ]
    return "\n".join(lines)


__all__ = ["render_fitness_plot"]
