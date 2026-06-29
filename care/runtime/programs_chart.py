"""Text-mode "pie chart" for the EvolutionScreen's Programs tab.

The web UI renders a matplotlib donut showing
``programs_valid_count`` vs ``programs_invalid_count``. The TUI
can't draw raster images cheaply, so we ship a unicode-block
proportional bar + a stat block instead. Mirrors the web UI's
colour palette (`#13c1ac` valid / `#a8afba` invalid) by routing
the block characters through Rich-style markup.

Pure function — no Textual / Rich imports here so the helper
is unit-testable without the renderer dependency.
"""

from __future__ import annotations


def _format_pct(part: int, total: int) -> str:
    if total <= 0:
        return "—"
    pct = (part / total) * 100.0
    return f"{pct:5.1f}%"


def render_programs_pie(
    valid: int,
    invalid: int,
    *,
    width: int = 48,
) -> str:
    """Render a unicode "proportional bar" for valid / invalid programs.

    Args:
        valid: Latest ``programs_valid_count`` from the runner.
            Pass ``-1`` (or any negative number) when the metric
            hasn't been reported yet.
        invalid: Latest ``programs_invalid_count``.
        width: Width of the proportional bar in cells (defaults
            to 48, which fits comfortably in the screen's
            ``#evolution-programs`` pane).

    Returns:
        Multi-line plot text. Empty string when neither count has
        landed yet so the caller can render a "(no program data
        yet)" placeholder above it.
    """
    have_valid = valid >= 0
    have_invalid = invalid >= 0
    if not (have_valid or have_invalid):
        return ""

    v = max(valid, 0) if have_valid else 0
    i = max(invalid, 0) if have_invalid else 0
    total = v + i
    if total <= 0:
        return ""

    # Proportional bar — full blocks for valid, lighter shade
    # for invalid so colour-blind users can still see the split.
    bar_width = max(8, int(width))
    v_cells = round(bar_width * v / total)
    v_cells = min(v_cells, bar_width)
    i_cells = bar_width - v_cells

    valid_bar = "[green]" + ("█" * v_cells) + "[/]"
    invalid_bar = "[grey50]" + ("▓" * i_cells) + "[/]"
    bar_line = valid_bar + invalid_bar

    rows = [
        f"Total programs: {total}",
        "",
        bar_line,
        "",
        f"[green]█[/] Valid    {v:5d}  ({_format_pct(v, total)})",
        f"[grey50]▓[/] Invalid  {i:5d}  ({_format_pct(i, total)})",
    ]
    return "\n".join(rows)


_TREND_GLYPHS = "▁▂▃▄▅▆▇█"


def render_programs_trend(curve, *, width: int = 40) -> str:
    """Render a one-line valid-program sparkline over generations.

    ``curve`` is an iterable of ``(generation, valid, invalid)`` rows
    (``EvolutionProgressTracker.programs_curve()``). Returns an empty
    string when there are fewer than two generations of data — a single
    point isn't a trend and the bar chart already shows the latest
    count. Pure-stdlib unicode sparkline so the default install renders
    it without an extra dep."""
    rows = [
        r
        for r in curve
        if isinstance(r, (tuple, list))
        and len(r) >= 2
        and isinstance(r[1], int)
        and r[1] >= 0
    ]
    if len(rows) < 2:
        return ""
    tail = rows[-max(1, int(width)):]
    ys = [int(r[1]) for r in tail]
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        glyph = _TREND_GLYPHS[len(_TREND_GLYPHS) // 2]
        spark = glyph * len(ys)
    else:
        span = ymax - ymin
        top = len(_TREND_GLYPHS) - 1
        spark = "".join(
            _TREND_GLYPHS[min(top, int(((y - ymin) / span) * top))] for y in ys
        )
    first_gen = tail[0][0]
    last_gen = tail[-1][0]
    return (
        f"valid-program trend (gen {first_gen}..{last_gen}):\n"
        f"[green]{spark}[/green]  (min {ymin} → max {ymax})"
    )
