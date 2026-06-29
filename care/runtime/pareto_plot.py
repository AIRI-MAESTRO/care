"""Text-mode Pareto scatter (TODO §5 P1).

When an evolution run has ≥ 2 objectives, the EvolutionScreen
already exposes a one-line Pareto-front summary plus a fitness-
vs-generation plot. This module adds a small 2D scatter of
``objective_0 × objective_1`` so the user can eyeball the
front shape at a glance.

Two render modes (mirroring :mod:`care.runtime.fitness_plot`):

* ``plotext`` (extras-gated `care[plots]`). All individuals
  land as small dots; Pareto-front individuals overlay as a
  distinct marker. Axes labelled with the two objective keys.
* Pure-stdlib textual summary. Reads as a header + a table of
  ``id  x=…  y=…  ★ front`` rows, capped to ``_FALLBACK_ROW_CAP``
  so a large population doesn't blow the pane height.

Inputs are duck-typed (`individual_id`, `objectives` mapping or
`Iterable[(key, value)]`) so the screen passes
:class:`care.screens.evolution.EvolutionIndividual` rows directly
without an adapter pass.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

_log = logging.getLogger("care.runtime.pareto_plot")

_FALLBACK_ROW_CAP = 12
"""Cap the fallback render to this many rows so the pane height
stays predictable on large populations."""


def _objectives_dict(individual: Any) -> dict[str, float]:
    """Coerce an individual's `objectives` to a key→float dict.

    Accepts: a `dict`, a `tuple[(key, value), ...]`, or any
    iterable of `(key, value)` pairs (the shape
    `EvolutionIndividual.objectives` carries). Returns an empty
    dict when the field is missing / unparseable so the caller
    can short-circuit on `len(...) < 2`.
    """
    raw = getattr(individual, "objectives", None)
    if raw is None:
        return {}
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        items: Iterable = raw.items()
    else:
        items = raw
    for item in items:
        try:
            k, v = item
        except Exception:
            continue
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _common_axes(
    individuals: list[Any],
) -> tuple[str, str] | None:
    """Pick the first two objective keys shared by at least one
    individual. Preserves insertion order from the first
    individual that exposes ≥ 2 objectives so a stable axis
    layout survives across refreshes."""
    for ind in individuals:
        objs = _objectives_dict(ind)
        if len(objs) >= 2:
            keys = list(objs.keys())
            return keys[0], keys[1]
    return None


def render_pareto_scatter(
    individuals: Iterable[Any],
    *,
    front_ids: Iterable[str] = (),
    x_obj: str | None = None,
    y_obj: str | None = None,
    width: int = 60,
    height: int = 12,
    use_plotext: bool | None = None,
) -> str:
    """Render the Pareto scatter as a multi-line text plot.

    Args:
        individuals: Iterable of individuals (each carrying
            ``individual_id`` + ``objectives``). The
            EvolutionScreen passes `self.run.individuals`
            directly.
        front_ids: Iterable of `individual_id`s that sit on
            the current Pareto front. Front individuals are
            marked distinctly in the plotext path + tagged
            ``★ front`` in the textual fallback.
        x_obj / y_obj: Explicit objective keys for the two
            axes. When unset, the function picks the first two
            keys from whichever individual exposes ≥ 2
            objectives (insertion order preserved).
        width / height: Target plot dimensions (chars / rows).
        use_plotext: `None` (auto), `True` (force plotext —
            raises on import / render failure, useful for
            tests), `False` (force textual fallback).

    Returns:
        Plot text. Empty string when fewer than two
        individuals carry the required two objectives — the
        caller decides whether to render a header.
    """
    rows = [ind for ind in individuals if ind is not None]
    if not rows:
        return ""

    if x_obj is None or y_obj is None:
        axes = _common_axes(rows)
        if axes is None:
            return ""
        x_obj, y_obj = axes

    points: list[tuple[str, float, float]] = []
    for ind in rows:
        objs = _objectives_dict(ind)
        if x_obj not in objs or y_obj not in objs:
            continue
        points.append(
            (str(getattr(ind, "individual_id", "?")),
             objs[x_obj], objs[y_obj]),
        )
    if len(points) < 2:
        # Need at least two well-formed points before the
        # scatter is meaningful.
        return ""

    front_set = {str(i) for i in front_ids}

    if use_plotext is False:
        return _render_fallback(
            points, front_set, x_obj=x_obj, y_obj=y_obj,
        )

    plot = _try_plotext(
        points, front_set,
        x_obj=x_obj, y_obj=y_obj,
        width=width, height=height,
    )
    if plot is not None:
        return plot
    if use_plotext is True:
        raise RuntimeError(
            "render_pareto_scatter(use_plotext=True) "
            "requested but plotext failed or isn't installed"
        )
    return _render_fallback(
        points, front_set, x_obj=x_obj, y_obj=y_obj,
    )


def _try_plotext(
    points: list[tuple[str, float, float]],
    front_set: set[str],
    *,
    x_obj: str,
    y_obj: str,
    width: int,
    height: int,
) -> str | None:
    """Best-effort plotext scatter. Returns ``None`` when
    plotext isn't installed; logs at DEBUG + falls back on any
    other rendering exception."""
    try:
        import plotext as plt  # type: ignore
    except ImportError:
        return None
    try:
        plt.clear_figure()
        # All individuals first (small dot).
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        plt.theme("clear")
        plt.scatter(xs, ys, marker="dot")
        # Overlay the front with a distinct marker so a busy
        # population doesn't obscure it.
        front = [p for p in points if p[0] in front_set]
        if front:
            fxs = [p[1] for p in front]
            fys = [p[2] for p in front]
            plt.scatter(fxs, fys, marker="hd")
        plt.title("pareto front")
        plt.xlabel(x_obj)
        plt.ylabel(y_obj)
        plt.plotsize(width, height)
        text = plt.build()
        plt.clear_figure()
        return text
    except Exception as exc:  # noqa: BLE001
        _log.debug("plotext pareto render failed: %s", exc)
        return None


def _render_fallback(
    points: list[tuple[str, float, float]],
    front_set: set[str],
    *,
    x_obj: str,
    y_obj: str,
) -> str:
    """Pure-stdlib fallback when `plotext` isn't installed.

    Format::

        pareto: <x_obj> × <y_obj>
        ind-A  x=0.870  y=0.420  ★ front
        ind-B  x=0.810  y=0.510  ★ front
        ind-C  x=0.760  y=0.380
        …
        (showing 3/12 — install care[plots] for a 2D scatter)

    Front individuals sort first; within each group, sort by
    `x` descending so the visually-leftmost-on-a-real-scatter
    front individual lands at the top.
    """
    sorted_points = sorted(
        points,
        key=lambda p: (0 if p[0] in front_set else 1, -p[1]),
    )
    visible = sorted_points[:_FALLBACK_ROW_CAP]
    lines = [f"pareto: {x_obj} × {y_obj}"]
    for ind_id, x, y in visible:
        tag = "  ★ front" if ind_id in front_set else ""
        lines.append(
            f"{ind_id}  {x_obj}={x:.3f}  {y_obj}={y:.3f}{tag}"
        )
    if len(sorted_points) > _FALLBACK_ROW_CAP:
        lines.append(
            f"(showing {_FALLBACK_ROW_CAP}/{len(sorted_points)} — "
            f"install care[plots] for a 2D scatter)"
        )
    return "\n".join(lines)


__all__ = ["render_pareto_scatter"]
