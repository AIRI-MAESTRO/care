"""EvolutionCompareModal — side-by-side fitness curves for two
evolution runs (TODO §5 P1).

Pushed from the §5 P0 :class:`EvolutionDashboard` after the
user picks exactly two rows with the multi-select binding
(`space`) and presses `c` to compare. The modal fetches each
run's state via the platform facade
(``platform.get_evolution(run_id)``), projects per-generation
fitness records, and renders the two fitness sparklines /
plotext charts side by side via the existing
:func:`care.runtime.fitness_plot.render_fitness_plot`.

Failure modes degrade gracefully:

* Missing platform facade → error line per-side ("(no platform
  configured)").
* `get_evolution` raises → error line carrying the exception
  message.
* State payload has no fitness history → blank plot pane +
  ``"(no fitness data yet)"`` placeholder under the run header.

Pre-loaded mode: tests + future drill-from-dashboard paths can
construct with explicit ``left_state`` / ``right_state`` payloads
to skip the fetch.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from care.runtime.fitness_plot import render_fitness_plot
from care.runtime.i18n import t

_log = logging.getLogger("care.screen.evolution_compare")


@dataclass(frozen=True)
class _FitnessRecord:
    """Duck-typed projection that satisfies
    :func:`render_fitness_plot`."""

    generation: int
    best_fitness: float


@dataclass(frozen=True)
class _CompareSummary:
    """Headline metrics shown beneath each run's fitness chart."""

    generation: int | None = None
    best_fitness: float | None = None
    current_fitness: float | None = None
    programs_valid: int | None = None
    programs_invalid: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def has_any(self) -> bool:
        return any(
            v is not None
            for v in (
                self.generation,
                self.best_fitness,
                self.current_fitness,
                self.programs_valid,
                self.programs_invalid,
                self.total_tokens,
                self.cost_usd,
            )
        )


def _metrics_view(state: Any) -> dict[str, Any]:
    """Merge the metric dicts an evolution state can carry.

    ``get_evolution`` returns top-level ``best_fitness``/``generation``
    plus a nested ``_raw.results.metrics`` blob (fitness_history,
    programs_valid/invalid, current_fitness). Flatten both so the
    summary extractor reads from one place; top-level wins on conflict.
    """
    merged: dict[str, Any] = {}
    if not isinstance(state, Mapping):
        return merged
    raw = state.get("_raw")
    if isinstance(raw, Mapping):
        results = raw.get("results")
        if isinstance(results, Mapping):
            rm = results.get("metrics")
            if isinstance(rm, Mapping):
                merged.update(rm)
    top_metrics = state.get("metrics")
    if isinstance(top_metrics, Mapping):
        merged.update(top_metrics)
    return merged


def extract_compare_summary(state: Any) -> _CompareSummary:
    """Project headline metrics (fitness / programs / cost) out of an
    evolution state payload for the side-by-side summary.

    Defensive like :func:`extract_fitness_records`: reads from the
    top-level keys and the nested ``_raw.results.metrics`` blob,
    tolerating absent fields so a partial payload still renders what it
    has."""
    if not isinstance(state, Mapping):
        return _CompareSummary()
    metrics = _metrics_view(state)

    def _pick(*keys: str):
        for src in (state, metrics):
            for key in keys:
                if isinstance(src, Mapping) and src.get(key) is not None:
                    return src.get(key)
        return None

    def _as_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            iv = int(value)
            return iv if iv >= 0 else None
        return None

    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    return _CompareSummary(
        generation=_as_int(_pick("generation", "gen")),
        best_fitness=_as_float(_pick("best_fitness", "best")),
        current_fitness=_as_float(_pick("current_fitness")),
        programs_valid=_as_int(_pick("programs_valid", "programs_valid_count")),
        programs_invalid=_as_int(
            _pick("programs_invalid", "programs_invalid_count")
        ),
        total_tokens=_as_int(_pick("total_tokens", "tokens")),
        cost_usd=_as_float(_pick("cost_usd", "total_cost_usd", "cost")),
    )


def format_compare_summary_lines(summary: _CompareSummary) -> list[str]:
    """Render a compact metric block for the compare pane footer."""
    if not summary.has_any:
        return []
    lines: list[str] = []
    top: list[str] = []
    if summary.generation is not None:
        top.append(f"gen {summary.generation}")
    if summary.best_fitness is not None:
        top.append(f"best [green]{summary.best_fitness:.4f}[/green]")
    if summary.current_fitness is not None:
        top.append(f"current {summary.current_fitness:.4f}")
    if top:
        lines.append(" · ".join(top))
    pv, pi = summary.programs_valid, summary.programs_invalid
    if pv is not None or pi is not None:
        v = pv if pv is not None else 0
        i = pi if pi is not None else 0
        total = v + i
        pct = f" ({v / total * 100:.0f}% valid)" if total else ""
        lines.append(f"programs: [green]{v}[/green] valid / {i} invalid{pct}")
    cost_bits: list[str] = []
    if summary.total_tokens is not None:
        cost_bits.append(f"{summary.total_tokens:,} tokens")
    if summary.cost_usd is not None and summary.cost_usd > 0:
        cost_bits.append(f"${summary.cost_usd:,.2f}")
    if cost_bits:
        lines.append("cost: " + " · ".join(cost_bits))
    return lines


def extract_fitness_records(state: Any) -> tuple[_FitnessRecord, ...]:
    """Project per-generation fitness records out of an
    evolution state payload.

    Accepts ``state["fitness_history"]``,
    ``state["generations"]``, or the nested
    ``state["progress"]["fitness_history"]`` location, and
    tolerates ``generation``/``gen`` + ``best_fitness``/
    ``fitness``/``best`` key variants per entry. Malformed
    entries silently drop so a partial payload still renders.

    Mirrors the (private) `ChatScreen._extract_fitness_records`
    helper used by the inline `/evolution <id>` snapshot — filed
    as a follow-up to dedupe across the two surfaces once the
    Platform's SSE schema is finalized.
    """
    if not isinstance(state, Mapping):
        return ()
    raw: Any = None
    for path in (
        "fitness_history",
        "generations",
        "progress.fitness_history",
    ):
        cursor: Any = state
        for segment in path.split("."):
            if isinstance(cursor, Mapping) and segment in cursor:
                cursor = cursor[segment]
            else:
                cursor = None
                break
        if isinstance(cursor, list) and cursor:
            raw = cursor
            break
    if raw is None:
        # Chain experiments (``exp_*``) carry the curve nested under
        # ``_raw.results.metrics.fitness_history`` (where ``get_evolution``
        # puts it), not at the top level — consult the merged metric view
        # so the curve renders for chain runs, matching where
        # ``extract_compare_summary`` already reads from.
        nested = _metrics_view(state).get("fitness_history")
        if isinstance(nested, list) and nested:
            raw = nested
    if raw is None:
        return ()
    out: list[_FitnessRecord] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        gen = entry.get("generation")
        if gen is None:
            gen = entry.get("gen")
        fitness = entry.get("best_fitness")
        if fitness is None:
            fitness = entry.get("fitness")
        if fitness is None:
            fitness = entry.get("best")
        if gen is None or fitness is None:
            continue
        try:
            out.append(_FitnessRecord(
                generation=int(gen),
                best_fitness=float(fitness),
            ))
        except (TypeError, ValueError):
            continue
    return tuple(out)


class EvolutionCompareModal(ModalScreen[None]):
    """Side-by-side fitness curves for two evolution runs.

    Construct with the two `run_id` strings the dashboard
    multi-selected. Pre-load `left_state` / `right_state` to
    skip the platform fetch — useful for tests + future
    cached-state replay.
    """

    DEFAULT_CSS = """
    EvolutionCompareModal {
        align: center middle;
    }
    EvolutionCompareModal #evo-compare-box {
        width: 130;
        max-width: 95%;
        height: 30;
        max-height: 90%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    EvolutionCompareModal #evo-compare-title {
        text-style: bold;
        padding-bottom: 1;
    }
    EvolutionCompareModal #evo-compare-body {
        height: 1fr;
    }
    EvolutionCompareModal .compare-side {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        background: $panel;
        border: tall $primary 30%;
    }
    EvolutionCompareModal .compare-side-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    EvolutionCompareModal #evo-compare-actions {
        height: 3;
        align-horizontal: right;
        margin-top: 1;
    }
    EvolutionCompareModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("r", "refetch", "Refresh", show=True),
    ]

    _SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    """Braille spinner frames for the in-flight fetch animation."""

    _FETCH_TIMEOUT_SECONDS = 15.0
    """Per-side fetch deadline so a hung Platform can't freeze "Loading"."""

    def __init__(
        self,
        *,
        left_run_id: str,
        right_run_id: str,
        platform: Any = None,
        left_state: Any = None,
        right_state: Any = None,
    ) -> None:
        super().__init__()
        if not left_run_id or not right_run_id:
            raise ValueError(
                "both left_run_id and right_run_id must be "
                "non-empty"
            )
        self.left_run_id = left_run_id
        self.right_run_id = right_run_id
        self._platform = platform
        self._left_state = left_state
        self._right_state = right_state
        self._left_error: str | None = None
        self._right_error: str | None = None
        self.action_log: list[tuple[str, str]] = []
        # Spinner state for the in-flight fetch animation.
        self._pending_sides: set[str] = set()
        self._spinner_idx: int = 0
        self._spinner_timer: Any = None

    def compose(self) -> ComposeResult:
        with Vertical(id="evo-compare-box"):
            yield Label(
                t("evolutionCompare.title"),
                id="evo-compare-title",
            )
            with Horizontal(id="evo-compare-body"):
                with Vertical(
                    id="evo-compare-left",
                    classes="compare-side",
                ):
                    yield Label(
                        t("evolutionCompare.left", id=self.left_run_id),
                        classes="compare-side-title",
                    )
                    yield Static(
                        t("common.loading"),
                        id="evo-compare-left-text",
                    )
                with Vertical(
                    id="evo-compare-right",
                    classes="compare-side",
                ):
                    yield Label(
                        t("evolutionCompare.right", id=self.right_run_id),
                        classes="compare-side-title",
                    )
                    yield Static(
                        t("common.loading"),
                        id="evo-compare-right-text",
                    )
            with Horizontal(id="evo-compare-actions"):
                yield Button(t("common.close"), id="evo-compare-btn-close")

    def on_mount(self) -> None:
        if (
            self._left_state is not None
            and self._right_state is not None
        ):
            # Pre-loaded path — render synchronously and skip
            # the fetch worker entirely.
            self._render_side("left", self._left_state)
            self._render_side("right", self._right_state)
            return
        self._start_spinner(("left", "right"))
        self.run_worker(
            self._fetch_and_render(),
            name="evo_compare_fetch",
            group="evo_compare",
            exclusive=True,
            exit_on_error=False,
        )

    # ------------------------------------------------------------------
    # Loading spinner
    # ------------------------------------------------------------------

    def _motion_enabled(self) -> bool:
        """True when the app permits the loading spinner to animate.
        Reduced-motion (``animation_level == "none"``) freezes it on the
        first frame — the "Loading" indicator still paints, it just doesn't
        spin (and the interval is never scheduled, so no idle repaints)."""
        try:
            return getattr(self.app, "animation_level", "none") != "none"
        except Exception:
            return False

    def _start_spinner(self, sides: tuple[str, ...]) -> None:
        """Animate the loading panes for ``sides`` until each renders.

        Under reduced motion the interval is skipped entirely; the single
        `_tick_spinner()` below still paints one static frame so the panes
        read as "Loading" without repainting on a timer."""
        self._pending_sides = set(sides)
        self._spinner_idx = 0
        if (
            self._spinner_timer is None
            and self.is_mounted
            and self._motion_enabled()
        ):
            try:
                self._spinner_timer = self.set_interval(0.12, self._tick_spinner)
            except Exception:
                self._spinner_timer = None
        self._tick_spinner()

    def _tick_spinner(self) -> None:
        if not self._pending_sides:
            self._stop_spinner()
            return
        frame = self._SPINNER_FRAMES[self._spinner_idx % len(self._SPINNER_FRAMES)]
        self._spinner_idx += 1
        for side in tuple(self._pending_sides):
            try:
                pane = self.query_one(f"#evo-compare-{side}-text", Static)
                pane.update(f"{frame} {t('common.loading')}")
            except Exception:
                pass

    def _finish_side(self, side: str) -> None:
        """Mark a side done; stop the spinner once both have landed."""
        self._pending_sides.discard(side)
        if not self._pending_sides:
            self._stop_spinner()

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            try:
                self._spinner_timer.stop()
            except Exception:
                pass
            self._spinner_timer = None

    async def _fetch_and_render(self) -> None:
        for side, run_id in (
            ("left", self.left_run_id),
            ("right", self.right_run_id),
        ):
            state, error = await self._fetch_state(run_id)
            if error:
                self._set_error(side, error)
                continue
            if side == "left":
                self._left_state = state
            else:
                self._right_state = state
            self._render_side(side, state)

    async def _fetch_state(
        self, run_id: str,
    ) -> tuple[Any, str | None]:
        if self._platform is None:
            return None, t("evolutionCompare.noPlatform")
        getter = getattr(self._platform, "get_evolution", None)
        if not callable(getter):
            return None, (
                t("evolutionCompare.noGetEvolution")
            )
        try:
            state = await asyncio.wait_for(
                asyncio.to_thread(getter, run_id),
                timeout=self._FETCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            _log.warning("compare fetch %s timed out", run_id)
            return None, (
                f"timed out after {self._FETCH_TIMEOUT_SECONDS:.0f}s — the "
                f"Platform didn't respond (press `r` to retry)"
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "compare fetch %s failed: %s", run_id, exc,
                exc_info=False,
            )
            return None, f"{type(exc).__name__}: {exc}"
        return state, None

    def _render_side(self, side: str, state: Any) -> None:
        self._finish_side(side)
        try:
            pane = self.query_one(
                f"#evo-compare-{side}-text", Static,
            )
        except Exception:
            return
        records = extract_fitness_records(state)
        summary_lines = format_compare_summary_lines(
            extract_compare_summary(state)
        )
        if not records:
            # No curve yet — still surface any headline metrics
            # (cost / programs / best so far) so the side isn't blank.
            body = t("evolutionCompare.noFitnessData")
            if summary_lines:
                body += "\n\n" + "\n".join(summary_lines)
            pane.update(body)
            return
        plot = render_fitness_plot(records, width=58, height=12)
        body = plot or t("evolutionCompare.noFitnessData")
        if summary_lines:
            body += "\n\n" + "\n".join(summary_lines)
        pane.update(body)

    def _set_error(self, side: str, message: str) -> None:
        self._finish_side(side)
        if side == "left":
            self._left_error = message
        else:
            self._right_error = message
        try:
            pane = self.query_one(
                f"#evo-compare-{side}-text", Static,
            )
        except Exception:
            return
        pane.update(f"⚠ {message}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "evo-compare-btn-close":
            self.action_close()

    def action_close(self) -> None:
        self.action_log.append(("close", ""))
        self._stop_spinner()
        self.dismiss(None)

    def action_refetch(self) -> None:
        self.action_log.append(("refetch", ""))
        self._left_state = None
        self._right_state = None
        self._left_error = None
        self._right_error = None
        for side in ("left", "right"):
            try:
                pane = self.query_one(
                    f"#evo-compare-{side}-text", Static,
                )
                pane.update(t("common.loading"))
            except Exception:
                pass
        self._start_spinner(("left", "right"))
        self.run_worker(
            self._fetch_and_render(),
            name="evo_compare_fetch",
            group="evo_compare",
            exclusive=True,
            exit_on_error=False,
        )


__all__ = [
    "EvolutionCompareModal",
    "extract_compare_summary",
    "extract_fitness_records",
    "format_compare_summary_lines",
]
