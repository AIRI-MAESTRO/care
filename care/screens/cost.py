"""CostDashboardScreen — token + spend rollups (TODO §6 P2).

Aggregates the same `~/.cache/care/runs/<YYYY-MM-DD>.jsonl`
records the §6 P1 `/runs` screen displays, broken out by
provider / chain / mode plus a single-line overall header.

The screen is read-only — no records are mutated; reload
happens via the `r` binding.

Layout (top → bottom):

* Overall header: total runs, success rate, total tokens
  (in / out / sum), total cost, total wall-clock duration.
* Per-provider table: `provider`, `runs`, `tokens`, `cost`.
* Per-chain table: `chain`, `runs`, `tokens`, `cost`.
* Per-mode table: `mode`, `runs`, `tokens`, `cost`.

Each per-X table is sorted by ``cost_usd`` desc so the
biggest spenders sit at the top.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Label, Static

from care.runtime.cost_rollups import (
    OverallTotals,
    RollupRow,
    compute_overall,
    compute_per_chain,
    compute_per_mode,
    compute_per_provider,
)
from care.runtime.i18n import t
from care.runtime.local_run_history import (
    LocalRunEntry,
    load_local_runs,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader

_log = logging.getLogger("care.screen.cost")


_TABLE_COLUMNS: tuple[str, ...] = (
    "Key", "Runs", "Tokens", "Cost",
)


class CostDashboardScreen(Screen):
    """Token + spend rollups across local run history."""

    DEFAULT_CSS = """
    CostDashboardScreen {
        layout: vertical;
    }
    CostDashboardScreen #cost-body {
        height: 1fr;
        padding: 0 1;
    }
    CostDashboardScreen #cost-overall {
        padding: 1 2;
        color: $accent;
        text-style: bold;
    }
    CostDashboardScreen .pane-title {
        text-style: bold;
        color: $accent;
        padding: 0 2;
    }
    CostDashboardScreen DataTable {
        height: auto;
        max-height: 12;
        margin-bottom: 1;
    }
    CostDashboardScreen #cost-empty {
        padding: 1 2;
        color: $text-muted;
    }
    CostDashboardScreen #cost-status {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    DEFAULT_LIMIT: int = 1000
    """Cap input rows so a heavy user doesn't trigger an
    expensive aggregation. Spend rolls up newest-first so
    the cap drops the oldest history rather than the most
    recent (and most relevant) data."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        self._cache_dir = cache_dir
        self._limit = (
            self.DEFAULT_LIMIT if limit is None else limit
        )
        self.entries: list[LocalRunEntry] = []
        self.overall: OverallTotals = OverallTotals()
        self.by_provider: list[RollupRow] = []
        self.by_chain: list[RollupRow] = []
        self.by_mode: list[RollupRow] = []
        self.last_error: str | None = None
        self.action_log: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with VerticalScroll(id="cost-body"):
            with Vertical():
                yield Static(" ", id="cost-overall")
                yield Label(
                    t("cost.byProvider"), classes="pane-title",
                )
                yield DataTable(id="cost-provider-table")
                yield Label(t("cost.byChain"), classes="pane-title")
                yield DataTable(id="cost-chain-table")
                yield Label(t("cost.byMode"), classes="pane-title")
                yield DataTable(id="cost-mode-table")
                yield Static(" ", id="cost-empty")
        yield Static(" ", id="cost-status")
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="CostDashboardScreen",
                breadcrumb=(t("header.breadcrumb.cost"),),
            )
        except Exception:
            pass
        for table_id in (
            "#cost-provider-table",
            "#cost-chain-table",
            "#cost-mode-table",
        ):
            try:
                table = self.query_one(table_id, DataTable)
                for col in _TABLE_COLUMNS:
                    table.add_column(col, key=col)
                table.cursor_type = "row"
                table.zebra_stripes = True
            except Exception:
                pass
        self.app.call_after_refresh(self.refresh_rollups)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_rollups(self) -> None:
        try:
            self.entries = load_local_runs(
                cache_dir=self._cache_dir, limit=self._limit,
            )
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.entries = []
            self.last_error = f"{type(exc).__name__}: {exc}"
        self.overall = compute_overall(self.entries)
        self.by_provider = compute_per_provider(self.entries)
        self.by_chain = compute_per_chain(self.entries)
        self.by_mode = compute_per_mode(self.entries)
        self._apply_view()

    def action_refresh(self) -> None:
        self.action_log.append(("refresh", ""))
        self.refresh_rollups()

    def action_back(self) -> None:
        self.action_log.append(("back", ""))
        try:
            self.app.pop_screen()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _apply_view(self) -> None:
        try:
            overall_widget = self.query_one(
                "#cost-overall", Static,
            )
            empty = self.query_one("#cost-empty", Static)
            status = self.query_one("#cost-status", Static)
            provider_table = self.query_one(
                "#cost-provider-table", DataTable,
            )
            chain_table = self.query_one(
                "#cost-chain-table", DataTable,
            )
            mode_table = self.query_one(
                "#cost-mode-table", DataTable,
            )
        except Exception:
            return

        is_empty = not self.entries

        overall_widget.update(format_overall(self.overall))

        for table, rows in (
            (provider_table, self.by_provider),
            (chain_table, self.by_chain),
            (mode_table, self.by_mode),
        ):
            table.clear()
            for row in rows:
                table.add_row(
                    row.label or row.key,
                    str(row.runs),
                    str(row.tokens_total),
                    format_cost(row.cost_usd),
                    key=row.key,
                )

        empty.display = is_empty and not self.last_error
        if is_empty and not self.last_error:
            empty.update(t("cost.empty"))
        else:
            empty.update(" ")
        if self.last_error:
            status.update(f"⚠ {self.last_error}")
        else:
            status.update(
                f"{self.overall.runs} run(s)  ·  "
                f"capped at {self._limit}",
            )


# ---------------------------------------------------------------------------
# Pure formatters
# ---------------------------------------------------------------------------


def format_overall(totals: OverallTotals) -> str:
    if totals.runs == 0:
        return "no runs yet"
    parts = [
        f"runs: {totals.runs}",
        f"success: "
        f"{int((totals.success_rate or 0) * 100)}%",
        f"tokens: {totals.tokens_total}",
        f"cost: {format_cost(totals.cost_usd)}",
        f"wall: {format_duration(totals.total_duration_seconds)}",
    ]
    return "  ·  ".join(parts)


def format_cost(value: float | None) -> str:
    if value is None or value == 0:
        return "$0.0000"
    return f"${value:.4f}"


def format_duration(value: float | None) -> str:
    if value is None or value <= 0:
        return "0s"
    if value < 60:
        return f"{value:.1f}s"
    mins, secs = divmod(int(value), 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m"


def _suppress_unused(*_: Any) -> None:
    """Reserve for future Any-typed helpers; kept so the
    import line stays stable across refactors."""


__all__ = [
    "CostDashboardScreen",
    "format_cost",
    "format_duration",
    "format_overall",
]
