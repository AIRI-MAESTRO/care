"""LineageModal — view ancestry DAG of a saved chain
(TODO §1.1 P0.26).

Pushed when the user invokes the `L` (Show lineage) row
action on LibraryScreen / InspectionScreen. Calls
:func:`care.runtime.lineage.fetch_chain_lineage` on mount,
projects into :class:`LineageGraph`, and renders the DAG via
`graph.layers()`. Per-node "Re-run from here" action calls
:func:`care.runtime.prime_from_saved_chain(..., version_id=
node.version_id)` — exposed via the dismiss envelope so the
host screen pushes the next screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from care.runtime.i18n import t
from care.runtime.lineage import (
    LineageGraph,
    LineageNode,
    fetch_chain_lineage,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


@dataclass(frozen=True)
class LineageResult:
    """Dismiss envelope.

    ``rerun_version_id`` is non-``None`` when the user picks
    "Re-run from here" — the host then primes a re-run via
    `prime_from_saved_chain`."""

    rerun_version_id: str | None = None


class LineageModal(ModalScreen[LineageResult]):
    """Modal lineage viewer.

    Construct with `entity_id` + the `CareMemory`-like
    facade. `on_mount` fires the lineage worker; the result
    populates a layered ASCII tree (one row per node, indented
    by depth) and a DataTable for richer drill-down. The
    selected row's "Re-run from here" button dismisses with
    the chosen version id."""

    DEFAULT_CSS = """
    LineageModal {
        align: center middle;
    }
    LineageModal #lineage-box {
        width: 100;
        max-width: 95%;
        height: 30;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    LineageModal #lineage-title {
        text-style: bold;
        padding-bottom: 1;
    }
    LineageModal #lineage-body {
        height: 1fr;
    }
    LineageModal #lineage-tree {
        width: 1fr;
        padding-right: 1;
    }
    LineageModal #lineage-table {
        width: 2fr;
    }
    LineageModal #lineage-actions {
        height: 3;
        align-horizontal: right;
    }
    LineageModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        entity_id: str,
        *,
        memory: Any = None,
    ) -> None:
        super().__init__()
        self.entity_id = entity_id
        self._memory = memory
        self.graph: LineageGraph = LineageGraph(
            entity_id=entity_id,
            root_version_id="",
        )
        self.load_error: str | None = None
        self.selected_version: str | None = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="lineage-box"):
            yield CareHeader()
            yield Static(
                t("lineage.title", id=self.entity_id),
                id="lineage-title",
            )
            with Horizontal(id="lineage-body"):
                yield VerticalScroll(id="lineage-tree")
                yield DataTable(id="lineage-table")
            with Horizontal(id="lineage-actions"):
                yield Button(t("common.cancel"), id="lineage-btn-cancel")
                yield Button(
                    t("lineage.rerun"),
                    id="lineage-btn-rerun",
                    variant="primary",
                )
            yield CareFooter()

    def on_mount(self) -> None:
        try:
            table = self.query_one("#lineage-table", DataTable)
            table.add_columns(
                t("lineage.colVersion"),
                t("lineage.colDepth"),
                t("lineage.colGen"),
                t("lineage.colFitness"),
                t("lineage.colSummary"),
            )
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        if self._memory is None:
            self.load_error = t("lineage.noMemory")
            self._loaded = True
            self._render_panes()
            return
        # Native animated loading overlay while the lineage graph fetches;
        # cleared in `_render_panes` (every `_load` exit path calls it).
        self.loading = True
        self.run_worker(
            self._load(),
            name="lineage_load",
            group="lineage",
            exclusive=True,
            exit_on_error=False,
        )

    async def _load(self) -> None:
        try:
            graph = await fetch_chain_lineage(
                self._memory, self.entity_id,
            )
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._loaded = True
            self._render_panes()
            return
        self.graph = graph
        self.load_error = None
        self._loaded = True
        # Default selection: the best (highest-fitness) node
        # when available, else the root.
        best = graph.best()
        if best is not None:
            self.selected_version = best.version_id
        elif graph.root is not None:
            self.selected_version = graph.root.version_id
        self._render_panes()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_panes(self) -> None:
        # The fetch worker resolves into here on every path — drop the
        # loading overlay armed in `on_mount`.
        self.loading = False
        try:
            tree = self.query_one("#lineage-tree", VerticalScroll)
            table = self.query_one("#lineage-table", DataTable)
        except Exception:
            return
        try:
            for child in list(tree.children):
                child.remove()
        except Exception:
            pass
        if self.load_error:
            tree.mount(Static(f"⚠ {self.load_error}"))
            return
        if not self.graph.nodes:
            tree.mount(Static(t("lineage.noData")))
            return
        for layer in self.graph.layers():
            for node in layer:
                tree.mount(Static(self._tree_row(node)))
        try:
            table.clear()
        except Exception:
            pass
        for node in self.graph.nodes:
            table.add_row(*self._table_row(node), key=node.version_id)

    @staticmethod
    def _tree_row(node: LineageNode) -> str:
        indent = "  " * node.depth
        marker = "✦" if node.is_crossover else "•"
        bits = [f"{indent}{marker} v{node.version_number}"]
        if node.fitness is not None:
            bits.append(f"fit={node.fitness:.3f}")
        if node.mutation_kind:
            bits.append(node.mutation_kind)
        return "  ".join(bits)

    @staticmethod
    def _table_row(node: LineageNode) -> tuple[str, ...]:
        fitness = (
            f"{node.fitness:.3f}" if node.fitness is not None else "—"
        )
        generation = (
            str(node.generation) if node.generation is not None else "—"
        )
        summary = (node.change_summary or "")[:60]
        return (
            f"v{node.version_number}",
            str(node.depth),
            generation,
            fitness,
            summary,
        )

    # ------------------------------------------------------------------
    # Selection / actions
    # ------------------------------------------------------------------

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected,
    ) -> None:
        if event.data_table.id != "lineage-table":
            return
        try:
            self.selected_version = str(event.row_key.value or "")
        except Exception:
            self.selected_version = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "lineage-btn-cancel":
            self.action_cancel()
        elif bid == "lineage-btn-rerun":
            self._rerun_selected()

    def _rerun_selected(self) -> None:
        if not self.selected_version:
            return
        self.dismiss(
            LineageResult(rerun_version_id=self.selected_version),
        )

    def action_cancel(self) -> None:
        self.dismiss(LineageResult(rerun_version_id=None))


__all__ = [
    "LineageModal",
    "LineageResult",
]
