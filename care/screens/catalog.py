"""CatalogScreen — browse installed capabilities
(§8 P1 [DONE — CLI half] → fully DONE).

Pushed from the LibraryScreen's "Browse capabilities" CTA (or
from any host that wants to surface the four kinds of locally-
discoverable agent capabilities). Wraps the shipped
:func:`care.build_catalog` data layer + the
:class:`care.CapabilityCatalog` aggregate the CLI's
`care catalog` subcommand already consumes — so the TUI and CLI
render the same entries / errors shape.

Layout:

* **Kind filter sidebar** — five chip-buttons (``all`` +
  one per :class:`EntryKind`). The active filter narrows the
  results table.
* **Results DataTable** — kind / name / source / tags + summary
  preview. Cursor selection drives the Promote action.
* **Warnings panel** — collapsible footer listing
  :attr:`CapabilityCatalog.errors` (broken SKILL.md / 503 from
  Memory / etc.). Empty when discovery completed cleanly.
* **Status line** — `N entries [· filtered to <kind>]` / error
  text on failure.

The screen is a pure consumer of the data layer — no new
projection logic ships here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Static

from care.catalog import CapabilityCatalog, CapabilityCatalogEntry, EntryKind
from care.runtime.i18n import t
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


_KINDS: tuple[EntryKind, ...] = (
    "agent_skill",
    "mcp_server",
    "tool",
    "memory_card",
)


@dataclass(frozen=True)
class CatalogPromoteRequest:
    """Payload posted when the user fires the Promote action on
    an `agent_skill` entry — the screen doesn't perform the
    Memory write itself so the host can decide whether to
    prompt for confirmation / source URI / etc."""

    entry: CapabilityCatalogEntry


class CatalogScreen(Screen):
    """Browse installed capabilities discovered via
    :func:`care.build_catalog`.

    Construct with a pre-built :class:`CapabilityCatalog`
    (preferred — keeps discovery scheduling out of the screen)
    or a lazy ``catalog_factory`` callable that builds one on
    demand. Tests almost always pass the catalog directly."""

    DEFAULT_CSS = """
    CatalogScreen {
        layout: vertical;
    }
    CatalogScreen #catalog-body {
        height: 1fr;
    }
    CatalogScreen #catalog-kinds {
        width: 22;
        padding: 1 2;
        border-right: solid $primary;
    }
    CatalogScreen #catalog-results {
        width: 1fr;
        padding: 1 2;
    }
    CatalogScreen .pane-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    CatalogScreen .kind-chip {
        margin-bottom: 1;
    }
    CatalogScreen #catalog-status {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    CatalogScreen #catalog-warnings {
        height: auto;
        max-height: 6;
        padding: 0 2;
        color: $warning;
    }
    CatalogScreen #catalog-actions {
        height: 3;
        padding: 0 2;
        align-horizontal: right;
    }
    CatalogScreen #catalog-actions Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("p", "promote_selected", "Promote", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    selected_kind: reactive[str] = reactive("all", init=False)

    class PromoteRequested(Message):
        """Posted when the user fires the Promote action so the
        host can drive the Memory write (or push a confirm modal
        first)."""

        def __init__(self, entry: CapabilityCatalogEntry) -> None:
            super().__init__()
            self.entry = entry

    def __init__(
        self,
        catalog: CapabilityCatalog | None = None,
        *,
        catalog_factory: Any = None,
        focus_entry_id: str | None = None,
    ) -> None:
        super().__init__()
        # Either a pre-built catalog (tests + most hosts) or a
        # lazy builder fired on mount.
        self.catalog: CapabilityCatalog = (
            catalog if catalog is not None else CapabilityCatalog()
        )
        self._catalog_factory = catalog_factory
        # Optional entry_id to land focused on. Used by the
        # palette dispatch when the user picks an agent_skill
        # row — the screen lands focused on that specific
        # skill rather than the top of the list. Empty / None
        # → no auto-focus (default behaviour).
        self._focus_entry_id = focus_entry_id or None
        self.selected_entry: CapabilityCatalogEntry | None = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Horizontal(id="catalog-body"):
            with Vertical(id="catalog-kinds"):
                yield Label(t("catalog.kinds"), classes="pane-title")
                yield VerticalScroll(id="catalog-kind-list")
            with Vertical(id="catalog-results"):
                yield Label(t("catalog.entries"), classes="pane-title")
                yield DataTable(id="catalog-table")
        yield Static("", id="catalog-status")
        yield Static("", id="catalog-warnings")
        with Horizontal(id="catalog-actions"):
            yield Button(t("common.back"), id="catalog-btn-back")
            yield Button(
                t("catalog.promote"),
                id="catalog-btn-promote",
                variant="primary",
            )
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="CatalogScreen",
                breadcrumb=(t("header.breadcrumb.library"), t("header.breadcrumb.catalog")),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="CatalogScreen",
                scope="screen",
            )
        except Exception:
            pass
        try:
            table = self.query_one("#catalog-table", DataTable)
            table.add_columns(
                t("catalog.colKind"),
                t("catalog.colName"),
                t("catalog.colSource"),
                t("catalog.colTags"),
                t("catalog.colSummary"),
            )
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        # Lazy build path — only when no catalog supplied.
        if self.catalog.is_empty and self._catalog_factory is not None:
            try:
                self.catalog = self._catalog_factory()
            except Exception:  # noqa: BLE001
                # Best-effort; render an empty catalog rather
                # than crash the screen on a misbehaving factory.
                self.catalog = CapabilityCatalog()
        self._refresh_panes()
        if self._focus_entry_id:
            self._focus_on_entry_id(self._focus_entry_id)

    def _focus_on_entry_id(self, entry_id: str) -> None:
        """Move the DataTable cursor onto the row whose
        underlying entry matches ``entry_id``.

        Catalog entries don't carry their own `entity_id`
        (memory_card rows store one in `source` as
        ``memory://<id>``; the other kinds use file paths).
        Match strategy:
          1. `source` exactly equals ``entry_id`` (most cases).
          2. `source` is ``memory://<entry_id>`` (memory_card
             palette picks).
          3. `name` equals ``entry_id`` (palette picks for
             agent_skill entries use the Memory entity_id but
             the catalog entry's source is the SKILL.md path —
             no match here; fall through to name match for
             entries the catalog also tags by name).

        Best-effort: an entry_id that doesn't resolve to a
        visible row leaves the cursor at the default position
        (no exception, no error toast — the user can still
        scroll manually).
        """
        target = self._find_entry_for_focus(entry_id)
        if target is None:
            return
        self.selected_entry = target
        try:
            from textual.coordinate import Coordinate

            table = self.query_one("#catalog-table", DataTable)
            row_index = table.get_row_index(self._row_key_for(target))
            table.move_cursor(
                coordinate=Coordinate(row_index, 0),
                animate=False,
            )
        except Exception:
            pass

    def _find_entry_for_focus(
        self, entry_id: str,
    ) -> CapabilityCatalogEntry | None:
        memory_uri = f"memory://{entry_id}"
        for entry in self.catalog.entries:
            if entry.source == entry_id:
                return entry
            if entry.source == memory_uri:
                return entry
        # Fallback by name (palette-pick entries from Memory
        # share the same `name` as the catalog's
        # frontmatter-`name` field for agent_skill rows that
        # were also promoted).
        for entry in self.catalog.entries:
            if entry.name == entry_id:
                return entry
        return None

    # ------------------------------------------------------------------
    # Visible-entry projection
    # ------------------------------------------------------------------

    def visible_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        """Entries after the active kind filter is applied."""
        if self.selected_kind == "all":
            return self.catalog.entries
        # Cast: selected_kind is a `str` reactive but at this
        # point we've vetted it against `_KINDS` in the chip
        # handler.
        return self.catalog.by_kind(self.selected_kind)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Chip handling
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "catalog-btn-back":
            self.action_back()
        elif bid == "catalog-btn-promote":
            self.action_promote_selected()
        elif bid.startswith("catalog-kind-chip-"):
            chip_kind = bid.removeprefix("catalog-kind-chip-")
            self._select_kind(chip_kind)

    def _select_kind(self, kind: str) -> None:
        # Re-clicking the active chip clears the filter
        # (matches the marketplace screen's chip toggle).
        if self.selected_kind == kind:
            self.selected_kind = "all"
        else:
            self.selected_kind = kind
        self._render_results()
        self._render_status()

    # ------------------------------------------------------------------
    # Selection + actions
    # ------------------------------------------------------------------

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected,
    ) -> None:
        if event.data_table.id != "catalog-table":
            return
        key = str(event.row_key.value or "")
        self.selected_entry = self._entry_by_key(key)

    def _entry_by_key(self, key: str) -> CapabilityCatalogEntry | None:
        # Row key format: ``"<kind>::<name>::<source>"`` —
        # matches what `_render_results` writes. Source can
        # contain colons (e.g. ``memory://ent-1``) so split
        # with a max-count.
        parts = key.split("::", 2)
        if len(parts) != 3:
            return None
        kind, name, source = parts
        for entry in self.visible_entries():
            if entry.kind == kind and entry.name == name and entry.source == source:
                return entry
        return None

    @staticmethod
    def _row_key_for(entry: CapabilityCatalogEntry) -> str:
        return f"{entry.kind}::{entry.name}::{entry.source}"

    def action_promote_selected(self) -> None:
        """Fire the Promote message for the selected entry.

        The screen does NOT call ``promote_skill_to_memory``
        itself — the host owns the confirmation flow + Memory
        wiring. Useful for: a host that wants to push a
        confirm modal asking for ``source_uri``; the future
        LibraryScreen integration that pushes the promote action
        through its own worker pool with the existing toast
        host. This keeps the screen pure-presentation.
        """
        entry = self.selected_entry
        if entry is None:
            return
        if entry.kind != "agent_skill":
            # Promote only makes sense for agent_skill entries;
            # MCP servers / tools / memory_cards already live
            # somewhere else. Surface a friendly note rather
            # than fire a message the host would have to filter.
            self._push_toast(
                t("catalog.promoteOnlySkills", kind=entry.kind),
                severity="warning",
            )
            return
        self.post_message(self.PromoteRequested(entry))

    def _push_toast(self, message: str, *, severity: str) -> None:
        try:
            self.app.push_toast(message, severity=severity)
        except Exception:
            pass

    def action_back(self) -> None:
        try:
            self.app.pop_screen()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh_panes(self) -> None:
        self._render_kind_chips()
        self._render_results()
        self._render_status()
        self._render_warnings()

    def _render_kind_chips(self) -> None:
        try:
            container = self.query_one(
                "#catalog-kind-list", VerticalScroll,
            )
        except Exception:
            return
        for child in list(container.children):
            try:
                child.remove()
            except Exception:
                pass
        # Always include "all" + one chip per Kind that has at
        # least one entry. Hiding empty-kind chips keeps the
        # sidebar lean on partial catalogs.
        kinds_present = {e.kind for e in self.catalog.entries}
        labels: list[tuple[str, str]] = [("all", "all")]
        for kind in _KINDS:
            if kind in kinds_present:
                labels.append((kind, kind))
        for chip_id, label in labels:
            text = f"#{label}"
            if self.selected_kind == chip_id:
                text = f"✓ {text}"
            container.mount(
                Button(
                    text,
                    id=f"catalog-kind-chip-{chip_id}",
                    classes="kind-chip",
                ),
            )

    def _render_results(self) -> None:
        try:
            table = self.query_one("#catalog-table", DataTable)
        except Exception:
            return
        try:
            table.clear()
        except Exception:
            pass
        for entry in self.visible_entries():
            tags = ", ".join(entry.tags) if entry.tags else ""
            summary = entry.summary
            if len(summary) > 60:
                summary = summary[:57] + "…"
            table.add_row(
                entry.kind,
                entry.name,
                self._truncate_source(entry.source),
                tags,
                summary,
                key=self._row_key_for(entry),
            )

    def _render_status(self) -> None:
        try:
            target = self.query_one("#catalog-status", Static)
        except Exception:
            return
        visible = self.visible_entries()
        if self.catalog.is_empty:
            target.update(t("catalog.empty"))
            return
        filter_text = (
            t("catalog.filteredTo", kind=self.selected_kind)
            if self.selected_kind != "all" else ""
        )
        count_key = (
            "catalog.entryCount.one"
            if len(visible) == 1 else "catalog.entryCount.many"
        )
        target.update(
            t(count_key, count=len(visible), filter=filter_text),
        )

    def _render_warnings(self) -> None:
        try:
            target = self.query_one("#catalog-warnings", Static)
        except Exception:
            return
        if not self.catalog.errors:
            target.update("")
            return
        lines = [
            t("catalog.warningsHeader", count=len(self.catalog.errors)),
        ]
        for err in self.catalog.errors[:5]:
            lines.append(f"  · {err}")
        if len(self.catalog.errors) > 5:
            lines.append(
                t("catalog.warningsMore", count=len(self.catalog.errors) - 5),
            )
        target.update("\n".join(lines))

    @staticmethod
    def _truncate_source(source: str, *, n: int = 50) -> str:
        if len(source) <= n:
            return source
        # Keep the tail (the actual file name) — paths matter
        # more at the end than the beginning.
        return "…" + source[-(n - 1):]


__all__ = [
    "CatalogPromoteRequest",
    "CatalogScreen",
]
