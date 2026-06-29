"""MarketplaceScreen — search shared `agent_skill` listings
(§8 P2 [DONE — data half] → fully DONE).

Pushed from the LibraryScreen's "Find capability" CTA (or
from a future `care marketplace <query>` CLI subcommand).
Wraps the shipped :func:`care.search_marketplace` data layer:

* Top input bar bound to ``self.query_text`` reactive. ``Enter``
  submits explicitly; typing debounces 300ms before kicking
  off a worker that calls ``search_marketplace`` so the user
  can type freely without flooding the backend.
* Results :class:`DataTable` sorted by
  :attr:`MarketplaceListing.score` descending. Columns:
  ★ (high-signal-match badge), score, name, matched_via,
  tags.
* Tag-chip sidebar shows every distinct tag in the current
  result set. Picking a chip narrows the visible rows via
  :meth:`MarketplaceResult.by_tag` — no backend round-trip.
* ``I`` (or the Install button) dispatches the selected
  listing to the host's
  ``memory.client.save_agent_skill(...)`` so the listed
  skill lands in the user's namespace. Failures surface as a
  toast through the app's ``push_toast`` host.

The screen is purely a consumer of the data layer — no new
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
from textual.widgets import Button, DataTable, Input, Label, Static

from care.marketplace import (
    MarketplaceError,
    MarketplaceListing,
    MarketplaceResult,
    search_marketplace,
)
from care.runtime.i18n import t
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


@dataclass(frozen=True)
class MarketplaceInstalled:
    """Payload posted when the user installs a listing.

    Carries the listing for telemetry / future "show
    installed" filters; ``entity_id`` is the Memory
    identifier of the newly-saved local copy (returned by
    ``save_agent_skill``)."""

    listing: MarketplaceListing
    saved_entity_id: str


class MarketplaceScreen(Screen):
    """AgentSkill marketplace browser.

    Construct with the host's `CareMemory`-like facade.
    Tests inject a duck-typed stub exposing
    ``find_capability_matches`` (for the search) +
    ``client.save_agent_skill`` (for the install action)."""

    DEFAULT_CSS = """
    MarketplaceScreen {
        layout: vertical;
    }
    MarketplaceScreen #marketplace-search-bar {
        height: 3;
        padding: 0 2;
    }
    MarketplaceScreen #marketplace-search-input {
        width: 1fr;
    }
    MarketplaceScreen #marketplace-body {
        height: 1fr;
    }
    MarketplaceScreen #marketplace-tags {
        width: 24;
        padding: 1 2;
        border-right: solid $primary;
    }
    MarketplaceScreen #marketplace-results {
        width: 1fr;
        padding: 1 2;
    }
    MarketplaceScreen .pane-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    MarketplaceScreen .tag-chip {
        margin-bottom: 1;
    }
    MarketplaceScreen #marketplace-status {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    MarketplaceScreen #marketplace-actions {
        height: 3;
        padding: 0 2;
        align-horizontal: right;
    }
    MarketplaceScreen #marketplace-actions Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("enter", "submit_search", "Search", show=False),
        Binding("i", "install_selected", "Install", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    # Debounce window between keystrokes before the worker fires.
    _SEARCH_DEBOUNCE_SEC = 0.3

    # Note: named `query_text`, NOT `query`, to avoid shadowing
    # the inherited `Screen.query(selector)` DOM-walker method
    # that Textual relies on for auto-focus + child lookups.
    query_text: reactive[str] = reactive("", init=False)
    selected_tag: reactive[str | None] = reactive(None, init=False)

    class Installed(Message):
        """Posted after a successful install for the host to react
        to (push_toast / refresh library / etc.)."""

        def __init__(
            self,
            listing: MarketplaceListing,
            saved_entity_id: str,
        ) -> None:
            super().__init__()
            self.listing = listing
            self.saved_entity_id = saved_entity_id

    def __init__(
        self,
        *,
        memory: Any = None,
        top_k: int = 20,
        min_score: float = 0.0,
        namespace: str | None = None,
        deep: bool = False,
        initial_query: str = "",
    ) -> None:
        super().__init__()
        self._memory = memory
        self._top_k = top_k
        self._min_score = min_score
        self._namespace = namespace
        self._deep = deep
        self.result: MarketplaceResult = MarketplaceResult()
        self.last_error: str | None = None
        # Currently selected row's listing (drives Install).
        self.selected_listing: MarketplaceListing | None = None
        # Debounce-timer handle so back-to-back keystrokes
        # coalesce into one worker run.
        self._debounce_timer = None
        self._initial_query = initial_query

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Horizontal(id="marketplace-search-bar"):
            yield Input(
                placeholder=t("marketplace.searchPlaceholder"),
                id="marketplace-search-input",
                value=self._initial_query,
            )
        with Horizontal(id="marketplace-body"):
            with Vertical(id="marketplace-tags"):
                yield Label(t("marketplace.tags"), classes="pane-title")
                yield VerticalScroll(id="marketplace-tag-list")
            with Vertical(id="marketplace-results"):
                yield Label(t("marketplace.results"), classes="pane-title")
                yield DataTable(id="marketplace-table")
        yield Static(t("marketplace.typeToSearch"), id="marketplace-status")
        with Horizontal(id="marketplace-actions"):
            yield Button(t("common.back"), id="marketplace-btn-back")
            yield Button(
                t("marketplace.install"),
                id="marketplace-btn-install",
                variant="success",
            )
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="MarketplaceScreen",
                breadcrumb=(t("header.breadcrumb.library"), t("header.breadcrumb.marketplace")),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="MarketplaceScreen",
                scope="screen",
            )
        except Exception:
            pass
        try:
            table = self.query_one(
                "#marketplace-table", DataTable,
            )
            table.add_columns(
                t("marketplace.colStar"),
                t("marketplace.colScore"),
                t("marketplace.colName"),
                t("marketplace.colMatched"),
                t("marketplace.colTags"),
            )
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        # If the host pre-seeded a query, fire one search now.
        if self._initial_query:
            self.query_text = self._initial_query
            self._run_search_worker()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "marketplace-search-input":
            return
        # Textual's `Input(value=...)` constructor emits a
        # Changed event when the widget mounts. Skip it so the
        # initial_query search isn't fired twice + a stale
        # `last_error` from a follow-up action doesn't get
        # clobbered by a no-op re-search.
        if event.value == self.query_text:
            return
        self.query_text = event.value
        self._schedule_search()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "marketplace-search-input":
            return
        self.query_text = event.value
        self.action_submit_search()

    def action_submit_search(self) -> None:
        """`Enter` from anywhere fires an immediate search."""
        if self._debounce_timer is not None:
            try:
                self._debounce_timer.stop()
            except Exception:
                pass
            self._debounce_timer = None
        self._run_search_worker()

    def _schedule_search(self) -> None:
        """Coalesce back-to-back keystrokes into one worker call.

        Cancels any pending timer + reschedules. The worker
        only fires once the user pauses typing for
        ``_SEARCH_DEBOUNCE_SEC`` seconds.
        """
        if self._debounce_timer is not None:
            try:
                self._debounce_timer.stop()
            except Exception:
                pass
        try:
            self._debounce_timer = self.set_timer(
                self._SEARCH_DEBOUNCE_SEC, self._run_search_worker,
            )
        except Exception:
            # Outside the Textual loop (rare; mostly tests
            # that drive the action directly). Just run.
            self._run_search_worker()

    def _run_search_worker(self) -> None:
        self._debounce_timer = None
        self.run_worker(
            self._search(),
            name="marketplace_search",
            group="marketplace",
            exclusive=True,
            exit_on_error=False,
        )

    async def _search(self) -> None:
        query = self.query_text
        if self._memory is None:
            self.last_error = t("marketplace.noMemoryFacade")
            self.result = MarketplaceResult(query=query)
            self._refresh_panes()
            return
        try:
            result = search_marketplace(
                self._memory,
                query,
                top_k=self._top_k,
                min_score=self._min_score,
                namespace=self._namespace,
                deep=self._deep,
            )
        except MarketplaceError as exc:
            self.last_error = str(exc)
            self.result = MarketplaceResult(query=query)
            self._refresh_panes()
            return
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.result = MarketplaceResult(query=query)
            self._refresh_panes()
            return
        self.last_error = None
        self.result = result
        # A new search invalidates the tag filter (the chip the
        # user picked may not be in the new result set).
        self.selected_tag = None
        self.selected_listing = None
        self._refresh_panes()

    # ------------------------------------------------------------------
    # Tag-chip sidebar
    # ------------------------------------------------------------------

    def visible_listings(self) -> tuple[MarketplaceListing, ...]:
        """Listings after the active tag-chip filter is applied."""
        if not self.selected_tag:
            return self.result.listings
        return self.result.by_tag(self.selected_tag)

    def collect_tags(self) -> tuple[str, ...]:
        """Distinct tags across the full result set (not
        post-filter) so the user can switch between chips."""
        seen: dict[str, None] = {}
        for li in self.result.listings:
            for tag in li.tags:
                if tag not in seen:
                    seen[tag] = None
        return tuple(seen.keys())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "marketplace-btn-back":
            self.action_back()
        elif bid == "marketplace-btn-install":
            self.action_install_selected()
        elif bid.startswith("marketplace-tag-"):
            tag = bid.removeprefix("marketplace-tag-")
            self._select_tag(tag)

    def _select_tag(self, tag: str) -> None:
        # Toggle: re-clicking the active tag clears the filter.
        if self.selected_tag == tag:
            self.selected_tag = None
        else:
            self.selected_tag = tag
        self._render_results()
        self._render_status()

    # ------------------------------------------------------------------
    # Selection + install
    # ------------------------------------------------------------------

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected,
    ) -> None:
        if event.data_table.id != "marketplace-table":
            return
        row_key = str(event.row_key.value or "")
        self.selected_listing = self._listing_by_id(row_key)

    def _listing_by_id(
        self, entity_id: str,
    ) -> MarketplaceListing | None:
        for li in self.visible_listings():
            if li.entity_id == entity_id:
                return li
        return None

    def action_install_selected(self) -> None:
        listing = self.selected_listing
        if listing is None:
            return
        self.run_worker(
            self._install(listing),
            name="marketplace_install",
            group="marketplace_install",
            exclusive=True,
            exit_on_error=False,
        )

    async def _install(self, listing: MarketplaceListing) -> None:
        client = getattr(self._memory, "client", None) or self._memory
        save = getattr(client, "save_agent_skill", None)
        if not callable(save):
            self.last_error = t("marketplace.saveUnavailable")
            self._render_status()
            self._push_toast(self.last_error, severity="error")
            return
        try:
            response = save(
                entity_id=listing.entity_id,
                # Production callers will pull the full skill
                # spec from Memory before saving; for the screen
                # the entity_id + name carry enough for the SDK's
                # idempotent save path (existing entity_id →
                # version bump).
                name=listing.name,
            )
        except Exception as exc:  # noqa: BLE001
            error_text = t("marketplace.installFailed", error=exc)
            self.last_error = error_text
            # Render + toast best-effort. If either blows up we
            # still want `last_error` to be observable so the
            # outer test can assert on the failure.
            try:
                self._render_status()
            except Exception:
                pass
            try:
                self._push_toast(error_text, severity="error")
            except Exception:
                pass
            return
        saved_id = self._extract_saved_entity_id(response, listing)
        self._push_toast(
            t("marketplace.installed", name=listing.name), severity="success",
        )
        self.post_message(self.Installed(listing, saved_id))

    @staticmethod
    def _extract_saved_entity_id(
        response: Any, listing: MarketplaceListing,
    ) -> str:
        """SDK methods return varied shapes — probe in order
        and fall back to the listing's own id."""
        if isinstance(response, str):
            return response
        for attr in ("entity_id", "id"):
            value = getattr(response, attr, None)
            if isinstance(value, str) and value:
                return value
        if isinstance(response, dict):
            for key in ("entity_id", "id"):
                value = response.get(key)
                if isinstance(value, str) and value:
                    return value
        return listing.entity_id

    def _push_toast(self, message: str, *, severity: str) -> None:
        try:
            self.app.push_toast(message, severity=severity)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_back(self) -> None:
        try:
            self.app.pop_screen()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh_panes(self) -> None:
        self._render_tag_chips()
        self._render_results()
        self._render_status()

    def _render_tag_chips(self) -> None:
        try:
            container = self.query_one(
                "#marketplace-tag-list", VerticalScroll,
            )
        except Exception:
            return
        for child in list(container.children):
            try:
                child.remove()
            except Exception:
                pass
        tags = self.collect_tags()
        if not tags:
            container.mount(Static(t("marketplace.noTags")))
            return
        for tag in tags:
            label = f"#{tag}"
            if self.selected_tag == tag:
                label = f"✓ {label}"
            container.mount(
                Button(
                    label,
                    id=f"marketplace-tag-{tag}",
                    classes="tag-chip",
                ),
            )

    def _render_results(self) -> None:
        try:
            table = self.query_one(
                "#marketplace-table", DataTable,
            )
        except Exception:
            return
        try:
            table.clear()
        except Exception:
            pass
        for li in self.visible_listings():
            badge = "★" if li.matched_via == "skill_description" else ""
            score = f"{li.score:.3f}"
            tags = ", ".join(li.tags) if li.tags else ""
            table.add_row(
                badge,
                score,
                li.name,
                li.matched_via or "—",
                tags,
                key=li.entity_id,
            )

    def _render_status(self) -> None:
        try:
            target = self.query_one("#marketplace-status", Static)
        except Exception:
            return
        if self.last_error:
            target.update(f"⚠ {self.last_error}")
            return
        listings = self.visible_listings()
        if not listings:
            if not self.query_text.strip():
                target.update(t("marketplace.typeToSearch"))
            else:
                target.update(
                    t("marketplace.noResults", query=repr(self.query_text)),
                )
            return
        suffix = (
            t("marketplace.filteredBy", tag=self.selected_tag)
            if self.selected_tag else ""
        )
        target.update(
            t(
                "marketplace.listingCount",
                count=len(listings),
                query=repr(self.query_text),
                suffix=suffix,
            ),
        )


__all__ = [
    "MarketplaceInstalled",
    "MarketplaceScreen",
]
