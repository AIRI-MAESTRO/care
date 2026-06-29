"""LibraryScreen sidebar — filter chips + search (TODO §1.1 P0.8).

Vertical widget mounted in the LibraryScreen's left rail. Hosts:

* A search input (``Input#library-sidebar-search``) bound to
  :attr:`LibraryFilters.search`. Updates fire on every
  keystroke.
* A status radio set (``RadioSet#library-sidebar-status``)
  binding to :attr:`LibraryFilters.status` (``draft`` /
  ``runnable`` / ``evolved`` / All).
* A favourites-only checkbox
  (``Checkbox#library-sidebar-favourites``) binding to
  :attr:`LibraryFilters.favourites_only`.

Every change posts a :class:`LibrarySidebar.FiltersChanged`
message carrying the new :class:`LibraryFilters`. The host
screen (`LibraryScreen`) subscribes via
``on_library_sidebar_filters_changed`` and re-runs its fetch
worker.

Future P0.14 (collections tree) and tag-chip multi-select land
on top of this scaffold without changing the message
contract.
"""

from __future__ import annotations

from typing import Iterable

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Checkbox,
    Input,
    OptionList,
    RadioButton,
    RadioSet,
    Static,
)
from textual.widgets.option_list import Option

from care.runtime.collections import (
    Collection,
    active_collection_name,
)
from care.runtime.i18n import t
from care.runtime.library_view import (
    LibraryFilters,
    with_favourites_only,
    with_search,
    with_status,
    with_tags,
)


_COLLECTIONS_ALL_ID = "__all__"
"""Sentinel option id meaning ``filter_by_collection(filters, None)``."""


_STATUS_BY_BUTTON_ID = {
    "library-sidebar-status-all": None,
    "library-sidebar-status-draft": "draft",
    "library-sidebar-status-runnable": "runnable",
    "library-sidebar-status-evolved": "evolved",
}
_BUTTON_ID_BY_STATUS = {v: k for k, v in _STATUS_BY_BUTTON_ID.items()}

_TAG_CHIP_ID_PREFIX = "library-sidebar-tag-"
"""Prefix on every tag-chip widget id. Used by
:meth:`LibrarySidebar.on_checkbox_changed` to route tag toggles
without colliding with the favourites checkbox."""

_TAG_POOL_CAP = 24
"""Maximum number of tag chips the sidebar renders. Pools
larger than this are truncated (most-common-first ordering is
the caller's responsibility) so the sidebar layout stays
predictable."""


def _tag_chip_id(tag: str) -> str:
    """Sanitise ``tag`` into a Textual-safe widget id.

    Textual ids must match ``[a-zA-Z_][a-zA-Z0-9_-]*``. Tags
    can contain ``:`` / ``.`` / spaces / other punctuation
    (eg. ``"domain:weather"``, ``"v0.1"``). Replace anything
    outside the allowed character set with ``_`` so the id
    remains stable + reversible enough for tests.
    """
    sanitised = "".join(
        ch if (ch.isalnum() or ch in "-_") else "_" for ch in tag
    )
    return f"{_TAG_CHIP_ID_PREFIX}{sanitised}"


class LibrarySidebar(VerticalScroll):
    """Filter sidebar bound to a :class:`LibraryFilters`.

    Construct without args; the host screen sets initial
    filters via the constructor kwarg or :meth:`set_filters`.
    Every interactive change emits a
    :class:`FiltersChanged` message — the screen listens +
    re-runs the worker.

    Extends :class:`VerticalScroll` so the filter stack scrolls
    when the status radios + collections list + tag chips
    outgrow the available height instead of clipping off-screen.
    """

    DEFAULT_CSS = """
    LibrarySidebar {
        padding: 1;
        background: $panel;
        scrollbar-size-vertical: 1;
    }
    LibrarySidebar #library-sidebar-search {
        margin-bottom: 1;
    }
    LibrarySidebar .library-sidebar-section-title {
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }
    """

    class FiltersChanged(Message):
        """Posted on every filter-state mutation."""

        def __init__(self, filters: LibraryFilters) -> None:
            super().__init__()
            self.filters = filters

    class CollectionSelected(Message):
        """Posted when the user picks a collection node from
        the sidebar. ``name`` is the bare collection name, or
        ``None`` for the "All" / clear-filter entry."""

        def __init__(self, name: str | None) -> None:
            super().__init__()
            self.name = name

    class CollectionActionRequested(Message):
        """Posted when the user invokes a per-collection
        action from the sidebar's keyboard handlers (rename /
        delete). The host screen routes the request through
        the matching `apply_rename_collection` /
        `apply_delete_collection` worker."""

        def __init__(self, action: str, name: str) -> None:
            super().__init__()
            self.action = action
            self.name = name

    def __init__(self, filters: LibraryFilters | None = None) -> None:
        super().__init__()
        self._filters: LibraryFilters = (
            filters if filters is not None else LibraryFilters()
        )
        self._collections: tuple[Collection, ...] = ()
        # §4 P0 — known tags rendered as toggle chips below.
        # Populated by `set_tag_pool` after the LibraryScreen
        # fetches the current view; per-tag Checkboxes mount
        # into `#library-sidebar-tag-list`. Capped at
        # `_TAG_POOL_CAP` so a 200-tag namespace doesn't blow
        # the sidebar layout.
        self._tag_pool: tuple[str, ...] = ()
        # Re-entry guard for sync-driven widget mutations.
        # `set_filters` programmatically swaps Input.value /
        # RadioButton.value / Checkbox.value — those setters
        # schedule async events that fire AFTER the
        # synchronous `set_filters` call returns. A simple
        # boolean reset doesn't span the gap; the counter
        # tracks how many "synthetic" events we expect and
        # each handler decrements + suppresses while > 0.
        self._suppress_count = 0
        # Same trick for the collections OptionList — when
        # we programmatically highlight the active node, the
        # change event would fire a no-op CollectionSelected
        # otherwise.
        self._collection_suppress_count = 0
        # §4 P2 — one-shot absorber for the leading `/`
        # keystroke that focuses the search input. When set,
        # the next `Input.Changed` with value=="/" is dropped
        # + the input is reset to "". `Ctrl+F` doesn't set the
        # flag so its activation keystroke (which Textual
        # doesn't redeliver to the focused widget anyway)
        # leaves the input alone.
        self._absorb_next_search_keystroke = False

    @property
    def filters(self) -> LibraryFilters:
        """Read-only snapshot — tests + telemetry."""
        return self._filters

    @property
    def collections(self) -> tuple[Collection, ...]:
        """Read-only snapshot of the collections rendered in
        the sidebar (without the synthetic "All" entry)."""
        return self._collections

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(
            t("library.sidebar.filters"),
            classes="library-sidebar-section-title",
        )
        yield Input(
            placeholder=t("library.sidebar.searchPlaceholder"),
            value=self._filters.search,
            id="library-sidebar-search",
        )
        yield Static(
            t("library.sidebar.status"),
            classes="library-sidebar-section-title",
        )
        active_status = self._filters.status
        with RadioSet(id="library-sidebar-status"):
            yield RadioButton(
                t("library.sidebar.statusAll"),
                value=active_status is None,
                id="library-sidebar-status-all",
            )
            yield RadioButton(
                t("library.sidebar.statusDraft"),
                value=active_status == "draft",
                id="library-sidebar-status-draft",
            )
            yield RadioButton(
                t("library.sidebar.statusRunnable"),
                value=active_status == "runnable",
                id="library-sidebar-status-runnable",
            )
            yield RadioButton(
                t("library.sidebar.statusEvolved"),
                value=active_status == "evolved",
                id="library-sidebar-status-evolved",
            )
        yield Static(
            t("library.sidebar.options"),
            classes="library-sidebar-section-title",
        )
        yield Checkbox(
            t("library.sidebar.favouritesOnly"),
            value=self._filters.favourites_only,
            id="library-sidebar-favourites",
        )
        yield Static(
            t("library.sidebar.collections"),
            classes="library-sidebar-section-title",
        )
        yield OptionList(
            *self._collection_options(),
            id="library-sidebar-collections",
        )
        # §4 P0 — Tag chips. The Vertical mounts empty; the
        # LibraryScreen's `_refresh` worker calls
        # `set_tag_pool(...)` after each fetch to mount one
        # Checkbox per known tag. Empty pool keeps the section
        # title visible so the user sees "Tags" exists even
        # before the first fetch returns.
        yield Static(
            t("library.sidebar.tags"),
            classes="library-sidebar-section-title",
        )
        yield Vertical(id="library-sidebar-tag-list")

    # ------------------------------------------------------------------
    # External hooks
    # ------------------------------------------------------------------

    def set_filters(self, filters: LibraryFilters) -> None:
        """Sync widget state to ``filters`` without emitting a
        :class:`FiltersChanged` message. Used when the host
        screen mutates filters elsewhere (e.g. collections
        sidebar in P0.14 picking a tag) and wants the chip set
        to reflect the new state.

        Suppresses up to 3 pending change events — one per
        widget that might fire on its `value` setter.
        """
        if not self.is_mounted:
            self._filters = filters
            return
        self._filters = filters

        search = self.query_one("#library-sidebar-search", Input)
        if search.value != filters.search:
            self._suppress_count += 1
            search.value = filters.search
        fav = self.query_one("#library-sidebar-favourites", Checkbox)
        if fav.value != filters.favourites_only:
            self._suppress_count += 1
            fav.value = filters.favourites_only
        target_id = _BUTTON_ID_BY_STATUS.get(filters.status)
        if target_id is not None:
            target = self.query_one(f"#{target_id}", RadioButton)
            if not target.value:
                self._suppress_count += 1
                target.value = True
        # §4 P0 — sync tag chip values to the new filter state.
        # Skip silently when no tag chips have been mounted
        # yet (set_tag_pool hasn't been called).
        active = filters.tag_set
        for tag in self._tag_pool:
            chip_id = _tag_chip_id(tag)
            try:
                chip = self.query_one(f"#{chip_id}", Checkbox)
            except Exception:
                continue
            want = tag in active
            if chip.value != want:
                self._suppress_count += 1
                chip.value = want

    def set_tag_pool(self, tags: Iterable[str]) -> None:
        """Replace the rendered tag chip set with ``tags``
        (§4 P0).

        Caller passes the most-common-first ordering they want
        — typically harvested from the current
        :class:`LibraryView` rows + capped to ``_TAG_POOL_CAP``
        chips. Existing chips are removed and re-mounted with
        the new pool; chips whose tag is in the current
        ``LibraryFilters.tags`` start pre-checked.

        Idempotent — safe to call on every fetch. Suppression
        increments per pre-checked chip so the initial
        Checkbox.value=True doesn't fire a FiltersChanged for
        each re-mount.
        """
        deduped: list[str] = []
        for tag in tags:
            if tag and tag not in deduped:
                deduped.append(tag)
            if len(deduped) >= _TAG_POOL_CAP:
                break
        new_pool = tuple(deduped)
        # Skip the rebuild when the pool is unchanged — saves
        # the `remove_children` / `mount` churn that would
        # otherwise produce `DuplicateIds` on rapid refreshes
        # (mount is async; subsequent calls land before the
        # previous remove finishes).
        if new_pool == self._tag_pool:
            return
        self._tag_pool = new_pool
        if not self.is_mounted:
            return
        try:
            container = self.query_one(
                "#library-sidebar-tag-list", Vertical,
            )
        except Exception:
            return
        container.remove_children()
        active = self._filters.tag_set
        pool = self._tag_pool

        def _mount_chips() -> None:
            # `remove_children()` is async-deferred; mounting the new
            # chips synchronously right after races the removal and, when
            # the old + new pools share a tag, collides on the chip id
            # (`DuplicateIds`) — which stalls the message pump and times
            # out `pilot.pause()`. Defer the mount one refresh so the
            # removal settles first.
            try:
                target = self.query_one(
                    "#library-sidebar-tag-list", Vertical,
                )
            except Exception:
                return
            for tag in pool:
                # Pass `value=` in the constructor instead of post-mount
                # assignment — the post-mount setter fires a
                # Checkbox.Changed that interferes with pilot-driven
                # message-pump waits. Constructor-set value paints clean.
                target.mount(
                    Checkbox(tag, value=(tag in active), id=_tag_chip_id(tag)),
                )

        self.call_after_refresh(_mount_chips)

    @property
    def tag_pool(self) -> tuple[str, ...]:
        """Read-only snapshot of the rendered tag chips."""
        return self._tag_pool

    def set_collections(
        self, collections: tuple[Collection, ...] | list[Collection],
    ) -> None:
        """Replace the rendered collections list with a fresh
        snapshot. Idempotent — safe to call from the host
        screen's `_refresh` worker on every fetch."""
        self._collections = tuple(collections)
        if not self.is_mounted:
            return
        try:
            options = self.query_one(
                "#library-sidebar-collections", OptionList,
            )
        except Exception:
            return
        self._collection_suppress_count += 1
        options.clear_options()
        for opt in self._collection_options():
            options.add_option(opt)
        self._sync_collection_highlight()

    def _collection_options(self) -> list[Option]:
        opts: list[Option] = [
            Option(t("library.sidebar.allCollections"), id=_COLLECTIONS_ALL_ID),
        ]
        for collection in self._collections:
            label = (
                f"{collection.name} ({collection.member_count})"
                if collection.member_count
                else collection.name
            )
            opts.append(Option(label, id=collection.name))
        return opts

    def _sync_collection_highlight(self) -> None:
        if not self.is_mounted:
            return
        try:
            options = self.query_one(
                "#library-sidebar-collections", OptionList,
            )
        except Exception:
            return
        active = active_collection_name(self._filters)
        target = active if active else _COLLECTIONS_ALL_ID
        try:
            idx = options.get_option_index(target)
        except Exception:
            return
        if options.highlighted != idx:
            self._collection_suppress_count += 1
            options.highlighted = idx

    def focus_search(
        self, *, absorb_next_keystroke: bool = False,
    ) -> None:
        """Focus the search input — wired to Ctrl+F on the
        host screen.

        When ``absorb_next_keystroke`` is True, the next
        ``Input.Changed`` event with value ``"/"`` will be
        swallowed + the input reset to empty. This is for the
        ``/`` keybinding so the triggering character doesn't
        end up in the search prompt (vim-style search).
        """
        if not self.is_mounted:
            return
        if absorb_next_keystroke:
            self._absorb_next_search_keystroke = True
        self.query_one("#library-sidebar-search", Input).focus()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _consume_suppression(self) -> bool:
        """Decrement the suppression counter; return True
        when the event should be dropped (counter was > 0)."""
        if self._suppress_count > 0:
            self._suppress_count -= 1
            return True
        return False

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "library-sidebar-search":
            return
        # §4 P2 — when the LibraryScreen's `/` binding focuses
        # the search input, Textual redelivers the activating
        # `/` keystroke to the now-focused widget on the next
        # dispatch tick. Drop that single leading `/` so the
        # prompt opens empty.
        if (
            self._absorb_next_search_keystroke
            and event.value == "/"
        ):
            self._absorb_next_search_keystroke = False
            # The reset itself fires a secondary on_input_changed
            # with value=""; consume it via the suppress counter
            # so the empty search doesn't broadcast a no-op
            # FiltersChanged.
            self._suppress_count += 1
            event.input.value = ""
            return
        # Any other Changed clears the one-shot — we don't want
        # to swallow a literal `/` the user types later.
        self._absorb_next_search_keystroke = False
        if self._consume_suppression():
            return
        self._filters = with_search(self._filters, event.value)
        self.post_message(self.FiltersChanged(self._filters))

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id != "library-sidebar-status":
            return
        if self._consume_suppression():
            return
        pressed_id = event.pressed.id if event.pressed is not None else None
        new_status = _STATUS_BY_BUTTON_ID.get(pressed_id or "")
        self._filters = with_status(self._filters, new_status)
        self.post_message(self.FiltersChanged(self._filters))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        cb_id = event.checkbox.id
        if cb_id == "library-sidebar-favourites":
            if self._consume_suppression():
                return
            self._filters = with_favourites_only(
                self._filters, event.value,
            )
            self.post_message(self.FiltersChanged(self._filters))
            return
        if cb_id is not None and cb_id.startswith(_TAG_CHIP_ID_PREFIX):
            # §4 P0 — tag chip toggle. Suppression also drains
            # here so a `set_tag_pool(...)` rebuild that
            # pre-checks active chips doesn't broadcast a
            # FiltersChanged per pre-checked entry.
            if self._consume_suppression():
                return
            tag = self._tag_for_chip_id(cb_id)
            if tag is None:
                return
            active = set(self._filters.tags)
            if event.value:
                active.add(tag)
            else:
                active.discard(tag)
            new_filters = with_tags(self._filters, sorted(active))
            self._filters = new_filters
            self.post_message(self.FiltersChanged(self._filters))

    def _tag_for_chip_id(self, chip_id: str) -> str | None:
        """Reverse-map a tag chip widget id back to its source
        tag. Sanitisation in :func:`_tag_chip_id` isn't perfectly
        invertible (e.g. `:` → `_`), so we look the tag up in
        the current pool by comparing sanitised ids."""
        for tag in self._tag_pool:
            if _tag_chip_id(tag) == chip_id:
                return tag
        return None

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected,
    ) -> None:
        if event.option_list.id != "library-sidebar-collections":
            return
        if self._collection_suppress_count > 0:
            self._collection_suppress_count -= 1
            return
        option_id = event.option.id
        if option_id is None or option_id == _COLLECTIONS_ALL_ID:
            self.post_message(self.CollectionSelected(None))
        else:
            self.post_message(self.CollectionSelected(option_id))


__all__ = ["LibrarySidebar"]
