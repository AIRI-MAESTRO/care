"""Pilot tests for `LibrarySidebar` (TODO §1.1 P0.8).

Mounts the widget inside a minimal host App, simulates user
interaction (typing in search, clicking radio buttons, toggling
the favourites checkbox) and asserts the `FiltersChanged`
message fires with the expected `LibraryFilters` snapshot.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Input, RadioButton

from care.runtime.library_view import LibraryFilters
from care.widgets.library_sidebar import LibrarySidebar


# ---------------------------------------------------------------------------
# Host App + message recorder
# ---------------------------------------------------------------------------


class _SidebarHostApp(App):
    """Mounts `LibrarySidebar` on boot and records every
    FiltersChanged message that bubbles up."""

    def __init__(self, *, filters: LibraryFilters | None = None) -> None:
        super().__init__()
        self._initial_filters = filters
        self.observed: list[LibraryFilters] = []

    def compose(self) -> ComposeResult:
        self.sidebar = LibrarySidebar(filters=self._initial_filters)
        yield self.sidebar

    def on_library_sidebar_filters_changed(
        self, event: LibrarySidebar.FiltersChanged,
    ) -> None:
        self.observed.append(event.filters)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_filters(self):
        sidebar = LibrarySidebar()
        assert sidebar.filters == LibraryFilters()

    def test_explicit_filters(self):
        filters = LibraryFilters(search="weather", favourites_only=True)
        sidebar = LibrarySidebar(filters=filters)
        assert sidebar.filters is filters

    @pytest.mark.asyncio
    async def test_initial_widgets_reflect_filters(self):
        filters = LibraryFilters(
            search="weather",
            favourites_only=True,
            status="evolved",
        )
        app = _SidebarHostApp(filters=filters)
        async with app.run_test() as pilot:
            await pilot.pause()
            search = app.sidebar.query_one("#library-sidebar-search", Input)
            fav = app.sidebar.query_one(
                "#library-sidebar-favourites", Checkbox,
            )
            evolved = app.sidebar.query_one(
                "#library-sidebar-status-evolved", RadioButton,
            )
            assert search.value == "weather"
            assert fav.value is True
            assert evolved.value is True


# ---------------------------------------------------------------------------
# Search input emits FiltersChanged
# ---------------------------------------------------------------------------


class TestSearchInput:
    @pytest.mark.asyncio
    async def test_typing_updates_filters(self):
        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            search = app.sidebar.query_one("#library-sidebar-search", Input)
            search.value = "storm"
            await pilot.pause()
            assert app.observed
            assert app.observed[-1].search == "storm"

    @pytest.mark.asyncio
    async def test_focus_search_method(self):
        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.sidebar.focus_search()
            await pilot.pause()
            search = app.sidebar.query_one("#library-sidebar-search", Input)
            assert search.has_focus


# ---------------------------------------------------------------------------
# Status radio set
# ---------------------------------------------------------------------------


class TestStatusRadio:
    @pytest.mark.asyncio
    async def test_selecting_draft_updates_filters(self):
        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            draft = app.sidebar.query_one(
                "#library-sidebar-status-draft", RadioButton,
            )
            draft.value = True
            await pilot.pause()
            assert app.observed
            assert app.observed[-1].status == "draft"

    @pytest.mark.asyncio
    async def test_selecting_evolved_updates_filters(self):
        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            evolved = app.sidebar.query_one(
                "#library-sidebar-status-evolved", RadioButton,
            )
            evolved.value = True
            await pilot.pause()
            assert app.observed[-1].status == "evolved"

    @pytest.mark.asyncio
    async def test_selecting_all_clears_status(self):
        # Start with status="draft"; switch to all.
        app = _SidebarHostApp(filters=LibraryFilters(status="draft"))
        async with app.run_test() as pilot:
            await pilot.pause()
            all_button = app.sidebar.query_one(
                "#library-sidebar-status-all", RadioButton,
            )
            all_button.value = True
            await pilot.pause()
            assert app.observed[-1].status is None


# ---------------------------------------------------------------------------
# Favourites checkbox
# ---------------------------------------------------------------------------


class TestTagChips:
    """TODO §4 P0 — tag chips populate from
    `set_tag_pool(...)` + toggle `LibraryFilters.tags`.
    """

    @pytest.mark.asyncio
    async def test_set_tag_pool_mounts_one_checkbox_per_tag(self):
        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.sidebar.set_tag_pool(["alpha", "beta", "gamma"])
            await pilot.pause()
            assert app.sidebar.tag_pool == ("alpha", "beta", "gamma")
            # One Checkbox per tag inside the dedicated container.
            chips = list(
                app.sidebar.query("#library-sidebar-tag-list Checkbox")
            )
            labels = {str(cb.label) for cb in chips}
            assert labels == {"alpha", "beta", "gamma"}

    @pytest.mark.asyncio
    async def test_set_tag_pool_deduplicates_and_caps(self):
        from care.widgets.library_sidebar import _TAG_POOL_CAP

        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Pool larger than the cap + dupes — output is
            # deduped and truncated to the cap.
            pool = (
                ["weather"] * 3
                + [f"t{i}" for i in range(_TAG_POOL_CAP + 5)]
            )
            app.sidebar.set_tag_pool(pool)
            await pilot.pause()
            assert len(app.sidebar.tag_pool) == _TAG_POOL_CAP
            # First entry survives (dedup preserves first-seen
            # order).
            assert app.sidebar.tag_pool[0] == "weather"

    @pytest.mark.asyncio
    async def test_toggling_chip_updates_filters_tags(self):
        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.sidebar.set_tag_pool(["weather", "finance"])
            await pilot.pause()
            chip = app.sidebar.query_one(
                "#library-sidebar-tag-weather", Checkbox,
            )
            chip.value = True
            await pilot.pause()
            assert app.observed
            assert app.observed[-1].tag_set == frozenset({"weather"})
            # Toggle a second on.
            chip2 = app.sidebar.query_one(
                "#library-sidebar-tag-finance", Checkbox,
            )
            chip2.value = True
            await pilot.pause()
            assert app.observed[-1].tag_set == frozenset(
                {"weather", "finance"},
            )
            # Toggle the first off.
            chip.value = False
            await pilot.pause()
            assert app.observed[-1].tag_set == frozenset({"finance"})

    @pytest.mark.asyncio
    async def test_active_tags_start_checked_after_set_tag_pool(self):
        """When tags are already in `LibraryFilters.tags`,
        the matching chip mounts pre-checked + no synthetic
        FiltersChanged is broadcast for it (suppression
        consumes the value setter event)."""
        app = _SidebarHostApp(
            filters=LibraryFilters(tags=("weather",)),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.observed == []
            app.sidebar.set_tag_pool(["weather", "finance"])
            await pilot.pause()
            chip = app.sidebar.query_one(
                "#library-sidebar-tag-weather", Checkbox,
            )
            assert chip.value is True
            # No synthetic FiltersChanged for the suppressed
            # pre-check (the only event in `observed` would be
            # one we didn't want).
            assert app.observed == []

    @pytest.mark.asyncio
    async def test_chip_id_sanitises_colons_in_tag_names(self):
        """Tags like `"domain:weather"` get their `:` replaced
        with `_` so the resulting widget id is Textual-safe.
        The toggle round-trip still updates filters with the
        original tag (not the sanitised id)."""
        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.sidebar.set_tag_pool(["domain:weather"])
            await pilot.pause()
            chip = app.sidebar.query_one(
                "#library-sidebar-tag-domain_weather", Checkbox,
            )
            chip.value = True
            await pilot.pause()
            assert app.observed[-1].tag_set == frozenset(
                {"domain:weather"},
            )


class TestFavouritesCheckbox:
    @pytest.mark.asyncio
    async def test_toggle_emits_filters_changed(self):
        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            fav = app.sidebar.query_one(
                "#library-sidebar-favourites", Checkbox,
            )
            fav.value = True
            await pilot.pause()
            assert app.observed
            assert app.observed[-1].favourites_only is True

    @pytest.mark.asyncio
    async def test_toggle_off(self):
        app = _SidebarHostApp(
            filters=LibraryFilters(favourites_only=True),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            fav = app.sidebar.query_one(
                "#library-sidebar-favourites", Checkbox,
            )
            fav.value = False
            await pilot.pause()
            assert app.observed[-1].favourites_only is False


# ---------------------------------------------------------------------------
# set_filters external sync
# ---------------------------------------------------------------------------


class TestSetFiltersExternal:
    @pytest.mark.asyncio
    async def test_set_filters_does_not_emit_event(self):
        # `set_filters` is for host-driven sync; emitting an
        # event would loop infinitely against `refresh_library`.
        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            initial_count = len(app.observed)
            app.sidebar.set_filters(
                LibraryFilters(search="external", favourites_only=True)
            )
            await pilot.pause()
            await pilot.pause()
            assert len(app.observed) == initial_count

    @pytest.mark.asyncio
    async def test_set_filters_updates_widgets(self):
        app = _SidebarHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.sidebar.set_filters(
                LibraryFilters(
                    search="external",
                    favourites_only=True,
                    status="evolved",
                )
            )
            await pilot.pause()
            search = app.sidebar.query_one("#library-sidebar-search", Input)
            fav = app.sidebar.query_one(
                "#library-sidebar-favourites", Checkbox,
            )
            evolved = app.sidebar.query_one(
                "#library-sidebar-status-evolved", RadioButton,
            )
            assert search.value == "external"
            assert fav.value is True
            assert evolved.value is True

    def test_set_filters_pre_mount_no_crash(self):
        sidebar = LibrarySidebar()
        sidebar.set_filters(LibraryFilters(search="x"))
        assert sidebar.filters.search == "x"


# ---------------------------------------------------------------------------
# LibraryScreen consumption
# ---------------------------------------------------------------------------


class TestLibraryScreenIntegration:
    @pytest.mark.asyncio
    async def test_screen_reacts_to_sidebar_filters_changed(self):
        # Building the full LibraryScreen consumes the
        # FiltersChanged message, updates its `filters`, and
        # re-runs the worker.
        from care.screens.library import LibraryScreen

        class _StubClient:
            def __init__(self):
                self.calls = []

            def list_chains(self, **kw):
                self.calls.append(kw)
                return []

        class _StubMemory:
            def __init__(self, client):
                self.client = client

        class _LibraryHostApp(App):
            def __init__(self, *, memory):
                super().__init__()
                self.memory = memory

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(LibraryScreen())

        client = _StubClient()
        app = _LibraryHostApp(memory=_StubMemory(client))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            initial_calls = len(client.calls)
            # Drive a sidebar filter change.
            sidebar = app.screen.query_one(LibrarySidebar)
            search = sidebar.query_one("#library-sidebar-search", Input)
            search.value = "weather"
            await pilot.pause()
            await pilot.pause()
            # Screen's filters now reflect the search; another
            # fetch fired.
            assert app.screen.filters.search == "weather"
            assert len(client.calls) > initial_calls
            # At least one call carries the search arg (the
            # filtered fetch — the P0.14 collections aggregator
            # also calls list_chains without `q`, so don't pin
            # on `calls[-1]`).
            assert any(c.get("q") == "weather" for c in client.calls)

    @pytest.mark.asyncio
    async def test_ctrl_f_focuses_search(self):
        from care.screens.library import LibraryScreen

        class _StubMemory:
            class client:
                @staticmethod
                def list_chains(**kw):
                    return []

        class _LibraryHostApp(App):
            def __init__(self, *, memory):
                super().__init__()
                self.memory = memory

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(LibraryScreen())

        app = _LibraryHostApp(memory=_StubMemory())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("ctrl+f")
            await pilot.pause()
            sidebar = app.screen.query_one(LibrarySidebar)
            search = sidebar.query_one("#library-sidebar-search", Input)
            assert search.has_focus


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_widgets_re_exports_library_sidebar(self):
        from care.widgets import LibrarySidebar as ReExported

        assert ReExported is LibrarySidebar
