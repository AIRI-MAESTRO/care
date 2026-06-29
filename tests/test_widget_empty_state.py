"""Pilot tests for `EmptyStateView` (TODO §1.1 P0.9).

Mounts the widget inside a minimal host App, drives it via
`set_state`, and asserts the right Statics + Button render
across every shipped EmptyState template.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from care.runtime.empty_state import EmptyState, classify_empty_state
from care.widgets.empty_state import EmptyStateView


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _EmptyHostApp(App):
    def __init__(self, *, state: EmptyState | None = None) -> None:
        super().__init__()
        self._initial_state = state
        self.fired: list[str] = []

    def compose(self) -> ComposeResult:
        self.view = EmptyStateView(state=self._initial_state)
        yield self.view

    def on_empty_state_view_action_fired(
        self, event: EmptyStateView.ActionFired,
    ) -> None:
        self.fired.append(event.action_kind)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_state_none(self):
        view = EmptyStateView()
        assert view.state is None

    def test_explicit_state(self):
        state = classify_empty_state(None)
        view = EmptyStateView(state=state)
        assert view.state is state


# ---------------------------------------------------------------------------
# Compose: each shipped template renders the right widgets
# ---------------------------------------------------------------------------


class TestComposeNoLibrary:
    @pytest.mark.asyncio
    async def test_no_library_renders_cta(self):
        from care.runtime.library_view import LibraryView

        state = classify_empty_state(LibraryView())
        assert state.kind == "no_library"
        app = _EmptyHostApp(state=state)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Title + message + CTA button.
            assert app.view.query_one("#empty-state-title", Static) is not None
            assert (
                app.view.query_one("#empty-state-message", Static) is not None
            )
            cta = app.view.query_one("#empty-state-cta", Button)
            assert cta is not None
            assert str(cta.label) == "Create your first chain"
            # Hint surfaces for no_library.
            assert app.view.query_one("#empty-state-hint", Static) is not None


class TestComposeNoResults:
    @pytest.mark.asyncio
    async def test_no_results_renders_clear_filters_cta(self):
        from care.runtime.library_view import LibraryFilters, LibraryView

        state = classify_empty_state(
            LibraryView(),
            filters=LibraryFilters(search="storm"),
        )
        assert state.kind == "no_results"
        app = _EmptyHostApp(state=state)
        async with app.run_test() as pilot:
            await pilot.pause()
            cta = app.view.query_one("#empty-state-cta", Button)
            assert str(cta.label) == "Clear filters"


class TestComposeLoading:
    @pytest.mark.asyncio
    async def test_loading_no_cta(self):
        state = classify_empty_state(None, is_loading=True)
        assert state.kind == "loading"
        app = _EmptyHostApp(state=state)
        async with app.run_test() as pilot:
            await pilot.pause()
            # No CTA button (primary_action_kind == "noop").
            buttons = app.view.query(Button)
            assert len(buttons) == 0
            # Title still rendered.
            assert app.view.query_one("#empty-state-title", Static) is not None


class TestComposeError:
    @pytest.mark.asyncio
    async def test_error_renders_retry_and_detail(self):
        state = classify_empty_state(None, error="connection refused")
        assert state.kind == "error"
        app = _EmptyHostApp(state=state)
        async with app.run_test() as pilot:
            await pilot.pause()
            cta = app.view.query_one("#empty-state-cta", Button)
            assert str(cta.label) == "Retry"
            detail = app.view.query_one(
                "#empty-state-error-detail", Static,
            )
            # The Static is mounted; the model carries the
            # detail text.
            assert detail is not None
            assert "connection refused" in state.error_detail


class TestComposeEmpty:
    @pytest.mark.asyncio
    async def test_none_state_renders_nothing(self):
        # No state → no Statics, no buttons.
        app = _EmptyHostApp(state=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            statics = app.view.query(Static)
            buttons = app.view.query(Button)
            assert len(statics) == 0
            assert len(buttons) == 0


# ---------------------------------------------------------------------------
# set_state recomposes
# ---------------------------------------------------------------------------


class TestSetState:
    @pytest.mark.asyncio
    async def test_set_state_recomposes(self):
        from care.runtime.library_view import LibraryView

        no_lib = classify_empty_state(LibraryView())
        app = _EmptyHostApp(state=no_lib)
        async with app.run_test() as pilot:
            await pilot.pause()
            initial_cta = app.view.query_one("#empty-state-cta", Button)
            assert str(initial_cta.label) == "Create your first chain"
            # Swap to error state.
            err = classify_empty_state(None, error="503")
            app.view.set_state(err)
            await pilot.pause()
            await pilot.pause()
            new_cta = app.view.query_one("#empty-state-cta", Button)
            assert str(new_cta.label) == "Retry"

    def test_set_state_pre_mount_no_crash(self):
        view = EmptyStateView()
        state = classify_empty_state(None)
        view.set_state(state)
        assert view.state is state


# ---------------------------------------------------------------------------
# CTA button fires ActionFired
# ---------------------------------------------------------------------------


class TestActionFired:
    @pytest.mark.asyncio
    async def test_no_library_cta_fires_create_first_agent(self):
        from care.runtime.library_view import LibraryView

        state = classify_empty_state(LibraryView())
        app = _EmptyHostApp(state=state)
        async with app.run_test() as pilot:
            await pilot.pause()
            cta = app.view.query_one("#empty-state-cta", Button)
            cta.press()
            await pilot.pause()
            assert app.fired == ["create_first_agent"]

    @pytest.mark.asyncio
    async def test_no_results_cta_fires_clear_filters(self):
        from care.runtime.library_view import LibraryFilters, LibraryView

        state = classify_empty_state(
            LibraryView(), filters=LibraryFilters(search="x"),
        )
        app = _EmptyHostApp(state=state)
        async with app.run_test() as pilot:
            await pilot.pause()
            cta = app.view.query_one("#empty-state-cta", Button)
            cta.press()
            await pilot.pause()
            assert app.fired == ["clear_filters"]

    @pytest.mark.asyncio
    async def test_error_cta_fires_retry(self):
        state = classify_empty_state(None, error="boom")
        app = _EmptyHostApp(state=state)
        async with app.run_test() as pilot:
            await pilot.pause()
            cta = app.view.query_one("#empty-state-cta", Button)
            cta.press()
            await pilot.pause()
            assert app.fired == ["retry"]

    @pytest.mark.asyncio
    async def test_no_library_renders_secondary_back_to_chat(self):
        """TODO §4 P0 — the no-library card mounts a secondary
        button alongside the primary `Create first agent` CTA;
        clicking it fires the `back_to_chat` action so the
        LibraryScreen handler can return the user to chat."""
        from care.runtime.library_view import LibraryView

        state = classify_empty_state(LibraryView())
        assert state.secondary_action_kind == "back_to_chat"
        app = _EmptyHostApp(state=state)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Both buttons present.
            buttons = list(app.view.query(Button))
            ids = {b.id for b in buttons}
            assert "empty-state-cta" in ids
            assert "empty-state-cta-secondary" in ids
            secondary = app.view.query_one(
                "#empty-state-cta-secondary", Button,
            )
            assert str(secondary.label) == "Back to chat"
            secondary.press()
            await pilot.pause()
            assert app.fired == ["back_to_chat"]


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_widgets_re_exports_empty_state_view(self):
        from care.widgets import EmptyStateView as ReExported

        assert ReExported is EmptyStateView


# ---------------------------------------------------------------------------
# LibraryScreen integration
# ---------------------------------------------------------------------------


class TestLibraryScreenIntegration:
    @pytest.mark.asyncio
    async def test_no_memory_renders_no_library(self):
        from care.screens.library import LibraryScreen

        class _LibHost(App):
            memory = None

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(LibraryScreen())

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # Empty-state view rendered + visible.
            view = app.screen.query_one(EmptyStateView)
            assert view.state is not None
            assert view.state.kind == "no_library"
            assert view.display is True

    @pytest.mark.asyncio
    async def test_rows_present_hides_empty_state(self):
        from care.screens.library import LibraryScreen
        from textual.widgets import DataTable

        def _row(entity_id="a"):
            return {
                "entity_type": "chain",
                "entity_id": entity_id,
                "version_id": "v",
                "channel": "latest",
                "etag": "e",
                "favourite": False,
                "run_count": 0,
                "last_run_at": None,
                "display_name": "Alpha",
                "description": "d",
                "meta": {"tags": [], "name": "n"},
                "content": {"steps": []},
                "evolution_meta": None,
            }

        class _StubClient:
            def list_chains(self, **kw):
                return [_row()]

        class _StubMemory:
            def __init__(self):
                self.client = _StubClient()

        class _LibHost(App):
            memory = _StubMemory()

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(LibraryScreen())

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            view = app.screen.query_one(EmptyStateView)
            table = app.screen.query_one("#library-table", DataTable)
            # Rows present → empty-state hidden, table shown.
            assert view.display is False
            assert table.display is True
            assert table.row_count == 1

    @pytest.mark.asyncio
    async def test_clear_filters_action_resets_filters(self):
        from care.runtime.library_view import LibraryFilters
        from care.screens.library import LibraryScreen
        from care.widgets.library_sidebar import LibrarySidebar

        def _empty(**kw):
            return []

        class _StubClient:
            def list_chains(self, **kw):
                return _empty(**kw)

        class _StubMemory:
            def __init__(self):
                self.client = _StubClient()

        class _LibHost(App):
            memory = _StubMemory()

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(
                    LibraryScreen(filters=LibraryFilters(search="storm"))
                )

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # No results because filters return zero rows.
            view = app.screen.query_one(EmptyStateView)
            assert view.state is not None
            assert view.state.kind == "no_results"
            cta = view.query_one("#empty-state-cta", Button)
            cta.press()
            await pilot.pause()
            await pilot.pause()
            # Filters reset.
            assert app.screen.filters.search == ""
            sidebar = app.screen.query_one(LibrarySidebar)
            assert sidebar.filters.search == ""

    @pytest.mark.asyncio
    async def test_create_first_agent_logs_cta(self):
        from care.screens.library import LibraryScreen

        class _LibHost(App):
            memory = None

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(LibraryScreen())

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            library = app.screen_stack[-1]
            assert isinstance(library, LibraryScreen)
            cta = library.query_one("#empty-state-cta", Button)
            cta.press()
            await pilot.pause()
            assert "create_first_agent" in library._cta_log

    @pytest.mark.asyncio
    async def test_back_to_chat_secondary_routes_to_palette_opener(self):
        """TODO §4 P0 — the no-library card's secondary CTA
        fires `back_to_chat`. The LibraryScreen handler should
        log the gesture + invoke the host's
        `action_palette_open_chat` (so palette + slash + this
        CTA share one return-to-chat path). Spy on the host to
        confirm dispatch lands."""
        from care.screens.library import LibraryScreen

        calls: list[int] = []

        class _LibHost(App):
            memory = None

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(LibraryScreen())

            def action_palette_open_chat(self) -> None:  # noqa: D401
                calls.append(1)

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            library = app.screen_stack[-1]
            assert isinstance(library, LibraryScreen)
            secondary = library.query_one(
                "#empty-state-cta-secondary", Button,
            )
            secondary.press()
            await pilot.pause()
            assert calls == [1], (
                f"action_palette_open_chat should fire once; "
                f"got {len(calls)}"
            )
            assert "back_to_chat" in library._cta_log

    @pytest.mark.asyncio
    async def test_error_state_retry_reruns_fetch(self):
        from care.screens.library import LibraryScreen

        class _BoomClient:
            def __init__(self):
                self.calls = 0

            def list_chains(self, **kw):
                self.calls += 1
                raise RuntimeError("503")

        client = _BoomClient()

        class _StubMemory:
            def __init__(self):
                self.client = client

        class _LibHost(App):
            memory = _StubMemory()

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(LibraryScreen())

        app = _LibHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # Error state renders.
            view = app.screen.query_one(EmptyStateView)
            assert view.state.kind == "error"
            # Click retry → fetch fires again.
            cta = view.query_one("#empty-state-cta", Button)
            initial_calls = client.calls
            cta.press()
            await pilot.pause()
            await pilot.pause()
            assert client.calls > initial_calls
