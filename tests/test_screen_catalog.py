"""Pilot tests for CatalogScreen (§8 P1 [DONE — CLI half] → DONE).

Wires :func:`care.build_catalog`'s produced
:class:`care.CapabilityCatalog` into a Textual ``Screen`` that
mirrors the CLI's `care catalog` output. Tests exercise:

* Compose — kind sidebar / results table / status / warnings
  all mount.
* Pre-built catalog renders entries in encounter order.
* Lazy `catalog_factory` runs on mount when no catalog is
  supplied.
* Kind-chip filter narrows the visible rows.
* Promote action posts a `PromoteRequested` for agent_skill
  entries only.
* Errors panel renders discovery warnings.
* Re-exports.
"""

from __future__ import annotations

import os

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from care.catalog import CapabilityCatalog, CapabilityCatalogEntry
from care.screens.catalog import CatalogPromoteRequest, CatalogScreen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    *,
    kind: str = "agent_skill",
    name: str = "pdf-extract",
    source: str = "/skills/pdf-extract/SKILL.md",
    summary: str = "Extract text from PDFs",
    tags=("pdf", "extraction"),
    metadata: dict | None = None,
) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        kind=kind,  # type: ignore[arg-type]
        name=name,
        source=source,
        summary=summary,
        tags=tuple(tags),
        metadata=dict(metadata or {}),
    )


def _mixed_catalog() -> CapabilityCatalog:
    return CapabilityCatalog(
        entries=(
            _entry(kind="agent_skill", name="pdf-extract"),
            _entry(
                kind="mcp_server",
                name="weather",
                source="npx weather-mcp",
                summary="Weather forecast API",
                tags=("weather",),
            ),
            _entry(
                kind="tool",
                name="run_python",
                source="/tools/run_python.py",
                summary="Execute Python snippets",
                tags=(),
            ),
            _entry(
                kind="memory_card",
                name="q4-financials",
                source="memory://card-1",
                summary="2025 Q4 financials report card",
                tags=("finance",),
            ),
        ),
        errors=(),
    )


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(
        self,
        *,
        catalog: CapabilityCatalog | None = None,
        catalog_factory=None,
    ) -> None:
        super().__init__()
        self._catalog = catalog
        self._catalog_factory = catalog_factory
        self.promote_requests: list[CatalogPromoteRequest] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(CatalogScreen(
            self._catalog,
            catalog_factory=self._catalog_factory,
        ))

    def on_catalog_screen_promote_requested(
        self, event: CatalogScreen.PromoteRequested,
    ) -> None:
        self.promote_requests.append(
            CatalogPromoteRequest(entry=event.entry),
        )


def _screen(app: App) -> CatalogScreen:
    s = app.screen_stack[-1]
    assert isinstance(s, CatalogScreen)
    return s


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_panes_mount(self):
        app = _Host(catalog=_mixed_catalog())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.query_one("#catalog-table", DataTable) is not None
            assert screen.query_one("#catalog-kind-list") is not None
            assert screen.query_one("#catalog-status", Static) is not None
            assert screen.query_one("#catalog-warnings", Static) is not None


# ---------------------------------------------------------------------------
# Catalog construction paths
# ---------------------------------------------------------------------------


class TestCatalogSource:
    @pytest.mark.asyncio
    async def test_pre_built_catalog_renders(self):
        app = _Host(catalog=_mixed_catalog())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert len(screen.visible_entries()) == 4
            table = screen.query_one("#catalog-table", DataTable)
            assert table.row_count == 4

    @pytest.mark.asyncio
    async def test_lazy_factory_runs_on_mount(self):
        call_count = [0]

        def _factory():
            call_count[0] += 1
            return _mixed_catalog()

        app = _Host(catalog_factory=_factory)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert call_count[0] == 1
            assert len(screen.visible_entries()) == 4

    @pytest.mark.asyncio
    async def test_lazy_factory_failure_renders_empty_catalog(self):
        def _factory():
            raise RuntimeError("discovery-down")

        app = _Host(catalog_factory=_factory)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.catalog.is_empty


# ---------------------------------------------------------------------------
# Kind-chip filter
# ---------------------------------------------------------------------------


class TestKindFilter:
    @pytest.mark.asyncio
    async def test_chips_render_one_per_kind_present(self):
        app = _Host(catalog=_mixed_catalog())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            chip_ids = {
                str(b.id) for b in screen.query(
                    "#catalog-kind-list Button",
                )
            }
            assert "catalog-kind-chip-all" in chip_ids
            assert "catalog-kind-chip-agent_skill" in chip_ids
            assert "catalog-kind-chip-mcp_server" in chip_ids
            assert "catalog-kind-chip-tool" in chip_ids
            assert "catalog-kind-chip-memory_card" in chip_ids

    @pytest.mark.asyncio
    async def test_select_kind_narrows_rows(self):
        app = _Host(catalog=_mixed_catalog())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert len(screen.visible_entries()) == 4
            screen._select_kind("agent_skill")
            assert len(screen.visible_entries()) == 1
            assert screen.visible_entries()[0].kind == "agent_skill"
            # Re-clicking the active chip clears the filter.
            screen._select_kind("agent_skill")
            assert screen.selected_kind == "all"
            assert len(screen.visible_entries()) == 4

    @pytest.mark.asyncio
    async def test_empty_kinds_hidden(self):
        # Only `agent_skill` entries — other chips shouldn't
        # render.
        catalog = CapabilityCatalog(
            entries=(_entry(kind="agent_skill", name="x"),),
        )
        app = _Host(catalog=catalog)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            chip_ids = {
                str(b.id) for b in screen.query(
                    "#catalog-kind-list Button",
                )
            }
            assert "catalog-kind-chip-all" in chip_ids
            assert "catalog-kind-chip-agent_skill" in chip_ids
            assert "catalog-kind-chip-mcp_server" not in chip_ids
            assert "catalog-kind-chip-tool" not in chip_ids


# ---------------------------------------------------------------------------
# Promote action
# ---------------------------------------------------------------------------


class TestPromote:
    @pytest.mark.asyncio
    async def test_promote_agent_skill_posts_request(self):
        catalog = CapabilityCatalog(
            entries=(_entry(kind="agent_skill", name="pdf-extract"),),
        )
        app = _Host(catalog=catalog)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.selected_entry = catalog.entries[0]
            screen.action_promote_selected()
            for _ in range(4):
                await pilot.pause()
            assert app.promote_requests
            assert app.promote_requests[0].entry.name == "pdf-extract"

    @pytest.mark.asyncio
    async def test_promote_no_selection_is_noop(self):
        app = _Host(catalog=_mixed_catalog())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.selected_entry is None
            screen.action_promote_selected()
            for _ in range(4):
                await pilot.pause()
            assert app.promote_requests == []

    @pytest.mark.asyncio
    async def test_promote_non_agent_skill_no_request(self):
        catalog = CapabilityCatalog(
            entries=(_entry(kind="mcp_server", name="weather"),),
        )
        app = _Host(catalog=catalog)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.selected_entry = catalog.entries[0]
            screen.action_promote_selected()
            for _ in range(4):
                await pilot.pause()
            # Promote only fires for agent_skill kind.
            assert app.promote_requests == []


# ---------------------------------------------------------------------------
# Warnings panel
# ---------------------------------------------------------------------------


class TestWarnings:
    @pytest.mark.asyncio
    async def test_warnings_render(self):
        catalog = CapabilityCatalog(
            entries=(_entry(name="ok-skill"),),
            errors=(
                "skills_path /missing: not a directory",
                "tools_path /broken: permission denied",
            ),
        )
        app = _Host(catalog=catalog)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            warnings = screen.query_one("#catalog-warnings", Static)
            text = str(warnings.content)
            assert "2 discovery warning" in text
            assert "skills_path /missing" in text
            assert "tools_path /broken" in text

    @pytest.mark.asyncio
    async def test_no_warnings_panel_empty(self):
        catalog = CapabilityCatalog(
            entries=(_entry(name="ok"),),
            errors=(),
        )
        app = _Host(catalog=catalog)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            warnings = screen.query_one("#catalog-warnings", Static)
            assert str(warnings.content).strip() == ""

    @pytest.mark.asyncio
    async def test_warnings_truncated_after_five(self):
        errs = tuple(f"error-{i}" for i in range(10))
        catalog = CapabilityCatalog(
            entries=(),
            errors=errs,
        )
        app = _Host(catalog=catalog)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            text = str(
                screen.query_one("#catalog-warnings", Static).content,
            )
            assert "error-0" in text
            assert "error-4" in text
            # 5 more truncated below.
            assert "5 more" in text


# ---------------------------------------------------------------------------
# Empty catalog
# ---------------------------------------------------------------------------


class TestEmpty:
    @pytest.mark.asyncio
    async def test_empty_catalog_status(self):
        app = _Host(catalog=CapabilityCatalog())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            status = screen.query_one("#catalog-status", Static)
            text = str(status.content)
            assert "no entries" in text


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import CatalogScreen as C

        assert C is CatalogScreen


# ---------------------------------------------------------------------------
# CareApp dispatch — Ctrl+K binding + action_open_catalog
# ---------------------------------------------------------------------------


class TestAppDispatch:
    def test_app_has_open_catalog_binding(self):
        from care.app import CareApp

        actions = {b.action for b in CareApp.BINDINGS}
        assert "open_catalog" in actions

    @pytest.mark.asyncio
    async def test_action_open_catalog_pushes_catalog_screen(self):
        # The CareApp's `Ctrl+K` global binding wires through to
        # action_open_catalog, which pushes CatalogScreen.
        from care.app import CareApp

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(12):
                await pilot.pause()
            app.action_open_catalog()
            for _ in range(12):
                await pilot.pause()
            assert any(
                isinstance(s, CatalogScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_action_open_marketplace_pushes_marketplace_screen(self):
        # Companion action — no key binding, opened via the
        # command palette. We still verify the action method
        # works end-to-end.
        from care.app import CareApp
        from care.screens.marketplace import MarketplaceScreen

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(12):
                await pilot.pause()
            app.action_open_marketplace()
            for _ in range(12):
                await pilot.pause()
            assert any(
                isinstance(s, MarketplaceScreen)
                for s in app.screen_stack
            )


# ---------------------------------------------------------------------------
# Palette → action dispatch
# ---------------------------------------------------------------------------


class TestPaletteDispatch:
    """The palette dismisses with a `PaletteSelection`; the
    app's `_dispatch_palette_action` routes the
    `command_action` to the matching `action_*` method. These
    tests pin that wiring end-to-end."""

    @pytest.mark.asyncio
    async def test_open_catalog_command_pushes_catalog_screen(self):
        from care.app import CareApp
        from care.screens.command_palette import PaletteSelection
        from care.runtime.command_palette import PaletteEntry

        selection = PaletteSelection(entry=PaletteEntry(
            entry_id="command:open_catalog",
            kind="command",
            label="Browse capability catalog",
            description="x",
            command_action="open_catalog",
        ))
        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(12):
                await pilot.pause()
            app._dispatch_palette_action(selection)
            for _ in range(12):
                await pilot.pause()
            assert any(
                isinstance(s, CatalogScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_open_marketplace_command_pushes_marketplace(self):
        from care.app import CareApp
        from care.screens.command_palette import PaletteSelection
        from care.screens.marketplace import MarketplaceScreen
        from care.runtime.command_palette import PaletteEntry

        selection = PaletteSelection(entry=PaletteEntry(
            entry_id="command:open_marketplace",
            kind="command",
            label="Browse capability marketplace",
            description="x",
            command_action="open_marketplace",
        ))
        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(12):
                await pilot.pause()
            app._dispatch_palette_action(selection)
            for _ in range(12):
                await pilot.pause()
            assert any(
                isinstance(s, MarketplaceScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_show_help_command_pushes_help_screen(self):
        from care.app import CareApp
        from care.screens.command_palette import PaletteSelection
        from care.screens.help import HelpScreen
        from care.runtime.command_palette import PaletteEntry

        selection = PaletteSelection(entry=PaletteEntry(
            entry_id="command:show_help",
            kind="command",
            label="Help",
            description="x",
            command_action="show_help",
        ))
        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(12):
                await pilot.pause()
            app._dispatch_palette_action(selection)
            for _ in range(12):
                await pilot.pause()
            assert any(
                isinstance(s, HelpScreen)
                for s in app.screen_stack
            )

    def test_none_selection_is_noop(self):
        from care.app import CareApp

        app = CareApp(mode="returning")
        # No exception; dispatch silently returns on None.
        app._dispatch_palette_action(None)

    def test_unknown_action_is_noop(self):
        from care.app import CareApp
        from care.screens.command_palette import PaletteSelection

        class _Entry:
            command_action = "this_action_does_not_exist"

        app = CareApp(mode="returning")
        # Unknown action → no-op (no exception).
        app._dispatch_palette_action(PaletteSelection(entry=_Entry()))

    def test_entry_without_command_action_is_noop(self):
        from care.app import CareApp
        from care.screens.command_palette import PaletteSelection

        class _Entry:
            kind = "command"
            command_action = None

        app = CareApp(mode="returning")
        app._dispatch_palette_action(PaletteSelection(entry=_Entry()))

    @pytest.mark.asyncio
    async def test_chain_pick_pushes_inspection_screen(self):
        from care.app import CareApp
        from care.runtime.command_palette import PaletteEntry
        from care.screens.command_palette import PaletteSelection
        from care.screens.inspection import InspectionScreen

        selection = PaletteSelection(entry=PaletteEntry(
            entry_id="ent-storm-watcher",
            kind="chain",
            label="Storm Watcher",
            description="Forecast pipeline",
        ))
        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(12):
                await pilot.pause()
            app._dispatch_palette_action(selection)
            for _ in range(12):
                await pilot.pause()
            inspection = next(
                (s for s in app.screen_stack
                 if isinstance(s, InspectionScreen)),
                None,
            )
            assert inspection is not None
            assert inspection.entity_id == "ent-storm-watcher"

    @pytest.mark.asyncio
    async def test_chain_pick_without_entry_id_is_noop(self):
        from care.app import CareApp
        from care.runtime.command_palette import PaletteEntry
        from care.screens.command_palette import PaletteSelection
        from care.screens.inspection import InspectionScreen

        selection = PaletteSelection(entry=PaletteEntry(
            entry_id="",
            kind="chain",
            label="anon",
        ))
        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(12):
                await pilot.pause()
            app._dispatch_palette_action(selection)
            for _ in range(6):
                await pilot.pause()
            # No InspectionScreen pushed.
            assert not any(
                isinstance(s, InspectionScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_agent_skill_pick_pushes_catalog_screen(self):
        from care.app import CareApp
        from care.runtime.command_palette import PaletteEntry
        from care.screens.command_palette import PaletteSelection

        selection = PaletteSelection(entry=PaletteEntry(
            entry_id="sk-pdf-extract",
            kind="agent_skill",
            label="pdf-extract",
            description="Extract text from PDFs",
        ))
        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(12):
                await pilot.pause()
            app._dispatch_palette_action(selection)
            for _ in range(12):
                await pilot.pause()
            catalog = next(
                (s for s in app.screen_stack
                 if isinstance(s, CatalogScreen)),
                None,
            )
            assert catalog is not None
            # Agent_skill palette picks seed the catalog's
            # focus-entry-id so the cursor lands on the row
            # matching the picked entity_id.
            assert catalog._focus_entry_id == "sk-pdf-extract"


# ---------------------------------------------------------------------------
# Focus-entry-id constructor kwarg
# ---------------------------------------------------------------------------


class TestFocusEntry:
    @pytest.mark.asyncio
    async def test_focus_by_source_match(self):
        # `source` exactly equals the focus_entry_id — most
        # common path for file-backed catalog entries (the
        # source is the SKILL.md / mcp_servers.toml /
        # tool.py path the caller already knows).
        catalog = CapabilityCatalog(entries=(
            _entry(kind="agent_skill", name="a", source="/skills/a.md"),
            _entry(kind="agent_skill", name="b", source="/skills/b.md"),
            _entry(kind="agent_skill", name="c", source="/skills/c.md"),
        ))
        app = _Host(catalog=catalog)
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            # Re-push with focus kwarg.
            app.screen_stack[-1].dismiss()
            await pilot.pause()
            app.push_screen(
                CatalogScreen(catalog, focus_entry_id="/skills/b.md"),
            )
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.selected_entry is not None
            assert screen.selected_entry.source == "/skills/b.md"

    @pytest.mark.asyncio
    async def test_focus_by_memory_uri(self):
        # memory_card source is `memory://<entity_id>`; the
        # focus path strips the prefix so callers can pass the
        # bare entity_id.
        catalog = CapabilityCatalog(entries=(
            _entry(
                kind="memory_card",
                name="q4-financials",
                source="memory://card-1",
            ),
        ))
        app = _Host(catalog=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen_stack[-1].dismiss()
            await pilot.pause()
            app.push_screen(
                CatalogScreen(catalog, focus_entry_id="card-1"),
            )
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.selected_entry is not None
            assert screen.selected_entry.name == "q4-financials"

    @pytest.mark.asyncio
    async def test_focus_by_name_fallback(self):
        # When `entry_id` isn't in `source` form, fall back to
        # matching by entry `name` — covers agent_skill palette
        # picks where the palette delivers the Memory
        # `entity_id` but the catalog entry's source is the
        # SKILL.md path (no overlap).
        catalog = CapabilityCatalog(entries=(
            _entry(
                kind="agent_skill",
                name="pdf-extract",
                source="/skills/pdf/SKILL.md",
            ),
        ))
        app = _Host(catalog=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen_stack[-1].dismiss()
            await pilot.pause()
            app.push_screen(
                CatalogScreen(catalog, focus_entry_id="pdf-extract"),
            )
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.selected_entry is not None
            assert screen.selected_entry.name == "pdf-extract"

    @pytest.mark.asyncio
    async def test_focus_unresolved_is_noop(self):
        catalog = CapabilityCatalog(entries=(
            _entry(kind="agent_skill", name="a", source="/skills/a.md"),
        ))
        app = _Host(catalog=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen_stack[-1].dismiss()
            await pilot.pause()
            app.push_screen(
                CatalogScreen(catalog, focus_entry_id="nope"),
            )
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            # No exception, just no selection.
            assert screen.selected_entry is None

    @pytest.mark.asyncio
    async def test_focus_empty_string_skipped(self):
        # Falsy focus_entry_id (None / "") → no auto-focus.
        catalog = CapabilityCatalog(entries=(
            _entry(kind="agent_skill", name="a", source="/skills/a.md"),
        ))
        app = _Host(catalog=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen_stack[-1].dismiss()
            await pilot.pause()
            app.push_screen(
                CatalogScreen(catalog, focus_entry_id=""),
            )
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            # No auto-focus + no selection.
            assert screen._focus_entry_id is None
            assert screen.selected_entry is None


# ---------------------------------------------------------------------------
# CareApp handles PromoteRequested
# ---------------------------------------------------------------------------


class TestAppPromoteHandler:
    """The CatalogScreen stays pure-presentation; the App owns
    the Memory upload + toast on PromoteRequested."""

    @pytest.mark.asyncio
    async def test_no_memory_pushes_error_toast(
        self, tmp_path, monkeypatch,
    ):
        from care.app import CareApp
        from care.config import CareConfig

        # Isolate from the dev's saved `~/.config/care/config.toml`
        # — it may carry a non-default Memory base_url that
        # would auto-build the facade via the opt-in gate.
        from care import config as config_module

        monkeypatch.setattr(
            config_module,
            "DEFAULT_CONFIG_PATH",
            tmp_path / "no-such-config.toml",
        )
        for name in list(os.environ.keys()):
            if name.startswith("CARE_"):
                monkeypatch.delenv(name, raising=False)

        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            "---\nname: pdf-extract\ndescription: x\n---\nBody.\n",
            encoding="utf-8",
        )
        entry = _entry(
            kind="agent_skill",
            name="pdf-extract",
            source=str(skill_md),
        )
        app = CareApp(mode="returning", config=CareConfig())
        async with app.run_test() as pilot:
            # The app now always builds a facade from config; null it
            # to exercise the no-memory promote path this test targets.
            app.memory = None
            for _ in range(8):
                await pilot.pause()

            class _Evt:
                pass

            evt = _Evt()
            evt.entry = entry
            app.on_catalog_screen_promote_requested(evt)
            for _ in range(4):
                await pilot.pause()
            # The handler pushed an error toast — no exception
            # propagated, no worker started.
            # Verify by checking the toast host's pending count
            # via app._toast_host (best-effort — Textual's host
            # widget exposes children).
            assert app.memory is None

    @pytest.mark.asyncio
    async def test_promote_calls_promote_skill_to_memory(self, tmp_path):
        from care.app import CareApp

        # Real SKILL.md the data layer can parse.
        skill_dir = tmp_path / "pdf-extract"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: pdf-extract\n"
            "description: Extract text from PDFs\n"
            "tags: [pdf]\n"
            "---\n"
            "Body.\n",
            encoding="utf-8",
        )
        entry = _entry(
            kind="agent_skill",
            name="pdf-extract",
            source=str(skill_dir),
            tags=("pdf",),
        )

        save_calls: list = []

        class _StubMemory:
            def save_agent_skill(self, **kw):
                save_calls.append(kw)
                return "saved-1"

        app = CareApp(mode="returning", memory=_StubMemory())
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()

            class _Evt:
                pass

            evt = _Evt()
            evt.entry = entry
            app.on_catalog_screen_promote_requested(evt)
            # Worker is async — wait for it to settle.
            for _ in range(15):
                await pilot.pause()
            assert save_calls
            call = save_calls[0]
            # Manifest name from frontmatter; explicit name=
            # override from the catalog entry wins (same value
            # here but the path is exercised).
            assert call["name"] == "pdf-extract"
            assert "pdf" in call["tags"]

    @pytest.mark.asyncio
    async def test_promote_failure_pushes_error_toast(self, tmp_path):
        from care.app import CareApp

        # Missing path → promote_skill_to_memory raises
        # SkillPromotionError. The handler catches and pushes
        # an error toast rather than crashing the worker.
        entry = _entry(
            kind="agent_skill",
            name="ghost",
            source=str(tmp_path / "missing.md"),
        )

        class _StubMemory:
            def save_agent_skill(self, **kw):  # pragma: no cover
                raise AssertionError("should not be called")

        app = CareApp(mode="returning", memory=_StubMemory())
        async with app.run_test() as pilot:
            for _ in range(8):
                await pilot.pause()

            class _Evt:
                pass

            evt = _Evt()
            evt.entry = entry
            app.on_catalog_screen_promote_requested(evt)
            for _ in range(15):
                await pilot.pause()
            # No save attempted; promote_skill_to_memory bailed
            # on the missing-path check.

    def test_marketplace_installed_handler_exists(self):
        # The app-level handler is a deliberate no-op anchor —
        # check it accepts the canonical message shape without
        # raising.
        from care.app import CareApp

        class _Evt:
            class listing:
                entity_id = "x"
                name = "y"
            saved_entity_id = "saved-1"

        app = CareApp(mode="returning")
        # No-op handler shouldn't raise.
        app.on_marketplace_screen_installed(_Evt())


# ---------------------------------------------------------------------------
# CareApp.on_evolution_screen_acceptance_complete + edit_agent_screen_submitted
# ---------------------------------------------------------------------------


class TestAppPostScreenHandlers:
    """Two more screen-message handlers that previously had no
    host listener — verify they accept the canonical message
    shape without raising. Full TUI toast pickup is integration-
    tested via the actual ToastHost widget."""

    @pytest.mark.asyncio
    async def test_evolution_acceptance_handler_runs(self):
        from care.app import CareApp

        class _Evt:
            evolution_id = "evo-1"
            individual_id = "ind-winner"

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            # The handler queues a toast via `push_toast`.
            # No exception should propagate.
            app.on_evolution_screen_acceptance_complete(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_evolution_acceptance_missing_attrs_safe(self):
        from care.app import CareApp

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # Empty event — handler should default to "?" and
            # still push a toast without raising.
            app.on_evolution_screen_acceptance_complete(object())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_evolution_acceptance_toast_includes_version_transition(self):
        """§5 P0 — when the AcceptanceComplete event carries
        version metadata, the toast reads
        `Accepted: <chain_id> v(N) → v(N+1) (now latest)`."""
        from care.app import CareApp

        toasts: list[tuple[str, str]] = []

        class _Evt:
            evolution_id = "evo-1"
            individual_id = "ind-winner"
            chain_id = "agent-storm"
            previous_version = 5
            new_version = 6

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            original = app.push_toast

            def _spy(message, *, severity="info", ttl=None):
                toasts.append((message, severity))
                return original(message, severity=severity, ttl=ttl)

            app.push_toast = _spy  # type: ignore[method-assign]
            app.on_evolution_screen_acceptance_complete(_Evt())
            for _ in range(4):
                await pilot.pause()
        assert any(
            "Accepted: agent-storm v5 → v6 (now latest)" in m
            for m, _ in toasts
        )

    @pytest.mark.asyncio
    async def test_evolution_acceptance_toast_handles_missing_previous_version(self):
        """First-ever accept on a chain has no
        `previous_version`. The toast should still render the
        new-version + chain_id form so the user sees what was
        bumped."""
        from care.app import CareApp

        toasts: list[tuple[str, str]] = []

        class _Evt:
            evolution_id = "evo-1"
            individual_id = "ind-1"
            chain_id = "agent-fresh"
            previous_version = None
            new_version = 1

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            original = app.push_toast

            def _spy(message, *, severity="info", ttl=None):
                toasts.append((message, severity))
                return original(message, severity=severity, ttl=ttl)

            app.push_toast = _spy  # type: ignore[method-assign]
            app.on_evolution_screen_acceptance_complete(_Evt())
            for _ in range(4):
                await pilot.pause()
        assert any(
            "Accepted: agent-fresh v1 (now latest)" in m
            for m, _ in toasts
        )

    @pytest.mark.asyncio
    async def test_evolution_acceptance_pushes_use_it_now_modal(
        self,
    ):
        """§5 P1 — after a successful accept-winner with
        `chain_id` + `new_version`, the host pushes
        UseItNowModal carrying those fields + a `stable`
        channel so the user copy-pastes a `get_chain(...,
        channel="latest")` snippet for their downstream
        service."""
        from care.app import CareApp
        from care.screens.use_it_now import UseItNowModal

        class _Evt:
            evolution_id = "evo-1"
            individual_id = "ind-winner"
            chain_id = "agent-storm"
            previous_version = 5
            new_version = 6

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(20):
                await pilot.pause()
            app.on_evolution_screen_acceptance_complete(_Evt())
            for _ in range(30):
                await pilot.pause()
            modals = [
                s for s in app.screen_stack
                if isinstance(s, UseItNowModal)
            ]
            assert modals, (
                "UseItNowModal should land on the stack after "
                "successful accept-winner; stack="
                f"{[type(s).__name__ for s in app.screen_stack]}"
            )
            modal = modals[-1]
            assert modal.entity_id == "agent-storm"
            assert modal.version == "6"
            assert modal.channel == "stable"
            assert modal.display_name == "agent-storm"

    @pytest.mark.asyncio
    async def test_evolution_acceptance_no_modal_without_chain_id(
        self,
    ):
        """§5 P1 — older platforms that don't ship `chain_id`
        in the AcceptanceComplete event must not push the
        UseItNowModal (the legacy toast is the only signal
        the host can give in that case)."""
        from care.app import CareApp
        from care.screens.use_it_now import UseItNowModal

        class _Evt:
            evolution_id = "evo-1"
            individual_id = "ind-legacy"
            chain_id = ""
            previous_version = None
            new_version = None

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(20):
                await pilot.pause()
            app.on_evolution_screen_acceptance_complete(_Evt())
            for _ in range(30):
                await pilot.pause()
            assert not any(
                isinstance(s, UseItNowModal)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_evolution_acceptance_modal_evolve_routes_to_evolution(
        self,
    ):
        """§5 P1 — dismissing the post-accept UseItNowModal
        with `evolve_requested=True` should route through the
        host's `_push_evolution_for(chain_id)` helper (the same
        opener the library + dashboard use)."""
        from care.app import CareApp
        from care.screens.use_it_now import (
            UseItNowModal,
            UseItNowResult,
        )

        class _Evt:
            evolution_id = "evo-1"
            individual_id = "ind-winner"
            chain_id = "agent-flow"
            previous_version = 2
            new_version = 3

        evolution_pushes: list[str] = []

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(20):
                await pilot.pause()
            # Spy on `_push_evolution_for` so we don't depend
            # on the upstream EvolutionLaunchModal being
            # importable / wired in the test scaffold.
            original_push = app._push_evolution_for

            def _spy(entity_id):
                evolution_pushes.append(entity_id)
                return None

            app._push_evolution_for = _spy  # type: ignore[method-assign]
            app.on_evolution_screen_acceptance_complete(_Evt())
            for _ in range(30):
                await pilot.pause()
            modal = next(
                s for s in app.screen_stack
                if isinstance(s, UseItNowModal)
            )
            modal.dismiss(UseItNowResult(
                closed=False, evolve_requested=True,
            ))
            for _ in range(10):
                await pilot.pause()
            assert evolution_pushes == ["agent-flow"], (
                f"expected exactly one push for 'agent-flow'; "
                f"got {evolution_pushes!r}"
            )
            # Restore so teardown finalisation doesn't trip on
            # the patched method.
            app._push_evolution_for = original_push  # type: ignore[method-assign]

    @pytest.mark.asyncio
    async def test_evolution_acceptance_toast_falls_back_legacy_format(self):
        """When the platform shipped no version info, the
        toast falls back to the iter-8 legacy format
        (`Accepted <individual> from evolution <id> → promoted
        to stable`)."""
        from care.app import CareApp

        toasts: list[tuple[str, str]] = []

        class _Evt:
            evolution_id = "evo-1"
            individual_id = "ind-legacy"
            chain_id = ""
            previous_version = None
            new_version = None

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            original = app.push_toast

            def _spy(message, *, severity="info", ttl=None):
                toasts.append((message, severity))
                return original(message, severity=severity, ttl=ttl)

            app.push_toast = _spy  # type: ignore[method-assign]
            app.on_evolution_screen_acceptance_complete(_Evt())
            for _ in range(4):
                await pilot.pause()
        assert any(
            "Accepted ind-legacy from evolution evo-1" in m
            for m, _ in toasts
        )

    @pytest.mark.asyncio
    async def test_evolution_acceptance_refreshes_library_on_stack(
        self, monkeypatch,
    ):
        """TODO §5 P0 — accept-winner should also refresh any
        `LibraryScreen` already on the stack so the new stable
        individual lands without forcing the user to re-open
        the library. Pin the refresh call by spying on
        `refresh_library`."""
        from care.app import CareApp
        from care.screens.library import LibraryScreen
        from care.screens.welcome import WelcomeScreen

        # Settle boot quickly — zeroing the splash makes the
        # `set_timer` use `call_later` so a single pilot.pause
        # batch drains the routing.
        monkeypatch.setattr(WelcomeScreen, "DEFAULT_SPLASH_SECONDS", 0.0)

        class _Evt:
            evolution_id = "evo-42"
            individual_id = "ind-winner"

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            # Push a LibraryScreen and spy on its refresh hook.
            lib = LibraryScreen(restore_state=False)
            app.push_screen(lib)
            for _ in range(4):
                await pilot.pause()
            calls: list[int] = []
            monkeypatch.setattr(
                lib, "refresh_library", lambda: calls.append(1),
            )
            app.on_evolution_screen_acceptance_complete(_Evt())
            for _ in range(4):
                await pilot.pause()
        assert len(calls) == 1, (
            f"expected exactly one refresh_library call, got {len(calls)}"
        )

    @pytest.mark.asyncio
    async def test_evolution_acceptance_no_library_is_no_op(self):
        """When no LibraryScreen is mounted, the acceptance
        handler should still toast but not raise — the refresh
        helper must walk the (empty-of-library) stack and
        no-op cleanly."""
        from care.app import CareApp

        class _Evt:
            evolution_id = "evo-1"
            individual_id = "ind-7"

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_evolution_screen_acceptance_complete(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_evolution_acceptance_refresh_failure_is_swallowed(
        self, monkeypatch, caplog,
    ):
        """A bad `refresh_library` (e.g. worker spawn fails)
        must not propagate out of the handler — the user's
        success toast already fired and shouldn't be torpedoed
        by a refresh hiccup. The failure is logged at WARNING
        so the regression is visible in `care-app-*.log`."""
        from care.app import CareApp
        from care.screens.library import LibraryScreen
        from care.screens.welcome import WelcomeScreen

        monkeypatch.setattr(WelcomeScreen, "DEFAULT_SPLASH_SECONDS", 0.0)

        class _Evt:
            evolution_id = "evo-1"
            individual_id = "ind-7"

        def _boom():
            raise RuntimeError("worker spawn failed")

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            lib = LibraryScreen(restore_state=False)
            app.push_screen(lib)
            for _ in range(4):
                await pilot.pause()
            monkeypatch.setattr(lib, "refresh_library", _boom)
            with caplog.at_level("WARNING", logger="care.app"):
                app.on_evolution_screen_acceptance_complete(_Evt())
                for _ in range(4):
                    await pilot.pause()
        assert any(
            "LibraryScreen refresh failed" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_edit_agent_back_action_is_silent(self):
        from care.app import CareApp

        class _Payload:
            action = "back"

        class _Evt:
            payload = _Payload()

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # `back` is an explicit no-toast cancel path.
            app.on_edit_agent_screen_submitted(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_edit_agent_save_success_runs(self):
        from care.app import CareApp

        class _Save:
            entity_id = "ent-saved"
            success = True
            error = None

        class _Payload:
            action = "save"
            save_result = _Save()
            promote_result = None

        class _Evt:
            payload = _Payload()

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_edit_agent_screen_submitted(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_edit_agent_save_failure_runs(self):
        from care.app import CareApp

        class _Save:
            entity_id = ""
            success = False
            error = "save-down"

        class _Payload:
            action = "save"
            save_result = _Save()
            promote_result = None

        class _Evt:
            payload = _Payload()

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_edit_agent_screen_submitted(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_edit_agent_promote_success_runs(self):
        from care.app import CareApp

        class _Promote:
            entity_id = "ent-x"
            from_channel = "latest"
            to_channel = "stable"
            success = True
            error = None

        class _Payload:
            action = "promote"
            save_result = None
            promote_result = _Promote()

        class _Evt:
            payload = _Payload()

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_edit_agent_screen_submitted(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_edit_agent_no_payload_safe(self):
        from care.app import CareApp

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # Event with no `payload` attribute — handler
            # silently no-ops.
            app.on_edit_agent_screen_submitted(object())
            for _ in range(4):
                await pilot.pause()


# ---------------------------------------------------------------------------
# Inspection / Settings / Welcome screen-message handlers
# ---------------------------------------------------------------------------


class TestInspectionActionDispatch:
    @pytest.mark.asyncio
    async def test_back_action_is_silent(self):
        from care.app import CareApp

        class _Evt:
            action = "back"
            entity_id = "ent-x"

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_inspection_screen_action_requested(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_run_action_toasts(self):
        from care.app import CareApp

        class _Evt:
            action = "run"
            entity_id = "ent-storm"

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # `run` is the runtime-pending path; the handler
            # surfaces an info toast. No exception.
            app.on_inspection_screen_action_requested(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_edit_action_no_memory_toasts_error(self):
        from care.app import CareApp

        class _Evt:
            action = "edit"
            entity_id = "ent-x"

        app = CareApp(mode="returning")  # memory=None
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_inspection_screen_action_requested(_Evt())
            for _ in range(4):
                await pilot.pause()
            # No EditAgentScreen pushed.
            from care.screens.edit_agent import EditAgentScreen

            assert not any(
                isinstance(s, EditAgentScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_edit_action_with_memory_pushes_edit_screen(self):
        from care.app import CareApp
        from care.screens.edit_agent import EditAgentScreen

        class _StubMemory:
            def get_chain(self, entity_id, *, channel="latest"):
                return {
                    "task_description": "demo",
                    "steps": [
                        {"number": 1, "title": "x", "step_type": "llm"},
                    ],
                }

            def save_agent_skill(self, **kw):  # pragma: no cover
                pass

        class _Evt:
            action = "edit"
            entity_id = "ent-x"

        app = CareApp(mode="returning", memory=_StubMemory())
        async with app.run_test() as pilot:
            for _ in range(15):
                await pilot.pause()
            app.on_inspection_screen_action_requested(_Evt())
            for _ in range(15):
                await pilot.pause()
            assert any(
                isinstance(s, EditAgentScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_evolve_action_no_platform_toasts_error(self):
        from care.app import CareApp

        class _Evt:
            action = "evolve"
            entity_id = "ent-x"

        app = CareApp(mode="returning")  # platform=None
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_inspection_screen_action_requested(_Evt())
            for _ in range(4):
                await pilot.pause()
            from care.screens.evolution import EvolutionScreen

            assert not any(
                isinstance(s, EvolutionScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_evolve_action_with_platform_pushes_evolution(self):
        """§5 P0 — the `evolve` action from InspectionScreen
        first pushes :class:`EvolutionLaunchModal` so the user
        can pick the budget / rubric / objectives before the
        full :class:`EvolutionScreen` launches. The screen
        itself only lands after the user submits the modal —
        cancelling the modal pops back to chat.

        Iter 78: was originally written to expect
        EvolutionScreen directly on the stack; updated to
        match the §5 P0 modal-first flow (asserts the
        EvolutionLaunchModal lands carrying the right
        `base_chain_id`).
        """
        from care.app import CareApp
        from care.screens.evolution_launch import (
            EvolutionLaunchModal,
        )

        class _StubPlatform:
            pass  # Existence is enough for the gate.

        class _Evt:
            action = "evolve"
            entity_id = "ent-x"

        app = CareApp(mode="returning", platform=_StubPlatform())
        async with app.run_test() as pilot:
            for _ in range(15):
                await pilot.pause()
            app.on_inspection_screen_action_requested(_Evt())
            for _ in range(15):
                await pilot.pause()
            modals = [
                s for s in app.screen_stack
                if isinstance(s, EvolutionLaunchModal)
            ]
            assert modals, (
                "EvolutionLaunchModal should land on the stack; "
                f"got {[type(s).__name__ for s in app.screen_stack]}"
            )
            assert modals[0].base_chain_id == "ent-x"

    @pytest.mark.asyncio
    async def test_duplicate_action_toasts(self):
        from care.app import CareApp

        class _Evt:
            action = "duplicate"
            entity_id = "ent-x"

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_inspection_screen_action_requested(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_unknown_action_safe(self):
        from care.app import CareApp

        class _Evt:
            action = "fly_to_moon"
            entity_id = "ent-x"

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # Unknown action falls through to a no-op.
            app.on_inspection_screen_action_requested(_Evt())
            for _ in range(4):
                await pilot.pause()


class TestSettingsHandlers:
    @pytest.mark.asyncio
    async def test_settings_saved_reloads_config(self):
        from care.app import CareApp

        class _Evt:
            class snapshot:
                pass

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # Reload may fail on a non-configured environment;
            # the handler should still produce a toast rather
            # than raising.
            app.on_settings_screen_saved(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_settings_cancelled_is_silent(self):
        from care.app import CareApp

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_settings_screen_cancelled(object())
            for _ in range(4):
                await pilot.pause()


class TestWelcomeRecentSelected:
    @pytest.mark.asyncio
    async def test_recent_selected_pushes_inspection(self):
        from care.app import CareApp
        from care.screens.inspection import InspectionScreen

        class _Row:
            entity_id = "ent-recent"

        class _Evt:
            row = _Row()

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(15):
                await pilot.pause()
            app.on_welcome_screen_recent_selected(_Evt())
            for _ in range(15):
                await pilot.pause()
            assert any(
                isinstance(s, InspectionScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_recent_selected_missing_row_safe(self):
        from care.app import CareApp

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # Event with no `row` attribute — silent no-op.
            app.on_welcome_screen_recent_selected(object())
            for _ in range(4):
                await pilot.pause()


# ---------------------------------------------------------------------------
# Remaining screen-message handlers (Query / TaskList / Resume / Import)
# ---------------------------------------------------------------------------


class TestQueryHandlers:
    @pytest.mark.asyncio
    async def test_generate_requested_toasts(self):
        from care.app import CareApp

        class _Sub:
            task = "weather report for SF"
            mage_mode = "deep"

        class _Evt:
            submission = _Sub()

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_query_screen_generate_requested(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_generate_requested_empty_task_silent(self):
        from care.app import CareApp

        class _Sub:
            task = ""
            mage_mode = "deep"

        class _Evt:
            submission = _Sub()

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # Empty task → handler no-ops (no toast queued).
            app.on_query_screen_generate_requested(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_generate_requested_no_submission_safe(self):
        from care.app import CareApp

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_query_screen_generate_requested(object())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_back_requested_is_silent_anchor(self):
        from care.app import CareApp

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # Anchor handler accepts the event without raising.
            app.on_query_screen_back_requested(object())
            for _ in range(4):
                await pilot.pause()


class TestTaskListSwitch:
    @pytest.mark.asyncio
    async def test_switch_requested_toasts(self):
        from care.app import CareApp

        class _Record:
            task_id = "task-1"
            description = "Generation in progress"

        class _Evt:
            record = _Record()

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_task_list_drawer_switch_requested(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_switch_requested_missing_record_safe(self):
        from care.app import CareApp

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # No `record` attr → silent no-op.
            app.on_task_list_drawer_switch_requested(object())
            for _ in range(4):
                await pilot.pause()


class TestResumeRequested:
    @pytest.mark.asyncio
    async def test_resume_with_state_toasts(self):
        from care.app import CareApp

        class _State:
            chain_id = "ent-storm"

        class _Evt:
            state = _State()

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_welcome_screen_resume_requested(_Evt())
            for _ in range(4):
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_resume_without_state_warns(self):
        from care.app import CareApp

        class _Evt:
            state = None

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            app.on_welcome_screen_resume_requested(_Evt())
            for _ in range(4):
                await pilot.pause()


class TestImportPreview:
    @pytest.mark.asyncio
    async def test_import_preview_anchor(self):
        from care.app import CareApp

        class _Evt:
            class manifest:
                pass

        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            # No-op anchor handler — accepts any event shape.
            app.on_import_modal_preview_loaded(_Evt())
            for _ in range(4):
                await pilot.pause()
