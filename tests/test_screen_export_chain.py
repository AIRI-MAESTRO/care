"""Tests for the §5 P1 `ExportChainModal` + EvolutionScreen's
`x` Export-individual binding."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from care.screens.evolution import (
    EvolutionIndividual,
    EvolutionScreen,
)
from care.screens.export_chain import (
    ExportChainModal,
    ExportChainResult,
    _default_stem,
    _safe_filename_stem,
)


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------


class TestSafeFilenameStem:
    def test_keeps_alnum_dash_underscore(self):
        assert _safe_filename_stem("alpha-beta_42") == "alpha-beta_42"

    def test_lowercases(self):
        assert _safe_filename_stem("Weather Forecaster") == "weather-forecaster"

    def test_collapses_runs_of_specials(self):
        assert _safe_filename_stem("a // b") == "a-b"

    def test_trims_dashes(self):
        assert _safe_filename_stem("/foo/bar/") == "foo-bar"

    def test_empty(self):
        assert _safe_filename_stem("") == ""

    def test_only_specials(self):
        assert _safe_filename_stem("// ?? !!") == ""


class TestDefaultStem:
    def test_id_and_version(self):
        assert _default_stem("5bfa49dfc6c2", "v7", "Name") == "chain_5bfa49dfc6c2_v7"

    def test_bare_numeric_version_gets_v_prefix(self):
        assert _default_stem("abc", "3", "Name") == "chain_abc_v3"

    def test_id_without_version(self):
        assert _default_stem("abc", "", "Name") == "chain_abc"

    def test_falls_back_to_display_name_without_id(self):
        assert _default_stem("", "v1", "Weather Bot") == "weather-bot"

    def test_falls_back_to_chain_when_nothing(self):
        assert _default_stem("", "", "") == "chain"


# ---------------------------------------------------------------------------
# ExportChainResult dataclass
# ---------------------------------------------------------------------------


class TestExportChainResult:
    def test_defaults(self):
        r = ExportChainResult()
        assert r.ok is False
        assert r.path is None
        assert r.bytes_written == 0
        assert r.format == "json"
        assert r.error == ""


# ---------------------------------------------------------------------------
# Modal compose + submit
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, modal: ExportChainModal):
        super().__init__()
        self._modal = modal
        self.dismissed: list[ExportChainResult | None] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self._modal, self.dismissed.append)


def _chain() -> dict:
    return {
        "name": "test-chain",
        "steps": [
            {"name": "Stage A", "type": "llm"},
            {"name": "Stage B", "type": "tool"},
        ],
        "metadata": {"care": {"fitness_score": 0.87}},
    }


class TestExportChainModalCompose:
    @pytest.mark.asyncio
    async def test_default_path_lands_in_current_folder(self):
        """With no explicit path, the default lands in the CURRENT working
        directory (not the home dir), so exports are easy to find."""
        from pathlib import Path

        modal = ExportChainModal(
            chain=_chain(), display_name="Weather Forecaster",
            default_format="markdown",
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            assert modal._default_path == str(
                Path.cwd() / "weather-forecaster.md"
            )

    @pytest.mark.asyncio
    async def test_default_name_uses_entity_id_and_version(self):
        """With an entity_id + version, the default name is traceable:
        ``chain_<id>_v<version>.<ext>``."""
        from pathlib import Path

        modal = ExportChainModal(
            chain=_chain(), display_name="Weather Forecaster",
            entity_id="5bfa49dfc6c2", version="v7",
            default_format="markdown",
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            assert modal._default_path == str(
                Path.cwd() / "chain_5bfa49dfc6c2_v7.md"
            )

    @pytest.mark.asyncio
    async def test_default_path_uses_display_name(
        self, tmp_path,
    ):
        modal = ExportChainModal(
            chain=_chain(), display_name="Weather Forecaster",
            default_path=tmp_path / "out.json",
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            assert modal._default_path == str(
                tmp_path / "out.json"
            )

    @pytest.mark.asyncio
    async def test_compose_renders_title_path_radio(
        self, tmp_path,
    ):
        from textual.widgets import Input, RadioSet
        modal = ExportChainModal(
            chain=_chain(),
            display_name="My Chain",
            default_path=tmp_path / "chain.json",
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            path_input = modal.query_one(
                "#export-chain-path", Input,
            )
            assert path_input.value.endswith("chain.json")
            radio = modal.query_one(
                "#export-chain-format", RadioSet
            )
            assert radio is not None

    @pytest.mark.asyncio
    async def test_markdown_is_first_format_option(self, tmp_path):
        from textual.widgets import RadioButton, RadioSet
        modal = ExportChainModal(
            chain=_chain(),
            display_name="My Chain",
            default_path=tmp_path / "chain.md",
            default_format="markdown",
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            radio = modal.query_one("#export-chain-format", RadioSet)
            buttons = list(radio.query(RadioButton))
            assert (buttons[0].id or "").endswith("-markdown")
            assert modal._read_format() == "markdown"

    @pytest.mark.asyncio
    async def test_changing_format_syncs_path_extension(self, tmp_path):
        from textual.widgets import Input, RadioButton
        modal = ExportChainModal(
            chain=_chain(),
            display_name="My Chain",
            default_path=tmp_path / "chain.md",
            default_format="markdown",
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            path = modal.query_one("#export-chain-path", Input)
            assert path.value.endswith(".md")
            # Switch to JSON → extension follows.
            modal.query_one(
                "#export-chain-format-json", RadioButton,
            ).value = True
            for _ in range(3):
                await pilot.pause()
            assert modal._read_format() == "json"
            assert path.value.endswith(".json")
            assert not path.value.endswith(".md")
            # Switch to Python → extension follows.
            modal.query_one(
                "#export-chain-format-python", RadioButton,
            ).value = True
            for _ in range(3):
                await pilot.pause()
            assert path.value.endswith(".py")


class TestExportChainModalBrowse:
    @pytest.mark.asyncio
    async def test_browse_opens_directory_picker(self, tmp_path):
        from textual.widgets import Button

        from care.screens.directory_picker import DirectoryPickerModal

        modal = ExportChainModal(
            chain=_chain(), display_name="c",
            default_path=tmp_path / "c.md", default_format="markdown",
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal.on_button_pressed(
                Button.Pressed(modal.query_one("#export-chain-btn-browse", Button)),
            )
            for _ in range(3):
                await pilot.pause()
            assert isinstance(app.screen_stack[-1], DirectoryPickerModal)

    @pytest.mark.asyncio
    async def test_picked_folder_rewrites_path_keeping_filename(self, tmp_path):
        from textual.widgets import Button, Input

        from care.screens.directory_picker import DirectoryPickerModal

        dest_dir = tmp_path / "out"
        dest_dir.mkdir()
        modal = ExportChainModal(
            chain=_chain(), display_name="c",
            default_path=tmp_path / "weather.md", default_format="markdown",
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal.on_button_pressed(
                Button.Pressed(modal.query_one("#export-chain-btn-browse", Button)),
            )
            for _ in range(3):
                await pilot.pause()
            picker = app.screen_stack[-1]
            assert isinstance(picker, DirectoryPickerModal)
            # Pick `dest_dir` and confirm.
            picker.on_directory_tree_directory_selected(
                type("E", (), {"path": dest_dir})(),  # type: ignore[arg-type]
            )
            picker.on_button_pressed(
                Button.Pressed(picker.query_one("#dirpick-btn-select", Button)),
            )
            for _ in range(3):
                await pilot.pause()
            # The path field now points into the chosen folder, filename kept.
            field = modal.query_one("#export-chain-path", Input)
            assert field.value == str(dest_dir / "weather.md")


class TestExportChainModalSubmit:
    @pytest.mark.asyncio
    async def test_submit_writes_json_and_dismisses(
        self, tmp_path,
    ):
        dest = tmp_path / "evolved.json"
        modal = ExportChainModal(
            chain=_chain(), display_name="evo",
            default_path=dest,
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal.action_submit()
            for _ in range(3):
                await pilot.pause()
            assert dest.is_file()
            body = dest.read_text()
            assert '"test-chain"' in body
            assert app.dismissed
            result = app.dismissed[-1]
            assert isinstance(result, ExportChainResult)
            assert result.ok
            assert result.path == dest
            assert result.format == "json"
            assert result.bytes_written > 0

    @pytest.mark.asyncio
    async def test_submit_to_unwritable_path_renders_error(
        self, tmp_path,
    ):
        # A path under a missing parent directory triggers
        # ChainExportError without crashing the modal.
        dest = tmp_path / "missing-parent-dir" / "out.json"
        modal = ExportChainModal(
            chain=_chain(), display_name="evo",
            default_path=dest,
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal.action_submit()
            for _ in range(3):
                await pilot.pause()
            assert not dest.exists()
            assert modal.last_error
            # Modal stayed open — failures don't auto-dismiss.
            assert not app.dismissed or app.dismissed == [None]

    @pytest.mark.asyncio
    async def test_cancel_dismisses_with_none(self, tmp_path):
        modal = ExportChainModal(
            chain=_chain(), display_name="evo",
            default_path=tmp_path / "out.json",
        )
        app = _Host(modal)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal.action_cancel()
            await pilot.pause()
            assert app.dismissed == [None]


# ---------------------------------------------------------------------------
# EvolutionScreen `x` Export-individual binding
# ---------------------------------------------------------------------------


class _EvoHost(App):
    def __init__(self, screen: EvolutionScreen):
        super().__init__()
        self._screen = screen
        self.toasts: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self._screen)

    def push_toast(
        self, message, *, severity="info", ttl=None,
    ) -> None:  # type: ignore[override]
        self.toasts.append((message, severity))


class TestEvolutionExportBinding:
    def test_x_bound_to_export_individual(self):
        action_by_key = {
            b.key: getattr(b, "action", None)
            for b in EvolutionScreen.BINDINGS
        }
        assert action_by_key.get("x") == "export_individual", (
            f"`x` must bind to export_individual; got "
            f"{action_by_key.get('x')!r}"
        )

    @pytest.mark.asyncio
    async def test_export_with_no_highlight_toasts(self):
        screen = EvolutionScreen(base_chain_id="base-1")
        app = _EvoHost(screen)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            screen._highlighted_individual = None
            screen.selected_individual = None
            screen.action_export_individual()
            await pilot.pause()
            assert any(
                "Highlight" in m for m, _ in app.toasts
            ), f"toasts: {app.toasts}"

    @pytest.mark.asyncio
    async def test_export_individual_without_chain_toasts(
        self,
    ):
        screen = EvolutionScreen(base_chain_id="base-1")
        # Individual with no chain payload.
        screen.run.individuals = [
            EvolutionIndividual(
                individual_id="ind-1", fitness=0.5,
                chain_dict=None,
            ),
        ]
        app = _EvoHost(screen)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            screen._highlighted_individual = "ind-1"
            screen.action_export_individual()
            await pilot.pause()
            assert any(
                "no chain payload" in m for m, _ in app.toasts
            ), f"toasts: {app.toasts}"

    @pytest.mark.asyncio
    async def test_export_pushes_modal_for_individual_with_chain(
        self,
    ):
        screen = EvolutionScreen(base_chain_id="base-1")
        screen.run.individuals = [
            EvolutionIndividual(
                individual_id="ind-A", fitness=0.85,
                summary="evolved-A",
                chain_dict={"name": "evolved-A",
                            "steps": [{"name": "S1"}]},
            ),
        ]
        app = _EvoHost(screen)
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            screen._highlighted_individual = "ind-A"
            screen.action_export_individual()
            for _ in range(3):
                await pilot.pause()
            modals = [
                s for s in app.screen_stack
                if isinstance(s, ExportChainModal)
            ]
            assert modals, (
                "ExportChainModal should land on the stack"
            )
            # display_name → "evolved-A" → sanitized stem
            # "evolved-a" (lowercased).
            assert "evolved-a" in modals[-1]._default_path


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import (
            ExportChainModal as M,
            ExportChainResult as R,
        )

        assert M is ExportChainModal
        assert R is ExportChainResult
