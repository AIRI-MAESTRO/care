"""Pilot tests for FilePickerModal + the Evolution-launch Browse wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input

from care.screens.file_picker import FilePickerModal, _ExtFilteredDirectoryTree


class _Host(App):
    def __init__(self, modal: FilePickerModal) -> None:
        super().__init__()
        self._modal = modal
        self.dismissed: list[Path | None] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self._modal, self.dismissed.append)


def _press(modal: FilePickerModal, bid: str) -> None:
    modal.on_button_pressed(Button.Pressed(modal.query_one(f"#{bid}", Button)))


class TestFilePicker:
    @pytest.mark.asyncio
    async def test_file_selected_then_select_dismisses_with_path(self, tmp_path):
        f = tmp_path / "eval.jsonl"
        f.write_text('{"input":"x"}\n')
        modal = FilePickerModal(start=tmp_path)
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal.on_directory_tree_file_selected(
                type("E", (), {"path": f})(),  # type: ignore[arg-type]
            )
            await pilot.pause()
            _press(modal, "filepick-btn-select")
            await pilot.pause()
            assert app.dismissed == [f]

    @pytest.mark.asyncio
    async def test_select_without_a_file_keeps_modal_open(self, tmp_path):
        modal = FilePickerModal(start=tmp_path)
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            _press(modal, "filepick-btn-select")
            await pilot.pause()
            # No file picked → no dismissal.
            assert app.dismissed == []

    @pytest.mark.asyncio
    async def test_cancel_returns_none(self, tmp_path):
        modal = FilePickerModal(start=tmp_path)
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            _press(modal, "filepick-btn-cancel")
            await pilot.pause()
            assert app.dismissed == [None]

    @pytest.mark.asyncio
    async def test_file_start_preselects_and_roots_at_parent(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text("{}\n")
        modal = FilePickerModal(start=f)
        assert modal._start == tmp_path
        assert modal._selected == f
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            # The pre-selected file confirms immediately.
            _press(modal, "filepick-btn-select")
            await pilot.pause()
            assert app.dismissed == [f]

    def test_extension_filter_hides_non_matching_files(self, tmp_path):
        (tmp_path / "keep.jsonl").write_text("{}\n")
        (tmp_path / "skip.txt").write_text("x")
        sub = tmp_path / "subdir"
        sub.mkdir()
        tree = _ExtFilteredDirectoryTree(
            str(tmp_path), extensions=(".jsonl",),
        )
        kept = {p.name for p in tree.filter_paths(tmp_path.iterdir())}
        assert "keep.jsonl" in kept
        assert "subdir" in kept  # directories always stay visible
        assert "skip.txt" not in kept

    def test_no_extensions_shows_all_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.csv").write_text("y")
        tree = _ExtFilteredDirectoryTree(str(tmp_path), extensions=())
        kept = {p.name for p in tree.filter_paths(tmp_path.iterdir())}
        assert {"a.txt", "b.csv"} <= kept


class TestEvolutionLaunchBrowse:
    @pytest.mark.asyncio
    async def test_browse_button_opens_file_picker(self, tmp_path):
        from care.screens.evolution_launch import EvolutionLaunchModal

        class _LaunchHost(App):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(EvolutionLaunchModal(base_chain_id="c1"))

        app = _LaunchHost()
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal = app.screen
            modal.on_button_pressed(
                Button.Pressed(
                    modal.query_one("#launch-dataset-browse", Button),
                ),
            )
            for _ in range(3):
                await pilot.pause()
            assert isinstance(app.screen, FilePickerModal)

    @pytest.mark.asyncio
    async def test_picked_file_populates_dataset_field(self, tmp_path):
        from care.screens.evolution_launch import EvolutionLaunchModal

        f = tmp_path / "eval.jsonl"
        f.write_text('{"input":"x","expected":"y"}\n')

        class _LaunchHost(App):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(EvolutionLaunchModal(base_chain_id="c1"))

        app = _LaunchHost()
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            launch = app.screen
            launch.on_button_pressed(
                Button.Pressed(
                    launch.query_one("#launch-dataset-browse", Button),
                ),
            )
            for _ in range(3):
                await pilot.pause()
            picker = app.screen
            assert isinstance(picker, FilePickerModal)
            picker.on_directory_tree_file_selected(
                type("E", (), {"path": f})(),  # type: ignore[arg-type]
            )
            picker.on_button_pressed(
                Button.Pressed(
                    picker.query_one("#filepick-btn-select", Button),
                ),
            )
            for _ in range(4):
                await pilot.pause()
            # Back on the launch modal, the dataset field is filled.
            assert launch.query_one("#launch-dataset", Input).value == str(f)
            # ...and it flows into the collected spec.
            assert launch.collect_spec().dataset_path == str(f)
