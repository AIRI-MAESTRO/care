"""Pilot tests for DirectoryPickerModal — browse + pick a folder."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from care.screens.directory_picker import DirectoryPickerModal


class _Host(App):
    def __init__(self, modal: DirectoryPickerModal) -> None:
        super().__init__()
        self._modal = modal
        self.dismissed: list[Path | None] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self._modal, self.dismissed.append)


def _press(modal: DirectoryPickerModal, bid: str) -> None:
    modal.on_button_pressed(Button.Pressed(modal.query_one(f"#{bid}", Button)))


class TestDirectoryPicker:
    @pytest.mark.asyncio
    async def test_select_returns_start_dir_by_default(self, tmp_path):
        modal = DirectoryPickerModal(start=tmp_path)
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            _press(modal, "dirpick-btn-select")
            await pilot.pause()
            assert app.dismissed == [tmp_path]

    @pytest.mark.asyncio
    async def test_directory_selected_updates_choice(self, tmp_path):
        sub = tmp_path / "exports"
        sub.mkdir()
        modal = DirectoryPickerModal(start=tmp_path)
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Simulate the tree emitting DirectorySelected for the subdir.
            modal.on_directory_tree_directory_selected(
                type("E", (), {"path": sub})(),  # type: ignore[arg-type]
            )
            await pilot.pause()
            _press(modal, "dirpick-btn-select")
            await pilot.pause()
            assert app.dismissed == [sub]

    @pytest.mark.asyncio
    async def test_cancel_returns_none(self, tmp_path):
        modal = DirectoryPickerModal(start=tmp_path)
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            _press(modal, "dirpick-btn-cancel")
            await pilot.pause()
            assert app.dismissed == [None]

    @pytest.mark.asyncio
    async def test_up_reroots_to_parent(self, tmp_path):
        sub = tmp_path / "nested"
        sub.mkdir()
        modal = DirectoryPickerModal(start=sub)
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            _press(modal, "dirpick-btn-up")
            await pilot.pause()
            # Selecting now returns the parent (tmp_path).
            _press(modal, "dirpick-btn-select")
            await pilot.pause()
            assert app.dismissed == [tmp_path]

    @pytest.mark.asyncio
    async def test_nonexistent_start_falls_back_to_parent_or_cwd(self, tmp_path):
        missing = tmp_path / "does-not-exist" / "deep"
        modal = DirectoryPickerModal(start=missing)
        # Start resolves to an existing directory (never the missing path).
        assert modal._start.is_dir()
