"""Smoke + behaviour tests for `ImportModal` (TODO §8 P1)."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from care.screens.import_bundle import ImportModal


class _StubMemory:
    """Bare facade — the modal only touches the memory facade
    inside its async worker, which we don't fire from the
    smoke tests. Smoke tests live and die on compose +
    button-routing assertions."""


class _Host(App):
    def __init__(self, *, default_path: str = "~/care-export.tar.gz"):
        super().__init__()
        self._default_path = default_path
        self.dismissed: list[object] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(
            ImportModal(
                memory=_StubMemory(),
                default_path=self._default_path,
            ),
            self._on_dismiss,
        )

    def _on_dismiss(self, result) -> None:
        self.dismissed.append(result)


def _modal(app: _Host) -> ImportModal:
    for s in app.screen_stack:
        if isinstance(s, ImportModal):
            return s
    raise AssertionError("ImportModal not on stack")


class TestCompose:
    @pytest.mark.asyncio
    async def test_mount_does_not_raise(self) -> None:
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, ImportModal)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_default_path_prefills_input(self) -> None:
        app = _Host(default_path="/tmp/example.tar.gz")
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            inp = modal.query_one("#import-path", Input)
            assert inp.value == "/tmp/example.tar.gz"

    @pytest.mark.asyncio
    async def test_action_buttons_present(self) -> None:
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            # Three buttons: Cancel / Preview / Import.
            ids = {
                b.id for b in modal.query("Button").results(Button)
            }
            assert "import-btn-cancel" in ids
            assert "import-btn-preview" in ids
            assert "import-btn-submit" in ids


class TestActions:
    @pytest.mark.asyncio
    async def test_cancel_dismisses_with_none(self) -> None:
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            modal.action_cancel()
            for _ in range(4):
                await pilot.pause()
            assert app.dismissed == [None]

    @pytest.mark.asyncio
    async def test_missing_default_path_renders_placeholder_preview(
        self, tmp_path,
    ) -> None:
        # The default path points at a non-existent file →
        # the preview pane should render the
        # "(no preview)" placeholder (or a friendly-error
        # variant) rather than raising.
        app = _Host(
            default_path=str(tmp_path / "missing.tar.gz"),
        )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            preview = modal.query_one(
                "#import-preview", Static,
            )
            text = str(preview.render())
            # Either the placeholder OR a "couldn't open"
            # error variant is acceptable; what we don't
            # want is a crash or an empty render.
            assert text.strip() != ""


class TestReExports:
    def test_screens_re_exports_import_modal(self) -> None:
        from care.screens import ImportModal as I

        assert I is ImportModal
