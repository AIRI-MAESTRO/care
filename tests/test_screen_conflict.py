"""Pilot tests for ConflictModal (§3 P1 [DONE — data layer] → fully DONE).

Wires :func:`care.detect_conflict`'s produced
:class:`ConflictReport` into the modal that lets the user pick a
resolution. Tests exercise:

* Compose — the three resolution buttons mount.
* Each button dismisses with the right resolution literal.
* Escape dismisses with ``cancelled=True``, ``resolution=None``.
* No-conflict reports still render (informational mode).
* Re-exports from ``care.screens``.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from care.conflict import ConflictReport
from care.screens.conflict import ConflictModal, ConflictModalResult


# ---------------------------------------------------------------------------
# Sample reports
# ---------------------------------------------------------------------------


def _conflict_report() -> ConflictReport:
    return ConflictReport(
        existing_entity_id="ent-123",
        existing_sha256="a" * 64,
        incoming_sha256="b" * 64,
        is_conflict=True,
        existing_content={"steps": [{"prompt": "old"}]},
        incoming_content={"steps": [{"prompt": "new"}]},
        diff_lines=(
            "--- Storm Watcher (existing)",
            "+++ Storm Watcher (incoming)",
            "-  \"prompt\": \"old\"",
            "+  \"prompt\": \"new\"",
        ),
        name="Storm Watcher",
        entity_type="chain",
    )


def _identical_report() -> ConflictReport:
    sha = "c" * 64
    return ConflictReport(
        existing_entity_id="ent-456",
        existing_sha256=sha,
        incoming_sha256=sha,
        is_conflict=False,
        existing_content={"steps": []},
        incoming_content={"steps": []},
        diff_lines=(),
        name="Quiet Watcher",
        entity_type="chain",
    )


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, *, report: ConflictReport) -> None:
        super().__init__()
        self._report = report
        self.dismissed: list[ConflictModalResult] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(result):
            self.dismissed.append(result)

        self.push_screen(ConflictModal(self._report), _on_dismiss)


def _modal(app: App) -> ConflictModal:
    s = app.screen_stack[-1]
    assert isinstance(s, ConflictModal)
    return s


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_three_buttons_mount(self):
        app = _Host(report=_conflict_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.query_one(
                "#conflict-btn-keep-existing", Button,
            ) is not None
            assert modal.query_one(
                "#conflict-btn-accept-incoming", Button,
            ) is not None
            assert modal.query_one(
                "#conflict-btn-new-version", Button,
            ) is not None
            assert modal.query_one(
                "#conflict-btn-cancel", Button,
            ) is not None

    @pytest.mark.asyncio
    async def test_diff_body_renders_diff_lines(self):
        app = _Host(report=_conflict_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            body = modal.query_one("#conflict-diff-body", Static)
            assert body is not None
            text = modal._diff_text()
            assert "+++ Storm Watcher" in text
            assert "old" in text
            assert "new" in text

    @pytest.mark.asyncio
    async def test_summary_for_identical(self):
        app = _Host(report=_identical_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            text = modal._summary_line()
            assert "identical" in text

    @pytest.mark.asyncio
    async def test_no_diff_text_for_identical(self):
        app = _Host(report=_identical_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal._diff_text() == "(no differences)"


# ---------------------------------------------------------------------------
# Button dispatch
# ---------------------------------------------------------------------------


class TestDismiss:
    @pytest.mark.asyncio
    async def test_keep_existing_dismisses_with_resolution(self):
        app = _Host(report=_conflict_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#conflict-btn-keep-existing", Button,
            ).press()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].resolution == "keep_existing"
            assert app.dismissed[0].cancelled is False
            assert app.dismissed[0].report is not None

    @pytest.mark.asyncio
    async def test_accept_incoming_dismisses_with_resolution(self):
        app = _Host(report=_conflict_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#conflict-btn-accept-incoming", Button,
            ).press()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].resolution == "accept_incoming"
            assert app.dismissed[0].cancelled is False

    @pytest.mark.asyncio
    async def test_new_version_dismisses_with_resolution(self):
        app = _Host(report=_conflict_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#conflict-btn-new-version", Button,
            ).press()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].resolution == "new_version"
            assert app.dismissed[0].cancelled is False

    @pytest.mark.asyncio
    async def test_cancel_button_dismisses_cancelled(self):
        app = _Host(report=_conflict_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#conflict-btn-cancel", Button).press()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].resolution is None
            assert app.dismissed[0].cancelled is True

    @pytest.mark.asyncio
    async def test_escape_dismisses_cancelled(self):
        app = _Host(report=_conflict_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_cancel()
            for _ in range(3):
                await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].resolution is None
            assert app.dismissed[0].cancelled is True


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import ConflictModal as M
        from care.screens import ConflictModalResult as R

        assert M is ConflictModal
        assert R is ConflictModalResult
