"""Pilot tests for ExportModal (TODO §1.1 P0.30).

Exercises:
* Composition — output Input + skills Checkbox + summary line.
* `current_request()` reads the form values.
* Submit fires `export_library_bundle` against a tmp path
  and renders the result line.
* Submit on success auto-dismisses with `BundleExportResult`.
* Cancel / Escape dismiss with `None`.
"""

from __future__ import annotations


import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, Static

from care.runtime.library_bundle import BundleExportResult
from care.screens.export import ExportModal, ExportRequest


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(self):
        self.get_chain_calls: list[tuple] = []

    def get_chain_dict(self, entity_id, channel):
        self.get_chain_calls.append((entity_id, channel))
        return {
            "entity_id": entity_id,
            "channel": channel,
            "content": {"steps": [{"name": "fetch"}]},
            "meta": {"display_name": entity_id.title(), "tags": []},
        }

    def get_agent_skill_dict(self, entity_id, channel):
        return {
            "entity_id": entity_id,
            "channel": channel,
            "content": {"manifest": {}, "sha256": "x"},
            "meta": {"name": entity_id, "tags": []},
        }


class _StubMemory:
    def __init__(self):
        self.client = _StubClient()


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self.dismissed: list[BundleExportResult | None] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(result):
            self.dismissed.append(result)

        self.push_screen(ExportModal(**self._kwargs), _on_dismiss)


def _modal(app: App) -> ExportModal:
    s = app.screen_stack[-1]
    assert isinstance(s, ExportModal)
    return s


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_widgets_mount(self, tmp_path):
        app = _Host(
            memory=_StubMemory(),
            entity_ids=("agent-1", "agent-2"),
            default_path=tmp_path / "out.tar.gz",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.query_one("#export-path", Input) is not None
            assert modal.query_one("#export-skills", Checkbox) is not None
            assert modal.query_one("#export-summary", Static) is not None

    @pytest.mark.asyncio
    async def test_summary_reflects_counts(self, tmp_path):
        app = _Host(
            memory=_StubMemory(),
            entity_ids=("a", "b", "c"),
            skill_entity_ids=("skill-x",),
            default_path=tmp_path / "out.tar.gz",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert "3 chains" in modal._summary_text()
            assert "1 skill" in modal._summary_text()


# ---------------------------------------------------------------------------
# Form snapshot
# ---------------------------------------------------------------------------


class TestRequest:
    @pytest.mark.asyncio
    async def test_current_request_reads_form(self, tmp_path):
        path = tmp_path / "lib.tar.gz"
        app = _Host(
            memory=_StubMemory(),
            entity_ids=("agent-1",),
            skill_entity_ids=(),
            default_path=path,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            req = modal.current_request()
            assert isinstance(req, ExportRequest)
            assert req.output_path == path
            assert req.include_skills is False

    @pytest.mark.asyncio
    async def test_skills_checkbox_default_when_skill_ids_present(
        self, tmp_path,
    ):
        app = _Host(
            memory=_StubMemory(),
            entity_ids=("agent-1",),
            skill_entity_ids=("skill-x",),
            default_path=tmp_path / "lib.tar.gz",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            req = modal.current_request()
            assert req.include_skills is True


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


class TestSubmit:
    @pytest.mark.asyncio
    async def test_submit_writes_bundle_and_dismisses(self, tmp_path):
        path = tmp_path / "out.tar.gz"
        app = _Host(
            memory=_StubMemory(),
            entity_ids=("agent-1",),
            default_path=path,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#export-btn-submit", Button).press()
            for _ in range(8):
                await pilot.pause()
            assert len(app.dismissed) == 1
            result = app.dismissed[0]
            assert isinstance(result, BundleExportResult)
            assert result.success is True
            assert result.chain_count == 1
            assert path.exists()

    @pytest.mark.asyncio
    async def test_submit_failure_keeps_modal_open(self, tmp_path):
        # Pass an unwritable parent (a file masquerading as a
        # directory) so the bundle helper fails before the
        # tarball write.
        sentinel = tmp_path / "blocker"
        sentinel.write_text("x")  # creates a file at parent
        bad_path = sentinel / "out.tar.gz"
        app = _Host(
            memory=_StubMemory(),
            entity_ids=("agent-1",),
            default_path=bad_path,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_submit()
            for _ in range(8):
                await pilot.pause()
            # No dismiss yet; result captures the error.
            assert app.dismissed == []
            assert modal.last_result is not None
            assert modal.last_result.error is not None


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_dismisses_with_none(self, tmp_path):
        app = _Host(
            memory=_StubMemory(),
            entity_ids=("agent-1",),
            default_path=tmp_path / "x.tar.gz",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_cancel()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed == [None]

    @pytest.mark.asyncio
    async def test_escape_dismisses_with_none(self, tmp_path):
        app = _Host(
            memory=_StubMemory(),
            entity_ids=("agent-1",),
            default_path=tmp_path / "x.tar.gz",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed == [None]


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import ExportModal as M
        from care.screens import ExportRequest as R

        assert M is ExportModal
        assert R is ExportRequest
