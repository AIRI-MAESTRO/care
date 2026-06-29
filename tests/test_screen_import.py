"""Pilot tests for ImportModal (TODO §1.1 P0.31).

Exercises:
* Composition — path Input + collision RadioSet + dry-run
  Checkbox + preview Static.
* On-mount preview parses an existing tarball via
  `read_bundle_manifest`.
* Submit fires `import_library_bundle` and dismisses on
  success.
* Dry-run keeps the modal open.
* Cancel / Escape dismiss with `None`.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, RadioSet

from care.runtime.library_bundle import (
    BundleImportResult,
    export_library_bundle,
)
from care.screens.import_bundle import ImportModal, ImportRequest


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(self):
        self.bulk_save_calls: list = []

    def get_chain_dict(self, entity_id, channel):
        return {
            "entity_id": entity_id,
            "entity_type": "chain",
            "channel": channel,
            "content": {"steps": [{"name": "fetch"}]},
            "meta": {
                "display_name": entity_id.title(),
                "tags": [],
                "name": entity_id,
            },
        }

    def bulk_save(self, items, **kw):
        self.bulk_save_calls.append({"items": list(items), **kw})
        return {"created": [f"new-{i}" for i in range(len(items))]}


class _StubMemory:
    def __init__(self):
        self.client = _StubClient()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def bundle_path(tmp_path):
    """Build a real bundle via the shipped export helper so
    the import path exercises the actual on-disk format."""
    path = tmp_path / "lib.tar.gz"
    await export_library_bundle(
        _StubMemory(),
        ("agent-1", "agent-2"),
        path,
    )
    assert path.exists()
    return path


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self.dismissed: list = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(result):
            self.dismissed.append(result)

        self.push_screen(ImportModal(**self._kwargs), _on_dismiss)


def _modal(app: App) -> ImportModal:
    s = app.screen_stack[-1]
    assert isinstance(s, ImportModal)
    return s


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_widgets_mount(self, tmp_path):
        app = _Host(
            memory=_StubMemory(),
            default_path=tmp_path / "missing.tar.gz",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.query_one("#import-path", Input) is not None
            assert modal.query_one(
                "#import-collision", RadioSet,
            ) is not None
            assert modal.query_one("#import-dry-run", Checkbox) is not None

    @pytest.mark.asyncio
    async def test_missing_file_renders_placeholder(self, tmp_path):
        app = _Host(
            memory=_StubMemory(),
            default_path=tmp_path / "missing.tar.gz",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.manifest is None
            assert modal.preview_error is None


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


class TestPreview:
    @pytest.mark.asyncio
    async def test_existing_bundle_parses_on_mount(self, bundle_path):
        app = _Host(
            memory=_StubMemory(),
            default_path=bundle_path,
        )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            assert modal.manifest is not None
            assert len(modal.manifest.chains) == 2

    @pytest.mark.asyncio
    async def test_invalid_tarball_lands_on_error(self, tmp_path):
        bad = tmp_path / "garbage.tar.gz"
        bad.write_text("not a tarball")
        app = _Host(
            memory=_StubMemory(),
            default_path=bad,
        )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            assert modal.manifest is None
            assert modal.preview_error is not None


# ---------------------------------------------------------------------------
# Form snapshot
# ---------------------------------------------------------------------------


class TestRequest:
    @pytest.mark.asyncio
    async def test_current_request_defaults(self, tmp_path):
        app = _Host(
            memory=_StubMemory(),
            default_path=tmp_path / "x.tar.gz",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            req = modal.current_request()
            assert isinstance(req, ImportRequest)
            assert req.on_collision == "skip"
            assert req.dry_run is False


# ---------------------------------------------------------------------------
# Submit + dismiss
# ---------------------------------------------------------------------------


class TestSubmit:
    @pytest.mark.asyncio
    async def test_submit_imports_and_dismisses(self, bundle_path):
        memory = _StubMemory()
        app = _Host(
            memory=memory,
            default_path=bundle_path,
        )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            assert modal.manifest is not None
            modal.query_one("#import-btn-submit", Button).press()
            for _ in range(8):
                await pilot.pause()
            assert len(app.dismissed) == 1
            result = app.dismissed[0]
            assert isinstance(result, BundleImportResult)
            assert result.success is True
            assert memory.client.bulk_save_calls != []

    @pytest.mark.asyncio
    async def test_dry_run_keeps_modal_open(self, bundle_path):
        memory = _StubMemory()
        app = _Host(
            memory=memory,
            default_path=bundle_path,
        )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            modal.query_one("#import-dry-run", Checkbox).value = True
            await pilot.pause()
            modal.action_submit()
            for _ in range(6):
                await pilot.pause()
            # No bulk_save call, no dismiss.
            assert memory.client.bulk_save_calls == []
            assert app.dismissed == []
            assert modal.last_result is not None

    @pytest.mark.asyncio
    async def test_submit_without_manifest_is_noop(self, tmp_path):
        app = _Host(
            memory=_StubMemory(),
            default_path=tmp_path / "missing.tar.gz",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert modal.importing is False
            assert app.dismissed == []


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_dismisses_with_none(self, tmp_path):
        app = _Host(
            memory=_StubMemory(),
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
        from care.screens import ImportModal as M
        from care.screens import ImportRequest as R

        assert M is ImportModal
        assert R is ImportRequest
