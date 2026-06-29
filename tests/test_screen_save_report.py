"""Pilot tests for `SaveReport` modal (TODO §3 P1)."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from care.screens.save_report import (
    SaveReport,
    SaveReportResult,
    SaveReportRow,
    _format_entity_id,
    _format_error,
    _format_status,
)


# ---------------------------------------------------------------------------
# Pure formatters + dataclass behaviour
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_status_success(self):
        row = SaveReportRow(
            artifact_id="a", title="t", status="success",
        )
        assert _format_status(row) == "✓ success"

    def test_format_status_failure(self):
        row = SaveReportRow(
            artifact_id="a", title="t", status="failure",
        )
        assert _format_status(row) == "✗ failure"

    def test_format_entity_id_empty(self):
        assert _format_entity_id("") == "—"

    def test_format_entity_id_short(self):
        assert _format_entity_id("abc-123") == "abc-123"

    def test_format_entity_id_truncates_long(self):
        out = _format_entity_id("a" * 30)
        assert len(out) <= 16
        assert out.endswith("…")

    def test_format_error_empty(self):
        assert _format_error("") == ""

    def test_format_error_truncates_long(self):
        long = "x" * 100
        out = _format_error(long)
        assert len(out) <= 56
        assert out.endswith("…")


class TestRowDataclass:
    def test_ok_property(self):
        success = SaveReportRow(
            artifact_id="x", title="t", status="success",
        )
        assert success.ok is True
        failure = SaveReportRow(
            artifact_id="x", title="t", status="failure",
        )
        assert failure.ok is False


# ---------------------------------------------------------------------------
# Pilot scaffolding
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(
        self, rows: tuple[SaveReportRow, ...] = (),
    ):
        super().__init__()
        self._rows = rows
        self.dismissed: list[SaveReportResult] = []
        self.toasts: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(SaveReport(self._rows), self._on_dismiss)

    def _on_dismiss(self, result: SaveReportResult | None) -> None:
        if result is not None:
            self.dismissed.append(result)

    def push_toast(
        self, message, *, severity="info", ttl=None,
    ) -> None:  # type: ignore[override]
        self.toasts.append((message, severity))


def _report(app: _Host) -> SaveReport:
    for s in app.screen_stack:
        if isinstance(s, SaveReport):
            return s
    raise AssertionError("SaveReport not on stack")


class TestCompose:
    @pytest.mark.asyncio
    async def test_mount_does_not_raise(self):
        rows = (
            SaveReportRow(
                artifact_id="a", title="t",
                status="success", entity_id="ENT-1",
            ),
        )
        app = _Host(rows)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _report(app)
            table = modal.query_one(
                "#save-report-table", DataTable,
            )
            assert table.row_count == 1

    @pytest.mark.asyncio
    async def test_summary_text_all_success(self):
        rows = (
            SaveReportRow(
                artifact_id="a", title="t",
                status="success", entity_id="ENT-1",
            ),
            SaveReportRow(
                artifact_id="b", title="u",
                status="success", entity_id="ENT-2",
            ),
        )
        app = _Host(rows)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _report(app)
            assert modal.saved_count == 2
            assert modal.failed_count == 0
            summary = modal.query_one(
                "#save-report-summary", Static,
            )
            assert "saved all 2" in str(summary.render())

    @pytest.mark.asyncio
    async def test_summary_text_partial(self):
        rows = (
            SaveReportRow(
                artifact_id="a", title="t",
                status="success", entity_id="ENT-1",
            ),
            SaveReportRow(
                artifact_id="b", title="u", status="failure",
                error="boom",
            ),
        )
        app = _Host(rows)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _report(app)
            summary = modal.query_one(
                "#save-report-summary", Static,
            )
            text = str(summary.render())
            assert "saved 1 of 2" in text
            assert "1 failed" in text


class TestActions:
    @pytest.mark.asyncio
    async def test_close_dismisses_empty(self):
        rows = (
            SaveReportRow(
                artifact_id="a", title="t",
                status="success", entity_id="ENT-1",
            ),
        )
        app = _Host(rows)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _report(app)
            modal.action_close()
            await pilot.pause()
            assert app.dismissed
            assert app.dismissed[0].show_id == ""
            assert app.dismissed[0].closed is True

    @pytest.mark.asyncio
    async def test_show_id_on_success_row_dismisses_with_id(
        self,
    ):
        rows = (
            SaveReportRow(
                artifact_id="art-A", title="t",
                status="success", entity_id="ENT-X",
            ),
        )
        app = _Host(rows)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _report(app)
            modal.action_show_id()
            await pilot.pause()
            assert app.dismissed
            assert app.dismissed[0].show_id == "art-A"
            assert app.dismissed[0].closed is False

    @pytest.mark.asyncio
    async def test_show_id_on_failure_row_toasts(self):
        rows = (
            SaveReportRow(
                artifact_id="art-B", title="bad",
                status="failure", error="503",
            ),
        )
        app = _Host(rows)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _report(app)
            modal.action_show_id()
            await pilot.pause()
            assert any(
                "failed" in m for m, _ in app.toasts
            )
            # No dismiss yet — modal stays open.
            assert not app.dismissed


class TestSaveAllIntegration:
    """ArtifactsScreen pushes SaveReport when batch ≥ 5 or
    any row failed."""

    @pytest.mark.asyncio
    async def test_large_batch_pushes_report(self):
        from care.runtime.session_artifacts import (
            SessionArtifactStore,
        )
        from care.screens.artifacts import ArtifactsScreen
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        class _Mem:
            def __init__(self): self.calls = 0
            def save_chain(self, chain, *, name=None, tags=None):
                self.calls += 1
                return f"ENT-{self.calls}"

        class _Host2(App):
            def __init__(self):
                super().__init__()
                self.store = SessionArtifactStore()
                self.memory = _Mem()
                self.toasts: list[tuple[str, str]] = []

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(ArtifactsScreen(self.store))

            def push_toast(
                self, message, *, severity="info", ttl=None,
            ) -> None:  # type: ignore[override]
                self.toasts.append((message, severity))

        app = _Host2()
        # 5 unsaved chains → triggers SaveReport.
        for i in range(5):
            app.store.append_chain(
                chain={}, title=f"c{i}", summary="",
            )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = next(
                s for s in app.screen_stack
                if isinstance(s, ArtifactsScreen)
            )
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            tag_modal.dismiss(TagEditorResult(submitted=True))
            for _ in range(10):
                await pilot.pause()
            # SaveReport should land on the stack.
            assert any(
                isinstance(s, SaveReport)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_small_clean_batch_skips_report(self):
        from care.runtime.session_artifacts import (
            SessionArtifactStore,
        )
        from care.screens.artifacts import ArtifactsScreen
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                return "ENT-OK"

        class _Host3(App):
            def __init__(self):
                super().__init__()
                self.store = SessionArtifactStore()
                self.memory = _Mem()
                self.toasts: list[tuple[str, str]] = []

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(ArtifactsScreen(self.store))

            def push_toast(
                self, message, *, severity="info", ttl=None,
            ) -> None:  # type: ignore[override]
                self.toasts.append((message, severity))

        app = _Host3()
        # 2 unsaved chains, all succeed → no report.
        for i in range(2):
            app.store.append_chain(
                chain={}, title=f"c{i}", summary="",
            )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = next(
                s for s in app.screen_stack
                if isinstance(s, ArtifactsScreen)
            )
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            tag_modal.dismiss(TagEditorResult(submitted=True))
            for _ in range(10):
                await pilot.pause()
            assert not any(
                isinstance(s, SaveReport)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_any_failure_pushes_report_even_small(self):
        from care.runtime.session_artifacts import (
            SessionArtifactStore,
        )
        from care.screens.artifacts import ArtifactsScreen
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        class _MemBad:
            def __init__(self): self.calls = 0
            def save_chain(self, chain, *, name=None, tags=None):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("503")
                return f"ENT-{self.calls}"

        class _Host4(App):
            def __init__(self):
                super().__init__()
                self.store = SessionArtifactStore()
                self.memory = _MemBad()
                self.toasts: list[tuple[str, str]] = []

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(ArtifactsScreen(self.store))

            def push_toast(
                self, message, *, severity="info", ttl=None,
            ) -> None:  # type: ignore[override]
                self.toasts.append((message, severity))

        app = _Host4()
        for i in range(2):
            app.store.append_chain(
                chain={}, title=f"c{i}", summary="",
            )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = next(
                s for s in app.screen_stack
                if isinstance(s, ArtifactsScreen)
            )
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            tag_modal.dismiss(TagEditorResult(submitted=True))
            for _ in range(10):
                await pilot.pause()
            # Failure → report should land.
            assert any(
                isinstance(s, SaveReport)
                for s in app.screen_stack
            )


class TestShowIdRoutesToUseItNow:
    """§3 P2 — pressing Enter on a successful SaveReport row
    pushes UseItNowModal for that artifact (re-using the
    existing `_push_use_it_now` plumbing)."""

    @pytest.mark.asyncio
    async def test_show_id_pushes_use_it_now_modal(self):
        from care.runtime.session_artifacts import (
            SessionArtifactStore,
        )
        from care.screens.artifacts import ArtifactsScreen
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )
        from care.screens.use_it_now import UseItNowModal

        class _Mem:
            def __init__(self):
                self.calls = 0

            def save_chain(self, chain, *, name=None, tags=None):
                self.calls += 1
                return f"ENT-{self.calls:04d}"

        class _HostUIN(App):
            def __init__(self):
                super().__init__()
                self.store = SessionArtifactStore()
                self.memory = _Mem()
                self.toasts: list[tuple[str, str]] = []

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(ArtifactsScreen(self.store))

            def push_toast(
                self, message, *, severity="info", ttl=None,
            ) -> None:  # type: ignore[override]
                self.toasts.append((message, severity))

        app = _HostUIN()
        for i in range(5):
            app.store.append_chain(
                chain={"name": f"c{i}"},
                title=f"chain-{i}",
                summary="",
            )
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            artifacts = next(
                s for s in app.screen_stack
                if isinstance(s, ArtifactsScreen)
            )
            artifacts.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            tag_modal.dismiss(TagEditorResult(submitted=True))
            for _ in range(10):
                await pilot.pause()
            report = next(
                s for s in app.screen_stack
                if isinstance(s, SaveReport)
            )
            # Pick the first successful row + drive show_id.
            ok_row = next(r for r in report.rows if r.ok)
            table = report.query_one("#save-report-table", DataTable)
            idx = next(
                i for i, r in enumerate(report.rows)
                if r.artifact_id == ok_row.artifact_id
            )
            table.cursor_coordinate = (idx, 0)
            await pilot.pause()
            report.action_show_id()
            for _ in range(8):
                await pilot.pause()
            # SaveReport dismissed, UseItNowModal pushed in
            # its place — `_on_dismiss` looked up the artifact
            # in the store + routed through `_push_use_it_now`.
            assert not any(
                isinstance(s, SaveReport)
                for s in app.screen_stack
            )
            uin = next(
                (s for s in app.screen_stack
                 if isinstance(s, UseItNowModal)),
                None,
            )
            assert uin is not None, (
                "UseItNowModal should land on the stack after "
                "SaveReport's show_id dismiss"
            )
            # Modal carries the same entity_id the save flow
            # minted for that artifact.
            artifact = app.store.get(ok_row.artifact_id)
            assert uin.entity_id == artifact.memory_entity_id
            assert uin.display_name == artifact.title


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import (
            SaveReport as M,
            SaveReportResult as R,
            SaveReportRow as W,
        )

        assert M is SaveReport
        assert R is SaveReportResult
        assert W is SaveReportRow
