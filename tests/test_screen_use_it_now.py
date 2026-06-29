"""Pilot tests for `UseItNowModal` (TODO §3 P0)."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static, TextArea

from care.screens.use_it_now import (
    UseItNowModal,
    UseItNowResult,
)


# ---------------------------------------------------------------------------
# Pure projection (no Textual)
# ---------------------------------------------------------------------------


class TestSnippetRendering:
    def test_python_snippet_carries_entity_id_and_channel(self) -> None:
        modal = UseItNowModal(
            entity_id="chain-abc",
            version="v3",
            channel="stable",
        )
        body = modal.render_snippet("python")
        assert "from gigaevo_client import GigaEvoClient, GigaEvoConfig" in body
        assert "GigaEvoConfig.from_env()" in body
        assert "GigaEvoClient.from_config" in body
        assert '"chain-abc"' in body
        assert 'channel="stable"' in body
        assert "client.run(" not in body
        assert "get_chain(" in body
        assert "DAGExecutor" in body
        assert ".execute(chain.steps" in body

    def test_curl_snippet_uses_supplied_memory_base_url(self) -> None:
        modal = UseItNowModal(
            entity_id="chain-x",
            memory_base_url="https://memory.example.com",
        )
        body = modal.render_snippet("curl")
        assert "https://memory.example.com/v1/chains/chain-x" in body
        assert "CARE_MEMORY__API_KEY" in body
        assert "channel=latest" in body

    def test_curl_snippet_falls_back_to_env_var(self) -> None:
        modal = UseItNowModal(entity_id="chain-x")
        body = modal.render_snippet("curl")
        assert "${CARE_MEMORY__BASE_URL}" in body

    def test_cli_snippet_uses_care_run(self) -> None:
        modal = UseItNowModal(entity_id="chain-y")
        body = modal.render_snippet("cli")
        assert "care run chain-y" in body
        assert "--execute" in body
        assert "--task" in body

    def test_unknown_lang_raises(self) -> None:
        modal = UseItNowModal(entity_id="x")
        with pytest.raises(ValueError, match="unknown snippet lang"):
            modal.render_snippet("ruby")  # type: ignore[arg-type]


class TestConstruction:
    def test_empty_entity_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="entity_id"):
            UseItNowModal(entity_id="")

    def test_defaults(self) -> None:
        modal = UseItNowModal(entity_id="x")
        assert modal.version == "latest"
        assert modal.channel == "latest"
        assert modal.active_lang == "python"
        assert modal.display_name == ""


# ---------------------------------------------------------------------------
# Pilot — host scaffold
# ---------------------------------------------------------------------------


class _Host(App):
    def __init__(self, *, modal: UseItNowModal):
        super().__init__()
        self._modal = modal
        self.toasts: list[tuple[str, str]] = []
        self.dismissed: list[UseItNowResult] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self._modal, self._on_dismiss)

    def _on_dismiss(self, result: UseItNowResult | None) -> None:
        if result is not None:
            self.dismissed.append(result)

    def push_toast(
        self, message, *, severity="info", ttl=None,
    ) -> None:  # type: ignore[override]
        self.toasts.append((message, severity))


def _modal(app: _Host) -> UseItNowModal:
    for screen in app.screen_stack:
        if isinstance(screen, UseItNowModal):
            return screen
    raise AssertionError("UseItNowModal not on stack")


class TestCompose:
    @pytest.mark.asyncio
    async def test_mounts_with_meta_and_snippet_sections(self):
        app = _Host(modal=UseItNowModal(
            entity_id="chain-abc",
            version="v2",
            channel="stable",
            display_name="Forecaster",
        ))
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            title = modal.query_one("#use-it-now-title", Static)
            meta = modal.query_one("#use-it-now-meta", Static)
            ident = modal.query_one("#use-it-now-id", Static)
            snippet = modal.query_one(
                "#use-it-now-snippet", TextArea,
            )
            assert "Forecaster" in str(title.render())
            assert "version: v2" in str(meta.render())
            assert "channel: stable" in str(meta.render())
            assert "chain-abc" in str(ident.render())
            # Snippet defaults to python on mount (read-only code viewer).
            assert "GigaEvoClient" in snippet.text
            assert snippet.language == "python"


class TestBindings:
    @pytest.mark.asyncio
    async def test_cycle_lang_walks_python_curl_cli(self):
        app = _Host(modal=UseItNowModal(entity_id="chain-x"))
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            assert modal.active_lang == "python"
            modal.action_cycle_lang()
            assert modal.active_lang == "curl"
            modal.action_cycle_lang()
            assert modal.active_lang == "cli"
            modal.action_cycle_lang()
            assert modal.active_lang == "python"

    @pytest.mark.asyncio
    async def test_cycle_lang_updates_snippet_pane(self):
        app = _Host(modal=UseItNowModal(entity_id="chain-x"))
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            modal.action_cycle_lang()  # python -> curl
            await pilot.pause()
            snippet = modal.query_one(
                "#use-it-now-snippet", TextArea,
            )
            assert "curl" in snippet.text
            assert snippet.language == "bash"

    @pytest.mark.asyncio
    async def test_copy_id_logs_and_toasts(self, monkeypatch):
        captured: list[str] = []

        def _fake_copy(text: str) -> None:
            captured.append(text)

        monkeypatch.setattr(
            "care.runtime.clipboard.copy_text", _fake_copy,
        )
        app = _Host(modal=UseItNowModal(entity_id="chain-CC"))
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            modal.action_copy_id()
            await pilot.pause()
            assert captured == ["chain-CC"]
            assert ("copy_id", "chain-CC") in modal.action_log
            assert any(
                "Copied id" in m for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_copy_snippet_copies_active_lang(self, monkeypatch):
        captured: list[str] = []

        def _fake_copy(text: str) -> None:
            captured.append(text)

        monkeypatch.setattr(
            "care.runtime.clipboard.copy_text", _fake_copy,
        )
        app = _Host(modal=UseItNowModal(
            entity_id="chain-snip", version="v9",
        ))
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            # Cycle to cli then copy.
            modal.action_cycle_lang()
            modal.action_cycle_lang()  # now cli
            assert modal.active_lang == "cli"
            modal.action_copy_snippet()
            await pilot.pause()
            assert captured
            assert "care run chain-snip" in captured[0]

    @pytest.mark.asyncio
    async def test_close_dismisses_with_evolve_false(self):
        app = _Host(modal=UseItNowModal(entity_id="chain-x"))
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            modal.action_close()
            await pilot.pause()
            assert app.dismissed
            assert app.dismissed[0].closed is True
            assert app.dismissed[0].evolve_requested is False

    @pytest.mark.asyncio
    async def test_evolve_dismisses_with_evolve_true(self):
        app = _Host(modal=UseItNowModal(entity_id="chain-evolve"))
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            modal.action_evolve()
            await pilot.pause()
            assert app.dismissed
            assert app.dismissed[0].evolve_requested is True
            assert (
                "evolve", "chain-evolve",
            ) in modal.action_log


# ---------------------------------------------------------------------------
# ArtifactsScreen integration (§3 P0 hand-off)
# ---------------------------------------------------------------------------


class TestArtifactsIntegration:
    @pytest.mark.asyncio
    async def test_save_pushes_use_it_now_after_success(self):
        from care.runtime.session_artifacts import SessionArtifactStore
        from care.screens.artifacts import ArtifactsScreen
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                return "ENT-ui-test"

        class _Host2(App):
            def __init__(self, store):
                super().__init__()
                self.store = store
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

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="demo", summary="")
        app = _Host2(store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = next(
                s for s in app.screen_stack
                if isinstance(s, ArtifactsScreen)
            )
            screen.action_save()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            tag_modal.dismiss(TagEditorResult(submitted=True))
            for _ in range(8):
                await pilot.pause()
            # After the save worker completes the UseItNow
            # modal should land.
            assert any(
                isinstance(s, UseItNowModal)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_evolve_request_routes_through_push_evolution_for(
        self,
    ):
        """The artifacts screen's `_open_evolution_launch`
        helper should fire on the app's
        `_push_evolution_for` when the UseItNowModal
        dismisses with `evolve_requested=True`."""
        from care.runtime.session_artifacts import SessionArtifactStore
        from care.screens.artifacts import ArtifactsScreen

        invocations: list[str] = []

        class _Host3(App):
            def __init__(self):
                super().__init__()
                self.store = SessionArtifactStore()
                self.memory = object()
                self.toasts: list[tuple[str, str]] = []

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(ArtifactsScreen(self.store))

            def _push_evolution_for(self, entity_id: str) -> None:
                invocations.append(entity_id)

            def push_toast(self, message, *, severity="info", ttl=None):  # type: ignore[override]
                self.toasts.append((message, severity))

        app = _Host3()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = next(
                s for s in app.screen_stack
                if isinstance(s, ArtifactsScreen)
            )
            screen._open_evolution_launch("chain-evo-1")
            await pilot.pause()
            assert invocations == ["chain-evo-1"]


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self) -> None:
        from care.screens import (
            UseItNowModal as M,
            UseItNowResult as R,
        )

        assert M is UseItNowModal
        assert R is UseItNowResult
