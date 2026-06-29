"""Pilot tests for `CareFooter` (TODO §1.1 P0.4).

Mounts the widget inside a minimal host App, drives it via
`set_model` / `refresh_from_app`, and asserts the hint Static
children update across screen-scope changes.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from care.runtime.global_bindings import (
    FooterHint,
    FooterModel,
    GlobalBinding,
    build_footer,
)
from care.widgets.footer import CareFooter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_model_is_empty_footer(self):
        widget = CareFooter()
        assert len(widget.model) == 0
        assert widget.model.active_screen == ""

    def test_explicit_model(self):
        model = FooterModel(
            hints=(
                FooterHint(key="Ctrl+P", label="Palette", action_id="open_command_palette"),
            ),
            active_screen="LibraryScreen",
        )
        widget = CareFooter(model)
        assert widget.model is model


# ---------------------------------------------------------------------------
# Mount + compose
# ---------------------------------------------------------------------------


class _FooterHostApp(App):
    def __init__(self, *, model: FooterModel | None = None) -> None:
        super().__init__()
        self._initial_model = model

    def compose(self) -> ComposeResult:
        self.footer = CareFooter(self._initial_model)
        yield self.footer


class TestStatusSegment:
    @pytest.mark.asyncio
    async def test_set_status_updates_segment(self):
        app = _FooterHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.footer.set_status("▶ 3 evolving")
            await pilot.pause()
            seg = app.footer.query_one(f"#{CareFooter.STATUS_ID}", Static)
            assert "3 evolving" in str(seg.render())
            assert app.footer.status_text == "▶ 3 evolving"

    @pytest.mark.asyncio
    async def test_status_survives_hint_recompose(self):
        app = _FooterHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.footer.set_status("▶ 2 evolving")
            await pilot.pause()
            # A hint refresh recomposes children — status must persist.
            app.footer.refresh_from_app(active_screen="LibraryScreen")
            await pilot.pause()
            seg = app.footer.query_one(f"#{CareFooter.STATUS_ID}", Static)
            assert "2 evolving" in str(seg.render())

    @pytest.mark.asyncio
    async def test_empty_status_clears_segment(self):
        app = _FooterHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.footer.set_status("▶ 1 evolving")
            await pilot.pause()
            app.footer.set_status("")
            await pilot.pause()
            seg = app.footer.query_one(f"#{CareFooter.STATUS_ID}", Static)
            assert str(seg.render()).strip() == ""


class TestMount:
    @pytest.mark.asyncio
    async def test_renders_spacer_and_no_hints_when_empty(self):
        app = _FooterHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Spacer always present.
            assert (
                app.footer.query_one("#footer-spacer", Static) is not None
            )
            # No hint children when the model is empty.
            assert len(app.footer.query(f".{CareFooter.HINT_CLASS}")) == 0

    @pytest.mark.asyncio
    async def test_renders_hint_per_binding(self):
        model = build_footer(
            active_screen="LibraryScreen", scope="screen",
        )
        app = _FooterHostApp(model=model)
        async with app.run_test() as pilot:
            await pilot.pause()
            hints = app.footer.query(f".{CareFooter.HINT_CLASS}")
            assert len(hints) == len(model.hints)

    @pytest.mark.asyncio
    async def test_each_hint_id_namespaced_by_action(self):
        model = build_footer(active_screen="LibraryScreen", scope="screen")
        app = _FooterHostApp(model=model)
        async with app.run_test() as pilot:
            await pilot.pause()
            for hint in model.hints:
                node = app.footer.query_one(
                    f"#footer-hint-{hint.action_id}", Static
                )
                assert node is not None

    @pytest.mark.asyncio
    async def test_modal_scope_drops_screen_only_hints(self):
        # Modal scope hides Save + Re-run (screen-scoped).
        model = build_footer(active_screen="modal", scope="modal")
        app = _FooterHostApp(model=model)
        async with app.run_test() as pilot:
            await pilot.pause()
            with pytest.raises(Exception):
                # save_artifact hint shouldn't exist on a modal.
                app.footer.query_one(
                    "#footer-hint-save_artifact", Static
                )
            # back / quit / palette all still present.
            assert app.footer.query_one(
                "#footer-hint-back", Static
            ) is not None
            assert app.footer.query_one(
                "#footer-hint-quit", Static
            ) is not None


# ---------------------------------------------------------------------------
# set_model rebuilds children
# ---------------------------------------------------------------------------


class TestSetModel:
    @pytest.mark.asyncio
    async def test_set_model_rebuilds_hints(self):
        # Start empty, swap to screen-scoped model, hints appear.
        app = _FooterHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert len(app.footer.query(f".{CareFooter.HINT_CLASS}")) == 0
            new_model = build_footer(scope="screen")
            app.footer.set_model(new_model)
            await pilot.pause()
            hints = app.footer.query(f".{CareFooter.HINT_CLASS}")
            assert len(hints) == len(new_model.hints)
            assert app.footer.model is new_model

    @pytest.mark.asyncio
    async def test_set_model_replaces_old_hints(self):
        # screen → modal should drop the screen-only hints
        # and leave only the always-scoped ones.
        screen_model = build_footer(scope="screen")
        app = _FooterHostApp(model=screen_model)
        async with app.run_test() as pilot:
            await pilot.pause()
            initial = len(app.footer.query(f".{CareFooter.HINT_CLASS}"))
            modal_model = build_footer(scope="modal")
            app.footer.set_model(modal_model)
            await pilot.pause()
            after = len(app.footer.query(f".{CareFooter.HINT_CLASS}"))
            assert after < initial
            assert after == len(modal_model.hints)

    def test_set_model_before_mount_no_crash(self):
        widget = CareFooter()
        new_model = FooterModel(
            hints=(
                FooterHint(key="X", label="X", action_id="quit"),
            )
        )
        widget.set_model(new_model)
        assert widget.model is new_model


# ---------------------------------------------------------------------------
# refresh_from_app
# ---------------------------------------------------------------------------


class TestRefreshFromApp:
    @pytest.mark.asyncio
    async def test_refresh_builds_via_build_footer(self):
        app = _FooterHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.footer.refresh_from_app(
                active_screen="LibraryScreen", scope="screen",
            )
            await pilot.pause()
            expected = build_footer(
                active_screen="LibraryScreen", scope="screen",
            )
            assert app.footer.model == expected
            assert app.footer.model.active_screen == "LibraryScreen"

    @pytest.mark.asyncio
    async def test_refresh_with_modal_scope_filters_hints(self):
        app = _FooterHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.footer.refresh_from_app(
                active_screen="ConfirmModal", scope="modal",
            )
            await pilot.pause()
            action_ids = {h.action_id for h in app.footer.model.hints}
            assert "save_artifact" not in action_ids
            assert "rerun_artifact" not in action_ids

    @pytest.mark.asyncio
    async def test_refresh_with_custom_registry(self):
        # Provide a custom one-binding registry.
        custom = (
            GlobalBinding(
                action_id="quit", key="Ctrl+X", label="Bye",
                scope="always",
            ),
        )
        app = _FooterHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.footer.refresh_from_app(
                active_screen="Whatever",
                scope="screen",
                registry=custom,
            )
            await pilot.pause()
            assert len(app.footer.model.hints) == 1
            assert app.footer.model.hints[0].key == "Ctrl+X"

    @pytest.mark.asyncio
    async def test_multi_transition_refresh(self):
        # Simulate screen-stack transitions calling refresh_from_app
        # on every push/pop.
        app = _FooterHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            for screen, scope in [
                ("LibraryScreen", "screen"),
                ("CommandPaletteModal", "modal"),
                ("InspectionScreen", "screen"),
            ]:
                app.footer.refresh_from_app(
                    active_screen=screen, scope=scope,
                )
                await pilot.pause()
                assert app.footer.model.active_screen == screen


# ---------------------------------------------------------------------------
# Hint formatting
# ---------------------------------------------------------------------------


class TestHintFormatting:
    def test_format_hint_bracketed(self):
        assert CareFooter._format_hint("Ctrl+P", "Palette") == "[Ctrl+P] Palette"

    def test_format_hint_escape(self):
        assert CareFooter._format_hint("Esc", "Back") == "[Esc] Back"

    def test_hint_id_namespaced(self):
        assert CareFooter._hint_id_for("save_artifact") == "footer-hint-save_artifact"


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_widgets_re_exports_care_footer(self):
        from care.widgets import CareFooter as ReExported

        assert ReExported is CareFooter
