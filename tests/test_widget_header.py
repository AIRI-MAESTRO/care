"""Pilot tests for `CareHeader` (TODO §1.1 P0.3).

Mounts the widget inside a minimal host App, drives it via
`set_model` / `refresh_from_app`, and asserts the three Static
children update.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from care.runtime.global_bindings import HeaderModel, build_header
from care.widgets.header import CareHeader


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_model_is_empty_header(self):
        widget = CareHeader()
        assert widget.model.title == "MAESTRO"
        assert widget.model.breadcrumb == ()
        assert widget.model.version == ""

    def test_explicit_model(self):
        model = HeaderModel(
            title="MAESTRO", breadcrumb=("Library", "v3"), version="0.5.0",
        )
        widget = CareHeader(model)
        assert widget.model is model


# ---------------------------------------------------------------------------
# Mount + compose
# ---------------------------------------------------------------------------


class _HeaderHostApp(App):
    def __init__(
        self,
        *,
        model: HeaderModel | None = None,
        badge: str = "",
        library_button: bool = False,
    ) -> None:
        super().__init__()
        self._initial_model = model
        self._initial_badge = badge
        self._initial_library_button = library_button

    def compose(self) -> ComposeResult:
        self.header = CareHeader(
            self._initial_model,
            badge=self._initial_badge,
            library_button=self._initial_library_button,
        )
        yield self.header


class TestMount:
    @pytest.mark.asyncio
    async def test_renders_three_statics(self):
        app = _HeaderHostApp(
            model=HeaderModel(
                title="MAESTRO",
                breadcrumb=("Library", "Weather report"),
                version="0.3.0",
            )
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            # The three Statics are queryable by id.
            assert app.header.query_one("#header-title", Static) is not None
            assert (
                app.header.query_one("#header-breadcrumb", Static) is not None
            )
            assert app.header.query_one("#header-version", Static) is not None
            # Model carries the rendered values.
            assert app.header.model.title == "MAESTRO"
            assert app.header.model.breadcrumb_text == "Library › Weather report"
            assert app.header.model.version == "0.3.0"

    @pytest.mark.asyncio
    async def test_default_renders_empty_breadcrumb(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.header.model.breadcrumb_text == ""

    @pytest.mark.asyncio
    async def test_breadcrumb_separator(self):
        # `HeaderModel.breadcrumb_text` joins with " › ".
        app = _HeaderHostApp(
            model=HeaderModel(breadcrumb=("A", "B", "C")),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.header.model.breadcrumb_text == "A › B › C"


# ---------------------------------------------------------------------------
# set_model + refresh_from_app
# ---------------------------------------------------------------------------


class TestSetModel:
    @pytest.mark.asyncio
    async def test_set_model_repaints(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            new_model = HeaderModel(
                title="MAESTRO",
                breadcrumb=("Settings",),
                version="0.4.0",
            )
            app.header.set_model(new_model)
            await pilot.pause()
            assert app.header.model is new_model
            # Statics still present (no torn DOM).
            assert app.header.query_one("#header-title", Static) is not None
            assert (
                app.header.query_one("#header-breadcrumb", Static) is not None
            )
            assert app.header.query_one("#header-version", Static) is not None

    def test_set_model_before_mount_no_crash(self):
        widget = CareHeader()
        # No mount → `is_mounted` False → set_model just
        # updates the snapshot without trying to query the
        # children.
        new_model = HeaderModel(version="0.9.0")
        widget.set_model(new_model)
        assert widget.model is new_model

    @pytest.mark.asyncio
    async def test_refresh_from_app_builds_via_build_header(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.refresh_from_app(
                active_screen="InspectionScreen",
                breadcrumb=["Library", "Weather"],
                version="0.5.0",
            )
            await pilot.pause()
            # Same shape `build_header` would have produced.
            expected = build_header(
                active_screen="InspectionScreen",
                breadcrumb=["Library", "Weather"],
                version="0.5.0",
            )
            assert app.header.model == expected
            assert app.header.model.active_screen == "InspectionScreen"

    @pytest.mark.asyncio
    async def test_refresh_filters_empty_breadcrumb(self):
        # `build_header` filters empty strings — header
        # surfaces that without a separate check.
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.refresh_from_app(
                breadcrumb=["A", "", "B"], version="x",
            )
            await pilot.pause()
            assert app.header.model.breadcrumb == ("A", "B")


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


class TestVersionDetection:
    @pytest.mark.asyncio
    async def test_reads_app_version_attribute(self):
        class _VersionedHostApp(App):
            version = "1.2.3"

            def compose(self) -> ComposeResult:
                self.header = CareHeader()
                yield self.header

        app = _VersionedHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.refresh_from_app(breadcrumb=("x",))
            await pilot.pause()
            assert app.header.model.version == "1.2.3"

    @pytest.mark.asyncio
    async def test_reads_config_version_attribute(self):
        class _Config:
            version = "9.9.9"

        class _ConfigHostApp(App):
            config = _Config()

            def compose(self) -> ComposeResult:
                self.header = CareHeader()
                yield self.header

        app = _ConfigHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.refresh_from_app()
            await pilot.pause()
            assert app.header.model.version == "9.9.9"

    @pytest.mark.asyncio
    async def test_explicit_version_overrides_app_detection(self):
        class _VersionedHostApp(App):
            version = "0.0.0"

            def compose(self) -> ComposeResult:
                self.header = CareHeader()
                yield self.header

        app = _VersionedHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.refresh_from_app(version="explicit")
            await pilot.pause()
            assert app.header.model.version == "explicit"

    @pytest.mark.asyncio
    async def test_missing_version_falls_back_to_empty(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.refresh_from_app()
            await pilot.pause()
            assert app.header.model.version == ""


class TestChatNavButtons:
    """The Chat top-bar nav cluster: My chains / Evolution links (left of
    the Artifacts pill) + a Help link, all hidden by default."""

    @pytest.mark.asyncio
    async def test_nav_buttons_hidden_by_default(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            for wid in (
                "#header-evolution-btn", "#header-help-btn",
            ):
                assert app.header.query_one(wid, Static).display is False
            assert app.header.evolution_button_visible is False
            assert app.header.help_button_visible is False

    @pytest.mark.asyncio
    async def test_toggles_show_buttons(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.set_library_button(True)
            app.header.set_evolution_button(True)
            app.header.set_help_button(True)
            await pilot.pause()
            assert app.header.query_one("#header-library-btn", Static).display
            assert app.header.query_one("#header-evolution-btn", Static).display
            assert app.header.query_one("#header-help-btn", Static).display

    @pytest.mark.asyncio
    async def test_my_chains_and_evolution_sit_left_of_artifacts(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.set_library_button(True)
            app.header.set_evolution_button(True)
            app.header.set_help_button(True)
            app.header.set_artifact_pill("Artifacts (1 unsaved)")
            await pilot.pause()
            lib = app.header.query_one("#header-library-btn", Static).region.x
            evo = app.header.query_one("#header-evolution-btn", Static).region.x
            art = app.header.query_one("#header-artifact-pill", Static).region.x
            hlp = app.header.query_one("#header-help-btn", Static).region.x
            assert lib < evo < art < hlp

    @pytest.mark.asyncio
    async def test_labels_localized(self):
        from care.runtime.i18n import t
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert t("header.evolution") in str(
                app.header.query_one("#header-evolution-btn", Static).render()
            )
            assert t("header.help") in str(
                app.header.query_one("#header-help-btn", Static).render()
            )


# ---------------------------------------------------------------------------
# Integration with shipped HeaderModel + build_header
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_uses_shipped_header_model(self):
        # Round-trip a build_header output through the widget
        # without losing fidelity.
        model = build_header(
            active_screen="LibraryScreen",
            breadcrumb=["Library"],
            version="0.3.0",
        )
        widget = CareHeader(model)
        assert widget.model.active_screen == "LibraryScreen"
        assert widget.model.breadcrumb == ("Library",)
        assert widget.model.breadcrumb_text == "Library"

    @pytest.mark.asyncio
    async def test_consume_changes_on_every_refresh(self):
        # Simulates the `CareApp.watch_current_screen` flow —
        # one refresh per screen transition.
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            transitions = [
                ("LibraryScreen", ()),
                ("InspectionScreen", ("Library", "Weather")),
                ("EditAgentScreen", ("Library", "Weather", "Edit")),
            ]
            for screen_name, breadcrumb in transitions:
                app.header.refresh_from_app(
                    active_screen=screen_name,
                    breadcrumb=breadcrumb,
                    version="0.3.0",
                )
                await pilot.pause()
                assert app.header.model.active_screen == screen_name
                assert app.header.model.breadcrumb == breadcrumb


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_widgets_re_exports_care_header(self):
        from care.widgets import CareHeader as ReExported

        assert ReExported is CareHeader


class TestBadge:
    """Phase 1 P1 — mode badge between breadcrumb and version."""

    @pytest.mark.asyncio
    async def test_badge_widget_mounts(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.header.query_one("#header-badge", Static) is not None
            # Empty by default → collapsed.
            assert app.header.badge == ""
            assert app.header.query_one("#header-badge", Static).display is False

    @pytest.mark.asyncio
    async def test_initial_badge_kwarg_shown_on_mount(self):
        app = _HeaderHostApp(badge="AD-HOC")
        async with app.run_test() as pilot:
            await pilot.pause()
            badge = app.header.query_one("#header-badge", Static)
            assert app.header.badge == "AD-HOC"
            assert badge.display is True
            assert "AD-HOC" in str(badge.render())

    @pytest.mark.asyncio
    async def test_set_badge_repaints_and_toggles_visibility(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            badge = app.header.query_one("#header-badge", Static)
            # Show.
            app.header.set_badge("PROD")
            await pilot.pause()
            assert badge.display is True
            assert "PROD" in str(badge.render())
            # Hide on empty.
            app.header.set_badge("")
            await pilot.pause()
            assert badge.display is False

    def test_set_badge_before_mount_no_crash(self):
        widget = CareHeader()
        widget.set_badge("AD-HOC")
        # Stash survives — mount will apply.
        assert widget.badge == "AD-HOC"


class TestLibraryButton:
    """Quick-access Library link sitting just left of the mode badge.
    Hidden by default; the Chat surface turns it on."""

    @pytest.mark.asyncio
    async def test_library_button_hidden_by_default(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.header.query_one("#header-library-btn", Static)
            assert btn is not None
            assert app.header.library_button_visible is False
            assert btn.display is False

    @pytest.mark.asyncio
    async def test_set_library_button_toggles_visibility(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.header.query_one("#header-library-btn", Static)
            app.header.set_library_button(True)
            await pilot.pause()
            assert btn.display is True
            app.header.set_library_button(False)
            await pilot.pause()
            assert btn.display is False

    @pytest.mark.asyncio
    async def test_initial_library_kwarg_shown_on_mount(self):
        app = _HeaderHostApp(library_button=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.header.query_one("#header-library-btn", Static)
            assert app.header.library_button_visible is True
            assert btn.display is True

    def test_set_library_button_before_mount_no_crash(self):
        widget = CareHeader()
        widget.set_library_button(True)
        assert widget.library_button_visible is True

    @pytest.mark.asyncio
    async def test_library_button_relocalizes_with_ui_language(self):
        from care.runtime import i18n

        i18n.set_ui_language("ru")
        app = _HeaderHostApp(library_button=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.header.query_one("#header-library-btn", Static)
            assert str(btn.render()) == "Мои цепочки"
            i18n.set_ui_language("en")
            app.header.relocalize()
            await pilot.pause()
            assert str(btn.render()) == "My chains"
        i18n.set_ui_language("en")


class TestBackHint:
    """The `Esc to go back` hint sits next to the breadcrumb on every
    screen except Chat, so the user always sees how to leave a pushed
    screen and climb back toward Chat."""

    @pytest.mark.asyncio
    async def test_hint_widget_mounts_collapsed_by_default(self):
        # Default model has an empty `active_screen` → nothing to go
        # back from yet, so the hint stays collapsed.
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            hint = app.header.query_one("#header-back-hint", Static)
            assert hint is not None
            assert hint.display is False
            assert "Esc" in str(hint.render())

    @pytest.mark.asyncio
    async def test_hint_shown_for_non_chat_screen(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.set_model(
                HeaderModel(
                    breadcrumb=("Library",),
                    active_screen="LibraryScreen",
                ),
            )
            await pilot.pause()
            hint = app.header.query_one("#header-back-hint", Static)
            assert hint.display is True
            assert "Esc to go back" in str(hint.render())

    @pytest.mark.asyncio
    async def test_hint_hidden_on_chat_screen(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.set_model(HeaderModel(active_screen="ChatScreen"))
            await pilot.pause()
            assert (
                app.header.query_one("#header-back-hint", Static).display
                is False
            )

    @pytest.mark.asyncio
    async def test_hint_toggles_across_transitions(self):
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            hint = app.header.query_one("#header-back-hint", Static)
            # Chat → hidden.
            app.header.set_model(HeaderModel(active_screen="ChatScreen"))
            await pilot.pause()
            assert hint.display is False
            # Push Settings → shown.
            app.header.set_model(HeaderModel(active_screen="SettingsScreen"))
            await pilot.pause()
            assert hint.display is True
            # Back to Chat → hidden again.
            app.header.set_model(HeaderModel(active_screen="ChatScreen"))
            await pilot.pause()
            assert hint.display is False

    @pytest.mark.asyncio
    async def test_hint_shown_via_refresh_from_app(self):
        # The real call path screens use.
        app = _HeaderHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.header.refresh_from_app(
                active_screen="InspectionScreen",
                breadcrumb=("Library", "Weather"),
                version="0.5.0",
            )
            await pilot.pause()
            assert (
                app.header.query_one("#header-back-hint", Static).display
                is True
            )
