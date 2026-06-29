"""Tests for the header/footer + global key bindings data layer
(TODO §1.1 P0).

The Textual header/footer widgets + key handler are gated on
§1 P0; this suite pins the registry + projection contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from care.runtime.global_bindings import (
    FooterModel,
    GlobalBinding,
    GlobalBindingError,
    HeaderModel,
    bindings_for_scope,
    build_footer,
    build_header,
    default_global_bindings,
    find_binding_by_action,
    find_binding_by_key,
    validate_registry,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_default_action_ids_match_spec(self):
        ids = {b.action_id for b in default_global_bindings()}
        expected = {
            "open_command_palette",
            "quit",
            "save_artifact",
            "rerun_artifact",
            "back",
        }
        assert ids == expected

    def test_canonical_keys_match_spec(self):
        registry = {b.action_id: b for b in default_global_bindings()}
        assert registry["open_command_palette"].key == "Ctrl+P"
        assert registry["quit"].key == "Ctrl+Q"
        assert registry["save_artifact"].key == "Ctrl+S"
        assert registry["rerun_artifact"].key == "Ctrl+R"
        assert registry["back"].key == "Esc"

    def test_binding_is_frozen(self):
        binding = default_global_bindings()[0]
        with pytest.raises(FrozenInstanceError):
            binding.key = "x"  # type: ignore[misc]

    def test_no_duplicate_keys_in_defaults(self):
        # Validator should pass on the canonical registry.
        validate_registry(default_global_bindings())

    def test_validate_registry_detects_duplicates(self):
        bad = (
            GlobalBinding(
                action_id="quit", key="Ctrl+P", label="Quit",
            ),
            GlobalBinding(
                action_id="open_command_palette",
                key="Ctrl+P",
                label="Palette",
            ),
        )
        with pytest.raises(GlobalBindingError, match="duplicate"):
            validate_registry(bad)


# ---------------------------------------------------------------------------
# Textual key normalisation
# ---------------------------------------------------------------------------


class TestTextualKey:
    def test_ctrl_p_normalised(self):
        b = GlobalBinding(action_id="open_command_palette", key="Ctrl+P", label="x")
        assert b.textual_key == "ctrl+p"

    def test_esc_normalised(self):
        b = GlobalBinding(action_id="back", key="Esc", label="x")
        # Textual uses `escape` (full word) for the Esc key.
        assert b.textual_key == "escape"

    def test_whitespace_stripped(self):
        b = GlobalBinding(action_id="quit", key="Ctrl + Q", label="x")
        assert b.textual_key == "ctrl+q"


# ---------------------------------------------------------------------------
# Scope semantics
# ---------------------------------------------------------------------------


class TestScope:
    def test_always_applies_to_all(self):
        b = GlobalBinding(action_id="quit", key="X", label="x", scope="always")
        assert b.applies_to("screen") is True
        assert b.applies_to("modal") is True

    def test_screen_only(self):
        b = GlobalBinding(
            action_id="save_artifact", key="X", label="x", scope="screen",
        )
        assert b.applies_to("screen") is True
        assert b.applies_to("modal") is False

    def test_modal_only(self):
        b = GlobalBinding(action_id="back", key="X", label="x", scope="modal")
        assert b.applies_to("modal") is True
        assert b.applies_to("screen") is False


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


class TestFindByKey:
    def test_match_canonical_key(self):
        binding = find_binding_by_key("Ctrl+P")
        assert binding is not None
        assert binding.action_id == "open_command_palette"

    def test_case_insensitive(self):
        assert find_binding_by_key("ctrl+p").action_id == "open_command_palette"
        assert find_binding_by_key("CTRL+P").action_id == "open_command_palette"

    def test_whitespace_tolerant(self):
        assert find_binding_by_key("ctrl + p").action_id == "open_command_palette"

    def test_esc_matches_escape(self):
        # Textual emits "escape"; canonical key is "Esc".
        assert find_binding_by_key("escape").action_id == "back"

    def test_unknown_key_returns_none(self):
        assert find_binding_by_key("Q") is None
        assert find_binding_by_key("") is None

    def test_scope_filter(self):
        # `save_artifact` is screen-scoped → won't match in modal scope.
        assert find_binding_by_key("Ctrl+S", scope="modal") is None
        assert (
            find_binding_by_key("Ctrl+S", scope="screen").action_id
            == "save_artifact"
        )

    def test_always_scoped_matches_modal(self):
        # `open_command_palette` is always-scoped → matches in modals too.
        assert (
            find_binding_by_key("Ctrl+P", scope="modal").action_id
            == "open_command_palette"
        )


class TestFindByAction:
    def test_known_action(self):
        binding = find_binding_by_action("save_artifact")
        assert binding is not None
        assert binding.key == "Ctrl+S"

    def test_unknown_action(self):
        # Passing through an unknown literal — type system would
        # reject this in production, but the lookup is forgiving.
        assert find_binding_by_action("not-a-real-action") is None  # type: ignore[arg-type]


class TestBindingsForScope:
    def test_screen_excludes_modal_only(self):
        scoped = bindings_for_scope("screen")
        # All default bindings are either `always` or `screen`,
        # so none are excluded.
        assert len(scoped) == len(default_global_bindings())

    def test_modal_excludes_screen_only(self):
        scoped = bindings_for_scope("modal")
        # `save_artifact` + `rerun_artifact` are screen-scoped.
        action_ids = {b.action_id for b in scoped}
        assert "save_artifact" not in action_ids
        assert "rerun_artifact" not in action_ids
        # `back`, `open_command_palette`, `quit` are always-scoped.
        assert "back" in action_ids
        assert "open_command_palette" in action_ids
        assert "quit" in action_ids

    def test_custom_registry_respected(self):
        custom = (
            GlobalBinding(action_id="quit", key="X", label="x"),
        )
        scoped = bindings_for_scope("screen", registry=custom)
        assert len(scoped) == 1


# ---------------------------------------------------------------------------
# Header / footer projection
# ---------------------------------------------------------------------------


class TestHeader:
    def test_build_header_defaults(self):
        header = build_header()
        assert header.title == "MAESTRO"
        assert header.breadcrumb == ()
        assert header.version == ""
        assert header.active_screen == ""

    def test_build_header_full(self):
        header = build_header(
            active_screen="InspectionScreen",
            breadcrumb=["Library", "Weather report"],
            version="0.3.0",
            title="CARE",
        )
        assert header.active_screen == "InspectionScreen"
        assert header.breadcrumb == ("Library", "Weather report")
        assert header.version == "0.3.0"
        assert header.breadcrumb_text == "Library › Weather report"

    def test_breadcrumb_filters_empty(self):
        header = build_header(breadcrumb=["Library", "", "v3"])
        assert header.breadcrumb == ("Library", "v3")

    def test_header_is_frozen(self):
        header = HeaderModel()
        with pytest.raises(FrozenInstanceError):
            header.title = "x"  # type: ignore[misc]


class TestFooter:
    def test_screen_scope_has_all_actions(self):
        footer = build_footer(active_screen="LibraryScreen", scope="screen")
        action_ids = {h.action_id for h in footer.hints}
        assert action_ids == {
            "open_command_palette", "save_artifact",
            "rerun_artifact", "back", "quit",
        }

    def test_modal_scope_drops_save_and_rerun(self):
        footer = build_footer(active_screen="ModalX", scope="modal")
        action_ids = {h.action_id for h in footer.hints}
        assert "save_artifact" not in action_ids
        assert "rerun_artifact" not in action_ids

    def test_active_screen_stamped(self):
        footer = build_footer(active_screen="ExecutionScreen")
        assert footer.active_screen == "ExecutionScreen"

    def test_iter_and_len(self):
        footer = build_footer()
        assert len(footer) == len(list(footer))

    def test_footer_is_frozen(self):
        footer = FooterModel()
        with pytest.raises(FrozenInstanceError):
            footer.active_screen = "x"  # type: ignore[misc]

    def test_hints_preserve_canonical_keys(self):
        # The footer renders the human-readable canonical key
        # ("Ctrl+P"), not Textual's lowercased form.
        footer = build_footer()
        keys = {h.key for h in footer.hints}
        assert "Ctrl+P" in keys
        assert "Esc" in keys

    def test_custom_registry(self):
        custom = (
            GlobalBinding(
                action_id="quit", key="Ctrl+X", label="Bye",
            ),
        )
        footer = build_footer(registry=custom)
        assert len(footer) == 1
        assert footer.hints[0].key == "Ctrl+X"
        assert footer.hints[0].label == "Bye"


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            FooterHint as FH,
            FooterModel as FM,
            GlobalBinding as B,
            GlobalBindingError as Err,
            HeaderModel as H,
            bindings_for_scope as bfs,
            build_footer as bf,
            build_header as bh,
            default_global_bindings as defaults,
            find_binding_by_action as fba,
            find_binding_by_key as fbk,
            validate_registry as validate,
        )

        assert B is GlobalBinding
        assert Err is GlobalBindingError
        assert H is HeaderModel
        assert FM is FooterModel
        assert defaults is default_global_bindings
        assert fbk is find_binding_by_key
        assert fba is find_binding_by_action
        assert bfs is bindings_for_scope
        assert bf is build_footer
        assert bh is build_header
        assert validate is validate_registry
        assert FH is not None
