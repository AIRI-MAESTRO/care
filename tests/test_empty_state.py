"""Tests for the LibraryScreen empty-state data layer (TODO §1.3 P0).

The empty-state widget is gated on §1 P0; this suite pins the
classifier contract the widget will bind to.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from care.runtime.empty_state import (
    EMPTY_STATE_TEMPLATES,
    EmptyState,
    classify_empty_state,
)
from care.runtime.library_view import LibraryFilters, LibraryRow, LibraryView


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _view(*, rows=None, filters=None) -> LibraryView:
    return LibraryView(
        rows=tuple(rows or ()),
        filters=filters or LibraryFilters(),
        total_returned=len(rows or ()),
    )


def _row(entity_id: str = "ent-1") -> LibraryRow:
    return LibraryRow(entity_id=entity_id, display_name="row")


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_returns_none_when_view_has_rows(self):
        view = _view(rows=[_row()])
        assert classify_empty_state(view) is None

    def test_no_library_when_zero_rows_and_no_filters(self):
        view = _view()
        state = classify_empty_state(view)
        assert state is not None
        assert state.kind == "no_library"
        assert state.primary_action_kind == "create_first_agent"
        assert state.primary_action_label  # non-empty

    def test_no_results_when_filters_active(self):
        view = _view(filters=LibraryFilters(search="storm"))
        state = classify_empty_state(view, filters=LibraryFilters(search="storm"))
        assert state is not None
        assert state.kind == "no_results"
        assert state.primary_action_kind == "clear_filters"

    def test_no_results_via_view_filters(self):
        # When caller doesn't pass filters=, classifier falls
        # back to the view's own filter state.
        view = _view(filters=LibraryFilters(domain="weather"))
        state = classify_empty_state(view)
        assert state is not None
        assert state.kind == "no_results"

    def test_loading_state_on_first_paint(self):
        state = classify_empty_state(None, is_loading=True)
        assert state is not None
        assert state.kind == "loading"
        assert state.primary_action_kind == "noop"

    def test_loading_does_not_blank_existing_view(self):
        # A refresh in flight WITH an existing view → screen
        # keeps showing the existing rows.
        view = _view(rows=[_row()])
        assert classify_empty_state(view, is_loading=True) is None

    def test_loading_ignored_when_view_present_and_empty(self):
        # Loading=True + view=empty → no_library wins because
        # the screen already painted once (the view exists,
        # just empty).
        view = _view()
        state = classify_empty_state(view, is_loading=True)
        assert state is not None
        assert state.kind == "no_library"

    def test_error_state_wins_over_loading(self):
        state = classify_empty_state(
            None, is_loading=True, error="connection refused"
        )
        assert state is not None
        assert state.kind == "error"
        assert state.primary_action_kind == "retry"
        assert "connection refused" in state.error_detail

    def test_error_state_wins_over_rows(self):
        # If error is set AND view has rows, error still wins
        # — fetch failed mid-refresh.
        view = _view(rows=[_row()])
        state = classify_empty_state(view, error="503")
        assert state is not None
        assert state.kind == "error"
        assert state.error_detail == "503"

    def test_error_empty_string_does_not_trigger_error_state(self):
        # Empty string error → no error state.
        view = _view()
        state = classify_empty_state(view, error="")
        assert state.kind == "no_library"

    def test_none_view_no_loading_falls_to_no_library(self):
        # Caller misuse (view=None, is_loading=False) — return
        # the zero-data state rather than crashing.
        state = classify_empty_state(None)
        assert state is not None
        assert state.kind == "no_library"

    def test_filters_param_precedence_over_view_filters(self):
        # Caller-supplied filters trump view's own filter state.
        view = _view(filters=LibraryFilters())  # no filters on view
        state = classify_empty_state(view, filters=LibraryFilters(search="x"))
        assert state is not None
        assert state.kind == "no_results"

    def test_view_filters_used_when_caller_param_missing(self):
        view = _view(filters=LibraryFilters(favourites_only=True))
        state = classify_empty_state(view)
        assert state is not None
        assert state.kind == "no_results"

    def test_caller_passed_empty_filters_uses_view_filters(self):
        # Passing an empty `LibraryFilters()` is_filtering=False;
        # the classifier falls through to the view's own filter
        # state (which may be filtering).
        view = _view(filters=LibraryFilters(search="x"))
        state = classify_empty_state(view, filters=LibraryFilters())
        # Caller's empty filters NOT filtering, but view IS.
        assert state.kind == "no_results"


# ---------------------------------------------------------------------------
# EmptyState shape
# ---------------------------------------------------------------------------


class TestEmptyStateShape:
    def test_frozen(self):
        state = EmptyState(kind="loading", title="x")
        with pytest.raises(FrozenInstanceError):
            state.title = "y"  # type: ignore[misc]

    def test_has_action_predicate(self):
        loading = EmptyState(kind="loading", title="x", primary_action_kind="noop")
        assert loading.has_action is False
        action = EmptyState(
            kind="no_library", title="x",
            primary_action_kind="create_first_agent",
        )
        assert action.has_action is True

    def test_has_secondary_action_predicate(self):
        bare = EmptyState(
            kind="no_library", title="x",
            primary_action_kind="create_first_agent",
        )
        assert bare.has_secondary_action is False
        with_secondary = EmptyState(
            kind="no_library", title="x",
            primary_action_kind="create_first_agent",
            secondary_action_kind="back_to_chat",
            secondary_action_label="Back",
        )
        assert with_secondary.has_secondary_action is True

    def test_templates_tuple(self):
        # EMPTY_STATE_TEMPLATES exposes all four canonical states.
        kinds = {t.kind for t in EMPTY_STATE_TEMPLATES}
        assert kinds == {"no_library", "no_results", "loading", "error"}

    def test_templates_have_non_empty_labels(self):
        for template in EMPTY_STATE_TEMPLATES:
            if template.has_action:
                assert template.primary_action_label
                assert template.title


# ---------------------------------------------------------------------------
# Message templates pin-tests
# ---------------------------------------------------------------------------


class TestMessageTemplates:
    def test_no_library_message(self):
        state = classify_empty_state(_view())
        # Pin user-facing copy so the screen + tests stay in lockstep.
        assert "library is empty" in state.title.lower()
        assert "Create your first chain" == state.primary_action_label

    def test_no_library_advertises_artifacts(self):
        """TODO §4 P0 — the no-library card surfaces /artifacts
        so users with an unsaved in-session chain learn the
        save path without needing /help. Also confirms the
        secondary back-to-chat CTA so the card has an explicit
        exit affordance."""
        state = classify_empty_state(_view())
        assert "/artifacts" in state.message
        assert state.has_secondary_action
        assert state.secondary_action_kind == "back_to_chat"
        assert state.secondary_action_label == "Back to chat"

    def test_no_results_message(self):
        state = classify_empty_state(
            _view(filters=LibraryFilters(search="x"))
        )
        assert "filter" in state.title.lower()
        assert "Clear filters" == state.primary_action_label

    def test_loading_message(self):
        state = classify_empty_state(None, is_loading=True)
        assert "Loading" in state.title
        # Loading state has no CTA.
        assert state.primary_action_kind == "noop"
        assert state.primary_action_label == ""

    def test_error_message_includes_detail(self):
        state = classify_empty_state(_view(), error="HTTP 503 from Memory")
        assert state.error_detail == "HTTP 503 from Memory"
        # The template message stays stable; the detail is
        # surfaced separately for diagnostic display.
        assert "Couldn't load" in state.title


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            EmptyState as S,
            EmptyStateKind,
            EmptyStateAction,
            classify_empty_state as classify,
            EMPTY_STATE_TEMPLATES as templates,
        )

        assert S is EmptyState
        assert classify is classify_empty_state
        # Literal types pass through.
        assert EmptyStateKind is not None
        assert EmptyStateAction is not None
        # Templates re-export.
        assert templates is EMPTY_STATE_TEMPLATES
