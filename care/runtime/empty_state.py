"""LibraryScreen empty-state data layer (TODO §1.3 P0).

The LibraryScreen renders different empty-state messages
depending on *why* the table has no rows:

* **No library** — Memory has zero saved agents in the
  namespace at all. Show a centered "Create your first agent"
  CTA that opens the QueryScreen.
* **No results** — the namespace HAS agents but the active
  filters or search query return zero matches. Show a
  "Clear filters" affordance instead of the create CTA.
* **Loading** — a fetch is in flight; show a placeholder.
* **Error** — the fetch failed. Show the error + a retry
  affordance.

The Textual widget choosing between these states is gated on
TODO §1 P0 multi-screen workflow, but the classifier + the
canonical message templates ship now so the widget is a thin
renderer.

What this module provides:

* :class:`EmptyStateKind` literal pinning the four cases.
* :class:`EmptyStateAction` literal pinning the four primary-
  CTA kinds the screen wires up.
* :class:`EmptyState` — frozen rendering payload (kind, title,
  message, primary CTA descriptor, optional hint).
* :func:`classify_empty_state` — pure projection from a
  :class:`LibraryView` + optional `is_loading` / `error`
  signals into an :class:`EmptyState`. Returns ``None`` when
  the view has rows (screen renders the table normally).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from care.runtime.i18n import t
from care.runtime.library_view import LibraryFilters, LibraryView


EmptyStateKind = Literal[
    "no_library",
    "no_results",
    "loading",
    "error",
]
"""The four canonical empty-state classifications. ``no_library``
is the first-run zero-data state; ``no_results`` means the
namespace HAS rows but filters / search hide them all.
"""


EmptyStateAction = Literal[
    "create_first_agent",
    "clear_filters",
    "retry",
    "back_to_chat",
    "noop",
]
"""CTA kinds the screen knows how to wire up:

* ``create_first_agent`` — pushes QueryScreen.
* ``clear_filters`` — resets the active filter chips.
* ``retry`` — re-runs the failed fetch.
* ``back_to_chat`` — pops the screen stack back to the
  ChatScreen (or pushes one when none is mounted).
* ``noop`` — render a label only, no clickable affordance
  (used for the loading state).
"""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmptyState:
    """Frozen rendering payload for the empty-state widget.

    The widget reads ``title`` + ``message`` for its header,
    renders the CTA labelled ``primary_action_label`` wired to
    ``primary_action_kind``, and surfaces ``hint`` as
    grey-text below the CTA when present. Optional
    ``secondary_action_*`` fields drive a second button rendered
    next to the primary — used by the no-library template so
    users with in-session chains can jump back to chat and save
    them via ``/artifacts`` instead of generating a new one.

    ``error_detail`` is populated only for ``kind="error"`` —
    carries the raw error message for diagnostic display.
    """

    kind: EmptyStateKind
    title: str
    message: str = ""
    primary_action_kind: EmptyStateAction = "noop"
    primary_action_label: str = ""
    secondary_action_kind: EmptyStateAction = "noop"
    secondary_action_label: str = ""
    hint: str = ""
    error_detail: str = ""

    @property
    def has_action(self) -> bool:
        """``True`` when the screen should render a clickable
        primary CTA. ``noop`` kinds (loading) render label
        only."""
        return self.primary_action_kind != "noop"

    @property
    def has_secondary_action(self) -> bool:
        """``True`` when a second CTA button should render
        alongside the primary. Templates default to ``noop`` so
        most empty states render a single button — only the
        no-library card opts in to the back-to-chat secondary
        today."""
        return self.secondary_action_kind != "noop"


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


# Templates are built fresh on each call (not pinned as module
# constants) so the copy resolves in the UI language active at render
# time, not whatever language was loaded at import.


def _no_library() -> EmptyState:
    return EmptyState(
        kind="no_library",
        title=t("library.empty.noLibrary.title"),
        message=t("library.empty.noLibrary.message"),
        primary_action_kind="create_first_agent",
        primary_action_label=t("library.empty.noLibrary.primary"),
        secondary_action_kind="back_to_chat",
        secondary_action_label=t("library.empty.noLibrary.secondary"),
        hint=t("library.empty.noLibrary.hint"),
    )


def _no_results() -> EmptyState:
    return EmptyState(
        kind="no_results",
        title=t("library.empty.noResults.title"),
        message=t("library.empty.noResults.message"),
        primary_action_kind="clear_filters",
        primary_action_label=t("library.empty.noResults.primary"),
    )


def _loading() -> EmptyState:
    return EmptyState(
        kind="loading",
        title=t("library.empty.loading.title"),
        message=t("library.empty.loading.message"),
        primary_action_kind="noop",
    )


def _error_template() -> EmptyState:
    return EmptyState(
        kind="error",
        title=t("library.empty.error.title"),
        message=t("library.empty.error.message"),
        primary_action_kind="retry",
        primary_action_label=t("library.empty.error.primary"),
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify_empty_state(
    view: Optional[LibraryView],
    *,
    filters: Optional[LibraryFilters] = None,
    is_loading: bool = False,
    error: Optional[str] = None,
) -> Optional[EmptyState]:
    """Pick the empty-state message that fits the current
    LibraryScreen condition.

    Precedence (top to bottom):

    1. ``error`` (truthy) → :data:`EmptyStateKind.error` —
       fetches failed. Returned regardless of `view` /
       `is_loading` so the user sees the failure rather than a
       stale "loading" or "no results" placeholder.
    2. ``is_loading=True`` AND ``view is None`` →
       :data:`EmptyStateKind.loading` — first-paint fetch in
       flight (the screen hasn't received any data yet).
    3. ``view.is_empty`` AND filters are active →
       :data:`EmptyStateKind.no_results` — namespace has data
       but the user's filters hide it.
    4. ``view.is_empty`` AND no filters active →
       :data:`EmptyStateKind.no_library` — first-run zero-data
       state.
    5. Otherwise: ``None`` — the screen has rows; render the
       table as usual.

    Args:
        view: Current :class:`LibraryView` (or ``None`` before
            the first fetch completes).
        filters: Active filter state. ``None`` is equivalent to
            an empty :class:`LibraryFilters` (no filtering).
            Used to disambiguate "no_library" vs.
            "no_results".
        is_loading: ``True`` when a fetch is in flight. Only
            surfaces the loading state when ``view`` is also
            ``None`` — a subsequent refresh with an existing
            view shouldn't blank the table.
        error: Optional error string. Truthy values flip the
            state to error and populate
            :attr:`EmptyState.error_detail`.

    Returns:
        :class:`EmptyState` or ``None``.
    """
    if error:
        return _make_error_state(error)

    if is_loading and view is None:
        return _loading()

    if view is None:
        # No fetch has even started; treat as a degenerate
        # zero-data state (the wrapping screen typically only
        # reaches this branch via misuse — first-paint should
        # always set is_loading=True).
        return _no_library()

    if not view.is_empty:
        return None

    if filters is not None and filters.is_filtering:
        return _no_results()
    if view.filters.is_filtering:
        # Caller didn't supply `filters=` but the view itself
        # carries them — honour the view's filter state.
        return _no_results()

    return _no_library()


def _make_error_state(error: str) -> EmptyState:
    """Stamp the raw error onto the template's diagnostic
    field. Done via :func:`dataclasses.replace` so the template
    stays a frozen sentinel."""
    from dataclasses import replace

    return replace(_error_template(), error_detail=error)


# The canonical templates as a tuple so callers can iterate them
# (e.g. a "preview every empty state" affordance) and tests can pin
# the kinds without importing internals. Built once at import in the
# language active then — the live LibraryScreen always re-resolves via
# `classify_empty_state`, so this snapshot is only a preview artifact.
EMPTY_STATE_TEMPLATES: tuple[EmptyState, ...] = (
    _no_library(),
    _no_results(),
    _loading(),
    _error_template(),
)


# Re-export the unused field marker for future dataclass
# extensions.
_ = field


__all__ = [
    "EMPTY_STATE_TEMPLATES",
    "EmptyState",
    "EmptyStateAction",
    "EmptyStateKind",
    "classify_empty_state",
]
