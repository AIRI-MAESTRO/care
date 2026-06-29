"""EmptyStateView widget (TODO §1.1 P0.9).

Centred empty-state panel the LibraryScreen (and future
QueryHistory / RunHistory / DiffModal / etc.) shows when its
content area has no rows to render. Consumes the shipped
:class:`care.runtime.EmptyState` projection — `classify_empty_state(...)`
picks the right template; this widget renders it.

Render shape (vertical centre, with each row a `Static`):

    Title (bold)
    Message
    [ Primary action button ]
    Hint (dim)
    Error detail (dim, error templates only)

Posts :class:`EmptyStateView.ActionFired(action_kind)` when the
user clicks the CTA — the host screen dispatches off the
:class:`EmptyStateAction` literal (`create_first_agent` /
`clear_filters` / `retry` / `noop`).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from care.runtime.empty_state import EmptyState, EmptyStateAction


class EmptyStateView(Vertical):
    """Frozen-state rendering of a :class:`EmptyState`.

    Construct with optional initial state; tests + host code
    swap states via :meth:`set_state`. Rendering rebuilds via
    `recompose()` because the visible widgets vary by state
    kind (loading has no CTA, error has an extra detail row).
    """

    DEFAULT_CSS = """
    EmptyStateView {
        align-horizontal: center;
        align-vertical: middle;
        background: $background;
        padding: 1;
    }
    EmptyStateView #empty-state-title {
        text-style: bold;
        text-align: center;
        content-align: center middle;
        width: 100%;
        margin-bottom: 1;
    }
    EmptyStateView #empty-state-message {
        text-align: center;
        content-align: center middle;
        width: 100%;
        margin-bottom: 1;
    }
    EmptyStateView #empty-state-hint {
        text-style: dim italic;
        text-align: center;
        content-align: center middle;
        width: 100%;
        margin-top: 1;
    }
    EmptyStateView #empty-state-error-detail {
        text-style: dim;
        text-align: center;
        content-align: center middle;
        width: 100%;
        margin-top: 1;
        color: $error;
    }
    EmptyStateView #empty-state-cta-row {
        align-horizontal: center;
        width: 100%;
        height: auto;
        margin-top: 1;
    }
    EmptyStateView Button {
        width: auto;
    }
    """

    class ActionFired(Message):
        """Posted when the user clicks the CTA button."""

        def __init__(self, action_kind: EmptyStateAction) -> None:
            super().__init__()
            self.action_kind = action_kind

    def __init__(self, state: EmptyState | None = None) -> None:
        super().__init__()
        self._state: EmptyState | None = state

    @property
    def state(self) -> EmptyState | None:
        """Read-only snapshot — tests + telemetry."""
        return self._state

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        state = self._state
        if state is None:
            # Empty content — the widget's `display=False` flip
            # on the host side hides it from view; render
            # nothing in case the host forgets.
            return

        yield Static(state.title, id="empty-state-title")
        if state.message:
            yield Static(state.message, id="empty-state-message")
        if state.has_action or state.has_secondary_action:
            with Horizontal(id="empty-state-cta-row"):
                if state.has_action:
                    yield Button(
                        state.primary_action_label,
                        id="empty-state-cta",
                        variant="primary",
                    )
                if state.has_secondary_action:
                    yield Button(
                        state.secondary_action_label,
                        id="empty-state-cta-secondary",
                        variant="default",
                    )
        if state.hint:
            yield Static(state.hint, id="empty-state-hint")
        if state.error_detail:
            yield Static(
                state.error_detail,
                id="empty-state-error-detail",
            )

    # ------------------------------------------------------------------
    # State swap
    # ------------------------------------------------------------------

    def set_state(self, state: EmptyState | None) -> None:
        """Replace the current state + rebuild via
        :meth:`recompose`. Like the footer widget, `recompose`
        is async so we schedule it on the app loop —
        callers stay synchronous."""
        self._state = state
        if not self.is_mounted:
            return
        self.app.call_later(self.recompose)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._state is None:
            return
        if (
            event.button.id == "empty-state-cta"
            and self._state.has_action
        ):
            self.post_message(
                self.ActionFired(self._state.primary_action_kind),
            )
        elif (
            event.button.id == "empty-state-cta-secondary"
            and self._state.has_secondary_action
        ):
            self.post_message(
                self.ActionFired(self._state.secondary_action_kind),
            )


__all__ = ["EmptyStateView"]
