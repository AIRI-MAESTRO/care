"""ConflictModal — resolve a save-time naming conflict
(TODO §3 P1 [DONE — data layer] → fully DONE).

Pushed on top of `SaveAgentModal` (or any screen that drives a
``save_*`` flow) when :func:`care.detect_conflict` reports an
existing entity with the same display name but a different SHA.
The modal renders the unified-diff lines pre-computed by the data
layer and presents the three resolutions the user can pick:

* **Keep existing** — abort the save; reuse the existing
  ``entity_id``.
* **New version** *(default)* — write the incoming content as a
  fresh version of the same ``entity_id``, preserving history.
* **Accept incoming** — overwrite the existing entity in place.

The modal is purely presentational; the host calls
:func:`care.apply_resolution` with the dismissed
:class:`ConflictResolution`. This keeps the modal free of memory
side-effects so it stays trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from care.conflict import ConflictReport, ConflictResolution
from care.runtime.i18n import t


@dataclass(frozen=True)
class ConflictModalResult:
    """Dismiss envelope for :class:`ConflictModal`.

    Fields:
        resolution: The resolution the user picked, or ``None``
            when they dismissed via Escape without choosing.
        report: The :class:`ConflictReport` the modal rendered.
            Returned verbatim so the host can pass it straight to
            :func:`care.apply_resolution` without re-detecting.
        cancelled: ``True`` when the user hit Escape / Cancel; the
            host should treat this as a "back out of save" gesture
            (no Memory mutation).
    """

    resolution: ConflictResolution | None = None
    report: ConflictReport | None = None
    cancelled: bool = False


class ConflictModal(ModalScreen[ConflictModalResult]):
    """Three-button conflict-resolution modal.

    Construct with a :class:`ConflictReport` from
    :func:`care.detect_conflict`. On mount the diff lines render
    into a scrollable pane and three buttons land on the bottom
    row. Pressing one dismisses with the matching
    :class:`ConflictResolution`. Escape cancels without choosing.
    """

    DEFAULT_CSS = """
    ConflictModal {
        align: center middle;
    }
    ConflictModal #conflict-box {
        width: 100;
        max-width: 95%;
        height: 30;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }
    ConflictModal #conflict-title {
        text-style: bold;
        color: $warning;
        padding-bottom: 1;
    }
    ConflictModal #conflict-summary {
        height: auto;
        margin-bottom: 1;
        color: $text-muted;
    }
    ConflictModal #conflict-diff {
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
    }
    ConflictModal #conflict-actions {
        height: 3;
        align-horizontal: right;
    }
    ConflictModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, report: ConflictReport) -> None:
        super().__init__()
        self.report = report

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="conflict-box"):
            yield Static(
                t("conflict.title", name=self.report.name or t("conflict.unnamed")),
                id="conflict-title",
            )
            yield Static(self._summary_line(), id="conflict-summary")
            with VerticalScroll(id="conflict-diff"):
                yield Static(self._diff_text(), id="conflict-diff-body")
            with Horizontal(id="conflict-actions"):
                yield Button(t("common.cancel"), id="conflict-btn-cancel")
                yield Button(
                    t("conflict.keepExisting"),
                    id="conflict-btn-keep-existing",
                )
                yield Button(
                    t("conflict.acceptIncoming"),
                    id="conflict-btn-accept-incoming",
                    variant="warning",
                )
                yield Button(
                    t("conflict.newVersion"),
                    id="conflict-btn-new-version",
                    variant="primary",
                )

    def on_mount(self) -> None:
        # Default focus on "New version" — the recommended choice
        # the data-layer docstring calls out.
        try:
            self.query_one("#conflict-btn-new-version", Button).focus()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _summary_line(self) -> str:
        kind = self.report.entity_type or t("conflict.entity")
        existing = self.report.existing_entity_id or "?"
        if not self.report.is_conflict:
            return t(
                "conflict.summaryIdentical", kind=kind, existing=repr(existing),
            )
        return t(
            "conflict.summaryDifferent", kind=kind, existing=repr(existing),
        )

    def _diff_text(self) -> str:
        if not self.report.is_conflict:
            return t("conflict.noDifferences")
        if not self.report.diff_lines:
            return t("conflict.diffUnavailable")
        return "\n".join(self.report.diff_lines)

    # ------------------------------------------------------------------
    # Dismiss
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "conflict-btn-cancel":
            self.action_cancel()
        elif bid == "conflict-btn-keep-existing":
            self._dismiss_with("keep_existing")
        elif bid == "conflict-btn-accept-incoming":
            self._dismiss_with("accept_incoming")
        elif bid == "conflict-btn-new-version":
            self._dismiss_with("new_version")

    def action_cancel(self) -> None:
        self.dismiss(
            ConflictModalResult(
                resolution=None, report=self.report, cancelled=True,
            ),
        )

    def _dismiss_with(self, resolution: ConflictResolution) -> None:
        self.dismiss(
            ConflictModalResult(
                resolution=resolution,
                report=self.report,
                cancelled=False,
            ),
        )


__all__ = ["ConflictModal", "ConflictModalResult"]
