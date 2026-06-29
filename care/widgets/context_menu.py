"""ContextMenu modal (TODO §1.1 P0.12).

A :class:`ModalScreen` that pops up over the LibraryScreen to
render the per-row actions returned by
:func:`care.runtime.row_actions.actions_for_row` for the focused
row. The user picks an entry; the modal dismisses with the
selected :class:`care.runtime.row_actions.RowActionKind` (or
``None`` on Escape / cancel) so the caller can route the choice
through the same `_dispatch_row_action` path the key bindings
use.

Pure presentation — destination wiring stays on the screen
that pushes the modal.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from care.runtime.row_actions import RowAction, RowActionKind


class ContextMenu(ModalScreen[RowActionKind | None]):
    """Right-click / `Menu`-key affordance for the focused row.

    Args:
        actions: Tuple of :class:`RowAction` to render. The
            screen passes ``actions_for_row(focused)`` so
            status-gated kinds (`evolve` / `show_lineage` on
            draft rows) are pre-filtered out.
        anchor: Optional ``(x, y)`` screen coordinates so the
            modal can later position itself near the click;
            current implementation centers the menu (Textual
            doesn't expose pixel anchoring on overlay screens
            without a custom layout — pinned for a future
            refinement).
    """

    DEFAULT_CSS = """
    ContextMenu {
        align: center middle;
        background: $background 40%;
    }
    ContextMenu #context-menu-box {
        width: 32;
        max-width: 80%;
        padding: 0 1;
        border: thick $primary;
        background: $surface;
    }
    ContextMenu OptionList {
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        *,
        actions: tuple[RowAction, ...],
        anchor: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self._actions = actions
        self._anchor = anchor

    def compose(self) -> ComposeResult:
        with Vertical(id="context-menu-box"):
            options = [
                Option(action.label, id=action.kind)
                for action in self._actions
            ]
            yield OptionList(*options, id="context-menu-list")

    def on_mount(self) -> None:
        # Auto-focus the option list so arrow keys work
        # immediately.
        try:
            self.query_one(OptionList).focus()
        except Exception:
            pass

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected,
    ) -> None:
        kind = event.option.id
        # `id` is always a `RowActionKind` because we set it
        # at compose time; cast for the dismiss return.
        self.dismiss(kind)  # type: ignore[arg-type]

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["ContextMenu"]
