"""DirectoryPickerModal — browse the filesystem and pick a folder.

A small reusable modal wrapping a :class:`FilteredDirectoryTree` so the
user can navigate directories (drill in by clicking a folder, ``Up`` to
re-root at the parent) and confirm a target directory. Dismisses with the
chosen :class:`pathlib.Path`, or ``None`` on cancel.

Used by :class:`care.screens.export_chain.ExportChainModal`'s *Browse…*
button to choose the export destination folder.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Label, Static

from care.runtime.i18n import t
from care.widgets.task_setup import FilteredDirectoryTree


class DirectoryPickerModal(ModalScreen["Path | None"]):
    """Folder browser. Construct with a ``start`` directory; dismiss with
    the selected :class:`Path` (Select) or ``None`` (Cancel / Esc)."""

    DEFAULT_CSS = """
    DirectoryPickerModal {
        align: center middle;
    }
    DirectoryPickerModal #dirpick-box {
        width: 80;
        max-width: 90%;
        height: 30;
        max-height: 90%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    DirectoryPickerModal #dirpick-title {
        text-style: bold;
        padding-bottom: 1;
    }
    DirectoryPickerModal #dirpick-current {
        color: $accent;
        padding-bottom: 1;
    }
    DirectoryPickerModal #dirpick-tree {
        height: 1fr;
        border: round $primary 40%;
    }
    DirectoryPickerModal #dirpick-buttons {
        height: auto;
        align-horizontal: right;
        padding-top: 1;
    }
    DirectoryPickerModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, *, start: Path | str | None = None) -> None:
        super().__init__()
        start_path = Path(str(start)).expanduser() if start else Path.cwd()
        if not start_path.is_dir():
            parent = start_path.parent
            start_path = parent if parent.is_dir() else Path.cwd()
        self._start = start_path
        self._selected = start_path

    def compose(self) -> ComposeResult:
        with Vertical(id="dirpick-box"):
            yield Label(t("dirPicker.title"), id="dirpick-title")
            yield Static(str(self._selected), id="dirpick-current")
            yield FilteredDirectoryTree(str(self._start), id="dirpick-tree")
            with Horizontal(id="dirpick-buttons"):
                yield Button(t("dirPicker.up"), id="dirpick-btn-up")
                yield Button(t("common.cancel"), id="dirpick-btn-cancel")
                yield Button(
                    t("dirPicker.select"),
                    id="dirpick-btn-select", variant="primary",
                )

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected,
    ) -> None:
        self._selected = Path(event.path)
        self._refresh_current()

    def _refresh_current(self) -> None:
        try:
            self.query_one("#dirpick-current", Static).update(str(self._selected))
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "dirpick-btn-cancel":
            self.dismiss(None)
        elif bid == "dirpick-btn-select":
            self.dismiss(self._selected)
        elif bid == "dirpick-btn-up":
            self._go_up()

    def _go_up(self) -> None:
        """Re-root the tree at the parent of its current root so the user
        can navigate above the starting directory."""
        try:
            tree = self.query_one("#dirpick-tree", FilteredDirectoryTree)
        except Exception:
            return
        current_root = Path(str(tree.path)).expanduser()
        parent = current_root.parent
        if parent == current_root:
            return  # already at filesystem root
        tree.path = str(parent)
        try:
            tree.reload()
        except Exception:
            pass
        self._selected = parent
        self._refresh_current()

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["DirectoryPickerModal"]
