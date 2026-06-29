"""FilePickerModal — browse the filesystem and pick a file.

A small reusable modal wrapping a :class:`FilteredDirectoryTree` so the
user can navigate directories (drill in by clicking a folder, ``Up`` to
re-root at the parent) and click a file to select it. Dismisses with the
chosen :class:`pathlib.Path`, or ``None`` on cancel.

The sibling of :class:`care.screens.directory_picker.DirectoryPickerModal`
(which picks a folder). Used by the Evolution launch modal's *Browse…*
button to choose the evaluation dataset (a ``.jsonl`` file).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DirectoryTree, Label, Static

from care.runtime.i18n import t
from care.screens._animated_modal import AnimatedModalScreen
from care.widgets.task_setup import FilteredDirectoryTree


class _ExtFilteredDirectoryTree(FilteredDirectoryTree):
    """A `FilteredDirectoryTree` that also hides files whose suffix isn't in
    ``extensions``. Directories always stay visible so the user can keep
    navigating; an empty ``extensions`` shows every file."""

    def __init__(
        self, path: str, *, extensions: tuple[str, ...] = (), **kwargs,
    ) -> None:
        super().__init__(path, **kwargs)
        self._extensions = tuple(e.lower() for e in extensions)

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        base = list(super().filter_paths(paths))
        if not self._extensions:
            return base
        return [
            p for p in base
            if p.is_dir() or p.suffix.lower() in self._extensions
        ]


class FilePickerModal(AnimatedModalScreen["Path | None"]):
    """File browser. Construct with a ``start`` directory and an optional
    ``extensions`` filter (e.g. ``(".jsonl",)``); dismiss with the selected
    file :class:`Path` (Select) or ``None`` (Cancel / Esc)."""

    DEFAULT_CSS = """
    FilePickerModal {
        align: center middle;
    }
    FilePickerModal #filepick-box {
        width: 80;
        max-width: 90%;
        height: 30;
        max-height: 90%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    FilePickerModal #filepick-title {
        text-style: bold;
        padding-bottom: 1;
    }
    FilePickerModal #filepick-current {
        color: $accent;
        padding-bottom: 1;
    }
    FilePickerModal #filepick-tree {
        height: 1fr;
        border: round $primary 40%;
    }
    FilePickerModal #filepick-buttons {
        height: auto;
        align-horizontal: right;
        padding-top: 1;
    }
    FilePickerModal Button {
        margin-left: 1;
    }
    """

    ANIM_BOX_ID = "filepick-box"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        *,
        start: Path | str | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        start_path = Path(str(start)).expanduser() if start else Path.cwd()
        # If `start` points at a file, root the tree at its parent so the
        # file itself is visible + pre-selected.
        self._selected: Path | None = None
        if start_path.is_file():
            self._selected = start_path
            start_path = start_path.parent
        if not start_path.is_dir():
            parent = start_path.parent
            start_path = parent if parent.is_dir() else Path.cwd()
        self._start = start_path
        self._extensions = tuple(extensions or ())

    def compose(self) -> ComposeResult:
        with Vertical(id="filepick-box"):
            yield Label(t("filePicker.title"), id="filepick-title")
            yield Static(
                self._current_text(), id="filepick-current", markup=False,
            )
            yield _ExtFilteredDirectoryTree(
                str(self._start),
                extensions=self._extensions,
                id="filepick-tree",
            )
            with Horizontal(id="filepick-buttons"):
                yield Button(t("dirPicker.up"), id="filepick-btn-up")
                yield Button(t("common.cancel"), id="filepick-btn-cancel")
                yield Button(
                    t("filePicker.select"),
                    id="filepick-btn-select", variant="primary",
                )

    def _current_text(self) -> str:
        if self._selected is None:
            return t("filePicker.noneSelected")
        return str(self._selected)

    def on_mount(self) -> None:
        self._animate_modal_in()

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected,
    ) -> None:
        self._selected = Path(event.path)
        self._refresh_current()

    def _refresh_current(self) -> None:
        try:
            self.query_one("#filepick-current", Static).update(
                self._current_text(),
            )
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "filepick-btn-cancel":
            self.dismiss(None)
        elif bid == "filepick-btn-select":
            # Only confirm when a file is actually picked — otherwise keep
            # the modal open so the user can click one.
            if self._selected is not None and self._selected.is_file():
                self.dismiss(self._selected)
        elif bid == "filepick-btn-up":
            self._go_up()

    def _go_up(self) -> None:
        """Re-root the tree at the parent of its current root so the user
        can navigate above the starting directory."""
        try:
            tree = self.query_one("#filepick-tree", _ExtFilteredDirectoryTree)
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

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["FilePickerModal"]
