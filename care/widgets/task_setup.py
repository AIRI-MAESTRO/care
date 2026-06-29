"""Task setup widget — task description editor + context file picker.

Organises the New-Agent form into a :class:`TabbedContent` so the
panel fits short terminals: only the task description + action
buttons live on the first tab; file browsing, the context-doc
table, and the optional generation hints sit on their own tabs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    DirectoryTree,
    Input,
    Label,
    RadioButton,
    RadioSet,
    TabbedContent,
    TabPane,
    TextArea,
)

from care.runtime.i18n import t


SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules"}


TargetRuntime = Literal["local", "docker", "e2b"]
MageMode = Literal["fast", "deep"]


class FilteredDirectoryTree(DirectoryTree):
    """DirectoryTree that hides common noise directories."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [p for p in paths if not (p.is_dir() and p.name in SKIP_DIRS)]


class TaskSetup(Widget):
    """+ New agent form, organised as four tabs so the screen
    fits short terminals."""

    DEFAULT_CSS = """
    TaskSetup {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
    }
    TaskSetup TabbedContent {
        height: 1fr;
    }
    TaskSetup TabPane {
        padding: 1 1;
    }
    TaskSetup .section-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    TaskSetup #task-input {
        height: 1fr;
        min-height: 5;
        margin-bottom: 1;
    }
    TaskSetup FilteredDirectoryTree {
        height: 1fr;
        min-height: 6;
        border: round $primary 30%;
        margin-bottom: 1;
    }
    TaskSetup .tree-nav {
        height: 3;
        margin-bottom: 1;
    }
    TaskSetup .tree-nav Button {
        margin-right: 1;
        min-width: 10;
    }
    TaskSetup #tree-path {
        color: $text-muted;
        margin-bottom: 1;
    }
    TaskSetup #context-table {
        height: 1fr;
        min-height: 5;
        border: round $primary 30%;
        margin-bottom: 1;
    }
    TaskSetup .actions {
        height: 3;
        align-horizontal: right;
    }
    TaskSetup .actions Button {
        margin-left: 1;
    }
    TaskSetup #tab-hints Input {
        margin-bottom: 1;
    }
    TaskSetup #tab-hints RadioSet {
        height: 5;
        margin-bottom: 1;
    }
    """

    class GenerateRequested(Message):
        def __init__(self, task: str, files: tuple[Path, ...]) -> None:
            super().__init__()
            self.task = task
            self.files = files

    @property
    def DEFAULT_TASK(self) -> str:
        """Default task text seeded into the editor. Resolved via
        :func:`t` at access time so a language change re-seeds the
        localized default on the next compose."""
        return t("taskSetup.defaultTask")

    def __init__(
        self,
        *,
        initial_runtime: TargetRuntime = "local",
        initial_mode: MageMode = "deep",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._files: list[Path] = []
        self._initial_runtime: TargetRuntime = initial_runtime
        self._initial_mode: MageMode = initial_mode

    def compose(self) -> ComposeResult:
        with TabbedContent(id="task-tabs"):
            with TabPane(t("taskSetup.tabTask"), id="tab-task"):
                yield Label(
                    t("taskSetup.taskPrompt"),
                    classes="section-hint",
                )
                yield TextArea(self.DEFAULT_TASK, id="task-input")
                with Horizontal(classes="actions"):
                    yield Button(t("common.clear"), id="btn-clear")
                    yield Button(
                        t("taskSetup.generate"),
                        id="btn-generate",
                        variant="primary",
                    )

            with TabPane(t("taskSetup.tabFiles"), id="tab-files"):
                yield Label(
                    t("taskSetup.filesHint"),
                    classes="section-hint",
                )
                with Horizontal(classes="tree-nav"):
                    yield Button(t("common.home"), id="btn-tree-home")
                    yield Button(t("common.up"), id="btn-tree-up")
                yield Label(
                    self._format_root_label(Path.cwd()), id="tree-path",
                )
                yield FilteredDirectoryTree(str(Path.cwd()), id="file-tree")

            with TabPane(t("taskSetup.tabContext"), id="tab-context"):
                yield Label(
                    t("taskSetup.contextHint"),
                    classes="section-hint",
                )
                yield DataTable(
                    id="context-table",
                    cursor_type="row",
                    zebra_stripes=True,
                )

            with TabPane(t("taskSetup.tabHints"), id="tab-hints"):
                with Vertical():
                    yield Label(
                        t("taskSetup.hintsHint"),
                        classes="section-hint",
                    )
                    yield Input(
                        placeholder=t("taskSetup.domainPlaceholder"),
                        id="query-domain-hint",
                    )
                    yield Input(
                        placeholder=t("taskSetup.maxStepsPlaceholder"),
                        id="query-max-steps",
                    )
                    yield Label(t("taskSetup.runtime"), classes="section-hint")
                    with RadioSet(id="query-runtime"):
                        yield RadioButton(
                            t("taskSetup.runtimeLocal"),
                            value=self._initial_runtime == "local",
                            id="query-runtime-local",
                        )
                        yield RadioButton(
                            t("taskSetup.runtimeDocker"),
                            value=self._initial_runtime == "docker",
                            id="query-runtime-docker",
                        )
                        yield RadioButton(
                            t("taskSetup.runtimeE2b"),
                            value=self._initial_runtime == "e2b",
                            id="query-runtime-e2b",
                        )
                    yield Checkbox(
                        t("taskSetup.fastMode"),
                        value=self._initial_mode == "fast",
                        id="query-mage-fast",
                    )

    def on_mount(self) -> None:
        table = self.query_one("#context-table", DataTable)
        table.add_columns(t("taskSetup.colPath"), t("taskSetup.colSize"))

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        event.stop()
        path = event.path.resolve()
        if path in self._files:
            self.app.bell()
            return
        self._files.append(path)
        table = self.query_one("#context-table", DataTable)
        table.add_row(self._display_path(path), self._format_size(path), key=str(path))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Pressing Enter on a row removes it.
        self._remove_row(event.row_key.value)

    def key_delete(self) -> None:
        table = self.query_one("#context-table", DataTable)
        if table.has_focus and table.cursor_row >= 0 and table.row_count > 0:
            row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
            self._remove_row(row_key.value)

    def _remove_row(self, key: str | None) -> None:
        if key is None:
            return
        target = Path(key)
        if target in self._files:
            self._files.remove(target)
        table = self.query_one("#context-table", DataTable)
        try:
            table.remove_row(key)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-clear":
            self._files.clear()
            self.query_one("#context-table", DataTable).clear()
            self.query_one("#task-input", TextArea).load_text("")
        elif event.button.id == "btn-generate":
            task = self.query_one("#task-input", TextArea).text.strip()
            self.post_message(self.GenerateRequested(task, tuple(self._files)))
        elif event.button.id == "btn-tree-home":
            self._reroot_tree(Path.home())
        elif event.button.id == "btn-tree-up":
            tree = self.query_one("#file-tree", FilteredDirectoryTree)
            current = Path(str(tree.path)).resolve()
            if current.parent != current:
                self._reroot_tree(current.parent)
            else:
                self.app.bell()

    def _reroot_tree(self, new_root: Path) -> None:
        new_root = new_root.expanduser().resolve()
        tree = self.query_one("#file-tree", FilteredDirectoryTree)
        tree.path = str(new_root)
        try:
            tree.reload()
        except Exception:
            pass
        self.query_one("#tree-path", Label).update(self._format_root_label(new_root))

    @staticmethod
    def _format_root_label(path: Path) -> str:
        try:
            home = Path.home()
            if path == home or path.is_relative_to(home):
                return f"~/{path.relative_to(home)}" if path != home else "~/"
        except (ValueError, AttributeError):
            pass
        return str(path)

    @staticmethod
    def _display_path(path: Path) -> str:
        cwd = Path.cwd()
        try:
            return str(path.relative_to(cwd))
        except ValueError:
            return str(path)

    @staticmethod
    def _format_size(path: Path) -> str:
        try:
            size = path.stat().st_size
        except OSError:
            return "?"
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
