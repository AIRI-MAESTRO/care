"""ExportChainModal — write one chain dict to disk (TODO §5 P1).

Pushed from EvolutionScreen's `x` row action when the user
wants to save a Pareto-front individual's chain payload to a
file before deciding whether to accept it as `latest`. Also
re-usable from anywhere holding a chain dict (Library row
inspect → "Save as file"; LineageModal compare → "Diff vs.
on-disk copy").

Distinct from :class:`care.screens.export.ExportModal`, which
packages saved-Memory entities into a tarball — this modal
operates on a single in-memory chain dict and routes through
:func:`care.chain_export.export_chain` so the user picks
JSON (round-trip safe) or Python (runnable script).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
)

from care.chain_export import (
    ChainExportError,
    ExportFormat,
    ExportResult,
    export_chain,
)
from care.runtime.i18n import t


_FORMAT_EXTENSIONS: dict[ExportFormat, str] = {
    "json": ".json",
    "python": ".py",
    "markdown": ".md",
}


@dataclass(frozen=True)
class ExportChainResult:
    """Dismiss envelope.

    ``ok=True`` carries the :class:`ExportResult` from a
    successful write; ``ok=False`` carries the error string the
    user saw before cancelling. ``None`` (no result) means the
    user pressed Cancel before submitting.
    """

    ok: bool = False
    path: Path | None = None
    bytes_written: int = 0
    format: ExportFormat = "json"
    error: str = ""


class ExportChainModal(ModalScreen[ExportChainResult | None]):
    """Output-path + format picker for a single chain dict.

    Construct with the chain payload + a friendly display name
    (used in the title + default filename). Submitting calls
    :func:`care.chain_export.export_chain` and dismisses with
    the result envelope.
    """

    DEFAULT_CSS = """
    ExportChainModal {
        align: center middle;
    }
    ExportChainModal #export-chain-box {
        width: 70;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    ExportChainModal #export-chain-title {
        text-style: bold;
        padding-bottom: 1;
    }
    ExportChainModal #export-chain-summary {
        color: $text-muted;
        margin-bottom: 1;
    }
    ExportChainModal Input {
        margin-bottom: 1;
    }
    ExportChainModal #export-chain-path-row {
        height: auto;
    }
    ExportChainModal #export-chain-path-row #export-chain-path {
        width: 1fr;
    }
    ExportChainModal #export-chain-btn-browse {
        width: auto;
        margin-left: 1;
    }
    ExportChainModal RadioSet {
        margin-bottom: 1;
    }
    ExportChainModal #export-chain-result {
        color: $accent;
        margin-bottom: 1;
    }
    ExportChainModal #export-chain-buttons {
        height: auto;
        align-horizontal: right;
    }
    ExportChainModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        *,
        chain: dict,
        display_name: str = "",
        entity_id: str = "",
        version: str = "",
        default_path: Path | str | None = None,
        default_format: ExportFormat = "json",
    ) -> None:
        super().__init__()
        self._chain = dict(chain)
        self._display_name = display_name or "chain"
        if default_path is None:
            stem = _default_stem(entity_id, version, display_name)
            ext = _FORMAT_EXTENSIONS.get(default_format, ".json")
            # Default to the CURRENT folder (where CARE was launched) so the
            # export lands somewhere obvious; the path field lets the user
            # pick anywhere else.
            default_path = Path.cwd() / f"{stem}{ext}"
        self._default_path = str(default_path)
        self._default_format: ExportFormat = default_format
        self.last_result: ExportResult | None = None
        self.last_error: str = ""
        self.exporting: bool = False

    def compose(self) -> ComposeResult:
        with Vertical(id="export-chain-box"):
            yield Label(
                t("exportChain.title", name=self._display_name),
                id="export-chain-title",
            )
            yield Static(
                self._summary_text(),
                id="export-chain-summary",
            )
            yield Label(t("exportChain.outputPath"))
            with Horizontal(id="export-chain-path-row"):
                yield Input(
                    value=self._default_path,
                    id="export-chain-path",
                )
                yield Button(
                    t("exportChain.browse"),
                    id="export-chain-btn-browse",
                )
            yield Label(t("exportChain.format"))
            with RadioSet(id="export-chain-format"):
                yield RadioButton(
                    t("exportChain.formatMarkdown"),
                    value=self._default_format == "markdown",
                    id="export-chain-format-markdown",
                )
                yield RadioButton(
                    t("exportChain.formatJson"),
                    value=self._default_format == "json",
                    id="export-chain-format-json",
                )
                yield RadioButton(
                    t("exportChain.formatPython"),
                    value=self._default_format == "python",
                    id="export-chain-format-python",
                )
            yield Static("", id="export-chain-result")
            with Horizontal(id="export-chain-buttons"):
                yield Button(t("common.cancel"), id="export-chain-btn-cancel")
                yield Button(
                    t("exportChain.export"),
                    id="export-chain-btn-submit",
                    variant="primary",
                )

    def _summary_text(self) -> str:
        steps = self._chain.get("steps")
        n_steps = len(steps) if isinstance(steps, list) else 0
        key = (
            "exportChain.stepCount.one" if n_steps == 1
            else "exportChain.stepCount.many"
        )
        return t(key, count=n_steps)

    def _read_path(self) -> Path:
        try:
            widget = self.query_one("#export-chain-path", Input)
        except Exception:
            return Path(self._default_path).expanduser()
        return Path(widget.value or self._default_path).expanduser()

    def _read_format(self) -> ExportFormat:
        try:
            radio = self.query_one("#export-chain-format", RadioSet)
        except Exception:
            return self._default_format
        pressed = radio.pressed_button
        if pressed is None:
            return self._default_format
        pid = pressed.id or ""
        if pid.endswith("-python"):
            return "python"
        if pid.endswith("-markdown"):
            return "markdown"
        return "json"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "export-chain-btn-cancel":
            self.action_cancel()
        elif bid == "export-chain-btn-submit":
            self.action_submit()
        elif bid == "export-chain-btn-browse":
            self._open_directory_picker()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Keep the output path's extension in sync with the chosen format
        — otherwise switching from Markdown to JSON/Python would still write
        a ``.md``-named file with JSON/Python content."""
        if event.radio_set.id != "export-chain-format":
            return
        self._sync_path_extension(self._read_format())

    def _sync_path_extension(self, fmt: ExportFormat) -> None:
        ext = _FORMAT_EXTENSIONS.get(fmt, ".json")
        try:
            field = self.query_one("#export-chain-path", Input)
        except Exception:
            return
        current = Path(field.value or self._default_path)
        if current.suffix.lower() == ext:
            return
        new_value = str(current.with_suffix(ext))
        if new_value != field.value:
            field.value = new_value

    def _open_directory_picker(self) -> None:
        """Browse for a destination folder; on selection, rewrite the path
        field to ``<chosen folder>/<current filename>`` (keeping the chosen
        filename + extension)."""
        from care.screens.directory_picker import DirectoryPickerModal

        current = self._read_path()
        start = current.parent if current.parent.is_dir() else Path.cwd()

        def _on_pick(folder: Path | None) -> None:
            if folder is None:
                return
            try:
                field = self.query_one("#export-chain-path", Input)
            except Exception:
                return
            filename = Path(field.value or self._default_path).name or "chain"
            field.value = str(Path(folder) / filename)

        self.app.push_screen(DirectoryPickerModal(start=start), _on_pick)

    def action_submit(self) -> None:
        if self.exporting:
            return
        self.exporting = True
        path = self._read_path()
        fmt = self._read_format()
        try:
            result = export_chain(
                self._chain, path, format=fmt,
            )
        except ChainExportError as exc:
            self.last_error = str(exc)
            self.last_result = None
            self._render_result()
            self.exporting = False
            return
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_result = None
            self._render_result()
            self.exporting = False
            return
        self.last_result = result
        self.last_error = ""
        self._render_result()
        self.exporting = False
        self.dismiss(ExportChainResult(
            ok=True,
            path=result.path,
            bytes_written=result.bytes_written,
            format=result.format,
        ))

    def _render_result(self) -> None:
        try:
            target = self.query_one("#export-chain-result", Static)
        except Exception:
            return
        if self.last_error:
            target.update(f"⚠ {self.last_error}")
            return
        if self.last_result is None:
            target.update("")
            return
        result = self.last_result
        target.update(
            t(
                "exportChain.wrote",
                name=result.path.name,
                bytes=f"{result.bytes_written:,}",
                format=result.format,
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


def _default_stem(entity_id: str, version: str, display_name: str) -> str:
    """Build the default export filename stem.

    Prefers ``chain_<id>[_v<version>]`` when an ``entity_id`` is known (the
    Library → Inspection case) so exports are traceable back to the saved
    chain + version; falls back to the sanitized display name (e.g.
    evolution candidates that aren't saved yet), then plain ``chain``."""
    eid = _safe_filename_stem(entity_id)
    if eid:
        stem = f"chain_{eid}"
        ver = _safe_filename_stem(version).lstrip("vV")
        if ver:
            stem = f"{stem}_v{ver}"
        return stem
    return _safe_filename_stem(display_name) or "chain"


def _safe_filename_stem(name: str) -> str:
    """Sanitize a display name into a filename stem.

    Keeps alnum + dash + underscore; collapses runs of other
    characters into a single dash; trims leading/trailing dashes.
    """
    if not name:
        return ""
    out: list[str] = []
    last_dash = False
    for ch in name.strip().lower():
        if ch.isalnum() or ch in "-_":
            out.append(ch)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    return "".join(out).strip("-_")


__all__ = [
    "ExportChainModal",
    "ExportChainResult",
    "_default_stem",
    "_safe_filename_stem",
]
