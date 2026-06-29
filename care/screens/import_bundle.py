"""ImportModal — restore a bundle into Memory (TODO §1.1 P0.31).

Pushed by the `Import bundle` command (palette entry +
LibraryScreen menu entry). The user picks a tarball, the
modal previews the manifest via
:func:`care.runtime.library_bundle.read_bundle_manifest`,
and on submit fires
:func:`care.runtime.library_bundle.import_library_bundle`
against the configured Memory + namespace.

The collision policy + dry-run toggle live on the modal so
the host doesn't have to thread them through separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
)

from care.runtime.i18n import t
from care.runtime.library_bundle import (
    BundleImportResult,
    BundleManifest,
    LibraryBundleError,
    import_library_bundle,
    read_bundle_manifest,
)


CollisionPolicy = Literal["skip", "overwrite", "raise"]


_POLICY_BY_BUTTON_ID = {
    "import-collision-skip": "skip",
    "import-collision-overwrite": "overwrite",
    "import-collision-raise": "raise",
}


@dataclass(frozen=True)
class ImportRequest:
    """Snapshot of the user-supplied form state."""

    tarball_path: Path
    on_collision: CollisionPolicy
    dry_run: bool


class ImportModal(ModalScreen[BundleImportResult | None]):
    """Bundle import flow.

    Construct with the `CareMemory`-like facade + optional
    namespace + default tarball path. The user types a path,
    the modal previews the manifest, then `Import` calls
    `import_library_bundle`."""

    DEFAULT_CSS = """
    ImportModal {
        align: center middle;
    }
    ImportModal #import-box {
        width: 75;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    ImportModal #import-title {
        text-style: bold;
        padding-bottom: 1;
    }
    ImportModal Input {
        margin-bottom: 1;
    }
    ImportModal #import-preview {
        color: $text-muted;
        margin-bottom: 1;
    }
    ImportModal #import-result {
        color: $accent;
        margin-bottom: 1;
    }
    ImportModal RadioSet {
        height: 5;
    }
    ImportModal #import-buttons {
        height: auto;
        align-horizontal: right;
    }
    ImportModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    class PreviewLoaded(Message):
        """Posted after a successful preview parse so the host
        can record telemetry. Not used by the modal itself."""

        def __init__(self, manifest: BundleManifest) -> None:
            super().__init__()
            self.manifest = manifest

    def __init__(
        self,
        *,
        memory: Any,
        default_path: Path | str = "~/care-export.tar.gz",
        namespace: str | None = None,
    ) -> None:
        super().__init__()
        self._memory = memory
        self._default_path = str(default_path)
        self._namespace = namespace
        # Last preview / result — exposed for tests + future
        # telemetry.
        self.manifest: BundleManifest | None = None
        self.preview_error: str | None = None
        self.last_result: BundleImportResult | None = None
        self.importing: bool = False

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="import-box"):
            yield Label(t("importBundle.title"), id="import-title")
            yield Label(t("importBundle.tarballPath"))
            yield Input(
                value=self._default_path,
                id="import-path",
            )
            yield Static(t("importBundle.noPreview"), id="import-preview")
            yield Label(t("importBundle.onCollision"))
            with RadioSet(id="import-collision"):
                yield RadioButton(
                    t("importBundle.collisionSkip"),
                    value=True,
                    id="import-collision-skip",
                )
                yield RadioButton(
                    t("importBundle.collisionOverwrite"),
                    id="import-collision-overwrite",
                )
                yield RadioButton(
                    t("importBundle.collisionRaise"),
                    id="import-collision-raise",
                )
            yield Checkbox(
                t("importBundle.dryRun"),
                value=False,
                id="import-dry-run",
            )
            yield Static("", id="import-result")
            with Horizontal(id="import-buttons"):
                yield Button(t("common.cancel"), id="import-btn-cancel")
                yield Button(
                    t("importBundle.preview"),
                    id="import-btn-preview",
                )
                yield Button(
                    t("importBundle.import"),
                    id="import-btn-submit",
                    variant="primary",
                )

    def on_mount(self) -> None:
        # Kick off a first preview pass when the default path
        # exists; otherwise leave the placeholder so the user
        # types their own.
        self._refresh_preview()

    # ------------------------------------------------------------------
    # Field reads
    # ------------------------------------------------------------------

    def _read_path(self) -> Path:
        try:
            widget = self.query_one("#import-path", Input)
        except Exception:
            return Path(self._default_path).expanduser()
        return Path(widget.value or self._default_path).expanduser()

    def _read_collision(self) -> CollisionPolicy:
        try:
            rs = self.query_one("#import-collision", RadioSet)
        except Exception:
            return "skip"
        pressed = rs.pressed_button
        pid = pressed.id if pressed is not None else None
        return _POLICY_BY_BUTTON_ID.get(pid or "", "skip")

    def _read_dry_run(self) -> bool:
        try:
            cb = self.query_one("#import-dry-run", Checkbox)
        except Exception:
            return False
        return bool(cb.value)

    def current_request(self) -> ImportRequest:
        """Snapshot the form values — exposed for tests."""
        return ImportRequest(
            tarball_path=self._read_path(),
            on_collision=self._read_collision(),
            dry_run=self._read_dry_run(),
        )

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "import-path":
            return
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        path = self._read_path()
        if not path.exists():
            self.manifest = None
            self.preview_error = None
            self._render_preview("(file not found — type a valid path)")
            return
        try:
            manifest = read_bundle_manifest(path)
        except LibraryBundleError as exc:
            self.manifest = None
            self.preview_error = str(exc)
            self._render_preview(f"⚠ {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            self.manifest = None
            self.preview_error = f"{type(exc).__name__}: {exc}"
            self._render_preview(f"⚠ {self.preview_error}")
            return
        self.manifest = manifest
        self.preview_error = None
        self._render_preview(self._format_preview(manifest))
        self.post_message(self.PreviewLoaded(manifest))

    @staticmethod
    def _format_preview(manifest: BundleManifest) -> str:
        bits = [
            f"schema v{manifest.schema_version}",
            f"{len(manifest.chains)} chain{'s' if len(manifest.chains) != 1 else ''}",
        ]
        if manifest.agent_skills:
            bits.append(
                f"{len(manifest.agent_skills)} skill{'s' if len(manifest.agent_skills) != 1 else ''}",
            )
        if manifest.source_namespace:
            bits.append(f"from `{manifest.source_namespace}`")
        return "  ·  ".join(bits)

    def _render_preview(self, text: str) -> None:
        try:
            target = self.query_one("#import-preview", Static)
        except Exception:
            return
        target.update(text)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "import-btn-cancel":
            self.action_cancel()
        elif bid == "import-btn-preview":
            self._refresh_preview()
        elif bid == "import-btn-submit":
            self.action_submit()

    def action_submit(self) -> None:
        if self.importing:
            return
        if self.manifest is None:
            return  # Block until a valid preview lands.
        self.importing = True
        self.run_worker(
            self._import_worker(),
            name="import_bundle",
            group="import",
            exclusive=True,
            exit_on_error=False,
        )

    async def _import_worker(self) -> None:
        request = self.current_request()
        try:
            result = await import_library_bundle(
                self._memory,
                request.tarball_path,
                on_collision=request.on_collision,
                namespace=self._namespace,
                dry_run=request.dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            self.last_result = BundleImportResult(
                error=f"{type(exc).__name__}: {exc}",
                manifest=self.manifest,
            )
            self._render_result()
            self.importing = False
            return
        self.last_result = result
        self._render_result()
        self.importing = False
        # Auto-dismiss successful (non-dry-run) imports so the
        # host can refresh the library. Dry-run + failures keep
        # the modal open so the user reads the report.
        if result.success and not request.dry_run:
            self.dismiss(result)

    def _render_result(self) -> None:
        try:
            target = self.query_one("#import-result", Static)
        except Exception:
            return
        if self.last_result is None:
            target.update("")
            return
        result = self.last_result
        if result.error:
            target.update(f"⚠ {result.error}")
            return
        bits = [
            f"imported: {result.imported_count}",
        ]
        if result.skipped_count:
            bits.append(f"skipped: {result.skipped_count}")
        if result.failed_count:
            bits.append(f"failed: {result.failed_count}")
        target.update("  ·  ".join(bits))

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = [
    "CollisionPolicy",
    "ImportModal",
    "ImportRequest",
]
