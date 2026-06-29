"""ExportModal — pack selected agents into a tarball
(TODO §1.1 P0.30).

Pushed by the `Export library` command (palette entry +
optional bulk-action affordance). Collects an output path
plus an "include skills" toggle, then calls
:func:`care.runtime.export_library_bundle` on the worker pool.

The modal stays open until the worker settles so the user
sees the resulting summary inline before dismissing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Static

from care.runtime.i18n import t
from care.runtime.library_bundle import (
    BundleExportResult,
    export_library_bundle,
)


@dataclass(frozen=True)
class ExportRequest:
    """Snapshot of the user-supplied form state."""

    output_path: Path
    include_skills: bool


class ExportModal(ModalScreen[BundleExportResult | None]):
    """Output-path selector + "include skills" checkbox.

    Construct with the `CareMemory`-like facade + the chain
    entity ids to export, plus optionally the AgentSkill ids
    the chains depend on. Submitting fires
    `export_library_bundle` and dismisses with the
    :class:`BundleExportResult` (or ``None`` on cancel).
    """

    DEFAULT_CSS = """
    ExportModal {
        align: center middle;
    }
    ExportModal #export-box {
        width: 70;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    ExportModal #export-title {
        text-style: bold;
        padding-bottom: 1;
    }
    ExportModal Input {
        margin-bottom: 1;
    }
    ExportModal #export-summary {
        color: $text-muted;
        margin-bottom: 1;
    }
    ExportModal #export-result {
        color: $accent;
        margin-bottom: 1;
    }
    ExportModal #export-buttons {
        height: auto;
        align-horizontal: right;
    }
    ExportModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        *,
        memory: Any,
        entity_ids: Iterable[str],
        skill_entity_ids: Iterable[str] = (),
        default_path: Path | str = "~/care-export.tar.gz",
        namespace: str | None = None,
    ) -> None:
        super().__init__()
        self._memory = memory
        self._entity_ids = tuple(entity_ids)
        self._skill_ids = tuple(skill_entity_ids)
        self._default_path = str(default_path)
        self._namespace = namespace
        # Last export result — exposed for tests + future
        # telemetry.
        self.last_result: BundleExportResult | None = None
        self.exporting: bool = False

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="export-box"):
            yield Label(t("export.title"), id="export-title")
            yield Static(
                self._summary_text(), id="export-summary",
            )
            yield Label(t("export.outputPath"))
            yield Input(
                value=self._default_path,
                id="export-path",
            )
            yield Checkbox(
                t("export.includeSkills"),
                value=bool(self._skill_ids),
                id="export-skills",
            )
            yield Static("", id="export-result")
            with Horizontal(id="export-buttons"):
                yield Button(t("common.cancel"), id="export-btn-cancel")
                yield Button(
                    t("export.export"),
                    id="export-btn-submit",
                    variant="primary",
                )

    def _summary_text(self) -> str:
        n_chains = len(self._entity_ids)
        bits = [
            t(
                "export.summaryChains.one" if n_chains == 1
                else "export.summaryChains.many",
                n=n_chains,
            ),
        ]
        if self._skill_ids:
            n_skills = len(self._skill_ids)
            bits.append(
                t(
                    "export.summarySkills.one" if n_skills == 1
                    else "export.summarySkills.many",
                    n=n_skills,
                ),
            )
        return " + ".join(bits)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _read_path(self) -> Path:
        try:
            widget = self.query_one("#export-path", Input)
        except Exception:
            return Path(self._default_path).expanduser()
        return Path(widget.value or self._default_path).expanduser()

    def _read_include_skills(self) -> bool:
        try:
            checkbox = self.query_one("#export-skills", Checkbox)
        except Exception:
            return bool(self._skill_ids)
        return bool(checkbox.value)

    def current_request(self) -> ExportRequest:
        """Snapshot the form values — exposed for tests so they
        don't have to scrape widgets."""
        return ExportRequest(
            output_path=self._read_path(),
            include_skills=self._read_include_skills(),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "export-btn-cancel":
            self.action_cancel()
        elif bid == "export-btn-submit":
            self.action_submit()

    def action_submit(self) -> None:
        if self.exporting:
            return
        self.exporting = True
        self.run_worker(
            self._export_worker(),
            name="export_bundle",
            group="export",
            exclusive=True,
            exit_on_error=False,
        )

    async def _export_worker(self) -> None:
        request = self.current_request()
        skill_ids = self._skill_ids if request.include_skills else ()
        try:
            result = await export_library_bundle(
                self._memory,
                self._entity_ids,
                request.output_path,
                skill_entity_ids=skill_ids,
                source_namespace=self._namespace,
            )
        except Exception as exc:  # noqa: BLE001
            self.last_result = BundleExportResult(
                path=request.output_path,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._render_result()
            self.exporting = False
            return
        self.last_result = result
        self._render_result()
        self.exporting = False
        # Auto-dismiss successful exports so the host can
        # surface a toast; failures keep the modal open so the
        # user can see what went wrong + retry.
        if result.success:
            self.dismiss(result)

    def _render_result(self) -> None:
        try:
            target = self.query_one("#export-result", Static)
        except Exception:
            return
        if self.last_result is None:
            target.update("")
            return
        result = self.last_result
        if result.error:
            target.update(f"⚠ {result.error}")
            return
        n_written = result.total_written
        bits = [
            t(
                "export.wroteItems.one" if n_written == 1
                else "export.wroteItems.many",
                n=n_written,
            ),
            t("export.bytes", n=f"{result.bytes_written:,}"),
        ]
        if result.skipped_chains or result.skipped_skills:
            bits.append(
                t(
                    "export.skipped",
                    n=len(result.skipped_chains) + len(result.skipped_skills),
                )
            )
        target.update("  ·  ".join(bits))

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["ExportModal", "ExportRequest"]
