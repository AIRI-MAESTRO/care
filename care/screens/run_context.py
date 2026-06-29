"""RunContextModal — adjust task + files for a re-run
(TODO §1.1 P0.21).

Pushed when the user invokes `Run` on a saved agent and
elects "Re-run with new inputs". Bound to the shipped
:mod:`care.runtime.run_context_draft` data layer:

* :func:`extract_run_context_draft(chain)` on construction —
  seeds the form from the saved chain's CARE metadata.
* :func:`set_task` / :func:`drop_file` / :func:`restore_file` /
  :func:`set_model_override` on each field edit.
* :func:`validate_run_context_draft(draft)` runs on every
  change.
* :func:`apply_overrides(config, draft)` + :func:`build_extra_kwargs(draft)`
  are exposed via :class:`RunContextResult` so the host
  screen can hand them to `execute_library_run`.

Pure presentation — the modal does not run the chain itself
(that lives on the future ExecutionScreen, P0.22).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Label,
    Static,
    TextArea,
)

from care.runtime.i18n import t
from care.runtime.run_context_draft import (
    ContextFile,
    RunContextDraft,
    RunContextIssue,
    attach_path,
    drop_file,
    extract_run_context_draft,
    missing_active_files,
    resolve_file_arg,
    restore_file,
    set_model_override,
    set_task,
    validate_run_context_draft,
)


def _chain_reads_document(chain: Any) -> bool:
    """Best-effort: does ``chain`` have a document-reading skill step?

    Accepts a dict or a ``ReasoningChain`` object (via ``to_dict``). Any
    failure → ``False`` (no banner)."""
    try:
        from care.skill_file_inputs import requires_file_input

        if isinstance(chain, dict):
            data = chain
        else:
            to_dict = getattr(chain, "to_dict", None)
            if not callable(to_dict):
                return False
            try:
                data = to_dict(full=True)
            except TypeError:
                data = to_dict()
        return bool(isinstance(data, dict) and requires_file_input(data))
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class RunContextResult:
    """Dismiss envelope.

    ``submitted`` is ``True`` when the user clicked "Run" /
    "Run (modified)"; ``False`` for cancel. The host screen
    routes only on submit — `apply_overrides` +
    `build_extra_kwargs` consume the carried draft."""

    submitted: bool
    draft: RunContextDraft


class RunContextModal(ModalScreen[RunContextResult]):
    """Modal that lets the user edit the task description + the
    context-file set + per-run overrides before re-running a
    saved chain."""

    DEFAULT_CSS = """
    RunContextModal {
        align: center middle;
    }
    RunContextModal #run-context-box {
        width: 80;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    RunContextModal #run-context-title {
        text-style: bold;
        padding-bottom: 1;
    }
    RunContextModal #run-context-task {
        height: 5;
        margin-bottom: 1;
    }
    RunContextModal #run-context-required {
        color: $warning;
        margin-bottom: 1;
    }
    RunContextModal #run-context-files {
        height: 7;
        border: round $primary 30%;
        margin-bottom: 1;
    }
    /* File rows are `compact` Buttons (borderless, height 1) so the label
       isn't clipped — a default Button's `border: tall` is 3 cells high and
       at height 1 only the top border showed, hiding the file name. */
    RunContextModal .file-row {
        width: 100%;
        margin: 0;
        content-align: left middle;
        text-align: left;
    }
    RunContextModal #run-context-attach-row {
        height: auto;
        margin-bottom: 1;
    }
    RunContextModal #run-context-attach {
        width: 1fr;
    }
    RunContextModal #run-context-issues {
        color: $warning;
        margin-bottom: 1;
    }
    RunContextModal #run-context-buttons {
        height: auto;
        align-horizontal: right;
    }
    RunContextModal .run-context-hint {
        color: $text-muted;
    }
    RunContextModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        chain: Any,
        *,
        source_name: str = "",
    ) -> None:
        super().__init__()
        self.draft: RunContextDraft = extract_run_context_draft(
            chain, source_name=source_name,
        )
        self.issues: tuple[RunContextIssue, ...] = ()
        self._suppress_count: int = 0
        # Does this chain READ a document (docx/pdf/… skill)? Drives a
        # proactive "attach a file" banner. Keyword heuristic (sync, no LLM
        # in the modal) — good enough for a hint.
        self._reads_doc: bool = _chain_reads_document(chain)

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def _model_placeholder(self) -> str:
        """Placeholder for the model field: shows the CURRENT default model
        (from config) so the user sees what 'leave blank' resolves to;
        falls back to a generic hint when the config model isn't known."""
        cfg = getattr(getattr(self, "app", None), "config", None)
        mage = getattr(cfg, "mage", None) if cfg is not None else None
        model = (getattr(mage, "model", "") or "").strip() if mage else ""
        if model:
            return t("runContext.modelPlaceholderNamed", model=model)
        return t("runContext.modelPlaceholder")

    def compose(self) -> ComposeResult:
        with Vertical(id="run-context-box"):
            title = (
                t("runContext.titleNamed", name=self.draft.source_name)
                if self.draft.source_name
                else t("runContext.title")
            )
            yield Label(title, id="run-context-title")
            yield Label(t("common.taskDescription"))
            yield TextArea(
                self.draft.task_description,
                id="run-context-task",
            )
            yield Label(t("runContext.contextFiles"))
            yield Static("", id="run-context-required")
            with VerticalScroll(id="run-context-files"):
                if self.draft.files:
                    for cf in self.draft.files:
                        yield self._compose_file_button(cf)
                else:
                    yield Static(
                        t("runContext.noContextFiles"),
                        id="run-context-no-files",
                    )
                    yield Static(
                        t("runContext.noContextFilesHint"),
                        id="run-context-no-files-hint",
                        classes="run-context-hint",
                    )
            with Horizontal(id="run-context-attach-row"):
                yield Input(
                    placeholder=t("runContext.attachPlaceholder"),
                    id="run-context-attach",
                )
                yield Button(
                    t("runContext.browse"),
                    id="run-context-attach-browse",
                )
            yield Label(t("runContext.chooseModel"))
            yield Input(
                value=self.draft.model_override or "",
                placeholder=self._model_placeholder(),
                id="run-context-model",
            )
            with Collapsible(
                title=t("runContext.connection"),
                collapsed=True,
                id="run-context-connection",
            ):
                yield Label(t("runContext.baseUrl"))
                yield Input(
                    value=self.draft.base_url_override or "",
                    placeholder=t("runContext.baseUrlPlaceholder"),
                    id="run-context-base-url",
                )
                yield Label(t("runContext.apiKey"))
                yield Input(
                    value=self.draft.api_key_override or "",
                    placeholder=t("runContext.apiKeyPlaceholder"),
                    password=True,
                    id="run-context-api-key",
                )
            yield Checkbox(
                t("runContext.streaming"),
                value=self.draft.streaming_enabled,
                id="run-context-streaming",
            )
            yield Static("", id="run-context-issues")
            with Horizontal(id="run-context-buttons"):
                yield Button(t("common.cancel"), id="run-context-cancel")
                yield Button(
                    t("runContext.run"),
                    id="run-context-submit",
                    variant="primary",
                )

    def on_mount(self) -> None:
        self._refresh_submit_label()
        self._run_validate()
        # Defer the required-files banner until the file rows are laid
        # out — querying the Static during the screen's own on_mount can
        # race the child mount and silently no-op.
        self.call_after_refresh(self._refresh_required_banner)

    # ------------------------------------------------------------------
    # File list
    # ------------------------------------------------------------------

    def _render_file_list(self) -> None:
        """Reconcile the file-row buttons with ``self.draft.files``.

        Incremental, NOT a clear-and-remount, which keeps it free of the
        ``DuplicateIds`` race that the async-deferred ``remove()`` /
        ``remove_children()`` would otherwise create:

        * a button is mounted ONLY when its stable id is absent from the
          live DOM (``button_id not in present``), so a mount never
          collides with an existing (or mid-removal) widget;
        * a button is removed ONLY when its file left the draft (a
          user-added row that was dropped vanishes outright);
        * every surviving row's label is updated in place so ``Drop``
          flips to ``Restore`` and back.

        Call sites are single user events (a button press / an attach),
        so a removal scheduled by one render always settles before the
        next — there is no sub-refresh add→drop→re-add path reachable
        from the UI. The empty-state placeholder is swapped in/out as the
        set crosses zero rows."""
        if not self.is_mounted:
            return
        try:
            container = self.query_one("#run-context-files", VerticalScroll)
        except Exception:
            return
        desired = {
            self._file_button_id(cf.path): cf
            for cf in self.draft.files
            if cf.path
        }
        # Drop buttons whose file is gone (added-then-dropped rows).
        for button in list(container.query(Button)):
            if button.id not in desired:
                button.remove()
        present = {b.id for b in container.query(Button)}
        for button_id, cf in desired.items():
            if button_id in present:
                try:
                    container.query_one(f"#{button_id}", Button).label = (
                        self._file_button_label(cf)
                    )
                except Exception:
                    continue
            else:
                self._remove_placeholders(container)
                container.mount(self._compose_file_button(cf))
        if not desired:
            self._ensure_placeholders(container)

    def _remove_placeholders(self, container: VerticalScroll) -> None:
        for pid in ("run-context-no-files", "run-context-no-files-hint"):
            try:
                container.query_one(f"#{pid}", Static).remove()
            except Exception:
                continue

    def _ensure_placeholders(self, container: VerticalScroll) -> None:
        try:
            container.query_one("#run-context-no-files", Static)
            return  # already present
        except Exception:
            pass
        container.mount(
            Static(t("runContext.noContextFiles"), id="run-context-no-files"),
        )
        container.mount(
            Static(
                t("runContext.noContextFilesHint"),
                id="run-context-no-files-hint",
                classes="run-context-hint",
            ),
        )

    def _refresh_required_banner(self) -> None:
        """Surface the active context files the chain expects but that
        no longer resolve on disk — the set that would otherwise have
        CARL prime the run with a ``[missing context file: …]``
        placeholder. Cleared when nothing's missing."""
        if not self.is_mounted:
            return
        try:
            target = self.query_one("#run-context-required", Static)
        except Exception:
            return
        missing = missing_active_files(self.draft)
        if missing:
            target.update(t("runContext.requiredMissing", count=len(missing)))
            return
        # Proactive nudge: the chain reads a document but nothing's attached.
        if self._reads_doc and not self.draft.active_files:
            target.update(t("runContext.docNeeded"))
            return
        target.update("")

    @staticmethod
    def _file_button_label(cf: ContextFile) -> str:
        dropped = cf.status == "dropped"
        badge = "✗" if dropped else "·"
        verb = (
            t("runContext.restore") if dropped else t("runContext.drop")
        )
        name = Path(cf.path).name or cf.path
        if len(name) > 40:  # ellipsize so the row doesn't overflow
            name = f"{name[:18]}…{name[-18:]}"
        size = RunContextModal._human_size(cf.size_bytes)
        tail = f" ({size})" if size else ""
        return f"{badge} {name}{tail}  [{verb}]"

    @staticmethod
    def _human_size(n: int) -> str:
        """Human-readable byte count (``""`` for unknown / zero)."""
        if not n or n <= 0:
            return ""
        size = float(n)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return ""

    def _compose_file_button(self, cf: ContextFile) -> Button:
        return Button(
            self._file_button_label(cf),
            id=self._file_button_id(cf.path),
            classes="file-row",
            compact=True,
        )

    @staticmethod
    def _file_button_id(path: str) -> str:
        """Generate a stable, Textual-id-compatible button id
        for ``path``. Textual ids must start with a letter and
        contain only ``[A-Za-z0-9_-]``."""
        from hashlib import sha1

        digest = sha1(path.encode("utf-8")).hexdigest()[:12]
        return f"run-context-file-{digest}"

    # ------------------------------------------------------------------
    # Field handlers
    # ------------------------------------------------------------------

    def _consume_suppression(self) -> bool:
        if self._suppress_count > 0:
            self._suppress_count -= 1
            return True
        return False

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "run-context-task":
            return
        if self._consume_suppression():
            return
        self.draft = set_task(self.draft, event.text_area.text)
        self._refresh_submit_label()
        self._run_validate()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id not in (
            "run-context-model", "run-context-base-url", "run-context-api-key",
        ):
            return
        if self._consume_suppression():
            return
        self._sync_overrides_from_inputs()
        self._refresh_submit_label()
        self._run_validate()

    def _sync_overrides_from_inputs(self) -> None:
        """Read the model + base-URL + API-key fields and fold all three onto
        the draft together (so editing one never clears the others)."""

        def _read(widget_id: str, current: str | None) -> str | None:
            try:
                return self.query_one(f"#{widget_id}", Input).value.strip() or None
            except Exception:
                return current

        self.draft = set_model_override(
            self.draft,
            model=_read("run-context-model", self.draft.model_override),
            base_url=_read("run-context-base-url", self.draft.base_url_override),
            api_key=_read("run-context-api-key", self.draft.api_key_override),
        )

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id != "run-context-streaming":
            return
        if self._consume_suppression():
            return
        self.draft = replace(self.draft, streaming_enabled=event.value)
        self._refresh_submit_label()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "run-context-cancel":
            self.action_cancel()
            return
        if bid == "run-context-submit":
            self._submit()
            return
        if bid == "run-context-attach-browse":
            self._open_file_picker()
            return
        if bid.startswith("run-context-file-"):
            self._toggle_file_for_button(bid)

    def _toggle_file_for_button(self, button_id: str) -> None:
        for cf in self.draft.files:
            if self._file_button_id(cf.path) == button_id:
                if cf.status == "dropped":
                    self.draft = restore_file(self.draft, cf.path)
                else:
                    self.draft = drop_file(self.draft, cf.path)
                self._render_file_list()
                self._refresh_required_banner()
                self._refresh_submit_label()
                self._run_validate()
                return

    # ------------------------------------------------------------------
    # Attach files (Browse… / @path)
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the quick-attach field attaches the typed path."""
        if event.input.id != "run-context-attach":
            return
        if self._attach_path(event.value):
            event.input.value = ""

    def _open_file_picker(self) -> None:
        """Push the shared file browser (the sibling of the export
        screen's folder picker) and attach whatever the user selects."""
        from care.screens.file_picker import FilePickerModal

        self.app.push_screen(
            FilePickerModal(start=self._attach_start_dir()),
            self._on_file_picked,
        )

    def _attach_start_dir(self) -> Path:
        """Root the browser at the most-recently-referenced file's
        directory when there is one, else the working directory."""
        for cf in reversed(self.draft.files):
            try:
                parent = Path(cf.path).expanduser().parent
            except Exception:
                continue
            if parent.is_dir():
                return parent
        return Path.cwd()

    def _on_file_picked(self, path: Any) -> None:
        if path is None:
            return
        self._attach_path(str(path))

    def _attach_path(self, raw: str) -> bool:
        """Resolve ``raw`` (a path or ``@path``) and attach it. Returns
        ``True`` when a file was attached, ``False`` (with an inline
        hint) when the reference doesn't point at a real file."""
        resolved = resolve_file_arg(raw)
        if not resolved or not Path(resolved).is_file():
            self._flash_attach_error(resolved or raw.strip())
            return False
        self.draft = attach_path(self.draft, raw)
        self._render_file_list()
        self._refresh_required_banner()
        self._refresh_submit_label()
        self._run_validate()
        self._notify_attached(resolved)
        return True

    def _notify_attached(self, path: str) -> None:
        """Explicit confirmation that a file was attached — so it's never
        ambiguous whether the attach took (the user-reported confusion)."""
        toast = getattr(self.app, "push_toast", None)
        if not callable(toast):
            return
        try:
            name = Path(path).name or path
            size = self._human_size(Path(path).stat().st_size)
            label = f"{name} ({size})" if size else name
            toast(t("runContext.attached", name=label), severity="information")
        except Exception:  # noqa: BLE001
            pass

    def _flash_attach_error(self, path: str) -> None:
        if not self.is_mounted:
            return
        try:
            self.query_one("#run-context-required", Static).update(
                t("runContext.attachNotFound", path=path),
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Submit label + validation
    # ------------------------------------------------------------------

    def _refresh_submit_label(self) -> None:
        if not self.is_mounted:
            return
        try:
            submit = self.query_one("#run-context-submit", Button)
        except Exception:
            return
        submit.label = (
            t("runContext.runModified") if self.draft.has_edits
            else t("runContext.run")
        )

    def _run_validate(self) -> None:
        # No file IO during tests: the validator's `check_files`
        # default does `Path.exists()` per file. Skip when the
        # active set is empty so tests with synthetic files
        # don't see warnings.
        self.issues = validate_run_context_draft(
            self.draft, check_files=False,
        )
        self._render_issues()

    def _render_issues(self) -> None:
        if not self.is_mounted:
            return
        try:
            target = self.query_one("#run-context-issues", Static)
        except Exception:
            return
        if not self.issues:
            target.update("")
            return
        lines = []
        for issue in self.issues:
            badge = "⚠" if issue.severity == "warning" else "✗"
            lines.append(f"{badge} {issue.field}: {issue.message}")
        target.update("\n".join(lines))

    @property
    def has_blocking_issues(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    # ------------------------------------------------------------------
    # Submit / cancel
    # ------------------------------------------------------------------

    def _submit(self) -> None:
        if self.has_blocking_issues:
            return
        self.dismiss(
            RunContextResult(submitted=True, draft=self.draft),
        )

    def action_cancel(self) -> None:
        self.dismiss(
            RunContextResult(submitted=False, draft=self.draft),
        )


__all__ = [
    "RunContextModal",
    "RunContextResult",
]
