"""EditAgentScreen — mutate a saved agent (TODO §1.1 P0.23).

Pushed when the user invokes `Edit` on a saved agent.
Wired entirely through the shipped
:mod:`care.runtime.edit_draft` data layer:

* :func:`extract_edit_draft(chain)` on construction — seeds
  the form from the saved chain.
* :func:`set_display_name` / :func:`set_description` /
  :func:`set_tags` / :func:`set_task_description` /
  :func:`set_change_summary` on each field edit.
* :func:`validate_edit_draft(draft)` runs on every change.
* :func:`save_edit_as_new_version(memory, draft)` runs on
  Save.
* :func:`promote_to_stable(memory, draft)` runs on Promote.

The Textual form is a thin renderer; the data layer owns the
actual mutation contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from care.runtime.i18n import t
from care.runtime.edit_draft import (
    EditAgentDraft,
    EditDraftIssue,
    PromoteResult,
    SaveEditResult,
    extract_edit_draft,
    promote_to_stable,
    save_edit_as_new_version,
    set_change_summary,
    set_description,
    set_display_name,
    set_tags,
    set_task_description,
    update_chain,
    validate_edit_draft,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


EditAction = Literal["save", "promote", "back"]


@dataclass(frozen=True)
class EditAgentEvent:
    """Envelope the screen posts on each terminal action so the
    host app can route to the next screen / refresh the
    library."""

    action: EditAction
    draft: EditAgentDraft
    save_result: SaveEditResult | None = None
    promote_result: PromoteResult | None = None


class EditAgentScreen(Screen):
    """Form for mutating a saved agent.

    Construct with a saved chain object + the
    `CareMemory`-like facade. Renders editable fields for
    display name / description / tags / task description /
    change summary, runs `validate_edit_draft` on every change,
    and exposes the three action buttons (Save / Promote /
    Back)."""

    DEFAULT_CSS = """
    EditAgentScreen {
        layout: vertical;
    }
    EditAgentScreen #edit-tabs {
        height: 1fr;
    }
    EditAgentScreen #edit-body {
        height: 1fr;
        padding: 1 2;
    }
    EditAgentScreen #edit-description {
        height: 4;
        margin-bottom: 1;
    }
    EditAgentScreen #edit-task {
        height: 4;
        margin-bottom: 1;
    }
    EditAgentScreen #edit-summary {
        height: 3;
        margin-bottom: 1;
    }
    EditAgentScreen #edit-issues {
        color: $warning;
        margin-bottom: 1;
    }
    EditAgentScreen #edit-content-body {
        height: 1fr;
        padding: 1 2;
    }
    EditAgentScreen #edit-content-hint {
        color: $text-muted;
        width: 1fr;
        height: auto;
        margin-bottom: 1;
    }
    EditAgentScreen #edit-content-json {
        height: 1fr;
        margin-bottom: 1;
    }
    EditAgentScreen #edit-content-error {
        color: $error;
    }
    EditAgentScreen #edit-actions {
        height: 3;
        padding: 0 1;
    }
    EditAgentScreen #edit-actions Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "save_edit", "Save", show=True),
        Binding("ctrl+p", "promote", "Promote", show=False),
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(
        self,
        chain: Any,
        *,
        memory: Any = None,
        entity_id: str | None = None,
        entity_type: Literal[
            "chain", "agent", "agent_skill"
        ] = "chain",
    ) -> None:
        super().__init__()
        self._memory = memory
        resolved_id = entity_id or self._read_entity_id(chain)
        self.draft: EditAgentDraft = extract_edit_draft(
            chain, resolved_id, entity_type=entity_type,
        )
        self.issues: tuple[EditDraftIssue, ...] = ()
        self.last_save: SaveEditResult | None = None
        self.last_promote: PromoteResult | None = None
        self._suppress_count: int = 0
        # Content-tab state. ``_original_content_obj`` is the JSON-
        # roundtripped snapshot of the saved chain content; the
        # Content tab compares the user's parsed edit against it so a
        # whitespace-only reformat (or the initial mount seed) doesn't
        # flag a structural edit. ``_content_error`` carries the live
        # JSON-parse error string (blocks Save while non-empty).
        self._original_content_obj: Any = self._content_to_obj(
            self.draft.chain_content,
        )
        self._content_error: str = ""

    @staticmethod
    def _read_entity_id(chain: Any) -> str:
        if chain is None:
            return ""
        if isinstance(chain, dict):
            return str(chain.get("entity_id") or chain.get("id") or "")
        return str(
            getattr(chain, "entity_id", None) or getattr(chain, "id", "") or ""
        )

    # ------------------------------------------------------------------
    # Chain-content (Content tab) serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _content_to_obj(content: Any) -> Any:
        """Project the draft's ``chain_content`` into a plain
        JSON-serialisable object (dict / list).

        Accepts the raw ``get_chain`` dict, a CARL ``ReasoningChain``
        (``to_dict`` / ``model_dump``), or ``None``. Falls back to an
        empty dict so the Content editor always has something valid to
        render."""
        if content is None:
            return {}
        if isinstance(content, (dict, list)):
            return content
        for attr in ("to_dict", "model_dump"):
            fn = getattr(content, attr, None)
            if callable(fn):
                try:
                    data = fn()
                except Exception:  # noqa: BLE001 — try the next projector
                    continue
                if isinstance(data, (dict, list)):
                    return data
        return {}

    def _content_text(self) -> str:
        """Pretty-printed JSON for the Content tab's editor."""
        try:
            return json.dumps(
                self._original_content_obj, indent=2, ensure_ascii=False,
            )
        except (TypeError, ValueError):
            return "{}"

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with TabbedContent(id="edit-tabs"):
            with TabPane(
                t("editAgent.tabContent"), id="edit-tab-content",
            ):
                with Vertical(id="edit-content-body"):
                    yield Static(
                        t("editAgent.contentHint"), id="edit-content-hint",
                    )
                    yield TextArea(
                        self._content_text(),
                        language="json",
                        show_line_numbers=True,
                        id="edit-content-json",
                    )
                    yield Static("", id="edit-content-error")
            with TabPane(
                t("editAgent.tabMetadata"), id="edit-tab-metadata",
            ):
                with VerticalScroll(id="edit-body"):
                    with Vertical():
                        yield Label(t("common.displayName"))
                        yield Input(
                            value=self.draft.display_name,
                            id="edit-display-name",
                        )
                        yield Label(t("common.description"))
                        yield TextArea(
                            self.draft.description,
                            id="edit-description",
                        )
                        yield Label(t("common.tags"))
                        yield Input(
                            value=", ".join(self.draft.tags),
                            placeholder=t("common.tagsPlaceholder"),
                            id="edit-tags",
                        )
                        yield Label(t("common.taskDescription"))
                        yield TextArea(
                            self.draft.task_description,
                            id="edit-task",
                        )
                        yield Label(t("editAgent.changeSummary"))
                        yield TextArea(
                            self.draft.change_summary,
                            id="edit-summary",
                        )
                        yield Static("", id="edit-issues")
        with Horizontal(id="edit-actions"):
            yield Button(t("common.back"), id="edit-btn-back")
            yield Button(
                t("editAgent.promoteStable"),
                id="edit-btn-promote",
            )
            yield Button(
                t("common.save"),
                id="edit-btn-save",
                variant="primary",
            )
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="EditAgentScreen",
                breadcrumb=(t("header.breadcrumb.library"), t("header.breadcrumb.edit")),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="EditAgentScreen",
                scope="screen",
            )
        except Exception:
            pass
        self._refresh_save_label()
        self._run_validate()

    # ------------------------------------------------------------------
    # Field edits
    # ------------------------------------------------------------------

    def _consume_suppression(self) -> bool:
        if self._suppress_count > 0:
            self._suppress_count -= 1
            return True
        return False

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._consume_suppression():
            return
        if event.input.id == "edit-display-name":
            self.draft = set_display_name(self.draft, event.value)
        elif event.input.id == "edit-tags":
            tags = [t.strip() for t in event.value.split(",") if t.strip()]
            self.draft = set_tags(self.draft, tags)
        else:
            return
        self._refresh_save_label()
        self._run_validate()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "edit-content-json":
            # Content tab owns its own dirty/validation handling and is
            # value-compared against the original snapshot, so it doesn't
            # ride the suppression counter (the mount seed re-parses to
            # the same object → clean).
            self._apply_content_edit(event.text_area.text)
            return
        if self._consume_suppression():
            return
        if event.text_area.id == "edit-description":
            self.draft = set_description(self.draft, event.text_area.text)
        elif event.text_area.id == "edit-task":
            self.draft = set_task_description(
                self.draft, event.text_area.text,
            )
        elif event.text_area.id == "edit-summary":
            self.draft = set_change_summary(
                self.draft, event.text_area.text,
            )
        else:
            return
        self._refresh_save_label()
        self._run_validate()

    def _apply_content_edit(self, text: str) -> None:
        """Parse the Content tab's JSON and fold it into the draft.

        Invalid JSON sets ``_content_error`` (which blocks Save) and
        leaves the draft's ``chain_content`` untouched. Valid JSON
        stamps the draft via :func:`update_chain`, flagging it dirty
        only when the parsed object differs from the saved snapshot so
        a no-op reformat doesn't demand a change summary."""
        stripped = text.strip()
        if not stripped:
            parsed: Any = {}
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                self._content_error = t(
                    "editAgent.contentInvalidJson",
                    error=f"{exc.msg} (line {exc.lineno}, col {exc.colno})",
                )
                self._render_content_error()
                self._refresh_save_label()
                return
        self._content_error = ""
        self._render_content_error()
        dirty = parsed != self._original_content_obj
        self.draft = update_chain(self.draft, parsed, dirty=dirty)
        self._refresh_save_label()
        self._run_validate()

    def _render_content_error(self) -> None:
        if not self.is_mounted:
            return
        try:
            target = self.query_one("#edit-content-error", Static)
        except Exception:
            return
        if self._content_error:
            target.update(self._content_error)
            return
        # No JSON-parse error — surface any blocking content-side
        # validation issue (e.g. "change summary required") here too.
        for issue in self.issues:
            if issue.field == "chain_content" and issue.severity == "error":
                target.update(f"⚠ {issue.message}")
                return
        target.update("")

    # ------------------------------------------------------------------
    # Validation + save label
    # ------------------------------------------------------------------

    def _run_validate(self) -> None:
        self.issues = validate_edit_draft(self.draft)
        self._render_issues()
        # Keep the Content tab's notice in sync so a content edit shows
        # *why* Save is blocked (e.g. change summary required) without
        # the user having to flip back to the Metadata tab to find out.
        self._render_content_error()

    def _render_issues(self) -> None:
        if not self.is_mounted:
            return
        try:
            target = self.query_one("#edit-issues", Static)
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
        if self._content_error:
            return True
        return any(i.severity == "error" for i in self.issues)

    def _refresh_save_label(self) -> None:
        if not self.is_mounted:
            return
        try:
            save = self.query_one("#edit-btn-save", Button)
        except Exception:
            return
        if self.draft.is_dirty():
            n = len(self.draft.dirty_fields())
            save.label = f"Save ({n} change{'s' if n != 1 else ''})"
        else:
            save.label = "Save"

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "edit-btn-save":
            self.action_save_edit()
        elif bid == "edit-btn-promote":
            self.action_promote()
        elif bid == "edit-btn-back":
            self.action_back()

    def action_save_edit(self) -> None:
        # A content edit shouldn't force the user to hand-write a change
        # summary before Save will do anything — that hidden requirement
        # was the "I clicked Save and nothing happened" trap. Auto-fill a
        # sensible default (the user can still override it) so editing the
        # JSON and hitting Save just works.
        self._autofill_change_summary()
        if self.has_blocking_issues:
            # Any *remaining* blocker (e.g. an empty display name) is real
            # — don't fail silently: navigate to the offending field and
            # tell the user what's missing.
            self._announce_blocked_save()
            return
        self.run_worker(
            self._save_worker(),
            name="edit_save",
            group="edit",
            exclusive=True,
            exit_on_error=False,
        )

    def _autofill_change_summary(self) -> None:
        """If the draft has a structural (content) edit but no change
        summary, stamp a default one so the change-summary requirement
        never silently blocks a content save. No-op when the user already
        wrote a summary or there's no structural edit."""
        if not self.draft.is_structural_edit:
            return
        if self.draft.change_summary.strip():
            return
        default = t("editAgent.defaultChangeSummary")
        self.draft = set_change_summary(self.draft, default)
        # Reflect it in the (Metadata-tab) summary field, suppressing the
        # echoed Changed event so it doesn't re-enter the handler.
        try:
            summary = self.query_one("#edit-summary", TextArea)
            if summary.text != default:
                self._suppress_count += 1
                summary.load_text(default)
        except Exception:
            pass
        self._run_validate()

    def _announce_blocked_save(self) -> None:
        """Surface why a Save was rejected: focus the offending field,
        switch to its tab, and notify. Covers the JSON-parse error
        (Content tab) and the inline validation errors (Metadata tab)."""
        if self._content_error:
            self._activate_tab("edit-tab-content")
            self._focus_widget("#edit-content-json")
            self._notify_blocked(self._content_error)
            return
        blockers = [i for i in self.issues if i.severity == "error"]
        if not blockers:
            return
        issue = blockers[0]
        if issue.field == "chain_content":
            # The only chain_content error is the change-summary
            # requirement — its field lives on the Metadata tab.
            self._activate_tab("edit-tab-metadata")
            self._focus_widget("#edit-summary")
        elif issue.field == "display_name":
            self._activate_tab("edit-tab-metadata")
            self._focus_widget("#edit-display-name")
        self._notify_blocked(
            t("editAgent.saveBlocked", reason=issue.message),
        )

    def _activate_tab(self, tab_id: str) -> None:
        try:
            self.query_one("#edit-tabs", TabbedContent).active = tab_id
        except Exception:
            pass

    def _focus_widget(self, selector: str) -> None:
        try:
            self.query_one(selector).focus()
        except Exception:
            pass

    def _notify_blocked(self, message: str) -> None:
        push = getattr(self.app, "push_toast", None)
        if callable(push):
            try:
                push(message, severity="warning")
                return
            except Exception:
                pass
        try:
            self.notify(message, severity="warning")
        except Exception:
            pass

    def action_promote(self) -> None:
        self.run_worker(
            self._promote_worker(),
            name="edit_promote",
            group="edit",
            exclusive=True,
            exit_on_error=False,
        )

    def action_back(self) -> None:
        self.post_message(self._envelope("back"))
        try:
            self.app.pop_screen()
        except Exception:
            pass

    async def _save_worker(self) -> None:
        if self._memory is None:
            self.last_save = None
            self.post_message(self._envelope("save"))
            return
        result = await save_edit_as_new_version(self._memory, self.draft)
        self.last_save = result
        if (
            result is not None
            and getattr(result, "success", False)
            and not getattr(result, "error", None)
        ):
            # Re-baseline the form so the screen visibly reflects the
            # save: the dirty markers clear and the Save label drops back
            # to "Save" instead of lingering on "(N changes)".
            self._commit_saved_state()
        self.post_message(self._envelope("save"))

    def _commit_saved_state(self) -> None:
        """Fold the just-saved edits into the draft's ``original_*``
        snapshots so :meth:`EditAgentDraft.is_dirty` reports clean."""
        self.draft = replace(
            self.draft,
            original_display_name=self.draft.display_name,
            original_description=self.draft.description,
            original_tags=self.draft.tags,
            original_task_description=self.draft.task_description,
            chain_content_dirty=False,
        )
        self._original_content_obj = self._content_to_obj(
            self.draft.chain_content,
        )
        self._refresh_save_label()
        self._run_validate()

    async def _promote_worker(self) -> None:
        if self._memory is None:
            self.last_promote = None
            self.post_message(self._envelope("promote"))
            return
        result = await promote_to_stable(self._memory, self.draft)
        self.last_promote = result
        self.post_message(self._envelope("promote"))

    def _envelope(self, action: EditAction) -> "EditAgentScreen.Submitted":
        return EditAgentScreen.Submitted(
            EditAgentEvent(
                action=action,
                draft=self.draft,
                save_result=self.last_save,
                promote_result=self.last_promote,
            ),
        )

    class Submitted(Message):
        """Posted after Save / Promote / Back completes. The host
        app reads `event.payload` (the frozen
        :class:`EditAgentEvent`) to route the next push."""

        def __init__(self, payload: EditAgentEvent) -> None:
            super().__init__()
            self.payload = payload


__all__ = [
    "EditAgentEvent",
    "EditAgentScreen",
]
