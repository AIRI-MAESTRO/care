"""SaveAgentModal — promote a draft into a saved agent
(TODO §1.1 P0.18).

Pushed on top of `GenerationScreen` when MAGE finishes a run.
Bound to the shipped :mod:`care.runtime.save_agent_form` data
layer:

* `seed_save_agent_form(...)` on mount — pre-fills the form
  from the MAGE result.
* `set_display_name` / `set_description` / `set_tags` /
  `toggle_favourite` / `set_keep_context` on each field edit.
* `validate_save_agent_form(...)` runs on every change (with
  a short debounce so the user can keep typing).
* `apply_save_agent_form(...)` runs on submit. The three
  shipped action buttons (`Save & Inspect`, `Save & Run`,
  `Discard`) all dismiss the modal with a typed action
  envelope so the host screen routes the next push.

This widget is wired entirely through the data layer; the
modal does not call HTTP itself, doesn't import MAGE, and
treats `memory` + `session` as opaque facades. Tests inject
stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Label, Static, TextArea

from care.runtime.i18n import t
from care.runtime.save_agent_form import (
    SaveAgentForm,
    SaveAgentIssue,
    SaveAgentOutcome,
    apply_save_agent_form,
    seed_save_agent_form,
    set_description,
    set_display_name,
    set_keep_context,
    set_tags,
    toggle_favourite,
    validate_save_agent_form,
)
from care.screens._animated_modal import AnimatedModalScreen


SaveAgentAction = Literal["save_inspect", "save_run", "discard"]
"""Three shipped button actions per the §3 action mapping table."""


@dataclass(frozen=True)
class SaveAgentResult:
    """Dismiss envelope for the modal.

    Carries the action the user picked plus the post-apply
    outcome (for the save actions) so the host screen can
    route the next push without re-running the data layer.
    """

    action: SaveAgentAction
    form: SaveAgentForm
    outcome: SaveAgentOutcome | None = None


class SaveAgentModal(AnimatedModalScreen[SaveAgentResult]):
    """Modal that promotes a draft into a saved agent.

    Construct with the `CareMemory`-like facade + the
    :class:`care.runtime.draft.DraftSession` returned by the
    upstream auto-save worker, plus the optional MAGE-result
    seed (`query`, `mage_metadata`, `context_files`,
    `suggested_name_override`).
    """

    DEFAULT_CSS = """
    SaveAgentModal {
        align: center middle;
    }
    SaveAgentModal #save-agent-box {
        width: 70;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    SaveAgentModal #save-agent-title {
        text-style: bold;
        padding-bottom: 1;
    }
    SaveAgentModal #save-agent-description {
        height: 5;
        margin-bottom: 1;
    }
    SaveAgentModal #save-agent-issues {
        color: $warning;
        margin-bottom: 1;
    }
    SaveAgentModal #save-agent-buttons {
        height: auto;
        align-horizontal: right;
    }
    SaveAgentModal Button {
        margin-left: 1;
    }
    """

    ANIM_BOX_ID = "save-agent-box"

    BINDINGS = [
        Binding("escape", "discard", "Discard", show=False),
    ]

    def __init__(
        self,
        *,
        memory: Any = None,
        session: Any = None,
        query: str = "",
        mage_metadata: Any = None,
        context_files: Any = None,
        suggested_name_override: str | None = None,
    ) -> None:
        super().__init__()
        self._memory = memory
        self._session = session
        self.form: SaveAgentForm = seed_save_agent_form(
            query=query,
            mage_metadata=mage_metadata,
            context_files=context_files,
            suggested_name_override=suggested_name_override,
        )
        # Latest validation result (debounced).
        self.issues: tuple[SaveAgentIssue, ...] = ()
        # Last submission outcome — exposed for tests +
        # post-apply telemetry.
        self.last_outcome: SaveAgentOutcome | None = None
        # Timer handle for the validate debounce. Stored so a
        # rapid sequence of keystrokes coalesces to one
        # validate call.
        self._validate_timer = None
        # Re-entry guard for set_*() programmatic widget
        # writes (mirrors LibrarySidebar's _suppress_count
        # pattern).
        self._suppress_count = 0

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="save-agent-box"):
            yield Label(t("saveAgent.title"), id="save-agent-title")
            yield Label(t("common.displayName"))
            yield Input(
                value=self.form.display_name,
                placeholder=t("saveAgent.namePlaceholder"),
                id="save-agent-display-name",
            )
            yield Label(t("common.description"))
            yield TextArea(
                self.form.description,
                id="save-agent-description",
            )
            yield Label(t("common.tags"))
            yield Input(
                value=", ".join(self.form.tags),
                placeholder=t("common.tagsPlaceholder"),
                id="save-agent-tags",
            )
            yield Checkbox(
                t("saveAgent.favourite"),
                value=self.form.favourite,
                id="save-agent-favourite",
            )
            yield Checkbox(
                t("saveAgent.keepContext"),
                value=self.form.keep_context,
                id="save-agent-keep-context",
            )
            yield Static("", id="save-agent-issues")
            with Horizontal(id="save-agent-buttons"):
                yield Button(t("common.discard"), id="save-agent-discard")
                yield Button(
                    t("saveAgent.saveInspect"),
                    id="save-agent-save-inspect",
                    variant="primary",
                )
                yield Button(
                    t("saveAgent.saveRun"),
                    id="save-agent-save-run",
                    variant="success",
                )

    def on_mount(self) -> None:
        self._animate_modal_in()
        # Kick a first validation pass so any pre-existing
        # issues (empty name etc.) surface immediately.
        self._schedule_validate(delay=0.0)

    # ------------------------------------------------------------------
    # Field handlers
    # ------------------------------------------------------------------

    def _consume_suppression(self) -> bool:
        if self._suppress_count > 0:
            self._suppress_count -= 1
            return True
        return False

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._consume_suppression():
            return
        if event.input.id == "save-agent-display-name":
            self.form = set_display_name(self.form, event.value)
        elif event.input.id == "save-agent-tags":
            tags = [t.strip() for t in event.value.split(",") if t.strip()]
            self.form = set_tags(self.form, tags)
        else:
            return
        self._schedule_validate()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "save-agent-description":
            return
        if self._consume_suppression():
            return
        self.form = set_description(self.form, event.text_area.text)
        self._schedule_validate()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if self._consume_suppression():
            return
        if event.checkbox.id == "save-agent-favourite":
            current = self.form.favourite
            if current != event.value:
                self.form = toggle_favourite(self.form)
        elif event.checkbox.id == "save-agent-keep-context":
            self.form = set_keep_context(self.form, event.value)
        else:
            return
        self._schedule_validate()

    # ------------------------------------------------------------------
    # Validation (debounced)
    # ------------------------------------------------------------------

    _VALIDATE_DEBOUNCE_SEC = 0.25

    def _schedule_validate(self, *, delay: float | None = None) -> None:
        if not self.is_mounted:
            return
        if self._validate_timer is not None:
            try:
                self._validate_timer.stop()
            except Exception:
                pass
        wait = self._VALIDATE_DEBOUNCE_SEC if delay is None else delay
        if wait <= 0:
            self._run_validate()
            return
        self._validate_timer = self.set_timer(wait, self._run_validate)

    def _run_validate(self) -> None:
        self.run_worker(
            self._validate_form(),
            name="save_agent_validate",
            group="save_agent",
            exclusive=True,
            exit_on_error=False,
        )

    async def _validate_form(self) -> None:
        issues = await validate_save_agent_form(
            self.form, self._memory,
        )
        self.issues = issues
        self._render_issues()

    def _render_issues(self) -> None:
        if not self.is_mounted:
            return
        try:
            target = self.query_one("#save-agent-issues", Static)
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

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "save-agent-discard":
            self.action_discard()
        elif bid == "save-agent-save-inspect":
            self._submit("save_inspect")
        elif bid == "save-agent-save-run":
            self._submit("save_run")

    def action_discard(self) -> None:
        """Dismiss with the discard envelope. The host screen
        is responsible for calling `discard_draft` so this
        modal stays free of side effects (mirrors the
        ConfirmModal contract)."""
        self.dismiss(
            SaveAgentResult(action="discard", form=self.form, outcome=None),
        )

    def _submit(self, action: SaveAgentAction) -> None:
        self.run_worker(
            self._apply_and_dismiss(action),
            name="save_agent_apply",
            group="save_agent_apply",
            exclusive=True,
            exit_on_error=False,
        )

    async def _apply_and_dismiss(self, action: SaveAgentAction) -> None:
        if self._memory is None or self._session is None:
            # No facade wired (rare; mostly for direct tests
            # that don't intend to call apply). Dismiss with
            # the action verb so the host can still route.
            self.last_outcome = None
            self.dismiss(
                SaveAgentResult(action=action, form=self.form, outcome=None),
            )
            return
        outcome = await apply_save_agent_form(
            self._memory, self._session, self.form,
        )
        self.last_outcome = outcome
        self.dismiss(
            SaveAgentResult(
                action=action, form=self.form, outcome=outcome,
            ),
        )


__all__ = [
    "SaveAgentAction",
    "SaveAgentModal",
    "SaveAgentResult",
]
