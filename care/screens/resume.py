"""ResumeModal — prompt to resume an interrupted job
(TODO §1.1 P0.37).

Pushed by :class:`WelcomeScreen.on_mount` when
:func:`RunStateStore().load()` returns a non-``None``
snapshot. Renders the stored job summary plus two actions:

* `Resume` — dismiss with the original
  :class:`care.runtime.RunState`; the host re-primes from
  the payload.
* `Discard` — call :meth:`RunStateStore.clear()` then dismiss
  with ``None`` so the host proceeds with the normal boot.

The modal is a thin dispatcher — it doesn't try to re-run
the job itself. The host owns the per-kind resume policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from care.runtime.i18n import t
from care.runtime.run_state import RunState, RunStateStore


@dataclass(frozen=True)
class ResumeResult:
    """Dismiss envelope.

    ``action`` is one of:

    * ``"resume"`` — the host re-primes from ``state``.
    * ``"discard"`` — the store has been cleared; proceed with
      the normal boot flow.
    * ``"cancel"`` — Escape; treat like ``"discard"`` but
      *without* clearing the persisted state (the next launch
      will offer the same prompt again)."""

    action: str
    state: RunState | None


class ResumeModal(ModalScreen[ResumeResult]):
    """Two-button modal: Resume or Discard."""

    DEFAULT_CSS = """
    ResumeModal {
        align: center middle;
    }
    ResumeModal #resume-box {
        width: 70;
        max-width: 90%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    ResumeModal #resume-title {
        text-style: bold;
        padding-bottom: 1;
    }
    ResumeModal #resume-summary {
        color: $text-muted;
        padding-bottom: 1;
    }
    ResumeModal #resume-buttons {
        height: auto;
        align-horizontal: right;
    }
    ResumeModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        state: RunState,
        *,
        store: RunStateStore | None = None,
    ) -> None:
        super().__init__()
        self.state = state
        self._store = store or RunStateStore()

    def compose(self) -> ComposeResult:
        with Vertical(id="resume-box"):
            yield Label(t("resume.title"), id="resume-title")
            yield Static(
                self._summary_text(), id="resume-summary",
            )
            with Horizontal(id="resume-buttons"):
                yield Button(t("common.discard"), id="resume-btn-discard")
                yield Button(
                    t("resume.resume"),
                    id="resume-btn-resume",
                    variant="primary",
                )

    def on_mount(self) -> None:
        # Focus the Resume button so Enter accepts.
        try:
            self.query_one("#resume-btn-resume", Button).focus()
        except Exception:
            pass

    def _summary_text(self) -> str:
        ts = datetime.fromtimestamp(
            self.state.started_at, tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"{self.state.kind}: {self.state.label}\n"
            f"started {ts} · run_id {self.state.run_id[:12]}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "resume-btn-resume":
            self.action_resume()
        elif bid == "resume-btn-discard":
            self.action_discard()

    def action_resume(self) -> None:
        self.dismiss(
            ResumeResult(action="resume", state=self.state),
        )

    def action_discard(self) -> None:
        try:
            self._store.clear()
        except Exception:
            pass
        self.dismiss(
            ResumeResult(action="discard", state=None),
        )

    def action_cancel(self) -> None:
        # Escape closes the modal without clearing the store —
        # the next launch will offer the same prompt.
        self.dismiss(
            ResumeResult(action="cancel", state=self.state),
        )


__all__ = ["ResumeModal", "ResumeResult"]
