"""DiffModal — side-by-side compare two saved agents
(TODO §1.1 P0.27).

Pushed by the `D` action on a 2-row LibraryScreen bulk
selection. Calls
:func:`care.runtime.agent_diff.fetch_agent_diff(memory, left,
right)` on mount and renders the result:

* Header: left vs. right labels.
* Per-step rows from `diff.steps` (status badge + step title).
* Metadata diff section above the rows.
* Footer shows `diff.format_summary()`.

The modal is pure presentation — no mutation. Dismiss with a
:class:`DiffResult` envelope so the host can record telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from care.runtime.agent_diff import (
    AgentDiff,
    FieldDiff,
    MetadataDiff,
    StepDiff,
    diff_chains,
    fetch_agent_diff,
)
from care.runtime.i18n import t
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader


@dataclass(frozen=True)
class DiffResult:
    """Dismiss envelope. ``cancelled`` is `True` for Escape /
    Close gestures so the host can distinguish them from a
    "viewed → moved on" gesture; the carried `AgentDiff` is
    available either way (or `None` if the fetch failed)."""

    diff: AgentDiff | None = None
    cancelled: bool = False


_BADGES: dict[str, str] = {
    "added": "+",
    "removed": "−",
    "modified": "~",
    "unchanged": "·",
}


class DiffModal(ModalScreen[DiffResult]):
    """Side-by-side diff modal.

    Construct with ``(left_entity_id, right_entity_id)`` +
    the `CareMemory`-like facade. On mount fires the fetch
    worker; results populate the per-step rows + metadata
    section + footer summary."""

    DEFAULT_CSS = """
    DiffModal {
        align: center middle;
    }
    DiffModal #diff-box {
        width: 110;
        max-width: 95%;
        height: 32;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    DiffModal #diff-title {
        text-style: bold;
        padding-bottom: 1;
    }
    DiffModal #diff-metadata {
        height: auto;
        margin-bottom: 1;
        color: $text-muted;
    }
    DiffModal #diff-steps {
        height: 1fr;
    }
    DiffModal #diff-footer {
        height: 1;
        color: $accent;
    }
    DiffModal #diff-actions {
        height: 3;
        align-horizontal: right;
    }
    DiffModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        left_entity_id: str = "",
        right_entity_id: str = "",
        *,
        memory: Any = None,
        left_label: str = "",
        right_label: str = "",
        left_payload: Any = None,
        right_payload: Any = None,
    ) -> None:
        """Construct a DiffModal.

        Two modes:

        * **Memory-backed**: pass ``left_entity_id`` +
          ``right_entity_id`` + ``memory`` — the modal fetches
          both chains via `client.get_chain_dict` on mount.
        * **Pre-loaded** (§3 P1, in-session artifact compare):
          pass ``left_payload`` + ``right_payload`` chain dicts
          directly — the modal skips the fetch and computes
          the diff synchronously via :func:`diff_chains`.
          Useful for comparing two
          :class:`SessionArtifact` payloads before they've
          been persisted to Memory.

        ``left_label`` / ``right_label`` are used in the
        header for both modes; in pre-loaded mode they default
        to the bare strings the caller supplies + are also
        echoed into the synthesised `AgentDiff.left_label` /
        `.right_label` slots.
        """
        super().__init__()
        self.left_entity_id = left_entity_id
        self.right_entity_id = right_entity_id
        self.left_label = left_label
        self.right_label = right_label
        self._memory = memory
        self._left_payload = left_payload
        self._right_payload = right_payload
        self.diff: AgentDiff | None = None
        self.load_error: str | None = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="diff-box"):
            yield CareHeader()
            yield Static(
                self._title(), id="diff-title",
            )
            yield Static(t("common.loading"), id="diff-metadata")
            yield VerticalScroll(id="diff-steps")
            yield Static("", id="diff-footer")
            with Horizontal(id="diff-actions"):
                yield Button(t("common.close"), id="diff-btn-close")
            yield CareFooter()

    def _title(self) -> str:
        left = self.left_label or self.left_entity_id
        right = self.right_label or self.right_entity_id
        return t("diff.title", left=left, right=right)

    def on_mount(self) -> None:
        # Pre-loaded mode wins: when both payloads are
        # supplied, skip the Memory fetch entirely and compute
        # synchronously. The §3 P1 in-session artifact compare
        # uses this path.
        if (
            self._left_payload is not None
            and self._right_payload is not None
        ):
            try:
                self.diff = diff_chains(
                    self._left_payload,
                    self._right_payload,
                    left_entity_id=self.left_entity_id,
                    right_entity_id=self.right_entity_id,
                    left_label=self.left_label,
                    right_label=self.right_label,
                )
            except Exception as exc:  # noqa: BLE001
                self.load_error = f"{type(exc).__name__}: {exc}"
            self._loaded = True
            self._render_panes()
            return
        if self._memory is None:
            self.load_error = "no memory facade configured"
            self._loaded = True
            self._render_panes()
            return
        self.run_worker(
            self._load(),
            name="diff_load",
            group="diff",
            exclusive=True,
            exit_on_error=False,
        )

    async def _load(self) -> None:
        try:
            self.diff = await fetch_agent_diff(
                self._memory,
                self.left_entity_id,
                self.right_entity_id,
                left_label=self.left_label,
                right_label=self.right_label,
            )
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._loaded = True
        self._render_panes()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render_panes(self) -> None:
        try:
            metadata = self.query_one("#diff-metadata", Static)
            steps = self.query_one("#diff-steps", VerticalScroll)
            footer = self.query_one("#diff-footer", Static)
        except Exception:
            return
        try:
            for child in list(steps.children):
                child.remove()
        except Exception:
            pass
        if self.load_error:
            metadata.update(f"⚠ {self.load_error}")
            footer.update("")
            return
        if self.diff is None:
            metadata.update(t("common.loading"))
            footer.update("")
            return
        metadata.update(self._format_metadata(self.diff.metadata))
        for row in self.diff.steps:
            steps.mount(Static(self._format_step_row(row)))
            for f in row.fields:
                steps.mount(Static(f"    {self._format_field(f)}"))
        footer.update(self.diff.format_summary())

    @staticmethod
    def _format_metadata(meta: MetadataDiff) -> str:
        if not meta.has_changes:
            return t("diff.metadataUnchanged")
        lines = []
        for f in meta.fields:
            lines.append(
                f"{f.field}: {f.left_value!r} → {f.right_value!r}"
            )
        if meta.added_tags:
            lines.append(f"+tags: {', '.join(meta.added_tags)}")
        if meta.removed_tags:
            lines.append(f"-tags: {', '.join(meta.removed_tags)}")
        return "  ·  ".join(lines)

    @staticmethod
    def _format_step_row(row: StepDiff) -> str:
        badge = _BADGES.get(row.kind, "?")
        title = row.label
        return f"{badge} step {row.number}: {title}"

    @staticmethod
    def _format_field(f: FieldDiff) -> str:
        return (
            f"{f.field}: {_truncate(f.left_value)} → "
            f"{_truncate(f.right_value)}"
        )

    # ------------------------------------------------------------------
    # Dismiss
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "diff-btn-close":
            self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(
            DiffResult(diff=self.diff, cancelled=True),
        )


def _truncate(value: Any, *, n: int = 40) -> str:
    s = "" if value is None else str(value)
    return s if len(s) <= n else s[: n - 1] + "…"


__all__ = ["DiffModal", "DiffResult"]
