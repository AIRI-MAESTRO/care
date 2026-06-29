"""TagEditorModal — bulk tag editor (TODO §1.1 P0.28).

Pushed when the user presses `T` in LibraryScreen bulk-select
mode (P0.13). Composes the `add_tags` / `remove_tags` lists
that get passed to
:func:`care.runtime.bulk_ops.apply_tag_edits`.

The modal is intentionally minimal: two comma-separated Inputs
backed by :func:`merge_tags` for a live preview. Submitting
dismisses with :class:`TagEditorResult` carrying the cleaned
lists; the host screen fires the apply worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from care.runtime.bulk_ops import merge_tags
from care.runtime.i18n import t


@dataclass(frozen=True)
class TagEditorResult:
    """Dismiss envelope.

    ``submitted`` is `True` for an explicit "Apply" gesture
    and `False` for cancel; ``add_tags`` / ``remove_tags`` carry
    the cleaned lists ready to forward to
    :func:`care.runtime.apply_tag_edits`. Empty tuples on a
    cancel.

    ``title`` (§3 P3) carries the user-edited / accepted name
    when the modal was constructed with ``initial_title=…``
    (the ArtifactsScreen save-flow path). Empty string when
    the title field wasn't shown OR the user cleared it —
    callers fall back to whatever default they had previously.
    """

    submitted: bool = False
    add_tags: tuple[str, ...] = ()
    remove_tags: tuple[str, ...] = ()
    title: str = ""


class TagEditorModal(ModalScreen[TagEditorResult]):
    """Chip-style bulk tag editor.

    Construct with the union of currently-applied tags across
    the bulk selection (``initial_tags``); the modal renders
    them as a read-only chip list above two Inputs ("Add
    tags" / "Remove tags"). Submitting cleans both lists with
    :func:`merge_tags` semantics (whitespace stripped, empty
    skipped, dedup'd) before dismissing."""

    DEFAULT_CSS = """
    TagEditorModal {
        align: center middle;
    }
    TagEditorModal #tag-editor-box {
        width: 70;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    TagEditorModal #tag-editor-title {
        text-style: bold;
        padding-bottom: 1;
    }
    TagEditorModal #tag-editor-current {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }
    TagEditorModal #tag-editor-preview {
        height: auto;
        color: $accent;
        margin-bottom: 1;
    }
    TagEditorModal Input {
        margin-bottom: 1;
    }
    TagEditorModal #tag-editor-buttons {
        height: auto;
        align-horizontal: right;
    }
    TagEditorModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        *,
        initial_tags: Iterable[str] = (),
        target_count: int = 0,
        initial_title: str = "",
    ) -> None:
        super().__init__()
        self.initial_tags: tuple[str, ...] = tuple(
            t for t in (s.strip() for s in initial_tags) if t
        )
        self.target_count = target_count
        # §3 P3 — when non-empty, the modal renders an Input
        # for the chain name above the tag rows. The
        # ArtifactsScreen save-flow pre-computes the
        # LLM-suggested title (via `chain_title.suggest_chain_title`)
        # and seeds the input so the user can accept / tweak
        # / clear before persisting. Empty string keeps the
        # legacy compact form (just tag inputs).
        self.initial_title: str = str(initial_title).strip()

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="tag-editor-box"):
            yield Label(
                self._title_text(), id="tag-editor-title",
            )
            # §3 P3 — chain-name field. Only rendered when the
            # caller seeded an `initial_title`, so the
            # bulk-tag-edit path (LibraryScreen) keeps its
            # compact tag-only form.
            if self.initial_title:
                yield Label(t("tagEditor.name"))
                yield Input(
                    value=self.initial_title,
                    placeholder=t("tagEditor.namePlaceholder"),
                    id="tag-editor-name",
                )
            yield Static(
                self._current_text(), id="tag-editor-current",
            )
            yield Label(t("tagEditor.addTags"))
            yield Input(
                placeholder=t("tagEditor.addPlaceholder"),
                id="tag-editor-add",
            )
            yield Label(t("tagEditor.removeTags"))
            yield Input(
                placeholder=t("tagEditor.removePlaceholder"),
                id="tag-editor-remove",
            )
            yield Static("", id="tag-editor-preview")
            with Horizontal(id="tag-editor-buttons"):
                yield Button(t("common.cancel"), id="tag-editor-cancel")
                yield Button(
                    t("tagEditor.apply"),
                    id="tag-editor-apply",
                    variant="primary",
                )

    def on_mount(self) -> None:
        self._render_preview()

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _title_text(self) -> str:
        if self.target_count:
            return f"Edit tags · {self.target_count} row{'s' if self.target_count != 1 else ''}"
        return "Edit tags"

    def _current_text(self) -> str:
        if not self.initial_tags:
            return "(no tags currently applied)"
        return "Current: " + ", ".join(self.initial_tags)

    def _read_input(self, selector: str) -> list[str]:
        try:
            widget = self.query_one(selector, Input)
        except Exception:
            return []
        return [
            t.strip() for t in (widget.value or "").split(",") if t.strip()
        ]

    @property
    def add_tags(self) -> tuple[str, ...]:
        return tuple(self._read_input("#tag-editor-add"))

    @property
    def remove_tags(self) -> tuple[str, ...]:
        return tuple(self._read_input("#tag-editor-remove"))

    @property
    def current_title(self) -> str:
        """§3 P3 — return the name-Input's current value (post-
        any user edits), or empty string when the title field
        wasn't rendered. Falls back to ``initial_title`` if the
        widget exists but has been cleared so the caller can
        decide whether to treat empty input as "use the
        suggested default" or "respect the user's explicit
        clear"."""
        if not self.initial_title:
            return ""
        try:
            widget = self.query_one("#tag-editor-name", Input)
        except Exception:
            return self.initial_title
        return (widget.value or "").strip()

    def _preview_tags(self) -> tuple[str, ...]:
        """Show the user what the selection's tag set will
        look like after the apply. Pure projection through
        :func:`merge_tags`."""
        return tuple(
            merge_tags(
                self.initial_tags,
                add=self.add_tags,
                remove=self.remove_tags,
            )
        )

    # ------------------------------------------------------------------
    # Field handlers
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id not in ("tag-editor-add", "tag-editor-remove"):
            return
        self._render_preview()

    def _render_preview(self) -> None:
        try:
            target = self.query_one("#tag-editor-preview", Static)
        except Exception:
            return
        if not (self.add_tags or self.remove_tags):
            target.update("")
            return
        target.update(
            "Result: " + ", ".join(self._preview_tags() or ["(no tags)"]),
        )

    # ------------------------------------------------------------------
    # Dismiss
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "tag-editor-cancel":
            self.action_cancel()
        elif bid == "tag-editor-apply":
            self.action_apply()

    def action_apply(self) -> None:
        adds = self.add_tags
        removes = self.remove_tags
        title = self.current_title
        # §3 P3 — when the title field is rendered, a name edit
        # alone counts as a meaningful apply (so the user can
        # accept the suggestion without typing any tag input).
        # The save flow always benefits from running apply
        # because it carries the name; bulk-tag-edit paths
        # without a title still drop the no-op cancel.
        has_title_field = bool(self.initial_title)
        if not adds and not removes and not has_title_field:
            # Nothing to do — dismiss as cancel rather than
            # firing a no-op apply.
            self.action_cancel()
            return
        self.dismiss(
            TagEditorResult(
                submitted=True,
                add_tags=adds,
                remove_tags=removes,
                title=title,
            ),
        )

    def action_cancel(self) -> None:
        self.dismiss(TagEditorResult(submitted=False))


# Re-export `field` import so future dataclass extensions don't
# need a separate import.
_ = field


__all__ = [
    "TagEditorModal",
    "TagEditorResult",
]
