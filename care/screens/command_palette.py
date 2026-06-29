"""CommandPaletteModal — fuzzy palette over commands + library
(TODO §1.1 P0.25).

Bound to `Ctrl+P` from the global key bindings shipped in
P0.5. On mount fires
:func:`care.runtime.command_palette.fetch_palette_index` to
aggregate commands + library entries; every Input keystroke
re-runs :func:`search_palette` against the loaded index.
Selecting an entry dismisses the modal with a
:class:`PaletteSelection` envelope the host app uses to:

* dispatch the action via the canonical
  :data:`CommandActionId` (for `kind == "command"`), or
* push the destination screen (InspectionScreen / etc.) for
  entity entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from care.runtime.command_palette import (
    PaletteEntry,
    PaletteIndex,
    fetch_palette_index,
    search_palette,
)
from care.runtime.i18n import t
from care.screens._animated_modal import AnimatedModalScreen


@dataclass(frozen=True)
class PaletteSelection:
    """Dismiss envelope.

    ``entry`` is ``None`` when the user cancels (Escape or
    empty submit). Otherwise carries the picked
    :class:`PaletteEntry` verbatim so the host can route off
    `entry.command_action` / `entry.entry_id`."""

    entry: PaletteEntry | None


class CommandPaletteModal(AnimatedModalScreen[PaletteSelection]):
    """Fuzzy palette over commands + chains + agent skills.

    Construct with an optional `index`; otherwise on_mount
    fires `fetch_palette_index(app.memory)` and populates it
    lazily. The modal stays open until the user picks an
    entry or presses Escape."""

    DEFAULT_CSS = """
    CommandPaletteModal {
        align: center middle;
    }
    CommandPaletteModal #palette-box {
        width: 80;
        max-width: 90%;
        height: 20;
        padding: 0 1;
        border: thick $primary;
        background: $surface;
    }
    CommandPaletteModal #palette-input {
        margin-bottom: 1;
    }
    CommandPaletteModal OptionList {
        background: $surface;
        height: 1fr;
    }
    """

    ANIM_BOX_ID = "palette-box"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        *,
        index: PaletteIndex | None = None,
        memory: Any = None,
    ) -> None:
        super().__init__()
        self._memory = memory
        self.index: PaletteIndex = index or PaletteIndex()
        # Last rendered search results — exposed for tests +
        # future telemetry.
        self.results: tuple[PaletteEntry, ...] = ()
        # Last error from the aggregator (None on success).
        self.load_error: str | None = None
        # `True` once the on_mount aggregator settled.
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="palette-box"):
            yield Input(
                placeholder=t("commandPalette.searchPlaceholder"),
                id="palette-input",
            )
            yield OptionList(id="palette-list")

    def on_mount(self) -> None:
        try:
            self.query_one("#palette-input", Input).focus()
        except Exception:
            pass
        self._animate_modal_in()
        # Render whatever we have right now (commands sort
        # first so the empty-query case is useful even before
        # the aggregator returns).
        self._rerun_search("")
        if self._memory is not None:
            self.run_worker(
                self._load_index(),
                name="palette_load",
                group="palette",
                exclusive=True,
                exit_on_error=False,
            )

    async def _load_index(self) -> None:
        try:
            self.index = await fetch_palette_index(self._memory)
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._loaded = True
        self._rerun_search(self._current_query())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _current_query(self) -> str:
        if not self.is_mounted:
            return ""
        try:
            return self.query_one("#palette-input", Input).value
        except Exception:
            return ""

    def _rerun_search(self, query: str) -> None:
        self.results = search_palette(self.index, query)
        self._render_results()

    def _render_results(self) -> None:
        # Note: `self.is_mounted` is `False` while `on_mount` is
        # running on a ModalScreen even though children are
        # composed and queryable — fall through to a `query_one`
        # probe rather than gating on `is_mounted`.
        try:
            option_list = self.query_one("#palette-list", OptionList)
        except Exception:
            return
        try:
            option_list.clear_options()
        except Exception:
            pass
        for entry in self.results:
            option_list.add_option(self._render_entry(entry))

    @staticmethod
    def _render_entry(entry: PaletteEntry) -> Option:
        label = entry.label
        suffix_bits = []
        if entry.is_command:
            suffix_bits.append("⌘")
        elif entry.kind == "chain":
            suffix_bits.append(t("commandPalette.badgeChain"))
        elif entry.kind == "agent_skill":
            suffix_bits.append(t("commandPalette.badgeSkill"))
        if entry.description:
            suffix_bits.append(entry.description[:60])
        suffix = "  ·  ".join(suffix_bits)
        text = f"{label}  {suffix}" if suffix else label
        return Option(text, id=_palette_option_id(entry.entry_id))

    # ------------------------------------------------------------------
    # Field handlers
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "palette-input":
            return
        self._rerun_search(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "palette-input":
            return
        # Submit on the first entry; commands-first ordering
        # means an empty query + Enter picks the most useful
        # built-in.
        if not self.results:
            return
        self._dispatch(self.results[0])

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected,
    ) -> None:
        if event.option_list.id != "palette-list":
            return
        option_id = event.option.id
        if option_id is None:
            return
        entry = next(
            (
                e for e in self.results
                if _palette_option_id(e.entry_id) == option_id
            ),
            None,
        )
        if entry is None:
            return
        self._dispatch(entry)

    # ------------------------------------------------------------------
    # Dismiss
    # ------------------------------------------------------------------

    def _dispatch(self, entry: PaletteEntry) -> None:
        self.dismiss(PaletteSelection(entry=entry))

    def action_cancel(self) -> None:
        self.dismiss(PaletteSelection(entry=None))


def _palette_option_id(entry_id: str) -> str:
    """Project a :class:`PaletteEntry.entry_id` into a Textual
    OptionList-compatible id. Entry ids may contain colons
    (``command:create_new_agent``) which Textual rejects;
    replace them with underscores. Prefix with ``opt-`` so the
    leading char is always a letter."""
    cleaned = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in entry_id
    )
    return f"opt-{cleaned}"


__all__ = [
    "CommandPaletteModal",
    "PaletteSelection",
]
