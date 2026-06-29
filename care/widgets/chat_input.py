"""ChatInput — multi-line wrapping prompt with Input-compat surface.

Textual's :class:`~textual.widgets.Input` is single-line by design
— long prompts scroll horizontally and there's no native way to
soft-wrap. The chat surface needs a single-row default that
auto-grows up to four rows as the user types longer prompts and
then scrolls vertically beyond that, so we ride on top of
:class:`~textual.widgets.TextArea` instead.

The widget exposes the bits of :class:`Input`'s public surface
that the chat code + 263 existing test sites already depend on,
so callers don't have to learn a new API:

* ``.value`` (str) — alias for :attr:`TextArea.text`. Read +
  write. Setting clears the buffer and inserts ``value`` so the
  cursor lands at the end (matches Input's behaviour).
* ``.cursor_position`` (int) — linear character offset into the
  buffer. Read + write. Internally maps to TextArea's
  ``(row, column)`` tuple.
* ``action_submit()`` — posts :class:`Input.Submitted` so the
  existing ``on_input_submitted`` handlers fire unchanged.
  Bound to Enter; ``action_newline`` (Shift+Enter) inserts a
  soft newline instead, where the terminal can tell the two
  chords apart (see the BINDINGS note).

The widget also bridges :class:`TextArea.Changed` →
:class:`Input.Changed` so the slash-autocomplete + transcript
search keep working.
"""

from __future__ import annotations

from typing import ClassVar

from textual import events, on
from textual.binding import Binding, BindingType
from textual.widgets import Input, TextArea


class ChatInput(TextArea):
    """Multi-line wrapping chat prompt with Input-compat shims.

    Defaults: ``compact=True``, ``soft_wrap=True``, no line
    numbers, no syntax highlighting, ``tab_behavior="focus"`` so
    Tab still walks the focus chain (and the chat-level
    ``action_slash_autocomplete`` binding can claim it).
    Heights are clamped to ``[1, 4]`` rows via CSS so the strip
    auto-grows to fit the wrapped content and scrolls vertically
    once the user has typed more than four visual rows.
    """

    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        min-height: 1;
        max-height: 4;
        border: none;
        padding: 0;
        /* Slim vertical scrollbar so users see the
           affordance once the wrapped prompt exceeds the
           4-row cap. Horizontal scroll stays hidden — the
           widget soft-wraps so there's never anything to
           scroll horizontally. */
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
    }
    ChatInput:focus {
        border: none;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        # Enter submits; Shift+Enter inserts a soft newline. Both rely on the
        # terminal reporting them as DISTINCT chords. A terminal that sends a
        # bare CR for both (e.g. macOS Terminal.app — no Kitty keyboard
        # protocol) can't tell them apart, so there Shift+Enter is
        # indistinguishable from Enter and still submits. iTerm2 / Ghostty /
        # Kitty / WezTerm report `shift+enter` separately, so it works there.
        Binding(
            "enter",
            "submit",
            "Submit",
            show=False,
            priority=True,
        ),
        Binding(
            "shift+enter",
            "newline",
            "New line",
            show=False,
            priority=True,
        ),
    ]

    def __init__(
        self,
        *,
        placeholder: str = "",
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(
            text="",
            placeholder=placeholder,
            id=id,
            classes=classes,
            compact=True,
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
        )
        # Suppress the in-flight bridge during the
        # `.value = "..."` setter so a programmatic refill
        # doesn't fire a redundant `Input.Changed` (TextArea's
        # own `Changed` already fires; the bridge would
        # double-deliver to consumers that listen to both).
        # Bookkeeping only — read/written by the property
        # setter and the bridge handler.
        self._suppress_changed_bridge = False

    # ------------------------------------------------------------------
    # Input-compatibility surface
    # ------------------------------------------------------------------

    @property
    def value(self) -> str:
        """Alias for :attr:`TextArea.text`. Read returns the
        full multi-line buffer; write replaces it wholesale."""
        return self.text

    @value.setter
    def value(self, new_value: str) -> None:
        # Match Input's "set the whole thing, cursor at end"
        # semantics so existing test code (``inp.value = "/help"``)
        # behaves identically.
        self._suppress_changed_bridge = True
        try:
            self.text = new_value or ""
        finally:
            self._suppress_changed_bridge = False
        # Park the cursor at the end of the new buffer.
        self.cursor_position = len(self.text)
        # Hand-fire the bridge once so consumers see exactly one
        # Input.Changed per assignment (the same contract Input
        # offers via its reactive watcher).
        self.post_message(Input.Changed(input=self, value=self.text))

    @property
    def cursor_position(self) -> int:
        """Linear character offset into the buffer. Matches
        :attr:`Input.cursor_position` semantics so test code
        like ``inp.cursor_position = 4`` keeps working."""
        try:
            row, col = self.cursor_location
        except Exception:
            return 0
        return self._location_to_offset(row, col)

    @cursor_position.setter
    def cursor_position(self, offset: int) -> None:
        text = self.text
        if offset < 0:
            offset = 0
        elif offset > len(text):
            offset = len(text)
        row, col = self._offset_to_location(offset)
        try:
            self.cursor_location = (row, col)
        except Exception:
            # Pre-mount writes can race the TextArea internals;
            # silently swallow so the caller's set-then-mount
            # flow stays robust.
            pass

    def _location_to_offset(self, row: int, col: int) -> int:
        """Convert a ``(row, column)`` TextArea coordinate into
        a linear offset that mirrors what ``Input.cursor_position``
        would report for the same effective caret state."""
        lines = self.text.split("\n")
        if row >= len(lines):
            row = len(lines) - 1
            col = len(lines[row]) if lines else 0
        # Sum the lengths of all prior lines + their newline
        # characters, then add the in-line column.
        prefix = sum(len(lines[i]) + 1 for i in range(row))
        return prefix + min(col, len(lines[row]) if lines else 0)

    def _offset_to_location(self, offset: int) -> tuple[int, int]:
        """Reverse of :meth:`_location_to_offset`."""
        text = self.text
        if offset <= 0:
            return (0, 0)
        if offset >= len(text):
            lines = text.split("\n")
            return (len(lines) - 1, len(lines[-1]))
        row = 0
        col = 0
        for ch in text[:offset]:
            if ch == "\n":
                row += 1
                col = 0
            else:
                col += 1
        return (row, col)

    # ------------------------------------------------------------------
    # Event bridges
    # ------------------------------------------------------------------

    async def action_submit(self) -> None:
        """Post :class:`Input.Submitted` so existing
        ``on_input_submitted`` handlers keep firing on Enter.
        TextArea has no built-in "submit" gesture — we provide
        it via a screen-level Enter binding that takes priority
        over the default newline-insertion."""
        self.post_message(
            Input.Submitted(input=self, value=self.text),
        )

    def action_newline(self) -> None:
        """Insert a soft newline at the cursor (Shift+Enter), so a bare
        Enter can stay bound to submit. Only reachable when the terminal
        delivers ``shift+enter`` as a distinct chord (Kitty keyboard
        protocol); otherwise the ``enter`` binding wins and submits."""
        self.insert("\n")

    @on(TextArea.Changed)
    def _bridge_text_area_changed(
        self, event: TextArea.Changed,
    ) -> None:
        """Translate :class:`TextArea.Changed` →
        :class:`Input.Changed` so the slash-autocomplete +
        transcript-search handlers (``on_input_changed``) keep
        working without code rewrites in the chat screen."""
        if self._suppress_changed_bridge:
            return
        # TextArea.Changed isn't stopped here — consumers that
        # explicitly handle the TextArea event still get it.
        self.post_message(
            Input.Changed(input=self, value=self.text),
        )

    async def _on_key(self, event: events.Key) -> None:
        """Route Up / Down to the autocomplete popup when it's
        open instead of letting TextArea consume them for caret
        navigation.

        TextArea's built-in `up` / `down` bindings move the
        caret between visual rows; they fire BEFORE any
        screen-level binding because Textual dispatches focused-
        widget bindings first. The chat surface has ``up`` /
        ``down`` mapped to ``recall_prev`` / ``recall_next``,
        which in turn redirects to ``_move_autocomplete_selection``
        when the popup is open — but the screen handler never
        sees the key because the TextArea already ate it.

        We close that gap here: when the parent screen has the
        popup open with at least one match, swallow the key at
        the widget level and call the screen's nav action.
        Otherwise we let TextArea's default handling proceed,
        which preserves cursor movement inside the buffer and
        lets the screen's history-recall binding fire when
        nothing is intercepting.
        """
        if event.key in ("up", "down"):
            if self._autocomplete_popup_active():
                screen = self.screen
                action = (
                    "recall_prev"
                    if event.key == "up"
                    else "recall_next"
                )
                event.stop()
                event.prevent_default()
                try:
                    await screen.run_action(action)
                except Exception:
                    # `run_action` failures shouldn't crash key
                    # dispatch — fall back to a direct call when
                    # the screen exposes the handler.
                    handler = getattr(screen, f"action_{action}", None)
                    if handler is not None:
                        try:
                            handler()
                        except Exception:
                            pass
                return
        await super()._on_key(event)

    def _autocomplete_popup_active(self) -> bool:
        """Inspect the parent screen for an open autocomplete
        popup with at least one match. Returns ``False`` when
        the screen lacks the chat-specific attributes (e.g.
        when the widget is mounted in a non-ChatScreen host
        during tests)."""
        screen = self.screen
        is_open = getattr(screen, "_autocomplete_open", False)
        matches = getattr(screen, "_autocomplete_matches", None) or []
        return bool(is_open and matches)


__all__ = ["ChatInput"]
