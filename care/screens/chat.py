"""ChatScreen — Claude-Code-style primary surface.

A scrollable transcript pinned above a bottom prompt. Plain
prose messages drive MAGE generation; slash commands cover
the non-chat affordances (library / settings / clear / quit
/ etc.).

Why a chat surface:

* Single entry-point: users describe the task in natural
  language; CARE figures out whether to generate, run, or
  show a screen. Same gesture every time.
* Scrollable history: every previous interaction stays in
  view — past tasks, generated chains, stage events.
* Slash commands keep the legacy screens reachable. ChatScreen
  doesn't replace `LibraryScreen` / `SettingsScreen`; it
  surfaces them via `/library` / `/settings`.

The screen is self-contained — it does NOT route through the
existing `QueryScreen.GenerateRequested` → `App` flow. Instead
it spawns a MAGE worker locally with a :class:`MagePoster`
pointed at itself, so stage events come back as transcript
lines.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, Literal

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Input,
    Markdown,
    RadioButton,
    RadioSet,
    Select,
    Static,
)
from textual.worker import Worker, WorkerState

from care.config import ChatConfig, StagePolicy
from care.runtime.agent_hub import (
    HubClient,
    HubError,
    HubUnavailableError,
    ensure_hub_running,
    hub_env,
)
from care.runtime.i18n import get_ui_language, t
from care.runtime.hint_fit import fit_line, fit_segments
from care.runtime.open_url import open_url
from care.runtime.promote_gate import gate_promotion
from care.runtime.carl_streamer import (
    CarlStreamer,
    ChainCompleted,
    HumanInputRequested,
    LlmChunk as CarlLlmChunk,
    Progress as CarlProgress,
    StepCompleted,
    StepEvent as CarlStepEvent,
    StepStarted,
)
from care.runtime.deploy_gate import gate_chain_for_deploy
from care.runtime.mage_poster import (
    StageCompleted,
    StageError,
    StageProgress,
    StageRetry,
    StageStarted,
)
from care.widgets.chat_input import ChatInput
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader
from care.widgets.status_bar import StatusBar

_log = logging.getLogger("care.chat")

_CHAT_INPUT_HINT_KEYS = ("enter", "atFile", "library", "esc")
_AGENT_INPUT_HINT_KEYS = ("enter", "atFile", "dataset", "esc")


def _make_halfblock_renderer() -> Any:
    """Build a half-block renderer that fixes rich_pixels'
    asymmetric-cell bug.

    The stock ``HalfcellRenderer`` always emits ``▄`` (lower
    half block) with the lower pixel as the foreground and
    the upper pixel as the background. That works when both
    pixels are opaque, but when ONLY the upper pixel is
    opaque it falls back to ``" "`` with ``bg = upper``,
    which paints the full cell — visually smearing the
    single source pixel into TWO terminal rows. Switching
    to ``▀`` (upper half block) for upper-only cells
    restores the half-cell-per-pixel mapping.
    """
    from rich.segment import Segment
    from rich.style import Style
    from rich_pixels import HalfcellRenderer
    from rich_pixels._renderer import _get_color

    class _AsymmetricHalfblock(HalfcellRenderer):
        def _render_halfcell(self, *, x, y, get_pixel):  # type: ignore[override]
            upper = _get_color(get_pixel((x, y)), default_color=self.default_color)
            lower = _get_color(
                get_pixel((x, y + 1)), default_color=self.default_color,
            )
            if upper and lower:
                return Segment("▄", Style.parse(f"{lower} on {upper}"))
            if lower:
                return Segment("▄", Style.parse(lower))
            if upper:
                return Segment("▀", Style.parse(upper))
            return Segment(" ", self.null_style)

    return _AsymmetricHalfblock()


_HalfblockAsymmetricRenderer = _make_halfblock_renderer


async def _maybe_await(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn(*args, **kwargs)`` whether it's sync or async.

    Memory facades land sync methods on the typed protocol but the
    SDK occasionally exposes ``async`` variants in newer
    versions. Worker helpers that want to interop with both shapes
    use this to avoid hard-coding one calling convention.
    """
    import inspect

    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


_SEVERITY_TO_LEVEL: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


ChatRole = Literal["user", "assistant", "system", "tool"]


ChatMode = Literal["interactive", "production"]
"""Two operating modes the chat surface drives.

* ``interactive`` — MAGE generates a chain, CARL runs it on the spot,
  the assistant analyses the result and may loop (ReAct). Persistence
  is opt-in. (Renamed from the legacy ``ad_hoc`` literal — see
  :func:`normalise_mode` for the read-side alias.)
* ``production`` — MAGE generates a reproducible chain, CARE saves
  it to Memory, seeds a dataset card with a baseline run, and
  optionally fires a Platform evolution. The user copies the
  ``chain_id`` and wires it into their own service.

Default is driven by ``CARE_CHAT__DEFAULT_MODE``; the default is
``"interactive"`` so first-time users get the low-friction path.
"""

# Legacy mode values that must still deserialize (saved sessions,
# saved agents, persisted config, CLI input, telemetry replay). Only
# the canonical right-hand value is ever *written*.
_LEGACY_MODE_ALIASES: dict[str, str] = {
    "ad_hoc": "interactive",
    "ad-hoc": "interactive",
    "adhoc": "interactive",
}


def normalise_mode(value: str) -> ChatMode:
    """Map any externally-sourced mode string onto a canonical
    :data:`ChatMode`.

    Translates the legacy ``ad_hoc`` spellings to ``interactive`` and
    passes ``production`` (and an already-canonical ``interactive``)
    through unchanged. Unknown values are returned as-is (cast) so the
    caller decides whether to fall back — keeping this reader total.
    """
    key = (value or "").strip().lower()
    return _LEGACY_MODE_ALIASES.get(key, key)  # type: ignore[return-value]


DEFAULT_CHAT_MODE: ChatMode = "interactive"
"""Project-wide fallback when nothing else picks a mode."""


def _resolve_default_mode() -> ChatMode:
    """Read ``CARE_CHAT__DEFAULT_MODE`` from the environment.

    Unknown values fall back to :data:`DEFAULT_CHAT_MODE` rather than
    raising — the chat surface must boot even with a malformed env.
    """
    import os

    raw = normalise_mode(os.environ.get("CARE_CHAT__DEFAULT_MODE") or "")
    if raw == "interactive":
        return "interactive"
    if raw in ("production", "prod"):
        return "production"
    return DEFAULT_CHAT_MODE


# ---------------------------------------------------------------------------
# Modes redesign — pipeline stage model + per-mode presets
#
# Both modes run one pipeline (GENERATE → PREVIEW → RUN → SAVE → BASELINE →
# EVOLVE); they differ only in each stage's *policy*. GENERATE/PREVIEW are
# always auto (that's how the chain + schema come into existence), so only
# RUN/SAVE/BASELINE/EVOLVE are policy-configurable. See PLAN_MODES.md.
# ---------------------------------------------------------------------------


class Stage(StrEnum):
    """The ordered pipeline stages."""

    GENERATE = "generate"
    PREVIEW = "preview"
    RUN = "run"
    SAVE = "save"
    BASELINE = "baseline"
    EVOLVE = "evolve"


class StageOutcome(StrEnum):
    """Resolved result of a stage at runtime."""

    DONE = "done"        # executed successfully
    SKIPPED = "skipped"  # policy=skip, dependency unmet, or user declined
    FAILED = "failed"    # executed but raised


FollowupKind = Literal["reuse", "revise"]
"""How a mode treats follow-up prompts after the first chain.

* ``reuse`` — new generation / reuse a cached chain (interactive ReAct).
* ``revise`` — treat the prompt as a `/revise` edit of the saved agent.
"""


@dataclass(frozen=True)
class ModeSpec:
    """Per-mode pipeline preset. GENERATE/PREVIEW are always auto and so are
    intentionally absent — only the four configurable stages are listed."""

    label: str            # locale key, e.g. "chat.mode.chat"
    run: StagePolicy
    save: StagePolicy
    baseline: StagePolicy
    evolve: StagePolicy
    followup: FollowupKind


MODE_SPECS: dict[ChatMode, ModeSpec] = {
    "interactive": ModeSpec(
        label="chat.mode.chat",
        run=StagePolicy.ASK,        # confirm before running (configurable in /settings)
        # SAVE + EVOLVE are button-driven in Interactive (the chain-action
        # "Save to library" / "Evolve" buttons), NOT pipeline modal gates —
        # so the pipeline leaves them to the UI affordances.
        save=StagePolicy.SKIP,
        baseline=StagePolicy.SKIP,
        evolve=StagePolicy.SKIP,
        followup="reuse",
    ),
    "production": ModeSpec(
        label="chat.mode.agent",
        run=StagePolicy.ASK,        # validate once
        save=StagePolicy.AUTO,
        baseline=StagePolicy.AUTO,
        evolve=StagePolicy.AUTO,
        followup="revise",
    ),
}


def resolve_mode_spec(mode: ChatMode, cfg: "ChatConfig | None") -> ModeSpec:
    """Merge the mode preset with any per-stage config overrides.

    Precedence: preset default ← config override. A ``None`` override
    field (the default) defers to the preset, so an untouched config
    reproduces the preset exactly. ``cfg=None`` also returns the preset.
    """
    preset = MODE_SPECS[mode]
    if cfg is None:
        return preset
    override = getattr(cfg.mode, mode, None)
    if override is None:
        return preset
    return replace(
        preset,
        run=override.run or preset.run,
        save=override.save or preset.save,
        baseline=override.baseline or preset.baseline,
        evolve=override.evolve or preset.evolve,
    )


Reaction = Literal["up", "down"]
"""User-applied reaction on an assistant line. Surfaced via
Ctrl+T (up) / Ctrl+Shift+T (down). ``None`` is the default —
no reaction set."""


@dataclass
class ChatLine:
    """One immutable transcript entry.

    ``mode`` captures the chat mode active at post-time so the
    transcript carries a permanent audit trail of which user
    messages went out "for keeps" (Production). Switching modes
    later doesn't rewrite earlier lines.

    ``provenance`` (Phase 8 P2 #18) is an optional dict carrying
    the LLM call details that produced this line — model,
    duration, tokens, cost, iteration. Populated by the
    generation driver for assistant answers; ``None`` for
    user / system / tool lines (and assistant lines that
    weren't produced by a tracked call, e.g. canned welcome
    messages). Surfaced via the Ctrl+I "prompt inspector"
    binding.

    ``reaction`` (Phase 8 P2 #15) is the optional 👍/👎 the user
    applied via Ctrl+T / Ctrl+Shift+T. Lives only on assistant
    lines today (the toggle helpers only walk assistant
    candidates) — mutated in place via the action; never set
    at construction.
    """

    role: ChatRole
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    mode: ChatMode = DEFAULT_CHAT_MODE
    provenance: dict[str, Any] | None = None
    reaction: Reaction | None = None
    # When True, renderers skip the `[HH:MM] <role>` caption.
    # Used by chrome lines like the boot banner that should
    # land as a clean block rather than a labelled system
    # message.
    chrome: bool = False
    # When True, the Markdown projection wraps registered
    # `/command` tokens in clickable links (see the welcome
    # banner). Clicks route back through the slash dispatcher.
    linkify_commands: bool = False


CommandHandler = Callable[["ChatScreen", str], None]


# Markdown links in the welcome banner point at this synthetic href
# scheme so a click routes back through the slash dispatcher instead
# of the browser. See `ChatScreen.on_markdown_link_clicked`.
_COMMAND_LINK_SCHEME = "care-cmd:"


def _linkify_slash_commands(text: str) -> str:
    """Wrap registered ``/command`` tokens in Markdown links so the
    welcome banner renders them clickable.

    Only tokens that resolve to a real handler in
    :data:`_COMMAND_HANDLERS` are linkified — unknown ``/foo`` text is
    left untouched. The link target carries :data:`_COMMAND_LINK_SCHEME`
    so :meth:`ChatScreen.on_markdown_link_clicked` can tell a command
    click apart from a real URL and dispatch it immediately.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name.lower() not in _COMMAND_HANDLERS:
            return match.group(0)
        return f"[/{name}]({_COMMAND_LINK_SCHEME}/{name})"

    # Lookbehind keeps `://` and mid-word slashes (e.g. a URL path)
    # from being mistaken for a command token.
    return re.sub(r"(?<![\w:/])/([A-Za-z]+)", _replace, text)


class ChatScreen(Screen):
    """Claude-Code-style chat surface — primary CARE entry."""

    DEFAULT_CSS = """
    ChatScreen {
        layout: vertical;
    }
    /* `padding` order is `top right bottom left`. Right
       padding is zero so the vertical scrollbar sits flush
       against the screen edge instead of floating two cells
       in. Left padding is `2` — same value as
       `#chat-mode-row` so chat lines, mode toggle, and the
       boot-banner logo all share one consistent left
       gutter. Top + bottom padding gives the transcript
       breathing room. */
    ChatScreen #chat-transcript {
        height: 1fr;
        padding: 1 0 1 2;
        background: $background;
    }
    /* Single-mode release — the chat footer carries no shortcut hints
       (empty registry) and blends into the screen background so it reads
       as a blank trailing row rather than a panel-coloured strip. */
    ChatScreen CareFooter {
        background: $background;
    }
    /* Outer height is border-box in Textual: `border-top`
       claims one row, the rest is content. The row grows with
       the wrapped `ChatInput` so a long prompt expands
       downward instead of horizontal-scrolling, but capped so
       a runaway paste doesn't push the transcript off-screen.
       `min-height: 2` = 1 border + 1 content row; the cap of
       5 leaves room for 4 content rows before the
       `ChatInput`'s own vertical scroll engages.

       `align-vertical: top` keeps the `>` glyph + Stop
       button anchored to the first content row so they don't
       drift downward as the input expands. */
    ChatScreen #chat-input-row {
        height: auto;
        min-height: 3;
        max-height: 6;
        padding: 0 2;
        background: $boost;
        /* Top + bottom blue lines close the input into a clear box; the
           hint row below sits OUTSIDE these borders. */
        border-top: solid $primary 30%;
        border-bottom: solid $primary 30%;
        align-vertical: top;
    }
    /* §2 P1 — Production action toolbar. `display: none` for
       the hidden state (Ad-Hoc OR no chains yet); the
       `-hidden` class is removed by `_refresh_prod_toolbar`
       once the gate passes. Compact buttons sit flush against
       each other to keep the row narrow. */
    ChatScreen #chat-prod-toolbar {
        height: 1;
        padding: 0 2;
        background: $boost;
        align-vertical: middle;
    }
    ChatScreen #chat-prod-toolbar.-hidden {
        display: none;
    }
    ChatScreen #chat-prod-toolbar Button {
        margin-right: 1;
    }
    ChatScreen #chat-prompt {
        color: $accent;
        text-style: bold;
        width: 2;
        height: 1;
    }
    /* Textual's Input default is `height: 3` (`border: tall` adds
       a 1-row frame top + bottom), which renders the text on the
       middle row and leaves the top/bottom rows visually empty.
       The widget is constructed with `compact=True` in
       :meth:`compose` — that swap-in handles height / padding /
       border on its own (via the upstream `-textual-compact`
       rule). We only need to tint the background here so the
       input strip blends into the surrounding `$boost`. */
    ChatScreen #chat-input {
        background: $boost;
    }
    ChatScreen #chat-input-hints {
        height: 1;
        width: 100%;
        padding: 0 2;
        color: $text-muted;
        text-overflow: ellipsis;
        overflow: hidden;
    }
    ChatScreen .chat-line {
        margin: 0;
        height: auto;
    }
    /* Textual's `Markdown` widget DEFAULT_CSS pins
       `padding: 0 2 0 2` so every Markdown-rendered chat
       line (assistant + system) ends up 2 cells further to
       the right than the Static-rendered ones (user + tool).
       Strip the horizontal padding for chat-line Markdown
       widgets so every chat phrase sits at the same left
       column — `#chat-transcript`'s `padding-left: 2`. */
    ChatScreen Markdown.chat-line {
        padding: 0;
    }
    /* The `✓ Synthesising answer` tool line wears this
       class so it gets one row of bottom padding before the
       assistant answer that follows. Makes the synthesis
       step → final answer transition feel like a visual
       break instead of a tight Markdown shove. */
    ChatScreen .chat-line-pre-answer {
        margin-bottom: 1;
    }
    /* Light tint on user prompts so the eye can spot each
       request at a glance when scrolling a long transcript.
       White at 10 % alpha is theme-agnostic and reads as a
       soft wash on dark backgrounds (the dominant CARE
       theme) and as a faint highlight on light backgrounds
       — strong enough to anchor a turn boundary without
       overwhelming the foreground text.

       `padding: 0` keeps the `[HH:MM]` timestamp flush with
       the rest of the chat lines' left gutter — no extra
       column of whitespace before the timestamp. Symmetric
       `margin-top: 1` / `margin-bottom: 1` adds one row of
       breathing room on both sides of the tinted prompt so
       each user request stands apart from the tool /
       answer chatter above and below it. */
    ChatScreen .chat-line-user {
        color: $accent;
        text-style: bold;
        background: white 10%;
        padding: 0;
        margin-top: 1;
        margin-bottom: 1;
    }
    ChatScreen .chat-line-assistant {
        color: $foreground;
    }
    ChatScreen .chat-line-system {
        color: $text-muted;
        text-style: italic;
    }
    ChatScreen .chat-line-tool {
        color: $warning;
        /* A-4 — when a MAGE stage finishes and `.chat-line-stage-done`
           recolours the `▶ <stage>` row to muted grey, ease the colour
           change instead of snapping so the eye tracks forward progress.
           Auto-instant when the app animation level is "none". */
        transition: color 0.3s out_cubic;
    }
    /* Boot-banner / status chrome lines. White foreground so
       the version + model + cwd block reads as a header
       above the muted system chatter. Top + bottom padding
       gives the three rows a little breathing room so they
       don't feel cramped against the welcome block. The rule
       comes AFTER `.chat-line-system` so its colour wins
       even when the chrome flag rides on a system line. */
    ChatScreen .chat-line-chrome {
        color: white;
        text-style: not italic;
        padding: 1 0;
    }
    /* Boot-banner row layout: the AIRI logo (Static carrying
       `rich_pixels.Pixels`) sits to the left of the version /
       model / cwd Markdown block. Horizontal container is
       sized to the logo's natural footprint so the pixel art
       is neither clipped nor stretched.

       Sizing rationale: the default asset is 10×10 px (one
       of the four 8 / 10 / 12 / 16 variants). Rich-pixels
       half-block rendering uses 1 px per horizontal cell
       and 2 px per vertical cell, so the natural cell
       footprint is 10 wide × 5 tall. Terminal cells are
       typically ~1:2 (width:height), so a block with cell-
       ratio 2:1 (width:height) reads as a visual SQUARE on
       screen. ``_build_boot_logo_widget`` overrides these
       styles at mount time when a non-default size is
       picked. */
    /* The Horizontal row sizes to the logo's height (auto
       picks the taller child). The text widget's vertical
       centring is handled in Python via a computed
       `padding-top` (`_apply_banner_text_centering`) so the
       same rule works for every logo variant without a
       fragile flexbox-style wrapper. */
    ChatScreen .chat-banner-row {
        height: auto;
    }
    ChatScreen .chat-banner-logo {
        width: 10;
        height: 5;
        min-width: 10;
        min-height: 5;
        margin-right: 2;
    }
    /* The banner Markdown sizes to its content (3 rows). A
       `padding-top` set in Python via
       `_apply_banner_text_centering` shifts it down by the
       slack between the logo height and the text row count
       so the text lines up against the logo's middle row.
       Keeping the centring as inline-style overrides means
       the same CSS rule works for every logo variant
       (8/12/16 px) without per-size CSS rules. */
    ChatScreen .chat-banner-text {
        height: auto;
        width: 1fr;
        background: transparent;
    }
    /* New-conversation divider. Applied only to the first
       user line after `/new` so the seam between
       conversations is visually obvious. Consecutive prompts
       within the same conversation deliberately stay
       divider-free — the `.chat-line-user` tint already
       distinguishes them. `background: transparent` drops
       the tint that would otherwise wash over the border
       row, so the divider reads as a clean horizontal line
       against the chat background. */
    ChatScreen .chat-line-user-turn-boundary {
        border-top: solid $primary 30%;
        padding-top: 1;
        margin-top: 1;
        background: transparent;
    }
    /* Phase 8 P2 #21 — once a MAGE stage completes, the matching
       `▶ <stage> …` line gets this class added so the eye tracks
       forward motion: muted colour + strike-through visual. */
    ChatScreen .chat-line-stage-done {
        color: $text-muted;
        text-style: strike;
    }
    /* Per-iteration footer (`iter N · time · token split · model`)
       wears this class so it reads as quieter than the warning-
       yellow `chat-line-tool` default. Same muted grey as the
       stage-done rule, but without the strike-through — the line
       is metadata, not a finished step. */
    ChatScreen .chat-line-iter-footer {
        color: $text-muted;
    }
    /* Phase 8 P1 #10 — Ctrl+F search overlay. Hidden by default,
       Ctrl+F flips `.display` on. The Input gets focus and as the
       user types, matching ChatLines pick up the
       `.chat-line-search-match` class. Enter cycles + Esc
       closes. */
    ChatScreen #chat-search-row {
        height: 1;
        padding: 0 2;
        background: $boost;
        border-top: solid $primary 30%;
    }
    ChatScreen #chat-search-input {
        height: 1;
        border: none;
        background: transparent;
        width: 1fr;
    }
    ChatScreen #chat-search-count {
        height: 1;
        width: auto;
        color: $text-muted;
        text-style: italic;
    }
    ChatScreen .chat-line-search-match {
        background: $warning 20%;
        /* A-5 — fade the highlight in/out as matches are added/cleared and
           as the current match cycles, instead of a hard snap. */
        transition: background 0.15s linear;
    }
    ChatScreen .chat-line-search-current {
        background: $warning 40%;
        text-style: bold;
        transition: background 0.15s linear;
    }
    /* Phase 8 P0 #5 — slash-command autocomplete popup. Floating
       Static below the input row, populated when the input
       starts with `/`. Tab completes the top match; Esc hides. */
    ChatScreen #chat-autocomplete-row {
        height: auto;
        max-height: 13;
        padding: 0 2;
        background: $boost;
        border-top: solid $primary 30%;
        color: $text-muted;
    }
    /* Mode-toggle strip sits between the transcript and the
       prompt: short labels, accent-coloured active state, no
       border — feels like a segmented control. */
    /* Left padding matches `#chat-transcript` (`2` cells) so
       the Ad-Hoc / Production buttons sit at the same column
       as chat lines and the boot-banner logo. */
    ChatScreen #chat-pipeline-strip {
        /* Combined status line: pipeline cells + the "thinking…" spinner on
           one row — `◇ Generate → ○ Run? | ● thinking…`. One text row, no
           border (a `border-top` on a height:1 widget would clip the text);
           the input row below draws its own separator. */
        height: 1;
        padding: 0 2;
        background: $background;
        color: $accent;
        text-overflow: ellipsis;
        /* Collapsed when idle; shown whenever a pipeline is running OR a
           worker is "thinking" (see `_refresh_status_strip`). */
        display: none;
    }
    ChatScreen #chat-mode-row {
        height: auto;
        min-height: 3;
        padding: 0 2 1 2;
        background: $background;
        border-top: solid $primary 30%;
        /* Single-mode release — the interactive/production selector is
           hidden. The RadioSet stays mounted (mode logic + relocalize +
           tests still query it) but isn't shown. The input row's own top
           border keeps the visual separator from the transcript. */
        display: none;
    }
    /* Production / «Create agent» mode — hidden while chat-only UX ships. */
    ChatScreen #chat-mode-production {
        display: none;
    }
    ChatScreen #chat-mode {
        height: 3;
        width: auto;
        border: none;
        background: transparent;
    }
    ChatScreen #chat-mode RadioButton {
        margin-right: 2;
        background: transparent;
    }
    /* A-5 — ease the active-state accent tint in both directions when the
       mode toggle flips (base rule so on→off transitions too). */
    ChatScreen #chat-mode RadioButton > .toggle--button {
        transition: color 0.2s out_cubic;
    }
    ChatScreen #chat-mode RadioButton.-on > .toggle--button {
        color: $accent;
    }
    /* Phase 8 P1 #9 — "■ Stop" button in the input row. Same
       visibility lifecycle as the spinner: shown while any
       worker in `_CANCELLABLE_WORKER_GROUPS` is RUNNING,
       hidden otherwise. Click dispatches `action_interrupt`
       (the same code path Esc walks). */
    /* Icon-only Stop button. Width = 3 cells (1-cell glyph
       + 1 cell of padding either side) so the `■` sits
       visually centred. Textual's `Button` defaults
       (`min-width: 16`, `line-pad: 1`) get overridden here so
       the button stays tight against the input row's right
       edge. */
    ChatScreen #chat-stop-btn {
        width: 3;
        height: 1;
        min-width: 3;
        margin: 0;
        padding: 0;
        background: $error 40%;
        color: $text;
        border: none;
        content-align: center middle;
        text-align: center;
        /* A-5 — smooth the hover / shown-active background swell. */
        transition: background 0.15s out_cubic, color 0.15s out_cubic;
    }
    ChatScreen #chat-stop-btn:hover {
        background: $error;
        color: $background;
    }
    ChatScreen #chat-stop-btn.-active {
        background: $error;
    }
    /* Inline `Read full` button mounted after a successful
       generation. Compact + accent-tinted so it reads as an
       affordance under the stage trail without stealing the eye
       from the assistant answer that follows. `width: auto`
       keeps it hugging its label rather than spanning the
       transcript. */
    /* Persistent interactive chain action bar (replaces Read-full row). */
    ChatScreen .chat-chain-bar-row {
        height: auto;
        width: 1fr;
        margin-bottom: 1;
    }
    ChatScreen .chat-chain-bar-label {
        height: auto;
        width: 1fr;
        content-align: left top;
        color: $text-muted;
        margin-bottom: 1;
    }
    ChatScreen .chat-chain-bar-buttons {
        height: auto;
        width: 1fr;
    }
    ChatScreen .chat-chain-bar-hints {
        height: auto;
        width: 1fr;
        margin-top: 1;
    }
    ChatScreen .chat-chain-bar-hint {
        height: auto;
        width: 1fr;
        content-align: left top;
        color: $text-muted;
    }
    ChatScreen .chat-chain-version-row {
        height: auto;
        width: 1fr;
        margin-bottom: 1;
    }
    ChatScreen .chat-chain-version-label {
        width: auto;
        height: auto;
        content-align: left middle;
        color: $text-muted;
        margin-right: 1;
    }
    ChatScreen .chat-chain-version-select {
        width: 1fr;
        max-width: 24;
        height: auto;
        min-height: 1;
    }
    ChatScreen .chat-chain-act-btn {
        width: auto;
        height: auto;
        min-width: 0;
        margin: 0 1 0 0;
        padding: 0 1;
        background: $accent 25%;
        color: $text;
    }
    ChatScreen .chat-chain-act-btn:hover {
        background: $accent;
        color: $background;
    }
    ChatScreen .chat-chain-act-finish {
        background: $boost;
    }
    ChatScreen .chat-chain-act-finish:hover {
        background: $primary;
        color: $background;
    }
    /* Legacy production Read-full row (interactive mode uses chain bar). */
    ChatScreen .chat-readfull-row {
        height: auto;
        width: 1fr;
        margin-bottom: 1;
    }
    ChatScreen .chat-readfull-label {
        /* Match the 3-row height of the bordered button so the caption
           sits centred against it on the same baseline. */
        height: 3;
        width: auto;
        content-align: left middle;
        color: $text-muted;
        margin-right: 1;
    }
    ChatScreen .chat-readfull-btn {
        /* `Button` paints a `tall` top+bottom border that CSS can't
           clear, so it always claims two rows. `height: auto` lets the
           single label row sit between them — forcing `height: 1` would
           starve the content row and render a blank button. `width:
           auto` keeps it hugging the label rather than the 16-cell
           default. */
        width: auto;
        height: auto;
        min-width: 0;
        margin: 0;
        padding: 0 1;
        background: $accent 25%;
        color: $text;
    }
    ChatScreen .chat-readfull-btn:hover {
        background: $accent;
        color: $background;
    }
    /* Inline RUN-gate confirm row: guide text + [Run] [Save] [Edit] [Skip]. */
    ChatScreen .chat-confirm-row {
        height: auto;
        width: 1fr;
        margin-bottom: 1;
    }
    ChatScreen .chat-confirm-label {
        height: auto;
        width: 1fr;
        content-align: left top;
        color: $text;
        margin-bottom: 1;
    }
    ChatScreen .chat-confirm-buttons {
        height: auto;
        width: 1fr;
    }
    ChatScreen .chat-confirm-btn {
        width: auto;
        height: auto;
        min-width: 0;
        margin: 0 0 0 1;
        padding: 0 1;
        background: $accent 25%;
        color: $text;
    }
    ChatScreen .chat-confirm-btn:hover {
        background: $accent;
        color: $background;
    }
    ChatScreen .chat-confirm-no {
        background: $boost;
    }
    ChatScreen .chat-confirm-no:hover {
        background: $primary;
        color: $background;
    }
    /* Phase 9 P1 — persistent history sidebar (Ctrl+\\). Docks
       to the left edge of the screen with a fixed width; hidden
       by default. When `display: none`, the docked widget
       releases its width back to the layout so the transcript
       reclaims full-width. Click on any row prefills the chat
       input via `on_click` + `_sidebar_actions` lookup. */
    ChatScreen #chat-history-sidebar {
        dock: left;
        width: 32;
        height: 1fr;
        padding: 1 1;
        background: $boost;
        border-right: solid $primary 30%;
        display: none;
        overflow-y: auto;
        overflow-x: hidden;
    }
    ChatScreen .hist-section-title {
        color: $accent;
        text-style: bold;
        padding-bottom: 1;
        padding-top: 1;
    }
    ChatScreen .hist-row {
        height: auto;
        padding: 0 1;
        color: $foreground;
    }
    ChatScreen .hist-row:hover {
        background: $primary 30%;
        text-style: bold;
    }
    ChatScreen .hist-empty {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }
    /* Phase 9 P1 — turn-focus mode (Ctrl+0). Lines belonging
       to earlier turns get this class added; the CSS hides
       them so only the current turn is visible. Toggling
       focus mode off removes the class and the transcript
       restores. */
    ChatScreen .chat-line-turn-hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_transcript", "Clear", show=True),
        Binding("escape", "interrupt", "Interrupt", show=True),
        Binding("ctrl+up", "recall_prev", "↑ history", show=False),
        Binding("ctrl+down", "recall_next", "↓ history", show=False),
        # Plain arrows: navigate suggestion list when the
        # autocomplete popup is open, otherwise walk the prompt
        # history (Claude-Code-style). The branching lives inside
        # `action_recall_prev` / `action_recall_next`.
        Binding("up", "recall_prev", "↑", show=False),
        Binding("down", "recall_next", "↓", show=False),
        Binding("ctrl+y", "copy_last_reply", "Copy reply", show=True),
        Binding(
            "ctrl+shift+y",
            "copy_transcript",
            "Copy all",
            show=False,
        ),
        # Drag-select any text in the transcript / input row,
        # then Cmd+C (on terminals that forward the chord)
        # pipes it through the platform-aware `copy_text`
        # helper. We deliberately do NOT bind Ctrl+C — that
        # belongs to the app-level `global_quit` action so the
        # classic interrupt actually exits CARE. Drag-select
        # already auto-copies via `on_text_selected`, so users
        # don't need a keystroke for the common copy case.
        Binding(
            "super+c",
            "copy_text",
            "Copy selection",
            show=False,
        ),
        Binding(
            "ctrl+e",
            "toggle_step_bodies",
            "Expand steps",
            show=False,
        ),
        Binding(
            "ctrl+d",
            "toggle_compact_mode",
            "Compact",
            show=False,
        ),
        Binding(
            "ctrl+q",
            "quote_last_reply",
            "Quote",
            show=False,
        ),
        Binding(
            "ctrl+f",
            "search",
            "Find",
            show=True,
        ),
        Binding(
            "tab",
            "slash_autocomplete",
            "Complete",
            show=False,
        ),
        Binding(
            "ctrl+i",
            "inspect_last",
            "Inspect",
            show=False,
        ),
        Binding(
            "ctrl+t",
            "react_up",
            "👍",
            show=False,
        ),
        Binding(
            "ctrl+shift+t",
            "react_down",
            "👎",
            show=False,
        ),
        Binding(
            "ctrl+b",
            "copy_code_block",
            "Copy code",
            show=False,
        ),
        Binding(
            "ctrl+backslash",
            "toggle_history_sidebar",
            "History",
            show=True,
        ),
        # Phase 9 P2 — Alt+T cycles through `app.available_themes`
        # in alphabetical order. Ctrl+T is already react_up, so
        # Alt+T avoids the conflict.
        Binding(
            "alt+t",
            "cycle_theme",
            "Theme→",
            show=False,
        ),
        # Phase 9 P1 — Ctrl+0 toggles turn-focus mode: hide
        # every line that doesn't belong to the current turn
        # (collapse earlier turns), or restore the full
        # history. Delivers the collapse/expand UX from the
        # deferred Collapsible-cells spec without rewriting
        # the widget tree.
        Binding(
            "ctrl+0",
            "focus_current_turn",
            "Focus turn",
            show=False,
        ),
    ]

    @staticmethod
    def _welcome_text() -> str:
        return t("chat.welcome")

    # Mode-specific tail appended to the welcome banner so
    # first-time users see exactly what'll happen on their first
    # prompt. Stays separate from `WELCOME_TEXT` so callers /
    # tests that want the generic intro still see the legacy
    # substring.
    # Shared command rows — every line is shown in both modes.
    def _command_blurb(self, name: str) -> str:
        """One-line localized description for a slash command. Resolves
        via the ``chat.cmd.<name>`` catalog key, falling back to the
        English :attr:`_COMMAND_BLURBS` source when no translation
        exists (and to ``""`` for unknown commands)."""
        english = self._COMMAND_BLURBS.get(name)
        if english is None:
            return ""
        translated = t(f"chat.cmd.{name}")
        # `t` echoes the raw key back when the catalog has no entry.
        return english if translated == f"chat.cmd.{name}" else translated

    # /help command rows as (syntax, description) pairs. Syntax tokens
    # are language-neutral; descriptions resolve through the catalog at
    # render time, so /help follows the active UI language.
    def _help_common_rows(self) -> list[tuple[str, str]]:
        return [
            ("/help", self._command_blurb("help")),
            ("/tour", self._command_blurb("tour")),
            ("/mode [interactive|production]", self._command_blurb("mode")),
            ("/artifacts", self._command_blurb("artifacts")),
            ("/library", self._command_blurb("library")),
            ("/evolution [setup|list|<run_id>]", self._command_blurb("evolution")),
            ("/settings", self._command_blurb("settings")),
            ("/run <chain_id>", self._command_blurb("run")),
            ("/resume [latest|<filename>]", self._command_blurb("resume")),
            ("/theme [name]", self._command_blurb("theme")),
            ("/log [level] [module]", self._command_blurb("log")),
            ("/multi", self._command_blurb("multi")),
            ("/edit [N|list]", self._command_blurb("edit")),
            ("/history [N]", self._command_blurb("history")),
            ("/blocks [copy|save N [path]]", self._command_blurb("blocks")),
            (
                "/branch [name|list|switch <id>|delete <id>]",
                self._command_blurb("branch"),
            ),
            ("/imgpreview [status|<path>]", self._command_blurb("imgpreview")),
            ("/subagents [clear]", self._command_blurb("subagents")),
            ("/voice [status|transcribe <path>]", self._command_blurb("voice")),
            (
                "/export <md|mdx|json|html> [path]",
                self._command_blurb("export"),
            ),
            ("/clear", self._command_blurb("clear")),
            ("/quit", self._command_blurb("quit")),
        ]

    # Production-only rows — dataset + evolution sub-forms only make
    # sense once chains are being saved.
    def _help_production_rows(self) -> list[tuple[str, str]]:
        return [
            ("/dataset list <chain_id>", t("chat.help.datasetList")),
            (
                '/dataset add <chain_id> "<task>" --expected "<out>" '
                '[--rubric "<prompt>"]',
                "",
            ),
            ("/dataset run <chain_id>", t("chat.help.datasetRun")),
            (
                "/dataset export <chain_id> <path>",
                t("chat.help.datasetExport"),
            ),
            ("/upload <chain_id>", self._command_blurb("upload")),
            ("/forget <chain_id> [--force]", self._command_blurb("forget")),
            ("/evolution", t("chat.help.evolutionBare")),
            ("/evolution setup [<chain_id>]", t("chat.help.evolutionSetup")),
            ("/evolution list", t("chat.help.evolutionList")),
            ("/evolution <run_id>", t("chat.help.evolutionSnapshot")),
            ("/evolution watch <run_id>", t("chat.help.evolutionWatch")),
            (
                "/evolution accept <run_id> <individual_id>",
                t("chat.help.evolutionAccept"),
            ),
        ]

    @staticmethod
    def _format_command_rows(rows: list[tuple[str, str]]) -> str:
        """Render (syntax, description) pairs into aligned ``/help``
        lines. Short syntaxes pad to a common column; long ones keep
        the description inline so it never falls off a fixed column."""
        pad = 19
        out: list[str] = []
        for syntax, desc in rows:
            if not desc:
                out.append(f"  {syntax}")
            elif len(syntax) <= pad:
                out.append(f"  {syntax:<{pad}}  {desc}")
            else:
                out.append(f"  {syntax}  {desc}")
        return "\n".join(out)

    def _help_command_block(self) -> str:
        """The mode-filtered command list. Production swaps the
        `/evolution` umbrella + trailing `/clear`/`/quit` for the
        concrete dataset/evolution sub-forms, then re-appends the exit
        rows so they stay last."""
        common = self._help_common_rows()
        if self.mode != "production":
            return self._format_command_rows(common)
        filtered = [
            (syntax, desc) for syntax, desc in common
            if not syntax.startswith(("/clear", "/quit", "/evolution"))
        ]
        rows = (
            filtered
            + self._help_production_rows()
            + [
                ("/clear", self._command_blurb("clear")),
                ("/quit", self._command_blurb("quit")),
            ]
        )
        return self._format_command_rows(rows)

    def _render_help_text(self) -> str:
        """Build the `/help` system message in the active UI language.
        The Modes overview is always shown so users learn both modes;
        the command list + scope label follow `self.mode`.

        System lines render as Markdown — the structured blocks are
        wrapped in fenced code so single newlines survive (otherwise
        Markdown collapses them to spaces).
        """
        is_prod = self.mode == "production"
        label = (
            t("chat.help.modeLabelProduction") if is_prod
            else t("chat.help.modeLabelInteractive")
        )
        scope = (
            t("chat.help.scopeProduction") if is_prod
            else t("chat.help.scopeAdHoc")
        )
        return (
            t("chat.help.modesIntro") + "\n"
            "\n"
            "```\n"
            + t("chat.help.modesBlock") + "\n"
            "```\n"
            "\n"
            + t("chat.help.currentlyIn", label=label) + "\n"
            "\n"
            + t("chat.help.commandsHeader", scope=scope) + "\n"
            "\n"
            "```\n"
            + self._help_command_block()
            + "\n```\n"
            "\n"
            + t("chat.help.footer").replace("\n", "  \n")
        )

    # Phase 0 P1 — a11y descriptions for the mode toggle.
    # Surfaced as tooltips on each RadioButton so a hover (or a
    # screen-reader read of the widget's `tooltip` property)
    # documents what the mode does AND, for Production, the
    # behaviour when Memory isn't wired (auto-fallback rather
    # than greyed-out option, per Phase 0 §Production-gate
    # decision).
    @staticmethod
    def _interactive_tooltip() -> str:
        return t("chat.tooltip.adHoc")

    @staticmethod
    def _production_tooltip() -> str:
        return t("chat.tooltip.production")

    @staticmethod
    def _mode_welcome_tail(mode: ChatMode) -> str:
        if mode == "production":
            return t("chat.welcomeTail.agent")
        return t("chat.welcomeTail.chat")

    # Reactive so a watcher fires on every mutation — Phase 1 P1
    # hint-line + Phase 2 / 3 dispatch will subscribe to that. The
    # default lands from the env at construction time (so per-session
    # overrides survive), with :data:`DEFAULT_CHAT_MODE` as the final
    # fallback.
    mode: reactive[ChatMode] = reactive("interactive", init=False)

    def __init__(self, *, mode: ChatMode | None = None) -> None:
        super().__init__()
        self._lines: list[ChatLine] = []
        self._input_history: list[str] = []
        self._history_cursor: int = 0
        self._generating: bool = False
        self._line_counter: int = 0
        # Refs to the localized welcome/mode-tail lines so a UI-language
        # switch (Settings → Save) can re-render them in place via
        # :meth:`relocalize` instead of leaving stale-language text in
        # the transcript. Each pairs the backing ChatLine with its
        # mounted widget id.
        self._welcome_line: ChatLine | None = None
        self._welcome_line_id: str | None = None
        self._welcome_tail_line: ChatLine | None = None
        self._welcome_tail_line_id: str | None = None
        # Other chrome lines that carry localized copy (e.g. mode-flip
        # hints). Each entry is (widget_id, ChatLine, text_factory) so
        # :meth:`relocalize` can re-render them in the new language.
        # Cleared when the transcript is wiped.
        self._localizable_lines: list[
            tuple[str, ChatLine, Callable[[], str]]
        ] = []
        # §3 P0 — Session artifact store. Owns every CARL chain
        # generated during this chat session plus stage payloads
        # / synthesised answers / tool outputs / dataset rows so
        # the `/artifacts` screen can surface them. Listener
        # below repaints the header pill on every mutation.
        from care.runtime.session_artifacts import SessionArtifactStore
        from care.runtime.session_persistence import (
            attach_persistence,
            make_session_id,
            session_path,
        )

        self.artifact_store: SessionArtifactStore = SessionArtifactStore()
        self.artifact_store.add_listener(self._on_artifact_event)
        # §3 P1 — Persist artifacts under
        # `~/.cache/care/sessions/<id>.jsonl` so closing the app
        # doesn't drop them. The cache directory is materialised by
        # `ensure_user_dirs()` in `CareApp.__init__` (§7 P0). The
        # handle is detachable so `/resume` can swap stores
        # without dragging the old persistence along.
        self.session_id: str = make_session_id()
        try:
            self._session_persistence = attach_persistence(
                self.artifact_store,
                session_path(self.session_id),
            )
        except (OSError, ValueError):
            # Non-fatal: a permission failure on
            # ~/.cache/care/sessions/ shouldn't refuse boot.
            # The artifact store keeps working in-memory.
            self._session_persistence = None
        # Honour explicit kwarg → env → project default. Normalise the
        # kwarg so a legacy ``ad_hoc`` value resolves to ``interactive``
        # rather than being treated as a non-interactive (production) mode.
        self._initial_mode: ChatMode = (
            normalise_mode(mode) if mode else _resolve_default_mode()
        )
        # When True, ``watch_mode`` posts the "Now in X mode" hint
        # line. Stays False during the boot transition so users
        # who configured ``CARE_CHAT__DEFAULT_MODE`` don't get a
        # spurious "you switched to Production" line on launch —
        # they didn't, the env did.
        self._hint_armed: bool = False
        # Read by tests + future telemetry. Tracks the current
        # ReAct iteration; ``_run_generation`` resets it to 0
        # in its `finally` block so the next prompt starts fresh.
        self._loop_iteration: int = 0
        # Phase 6 P2 — Production-mode session log path. Lazy:
        # only resolved on the first Production-mode line so
        # Ad-Hoc users never touch disk. Stays set for the rest
        # of the screen lifetime so a flip-back-to-ad_hoc-then-
        # back-to-production resumes the SAME file rather than
        # spawning a fresh one mid-session.
        self._session_log_path: Path | None = None
        # Phase 6 P2 — first-boot guided tour state. ``None`` when
        # no tour is in flight; 1..N during the walk. Any input
        # while non-None advances the tour rather than running
        # MAGE; Esc exits cleanly via ``action_interrupt``.
        self._tour_step: int | None = None
        # Phase 2 P2 — collapsed step bodies. False by default so
        # long CARL step output / MagePoster events fold into a
        # one-liner with an expand hint. Ctrl+E flips the toggle
        # globally; ``ChatLine.text`` keeps the full body so a
        # toggle-on never loses information.
        self._step_bodies_expanded: bool = False
        # Phase 8 P2 #20 — compact / dense mode. When True, the
        # transcript drops the `[HH:MM] role  ` prefix from each
        # line so power users on small terminals see 2–3× more
        # content per screen. Initial value loaded from the
        # tutorial sidecar so the preference survives across
        # sessions. Ctrl+D toggles + re-renders every mounted
        # line widget.
        self._compact_mode: bool = self._tutorial_seen(
            self._COMPACT_MODE_SIDECAR_KEY,
        )
        # Phase 8 P1 #10 — Ctrl+F transcript search state. The
        # match index list tracks which ChatLine indexes
        # (1-based, matching the widget id scheme) currently
        # carry the highlight class so the toggle + Esc paths
        # can clean up without re-scanning. ``_search_cursor``
        # tracks the in-flight cycle position for Enter.
        self._search_open: bool = False
        self._search_matches: list[int] = []
        self._search_cursor: int = 0
        # Phase 8 P0 #5 — slash autocomplete state. Visible flag
        # tracks whether the popup is currently rendered (driven
        # by the `chat-input` value starting with "/"); matches
        # is the ranked candidate list backing Tab-complete.
        self._autocomplete_open: bool = False
        self._autocomplete_matches: list[str] = []
        # Kind drives both the rendering (prefix on each row)
        # and the completion path: "slash" replaces the whole
        # input with `/<name> `; "file" splices `<path>` over the
        # `@<token>` under the cursor.
        self._autocomplete_kind: str = ""
        self._autocomplete_selected: int = 0
        # Track the (start, end) span of the token being
        # completed in the chat input, so file completion can
        # splice into the middle of a longer prompt.
        self._autocomplete_span: tuple[int, int] = (0, 0)
        # Lazy file index for `@<path>` suggestions. Populated on
        # the first `@` keystroke; subsequent keystrokes filter
        # the cached list. Bounded so a giant repo doesn't blow
        # the popup or stall the event loop.
        self._file_index: list[str] | None = None
        # Per-root cache for non-cwd directories so `@../`
        # navigation doesn't rebuild the parent index on every
        # keystroke. Keyed by ``str(Path.resolve())``.
        self._file_index_roots: dict[str, list[str]] = {}
        # Set when `@` refs resolve successfully — triggers a
        # one-shot DataIntroModal before the next generation.
        self._file_ref_intro_pending: bool = False
        # MAGE stage → 1-based widget index for the matching
        # `▶ <label>…` tool line. Populated in
        # `on_stage_started` and consumed in
        # `_mark_stage_started_line_done` so a stage's "done"
        # state survives the friendly-label rendering and won't
        # mis-match unrelated lines via substring search.
        self._stage_started_indexes: dict[str, int] = {}
        # "Read full" inline-button bookkeeping. Each successful
        # generation mounts a button whose id keys into this map; the
        # stored payload (chain_dict + display name + saved chain_id)
        # lets the press handler open the DAG modal and, on evolve,
        # resolve a chain to submit. `_last_chain_action_payload` keeps
        # a live reference to the most-recent payload so the Production
        # save path can backfill the chain_id once it's known.
        self._chain_action_payloads: dict[str, dict[str, Any]] = {}
        self._chain_action_counter: int = 0
        self._last_chain_action_payload: dict[str, Any] | None = None
        self._last_chain_action_button_id: str | None = None
        # Ad-Hoc conversational context — user / assistant turns
        # from the live session, oldest first. The next
        # generation prepends this list as a "Previous
        # conversation" block so follow-up prompts (`а где
        # эссе?`) reference earlier ones. Cleared by `/new`,
        # `/clear`, and mode flips. Production runs deliberately
        # bypass this so each chain stays reproducible.
        self._interactive_history: list[tuple[str, str]] = []
        # In-session chain reuse: successful, parameterized chains keyed by
        # a normalized task template, so a similar follow-up ("погода в
        # Питере" after "погода в Москве") re-runs the cached chain with the
        # new task as outer_context instead of regenerating via MAGE. Reset
        # with the Ad-Hoc history on `/new` / `/clear` / mode flips.
        self._reuse_cache: list[dict[str, Any]] = []
        # Transient pipeline-strip state (Modes redesign). `None` spec ⇒
        # idle/hidden; populated by `_show_pipeline_strip` for a run.
        self._pipeline_spec: ModeSpec | None = None
        self._pipeline_outcomes: dict[Stage, StageOutcome] = {}
        # Combined status strip (pipeline cells + "thinking…"). The spinner
        # and the pipeline strip share one line: `◇ Generate → ○ Run? | ●
        # thinking…`. `_thinking` is on while a cancellable worker runs; it
        # gates the "thinking…" tail and the smooth marker pulse. `_status_phase`
        # advances on a timer to animate the active stage marker + the dot.
        self._thinking: bool = False
        self._status_phase: int = 0
        self._status_anim_timer: Any = None
        # Inline confirm (RUN gate) — the awaited Future is resolved by the
        # Run/Skip button press; `_confirm_counter` keys each row uniquely.
        self._pending_confirm: Any = None
        self._confirm_counter: int = 0
        # Interactive chain session — persistent action bar after generation.
        # Stays mounted until the user clicks Finish; Run/Save/Edit/View/Evolve
        # are non-terminal and keep the bar visible.
        self._chain_session: dict[str, Any] | None = None
        self._chain_session_waiter: Any = None
        self._chain_session_counter: int = 0
        # The raw user prompt for the in-flight generation.
        # `_handle_task` records it here BEFORE prepending the
        # context preamble; `_run_generation` reads it once the
        # assistant reply lands so the history entry stores the
        # user's literal prompt (not the preamble-bloated one).
        self._pending_user_turn: str | None = None
        # Production-mode follow-up target. The first Production
        # generation creates a chain from scratch; once saved, the
        # id lands here so the next plain prompt (until /new,
        # /clear, or a mode flip) is routed through MAGE's
        # chain-edit flow instead of generating a brand-new chain.
        # Cleared by `_reset_interactive_history` so all the existing
        # reset paths drop it without extra plumbing.
        self._production_chain_id: str | None = None
        # Set by `/new` so the NEXT user-line mount carries the
        # turn-boundary divider class. Within a single
        # conversation we deliberately don't draw a divider
        # between consecutive prompts — the boundary is
        # reserved for the "new conversation starts here"
        # moment, which is what /new signals. Cleared on the
        # next user line so the marker fires exactly once.
        self._new_conversation_pending: bool = False
        # Phase 8 P0 #3 — token-streaming state. ``_stream_widget_id``
        # captures the id of the tool-line widget currently
        # accumulating chunks; ``None`` means no active stream
        # so the next ``LlmChunk`` will spawn a fresh widget.
        # Cleared on ``StepCompleted`` (next step starts a new
        # preview) and on ``ChainCompleted`` (run finished).
        self._stream_widget_id: str | None = None
        self._stream_buffer: str = ""
        # Phase 8 P2 #16 — pending human-input future. When a
        # CARL ``HumanInputStep`` fires `on_human_input_requested`,
        # we stash the future here + intercept the next chat
        # submission as the response. ``None`` means no pending
        # request; ``on_input_submitted`` checks this BEFORE
        # routing the text to the slash dispatcher / task
        # handler.
        self._pending_human_input: Any | None = None
        # Phase 8 P3 — sub-agent activity log. Every CARL
        # ``StepEvent`` lands here as `(step_number,
        # event_type, payload_snapshot)`. Bounded to the last
        # `_STEP_EVENT_BUFFER_MAX` entries so long-running
        # debates / parallel sampling don't blow memory.
        # Surfaced via ``/subagents``.
        self._step_events: list[tuple[int, str, dict[str, Any]]] = []
        # Phase 9 P1 — persistent history sidebar (Ctrl+\). The
        # sidebar mounts hidden and is shown via
        # ``action_toggle_history_sidebar``. ``_sidebar_actions``
        # maps each row widget id → ``("prompt", text)`` or
        # ``("chain", chain_id)`` so the click handler knows
        # what to prefill the input with. Rebuilt from scratch
        # on every refresh so we never have to diff.
        self._history_sidebar_open: bool = False
        self._sidebar_actions: dict[str, tuple[str, str]] = {}
        # Phase 9 P2 — accumulate the raw LLM stream for the
        # current iteration so the prompt inspector (Ctrl+I)
        # can render the full response as a fenced code block.
        # Distinct lifecycle from ``_stream_buffer`` (which
        # resets per-STEP for the tool-line preview): this
        # buffer resets per-ITERATION so it captures the
        # whole generate→execute response stream.
        self._iteration_raw_response: list[str] = []
        # Phase 9 P3 — track every streaming tool-line preview
        # widget id created during the iteration so that the
        # assistant-line post can sweep them away. The spec's
        # "stream lands on the assistant Markdown widget"
        # ideal is hard to reach without a CARL signal for
        # "this step is the final answer". The practical
        # value is: by the time the user reads the rendered
        # Markdown answer, the duplicated truncated stream
        # preview should be gone.
        self._iteration_stream_widget_ids: list[str] = []
        # Phase 9 P1 — turn-focus mode (Ctrl+0). ``_current_turn``
        # tracks the 1-based turn number assigned to each new
        # widget at mount time (user lines bump it; assistant /
        # system / tool lines inherit the most-recent turn or
        # 0 for pre-turn content like the welcome block).
        # ``_turn_focus_mode`` toggles "show only the current
        # turn" — earlier turns get a hidden class added to
        # their widgets and the transcript reflows.
        self._current_turn: int = 0
        self._turn_focus_mode: bool = False
        # A newer maestro-care found on PyPI by the background version check
        # (None = up to date / not yet checked / offline). When set, it takes
        # over the resting input-hint line below the prompt.
        self._care_update_latest: str | None = None

    def on_mount(self) -> None:  # noqa: D401 — Textual override
        # Defer reactive set to mount so the watcher is wired before
        # the first value lands (otherwise `watch_mode` doesn't fire
        # for the initial value and downstream side-effects miss it).
        self.mode = self._initial_mode
        super_on_mount = getattr(super(), "on_mount", None)
        if callable(super_on_mount):
            super_on_mount()
        # Phase 8 P0 — spinner starts hidden; the worker-state
        # listener flips it on as soon as MAGE / CARL / dataset /
        # forget / evolution_stream activity lands.
        # Phase 8 P1 #9 — the visible "■ Stop" button shares the
        # same lifecycle so initialise both surfaces together.
        self._set_spinner_visible(False)
        self._set_stop_button_visible(False)
        # Phase 8 P1 #10 — search overlay starts hidden; Ctrl+F
        # flips it visible via `action_search`.
        self._set_search_visible(False)
        # Phase 8 P0 #5 — autocomplete popup starts hidden; the
        # `on_input_changed` handler flips it visible as soon as
        # the user types `/`.
        self._set_autocomplete_visible(False)
        # Phase 9 P2 — restore the user's last `/theme X` choice
        # from the sidecar before painting the welcome block so
        # the welcome line already reflects the chosen palette.
        self._apply_persisted_theme()
        self._post_welcome_and_focus()
        # Arm the hint AFTER the welcome line lands so it only fires
        # on user-driven mode flips during the session.
        self._hint_armed = True

    def _post_welcome_and_focus(self) -> None:
        try:
            header = self.query_one(CareHeader)
            header.refresh_from_app(
                active_screen="ChatScreen",
                breadcrumb=(t("header.breadcrumb.chat"),),
            )
            # Chat nav cluster lives in the top bar: "My chains" +
            # "Evolution" (left of the Artifacts pill) and "Help" (end).
            header.set_library_button(True)
            header.set_evolution_button(True)
            header.set_help_button(True)
            self.query_one(CareFooter).refresh_from_app(
                active_screen="ChatScreen",
                scope="screen",
                # Single-mode release: the chat surface shows no footer
                # shortcut hints — an empty registry leaves a blank row
                # (styled to match the background in CSS).
                registry=(),
            )
            self._refresh_footer_fit()
        except Exception:
            pass
        # Force-sync the badge to the current mode. `watch_mode`
        # skips when old==new, so a boot that lands on the class
        # default (ad_hoc) wouldn't otherwise stamp the badge.
        self._sync_header_badge(self.mode)
        # §3 P0 — initial paint of the session-artifacts pill.
        # The store is empty on boot so this hides the pill; the
        # listener wired in `__init__` keeps it in sync from
        # here on.
        self._sync_artifact_pill()
        # Boot banner: 3-line summary of "what version, which
        # model, where am I" so the user can sanity-check the
        # session before typing their first prompt.
        self._post_boot_header()
        self._post_welcome_block()
        self._refresh_bottom_chrome()
        self._start_evolution_footer_status()
        self._start_version_check()
        try:
            self.query_one("#chat-input", ChatInput).focus()
        except Exception:
            pass

    def _start_evolution_footer_status(self) -> None:
        """Begin polling the Platform for the count of running evolutions
        and surface it in the footer's right-most status segment.

        No-ops when no Platform facade is wired (the common no-evolution
        setup) so we never fire pointless network calls."""
        if getattr(self.app, "platform", None) is None:
            return
        try:
            import os as _os

            interval = float(
                _os.environ.get("CARE_CHAT__EVOLUTION_FOOTER_POLL_SECONDS", "15")
            )
        except (TypeError, ValueError):
            interval = 15.0
        self._evo_footer_base_interval = max(3.0, interval)
        # P-6 — kick the first poll; each poll reschedules the next tick
        # itself (one-shot timer) at a cadence that backs off while nothing
        # is evolving, so an idle session barely touches the Platform.
        self._refresh_evolution_footer_status()

    # P-6 — when no evolutions are running, poll this many times slower
    # (capped) than the base cadence.
    _EVO_FOOTER_IDLE_BACKOFF: float = 4.0
    _EVO_FOOTER_IDLE_MAX_SECONDS: float = 60.0

    def _schedule_evolution_footer_tick(self, delay: float) -> None:
        """Arm the next one-shot footer poll, replacing any pending tick."""
        timer = getattr(self, "_evo_footer_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        try:
            self._evo_footer_timer = self.set_timer(
                delay, self._refresh_evolution_footer_status,
            )
        except Exception:
            self._evo_footer_timer = None

    def _refresh_evolution_footer_status(self) -> None:
        """Timer tick → spawn a worker so the (sync) Platform call doesn't
        block the UI thread. The worker reschedules the next tick once it
        knows whether anything is running (P-6 adaptive backoff)."""
        try:
            self.run_worker(
                self._poll_evolution_footer_status(),
                name="evolution_footer_status",
                group="evolution_footer",
                exclusive=True,
                exit_on_error=False,
            )
        except Exception:
            pass

    async def _poll_evolution_footer_status(self) -> None:
        """Fetch the running-evolution count off-thread and update the
        footer, then reschedule the next tick. Best-effort — never raises
        into the UI."""
        platform = getattr(self.app, "platform", None)
        if platform is None:
            self._set_footer_status("")
            return
        count: int | None = None
        try:
            count = await asyncio.to_thread(platform.running_evolution_count)
        except Exception:
            pass  # leave the last value; transient errors shouldn't blank it
        else:
            self._set_footer_status(f"▶ {count} evolving" if count > 0 else "")
        # P-6 — base cadence while something runs; back off (capped) while
        # idle or after a transient error.
        base = getattr(self, "_evo_footer_base_interval", 15.0)
        if count and count > 0:
            delay = base
        else:
            delay = min(
                base * self._EVO_FOOTER_IDLE_BACKOFF,
                self._EVO_FOOTER_IDLE_MAX_SECONDS,
            )
        self._schedule_evolution_footer_tick(delay)

    def _set_footer_status(self, text: str) -> None:
        if not self.is_mounted:
            return
        try:
            self.query_one(CareFooter).set_status(text)
        except Exception:
            pass

    def _start_version_check(self) -> None:
        """Kick off a one-shot, off-thread check for a newer maestro-care on
        PyPI. Best-effort and non-blocking — never delays mount. The result
        lands in the input-hint strip via :meth:`_apply_care_update_notice`."""
        try:
            self.run_worker(
                self._run_version_check(),
                name="care_version_check",
                group="version_check",
                exclusive=True,
                exit_on_error=False,
            )
        except Exception:
            pass

    async def _run_version_check(self) -> None:
        """Resolve the latest maestro-care off-thread (daily-cached) and, if
        we're behind, surface the nudge. Best-effort — swallows everything."""
        try:
            from care.runtime import version_check

            latest = await asyncio.to_thread(version_check.available_update)
        except Exception:
            return
        if latest:
            self._apply_care_update_notice(latest)

    def _apply_care_update_notice(self, latest: str) -> None:
        """Record the available version and repaint so the notice replaces the
        resting hint line under the prompt."""
        if not self.is_mounted:
            return
        self._care_update_latest = latest
        self._refresh_input_hints()

    def _post_boot_header(self) -> None:
        """Mount the boot banner: AIRI logo (left) + 3-line
        ``CARE v<version>`` / ``<model>`` / ``<cwd>`` block
        (right), wrapped in a Horizontal container so the user
        sees a clean branded header before any chat content.

        Architecture:

        * The logo rides as a Static carrying a ``rich_pixels.Pixels``
          renderable so it draws via the standard ``Static.update``
          path (Static accepts any Rich renderable).
        * The text rides as a Markdown widget with hard-break
          (``"  \\n"``) row terminators so each row lands on its
          own visual line.
        * Both children mount inside an `Horizontal` with the
          ``chat-banner-row`` id so CSS can pin layout, colour,
          padding without per-widget tweaks.
        * A chrome ``ChatLine`` is still appended to ``_lines``
          so transcript / clipboard / sidebar refresh paths see
          the entry (they read the text body, which carries the
          three rows joined by plain ``\\n``).

        Logo failure paths (missing asset, ``rich_pixels`` not
        installed, decode error) fall back to the text-only
        banner so the screen never blanks out the boot header.
        """
        from textual.containers import Horizontal
        from textual.widgets import Markdown

        version = self._resolve_care_version()
        model = self._resolve_active_model()
        cwd = self._render_cwd()
        # Two parallel row lists:
        # * `rows` — Markdown source; the brand line wears
        #   `**…**` so the version reads bold inside the
        #   rendered widget.
        # * `plain_rows` — canonical text body for the
        #   `ChatLine.text` slot. Clipboard / `/export` /
        #   sidebar paths copy this, so the asterisks are
        #   stripped to keep the output copy-pastable.
        rows = [f"**MAESTRO v{version}**"]
        plain_rows = [f"MAESTRO v{version}"]
        if model:
            rows.append(model)
            plain_rows.append(model)
        rows.append(cwd)
        plain_rows.append(cwd)
        # Markdown soft-break collapses single `\n` to space;
        # use the standard `"  \n"` hard-break so each row
        # renders on its own visual line.
        markdown_body = "  \n".join(rows)
        # Record the line in the transcript history so /clear /
        # copy / sidebar paths see it. The canonical text uses
        # plain newlines — the hard-break suffix is a Markdown-
        # render concern, not a data concern.
        plain_body = "\n".join(plain_rows)
        line = ChatLine(
            role="system",
            text=plain_body,
            mode=self.mode,
            chrome=True,
        )
        self._lines.append(line)
        self._line_counter += 1
        line_id = f"chat-line-{self._line_counter}"
        css_classes = (
            f"chat-line chat-line-system chat-line-chrome "
            f"chat-line-turn-{self._current_turn}"
        )

        logo_widget = self._build_boot_logo_widget()
        text_widget = Markdown(markdown_body)
        text_widget.add_class("chat-banner-text")
        # Sit the text 1 row down from the top so the 3-line
        # `CARE / model / cwd` block lines up against the
        # middle row of the logo. With the default 12 px logo
        # (6 cells tall) and a 3-row text body, this gives
        # rows 1–3 of the text against rows 0–5 of the logo
        # — visually centred without a wrapper container.
        # `_apply_banner_text_centering` recomputes when a
        # non-default logo size is in play.
        self._apply_banner_text_centering(text_widget)

        try:
            transcript = self.query_one(
                "#chat-transcript", VerticalScroll,
            )
        except Exception:
            return
        if logo_widget is None:
            # No logo available → mount the text widget alone
            # so we never end up with an empty banner row.
            text_widget.id = line_id
            text_widget.add_class("chat-line")
            text_widget.add_class("chat-line-system")
            text_widget.add_class("chat-line-chrome")
            text_widget.add_class(f"chat-line-turn-{self._current_turn}")
            # No logo to centre against → drop the padding
            # offset so the text rides flush at the top.
            try:
                text_widget.styles.padding = (0, 0, 0, 0)
            except Exception:
                pass
            try:
                transcript.mount(text_widget)
                transcript.scroll_end(animate=False)
            except Exception:
                pass
            return
        logo_widget.add_class("chat-banner-logo")
        banner = Horizontal(
            logo_widget,
            text_widget,
            id=line_id,
            classes=css_classes,
        )
        banner.add_class("chat-banner-row")
        try:
            transcript.mount(banner)
            transcript.scroll_end(animate=False)
        except Exception:
            pass

    # Number of plain-text rows the banner emits (CARE +
    # optional model + cwd). Used by
    # `_apply_banner_text_centering` to compute the padding
    # that lines the text up with the logo's middle row.
    _BANNER_TEXT_ROW_COUNT: int = 3

    def _apply_banner_text_centering(self, text_widget: Any) -> None:
        """Push the Markdown text widget down by the integer
        difference between the logo's rendered cell-height and
        the text's row count, so the three rows sit centred
        against the logo's middle row.

        For the default 12 px logo: 6 cells tall − 3 text rows
        = 3 rows of slack → 1 row of top padding so the text
        occupies rows 1-3 of a 6-row layout (1 row above, 2
        below). For 16 px (8 cells tall): 5 rows of slack →
        2 rows of top padding.
        """
        size = self._boot_logo_size()
        logo_cell_height = max(1, size // 2)
        text_rows = self._BANNER_TEXT_ROW_COUNT
        slack = max(0, logo_cell_height - text_rows)
        # Split the slack so the text leans slightly toward
        # the visual centre. Integer division naturally puts
        # the larger half UNDER the text (`bottom = slack -
        # top`).
        top = slack // 2
        try:
            text_widget.styles.padding = (top, 0, 0, 0)
        except Exception:
            pass

    # Pixel sizes of the bundled `airi_logo_<N>.png` variants.
    # Larger = more detail + more screen real estate. The
    # default (10) reads as a 10×5 cell visual square on
    # typical macOS / iTerm terminals — sized just slightly
    # taller than the 3-row text banner so the brand stays
    # readable without dominating the welcome block.
    _BOOT_LOGO_SIZES: tuple[int, ...] = (8, 10, 12, 16)
    _DEFAULT_BOOT_LOGO_SIZE: int = 10

    @classmethod
    def _boot_logo_size(cls) -> int:
        """Resolve the active logo variant.

        Honours ``CARE_CHAT__BOOT_LOGO_SIZE`` (one of
        ``8`` / ``12`` / ``16``). Malformed / unsupported
        values fall back to the default so the banner never
        breaks on a typo.
        """
        import os

        raw = (
            os.environ.get("CARE_CHAT__BOOT_LOGO_SIZE") or ""
        ).strip()
        if not raw:
            return cls._DEFAULT_BOOT_LOGO_SIZE
        try:
            n = int(raw)
        except ValueError:
            return cls._DEFAULT_BOOT_LOGO_SIZE
        if n in cls._BOOT_LOGO_SIZES:
            return n
        return cls._DEFAULT_BOOT_LOGO_SIZE

    def _build_boot_logo_pixels(self) -> Any:
        """Build the Pixels renderable (without wrapping it in
        a widget). Centralised so the initial mount and the
        theme-change refresh path share the exact same
        compositing pipeline. Returns ``None`` when the
        renderer / asset / Pillow chain can't produce one.
        """
        try:
            from rich_pixels import Pixels
        except Exception as exc:  # noqa: BLE001
            _log.debug("rich_pixels unavailable: %s", exc)
            return None
        size = self._boot_logo_size()
        asset = self._boot_logo_path(size)
        if asset is None:
            return None
        try:
            from PIL import Image

            image = Image.open(str(asset)).convert("RGBA")
            # rich_pixels treats any `alpha > 0` as fully
            # opaque, so partial-alpha anti-alias edges
            # render as solid black cells. Binarise alpha to
            # 0 / 255 so transparent backdrop stays
            # transparent without a phantom black halo.
            alpha = image.getchannel("A").point(lambda v: 255 if v >= 128 else 0)
            image.putalpha(alpha)
            return Pixels.from_image(
                image, renderer=_HalfblockAsymmetricRenderer(),
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("boot logo render failed: %s", exc)
            return None

    def _build_boot_logo_widget(self) -> Any:
        """Build the Static widget that wraps the boot logo
        Pixels. Sizes the widget to the rendered cell
        footprint so the container doesn't clip or stretch.
        Returns ``None`` so the caller falls back to a
        text-only banner.
        """
        pixels = self._build_boot_logo_pixels()
        if pixels is None:
            return None
        size = self._boot_logo_size()
        from textual.widgets import Static

        widget = Static(pixels, id="boot-logo-widget")
        # Half-block rendering: 1 px per horizontal cell,
        # 2 px per vertical cell. Size the widget to the
        # exact footprint so the CSS-pinned default (16 × 8)
        # doesn't crop a smaller variant.
        try:
            widget.styles.width = size
            widget.styles.min_width = size
            widget.styles.height = size // 2
            widget.styles.min_height = size // 2
        except Exception:
            pass
        return widget

    @classmethod
    def _boot_logo_path(cls, size: int | None = None) -> Path | None:
        """Locate the bundled ``care/assets/airi_logo_<N>.png``
        asset for ``size``. Uses ``importlib.resources`` so
        the path resolves both in editable installs (where
        the file lives under the source tree) and in wheels
        (where it lives next to the installed package).

        Falls back to the un-sized ``airi_logo.png`` name when
        the size-specific variant is missing (older installs
        / external assets) so the banner stays robust.
        Returns ``None`` when even the fallback is missing —
        the caller then mounts a text-only banner.
        """
        try:
            from importlib.resources import files
        except Exception:
            return None
        candidates: list[str] = []
        if size is not None and size in cls._BOOT_LOGO_SIZES:
            # Prefer the transparent-backdrop variant when it
            # ships alongside the size-specific PNG so the
            # logo blends into any theme without runtime
            # compositing.
            candidates.append(f"airi_logo_{size}_transparent_bg.png")
            candidates.append(f"airi_logo_{size}.png")
        # Legacy fallback so a wheel built before the size
        # split still works.
        candidates.append("airi_logo.png")
        for name in candidates:
            try:
                ref = files("care.assets").joinpath(name)
            except Exception:
                continue
            try:
                with ref.open("rb") as fh:  # type: ignore[attr-defined]
                    _ = fh.read(1)
            except Exception:
                continue
            try:
                return Path(str(ref))
            except Exception:
                continue
        return None

    @staticmethod
    def _resolve_care_version() -> str:
        """Read the installed package version. The pyproject
        distribution is named ``maestro-care`` (the ``care``
        import name is the package, not the dist), so we ask
        for that first. Falls back to ``care.__version__`` for
        source checkouts, then to reading ``pyproject.toml``
        directly when even that's stale, and finally to
        ``"unknown"`` so the banner never crashes."""
        try:
            from importlib.metadata import PackageNotFoundError, version
        except Exception:
            PackageNotFoundError = Exception  # type: ignore[assignment]
            version = None  # type: ignore[assignment]
        if version is not None:
            for dist in ("maestro-care", "care"):
                try:
                    v = version(dist)
                    if v:
                        return v
                except PackageNotFoundError:
                    continue
                except Exception:
                    continue
        try:
            from care import __version__ as _pkg_version
            if isinstance(_pkg_version, str) and _pkg_version:
                return _pkg_version
        except Exception:
            pass
        try:
            import tomllib
            from pathlib import Path
            # care/screens/chat.py → parents[2] = repo root
            pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
            if pyproject.is_file():
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                v = data.get("project", {}).get("version")
                if isinstance(v, str) and v:
                    return v
        except Exception:
            pass
        return "unknown"

    def _resolve_active_model(self) -> str:
        """Pull the active MAGE model id off the app config.
        Returns ``""`` when no config is reachable so the
        caller can omit the line entirely rather than printing
        a blank row."""
        cfg = getattr(self.app, "config", None)
        if cfg is None:
            return ""
        mage = getattr(cfg, "mage", None)
        if mage is None:
            return ""
        return (getattr(mage, "model", "") or "").strip()

    @staticmethod
    def _render_cwd() -> str:
        """Render the cwd with ``~`` substitution. Falls back
        to the absolute path when the cwd isn't inside the
        user's home tree (or when ``HOME`` isn't set)."""
        try:
            cwd = Path.cwd()
        except Exception:
            return ""
        try:
            home = Path.home()
        except Exception:
            return str(cwd)
        try:
            relative = cwd.relative_to(home)
        except ValueError:
            return str(cwd)
        if relative == Path("."):
            return "~"
        return f"~/{relative}"

    def _post_welcome_block(self) -> None:
        """Post the generic welcome banner followed by a
        mode-specific tail explaining what'll happen on the
        first prompt. Called on boot and on /clear so users
        always see the current-mode contract.

        Phase 6 P2 — appends a one-line tour offer when the
        user hasn't engaged with `/tour` yet. The offer is
        persisted via the existing tutorial sidecar, so the
        second boot (or any boot after `/tour` runs) sees
        only the welcome+tail block.
        """
        # `chrome=True` strips the `[HH:MM] •` caption from
        # these chrome lines — the welcome / mode-tail /
        # tour-offer rows are part of the boot block, not
        # actual chat messages, so the timestamp prefix is
        # noise here. Real system lines (errors, warnings,
        # transcript notices) keep their captions.
        self._post_line(
            "system", self._welcome_text(), chrome=True, linkify_commands=True,
        )
        # Remember the welcome + tail lines so `relocalize` can swap
        # their text when the UI language changes mid-session.
        self._welcome_line = self._lines[-1]
        self._welcome_line_id = f"chat-line-{self._line_counter}"
        tail = self._mode_welcome_tail(self.mode)
        if tail:
            self._post_line("system", tail, chrome=True)
            self._welcome_tail_line = self._lines[-1]
            self._welcome_tail_line_id = f"chat-line-{self._line_counter}"
        else:
            self._welcome_tail_line = None
            self._welcome_tail_line_id = None
        if not self._tutorial_seen("tour_offer_shown"):
            self._post_line("system", t("chat.tour.offer"), chrome=True)
            self._register_localizable_line(lambda: t("chat.tour.offer"))
            # Mark as offered so the second boot (or /clear in the
            # same session) doesn't re-prompt — "first time here?"
            # is a one-shot signal, not a recurring nudge.
            self._mark_tutorial_seen("tour_offer_shown")

    def relocalize(self) -> None:
        """Re-render the screen's localized chrome after a UI-language
        switch (Settings → Save).

        `t()` reads the active language at call time, but widgets that
        were mounted under the old language keep their cached strings —
        so the welcome banner, mode toggle, and header badge stay in the
        old language until something re-renders them. The app calls this
        once the new language is live. The transcript *history* (past
        user/assistant turns) is intentionally left alone — only the
        welcome preamble + interactive chrome flip.

        Best-effort throughout: a missing widget (focus mode, mid-clear)
        is skipped rather than raised.
        """
        # 1 — welcome banner + mode tail, swapped in place.
        if self._welcome_line is not None and self._welcome_line_id:
            self._welcome_line.text = self._welcome_text()
            self._update_markdown_line(self._welcome_line_id, self._welcome_line)
        if self._welcome_tail_line is not None and self._welcome_tail_line_id:
            self._welcome_tail_line.text = self._mode_welcome_tail(self.mode)
            self._update_markdown_line(
                self._welcome_tail_line_id, self._welcome_tail_line,
            )
        # 1b — other localized chrome lines (mode-flip hints).
        for line_id, line, factory in self._localizable_lines:
            try:
                line.text = factory()
            except Exception:
                continue
            self._update_markdown_line(line_id, line)
        # 1c — input + search placeholders, spinner, Stop tooltip.
        self._relocalize_widget_text("#chat-input", placeholder="chat.inputPlaceholder")
        self._refresh_input_hints()
        self._relocalize_widget_text(
            "#chat-search-input", placeholder="chat.searchPlaceholder",
        )
        # The "thinking…" text now rides the combined status strip; repaint
        # it so a language switch re-localizes the tail.
        self._refresh_status_strip()
        self._relocalize_widget_text("#chat-stop-btn", tooltip="chat.stopTooltip")
        # 1d — Production action toolbar (labels + tooltips).
        self._relocalize_widget_text(
            "#chat-prod-btn-artifacts",
            label="chat.prod.artifacts", tooltip="chat.prod.artifactsTip",
        )
        self._relocalize_widget_text(
            "#chat-prod-btn-dataset",
            label="chat.prod.dataset", tooltip="chat.prod.datasetTip",
        )
        self._relocalize_widget_text(
            "#chat-prod-btn-evolve",
            label="chat.prod.evolve", tooltip="chat.prod.evolveTip",
        )
        # 2 — mode toggle labels + tooltips.
        try:
            interactive = self.query_one("#chat-mode-interactive", RadioButton)
            interactive.label = t("chat.mode.chat")
            interactive.tooltip = self._interactive_tooltip()
        except Exception:
            pass
        try:
            prod = self.query_one("#chat-mode-production", RadioButton)
            prod.label = t("chat.mode.agent")
            prod.tooltip = self._production_tooltip()
        except Exception:
            pass
        # The library / evolution / help nav links live in the header now;
        # their localized text is refreshed by CareHeader.relocalize().
        # 3 — header mode badge (chat.badge.*).
        self._sync_header_badge(self.mode)
        # 4 — header / footer chrome.
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="ChatScreen",
                breadcrumb=(t("header.breadcrumb.chat"),),
            )
            self.query_one(CareFooter).refresh_from_app(
                active_screen="ChatScreen",
                scope="screen",
                # Single-mode release: the chat surface shows no footer
                # shortcut hints — an empty registry leaves a blank row
                # (styled to match the background in CSS).
                registry=(),
            )
        except Exception:
            pass
        self._refresh_bottom_chrome()
        self._relocalize_chain_action_bar()

    def _resolve_inline_confirm(self, value: bool) -> None:
        """Resolve the pending inline-confirm Future from a button press."""
        fut = self._pending_confirm
        if fut is not None and not fut.done():
            fut.set_result(value)

    def _register_localizable_line(self, factory: Callable[[], str]) -> None:
        """Track the just-posted chrome line so :meth:`relocalize` can
        re-render it after a UI-language switch. ``factory`` returns the
        line's text in the active language; call this immediately after
        the ``_post_line`` that mounted it (it reads the line + id off
        the latest post)."""
        if not self._lines:
            return
        self._localizable_lines.append(
            (f"chat-line-{self._line_counter}", self._lines[-1], factory),
        )

    def _relocalize_widget_text(
        self,
        selector: str,
        *,
        label: str | None = None,
        placeholder: str | None = None,
        tooltip: str | None = None,
        update: str | None = None,
    ) -> None:
        """Re-apply localized text to one mounted widget by catalog key.
        Each keyword names the widget attribute to set (``label`` /
        ``placeholder`` / ``tooltip``) or ``update`` for ``Static.update``.
        Best-effort: a missing widget (e.g. toolbar not mounted) is
        skipped."""
        try:
            widget = self.query_one(selector)
        except Exception:
            return
        try:
            if label is not None:
                widget.label = t(label)
            if placeholder is not None:
                widget.placeholder = t(placeholder)
            if tooltip is not None:
                widget.tooltip = t(tooltip)
            if update is not None:
                widget.update(t(update))
        except Exception:
            pass

    def _update_markdown_line(self, line_id: str, line: ChatLine) -> None:
        """Repaint a single mounted Markdown chat line from its
        (already-updated) :class:`ChatLine`. No-op when the widget
        isn't currently mounted."""
        try:
            widget = self.query_one(f"#{line_id}", Markdown)
        except Exception:
            return
        try:
            widget.update(self._format_line_as_markdown_for_widget(line))
        except Exception:
            pass

    # Canonical mode → header-badge text. Short, all-caps to feel like a
    # tag; localized at render time.
    @staticmethod
    def _mode_badge(mode: ChatMode) -> str:
        if mode == "production":
            return t("chat.badge.agent")
        return t("chat.badge.chat")

    # Reactive watcher — Textual auto-calls on every `self.mode = ...`.
    def watch_mode(self, old: ChatMode, new: ChatMode) -> None:
        if old == new:
            return
        _log.info("chat mode -> %s (was %s)", new, old)
        # Mode flip wipes the Ad-Hoc context. Production runs
        # are deliberately single-shot reproducible chains, so
        # carrying the conversational state across the boundary
        # would muddle the audit trail; flipping the other way
        # (back into Ad-Hoc) starts a fresh thread.
        self._reset_interactive_history()
        # Header badge tracks the mode so users on compact terminals
        # (where the toggle row may scroll off) still see which mode
        # they're in at a glance. Done first so the badge reflects
        # the new mode even when the production gate reverts below.
        self._sync_header_badge(new)
        # Production-gate: Memory facade is mandatory for the
        # save → baseline → evolve flow. When it isn't wired the
        # user picked an impossible mode — auto-revert to Ad-Hoc
        # and explain why so they aren't stuck on a dead toggle.
        if new == "production" and self._production_blocked():
            _log.warning(
                "production mode requested but Memory facade is None; "
                "reverting to ad_hoc",
            )
            self._post_line(
                "system",
                t("chat.productionFallback"),
                severity="warning",
            )
            # Suppress the spurious "Now in Ad-Hoc mode" hint on
            # the revert — the user didn't pick Ad-Hoc, the
            # fallback did.
            armed = self._hint_armed
            self._hint_armed = False
            try:
                self.mode = "interactive"
            finally:
                self._hint_armed = armed
            return
        # Keep the visible toggle in lockstep with the reactive
        # when the mode changes programmatically (e.g. via the
        # /mode slash command). Pre-mount writes have no widget
        # yet — silently skip.
        self._sync_toggle_to_mode(new)
        # §2 P1 — flip the Production action toolbar with the
        # mode change so a flip from Production → Ad-Hoc hides
        # the buttons immediately + a flip back shows them
        # again (when chains are in the store).
        self._refresh_prod_toolbar()
        self._refresh_input_hints()
        # User-visible "you switched mode" hint. Suppressed during
        # boot so env-defaulted production users don't get a
        # surprise "you switched" line on launch.
        # `chrome=True` matches the welcome-block treatment —
        # this is a session-context callout, not a chat
        # message, so we drop the `[HH:MM] •` caption.
        if self._hint_armed:
            self._post_line(
                "system",
                self._mode_flip_hint(new),
                chrome=True,
            )
            # Track so a later UI-language switch re-renders this hint
            # (bind the flipped-to mode into the factory).
            self._register_localizable_line(
                lambda m=new: self._mode_flip_hint(m),
            )
            # Phase 6 P1 — first time the user toggles into
            # Production, follow the brief hint with a one-shot
            # tutorial walking them through what'll happen on
            # the next message. Sidecar flag suppresses on
            # subsequent flips so power users aren't pestered.
            if new == "production" and not self._tutorial_seen(
                "production",
            ):
                self._post_line(
                    "system", t("chat.productionTutorial"),
                )
                self._mark_tutorial_seen("production")
            # Cross-cutting Telemetry — user-driven flips only,
            # so env-defaulted boots and production-gate reverts
            # don't pollute the event stream.
            self._emit_telemetry(
                "chat.mode.flipped",
                {"old": old, "new": new},
            )

    def _sync_header_badge(self, mode: ChatMode) -> None:
        """Push the mode badge into the :class:`CareHeader`. Pre-mount
        writes are silently skipped — `_post_welcome_and_focus` re-applies
        on first paint.

        Single-mode release: the mode selector is hidden, so the badge
        (``CHAT`` / ``AGENT``) is redundant — we push an empty string,
        which collapses the badge. ``_mode_badge`` is retained for when
        multiple modes return."""
        try:
            header = self.query_one(CareHeader)
        except Exception:
            return
        header.set_badge("")

    def _on_artifact_event(self, _artifact: Any) -> None:
        """Listener subscribed to :attr:`artifact_store`. Fires
        on every append / mark_saved / forget. Marshals the
        repaint to the Textual loop so background workers (MAGE
        / CARL / dataset) can `append_chain(...)` from any
        thread."""
        def _repaint() -> None:
            self._sync_artifact_pill()
            self._refresh_prod_toolbar()
        try:
            self.app.call_from_thread(_repaint)
        except Exception:
            # Pre-mount listener fires (e.g. seed during __init__
            # before app loop runs) lack a live app loop;
            # fall back to a direct call so the initial paint
            # still happens.
            _repaint()

    def _refresh_prod_toolbar(self) -> None:
        """§2 P1 — show / hide the Production action toolbar.

        Visible iff ``self.mode == "production"`` AND at least
        one chain artifact has been generated in the current
        session. The toolbar starts hidden via the `-hidden`
        CSS class; this method flips the class based on the
        live gate so a mode change OR a fresh chain artifact
        toggles visibility automatically.

        Best-effort — pre-mount calls (e.g. from the boot
        listener seed) silently skip the DOM query rather than
        raising.
        """
        if not self.is_mounted:
            return
        try:
            toolbar = self.query_one("#chat-prod-toolbar", Horizontal)
        except Exception:
            return
        production = str(self.mode) == "production"
        has_chain = bool(
            self.artifact_store.list_artifacts(kind="chain"),
        )
        if production and has_chain:
            toolbar.remove_class("-hidden")
        else:
            toolbar.add_class("-hidden")

    def _sync_artifact_pill(self) -> None:
        """Render the header pill as ``"Artifacts"`` — plus a chain-status
        parenthetical once chains exist this session, e.g.
        ``"Artifacts (1 unsaved)"`` / ``"Artifacts (2 saved)"``. Empty when
        the store has no entries; the pill widget auto-hides.

        The parenthetical reflects **chains only** — tool outputs and stage
        payloads aren't persistable, so they never appear as a count.

        Pre-mount writes are silently skipped — the first post-mount paint
        runs from :meth:`_post_welcome_and_focus`'s explicit call below.
        """
        try:
            header = self.query_one(CareHeader)
        except Exception:
            return
        counts = self.artifact_store.counts()
        if counts.get("total", 0) == 0:
            header.set_artifact_pill("")
            return
        chain_total = counts.get("kind:chain", 0)
        unsaved = len(self.artifact_store.unsaved(kind="chain"))
        if chain_total == 0:
            # Only non-chain artifacts so far — bare label, no count.
            text = t("chat.artifactPill.label")
        elif unsaved:
            text = t("chat.artifactPill.unsaved", n=unsaved)
        else:
            text = t("chat.artifactPill.saved", n=chain_total)
        header.set_artifact_pill(text)

    def _stash_generation_artifact(
        self,
        *,
        initial_task: str,
        task: str,
        iteration: int,
        result: Any,
    ) -> str | None:
        """Append the just-finished MAGE generation to the
        session artifact store (§3 P0).

        Extracts the chain dict + a friendly title (suggested
        display name → first 60 chars of the prompt → "Chain")
        and stamps the origin with the iteration index + task
        snippet so the future ArtifactsScreen can group
        ReAct-loop iterations under one user turn. Silent on
        per-artifact failure — generation already succeeded;
        a failed listener / append shouldn't tear down the run.

        Returns the new artifact's id so the caller can flip it
        to ``saved`` (and refresh the header's unsaved pill) when
        the chain is later persisted from the DAG modal. Returns
        ``None`` when nothing was stored.
        """
        try:
            chain_dict = getattr(result, "chain_dict", None) or {}
            if not chain_dict:
                # MAGE returned no chain (an upstream warning
                # already fires in `_run_generation`); skip the
                # append so the store stays clean.
                return None
            from care.runtime.save_agent_form import sanitize_chain_name

            suggested = sanitize_chain_name(
                getattr(result, "suggested_display_name", "") or ""
            )
            description = (
                getattr(result, "suggested_description", "")
                or ""
            ).strip()
            prompt_snippet = initial_task.strip().splitlines()[0][:60]
            title = suggested or prompt_snippet or "Chain"
            summary = description or (
                prompt_snippet if prompt_snippet else "Generated agent chain"
            )
            origin = {
                "iteration": iteration,
                "user_query": initial_task,
                "task": task,
                "mode": str(self.mode),
            }
            artifact = self.artifact_store.append_chain(
                chain=chain_dict,
                title=title,
                summary=summary,
                origin=origin,
            )
            return artifact.id
        except Exception:  # noqa: BLE001
            _log.exception(
                "failed to stash MAGE chain into session artifact store"
            )
            return None

    def _maybe_fire_artifact_pill_click(self, widget_id: str | None) -> bool:
        """Header-pill click → fire the `/artifacts` command.

        Returns True when the click was on the pill (handled),
        False otherwise. Caller (`on_click` lower in this file)
        composes this with its own sidebar / refocus logic.
        """
        if widget_id != "header-artifact-pill":
            return False
        # Look up the handler from the command registry rather
        # than calling the function directly — keeps the click
        # path identical to the `/artifacts` slash and lets the
        # eventual real `ArtifactsScreen` swap in without
        # touching this handler.
        handler = _COMMAND_HANDLERS.get("artifacts")
        if handler is None:
            return True
        try:
            handler(self, "")
        except Exception:
            pass
        return True

    def _maybe_fire_library_button_click(self, widget_id: str | None) -> bool:
        """Header Library link click → fire the `/library` command.

        Returns True when the click was on the link (handled), False
        otherwise. Routed through the command registry so the click
        path stays identical to the `/library` slash.
        """
        if widget_id not in ("header-library-btn",):
            return False
        handler = _COMMAND_HANDLERS.get("library")
        if handler is None:
            return True
        try:
            handler(self, "")
        except Exception:
            pass
        return True

    def _maybe_fire_evolution_button_click(self, widget_id: str | None) -> bool:
        """Header «Evolution» link click → open the evolution primer."""
        if widget_id != "header-evolution-btn":
            return False
        try:
            self._open_evolution_intro()
        except Exception:
            pass
        return True

    def _maybe_fire_help_button_click(self, widget_id: str | None) -> bool:
        """Header «Help» link click → open the Help modal (data primer +
        coding-agent-skill stub)."""
        if widget_id != "header-help-btn":
            return False
        try:
            self._open_help_modal()
        except Exception:
            pass
        return True

    def _open_help_modal(self) -> None:
        """Push the Help modal and route the chosen action back to the
        existing handlers."""
        from care.screens.help_modal import HelpAction, HelpModal

        def _on_dismiss(action: "HelpAction | None") -> None:
            if action == "data":
                self._open_data_intro()
            elif action == "skill":
                self._toast_inline(
                    t("chat.mode.skillStubToast"), severity="info",
                )

        self.app.push_screen(HelpModal(), _on_dismiss)

    def _production_blocked(self) -> bool:
        """``True`` when Production mode should fall back to
        Ad-Hoc because Memory isn't wired.

        Subtlety: the gate must distinguish a real `CareApp` whose
        ``memory`` slot is explicitly ``None`` (block — the user
        hasn't configured Memory) from a bare test host that
        doesn't carry the slot at all (pass — the test scaffold
        doesn't model the gate). `CareApp.__init__` always assigns
        ``self.memory = memory`` so the attribute exists in
        production, defaulting to ``None``.
        """
        app = getattr(self, "app", None)
        if app is None:
            return False
        if not hasattr(app, "memory"):
            return False
        return app.memory is None

    # Phase 6 P2 — guided tour. Five steps, each posted as a
    # ``system`` line so the tour reads as a chat-native
    # walkthrough rather than a modal popup. The user advances
    # by submitting any input (slash command or prose); Esc
    # exits cleanly. After the last step the tour marks itself
    # complete and re-enables normal input dispatch. Step text
    # is localized (``chat.tour.*``); see :meth:`_tour_steps`.
    _TOUR_STEP_COUNT: int = 5

    def _tour_steps(self) -> list[str]:
        """The localized tour steps (``chat.tour.step1..N``), resolved
        fresh so the walkthrough follows the active UI language."""
        return [
            t(f"chat.tour.step{i}")
            for i in range(1, self._TOUR_STEP_COUNT + 1)
        ]

    # ------------------------------------------------------------------
    # Guided tour (Phase 6 P2)
    # ------------------------------------------------------------------

    def _start_tour(self) -> None:
        """Begin the 5-step tour. Idempotent — re-running while a
        tour is in flight restarts from step 1 rather than
        accumulating state. Also marks the tour as "seen" so the
        first-boot offer doesn't keep nagging after the user
        engages with it."""
        if self._generating:
            self._post_line(
                "system",
                t("chat.tour.busy"),
                severity="warning",
            )
            return
        self._tour_step = 1
        self._post_line("system", self._tour_steps()[0])
        # Engaging with /tour counts as "seen" so the offer doesn't
        # re-fire on the next boot even if the user Esc's mid-tour.
        self._mark_tutorial_seen("tour_offer_shown")

    def _advance_tour(self) -> None:
        """Move the tour to the next step. When advanced past the
        last step, mark the tour complete and re-enable normal
        input dispatch."""
        if self._tour_step is None:
            return
        steps = self._tour_steps()
        self._tour_step += 1
        if self._tour_step > len(steps):
            self._tour_step = None
            self._post_line("system", t("chat.tour.complete"))
            self._mark_tutorial_seen("tour_completed")
            return
        self._post_line(
            "system", steps[self._tour_step - 1],
        )

    def _exit_tour(self) -> None:
        """Cancel the tour mid-flow (Esc, /quit, etc.). Posts a
        short exit line so the user has a clear signal that
        normal input dispatch is back."""
        if self._tour_step is None:
            return
        self._tour_step = None
        self._post_line("system", t("chat.tour.exited"))

    # ------------------------------------------------------------------
    # Per-mode tutorial sidecar (Phase 6 P1)
    # ------------------------------------------------------------------

    @staticmethod
    def _tutorial_sidecar_path() -> Path:
        """Resolve the sidecar file that tracks which one-shot
        tutorials this user has already seen.

        Honours ``CARE_CHAT__TUTORIAL_SIDECAR`` so tests can
        redirect to a tmp file. Defaults to
        ``$XDG_STATE_HOME/care/chat_tutorial.json`` (or
        ``~/.local/state/care/chat_tutorial.json`` when XDG
        isn't set) — same family as the existing theme sidecar.
        """
        import os

        override = (
            os.environ.get("CARE_CHAT__TUTORIAL_SIDECAR") or ""
        ).strip()
        if override:
            return Path(override).expanduser()
        state_root = (
            os.environ.get("XDG_STATE_HOME") or ""
        ).strip() or "~/.local/state"
        return Path(state_root).expanduser() / "care" / "chat_tutorial.json"

    @classmethod
    def _read_tutorial_sidecar(cls) -> dict[str, bool]:
        """Read the sidecar dict. Missing file / malformed JSON
        / permission errors all degrade silently to an empty
        dict so a broken sidecar never blocks the chat."""
        import json

        path = cls._tutorial_sidecar_path()
        if not path.exists() or not path.is_file():
            return {}
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): bool(v) for k, v in data.items()}

    @classmethod
    def _tutorial_seen(cls, name: str) -> bool:
        return bool(cls._read_tutorial_sidecar().get(name))

    @classmethod
    def _mark_tutorial_seen(cls, name: str) -> None:
        """Persist that ``name`` has been shown. Failures degrade
        silently — the tutorial will fire again on the next
        boot, which is at worst mildly annoying (vs the chat
        crashing on a sidecar write fault)."""
        import json

        path = cls._tutorial_sidecar_path()
        current = cls._read_tutorial_sidecar()
        current[name] = True
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(current, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            _log.warning(
                "couldn't persist tutorial sidecar at %s: %s",
                path, exc,
            )

    @classmethod
    def _clear_tutorial_flag(cls, name: str) -> None:
        """Phase 8 P2 #20 — remove ``name`` from the sidecar.
        Used by sticky-preference toggles (compact mode) so the
        bit returns to its default-False state on disk rather
        than staying as a stale True. Missing-key and
        permission errors degrade silently."""
        import json

        path = cls._tutorial_sidecar_path()
        current = cls._read_tutorial_sidecar()
        if name not in current:
            return
        current.pop(name, None)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(current, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            _log.warning(
                "couldn't persist tutorial sidecar at %s: %s",
                path, exc,
            )

    @staticmethod
    def _mode_flip_hint(mode: ChatMode) -> str:
        if mode == "production":
            return t("chat.flipHint.agent")
        return t("chat.flipHint.chat")

    def _sync_toggle_to_mode(self, mode: ChatMode) -> None:
        target_id = self._MODE_TO_RADIO_ID.get(mode)
        if target_id is None:
            return
        # Phase 1: turn OFF every non-target button. Has to come
        # before the target gets turned on so RadioSet can't
        # snap-back the previous selection during a re-entrant
        # event (toggle click → watch_mode → sync → fresh Changed
        # event → handler bails when mode already matches, but
        # RadioSet may have already re-asserted the old button).
        for radio_mode, radio_id in self._MODE_TO_RADIO_ID.items():
            if radio_mode == mode:
                continue
            try:
                btn = self.query_one(f"#{radio_id}", RadioButton)
            except Exception:
                continue
            if btn.value:
                btn.value = False
        # Phase 2: turn ON the target. Idempotent — if it was
        # already on (e.g. user clicked the same button), skip.
        try:
            target = self.query_one(f"#{target_id}", RadioButton)
        except Exception:
            return
        if not target.value:
            target.value = True

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        # Phase 9 P1 — persistent left history sidebar. Mounts
        # hidden; Ctrl+\\ flips visibility via
        # `action_toggle_history_sidebar`. Docks to the left of
        # the screen so the transcript shrinks horizontally
        # when the sidebar is shown.
        yield VerticalScroll(id="chat-history-sidebar")
        yield VerticalScroll(id="chat-transcript")
        # Phase 8 P1 #10 — Ctrl+F search overlay. Lives between the
        # transcript and the mode toggle. Hidden by default; the
        # `action_search` binding flips display + focuses the
        # input so the user can immediately type a query.
        with Horizontal(id="chat-search-row"):
            yield Input(
                placeholder=t("chat.searchPlaceholder"),
                id="chat-search-input",
            )
            yield Static("", id="chat-search-count")
        # Modes redesign — transient pipeline strip. Shows the live
        # GENERATE → RUN → … status while a generation pipeline runs,
        # directly above the mode selector; hidden when idle.
        yield Static("", id="chat-pipeline-strip")
        with Vertical(id="chat-mode-row"):
            with RadioSet(id="chat-mode"):
                yield RadioButton(
                    t("chat.mode.chat"),
                    value=self._initial_mode == "interactive",
                    id="chat-mode-interactive",
                    tooltip=self._interactive_tooltip(),
                )
                yield RadioButton(
                    t("chat.mode.agent"),
                    value=self._initial_mode == "production",
                    id="chat-mode-production",
                    tooltip=self._production_tooltip(),
                )
            # The former quick-action strip (My chains / Evolution / Working
            # with data / Add as a skill) moved into the top-bar header:
            # "My chains" + "Evolution" links + a "Help" link whose modal
            # hosts "Working with data" and "Add as a coding-agent skill".
        # §2 P1 — Production-mode action toolbar above the
        # prompt. Hidden by default (toggled visible by
        # `_refresh_prod_toolbar` when (a) the chat is in
        # Production mode AND (b) at least one chain artifact
        # has been generated). Each button shortcuts to the
        # equivalent slash command so keyboard + click paths
        # converge on a single handler.
        with Horizontal(
            id="chat-prod-toolbar",
            classes="-hidden",
        ):
            yield Button(
                t("chat.prod.artifacts"),
                id="chat-prod-btn-artifacts",
                tooltip=t("chat.prod.artifactsTip"),
                compact=True,
            )
            yield Button(
                t("chat.prod.dataset"),
                id="chat-prod-btn-dataset",
                tooltip=t("chat.prod.datasetTip"),
                compact=True,
            )
            yield Button(
                t("chat.prod.evolve"),
                id="chat-prod-btn-evolve",
                tooltip=t("chat.prod.evolveTip"),
                compact=True,
            )
        with Horizontal(id="chat-input-row"):
            yield Static(">", id="chat-prompt")
            # `ChatInput` is a `TextArea` subclass with the
            # Input-compatible surface (`.value`, `.cursor_position`,
            # `action_submit`, `Input.Submitted` / `Input.Changed`
            # bridging). The widget auto-grows 1→4 rows and
            # soft-wraps long prompts so the input stops scrolling
            # horizontally past the visible width.
            yield ChatInput(
                placeholder=t("chat.inputPlaceholder"),
                id="chat-input",
            )
            # Phase 8 P1 #9 — visible "■" Stop affordance paired
            # with the spinner. Initially hidden; `_refresh_spinner`
            # flips it on whenever a cancellable worker is in
            # flight so the user has an eye-level cancel instead
            # of relying on a buried Esc key hint.
            # `compact=True` strips the invisible `border-top` /
            # `border-bottom` shadow rows that Textual's default
            # Button style adds — without it the glyph would
            # sit on the middle row of a 3-row button (the
            # shaded top/bottom rows look like empty space
            # around a centred square).
            yield Button(
                self._STOP_BUTTON_LABEL,
                id="chat-stop-btn",
                tooltip=t("chat.stopTooltip"),
                compact=True,
            )
        # Input-affordance hints ("Enter — send · @file — attach · …").
        # Sits OUTSIDE the input's blue-bordered box: the input row carries
        # its own top+bottom border, so this row is below the input's blue
        # lines rather than sandwiched inside them.
        yield Static("", id="chat-input-hints")
        # Phase 8 P0 #5 — slash-command autocomplete popup. Sits
        # below the input row and pops up as soon as the user
        # types `/`. Filtered against `_COMMAND_HANDLERS` via the
        # existing `fuzzy_score`; Tab inserts the top match.
        yield Static("", id="chat-autocomplete-row")
        yield CareFooter()

    # Map between widget id and ChatMode literal. This dict is the only
    # place that bridge happens.
    _MODE_TO_RADIO_ID: dict[ChatMode, str] = {
        "interactive": "chat-mode-interactive",
        "production": "chat-mode-production",
    }
    _RADIO_ID_TO_MODE: dict[str, ChatMode] = {
        v: k for k, v in _MODE_TO_RADIO_ID.items()
    }

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """User clicked / arrow-keyed to a different mode button.

        The reactive `self.mode = new` re-enters us through
        :meth:`watch_mode`, which is the single source of truth for
        side-effects on a mode flip (log, future Phase-1 hint line).
        Guard against re-entrancy: setting ``self.mode`` programmatically
        also flips the RadioSet via :meth:`watch_mode`, and we'd
        otherwise bounce back through this handler.
        """
        if event.radio_set.id != "chat-mode":
            return
        pressed_id = (event.pressed.id or "") if event.pressed else ""
        new_mode = self._RADIO_ID_TO_MODE.get(pressed_id)
        if new_mode is None or new_mode == self.mode:
            return
        self.mode = new_mode


    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Route a click on a welcome-banner command link back through
        the slash dispatcher so e.g. clicking ``/library`` runs it
        immediately — same as typing the command and pressing Enter.

        Only our synthetic :data:`_COMMAND_LINK_SCHEME` hrefs are
        intercepted; any other link falls through to the Markdown
        widget's default open-in-browser handling.
        """
        href = event.href or ""
        if not href.startswith(_COMMAND_LINK_SCHEME):
            return
        event.stop()
        command = href[len(_COMMAND_LINK_SCHEME):]
        if not command.startswith("/"):
            command = f"/{command}"
        self._handle_command(command)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Phase 8 P1 #10 — Enter on the search input cycles
        # through matches rather than running a chat command.
        if event.input.id == "chat-search-input":
            self._search_cycle_next()
            return
        if event.input.id != "chat-input":
            return
        text = event.value
        event.input.value = ""
        # Phase 8 P2 #16 — when a HumanInputStep is pending,
        # the next chat submission resolves its future
        # instead of routing through the slash dispatcher /
        # task handler. The raw value (not stripped) rides
        # through so the user can deliberately submit
        # whitespace-significant answers.
        if self._pending_human_input is not None:
            self._resolve_pending_human_input(text)
            return
        text = text.strip()
        if not text:
            return
        self._input_history.append(text)
        self._history_cursor = len(self._input_history)
        _log.info("user submitted: %r", text)
        # Phase 6 P2 — tour mode swallows all input as "advance"
        # so the user can press Enter to walk through. `/quit` and
        # `/exit` are honoured so the user can still leave CARE
        # without first running Esc.
        if self._tour_step is not None:
            if text.lower() in ("/quit", "/exit"):
                self._exit_tour()
                self._handle_command(text)
                return
            self._advance_tour()
            return
        # Explicit "remember this": a `#`-prefixed message merges its content
        # into long-term memory (LLM-adapted, supersedes stale facts) instead of
        # running generation. See `_remember_content`.
        if text.startswith("#"):
            self._post_line("user", text)
            self.run_worker(
                self._remember_content(text[1:]),
                name="chat_remember",
                group="memory",
                exit_on_error=False,
            )
            return
        if text.startswith("/"):
            self._handle_command(text)
        else:
            self._handle_task(text)

    def _handle_command(self, raw: str) -> None:
        parts = raw[1:].split(maxsplit=1)
        if not parts:
            return
        name = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        # Mirror the typed command into the transcript so the
        # user sees what they sent — same UX as a plain prompt.
        # ``/clear`` would wipe the line a moment later anyway,
        # so we don't special-case it. Unknown commands still
        # show the line + a "/foo: unknown" follow-up so the
        # user can spot the typo in context.
        self._post_line("user", raw)
        handler = _COMMAND_HANDLERS.get(name)
        if handler is None:
            _log.warning("unknown command /%s", name)
            self._post_line(
                "system",
                f"Unknown command: /{name}. Type /help for the list.",
                severity="warning",
            )
            return
        _log.info("command /%s arg=%r", name, arg)
        try:
            handler(self, arg)
        except Exception as exc:  # noqa: BLE001
            _log.exception("command /%s failed", name)
            self._post_line(
                "system", f"/{name} failed: {exc}", severity="error",
            )

    def _handle_task(self, task: str) -> None:
        self._post_line("user", task)
        if self._generating:
            _log.warning(
                "task %r ignored — generation already in flight", task,
            )
            self._post_line(
                "system",
                "Generation already running — press Esc to interrupt.",
            )
            return
        # Phase 8 P1 #6 — resolve `@<path>` tokens BEFORE routing so the
        # embedded file body rides into MAGE as part of the task. Warnings
        # (file missing / oversize / decode error) surface as system lines
        # but DON'T abort generation — the original tokens stay in place and
        # MAGE sees them as literal text.
        #
        # P-3 — a `@<path>` read can be a multi-MB PDF / Office doc and block
        # for seconds. When refs are present we offload the read to a worker
        # thread and dispatch once it resolves; the common no-ref path stays
        # fully synchronous (zero I/O, no worker), preserving the original
        # control flow for every prompt that doesn't attach a file.
        if not self._scan_at_refs(task):
            # No attachments — resolution is a no-op, route synchronously.
            self._dispatch_resolved_task(task)
            return
        self._generating = True  # lock the prompt while the file(s) read
        self.run_worker(
            self._resolve_then_dispatch(task),
            name="chat_resolve",
            group="generate",
            exclusive=True,
            exit_on_error=False,
        )

    async def _resolve_then_dispatch(self, task: str) -> None:
        """P-3 — read `@<path>` attachments off the UI thread, then re-enter
        the normal routing on the event loop. The blocking read runs in a
        thread (`_resolve_file_refs_async`); `_dispatch_resolved_task` then
        spawns the generation / revise worker exactly as the synchronous
        no-ref path does."""
        try:
            resolved = await self._resolve_file_refs_async(task)
        finally:
            # Release the read lock. `_dispatch_resolved_task` re-acquires
            # `_generating` synchronously (no await between), so the prompt
            # never appears unlocked to another turn.
            self._generating = False
        self._dispatch_resolved_task(resolved)

    async def _resolve_file_refs_async(self, task: str) -> str:
        """P-3 — :meth:`_resolve_file_refs` with the blocking read pushed to
        a worker thread so the UI thread stays responsive (and Esc-cancelable)
        while a large attachment is read. Attach / warning lines still surface
        on the event loop — ``_post_line`` marshals across thread boundaries."""
        if not self._scan_at_refs(task):
            return task
        self._post_line("tool", "▶ Reading attachments…")
        try:
            return await asyncio.to_thread(self._resolve_file_refs, task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system",
                f"Attachment read failed: {exc}",
                severity="warning",
            )
            return task

    def _dispatch_resolved_task(self, resolved_task: str) -> None:
        """Route an already-@-resolved prompt to the generation or
        production-revise worker. Extracted from :meth:`_handle_task` so the
        synchronous no-ref path and the threaded ref path share one routing
        body (P-3)."""
        # Stash the *user-facing* prompt before we layer on the Ad-Hoc
        # conversation context — `_run_generation` reads it in its `finally`
        # block to append a (user, …) entry to the history once the assistant
        # reply lands.
        self._pending_user_turn = resolved_task
        # Ad-Hoc keeps a live conversation context across generations so
        # follow-up prompts ("а где эссе?") can reference earlier ones.
        # Production deliberately skips this — each chain is reproducible.
        followup = self._current_mode_spec().followup
        if followup == "reuse" and self._interactive_history:
            resolved_task = self._build_interactive_prompt(resolved_task)
        # Production follow-up: once a chain is saved this session, treat the
        # next plain prompt as a NL revision instruction against that chain
        # rather than generating a brand-new one. The user resets this with
        # /new (or /clear / mode flip). Reuses the /revise worker so the
        # preview + ConfirmModal + versioned save flow is identical to typing
        # `/revise <id> <change>` by hand.
        if (
            followup == "revise"
            and self._production_chain_id is not None
        ):
            chain_id = self._production_chain_id
            self._generating = True
            _log.info(
                "production revise worker spawned: chain=%s instruction=%r",
                chain_id, resolved_task[:120],
            )
            self._post_line(
                "tool",
                f"Revising saved chain `{chain_id}` "
                "(use /new to start a fresh chain instead).",
            )
            self.run_worker(
                self._run_production_revise(chain_id, resolved_task),
                name="chat_edit",
                group="generate",
                exclusive=True,
                exit_on_error=False,
            )
            return
        self._generating = True
        _log.info(
            "generation worker spawned: task_len=%d task=%r",
            len(resolved_task), resolved_task[:120],
        )
        self._post_line("tool", t("chat.trace.generationStarted"))
        self.run_worker(
            self._run_generation(resolved_task),
            name="chat_generate",
            group="generate",
            exclusive=True,
            exit_on_error=False,
        )

    async def _run_production_revise(
        self, chain_id: str, instruction: str,
    ) -> None:
        """Production follow-up wrapper: route to ``_run_edit``
        with the active chain id prepended so the same parser
        path the `/revise <id> <change>` command takes also
        handles a bare follow-up prompt. ``self._generating`` is
        managed here (the underlying `_run_edit` doesn't touch
        it) so the input stays unlocked after the worker
        finishes.
        """
        try:
            await self._maybe_show_data_intro_from_file_ref()
            await self._run_edit(f"{chain_id} {instruction}")
        finally:
            self._generating = False

    # Cap on how many prior turns ride into the context preamble.
    # Six = three full back-and-forths; enough for "what did
    # you say earlier" follow-ups while keeping the MAGE prompt
    # bounded.
    _AD_HOC_HISTORY_MAX_TURNS: int = 6

    # Per-turn body truncation. Long assistant replies (essay
    # drafts, code dumps) get clipped so the preamble stays
    # under MAGE's prompt budget. Configurable via env for
    # power users who want more verbatim recall.
    _AD_HOC_HISTORY_BODY_MAX_CHARS: int = 1_200

    @classmethod
    def _interactive_history_max_turns(cls) -> int:
        import os

        raw = (
            os.environ.get("CARE_CHAT__AD_HOC_HISTORY_TURNS") or ""
        ).strip()
        if not raw:
            return cls._AD_HOC_HISTORY_MAX_TURNS
        try:
            n = int(raw)
        except ValueError:
            return cls._AD_HOC_HISTORY_MAX_TURNS
        return max(0, n)

    @classmethod
    def _interactive_history_body_chars(cls) -> int:
        import os

        raw = (
            os.environ.get("CARE_CHAT__AD_HOC_HISTORY_CHARS") or ""
        ).strip()
        if not raw:
            return cls._AD_HOC_HISTORY_BODY_MAX_CHARS
        try:
            n = int(raw)
        except ValueError:
            return cls._AD_HOC_HISTORY_BODY_MAX_CHARS
        return max(100, n)

    def _record_interactive_turn(self, user_text: str, assistant_text: str) -> None:
        """Append the just-finished turn to the Ad-Hoc context.

        Caps the buffer at twice :meth:`_interactive_history_max_turns`
        entries (one user + one assistant per turn) so an
        all-day session doesn't accumulate unbounded state. The
        preamble builder slices the tail when prompting, so the
        cap here is just a memory ceiling.
        """
        if not user_text.strip():
            return
        self._interactive_history.append(("user", user_text))
        self._interactive_history.append(("assistant", assistant_text or ""))
        # 2 entries per turn (user + assistant). Keep a few
        # extras above the prompt window so consecutive `/new`
        # bumps don't lose mid-session memory.
        limit = self._interactive_history_max_turns() * 2 + 4
        if limit > 0 and len(self._interactive_history) > limit:
            del self._interactive_history[: len(self._interactive_history) - limit]

    # ------------------------------------------------------------------
    # In-session chain reuse (Phase 2)
    # ------------------------------------------------------------------

    _REUSE_CACHE_MAX = 8

    @staticmethod
    def _task_template(task: str) -> str:
        """Normalize a task to a reuse key by masking its variable parts —
        quoted spans, numbers, and proper nouns (capitalized words past the
        first) collapse to ``*``, the rest lowercases. So "погода в Москве"
        and "погода в Питере" share the template "погода в *", while the
        non-variable skeleton still distinguishes intent ("новости о *" ≠
        "погода в *"). Heuristic — intentionally conservative."""
        import re as _re

        text = str(task or "").strip()
        if not text:
            return ""
        # Quoted spans are the variable payload — collapse them.
        text = _re.sub(r"[\"'«»“”][^\"'«»“”]*[\"'«»“”]", " * ", text)
        tokens = _re.findall(r"\w+|[^\w\s]", text, flags=_re.UNICODE)
        out: list[str] = []
        seen_word = False
        for tok in tokens:
            if not (tok[0].isalnum() or tok[0] == "_"):
                out.append(tok)  # punctuation kept verbatim
                continue
            if any(ch.isdigit() for ch in tok):
                out.append("*")
            elif seen_word and tok[:1].isupper():
                out.append("*")  # a proper noun mid-sentence — the variable
            else:
                out.append(tok.lower())
            seen_word = True
        return " ".join(out)

    @staticmethod
    def _chain_is_parameterized(chain_dict: Any) -> bool:
        """True when the chain references ``$outer_context`` somewhere, so
        re-running it with a different task actually changes its behaviour
        (a chain that hard-codes the entity isn't safe to reuse)."""
        if not isinstance(chain_dict, dict):
            return False
        import json as _json

        try:
            # default=str so a chain whose step_config was already upgraded to
            # Pydantic objects (e.g. a from_dict round-trip mutated it in place)
            # still serializes — the "$outer_context" marker survives the dump,
            # instead of raising TypeError and being misread as "not parameterized".
            blob = _json.dumps(chain_dict, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return False
        return "$outer_context" in blob

    def _cache_chain_for_reuse(self, task: str, chain_dict: Any) -> None:
        """Remember a successful, parameterized chain under its task
        template so a similar follow-up can reuse it. No-op for chains that
        aren't safe to re-run with a different task."""
        if not isinstance(chain_dict, dict) or not chain_dict:
            return
        if not self._chain_is_parameterized(chain_dict):
            return
        template = self._task_template(task)
        if not template:
            return
        import copy as _copy

        # Keep the freshest chain per template.
        self._reuse_cache = [
            e for e in self._reuse_cache if e.get("template") != template
        ]
        self._reuse_cache.append(
            {"template": template, "task": task, "chain": _copy.deepcopy(chain_dict)}
        )
        overflow = len(self._reuse_cache) - self._REUSE_CACHE_MAX
        if overflow > 0:
            del self._reuse_cache[:overflow]

    def _find_reusable_chain(self, task: str) -> dict | None:
        """Return a cached chain whose task template matches ``task`` (a
        deep copy, ready to re-run), or ``None``."""
        if self._current_mode_spec().followup != "reuse" or not self._reuse_cache:
            return None
        template = self._task_template(task)
        if not template:
            return None
        import copy as _copy

        for entry in reversed(self._reuse_cache):
            if entry.get("template") == template:
                return _copy.deepcopy(entry.get("chain"))
        return None

    async def _try_reuse_chain(self, task: str) -> bool:
        """Fast path: if a structurally-matching chain from earlier in this
        conversation is cached, re-run it with the new task as
        outer_context instead of regenerating via MAGE.

        Returns ``True`` when reuse produced an answer (the turn is fully
        handled); ``False`` to fall through to a fresh generation (no match,
        or the reused chain failed / produced nothing).
        """
        cached = self._find_reusable_chain(task)
        if cached is None:
            return False
        # Honour the interactive RUN gate before re-running a cached chain.
        if not await self._confirm_interactive_run(rich=False):
            self._post_line("system", t("chat.stage.runDeclined"))
            return True  # turn handled (user declined the run)
        self._post_line("tool", t("chat.misc.reusingChain"))
        # Reuse skips MAGE — the chain already exists, so GENERATE is done.
        self._update_pipeline_stage(Stage.GENERATE, StageOutcome.DONE)
        started = time.perf_counter()
        run_result = await self._execute_chain_interactive(task=task, chain_dict=cached)
        if run_result is None or self._chain_result_failed(run_result):
            self._update_pipeline_stage(Stage.RUN, StageOutcome.FAILED)
            self._post_line(
                "system",
                "the reused chain didn't fit — generating a fresh one…",
                severity="warning",
            )
            return False
        self._update_pipeline_stage(Stage.RUN, StageOutcome.DONE)
        synthesised = await self._synthesise_user_answer(
            task=task, run_result=run_result,
        )
        answer = (
            synthesised
            if synthesised is not None
            else self._format_carl_result(run_result)
        )
        visible_answer = self._strip_continuation_marker(answer)
        self._post_line("assistant", visible_answer)
        self._post_line(
            "tool",
            f"  ⎿ reused in {time.perf_counter() - started:.1f}s (no generation)",
        )
        if self._pending_user_turn is not None:
            self._record_interactive_turn(self._pending_user_turn, visible_answer)
            self._pending_user_turn = None
        return True

    def _reset_interactive_history(self) -> None:
        """Clear the running Ad-Hoc context. Invoked by `/new`,
        `/clear`, and mode flips so the next prompt starts from
        a blank slate.

        Also clears the §3 P0 session artifact store — the
        spec says the store is reset on `/new` / `/clear` so a
        fresh conversation doesn't carry artifacts from the
        previous one. `clear()` deliberately does NOT fire the
        per-artifact listeners (no single artifact to pass)
        so we manually re-sync the header pill afterwards.
        """
        self._interactive_history.clear()
        self._reuse_cache.clear()
        self._pending_user_turn = None
        self._production_chain_id = None
        self._remove_chain_action_bar(clear_session=True)
        # The pipeline strip lingers past a turn (until Finish / new message);
        # a /new, /clear, or mode flip is a fresh slate, so collapse it too.
        self._hide_pipeline_strip()
        try:
            self.artifact_store.clear()
            self._sync_artifact_pill()
        except Exception:
            pass

    def _build_interactive_prompt(self, current_task: str) -> str:
        """Wrap ``current_task`` with the running Ad-Hoc
        conversation context.

        Format is deliberately plain-text — MAGE consumes it
        as the user's task description. Long bodies get
        truncated so the preamble doesn't blow the prompt
        budget. ``/new`` clears the history.
        """
        max_turns = self._interactive_history_max_turns()
        body_cap = self._interactive_history_body_chars()
        if max_turns == 0 or not self._interactive_history:
            return current_task
        turns = self._interactive_history[-max_turns:]
        lines: list[str] = [
            "Previous conversation in this session "
            "(use it as context for the current request):",
            "",
        ]
        for role, body in turns:
            label = "User" if role == "user" else "Assistant"
            text = (body or "").strip()
            if len(text) > body_cap:
                text = text[: body_cap - 1].rstrip() + "…"
            lines.append(f"{label}: {text}")
        lines.append("")
        lines.append("Current request:")
        lines.append(current_task)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Generation retry policy
    # ------------------------------------------------------------------

    # Default number of attempts MAGE generation is allowed
    # before the iteration gives up. 3 is the standard
    # "try, retry once, retry once more" budget; raises survive
    # transient API hiccups + Pydantic-validation chain errors
    # (the LLM hallucinated a bad step_type — a fresh call
    # often returns a valid plan).
    _DEFAULT_GENERATION_MAX_ATTEMPTS: int = 3

    # Cap the exponential backoff so a long-running streak of
    # failures doesn't strand the user in a multi-minute wait.
    _GENERATION_BACKOFF_CAP_SECONDS: float = 8.0

    @classmethod
    def _generation_max_attempts(cls) -> int:
        """Resolve the per-iteration retry budget. Honours
        ``CARE_CHAT__GENERATION_MAX_ATTEMPTS`` (default 3,
        clamped to >=1 so the legacy "no retry" behaviour is
        opt-in via an explicit ``1``)."""
        import os

        raw = (
            os.environ.get("CARE_CHAT__GENERATION_MAX_ATTEMPTS") or ""
        ).strip()
        if not raw:
            return cls._DEFAULT_GENERATION_MAX_ATTEMPTS
        try:
            n = int(raw)
        except ValueError:
            return cls._DEFAULT_GENERATION_MAX_ATTEMPTS
        return max(1, n)

    async def _generate_with_retry(
        self,
        run_generation: Any,
        generator: Any,
        task: str,
        *,
        iteration: int,
    ) -> Any:
        """Drive MAGE generation with bounded retries.

        The MAGE pipeline can fail on a single roll due to:
        * transient provider errors (5xx, timeout)
        * Pydantic validation when the LLM hallucinates a
          ``step_type`` outside CARL's enum
        * malformed JSON from the planner

        None of these are deterministic — a fresh call with
        the same task usually succeeds. We rerun up to
        :meth:`_generation_max_attempts` times with exponential
        backoff (1, 2, 4, capped at
        :data:`_GENERATION_BACKOFF_CAP_SECONDS`). Each retry
        surfaces a system warning so the user sees why the
        chain looked busy for an extra few seconds. The final
        failure raises the last exception unchanged so the
        caller's error-rendering path stays in charge of the
        red "Generation failed" line.

        ``asyncio.CancelledError`` is re-raised immediately
        (Esc must always abort instantly).
        """
        max_attempts = self._generation_max_attempts()
        attempt = 0
        last_exc: BaseException | None = None
        while attempt < max_attempts:
            attempt += 1
            try:
                return await run_generation(generator, task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                _log.warning(
                    "generation attempt %d/%d failed (iter %d): %s",
                    attempt, max_attempts, iteration, exc,
                )
                if attempt >= max_attempts:
                    break
                short_reason = self._short_exception_label(exc)
                self._post_line(
                    "system",
                    f"Generation attempt {attempt}/{max_attempts} "
                    f"failed ({short_reason}); retrying…",
                    severity="warning",
                )
                backoff = min(
                    2 ** (attempt - 1),
                    self._GENERATION_BACKOFF_CAP_SECONDS,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _short_exception_label(exc: BaseException) -> str:
        """Build a one-line "Type: message" summary safe to ride
        in a system-line preview. Long messages get truncated so
        the retry hint stays on one row even for a Pydantic
        validation traceback."""
        name = type(exc).__name__
        msg = " ".join(str(exc).split())
        cap = 80
        if len(msg) > cap:
            msg = msg[: cap - 1] + "…"
        return f"{name}: {msg}" if msg else name

    # ------------------------------------------------------------------
    # Generation worker (mirrors CareApp._run_mage_generation but
    # streams events into the chat transcript instead of pushing
    # GenerationScreen)
    # ------------------------------------------------------------------

    def _user_context_bundle(self, query: str = "") -> tuple[str, Any, str]:
        """Standing user-context to inject everywhere: CARE.md + a recalled LTM
        digest, plus the live LTM store + its session id (for attaching to the
        run context + the post-turn save-decision). Best-effort —
        ``("", None, "default")`` on any failure; memory is never load-bearing.
        See :mod:`care.context_md` + :mod:`care.memory_ltm`."""
        cfg = getattr(self.app, "config", None)
        parts: list[str] = []
        ltm: Any = None
        session_id = "default"
        try:
            from care import context_md, memory_ltm

            ctx_cfg = getattr(cfg, "context", None)
            if ctx_cfg is None or getattr(ctx_cfg, "enabled", True):
                care_md = context_md.load_user_context(cfg)
                if care_md:
                    parts.append(care_md)
            session_id = memory_ltm.ltm_session_id(cfg)
            ltm = memory_ltm.build_ltm(cfg)
            if ltm is not None:
                cap = getattr(ctx_cfg, "ltm_inject_max_chars", 2000) if ctx_cfg else 2000
                digest = memory_ltm.recall_digest(
                    ltm, session_id, query=query, max_chars=cap,
                )
                if digest:
                    parts.append(digest)
        except Exception as exc:  # noqa: BLE001
            _log.info("user-context bundle failed: %s", exc)
        return ("\n\n".join(parts), ltm, session_id)

    def _ltm_llm_complete(self):
        """A sync ``(system, user) -> raw_json`` LLM callable for the memory
        passes (save-decision + explicit remember), or ``None`` when no LLM is
        configured (no api_key / base_url) — keeps unconfigured installs + the
        test suite offline. Best-effort."""
        cfg = getattr(self.app, "config", None)
        mage = getattr(cfg, "mage", None)
        if mage is None or not (
            getattr(mage, "api_key", None) or getattr(mage, "base_url", None)
        ):
            return None
        try:
            from care.runtime.llm_client import build_llm_client

            client = build_llm_client(mage)
        except Exception as exc:  # noqa: BLE001
            _log.info("LTM LLM client unavailable: %s", exc)
            return None
        model = getattr(mage, "model", "") or "gpt-4o-mini"

        def _complete(system: str, user: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return resp.choices[0].message.content or ""

        return _complete

    async def _maybe_save_ltm(self, *, query: str, answer: str) -> None:
        """Post-turn save-decision: conservatively persist durable USER facts
        (role / prefs / recurring constraints / projects) to CARL's LTM, deduped
        and superseding stale values, and surface ``🧠 remembered: …``. Runs ONE
        cheap LLM call off-thread; gated by ``config.context.ltm_autosave``.
        Best-effort — never raises into the turn. See :mod:`care.memory_ltm`."""
        cfg = getattr(self.app, "config", None)
        ctx_cfg = getattr(cfg, "context", None)
        if ctx_cfg is None or not getattr(ctx_cfg, "ltm_autosave", False):
            return
        if not (query or "").strip():
            return
        complete = self._ltm_llm_complete()
        if complete is None:
            return
        try:
            import asyncio

            from care import memory_ltm

            ltm = memory_ltm.build_ltm(cfg)
            if ltm is None:
                return
            session_id = memory_ltm.ltm_session_id(cfg)
            cap = getattr(ctx_cfg, "ltm_inject_max_chars", 2000)
            existing = memory_ltm.recall_digest(ltm, session_id, max_chars=cap)
            saved = await asyncio.to_thread(
                memory_ltm.save_from_turn,
                ltm,
                session_id,
                query=query,
                answer=answer,
                complete=complete,
                existing_digest=existing,
            )
        except Exception as exc:  # noqa: BLE001
            _log.info("LTM save-decision skipped: %s", exc)
            return
        line = memory_ltm.format_saved(saved)
        if line:
            self._post_line("system", line)

    async def _remember_content(self, content: str) -> None:
        """Explicit "remember this" (``#…`` / ``/remember``): LLM-merge the note
        into LTM — adapt, don't lose info, supersede stale/contradictory facts —
        with a raw-note fallback so an explicit request is never lost. Surfaces
        ``🧠 remembered: …``. Best-effort, off-thread."""
        content = (content or "").strip()
        if not content:
            self._post_line(
                "system",
                "Usage: `# <note to remember>`  (or `/remember <note>`)",
                severity="warning",
            )
            return
        cfg = getattr(self.app, "config", None)
        try:
            import asyncio

            from care import memory_ltm

            ltm = memory_ltm.build_ltm(cfg)
            if ltm is None:
                self._post_line(
                    "system", "Long-term memory is disabled.", severity="warning",
                )
                return
            session_id = memory_ltm.ltm_session_id(cfg)
            ctx_cfg = getattr(cfg, "context", None)
            cap = getattr(ctx_cfg, "ltm_inject_max_chars", 2000) if ctx_cfg else 2000
            existing = memory_ltm.recall_digest(ltm, session_id, max_chars=cap)
            # no LLM → a complete that fails, so remember_text's fallback stores
            # the raw note (an explicit "remember this" is never dropped).
            complete = self._ltm_llm_complete() or (lambda _s, _u: "")
            saved = await asyncio.to_thread(
                memory_ltm.remember_text,
                ltm,
                session_id,
                content=content,
                complete=complete,
                existing_digest=existing,
            )
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system", f"Couldn't remember that: {exc}", severity="error",
            )
            return
        self._post_line("system", memory_ltm.format_saved(saved) or "🧠 noted.")

    # ------------------------------------------------------------------
    # /memory — view + edit what CARE remembers (P6.8)
    # ------------------------------------------------------------------
    def _memory_show(self) -> None:
        """Render CARE.md context + the long-term-memory keys/values."""
        cfg = getattr(self.app, "config", None)
        blocks: list[str] = ["## Memory"]
        try:
            from care import context_md, memory_ltm

            care_md = context_md.load_user_context(cfg)
            blocks.append(care_md if care_md else "_(no CARE.md context)_")
            ltm = memory_ltm.build_ltm(cfg)
            sid = memory_ltm.ltm_session_id(cfg)
            digest = (
                memory_ltm.recall_digest(ltm, sid, max_chars=10_000)
                if ltm is not None else ""
            )
            blocks.append(digest if digest else "_(long-term memory empty)_")
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system", f"Couldn't read memory: {exc}", severity="error",
            )
            return
        self._post_line("system", "\n\n".join(blocks))

    def _memory_forget(self, key: str) -> None:
        """Delete one long-term-memory key (``/memory forget <key>``)."""
        if not key:
            self._post_line(
                "system", "Usage: /memory forget <key>", severity="warning",
            )
            return
        cfg = getattr(self.app, "config", None)
        try:
            from care import memory_ltm

            ltm = memory_ltm.build_ltm(cfg)
            if ltm is None:
                self._post_line(
                    "system", "Long-term memory is disabled.", severity="warning",
                )
                return
            removed = ltm.delete(key, session_id=memory_ltm.ltm_session_id(cfg))
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system", f"Couldn't forget {key!r}: {exc}", severity="error",
            )
            return
        if removed:
            self._post_line("system", f"🗑 forgot: {key}")
        else:
            self._post_line(
                "system", f"No long-term-memory key {key!r} to forget.",
                severity="warning",
            )

    def _memory_edit(self) -> None:
        """Open the global CARE.md in the user's editor (OS-aware)."""
        cfg = getattr(self.app, "config", None)
        try:
            from care import context_md

            ctx_cfg = getattr(cfg, "context", None)
            target = (
                getattr(ctx_cfg, "global_path", None)
                or context_md.default_global_care_md()
            )
            path = context_md.ensure_care_md(target)
            self._launch_editor(path)
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system", f"Couldn't open editor: {exc}", severity="error",
            )
            return
        self._post_line("system", f"📝 opened {path} in your editor.")

    @staticmethod
    def _resolve_editor() -> str:
        """The editor command to use, cross-platform: ``$VISUAL`` / ``$EDITOR``
        first, else ``notepad`` on Windows and ``nano`` → ``vi`` on unix."""
        import os
        import shutil
        import sys

        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if editor:
            return editor
        if os.name == "nt" or sys.platform.startswith("win"):
            return "notepad"
        return shutil.which("nano") or shutil.which("vi") or "vi"

    def _launch_editor(self, path: Any) -> None:
        """Run the resolved editor on ``path``, suspending the TUI while it's
        open. Best-effort — overridable/mockable in tests."""
        import subprocess

        cmd = [self._resolve_editor(), str(path)]
        suspend = getattr(self.app, "suspend", None)
        if callable(suspend):
            with suspend():
                subprocess.run(cmd, check=False)
        else:
            subprocess.run(cmd, check=False)

    def seed_input(self, text: str) -> None:
        """Prefill the chat input with ``text`` and focus it.

        Used by cross-screen hand-offs (e.g. the library's "Revise (AI)" action
        seeds ``/revise <id> `` so the user lands in chat ready to type the
        change). Best-effort — silently no-ops if the input isn't mounted.
        """
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            return
        inp.value = text
        try:
            inp.focus()
            inp.cursor_position = len(text)
        except Exception:
            pass

    async def _try_load_chain(
        self, memory: Any, entity_id: str
    ) -> dict[str, Any] | None:
        """Best-effort load of a saved chain dict by id via the memory client.

        Returns ``None`` when the id doesn't resolve — the caller then treats
        the token as the first word of the instruction (search-by-prose) rather
        than as an id.
        """
        client = getattr(memory, "client", None)
        getter = getattr(client, "get_chain_dict", None)
        if not callable(getter):
            return None
        try:
            content = await asyncio.to_thread(getter, entity_id, channel="latest")
        except Exception:  # noqa: BLE001 — unknown id / transport ⇒ "not an id"
            return None
        return content if isinstance(content, dict) else None

    async def _run_deploy(self, raw: str) -> None:
        """Worker for ``/deploy`` — ship a saved chain to the agent hub.

        Flow (PRODUCTION_TODO B2): resolve the chain (id-first, name-search
        fallback) on the requested channel → run the client-side deploy gate
        (loadability + template tool set + MAGE lint) → ensure the hub is up
        (autostart per ``[hub]``) → ``POST /deployments`` → post the agent URL
        and its personal Swagger link.
        """
        ref, channel, agent_name = _parse_deploy_args(raw)
        memory = getattr(self.app, "memory", None)
        client = getattr(memory, "client", None)
        if client is None:
            self._post_line(
                "system",
                "Memory is not configured — /deploy needs the chain registry "
                "(set [memory].base_url).",
                severity="error",
            )
            return

        # --- resolve: id first, unique name-search hit second -------------
        entity_id = ref
        record = await self._fetch_chain_record(client, ref, channel)
        if record is None:
            match = await self._find_chain_by_name(client, ref)
            if match is None:
                self._post_line(
                    "system",
                    f"Could not resolve {ref!r} to a saved chain (tried id and "
                    f"name search). If you copied it from a “Baseline recorded "
                    f"as run …” line, that's a run id — use the chain id from "
                    f"the “Saved chain …” line instead. Also: /deploy needs a "
                    f"version on channel {channel!r} — promote it first, or pass "
                    f"--channel latest.",
                    severity="error",
                )
                return
            entity_id = match
            record = await self._fetch_chain_record(client, entity_id, channel)
            if record is None:
                self._post_line(
                    "system",
                    f"Chain {entity_id} has no version on channel {channel!r} — "
                    f"promote it first (library P action) or pass --channel latest.",
                    severity="error",
                )
                return

        content = dict(getattr(record, "content", None) or {})
        meta = dict(getattr(record, "meta", None) or {})
        display_name = str(
            meta.get("display_name") or content.get("name") or entity_id
        )

        # --- gate-lite ------------------------------------------------------
        # Collect the chain's locally-synthesized tools so they ship WITH the
        # deployment (the hub bundles only builtins) — and so the gate counts
        # them as available instead of flagging them "missing".
        try:
            from care.tool_synthesis import bundled_tools_for_chain

            bundled = bundled_tools_for_chain(content, self.app.config)
        except Exception:  # noqa: BLE001 — bundling is best-effort
            bundled = []
        bundled_names = frozenset(t["name"] for t in bundled)
        issues = await asyncio.to_thread(gate_chain_for_deploy, content, bundled_names)
        if issues:
            self._post_line(
                "system",
                f"🛑 Deploy gate failed for {display_name} — nothing deployed:",
                severity="error",
            )
            for issue in issues:
                self._post_line("tool", f"  ⎿ {issue}")
            return
        self._post_line("tool", f"✓ Deploy gate passed for {display_name}")
        if bundled:
            self._post_line(
                "tool",
                f"  ⎿ bundling {len(bundled)} synthesized tool(s) into the agent: "
                + ", ".join(t["name"] for t in bundled),
            )

        # --- hub up + deploy --------------------------------------------------
        agent_name = agent_name or _slugify_agent_name(display_name)
        # Generate a per-agent API key (C4). It guards /invoke /chat /runs;
        # loopback requests bypass it, so the local demo /docs still works.
        api_key = secrets.token_urlsafe(24)
        try:
            hub = await ensure_hub_running(
                self.app.config.hub, env=hub_env(self.app.config)
            )
            deployment = await hub.deploy(
                {
                    "name": agent_name,
                    "entity_id": entity_id,
                    "channel": channel,
                    "api_key": api_key,
                    "extra_tools": bundled,
                }
            )
        except HubUnavailableError as exc:
            self._post_line("system", f"Hub unavailable: {exc}", severity="error")
            return
        except HubError as exc:
            self._post_line("system", f"Deploy rejected: {exc}", severity="error")
            return

        agent_url = hub.agent_url(deployment.name)
        self._post_line(
            "assistant",
            f"🚀 Deployed **{deployment.display_name}** ({deployment.version}) "
            f"as `{deployment.name}` → {agent_url}",
        )
        self._post_line("tool", f"  ⎿ docs: {hub.docs_url(deployment.name)}")
        self._post_line("tool", f"  ⎿ api key: {api_key}  (X-API-Key header for remote calls)")
        self._post_line(
            "tool",
            f"  ⎿ channel: {channel} — promote a new version to hot-reload "
            f"this agent; pin an old one to roll back",
        )
        if not deployment.ready:
            self._post_line(
                "system",
                f"⚠ Agent is not ready yet: {deployment.ready_reason}",
                severity="warning",
            )

    async def _run_metrics(self) -> None:
        """Worker for ``/metrics`` — per-deployment usage + USD cost (D4).

        Reads each agent's ``/metrics``. Like ``/deployments`` it never
        autostarts the hub (viewing must not spawn a process).
        """
        hub_config = self.app.config.hub
        client = HubClient(hub_config.base_url, timeout=hub_config.timeout)
        if await client.health() is None:
            self._post_line(
                "system",
                f"Hub at {hub_config.base_url} is not running — no metrics. "
                f"/deploy <chain> starts it.",
                severity="warning",
            )
            return
        try:
            deployments = await client.list_deployments()
        except (HubError, HubUnavailableError) as exc:
            self._post_line("system", f"Could not reach the hub: {exc}", severity="error")
            return
        if not deployments:
            self._post_line(
                "system", "No deployments yet — ship one with /deploy <chain-id|name>."
            )
            return
        self._post_line("system", f"Agent metrics on {hub_config.base_url}:")
        total_cost = 0.0
        any_priced = False
        for item in deployments:
            metrics = await client.agent_metrics(item.name)
            self._post_line("tool", _format_metrics_row(item.name, metrics))
            cost = metrics.get("total_cost_usd") if metrics else None
            if cost is not None:
                any_priced = True
                total_cost += float(cost)
        if any_priced:
            self._post_line(
                "tool", f"  ⎿ total spend across priced agents: ${total_cost:.4f}"
            )

    async def _run_deployments(self, action: str, name: str | None) -> None:
        """Worker for ``/deployments`` — list/inspect/operate hub deployments.

        Never autostarts the hub: looking at deployments should not spawn a
        process (``/deploy`` is the autostart path). A down hub posts a
        friendly hint instead.
        """
        hub_config = self.app.config.hub
        client = HubClient(hub_config.base_url, timeout=hub_config.timeout)
        if await client.health() is None:
            self._post_line(
                "system",
                f"Hub at {hub_config.base_url} is not running — nothing is "
                f"deployed. /deploy <chain> starts it automatically.",
                severity="warning",
            )
            return
        try:
            if action == "list":
                deployments = await client.list_deployments()
                if not deployments:
                    self._post_line(
                        "system",
                        "No deployments yet — ship one with /deploy <chain-id|name>.",
                    )
                    return
                self._post_line(
                    "system", f"Deployments on {hub_config.base_url}:"
                )
                for item in deployments:
                    status = "ready" if item.ready else f"⚠ {item.ready_reason}"
                    uptime = _format_uptime(item.deployed_at)
                    headline = (
                        f"● {item.name} — {item.display_name} ({item.version}) "
                        f"· {status} · runs {item.runs}"
                    )
                    if uptime:
                        headline += f" · up {uptime}"
                    self._post_line("tool", headline)
                    self._post_line(
                        "tool",
                        f"  ⎿ {client.agent_url(item.name)} · docs: "
                        f"{client.docs_url(item.name)}",
                    )
                self._post_line(
                    "tool",
                    "  ⎿ actions: /deployments undeploy <name> · reload <name> "
                    "· docs <name>",
                )
            elif action == "undeploy":
                await client.undeploy(name or "")
                self._post_line("assistant", f"🗑 Undeployed `{name}`.")
            elif action == "reload":
                reloaded, deployment = await client.reload(name or "")
                if reloaded:
                    self._post_line(
                        "assistant",
                        f"↻ Reloaded `{name}` → {deployment.version}.",
                    )
                else:
                    self._post_line(
                        "system",
                        f"Reload of `{name}` kept the previous version "
                        f"({deployment.version}) — the new one failed to load "
                        f"or preflight (see the hub log).",
                        severity="warning",
                    )
            elif action == "docs":
                url = client.docs_url(name or "")
                opened = open_url(url)
                suffix = " (opened in browser)" if opened else ""
                self._post_line("tool", f"docs: {url}{suffix}")
        except HubUnavailableError as exc:
            self._post_line("system", f"Hub unavailable: {exc}", severity="error")
        except HubError as exc:
            self._post_line("system", f"Hub error: {exc}", severity="error")

    async def _run_promote(self, raw: str) -> None:
        """Worker for ``/promote`` — gated release: latest → stable.

        Full gate (C1): artifact loads cleanly → a baseline run of the
        candidate SUCCEEDS (executed right here, recorded) → eval score beats
        the target channel's baseline when one exists. Any hard failure
        refuses the promote; ``--force`` bypasses the gate. Attached agents
        on the target channel hot-reload via their watcher.
        """
        ref, from_channel, to_channel, force = _parse_promote_args(raw)
        memory = getattr(self.app, "memory", None)
        client = getattr(memory, "client", None)
        if client is None:
            self._post_line(
                "system",
                "Memory is not configured — /promote needs the chain registry.",
                severity="error",
            )
            return

        entity_id = ref
        record = await self._fetch_chain_record(client, ref, from_channel)
        if record is None:
            match = await self._find_chain_by_name(client, ref)
            if match is None:
                self._post_line(
                    "system",
                    f"Could not resolve {ref!r} on channel {from_channel!r} — "
                    f"nothing to promote.",
                    severity="error",
                )
                return
            entity_id = match
        meta = dict(getattr(record, "meta", None) or {}) if record else {}
        display_name = str(meta.get("display_name") or entity_id)

        if force:
            self._post_line(
                "system",
                "⚠ --force: skipping the promotion gate.",
                severity="warning",
            )
        else:
            self._post_line(
                "tool", f"▶ Promotion gate for {display_name} "
                f"({from_channel} → {to_channel})…",
            )
            report = await gate_promotion(
                memory,
                self.app.config,
                entity_id,
                from_channel=from_channel,
                to_channel=to_channel,
            )
            for line in report.lines():
                self._post_line("tool", f"  ⎿ {line}")
            if not report.ok:
                self._post_line(
                    "system",
                    f"🛑 Promotion refused — the gate failed. Fix the issues "
                    f"above or use `/promote {ref} --force` to override.",
                    severity="error",
                )
                return

        try:
            await asyncio.to_thread(
                client.promote, entity_id, from_channel, to_channel
            )
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system", f"Promote failed: {exc}", severity="error"
            )
            return
        self._post_line(
            "assistant",
            f"⬆ Promoted **{display_name}**: `{from_channel}` → `{to_channel}`.",
        )
        self._post_line(
            "tool",
            "  ⎿ attached agents on this channel hot-reload automatically; "
            "`/rollback` repoints it back",
        )

    async def _run_rollback(self, raw: str) -> None:
        """Worker for ``/rollback`` — repoint a channel at an earlier version.

        Pin (not revert): the channel pointer moves to the target version, no
        new version is created — the honest rollback for deployments. Attached
        agents following the channel hot-reload via their watcher (A3).
        """
        ref, channel, to_version = _parse_rollback_args(raw)
        memory = getattr(self.app, "memory", None)
        client = getattr(memory, "client", None)
        if client is None:
            self._post_line(
                "system",
                "Memory is not configured — /rollback needs the chain registry.",
                severity="error",
            )
            return

        entity_id = ref
        record = await self._fetch_chain_record(client, ref, channel)
        if record is None:
            match = await self._find_chain_by_name(client, ref)
            if match is None:
                self._post_line(
                    "system",
                    f"Could not resolve {ref!r} on channel {channel!r} — "
                    f"nothing to roll back.",
                    severity="error",
                )
                return
            entity_id = match
            record = await self._fetch_chain_record(client, entity_id, channel)
            if record is None:
                self._post_line(
                    "system",
                    f"Chain {entity_id} has no version on channel {channel!r}.",
                    severity="error",
                )
                return

        meta = dict(getattr(record, "meta", None) or {})
        content = dict(getattr(record, "content", None) or {})
        display_name = str(
            meta.get("display_name") or content.get("name") or entity_id
        )
        current_number = getattr(record, "version_number", None)
        current_label = (
            f"v{current_number}" if current_number is not None else "current"
        )

        if to_version:
            target_id, target_label = to_version, to_version[:12]
        else:
            lister = getattr(client, "list_versions", None)
            try:
                versions = await asyncio.to_thread(lister, entity_id, "chain")
            except Exception as exc:  # noqa: BLE001
                self._post_line(
                    "system", f"Could not list versions: {exc}", severity="error"
                )
                return
            earlier = [
                v
                for v in versions or []
                if current_number is None
                or getattr(v, "version_number", 0) < current_number
            ]
            if not earlier:
                self._post_line(
                    "system",
                    f"{display_name}: {current_label} on {channel!r} is the "
                    f"earliest version — nothing to roll back to.",
                    severity="warning",
                )
                return
            previous = max(earlier, key=lambda v: getattr(v, "version_number", 0))
            target_id = previous.version_id
            target_label = f"v{previous.version_number}"

        try:
            await asyncio.to_thread(
                client.pin_channel, entity_id, channel, target_id
            )
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system", f"Rollback failed: {exc}", severity="error"
            )
            return

        self._post_line(
            "assistant",
            f"⏪ Rolled back **{display_name}** on `{channel}`: "
            f"{current_label} → {target_label}.",
        )
        self._post_line(
            "tool",
            "  ⎿ attached agents on this channel hot-reload automatically; "
            "`/deployments reload <name>` forces it",
        )

    async def _run_versions(self, raw: str) -> None:
        """Worker for ``/versions`` — list a chain's version history (C5).

        ``/versions <id|name>`` lists versions newest-first, marking which the
        ``latest`` / ``stable`` channels point at and showing each version's
        eval ``fitness_score`` when scored. ``/versions <ref> diff <vA> <vB>``
        prints the JSON-patch between two versions. Rollback is the existing
        ``/rollback <id> --to <vid>``; this screen surfaces the ids to pass.
        """
        ref, diff_pair = _parse_versions_args(raw)
        memory = getattr(self.app, "memory", None)
        client = getattr(memory, "client", None)
        if client is None:
            self._post_line(
                "system",
                "Memory is not configured — /versions needs the chain registry.",
                severity="error",
            )
            return

        entity_id = ref
        record = await self._fetch_chain_record(client, ref, "latest")
        if record is None:
            match = await self._find_chain_by_name(client, ref)
            if match is None:
                self._post_line(
                    "system",
                    f"Could not resolve {ref!r} to a saved chain.",
                    severity="error",
                )
                return
            entity_id = match

        if diff_pair is not None:
            await self._post_version_diff(client, entity_id, *diff_pair)
            return

        lister = getattr(client, "list_versions", None)
        try:
            versions = await asyncio.to_thread(lister, entity_id, "chain")
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system", f"Could not list versions: {exc}", severity="error"
            )
            return
        versions = list(versions or [])
        if not versions:
            self._post_line("system", f"No versions found for {entity_id}.")
            return

        # which version each channel points at (best-effort)
        channel_of: dict[str, list[str]] = {}
        for channel in ("latest", "stable"):
            rec = await self._fetch_chain_record(client, entity_id, channel)
            vid = str(getattr(rec, "version_id", "") or "") if rec else ""
            if vid:
                channel_of.setdefault(vid, []).append(channel)

        display = self._versions_display_name(record, entity_id)
        self._post_line("system", f"Versions of {display} ({entity_id}):")
        for version in sorted(
            versions, key=lambda v: getattr(v, "version_number", 0), reverse=True
        ):
            self._post_line("tool", _format_version_row(version, channel_of))
        self._post_line(
            "tool",
            "  ⎿ roll back: /rollback "
            f"{entity_id} --to <version-id> · diff: /versions {entity_id} diff <vA> <vB>",
        )

    @staticmethod
    def _versions_display_name(record: Any, entity_id: str) -> str:
        meta = dict(getattr(record, "meta", None) or {}) if record else {}
        content = dict(getattr(record, "content", None) or {}) if record else {}
        return str(meta.get("display_name") or content.get("name") or entity_id)

    async def _post_version_diff(
        self, client: Any, entity_id: str, from_version: str, to_version: str
    ) -> None:
        differ = getattr(client, "diff_versions", None)
        if not callable(differ):
            self._post_line(
                "system", "This Memory build has no version diff.", severity="warning"
            )
            return
        try:
            response = await asyncio.to_thread(
                differ, entity_id, from_version, to_version
            )
        except Exception as exc:  # noqa: BLE001
            self._post_line("system", f"Diff failed: {exc}", severity="error")
            return
        patch = getattr(response, "patch", None) or {}
        if not patch:
            self._post_line(
                "system", f"No differences between {from_version} and {to_version}."
            )
            return
        import json as _json

        self._post_line(
            "system", f"Diff {from_version[:12]} → {to_version[:12]}:"
        )
        self._post_line("tool", f"```json\n{_json.dumps(patch, indent=2, ensure_ascii=False)[:2000]}\n```")

    async def _fetch_chain_record(
        self, client: Any, entity_id: str, channel: str
    ) -> Any | None:
        """``get_chain_record`` that treats unknown-id/unpinned-channel as None."""
        getter = getattr(client, "get_chain_record", None)
        if not callable(getter):
            return None
        try:
            return await asyncio.to_thread(getter, entity_id, channel=channel)
        except Exception:  # noqa: BLE001 — unknown id / 404 / transport
            return None

    async def _find_chain_by_name(self, client: Any, query: str) -> str | None:
        """Unique name-search hit → entity_id; ambiguous/none → None (+ a hint)."""
        lister = getattr(client, "list_chains", None)
        if not callable(lister):
            return None
        try:
            hits = await asyncio.to_thread(lambda: lister(limit=5, q=query))
        except Exception:  # noqa: BLE001
            return None
        items = list(hits or [])
        if len(items) == 1:
            return str(getattr(items[0], "entity_id", "") or "") or None
        if len(items) > 1:
            names = ", ".join(
                str(getattr(item, "display_name", None) or getattr(item, "name", "?"))
                for item in items
            )
            self._post_line(
                "system",
                f"{query!r} matches several chains ({names}) — deploy by id.",
                severity="warning",
            )
        return None

    def _resolve_edit_chain_context(
        self,
        entity_id: str | None,
        chain: dict[str, Any] | None,
    ) -> tuple[str | None, dict[str, Any] | None, str | None, int | None, str]:
        """Prefer the active chain session (selected version) for ``/revise``.

        Returns ``(entity_id, chain_dict, parent_version_id,
        parent_version_number, display_name)``.
        """
        session = self._chain_session
        if session is None:
            name = ""
            if isinstance(chain, dict):
                name = str(chain.get("name") or "")
            return entity_id, chain, None, None, name
        payload = session.get("payload") or {}
        session_id = payload.get("chain_id")
        session_chain = payload.get("chain_dict")
        display_name = str(payload.get("display_name") or "")
        parent_vid = payload.get("version_id")
        parent_num = payload.get("version_number")
        if not isinstance(session_chain, dict) or not session_chain:
            return entity_id, chain, None, None, display_name
        if entity_id is None or (session_id and entity_id == session_id):
            return (
                str(session_id) if session_id else entity_id,
                session_chain,
                str(parent_vid) if parent_vid else None,
                int(parent_num) if parent_num is not None else None,
                display_name,
            )
        return entity_id, chain, None, None, display_name

    async def _run_edit(self, raw: str) -> None:
        """Worker for ``/revise`` — AI-edit a chain, preview, apply in-session.

        Interactive mode applies accepted edits to the active chain session
        (run / edit again / Save when ready). Library versioning happens only
        when the user presses Save on an already-saved chain with pending edits.
        Production mode still saves immediately after accept.
        """
        from care.generation import (
            GenerationError,
            build_mage_generator,
            run_edit,
        )
        from care.runtime.chain_edit_view import (
            format_revise_confirm_body,
            render_disambiguation_lines,
            render_edit_plan_lines,
            revise_result_has_changes,
        )
        from care.runtime.mage_poster import MagePoster
        from care.screens.confirm import ConfirmModal

        cfg = getattr(self.app, "config", None)
        if cfg is None:
            self._post_line("system", t("chat.misc.noAppConfig"), severity="error")
            return
        memory = getattr(self.app, "memory", None)

        # Parse "<id> <instruction>" (when the first token loads as a chain)
        # vs a bare "<instruction>" (MAGE resolves the chain by search).
        entity_id: str | None = None
        instruction = raw
        chain: dict[str, Any] | None = None
        parts = raw.split(maxsplit=1)
        if len(parts) == 2 and memory is not None:
            maybe_id, rest = parts
            loaded = await self._try_load_chain(memory, maybe_id)
            if loaded is not None:
                entity_id, instruction, chain = maybe_id, rest, loaded

        (
            entity_id,
            chain,
            parent_version_id,
            parent_version_number,
            session_display_name,
        ) = self._resolve_edit_chain_context(entity_id, chain)

        self._post_line("tool", f"▶ Revising {entity_id or '(resolving…)'}…")
        if parent_version_number is not None:
            self._post_line(
                "tool",
                f"  ⎿ {t('chat.chainBar.versionLabel', n=parent_version_number)}",
            )

        poster = MagePoster(
            self, token_counter=getattr(self.app, "token_counter", None),
        )
        try:
            generator = build_mage_generator(
                cfg, progress=poster, deployable=(self.mode == "production"),
            )
        except GenerationError as exc:
            self._post_line("system", f"Can't run agent chain generator: {exc}", severity="warning")
            return
        except Exception as exc:  # noqa: BLE001
            self._post_line("system", f"Agent chain generator setup failed: {exc}", severity="error")
            return

        try:
            result = await run_edit(
                generator, instruction,
                entity_id=entity_id, chain=chain, save=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._post_line("system", t("chat.revise.failed", error=exc), severity="error")
            return

        # Multiple chains matched a bare instruction — list them and stop.
        if getattr(result, "needs_disambiguation", False):
            for line in render_disambiguation_lines(result):
                self._post_line("tool", f"  ⎿ {line}")
            return

        if not revise_result_has_changes(result):
            self._post_line(
                "assistant",
                t("chat.revise.noChanges"),
            )
            return

        # Preview the plan (summary + edits + step delta) + a visual diff
        # of the chain's shape (added/changed/removed steps).
        for line in render_edit_plan_lines(result):
            self._post_line("tool", f"  ⎿ {line}")
        self._post_dag_diff(
            getattr(result, "before_chain_dict", None),
            getattr(result, "chain_dict", None),
        )

        target_id = entity_id or getattr(result, "entity_id", None)

        confirmed = await self.app.push_screen_wait(
            ConfirmModal(
                title=t("chat.revise.saveTitle"),
                body=format_revise_confirm_body(
                    result,
                    intro=t("chat.revise.saveBody"),
                    empty_preview=t("chat.revise.savePreviewEmpty"),
                ),
                confirm_label=t("chat.revise.saveLabel"),
                cancel_label=t("chat.revise.discardLabel"),
            )
        )
        if not confirmed:
            self._post_line("system", t("chat.revise.discarded"))
            return

        chain_updated = dict(result.chain_dict or {})
        display_name = (
            session_display_name
            or chain_updated.get("name")
            or t("chat.revise.defaultChainName")
        )

        if self.mode == "interactive":
            self._apply_chain_session_edit(
                chain_dict=chain_updated,
                display_name=str(display_name),
                run_task=instruction,
                edit_instruction=instruction,
            )
            return

        if memory is None:
            self._post_line(
                "system",
                t("chat.revise.noEntity"),
                severity="warning",
            )
            return

        await self._persist_revised_chain_to_memory(
            memory=memory,
            chain_dict=chain_updated,
            display_name=str(display_name),
            instruction=instruction,
            target_id=target_id,
            parent_version_id=parent_version_id,
            parent_version_number=parent_version_number,
        )

    async def _persist_revised_chain_to_memory(
        self,
        *,
        memory: Any,
        chain_dict: dict[str, Any],
        display_name: str,
        instruction: str,
        target_id: str | None,
        parent_version_id: str | None,
        parent_version_number: int | None,
    ) -> None:
        """Production / headless path — save a accepted revision immediately."""
        from care.screens.save_chain_name import SaveChainNameModal

        save_name = await self.app.push_screen_wait(
            SaveChainNameModal(
                default_name=display_name,
                title_key="chat.revise.saveNameTitle",
                hint_key="chat.revise.saveNameHint",
                confirm_key="chat.revise.saveNameConfirm",
            )
        )
        if not save_name:
            self._post_line("system", t("chat.revise.discarded"))
            return

        chain_to_save = dict(chain_dict)
        chain_to_save["name"] = save_name

        save_kwargs: dict[str, Any] = {
            "name": save_name,
            "query": instruction,
            "channel": "latest",
            "change_summary": instruction[:500],
        }
        if target_id:
            save_kwargs["entity_id"] = target_id
        if parent_version_id and target_id:
            save_kwargs["parent_version_id"] = parent_version_id

        try:
            new_id = await asyncio.to_thread(
                memory.save_chain,
                chain_to_save,
                **save_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            self._post_line("system", t("chat.artifacts.saveFailed", error=exc), severity="error")
            return

        saved_id = str(new_id or target_id or "")
        version_id, version_number = await self._fetch_latest_chain_version(
            saved_id,
        )
        if version_id:
            loaded = await self._load_chain_version_detail(
                saved_id, version_id,
            )
            if loaded and isinstance(loaded.get("content"), dict):
                chain_to_save = loaded["content"]
        if parent_version_number is not None and version_number is not None:
            self._post_line(
                "assistant",
                t(
                    "chat.revise.savedVersionFrom",
                    n=version_number,
                    name=save_name,
                    parent_n=parent_version_number,
                ),
            )
        elif version_number is not None:
            self._post_line(
                "assistant",
                t(
                    "chat.revise.savedVersion",
                    n=version_number,
                    name=save_name,
                ),
            )
        else:
            self._post_line(
                "assistant",
                t("chat.revise.savedVersionGeneric", id=saved_id),
            )
        self._refresh_chain_session_after_version_save(
            chain_dict=chain_to_save,
            chain_id=saved_id,
            display_name=save_name,
            version_id=version_id or None,
            version_number=version_number,
            run_task=instruction,
        )

    def _apply_chain_session_edit(
        self,
        *,
        chain_dict: dict[str, Any],
        display_name: str,
        run_task: str,
        edit_instruction: str,
    ) -> None:
        """Apply an accepted NL edit to the in-chat chain session (no Memory save)."""
        if self.mode != "interactive":
            return
        session = self._chain_session
        if session is None:
            self._begin_chain_session(
                chain_dict=chain_dict,
                display_name=display_name,
                task=run_task,
                label_key="chat.chainBar.titleUpdated",
            )
            session = self._chain_session
        else:
            payload = session["payload"]
            payload["chain_dict"] = chain_dict
            payload["display_name"] = display_name
            self._last_chain_action_payload = payload
            session["task"] = run_task.strip()
            session["last_edit_instruction"] = edit_instruction.strip()
            session["edit_dirty"] = True
            session["has_run"] = False
            self._mount_chain_action_bar(
                include_run=True,
                focus=False,
                label_key="chat.chainBar.titleUpdated",
            )
        if session is not None:
            session["last_edit_instruction"] = edit_instruction.strip()
            session["edit_dirty"] = True
        if session is not None and not session.get("_versioning_hint_shown"):
            session["_versioning_hint_shown"] = True
            self._toast_inline(t("chat.revise.versioningHint"))
        self._post_line("assistant", t("chat.revise.applied"))

    async def _run_generation(self, initial_task: str) -> None:
        """ReAct-style loop: generate → execute → check whether
        the answer signals continuation → maybe loop.

        Implements Phase 0 decision (b): the loop is driven by a
        *separate* MAGE call that reasons over the previous
        chain's result. Termination signal: absence of a
        ``[CONTINUE]`` / ``[CONTINUE: <next>]`` marker in the
        CARL answer (default = terminate, agent must opt in to
        more iterations).

        Production mode loops zero times — only the MAGE summary
        lands; Phase 3 will replace this with a save+baseline.
        Budget is bounded by `CARE_CHAT__LOOP_MAX_ITER` (default
        5) so a stuck agent can't drain LLM credits.
        """
        from care.generation import (
            GenerationError,
            build_mage_generator,
            run_generation,
        )
        from care.runtime.mage_poster import MagePoster

        await self._maybe_show_data_intro_from_file_ref()

        cfg = getattr(self.app, "config", None)
        if cfg is None:
            _log.error("generation aborted: no app.config available")
            self._post_line(
                "system", t("chat.misc.noAppConfig"), severity="error",
            )
            self._generating = False
            return

        # The RAW user prompt (before Ad-Hoc context-wrapping at
        # `_handle_task`) drives both reuse matching and the cache key —
        # `initial_task` may carry the conversation-context preamble for
        # follow-ups, which would never template-match.
        raw_task = self._pending_user_turn or initial_task

        # Modes redesign — reveal the transient pipeline strip above the
        # mode selector for the duration of this turn. Seeded all-pending;
        # `_update_pipeline_stage` lights cells as stages resolve, and the
        # `finally` hides it (covering cancel / error unwinds too).
        self._show_pipeline_strip(self._current_mode_spec())

        # Fast path (ad-hoc): reuse a structurally-matching chain from
        # earlier in this conversation — skips MAGE generation entirely so a
        # similar follow-up answers in a fraction of the time. Best-effort:
        # no match (or any failure) falls through to a normal generation.
        if self._current_mode_spec().followup == "reuse":
            try:
                if await self._try_reuse_chain(raw_task):
                    self._generating = False
                    self._loop_iteration = 0
                    self._pending_user_turn = None
                    self._refresh_status_bar()
                    return
            except asyncio.CancelledError:
                self._generating = False
                raise
            except Exception as exc:  # noqa: BLE001 — reuse never blocks a turn
                _log.warning("chain reuse attempt failed: %s", exc)

        mage_mode = getattr(cfg.mage, "mode", "deep") or "deep"
        # Phase 2 P2 — pass the app's running SessionTokenCounter so
        # per-stage usage folds into the cumulative total. The
        # per-iteration footer reads `.snapshot().total` below to
        # render a delta. Falls back to None for test scaffolding
        # without a counter slot.
        poster = MagePoster(
            self,
            token_counter=getattr(self.app, "token_counter", None),
        )
        _log.info(
            "building MAGE generator: model=%s mage_mode=%s",
            getattr(cfg.mage, "model", None) or "(default)",
            mage_mode,
        )
        try:
            generator = build_mage_generator(
                cfg, progress=poster, mode=mage_mode,  # type: ignore[arg-type]
                # Production chains get deployed + run headless → no `human_input`
                # (the agent must consume its input, not ask for it). Ad-hoc keeps it.
                deployable=(self.mode == "production"),
            )
        except GenerationError as exc:
            _log.warning("generation aborted: %s", exc)
            self._post_line(
                "system",
                f"Can't run agent chain generator: {exc}",
                severity="warning",
            )
            self._post_line(
                "system",
                "Tip: set CARE_MAGE__API_KEY in .env, or use "
                "/settings to configure a provider.",
                severity="warning",
            )
            self._generating = False
            return
        except Exception as exc:  # noqa: BLE001
            _log.error("MAGE setup failed: %s", exc, exc_info=True)
            self._post_line(
                "system", f"Agent chain generator setup failed: {exc}", severity="error",
            )
            self._generating = False
            return

        # §2 P1 — instrument MAGE's internal AsyncOpenAI client
        # so per-call `response.usage` lands in the session
        # token counter. The MagePoster.handle_stage_completed
        # path stays primary (reads
        # `MAGEResult.metadata.usage`); this wrap is the
        # fallback for providers that leave that field empty,
        # ensuring the iteration footer's
        # `in / out / total tok` reflects real MAGE
        # consumption instead of reading 0.
        token_counter = getattr(self.app, "token_counter", None)
        if token_counter is not None:
            try:
                from care.runtime.llm_client import (
                    instrument_mage_generator,
                )

                instrument_mage_generator(generator, token_counter)
            except Exception:  # noqa: BLE001
                # Best-effort — wrapping never blocks generation.
                pass

        # Tell MAGE which tools the executor will register (web_search,
        # fetch_url, calculator, …) so the planned `tool_name`s match
        # what actually resolves at run time. Defensive: priming is a
        # best-effort nudge — `None` falls back to prior behaviour.
        capabilities: Any = None
        try:
            from care.capability_priming import (
                build_capabilities_for_generation,
            )

            capabilities = build_capabilities_for_generation(
                cfg, query=initial_task,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("capability priming failed: %s", exc)

        # Bind the generation-only extras (today's-date preamble + the
        # primed capabilities) onto a plain ``(generator, task)`` callable
        # so :meth:`_generate_with_retry` stays a generic retry wrapper —
        # it must call its target with exactly those two positional args.
        # Standing user context (CARE.md + recalled LTM digest) — personalises
        # PLANNING so the chain is built aware of who the user is + prior asks.
        planning_user_context, _, _ = self._user_context_bundle(initial_task)

        async def gen_fn(gen: Any, tsk: str) -> Any:
            return await run_generation(
                gen, self._with_today_preamble(tsk),
                capabilities=capabilities, user_context=planning_user_context,
            )

        # Cross-cutting Telemetry — capture outcome metadata
        # for the `chat.task.completed` event emitted in
        # `finally`. Initial state is "interrupted"; if we
        # reach a terminal `return` we update before fall-through.
        task_started_at = time.perf_counter()
        outcome: dict[str, Any] = {
            "mode": self.mode,
            "status": "interrupted",
            "iterations": 0,
            "max_iterations": self._loop_max_iter(),
        }

        try:
            max_iter = outcome["max_iterations"]
            task = initial_task
            iteration = 0
            while True:
                iteration += 1
                self._loop_iteration = iteration
                outcome["iterations"] = iteration
                if iteration > 1:
                    self._post_line(
                        "tool",
                        f"↻ continuing — iteration {iteration}/{max_iter}",
                    )

                # Phase 2 P2 — iteration footer instrumentation.
                # Wall-clock around MAGE+CARL; token delta against
                # the running counter so cumulative MAGE generation
                # stages PLUS CARL LLM-step usage both surface.
                # Phase 8 P1 #8 — also snapshot the prompt /
                # completion split so the footer can render the
                # in/out breakdown + estimated USD cost.
                iter_started = time.perf_counter()
                tokens_before = self._snapshot_total_tokens()
                token_split_before = self._snapshot_token_split()
                # Phase 9 P2 — reset the raw-LLM-stream buffer
                # at the iteration boundary so each assistant
                # line's provenance captures only its own
                # iteration's response, not a cumulative blob
                # from earlier iterations.
                self._iteration_raw_response = []
                # Phase 9 P3 — reset the stream-preview widget
                # tracker too so the iteration-end cleanup
                # only removes the previews this iteration
                # actually spawned.
                self._iteration_stream_widget_ids = []

                try:
                    result = await self._generate_with_retry(
                        gen_fn, generator, task, iteration=iteration,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    _log.error(
                        "generation failed (iter %d): %s",
                        iteration, exc, exc_info=True,
                    )
                    self._update_pipeline_stage(
                        Stage.GENERATE, StageOutcome.FAILED,
                    )
                    self._post_line(
                        "system",
                        f"Generation failed after retries: {exc}",
                        severity="error",
                    )
                    outcome["status"] = "generation_failed"
                    return
                # GENERATE succeeded — light the strip's first cell.
                self._update_pipeline_stage(Stage.GENERATE, StageOutcome.DONE)

                summary_rows = self._format_result_summary_rows(result)
                _log.info(
                    "generation succeeded (iter %d); rows=%d",
                    iteration, len(summary_rows),
                )
                # §3 P0 — every freshly-generated chain lands
                # in the session artifact store so the header
                # pill, /artifacts screen, and save-all flow
                # all see it without per-flow plumbing.
                # Best-effort: store append failures are logged
                # but don't abort generation — the chain is
                # already rendered to the transcript.
                artifact_id = self._stash_generation_artifact(
                    initial_task=initial_task,
                    task=task,
                    iteration=iteration,
                    result=result,
                )
                # Render the metadata summary as `⎿`-prefixed
                # tool sub-rows under the freshly-finished
                # `✓ Describing steps` stage so it visually
                # groups with the stage trail above instead of
                # interrupting the chat as a standalone
                # assistant block.
                for row in summary_rows:
                    self._post_line("tool", f"  ⎿ {row}")

                # Inline chain actions after a successful generation.
                # Interactive gets the full chain-action bar (View / Save /
                # Run / Edit / Evolve / Finish) via the chain session — the
                # user drives the chain by hand. Production posts the lighter
                # `Read full` action only; the auto SAVE → BASELINE → EVOLVE
                # pipeline below backfills the saved chain_id onto the same
                # payload so the modal's evolve button reuses it.
                chain_dict = getattr(result, "chain_dict", None) or {}
                display_name = self._derive_chain_display_name(initial_task)
                if self.mode == "interactive":
                    self._begin_chain_session(
                        chain_dict=chain_dict,
                        display_name=display_name,
                        artifact_id=artifact_id,
                        task=task,
                    )
                else:
                    self._post_chain_actions(
                        chain_dict,
                        display_name=display_name,
                        artifact_id=artifact_id,
                    )

                # Production mode: save to Memory under a stable
                # chain_id + run one baseline + persist that
                # baseline as the first dataset entry. No ReAct
                # loop in Production — each task produces ONE
                # reproducible chain.
                if self.mode != "interactive":
                    # Modes redesign — drive SAVE → BASELINE → EVOLVE
                    # through the pipeline driver. The Production preset
                    # is all-`auto`, so this reproduces the legacy flow
                    # exactly (no confirm gates). Next plain prompt in
                    # Production rides the /revise path against the saved
                    # chain (see `_handle_task`).
                    spec = resolve_mode_spec(
                        self.mode, getattr(getattr(self.app, "config", None),
                                           "chat", None),
                    )
                    stage_outcomes = await self._drive_production_pipeline(
                        spec, task=initial_task, mage_result=result,
                    )
                    outcome["status"] = (
                        "production_saved"
                        if stage_outcomes[Stage.SAVE] is StageOutcome.DONE
                        else "production_dedup_or_failed"
                    )
                    return

                # Deterministic skill selection — if the user asked for a file
                # (or named a skill via `/skill` / «используй скилл»), guarantee
                # the chain uses the matching `agent_skill` step regardless of
                # whether the planner LLM chose one. See care.skill_enforcement.
                from care.skill_enforcement import (
                    detect_requested_skill,
                    ensure_skill_step,
                )

                requested_skill = detect_requested_skill(initial_task)

                # Interactive RUN gate. Ask once, before the first
                # execution of the turn (ReAct continuations don't
                # re-prompt). `auto` (or a non-interactive spec) returns
                # True immediately. Declining skips execution entirely.
                if iteration == 1 and not await self._confirm_interactive_run(rich=True):
                    outcome["status"] = "chain_finished"
                    return

                # Regenerate-on-execution-failure: a chain can generate fine
                # yet fail at run time (a step errors / a tool keeps 400ing).
                # Rather than dead-end the turn, regenerate a fresh chain —
                # nudging the planner to avoid the failing approach — up to
                # `_generation_max_attempts()` (default 3, the same knob
                # generation transient-retry uses). Cancellation re-raises.
                exec_max = self._generation_max_attempts()
                run_result = None
                for exec_attempt in range(1, exec_max + 1):
                    chain_dict = getattr(result, "chain_dict", None) or {}
                    if not chain_dict:
                        _log.warning(
                            "MAGE returned no chain_dict (iter %d) — "
                            "skipping execution",
                            iteration,
                        )
                        self._post_line(
                            "system",
                            "Agent chain generator returned no chain to execute.",
                            severity="warning",
                        )
                        outcome["status"] = "no_chain"
                        return

                    skill_chain = chain_dict
                    if requested_skill:
                        self._post_line("tool", f"🛠 skill: {requested_skill}")
                        skill_chain = ensure_skill_step(
                            chain_dict, requested_skill,
                        )

                    # Re-assert "thinking" for the execution phase. The RUN
                    # confirm (rich chain-action bar) hid the spinner while
                    # waiting on the user; the `_handle_task` worker never
                    # changed state, so `_refresh_spinner` didn't fire on its
                    # own — without this, the active RUN stage marker sits
                    # static (`○ Run?`) instead of animating while the chain
                    # executes. We're unconditionally executing here, so turn
                    # it on directly.
                    self._set_spinner_visible(True)
                    self._post_line("tool", t("chat.trace.executingChain"))
                    run_result = await self._execute_chain_interactive(
                        task=task, chain_dict=skill_chain,
                    )
                    if run_result is not None:
                        # RUN succeeded — light the strip's Run cell.
                        self._update_pipeline_stage(
                            Stage.RUN, StageOutcome.DONE,
                        )
                        break  # executed (CARL may still flag per-step issues)

                    # Execution failed (the error was already surfaced by
                    # `_execute_chain_interactive`). Regenerate unless we're out of
                    # attempts.
                    if exec_attempt >= exec_max:
                        outcome["status"] = "execution_failed"
                        if self.mode == "interactive" and iteration == 1:
                            self._move_chain_action_bar_to_end()
                        return
                    self._post_line(
                        "system",
                        "↻ chain execution failed — regenerating a different "
                        f"approach (attempt {exec_attempt + 1}/{exec_max})…",
                        severity="warning",
                    )
                    regen_task = (
                        f"{task}\n\n[The previously generated chain failed to "
                        "execute. Generate a different, simpler approach that "
                        "avoids the step that failed.]"
                    )
                    try:
                        result = await self._generate_with_retry(
                            gen_fn, generator, regen_task, iteration=iteration,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        _log.error(
                            "regeneration after exec failure failed "
                            "(iter %d): %s",
                            iteration, exc, exc_info=True,
                        )
                        self._post_line(
                            "system",
                            f"Regeneration failed after retries: {exc}",
                            severity="error",
                        )
                        outcome["status"] = "generation_failed"
                        return
                    for row in self._format_result_summary_rows(result):
                        self._post_line("tool", f"  ⎿ {row}")
                    self._update_chain_session_chain(
                        getattr(result, "chain_dict", None) or {},
                    )

                # A file-producing skill that wrote nothing (weak model couldn't
                # drive the sandbox tools). Build the file deterministically from
                # the generated text so the user still gets their artifact.
                if requested_skill:
                    self._fallback_build_file_if_missing(
                        run_result, skill=requested_skill, task=task,
                    )

                # Ad-Hoc post-processing: when the chain produced
                # ≥2 successful steps, fold their outputs into a
                # single coherent user-facing answer via one LLM
                # call. Otherwise the legacy terminal-step
                # extraction wins and we'd surface only the last
                # step (often just "strengthen the conclusion"
                # rather than the merged essay).
                synthesised = await self._synthesise_user_answer(
                    task=task, run_result=run_result,
                )
                if synthesised is not None:
                    answer = synthesised
                else:
                    answer = self._format_carl_result(run_result)
                visible_answer = self._strip_continuation_marker(answer)
                # Phase 8 P2 #18 — capture provenance for the
                # Ctrl+I inspector. Numbers come from the same
                # snapshots that drive the iteration footer; the
                # raw deltas live on the ChatLine so the user can
                # introspect any past assistant reply.
                provenance = self._capture_provenance(
                    iteration=iteration,
                    started_at=iter_started,
                    tokens_before=tokens_before,
                    token_split_before=token_split_before,
                )
                # Phase 9 P3 — sweep the streaming tool-line
                # previews this iteration spawned before
                # mounting the assistant Markdown widget. The
                # final formatted answer carries the full
                # body; leaving the truncated preview line
                # behind would just duplicate content.
                self._remove_iteration_stream_previews()
                self._post_line(
                    "assistant",
                    visible_answer,
                    provenance=provenance,
                )
                # Capture the turn into the Ad-Hoc context so
                # follow-up prompts ("а где эссе?") can reference
                # it. Only on the first iteration of a task —
                # ReAct continuations all belong to the same
                # logical turn. /new wipes the history.
                if (
                    self._current_mode_spec().followup == "reuse"
                    and iteration == 1
                    and self._pending_user_turn is not None
                ):
                    self._record_interactive_turn(
                        self._pending_user_turn, visible_answer,
                    )
                    self._pending_user_turn = None

                # Cache this successful, parameterized chain so a similar
                # follow-up reuses it without a fresh MAGE generation. Only
                # the first iteration's chain (ReAct continuations belong to
                # the same logical turn). See _try_reuse_chain.
                if (
                    self._current_mode_spec().followup == "reuse"
                    and iteration == 1
                ):
                    self._cache_chain_for_reuse(
                        raw_task, getattr(result, "chain_dict", None),
                    )

                # Phase 2 P2 — per-iteration footer. Renders the
                # iteration label, wall-clock elapsed, and tokens
                # spent (when the counter is wired). Skipped in
                # Production above (early return).
                # Phase 8 P1 #8 — pass the prompt/completion
                # split snapshot so the footer can also render
                # in/out breakdown + estimated USD cost when the
                # active MAGE model is in the pricing table.
                self._post_iteration_footer(
                    iteration=iteration,
                    started_at=iter_started,
                    tokens_before=tokens_before,
                    token_split_before=token_split_before,
                )
                if self.mode == "interactive" and iteration == 1:
                    self._mark_chain_session_ran()

                # ReAct termination check.
                next_request = self._extract_continuation(answer)
                if next_request is None:
                    outcome["status"] = "completed"
                    # Turn complete — run the post-turn save-decision over the
                    # user's request + the final answer (persist durable facts
                    # to LTM; best-effort, gated by ltm_autosave).
                    await self._maybe_save_ltm(
                        query=initial_task, answer=visible_answer,
                    )
                    return  # terminal — agent didn't ask for more

                if iteration >= max_iter:
                    _log.warning(
                        "loop budget exhausted at iteration %d; "
                        "agent requested continuation",
                        iteration,
                    )
                    self._post_line(
                        "system",
                        (
                            f"Reached loop budget "
                            f"({max_iter} iterations) — stopping. "
                            "Override via CARE_CHAT__LOOP_MAX_ITER."
                        ),
                        severity="warning",
                    )
                    outcome["status"] = "budget_exhausted"
                    return

                task = self._build_followup_task(
                    initial_task=initial_task,
                    prev_answer=visible_answer,
                    next_request=next_request,
                )
        finally:
            # Keep the pipeline strip (`◆ Generate → ◆ Run`) on screen after
            # an interactive turn so the user keeps seeing the outcome until
            # they click Finish (`_finish_chain_session`) or send a new
            # message (the next turn re-seeds it via `_show_pipeline_strip`).
            # When there's no live chain session to anchor it — production
            # mode, or an error before the session began — collapse it now.
            if self._chain_session is None:
                self._hide_pipeline_strip()
            else:
                # The worker is done: drop the "thinking…" tail but leave the
                # cells visible.
                self._set_spinner_visible(False)
            # Always reset so the next prompt is accepted, even
            # if a cancel exception unwound us mid-loop.
            self._generating = False
            self._loop_iteration = 0
            # Drop the staged user-turn buffer so a generation
            # that aborted before the assistant reply (cancel,
            # retries exhausted) doesn't leave it dangling and
            # mis-attribute the next prompt's context capture.
            self._pending_user_turn = None
            # Performance cross-cutting — kick the status bar
            # so the new token total surfaces immediately
            # instead of waiting for the next 5s tick.
            self._refresh_status_bar()
            # Telemetry cross-cutting — emit a single
            # `chat.task.completed` event with the outcome +
            # duration so a Langfuse-style backend can build
            # per-mode success-rate dashboards. Token total +
            # latency rideshares so the event is self-contained
            # (no need to join against a separate counter
            # stream).
            outcome["duration_seconds"] = round(
                time.perf_counter() - task_started_at, 3,
            )
            outcome["tokens_total"] = (
                self._snapshot_total_tokens() or 0
            )
            self._emit_telemetry("chat.task.completed", outcome)

    # ------------------------------------------------------------------
    # ReAct loop helpers
    # ------------------------------------------------------------------

    def _loop_max_iter(self) -> int:
        """Per Phase-0 decision: max 5 iterations by default,
        overridable via ``CARE_CHAT__LOOP_MAX_ITER``. Clamped to
        at least 1 so the env can't accidentally disable loops."""
        import os

        raw = (os.environ.get("CARE_CHAT__LOOP_MAX_ITER") or "").strip()
        if not raw:
            return 5
        try:
            n = int(raw)
        except ValueError:
            return 5
        return max(1, n)

    # ------------------------------------------------------------------
    # Iteration footer (Phase 2 P2)
    # ------------------------------------------------------------------

    def _snapshot_total_tokens(self) -> int | None:
        """Read the running cumulative token total from the app's
        :class:`SessionTokenCounter`. Returns ``None`` when the
        counter is missing (test scaffolding) or doesn't expose
        the expected ``snapshot().total`` API so the footer can
        degrade gracefully instead of crashing the loop."""
        counter = getattr(self.app, "token_counter", None)
        if counter is None:
            return None
        try:
            snap = counter.snapshot()
            total = getattr(snap, "total", None)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(total, int):
            return None
        return total

    def _snapshot_token_split(self) -> tuple[int, int, int] | None:
        """Phase 8 P1 #8 — read the input/output split from the
        session counter. Returns ``(prompt, completion, total)``
        as ints, or ``None`` when the counter is missing or the
        snapshot is malformed. Used by the richer iteration
        footer to surface the in/out breakdown alongside the
        total + estimated USD cost."""
        counter = getattr(self.app, "token_counter", None)
        if counter is None:
            return None
        try:
            snap = counter.snapshot()
            prompt = getattr(snap, "prompt", None)
            completion = getattr(snap, "completion", None)
            total = getattr(snap, "total", None)
        except Exception:  # noqa: BLE001
            return None
        if (
            not isinstance(prompt, int)
            or not isinstance(completion, int)
            or not isinstance(total, int)
        ):
            return None
        return prompt, completion, total

    def _resolve_active_model(self) -> str | None:
        """Read the MAGE model id from app config defensively.
        Returns ``None`` for test hosts without `config.mage.model`
        so the footer's model + cost segments degrade cleanly
        rather than crash."""
        cfg = getattr(self.app, "config", None)
        if cfg is None:
            return None
        mage_cfg = getattr(cfg, "mage", None)
        if mage_cfg is None:
            return None
        model = getattr(mage_cfg, "model", None)
        if not isinstance(model, str) or not model.strip():
            return None
        return model.strip()

    @staticmethod
    def _format_iteration_footer(
        iteration: int,
        elapsed_seconds: float,
        tokens_used: int | None,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        model: str | None = None,
        cost: float | None = None,
    ) -> str:
        """Project the per-iteration footer into the displayed
        ``tool`` line. Shape (segments joined with ``" · "``):

        - ``iter <n>`` — always.
        - ``<Xs>`` — wall-clock duration (one decimal under 10s,
          rounded int at / above).
        - ``in <P> / out <C> / total <T> tok`` — when prompt
          and completion deltas are both available (Phase 8 P1
          #8 richer split); falls back to ``<T> tok`` when only
          the total is available; segment omitted entirely when
          no token data exists at all.
        - ``$<cost>`` — estimated USD cost (Phase 8 P1 #8); only
          present when the active model resolves to a row in
          `care.runtime.pricing._PRICING_TABLE`.
        - ``<model>`` — the resolved MAGE model id; omitted when
          config doesn't surface one.

        Negative deltas (counter reset mid-loop) are coerced to
        ``0`` rather than rendered as a negative — a fresh
        ``0 tok`` is a cleaner signal."""
        from care.runtime.pricing import format_cost

        elapsed_seconds = max(0.0, float(elapsed_seconds))
        if elapsed_seconds < 10:
            duration = f"{elapsed_seconds:.1f}s"
        else:
            duration = f"{int(round(elapsed_seconds))}s"
        parts = [f"iter {iteration}", duration]

        # Token segment — prefer the split when both halves are
        # available; degrade to combined-only; omit when neither.
        if prompt_tokens is not None and completion_tokens is not None:
            prompt = max(0, int(prompt_tokens))
            completion = max(0, int(completion_tokens))
            total = max(0, int(tokens_used)) if tokens_used is not None else (
                prompt + completion
            )
            parts.append(
                f"in {prompt} / out {completion} / total {total} tok",
            )
        elif tokens_used is not None:
            tokens = max(0, int(tokens_used))
            parts.append(f"{tokens} tok")

        cost_segment = format_cost(cost)
        if cost_segment:
            parts.append(cost_segment)

        if model:
            parts.append(model.strip())

        return " · ".join(parts)

    def _post_iteration_footer(
        self,
        *,
        iteration: int,
        started_at: float,
        tokens_before: int | None,
        token_split_before: tuple[int, int, int] | None = None,
    ) -> None:
        """Compute the elapsed + token deltas for a finished
        iteration and post the rendered footer as a ``tool`` line.
        Best-effort — never raises into the calling loop.

        ``tokens_before`` / ``token_split_before`` of ``None``
        means the counter was unavailable when the iteration
        started; in that case the footer skips the token segment
        regardless of whether the counter materialised
        mid-iteration (avoids reporting a bogus delta against
        ``0``).

        Phase 8 P1 #8 — when the counter exposes the prompt /
        completion split AND the active model has a known
        pricing row, the footer also surfaces the in/out
        breakdown + estimated USD cost + model id. Falls back
        to the legacy ``iter N · Xs · Y tok`` form for unknown
        models or counters that only emit a total."""
        from care.runtime.pricing import estimate_cost

        elapsed = time.perf_counter() - started_at
        tokens_used: int | None = None
        prompt_delta: int | None = None
        completion_delta: int | None = None
        if tokens_before is not None:
            tokens_after = self._snapshot_total_tokens()
            if tokens_after is not None:
                tokens_used = tokens_after - tokens_before
        if token_split_before is not None:
            split_after = self._snapshot_token_split()
            if split_after is not None:
                prompt_after, completion_after, _ = split_after
                prompt_before, completion_before, _ = token_split_before
                prompt_delta = prompt_after - prompt_before
                completion_delta = completion_after - completion_before
        model = self._resolve_active_model()
        cost: float | None = None
        if (
            model is not None
            and prompt_delta is not None
            and completion_delta is not None
        ):
            cost = estimate_cost(
                model,
                max(0, prompt_delta),
                max(0, completion_delta),
            )
        line = self._format_iteration_footer(
            iteration=iteration,
            elapsed_seconds=elapsed,
            tokens_used=tokens_used,
            prompt_tokens=prompt_delta,
            completion_tokens=completion_delta,
            model=model,
            cost=cost,
        )
        # `chat-line-iter-footer` paints the row in `$text-muted`
        # so the per-iteration metadata (`iter 1 · 38s · in 889 / out 13 · …`)
        # reads as quieter chrome than the warning-yellow tool
        # chatter above it.
        self._post_line(
            "tool", line, extra_class="chat-line-iter-footer",
        )

    # Sentinel pattern the agent emits to request another
    # iteration. `[CONTINUE]` on its own loops with the previous
    # answer in scope; `[CONTINUE: <next request>]` swaps in a
    # fresh task description. Case-insensitive so the agent
    # doesn't have to remember capitalisation.
    _CONTINUATION_RE = __import__("re").compile(
        r"\[CONTINUE(?:\s*:\s*([^\]]+))?\]",
        __import__("re").IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Provenance capture + inspector (Phase 8 P2 #18)
    # ------------------------------------------------------------------

    def _capture_provenance(
        self,
        *,
        iteration: int,
        started_at: float,
        tokens_before: int | None,
        token_split_before: tuple[int, int, int] | None,
    ) -> dict[str, Any]:
        """Phase 8 P2 #18 — snapshot the LLM-call metadata that
        produced the assistant line about to land. Reuses the
        same snapshot machinery that drives the iteration
        footer so the inspector and the footer can never
        disagree. Returns a dict (never ``None``) so the
        downstream `_post_line` can simply attach it; missing
        fields stay absent rather than carry ``None``."""
        from care.runtime.pricing import estimate_cost

        provenance: dict[str, Any] = {
            "iteration": iteration,
            "duration_seconds": round(
                max(0.0, time.perf_counter() - started_at), 3,
            ),
            "mode": self.mode,
        }
        model = self._resolve_active_model()
        if model:
            provenance["model"] = model
        if tokens_before is not None:
            tokens_after = self._snapshot_total_tokens()
            if tokens_after is not None:
                provenance["tokens_total"] = max(
                    0, tokens_after - tokens_before,
                )
        if token_split_before is not None:
            split_after = self._snapshot_token_split()
            if split_after is not None:
                prompt_after, completion_after, _ = split_after
                prompt_before, completion_before, _ = token_split_before
                provenance["prompt_tokens"] = max(
                    0, prompt_after - prompt_before,
                )
                provenance["completion_tokens"] = max(
                    0, completion_after - completion_before,
                )
        if (
            model is not None
            and "prompt_tokens" in provenance
            and "completion_tokens" in provenance
        ):
            cost = estimate_cost(
                model,
                provenance["prompt_tokens"],
                provenance["completion_tokens"],
            )
            if cost is not None:
                provenance["cost_usd"] = cost
        # Phase 9 P2 — attach the captured raw LLM response so
        # the Ctrl+I inspector can render the full reproducible
        # stream. Trimmed to a bound so a runaway agent can't
        # poison memory: full body lives in app log anyway.
        raw_response = "".join(self._iteration_raw_response)
        if raw_response:
            provenance["raw_response"] = self._truncate_for_provenance(
                raw_response,
            )
        return provenance

    _RAW_CAPTURE_MAX_CHARS: int = 16_000
    _RAW_CAPTURE_ELLIPSIS: str = (
        "\n…\n[truncated — see logs/care-app-*.log for full body]\n"
    )

    @classmethod
    def _truncate_for_provenance(cls, body: str) -> str:
        """Bound the raw-stream body so a runaway iteration
        can't bloat the provenance dict (and the per-line
        memory cost). Keeps the head + tail (so users can
        see both the system prompt edge and the final
        completion) with a clear truncation marker between
        when the body is over budget."""
        if len(body) <= cls._RAW_CAPTURE_MAX_CHARS:
            return body
        head_budget = cls._RAW_CAPTURE_MAX_CHARS // 2
        tail_budget = cls._RAW_CAPTURE_MAX_CHARS - head_budget
        return (
            body[:head_budget]
            + cls._RAW_CAPTURE_ELLIPSIS
            + body[-tail_budget:]
        )

    def action_inspect_last(self) -> None:
        """``Ctrl+I`` — surface the provenance of the most
        recent assistant line as a system message. Falls
        through to a friendly "no provenance recorded" notice
        when the last assistant line was a canned response
        (welcome banner, /help output, /resume rehydration —
        none of those carry per-call metadata)."""
        target: ChatLine | None = next(
            (
                line for line in reversed(self._lines)
                if line.role == "assistant"
            ),
            None,
        )
        if target is None:
            self._post_line(
                "system",
                "No assistant reply to inspect yet.",
                severity="warning",
            )
            return
        if not target.provenance:
            self._post_line(
                "system",
                "No provenance recorded for the last reply "
                "(canned response or rehydrated session).",
            )
            return
        self._post_line(
            "system",
            self._format_provenance(target.provenance),
        )

    # ------------------------------------------------------------------
    # Reactions (Phase 8 P2 #15)
    # ------------------------------------------------------------------

    _REACTION_MARKERS: dict[str, str] = {
        "up": "👍",
        "down": "👎",
    }

    def action_react_up(self) -> None:
        """``Ctrl+T`` — toggle 👍 on the most recent assistant
        line. Toggling the SAME reaction off clears it;
        toggling a DIFFERENT reaction swaps in place."""
        self._toggle_reaction_on_last_assistant("up")

    def action_react_down(self) -> None:
        """``Ctrl+Shift+T`` — toggle 👎 on the most recent
        assistant line. See :meth:`action_react_up` for
        toggle / swap semantics."""
        self._toggle_reaction_on_last_assistant("down")

    def _toggle_reaction_on_last_assistant(
        self, reaction: Reaction,
    ) -> None:
        """Find the latest assistant line, toggle / swap its
        reaction, then re-render the mounted widget so the
        marker shows. Posts a warning when no assistant line
        exists yet so the user gets feedback about the
        attempted gesture."""
        target_idx: int | None = None
        target_line: ChatLine | None = None
        for idx in range(len(self._lines) - 1, -1, -1):
            if self._lines[idx].role == "assistant":
                target_idx = idx
                target_line = self._lines[idx]
                break
        if target_line is None or target_idx is None:
            self._post_line(
                "system",
                "No assistant reply to react to yet.",
                severity="warning",
            )
            return
        # Toggle (same reaction) or swap (different).
        if target_line.reaction == reaction:
            target_line.reaction = None
        else:
            target_line.reaction = reaction
        self._rerender_assistant_line(target_idx)

    def _rerender_assistant_line(self, idx: int) -> None:
        """Re-render the assistant widget at ``idx`` (0-based
        into `_lines`, +1 for the widget id) so the new
        reaction marker shows immediately. Best-effort: queries
        defensively, no-ops when the widget can't be found."""
        line = self._lines[idx]
        widget_id = f"#chat-line-{idx + 1}"
        try:
            widget = self.query_one(widget_id)
        except Exception:
            return
        if isinstance(widget, Markdown):
            try:
                widget.update(
                    self._format_line_as_markdown_for_widget(line),
                )
            except Exception:
                pass
        elif isinstance(widget, Static):
            try:
                widget.update(self._format_line_for_render(line))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Code-block copy (Phase 8 P2 #17)
    # ------------------------------------------------------------------

    # Greedy fenced-block matcher. Captures the optional language
    # hint (group 1) and the block body (group 2). DOTALL so the
    # body can span lines; non-greedy so consecutive blocks don't
    # merge into one match.
    _CODE_FENCE_RE = __import__("re").compile(
        r"```([a-zA-Z0-9_+.-]*)\s*\n(.*?)\n```",
        __import__("re").DOTALL,
    )

    @classmethod
    def _extract_code_blocks(
        cls, body: str,
    ) -> list[tuple[str, str]]:
        """Phase 8 P2 #17 — pull every fenced code block out of
        a Markdown body. Returns ``[(language, content), ...]``
        in document order. Language defaults to empty string
        when the fence carried no hint."""
        return [
            (m.group(1) or "", m.group(2))
            for m in cls._CODE_FENCE_RE.finditer(body or "")
        ]

    def action_copy_code_block(self) -> None:
        """``Ctrl+B`` — copy the first fenced code block from
        the most recent assistant reply to the system
        clipboard. Surfaces friendly warnings when there's no
        assistant reply yet, when the latest reply carries no
        code blocks, or when the clipboard channel isn't
        available."""
        from care.runtime.clipboard import copy_text

        target = next(
            (
                line for line in reversed(self._lines)
                if line.role == "assistant"
            ),
            None,
        )
        if target is None:
            self._post_line(
                "system",
                "No assistant reply to copy code from yet.",
                severity="warning",
            )
            return
        blocks = self._extract_code_blocks(target.text)
        if not blocks:
            self._post_line(
                "system",
                "No fenced code block in the last reply.",
                severity="warning",
            )
            return
        language, body = blocks[0]
        if not copy_text(self.app, body):
            self._post_line(
                "system",
                t("chat.clipboard.copyFailed"),
                severity="warning",
            )
            return
        lines = len(body.splitlines())
        scope = f"({language})" if language else "(plain)"
        self._post_line(
            "system",
            f"Copied first code block {scope} — {lines} line"
            f"{'s' if lines != 1 else ''} "
            f"({len(blocks)} block{'s' if len(blocks) != 1 else ''}"
            " in the reply).",
        )

    # ------------------------------------------------------------------
    # /blocks — list + act on any fenced code block (Phase 9 P2)
    # ------------------------------------------------------------------

    _BLOCKS_PREVIEW_MAX_CHARS: int = 60

    def _collect_all_code_blocks(
        self,
    ) -> list[tuple[str, str, str]]:
        """Walk the transcript and return ``[(role, lang, body),
        ...]`` in document order for every fenced block found
        in an assistant or system line. User / tool bodies are
        skipped — those render as plain text (the user's
        verbatim characters), so they aren't "code blocks" in
        the Markdown sense even when they look like one."""
        out: list[tuple[str, str, str]] = []
        for line in self._lines:
            if line.role not in self._MARKDOWN_ROLES:
                continue
            for lang, body in self._extract_code_blocks(line.text):
                out.append((line.role, lang, body))
        return out

    def _handle_blocks_command(self, arg: str) -> None:
        """``/blocks``                → list every fenced code
        block in the transcript with stable 1-based indices.
        ``/blocks copy N``           → copy the N-th block to
        the system clipboard. ``/blocks save N <path>``   →
        write the N-th block to ``path`` (parent dirs are
        created). Replaces the Phase 8 P2 #17 "first block
        only" limitation — users can now act on any block."""
        from care.runtime.clipboard import copy_text

        all_blocks = self._collect_all_code_blocks()

        tokens = (arg or "").split(maxsplit=2)
        if not tokens:
            self._render_blocks_listing(all_blocks)
            return

        sub = tokens[0].lower()

        if sub == "copy":
            if len(tokens) < 2:
                self._post_line(
                    "system",
                    "Usage: `/blocks copy <N>`. Run `/blocks` "
                    "to see the indexed list.",
                    severity="warning",
                )
                return
            target = self._resolve_block_index(
                tokens[1], all_blocks,
            )
            if target is None:
                return
            role, lang, body = target
            if not copy_text(self.app, body):
                self._post_line(
                    "system",
                    t("chat.clipboard.copyFailed"),
                    severity="warning",
                )
                return
            line_count = len(body.splitlines())
            scope = f"({lang})" if lang else "(plain)"
            self._post_line(
                "system",
                f"Copied block #{tokens[1]} {scope} — "
                f"{line_count} line"
                f"{'s' if line_count != 1 else ''}.",
            )
            return

        if sub == "save":
            if len(tokens) < 3:
                self._post_line(
                    "system",
                    "Usage: `/blocks save <N> <path>`.",
                    severity="warning",
                )
                return
            target = self._resolve_block_index(
                tokens[1], all_blocks,
            )
            if target is None:
                return
            _role, _lang, body = target
            dest = Path(tokens[2]).expanduser()
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(body, encoding="utf-8")
            except OSError as exc:
                self._post_line(
                    "system",
                    f"Couldn't save block to `{dest}`: {exc}",
                    severity="error",
                )
                return
            line_count = len(body.splitlines())
            self._post_line(
                "system",
                f"Saved block #{tokens[1]} → `{dest}` "
                f"({line_count} line"
                f"{'s' if line_count != 1 else ''}).",
            )
            return

        self._post_line(
            "system",
            f"Unknown /blocks sub-command `{sub}`. "
            "Use `/blocks`, `/blocks copy <N>`, or "
            "`/blocks save <N> <path>`.",
            severity="warning",
        )

    def _resolve_block_index(
        self,
        raw: str,
        all_blocks: list[tuple[str, str, str]],
    ) -> tuple[str, str, str] | None:
        """Parse + validate a 1-based block index. Posts a
        warning and returns ``None`` on bad input so the
        caller can early-return uniformly."""
        try:
            n = int(raw)
        except ValueError:
            self._post_line(
                "system",
                f"Block index must be an integer (got `{raw}`).",
                severity="warning",
            )
            return None
        if not all_blocks:
            self._post_line(
                "system",
                "No fenced code blocks in the transcript yet.",
                severity="warning",
            )
            return None
        if n < 1 or n > len(all_blocks):
            self._post_line(
                "system",
                f"Block index {n} out of range (have "
                f"{len(all_blocks)} block"
                f"{'s' if len(all_blocks) != 1 else ''}).",
                severity="warning",
            )
            return None
        return all_blocks[n - 1]

    def _render_blocks_listing(
        self,
        all_blocks: list[tuple[str, str, str]],
    ) -> None:
        """Post a single system line listing every fenced
        block with index, role, language, and a short
        preview. Suggests the action sub-commands so the user
        learns the workflow on first run."""
        if not all_blocks:
            self._post_line(
                "system",
                "No fenced code blocks in the transcript yet. "
                "Ask the agent for code and they'll appear here.",
                severity="warning",
            )
            return
        rows: list[str] = ["**Code blocks in the transcript:**"]
        for idx, (role, lang, body) in enumerate(all_blocks, start=1):
            line_count = len(body.splitlines())
            first_line = body.splitlines()[0] if body else ""
            preview = first_line
            if len(preview) > self._BLOCKS_PREVIEW_MAX_CHARS:
                preview = (
                    preview[: self._BLOCKS_PREVIEW_MAX_CHARS - 1]
                    .rstrip() + "…"
                )
            scope = lang or "plain"
            rows.append(
                f"  {idx}. [{role}/{scope}, {line_count} line"
                f"{'s' if line_count != 1 else ''}] "
                f"`{preview}`"
            )
        rows.append(
            "\nActions: `/blocks copy <N>`, "
            "`/blocks save <N> <path>`.",
        )
        self._post_line("system", "\n".join(rows))

    # ------------------------------------------------------------------
    # /branch — transcript checkpoints (Phase 9 P3)
    # ------------------------------------------------------------------

    _BRANCH_FILENAME_RE = __import__("re").compile(
        r"^[A-Za-z0-9_\-]+$",
    )

    @staticmethod
    def _branches_dir() -> Path:
        """Resolve the directory that holds branch sidecars.
        Honours ``CARE_CHAT__BRANCHES_DIR`` (test fixture
        redirects to ``tmp_path``); defaults to
        ``$XDG_STATE_HOME/care/branches`` or
        ``~/.local/state/care/branches``."""
        import os

        override = (
            os.environ.get("CARE_CHAT__BRANCHES_DIR") or ""
        ).strip()
        if override:
            return Path(override).expanduser()
        state_root = (
            os.environ.get("XDG_STATE_HOME") or ""
        ).strip() or "~/.local/state"
        return Path(state_root).expanduser() / "care" / "branches"

    @classmethod
    def _list_branch_files(cls) -> list[Path]:
        """Return the saved branch sidecar paths sorted by
        modified time (newest first). Missing dir yields an
        empty list."""
        path = cls._branches_dir()
        if not path.exists() or not path.is_dir():
            return []
        try:
            files = [
                f for f in path.iterdir()
                if f.is_file() and f.suffix == ".json"
            ]
        except OSError:
            return []
        try:
            files.sort(
                key=lambda f: f.stat().st_mtime, reverse=True,
            )
        except OSError:
            files.sort(key=lambda f: f.name, reverse=True)
        return files

    def _handle_branch_command(self, arg: str) -> None:
        """``/branch [name]``            → snapshot the current
        transcript as a checkpoint. ``/branch list``
        → show saved checkpoints. ``/branch switch <id>``
        → rehydrate a checkpoint (replaces the live
        transcript). ``/branch delete <id>`` → remove a
        checkpoint sidecar. Contained variant of the
        per-cell branching spec — delivers the
        "save / restore experimental forks" value without
        the dual-pane UI."""
        tokens = (arg or "").split(maxsplit=1)
        if not tokens:
            # Bare /branch saves an auto-named checkpoint.
            self._save_branch(name=None)
            return
        sub = tokens[0].lower()
        rest = tokens[1] if len(tokens) > 1 else ""
        if sub == "list":
            self._render_branches_listing()
            return
        if sub == "switch":
            if not rest:
                self._post_line(
                    "system",
                    "Usage: `/branch switch <id>`. "
                    "Run `/branch list` to see saved ids.",
                    severity="warning",
                )
                return
            self._switch_to_branch(rest.strip())
            return
        if sub == "delete":
            if not rest:
                self._post_line(
                    "system",
                    "Usage: `/branch delete <id>`.",
                    severity="warning",
                )
                return
            self._delete_branch(rest.strip())
            return
        # Anything else is treated as a "save with this
        # name" call so users can write `/branch v2-attempt`
        # without remembering a `save` sub-command.
        candidate = arg.strip()
        if not self._BRANCH_FILENAME_RE.fullmatch(candidate):
            self._post_line(
                "system",
                f"Branch name `{candidate}` is invalid. "
                "Use letters, digits, `_`, or `-` only.",
                severity="warning",
            )
            return
        self._save_branch(name=candidate)

    def _save_branch(self, *, name: str | None) -> None:
        import json

        if name is None:
            name = f"branch-{int(time.time())}"
        branches_dir = self._branches_dir()
        try:
            branches_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._post_line(
                "system",
                f"Couldn't create branches directory `{branches_dir}`: "
                f"{exc}",
                severity="error",
            )
            return
        path = branches_dir / f"{name}.json"
        if path.exists():
            self._post_line(
                "system",
                f"Branch `{name}` already exists. Pick a "
                "different name or `/branch delete` it first.",
                severity="warning",
            )
            return
        payload: dict[str, Any] = {
            "name": name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode,
            "current_turn": self._current_turn,
            "lines": [
                {
                    "role": line.role,
                    "text": line.text,
                    "timestamp": line.timestamp.isoformat(),
                    "mode": line.mode,
                    "provenance": line.provenance,
                    "reaction": line.reaction,
                }
                for line in self._lines
            ],
        }
        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            self._post_line(
                "system",
                f"Couldn't save branch to `{path}`: {exc}",
                severity="error",
            )
            return
        self._post_line(
            "system",
            f"Saved branch `{name}` ({len(self._lines)} line"
            f"{'s' if len(self._lines) != 1 else ''}). "
            "Switch back via `/branch switch "
            f"{name}`.",
        )

    def _render_branches_listing(self) -> None:
        files = self._list_branch_files()
        if not files:
            self._post_line(
                "system",
                "No saved branches yet. Run `/branch` to "
                "checkpoint the current transcript.",
                severity="warning",
            )
            return
        rows: list[str] = ["**Saved branches (newest first):**"]
        for f in files[:20]:
            name = f.stem
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                stamp = mtime.strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                stamp = "?"
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            rows.append(f"  `{name}` — {stamp} ({size} bytes)")
        rows.append(
            "\nActions: `/branch switch <id>`, "
            "`/branch delete <id>`.",
        )
        self._post_line("system", "\n".join(rows))

    def _switch_to_branch(self, name: str) -> None:
        import json

        path = self._branches_dir() / f"{name}.json"
        if not path.exists() or not path.is_file():
            self._post_line(
                "system",
                f"Branch `{name}` not found. "
                "Run `/branch list` to see saved ids.",
                severity="warning",
            )
            return
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8"),
            )
        except (OSError, json.JSONDecodeError) as exc:
            self._post_line(
                "system",
                f"Couldn't read branch `{name}`: {exc}",
                severity="error",
            )
            return
        lines_payload = payload.get("lines") or []
        # Wipe the live transcript before replaying — we're
        # switching, not appending.
        self._lines.clear()
        # Tracked localizable lines belong to the transcript we're
        # replacing; drop the stale refs so relocalize doesn't chase
        # removed widgets.
        self._localizable_lines.clear()
        try:
            transcript = self.query_one(
                "#chat-transcript", VerticalScroll,
            )
            transcript.remove_children()
        except Exception:
            pass
        self._current_turn = 0
        self._turn_focus_mode = False
        self._post_line(
            "system",
            f"↳ branched from `{name}` "
            f"(saved {payload.get('created_at', '?')}).",
        )
        for entry in lines_payload:
            role = entry.get("role")
            text = entry.get("text") or ""
            provenance = entry.get("provenance")
            if role not in {"user", "assistant", "system", "tool"}:
                continue
            self._post_line(role, text, provenance=provenance)

    def _delete_branch(self, name: str) -> None:
        path = self._branches_dir() / f"{name}.json"
        if not path.exists() or not path.is_file():
            self._post_line(
                "system",
                f"Branch `{name}` not found.",
                severity="warning",
            )
            return
        try:
            path.unlink()
        except OSError as exc:
            self._post_line(
                "system",
                f"Couldn't delete branch `{name}`: {exc}",
                severity="error",
            )
            return
        self._post_line(
            "system",
            f"Deleted branch `{name}`.",
        )

    # ------------------------------------------------------------------
    # /imgpreview — terminal-graphics protocol detection (Phase 9 P3)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_terminal_graphics_protocol() -> str | None:
        """Phase 9 P3 — sniff the environment to decide which
        terminal-graphics protocol (if any) the user's
        terminal supports.

        Returns ``"kitty"`` when ``KITTY_WINDOW_ID`` is set
        (the Kitty / kitten / Ghostty family), ``"iterm2"``
        when ``TERM_PROGRAM`` is ``iTerm.app`` or
        ``LC_TERMINAL`` is ``iTerm2``, ``"wezterm"`` when
        ``TERM_PROGRAM`` is ``WezTerm`` (which supports the
        iTerm2 protocol). Returns ``None`` otherwise — most
        notably Apple Terminal, xterm, gnome-terminal, and
        VS Code's terminal (none of which speak inline
        graphics protocols natively)."""
        import os

        if (os.environ.get("KITTY_WINDOW_ID") or "").strip():
            return "kitty"
        term_program = (
            os.environ.get("TERM_PROGRAM") or ""
        ).strip()
        if term_program == "iTerm.app":
            return "iterm2"
        if term_program == "WezTerm":
            return "wezterm"
        lc_terminal = (
            os.environ.get("LC_TERMINAL") or ""
        ).strip()
        if lc_terminal == "iTerm2":
            return "iterm2"
        return None

    @staticmethod
    def _build_kitty_image_sequence(
        image_bytes: bytes, *, fmt: str = "100",
    ) -> str:
        """Phase 9 P3 — Kitty graphics protocol escape sequence
        that asks the terminal to display ``image_bytes`` once,
        directly (``a=T`` means transmit-and-display).
        ``f=100`` declares the data as a self-describing image
        (PNG/JPG/...) — Kitty sniffs the actual format from
        the bytes. The base64 payload is split into chunks
        only when over the protocol's per-message budget
        (4096 bytes); below that, a single message works."""
        import base64

        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        return f"\x1b_Ga=T,f={fmt};{b64}\x1b\\"

    @staticmethod
    def _build_iterm2_image_sequence(
        image_bytes: bytes,
        *,
        name: str = "image",
        preserve_aspect: bool = True,
        inline: bool = True,
    ) -> str:
        """Phase 9 P3 — iTerm2 / WezTerm inline-image OSC 1337.
        Both terminals consume the same sequence. The name +
        size args show up in iTerm's "right-click → save"
        menu. ``inline=1`` means render in the cell flow;
        ``preserveAspectRatio=1`` is the friendlier default
        (no squashed thumbnails)."""
        import base64

        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        name_b64 = base64.standard_b64encode(
            name.encode("utf-8"),
        ).decode("ascii")
        args = [
            f"name={name_b64}",
            f"size={len(image_bytes)}",
            f"inline={1 if inline else 0}",
            f"preserveAspectRatio={1 if preserve_aspect else 0}",
        ]
        return f"\x1b]1337;File={';'.join(args)}:{b64}\x07"

    @classmethod
    def _build_image_sequence(
        cls, image_bytes: bytes, protocol: str, *, name: str = "image",
    ) -> str | None:
        """Dispatcher that returns the right escape sequence
        for the named protocol, or ``None`` for unsupported
        protocols (caller decides how to surface the failure)."""
        if protocol == "kitty":
            return cls._build_kitty_image_sequence(image_bytes)
        if protocol in {"iterm2", "wezterm"}:
            return cls._build_iterm2_image_sequence(
                image_bytes, name=name,
            )
        return None

    _PROTOCOL_LABELS: dict[str, str] = {
        "kitty": "Kitty graphics protocol",
        "iterm2": "iTerm2 inline-image protocol",
        "wezterm": "WezTerm (iTerm2-compatible)",
    }

    def _handle_imgpreview_command(self, arg: str) -> None:
        """``/imgpreview status``       → report which terminal-
        graphics protocol (if any) the running terminal
        supports. ``/imgpreview <path>``         → check that
        the file is a supported image, then build the
        protocol-specific escape sequence and report what
        WOULD be emitted (cell-rendering is gated by the
        terminal; Textual's TUI rendering layer doesn't pass
        the sequence through without infrastructure
        beyond this iteration)."""
        protocol = self._detect_terminal_graphics_protocol()
        tokens = (arg or "").split(maxsplit=1)
        if not tokens or tokens[0].lower() == "status":
            if protocol is None:
                self._post_line(
                    "system",
                    "No inline-graphics protocol detected. "
                    "Supported terminals: Kitty (sets "
                    "`KITTY_WINDOW_ID`), iTerm2 / WezTerm "
                    "(set `TERM_PROGRAM`).",
                    severity="warning",
                )
            else:
                label = self._PROTOCOL_LABELS.get(protocol, protocol)
                self._post_line(
                    "system",
                    f"Inline-graphics protocol detected: "
                    f"{label} (`{protocol}`).",
                )
            return
        # Treat the first token as a path.
        raw_path = arg.strip()
        try:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system",
                f"Couldn't resolve `{raw_path}`: {exc}",
                severity="warning",
            )
            return
        if not path.exists() or not path.is_file():
            self._post_line(
                "system",
                f"`{raw_path}` is not a regular file.",
                severity="warning",
            )
            return
        mime = self._mime_for_image_path(path)
        if mime is None:
            self._post_line(
                "system",
                f"`{path.name}` doesn't look like a supported "
                "image (.png / .jpg / .jpeg / .gif / .webp / "
                ".bmp).",
                severity="warning",
            )
            return
        max_bytes = self._image_ref_max_bytes()
        try:
            data = path.read_bytes()
        except OSError as exc:
            self._post_line(
                "system",
                f"Couldn't read `{path}`: {exc}",
                severity="error",
            )
            return
        if len(data) > max_bytes:
            self._post_line(
                "system",
                f"Image `{path.name}` is {len(data)} bytes — "
                f"over the {max_bytes}-byte cap. Bump "
                "`CARE_CHAT__IMAGE_REF_MAX_BYTES` or use a "
                "smaller image.",
                severity="warning",
            )
            return
        if protocol is None:
            self._post_line(
                "system",
                f"Detected image `{path.name}` ({len(data)} "
                f"bytes, {mime}), but no terminal-graphics "
                "protocol is available. Run `/imgpreview "
                "status` for the supported-terminal list.",
                severity="warning",
            )
            return
        sequence = self._build_image_sequence(
            data, protocol, name=path.name,
        )
        if sequence is None:
            self._post_line(
                "system",
                f"Internal: no builder for protocol `{protocol}`.",
                severity="error",
            )
            return
        label = self._PROTOCOL_LABELS.get(protocol, protocol)
        self._post_line(
            "system",
            f"Built {label} sequence for `{path.name}` "
            f"({len(data)} bytes, {mime}, "
            f"{len(sequence)} byte escape). The bytes are "
            "ready to emit; Textual's renderer doesn't pass "
            "raw escapes through, so actual inline display "
            "still needs a renderer hook (Phase 10+).",
        )

    @staticmethod
    def _format_provenance(provenance: dict[str, Any]) -> str:
        """Project a provenance dict into the system-line block
        the inspector renders. Stable key order so the layout
        stays scannable across calls; missing keys drop their
        row rather than render an empty value."""
        from care.runtime.pricing import format_cost

        lines = ["🔍 Prompt inspector:"]
        if (model := provenance.get("model")):
            lines.append(f"  model:      {model}")
        if (mode := provenance.get("mode")):
            lines.append(f"  mode:       {mode}")
        if (iteration := provenance.get("iteration")) is not None:
            lines.append(f"  iteration:  {iteration}")
        duration = provenance.get("duration_seconds")
        if duration is not None:
            try:
                lines.append(f"  duration:   {float(duration):.3f}s")
            except (TypeError, ValueError):
                lines.append(f"  duration:   {duration}")
        prompt_tokens = provenance.get("prompt_tokens")
        completion_tokens = provenance.get("completion_tokens")
        tokens_total = provenance.get("tokens_total")
        if (
            prompt_tokens is not None
            and completion_tokens is not None
        ):
            total = tokens_total if tokens_total is not None else (
                prompt_tokens + completion_tokens
            )
            lines.append(
                f"  tokens:     in {prompt_tokens} / out "
                f"{completion_tokens} / total {total}",
            )
        elif tokens_total is not None:
            lines.append(f"  tokens:     {tokens_total} total")
        if (cost := provenance.get("cost_usd")) is not None:
            formatted = format_cost(cost)
            if formatted:
                lines.append(f"  cost:       {formatted}")
        # Phase 9 P2 — render the captured raw LLM stream
        # (system prompt + user message + response when CARL
        # surfaces them; response-only today) as a fenced
        # code block so the user can copy the exact body that
        # produced the reply.
        raw_system_prompt = provenance.get("raw_system_prompt")
        if raw_system_prompt:
            lines.append("\n**system prompt:**")
            lines.append("```text")
            lines.append(str(raw_system_prompt).rstrip())
            lines.append("```")
        raw_prompt = provenance.get("raw_prompt")
        if raw_prompt:
            lines.append("\n**user prompt:**")
            lines.append("```text")
            lines.append(str(raw_prompt).rstrip())
            lines.append("```")
        raw_response = provenance.get("raw_response")
        if raw_response:
            lines.append("\n**raw response:**")
            lines.append("```text")
            lines.append(str(raw_response).rstrip())
            lines.append("```")
        return "\n".join(lines)

    @classmethod
    def _extract_continuation(cls, answer: str) -> str | None:
        """Parse `answer` for the `[CONTINUE]` sentinel. Returns
        the next-request text on continuation (an empty marker
        yields the default ``"Continue."``), or ``None`` when
        the answer is terminal."""
        if not isinstance(answer, str):
            return None
        match = cls._CONTINUATION_RE.search(answer)
        if match is None:
            return None
        next_text = (match.group(1) or "").strip()
        return next_text or "Continue."

    @classmethod
    def _strip_continuation_marker(cls, answer: str) -> str:
        """Strip the `[CONTINUE…]` marker so the user-visible
        answer stays clean. The marker is metadata, not
        user-facing prose."""
        if not isinstance(answer, str):
            return answer
        return cls._CONTINUATION_RE.sub("", answer).strip()

    @classmethod
    def _extract_force_marker(cls, task: str) -> tuple[bool, str]:
        """Strip an optional leading ``[FORCE]`` marker from
        ``task`` (case-insensitive, with optional surrounding
        whitespace). Returns ``(force, cleaned_task)``.

        Lets a user explicitly opt-out of the Phase-3-P2 dedup
        check by prepending ``[FORCE]`` to their task — the
        cleanest in-chat escape hatch. The marker is stripped
        before the task is hashed / saved / shown in the chain's
        display name, so the saved chain looks identical to a
        fresh first-time save.
        """
        import re

        match = re.match(r"^\s*\[FORCE\]\s*", task or "", re.IGNORECASE)
        if match is None:
            return False, task
        return True, task[match.end():]

    @classmethod
    def _find_existing_chain_by_task_hash(
        cls, memory: Any, task_hash: str,
    ) -> tuple[str, str] | None:
        """Look up the first chain tagged ``task-hash:<hash>``
        and return ``(chain_id, display_name)``, or ``None`` when
        no duplicate exists / the lookup itself fails.

        Failures are silent — we don't want a flaky listing call
        to block a real save. The dedup gate falls open when the
        signal is unavailable.
        """
        tag = f"task-hash:{task_hash}"
        try:
            rows = memory.list_entities(
                entity_type="chain",
                tags=[tag],
                limit=1,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "dedup lookup failed (%s); proceeding with save", exc,
            )
            return None
        if not rows:
            return None
        row = rows[0]
        chain_id = (
            row.get("entity_id")
            or row.get("id")
            or row.get("chain_id")
            or ""
        )
        if not chain_id:
            return None
        name = (
            row.get("name")
            or row.get("display_name")
            or ""
        )
        return chain_id, name

    @staticmethod
    def _task_hash(task: str) -> str:
        """Return the first 12 hex chars of ``sha256(task)``.

        Used as the ``task-hash:<hash>`` provenance tag so the
        Production save path can answer "have I saved this exact
        task before?" via a single
        `memory.search_hits(tag="task-hash:<hash>")` call. 12
        hex chars = 48 bits → collision-free up to ~16 M tasks
        per chain library, which is plenty for any one user.
        """
        import hashlib

        return hashlib.sha256(
            (task or "").encode("utf-8"),
        ).hexdigest()[:12]

    @staticmethod
    def _slugify_tag(value: str) -> str:
        """Project ``value`` into a tag-safe slug: letters,
        digits, dot, underscore, hyphen survive; everything else
        becomes ``-``. Idempotent + lowercase so semantically
        identical inputs collapse to the same tag."""
        import re

        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
        return cleaned.strip("-") or "unknown"

    @staticmethod
    def _with_today_preamble(task: str) -> str:
        """Prefix the generation task with today's date so MAGE plans
        against *now*, not its training cutoff.

        Without this the planner bakes a stale year into tool inputs
        (e.g. a web-search query ``"... latest track 2024"`` in 2026) and
        reasons about "today" from memory. Generation-only — the
        execution ``outer_context`` keeps the user's original text."""
        try:
            from care.builtin_tools import current_datetime

            return f"(Today is {current_datetime()}. Treat this as the current date/time.)\n\n{task}"
        except Exception:  # noqa: BLE001
            return task

    @staticmethod
    def _build_followup_task(
        *, initial_task: str, prev_answer: str, next_request: str,
    ) -> str:
        """Compose the next-iteration task description. Carries
        the original goal + the previous answer + the agent's
        continuation request so MAGE has full context."""
        return (
            f"Original task: {initial_task}\n\n"
            f"Previous answer: {prev_answer}\n\n"
            f"Next step: {next_request}"
        )

    async def _execute_chain_interactive(
        self,
        *,
        task: str,
        chain_dict: dict[str, Any],
        dataset_id: str = "",
        files: dict[str, Any] | None = None,
    ) -> Any:
        """Run a freshly-generated chain via CARL with ``task``
        as ``outer_context``. Returns the ``ReasoningResult`` on
        success, ``None`` on any error (already surfaced as a
        chat system line + logged).

        ``dataset_id``: optional dataset identifier (§6 P1
        Production hook). When set, the local run-history row
        records `extra["dataset"]=<id>` so the `/runs` screen
        / `/cost` rollups can group by dataset.

        Defensive at every layer — the user can land on this
        path with a busted CARL install, missing API key, or a
        malformed chain dict, and the chat should keep working
        rather than crash the screen.
        """
        from care.runtime.executor import (
            build_run_context,
            execute_chain_async,
        )
        from care.runtime.llm_client import (
            LLMClientError,
            build_carl_llm_client,
        )

        cfg = self.app.config
        try:
            # CARL's step executors call `get_response_with_retries`
            # on the API object — `build_carl_llm_client` returns
            # an `mmar_carl.llm.OpenAICompatibleClient` which
            # implements that contract. The plain `build_llm_client`
            # returns a raw `openai.OpenAI` SDK client (the shape
            # MAGE wants) and would crash every step.
            # Passing `token_counter` here wraps the client in a
            # subclass that folds `response.usage` into the
            # session counter on every step — without it the
            # StatusBar + iteration footer would always read
            # `in 0 / out 0` for CARL-driven runs.
            api = build_carl_llm_client(
                cfg.mage,
                token_counter=getattr(self.app, "token_counter", None),
            )
        except LLMClientError as exc:
            _log.warning("LLM client unavailable for CARL: %s", exc)
            self._post_line(
                "system",
                f"Can't execute chain: {exc}",
                severity="warning",
            )
            return None

        try:
            from mmar_carl import ReasoningChain
        except ImportError as exc:
            _log.warning("mmar_carl not installed: %s", exc)
            self._post_line(
                "system",
                "Agent chain format runtime isn't installed — "
                "`pip install \"care[carl]\"` to execute chains.",
                severity="warning",
            )
            return None

        # MAGE's richer topologies can author step types the *installed* CARL
        # can't load (debate / evaluation / …). Downgrade those to llm BEFORE
        # parsing so the chain stays runnable — the DAG shape is preserved.
        try:
            from care.tool_planning import downgrade_unsupported_step_types

            downgrade_unsupported_step_types(chain_dict)
        except Exception as exc:  # noqa: BLE001
            _log.warning("step-type downgrade skipped: %s", exc)

        # Route a hallucination-prone live-data LLM step to a real tool
        # (e.g. "what's today's date" → current_datetime) BEFORE parsing,
        # so the rewritten tool step is what executes / gets synthesised.
        try:
            from care.tool_planning import augment_chain_for_live_data

            aug = await augment_chain_for_live_data(
                chain_dict, task=task, api=api, config=cfg,
            )
            if aug.get("rewrote"):
                chain_dict = aug["chain_dict"]
                self._post_line(
                    "tool",
                    f"🔧 routing “{aug.get('step_title') or 'step'}” to "
                    f"tool: {aug['tool_name']}",
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("live-data tool routing skipped: %s", exc)

        try:
            # from_dict(use_typed_steps=True) mutates the dict IN PLACE
            # (step_config/llm_config sub-dicts → Pydantic objects). Copy first
            # so result.chain_dict stays plain JSON for the reuse cache and
            # downstream serialization (else `_chain_is_parameterized` and other
            # json.dumps over it break).
            import copy as _copy

            chain = ReasoningChain.from_dict(
                _copy.deepcopy(chain_dict), use_typed_steps=True,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("chain parse failed: %s", exc, exc_info=True)
            self._post_line(
                "system",
                f"Couldn't parse generated chain: {exc}",
                severity="error",
            )
            return None

        # Wire a CarlStreamer at `self` so per-step events stream
        # back into the transcript via the `on_step_*` handlers
        # below. Without it the user sees only the final answer
        # and has no signal that work is happening on long chains.
        streamer = CarlStreamer(self)
        try:
            # P6.8/LTM — attach CARL's long-term memory + inject the standing
            # user context (CARE.md + recalled LTM digest) into the grounding
            # prompt so the ANSWER (every chain step) is personalised, and the
            # chain can also recall LTM on demand via ``$ltm.<key>``.
            exec_user_context, ltm, ltm_session = self._user_context_bundle(task)
            ctx = build_run_context(
                query=task, api=api, config=cfg, streamer=streamer,
                long_term_memory=ltm, session_id=ltm_session,
                user_context=exec_user_context,
                # Document-skill inputs the user attached when the chain
                # needs a file (care/skill_file_inputs) — seeds
                # context.memory["input"][<basename>] so the rewritten
                # `$memory.input.<basename>` step ref resolves.
                files=files or None,
            )
        except Exception as exc:  # noqa: BLE001
            # ``ExecutionError`` is the documented contract, but
            # upstream CARL drift can surface a raw ``ValueError``
            # (Pydantic field rejection on the context model) or
            # similar — catch broadly so the user sees a friendly
            # line instead of a silently-dead worker.
            _log.error("context build failed: %s", exc, exc_info=True)
            self._post_line(
                "system",
                f"Couldn't build execution context: {exc}",
                severity="error",
            )
            return None

        # Self-healing: synthesise any tool the chain references but the
        # context lacks (e.g. a planner that invented `get_current_weather`)
        # so the run doesn't die with "Tool '<name>' not registered in
        # context". The generated code runs only in the Docker sandbox.
        try:
            from care.tool_synthesis import synthesize_missing_tools

            report = await synthesize_missing_tools(
                chain_dict, ctx, api=api, config=cfg,
                notify=lambda msg: self._post_line("tool", msg),
            )
            for nm in report.get("reused", []):
                self._post_line("tool", f"♻ reusing tool: {nm}")
            for nm in report.get("created", []):
                self._post_line("tool", f"🛠 created tool: {nm}")
            for nm, why in report.get("failed", []):
                self._post_line(
                    "system",
                    f"⚠ couldn't create tool '{nm}': {why}",
                    severity="warning",
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("tool synthesis skipped: %s", exc)

        _log.info(
            "executing chain via CARL: steps=%d",
            len(getattr(chain, "steps", []) or []),
        )
        # §6 P1 — wall-clock start so the local run-history
        # row carries a real `duration_seconds`.
        import time as _time

        started_at = _time.time()
        try:
            result = await execute_chain_async(chain, ctx)
        except Exception as exc:  # noqa: BLE001
            # Same broad catch — CARL step errors can surface as
            # ``ExecutionError``, ``LLMClientError`` from a step,
            # or upstream Pydantic / network errors. All belong in
            # the friendly system line, not the silent worker
            # graveyard.
            _log.error(
                "chain execution failed: %s", exc, exc_info=True,
            )
            self._post_line(
                "system",
                f"Chain execution failed: {exc}",
                severity="error",
            )
            self._record_local_run(
                chain=chain,
                task=task,
                result=None,
                started_at=started_at,
                duration=_time.time() - started_at,
                status="failure",
                error=str(exc),
                dataset_id=dataset_id,
            )
            return None
        # A CARL chain can return `success=False` with per-step
        # `error_message` fields even when no exception bubbles
        # (every step caught its own error). Surface those as
        # red lines so the user reads what broke instead of the
        # raw ReasoningResult dump.
        if self._chain_result_failed(result):
            self._post_chain_failures(result)
            self._record_local_run(
                chain=chain,
                task=task,
                result=result,
                started_at=started_at,
                duration=_time.time() - started_at,
                status="failure",
                error=self._extract_first_step_error(result),
                dataset_id=dataset_id,
            )
            return None
        _log.info("chain execution succeeded")
        # P6.5 — land any files the chain/skill produced in a stable,
        # cross-platform artifacts dir + surface each as a `📄 saved:` line.
        self._save_run_artifacts(result, task=task)
        self._record_local_run(
            chain=chain,
            task=task,
            result=result,
            started_at=started_at,
            duration=_time.time() - started_at,
            status="success",
            dataset_id=dataset_id,
        )
        return result

    def _save_run_artifacts(self, result: Any, *, task: str = "") -> list[Path]:
        """P6.5 — copy any files a chain/skill produced out of the throwaway
        sandbox into a stable artifacts dir (cross-platform; default
        ``~/.care/artifacts/<run>``, overridable via ``CARE_ARTIFACTS__DIR``)
        and surface each as a ``📄 saved: <path>`` line. Best-effort — never
        raises into the turn; a run with no output files is a silent no-op."""
        try:
            from care.runtime.artifacts import save_run_artifacts

            care_config = getattr(self.app, "config", None)
            saved = save_run_artifacts(
                result, care_config=care_config, slug=task,
            )
        except Exception as exc:  # noqa: BLE001
            _log.info("artifact save skipped: %s", exc)
            return []
        for path in saved:
            self._post_line("system", f"📄 saved: {path}")
        return saved

    def _fallback_build_file_if_missing(
        self, run_result: Any, *, skill: str, task: str = "",
    ) -> Path | None:
        """When a file-producing skill wrote nothing, build the file ourselves
        from the generated text — model/Docker/network-independent.

        Only fires when ``missing_required_output(run_result)`` is set (the skill
        was required to write a file but didn't). Currently builds ``.pptx`` via
        :mod:`care.runtime.deck_builder`; other file kinds fall back to an honest
        warning. Never raises into the turn."""
        try:
            from care.runtime.artifacts import (
                missing_required_output,
                resolve_artifacts_root,
            )

            if not missing_required_output(run_result):
                return None  # the skill actually produced a file — nothing to do

            content = self._skill_output_text(run_result)
            built: Path | None = None
            if skill == "pptx" and content:
                from care.runtime.deck_builder import (
                    build_pptx_from_text,
                    pptx_available,
                )

                if pptx_available():
                    root = resolve_artifacts_root(getattr(self.app, "config", None))
                    from care.runtime.artifacts import _slugify, _unique_path

                    name = (_slugify(task) or "presentation") + ".pptx"
                    dest = _unique_path(root / name)
                    built = build_pptx_from_text(content, dest, title=None)

            if built is not None:
                self._post_line(
                    "system",
                    f"📄 saved: {built}  (восстановлено из текста — навык не "
                    f"записал файл сам)",
                )
                return built

            # Couldn't build (unsupported kind / no content / no python-pptx).
            self._post_line(
                "system",
                "⚠ навык не создал файл — модель вернула только текст, "
                "файл на диск не записан.",
                severity="warning",
            )
        except Exception as exc:  # noqa: BLE001
            _log.info("file fallback skipped: %s", exc)
        return None

    @staticmethod
    def _skill_output_text(run_result: Any) -> str:
        """The richest text to build a fallback file from: the agent_skill step
        that failed to write a file (its prose IS the content), else the last
        non-empty step output."""
        steps = getattr(run_result, "step_results", None) or []
        # Prefer the step flagged no_output_file (the skill step).
        for step in steps:
            data = getattr(step, "result_data", None)
            if isinstance(data, dict) and data.get("no_output_file") is True:
                text = (getattr(step, "result", "") or "").strip()
                if text:
                    return text
        # Otherwise the last step with non-empty text.
        for step in reversed(steps):
            text = (getattr(step, "result", "") or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _extract_first_step_error(result: Any) -> str:
        """Pull the first non-empty `error_message` from a
        failed `ReasoningResult` so the local run-history
        row's `error` slot carries a useful one-liner. Falls
        back to a generic message when no per-step error
        surfaced."""
        for step in getattr(result, "step_results", None) or ():
            if getattr(step, "success", True) is False:
                msg = getattr(step, "error_message", "") or ""
                if msg:
                    return str(msg)[:280]
        return "chain reported failure with no per-step error"

    def _record_local_run(
        self,
        *,
        chain: Any,
        task: str,
        result: Any,
        started_at: float,
        duration: float,
        status: str,
        error: str = "",
        dataset_id: str = "",
    ) -> None:
        """§6 P1 — Append a row to
        `~/.cache/care/runs/<YYYY-MM-DD>.jsonl` so the
        `/runs` screen shows real history.

        Best-effort: any failure (disk full, malformed
        record) logs at WARNING and returns; we never let a
        recording problem leak into the user-facing
        chat surface."""
        import time as _time

        try:
            from care.runtime.local_run_history import (
                build_run_entry,
                record_local_run,
            )
        except Exception:
            return
        run_id = (
            "ad_hoc-"
            + _time.strftime(
                "%Y%m%dT%H%M%SZ", _time.gmtime(started_at),
            )
            + f"-{int((started_at * 1000) % 1000):03d}"
        )
        provider = ""
        try:
            provider = str(
                getattr(
                    getattr(self.app.config, "mage", None),
                    "provider", "",
                )
                or ""
            )
        except Exception:
            provider = ""
        extra: dict[str, Any] = {}
        if dataset_id:
            extra["dataset"] = dataset_id
        try:
            record_local_run(
                build_run_entry(
                    run_id=run_id,
                    chain=chain,
                    task=task,
                    result=result,
                    started_at=started_at,
                    duration=duration,
                    status=status,
                    error=error,
                    mode=str(getattr(self, "mode", "") or "interactive"),
                    provider=provider,
                    extra=extra,
                    write_replay=True,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "local run history record failed: %s",
                exc, exc_info=False,
            )

    @staticmethod
    def _chain_result_failed(result: Any) -> bool:
        """Return ``True`` when a CARL ``ReasoningResult`` reports
        any failed step. Reads ``success`` first, then falls back
        to scanning ``step_results`` so an older CARL that only
        flagged the per-step failure still counts. Duck-typed so
        tests can pass plain dataclasses."""
        if result is None:
            return False
        success = getattr(result, "success", None)
        if success is False:
            return True
        step_results = getattr(result, "step_results", None) or []
        for step in step_results:
            if getattr(step, "success", True) is False:
                return True
        return False

    def _post_chain_failures(self, result: Any) -> None:
        """Render every failed step from a CARL result as a red
        ``✗ step N: <title> — <error>`` line, plus a final
        summary line. Replaces the legacy ``str(result)`` dump
        path so the user reads the actual failure cause instead
        of a screen-full of dataclass repr."""
        step_results = getattr(result, "step_results", None) or []
        failure_count = 0
        for step in step_results:
            if getattr(step, "success", True) is not False:
                continue
            failure_count += 1
            number = getattr(step, "step_number", "?")
            title = (
                getattr(step, "step_title", None)
                or "(untitled step)"
            )
            error_message = (
                getattr(step, "error_message", None)
                or "no error message"
            )
            self._post_line(
                "system",
                f"✗ step {number}: {title} — {error_message}",
                severity="error",
            )
        # Final summary line so the user has a single place to
        # read the verdict. `summary` exposes the count;
        # `failure_reason` (when CARL set it) carries the
        # higher-level cause.
        total = len(step_results) if step_results else 0
        if failure_count == 0:
            # `success=False` with no failed steps is unusual —
            # surface a generic line rather than going silent.
            self._post_line(
                "system",
                "Chain reported failure but no step error details "
                "were captured.",
                severity="error",
            )
            return
        metadata = getattr(result, "metadata", None) or {}
        replan = metadata.get("replan") if isinstance(metadata, dict) else None
        failure_reason = ""
        if isinstance(replan, dict):
            failure_reason = (replan.get("failure_reason") or "").strip()
        verdict = (
            f"Chain failed: {failure_count}/{total} step"
            f"{'s' if total != 1 else ''} errored."
        )
        if failure_reason:
            verdict += f" ({failure_reason})"
        self._post_line("system", verdict, severity="error")

    # ------------------------------------------------------------------
    # Production-mode save
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Modes redesign — pipeline driver
    #
    # One driver walks the post-generation pipeline (RUN → SAVE →
    # BASELINE → EVOLVE) consulting the resolved ModeSpec. The
    # parity landing wires the PRODUCTION stages (save/baseline/evolve)
    # so the existing production flow is reproduced exactly; the
    # interactive RUN gate is layered in P1 (the interactive path still
    # runs through the legacy execution loop in `_run_generation`).
    # `_mark_stage_*` are intentionally thin (trail wiring is P2).
    # ------------------------------------------------------------------

    def _current_mode_spec(self) -> "ModeSpec":
        """Resolve the active mode's `ModeSpec` (preset ← config override)."""
        cfg = getattr(getattr(self, "app", None), "config", None)
        return resolve_mode_spec(self.mode, getattr(cfg, "chat", None))

    # Stages shown in the live pipeline strip, in order. PREVIEW is folded
    # into GENERATE (instantaneous), so it isn't a separate cell.
    _STRIP_STAGES: tuple = (
        Stage.GENERATE, Stage.RUN, Stage.SAVE, Stage.BASELINE, Stage.EVOLVE,
    )

    @staticmethod
    def _stage_policy(spec: "ModeSpec", stage: "Stage") -> "StagePolicy":
        """Policy for a strip stage. GENERATE is always auto; the rest read
        off the spec."""
        if stage in (Stage.GENERATE, Stage.PREVIEW):
            return StagePolicy.AUTO
        return getattr(spec, str(stage))

    # Smooth marker animation — each frame changes BOTH form (glyph) and
    # colour, ping-ponging so the motion reads as continuous rather than a
    # two-state blink. Grey-scale so it reads on any theme background.
    #
    # `●` thinking dot — a "breathing" dot that grows + brightens then
    # shrinks + dims:
    _DOT_FRAMES: tuple[tuple[str, str], ...] = (
        ("·", "#555555"),
        ("•", "#727272"),
        ("•", "#909090"),
        ("●", "#aeaeae"),
        ("●", "#d2d2d2"),
        ("●", "#f0f0f0"),
        ("●", "#d2d2d2"),
        ("●", "#aeaeae"),
        ("•", "#909090"),
        ("•", "#727272"),
    )
    # `◇` active stage marker — an outline diamond that fills + brightens
    # then empties + dims:
    _DIAMOND_FRAMES: tuple[tuple[str, str], ...] = (
        ("◇", "#5e5e5e"),
        ("◇", "#7c7c7c"),
        ("◈", "#9a9a9a"),
        ("◈", "#b8b8b8"),
        ("◆", "#dadada"),
        ("◆", "#f0f0f0"),
        ("◆", "#dadada"),
        ("◈", "#b8b8b8"),
        ("◈", "#9a9a9a"),
        ("◇", "#7c7c7c"),
    )
    # Fallback for any other marker (e.g. the `○` ask glyph) — colour-only
    # pulse over the same brightness ramp the dot/diamond use.
    _PULSE_COLORS: tuple[str, ...] = (
        "#555555", "#727272", "#909090", "#aeaeae", "#d2d2d2",
        "#f0f0f0", "#d2d2d2", "#aeaeae", "#909090", "#727272",
    )
    _STATUS_ANIM_INTERVAL: float = 0.1

    def _pulse(self, glyph: str) -> str:
        """Animate ``glyph`` for the current phase — morphing its form +
        colour (Rich markup) so the marker breathes smoothly.

        The two markers are deliberately DESYNCED: the stage marker (``◇``)
        runs the sequence straight while the thinking dot (``●``) runs it
        reversed — so when the stage marker is on its first frame the dot is
        on its last, and they pulse in mirror (one swelling as the other
        ebbs). Anything else gets a colour-only pulse keeping its shape."""
        phase = self._status_phase
        if glyph == "●":
            frames = self._DOT_FRAMES
            # Reversed: last frame when the stage marker is on its first.
            idx = (len(frames) - 1) - (phase % len(frames))
        elif glyph == "◇":
            frames = self._DIAMOND_FRAMES
            idx = phase % len(frames)  # straight
        else:
            colour = self._PULSE_COLORS[phase % len(self._PULSE_COLORS)]
            return f"[{colour}]{glyph}[/]"
        shape, colour = frames[idx]
        return f"[{colour}]{shape}[/]"

    def _render_pipeline_strip(
        self, spec: "ModeSpec", outcomes: "dict[Stage, StageOutcome]",
    ) -> str:
        """Pure render of the pipeline cells, e.g. ``◆ Generate → ○ Run?``.

        `skip`-policy stages are omitted. Glyphs: `done` → ``◆``,
        `failed` → ``✗``, dependency-`skipped` → ``○`` (dim), pending-`ask`
        → ``○ …?``, pending-`auto` → ``◇``. While a worker is "thinking",
        the FIRST still-pending stage's marker pulses smoothly so the eye
        tracks the active step.
        """
        cells: list[str] = []
        active_taken = False
        for stage in self._STRIP_STAGES:
            policy = self._stage_policy(spec, stage)
            if policy is StagePolicy.SKIP:
                continue
            label = t(f"chat.stage.label.{stage}")
            outcome = outcomes.get(stage)
            if outcome is StageOutcome.DONE:
                cells.append(f"◆ {label}")
            elif outcome is StageOutcome.FAILED:
                cells.append(f"✗ {label}")
            elif outcome is StageOutcome.SKIPPED:
                cells.append(f"○ {label}")
            elif self._thinking and not active_taken:
                # First unresolved stage while a worker runs = the stage being
                # actively processed. Always show the morphing diamond — a
                # RUNNING stage isn't "asking", so its `○ …?` ask-glyph would
                # otherwise only colour-pulse and read as static next to
                # Generate's `◇ → ◈ → ◆`. Drop the trailing "?" while active.
                active_taken = True
                cells.append(f"{self._pulse('◇')} {label}")
            else:
                # Pending but not actively running: static ask/auto glyph.
                marker = "○" if policy is StagePolicy.ASK else "◇"
                suffix = "?" if policy is StagePolicy.ASK else ""
                cells.append(f"{marker} {label}{suffix}")
        return " → ".join(cells)

    def _render_status_strip(self) -> str:
        """Combined status line: pipeline cells (when a run is active) and
        the pulsing ``● thinking…`` tail (when a worker is running), joined
        by ``|`` — e.g. ``◇ Generate → ○ Run? | ● thinking…``."""
        parts: list[str] = []
        spec = getattr(self, "_pipeline_spec", None)
        if spec is not None:
            cells = self._render_pipeline_strip(spec, self._pipeline_outcomes)
            if cells:
                parts.append(cells)
        if self._thinking:
            parts.append(f"{self._pulse('●')} {t('chat.spinner')}")
        return "  |  ".join(parts)

    def _status_strip_active(self) -> bool:
        """True when the combined strip should be visible (a pipeline is
        running OR a worker is thinking)."""
        return getattr(self, "_pipeline_spec", None) is not None or self._thinking

    def _refresh_status_strip(self) -> None:
        """Repaint the combined strip + sync its visibility + the animation
        timer. Safe before mount (degrades silently)."""
        try:
            w = self.query_one("#chat-pipeline-strip", Static)
        except Exception:
            return
        active = self._status_strip_active()
        if w.display != active:
            w.display = active
        w.update(self._render_status_strip() if active else "")
        self._sync_status_anim_timer()

    def _sync_status_anim_timer(self) -> None:
        """Run the marker-pulse timer only while a worker is thinking (the
        only time markers animate); pause it otherwise to avoid idle repaints."""
        want = self._thinking
        timer = self._status_anim_timer
        if want and timer is None:
            try:
                self._status_anim_timer = self.set_interval(
                    self._STATUS_ANIM_INTERVAL, self._tick_status_anim,
                )
            except Exception:
                self._status_anim_timer = None
        elif not want and timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            self._status_anim_timer = None
            self._status_phase = 0

    def _tick_status_anim(self) -> None:
        """Advance the pulse phase + repaint the markers."""
        self._status_phase += 1
        spec = getattr(self, "_pipeline_spec", None)
        if not self._thinking and spec is None:
            return
        try:
            self.query_one("#chat-pipeline-strip", Static).update(
                self._render_status_strip(),
            )
        except Exception:
            pass

    def _show_pipeline_strip(
        self, spec: "ModeSpec", outcomes: "dict[Stage, StageOutcome] | None" = None,
    ) -> None:
        """Reveal + seed the combined status strip for a run."""
        self._pipeline_spec = spec
        self._pipeline_outcomes = dict(outcomes or {})
        self._refresh_status_strip()

    def _update_pipeline_stage(
        self, stage: "Stage", outcome: "StageOutcome",
    ) -> None:
        """Record a stage outcome + repaint the strip (no-op when idle)."""
        if getattr(self, "_pipeline_spec", None) is None:
            return
        self._pipeline_outcomes[stage] = outcome
        self._refresh_status_strip()

    def _refresh_pipeline_strip(self) -> None:
        self._refresh_status_strip()

    def _hide_pipeline_strip(self) -> None:
        """Clear the pipeline portion. The strip stays visible if a worker
        is still thinking; otherwise it collapses."""
        self._pipeline_spec = None
        self._pipeline_outcomes = {}
        self._refresh_status_strip()

    async def _confirm_interactive_run(self, *, rich: bool = False) -> bool:
        """RUN-stage gate for the interactive flow.

        Returns True (run) immediately unless the resolved spec sets
        ``run=ask``. After a fresh generation (``rich=True``) the user
        picks **Run** or **Finish** on the persistent chain action bar;
        other bar actions leave the bar mounted. Reuse paths keep the
        compact two-button prompt.
        """
        if self._current_mode_spec().run is not StagePolicy.ASK:
            return True
        if rich:
            action = await self._wait_chain_session_action()
            return action == "run"
        return await self._confirm_stage(Stage.RUN)

    async def _fetch_chain_versions_catalog(
        self, entity_id: str,
    ) -> list[dict[str, Any]]:
        """List saved versions for the chain-bar dropdown."""
        import asyncio

        memory = getattr(self.app, "memory", None)
        if memory is None or not entity_id:
            return []
        client = getattr(memory, "client", None) or getattr(memory, "_client", None)
        lister = getattr(client, "list_versions", None) if client else None
        if not callable(lister):
            return []
        try:
            versions = await asyncio.to_thread(lister, entity_id, "chain")
        except Exception:  # noqa: BLE001
            return []
        catalog: list[dict[str, Any]] = []
        for version in versions or []:
            vid = str(getattr(version, "version_id", "") or "")
            num = getattr(version, "version_number", None)
            if vid and num is not None:
                try:
                    entry: dict[str, Any] = {
                        "version_id": vid,
                        "version_number": int(num),
                    }
                    summary = getattr(version, "change_summary", None)
                    if summary:
                        entry["change_summary"] = str(summary).strip()
                    catalog.append(entry)
                except (TypeError, ValueError):
                    continue
        if catalog:
            names = await asyncio.gather(
                *[
                    self._version_catalog_display_name(
                        entity_id, str(entry["version_id"]),
                    )
                    for entry in catalog
                ],
            )
            for entry, name in zip(catalog, names, strict=True):
                if name:
                    entry["display_name"] = name
        return catalog

    async def _version_catalog_display_name(
        self, entity_id: str, version_id: str,
    ) -> str:
        """Human name for a version row (chain ``name`` field)."""
        loaded = await self._load_chain_version_detail(entity_id, version_id)
        if loaded is None:
            return ""
        content = loaded.get("content")
        if isinstance(content, dict):
            return str(content.get("name") or "").strip()
        return ""

    async def _refresh_chain_versions_catalog(self) -> None:
        """Load version ids for the dropdown and refresh the action bar."""
        session = self._chain_session
        if session is None:
            return
        chain_id = (session.get("payload") or {}).get("chain_id")
        if not chain_id:
            return
        catalog = await self._fetch_chain_versions_catalog(str(chain_id))
        if not catalog:
            return
        if self._chain_session is not session:
            return
        session["versions_catalog"] = catalog
        self._mount_chain_action_bar(focus=False, skip_catalog_fetch=True)

    def _chain_bar_label_text(
        self,
        payload: dict[str, Any],
        *,
        label_key: str,
        label_params: dict[str, Any] | None = None,
        versioned: bool = False,
    ) -> str:
        """Build the chain action bar headline (name + version + action cue)."""
        params = dict(label_params or {})
        name = str(payload.get("display_name") or "chain")
        vnum = payload.get("version_number") if versioned else None
        chain_id = payload.get("chain_id")
        if chain_id and vnum is not None:
            head = t("chat.chainBar.titleWithVersion", name=name, n=vnum)
        elif label_key == "chat.chainBar.titleRevised":
            head = t("chat.chainBar.titleRevised")
            if vnum is not None:
                head += f" · {t('chat.chainBar.versionLabel', n=vnum)}"
        elif label_key == "chat.trace.chainGenerated":
            head = t("chat.chainBar.titleGenerated")
            if vnum is not None:
                head += f" · {t('chat.chainBar.versionLabel', n=vnum)}"
        else:
            if vnum is not None and "n" not in params:
                params["n"] = vnum
            head = t(label_key, **params) if params else t(label_key)
        return f"{head} — {t('chat.chainBar.titleAction')}"

    def _refresh_chain_session_after_version_save(
        self,
        *,
        chain_dict: Any,
        chain_id: str,
        display_name: str | None = None,
        version_id: str | None = None,
        version_number: int | None = None,
        run_task: str | None = None,
    ) -> None:
        """Re-show the action bar after a chain was saved to Memory (new version)."""
        if self.mode != "interactive":
            return
        name = display_name or (chain_dict or {}).get("name") or "chain"
        resolved_task = (
            str(run_task or "").strip()
            or self._chain_run_task_from_metadata(chain_dict or {})
            or ""
        )
        if self._chain_session is None:
            self._begin_chain_session(
                chain_dict=chain_dict or {},
                display_name=name,
                task=resolved_task or self._pending_user_turn or "",
                chain_id=chain_id or None,
                label_key="chat.chainBar.titleRevised",
                version_id=version_id,
                version_number=version_number,
                versioned=True,
            )
            return
        payload = self._chain_session["payload"]
        payload["chain_dict"] = chain_dict or {}
        if chain_id:
            payload["chain_id"] = chain_id
        payload["display_name"] = name
        if resolved_task:
            self._chain_session["task"] = resolved_task
        elif isinstance(chain_dict, dict):
            meta_task = self._chain_run_task_from_metadata(chain_dict)
            if meta_task:
                self._chain_session["task"] = meta_task
        if version_id:
            payload["version_id"] = version_id
        else:
            payload.pop("version_id", None)
        if version_number is not None:
            payload["version_number"] = version_number
        else:
            payload.pop("version_number", None)
        self._last_chain_action_payload = payload
        self._chain_session["has_run"] = False
        self._chain_session["edit_dirty"] = False
        self._chain_session["versioned"] = True
        self._mount_chain_action_bar(
            include_run=True,
            focus=False,
            label_key="chat.chainBar.titleRevised",
        )
        if chain_id and getattr(self.app, "memory", None) is not None:
            self.run_worker(
                self._refresh_chain_versions_catalog(),
                name="chain_versions_catalog",
                group="chat_chain_bar",
                exclusive=False,
                exit_on_error=False,
            )

    async def _fetch_latest_chain_version(
        self, entity_id: str,
    ) -> tuple[str, int | None]:
        """Return ``(version_id, version_number)`` for a saved chain head."""
        import asyncio

        memory = getattr(self.app, "memory", None)
        if memory is None or not entity_id:
            return "", None
        try:
            ent = await asyncio.to_thread(
                memory.get_entity,
                entity_id,
                entity_type="chain",
                channel="latest",
            )
        except Exception:  # noqa: BLE001
            return "", None
        version_id = str(ent.get("version_id") or "")
        raw_num = ent.get("version_number")
        version_number: int | None
        if raw_num is not None:
            try:
                version_number = int(raw_num)
            except (TypeError, ValueError):
                version_number = None
        else:
            version_number = None
        if version_id and version_number is None:
            detail = await self._load_chain_version_detail(
                entity_id, version_id,
            )
            if detail is not None:
                version_number = detail.get("version_number")
        return version_id, version_number

    async def _load_chain_version_detail(
        self, entity_id: str, version_id: str,
    ) -> dict[str, Any] | None:
        """Fetch one historical chain version from Memory."""
        import asyncio

        memory = getattr(self.app, "memory", None)
        if memory is None:
            return None
        client = getattr(memory, "_client", None)
        if client is None:
            return None
        try:
            detail = await asyncio.to_thread(
                client.get_version, entity_id, version_id, "chain",
            )
        except Exception:  # noqa: BLE001
            return None
        content = getattr(detail, "content", None)
        if not isinstance(content, dict):
            return None
        version_number = getattr(detail, "version_number", None)
        try:
            version_number = (
                int(version_number) if version_number is not None else None
            )
        except (TypeError, ValueError):
            version_number = None
        return {
            "content": content,
            "version_id": str(getattr(detail, "version_id", "") or version_id),
            "version_number": version_number,
        }

    @staticmethod
    def _chain_run_task_from_metadata(chain_dict: Any) -> str | None:
        """Recover the task string stamped on a saved chain version."""
        if not isinstance(chain_dict, dict):
            return None
        meta = chain_dict.get("metadata")
        if not isinstance(meta, dict):
            return None
        care = meta.get("care")
        if isinstance(care, dict):
            for key in ("task_description", "query", "description"):
                val = care.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        for key in ("task_description", "query"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    def _resolve_chain_run_task(
        self, chain_dict: Any, session_task: str,
    ) -> str:
        """Task passed to CARL as ``outer_context`` for a session run."""
        meta_task = self._chain_run_task_from_metadata(chain_dict)
        if meta_task:
            return meta_task
        return str(session_task or "").strip()

    def _chain_session_task(self, session: dict[str, Any]) -> str:
        """Best-effort task string for a chain-session re-run."""
        task = str(session.get("task") or "").strip()
        if task:
            return task
        if self._pending_user_turn:
            return str(self._pending_user_turn).strip()
        for role, text in reversed(self._interactive_history):
            if role == "user" and str(text).strip():
                return str(text).strip()
        return ""

    def _dispatch_chain_session_run(self) -> None:
        """Run the in-session chain on the user's task (post-revision re-run)."""
        if self._generating:
            self._post_line(
                "system", t("chat.chainBar.runBusy"), severity="warning",
            )
            return
        self.run_worker(
            self._run_chain_session_worker(),
            name="chat_chain_session_run",
            group="generation",
            exclusive=True,
            exit_on_error=False,
        )

    async def _run_chain_session_worker(self) -> None:
        """Execute the chain held in the active interactive session."""
        session = self._chain_session
        if session is None:
            return
        payload = session.get("payload") or {}
        chain_dict = dict(payload.get("chain_dict") or {})
        session_task = self._chain_session_task(session)
        task = self._resolve_chain_run_task(chain_dict, session_task)
        if not chain_dict:
            self._post_line(
                "system", t("chat.chainBar.noChain"), severity="warning",
            )
            return
        if not task:
            self._post_line(
                "system", t("chat.chainBar.noTask"), severity="warning",
            )
            return

        # File-input gate: when the chain reads a document (docx/pdf/…) and the
        # user didn't already inline one via @path, prompt to attach it and
        # wire it into the run. Otherwise the skill runs against prose — the
        # "it ran without my file" surprise. Cancelling aborts the run rather
        # than executing against an empty document. The model classifies
        # whether a doc-skill step actually READS a file (a keyword heuristic
        # is the fallback) — only doc-skill chains pay for the classification.
        files: dict[str, Any] | None = None
        reads = None
        try:
            from care.skill_file_inputs import (
                apply_file_inputs,
                classify_reads,
                doc_skill_steps,
                requires_file_input,
            )

            if doc_skill_steps(chain_dict):
                try:
                    from care.runtime.llm_client import build_carl_llm_client

                    reads = await classify_reads(
                        build_carl_llm_client(self.app.config.mage), chain_dict,
                    )
                except Exception:  # noqa: BLE001 — heuristic fallback
                    reads = None
                needs_file = requires_file_input(chain_dict, reads=reads)
            else:
                needs_file = False
        except Exception:  # noqa: BLE001
            needs_file = False
        if needs_file and not self._task_has_inline_file(task):
            attached = session.get("skill_files") or await self._collect_skill_files()
            if not attached:
                return  # cancelled / extraction failed — don't run fileless
            session["skill_files"] = attached
            chain_dict, files = apply_file_inputs(
                chain_dict, attached, reads=reads,
            )

        self._generating = True
        self._refresh_status_bar()
        try:
            self._post_line("tool", t("chat.trace.executingChain"))
            run_result = await self._execute_chain_interactive(
                task=task, chain_dict=chain_dict, files=files,
            )
            if run_result is None:
                self._move_chain_action_bar_to_end()
                return
            synthesised = await self._synthesise_user_answer(
                task=task, run_result=run_result,
            )
            answer = (
                synthesised
                if synthesised is not None
                else self._format_carl_result(run_result)
            )
            visible_answer = self._strip_continuation_marker(answer)
            self._post_line("assistant", visible_answer)
            if self._current_mode_spec().followup == "reuse":
                self._record_interactive_turn(task, visible_answer)
            self._mark_chain_session_ran()
        finally:
            self._generating = False
            self._refresh_status_bar()

    # ------------------------------------------------------------------
    # Document-skill file inputs (care.skill_file_inputs)
    # ------------------------------------------------------------------

    @staticmethod
    def _task_has_inline_file(task: str) -> bool:
        """``True`` when the task already carries an inlined file/image — i.e.
        the user attached one via ``@path`` at generation time (see
        :meth:`_read_file_ref`, which wraps content in ``<file …>`` /
        ``<image …>`` blocks). Used to skip the attach prompt."""
        return "<file " in task or "<image " in task

    async def _collect_skill_files(self) -> "list[tuple[str, str]] | None":
        """Prompt the user to attach the document the chain needs, extract its
        text (same path an ``@file`` ref uses), and return ``[(path, text)]``.

        Returns ``None`` when the user cancels or extraction fails — the caller
        aborts the run rather than executing the skill against an empty
        document."""
        from pathlib import Path

        from care.screens.file_picker import FilePickerModal

        self._post_line(
            "system", t("chat.skillFiles.needed"), severity="warning",
        )
        try:
            picked = await self.app.push_screen_wait(
                FilePickerModal(start=Path.cwd()),
            )
        except Exception:  # noqa: BLE001
            return None
        if picked is None:
            self._post_line("system", t("chat.skillFiles.cancelled"))
            return None
        text = self._extract_skill_file_text(str(picked))
        if text is None:
            self._post_line(
                "system",
                t("chat.skillFiles.extractFailed", path=str(picked)),
                severity="error",
            )
            return None
        self._post_line(
            "tool", t("chat.skillFiles.attached", name=Path(picked).name),
        )
        return [(str(picked), text)]

    @staticmethod
    def _extract_skill_file_text(path: str) -> "str | None":
        """Document → chain-ready content via the canonical loader
        (:func:`care.runtime.file_loading.load_file`): office/PDF extracted,
        images → data URI, binary-safe, size-capped. ``None`` on a hard
        failure so the caller can surface a clear error."""
        from care.runtime.file_loading import load_file

        loaded = load_file(path)
        if loaded.error and not loaded.memory_value:
            return None
        return loaded.memory_value

    async def _apply_chain_session_version(
        self,
        version_id: str,
        *,
        show_diff: bool = True,
    ) -> None:
        """Load a historical version into the active chain session."""
        session = self._chain_session
        if session is None:
            return
        payload = session.get("payload") or {}
        chain_id = payload.get("chain_id")
        if not chain_id:
            return
        before_dict = payload.get("chain_dict")
        before_vid = str(payload.get("version_id") or "")
        before_num = payload.get("version_number")
        loaded = await self._load_chain_version_detail(
            str(chain_id), version_id,
        )
        if loaded is None:
            self._post_line(
                "system",
                t("chat.chainBar.versionLoadFailed"),
                severity="error",
            )
            return
        after_num = loaded.get("version_number")
        if (
            show_diff
            and before_vid
            and before_vid != version_id
            and isinstance(before_dict, dict)
        ):
            if before_num is not None and after_num is not None:
                self._post_line(
                    "tool",
                    t(
                        "chat.chainBar.versionDiffHeader",
                        from_n=before_num,
                        to_n=after_num,
                    ),
                )
            self._post_dag_diff(before_dict, loaded["content"])
            memory = getattr(self.app, "memory", None)
            client = (
                getattr(memory, "client", None)
                or getattr(memory, "_client", None)
                if memory is not None
                else None
            )
            if client is not None:
                self._post_line("tool", t("chat.chainBar.versionPatchHeader"))
                await self._post_version_diff(
                    client, str(chain_id), before_vid, str(version_id),
                )
        payload["chain_dict"] = loaded["content"]
        payload["version_id"] = loaded["version_id"]
        if after_num is not None:
            payload["version_number"] = after_num
        meta_task = self._chain_run_task_from_metadata(loaded["content"])
        if meta_task:
            session["task"] = meta_task
        session["has_run"] = False
        if after_num is not None:
            self._post_line(
                "tool",
                t("chat.chainBar.versionSwitched", n=after_num),
            )
        self._mount_chain_action_bar(
            include_run=True,
            focus=False,
            label_key="chat.chainBar.titleRevised",
            skip_catalog_fetch=True,
        )

    def _begin_chain_session(
        self,
        *,
        chain_dict: Any,
        display_name: str,
        task: str,
        chain_id: str | None = None,
        artifact_id: str | None = None,
        label_key: str = "chat.trace.chainGenerated",
        version_id: str | None = None,
        version_number: int | None = None,
        versioned: bool = False,
    ) -> None:
        """Open (or refresh) the interactive chain session + action bar."""
        if not chain_dict:
            return
        self._chain_session_counter += 1
        sid = self._chain_session_counter
        payload = {
            "chain_dict": chain_dict,
            "display_name": display_name or "chain",
            "chain_id": chain_id,
            "artifact_id": artifact_id,
        }
        if version_id:
            payload["version_id"] = version_id
        if version_number is not None:
            payload["version_number"] = version_number
        self._chain_session = {
            "sid": sid,
            "payload": payload,
            "task": task,
            "row_id": "",
            "mount_n": 0,
            "has_run": False,
            "versioned": versioned,
            "edit_dirty": False,
        }
        self._last_chain_action_payload = payload
        self._mount_chain_action_bar(label_key=label_key)

    def _update_chain_session_chain(self, chain_dict: Any) -> None:
        """Refresh the in-session payload after regeneration."""
        if self._chain_session is None:
            return
        payload = self._chain_session["payload"]
        payload["chain_dict"] = chain_dict or {}
        self._last_chain_action_payload = payload
        self._mount_chain_action_bar(focus=False)

    def _mount_chain_action_bar(
        self,
        *,
        include_run: bool | None = None,
        focus: bool = True,
        label_key: str = "chat.trace.chainGenerated",
        label_params: dict[str, Any] | None = None,
        skip_catalog_fetch: bool = False,
    ) -> None:
        """Mount (or re-mount) the persistent chain action bar."""
        from textual.widgets import Button, Select, Static

        session = self._chain_session
        if session is None:
            return
        if include_run is None:
            include_run = not session.get("has_run")
        self._remove_chain_action_bar(clear_session=False)
        sid = session["sid"]
        mount_n = int(session.get("mount_n", 0)) + 1
        session["mount_n"] = mount_n
        row_id = f"chat-chain-bar-{sid}-{mount_n}"
        session["row_id"] = row_id
        try:
            transcript = self.query_one("#chat-transcript", VerticalScroll)
        except Exception:
            return
        payload = session.get("payload") or {}
        versioned = bool(session.get("versioned"))
        label_text = self._chain_bar_label_text(
            payload, label_key=label_key, label_params=label_params,
            versioned=versioned,
        )
        label = Static(
            f"✓ {label_text}",
            classes="chat-chain-bar-label",
            markup=False,
        )
        chain_id = payload.get("chain_id")
        catalog = list(session.get("versions_catalog") or []) if versioned else []
        version_row: Horizontal | None = None
        if chain_id and versioned and catalog:
            options = []
            for entry in sorted(
                catalog, key=lambda item: int(item["version_number"]),
            ):
                n = int(entry["version_number"])
                name = str(entry.get("display_name") or "").strip()
                if name:
                    option_label = t("chat.chainBar.versionOption", n=n, name=name)
                else:
                    option_label = t("chat.chainBar.versionLabel", n=n)
                options.append((option_label, str(entry["version_id"])))
            current = str(payload.get("version_id") or options[-1][1])
            if not any(opt[1] == current for opt in options):
                current = options[-1][1]
            session["_suppress_version_select"] = True
            version_row = Horizontal(
                Static(
                    t("chat.chainBar.versionSelectLabel"),
                    classes="chat-chain-version-label",
                    markup=False,
                ),
                Select(
                    options,
                    value=current,
                    id=f"chat-chain-version-select-{sid}",
                    allow_blank=False,
                    classes="chat-chain-version-select",
                ),
                classes="chat-chain-version-row",
            )
        action_order = ["view", "save"]
        if include_run:
            action_order.append("run")
        action_order.extend(["edit", "evolve", "finish"])
        specs: list[tuple[str, str, str]] = []
        for action in action_order:
            classes = "chat-chain-act-btn"
            if action == "finish":
                classes += " chat-chain-act-finish"
            specs.append((action, t(f"chat.chainBar.{action}"), classes))
        buttons = [
            Button(text, id=f"chat-chain-act-{action}-{sid}", classes=classes)
            for action, text, classes in specs
        ]
        hint_keys = list(action_order)
        if chain_id and versioned and catalog:
            hint_keys.insert(1, "versionSelect")
        hint_lines = [
            Static(
                t(f"chat.chainBar.{hint_key}Hint"),
                classes="chat-chain-bar-hint",
                markup=False,
            )
            for hint_key in hint_keys
        ]
        children: list[Any] = [label]
        if version_row is not None:
            children.append(version_row)
        children.extend([
            Horizontal(*buttons, classes="chat-chain-bar-buttons"),
            Vertical(*hint_lines, classes="chat-chain-bar-hints"),
        ])
        row = Vertical(
            *children,
            classes="chat-line chat-line-action chat-chain-bar-row",
            id=row_id,
        )
        try:
            transcript.mount(row)
            transcript.scroll_end(animate=False)
            if focus and buttons:
                buttons[0].focus()
        except Exception:
            return
        self._set_spinner_visible(False)

        if (
            chain_id
            and versioned
            and not skip_catalog_fetch
            and not catalog
            and getattr(self.app, "memory", None) is not None
        ):
            self.run_worker(
                self._refresh_chain_versions_catalog(),
                name="chain_versions_catalog",
                group="chat_chain_bar",
                exclusive=False,
                exit_on_error=False,
            )
        elif session.get("_suppress_version_select"):
            def _release_version_select() -> None:
                if self._chain_session is session:
                    session["_suppress_version_select"] = False

            self.call_later(_release_version_select)

    def _remove_chain_action_bar(self, *, clear_session: bool = False) -> None:
        """Remove the action bar widget; optionally end the session."""
        session = self._chain_session
        if session is None:
            return
        row_id = session.get("row_id")
        if row_id:
            try:
                self.query_one(f"#{row_id}").remove()
            except Exception:
                pass
        if clear_session:
            self._chain_session = None
            self._chain_session_waiter = None

    def _finish_chain_session(self) -> None:
        """User finished working with the current chain."""
        self._resolve_chain_session_action("finish")
        self._post_line("system", t("chat.stage.chainFinished"))
        self._remove_chain_action_bar(clear_session=True)
        # The pipeline strip lingered past the turn so the outcome stayed
        # visible — Finish is the user's "I'm done" signal, so collapse it.
        self._hide_pipeline_strip()

    async def _wait_chain_session_action(self) -> str:
        """Block until the user clicks Run or Finish on the action bar."""
        import asyncio

        if self._chain_session is None:
            return "run"
        prev = self._chain_session_waiter
        if prev is not None and not prev.done():
            prev.set_result("finish")
        fut: Any = asyncio.get_running_loop().create_future()
        self._chain_session_waiter = fut
        try:
            action = str(await fut)
            if action not in {"run", "finish"}:
                action = "finish"
            return action
        finally:
            self._chain_session_waiter = None

    def _resolve_chain_session_action(self, action: str) -> None:
        """Resolve a pending chain-session wait (Run / Finish only)."""
        fut = self._chain_session_waiter
        if fut is not None and not fut.done():
            fut.set_result(action)

    def _handle_chain_bar_button(self, button_id: str) -> None:
        """Route a persistent chain-bar press — non-terminal actions keep
        the bar mounted."""
        session = self._chain_session
        if session is None:
            return
        parts = (button_id or "").split("-")
        if len(parts) < 5 or parts[0] != "chat" or parts[1] != "chain":
            return
        action = parts[3]
        payload = session["payload"]
        if action == "view":
            self._open_dag_modal(payload=payload)
        elif action == "save":
            self._save_to_library(payload)
        elif action == "edit":
            self._open_dag_modal(payload=payload)
        elif action == "evolve":
            self._evolve_from_dag(payload)
        elif action == "run":
            if self._chain_session_waiter is not None:
                self._resolve_chain_session_action("run")
            else:
                self._dispatch_chain_session_run()
        elif action == "finish":
            self._finish_chain_session()

    def _move_chain_action_bar_to_end(self) -> None:
        """Re-mount the action bar at the bottom of the transcript."""
        session = self._chain_session
        if session is None:
            return
        self._mount_chain_action_bar(
            include_run=not session.get("has_run"),
            focus=False,
        )

    def _mark_chain_session_ran(self) -> None:
        """After the first successful run, drop Run and move the bar to the bottom."""
        session = self._chain_session
        if session is None:
            return
        session["has_run"] = True
        self._move_chain_action_bar_to_end()

    def _relocalize_chain_action_bar(self) -> None:
        """Refresh bar labels after a UI-language switch."""
        if self._chain_session is None:
            return
        self._mount_chain_action_bar(focus=False)

    def _stage_label(self, stage: "Stage") -> str:
        """Localized short label for a pipeline stage (e.g. ``Save``)."""
        return t(f"chat.stage.label.{stage}")

    def _stage_skip_reason_text(self, reason: str) -> str:
        """Localize a skip reason; pass through unknown reasons verbatim."""
        key = f"chat.stage.skipReason.{reason}"
        localized = t(key)
        return reason if localized == key else localized

    def _mark_stage_started(self, stage: "Stage") -> None:
        _log.debug("pipeline stage started: %s", stage)
        self._post_line("tool", f"▶ {self._stage_label(stage)}…")

    def _mark_stage_done(self, stage: "Stage") -> None:
        _log.debug("pipeline stage done: %s", stage)
        self._post_line("tool", f"✓ {self._stage_label(stage)}")

    def _mark_stage_skipped(self, stage: "Stage", *, reason: str) -> None:
        _log.debug("pipeline stage skipped: %s (%s)", stage, reason)
        self._post_line(
            "tool",
            f"○ {self._stage_label(stage)} — "
            f"{self._stage_skip_reason_text(reason)}",
        )

    def _mark_stage_failed(self, stage: "Stage", exc: BaseException) -> None:
        _log.warning("pipeline stage failed: %s (%s)", stage, exc)
        self._post_line(
            "tool", f"✗ {self._stage_label(stage)}", severity="warning",
        )

    @staticmethod
    def _dep_skip_reason(
        outcomes: "dict[Stage, StageOutcome]", *, needs: "tuple[Stage, ...]",
    ) -> str:
        """Human reason a stage was force-skipped: name the first
        unmet prerequisite."""
        for dep in needs:
            if outcomes.get(dep) is not StageOutcome.DONE:
                return f"needs {dep} to have succeeded"
        return "skipped"

    async def _confirm_stage(self, stage: "Stage") -> bool:
        """Ask the user to confirm an ``ask``-policy stage **inline** in the
        transcript (not a modal) — a "<prompt> [Run] [Skip]" row, mirroring
        the inline "✓ Chain generated… [Read full]" affordance.

        Returns True when the user accepts. The prompt copy is resolved from
        the `chat.stage.confirm.*` locale namespace.
        """
        return await self._inline_confirm(
            t(f"chat.stage.confirm.{stage}"),
            confirm_label=t("chat.stage.confirmRun"),
            cancel_label=t("chat.stage.cancelRun"),
        )

    async def _inline_confirm(
        self, prompt: str, *, confirm_label: str, cancel_label: str,
    ) -> bool:
        """Mount an inline confirm row and await the user's button press.

        The generation worker is suspended on the returned Future until
        `on_button_pressed` resolves it; the row is removed afterwards so
        the buttons can't be re-clicked. Falls back to ``True`` (run) when
        there's no transcript to mount into (test scaffolds)."""
        import asyncio

        try:
            transcript = self.query_one("#chat-transcript", VerticalScroll)
        except Exception:
            return True

        # Cancel any previous pending confirm (shouldn't happen — the worker
        # is exclusive — but stay defensive).
        prev = self._pending_confirm
        if prev is not None and not prev.done():
            prev.set_result(False)

        fut: Any = asyncio.get_running_loop().create_future()
        self._pending_confirm = fut
        self._confirm_counter += 1
        cid = self._confirm_counter
        label = Static(prompt, classes="chat-confirm-label", markup=False)
        yes = Button(
            confirm_label, id=f"chat-confirm-yes-{cid}",
            classes="chat-confirm-btn",
        )
        no = Button(
            cancel_label, id=f"chat-confirm-no-{cid}",
            classes="chat-confirm-btn chat-confirm-no",
        )
        row = Horizontal(
            label, yes, no,
            classes="chat-line chat-line-action chat-confirm-row",
            id=f"chat-confirm-row-{cid}",
        )
        try:
            transcript.mount(row)
            transcript.scroll_end(animate=False)
            yes.focus()  # Enter on the focused button confirms
        except Exception:
            self._pending_confirm = None
            return True
        # We're waiting on the user, not the model — drop the "thinking…"
        # spinner (the worker stays RUNNING while suspended on the Future,
        # so `_refresh_spinner` wouldn't hide it on its own).
        self._set_spinner_visible(False)
        result = False
        try:
            result = bool(await fut)
            return result
        finally:
            self._pending_confirm = None
            try:
                row.remove()
            except Exception:
                pass
            # Restore the spinner only when we're proceeding to run; on
            # decline / cancel the turn is ending and it stays hidden.
            self._set_spinner_visible(result)

    async def _resolve_stage(
        self,
        stage: "Stage",
        policy: "StagePolicy",
        *,
        deps_ok: bool,
        execute,
        skip_reason: str | None = None,
    ) -> "StageOutcome":
        """Resolve one pipeline stage against its policy + dependencies.

        ``execute`` is an async thunk. It signals "did not actually run"
        (e.g. a dedup/short-circuit) by returning ``False``; any other
        return value (including ``None`` from a fire-and-forget) counts
        as DONE, and raising counts as FAILED.
        """
        def _resolved(
            outcome: "StageOutcome", *, reason: str | None = None,
        ) -> "StageOutcome":
            # One structured event per stage so "why didn't it save?" is
            # answerable from telemetry/logs.
            self._emit_telemetry("chat.pipeline.stage", {
                "stage": str(stage),
                "policy": str(policy),
                "outcome": str(outcome),
                "skip_reason": reason or "",
            })
            self._update_pipeline_stage(stage, outcome)
            return outcome

        if not deps_ok or policy is StagePolicy.SKIP:
            reason = skip_reason or "skipped"
            self._mark_stage_skipped(stage, reason=reason)
            return _resolved(StageOutcome.SKIPPED, reason=reason)
        if policy is StagePolicy.ASK and not await self._confirm_stage(stage):
            self._mark_stage_skipped(stage, reason="declined")
            return _resolved(StageOutcome.SKIPPED, reason="declined")
        self._mark_stage_started(stage)
        try:
            result = await execute()
        except Exception as exc:  # noqa: BLE001 — surface, don't crash pipeline
            self._mark_stage_failed(stage, exc)
            return _resolved(StageOutcome.FAILED, reason=str(exc))
        if result is False:
            reason = skip_reason or "skipped"
            self._mark_stage_skipped(stage, reason=reason)
            return _resolved(StageOutcome.SKIPPED, reason=reason)
        self._mark_stage_done(stage)
        return _resolved(StageOutcome.DONE)

    async def _drive_production_pipeline(
        self, spec: "ModeSpec", *, task: str, mage_result: Any,
    ) -> "dict[Stage, StageOutcome]":
        """Drive SAVE → BASELINE → EVOLVE for Production.

        Reproduces the legacy production flow exactly: save to Memory,
        backfill the chain id onto the `Read full` payload, record a
        baseline run, and (only if the baseline succeeded) kick off
        evolution. The skip-cascade falls out of the dependency model —
        a `None`-returning save short-circuits BASELINE + EVOLVE.
        """
        out: dict[Stage, StageOutcome] = {
            Stage.GENERATE: StageOutcome.DONE,
            Stage.PREVIEW: StageOutcome.DONE,
        }
        saved: dict[str, tuple[str, str]] = {}

        async def _do_save():
            result = await self._save_chain_production(
                task=task, mage_result=mage_result,
            )
            if result is None:
                return False  # dedup / failed → not done
            chain_id, display_name = result
            saved["pair"] = result
            self._production_chain_id = chain_id
            if self._last_chain_action_payload is not None:
                self._last_chain_action_payload["chain_id"] = chain_id
                self._last_chain_action_payload["display_name"] = display_name

        out[Stage.SAVE] = await self._resolve_stage(
            Stage.SAVE, spec.save,
            deps_ok=out[Stage.GENERATE] is StageOutcome.DONE,
            execute=_do_save,
        )

        baseline: dict[str, Any] = {}

        async def _do_baseline():
            chain_id, display_name = saved["pair"]
            card = await self._record_production_baseline(
                chain_id=chain_id,
                display_name=display_name,
                task=task,
                mage_result=mage_result,
            )
            if card is None:
                return False  # baseline failed → no evolve
            baseline["card"] = card

        out[Stage.BASELINE] = await self._resolve_stage(
            Stage.BASELINE, spec.baseline,
            deps_ok=out[Stage.SAVE] is StageOutcome.DONE,
            execute=_do_baseline,
            skip_reason=self._dep_skip_reason(out, needs=(Stage.SAVE,)),
        )
        # In Production the baseline *is* the chain's run, so light the
        # strip's Run cell to match (Production has no standalone RUN stage).
        if out[Stage.BASELINE] is StageOutcome.DONE:
            out[Stage.RUN] = StageOutcome.DONE
            self._update_pipeline_stage(Stage.RUN, StageOutcome.DONE)

        async def _do_evolve():
            chain_id, display_name = saved["pair"]
            chain_content = getattr(mage_result, "chain_dict", None)
            if not isinstance(chain_content, dict):
                chain_content = None
            self._kickoff_evolution(
                chain_id=chain_id,
                display_name=display_name,
                base_chain_content=chain_content,
            )

        out[Stage.EVOLVE] = await self._resolve_stage(
            Stage.EVOLVE, spec.evolve,
            deps_ok=out[Stage.BASELINE] is StageOutcome.DONE,
            execute=_do_evolve,
            skip_reason=self._dep_skip_reason(out, needs=(Stage.BASELINE,)),
        )
        return out

    async def _save_chain_production(
        self, *, task: str, mage_result: Any, validated: bool = True,
    ) -> tuple[str, str] | None:
        """Persist the freshly-generated chain to Memory and
        surface its ``entity_id`` in the transcript.

        Returns ``(chain_id, display_name)`` on success, ``None``
        on any error (already surfaced as a chat system line +
        logged). The downstream baseline run reuses both values
        without re-deriving them.

        ``validated`` (Decision 2): when ``False`` the chain is saved
        but stamped with an ``unvalidated`` tag — the agent was never
        successfully run, so BASELINE/EVOLVE are expected to skip. The
        default (``True``) reflects today's Production flow, where the
        baseline run validates the chain right after the save.

        Defensive at every layer — the production gate in
        `watch_mode` already prevents users from being in
        Production mode without a Memory facade, but we
        double-check here so a programmatic mis-call (test, future
        slash command, etc.) doesn't blow up the worker.
        """
        memory = getattr(self.app, "memory", None)
        if memory is None:
            _log.error(
                "production save aborted: app.memory is None "
                "(should have been blocked by the mode gate)",
            )
            self._post_line(
                "system",
                "Memory facade isn't wired — can't save the chain.",
                severity="error",
            )
            return None

        chain_dict = getattr(mage_result, "chain_dict", None) or {}
        if not chain_dict:
            _log.warning(
                "production save skipped: MAGE returned no chain_dict",
            )
            self._post_line(
                "system",
                "Agent chain generator returned no chain to save.",
                severity="warning",
            )
            return None

        # Phase 3 P2 — Conflict resolution. Strip an optional
        # `[FORCE]` marker so users can re-save the same task on
        # purpose; otherwise look up the task-hash tag and bail
        # cleanly when a previous save already exists.
        force, task = self._extract_force_marker(task)
        task_hash = self._task_hash(task)
        if not force:
            existing = self._find_existing_chain_by_task_hash(
                memory, task_hash,
            )
            if existing is not None:
                existing_id, existing_name = existing
                _log.info(
                    "dedup: task already saved as chain %s; "
                    "skipping fresh save",
                    existing_id,
                )
                self._post_line(
                    "assistant",
                    (
                        f"⚠ You've already saved this exact task as "
                        f"`{existing_id}`"
                        + (f" (“{existing_name}”)" if existing_name else "")
                        + ".\n  Inspect it with `/run {existing_id}`, "
                        f"replay its dataset via "
                        f"`/dataset run {existing_id}`, or "
                        "prepend `[FORCE]` to your task to save a "
                        "fresh copy anyway."
                    ).replace("{existing_id}", existing_id),
                )
                return None

        # Parse to a ReasoningChain so CARE-side metadata can
        # be injected before the SDK serialises. Falls back to
        # the raw dict if mmar_carl isn't installed — Memory's
        # `save_chain` accepts either.
        chain: Any = chain_dict
        try:
            from mmar_carl import ReasoningChain

            chain = ReasoningChain.from_dict(
                chain_dict, use_typed_steps=True,
            )
        except ImportError:
            _log.info(
                "mmar_carl not installed — saving chain_dict as-is",
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "chain parse failed (%s); saving raw chain_dict",
                exc,
            )

        # Derive the display name from the user's task — first
        # 60 chars of the first non-empty line.
        first_line = next(
            (
                line.strip()
                for line in (task or "").splitlines()
                if line.strip()
            ),
            "Untitled chain",
        )
        display_name = (
            first_line if len(first_line) <= 60
            else first_line[:57].rstrip() + "…"
        )

        # MAGE metadata for evolution + dataset bookkeeping.
        mage_meta: dict[str, Any] = {}
        meta_obj = getattr(mage_result, "metadata", None)
        if meta_obj is not None:
            try:
                from pydantic import BaseModel

                if isinstance(meta_obj, BaseModel):
                    mage_meta = meta_obj.model_dump()
                elif isinstance(meta_obj, dict):
                    mage_meta = dict(meta_obj)
            except Exception:
                # Defensive — never let metadata serialisation
                # block the save itself.
                mage_meta = {}

        domain = mage_meta.get("domain")
        # Provenance tag set — every Production save carries:
        #   source:chat-prod        identifies the entry-point
        #   task-hash:<sha256-12>   first 12 hex chars of the
        #                           task sha256, so dedup /
        #                           "you've saved this before"
        #                           lookups are O(1) via
        #                           memory.search_hits(tag=...)
        #   mage:<mode>             generation mode (fast / deep)
        #   mage-model:<model>      LLM model id (best-effort —
        #                           skipped when missing)
        tags = ["source:chat-prod"]
        tags.append(f"task-hash:{task_hash}")
        if not validated:
            # Decision 2 — the chain was saved without a successful run.
            tags.append("unvalidated")
        mage_mode_value = (
            (mage_meta.get("mode") or "").strip()
            or (getattr(self.app.config.mage, "mode", "") or "").strip()
        )
        if mage_mode_value:
            tags.append(f"mage:{mage_mode_value}")
        model_value = (
            (mage_meta.get("model") or "").strip()
            or (getattr(self.app.config.mage, "model", "") or "").strip()
        )
        if model_value:
            # Tag values can't contain whitespace cleanly; slugify
            # the model id (e.g. `anthropic/claude-3.5-sonnet` →
            # `anthropic-claude-3-5-sonnet`).
            tags.append(f"mage-model:{self._slugify_tag(model_value)}")

        _log.info(
            "saving chain to Memory: name=%r domain=%s tags=%s",
            display_name, domain, tags,
        )
        try:
            chain_id = memory.save_chain(
                chain,
                name=display_name,
                query=task,
                domain=domain if isinstance(domain, str) else None,
                mage_metadata=mage_meta or None,
                tags=tags,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "Memory.save_chain failed: %s", exc, exc_info=True,
            )
            detail = self._http_error_detail(exc)
            message = f"Couldn't save chain to Memory: {exc}"
            if detail:
                message += f" — {detail}"
            self._post_line(
                "system",
                message,
                severity="error",
            )
            return None

        _log.info("chain saved: id=%s", chain_id)
        self._post_line(
            "assistant",
            (
                f"✓ Saved chain `{chain_id}` to Memory as "
                f"“{display_name}”. Inspect with `/library`, run with "
                f"`/run {chain_id}`, deploy with "
                f"`/deploy {chain_id} --channel latest`, copy the id with Ctrl+Y."
            ),
        )
        return chain_id, display_name

    @staticmethod
    def _http_error_detail(exc: Exception) -> str | None:
        """Best-effort extraction of a server's JSON ``detail`` from an
        httpx error so 4xx/5xx responses surface the *why* (e.g. CARL
        DAG validation messages) instead of a bare status line. Returns
        ``None`` when the exception carries no parseable response body."""
        response = getattr(exc, "response", None)
        if response is None:
            return None
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001
            return None
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
        return None

    # ------------------------------------------------------------------
    # Production-mode baseline (Phase 4 P0)
    # ------------------------------------------------------------------

    async def _record_production_baseline(
        self,
        *,
        chain_id: str,
        display_name: str,
        task: str,
        mage_result: Any,
    ) -> str | None:
        """Run the freshly-saved chain once on the original task
        and persist the result as a `memory_card` tagged
        ``dataset-entry:<chain_id>`` (per Phase-0 decision §2).

        Seeds the chain's quality dataset with one entry so the
        user can measure regression on future runs / evolution
        winners against this baseline.

        Returns the persisted ``memory_card`` entity_id on
        success, ``None`` on any error (already surfaced as a
        chat system line + logged).
        """
        memory = getattr(self.app, "memory", None)
        if memory is None:
            # Should never happen — the save_chain step would
            # have returned None already. Defensive.
            return None

        chain_dict = getattr(mage_result, "chain_dict", None) or {}
        if not chain_dict:
            return None  # save_chain already warned

        self._post_line(
            "tool", "Running baseline to seed dataset…",
        )
        run_result = await self._execute_chain_interactive(
            task=task, chain_dict=chain_dict,
        )
        if run_result is None:
            # _execute_chain_interactive already posted the error
            # system line — surface a hint about the missing
            # dataset entry so the user knows what didn't happen.
            self._post_line(
                "system",
                "Baseline run failed — no dataset entry recorded. "
                "Re-run later via `/run <chain_id>` to seed.",
                severity="warning",
            )
            return None

        from care.runtime.run_recorder import record_run_completion

        _log.info(
            "recording baseline for chain_id=%s", chain_id,
        )
        try:
            completion = record_run_completion(
                memory,
                agent_entity_id=chain_id,
                agent_name=display_name,
                result=run_result,
                query=task,
                extra_tags=[f"dataset-entry:{chain_id}"],
            )
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "record_run_completion failed for chain_id=%s: %s",
                chain_id, exc, exc_info=True,
            )
            self._post_line(
                "system",
                f"Baseline ran but couldn't be recorded: {exc}",
                severity="error",
            )
            return None

        status = completion.summary.status_label
        card_id = completion.memory_card_entity_id
        _log.info(
            "baseline recorded: card_id=%s status=%s", card_id, status,
        )
        # Show the actual answer too — users want to see what
        # the chain produced, not just "baseline saved".
        answer = self._format_carl_result(run_result)
        self._post_line("assistant", answer)
        self._post_line(
            "assistant",
            (
                f"✓ Baseline recorded as run `{card_id}` "
                f"(status: {status}) — that's the run/dataset id (for "
                "measuring future runs), NOT the chain. To deploy, use the "
                f"chain id: `/deploy {chain_id} --channel latest`."
            ),
        )
        return card_id

    # ------------------------------------------------------------------
    # Dataset commands (Phase 4 P1) — /dataset list / add / run
    # ------------------------------------------------------------------

    _DATASET_ENTRY_PREFIX = "dataset-entry:"

    async def _dataset_list(self, args: list[str]) -> None:
        if not args:
            self._post_line(
                "system",
                t("chat.dataset.listNeedsId"),
                severity="warning",
            )
            return
        chain_id = args[0]
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._post_line(
                "system",
                t("chat.dataset.listNoMemory"),
                severity="warning",
            )
            return
        entries = self._collect_dataset_entries(memory, chain_id)
        if not entries:
            self._post_line(
                "system",
                t("chat.dataset.noEntries", chainId=chain_id),
            )
            return
        # Render a compact summary — each entry's task + status.
        lines = [f"Dataset for `{chain_id}` ({len(entries)} entries):"]
        for entry in entries:
            row = entry.get("task") or entry.get("name") or "(no task)"
            status = entry.get("status") or "—"
            card_id = entry.get("entity_id") or entry.get("id") or "?"
            label = row if len(row) <= 60 else row[:57] + "…"
            lines.append(f"  • {label}  [status: {status}]  ({card_id})")
        self._post_line("assistant", "\n".join(lines))

    async def _dataset_add(self, args: list[str]) -> None:
        parsed = self._parse_dataset_add(args)
        if parsed is None:
            self._post_line(
                "system",
                t("chat.dataset.addUsage"),
                severity="warning",
            )
            return
        chain_id, task, expected, rubric = parsed
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._post_line(
                "system",
                t("chat.dataset.addNoMemory"),
                severity="warning",
            )
            return
        if not task or not expected:
            self._post_line(
                "system",
                t("chat.dataset.addNeedsTaskExpected"),
                severity="warning",
            )
            return

        from datetime import datetime, timezone

        finished_at = datetime.now(timezone.utc)
        run_id = f"dataset-{finished_at.strftime('%Y%m%dT%H%M%SZ')}"
        # The entry IS the test case — task + expected. CARL hasn't
        # run it yet (that's `/dataset run`'s job); we save with
        # `status:pending` so the next /dataset run pass can flip
        # it after diffing.
        content = {
            "kind": "dataset-entry",
            "chain_id": chain_id,
            "task": task,
            "expected": expected,
            # Phase 4 P3 — Optional LLM-as-judge prompt template
            # the scorer uses instead of substring matching.
            # Placeholders ``{actual}`` and ``{expected}`` are
            # substituted at run-time. Empty = use substring.
            "rubric": rubric,
            "actual": None,
            "status": "pending",
            "created_at": finished_at.isoformat(),
            "run_id": run_id,
        }
        tags = [
            f"{self._DATASET_ENTRY_PREFIX}{chain_id}",
            "agent_run",
            f"agent:{chain_id}",
            "status:pending",
        ]
        # Provenance tag — surface "this entry uses an LLM
        # judge" so a future `/dataset list` filter could
        # split rule-based vs judge-based entries without
        # reading every card body.
        if rubric:
            tags.append("scorer:rubric")
        name = (
            f"dataset · {chain_id} · "
            f"{task if len(task) <= 40 else task[:37] + '…'}"
        )
        try:
            card_id = memory.save_memory_card(
                content,
                name=name,
                tags=tags,
                when_to_use=(
                    f"Dataset test case for chain {chain_id}. "
                    "Re-run via /dataset run."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "/dataset add failed: %s", exc, exc_info=True,
            )
            self._post_line(
                "system",
                t("chat.dataset.saveFailed", error=exc),
                severity="error",
            )
            return
        _log.info(
            "/dataset add: card_id=%s chain=%s task_len=%d",
            card_id, chain_id, len(task),
        )
        # §3 P1 — record the dataset entry in the session
        # artifact store so it surfaces in /artifacts. Best-
        # effort; failures log + continue.
        try:
            self.artifact_store.append_dataset_row(
                row=content,
                title=name,
                summary=task[:80],
                origin={
                    "chain_id": chain_id,
                    "card_id": card_id,
                    "kind": "dataset-entry",
                    "status": "pending",
                },
            )
        except Exception:  # noqa: BLE001
            _log.exception(
                "failed to stash dataset entry in artifact store"
            )
        self._post_line(
            "assistant",
            t("chat.dataset.added", cardId=card_id, chainId=chain_id),
        )

    async def _dataset_run(self, args: list[str]) -> None:
        if not args:
            self._post_line(
                "system",
                t("chat.dataset.runNeedsId"),
                severity="warning",
            )
            return
        chain_id = args[0]
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._post_line(
                "system",
                t("chat.dataset.runNoMemory"),
                severity="warning",
            )
            return
        entries = self._collect_dataset_entries(memory, chain_id)
        if not entries:
            self._post_line(
                "system",
                t("chat.dataset.noEntries", chainId=chain_id),
            )
            return

        # Pull the saved chain once — every entry runs against it.
        try:
            chain_dict = memory.get_chain(chain_id)
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "/dataset run get_chain failed: %s",
                exc, exc_info=True,
            )
            self._post_line(
                "system",
                t("chat.dataset.fetchChainFailed", chainId=chain_id, error=exc),
                severity="error",
            )
            return

        passed = 0
        failed = 0
        self._post_line(
            "tool", t("chat.dataset.running", count=len(entries)),
        )
        for idx, entry in enumerate(entries, start=1):
            task = entry.get("task") or ""
            expected = entry.get("expected") or ""
            rubric = entry.get("rubric") or ""
            if not task:
                continue
            result = await self._execute_chain_interactive(
                task=task,
                chain_dict=chain_dict,
                dataset_id=chain_id,
            )
            if result is None:
                failed += 1
                self._post_line(
                    "tool",
                    t(
                        "chat.dataset.entryFailed",
                        idx=idx,
                        total=len(entries),
                    ),
                )
                continue
            actual = self._format_carl_result(result).strip()
            # Phase 4 P3 — entries with a `rubric` template
            # delegate to the LLM judge; everything else uses
            # the substring scorer. Judge failures (LLM
            # unavailable, malformed verdict, network) fall
            # back to substring so one flaky rubric doesn't
            # abort the run.
            scorer_label = "substring"
            if rubric:
                verdict = await self._judge_with_rubric(
                    actual=actual,
                    expected=expected,
                    rubric=rubric,
                )
                if verdict is None:
                    ok = self._dataset_entry_passes(actual, expected)
                    scorer_label = "rubric→substring"
                else:
                    ok = verdict
                    scorer_label = "rubric"
            else:
                ok = self._dataset_entry_passes(actual, expected)
            if ok:
                passed += 1
                self._post_line(
                    "tool",
                    t(
                        "chat.dataset.entryPassed",
                        idx=idx,
                        total=len(entries),
                        scorer=scorer_label,
                    ),
                )
            else:
                failed += 1
                preview = (
                    actual if len(actual) <= 60 else actual[:57] + "…"
                )
                self._post_line(
                    "tool",
                    t(
                        "chat.dataset.entryMismatch",
                        idx=idx,
                        total=len(entries),
                        scorer=scorer_label,
                        preview=repr(preview),
                    ),
                )
            # §3 P1 — record the scored dataset row in the
            # session artifact store (one row per entry).
            try:
                preview_actual = (
                    actual if len(actual) <= 120 else actual[:117] + "…"
                )
                scored_row = {
                    "kind": "dataset-scored",
                    "chain_id": chain_id,
                    "task": task,
                    "expected": expected,
                    "actual": actual,
                    "status": "pass" if ok else "fail",
                    "scorer": scorer_label,
                    "rubric": rubric,
                }
                self.artifact_store.append_dataset_row(
                    row=scored_row,
                    title=(
                        f"dataset score · {chain_id} · "
                        f"{idx}/{len(entries)}"
                    ),
                    summary=(
                        f"{'pass' if ok else 'fail'}: {preview_actual}"
                    ),
                    origin={
                        "chain_id": chain_id,
                        "entry_index": idx,
                        "score": 1.0 if ok else 0.0,
                        "scorer": scorer_label,
                    },
                )
            except Exception:  # noqa: BLE001
                _log.exception(
                    "failed to stash dataset score in artifact store"
                )
        total = passed + failed
        self._post_line(
            "assistant",
            t("chat.dataset.runSummary", passed=passed, total=total),
        )
        # Phase 8 P2 #22 — long-running completion toast. A user
        # who fires `/dataset run` across 10+ entries probably
        # tabbed away; a toast above the footer pulls their eye
        # back the moment results land. Below the threshold the
        # transcript line is enough and a toast would feel
        # spammy.
        self._toast_dataset_run_complete(passed=passed, total=total)

    _DATASET_RUN_TOAST_THRESHOLD: int = 10

    def _toast_dataset_run_complete(
        self, *, passed: int, total: int,
    ) -> None:
        """Push a completion toast when `/dataset run` covered
        at least :data:`_DATASET_RUN_TOAST_THRESHOLD` entries.
        Best-effort — degrades silently when the host doesn't
        wire `push_toast` (test scaffolding, future headless
        runners). Severity reflects the run's pass rate so the
        toast visually communicates "good" vs "bad" without
        needing the user to scroll up."""
        if total < self._DATASET_RUN_TOAST_THRESHOLD:
            return
        push = getattr(self.app, "push_toast", None)
        if push is None:
            return
        if total == 0:
            severity = "warning"
        elif passed == total:
            severity = "success"
        elif passed == 0:
            severity = "error"
        else:
            severity = "warning"
        try:
            push(
                t("chat.dataset.runToast", passed=passed, total=total),
                severity=severity,
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("dataset-run toast failed: %s", exc)

    async def _dataset_export(self, args: list[str]) -> None:
        """``/dataset export <chain_id> <path>``  → write every
        entry to ``path`` as JSONL (one entry per line). Pluggable
        into external eval frameworks (Inspect-AI / promptfoo /
        OpenAI evals / etc.) that all accept JSONL natively.
        """
        if len(args) < 2:
            self._post_line(
                "system",
                t("chat.dataset.exportUsage"),
                severity="warning",
            )
            return
        chain_id = args[0]
        out_path = args[1]
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._post_line(
                "system",
                t("chat.dataset.exportNoMemory"),
                severity="warning",
            )
            return
        entries = self._collect_dataset_entries(memory, chain_id)
        if not entries:
            self._post_line(
                "system",
                t("chat.dataset.noEntriesExport", chainId=chain_id),
                severity="warning",
            )
            return

        import json
        from pathlib import Path

        target = Path(out_path).expanduser().resolve()
        # Create parent dirs so users can `/dataset export X
        # ~/evals/care/X.jsonl` without pre-mkdir.
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._post_line(
                "system",
                t("chat.dataset.mkdirFailed", target=target, error=exc),
                severity="error",
            )
            return

        try:
            with target.open("w", encoding="utf-8") as fp:
                for entry in entries:
                    payload = {
                        "chain_id": chain_id,
                        "card_id": entry.get("entity_id"),
                        "task": entry.get("task") or "",
                        "expected": entry.get("expected") or "",
                        "rubric": entry.get("rubric") or "",
                        "actual": entry.get("actual"),
                        "status": entry.get("status"),
                    }
                    fp.write(json.dumps(payload, ensure_ascii=False))
                    fp.write("\n")
        except OSError as exc:
            self._post_line(
                "system",
                t("chat.dataset.writeFailed", target=target, error=exc),
                severity="error",
            )
            return

        _log.info(
            "/dataset export: chain=%s entries=%d path=%s",
            chain_id, len(entries), target,
        )
        self._post_line(
            "assistant",
            t(
                "chat.dataset.exported",
                count=len(entries),
                chainId=chain_id,
                target=target,
            ),
        )

    # ------- dataset helpers -------------------------------------------

    @staticmethod
    def _parse_dataset_add(
        args: list[str],
    ) -> tuple[str, str, str, str] | None:
        """Parse ``/dataset add <chain_id> "<task>" --expected
        "<output>" [--rubric "<template>"]`` into
        ``(chain_id, task, expected, rubric)``.

        Returns ``None`` when there aren't enough positional
        arguments. Empty / missing ``--expected`` and
        ``--rubric`` are permitted at the parse level — callers
        validate (only ``--expected`` is required for the
        substring scorer; ``--rubric`` is Phase 4 P3's
        opt-in LLM-judge override).
        """
        if len(args) < 2:
            return None
        chain_id = args[0]
        rest = args[1:]
        expected = ""
        rubric = ""
        task_parts: list[str] = []
        i = 0
        while i < len(rest):
            token = rest[i]
            if token == "--expected" and i + 1 < len(rest):
                expected = rest[i + 1]
                i += 2
                continue
            if token == "--rubric" and i + 1 < len(rest):
                rubric = rest[i + 1]
                i += 2
                continue
            task_parts.append(token)
            i += 1
        return (
            chain_id,
            " ".join(task_parts).strip(),
            expected.strip(),
            rubric.strip(),
        )

    @classmethod
    def _collect_dataset_entries(
        cls, memory: Any, chain_id: str,
    ) -> list[dict[str, Any]]:
        """Return every dataset entry for ``chain_id`` as a flat
        list of dicts (``task`` / ``expected`` / ``actual`` /
        ``status`` / ``entity_id`` / ``name``).

        Walks every memory_card the SDK returns and filters on the
        ``dataset-entry:<chain_id>`` tag. List-by-tag isn't on the
        SDK surface yet so we list + filter — fine for v1 dataset
        sizes (<200 entries per chain).
        """
        tag = f"{cls._DATASET_ENTRY_PREFIX}{chain_id}"
        try:
            rows = memory.list_entities(
                entity_type="memory_card", limit=500,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("list_entities failed: %s", exc)
            return []
        out: list[dict[str, Any]] = []
        for row in rows or []:
            tags = (
                row.get("tags")
                or (row.get("meta") or {}).get("tags")
                or []
            )
            if tag not in tags:
                continue
            content = row.get("content") or row.get("content_json") or {}
            if not isinstance(content, dict):
                content = {}
            out.append({
                "entity_id": row.get("entity_id") or row.get("id"),
                "name": row.get("name") or row.get("display_name"),
                "task": content.get("task"),
                "expected": content.get("expected"),
                "rubric": content.get("rubric") or "",
                "actual": content.get("actual"),
                "status": (
                    content.get("status")
                    or next(
                        (
                            t.split(":", 1)[1]
                            for t in tags
                            if t.startswith("status:")
                        ),
                        None,
                    )
                ),
            })
        return out

    @staticmethod
    def _dataset_entry_passes(actual: str, expected: str) -> bool:
        """First-cut scorer: case-insensitive substring match
        of the expected text inside the actual output. Cheap +
        deterministic — used as the default + as the fallback
        when an LLM judge can't be reached.
        """
        if not expected:
            return False
        return expected.strip().lower() in actual.strip().lower()

    # ------------------------------------------------------------------
    # /forget — delete a chain + its dataset (privacy P3)
    # ------------------------------------------------------------------

    async def _forget_chain_and_dataset(
        self, chain_id: str, *, force: bool,
    ) -> None:
        """Drive the `/forget` command.

        ``force=False`` is the preview mode: gather counts and
        post a system line explaining what would happen,
        instructing the user to re-run with `--force`. No
        deletions land. ``force=True`` calls
        :meth:`client._delete_entity` on the chain itself plus
        every dataset card tagged
        :data:`_DATASET_ENTRY_PREFIX` ``+ chain_id``.

        Memory's delete is a soft-delete (sets ``deleted_at``)
        so the chain stays recoverable via Memory's trash —
        `/forget` is intentionally the lighter-weight escape
        hatch, not a permanent purge.
        """
        memory = getattr(self.app, "memory", None)
        if memory is None:
            return  # gated upstream; defensive

        # Verify the chain exists so the preview / delete tells
        # the truth. Older Memory deployments don't have
        # `get_chain` — degrade by skipping the verification.
        chain_exists = True
        get_chain = getattr(memory, "get_chain", None)
        if callable(get_chain):
            try:
                got = await _maybe_await(get_chain, chain_id)
                chain_exists = got is not None
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "/forget get_chain %s failed: %s",
                    chain_id, exc,
                )
                chain_exists = True  # err on the side of trying

        dataset = self._collect_dataset_entries(memory, chain_id)
        if not chain_exists and not dataset:
            self._post_line(
                "system",
                f"No saved chain `{chain_id}` and no dataset "
                f"entries — nothing to forget.",
                severity="warning",
            )
            return

        if not force:
            chain_blurb = (
                f"chain `{chain_id}`" if chain_exists
                else "no chain row"
            )
            self._post_line(
                "assistant",
                f"`/forget {chain_id}` would delete: "
                f"{chain_blurb} + {len(dataset)} dataset "
                f"entr{'y' if len(dataset) == 1 else 'ies'}. "
                f"Re-run with `--force` to actually delete "
                f"(`/forget {chain_id} --force`).",
            )
            return

        # Force path.
        client = getattr(memory, "client", None)
        delete_fn = getattr(client, "_delete_entity", None)
        if not callable(delete_fn):
            self._post_line(
                "system",
                "Memory client doesn't expose _delete_entity — "
                "can't forget via this facade.",
                severity="error",
            )
            return

        chain_ok = True
        if chain_exists:
            try:
                await _maybe_await(delete_fn, "chain", chain_id)
            except Exception as exc:  # noqa: BLE001
                _log.error(
                    "/forget failed for chain %s: %s",
                    chain_id, exc, exc_info=True,
                )
                self._post_line(
                    "system",
                    f"Couldn't delete chain `{chain_id}`: {exc}",
                    severity="error",
                )
                chain_ok = False

        deleted_cards = 0
        failed_cards = 0
        for entry in dataset:
            card_id = entry.get("entity_id")
            if not card_id:
                continue
            try:
                await _maybe_await(
                    delete_fn, "memory_card", card_id,
                )
                deleted_cards += 1
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "/forget failed for card %s: %s",
                    card_id, exc,
                )
                failed_cards += 1

        if chain_ok and failed_cards == 0:
            self._post_line(
                "assistant",
                f"✓ Forgot chain `{chain_id}` "
                f"({deleted_cards} dataset "
                f"entr{'y' if deleted_cards == 1 else 'ies'} "
                "also deleted).",
            )
        else:
            self._post_line(
                "system",
                f"/forget {chain_id}: partial — chain "
                f"{'ok' if chain_ok else 'FAILED'}, dataset "
                f"{deleted_cards} ok / {failed_cards} failed.",
                severity="warning",
            )

    # ------------------------------------------------------------------
    # LLM-as-judge scoring (Phase 4 P3)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_judge_verdict(text: str) -> bool | None:
        """Project an LLM judge's reply into a pass/fail bool.
        Returns ``True`` if the model said PASS, ``False`` for
        FAIL, and ``None`` for "I couldn't tell" (the caller
        falls back to the substring scorer in that case).

        Case-insensitive; we look for PASS/FAIL as whole tokens
        so a "PASSAGE" in an explanatory paragraph doesn't
        accidentally count as PASS.
        """
        import re

        if not text:
            return None
        upper = text.upper()
        has_pass = bool(re.search(r"\bPASS\b", upper))
        has_fail = bool(re.search(r"\bFAIL\b", upper))
        if has_pass and not has_fail:
            return True
        if has_fail and not has_pass:
            return False
        return None

    @staticmethod
    def _format_rubric_prompt(
        rubric: str, *, actual: str, expected: str,
    ) -> str | None:
        """Substitute ``{actual}`` / ``{expected}`` placeholders
        in the user's rubric template. Returns ``None`` when the
        template references unknown placeholders (the LLM call
        is skipped and the caller falls back to the substring
        scorer — better than crashing the whole `/dataset run`
        on one bad entry).

        Uses :py:meth:`str.format_map` over plain ``str.format``
        so braces in the template body don't trigger
        ``IndexError`` on bare ``{}``.
        """

        class _Mapping(dict):
            def __missing__(self, key):  # noqa: ANN001 — duck type
                raise KeyError(key)

        try:
            return rubric.format_map(
                _Mapping(actual=actual, expected=expected),
            )
        except (KeyError, ValueError, IndexError):
            return None

    async def _judge_with_rubric(
        self,
        *,
        actual: str,
        expected: str,
        rubric: str,
    ) -> bool | None:
        """Run the LLM-judge for one dataset entry.

        Returns ``True`` / ``False`` for a confident verdict,
        ``None`` to signal "judge unavailable / inconclusive"
        so the caller can fall back to the substring scorer.
        Never raises into the dataset-run loop — a flaky LLM
        endpoint must not abort the whole pass.
        """
        prompt = self._format_rubric_prompt(
            rubric, actual=actual, expected=expected,
        )
        if prompt is None:
            return None
        cfg = getattr(self.app, "config", None)
        if cfg is None or getattr(cfg, "mage", None) is None:
            return None
        try:
            from care.runtime.llm_client import build_llm_client
        except Exception:  # noqa: BLE001
            return None
        try:
            client = build_llm_client(cfg.mage)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "rubric judge unavailable (build_llm_client): %s", exc,
            )
            return None
        model = getattr(cfg.mage, "model", "") or "gpt-4o-mini"
        try:
            import asyncio

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an evaluation judge. Read the "
                            "rubric and return exactly one token: "
                            "PASS or FAIL."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("rubric judge call failed: %s", exc)
            return None
        try:
            text = response.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return None
        return self._parse_judge_verdict(text)

    # ------------------------------------------------------------------
    # Production-mode evolution kickoff (Phase 5 P1)
    # ------------------------------------------------------------------

    def _session_base_chain_id(self) -> str:
        """Best-effort id of the chain the user is most likely to evolve:
        the most-recent saved chain in this session (the `Read full` /
        save flow stamps `_last_chain_action_payload`). Empty when nothing
        has been saved yet — the setup modal then opens cold so the user
        types the chain id themselves."""
        payload = self._last_chain_action_payload
        if isinstance(payload, dict):
            cid = payload.get("chain_id")
            if isinstance(cid, str) and cid:
                return cid
        return ""

    def _open_evolution_dashboard(self) -> None:
        """Push the evolution runs dashboard."""
        try:
            from care.screens.evolution_dashboard import EvolutionDashboard

            self.app.push_screen(EvolutionDashboard())
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system",
                t("chat.evolution.dashboardFailed", error=exc),
                severity="error",
            )

    def _open_evolution_intro(self, *, dismiss_to_dashboard: bool = False) -> None:
        """Show the evolution primer modal.

        * Mode-row quick link — intro only; «Open runs list» continues.
        * Bare ``/evolution`` — both buttons open the dashboard after
          the primer (Esc still cancels).
        """
        from care.screens.evolution_intro import (
            EvolutionIntroModal,
            EvolutionIntroResult,
        )

        def _on_dismiss(result: EvolutionIntroResult | None) -> None:
            if result is None or not result.open_dashboard:
                return
            self._open_evolution_dashboard()

        self.app.push_screen(
            EvolutionIntroModal(dismiss_to_dashboard=dismiss_to_dashboard),
            _on_dismiss,
        )

    def _open_evolution_setup(self, chain_id: str = "") -> None:
        """Open the shared :class:`EvolutionLaunchModal` (dataset + budget
        + rubric), pre-bound to ``chain_id`` when known. Routes through
        ``app._push_evolution_for`` so `/evolution` and the Library /
        Inspection Evolve buttons converge on one launch path.

        Mirrors the platform-facade guard the other `/evolution`
        sub-commands use so the user gets an inline hint instead of a
        silent no-op when Platform isn't wired."""
        platform = getattr(self.app, "platform", None)
        if platform is None:
            self._post_line(
                "system",
                t("chat.evolution.setupNoPlatform"),
                severity="warning",
            )
            return
        opener = getattr(self.app, "_push_evolution_for", None)
        if not callable(opener):
            self._post_line(
                "system",
                t("chat.evolution.setupUnavailable"),
                severity="warning",
            )
            return
        opener(chain_id or "")

    def _kickoff_evolution(
        self,
        *,
        chain_id: str,
        display_name: str,
        base_chain_content: dict[str, Any] | None = None,
    ) -> str | None:
        """Fire-and-forget evolution submit on the freshly-saved
        chain. Synchronous because the data layer's
        `start_evolution` is sync; the network call is tiny
        (single POST returning a run id).

        Behaviours:
        * No Platform facade → `tool` line "Evolution skipped"
          (documented behaviour, not a warning — many users run
          without Platform on purpose).
        * Success → `assistant` line with the evolution run id
          + `/evolution <run_id>` hint for Phase 5 P2.
        * Submit raises → `warning` system line; the chain +
          baseline are still saved, only evolution failed.

        Returns the evolution_id on success, ``None`` otherwise.
        """
        platform = getattr(self.app, "platform", None)
        if platform is None:
            _log.info(
                "evolution skipped: Platform facade not configured",
            )
            self._post_line(
                "tool",
                t("chat.evolution.skipped"),
            )
            return None

        _log.info("kicking off evolution: base_chain_id=%s", chain_id)
        try:
            ref = platform.start_evolution(
                base_chain_id=chain_id,
                tags=["source:chat-prod", f"chain:{chain_id}"],
                base_chain_content=base_chain_content,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "Platform.start_evolution failed for chain_id=%s: %s",
                chain_id, exc, exc_info=True,
            )
            self._post_line(
                "system",
                t("chat.evolution.startFailed", error=exc, id=chain_id),
                severity="warning",
            )
            return None

        evo_id = ref.evolution_id
        status = ref.status or "queued"
        _log.info(
            "evolution started: evolution_id=%s status=%s",
            evo_id, status,
        )
        self._post_line(
            "assistant",
            t(
                "chat.evolution.started",
                id=evo_id,
                status=status,
                name=display_name,
            ),
        )
        return evo_id

    async def _stream_evolution(self, run_id: str) -> None:
        """Pump events from ``platform.stream_events(run_id)``
        into the transcript. Cancellation-safe — the worker is
        in the ``"evolution_stream"`` group, so Esc-cancel
        unwinds the loop via ``CancelledError``.
        """
        import asyncio

        platform = getattr(self.app, "platform", None)
        if platform is None:  # belt-and-suspenders
            return
        self._post_line(
            "tool",
            t("chat.evolution.watching", runId=run_id),
        )
        try:
            iterator = iter(platform.stream_events(run_id))
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "stream_events(%s) failed at open: %s",
                run_id, exc, exc_info=True,
            )
            self._post_line(
                "system",
                t("chat.evolution.streamOpenFailed", runId=run_id, error=exc),
                severity="error",
            )
            return

        count = 0
        _sentinel = object()
        try:
            while True:
                # Pump the sync generator off-thread so the UI
                # stays responsive while we wait for the next
                # SSE frame.
                try:
                    event = await asyncio.to_thread(
                        next, iterator, _sentinel,
                    )
                except Exception as exc:  # noqa: BLE001
                    _log.error(
                        "stream_events(%s) failed mid-stream: %s",
                        run_id, exc, exc_info=True,
                    )
                    self._post_line(
                        "system",
                        t("chat.evolution.streamErrored", runId=run_id, error=exc),
                        severity="error",
                    )
                    return
                if event is _sentinel:
                    break  # stream closed cleanly
                count += 1
                line = self._format_evolution_event(event)
                if line:
                    self._post_line("tool", line)
        finally:
            self._post_line(
                "tool",
                t("chat.evolution.streamEnded", count=count),
            )

    @classmethod
    def _format_evolution_event(cls, event: Any) -> str:
        """Project a single Platform SSE event dict into one
        chat line.

        Recognises the common shapes the Platform emits
        (generation_*, individual_evaluated, evolution_*,
        error) and falls back to the raw key/value list for
        anything unknown so the user still sees the payload.
        """
        if not isinstance(event, dict):
            return f"· {event}"
        kind = (
            event.get("event")
            or event.get("type")
            or event.get("kind")
            or ""
        )
        kind_lc = str(kind).lower()
        # High-frequency / chart-only frames carry no useful single-line
        # text for a linear transcript — drop them (the caller skips ""),
        # they belong in the EvolutionScreen's panes.
        if kind_lc in {
            "heartbeat",
            "cost_tick",
            "fitness_history_snapshot",
            "frontier_programs_snapshot",
            "programs_snapshot",
        }:
            return ""
        # Metric fields ride under ``event["data"]`` (the poll path wraps
        # them and the SDK's SSE normaliser does too); fall back to the
        # flat event for older/looser shapes.
        payload = event.get("data")
        if not isinstance(payload, dict):
            payload = event
        if kind_lc == "generation_started":
            gen = payload.get("generation")
            total = payload.get("total") or payload.get("max")
            if total:
                return f"▶ generation {gen}/{total} started"
            return f"▶ generation {gen} started"
        if kind_lc == "generation_completed":
            gen = payload.get("generation")
            best = payload.get("best_score") or payload.get("score")
            if best is not None:
                try:
                    return (
                        f"✓ generation {gen} complete "
                        f"(best={float(best):.3f})"
                    )
                except (TypeError, ValueError):
                    return f"✓ generation {gen} complete (best={best})"
            return f"✓ generation {gen} complete"
        if kind_lc == "individual_evaluated":
            ind_id = payload.get("individual_id") or payload.get("id")
            score = payload.get("score") or payload.get("fitness")
            if ind_id and score is not None:
                try:
                    return f"  · {ind_id}: {float(score):.3f}"
                except (TypeError, ValueError):
                    return f"  · {ind_id}: {score}"
            return f"  · evaluated: {payload}"
        if kind_lc == "best_updated":
            gen = payload.get("generation")
            best = payload.get("best_fitness") or payload.get("fitness")
            if best is not None:
                try:
                    return f"★ new best (gen {gen}): {float(best):.3f}"
                except (TypeError, ValueError):
                    return f"★ new best (gen {gen}): {best}"
            return f"★ new best (gen {gen})"
        if kind_lc == "status":
            return f"· status: {payload.get('status') or '?'}"
        if kind_lc in {"evolution_completed", "completed"}:
            winner = (
                payload.get("winner_id")
                or payload.get("best_id")
                or payload.get("individual_id")
            )
            if winner:
                return f"🏁 evolution finished — winner: {winner}"
            return "🏁 evolution finished"
        if kind_lc in {"failed", "cancelled"}:
            label = "cancelled" if kind_lc == "cancelled" else "failed"
            msg = payload.get("error") or payload.get("message") or ""
            return f"✗ evolution {label}: {msg}" if msg else f"✗ evolution {label}"
        if kind_lc == "error":
            msg = payload.get("message") or payload.get("error") or payload
            return f"✗ error: {msg}"
        # Unknown event — render kind + the remaining keys
        # compactly so the user sees the payload.
        rest = {
            k: v for k, v in payload.items()
            if k not in {"event", "type", "kind", "experiment_id"}
        }
        if kind:
            return f"· {kind}: {rest}" if rest else f"· {kind}"
        return f"· {event}"

    @classmethod
    def _format_evolution_state(
        cls, run_id: str, state: Any,
    ) -> str:
        """Project a Platform-side evolution-state dict into a
        compact multi-line block ChatScreen renders as one
        assistant message. Reads every field defensively — the
        Platform schema is younger than CARE and may carry
        either flat keys (``status``, ``generation``) or
        nested ones (``progress.generation``,
        ``best.fitness``).

        Falls back to "evolution `<id>` returned no data" when
        the payload is missing / malformed so the user sees
        something rather than a blank assistant line.
        """
        if not isinstance(state, dict):
            return (
                f"Evolution `{run_id}` returned no data (got "
                f"{type(state).__name__})."
            )
        status = cls._evo_pick(state, "status") or "unknown"
        gen_current = cls._evo_pick(
            state, "generation", "current_generation",
            "progress.generation",
        )
        gen_total = cls._evo_pick(
            state, "max_iterations", "max_generations",
            "progress.total",
        )
        pop = cls._evo_pick(state, "population_size", "population")
        best_score = cls._evo_pick(
            state, "best_score", "best.fitness",
            "best_individual.score",
        )
        pareto_size = cls._evo_pick(
            state, "pareto_front_size", "pareto_size",
        )
        if (
            pareto_size is None
            and isinstance(state.get("pareto_front"), list)
        ):
            pareto_size = len(state["pareto_front"])

        lines = [f"🧬 Evolution `{run_id}` — status: {status}"]
        if gen_current is not None or gen_total is not None:
            shown_current = "?" if gen_current is None else gen_current
            shown_total = "?" if gen_total is None else gen_total
            lines.append(
                f"  generation: {shown_current} / {shown_total}",
            )
        if pop is not None:
            lines.append(f"  population: {pop}")
        if best_score is not None:
            try:
                lines.append(f"  best score: {float(best_score):.3f}")
            except (TypeError, ValueError):
                lines.append(f"  best score: {best_score}")
        if pareto_size is not None:
            lines.append(f"  pareto front: {pareto_size} individuals")
        # §5 P1 — append a text-mode fitness curve when the
        # state payload carries per-generation history. The
        # extractor tolerates several shapes the upstream
        # Platform versions use; the renderer (same one the
        # dedicated EvolutionScreen uses) falls back to a
        # unicode sparkline when `plotext` isn't installed.
        records = cls._extract_fitness_records(state)
        if records:
            from care.runtime.fitness_plot import (
                render_fitness_plot,
            )

            plot = render_fitness_plot(records, width=60, height=8)
            if plot:
                lines.append("")
                lines.append(plot)
        return "\n".join(lines)

    @staticmethod
    def _extract_fitness_records(state: Any) -> tuple:
        """Project per-generation fitness records out of the
        state payload (§5 P1).

        Accepts the following shapes:

        * ``state["fitness_history"]`` — list of dicts with
          ``generation`` / ``gen`` and ``best_fitness`` /
          ``fitness`` / ``best`` keys.
        * ``state["generations"]`` — same shape.
        * ``state["progress"]["fitness_history"]`` — nested
          variant.

        Returns a tuple of duck-typed objects exposing
        ``generation`` + ``best_fitness`` (the
        :func:`render_fitness_plot` contract). Empty tuple
        when the payload has nothing usable.
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _Stat:
            generation: int
            best_fitness: float

        if not isinstance(state, Mapping):
            return ()
        raw: Any = None
        for path in (
            "fitness_history",
            "generations",
            "progress.fitness_history",
        ):
            cursor: Any = state
            for segment in path.split("."):
                if isinstance(cursor, Mapping) and segment in cursor:
                    cursor = cursor[segment]
                else:
                    cursor = None
                    break
            if isinstance(cursor, list) and cursor:
                raw = cursor
                break
        if raw is None:
            return ()
        out: list[_Stat] = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            gen = entry.get("generation")
            if gen is None:
                gen = entry.get("gen")
            fitness = entry.get("best_fitness")
            if fitness is None:
                fitness = entry.get("fitness")
            if fitness is None:
                fitness = entry.get("best")
            if gen is None or fitness is None:
                continue
            try:
                out.append(_Stat(
                    generation=int(gen),
                    best_fitness=float(fitness),
                ))
            except (TypeError, ValueError):
                continue
        return tuple(out)

    @staticmethod
    def _evo_pick(state: dict, *paths: str) -> Any:
        """Read the first non-None value from ``state`` across a
        list of dotted paths (e.g. ``"progress.generation"``).
        Lets the renderer accept whichever shape the Platform
        version returns without an explicit version branch."""
        for path in paths:
            cursor: Any = state
            for segment in path.split("."):
                if isinstance(cursor, dict) and segment in cursor:
                    cursor = cursor[segment]
                else:
                    cursor = None
                    break
            if cursor is not None:
                return cursor
        return None

    async def _synthesise_user_answer(
        self, *, task: str, run_result: Any,
    ) -> str | None:
        """Ad-Hoc post-processing — fold every successful step
        result into a single coherent user-facing answer via
        one LLM call. Returns the synthesised text on success,
        ``None`` when synthesis isn't applicable or the call
        failed (caller falls back to terminal-step extraction).

        Why this exists: MAGE often plans a chain like
        ``draft → review → revise → strengthen-conclusion``.
        The legacy "last successful step's result" projection
        then surfaces ONLY the strengthened conclusion — the
        user sees a stub instead of the full essay. Running a
        synthesis pass over every step's output produces the
        merged answer the user actually asked for.

        Production runs deliberately bypass this — the chain
        itself is the deliverable in Production mode.
        """
        if self._current_mode_spec().followup != "reuse":
            return None
        steps = self._collect_successful_steps(run_result)
        if len(steps) < 2:
            # Single-step (or no successful steps) — nothing
            # meaningful to merge; the legacy extractor already
            # gives the user the right answer.
            return None
        cfg = getattr(self.app, "config", None)
        if cfg is None:
            return None
        try:
            from care.runtime.llm_client import (
                LLMClientError,
                build_llm_client,
            )
        except Exception:
            return None
        try:
            client = build_llm_client(cfg.mage)
        except LLMClientError as exc:
            _log.warning("synthesis aborted (client build): %s", exc)
            return None
        prompt = self._build_synthesis_prompt(task, steps)
        self._post_line("tool", f"▶ {t('chat.stage.synthesising')}…")
        try:
            text = await asyncio.to_thread(
                self._call_synthesis_llm,
                client=client,
                model=cfg.mage.model,
                prompt=prompt,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("synthesis LLM call failed: %s", exc)
            self._post_line(
                "tool",
                "  ⎿ synthesis failed — using terminal step output",
            )
            return None
        text = (text or "").strip()
        if not text:
            self._post_line(
                "tool",
                "  ⎿ synthesis returned empty — using terminal step output",
            )
            return None
        self._post_line(
            "tool",
            f"✓ {t('chat.stage.synthesising')}",
            extra_class="chat-line-pre-answer",
        )
        # §3 P1 — stash the merged answer in the session
        # artifact store so the user can revisit it later via
        # `/artifacts`. Best-effort; appending failures get
        # logged but never propagate back to the synthesis
        # flow.
        try:
            self.artifact_store.append_synthesised_answer(
                answer=text,
                origin={
                    "iteration": getattr(self, "_iteration", 0),
                    "step_count": len(steps),
                    "task": task,
                },
            )
        except Exception:  # noqa: BLE001
            _log.exception(
                "failed to stash synthesised answer in artifact store"
            )
        return text

    @staticmethod
    def _collect_successful_steps(run_result: Any) -> list[Any]:
        """Return every step whose ``success`` is truthy AND
        whose textual ``result`` (or stringified ``result_data``)
        carries non-empty content. Order preserved so the
        synthesis pass sees the chain in execution sequence."""
        step_results = getattr(run_result, "step_results", None) or []
        kept: list[Any] = []
        for step in step_results:
            if getattr(step, "success", False) is not True:
                continue
            body = (getattr(step, "result", "") or "").strip()
            if not body:
                data = getattr(step, "result_data", None)
                if data is not None:
                    body = str(data).strip()
            if body:
                kept.append(step)
        return kept

    @staticmethod
    def _build_synthesis_prompt(task: str, steps: list[Any]) -> str:
        """Build the user-prompt for the synthesis LLM call.
        Lists every step in execution order so the model has
        the full intermediate trail when composing the merged
        answer."""
        parts = [
            "You are finalising the answer to a user request that "
            "was split into multiple intermediate steps. Below are "
            "the intermediate step results in execution order. "
            "Produce ONE coherent, complete response that directly "
            "satisfies the original user request — merge the "
            "intermediate work into a single deliverable.",
            "",
            "Rules:",
            "1. Do NOT mention the existence of intermediate steps "
            "or any meta-process. The user must see only the "
            "finished result.",
            "2. Preserve every relevant detail; do not drop "
            "content that the steps produced.",
            "3. Match the user's language (Russian → Russian, "
            "English → English, etc.).",
            "4. If the request was \"write me an essay\", return "
            "the full essay. If it was a list, return the list. "
            "Don't paraphrase the request itself.",
            "",
            f"Original user request:\n{task}",
            "",
            "Intermediate step results:",
        ]
        for idx, step in enumerate(steps, start=1):
            title = (
                getattr(step, "step_title", None)
                or f"Step {idx}"
            ).strip()
            body = (getattr(step, "result", "") or "").strip()
            if not body:
                data = getattr(step, "result_data", None)
                if data is not None:
                    body = str(data).strip()
            parts.append(f"### Step {idx} — {title}")
            parts.append(body)
            parts.append("")
        parts.append(
            "Now produce the final, merged response for the user "
            "according to the rules above."
        )
        return "\n".join(parts)

    @staticmethod
    def _call_synthesis_llm(
        *, client: Any, model: str, prompt: str,
    ) -> str:
        """Single sync OpenAI-compatible chat call used by
        :meth:`_synthesise_user_answer`. Lives as a staticmethod
        so tests can monkeypatch it without spinning a real
        client."""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""

    @staticmethod
    def _format_carl_result(result: Any) -> str:
        """Project a CARL ``ReasoningResult`` into a single
        assistant-line string.

        Thin re-export of :func:`care.carl_summary.summarise_carl_result`
        so tests can monkeypatch the screen-level seam without
        touching the data layer.
        """
        from care.carl_summary import summarise_carl_result

        return summarise_carl_result(result)

    @staticmethod
    def _format_result_summary_rows(result: Any) -> list[str]:
        """Project a MAGE generation summary into a list of
        plain-text rows fit for ``⎿``-prefixed tool sub-lines
        under the freshly-finished ``✓ Describing steps``
        stage.

        We condense the full ``MetadataSummary`` to the three
        facts users actually scan for:

        * **header** — step count + wall clock so the user
          knows the chain is built and roughly how long it
          took.
        * **provenance** — model + mode + domain, the bits
          they need to debug a surprising plan.
        * **research footprint** — memory / web hits + cold-
          start flag when any of them fired.

        The legacy stage trail (``domain_analysis,
        topology_selection, memory_research,
        capability_lookup_injected, …``) is intentionally
        dropped: that's the same trail the per-stage
        ``▶ /✓`` tool lines already render in execution
        order, so re-listing it as one comma-joined string is
        chrome.

        Returns an empty list when no summary is available;
        the caller skips emitting any sub-lines in that case.
        """
        try:
            from care.mage_summary import summarise_mage_result

            summary = summarise_mage_result(result)
        except Exception:
            return []
        if summary is None:
            return []
        rows: list[str] = []
        # 1. Header — steps + wall clock.
        steps = summary.num_steps
        step_label = ChatScreen._plural(steps, "chat.trace.step")
        time_part = ""
        if summary.generation_time_seconds:
            time_part = t(
                "chat.trace.inTime",
                secs=f"{summary.generation_time_seconds:.1f}",
            )
        rows.append(
            f"{t('chat.trace.chainReady')} · {steps} {step_label}{time_part}"
        )
        # 2. Provenance — only the parts that resolved.
        meta_parts: list[str] = []
        if summary.model:
            meta_parts.append(summary.model)
        if summary.mode:
            meta_parts.append(t("chat.trace.modeLabel", mode=summary.mode))
        if summary.domain and summary.domain != "general":
            meta_parts.append(t("chat.trace.domain", domain=summary.domain))
        if meta_parts:
            rows.append(" · ".join(meta_parts))
        # 3. Research footprint — skip when nothing fired.
        research_bits: list[str] = []
        if summary.memory_hits_used:
            research_bits.append(
                f"{summary.memory_hits_used} "
                f"{ChatScreen._plural(summary.memory_hits_used, 'chat.trace.memoryHit')}"
            )
        if summary.web_results_used:
            research_bits.append(
                f"{summary.web_results_used} "
                f"{ChatScreen._plural(summary.web_results_used, 'chat.trace.webResult')}"
            )
        if summary.was_cold_start:
            research_bits.append(t("chat.trace.coldStart"))
        if research_bits:
            rows.append(" · ".join(research_bits))
        return rows

    @staticmethod
    def _format_result_summary(result: Any) -> str:
        """Multi-row Markdown projection of the same summary
        produced by :meth:`_format_result_summary_rows`. Kept
        for the Production save path (where the metadata
        summary lands as the assistant reply) and for unit
        tests that prefer the joined-string shape."""
        rows = ChatScreen._format_result_summary_rows(result)
        if not rows:
            return t("chat.trace.generationComplete")
        # Bold the header row so it anchors the eye when
        # rendered through the Markdown widget.
        rows[0] = f"**{rows[0]}**"
        # Markdown hard-break (`"  \n"`) keeps each row on its
        # own visual line inside the single Markdown widget.
        return "  \n".join(rows)

    # ------------------------------------------------------------------
    # MagePoster event sinks → chat lines
    # ------------------------------------------------------------------

    # Friendly verbs for known MAGE stages. Surfaced in the
    # transcript instead of the raw snake_case key so the user
    # reads "Analysing domain…" rather than "domain_analysis …".
    # Unknown stages fall through to a title-cased form so we
    # never block on a missing entry — new MAGE pipeline steps
    # render legibly out of the box.
    # Raw MAGE stage key → ``chat.stage.<suffix>`` catalog key. Resolved
    # to the active UI language at render time (see _friendly_stage_label).
    _STAGE_LABEL_KEYS: dict[str, str] = {
        "domain_analysis": "domainAnalysis",
        "capability_lookup": "capabilityLookup",
        "memory_research": "memoryResearch",
        "topology_selection": "topologySelection",
        "step_planning": "planningSteps",
        "step_plan": "planningSteps",
        "dag_building": "buildingDag",
        "dag": "buildingDag",
        "step_describing": "describingSteps",
        "critique": "critique",
        "verification": "verification",
        "refine": "refine",
    }

    @classmethod
    def _friendly_stage_label(cls, stage: str) -> str:
        """Project a raw MAGE stage key onto a localized human-readable
        verb. Unknown keys get title-cased so new stages still read
        naturally without a code change."""
        raw = (stage or "").strip()
        if not raw:
            return t("chat.stage.working")
        key = cls._STAGE_LABEL_KEYS.get(raw)
        if key is not None:
            return t(f"chat.stage.{key}")
        # `domain_analysis` → "Domain analysis"; preserves the
        # spacing already used by intermediate_artifacts._STAGE_HEADERS.
        spaced = raw.replace("_", " ").replace("-", " ").strip()
        return spaced[:1].upper() + spaced[1:] if spaced else raw

    @staticmethod
    def _plural(n: int, base: str) -> str:
        """Pick the localized plural form for ``n`` from a catalog base
        with ``.one`` / ``.few`` / ``.many`` keys. English collapses to
        one/many; Russian applies the CLDR one/few/many rule."""
        if get_ui_language() == "ru":
            n100, n10 = abs(n) % 100, abs(n) % 10
            if 11 <= n100 <= 14:
                form = "many"
            elif n10 == 1:
                form = "one"
            elif 2 <= n10 <= 4:
                form = "few"
            else:
                form = "many"
        else:
            form = "one" if n == 1 else "many"
        return t(f"{base}.{form}")

    def on_stage_started(self, event: StageStarted) -> None:
        label = self._friendly_stage_label(event.stage)
        # Stash the line index keyed by the RAW stage key so the
        # completion handler can mute the started row regardless
        # of how friendly the rendered label is.
        self._post_line("tool", f"▶ {label}…")
        self._stage_started_indexes[event.stage] = self._line_counter

    def on_stage_completed(self, event: StageCompleted) -> None:
        label = self._friendly_stage_label(event.stage)
        self._post_line("tool", f"✓ {label}")
        # Phase 8 P2 #21 — mute the matching `▶ <stage>…` line
        # so the eye tracks forward progress through the
        # MAGE pipeline without confusion about which stage
        # is still active.
        self._mark_stage_started_line_done(event.stage)
        payload = self._project_stage_payload(event.result)
        # DAG terminal rendering — when the build-DAG stage lands, draw
        # the step graph inline as box-and-arrow `⎿` sub-rows so the user
        # sees the chain's shape, not just a node/edge count. MagePoster
        # emits this stage as `dag_building`; `dag` is kept as a defensive
        # alias for any path using the artifact-vocabulary name.
        if event.stage in ("dag_building", "dag"):
            self._post_dag_boxes(payload)
        # §3 P2 — stash the MAGE stage result in the artifact
        # store so the ArtifactsScreen can drill into each
        # generation's plan / DAG / critique / verification.
        # Best-effort; failures log but never block the screen.
        try:
            self.artifact_store.append_stage_payload(
                stage=event.stage,
                payload=payload,
                title=f"Agent chain generator {label}",
                origin={
                    "stage": event.stage,
                    "iteration": getattr(self, "_iteration", 0),
                },
            )
        except Exception:  # noqa: BLE001
            _log.exception(
                "failed to stash MAGE stage payload in artifact store",
            )

    @staticmethod
    def _project_stage_payload(result: Any) -> Any:
        """Best-effort projection of a MAGE stage result into a
        store-friendly shape. Tries `result.model_dump()` (Pydantic),
        falls back to ``dict(result)`` for Mapping-like, then
        ``result`` verbatim. The artifact store accepts `Any` so
        non-dict shapes pass through unchanged."""
        if result is None:
            return None
        dump = getattr(result, "model_dump", None)
        if callable(dump):
            try:
                return dump()
            except Exception:
                pass
        if isinstance(result, dict):
            return dict(result)
        return result

    def _post_dag_boxes(self, payload: Any) -> None:
        """Render the `build_dag` payload as an inline box-and-arrow
        graph under the `✓ Building DAG` stage line — tinted by step type
        (AI / Tool / MCP / Code) when the payload carries types. Best-
        effort: any failure (or an unrecognised payload shape) falls back
        to the terse node/edge count so the trail never breaks
        mid-stream."""
        from rich.text import Text

        from care.runtime.dag_view import dag_display_opts, render_dag_styled

        try:
            lines = render_dag_styled(
                payload,
                max_graph_width=self._dag_graph_width(),
                **dag_display_opts(getattr(self.app, "config", None)),
            )
        except Exception:  # noqa: BLE001
            _log.exception("failed to render DAG box graph")
            lines = []
        if not lines:
            from care.intermediate_artifacts import _summarise_stage

            try:
                summary = _summarise_stage("dag", payload)
            except Exception:  # noqa: BLE001
                summary = ""
            if summary:
                self._post_line("tool", f"  ⎿ {summary}")
            return
        for styled in lines:
            # Prefix the ⎿ sub-row marker, then mount the coloured Text
            # straight onto the tool Static (the plain mirror rides the
            # `text` arg for logging / collapse).
            prefixed = Text("  ⎿ ") + styled
            self._post_line(
                "tool", prefixed.plain, rich_override=prefixed,
            )

    def _post_dag_diff(self, before: Any, after: Any) -> None:
        """Render a before→after DAG diff under a `/revise` preview —
        added steps green, changed amber, removed listed in red — as
        colour-tinted `⎿` rows so the user *sees* the structural change
        before confirming the save. Best-effort: silently skips when the
        payloads don't render."""
        if not isinstance(after, dict):
            return
        from rich.text import Text

        from care.runtime.dag_view import dag_display_opts, render_dag_diff

        try:
            lines = render_dag_diff(
                before, after,
                max_graph_width=self._dag_graph_width(),
                **dag_display_opts(getattr(self.app, "config", None)),
            )
        except Exception:  # noqa: BLE001
            _log.exception("failed to render DAG diff")
            return
        if not lines:
            return
        self._post_line("tool", "  ⎿ diff (green=added, amber=changed):")
        for styled in lines:
            prefixed = Text("  ⎿ ") + styled
            self._post_line(
                "tool", prefixed.plain, rich_override=prefixed,
            )

    def _dag_graph_width(self) -> int:
        """Width budget for the inline DAG graph before it collapses to
        the compact number-box + legend variant. Tracks the live chat
        width (minus the `  ⎿ ` sub-row prefix and a little breathing
        room) so wide graphs compact on narrow terminals; falls back to
        a sane default when the size isn't known yet."""
        from care.runtime.dag_view import _DEFAULT_MAX_GRAPH_WIDTH

        try:
            width = int(self.size.width) or int(self.app.size.width)
        except Exception:  # noqa: BLE001
            return _DEFAULT_MAX_GRAPH_WIDTH
        if width <= 0:
            return _DEFAULT_MAX_GRAPH_WIDTH
        # `  ⎿ ` prefix (4) + chat gutter padding (~6).
        return max(24, width - 10)

    @staticmethod
    def _derive_chain_display_name(task: str | None) -> str:
        """First non-empty line of the task, trimmed to ≤60 chars —
        the same shape the Production save path derives, so the DAG
        modal header reads the way the library row will."""
        first_line = next(
            (
                line.strip()
                for line in (task or "").splitlines()
                if line.strip()
            ),
            t("chat.dag.untitled"),
        )
        if len(first_line) <= 60:
            return first_line
        return first_line[:57].rstrip() + "…"

    def _post_chain_actions(
        self,
        chain_dict: Any,
        *,
        display_name: str,
        chain_id: str | None = None,
        artifact_id: str | None = None,
        task: str = "",
    ) -> None:
        """Mount a `Read full` button inline under a freshly-generated
        chain's metadata rows. Clicking it opens the DAG modal (full
        graph + clickable steps in CARL format + evolve hand-off).

        The chain payload is stashed in ``_chain_action_payloads`` keyed
        by the button id so the press handler can reopen it without
        re-deriving anything. ``_last_chain_action_payload`` tracks the
        most-recent entry so the Production save path can backfill the
        saved ``chain_id`` after this button is already mounted.
        ``artifact_id`` links the row to its session-store entry so a
        save from the DAG modal can flip it to ``saved`` and refresh the
        header's unsaved pill.
        """
        if not chain_dict:
            return
        try:
            transcript = self.query_one("#chat-transcript", VerticalScroll)
        except Exception:
            return
        self._chain_action_counter += 1
        bid = f"chat-readfull-btn-{self._chain_action_counter}"
        payload = {
            "chain_dict": chain_dict,
            "display_name": display_name or "chain",
            "chain_id": chain_id,
            "artifact_id": artifact_id,
            # Original generation request — stamped as the saved chain's
            # task_description so a later re-run pre-fills the task field.
            "task": task,
        }
        self._chain_action_payloads[bid] = payload
        self._last_chain_action_payload = payload
        self._last_chain_action_button_id = bid
        # Mount the phrase + button as ONE inline row so the affordance
        # reads as part of the trace ("✓ Chain generated successfully!
        # [Read full]") rather than a button floating with no caption.
        label = Static(
            f"✓ {t('chat.trace.chainGenerated')}",
            classes="chat-readfull-label",
            markup=False,
        )
        button = Button(
            t("chat.trace.readFull"),
            id=bid,
            classes="chat-readfull-btn",
        )
        row = Horizontal(
            label,
            button,
            classes="chat-line chat-line-action chat-readfull-row",
        )
        try:
            transcript.mount(row)
            transcript.scroll_end(animate=False)
        except Exception:
            self._chain_action_payloads.pop(bid, None)

    def on_stage_progress(self, event: StageProgress) -> None:
        from care.screens.generation import _read_artifact_label

        label = _read_artifact_label(event.artifact)
        if not label:
            label = self._friendly_stage_label(event.stage)
        # `⎿ ` is a tree-branch glyph that visually nests the
        # sub-row under its parent `▶ <stage>…` line — same
        # convention Claude Code uses for tool sub-output.
        self._post_line("tool", f"  ⎿ {label}")

    def on_stage_error(self, event: StageError) -> None:
        _log.error(
            "stage %s failed: %s: %s",
            event.stage,
            type(event.error).__name__,
            event.error,
        )
        self._post_line(
            "system",
            f"✗ {event.stage} failed: "
            f"{type(event.error).__name__}: {event.error}",
            severity="error",
        )
        self._generating = False

    def on_stage_retry(self, event: StageRetry) -> None:
        self._post_line(
            "tool",
            f"  retry {event.attempt} on {event.stage}: {event.error}",
        )

    # Phase 8 P2 #21 — CSS class applied to the `▶ <stage> …`
    # line once the matching `StageCompleted` event fires so the
    # eye visually folds completed stages into a muted /
    # strike-through row.
    _STAGE_DONE_CSS_CLASS: str = "chat-line-stage-done"

    def _mark_stage_started_line_done(self, stage: str) -> None:
        """Add :data:`_STAGE_DONE_CSS_CLASS` to the widget for
        the started line that matches ``stage`` so the eye
        tracks forward progress through MAGE's discover → plan
        → assemble → validate pipeline.

        Uses :attr:`_stage_started_indexes` (populated in
        :meth:`on_stage_started`) keyed by the raw stage string,
        which keeps the lookup robust to friendly-label rendering
        changes. Falls back to a reverse text scan when the
        sidecar map is empty — old tests / replay paths that
        drive only ``StageCompleted`` events still light up.
        Idempotent and defensive: missing widget queries fail
        silently.
        """
        if not stage:
            return
        idx_one_based = self._stage_started_indexes.get(stage)
        if idx_one_based is not None:
            widget_id = f"#chat-line-{idx_one_based}"
            try:
                widget = self.query_one(widget_id)
            except Exception:
                return
            try:
                widget.add_class(self._STAGE_DONE_CSS_CLASS)
            except Exception:
                pass
            return
        # Fallback: walk back for the exact `▶ <friendly>…` or
        # legacy `▶ <stage> …` line. Exact match keeps unrelated
        # CARL step lines (`▶ step 1: parse PDF`) from getting
        # mis-marked when a stage shares a substring.
        friendly = self._friendly_stage_label(stage)
        needles = {
            f"▶ {friendly}…",
            f"▶ {stage}…",
            # Legacy format produced before the friendly-label
            # rendering landed — kept so any external driver
            # that still posts the old shape keeps working.
            f"▶ {stage} …",
        }
        for idx in range(len(self._lines) - 1, -1, -1):
            line = self._lines[idx]
            if line.role != "tool" or line.text not in needles:
                continue
            widget_id = f"#chat-line-{idx + 1}"
            try:
                widget = self.query_one(widget_id)
            except Exception:
                return
            try:
                widget.add_class(self._STAGE_DONE_CSS_CLASS)
            except Exception:
                pass
            return

    # ------------------------------------------------------------------
    # CarlStreamer event sinks → chat lines
    #
    # CARL fires one StepStarted+StepCompleted pair per chain step
    # plus a Progress message after each completion. We render
    # Started + Progress for visible motion; Completed is silent
    # because Progress already says "N of M done". ChainCompleted
    # is silent too — the final answer is rendered by the awaiting
    # caller via `_format_carl_result`.
    # ------------------------------------------------------------------

    def on_step_started(self, event: StepStarted) -> None:
        title = (event.step_title or "").strip() or "(untitled)"
        self._post_line(
            "tool",
            f"▶ {t('chat.trace.stepPrefix')} {event.step_number}: {title}",
        )

    def on_step_completed(self, event: StepCompleted) -> None:
        # Intentionally silent — Progress carries the "N done"
        # message; an extra "step done" line would double-print.
        # Phase 8 P0 #3 — reset the token-stream state so the
        # next step's `LlmChunk` events spawn a fresh preview.
        self._reset_stream_state()
        # §3 P2 — stash the step result in the artifact store
        # so the ArtifactsScreen can drill into per-step
        # outputs alongside the chain-level entry the
        # generation hook records. Best-effort throughout.
        try:
            meta = self._project_step_result(event.result)
            if meta is None:
                return
            self.artifact_store.append_tool_output(
                tool=meta["tool"],
                output=meta["output"],
                title=meta["title"],
                summary=meta["summary"],
                origin=meta["origin"],
            )
        except Exception:  # noqa: BLE001
            _log.exception(
                "failed to stash CARL step result in artifact store",
            )

    @staticmethod
    def _project_step_result(result: Any) -> dict[str, Any] | None:
        """§3 P2 — project a CARL `StepExecutionResult` into the
        kwargs the artifact store expects.

        Defensive duck-typing — CARL's result objects expose
        many attribute shapes across versions, so we probe a
        handful of likely names per field. Returns ``None``
        when nothing usable surfaces (so the caller can skip
        the append rather than store an opaque blob).
        """
        if result is None:
            return None
        step_number = (
            getattr(result, "step_number", None)
            or getattr(result, "number", None)
            or 0
        )
        step_title = (
            getattr(result, "step_title", "")
            or getattr(result, "title", "")
            or ""
        )
        step_type = (
            getattr(result, "step_type", "")
            or getattr(result, "type", "")
            or ""
        )
        tool_name = (
            getattr(result, "tool_name", "")
            or getattr(result, "tool", "")
            or ""
        )
        # CARL exposes either `.result` (final text) or
        # `.result_data` (structured) or both.
        output = (
            getattr(result, "result_data", None)
            if getattr(result, "result_data", None) is not None
            else getattr(result, "result", None)
        )
        success = bool(getattr(result, "success", True))
        # Prefer the tool name when present, else the step
        # title / type label. Cap to keep the store title
        # column readable.
        tool_label = (
            tool_name
            or step_title
            or step_type
            or f"{t('chat.trace.stepPrefix')} {step_number}"
        )
        title = (
            t("chat.trace.stepTitle", num=step_number, label=tool_label)
            if step_number
            else t("chat.trace.toolTitle", label=tool_label)
        )
        summary_text = (
            "" if output is None
            else str(output).strip().splitlines()[0][:120]
        )
        origin: dict[str, Any] = {
            "step_id": str(step_number),
            "step_number": step_number,
            "step_title": step_title,
            "step_type": step_type,
            "tool_name": tool_name,
            "success": success,
        }
        return {
            "tool": tool_label,
            "output": output,
            "title": title[:80],
            "summary": summary_text,
            "origin": origin,
        }

    def on_progress(self, event: CarlProgress) -> None:
        self._post_line(
            "tool",
            "  " + t(
                "chat.trace.stepsDone",
                done=event.completed,
                total=event.total,
                unit=self._plural(event.total, "chat.trace.step"),
            ),
        )

    def on_chain_completed(self, event: ChainCompleted) -> None:
        # The final answer is rendered by the `_run_generation`
        # awaiter via `_format_carl_result`; this handler exists
        # so future hooks (telemetry, save-result prompt) can
        # subscribe without changing the streaming surface.
        # Phase 8 P0 #3 — clear streaming state so the next
        # task's tokens start with a clean slate.
        self._reset_stream_state()
        _ = event

    # ------------------------------------------------------------------
    # Token streaming (Phase 8 P0 #3)
    # ------------------------------------------------------------------

    _STREAM_PREVIEW_MAX_CHARS: int = 80

    def on_llm_chunk(self, event: CarlLlmChunk) -> None:
        """Phase 8 P0 #3 — accumulate LLM-streamed tokens into a
        live tool line that updates in place as chunks arrive.
        Provides visible motion during long LLM calls instead
        of the user staring at a static `▶ step N: …` line
        until the step completes.

        Phase 9 P2 — also append to ``_iteration_raw_response``
        so the Ctrl+I inspector can replay the full raw stream
        for any past assistant line. Different lifecycle from
        ``_stream_buffer``: that resets per-step (tool preview
        rolls forward), this resets per-iteration (the whole
        generate→execute response).
        """
        chunk = (event.chunk or "")
        if not chunk:
            return
        # Phase 9 P2 — record every chunk for the iteration
        # capture buffer regardless of step boundaries.
        self._iteration_raw_response.append(chunk)
        if self._stream_widget_id is None:
            # First chunk of a fresh stream — post a new tool
            # line and capture its id so subsequent chunks
            # update the same widget instead of mounting more.
            self._stream_buffer = chunk
            self._post_line(
                "tool", self._format_stream_preview(chunk),
            )
            self._stream_widget_id = (
                f"chat-line-{self._line_counter}"
            )
            # Phase 9 P3 — accrue the widget id so the
            # iteration-end cleanup can remove these
            # previews before the assistant Markdown line
            # lands; the user reads the final formatted
            # answer without a duplicate truncated stream
            # sitting above it.
            self._iteration_stream_widget_ids.append(
                self._stream_widget_id,
            )
            return
        self._stream_buffer += chunk
        try:
            widget = self.query_one(
                f"#{self._stream_widget_id}", Static,
            )
            widget.update(
                self._format_stream_preview(self._stream_buffer),
            )
        except Exception:
            # Lost the widget (e.g. /clear mid-stream) — drop
            # the stream state so the next chunk spawns a
            # fresh preview.
            self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        """Clear the streaming widget reference + accumulator.
        Called from `StepCompleted` / `ChainCompleted` (next
        step starts a new preview) and from any failed
        update path."""
        self._stream_widget_id = None
        self._stream_buffer = ""

    def _remove_iteration_stream_previews(self) -> int:
        """Phase 9 P3 — sweep the streaming tool-line preview
        widgets created during this iteration off the
        transcript so the assistant Markdown line is the
        single visible carrier of the answer body. Also
        prunes the corresponding ``_lines`` entries so the
        canonical transcript matches what's on-screen.
        Returns the number of widgets actually removed (for
        confirmation logging / tests)."""
        removed = 0
        ids = list(self._iteration_stream_widget_ids)
        if not ids:
            return 0
        self._iteration_stream_widget_ids = []
        id_set = set(ids)
        for widget_id in ids:
            try:
                widget = self.query_one(f"#{widget_id}")
            except Exception:
                continue
            try:
                widget.remove()
                removed += 1
            except Exception:
                continue
        # Keep ``_lines`` consistent with the on-screen
        # transcript — drop any tool line whose widget id
        # matches one we just removed. Match by 1-based
        # index against the canonical ``chat-line-N`` scheme.
        if id_set:
            kept: list[ChatLine] = []
            for idx, line in enumerate(self._lines, start=1):
                if f"chat-line-{idx}" in id_set:
                    continue
                kept.append(line)
            self._lines = kept
        return removed

    @classmethod
    def _format_stream_preview(cls, buffer: str) -> str:
        """Project the running stream buffer into the single-
        line preview the tool widget renders. Flattens
        newlines to spaces + truncates over the budget so the
        transcript doesn't grow vertically with the stream."""
        flat = " ".join((buffer or "").split())
        if len(flat) > cls._STREAM_PREVIEW_MAX_CHARS:
            flat = flat[: cls._STREAM_PREVIEW_MAX_CHARS - 1].rstrip() + "…"
        return f"… {flat}"

    # ------------------------------------------------------------------
    # Human-input / tool-use confirms (Phase 8 P2 #16)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sub-agent activity log (Phase 8 P3)
    # ------------------------------------------------------------------

    _STEP_EVENT_BUFFER_MAX: int = 200
    _SUBAGENTS_PREVIEW_MAX_CHARS: int = 80

    def on_step_event(self, event: CarlStepEvent) -> None:
        """Capture every CARL ``StepEvent`` so `/subagents`
        can render a tree. Defensive: payload snapshots a
        shallow dict copy so later mutations from CARL don't
        retroactively alter our log; non-dict payloads coerce
        to ``{}`` for uniform downstream rendering. Bounded
        to the last :data:`_STEP_EVENT_BUFFER_MAX` entries
        so a debate with thousands of rounds doesn't drown
        memory."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        self._step_events.append(
            (event.step_number, event.event_type, dict(payload)),
        )
        if len(self._step_events) > self._STEP_EVENT_BUFFER_MAX:
            # Drop the oldest events to keep the bound honoured.
            overflow = len(self._step_events) - self._STEP_EVENT_BUFFER_MAX
            self._step_events = self._step_events[overflow:]

    def _handle_subagents_command(self, arg: str) -> None:
        """``/subagents`` body — render the captured step
        events as a tree grouped by step number. Empty buffer
        warns; `/subagents clear` flushes the buffer (useful
        between runs when the user wants a clean slate)."""
        if arg.lower() == "clear":
            count = len(self._step_events)
            self._step_events.clear()
            self._post_line(
                "system",
                f"Cleared {count} sub-agent event"
                f"{'s' if count != 1 else ''}.",
            )
            return
        if not self._step_events:
            self._post_line(
                "system",
                "No sub-agent events captured yet — run an agent "
                "chain that emits StepEvent calls (debate / "
                "parallel sampling / supervisor / tool calls) "
                "to populate this view.",
                severity="warning",
            )
            return
        rendered = self._render_subagent_tree(self._step_events)
        self._post_line("system", rendered)

    @classmethod
    def _render_subagent_tree(
        cls,
        events: list[tuple[int, str, dict[str, Any]]],
    ) -> str:
        """Project the event list into a step-grouped tree
        block. Steps render as `▶ step <N>`; each event under
        the step indents with `  · <event_type> <preview>`.
        Payload preview keeps the line scannable by collapsing
        whitespace + truncating over the budget."""
        lines = [
            f"Sub-agent activity ({len(events)} events):",
        ]
        # Preserve order but group by step number — events for
        # the same step coalesce so the eye reads top-down per
        # step rather than per-event chronology across steps.
        groups: dict[int, list[tuple[str, dict[str, Any]]]] = {}
        for step_num, ev_type, payload in events:
            groups.setdefault(step_num, []).append((ev_type, payload))
        for step_num in sorted(groups):
            entries = groups[step_num]
            lines.append(
                f"▶ {t('chat.trace.stepPrefix')} {step_num} "
                f"({len(entries)} "
                f"{ChatScreen._plural(len(entries), 'chat.trace.event')})",
            )
            for ev_type, payload in entries:
                preview = cls._render_subagent_payload_preview(payload)
                if preview:
                    lines.append(f"  · {ev_type}  {preview}")
                else:
                    lines.append(f"  · {ev_type}")
        return "\n".join(lines)

    @classmethod
    def _render_subagent_payload_preview(
        cls, payload: dict[str, Any],
    ) -> str:
        """Render a payload dict into a single scan-friendly
        line. Picks a few well-known keys (id, role, name,
        result, score) for compact preview; falls back to the
        raw key=value list when none of those match."""
        if not payload:
            return ""
        # Preferred keys land first in display order; anything
        # else falls through alphabetically.
        preferred = (
            "id", "agent_id", "round", "branch", "name",
            "role", "tool", "route", "score", "result",
        )
        pairs: list[str] = []
        used: set[str] = set()
        for key in preferred:
            if key in payload:
                pairs.append(f"{key}={cls._compact_value(payload[key])}")
                used.add(key)
        for key in sorted(payload):
            if key in used:
                continue
            pairs.append(f"{key}={cls._compact_value(payload[key])}")
        joined = ", ".join(pairs)
        if len(joined) > cls._SUBAGENTS_PREVIEW_MAX_CHARS:
            joined = joined[: cls._SUBAGENTS_PREVIEW_MAX_CHARS - 1].rstrip() + "…"
        return joined

    @staticmethod
    def _compact_value(value: Any) -> str:
        """Flatten a payload value to a compact one-line
        repr — strings stay as-is (with whitespace collapsed),
        complex values fall through to ``repr`` truncated to
        30 chars."""
        if isinstance(value, str):
            collapsed = " ".join(value.split())
            return repr(collapsed) if len(collapsed) <= 30 else repr(
                collapsed[:27].rstrip() + "…",
            )
        as_repr = repr(value)
        return as_repr if len(as_repr) <= 30 else as_repr[:27].rstrip() + "…"

    def on_human_input_requested(
        self, event: HumanInputRequested,
    ) -> None:
        """Phase 8 P2 #16 — when a CARL ``HumanInputStep``
        fires, post the prompt as a system line and stash the
        future. The next chat submission resolves it via
        :meth:`_resolve_pending_human_input`. When a second
        request arrives while one is pending, cancel the new
        one immediately with a warning (a serialised flow
        keeps the user from getting stranded between two
        unresolved prompts)."""
        if self._pending_human_input is not None:
            self._post_line(
                "system",
                "A previous human-input request is still "
                "pending — declining the new one.",
                severity="warning",
            )
            # Resolve the NEW request with a default-decline
            # so the chain doesn't deadlock. The set_result
            # call is best-effort; broken futures fall through
            # silently.
            try:
                event.future.set_result("")
            except Exception:
                pass
            return
        self._pending_human_input = event.future
        self._post_line(
            "system",
            f"[?] {event.prompt}\n"
            "(Type your answer + Enter. Empty input cancels.)",
        )

    def _resolve_pending_human_input(self, text: str) -> None:
        """Resolve the pending human-input future with ``text``.
        Empty input cancels via the future's ``set_exception``
        when supported, or falls through to ``set_result("")``
        for futures that don't expose cancellation."""
        future = self._pending_human_input
        self._pending_human_input = None
        if future is None:
            return
        stripped = text.strip()
        try:
            future.set_result(text)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "human-input future couldn't be resolved: %s",
                exc,
            )
            self._post_line(
                "system",
                f"Couldn't deliver your answer to the chain: {exc}",
                severity="error",
            )
            return
        if stripped:
            preview = (
                stripped if len(stripped) <= 60
                else stripped[:57].rstrip() + "…"
            )
            self._post_line("system", f"→ answered: {preview!r}")
        else:
            self._post_line(
                "system",
                "→ empty answer delivered (chain may treat as skip).",
            )

    # ------------------------------------------------------------------
    # Bindings
    # ------------------------------------------------------------------

    def action_clear_transcript(self) -> None:
        self._lines.clear()
        # The tracked localizable lines are gone with the transcript;
        # `_post_welcome_block` below re-registers the fresh welcome.
        self._localizable_lines.clear()
        try:
            transcript = self.query_one("#chat-transcript", VerticalScroll)
            transcript.remove_children()
        except Exception:
            return
        # Clearing the visible transcript also blows away the
        # Ad-Hoc conversation context — if the user wanted to
        # keep talking with the same memory, they'd have just
        # typed the next prompt instead.
        self._reset_interactive_history()
        self._post_line("system", t("chat.transcriptCleared"))
        # Re-show the mode-aware welcome block so a fresh `/clear`
        # behaves like a fresh boot — the user sees the current
        # contract before their next prompt.
        self._post_welcome_block()

    # Cross-cutting Cancellation — every long-running worker
    # group the chat surface spawns. Esc walks the full list
    # so users can stop a `/dataset run` / `/evolution watch`
    # / `/forget` mid-flight, not just MAGE generation.
    _CANCELLABLE_WORKER_GROUPS: tuple[str, ...] = (
        "generate",
        "dataset",
        "evolution_stream",
        "forget",
    )

    # Phase 8 P0 — "thinking…" spinner label. Sits next to the
    # animated LoadingIndicator so the visual + text together
    # tell the user CARE is working.
    # Phase 8 P1 #9 — Stop-button label is just the square glyph: a
    # centered icon-only button avoids the off-center "■ Stop" pairing
    # where the square drifted to the left edge. The tooltip is
    # localized at render time via ``t("chat.stopTooltip")``.
    _STOP_BUTTON_LABEL: str = "■"

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Phase 8 P0 — every chat-spawned worker transition
        re-evaluates spinner + Stop-button visibility. Filters
        to the cancellable groups so unrelated workers
        (StatusBar refresh, health probes) don't keep them
        alive forever."""
        worker = event.worker
        if getattr(worker, "group", None) not in self._CANCELLABLE_WORKER_GROUPS:
            return
        self._refresh_spinner()

    def _refresh_spinner(
        self,
        workers: Any = None,
    ) -> None:
        """Recompute spinner + Stop-button visibility from the
        live worker manager. Both surfaces shown iff at least
        one worker in our cancellable groups is RUNNING (i.e.
        not PENDING / SUCCESS / ERROR / CANCELLED).

        ``workers`` is injectable so unit tests can drive the
        helper without spinning a real long-lived worker. When
        omitted, reads ``self.workers`` defensively (Textual's
        ``WorkerManager`` is iterable; missing / raising falls
        back to "no workers", which hides both surfaces)."""
        if workers is None:
            try:
                workers = list(self.workers)
            except Exception:
                workers = []
        running = 0
        for worker in workers:
            group = getattr(worker, "group", None)
            state = getattr(worker, "state", None)
            if (
                group in self._CANCELLABLE_WORKER_GROUPS
                and state == WorkerState.RUNNING
            ):
                running += 1
        visible = running > 0
        self._set_spinner_visible(visible)
        self._set_stop_button_visible(visible)

    def _set_spinner_visible(self, visible: bool) -> None:
        """Flip the "thinking…" state on the combined status strip. Pure
        method so tests can drive it without spinning real workers; safe to
        call before mount (the strip repaint degrades silently)."""
        if self._thinking == visible:
            # Still repaint — a stage may have advanced under a steady
            # thinking state — but skip the timer churn.
            self._refresh_status_strip()
            return
        self._thinking = visible
        self._refresh_status_strip()

    def _set_stop_button_visible(self, visible: bool) -> None:
        """Phase 8 P1 #9 — toggle the Stop button's visibility.
        Same pattern as `_set_spinner_visible` — pure, safe
        before mount, idempotent. The button stays in the
        layout (no remount thrash); only its ``display`` flips."""
        try:
            btn = self.query_one("#chat-stop-btn", Button)
        except Exception:
            return
        if btn.display != visible:
            btn.display = visible

    def on_select_changed(self, event: "Select.Changed") -> None:
        """Chain-bar version dropdown — load + diff the picked revision."""
        from textual.widgets import Select

        try:
            select_id = event.select.id or ""
        except Exception:
            return
        if not select_id.startswith("chat-chain-version-select-"):
            return
        if event.value in (None, Select.BLANK):
            return
        session = self._chain_session
        if session is None or session.get("_suppress_version_select"):
            return
        payload = session.get("payload") or {}
        version_id = str(event.value)
        if version_id == str(payload.get("version_id") or ""):
            return
        self.run_worker(
            self._apply_chain_session_version(version_id),
            name="chat_chain_session_version",
            group="generation",
            exclusive=False,
            exit_on_error=False,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Phase 8 P1 #9 — Stop button forwards to the existing
        cancel path. ``action_interrupt`` already walks every
        long-running worker group AND posts the "Interrupted."
        line when MAGE generation was in flight, so the click
        gets identical behaviour to pressing Esc."""
        bid = event.button.id or ""
        if bid == "chat-stop-btn":
            self.action_interrupt()
            return
        if bid.startswith("chat-chain-act-"):
            self._handle_chain_bar_button(bid)
            return
        # Compact two-button confirm (reuse / pipeline stages).
        if bid.startswith("chat-confirm-yes-"):
            self._resolve_inline_confirm(True)
            return
        if bid.startswith("chat-confirm-no-"):
            self._resolve_inline_confirm(False)
            return
        # §2 P1 — Production action toolbar buttons. Each
        # forwards to the equivalent slash command handler so
        # keyboard + click paths converge on a single
        # implementation.
        if bid == "chat-prod-btn-artifacts":
            self._dispatch_prod_toolbar_artifacts()
            return
        if bid == "chat-prod-btn-dataset":
            self._dispatch_prod_toolbar_dataset()
            return
        if bid == "chat-prod-btn-evolve":
            self._dispatch_prod_toolbar_evolve()
            return
        # "Read full" — open the DAG modal for the chain this button
        # was mounted under.
        if bid.startswith("chat-readfull-btn-"):
            self._open_dag_modal(bid)
            return

    def _open_dag_modal(
        self,
        button_id: str | None = None,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Push the full-DAG modal for the chain stashed under
        ``button_id`` or passed directly as ``payload``."""
        if payload is None:
            payload = self._chain_action_payloads.get(button_id or "")
        if payload is None:
            return
        from care.screens.chain_dag import ChainDagModal

        modal = ChainDagModal(
            chain_dict=payload.get("chain_dict"),
            display_name=payload.get("display_name") or "chain",
            chain_id=payload.get("chain_id"),
        )
        # The modal can't reach Memory itself — it triggers the save
        # through this handler and we report the result back onto its
        # button via mark_saved / mark_save_failed.
        modal.save_handler = lambda: self._save_to_library(
            payload, modal=modal,
        )

        def _on_dismiss(result: str | None) -> None:
            if result == "evolve":
                self._evolve_from_dag(payload)
            elif result == "edit":
                self._seed_revise_for_step(payload, modal.edit_step_number)
            # Keep the action bar labels/focus fresh after closing the modal.
            if self._chain_session is not None:
                try:
                    row_id = self._chain_session.get("row_id") or ""
                    if row_id:
                        self.query_one(f"#{row_id}").focus()
                except Exception:
                    pass

        self.app.push_screen(modal, _on_dismiss)

    def _seed_revise_for_step(
        self, payload: dict[str, Any], step_number: int | None,
    ) -> None:
        """Seed the chat input with a `/revise` targeting one step, handed
        off from the DAG modal's *Edit step* action. Uses the saved chain
        id when known so the edit lands on the right chain; the user types
        the actual change after the seeded prefix."""
        chain_id = payload.get("chain_id")
        target = f"{chain_id} " if chain_id else ""
        step_hint = (
            f"step {step_number}: " if isinstance(step_number, int) else ""
        )
        self.seed_input(f"/revise {target}{step_hint}")

    def _evolve_from_dag(self, payload: dict[str, Any]) -> None:
        """Evolve the chain behind a DAG modal — opens the shared
        :class:`EvolutionLaunchModal` (dataset + budget + rubric) so the
        user configures the run, same as the Library / Inspection Evolve
        buttons.

        When the chain is already saved we have its id and open the setup
        modal pre-bound immediately; otherwise (unsaved Ad-Hoc chain) we
        save it first — async, so it rides a worker — then open the modal
        bound to the new id."""
        chain_dict = payload.get("chain_dict") or {}
        display_name = payload.get("display_name") or "chain"
        chain_id = payload.get("chain_id")
        if not chain_dict:
            return
        if chain_id:
            self._open_evolution_setup(chain_id)
            return
        self.run_worker(
            self._save_then_open_evolution_setup(
                chain_dict=chain_dict, display_name=display_name,
            ),
            name="chat_dag_evolve",
            group="evolution_stream",
            exclusive=False,
            exit_on_error=False,
        )

    def _save_to_library(
        self, payload: dict[str, Any], *, modal: Any = None,
    ) -> None:
        """Save the chain behind a DAG modal or chain-session bar to Memory."""
        chain_dict = payload.get("chain_dict") or {}
        display_name = payload.get("display_name") or t("chat.dag.defaultName")
        chain_id = payload.get("chain_id")
        if not chain_dict:
            return
        session = self._chain_session
        session_payload = (
            session.get("payload") if session is not None else None
        )
        is_session_bar = session_payload is payload
        if (
            chain_id
            and is_session_bar
            and session is not None
            and session.get("edit_dirty")
        ):
            self.run_worker(
                self._save_chain_session_new_version(
                    payload=payload,
                    chain_dict=chain_dict,
                    display_name=display_name,
                    modal=modal,
                ),
                name="chat_save_session_version",
                group="chat_dag_save",
                exclusive=False,
                exit_on_error=False,
            )
            return
        if chain_id:
            if modal is not None:
                modal.mark_saved()
            self._post_line(
                "assistant", t("chat.dag.alreadySaved", id=chain_id),
            )
            return
        self.run_worker(
            self._save_to_library_with_name_prompt(
                payload=payload,
                chain_dict=chain_dict,
                display_name=display_name,
                modal=modal,
            ),
            name="chat_save_name_prompt",
            group="chat_dag_save",
            exclusive=False,
            exit_on_error=False,
        )

    async def _save_chain_session_new_version(
        self,
        *,
        payload: dict[str, Any],
        chain_dict: dict[str, Any],
        display_name: str,
        modal: Any = None,
    ) -> None:
        """Persist a pending in-session edit as a new library version."""
        from care.screens.save_chain_name import SaveChainNameModal

        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._post_line(
                "system", t("chat.dag.saveMemoryNotWired"), severity="warning",
            )
            return
        chain_id = str(payload.get("chain_id") or "")
        if not chain_id:
            return
        session = self._chain_session
        instruction = ""
        parent_version_id: str | None = None
        parent_version_number: int | None = None
        if session is not None:
            instruction = str(
                session.get("last_edit_instruction")
                or session.get("task")
                or "",
            ).strip()
            parent_version_id = payload.get("version_id")
            parent_version_number = payload.get("version_number")
        save_name = await self.app.push_screen_wait(
            SaveChainNameModal(
                default_name=display_name,
                title_key="chat.revise.saveNameTitle",
                hint_key="chat.revise.saveNameHint",
                confirm_key="chat.revise.saveNameConfirm",
            )
        )
        if not save_name:
            return
        chain_to_save = dict(chain_dict)
        chain_to_save["name"] = save_name
        save_kwargs: dict[str, Any] = {
            "name": save_name,
            "query": instruction or None,
            "channel": "latest",
            "change_summary": (instruction or save_name)[:500],
            "entity_id": chain_id,
        }
        if parent_version_id:
            save_kwargs["parent_version_id"] = str(parent_version_id)
        self._post_line("tool", t("chat.dag.savingToLibrary"))
        try:
            await asyncio.to_thread(
                memory.save_chain,
                chain_to_save,
                **save_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("chain version save failed: %s", exc, exc_info=True)
            self._post_line(
                "system",
                t("chat.dag.saveFailed", error=str(exc)),
                severity="error",
            )
            if modal is not None:
                modal.mark_save_failed()
            return
        version_id, version_number = await self._fetch_latest_chain_version(
            chain_id,
        )
        if version_id:
            loaded = await self._load_chain_version_detail(
                chain_id, version_id,
            )
            if loaded and isinstance(loaded.get("content"), dict):
                chain_to_save = loaded["content"]
        if parent_version_number is not None and version_number is not None:
            self._post_line(
                "assistant",
                t(
                    "chat.revise.savedVersionFrom",
                    n=version_number,
                    name=save_name,
                    parent_n=parent_version_number,
                ),
            )
        elif version_number is not None:
            self._post_line(
                "assistant",
                t(
                    "chat.revise.savedVersion",
                    n=version_number,
                    name=save_name,
                ),
            )
        else:
            self._post_line(
                "assistant",
                t("chat.dag.savedAs", id=chain_id, name=save_name),
            )
        self._refresh_chain_session_after_version_save(
            chain_dict=chain_to_save,
            chain_id=chain_id,
            display_name=save_name,
            version_id=version_id or None,
            version_number=version_number,
            run_task=instruction or None,
        )
        self._mark_chain_artifact_saved(payload.get("artifact_id"), chain_id)
        if modal is not None:
            modal.mark_saved()

    async def _save_to_library_with_name_prompt(
        self,
        *,
        payload: dict[str, Any],
        chain_dict: dict[str, Any],
        display_name: str,
        modal: Any = None,
    ) -> None:
        """Ask for a library name, then persist the chain."""
        from care.screens.save_chain_name import SaveChainNameModal

        name = await self.app.push_screen_wait(
            SaveChainNameModal(default_name=display_name),
        )
        if not name:
            return
        payload["display_name"] = name
        await self._save_chain_to_library(
            payload=payload,
            chain_dict=chain_dict,
            display_name=name,
            modal=modal,
        )

    async def _persist_chain_to_memory(
        self,
        *,
        chain_dict: dict[str, Any],
        display_name: str,
        source_tag: str,
        progress_key: str,
        not_wired_key: str,
        query: str | None = None,
    ) -> str | None:
        """Save ``chain_dict`` to Memory under ``display_name``, posting
        the localized progress / success / failure lines. ``query`` is the
        original generation request — stamped as the chain's
        ``task_description`` so a later re-run pre-fills the task field.
        Returns the new entity id, or ``None`` when Memory isn't wired or
        the save fails (the user already saw the reason)."""
        memory = getattr(self.app, "memory", None)
        if memory is None:
            self._post_line("system", t(not_wired_key), severity="warning")
            return None
        self._post_line("tool", t(progress_key))
        try:
            save_kwargs: dict[str, Any] = {
                "name": display_name,
                "tags": [source_tag],
            }
            if query:
                save_kwargs["query"] = query
            chain_id = await asyncio.to_thread(
                memory.save_chain,
                chain_dict,
                **save_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("chain save failed: %s", exc, exc_info=True)
            self._post_line(
                "system",
                t("chat.dag.saveFailed", error=str(exc)),
                severity="error",
            )
            return None
        self._post_line(
            "assistant",
            t("chat.dag.savedAs", id=chain_id, name=display_name),
        )
        return chain_id

    async def _save_chain_to_library(
        self,
        *,
        payload: dict[str, Any],
        chain_dict: dict[str, Any],
        display_name: str,
        modal: Any = None,
    ) -> None:
        """Worker: persist an unsaved chain to the library and backfill
        the modal payload's ``chain_id`` so a later evolve reuses it
        instead of saving a second copy. Reports the outcome onto the
        modal's Save button (locked on success, re-enabled on failure)."""
        session = self._chain_session
        query = None
        if session is not None and session.get("payload") is payload:
            query = str(session.get("task") or "").strip() or None
        # Fall back to the original generation request carried on the payload
        # so the saved chain's ``query`` is stamped even when there's no live
        # session (e.g. saving straight from the DAG modal) — a later re-run
        # then pre-fills the task.
        if not query:
            query = str(payload.get("task") or "").strip() or None
        chain_id = await self._persist_chain_to_memory(
            chain_dict=chain_dict,
            display_name=display_name,
            source_tag="source:chat-dag-save",
            progress_key="chat.dag.savingToLibrary",
            not_wired_key="chat.dag.saveMemoryNotWired",
            query=query,
        )
        if chain_id is not None:
            payload["chain_id"] = chain_id
            if session is not None and session.get("payload") is payload:
                session["edit_dirty"] = False
                vid, _vnum = await self._fetch_latest_chain_version(chain_id)
                if vid:
                    payload["version_id"] = vid
                # Version label / dropdown only after a library version save.
            # Flip the matching session artifact to `saved` so the
            # header's "N unsaved" pill drops. mark_saved notifies the
            # store listener, which repaints the pill on the UI loop.
            self._mark_chain_artifact_saved(
                payload.get("artifact_id"), chain_id,
            )
            if modal is not None:
                modal.mark_saved()
        elif modal is not None:
            modal.mark_save_failed()

    def _mark_chain_artifact_saved(
        self, artifact_id: str | None, chain_id: str,
    ) -> None:
        """Flip the session-store artifact behind a saved chain to
        ``saved`` (best-effort) so the header's unsaved pill refreshes.
        No-op when the row carries no artifact id (e.g. the store append
        failed at generation time)."""
        if not artifact_id:
            return
        try:
            self.artifact_store.mark_saved(
                artifact_id, memory_entity_id=chain_id,
            )
        except Exception:  # noqa: BLE001
            _log.warning(
                "couldn't mark artifact %s saved", artifact_id,
                exc_info=True,
            )

    async def _save_then_open_evolution_setup(
        self, *, chain_dict: dict[str, Any], display_name: str,
    ) -> None:
        """Save an unsaved (Ad-Hoc) chain to Memory, then open the shared
        :class:`EvolutionLaunchModal` (dataset + budget + rubric) bound to
        the resulting entity id. Posts a friendly line and bails when
        Memory / Platform aren't wired."""
        chain_id = await self._persist_chain_to_memory(
            chain_dict=chain_dict,
            display_name=display_name,
            source_tag="source:chat-dag-evolve",
            progress_key="chat.dag.savingBeforeEvolve",
            not_wired_key="chat.dag.memoryNotWired",
        )
        if chain_id is None:
            return
        self._open_evolution_setup(chain_id)

    def _latest_chain_artifact(self) -> Any:
        """Return the most-recently-appended chain artifact, or
        ``None`` when the store has none. Used by the
        Production toolbar handlers to resolve a chain_id
        argument for the dataset / evolve dispatches."""
        chains = self.artifact_store.list_artifacts(kind="chain")
        return chains[0] if chains else None

    def _dispatch_prod_toolbar_artifacts(self) -> None:
        """`Artifacts` button → equivalent to `/artifacts`.
        The /artifacts handler is already wired below
        (`@_register("artifacts")`); we route through the same
        command dispatch path the slash keypress takes so the
        buttons stay a thin shortcut layer."""
        handler = _COMMAND_HANDLERS.get("artifacts")
        if handler is None:
            return
        try:
            handler(self, "")
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "Production toolbar `Artifacts` dispatch failed: %s",
                exc,
            )

    def _dispatch_prod_toolbar_dataset(self) -> None:
        """`Run on dataset` → `/dataset run <chain_id>` for the
        most recent saved chain. Toasts a friendly hint when
        no saved chain is in scope yet."""
        artifact = self._latest_chain_artifact()
        chain_id = (
            artifact.memory_entity_id
            if artifact is not None else ""
        ) or ""
        if not chain_id:
            self._toast_inline(
                "No saved chain yet — generate + save one "
                "before running it on a dataset.",
                severity="info",
            )
            return
        handler = _COMMAND_HANDLERS.get("dataset")
        if handler is None:
            return
        try:
            handler(self, f"run {chain_id}")
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "Production toolbar `Run on dataset` dispatch failed: %s",
                exc,
            )

    def _dispatch_prod_toolbar_evolve(self) -> None:
        """`Evolve` → `/evolution <chain_id>` for the most
        recent saved chain. Friendly toast when no saved chain
        is available."""
        artifact = self._latest_chain_artifact()
        chain_id = (
            artifact.memory_entity_id
            if artifact is not None else ""
        ) or ""
        if not chain_id:
            self._toast_inline(
                "No saved chain yet — generate + save one "
                "before evolving it.",
                severity="info",
            )
            return
        handler = _COMMAND_HANDLERS.get("evolution")
        if handler is None:
            return
        try:
            handler(self, chain_id)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "Production toolbar `Evolve` dispatch failed: %s",
                exc,
            )

    def _toast_inline(self, message: str, *, severity: str = "info") -> None:
        """Best-effort `push_toast` shim — falls back to a
        chat system line when the host doesn't expose
        `push_toast`."""
        push = getattr(self.app, "push_toast", None)
        if callable(push):
            try:
                push(message, severity=severity)
                return
            except Exception:
                pass
        self._post_line("system", message, severity=severity)

    def action_interrupt(self) -> None:
        # Phase 8 P0 #5 — Esc dismisses the autocomplete popup
        # before anything else so a user who pulled it up by
        # accident (typed `/` to start a name then changed
        # mind) doesn't also trigger a worker cancel.
        if self._autocomplete_open:
            self._set_autocomplete_visible(False)
            return
        # Phase 8 P1 #10 — Esc closes the search overlay first
        # so users can dismiss the find UI without also
        # cancelling an unrelated worker. Falls through to the
        # worker cancel path only when no overlay was open.
        if self._search_open:
            self._set_search_visible(False)
            return
        # Phase 6 P2 — Esc exits the tour cleanly before falling
        # through to the worker-cancel path.
        if self._tour_step is not None:
            self._exit_tour()
            return
        # Cross-cutting Cancellation — cancel every long-running
        # group regardless of which command spawned it. Empty
        # groups no-op so we can call all four unconditionally.
        for group_name in self._CANCELLABLE_WORKER_GROUPS:
            try:
                self.workers.cancel_group(self, group_name)
            except Exception:
                pass
        # Preserve the existing "Interrupted." line on
        # `/generation` — that's the only worker whose
        # in-flight state is gated by `_generating`. Other
        # workers post their own completion / error messages
        # via the CancelledError propagation in their own
        # `finally` blocks.
        if not self._generating:
            return
        self._post_line("system", t("chat.misc.interrupted"))
        self._generating = False

    def action_toggle_step_bodies(self) -> None:
        """Ctrl+E — flip the screen-wide collapse toggle, then
        re-render every tool ChatLine whose body fell over the
        threshold. The ChatLine list keeps the FULL text so the
        toggle is reversible without information loss.

        Walks the lines list rather than the transcript widget
        children so newly-mounted lines after a long-running
        chain get the right rendering automatically (post-time
        check in ``_format_line_for_render``).
        """
        self._step_bodies_expanded = not self._step_bodies_expanded
        affected = 0
        wmap = self._line_widget_map()
        for idx, line in enumerate(self._lines, start=1):
            if not self._should_collapse(line):
                continue
            widget = wmap.get(f"chat-line-{idx}")
            if not isinstance(widget, Static):
                continue
            try:
                # Phase 8 P1 #11 — tool lines were mounted with
                # `markup=True`, so re-render through the
                # markup-aware helper to keep inline-code spans
                # styled across toggle cycles. Other roles fall
                # through to the plain renderer.
                if line.role == "tool":
                    widget.update(self._render_tool_content(line))
                else:
                    widget.update(self._format_line_for_render(line))
                affected += 1
            except Exception:
                continue
        verb = "Expanded" if self._step_bodies_expanded else "Collapsed"
        plural = "" if affected == 1 else "s"
        self._post_line(
            "system",
            f"{verb} {affected} long step line{plural}. "
            "Press Ctrl+E again to toggle.",
        )

    def action_toggle_compact_mode(self) -> None:
        """Phase 8 P2 #20 — Ctrl+D flips compact / dense mode
        screen-wide. Walks every mounted line widget and
        re-renders it with the new prefix-stripped (compact)
        or full (default) projection. Persists the preference
        via the existing tutorial sidecar so the choice
        survives across sessions."""
        self._compact_mode = not self._compact_mode
        # Walk every line and refresh its mounted widget. We
        # branch on widget type rather than role so a future
        # tweak that re-roles a widget can't leave a stale
        # rendering behind. P-2 — resolve all line widgets in ONE DOM pass
        # (was a per-line `query_one`, i.e. O(n²) on a long transcript).
        wmap = self._line_widget_map()
        for idx, line in enumerate(self._lines, start=1):
            widget = wmap.get(f"chat-line-{idx}")
            if widget is None:
                continue
            try:
                if isinstance(widget, Markdown):
                    widget.update(
                        self._format_line_as_markdown_for_widget(line),
                    )
                elif line.role == "tool":
                    widget.update(self._render_tool_content(line))
                else:
                    widget.update(self._format_line_for_render(line))
            except Exception:
                continue
        # Persist as a sticky preference, not a one-shot
        # tutorial flag — the sidecar machinery stores the
        # bit either way, but we only mark it when the new
        # state is True so toggling back to default removes
        # the override on next read. (`_tutorial_seen` returns
        # the stored value; absence == False == default mode.)
        if self._compact_mode:
            self._mark_tutorial_seen(self._COMPACT_MODE_SIDECAR_KEY)
        else:
            self._clear_tutorial_flag(self._COMPACT_MODE_SIDECAR_KEY)
        verb = "Compact mode on" if self._compact_mode else "Compact mode off"
        self._post_line(
            "system",
            f"{verb}. Press Ctrl+D to toggle.",
        )

    # ------------------------------------------------------------------
    # Turn-focus mode (Phase 9 P1, Ctrl+0)
    # ------------------------------------------------------------------

    _TURN_HIDDEN_CLASS: str = "chat-line-turn-hidden"

    def action_focus_current_turn(self) -> None:
        """Phase 9 P1 — Ctrl+0 toggles "show only the current
        turn" focus mode. When ON, all line widgets from
        earlier turns get the ``chat-line-turn-hidden`` class
        (CSS ``display: none``) so only the current turn is
        visible. Toggling OFF restores the full transcript.
        Delivers the collapse/expand UX from the deferred
        Collapsible-cells spec without rewriting the widget
        tree."""
        if self._current_turn < 1:
            # No turns yet — nothing to focus / unfold.
            self._post_line(
                "system",
                "No turns yet — type a prompt to start one.",
                severity="warning",
            )
            return
        self._turn_focus_mode = not self._turn_focus_mode
        if self._turn_focus_mode:
            hidden = self._apply_turn_focus_classes()
            self._post_line(
                "system",
                f"Focus mode on — hid {hidden} line"
                f"{'s' if hidden != 1 else ''} from "
                f"{self._current_turn - 1} earlier turn"
                f"{'s' if self._current_turn != 2 else ''}. "
                "Press Ctrl+0 to expand all.",
            )
        else:
            restored = self._clear_turn_focus_classes()
            self._post_line(
                "system",
                f"Focus mode off — restored {restored} line"
                f"{'s' if restored != 1 else ''}. "
                "Press Ctrl+0 to refocus.",
            )

    def _apply_turn_focus_classes(self) -> int:
        """Hide widgets whose turn-N class doesn't match the
        current turn. Turn 0 (welcome / preamble) stays
        visible — it's not "an earlier turn", it's pre-turn
        scaffolding. Returns the number of widgets hidden so
        the action can render a confirmation count."""
        hidden = 0
        target = f"chat-line-turn-{self._current_turn}"
        try:
            widgets = list(self.query(".chat-line").results())
        except Exception:
            return 0
        for widget in widgets:
            classes = set(widget.classes or [])
            if target in classes:
                # Current turn — make sure it's visible (the
                # widget may have been hidden by a prior
                # focus-on cycle pointed at a different turn).
                if self._TURN_HIDDEN_CLASS in classes:
                    try:
                        widget.remove_class(self._TURN_HIDDEN_CLASS)
                    except Exception:
                        pass
                continue
            # Turn-0 scaffolding (welcome) stays visible too.
            if "chat-line-turn-0" in classes:
                if self._TURN_HIDDEN_CLASS in classes:
                    try:
                        widget.remove_class(self._TURN_HIDDEN_CLASS)
                    except Exception:
                        pass
                continue
            if self._TURN_HIDDEN_CLASS not in classes:
                try:
                    widget.add_class(self._TURN_HIDDEN_CLASS)
                    hidden += 1
                except Exception:
                    continue
            else:
                hidden += 1
        return hidden

    def _clear_turn_focus_classes(self) -> int:
        """Strip the hidden class from every line widget so the
        full transcript is visible again. Returns the count
        of widgets whose class was removed (for confirmation
        text)."""
        restored = 0
        try:
            widgets = list(self.query(".chat-line").results())
        except Exception:
            return 0
        for widget in widgets:
            classes = set(widget.classes or [])
            if self._TURN_HIDDEN_CLASS in classes:
                try:
                    widget.remove_class(self._TURN_HIDDEN_CLASS)
                    restored += 1
                except Exception:
                    continue
        return restored

    def action_recall_prev(self) -> None:
        # When the autocomplete popup is open, ↑ moves the
        # selection up rather than recalling history — the user
        # is mid-completion, not mid-edit.
        if self._autocomplete_open and self._autocomplete_matches:
            self._move_autocomplete_selection(-1)
            return
        if not self._input_history:
            return
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            return
        self._history_cursor = max(0, self._history_cursor - 1)
        inp.value = self._input_history[self._history_cursor]
        try:
            inp.cursor_position = len(inp.value)
        except Exception:
            pass

    # Phase 8 P2 #14 — quote-reply line-length budget. Long
    # assistant answers collapse to a single-line preview so
    # the single-line Input stays readable; the user appends
    # their follow-up after the preview. Once the multi-line
    # TextArea P0 lands, the full body can ride through.
    _QUOTE_MAX_LEN: int = 120

    # ------------------------------------------------------------------
    # Transcript search (Phase 8 P1 #10)
    # ------------------------------------------------------------------

    _SEARCH_MATCH_CLASS: str = "chat-line-search-match"
    _SEARCH_CURRENT_CLASS: str = "chat-line-search-current"

    def action_search(self) -> None:
        """Phase 8 P1 #10 — Ctrl+F shows the search overlay and
        focuses its input. Re-pressing while the overlay is
        already open re-focuses (and selects-all) the search
        input so the user can quickly retry a query."""
        self._set_search_visible(True)
        try:
            inp = self.query_one("#chat-search-input", Input)
            inp.focus()
            # Select-all so retyping replaces the previous
            # query rather than appending to it.
            inp.cursor_position = len(inp.value)
        except Exception:
            pass

    def _set_search_visible(self, visible: bool) -> None:
        """Toggle the search row's ``display``. When hiding,
        also clear any active highlights + reset state so a
        future Ctrl+F starts from a clean slate."""
        try:
            row = self.query_one("#chat-search-row", Horizontal)
        except Exception:
            self._search_open = visible
            return
        if row.display != visible:
            row.display = visible
        self._search_open = visible
        if not visible:
            self._clear_search_highlights()
            try:
                inp = self.query_one("#chat-search-input", Input)
                inp.value = ""
            except Exception:
                pass
            try:
                self.query_one("#chat-search-count", Static).update("")
            except Exception:
                pass
            # Return focus to the main chat input so the user
            # can resume typing immediately.
            try:
                self.query_one("#chat-input", ChatInput).focus()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Persistent history sidebar (Phase 9 P1, Ctrl+\)
    # ------------------------------------------------------------------

    _HISTORY_SIDEBAR_LIMIT: int = 10

    def action_toggle_history_sidebar(self) -> None:
        """Phase 9 P1 — Ctrl+\\ toggles the persistent left
        history sidebar. First open paints the rows; subsequent
        opens get the live snapshot because `_post_line`
        refreshes the sidebar whenever it's visible."""
        try:
            sidebar = self.query_one(
                "#chat-history-sidebar", VerticalScroll,
            )
        except Exception:
            return
        new_visible = not sidebar.display
        sidebar.display = new_visible
        self._history_sidebar_open = new_visible
        if new_visible:
            self._refresh_history_sidebar()

    def _refresh_history_sidebar(self) -> None:
        """Rebuild the sidebar from the live history helpers.
        Cheap no-op when hidden — every `_post_line` calls us
        for free, and the cost is bounded to ~20 row mounts
        when the sidebar IS open."""
        try:
            sidebar = self.query_one(
                "#chat-history-sidebar", VerticalScroll,
            )
        except Exception:
            return
        if not sidebar.display:
            return
        try:
            sidebar.remove_children()
        except Exception:
            return
        self._sidebar_actions = {}
        row_counter = 0

        def _row_id() -> str:
            nonlocal row_counter
            row_counter += 1
            return f"hist-row-{row_counter}"

        # --- Recent prompts (newest first) ---
        try:
            sidebar.mount(Static(
                "Recent prompts",
                classes="hist-section-title",
            ))
        except Exception:
            return
        user_lines = self._collect_user_lines()
        if not user_lines:
            try:
                sidebar.mount(Static(
                    "(submit a prompt)",
                    classes="hist-empty",
                ))
            except Exception:
                pass
        else:
            for turn, line in list(
                reversed(user_lines),
            )[: self._HISTORY_SIDEBAR_LIMIT]:
                full_text = line.text or ""
                first_line = (
                    full_text.splitlines()[0] if full_text else ""
                )
                preview = first_line
                if len(preview) > self._HISTORY_PREVIEW_MAX_CHARS:
                    preview = (
                        preview[: self._HISTORY_PREVIEW_MAX_CHARS - 1]
                        .rstrip() + "…"
                    )
                rid = _row_id()
                # Store the FULL (untruncated, multi-line) text
                # under the row id so the click handler can
                # restore it verbatim — the preview is for
                # display only.
                self._sidebar_actions[rid] = ("prompt", full_text)
                try:
                    sidebar.mount(Static(
                        f"{turn}. {preview}",
                        id=rid,
                        classes="hist-row",
                    ))
                except Exception:
                    continue

        # --- Saved chains (newest first) ---
        try:
            sidebar.mount(Static(
                "Saved chains",
                classes="hist-section-title",
            ))
        except Exception:
            return
        chain_lines = self._render_history_chains(
            self._HISTORY_SIDEBAR_LIMIT,
        )
        if not chain_lines:
            try:
                sidebar.mount(Static(
                    "(none saved)",
                    classes="hist-empty",
                ))
            except Exception:
                pass
            return
        for raw in chain_lines:
            stripped = raw.strip()
            chain_id = self._extract_chain_id_from_row(stripped)
            if not chain_id:
                continue
            rid = _row_id()
            self._sidebar_actions[rid] = ("chain", chain_id)
            try:
                sidebar.mount(Static(
                    stripped,
                    id=rid,
                    classes="hist-row",
                ))
            except Exception:
                continue

    @staticmethod
    def _extract_chain_id_from_row(row: str) -> str:
        r"""Pull the chain id out of a ``\`chain_id\` — name``
        row produced by :meth:`_render_history_chains`. Returns
        ``""`` when the row doesn't carry a backtick-wrapped
        id (defensive — `_render_history_chains` always emits
        the id wrapped, but a future format change shouldn't
        crash the sidebar)."""
        if "`" not in row:
            return ""
        parts = row.split("`")
        if len(parts) < 3:
            return ""
        return parts[1].strip()

    def _handle_sidebar_row_click(self, widget_id: str) -> bool:
        """Look up ``widget_id`` in :attr:`_sidebar_actions` and
        prefill the chat input. Returns ``True`` if the click
        was handled (the id matched a known row), ``False``
        otherwise — so the click event keeps bubbling for any
        other listener."""
        action = self._sidebar_actions.get(widget_id)
        if action is None:
            return False
        kind, value = action
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            return False
        if kind == "prompt":
            inp.value = value
        elif kind == "chain":
            inp.value = f"/run {value}"
        else:
            return False
        try:
            inp.cursor_position = len(inp.value)
        except Exception:
            pass
        try:
            inp.focus()
        except Exception:
            pass
        return True

    def on_click(self, event: Any) -> None:  # noqa: D401 — Textual
        """Two jobs:

        * Bridge sidebar-row clicks to the prefill helper so
          clicking a history entry drops the prompt into the
          input.
        * Re-focus the chat input on any click in the chat
          surface (transcript, mode toggle, autocomplete row,
          status bar). Without this, drag-selecting text or
          clicking a tool-line to read it steals focus and the
          user has to manually click the prompt before they
          can type — exactly the friction the user reported.

        Widgets that DO want to own focus (the search input,
        the chat-input itself, interactive form controls) get
        skipped: they're already focused as a side-effect of
        the click, and re-stealing focus would defeat the
        click's purpose.
        """
        widget = (
            getattr(event, "widget", None)
            or getattr(event, "control", None)
        )
        # First, handle any sidebar-row clicks (they prefill
        # the input AND focus it inside `_handle_sidebar_row_click`).
        wid = getattr(widget, "id", None) if widget is not None else None
        if wid and self._handle_sidebar_row_click(wid):
            return
        # §3 P0 — header artifacts pill click → /artifacts.
        # Routed via `_maybe_fire_artifact_pill_click` so the
        # dispatch stays adjacent to `_sync_artifact_pill`. The
        # helper returns True when it handled the click, letting
        # us short-circuit the refocus-chat-input fallback.
        if self._maybe_fire_artifact_pill_click(wid):
            return
        # Header Library link click → /library.
        if self._maybe_fire_library_button_click(wid):
            return
        # Header Evolution link → evolution primer.
        if self._maybe_fire_evolution_button_click(wid):
            return
        # Header Help link → Help modal (data primer + skill stub).
        if self._maybe_fire_help_button_click(wid):
            return
        # Otherwise, refocus the chat input so the next
        # keystroke lands there. The skip-set covers widgets
        # that legitimately own focus after their own click.
        if wid in self._CLICK_FOCUS_SKIP_IDS:
            return
        try:
            self.query_one("#chat-input", ChatInput).focus()
        except Exception:
            pass

    # Widget ids whose post-click focus we DON'T steal. The
    # search-input owns focus while Ctrl+F is open; the stop
    # button needs to fire its press handler; the mode
    # RadioButtons need keyboard navigation to work after
    # being clicked. Everything else (transcript lines,
    # autocomplete row, status bar, prompt glyph, etc.) hands
    # focus back to the chat input.
    _CLICK_FOCUS_SKIP_IDS: frozenset[str] = frozenset(
        {
            "chat-input",
            "chat-search-input",
            "chat-stop-btn",
            "chat-mode",
            "chat-mode-interactive",
            "chat-mode-production",
        },
    )

    def on_text_selected(self, event: events.TextSelected) -> None:
        """Auto-copy the highlighted text the moment a
        drag-select releases.

        macOS Terminal.app and iTerm2 (default config)
        intercept ``Cmd+C`` before it reaches the TUI, so
        binding ``super+c`` to a copy action is unreliable on
        the most common Mac terminals. Copying on selection
        completion sidesteps the keypress entirely — the user
        just drags, releases, and the clipboard is updated.
        ``Ctrl+C`` (and ``Cmd+C`` on terminals that DO forward
        it) still triggers the explicit ``action_copy_text``
        path for users who want a deliberate gesture.

        Feedback rides through ``App.notify`` (a transient
        toast that auto-dismisses) rather than the transcript
        so a chatty selection pattern doesn't push earlier
        replies off-screen.
        """
        from care.runtime.clipboard import copy_text

        # Claim the event so the app-level `on_text_selected` fallback
        # (which gives every other screen the same gesture) doesn't
        # re-copy and double-toast on top of this bespoke handler.
        try:
            event.stop()
        except Exception:
            pass

        try:
            selection = self.get_selected_text() or ""
        except Exception:
            return
        selection = selection.strip("\n")
        if not selection.strip():
            return
        if not copy_text(self.app, selection):
            return
        chars = len(selection)
        preview = (
            selection if chars <= 40 else selection[:37] + "…"
        )
        preview_one_line = " ".join(preview.split())
        key = "chat.copiedChars.one" if chars == 1 else "chat.copiedChars.many"
        try:
            self.app.notify(
                t(key, chars=chars, preview=preview_one_line),
                timeout=2.0,
            )
        except Exception:
            # `notify` is best-effort: scripts driving the
            # screen outside `App.run_test()` may not have a
            # notify channel; the clipboard was still written.
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        """Phase 8 P1 #10 — every keystroke in the search input
        re-runs the highlight pass. Phase 8 P0 #5 — every
        keystroke in the chat input drives the slash-autocomplete
        popup (showing when value starts with `/`, hiding
        otherwise)."""
        if event.input.id == "chat-search-input":
            # P-4 — coalesce the per-keystroke highlight pass (which scans
            # every line) so a fast typist triggers one pass per pause.
            self._debounce_search(event.value)
            return
        if event.input.id == "chat-input":
            self._refresh_autocomplete(event.value or "")
            self._refresh_input_hints(value=event.value or "")

    # P-4 — debounce window for the search-highlight pass.
    _SEARCH_DEBOUNCE_SECONDS: float = 0.15

    def _debounce_search(self, value: str) -> None:
        """P-4 — collapse a burst of search keystrokes into a single
        highlight pass fired ``_SEARCH_DEBOUNCE_SECONDS`` after the user
        stops typing. Runs synchronously (no debounce) when motion is
        disabled — reduced motion / headless tests — so the highlight is
        observable on the next ``pilot.pause()``."""
        if not self._motion_enabled():
            self._apply_search_query(value)
            return
        timer = getattr(self, "_search_debounce_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._search_debounce_timer = self.set_timer(
            self._SEARCH_DEBOUNCE_SECONDS,
            lambda: self._apply_search_query(value),
        )

    def on_resize(self, event: events.Resize) -> None:
        """Reflow bottom chrome when the terminal width changes."""
        self._refresh_bottom_chrome()

    def _content_width(self, *, pad: int = 4) -> int:
        """Usable text width inside the chat screen gutter."""
        try:
            width = int(self.size.width) or int(self.app.size.width)
        except Exception:  # noqa: BLE001
            width = 80
        if width <= 0:
            width = 80
        return max(20, width - pad)

    def _input_hint_segments(self, *, value: str = "") -> list[str]:
        """Ordered hint fragments for the strip under the prompt."""
        if self._detect_file_ref_span(value, len(value)) is not None:
            return [t("chat.inputHints.atRef")]
        keys = (
            _AGENT_INPUT_HINT_KEYS
            if self.mode == "production"
            else _CHAT_INPUT_HINT_KEYS
        )
        return [t(f"chat.inputHints.segments.{key}") for key in keys]

    def _input_hint_text(self, *, value: str = "") -> str:
        """Contextual one-line hint below the chat prompt.

        A pending CARE update (set by the background version check) takes over
        the *resting* hint line — i.e. while the input is empty. The functional
        hints (@file, /library, Esc, …) return the moment the user starts
        typing, so the update notice never gets in the way mid-prompt."""
        width = self._content_width(pad=4)
        if not value and self._care_update_latest:
            return fit_line(
                t("chat.updateAvailable", latest=self._care_update_latest), width
            )
        return fit_segments(self._input_hint_segments(value=value), width)

    def _refresh_bottom_chrome(self) -> None:
        """Keep input hints + footer within the viewport. (The former
        mode-row quick actions moved into the top-bar header, which sizes
        its own links.)"""
        self._refresh_input_hints()
        self._refresh_footer_fit()

    def _refresh_footer_fit(self) -> None:
        try:
            footer = self.query_one(CareFooter)
            footer.fit_to_width(self._content_width(pad=2))
        except Exception:
            pass

    def _refresh_input_hints(self, *, value: str | None = None) -> None:
        """Update the hint strip under the chat input row."""
        if not self.is_mounted:
            return
        if value is None:
            try:
                inp = self.query_one("#chat-input", ChatInput)
                value = inp.value or ""
            except Exception:
                value = ""
        try:
            hints = self.query_one("#chat-input-hints", Static)
        except Exception:
            return
        hints.update(self._input_hint_text(value=value))

    @staticmethod
    def _format_file_size(num_bytes: int) -> str:
        """Human-readable byte count for @-ref tool lines."""
        if num_bytes < 1024:
            return f"{num_bytes} B"
        if num_bytes < 1024 * 1024:
            return f"{num_bytes / 1024:.1f} KB"
        return f"{num_bytes / (1024 * 1024):.1f} MB"

    def _file_ref_byte_size(self, ref_path: str) -> int:
        """Best-effort on-disk size for a resolved @-ref path."""
        try:
            path = Path(ref_path).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            return path.stat().st_size if path.is_file() else 0
        except OSError:
            return 0

    def _post_file_ref_attached(self, ref_path: str) -> None:
        """Surface a successful @-attachment in the transcript."""
        size = self._format_file_size(self._file_ref_byte_size(ref_path))
        self._post_line(
            "tool",
            t("chat.fileRef.attached", path=ref_path, size=size),
        )
        self._file_ref_intro_pending = True

    def _open_data_intro(self) -> None:
        """Show the data-workflows primer modal (manual entry)."""
        from care.screens.data_intro import DataIntroModal, DataIntroResult

        def _on_dismiss(result: DataIntroResult | None) -> None:
            if result is not None and result.open_help:
                self._handle_command("/help")

        self.app.push_screen(DataIntroModal(), _on_dismiss)

    async def _await_data_intro_modal(self) -> None:
        """One-shot intro before the first @-attach or Library Run."""
        from care.screens.data_intro import DataIntroModal

        try:
            result = await self.app.push_screen_wait(DataIntroModal())
        except Exception:  # noqa: BLE001
            return
        if result is not None and result.open_help:
            self._handle_command("/help")

    async def _maybe_show_data_intro_from_file_ref(self) -> None:
        """One-shot DataIntroModal after the first successful @-attach."""
        if not self._file_ref_intro_pending:
            return
        self._file_ref_intro_pending = False
        if self._tutorial_seen("data_intro_shown"):
            return
        self._mark_tutorial_seen("data_intro_shown")
        await self._await_data_intro_modal()

    def _apply_search_query(self, query: str) -> None:
        """Walk every :class:`ChatLine` and update the
        ``chat-line-search-match`` CSS class on the matching
        mounted widget. Case-insensitive substring against
        ``ChatLine.text`` so the canonical body (not the
        rendered prefix or Markdown source) drives matching —
        the user searches for their words, not the timestamp."""
        # P-2 — one DOM pass per keystroke, shared by the clear + tag +
        # current-marker steps (was a `query_one` per line in each, i.e.
        # O(n²) every keystroke on a long transcript).
        wmap = self._line_widget_map()
        self._clear_search_highlights(wmap)
        needle = (query or "").strip().lower()
        count_widget = None
        try:
            count_widget = self.query_one(
                "#chat-search-count", Static,
            )
        except Exception:
            pass
        if not needle:
            if count_widget is not None:
                count_widget.update("")
            return
        matches: list[int] = []
        for idx, line in enumerate(self._lines, start=1):
            if needle in (line.text or "").lower():
                matches.append(idx)
                widget = wmap.get(f"chat-line-{idx}")
                if widget is not None:
                    widget.add_class(self._SEARCH_MATCH_CLASS)
        self._search_matches = matches
        self._search_cursor = 0
        if count_widget is not None:
            if matches:
                count_widget.update(f"{len(matches)} match{'es' if len(matches) != 1 else ''}")
            else:
                count_widget.update("No matches")
        # Auto-scroll to the first match so the user sees the
        # match without an explicit Enter press.
        if matches:
            self._mark_current_search_match(0, wmap)

    def _clear_search_highlights(
        self, wmap: "dict[str, Widget] | None" = None,
    ) -> None:
        """Strip both highlight classes from every previously
        matched widget. Idempotent — safe to call before /
        after every query change. ``wmap`` lets a caller share a
        single ``_line_widget_map`` pass (P-2)."""
        if wmap is None:
            wmap = self._line_widget_map()
        for idx in list(self._search_matches):
            widget = wmap.get(f"chat-line-{idx}")
            if widget is None:
                continue
            try:
                widget.remove_class(self._SEARCH_MATCH_CLASS)
                widget.remove_class(self._SEARCH_CURRENT_CLASS)
            except Exception:
                continue
        self._search_matches = []
        self._search_cursor = 0

    def _mark_current_search_match(
        self, cursor: int, wmap: "dict[str, Widget] | None" = None,
    ) -> None:
        """Move the "current match" visual marker to the
        match at ``cursor`` in ``self._search_matches`` and
        scroll it into view. Wraps around naturally because
        ``cursor`` is clamped modulo the match list size.
        ``wmap`` lets a caller share a single
        ``_line_widget_map`` pass (P-2)."""
        if not self._search_matches:
            return
        if wmap is None:
            wmap = self._line_widget_map()
        cursor = cursor % len(self._search_matches)
        self._search_cursor = cursor
        # Strip any prior current marker.
        for idx in self._search_matches:
            widget = wmap.get(f"chat-line-{idx}")
            if widget is not None:
                try:
                    widget.remove_class(self._SEARCH_CURRENT_CLASS)
                except Exception:
                    continue
        target_idx = self._search_matches[cursor]
        widget = wmap.get(f"chat-line-{target_idx}")
        if widget is not None:
            try:
                widget.add_class(self._SEARCH_CURRENT_CLASS)
                widget.scroll_visible(animate=self._motion_enabled())
            except Exception:
                pass

    def _search_cycle_next(self) -> None:
        """Enter on the search input: advance the cursor to
        the next match, wrap to start when at the end."""
        if not self._search_matches:
            return
        self._mark_current_search_match(self._search_cursor + 1)

    # ------------------------------------------------------------------
    # Slash autocomplete (Phase 8 P0 #5)
    # ------------------------------------------------------------------

    # One-line blurbs per registered slash command. Pulled into
    # the autocomplete popup so users can scan the rows without
    # needing to remember each command's purpose. Kept in lockstep
    # with `_COMMAND_HANDLERS` registrations below.
    _COMMAND_BLURBS: dict[str, str] = {
        "help": "Show this help",
        "memory": "View / edit what MAESTRO remembers (CARE.md + long-term memory)",
        "remember": "Remember a note in long-term memory (also: a `#…` message)",
        "tour": "5-step guided walkthrough",
        "mode": "Switch / show current chat mode",
        "artifacts": "Browse chains generated in this chat",
        "status": "Print agent chain generator / Memory / Platform health report",
        "library": "Open your saved chains library",
        "revise": "AI-edit a saved chain: /revise <id> <change>",
        "deploy": "Deploy a chain as an HTTP agent: /deploy <id|name> [--channel stable]",
        "deployments": "List/manage hub agents: /deployments [undeploy|reload|docs <name>]",
        "metrics": "Per-agent usage + USD cost from the hub: /metrics",
        "rollback": "Repoint a channel one version back: /rollback <id|name> [--to <vid>]",
        "versions": "Chain version history: /versions <id|name> [diff <vA> <vB>]",
        "promote": "Gated release latest→stable: /promote <id|name> [--force]",
        "marketplace": "Browse the agent_skill marketplace",
        "runs": "Open local run history (~/.cache/care/runs)",
        "sandbox": "Audit + revoke trusted AgentSkills",
        "cost": "Token + spend dashboard",
        "logs": "Open in-app log viewer",
        "profile": "List credential profiles",
        "settings": "Open settings",
        "run": "Open a saved chain for execution",
        "resume": "Rehydrate a saved session",
        "sessions": "List / rehydrate persisted artifact sessions",
        "theme": "List or switch UI theme",
        "log": "Tail the rolling app log",
        "multi": "Open a multi-line task composer",
        "edit": "Re-edit a past user prompt (1-based turn)",
        "history": "List recent prompts + saved chains",
        "blocks": "List / copy / save fenced code blocks",
        "branch": "Save / list / switch / delete transcript checkpoints",
        "imgpreview": "Detect terminal-graphics support / build image sequence",
        "subagents": "Render captured agent chain step events as a tree",
        "voice": "Voice transcription status (Whisper backend detection)",
        "export": "Save transcript to disk (md / mdx / json / html)",
        "clear": "Clear the transcript",
        "new": "Start a new Interactive conversation (drops context)",
        "quit": "Exit MAESTRO",
        "dataset": "Manage dataset entries (list / add / run / export)",
        "upload": "POST a saved chain to CARE_UPLOAD__URL",
        "forget": "Soft-delete a chain + its dataset",
        "evolution": "Evolution primer and runs dashboard",
    }

    # 16 keeps /help (position 13 of 39 commands alphabetically) visible on a
    # bare "/" — the discovery command must surface in the unfiltered popup.
    _AUTOCOMPLETE_MAX_ROWS: int = 16

    def action_slash_autocomplete(self) -> None:
        """Tab — commit the highlighted autocomplete row into the
        chat input. Handles both the slash-command popup and the
        `@`-file popup; no-op when no popup is visible (Tab then
        keeps its default focus-cycling behaviour through
        Textual's own dispatch)."""
        if not self._autocomplete_open or not self._autocomplete_matches:
            return
        idx = max(
            0,
            min(self._autocomplete_selected, len(self._autocomplete_matches) - 1),
        )
        choice = self._autocomplete_matches[idx]
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            return
        if self._autocomplete_kind == "file":
            start, end = self._autocomplete_span
            current = inp.value or ""
            new_value = (
                current[:start] + f"@{choice} " + current[end:]
            )
            inp.value = new_value
            try:
                inp.cursor_position = start + len(choice) + 2
            except Exception:
                pass
        else:
            # Default: slash command. Insert `/<name> ` with a
            # trailing space so the next keystroke types the
            # argument immediately.
            inp.value = f"/{choice} "
            try:
                inp.cursor_position = len(inp.value)
            except Exception:
                pass
        # Hide the popup — the input now carries the chosen
        # token, the user is past the disambiguation step.
        self._set_autocomplete_visible(False)

    def _set_autocomplete_visible(self, visible: bool, kind: str = "") -> None:
        """Toggle the popup row's display + reset state when
        hiding. Safe before mount (returns silently). ``kind``
        ("slash" / "file") is recorded so Tab / Up / Down route
        completion to the right path."""
        try:
            row = self.query_one("#chat-autocomplete-row", Static)
        except Exception:
            self._autocomplete_open = visible
            if not visible:
                self._autocomplete_matches = []
                self._autocomplete_kind = ""
                self._autocomplete_selected = 0
            elif kind:
                self._autocomplete_kind = kind
            return
        if row.display != visible:
            row.display = visible
        self._autocomplete_open = visible
        if not visible:
            self._autocomplete_matches = []
            self._autocomplete_kind = ""
            self._autocomplete_selected = 0
            try:
                row.update("")
            except Exception:
                pass
        elif kind:
            self._autocomplete_kind = kind

    def _move_autocomplete_selection(self, delta: int) -> None:
        """Walk the selection cursor by ``delta`` rows (clamped
        to the visible window) and re-render so the ► marker
        moves to the new row."""
        if not self._autocomplete_matches:
            return
        cap = min(len(self._autocomplete_matches), self._AUTOCOMPLETE_MAX_ROWS)
        new_idx = self._autocomplete_selected + delta
        if new_idx < 0:
            new_idx = 0
        elif new_idx >= cap:
            new_idx = cap - 1
        if new_idx == self._autocomplete_selected:
            return
        self._autocomplete_selected = new_idx
        self._render_autocomplete_rows()

    def _refresh_autocomplete(self, value: str) -> None:
        """Recompute popup state from the live chat-input value.

        Dispatches between two popup kinds:
        * ``/`` at the very start of the input → slash-command
          palette (ranked via
          :func:`care.runtime.command_palette.fuzzy_score`).
        * ``@<token>`` under the caret → file-path suggestions
          drawn from the lazy file index.
        Anything else hides the popup.
        """
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            inp = None
        cursor = (
            inp.cursor_position
            if inp is not None and inp.cursor_position is not None
            else len(value)
        )
        # @<token> takes precedence so a prompt like
        # "/refactor @care/screens/chat.py" still pops the file
        # list when the caret is over the path.
        file_span = self._detect_file_ref_span(value, cursor)
        if file_span is not None:
            start, end = file_span
            needle = value[start + 1 : end]
            self._refresh_file_autocomplete(needle, start, end)
            return
        if value.startswith("/") and " " not in value:
            needle = value[1:].lower().strip()
            ranked = self._rank_command_names(needle)
            if not ranked:
                self._set_autocomplete_visible(False)
                return
            # Reset selection whenever the candidate list shape
            # changes — keeping a stale index would point ► at
            # the wrong row.
            if (
                self._autocomplete_kind != "slash"
                or ranked != self._autocomplete_matches
            ):
                self._autocomplete_selected = 0
            self._autocomplete_matches = ranked
            self._autocomplete_span = (0, len(value))
            self._set_autocomplete_visible(True, kind="slash")
            self._render_autocomplete_rows()
            return
        self._set_autocomplete_visible(False)

    # ------------------------------------------------------------------
    # @-file autocomplete (file index + popup wiring)
    # ------------------------------------------------------------------

    # Number of paths a single `@` query can surface. Independent
    # of `_AUTOCOMPLETE_MAX_ROWS` only so the constant naming
    # stays self-documenting.
    _FILE_AUTOCOMPLETE_RANK_CAP: int = 200

    @staticmethod
    def _detect_file_ref_span(
        value: str, cursor: int,
    ) -> tuple[int, int] | None:
        """Find the `@<token>` span the caret currently sits in.

        Returns ``(start, end)`` where ``start`` points at the
        leading `@` and ``end`` is one past the last token char.
        Returns ``None`` when the caret isn't inside an
        ``@``-token. The `@` must be at the start of input or
        preceded by whitespace so an email like
        ``user@example.com`` isn't treated as a file ref —
        consistent with `_FILE_REF_RE` used at submit time.
        """
        if not value:
            return None
        # Clamp the caret to the value range — callers may pass
        # a stale `cursor_position` that points past the end of
        # the string (e.g. after the input value was rewritten).
        cursor = max(0, min(cursor, len(value)))
        # Walk backwards from the caret to find the start of the
        # current token (run of non-whitespace chars).
        start = cursor
        while start > 0 and not value[start - 1].isspace():
            start -= 1
        if start >= len(value) or value[start] != "@":
            return None
        # The char preceding `@` must be whitespace or BOL.
        if start > 0 and not value[start - 1].isspace():
            return None
        # Token end = next whitespace, or EOL.
        end = cursor
        while end < len(value) and not value[end].isspace():
            end += 1
        return (start, end)

    def _refresh_file_autocomplete(
        self, needle: str, start: int, end: int,
    ) -> None:
        """Rank the lazy file index against ``needle`` and show
        the popup. Empty needle → top-of-index slice so the user
        sees something to scroll the moment they type `@`.

        A leading ``../`` chain in ``needle`` re-roots the index
        to the corresponding parent directory so the user can
        reach files outside the project. The chain is preserved
        in each candidate so Tab-completion produces a valid
        `@../<rest>` token the downstream resolver can read.
        """
        prefix, base, inner = self._resolve_navigation_prefix(needle)
        index = self._get_file_index_for(base)
        if not index:
            self._set_autocomplete_visible(False)
            return
        ranked_inner = self._rank_file_paths(inner, index)
        if not ranked_inner:
            self._set_autocomplete_visible(False)
            return
        ranked = (
            [f"{prefix}{entry}" for entry in ranked_inner]
            if prefix
            else ranked_inner
        )
        if (
            self._autocomplete_kind != "file"
            or ranked != self._autocomplete_matches
        ):
            self._autocomplete_selected = 0
        self._autocomplete_matches = ranked
        self._autocomplete_span = (start, end)
        self._set_autocomplete_visible(True, kind="file")
        self._render_autocomplete_rows()

    def _resolve_navigation_prefix(
        self, needle: str,
    ) -> tuple[str, Path, str]:
        """Peel leading ``../`` segments off ``needle``.

        Returns ``(display_prefix, base_path, inner_needle)``:
        * ``display_prefix`` is the consumed `../` chain — empty
          string for everyday cwd-relative input.
        * ``base_path`` is the directory the file index should
          be rooted at. Defaults to ``Path.cwd()``.
        * ``inner_needle`` is the substring left over after the
          chain is consumed; this drives ranking.
        Stops climbing at the filesystem root — extra ``..``
        segments past `/` get folded back into ``inner_needle``
        so the user sees a (likely empty) match list rather
        than the helper silently swallowing them.
        """
        prefix = ""
        base = Path.cwd()
        remaining = needle
        while remaining.startswith("../"):
            parent = base.parent
            if parent == base:
                break
            base = parent
            prefix += "../"
            remaining = remaining[3:]
        return prefix, base, remaining

    def _render_autocomplete_rows(self) -> None:
        """Paint the popup. Shared by both kinds — only the
        row prefix (``/`` vs ``@``) and the trailing blurb
        differ."""
        try:
            row = self.query_one("#chat-autocomplete-row", Static)
        except Exception:
            return
        rows: list[str] = []
        visible = self._autocomplete_matches[: self._AUTOCOMPLETE_MAX_ROWS]
        for idx, name in enumerate(visible):
            marker = "►" if idx == self._autocomplete_selected else " "
            if self._autocomplete_kind == "file":
                rows.append(f"{marker} @{name}")
            else:
                blurb = self._command_blurb(name)
                rows.append(f"{marker} /{name}  {blurb}".rstrip())
        try:
            row.update("\n".join(rows))
        except Exception:
            pass

    def _get_file_index_for(self, root: Path) -> list[str]:
        """Return (and lazily build) the cached file index for
        ``root``. First lookup at a given root pays the scan
        cost; later lookups hit the cache. Cwd lookups go
        through the dedicated ``_file_index`` slot so existing
        tests / pilots keep their hook point."""
        try:
            cwd = Path.cwd()
        except Exception:
            cwd = root
        if root == cwd:
            if self._file_index is not None:
                return self._file_index
            self._file_index = self._build_file_index(root)
            return self._file_index
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        cached = self._file_index_roots.get(key)
        if cached is not None:
            return cached
        index = self._build_file_index(root)
        self._file_index_roots[key] = index
        return index

    # Cap on entries that ride in the index so a giant repo
    # doesn't blow memory or stall the popup render. Hit this
    # ceiling and the deeper paths get dropped (the user can
    # still ref them by typing the full path).
    _FILE_INDEX_MAX_ENTRIES: int = 5000

    # Directories we never descend into when walking the project
    # — keeps the index focused on source code rather than
    # vendored / cache content.
    _FILE_INDEX_SKIP_DIRS: frozenset[str] = frozenset(
        {
            ".git",
            ".hg",
            ".svn",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".tox",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "dist",
            "build",
            "target",
            ".idea",
            ".vscode",
            ".next",
            ".cache",
        },
    )

    def _build_file_index(self, root: Path | None = None) -> list[str]:
        """Walk ``root`` (default: cwd) and collect relative
        file paths. Uses ``git ls-files`` when inside a git
        repo so .gitignore'd content is excluded for free;
        falls back to a manual walk that skips hidden dirs and
        the common cache / vendor directories."""
        import subprocess

        if root is None:
            root = Path.cwd()
        entries: list[str] = []
        try:
            completed = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=str(root),
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in completed.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                entries.append(line)
                if len(entries) >= self._FILE_INDEX_MAX_ENTRIES:
                    break
            if entries:
                return entries
        except Exception:
            pass
        # Manual fallback: walk the tree, skip hidden +
        # high-volume dirs.
        for current, dirs, files in __import__("os").walk(str(root)):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d not in self._FILE_INDEX_SKIP_DIRS
            ]
            current_path = Path(current)
            for name in files:
                if name.startswith("."):
                    continue
                rel = (current_path / name).relative_to(root)
                entries.append(str(rel))
                if len(entries) >= self._FILE_INDEX_MAX_ENTRIES:
                    return entries
        return entries

    @classmethod
    def _rank_file_paths(
        cls, needle: str, index: list[str],
    ) -> list[str]:
        """Rank ``index`` paths against ``needle``.

        Scoring (higher first):
        * exact basename match — 1000
        * basename prefix — 500
        * basename contains — 300
        * path-anywhere contains — 100
        * subsequence match — `fuzzy_score`-derived signal
        Ties break on shorter path then alphabetical so the
        nearest / least-nested match floats up.
        """
        if not needle:
            # Empty query → return a stable prefix of the index
            # so the popup has something to show right after the
            # user types `@`.
            return sorted(index)[: cls._AUTOCOMPLETE_MAX_ROWS]
        from care.runtime.command_palette import fuzzy_score

        needle_lc = needle.lower()
        scored: list[tuple[float, int, str]] = []
        for path in index:
            base = path.rsplit("/", 1)[-1]
            base_lc = base.lower()
            path_lc = path.lower()
            score = 0.0
            if base_lc == needle_lc:
                score = 1000.0
            elif base_lc.startswith(needle_lc):
                score = 500.0
            elif needle_lc in base_lc:
                score = 300.0
            elif needle_lc in path_lc:
                score = 100.0
            else:
                fuzzy = fuzzy_score(needle_lc, path_lc)
                if fuzzy <= 0:
                    continue
                score = fuzzy
            scored.append((-score, len(path), path))
        scored.sort()
        return [path for _, _, path in scored[: cls._AUTOCOMPLETE_MAX_ROWS]]

    @classmethod
    def _rank_command_names(cls, needle: str) -> list[str]:
        """Return registered command names ranked by relevance
        to ``needle``. Empty needle → alphabetical full list
        (the user just typed `/`). Non-empty needle → matches
        ranked by `fuzzy_score`, descending, then alphabetical
        within ties. Names that don't match at all are
        filtered out."""
        from care.runtime.command_palette import fuzzy_score

        names = sorted(_COMMAND_HANDLERS.keys())
        if not needle:
            return names
        scored: list[tuple[float, str]] = []
        for name in names:
            score = fuzzy_score(needle, name)
            if score > 0:
                scored.append((score, name))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [name for _, name in scored]

    def action_quote_last_reply(self) -> None:
        """``Ctrl+Q`` — prefix the chat input with a `> body`
        block quoting the most recent assistant reply. Falls
        back to the last user / tool line when no assistant
        reply exists yet (so power users can quote their own
        last prompt). Bodies that exceed
        :data:`_QUOTE_MAX_LEN` chars collapse to a single-line
        preview with ellipsis so the prompt row stays readable.
        Multi-line bodies fold into one line via space-join so
        the single-line Input doesn't truncate awkwardly.
        Cursor lands at the end of the input so the next
        keystroke continues the prompt naturally.
        """
        target = self._pick_quote_target()
        if target is None:
            self._post_line(
                "system",
                t("chat.clipboard.nothingToQuote"),
                severity="warning",
            )
            return
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            return
        quote = self._build_quote_prefix(target.text)
        inp.value = quote + (inp.value or "")
        try:
            inp.cursor_position = len(inp.value)
        except Exception:
            pass
        try:
            inp.focus()
        except Exception:
            pass

    def _pick_quote_target(self) -> ChatLine | None:
        """Find the line `Ctrl+Q` should quote. Prefers the
        most recent ``assistant`` reply (the typical "quote
        the answer" gesture); falls back to the last user /
        tool line so the affordance still works in fresh
        sessions where no assistant has spoken yet."""
        for line in reversed(self._lines):
            if line.role == "assistant":
                return line
        for line in reversed(self._lines):
            if line.role in ("user", "tool"):
                return line
        return None

    @classmethod
    def _build_quote_prefix(cls, body: str) -> str:
        """Project a multi-line body into a single-line
        `> preview ` prefix safe to prepend to the chat input.
        Collapses internal whitespace + truncates over the
        :data:`_QUOTE_MAX_LEN` budget with an ellipsis."""
        squashed = " ".join(body.split())
        if len(squashed) > cls._QUOTE_MAX_LEN:
            squashed = squashed[: cls._QUOTE_MAX_LEN].rstrip() + "…"
        return f"> {squashed} "

    def action_copy_last_reply(self) -> None:
        """`Ctrl+Y` — copy the most recent ``assistant`` line
        to the system clipboard. Falls back to the last
        non-system line so command output (e.g. /help) is
        copyable too."""
        from care.runtime.clipboard import copy_text

        for line in reversed(self._lines):
            if line.role == "assistant":
                target = line.text
                break
        else:
            non_system = [
                line for line in self._lines if line.role != "system"
            ]
            if not non_system:
                self._post_line("system", t("chat.clipboard.nothingToCopy"))
                return
            target = non_system[-1].text
        if copy_text(self.app, target):
            preview = target if len(target) <= 40 else target[:37] + "…"
            self._post_line("system", f"Copied: {preview!r}")
        else:
            self._post_line(
                "system",
                t("chat.clipboard.copyFailed"),
                severity="warning",
            )

    def action_copy_transcript(self) -> None:
        """`Ctrl+Shift+Y` — copy the entire transcript as plain
        text. Useful for pasting a session into a bug report."""
        from care.runtime.clipboard import copy_text

        if not self._lines:
            self._post_line("system", t("chat.clipboard.transcriptEmpty"))
            return
        rendered = "\n".join(self._format_line(line) for line in self._lines)
        if copy_text(self.app, rendered):
            self._post_line(
                "system",
                f"Copied {len(self._lines)} lines to clipboard.",
            )
        else:
            self._post_line(
                "system",
                t("chat.clipboard.copyFailed"),
                severity="warning",
            )

    def action_copy_text(self) -> None:
        """`Ctrl+C` / `Cmd+C` — copy any text currently selected
        in the chat surface to the system clipboard.

        Overrides :meth:`Screen.action_copy_text` so the value
        rides through :func:`care.runtime.clipboard.copy_text`
        (which combines OSC 52 with the platform-native helper
        — ``pbcopy`` on macOS, ``xclip`` / ``xsel`` / ``wl-copy``
        on Linux, ``clip`` on Windows). The base class only
        emits OSC 52, which macOS Terminal.app disables.

        Empty selection surfaces a one-line warning rather than
        bubbling to the app-level quit confirmation so an
        accidental Ctrl+C doesn't surprise the user with a
        modal dialog.
        """
        from care.runtime.clipboard import copy_text

        selection = self.get_selected_text() or ""
        if not selection.strip():
            self._post_line(
                "system",
                "Nothing selected — drag to highlight text first, "
                "or use Ctrl+Y to copy the last reply.",
                severity="warning",
            )
            return
        if not copy_text(self.app, selection):
            self._post_line(
                "system",
                t("chat.clipboard.copyFailed"),
                severity="warning",
            )
            return
        chars = len(selection)
        preview = (
            selection if chars <= 40 else selection[:37] + "…"
        )
        # Collapse internal newlines so the system-line preview
        # stays on a single row.
        preview_one_line = " ".join(preview.split())
        plural = "s" if chars != 1 else ""
        self._post_line(
            "system",
            f"Copied {chars} char{plural}: {preview_one_line!r}",
        )

    def action_recall_next(self) -> None:
        # Mirror of `action_recall_prev`: when the popup is up,
        # ↓ walks the suggestion list instead of history.
        if self._autocomplete_open and self._autocomplete_matches:
            self._move_autocomplete_selection(1)
            return
        if not self._input_history:
            return
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            return
        if self._history_cursor < len(self._input_history) - 1:
            self._history_cursor += 1
            inp.value = self._input_history[self._history_cursor]
        else:
            self._history_cursor = len(self._input_history)
            inp.value = ""
        try:
            inp.cursor_position = len(inp.value)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Transcript rendering
    # ------------------------------------------------------------------

    def _post_line(
        self,
        role: ChatRole,
        text: str,
        *,
        severity: str | None = None,
        provenance: dict[str, Any] | None = None,
        chrome: bool = False,
        extra_class: str | None = None,
        linkify_commands: bool = False,
        rich_override: Any = None,
    ) -> None:
        # P-3 — `_post_line` mounts widgets, which Textual only permits on the
        # main thread. File-ref resolution runs the blocking read in
        # `asyncio.to_thread`, and its attach/warning lines flow through here,
        # so marshal back to the event loop when we're off it. `call_from_thread`
        # runs the call on the loop and blocks only the worker thread — the loop
        # is free (awaiting the to_thread future), so there's no deadlock. The
        # guard is inert on the main thread (every existing caller), so it can't
        # affect the synchronous hot path.
        if threading.current_thread() is not threading.main_thread():
            try:
                self.app.call_from_thread(
                    self._post_line,
                    role,
                    text,
                    severity=severity,
                    provenance=provenance,
                    chrome=chrome,
                    extra_class=extra_class,
                    linkify_commands=linkify_commands,
                    rich_override=rich_override,
                )
            except Exception:
                pass
            return
        line = ChatLine(
            role=role,
            text=text,
            mode=self.mode,
            provenance=provenance,
            chrome=chrome,
            linkify_commands=linkify_commands,
        )
        self._lines.append(line)
        self._line_counter += 1
        # Mirror every chat entry into the app log so the
        # full session transcript survives in
        # `logs/care-app-<ts>.log`. Default level by role:
        # tool/stage chatter rides DEBUG (high volume);
        # user / assistant / system lines ride INFO. Error
        # paths pass `severity="warning"` / `"error"` so the
        # mirror is grep-friendly alongside the raw warning
        # log already emitted at the call site.
        if severity is not None:
            log_level = _SEVERITY_TO_LEVEL.get(
                severity.lower(), logging.INFO,
            )
        elif role == "tool":
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        _log.log(log_level, "chat [%s] %s", role, text)
        # Phase 6 P2 — Production-mode session log. Off by
        # default in Ad-Hoc so casual users never touch disk;
        # in Production, user/assistant/system lines stream to
        # `~/.local/state/care/sessions/care-session-<ts>.md`
        # so Maria can revisit "what did I save / why" later.
        # `tool` chatter (CARL step progress, MagePoster, etc.)
        # is high-volume and stays in the live transcript only.
        if self.mode == "production" and role != "tool":
            self._append_to_session_log(line)
        line_id = f"chat-line-{self._line_counter}"
        try:
            transcript = self.query_one("#chat-transcript", VerticalScroll)
        except Exception:
            return
        # Phase 8 P0 #2 — assistant + system lines mount as
        # `Markdown` widgets so fenced code blocks, headings,
        # lists, and inline backticks all render properly.
        # `tool` chatter is high-volume + the existing collapse
        # logic walks Static widgets, so tool stays Static.
        # `user` lines render verbatim (their formatting was
        # the user's intent — never reflow it).
        # Phase 8 P1 #7 — user lines past the first get a turn
        # boundary class so a top border + extra padding
        # visually groups each user→assistant exchange. We
        # check `_lines` BEFORE the current line was appended
        # to count prior user lines (the new line is already
        # in `_lines` at this point, hence `len(...) - 1`).
        css_classes = f"chat-line chat-line-{role}"
        if extra_class:
            # Per-call CSS hook for cosmetic tweaks that
            # don't fit a role / chrome / turn pattern (e.g.
            # the synthesis-done line wears
            # `chat-line-pre-answer` so it carries a single
            # row of bottom padding before the assistant
            # answer that follows).
            css_classes += f" {extra_class}"
        if line.chrome:
            # Boot banner + future status panels — pure
            # sign-posting, styled independently of the
            # role-specific colour rules. The class lets the
            # CSS pin a white foreground and add a little
            # breathing room between rows.
            css_classes += " chat-line-chrome"
        if role == "user":
            prior_user_count = sum(
                1 for prior in self._lines[:-1]
                if prior.role == "user"
            )
            # The boundary divider only fires when the user
            # explicitly opened a fresh conversation via
            # `/new` — it marks the seam between
            # conversations, not between consecutive prompts
            # in the same session. The flag is cleared so the
            # divider lands on exactly one user line.
            if (
                self._new_conversation_pending
                and prior_user_count > 0
            ):
                css_classes += (
                    f" {self._USER_TURN_BOUNDARY_CLASS}"
                )
            self._new_conversation_pending = False
            # Phase 9 P1 — a new user line opens a new turn.
            # Bump the counter BEFORE adding the turn-N class
            # so the user-line carries the new turn's tag.
            self._current_turn += 1
            # If focus mode is on, the user has explicitly
            # asked "show me only the current turn" — hide
            # widgets from the prior turn so the new turn is
            # the only thing visible.
            if self._turn_focus_mode and prior_user_count > 0:
                self._apply_turn_focus_classes()
        # Tag every widget with the turn it belongs to so the
        # focus-mode toggle can flip visibility without
        # remounting. Turn 0 is the welcome/preamble (no user
        # line yet) and stays visible across focus toggles.
        css_classes += f" chat-line-turn-{self._current_turn}"
        if role in self._MARKDOWN_ROLES:
            widget: Static | Markdown = Markdown(
                self._format_line_as_markdown_for_widget(line),
                id=line_id,
                classes=css_classes,
                # Command links must NOT open in a browser — the screen
                # handler intercepts them. Real links on other lines keep
                # the default open-in-browser behaviour.
                open_links=not linkify_commands,
            )
        elif role == "tool":
            # Phase 8 P1 #11 — tool lines get the lightweight
            # inline-code styling so ids like `chain-X` read
            # visually distinct from prose. The escape pass on
            # source brackets means user content can't spoof
            # markup tags; the Static is built with
            # `markup=True` so the substituted [reverse]…[/]
            # spans render.
            #
            # `rich_override` lets a caller hand a pre-styled Rich
            # renderable straight to the Static (markup off, since
            # it carries its own spans) — used for the colour-tinted
            # DAG trail, whose box glyphs can't survive the markup
            # escape pass as a plain string. `line.text` still holds
            # the plain mirror for logging / collapse / transcript.
            widget = Static(
                rich_override if rich_override is not None
                else self._render_tool_content(line),
                id=line_id,
                classes=css_classes,
                markup=rich_override is None,
            )
        else:
            widget = Static(
                self._format_line_for_render(line),
                id=line_id,
                classes=css_classes,
            )
        try:
            transcript.mount(widget)
            # A-1 — fade the new line in (no-op under reduced motion / tests).
            self._animate_line_in(widget)
            # P-1 — bound the mounted widget count so long sessions don't
            # accumulate an unbounded, increasingly slow DOM.
            self._prune_transcript(transcript)
            # A-6 — smooth-follow scroll for low-frequency lines; keep the
            # high-volume `tool` stream (and reduced-motion / tests) instant
            # so a fast CARL run doesn't lag behind a queue of scroll tweens.
            transcript.scroll_end(
                animate=self._motion_enabled() and role != "tool",
            )
        except Exception:
            pass
        # Phase 9 P1 — keep the history sidebar in sync with the
        # live transcript. The refresh is a no-op when the
        # sidebar is hidden so the hot path stays cheap (zero
        # widget mutation when the user hasn't opened it).
        try:
            self._refresh_history_sidebar()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Motion + transcript bounds (TODO §Animations A-1/A-6, §Perf P-1)
    # ------------------------------------------------------------------

    #: P-1 — hard cap on mounted transcript widgets. The full `self._lines`
    #: model is unbounded (search / export / relocalize read it); only the
    #: oldest *mounted* widgets past this many get unmounted. Sized well above
    #: any realistic single screenful so only genuinely-scrolled-away history
    #: is dropped. Tests never post this many lines, so they're unaffected.
    _MAX_RENDERED_LINES: int = 500

    def _motion_enabled(self) -> bool:
        """True when UI animations should play. False under reduced motion
        and headless tests — both pin the app animation level to ``"none"``,
        which is also what makes every ``styles.animate`` / CSS ``transition``
        resolve instantly to its final value."""
        try:
            return getattr(self.app, "animation_level", "none") != "none"
        except Exception:
            return False

    def _animate_line_in(self, widget: "Widget") -> None:
        """A-1 — fade a freshly-mounted chat line in. Deliberately a no-op
        (the line stays at its default full opacity) when motion is disabled,
        so a headless pilot never observes a half-tweened ``opacity`` and a
        reduced-motion session never pays for the animation."""
        if not self._motion_enabled():
            return
        try:
            widget.styles.opacity = 0.0
            widget.styles.animate(
                "opacity", value=1.0, duration=0.18, easing="out_cubic",
            )
        except Exception:
            pass

    def _prune_transcript(self, transcript: "VerticalScroll") -> None:
        """P-1 — keep the number of mounted transcript widgets bounded.

        Removes the oldest mounted widgets once the count exceeds
        :attr:`_MAX_RENDERED_LINES`. Operates on the live child list (not on
        ``self._lines``) so it's robust to widgets mounted outside
        ``_post_line`` (chain-action bars, confirm rows, the boot banner).
        The dropped widgets are far above the viewport; the canonical
        transcript still lives in ``self._lines`` and the app log."""
        cap = self._MAX_RENDERED_LINES
        if cap <= 0:
            return
        try:
            children = list(transcript.children)
        except Exception:
            return
        overflow = len(children) - cap
        if overflow <= 0:
            return
        for widget in children[:overflow]:
            try:
                widget.remove()
            except Exception:
                pass

    def _line_widget_map(self) -> "dict[str, Widget]":
        """P-2 — map ``chat-line-N`` id → mounted widget in a SINGLE DOM
        pass. Callers that touch every line (compact toggle, step-body
        toggle, the per-keystroke search highlight) previously did a
        ``query_one("#chat-line-{idx}")`` per line — each an O(n) tree walk,
        so the whole loop was O(n²). Building the lookup once turns those
        into O(n)."""
        out: "dict[str, Widget]" = {}
        try:
            for widget in self.query(".chat-line"):
                if widget.id:
                    out[widget.id] = widget
        except Exception:
            pass
        return out

    def post_settings_updated(self, changes: list[str] | None = None) -> None:
        """Leave a breadcrumb in the chat trace after the user saves
        the SettingsScreen, so the `/settings` round-trip is visible
        instead of silently swapping config under the user.

        The app calls this once it has reloaded config and popped back
        to chat. ``changes`` is the masked field-level diff from
        :func:`care.config.summarize_config_changes` — each row mounts
        as a ``⎿`` tool sub-row under the headline, mirroring the
        stage-trail convention. An empty / ``None`` diff still confirms
        the save (the user pressed Save) but says so plainly.
        """
        if changes:
            self._post_line("system", t("chat.settingsUpdated"))
            for row in changes:
                self._post_line("tool", f"  ⎿ {row}")
        else:
            self._post_line("system", t("chat.settingsSaved"))

    # Phase 1 P2 — Production-mode "for keeps" marker. Only
    # rides on USER lines so the audit trail surfaces the
    # messages that produced saved chains; assistant /
    # system / tool lines stay unprefixed to keep the
    # visual diff between user-attributable and
    # system-generated content unambiguous.
    _PRODUCTION_USER_MARKER: str = "🛡 "

    # Phase 8 P0 #2 — roles whose body gets rendered through the
    # Textual ``Markdown`` widget (fenced code blocks, headings,
    # lists, inline backticks). ``user`` stays Static so the
    # user's verbatim prompt is never silently reformatted;
    # ``tool`` stays Static so high-volume CARL / MagePoster
    # chatter doesn't pay the Markdown rebuild cost on every
    # step event (also keeps the existing collapse logic
    # working — it walks Static widgets via ``widget.update``).
    _MARKDOWN_ROLES: frozenset[str] = frozenset({"assistant", "system"})

    # Phase 8 P1 #7 — CSS class applied to each USER line after
    # the first so a top border + padding visually groups
    # user→assistant turns. The first user line stays unmarked
    # so the welcome block flows naturally into the first
    # exchange.
    _USER_TURN_BOUNDARY_CLASS: str = "chat-line-user-turn-boundary"

    # Phase 8 P2 #20 — sidecar key for the compact-mode bit.
    # Reuses the existing tutorial sidecar machinery rather
    # than introducing a parallel preference file.
    _COMPACT_MODE_SIDECAR_KEY: str = "compact_mode_enabled"

    # Phase 2 P2 — collapsed step body thresholds. A tool line
    # exceeding EITHER bound folds to a one-liner with a
    # "+N more lines, Ctrl+E to expand" hint until the user
    # presses the toggle. Overridable via env so power users
    # can dial them up/down without forking the source.
    _COLLAPSE_MAX_LINES: int = 3
    _COLLAPSE_MAX_CHARS: int = 240
    _COLLAPSE_EXPAND_HINT: str = "Ctrl+E to expand"

    def _emit_telemetry(
        self,
        kind: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Forward one event to the host's
        :class:`TelemetrySink` if wired.

        Never raises into the chat code path — telemetry is a
        best-effort observability hook, NOT a load-bearing
        contract. Missing sink, broken backend, malformed event
        all degrade silently.

        Reads ``app.telemetry_sink`` so tests can inject a
        capturing stub by setting the attribute on the host.
        Production-wired callers attach a sink built via
        :func:`care.runtime.build_telemetry_sink` on the
        :class:`CareApp` directly.
        """
        sink = getattr(self.app, "telemetry_sink", None)
        if sink is None:
            return
        try:
            from care.runtime.telemetry import TelemetryEvent

            sink.record(TelemetryEvent(
                kind=kind,
                attributes=dict(attributes or {}),
            ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("telemetry emit failed (%s): %s", kind, exc)

    def _refresh_status_bar(self) -> None:
        """Kick the mounted :class:`StatusBar` so the new
        cumulative token / Memory / Platform snapshot lands
        immediately after a generation. Best-effort — the
        widget may not be present in unit tests (composed-less
        hosts), so we degrade silently."""
        try:
            bar = self.query_one(StatusBar)
        except Exception:
            return
        try:
            bar.refresh_snapshot()
        except Exception:
            pass

    @staticmethod
    def _format_line(line: ChatLine) -> str:
        # Chrome lines (e.g. the boot banner) skip the caption
        # entirely — they're sign-posts, not chat messages, so
        # the timestamp + role marker would just be noise.
        if line.chrome:
            return line.text
        # User AND assistant lines drop the role prefix
        # entirely — the message body already speaks for the
        # role, and `[12:34] care` before every answer is
        # visual noise. System (•) and tool (space) keep their
        # markers so the eye can still split turn boundaries
        # from running chatter.
        prefix = {
            "user": "",
            "assistant": "",
            "system": "•",
            "tool": " ",
        }.get(line.role, line.role)
        ts = line.timestamp.strftime("%H:%M")
        body = line.text
        if line.role == "user" and line.mode == "production":
            body = f"{ChatScreen._PRODUCTION_USER_MARKER}{body}"
        # Phase 8 P2 #15 — reaction marker rides on assistant
        # lines only. Sits between the role prefix and the
        # body so it stays visible in compact mode too.
        marker = ChatScreen._reaction_marker(line)
        if marker:
            body = f"{marker} {body}"
        if not prefix:
            return f"[{ts}] {body}"
        return f"[{ts}] {prefix}  {body}"

    @staticmethod
    def _format_line_compact(line: ChatLine) -> str:
        """Phase 8 P2 #20 — compact-mode line projection. Drops
        the ``[HH:MM] role`` prefix and just emits the body so
        power users get 2–3× more content per screen. The
        Production-user marker stays on so the audit trail
        survives even in dense mode."""
        body = line.text
        if line.role == "user" and line.mode == "production":
            body = f"{ChatScreen._PRODUCTION_USER_MARKER}{body}"
        marker = ChatScreen._reaction_marker(line)
        if marker:
            body = f"{marker} {body}"
        return body

    @staticmethod
    def _reaction_marker(line: ChatLine) -> str:
        """Phase 8 P2 #15 — return the emoji marker for the
        line's reaction, or empty string when unset. Only
        applies to assistant lines so a stray reaction on a
        user / system line (shouldn't happen, but defensive)
        doesn't surface."""
        if line.role != "assistant" or line.reaction is None:
            return ""
        return ChatScreen._REACTION_MARKERS.get(line.reaction, "")

    @staticmethod
    def _format_line_as_markdown(line: ChatLine) -> str:
        """Phase 8 P0 #2 — project a chat line into the body the
        Textual ``Markdown`` widget renders.

        Layout: bold caption (``**[HH:MM] role**``) followed by
        the body. When the body's first non-empty line is inline
        prose, the caption joins it in the same paragraph so the
        rendered output reads ``[HH:MM] care The summary…`` —
        no blank gap between the caption and the answer. When
        the body opens with a Markdown block element (heading,
        fenced code, list, blockquote, table, thematic break),
        a ``\\n\\n`` separator goes in instead so the block
        renders as a block instead of being swallowed into the
        caption paragraph.

        Clipboard (``Ctrl+Y`` / ``Ctrl+Shift+Y``) continues to
        read ``ChatLine.text`` via ``_format_line`` — the caption
        markup never leaks into the canonical body.
        """
        # Chrome lines (boot banner, future status panels)
        # skip the caption AND the timestamp completely so
        # they read as a block of sign-posting rather than
        # another system message.
        if line.chrome:
            return line.text
        # Assistant lines render the body alone — no `[12:34]
        # care` preamble. System lines keep the • caption so
        # they stand apart from a final answer that follows.
        prefix = {
            "assistant": "",
            "system": "•",
        }.get(line.role, line.role)
        ts = line.timestamp.strftime("%H:%M")
        body = line.text
        # Phase 8 P2 #15 — reaction marker on assistant lines.
        marker = ChatScreen._reaction_marker(line)
        if marker:
            body = f"{marker} {body}"
        if not prefix:
            # Reaction markers stay even on caption-less lines
            # so 👍/👎 are visible without scrolling.
            return body
        separator = (
            "\n\n"
            if ChatScreen._body_opens_with_block(body)
            else " "
        )
        return f"**[{ts}] {prefix}**{separator}{body}"

    # Markdown block-element openers checked when deciding
    # whether the caption can ride inline with the body. Order
    # doesn't matter — we test each line-start prefix.
    _MARKDOWN_BLOCK_PREFIXES: tuple[str, ...] = (
        "#",      # ATX headings (validated further below)
        ">",      # blockquote
        "- ",     # unordered list
        "* ",     # unordered list
        "+ ",     # unordered list
        "```",    # fenced code
        "~~~",    # fenced code
        "|",      # table row
        "---",    # thematic break / setext underline
        "===",    # setext underline
    )

    @staticmethod
    def _body_opens_with_block(body: str) -> bool:
        """Return ``True`` when ``body``'s first non-empty line
        is a Markdown block element that needs the ``\\n\\n``
        separator from the caption to render correctly."""
        if not body:
            return False
        # Find the first non-whitespace line — leading blank
        # lines don't change the answer; the OPENER does.
        for raw in body.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                # ATX heading requires `# ` (hash + space) or up
                # to six hashes. Bare `#word` (a comment in code
                # samples) shouldn't trip the gap.
                hashes = len(line) - len(line.lstrip("#"))
                if 1 <= hashes <= 6 and line[hashes:hashes + 1] in (" ", ""):
                    return True
                return False
            if line[:1].isdigit():
                # Ordered list — digit(s) then `.` then space.
                stripped = line.lstrip("0123456789")
                if stripped.startswith(". "):
                    return True
            for prefix in ChatScreen._MARKDOWN_BLOCK_PREFIXES:
                if prefix == "#":
                    continue  # handled above
                if line.startswith(prefix):
                    return True
            return False
        return False

    @staticmethod
    def _format_line_as_markdown_compact(line: ChatLine) -> str:
        """Phase 8 P2 #20 — compact-mode Markdown variant.
        Drops the bold caption row + blank separator so the
        Markdown widget renders just the body. Power users
        on small terminals see 2–3× more content per screen.

        Phase 8 P2 #15 — a reaction marker still rides at the
        front so 👍/👎 stays visible even in compact mode.
        """
        body = line.text
        marker = ChatScreen._reaction_marker(line)
        if marker:
            body = f"{marker} {body}"
        return body

    def _format_line_as_markdown_for_widget(self, line: ChatLine) -> str:
        """Pick the right Markdown projection based on the
        compact-mode flag. Used by both the initial mount in
        ``_post_line`` and the Ctrl+D toggle's re-render walk so
        the two paths stay in lockstep."""
        if self._compact_mode:
            md = self._format_line_as_markdown_compact(line)
        else:
            md = self._format_line_as_markdown(line)
        if line.linkify_commands:
            md = _linkify_slash_commands(md)
        return md

    # ------------------------------------------------------------------
    # Collapsed step bodies (Phase 2 P2)
    # ------------------------------------------------------------------

    @classmethod
    def _collapse_thresholds(cls) -> tuple[int, int]:
        """Resolve the (max_lines, max_chars) thresholds. Honours
        ``CARE_CHAT__COLLAPSE_MAX_LINES`` / ``CARE_CHAT__COLLAPSE_MAX_CHARS``
        for power users who want a different cutoff. Malformed
        values fall back to the class defaults — the toggle must
        never crash the chat.
        """
        import os

        def _resolve(env: str, default: int) -> int:
            raw = (os.environ.get(env) or "").strip()
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError:
                return default
            return max(1, value)

        return (
            _resolve("CARE_CHAT__COLLAPSE_MAX_LINES", cls._COLLAPSE_MAX_LINES),
            _resolve("CARE_CHAT__COLLAPSE_MAX_CHARS", cls._COLLAPSE_MAX_CHARS),
        )

    @classmethod
    def _should_collapse(cls, line: ChatLine) -> bool:
        """Only tool-role lines collapse. User / assistant /
        system stay full-width because each represents a
        deliberately user-visible event (their prompt, the
        agent's answer, a mode-flip notification) where
        truncating to a one-liner would be hostile rather than
        scan-friendly."""
        if line.role != "tool":
            return False
        max_lines, max_chars = cls._collapse_thresholds()
        if "\n" in line.text:
            return line.text.count("\n") + 1 > max_lines
        return len(line.text) > max_chars

    @classmethod
    def _collapse_text(cls, text: str) -> str:
        """Render the collapsed preview. Multi-line text shows
        line 1 plus "+N more lines" hint; over-long single lines
        truncate mid-string with an ellipsis. Either form ends
        in the same "Ctrl+E to expand" suffix so the user always
        knows the affordance."""
        _max_lines, max_chars = cls._collapse_thresholds()
        lines = text.split("\n")
        if len(lines) > 1:
            extra = len(lines) - 1
            return (
                f"{lines[0]} "
                f"[+{extra} more line{'s' if extra != 1 else ''}, "
                f"{cls._COLLAPSE_EXPAND_HINT}]"
            )
        truncated = text[: max_chars].rstrip()
        return (
            f"{truncated}… "
            f"[+{len(text) - max_chars} more chars, "
            f"{cls._COLLAPSE_EXPAND_HINT}]"
        )

    # ------------------------------------------------------------------
    # Tool-line inline code styling (Phase 8 P1 #11)
    # ------------------------------------------------------------------

    _INLINE_CODE_OPEN: str = "[reverse]"
    _INLINE_CODE_CLOSE: str = "[/reverse]"

    @classmethod
    def _apply_tool_inline_code(cls, plain: str) -> str:
        """Phase 8 P1 #11 — convert ``\\`segment\\``-delimited
        substrings of a tool line into Textual markup so ids
        like ``chain-X`` render as code-styled (reverse video)
        rather than as literal backticks.

        Source brackets are escaped via :func:`rich.markup.escape`
        first so user content (anything CARE renders through the
        tool surface) can't spoof markup tags. Backticks pass
        through ``escape`` unchanged, so the post-escape
        substitution targets only the user-intended ones.

        Returns the marked-up string. Callers should pass
        ``markup=True`` to the consuming Static so the tags
        render rather than appear literally.
        """
        import re
        from rich.markup import escape

        escaped = escape(plain)
        return re.sub(
            r"`([^`]+)`",
            lambda m: f"{cls._INLINE_CODE_OPEN}{m.group(1)}{cls._INLINE_CODE_CLOSE}",
            escaped,
        )

    def _render_tool_content(self, line: ChatLine) -> str:
        """Compute the markup string a tool Static widget should
        consume. Runs the line through the existing collapse-aware
        renderer first, then applies the inline-code substitution
        so the toggle round-trip preserves styling."""
        return self._apply_tool_inline_code(
            self._format_line_for_render(line),
        )

    def _format_line_for_render(self, line: ChatLine) -> str:
        """Render a chat line for live mount, taking the
        screen-wide ``_step_bodies_expanded`` toggle AND the
        Phase 8 P2 #20 ``_compact_mode`` flag into account.
        ``_format_line`` stays static so transcript copy +
        tests can read the full canonical text."""
        formatter = (
            self._format_line_compact
            if self._compact_mode
            else self._format_line
        )
        if (
            not self._step_bodies_expanded
            and self._should_collapse(line)
        ):
            preview_line = ChatLine(
                role=line.role,
                text=self._collapse_text(line.text),
                timestamp=line.timestamp,
                mode=line.mode,
            )
            return formatter(preview_line)
        return formatter(line)

    # ------------------------------------------------------------------
    # Production session log (Phase 6 P2)
    # ------------------------------------------------------------------

    @staticmethod
    def _session_log_dir() -> Path:
        """Resolve the directory Production-mode session logs
        land in. Honours ``CARE_CHAT__SESSION_LOG_DIR`` (tests
        redirect to ``tmp_path``); defaults to
        ``$XDG_STATE_HOME/care/sessions`` or
        ``~/.local/state/care/sessions``."""
        import os

        override = (
            os.environ.get("CARE_CHAT__SESSION_LOG_DIR") or ""
        ).strip()
        if override:
            return Path(override).expanduser()
        state_root = (
            os.environ.get("XDG_STATE_HOME") or ""
        ).strip() or "~/.local/state"
        return Path(state_root).expanduser() / "care" / "sessions"

    def _resolve_session_log_path(self) -> Path | None:
        """Lazy-create the per-session markdown log path.
        Returns ``None`` when the directory can't be created
        (permission errors, full disk, etc.) — the log stays
        best-effort. Once set, the path is stable for the rest
        of the screen lifetime so flipping ad_hoc ↔ production
        resumes the same file."""
        if self._session_log_path is not None:
            return self._session_log_path
        from datetime import datetime as _dt

        directory = self._session_log_dir()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _log.warning(
                "couldn't create session log dir %s: %s",
                directory, exc,
            )
            return None
        ts = _dt.now().strftime("%Y%m%dT%H%M%S")
        path = directory / f"care-session-{ts}.md"
        # Initialize with a header so the file is readable even
        # before any content lands.
        try:
            path.write_text(
                f"# CARE Production session — "
                f"{_dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                encoding="utf-8",
            )
        except OSError as exc:
            _log.warning(
                "couldn't initialise session log %s: %s",
                path, exc,
            )
            return None
        self._session_log_path = path
        _log.info("session log opened: %s", path)
        return path

    def _append_to_session_log(self, line: ChatLine) -> None:
        """Append one transcript entry to the session-log
        markdown file. Silently degrades on any I/O failure —
        the chat must keep working even when disk is
        unavailable."""
        path = self._resolve_session_log_path()
        if path is None:
            return
        ts = line.timestamp.strftime("%H:%M:%S")
        # Indent multi-line text two spaces so markdown keeps it
        # under the role bullet rather than starting a new list
        # entry.
        body = line.text.replace("\n", "\n  ")
        try:
            with path.open("a", encoding="utf-8") as fp:
                fp.write(f"- [{ts}] **{line.role}**: {body}\n")
        except OSError as exc:
            _log.warning(
                "session log append to %s failed: %s", path, exc,
            )

    # ------------------------------------------------------------------
    # Session resume (Phase 8 P1 #12)
    # ------------------------------------------------------------------

    # Matches one bullet line in the session-log markdown:
    #   `- [HH:MM:SS] **role**: body`
    # Body may be empty; continuation lines are indented with two
    # spaces and joined back when parsing.
    _SESSION_LOG_ENTRY_RE = __import__("re").compile(
        r"^- \[(\d{2}:\d{2}:\d{2})\] \*\*([a-zA-Z_]+)\*\*: ?(.*)$",
    )

    # Roles we re-post when rehydrating. Tool lines were never
    # written to the log (Phase 6 P2 skipped them), so this guard
    # also defends against future writers that might include them.
    _RESUMABLE_ROLES: frozenset[str] = frozenset(
        {"user", "assistant", "system"},
    )

    @classmethod
    def _parse_session_log(cls, path: Path) -> list[tuple[str, str]]:
        """Read a Production session-log markdown file and return
        a list of ``(role, body)`` tuples in file order.

        The writer in :meth:`_append_to_session_log` produces
        bullet lines like ``- [HH:MM:SS] **role**: body`` with
        any embedded newlines indented two spaces; this parser
        is the inverse — it joins indented continuation lines
        back into the original body separated by ``\\n``.

        Lines that don't match (the markdown header, blank
        lines, freeform commentary) terminate the current entry
        and are otherwise ignored, so a partially-corrupted log
        still produces the well-formed entries that DO match.
        All I/O failures degrade to an empty list — a broken
        sidecar must never crash the chat.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        entries: list[tuple[str, str]] = []
        current_role: str | None = None
        current_body: list[str] = []
        for raw in text.splitlines():
            match = cls._SESSION_LOG_ENTRY_RE.match(raw)
            if match is not None:
                if current_role is not None:
                    entries.append(
                        (current_role, "\n".join(current_body)),
                    )
                current_role = match.group(2).lower()
                current_body = [match.group(3)]
                continue
            if current_role is not None and raw.startswith("  "):
                current_body.append(raw[2:])
                continue
            if current_role is not None and not raw.strip():
                # Blank line ends the current entry but is
                # otherwise ignored (matches the writer's
                # one-line-per-entry shape).
                entries.append(
                    (current_role, "\n".join(current_body)),
                )
                current_role = None
                current_body = []
        if current_role is not None:
            entries.append((current_role, "\n".join(current_body)))
        return entries

    @classmethod
    def _list_session_logs(cls) -> list[Path]:
        """Return ``*.md`` files in :meth:`_session_log_dir`,
        newest first by mtime. Returns ``[]`` when the directory
        is missing / unreadable so the caller surfaces a clean
        "no sessions" line rather than a stacktrace."""
        directory = cls._session_log_dir()
        if not directory.exists() or not directory.is_dir():
            return []
        try:
            files = [
                p for p in directory.iterdir()
                if p.is_file() and p.suffix == ".md"
            ]
        except OSError:
            return []
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    @staticmethod
    def _first_user_preview(entries: list[tuple[str, str]]) -> str:
        """Pull the first ``user`` body from a parsed entry list
        and trim to ≤60 chars for the `/resume` listing. Falls
        back to a placeholder so a session that recorded only
        system / assistant lines still surfaces something."""
        first = next(
            (body for role, body in entries if role == "user"),
            "",
        )
        first = (first or "<no user prompt>").splitlines()[0]
        if len(first) <= 60:
            return first
        return first[:57].rstrip() + "…"

    def _select_session_for_resume(
        self,
        target: str,
        files: list[Path],
    ) -> Path | None:
        """Pick the right session file from ``files`` given the
        user's ``target``. ``"latest"`` returns the newest;
        otherwise tries exact filename match, then case-
        insensitive substring against the basename. Returns
        ``None`` when no match — the caller surfaces a friendly
        warning."""
        if target.lower() == "latest":
            return files[0]
        needle = target.lower()
        exact = next(
            (p for p in files if p.name.lower() == needle),
            None,
        )
        if exact is not None:
            return exact
        return next(
            (p for p in files if needle in p.name.lower()),
            None,
        )

    def _resume_session(self, target: str) -> None:
        """Phase 8 P1 #12 — the user-facing body of ``/resume``.

        - Empty ``target``: list up to 10 most recent sessions
          with the first user prompt as a preview.
        - ``"latest"``: resume the newest session.
        - Filename / substring: resume the matching session.

        Rehydrates by posting each parsed entry back through
        :meth:`_post_line` so it picks up the current chat
        surface's renderers (Markdown for assistant / system,
        Static for user / tool). Tool lines are never in the
        log so the rehydration is naturally collapse-free.
        """
        files = self._list_session_logs()
        if not files:
            self._post_line(
                "system",
                "No saved sessions found in "
                f"{self._session_log_dir()}.",
                severity="warning",
            )
            return
        target = (target or "").strip()
        if not target:
            lines = ["Recent sessions (newest first):"]
            for path in files[:10]:
                preview = self._first_user_preview(
                    self._parse_session_log(path),
                )
                lines.append(f"  {path.name} — {preview}")
            lines.append(
                "\nResume with /resume latest or /resume <filename>.",
            )
            self._post_line("system", "\n".join(lines))
            return

        path = self._select_session_for_resume(target, files)
        if path is None:
            self._post_line(
                "system",
                f"No session matching '{target}'. "
                "Try /resume (no args) to list saved sessions.",
                severity="warning",
            )
            return

        entries = self._parse_session_log(path)
        if not entries:
            self._post_line(
                "system",
                f"Session {path.name} is empty or malformed.",
                severity="warning",
            )
            return
        # Marker bracketing the rehydrated block so the user can
        # see exactly where the loaded session begins.
        self._post_line(
            "system",
            f"— resumed from `{path.name}` "
            f"({len(entries)} line{'s' if len(entries) != 1 else ''}) —",
        )
        for role, body in entries:
            if role in self._RESUMABLE_ROLES:
                self._post_line(role, body)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Theme switching (Phase 8 P2 #19)
    # ------------------------------------------------------------------

    def _available_theme_names(self) -> list[str]:
        """Return the sorted list of theme names Textual currently
        registers on the host app. Defensive against test hosts
        without a real app (``getattr`` falls back to an empty
        mapping)."""
        themes = getattr(self.app, "available_themes", None) or {}
        try:
            return sorted(themes)
        except Exception:
            return []

    def _current_theme_name(self) -> str | None:
        """Read the active theme name from the host app. Returns
        ``None`` when the app doesn't surface one (test scaffold,
        very early in mount, etc.)."""
        name = getattr(self.app, "theme", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None

    def _apply_theme_to_app(self, name: str) -> None:
        """Thin seam over ``self.app.theme = name`` so tests can
        monkeypatch this method to simulate a backend that
        raises without having to replace the read-only
        ``self.app`` property itself."""
        self.app.theme = name

    # ------------------------------------------------------------------
    # /theme sidecar persistence (Phase 9 P2)
    # ------------------------------------------------------------------

    @staticmethod
    def _theme_sidecar_path() -> Path:
        """Resolve the file that persists the user's chosen
        theme across sessions. Honours
        ``CARE_CHAT__THEME_SIDECAR`` (tests redirect to
        ``tmp_path``); defaults to
        ``$XDG_STATE_HOME/care/theme_preference.txt`` or
        ``~/.local/state/care/theme_preference.txt``."""
        import os

        override = (
            os.environ.get("CARE_CHAT__THEME_SIDECAR") or ""
        ).strip()
        if override:
            return Path(override).expanduser()
        state_root = (
            os.environ.get("XDG_STATE_HOME") or ""
        ).strip() or "~/.local/state"
        return (
            Path(state_root).expanduser() / "care" / "theme_preference.txt"
        )

    @classmethod
    def _read_theme_preference(cls) -> str | None:
        """Read the persisted theme name. Returns ``None`` when
        the sidecar is missing, unreadable, or empty — any of
        which means "no preference, use the Textual default"."""
        path = cls._theme_sidecar_path()
        if not path.exists() or not path.is_file():
            return None
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return raw or None

    @classmethod
    def _persist_theme_preference(cls, name: str) -> None:
        """Write the theme name to the sidecar so the next
        boot picks it up. Failures degrade silently — at
        worst the user has to re-run `/theme X` next session,
        which is not the end of the world."""
        path = cls._theme_sidecar_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        except OSError as exc:
            _log.warning(
                "couldn't persist theme preference at %s: %s",
                path, exc,
            )

    @classmethod
    def _clear_theme_preference(cls) -> None:
        """Remove the sidecar so the next boot uses Textual's
        default. Exposed for tests + future reset commands."""
        path = cls._theme_sidecar_path()
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            _log.warning(
                "couldn't clear theme preference at %s: %s",
                path, exc,
            )

    def _apply_persisted_theme(self) -> None:
        """On mount, restore the user's last-saved theme.
        Defensive at every layer: missing sidecar / unreadable
        file / theme no longer in the registry / setter raises
        all degrade silently so a corrupted preference can't
        block boot."""
        name = self._read_theme_preference()
        if not name:
            return
        available = set(self._available_theme_names())
        if name not in available:
            _log.warning(
                "persisted theme %r is not registered; skipping",
                name,
            )
            return
        current = self._current_theme_name()
        if name == current:
            return
        try:
            self._apply_theme_to_app(name)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "couldn't apply persisted theme %r: %s",
                name, exc,
            )

    def _handle_theme_command(self, target: str) -> None:
        """Phase 8 P2 #19 — `/theme` dispatcher.

        Bare ``target`` lists every Textual theme currently
        registered on the app plus highlights the active one;
        non-empty ``target`` validates against the registry and
        sets ``app.theme`` to apply. Unknown names surface a
        warning that lists the available set so the user can
        recover without leaving chat.
        """
        themes = self._available_theme_names()
        if not themes:
            self._post_line(
                "system",
                "No themes registered on the host app. "
                "(This is unusual — Textual ships built-in "
                "themes by default.)",
                severity="warning",
            )
            return
        current = self._current_theme_name()
        if not target:
            # System lines render as Markdown — single newlines
            # collapse to spaces, so the theme list has to ride
            # inside a fenced code block to render one-per-line.
            rows = [
                f"  {'*' if name == current else ' '} {name}"
                for name in themes
            ]
            body = (
                "Available themes (current marked with *):\n"
                "\n"
                "```\n"
                + "\n".join(rows)
                + "\n```\n"
                "\nApply with `/theme <name>`."
            )
            self._post_line("system", body)
            return
        if target not in themes:
            self._post_line(
                "system",
                f"Unknown theme `{target}`. "
                f"Available: {', '.join(themes)}.",
                severity="warning",
            )
            return
        if target == current:
            self._post_line(
                "system",
                f"Already using theme `{target}`.",
            )
            return
        try:
            self._apply_theme_to_app(target)
        except Exception as exc:  # noqa: BLE001
            _log.error("/theme %s failed: %s", target, exc, exc_info=True)
            self._post_line(
                "system",
                f"Couldn't apply theme `{target}`: {exc}",
                severity="error",
            )
            return
        # Phase 9 P2 — persist after the apply succeeded so a
        # broken theme name never lands in the sidecar. Failures
        # to write are logged but don't surface to the user; the
        # /theme command itself succeeded, persistence is a
        # nice-to-have on top. The inner helper already swallows
        # OSError, but a monkeypatched test (or future refactor)
        # could still raise — guard so the confirmation line
        # always lands.
        try:
            self._persist_theme_preference(target)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "couldn't persist theme preference for %r: %s",
                target, exc,
            )
        _log.info("theme switched to %s", target)
        self._post_line(
            "assistant",
            f"✓ Switched to theme `{target}`.",
        )

    # ------------------------------------------------------------------
    # Alt+T theme cycle (Phase 9 P2)
    # ------------------------------------------------------------------

    def action_cycle_theme(self) -> None:
        """Phase 9 P2 — Alt+T cycles through the registered
        themes in alphabetical order. Wraps around. Routes
        through ``_handle_theme_command`` so the apply,
        persistence, and confirmation message reuse the same
        code path as ``/theme <name>`` — keeps the contract
        identical regardless of how the user picks a theme."""
        themes = self._available_theme_names()
        if not themes:
            self._post_line(
                "system",
                "No themes registered on the host app — "
                "nothing to cycle.",
                severity="warning",
            )
            return
        if len(themes) == 1:
            self._post_line(
                "system",
                f"Only one theme available (`{themes[0]}`).",
            )
            return
        current = self._current_theme_name()
        try:
            idx = themes.index(current) if current else -1
        except ValueError:
            # Current theme isn't in the registry — fall back
            # to starting from the first theme so the cycle
            # still makes forward progress.
            idx = -1
        nxt = themes[(idx + 1) % len(themes)]
        self._handle_theme_command(nxt)

    # ------------------------------------------------------------------
    # /log inspector (Phase 8 P1 #13)
    # ------------------------------------------------------------------

    _LOG_LINE_RE = __import__("re").compile(
        r"^(?P<ts>\S+)\s+\[(?P<level>[A-Z]+)\]\s+(?P<name>[^:]+):\s+(?P<msg>.*)$",
    )

    _LOG_LEVEL_ORDER: dict[str, int] = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }

    _DEFAULT_LOG_VIEW_TAIL: int = 200
    _LOG_VIEW_MAX_DISPLAY: int = 40

    @classmethod
    def _log_view_tail(cls) -> int:
        """Number of records to scan from the END of the log
        file before applying filters. Honours
        ``CARE_CHAT__LOG_VIEW_TAIL`` so power users can crank
        the window up; clamped to ≥1 so a malformed env can't
        disable the command."""
        import os

        raw = (os.environ.get("CARE_CHAT__LOG_VIEW_TAIL") or "").strip()
        if not raw:
            return cls._DEFAULT_LOG_VIEW_TAIL
        try:
            n = int(raw)
        except ValueError:
            return cls._DEFAULT_LOG_VIEW_TAIL
        return max(1, n)

    # ------------------------------------------------------------------
    # @-file references (Phase 8 P1 #6)
    # ------------------------------------------------------------------

    # Matches `@<path>` where path is anything not whitespace. The
    # `@` must be at the start of input OR preceded by whitespace
    # so an email like `user@example.com` doesn't get rewritten.
    _FILE_REF_RE = __import__("re").compile(
        r"(?:^|(?<=\s))@(\S+)",
    )

    _DEFAULT_FILE_REF_MAX_BYTES: int = 51200

    @classmethod
    def _file_ref_max_bytes(cls) -> int:
        """Per-file size cap for `@<path>` embeds. Honours
        ``CARE_CHAT__FILE_REF_MAX_BYTES`` (default 50 KB) so
        accidental refs to a large log don't blow up the
        MAGE prompt. Clamped to ≥1 byte; malformed values
        fall back to the default."""
        import os

        raw = (
            os.environ.get("CARE_CHAT__FILE_REF_MAX_BYTES") or ""
        ).strip()
        if not raw:
            return cls._DEFAULT_FILE_REF_MAX_BYTES
        try:
            n = int(raw)
        except ValueError:
            return cls._DEFAULT_FILE_REF_MAX_BYTES
        return max(1, n)

    # Tiny extension → fence-language map so the embedded block
    # carries a Markdown hint MAGE / Markdown renderers can pick
    # up. Unknown extensions fall through to a bare fence.
    _FILE_REF_LANG_MAP: dict[str, str] = {
        ".py": "python",
        ".sh": "bash",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".rs": "rust",
        ".go": "go",
        ".rb": "ruby",
        ".java": "java",
        ".kt": "kotlin",
        ".swift": "swift",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".css": "css",
        ".html": "html",
        ".xml": "xml",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".sql": "sql",
        ".dockerfile": "dockerfile",
    }

    @classmethod
    def _fence_lang_for_path(cls, path: Path) -> str:
        return cls._FILE_REF_LANG_MAP.get(path.suffix.lower(), "")

    def _resolve_file_refs(self, task: str) -> str:
        """Phase 8 P1 #6 — scan ``task`` for `@<path>` tokens,
        read each file (size-capped via
        :meth:`_file_ref_max_bytes`), and substitute the body
        inline as a fenced Markdown block so MAGE sees the
        file content as part of the task.

        Resolution rules:

        * ``@"path with spaces"`` / ``@'path'`` — explicit
          quoted form. Everything between the quotes is the
          path, even if it spans multiple whitespace-separated
          words.
        * ``@<path>`` — bare form. The match starts at the
          first non-whitespace run after ``@`` and *extends
          greedily* across whitespace as long as the candidate
          path keeps existing on disk. So
          ``@../My Notes.md`` resolves to that exact file
          without the user needing to quote, while
          ``@notes.md please summarize`` keeps just
          ``notes.md`` (the trailing prose isn't a valid file).
          Trailing sentence punctuation (``,.;:!?)``) is
          stripped before disk-checking so
          ``see @notes.md.`` matches the file at ``notes.md``.
        * Extension stops at line breaks and at the next
          ``@<word>`` boundary, so two adjacent refs on the
          same line never collide.

        Failures (missing file, oversize, binary / decode
        error, OSError) surface as `system` warning lines but
        DON'T abort generation — the original `@<path>` token
        stays in place so the user can fix the ref or just
        let MAGE see the literal text.
        """
        spans = self._scan_at_refs(task)
        if not spans:
            return task
        max_bytes = self._file_ref_max_bytes()
        result_parts: list[str] = []
        cursor = len(task)
        # Walk back-to-front so earlier offsets stay valid as
        # we substitute.
        for start, end, raw_path, trailing in reversed(spans):
            substitution = self._read_file_ref(
                raw_path, max_bytes=max_bytes,
            )
            result_parts.append(task[end:cursor])
            if substitution is None:
                # Leave the original literal span in place so
                # the user sees their typo and MAGE doesn't
                # ingest a half-baked embed.
                result_parts.append(task[start:end])
            else:
                result_parts.append(trailing)
                result_parts.append(substitution)
                self._post_file_ref_attached(raw_path)
            cursor = start
        result_parts.append(task[:cursor])
        return "".join(reversed(result_parts))

    # Trailing sentence punctuation peeled off bare-form paths.
    # Doesn't apply inside `@"…"` quoted refs — the user is
    # being explicit about what belongs to the path.
    _AT_REF_TRAILING_PUNCT: str = ",.;:!?)"

    def _scan_at_refs(
        self, task: str,
    ) -> list[tuple[int, int, str, str]]:
        """Walk ``task`` and return every ``@<path>`` span in
        document order.

        Each tuple is ``(start, end, path, trailing)`` where:

        * ``start`` / ``end`` cover the full literal substring
          consumed by the ref, including the leading ``@``.
        * ``path`` is the resolved path candidate (already
          stripped of trailing sentence punctuation and / or
          enclosing quotes).
        * ``trailing`` is the tail of the matched span that
          isn't part of the path — re-emitted after the
          substituted file body so prose like
          ``see @notes.md, please`` keeps the trailing
          comma+space in the output.

        Bare form is greedy: starting from the
        non-whitespace token after ``@``, the helper appends
        whitespace-separated chunks while the resulting path
        keeps existing on disk and remembers the longest match
        that resolved. That makes paths with spaces work
        without quoting, while ``@notes.md please summarize``
        still collapses to just ``notes.md`` (the trailing
        prose isn't a file).

        Quoted form (``@"…"`` / ``@'…'``) takes the entire
        quoted region verbatim — no greedy extension, no
        punctuation stripping.
        """
        spans: list[tuple[int, int, str, str]] = []
        i = 0
        n = len(task)
        while i < n:
            if task[i] != "@":
                i += 1
                continue
            # `@` must be at start-of-input or preceded by
            # whitespace, otherwise it's an email-style use
            # ("user@example.com") and we leave it alone.
            if i > 0 and not task[i - 1].isspace():
                i += 1
                continue
            after_at = i + 1
            if after_at >= n:
                break
            quote = task[after_at]
            if quote in ('"', "'"):
                closing = task.find(quote, after_at + 1)
                if closing == -1:
                    # Unclosed quote — bail to the next char.
                    i = after_at
                    continue
                raw_path = task[after_at + 1 : closing]
                end = closing + 1
                spans.append((i, end, raw_path, ""))
                i = end
                continue
            # Bare form. Find the end of the initial token
            # (until whitespace or end-of-line).
            tok_end = after_at
            while tok_end < n and not task[tok_end].isspace():
                tok_end += 1
            if tok_end == after_at:
                # `@` followed by whitespace → ignore.
                i = after_at
                continue
            initial_token = task[after_at:tok_end]
            best_end, best_path, best_trailing = (
                self._greedy_extend_at_ref(
                    task, after_at, tok_end, initial_token,
                )
            )
            spans.append((i, best_end, best_path, best_trailing))
            i = best_end
        return spans

    def _greedy_extend_at_ref(
        self,
        task: str,
        after_at: int,
        initial_end: int,
        initial_token: str,
    ) -> tuple[int, str, str]:
        """Walk forward from ``initial_end`` adding whitespace
        + word chunks while the resulting path keeps resolving
        to an existing file. Returns the span's final
        ``(end, path, trailing)``.

        The bare-form fallback (when nothing on disk matches
        any extension): the initial token is used verbatim and
        the caller emits the standard "not found" warning so
        the user knows their ref didn't resolve. This keeps
        the legacy behaviour for misspelled @-refs intact.
        """
        import re

        # Initial candidate — same shape as the legacy path.
        initial_cleaned = re.sub(
            r"[" + re.escape(self._AT_REF_TRAILING_PUNCT) + r"]+$",
            "",
            initial_token,
        )
        initial_trailing = initial_token[len(initial_cleaned):]

        # Track the best EXISTING extension we've seen so far.
        # If nothing on disk ever resolves, we fall back to
        # the initial token (so the user gets the normal
        # "not found" warning).
        best_end = initial_end
        best_path = initial_cleaned
        best_trailing = initial_trailing
        if self._at_ref_path_exists(initial_cleaned):
            # initial token already maps to a file — keep it
            # as the floor and look for a longer match.
            pass

        cursor = initial_end
        n = len(task)
        while cursor < n:
            # Walk over the inter-word whitespace (but stop at
            # a newline — refs don't cross line boundaries).
            ws_start = cursor
            while cursor < n and task[cursor] in (" ", "\t"):
                cursor += 1
            if cursor >= n or task[cursor] == "\n":
                break
            if cursor == ws_start:
                # No whitespace consumed → end of extendable
                # region (shouldn't normally happen but stays
                # defensive against malformed input).
                break
            # Stop if the next word is the start of ANOTHER
            # @-ref (whitespace + `@` + non-space).
            if (
                task[cursor] == "@"
                and cursor + 1 < n
                and not task[cursor + 1].isspace()
            ):
                break
            # Advance to end of the next word.
            word_end = cursor
            while word_end < n and not task[word_end].isspace():
                word_end += 1
            candidate_raw = task[after_at:word_end]
            candidate_cleaned = re.sub(
                r"[" + re.escape(self._AT_REF_TRAILING_PUNCT) + r"]+$",
                "",
                candidate_raw,
            )
            if self._at_ref_path_exists(candidate_cleaned):
                best_end = word_end
                best_path = candidate_cleaned
                best_trailing = candidate_raw[len(candidate_cleaned):]
            cursor = word_end
        return best_end, best_path, best_trailing

    @staticmethod
    def _at_ref_path_exists(raw: str) -> bool:
        """Cheap "is this a real file?" check used by the
        greedy extender. ``expanduser`` + cwd-relative
        resolution mirrors :meth:`_read_file_ref` so we don't
        promise a file resolves and then fail at read time."""
        if not raw:
            return False
        try:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            return path.is_file()
        except OSError:
            return False

    # Phase 8 P3 — image extensions get base64-encoded into a
    # `<image>` envelope instead of trying to decode as UTF-8.
    # Vision-capable models (Claude / GPT-4V / Gemini) consume
    # `data:image/<fmt>;base64,...` natively.
    _IMAGE_EXTENSIONS: dict[str, str] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    # Larger budget for images — typical screenshot is 100-500 KB,
    # well above the 50 KB text budget. Independent of the text
    # cap so the user can crank either knob without affecting
    # the other.
    _DEFAULT_IMAGE_REF_MAX_BYTES: int = 1_048_576  # 1 MB

    # PDF text extraction has its own budget. A 50-page paper is
    # easily a few MB on disk but reads down to ~150 KB of
    # extracted text — what MAGE actually sees. We cap the
    # on-disk PDF and the extracted text separately so neither
    # blows up the prompt.
    _DEFAULT_PDF_REF_MAX_BYTES: int = 10 * 1_048_576  # 10 MB on disk
    _DEFAULT_PDF_TEXT_MAX_CHARS: int = 200_000

    # Office / rich-text documents (docx, pptx, xlsx, odt, rtf, …) get the
    # same two-cap treatment as PDFs: a generous on-disk cap (these
    # containers are zip-compressed, so 25 MB holds a lot of content) plus
    # a separate extracted-text cap so a huge spreadsheet doesn't blow the
    # MAGE prompt budget.
    _DEFAULT_DOC_REF_MAX_BYTES: int = 25 * 1_048_576  # 25 MB on disk
    _DEFAULT_DOC_TEXT_MAX_CHARS: int = 200_000

    @classmethod
    def _pdf_ref_max_bytes(cls) -> int:
        """Cap on PDF size in bytes. Honours
        ``CARE_CHAT__PDF_REF_MAX_BYTES`` (default 10 MB)."""
        import os

        raw = (
            os.environ.get("CARE_CHAT__PDF_REF_MAX_BYTES") or ""
        ).strip()
        if not raw:
            return cls._DEFAULT_PDF_REF_MAX_BYTES
        try:
            n = int(raw)
        except ValueError:
            return cls._DEFAULT_PDF_REF_MAX_BYTES
        return max(1, n)

    @classmethod
    def _pdf_text_max_chars(cls) -> int:
        """Cap on extracted PDF text in characters. Honours
        ``CARE_CHAT__PDF_TEXT_MAX_CHARS`` (default 200000)."""
        import os

        raw = (
            os.environ.get("CARE_CHAT__PDF_TEXT_MAX_CHARS") or ""
        ).strip()
        if not raw:
            return cls._DEFAULT_PDF_TEXT_MAX_CHARS
        try:
            n = int(raw)
        except ValueError:
            return cls._DEFAULT_PDF_TEXT_MAX_CHARS
        return max(1, n)

    @classmethod
    def _doc_ref_max_bytes(cls) -> int:
        """Cap on office-document size in bytes. Honours
        ``CARE_CHAT__DOC_REF_MAX_BYTES`` (default 25 MB)."""
        import os

        raw = (
            os.environ.get("CARE_CHAT__DOC_REF_MAX_BYTES") or ""
        ).strip()
        if not raw:
            return cls._DEFAULT_DOC_REF_MAX_BYTES
        try:
            n = int(raw)
        except ValueError:
            return cls._DEFAULT_DOC_REF_MAX_BYTES
        return max(1, n)

    @classmethod
    def _doc_text_max_chars(cls) -> int:
        """Cap on extracted document text in characters. Honours
        ``CARE_CHAT__DOC_TEXT_MAX_CHARS`` (default 200000)."""
        import os

        raw = (
            os.environ.get("CARE_CHAT__DOC_TEXT_MAX_CHARS") or ""
        ).strip()
        if not raw:
            return cls._DEFAULT_DOC_TEXT_MAX_CHARS
        try:
            n = int(raw)
        except ValueError:
            return cls._DEFAULT_DOC_TEXT_MAX_CHARS
        return max(1, n)

    @classmethod
    def _image_ref_max_bytes(cls) -> int:
        """Per-image size cap. Honours
        ``CARE_CHAT__IMAGE_REF_MAX_BYTES`` (default 1 MB).
        Independent of the text-file budget so power users
        can crank either knob in isolation."""
        import os

        raw = (
            os.environ.get("CARE_CHAT__IMAGE_REF_MAX_BYTES") or ""
        ).strip()
        if not raw:
            return cls._DEFAULT_IMAGE_REF_MAX_BYTES
        try:
            n = int(raw)
        except ValueError:
            return cls._DEFAULT_IMAGE_REF_MAX_BYTES
        return max(1, n)

    @classmethod
    def _mime_for_image_path(cls, path: Path) -> str | None:
        """Map an image-file extension to its MIME type.
        Returns ``None`` for non-image extensions."""
        return cls._IMAGE_EXTENSIONS.get(path.suffix.lower())

    def _read_file_ref(
        self, ref_path: str, *, max_bytes: int,
    ) -> str | None:
        """Read a single `@<path>` target and project the body
        into a fenced Markdown block (text files) or a base64
        `<image>` envelope (image files). Returns ``None`` on
        any failure (after surfacing a warning) so the caller
        can keep the original token in place."""
        try:
            path = Path(ref_path).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system",
                f"@-ref `{ref_path}` couldn't resolve: {exc}",
                severity="warning",
            )
            return None
        if not path.exists():
            self._post_line(
                "system",
                f"@-ref `{ref_path}` not found.",
                severity="warning",
            )
            return None
        if not path.is_file():
            self._post_line(
                "system",
                f"@-ref `{ref_path}` is not a regular file.",
                severity="warning",
            )
            return None
        # Phase 8 P3 — image branch. Detected by extension so
        # we don't try to decode binary bytes as UTF-8.
        mime = self._mime_for_image_path(path)
        if mime is not None:
            return self._read_image_ref(path, mime, ref_path)
        # PDF branch. Binary container, so we can't UTF-8 decode
        # it; instead extract the text layer via pypdf and embed
        # as a fenced block. Independent (larger) size cap so a
        # multi-page PDF doesn't bump up against the text-file
        # limit.
        if path.suffix.lower() == ".pdf":
            return self._read_pdf_ref(path, ref_path)
        # Office / rich-text document branch (docx, pptx, xlsx, odt, rtf,
        # …). Binary containers like PDFs: extract a plain-text projection
        # via the format's library and embed as a fenced <file> block with
        # its own (larger) size/text cap.
        from care.runtime.document_extract import ROUTABLE_EXTENSIONS

        if path.suffix.lower() in ROUTABLE_EXTENSIONS:
            return self._read_document_ref(path, ref_path)
        try:
            size = path.stat().st_size
        except OSError as exc:
            self._post_line(
                "system",
                f"@-ref `{ref_path}` stat failed: {exc}",
                severity="warning",
            )
            return None
        if size > max_bytes:
            self._post_line(
                "system",
                f"@-ref `{ref_path}` is {size} bytes (> "
                f"{max_bytes} cap) — set "
                "`CARE_CHAT__FILE_REF_MAX_BYTES` to raise the "
                "limit.",
                severity="warning",
            )
            return None
        try:
            body = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            self._post_line(
                "system",
                f"@-ref `{ref_path}` isn't valid UTF-8 "
                "(binary file?).",
                severity="warning",
            )
            return None
        except OSError as exc:
            self._post_line(
                "system",
                f"@-ref `{ref_path}` read failed: {exc}",
                severity="warning",
            )
            return None
        lang = self._fence_lang_for_path(path)
        fence_open = f"```{lang}".rstrip()
        return f"\n<file path=\"{path}\">\n{fence_open}\n{body}\n```\n</file>\n"

    def _read_image_ref(
        self, path: Path, mime: str, ref_path: str,
    ) -> str | None:
        """Read an image @-ref + project it as a base64
        ``<image>`` envelope. Size-capped via
        :meth:`_image_ref_max_bytes` (default 1 MB) — typical
        screenshots fit comfortably while accidental
        ref-to-a-huge-png-asset gets warned. Surfaces a
        friendly hint when the MAGE model isn't known to be
        vision-capable (best-effort — the substring heuristic
        below covers Claude / GPT-4o / Gemini and silently
        passes through unknown models)."""
        import base64

        try:
            size = path.stat().st_size
        except OSError as exc:
            self._post_line(
                "system",
                f"@-image `{ref_path}` stat failed: {exc}",
                severity="warning",
            )
            return None
        max_bytes = self._image_ref_max_bytes()
        if size > max_bytes:
            self._post_line(
                "system",
                f"@-image `{ref_path}` is {size} bytes (> "
                f"{max_bytes} cap) — set "
                "`CARE_CHAT__IMAGE_REF_MAX_BYTES` to raise it.",
                severity="warning",
            )
            return None
        try:
            raw = path.read_bytes()
        except OSError as exc:
            self._post_line(
                "system",
                f"@-image `{ref_path}` read failed: {exc}",
                severity="warning",
            )
            return None
        encoded = base64.b64encode(raw).decode("ascii")
        # Hint the user when the current model probably can't
        # consume vision — best-effort, only fires when we
        # explicitly recognise a text-only model id.
        if not self._model_likely_supports_vision():
            self._post_line(
                "system",
                f"@-image `{ref_path}` embedded as base64 — "
                "current model may not support vision input. "
                "Switch to Claude 3+ / GPT-4o / Gemini for "
                "image-aware responses.",
                severity="warning",
            )
        return (
            f"\n<image path=\"{path}\" mime=\"{mime}\" "
            f"size_bytes=\"{size}\">\n"
            f"data:{mime};base64,{encoded}\n"
            "</image>\n"
        )

    def _read_pdf_ref(self, path: Path, ref_path: str) -> str | None:
        """Read a PDF @-ref, extract its text layer via ``pypdf``,
        and project it as a fenced ``<file>`` block. Returns
        ``None`` on any failure (after surfacing a warning) so
        the caller keeps the original token in place.

        Two independent caps:

        * On-disk size capped by :meth:`_pdf_ref_max_bytes`
          (default 10 MB) so an accidental ref to a massive
          PDF doesn't OOM the reader.
        * Extracted text capped by :meth:`_pdf_text_max_chars`
          (default 200 000 chars) so a 500-page PDF doesn't
          blow the MAGE prompt budget — truncation wins a
          partial summary over an outright fail.
        """
        try:
            size = path.stat().st_size
        except OSError as exc:
            self._post_line(
                "system",
                f"@-pdf `{ref_path}` stat failed: {exc}",
                severity="warning",
            )
            return None
        max_bytes = self._pdf_ref_max_bytes()
        if size > max_bytes:
            self._post_line(
                "system",
                f"@-pdf `{ref_path}` is {size} bytes (> "
                f"{max_bytes} cap) — set "
                "`CARE_CHAT__PDF_REF_MAX_BYTES` to raise it.",
                severity="warning",
            )
            return None
        try:
            from pypdf import PdfReader
        except ImportError:
            self._post_line(
                "system",
                f"@-pdf `{ref_path}` can't be embedded — "
                "`pypdf` isn't installed. `pip install pypdf` "
                "to enable PDF refs.",
                severity="warning",
            )
            return None
        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system",
                f"@-pdf `{ref_path}` couldn't be opened: {exc}",
                severity="warning",
            )
            return None
        pages_text: list[str] = []
        running = 0
        cap = self._pdf_text_max_chars()
        truncated = False
        try:
            page_count = len(reader.pages)
        except Exception:  # noqa: BLE001
            page_count = 0
        for page_index in range(page_count):
            try:
                text = reader.pages[page_index].extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "pdf @-ref %s page %d extract failed: %s",
                    ref_path, page_index + 1, exc,
                )
                continue
            text = text.strip()
            if not text:
                continue
            chunk = f"--- page {page_index + 1} ---\n{text}"
            if running + len(chunk) > cap:
                remaining = cap - running
                if remaining > 0:
                    pages_text.append(chunk[:remaining])
                truncated = True
                break
            pages_text.append(chunk)
            running += len(chunk)
        if not pages_text:
            self._post_line(
                "system",
                f"@-pdf `{ref_path}` yielded no extractable text — "
                "scanned PDFs need OCR (not built in).",
                severity="warning",
            )
            return None
        body = "\n\n".join(pages_text)
        suffix = (
            "\n[truncated — set CARE_CHAT__PDF_TEXT_MAX_CHARS to raise]"
            if truncated
            else ""
        )
        return (
            f"\n<file path=\"{path}\" type=\"pdf\" "
            f"pages=\"{page_count}\">\n```\n{body}{suffix}\n```\n"
            "</file>\n"
        )

    def _read_document_ref(self, path: Path, ref_path: str) -> str | None:
        """Read an office / rich-text document @-ref (docx, pptx, xlsx,
        odt, rtf, …), extract a plain-text projection via the format's
        library, and project it as a fenced ``<file>`` block. Mirrors
        :meth:`_read_pdf_ref`'s two-cap + friendly-failure contract:

        * on-disk size capped by :meth:`_doc_ref_max_bytes` (default
          25 MB) so an accidental ref to a giant file doesn't OOM the
          parser;
        * extracted text capped by :meth:`_doc_text_max_chars` (default
          200 000 chars) so a huge spreadsheet doesn't blow the MAGE
          prompt — truncation wins a partial read over an outright fail.

        Returns ``None`` on any failure (after surfacing a warning) so the
        caller keeps the original @-token in place.
        """
        try:
            size = path.stat().st_size
        except OSError as exc:
            self._post_line(
                "system",
                f"@-doc `{ref_path}` stat failed: {exc}",
                severity="warning",
            )
            return None
        max_bytes = self._doc_ref_max_bytes()
        if size > max_bytes:
            self._post_line(
                "system",
                f"@-doc `{ref_path}` is {size} bytes (> "
                f"{max_bytes} cap) — set "
                "`CARE_CHAT__DOC_REF_MAX_BYTES` to raise it.",
                severity="warning",
            )
            return None
        from care.runtime.document_extract import (
            DocumentExtractionError,
            extract_document_text,
        )

        try:
            text = extract_document_text(path)
        except DocumentExtractionError as exc:
            self._post_line(
                "system",
                f"@-doc `{ref_path}` couldn't be read — {exc}",
                severity="warning",
            )
            return None
        text = (text or "").strip()
        if not text:
            self._post_line(
                "system",
                f"@-doc `{ref_path}` yielded no extractable text.",
                severity="warning",
            )
            return None
        cap = self._doc_text_max_chars()
        truncated = len(text) > cap
        if truncated:
            text = text[:cap]
        suffix = (
            "\n[truncated — set CARE_CHAT__DOC_TEXT_MAX_CHARS to raise]"
            if truncated
            else ""
        )
        doc_type = path.suffix.lower().lstrip(".")
        return (
            f"\n<file path=\"{path}\" type=\"{doc_type}\">\n"
            f"```\n{text}{suffix}\n```\n"
            "</file>\n"
        )

    # Substring patterns that identify known-vision-capable
    # model families. Best-effort — unknown models default to
    # True (pass-through) so we don't spuriously warn on a
    # vision-capable model that happens to use a non-matching
    # slug.
    _VISION_CAPABLE_PATTERNS: tuple[str, ...] = (
        "claude-3",
        "claude-3-5",
        "claude-3.5",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-haiku-4",
        "gpt-4o",
        "gpt-4-vision",
        "gpt-4.1",
        "gemini",
        "o1",
    )

    _TEXT_ONLY_PATTERNS: tuple[str, ...] = (
        "gpt-3.5",
        "claude-instant",
    )

    def _model_likely_supports_vision(self) -> bool:
        """Return False when the active model is in the
        text-only list. Defaults to True for unknown models so
        the warning doesn't spuriously fire on a future
        vision-capable provider."""
        model = self._resolve_active_model()
        if not model:
            return True
        needle = model.lower()
        if any(p in needle for p in self._TEXT_ONLY_PATTERNS):
            return False
        return True

    def _resolve_log_file(self) -> Path | None:
        """Find the active log file. Prefers ``CARE_LOG_FILE``
        env var (which the launcher / `configure_from_env`
        also reads) so the chat surface stays in lockstep with
        the file logger. Falls back to walking root-logger
        handlers for a `baseFilename` so users who configured
        logging programmatically still get a working `/log`."""
        import os

        raw = (os.environ.get("CARE_LOG_FILE") or "").strip()
        if raw:
            return Path(raw).expanduser()
        root = logging.getLogger()
        for handler in root.handlers:
            base = getattr(handler, "baseFilename", None)
            if isinstance(base, str) and base:
                return Path(base)
        return None

    @classmethod
    def _parse_log_args(
        cls, arg: str,
    ) -> tuple[int | None, str | None]:
        """Project the `/log` argv into ``(min_level, module)``.

        - First token, when matching a known level
          (``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR`` /
          ``CRITICAL``, case-insensitive), sets the minimum
          severity threshold.
        - Remaining tokens (or the first token when no level
          was named) supply the case-insensitive module-name
          substring filter. Empty string ⇒ no filter.
        """
        parts = arg.split()
        if not parts:
            return None, None
        first_upper = parts[0].upper()
        if first_upper in cls._LOG_LEVEL_ORDER:
            level = cls._LOG_LEVEL_ORDER[first_upper]
            module = " ".join(parts[1:]).strip() or None
            return level, module
        return None, " ".join(parts).strip() or None

    def _handle_log_command(self, arg: str) -> None:
        """Phase 8 P1 #13 — main `/log` dispatcher. Reads the
        active log file, tails it to :meth:`_log_view_tail`
        records, filters them by the parsed level + module
        args, and posts the matched records as a system line.
        Best-effort everywhere — missing log file / read
        failure / parse misses all surface friendly warnings
        rather than crashing."""
        level_filter, module_filter = self._parse_log_args(arg)
        path = self._resolve_log_file()
        if path is None or not path.exists():
            self._post_line(
                "system",
                "No log file is active. Set `CARE_LOG_FILE` "
                "in `.env` to enable file logging.",
                severity="warning",
            )
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _log.error("/log read failed: %s", exc, exc_info=True)
            self._post_line(
                "system",
                f"Couldn't read log file `{path.name}`: {exc}",
                severity="error",
            )
            return
        raw_lines = text.splitlines()
        tail = self._log_view_tail()
        recent = raw_lines[-tail:] if len(raw_lines) > tail else raw_lines
        matched = self._filter_log_records(
            recent, level_filter, module_filter,
        )
        if not matched:
            self._post_line(
                "system",
                "No log records match the filter "
                f"(scanned last {len(recent)} of {len(raw_lines)} lines "
                f"in `{path.name}`).",
            )
            return
        display = matched[-self._LOG_VIEW_MAX_DISPLAY:]
        scope_bits: list[str] = []
        if level_filter is not None:
            scope_bits.append(
                f"level≥{self._level_name(level_filter)}",
            )
        if module_filter:
            scope_bits.append(f"module≈{module_filter}")
        scope = f" ({', '.join(scope_bits)})" if scope_bits else ""
        header = (
            f"Last {len(display)} of {len(matched)} matching "
            f"record(s) from `{path.name}`{scope}:\n"
        )
        self._post_line("system", header + "\n".join(display))

    @classmethod
    def _filter_log_records(
        cls,
        lines: list[str],
        level_filter: int | None,
        module_filter: str | None,
    ) -> list[str]:
        """Apply level + module filters. Lines that don't
        match the expected format are silently skipped so a
        partially-corrupted log still surfaces well-formed
        records."""
        module_needle = (module_filter or "").lower()
        matched: list[str] = []
        for raw in lines:
            match = cls._LOG_LINE_RE.match(raw)
            if match is None:
                continue
            if level_filter is not None:
                line_level = cls._LOG_LEVEL_ORDER.get(
                    match.group("level"), 0,
                )
                if line_level < level_filter:
                    continue
            if module_needle and module_needle not in match.group("name").lower():
                continue
            matched.append(raw)
        return matched

    @classmethod
    def _level_name(cls, level: int) -> str:
        """Reverse-map the numeric level to its name. Helper
        for the scope display in the header."""
        for name, value in cls._LOG_LEVEL_ORDER.items():
            if value == level:
                return name
        return str(level)

    # ------------------------------------------------------------------
    # Transcript export (Phase 8 P3 — /export)
    # ------------------------------------------------------------------

    _EXPORT_FORMATS: frozenset[str] = frozenset(
        {"md", "mdx", "json", "html"},
    )

    # ------------------------------------------------------------------
    # /voice — transcription status (Phase 8 P3 stub)
    # ------------------------------------------------------------------

    _VOICE_PACKAGES: tuple[str, ...] = ("whisper", "faster_whisper")
    _VOICE_AUDIO_EXTENSIONS: frozenset[str] = frozenset({
        ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm",
        ".aac", ".aiff", ".aif", ".opus",
    })

    @classmethod
    def _detect_voice_backend(cls) -> str | None:
        """Return the name of the first importable Whisper
        backend, or ``None`` when none are installed.
        ``whisper`` is the OpenAI reference impl; the
        ``faster_whisper`` fork is a CTranslate2-backed
        speedup popular for on-device transcription. Either
        works — the chat surface doesn't care which."""
        import importlib.util

        for name in cls._VOICE_PACKAGES:
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ValueError):
                spec = None
            if spec is not None:
                return name
        return None

    @staticmethod
    def _invoke_whisper(backend: str, path: Path) -> str:
        """Phase 9 P3 — thin shim that dispatches to the
        detected Whisper backend and returns the transcribed
        text. Tests monkeypatch this seam so the suite stays
        independent of an installed Whisper. The backend
        APIs differ: ``whisper.load_model("base").transcribe(path)``
        returns a dict with ``text``; ``faster_whisper`` returns
        ``(segments, info)`` where ``segments`` is an iterable
        with a ``text`` attribute per chunk. Both paths produce
        a single concatenated string for the caller."""
        if backend == "whisper":
            import whisper  # type: ignore[import-not-found]

            model = whisper.load_model("base")
            result = model.transcribe(str(path))
            return str(result.get("text", "")).strip()
        if backend == "faster_whisper":
            from faster_whisper import (  # type: ignore[import-not-found]
                WhisperModel,
            )

            model = WhisperModel("base")
            segments, _info = model.transcribe(str(path))
            return " ".join(
                (seg.text or "").strip() for seg in segments
            ).strip()
        raise RuntimeError(f"Unknown voice backend: {backend!r}")

    def _handle_voice_command(self, arg: str) -> None:
        """``/voice``                 → backend status.
        ``/voice transcribe <path>``  → run Whisper on an
        audio file and prefill the chat input with the
        transcript. Phase 9 P3 — file-based transcription is
        the contained half of the spec; the live recording
        (push-to-talk via sounddevice) still needs its own
        iteration but can plug into the same Whisper helper."""
        tokens = (arg or "").split(maxsplit=1)
        if not tokens or tokens[0].lower() == "status":
            self._render_voice_status()
            return
        if tokens[0].lower() == "transcribe":
            rest = tokens[1] if len(tokens) > 1 else ""
            if not rest:
                self._post_line(
                    "system",
                    "Usage: `/voice transcribe <path-to-audio>`. "
                    "Supported: .wav .mp3 .m4a .ogg .flac .webm "
                    ".aac .aiff .opus.",
                    severity="warning",
                )
                return
            self._voice_transcribe(rest.strip())
            return
        self._post_line(
            "system",
            f"Unknown /voice sub-command `{tokens[0]}`. "
            "Use `/voice`, `/voice status`, or "
            "`/voice transcribe <path>`.",
            severity="warning",
        )

    def _render_voice_status(self) -> None:
        backend = self._detect_voice_backend()
        if backend is None:
            self._post_line(
                "system",
                "Voice transcription not available — no Whisper "
                "backend installed.\n"
                "Install one of:\n"
                "  pip install openai-whisper       # reference impl\n"
                "  pip install faster-whisper       # CTranslate2 speedup\n"
                "Then re-run /voice to confirm. Live audio capture "
                "(push-to-talk) lands in a future iteration; "
                "`/voice transcribe <path>` works today against any "
                "audio file on disk.",
                severity="warning",
            )
            return
        self._post_line(
            "system",
            f"Voice backend detected: `{backend}` is importable.\n"
            "Run `/voice transcribe <path-to-audio>` to drop a "
            "transcribed body into the chat input. Live capture "
            "(push-to-talk) still pending.",
        )

    def _voice_transcribe(self, raw_path: str) -> None:
        backend = self._detect_voice_backend()
        if backend is None:
            self._post_line(
                "system",
                "Voice transcription not available — no Whisper "
                "backend installed. Run `/voice status` for "
                "install instructions.",
                severity="warning",
            )
            return
        try:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
        except Exception as exc:  # noqa: BLE001
            self._post_line(
                "system",
                f"Couldn't resolve `{raw_path}`: {exc}",
                severity="warning",
            )
            return
        if not path.exists() or not path.is_file():
            self._post_line(
                "system",
                f"Audio file `{raw_path}` not found or "
                "is not a regular file.",
                severity="warning",
            )
            return
        if path.suffix.lower() not in self._VOICE_AUDIO_EXTENSIONS:
            self._post_line(
                "system",
                f"`{path.name}` doesn't look like an audio file. "
                "Supported: "
                + ", ".join(sorted(self._VOICE_AUDIO_EXTENSIONS))
                + ".",
                severity="warning",
            )
            return
        self._post_line(
            "system",
            f"Transcribing `{path.name}` via `{backend}`… "
            "this may take a moment.",
        )
        try:
            text = self._invoke_whisper(backend, path)
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "voice transcribe failed for %s: %s",
                path, exc, exc_info=True,
            )
            self._post_line(
                "system",
                f"Transcription failed: {exc}",
                severity="error",
            )
            return
        cleaned = (text or "").strip()
        if not cleaned:
            self._post_line(
                "system",
                f"Transcription of `{path.name}` came back empty. "
                "Try a longer / clearer recording.",
                severity="warning",
            )
            return
        try:
            inp = self.query_one("#chat-input", ChatInput)
            inp.value = cleaned
            inp.cursor_position = len(inp.value)
            inp.focus()
        except Exception:
            pass
        word_count = len(cleaned.split())
        self._post_line(
            "system",
            f"Transcribed `{path.name}` → input "
            f"({word_count} word"
            f"{'s' if word_count != 1 else ''}). "
            "Edit and press Enter to submit.",
        )

    # ------------------------------------------------------------------
    # /sessions — persisted artifact-session rehydration (§3 P1)
    # ------------------------------------------------------------------

    def _handle_sessions_command(self, arg: str) -> None:
        """``/sessions`` body. Routes:

        * empty → list persisted sessions (newest first, up to 10).
        * ``"latest"`` → load the newest persisted session.
        * any other value → treat as a session id; load it.
        """
        from care.runtime.session_persistence import (
            list_sessions,
            load_session,
        )
        from care.runtime.session_artifacts import replay_into

        if not arg:
            sessions = list_sessions()
            if not sessions:
                self._post_line(
                    "system",
                    "No persisted sessions yet — artifacts you "
                    "generate this session will be saved under "
                    "`~/.cache/care/sessions/`.",
                    severity="warning",
                )
                return
            lines = ["**Persisted sessions** (newest first):"]
            for info in sessions[:10]:
                # ISO-ish summary so the user can pick a recent one.
                import time as _time
                stamp = _time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    _time.localtime(info.mtime),
                )
                lines.append(
                    f"- `{info.session_id}` — {stamp} "
                    f"({info.size_bytes} B)"
                )
            lines.append(
                "Resume with `/sessions latest` or "
                "`/sessions <id>`."
            )
            self._post_line("system", "\n".join(lines))
            return

        target_id: str
        if arg.strip() == "latest":
            sessions = list_sessions()
            if not sessions:
                self._post_line(
                    "system",
                    "No persisted sessions to resume.",
                    severity="warning",
                )
                return
            target_id = sessions[0].session_id
        else:
            target_id = arg.strip()

        try:
            artifacts = load_session(target_id)
        except (OSError, ValueError) as exc:
            self._post_line(
                "system",
                f"Couldn't load session `{target_id}`: {exc}",
                severity="error",
            )
            return
        if not artifacts:
            self._post_line(
                "system",
                f"Session `{target_id}` not found or empty.",
                severity="warning",
            )
            return
        if len(self.artifact_store):
            self._post_line(
                "system",
                f"Can't rehydrate `{target_id}` over a non-empty "
                "artifact store — run `/clear` first to start fresh.",
                severity="warning",
            )
            return
        replay_into(self.artifact_store, artifacts)
        # Manual notify so the header pill repaints.
        self._sync_artifact_pill()
        self._post_line(
            "system",
            f"Rehydrated **{len(artifacts)}** artifact(s) from "
            f"session `{target_id}`. `/artifacts` opens the list.",
        )

    # ------------------------------------------------------------------
    # /history — recent prompts + saved chains (Phase 8 P3)
    # ------------------------------------------------------------------

    _HISTORY_DEFAULT_LIMIT: int = 10
    _HISTORY_PREVIEW_MAX_CHARS: int = 60

    def _handle_history_command(self, arg: str) -> None:
        """``/history`` body. Lists the last N user prompts
        from this session AND the last N saved chains from
        Memory. ``N`` defaults to
        :data:`_HISTORY_DEFAULT_LIMIT` (10); user can override
        via ``/history 25``."""
        limit = self._HISTORY_DEFAULT_LIMIT
        if arg:
            try:
                limit = int(arg)
            except ValueError:
                self._post_line(
                    "system",
                    f"/history needs an integer limit (got {arg!r}). "
                    "Try `/history 25` to widen the list.",
                    severity="warning",
                )
                return
            limit = max(1, limit)
        prompt_lines = self._render_history_prompts(limit)
        chain_lines = self._render_history_chains(limit)
        if not prompt_lines and not chain_lines:
            self._post_line(
                "system",
                "No history yet — submit a prompt or save a "
                "chain (Production mode) to populate this list.",
                severity="warning",
            )
            return
        body_parts: list[str] = []
        if prompt_lines:
            body_parts.append("**Recent user prompts (this session):**")
            body_parts.extend(prompt_lines)
        if chain_lines:
            if body_parts:
                body_parts.append("")
            body_parts.append("**Saved chains (newest first):**")
            body_parts.extend(chain_lines)
        if prompt_lines:
            body_parts.append(
                "\nTip: `/edit N` re-edits a past prompt; "
                "`/run <chain_id>` opens a saved chain.",
            )
        self._post_line("system", "\n".join(body_parts))

    def _render_history_prompts(self, limit: int) -> list[str]:
        """Last ``limit`` user prompts as
        ``  N. <preview>`` strings, newest first."""
        history = self._collect_user_lines()
        if not history:
            return []
        # Newest first.
        history = list(reversed(history))[:limit]
        rendered: list[str] = []
        for idx, line in history:
            preview = (line.text or "").splitlines()[0] if line.text else ""
            if len(preview) > self._HISTORY_PREVIEW_MAX_CHARS:
                preview = preview[: self._HISTORY_PREVIEW_MAX_CHARS - 1].rstrip() + "…"
            rendered.append(f"  {idx}. {preview}")
        return rendered

    def _render_history_chains(self, limit: int) -> list[str]:
        """Most recent ``limit`` saved chains from Memory (when
        wired). Empty list when Memory isn't configured or the
        listing call fails — `/history` falls back to showing
        only prompts."""
        memory = getattr(self.app, "memory", None)
        if memory is None:
            return []
        listing_fn = getattr(memory, "list_entities", None)
        if listing_fn is None:
            return []
        try:
            rows = listing_fn(
                entity_type="chain", limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "/history chain listing failed (%s); falling back "
                "to prompts only", exc,
            )
            return []
        rendered: list[str] = []
        for row in (rows or [])[:limit]:
            chain_id = (
                row.get("entity_id")
                or row.get("id")
                or row.get("chain_id")
                or "?"
            )
            name = (
                row.get("name")
                or row.get("display_name")
                or ""
            )
            preview = name
            if len(preview) > self._HISTORY_PREVIEW_MAX_CHARS:
                preview = preview[: self._HISTORY_PREVIEW_MAX_CHARS - 1].rstrip() + "…"
            label = f"`{chain_id}`"
            if preview:
                label += f" — {preview}"
            rendered.append(f"  {label}")
        return rendered

    # ------------------------------------------------------------------
    # /edit — replay a past prompt with edits (Phase 8 P3)
    # ------------------------------------------------------------------

    def _collect_user_lines(self) -> list[tuple[int, ChatLine]]:
        """Return `[(turn_index, ChatLine), …]` for every user
        *prompt* line in the transcript, in posting order.
        ``turn_index`` is 1-based — matches the `/edit N`
        user-facing numbering.

        Slash-command echoes (``/help``, ``/clear``, …) are
        skipped: they show in the visible transcript so the
        user can see what they sent, but ``/edit`` / ``/history``
        only care about actual prompts that drove generation.
        """
        out: list[tuple[int, ChatLine]] = []
        turn = 0
        for line in self._lines:
            if line.role != "user":
                continue
            if line.text.startswith("/"):
                continue
            turn += 1
            out.append((turn, line))
        return out

    def _handle_edit_command(self, arg: str) -> None:
        """``/edit`` (no arg) → edit the latest user prompt.
        ``/edit <N>`` → edit the N-th user prompt (1-based).
        ``/edit list`` → show indexed past prompts.

        Pre-fills the chat input with the chosen text and
        posts a marker so the transcript carries an audit
        trail. Submitting the (edited) text fires as a
        regular task via the normal `_handle_task` path."""
        history = self._collect_user_lines()
        if arg.lower() == "list":
            self._post_edit_list(history)
            return
        if not history:
            self._post_line(
                "system",
                "No prior user prompts to edit yet.",
                severity="warning",
            )
            return
        target_index: int
        if not arg:
            target_index = history[-1][0]
        else:
            try:
                target_index = int(arg)
            except ValueError:
                self._post_line(
                    "system",
                    f"/edit needs an integer turn number "
                    f"(got {arg!r}). Try /edit list.",
                    severity="warning",
                )
                return
        match = next(
            (line for idx, line in history if idx == target_index),
            None,
        )
        if match is None:
            self._post_line(
                "system",
                f"No user prompt at turn {target_index} "
                f"(have {len(history)}).",
                severity="warning",
            )
            return
        preview = match.text.splitlines()[0] if match.text else ""
        if len(preview) > 60:
            preview = preview[:57].rstrip() + "…"
        self._post_line(
            "system",
            f"↳ editing turn {target_index} (was: {preview!r}). "
            "Edit + Enter to submit; clear the input to abort.",
        )
        try:
            inp = self.query_one("#chat-input", ChatInput)
            inp.value = match.text
            inp.cursor_position = len(inp.value)
            inp.focus()
        except Exception:
            pass

    def _post_edit_list(
        self, history: list[tuple[int, ChatLine]],
    ) -> None:
        """Render the `/edit list` block listing every past
        user prompt with its turn index + first-line preview."""
        if not history:
            self._post_line(
                "system",
                "No prior user prompts to list.",
                severity="warning",
            )
            return
        lines = [f"Past user prompts (newest last, {len(history)} total):"]
        for idx, line in history:
            preview = (line.text or "").splitlines()[0] if line.text else ""
            if len(preview) > 60:
                preview = preview[:57].rstrip() + "…"
            lines.append(f"  {idx}. {preview}")
        lines.append("\nEdit with /edit <N> (default: latest).")
        self._post_line("system", "\n".join(lines))

    def _handle_export_command(self, arg: str) -> None:
        """``/export <format> <path>`` body. Validates the
        format, picks the right renderer, writes to disk."""
        parts = arg.split(maxsplit=1)
        if not parts or not parts[0]:
            self._post_line(
                "system",
                "/export usage: /export <format> [path]\n"
                "  formats: md, mdx, json, html",
                severity="warning",
            )
            return
        fmt = parts[0].lower().lstrip(".")
        if fmt not in self._EXPORT_FORMATS:
            self._post_line(
                "system",
                f"Unknown export format `{fmt}`. "
                f"Supported: {', '.join(sorted(self._EXPORT_FORMATS))}.",
                severity="warning",
            )
            return
        if len(parts) >= 2 and parts[1].strip():
            out_path = Path(parts[1]).expanduser()
        else:
            from datetime import datetime as _dt

            stamp = _dt.now().strftime("%Y%m%d-%H%M%S")
            out_path = Path.cwd() / f"care-transcript-{stamp}.{fmt}"
        body = self._render_export(self._lines, fmt)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(body, encoding="utf-8")
        except OSError as exc:
            _log.error(
                "/export %s %s failed: %s",
                fmt, out_path, exc, exc_info=True,
            )
            self._post_line(
                "system",
                f"Couldn't write `{out_path}`: {exc}",
                severity="error",
            )
            return
        line_count = len(self._lines)
        self._post_line(
            "assistant",
            f"✓ Exported {line_count} transcript line"
            f"{'s' if line_count != 1 else ''} to `{out_path}` "
            f"as `{fmt}`.",
        )

    @classmethod
    def _render_export(
        cls, lines: list[ChatLine], fmt: str,
    ) -> str:
        """Dispatch to the right renderer based on ``fmt``.
        Pure classmethod so tests can drive renderers without
        a mounted screen."""
        if fmt == "json":
            return cls._render_export_json(lines)
        if fmt == "html":
            return cls._render_export_html(lines)
        # md + mdx share the same body — MDX is markdown plus
        # JSX, and we're not emitting JSX components.
        return cls._render_export_markdown(lines)

    @staticmethod
    def _render_export_markdown(lines: list[ChatLine]) -> str:
        """Markdown export. One bullet per line, body indented
        two spaces on continuation lines so multi-line content
        stays nested under the role marker. Matches the
        Phase 6 P2 session-log shape so external tooling can
        parse either source."""
        from datetime import datetime as _dt

        out = [f"# CARE transcript — {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
        for line in lines:
            ts = line.timestamp.strftime("%H:%M:%S")
            body = (line.text or "").replace("\n", "\n  ")
            tail = ""
            if line.role == "assistant" and line.reaction:
                tail = f"  _(reaction: {line.reaction})_"
            out.append(f"- [{ts}] **{line.role}**: {body}{tail}")
        out.append("")
        return "\n".join(out)

    @staticmethod
    def _render_export_json(lines: list[ChatLine]) -> str:
        """JSON export. Each ChatLine becomes a dict with
        every field surfaced — including provenance (Phase 8
        P2 #18) and reaction (Phase 8 P2 #15) — so external
        tooling that ingests the transcript can rebuild every
        UX surface from the raw data."""
        import json

        rows: list[dict[str, Any]] = []
        for line in lines:
            rows.append({
                "role": line.role,
                "text": line.text,
                "timestamp": line.timestamp.isoformat(),
                "mode": line.mode,
                "reaction": line.reaction,
                "provenance": line.provenance,
            })
        return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"

    @staticmethod
    def _build_pygments_pieces() -> tuple[str, Any]:
        """Phase 9 P2 — return ``(css_block, highlight_callback)``
        for the HTML export. ``css_block`` is either the
        Pygments stylesheet pinned to ``.body pre code`` or an
        empty string when Pygments isn't installed.
        ``highlight_callback`` is either a markdown-it
        ``(code, lang, attrs) -> str`` hook that emits
        Pygments-highlighted ``<span>``-soup or ``None`` so the
        markdown-it default rendering applies. Both halves
        degrade together — the export still works without
        Pygments, just without colour."""
        try:
            from pygments import highlight as _pyg_highlight
            from pygments.formatters import HtmlFormatter
            from pygments.lexers import get_lexer_by_name
            from pygments.util import ClassNotFound
        except Exception:
            return "", None

        formatter = HtmlFormatter(nowrap=True, style="friendly")

        def _highlight(code: str, lang: str, _attrs: str) -> str:
            if not lang:
                return ""
            try:
                lexer = get_lexer_by_name(lang)
            except ClassNotFound:
                return ""
            try:
                inner = _pyg_highlight(code, lexer, formatter)
            except Exception:
                return ""
            # markdown-it expects the FULL <pre><code>…</code></pre>
            # wrapper when the highlight callback returns a
            # non-empty string. Wrap manually so the lang
            # class lands on <code> for downstream styling.
            from html import escape

            safe_lang = escape(lang)
            return (
                f'<pre><code class="language-{safe_lang}">'
                f"{inner}"
                "</code></pre>"
            )

        css = (
            HtmlFormatter(style="friendly").get_style_defs(".body pre code")
            + "\n"
        )
        return css, _highlight

    @classmethod
    def _render_export_html(cls, lines: list[ChatLine]) -> str:
        """Standalone HTML export. Carries inline CSS so the
        file opens cleanly in a browser without external
        assets. Assistant / system bodies render through
        `markdown-it-py` so fenced code blocks, headings,
        links etc. become real HTML. User / tool bodies
        escape to plain text (the user's exact characters are
        sacred). Falls back to escaped plain text everywhere
        when the markdown lib isn't importable so the export
        never crashes.

        Phase 9 P2 — fenced code blocks are syntax-highlighted
        via Pygments when both `markdown-it-py` AND `pygments`
        are importable. The relevant Pygments CSS is injected
        inline so the export stays a single file. When
        Pygments is missing, markdown-it still emits raw
        `<pre><code class="language-X">` so the export still
        works, just without colour."""
        from datetime import datetime as _dt
        from html import escape

        pygments_css, highlight_cb = cls._build_pygments_pieces()

        try:
            from markdown_it import MarkdownIt

            md_options: dict[str, Any] = {
                "breaks": True,
                "html": False,
            }
            if highlight_cb is not None:
                md_options["highlight"] = highlight_cb
            md = MarkdownIt("commonmark", md_options)
        except Exception:
            md = None

        body_rows: list[str] = []
        for line in lines:
            ts = line.timestamp.strftime("%H:%M:%S")
            css_class = f"line line-{line.role}"
            if line.role == "assistant" and line.reaction:
                css_class += f" reaction-{line.reaction}"
            body_html: str
            if md is not None and line.role in {"assistant", "system"}:
                try:
                    body_html = md.render(line.text or "")
                except Exception:
                    body_html = f"<pre>{escape(line.text or '')}</pre>"
            else:
                body_html = f"<pre>{escape(line.text or '')}</pre>"
            reaction_html = ""
            if line.role == "assistant" and line.reaction:
                marker = cls._REACTION_MARKERS.get(line.reaction, "")
                if marker:
                    reaction_html = (
                        f' <span class="reaction">{marker}</span>'
                    )
            body_rows.append(
                f'<div class="{css_class}">'
                f'<span class="meta">'
                f'<span class="ts">[{ts}]</span> '
                f'<span class="role">{escape(line.role)}</span>'
                f'{reaction_html}'
                f'</span>'
                f'<div class="body">{body_html}</div>'
                f'</div>'
            )
        title = (
            f"CARE transcript — "
            f"{_dt.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return (
            "<!doctype html>\n"
            '<html lang="en"><head><meta charset="utf-8">'
            f"<title>{escape(title)}</title>"
            "<style>\n"
            "body { font-family: system-ui, sans-serif; max-width: 900px; "
            "margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }\n"
            "h1 { font-size: 1.4rem; border-bottom: 1px solid #ccc; "
            "padding-bottom: .3rem; }\n"
            ".line { padding: .6rem 0; border-bottom: 1px solid #eee; }\n"
            ".meta { display: block; font-size: .85rem; color: #666; "
            "margin-bottom: .3rem; }\n"
            ".meta .ts { color: #999; }\n"
            ".meta .role { font-weight: bold; text-transform: uppercase; "
            "letter-spacing: .05em; }\n"
            ".line-user .meta .role { color: #d97706; }\n"
            ".line-assistant .meta .role { color: #2563eb; }\n"
            ".line-system .meta .role { color: #6b7280; }\n"
            ".line-tool .meta .role { color: #d97706; }\n"
            ".body pre { background: #f3f4f6; padding: .8rem; "
            "border-radius: .3rem; overflow-x: auto; }\n"
            ".body code { background: #f3f4f6; padding: .1rem .3rem; "
            "border-radius: .2rem; }\n"
            ".body pre code { background: transparent; padding: 0; }\n"
            ".reaction { margin-left: .4rem; }\n"
            + pygments_css
            + "</style></head><body>\n"
            f"<h1>{escape(title)}</h1>\n"
            + "\n".join(body_rows)
            + "\n</body></html>\n"
        )


# ---------------------------------------------------------------------------
# Slash-command registry
# ---------------------------------------------------------------------------


_COMMAND_HANDLERS: dict[str, CommandHandler] = {}


def _register(name: str) -> Callable[[CommandHandler], CommandHandler]:
    def deco(fn: CommandHandler) -> CommandHandler:
        _COMMAND_HANDLERS[name] = fn
        return fn

    return deco


@_register("help")
def _cmd_help(screen: ChatScreen, _arg: str) -> None:
    screen._post_line("system", screen._render_help_text())


@_register("library")
def _cmd_library(screen: ChatScreen, _arg: str) -> None:
    from care.screens.library import LibraryScreen

    screen.app.push_screen(LibraryScreen())


def _format_uptime(deployed_at_iso: str) -> str:
    """ISO deploy timestamp → compact "2h 5m" / "45s" age; '' when unparsable."""
    try:
        from datetime import datetime, timezone

        started = datetime.fromisoformat(deployed_at_iso)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        seconds = int(
            (datetime.now(timezone.utc) - started).total_seconds()
        )
    except Exception:  # noqa: BLE001 — cosmetic field only
        return ""
    if seconds < 0:
        return ""
    if seconds < 60:
        return f"{seconds}s"
    minutes, hours = (seconds // 60) % 60, seconds // 3600
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


_DEPLOYMENTS_ACTIONS = {"list", "undeploy", "reload", "docs"}


@_register("deployments")
def _cmd_deployments(screen: ChatScreen, arg: str) -> None:
    """``/deployments``                 → list hub deployments.
    ``/deployments undeploy <name>``  → remove an agent from the hub.
    ``/deployments reload <name>``    → re-fetch + preflight + swap now.
    ``/deployments docs <name>``      → open the agent's Swagger page."""
    tokens = (arg or "").split()
    action = tokens[0].lower() if tokens else "list"
    name = tokens[1] if len(tokens) > 1 else None
    if action not in _DEPLOYMENTS_ACTIONS or (action != "list" and not name):
        screen._post_line(
            "system",
            "Usage: /deployments [undeploy|reload|docs <name>] — bare "
            "/deployments lists everything on the hub.",
            severity="warning",
        )
        return
    screen.run_worker(
        screen._run_deployments(action, name),
        name="chat_deployments",
        group="generate",
        exclusive=True,
        exit_on_error=False,
    )


def _format_metrics_row(name: str, metrics: dict[str, Any] | None) -> str:
    """One chat line of an agent's usage + cost (D4)."""
    if not metrics:
        return f"● {name} — metrics unavailable"
    parts = [f"runs {metrics.get('run_count', 0)}", f"tokens {metrics.get('total_tokens', 0)}"]
    cost = metrics.get("total_cost_usd")
    if cost is None:
        parts.append("(unpriced)")
    else:
        parts.append(f"${float(cost):.4f}")
        budget = metrics.get("budget_usd")
        if budget is not None:
            remaining = metrics.get("remaining_usd")
            flag = " ⚠ OVER BUDGET" if metrics.get("over_budget") else ""
            if remaining is not None:
                parts.append(f"budget ${float(budget):.2f} (left ${float(remaining):.4f}){flag}")
            else:
                parts.append(f"budget ${float(budget):.2f}{flag}")
    return f"● {name} — " + " · ".join(parts)


@_register("metrics")
def _cmd_metrics(screen: ChatScreen, arg: str) -> None:
    """``/metrics`` — per-agent usage + USD cost from the hub (D4). Read-only;
    never autostarts the hub."""
    screen.run_worker(
        screen._run_metrics(),
        name="chat_metrics",
        group="generate",
        exclusive=True,
        exit_on_error=False,
    )


def _parse_promote_args(raw: str) -> tuple[str, str, str, bool]:
    """``/promote <ref> [--from latest] [--to stable] [--force]`` →
    (ref, from_channel, to_channel, force)."""
    tokens = (raw or "").split()
    from_channel, to_channel, force = "latest", "stable", False
    rest: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--from" and index + 1 < len(tokens):
            from_channel = tokens[index + 1]
            index += 2
        elif token == "--to" and index + 1 < len(tokens):
            to_channel = tokens[index + 1]
            index += 2
        elif token == "--force":
            force = True
            index += 1
        else:
            rest.append(token)
            index += 1
    return " ".join(rest).strip(), from_channel, to_channel, force


@_register("promote")
def _cmd_promote(screen: ChatScreen, arg: str) -> None:
    """``/promote <id|name> [--from latest] [--to stable] [--force]`` —
    gated release: artifact check → mandatory baseline run → eval-vs-baseline
    (when scores exist) → channel promote. ``--force`` skips the gate."""
    text = (arg or "").strip()
    if not text:
        screen._post_line(
            "system",
            "Usage: /promote <chain-id|name> [--from latest] [--to stable] "
            "[--force] — runs the promotion gate, then promotes the channel.",
            severity="warning",
        )
        return
    screen.run_worker(
        screen._run_promote(text),
        name="chat_promote",
        group="generate",
        exclusive=True,
        exit_on_error=False,
    )


def _parse_versions_args(raw: str) -> tuple[str, tuple[str, str] | None]:
    """``/versions <ref>`` or ``/versions <ref> diff <vA> <vB>`` →
    (ref, (from, to) | None)."""
    tokens = (raw or "").split()
    if len(tokens) >= 4 and tokens[-3] == "diff":
        ref = " ".join(tokens[:-3]).strip()
        return ref, (tokens[-2], tokens[-1])
    return " ".join(tokens).strip(), None


def _format_version_row(version: Any, channel_of: dict[str, list[str]]) -> str:
    number = getattr(version, "version_number", "?")
    vid = str(getattr(version, "version_id", "") or "")
    parts = [f"v{number}", vid[:12]]
    created = getattr(version, "created_at", None)
    if created is not None:
        parts.append(str(created)[:10])
    evo = getattr(version, "evolution_meta", None) or {}
    score = evo.get("fitness_score") if isinstance(evo, dict) else None
    if score is not None:
        parts.append(f"fitness {score}")
    summary = getattr(version, "change_summary", None)
    if summary:
        parts.append(str(summary)[:50])
    row = "● " + " · ".join(parts)
    channels = channel_of.get(vid)
    if channels:
        row += "  ← " + ", ".join(channels)
    return row


@_register("versions")
def _cmd_versions(screen: ChatScreen, arg: str) -> None:
    """``/versions <id|name>`` — list a chain's version history (which the
    latest/stable channels point at, eval scores). ``/versions <ref> diff
    <vA> <vB>`` shows the JSON patch. Roll back with /rollback <id> --to <vid>."""
    text = (arg or "").strip()
    if not text:
        screen._post_line(
            "system",
            "Usage: /versions <chain-id|name> [diff <vA> <vB>] — version history; "
            "roll back with /rollback <id> --to <version-id>.",
            severity="warning",
        )
        return
    screen.run_worker(
        screen._run_versions(text),
        name="chat_versions",
        group="generate",
        exclusive=True,
        exit_on_error=False,
    )


def _parse_rollback_args(raw: str) -> tuple[str, str, str | None]:
    """``/rollback <ref> [--channel stable] [--to <version-id>]`` →
    (ref, channel, to_version). Default channel ``stable`` — rollbacks target
    the released pointer deployments follow."""
    tokens = (raw or "").split()
    channel = "stable"
    to_version: str | None = None
    rest: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--channel" and index + 1 < len(tokens):
            channel = tokens[index + 1]
            index += 2
        elif token == "--to" and index + 1 < len(tokens):
            to_version = tokens[index + 1]
            index += 2
        else:
            rest.append(token)
            index += 1
    return " ".join(rest).strip(), channel, to_version


@_register("rollback")
def _cmd_rollback(screen: ChatScreen, arg: str) -> None:
    """``/rollback <id|name> [--channel stable] [--to <version-id>]`` — repoint
    the channel at the previous (or a specific) version. Attached agents
    hot-reload; nothing is deleted (pin, not revert)."""
    text = (arg or "").strip()
    if not text:
        screen._post_line(
            "system",
            "Usage: /rollback <chain-id|name> [--channel stable] "
            "[--to <version-id>] — repoints the channel one version back.",
            severity="warning",
        )
        return
    screen.run_worker(
        screen._run_rollback(text),
        name="chat_rollback",
        group="generate",
        exclusive=True,
        exit_on_error=False,
    )


def _parse_deploy_args(raw: str) -> tuple[str, str, str | None]:
    """``/deploy <ref> [--channel X] [--name Y]`` → (ref, channel, name).

    ``ref`` is a chain entity id or a name-search query (may contain spaces).
    Channel defaults to ``stable`` — production deploys follow the released
    channel, not the dev tip.
    """
    tokens = (raw or "").split()
    channel = "stable"
    name: str | None = None
    rest: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--channel" and index + 1 < len(tokens):
            channel = tokens[index + 1]
            index += 2
        elif token == "--name" and index + 1 < len(tokens):
            name = tokens[index + 1]
            index += 2
        else:
            rest.append(token)
            index += 1
    return " ".join(rest).strip(), channel, name


def _slugify_agent_name(text: str) -> str:
    """Display name → url-safe agent name (the hub mounts /agents/<name>)."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", (text or "").lower()).strip("-._")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:63] or "agent"


@_register("deploy")
def _cmd_deploy(screen: ChatScreen, arg: str) -> None:
    """``/deploy <id|name> [--channel stable] [--name x]`` → ship a saved
    chain to the agent hub as an HTTP agent with its own Swagger
    (``/agents/<name>/docs``). The deploy gate (loadability + template tool
    set + lint) runs first; the hub autostarts when allowed."""
    text = (arg or "").strip()
    if not text:
        screen._post_line(
            "system",
            "Usage: /deploy <chain-id|name> [--channel stable] [--name x] — "
            "deploys the chain as an HTTP agent on the hub.",
            severity="warning",
        )
        return
    screen.run_worker(
        screen._run_deploy(text),
        name="chat_deploy",
        group="generate",
        exclusive=True,
        exit_on_error=False,
    )


@_register("revise")
def _cmd_revise(screen: ChatScreen, arg: str) -> None:
    """``/revise <id> <instruction>``  → AI-edit a saved chain by id.
    ``/revise <instruction>``         → resolve the chain from your words
    via memory search (disambiguates on ties).

    MAGE plans a minimal, targeted edit, previews the diff, and — on
    confirm — saves it as a NEW VERSION of the chain (history kept)."""
    text = (arg or "").strip()
    if not text:
        screen._post_line(
            "system",
            "Usage: /revise <id> <instruction>  (or  /revise <instruction>  to "
            "resolve the chain from your words).",
            severity="warning",
        )
        return
    screen.run_worker(
        screen._run_edit(text),
        name="chat_edit",
        group="generate",
        exclusive=True,
        exit_on_error=False,
    )


@_register("marketplace")
def _cmd_marketplace(screen: ChatScreen, arg: str) -> None:
    """``/marketplace [query]`` — open the community
    `agent_skill` marketplace (§6 P1).

    Reads `app.memory.client` for the backend search +
    install flow; an unconfigured Memory facade surfaces a
    friendly warning on the screen itself rather than crashing
    the command. Optional ``[query]`` pre-fills the search
    input so users who already know what they're looking for
    can `/marketplace summarise pdf` and land on results
    immediately. Distinct from `/library` which lists the
    user's own saved chains."""
    from care.screens.marketplace import MarketplaceScreen

    memory = getattr(screen.app, "memory", None)
    target = (
        getattr(memory, "client", None) or memory
        if memory is not None else None
    )
    try:
        screen.app.push_screen(
            MarketplaceScreen(
                memory=target,
                initial_query=(arg or "").strip(),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        screen._post_line(
            "system",
            f"Couldn't open marketplace: {exc}",
            severity="error",
        )


@_register("profile")
def _cmd_profile(screen: ChatScreen, _arg: str) -> None:
    """``/profile`` — list credential profiles (§6 P2).

    Audits `~/.config/care/profiles/*.toml` + surfaces the
    currently-active selection (`CARE_PROFILE` env var). Useful
    for dev / prod cred splits without re-running onboarding.
    Switching requires `export CARE_PROFILE=<name>` + CARE
    restart — the screen surfaces the exact command."""
    try:
        from care.screens.profile import ProfileScreen

        screen.app.push_screen(ProfileScreen())
    except Exception as exc:  # noqa: BLE001
        screen._post_line(
            "system",
            f"Couldn't open profile screen: {exc}",
            severity="error",
        )


@_register("logs")
def _cmd_logs(screen: ChatScreen, _arg: str) -> None:
    """``/logs`` — open the in-app log viewer (§6 P2).

    Tails the active app log file (`CARE_LOG_FILE` env or the
    attached `care-app-file` handler). Distinct from the
    inline `/log [level] [module]` which prints filtered
    records into the chat transcript; `/logs` pushes a
    dedicated screen for sustained scrolling + level
    cycling."""
    try:
        from care.screens.logs import LogsScreen

        screen.app.push_screen(LogsScreen())
    except Exception as exc:  # noqa: BLE001
        screen._post_line(
            "system",
            f"Couldn't open logs screen: {exc}",
            severity="error",
        )


@_register("cost")
def _cmd_cost(screen: ChatScreen, _arg: str) -> None:
    """``/cost`` — open the token + spend dashboard (§6 P2).

    Aggregates every recorded chain run under
    `~/.cache/care/runs/<YYYY-MM-DD>.jsonl` into per-provider,
    per-chain, and per-mode rollups + an overall header
    (runs / success rate / tokens / cost / wall time)."""
    try:
        from care.screens.cost import CostDashboardScreen

        screen.app.push_screen(CostDashboardScreen())
    except Exception as exc:  # noqa: BLE001
        screen._post_line(
            "system",
            f"Couldn't open cost dashboard: {exc}",
            severity="error",
        )


@_register("sandbox")
def _cmd_sandbox(screen: ChatScreen, _arg: str) -> None:
    """``/sandbox`` — open the AgentSkill trust audit screen
    (§6 P1).

    Lists every skill the user has explicitly approved
    (SHA-pinned via the persistent `SkillTrustStore`) with
    revoke + refresh affordances. Independent of Memory."""
    try:
        from care.screens.sandbox_trust import SandboxTrustScreen

        screen.app.push_screen(SandboxTrustScreen())
    except Exception as exc:  # noqa: BLE001
        screen._post_line(
            "system",
            f"Couldn't open sandbox trust screen: {exc}",
            severity="error",
        )


@_register("runs")
def _cmd_runs(screen: ChatScreen, _arg: str) -> None:
    """``/runs`` — open the local run-history viewer (§6 P1).

    Reads `~/.cache/care/runs/<YYYY-MM-DD>.jsonl` (one row per
    recorded chain execution) and surfaces the rows
    newest-first in a DataTable. Independent of Memory —
    works without a configured Memory facade so users can
    audit local executions on a fresh install."""
    try:
        from care.screens.runs import RunsScreen

        screen.app.push_screen(RunsScreen())
    except Exception as exc:  # noqa: BLE001
        screen._post_line(
            "system",
            f"Couldn't open runs screen: {exc}",
            severity="error",
        )


@_register("artifacts")
def _cmd_artifacts(screen: ChatScreen, _arg: str) -> None:
    """Push :class:`care.screens.artifacts.ArtifactsScreen` for
    the current chat's session-artifacts store (§3 P0).

    The store is owned by ChatScreen and seeded by every
    successful MAGE generation (§3 P0 stash hook); the screen
    reads it live + subscribes to mutations.
    """
    from care.screens.artifacts import ArtifactsScreen

    try:
        screen.app.push_screen(ArtifactsScreen(screen.artifact_store))
    except Exception as exc:  # noqa: BLE001
        screen._post_line(
            "system",
            f"Couldn't open artifacts screen: {exc}",
            severity="error",
        )


@_register("memory")
def _cmd_memory(screen: ChatScreen, arg: str) -> None:
    """View / edit what CARE remembers (CARE.md + CARL long-term memory).

    ``/memory`` or ``/memory show`` — render CARE.md + LTM keys/values.
    ``/memory forget <key>`` — delete one LTM key.
    ``/memory edit`` — open CARE.md in your editor (OS-aware fallback).
    """
    parts = (arg or "").strip().split(maxsplit=1)
    sub = (parts[0].lower() if parts else "show") or "show"
    rest = parts[1].strip() if len(parts) > 1 else ""
    if sub == "show":
        screen._memory_show()
    elif sub == "forget":
        screen._memory_forget(rest)
    elif sub == "edit":
        screen._memory_edit()
    else:
        screen._post_line(
            "system",
            f"Unknown `/memory {sub}` — use: show | forget <key> | edit",
            severity="warning",
        )


@_register("remember")
def _cmd_remember(screen: ChatScreen, arg: str) -> None:
    """Remember a note in long-term memory (LLM-adapted; supersedes stale
    facts). Same as a ``#…`` message — ``/remember I prefer concise answers``.
    """
    screen.run_worker(
        screen._remember_content(arg),
        name="chat_remember",
        group="memory",
        exit_on_error=False,
    )


@_register("status")
def _cmd_status(screen: ChatScreen, _arg: str) -> None:
    """`/status` — print the live MAGE / Memory / Platform
    health report (§1 P0 health-check banner).

    Spawns a worker that runs the same `aggregate_status_bar`
    helper the bottom strip uses, then posts the formatted
    report as a system line so the user can scan the dots +
    error details without leaving the chat.
    """
    screen.run_worker(
        _status_worker(screen),
        name="chat_status_probe",
        group="status",
        exclusive=True,
        exit_on_error=False,
    )


async def _status_worker(screen: ChatScreen) -> None:
    """Worker body for `/status` — runs the SAME health probes as
    ``care doctor`` (``care.first_run.run_all_probes``) so the TUI and CLI
    reports never diverge, and includes the deep MAGE round-trip so an
    expired token shows red here too. The app's already-built
    ``memory`` / ``platform`` facades are injected so we don't spin up
    duplicate clients."""
    from care.first_run import run_all_probes

    app = screen.app
    config = getattr(app, "config", None)
    if config is None:
        screen._post_line(
            "system",
            "Status check needs an `app.config`; nothing to "
            "probe.",
            severity="warning",
        )
        return
    memory = getattr(app, "memory", None)
    platform = getattr(app, "platform", None)
    try:
        report = await run_all_probes(
            config,
            memory_factory=(lambda _c: memory) if memory is not None else None,
            platform_factory=(
                (lambda _c: platform) if platform is not None else None
            ),
            deep=True,
        )
    except Exception as exc:  # noqa: BLE001
        screen._post_line(
            "system",
            f"Status probe failed: {exc}", severity="error",
        )
        return
    screen._post_line(
        "system",
        "Health report (same probes as `care doctor`):\n"
        + report.format_text(),
        severity="info",
    )


@_register("settings")
def _cmd_settings(screen: ChatScreen, _arg: str) -> None:
    from care.screens.settings import SettingsScreen

    cfg = getattr(screen.app, "config", None)
    if cfg is None:
        from care.config import CareConfig

        cfg = CareConfig.load()
    screen.app.push_screen(SettingsScreen(cfg))


@_register("clear")
def _cmd_clear(screen: ChatScreen, _arg: str) -> None:
    screen.action_clear_transcript()


@_register("new")
def _cmd_new(screen: ChatScreen, _arg: str) -> None:
    """Drop the running Ad-Hoc conversation context so the next
    prompt starts from a blank slate. The visible transcript
    stays put — use `/clear` if you also want to wipe the
    on-screen history. Production runs ignore Ad-Hoc context
    anyway, so this is a no-op there beyond the confirmation
    line."""
    # Both contexts count: Ad-Hoc carries conversational
    # history; Production carries a "follow-ups revise this
    # chain" pointer. Either being live makes /new meaningful.
    had_history = bool(
        screen._interactive_history,
    ) or screen._production_chain_id is not None
    screen._reset_interactive_history()
    if had_history:
        # Arm the divider so the next user prompt visually
        # marks where the new conversation begins. We DON'T
        # arm it on the no-history branch — there's no prior
        # turn to separate from, so the divider would just
        # float above an empty backdrop.
        screen._new_conversation_pending = True
        screen._post_line(
            "system",
            t("chat.newConversation"),
        )
    else:
        screen._post_line(
            "system",
            t("chat.noContextToClear"),
        )


@_register("quit")
def _cmd_quit(screen: ChatScreen, _arg: str) -> None:
    screen.app.exit()


@_register("resume")
def _cmd_resume(screen: ChatScreen, arg: str) -> None:
    """``/resume``                 → list recent sessions.
    ``/resume latest``          → rehydrate the newest session.
    ``/resume <filename>``      → rehydrate a specific session.

    Phase 8 P1 #12 — reads the Production-mode session-log
    markdown sidecar (Phase 6 P2) and replays the
    user / assistant / system lines into the current
    transcript so the user can continue thinking from where
    they stopped. Pure transcript-restoration — no chains are
    re-executed.
    """
    screen._resume_session(arg or "")


@_register("theme")
def _cmd_theme(screen: ChatScreen, arg: str) -> None:
    """``/theme``                → list available themes + show current.
    ``/theme <name>``         → switch to the named theme.

    Phase 8 P2 #19 — wraps Textual's built-in theme registry
    (``app.available_themes`` / ``app.theme``) so users can
    cycle through dark / light / gruvbox / nord / dracula /
    tokyo-night / etc. without leaving the chat surface.
    Validates the name against the registry first so a typo
    surfaces a clean warning instead of a runtime exception.
    """
    screen._handle_theme_command((arg or "").strip())


@_register("multi")
def _cmd_multi(screen: ChatScreen, arg: str) -> None:
    """``/multi`` — open a multi-line task composer.

    Phase 8 P0 #1 — pushes
    :class:`care.screens.multi_compose.MultiLineComposer` onto
    the screen stack. The composer's TextArea accepts newlines
    naturally; Ctrl+J / Ctrl+S submits; Esc cancels. On submit,
    the typed text rides into ``_handle_task`` like any other
    chat submission (including @-file ref resolution from
    Phase 8 P1 #6). Optional ``arg`` pre-fills the composer so
    a user who started typing a prompt then realised they want
    multi-line can `/multi <partial>` and continue editing."""
    from care.screens.multi_compose import MultiLineComposer

    def _on_dismiss(result):
        # `result` is the submitted text (`str`) or `None` on
        # cancel. Whitespace-only submissions also come back
        # as None (the composer normalises that on its side).
        if result is None:
            screen._post_line(
                "system",
                "Multi-line composer cancelled.",
            )
            return
        screen._handle_task(result)

    initial = (arg or "").strip()
    screen.app.push_screen(
        MultiLineComposer(initial_text=initial),
        _on_dismiss,
    )


@_register("log")
def _cmd_log(screen: ChatScreen, arg: str) -> None:
    """``/log``                       → show recent log records.
    ``/log warning``              → only WARNING+ records.
    ``/log info care.chat``       → INFO+ records from care.chat.

    Phase 8 P1 #13 — tails the active rolling log file
    (driven by `CARE_LOG_FILE`) and posts the filtered
    records as a system line so users can debug from inside
    the chat surface. Severity filter matches the standard
    logging hierarchy (DEBUG < INFO < WARNING < ERROR <
    CRITICAL); module filter is a case-insensitive substring
    against the logger name."""
    screen._handle_log_command((arg or "").strip())


@_register("voice")
def _cmd_voice(screen: ChatScreen, arg: str) -> None:
    """``/voice`` — show voice-transcription status.

    Phase 8 P3 — local Whisper integration is gated behind
    optional deps (``pip install care[voice]``) so the bare
    install stays light. This command surfaces the
    current status: whether the ``whisper`` package is
    importable, and a friendly hint about the install path
    when it isn't. Real audio capture + transcription land
    in a follow-up iteration once microphone capture is
    wired."""
    screen._handle_voice_command((arg or "").strip())


@_register("subagents")
def _cmd_subagents(screen: ChatScreen, arg: str) -> None:
    """``/subagents`` — render captured CARL StepEvent
    activity as a tree grouped by step. ``/subagents clear``
    flushes the buffer.

    Phase 8 P3 — when a CARL chain emits ``StepEvent`` calls
    (debate round / parallel sampling / supervisor routing /
    tool invocations), the events accumulate in a bounded
    in-memory log. This command makes the log inspectable
    inline so users can see what each step's sub-agents did
    without leaving chat."""
    screen._handle_subagents_command((arg or "").strip())


@_register("sessions")
def _cmd_sessions(screen: ChatScreen, arg: str) -> None:
    """``/sessions``           → list persisted artifact sessions.
    ``/sessions latest``    → rehydrate the newest persisted session
                              into the current artifact store.
    ``/sessions <id>``      → rehydrate a specific session by id.

    §3 P1 — Artifacts are persisted to
    ``~/.cache/care/sessions/<id>.jsonl`` on every mutation so a
    crash / quit doesn't drop them. This command is the recovery
    surface: it surfaces known sessions and rehydrates one into
    the current store so `/artifacts` shows the old artifacts +
    save-all can retry."""
    screen._handle_sessions_command((arg or "").strip())


@_register("history")
def _cmd_history(screen: ChatScreen, arg: str) -> None:
    """``/history`` — list recent user prompts (this session)
    AND saved chains from Memory (when wired). ``/history N``
    caps both lists at N entries (default 10 each).

    Phase 8 P3 — Claude-Code-like history panel via the
    slash surface. The full collapsible sidebar UI is a
    layout-refactor follow-up; this command gives the user
    immediate visibility into past work without touching the
    `/library` screen."""
    screen._handle_history_command((arg or "").strip())


@_register("edit")
def _cmd_edit(screen: ChatScreen, arg: str) -> None:
    """``/edit``              → edit the most recent user prompt.
    ``/edit <N>``         → edit the N-th user prompt (1-based).
    ``/edit list``        → show indexed past user prompts.

    Phase 8 P3 — pre-fills the chat input with the chosen
    user submission so the user can tweak it and re-submit
    as a fresh task. Posts a `↳ editing turn N (was: …)`
    marker so the transcript carries an audit trail of
    which past prompt is being branched from."""
    screen._handle_edit_command((arg or "").strip())


@_register("imgpreview")
def _cmd_imgpreview(screen: ChatScreen, arg: str) -> None:
    """``/imgpreview status``        → detect terminal-graphics
    protocol. ``/imgpreview <path>``         → build the
    protocol-specific escape sequence for an image (Kitty /
    iTerm2 / WezTerm).

    Phase 9 P3 — terminal-graphics support detection +
    sequence builders. Inline rendering in Textual still
    needs a renderer hook beyond this iteration."""
    screen._handle_imgpreview_command((arg or "").strip())


@_register("branch")
def _cmd_branch(screen: ChatScreen, arg: str) -> None:
    """``/branch [name]``        → save the current transcript
    as a checkpoint. ``/branch list``         → show saved
    checkpoints. ``/branch switch <id>``  → rehydrate a
    checkpoint (replaces the live transcript).
    ``/branch delete <id>``  → remove a checkpoint.

    Phase 9 P3 — contained variant of the per-cell
    branching spec: save / restore experimental forks
    instead of dual-pane sibling threads."""
    screen._handle_branch_command((arg or "").strip())


@_register("blocks")
def _cmd_blocks(screen: ChatScreen, arg: str) -> None:
    """``/blocks``                → list every fenced code block.
    ``/blocks copy <N>``         → copy the N-th block.
    ``/blocks save <N> <path>``  → write the N-th to disk.

    Phase 9 P2 — extends the Phase 8 P2 #17 ``Ctrl+B``
    affordance (first block only) so users can act on any
    fenced code block in the transcript by stable index."""
    screen._handle_blocks_command((arg or "").strip())


@_register("export")
def _cmd_export(screen: ChatScreen, arg: str) -> None:
    """``/export <format> <path>`` — write the current
    transcript to ``path`` in the chosen format. Supported
    formats: ``md`` (Markdown), ``mdx`` (Markdown for MDX
    rendering), ``json`` (structured per-line data), ``html``
    (rendered standalone HTML).

    Phase 8 P3 — complements the existing Phase 6 P2 session-log
    sidecar (which is append-only and Production-only) with a
    deliberate user-driven export that works in any mode and
    supports multiple downstream formats. The output is
    self-contained: HTML carries inline CSS so it opens
    standalone in a browser without external assets; JSON
    carries every `ChatLine` field including provenance +
    reaction so external tooling can ingest the full
    transcript."""
    screen._handle_export_command((arg or "").strip())


@_register("tour")
def _cmd_tour(screen: ChatScreen, arg: str) -> None:
    """Start the 5-step guided tour. Ignores any argument so
    `/tour anything` still kicks off the walk — the slash
    command is a verb, not a query."""
    if arg.strip():
        screen._post_line(
            "system",
            "/tour ignores arguments — starting the walkthrough.",
        )
    screen._start_tour()


@_register("forget")
def _cmd_forget(screen: ChatScreen, arg: str) -> None:
    """``/forget <chain_id>`` — preview what would be deleted.
    ``/forget <chain_id> --force`` — soft-delete the chain AND
    every dataset card tagged ``dataset-entry:<chain_id>``.

    Mirrors Memory's soft-delete semantics (sets ``deleted_at``)
    so the chain stays recoverable via Memory's trash —
    `/forget` is the user-facing escape hatch for the privacy
    bullet in cross-cutting concerns. Two-step UX (preview →
    `--force`) keeps a single fat-finger from nuking a chain
    plus its whole dataset history.
    """
    parts = (arg or "").strip().split()
    if not parts:
        screen._post_line(
            "system",
            "/forget needs a chain id. Usage:\n"
            "  /forget <chain_id>          Preview what would be deleted\n"
            "  /forget <chain_id> --force  Actually delete",
            severity="warning",
        )
        return
    chain_id = parts[0]
    force = any(p.lower() == "--force" for p in parts[1:])
    memory = getattr(screen.app, "memory", None)
    if memory is None:
        screen._post_line(
            "system",
            "Memory facade isn't wired — can't /forget. "
            "Set CARE_MEMORY__BASE_URL.",
            severity="warning",
        )
        return
    screen.run_worker(
        screen._forget_chain_and_dataset(chain_id, force=force),
        name="forget",
        group="forget",
        exclusive=True,
        exit_on_error=False,
    )


@_register("run")
def _cmd_run(screen: ChatScreen, arg: str) -> None:
    arg = arg.strip()
    if not arg:
        screen._post_line(
            "system",
            "/run needs a chain id (try /library to find one).",
        )
        return
    try:
        from care.screens.inspection import InspectionScreen

        screen.app.push_screen(InspectionScreen(arg))
    except Exception as exc:  # noqa: BLE001
        screen._post_line("system", f"/run {arg} failed: {exc}")


@_register("upload")
def _cmd_upload(screen: ChatScreen, arg: str) -> None:
    """``/upload <chain_id>`` — fetch a saved chain from Memory
    and POST its JSON payload to the URL configured under
    ``[upload]`` / ``CARE_UPLOAD__URL``.

    The target service shape is intentionally open — different
    deployments hand chains to different downstream consumers
    (a CARE-companion bot, an internal "agent shelf", an
    eval pipeline). The slash command surfaces a friendly
    "config missing" line when ``upload.url`` is empty so
    first-time users see exactly what env var to set.
    """
    arg = arg.strip()
    if not arg:
        screen._post_line(
            "system",
            "/upload needs a chain id "
            "(try /library to find one).",
            severity="warning",
        )
        return
    cfg = getattr(screen.app, "config", None)
    upload_cfg = getattr(cfg, "upload", None) if cfg is not None else None
    url = (getattr(upload_cfg, "url", "") or "").strip()
    if not url:
        screen._post_line(
            "system",
            "Upload target not configured. Set "
            "CARE_UPLOAD__URL (and optionally "
            "CARE_UPLOAD__API_KEY) to enable /upload.",
            severity="warning",
        )
        return
    memory = getattr(screen.app, "memory", None)
    if memory is None:
        screen._post_line(
            "system",
            "Can't upload: Memory not configured. "
            "Set CARE_MEMORY__BASE_URL.",
            severity="warning",
        )
        return
    try:
        chain_payload = memory.get_chain(arg)
    except Exception as exc:  # noqa: BLE001
        screen._post_line(
            "system",
            f"Couldn't fetch chain `{arg}`: {exc}",
            severity="error",
        )
        return
    if chain_payload is None:
        screen._post_line(
            "system",
            f"No saved chain with id `{arg}` "
            "(try /library to list saved chains).",
            severity="warning",
        )
        return

    screen._post_line("tool", f"Uploading chain `{arg}` to {url}…")
    try:
        status, body_preview = _post_chain_to_upload(
            url=url,
            api_key=(getattr(upload_cfg, "api_key", None) or "") or None,
            auth_header=(
                getattr(upload_cfg, "auth_header", "Authorization")
                or "Authorization"
            ),
            timeout=float(getattr(upload_cfg, "timeout", 30.0) or 30.0),
            chain_id=arg,
            payload=chain_payload,
        )
    except Exception as exc:  # noqa: BLE001
        _log.error("/upload failed for %s: %s", arg, exc, exc_info=True)
        screen._post_line(
            "system",
            f"Upload failed: {exc}",
            severity="error",
        )
        return
    if 200 <= status < 300:
        screen._post_line(
            "assistant",
            f"✓ Uploaded chain `{arg}` (HTTP {status}).",
        )
    else:
        preview = body_preview.strip()
        suffix = f": {preview}" if preview else ""
        screen._post_line(
            "system",
            f"Upload rejected (HTTP {status}){suffix}",
            severity="error",
        )


def _post_chain_to_upload(
    *,
    url: str,
    api_key: str | None,
    auth_header: str,
    timeout: float,
    chain_id: str,
    payload: Any,
) -> tuple[int, str]:
    """POST the chain payload to ``url``. Returns ``(status,
    short_body_preview)``. Body preview is truncated to 200 chars
    so a verbose error response doesn't blow up the transcript."""
    import httpx

    headers = {"Content-Type": "application/json"}
    if api_key:
        # `Authorization` → `Bearer <key>`; any other header
        # name is treated as a raw value (e.g. `X-API-Key: <key>`).
        if auth_header.lower() == "authorization":
            headers[auth_header] = f"Bearer {api_key}"
        else:
            headers[auth_header] = api_key
    body = {"chain_id": chain_id, "chain": payload}
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=body, headers=headers)
    preview = (response.text or "")[:200]
    return response.status_code, preview


@_register("evolution")
def _cmd_evolution(screen: ChatScreen, arg: str) -> None:
    """``/evolution``                          → evolution primer, then
                                              the runs dashboard.
    ``/evolution setup [<chain_id>]``        → open the setup modal,
                                              optionally pre-bound to a
                                              chain.
    ``/evolution list``                      → open the runs dashboard.
    ``/evolution <run_id>``                  → snapshot state inline.
    ``/evolution watch <run_id>``            → stream live SSE
                                              events as `tool`
                                              lines until the
                                              stream closes or
                                              Esc fires.
    ``/evolution accept <run_id> <ind_id>``  → promote the
                                              winning individual
                                              to Memory's
                                              ``stable`` channel
                                              (Phase 5 P3).

    Setup routes through `app._push_evolution_for(chain_id)` (the shared
    modal → EvolutionScreen path). Snapshot reads
    `platform.get_evolution(run_id)`; watch reads
    `platform.stream_events(run_id)`; accept calls
    `platform.accept_individual(run_id, individual_id)` — the
    upstream SDK is idempotent on the same id (re-accept is a
    no-op; switch-after-accept surfaces 409 → friendly error).
    """
    import shlex

    try:
        parts = shlex.split(arg or "")
    except ValueError as exc:
        screen._post_line(
            "system",
            t("chat.evolution.parseFailed", error=exc),
            severity="warning",
        )
        return

    verb = parts[0].lower() if parts else ""

    # `/evolution list` (also `dashboard` / `runs`) → the runs dashboard
    # listing recent + active runs.
    if verb in ("list", "dashboard", "runs"):
        try:
            from care.screens.evolution_dashboard import EvolutionDashboard

            screen.app.push_screen(EvolutionDashboard())
        except Exception as exc:  # noqa: BLE001
            screen._post_line(
                "system",
                t("chat.evolution.dashboardFailed", error=exc),
                severity="error",
            )
        return

    # Bare `/evolution` → primer modal, then the runs dashboard.
    if not parts:
        screen._open_evolution_intro(dismiss_to_dashboard=True)
        return

    # `/evolution setup [<chain_id>]` → shared setup modal (dataset +
    # budget + rubric) and launch on submit.
    if verb in ("setup", "new", "launch"):
        chain_id = parts[1] if len(parts) > 1 else screen._session_base_chain_id()
        screen._open_evolution_setup(chain_id)
        return

    # `/evolution watch <run_id>` → live stream worker.
    if parts[0].lower() == "watch":
        if len(parts) < 2:
            screen._post_line(
                "system",
                t("chat.evolution.watchNeedsId"),
                severity="warning",
            )
            return
        run_id = parts[1]
        platform = getattr(screen.app, "platform", None)
        if platform is None:
            screen._post_line(
                "system",
                t("chat.evolution.watchNoPlatform"),
                severity="warning",
            )
            return
        screen.run_worker(
            screen._stream_evolution(run_id),
            name="evolution_stream",
            group="evolution_stream",
            exclusive=True,
            exit_on_error=False,
        )
        return

    # `/evolution accept <run_id> <individual_id>` → promote
    # the winner to Memory's `stable` channel. Phase 5 P3.
    if parts[0].lower() == "accept":
        if len(parts) < 3:
            screen._post_line(
                "system",
                t("chat.evolution.acceptUsage"),
                severity="warning",
            )
            return
        run_id = parts[1]
        individual_id = parts[2]
        platform = getattr(screen.app, "platform", None)
        if platform is None:
            screen._post_line(
                "system",
                t("chat.evolution.acceptNoPlatform"),
                severity="warning",
            )
            return
        screen._post_line(
            "tool",
            t(
                "chat.evolution.accepting",
                individualId=individual_id,
                runId=run_id,
            ),
        )
        # Chain experiments (``exp_*``) promote via Memory CARE-side, so the
        # facade needs a CareMemory handle — pass it like the EvolutionScreen
        # does (legacy ``evo_*`` accept ignores it). Retry without the kwarg
        # for older facades / test stubs that predate it.
        memory = getattr(screen.app, "memory", None)
        try:
            try:
                result = platform.accept_individual(
                    run_id, individual_id, memory=memory,
                )
            except TypeError:
                result = platform.accept_individual(
                    run_id, individual_id,
                )
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "/evolution accept %s %s failed: %s",
                run_id, individual_id, exc, exc_info=True,
            )
            screen._post_line(
                "system",
                t(
                    "chat.evolution.acceptFailed",
                    individualId=individual_id,
                    runId=run_id,
                    error=exc,
                ),
                severity="error",
            )
            return
        # Result shape varies per Platform version — surface
        # the channel/version fields if present, otherwise just
        # confirm the promotion landed.
        channel = ""
        version = ""
        if isinstance(result, dict):
            channel = str(result.get("channel") or "")
            version = str(
                result.get("version_id") or result.get("version") or "",
            )
        tail_parts: list[str] = []
        if channel:
            tail_parts.append(f"channel: {channel}")
        if version:
            tail_parts.append(f"version: {version}")
        tail = f" ({'; '.join(tail_parts)})" if tail_parts else ""
        screen._post_line(
            "assistant",
            t(
                "chat.evolution.accepted",
                individualId=individual_id,
                runId=run_id,
                tail=tail,
            ),
        )
        return

    # Default path — snapshot.
    run_id = parts[0]
    platform = getattr(screen.app, "platform", None)
    if platform is None:
        screen._post_line(
            "system",
            t("chat.evolution.snapshotNoPlatform"),
            severity="warning",
        )
        return
    try:
        state = platform.get_evolution(run_id)
    except Exception as exc:  # noqa: BLE001
        _log.error(
            "/evolution %s fetch failed: %s",
            run_id, exc, exc_info=True,
        )
        screen._post_line(
            "system",
            t("chat.evolution.fetchFailed", runId=run_id, error=exc),
            severity="error",
        )
        return
    rendered = screen._format_evolution_state(run_id, state)
    screen._post_line("assistant", rendered)


@_register("dataset")
def _cmd_dataset(screen: ChatScreen, arg: str) -> None:
    """``/dataset list <chain_id>``                     show entries
    ``/dataset add <chain_id> "<task>" --expected "<out>"``  add entry
    ``/dataset run <chain_id>``                       replay + score

    Subcommands fan out to async helpers on the screen so the
    chain-fetch + execute paths share the existing
    ``_execute_chain_interactive`` worker plumbing.
    """
    import shlex

    try:
        parts = shlex.split(arg or "")
    except ValueError as exc:
        screen._post_line(
            "system",
            t("chat.dataset.parseFailed", error=exc),
            severity="warning",
        )
        return
    if not parts:
        screen._post_line(
            "system",
            t("chat.dataset.usage"),
        )
        return
    sub = parts[0].lower()
    rest = parts[1:]
    if sub == "list":
        screen.run_worker(
            screen._dataset_list(rest),
            name="dataset_list",
            group="dataset",
            exclusive=False,
            exit_on_error=False,
        )
    elif sub == "add":
        screen.run_worker(
            screen._dataset_add(rest),
            name="dataset_add",
            group="dataset",
            exclusive=False,
            exit_on_error=False,
        )
    elif sub == "run":
        screen.run_worker(
            screen._dataset_run(rest),
            name="dataset_run",
            group="dataset",
            exclusive=False,
            exit_on_error=False,
        )
    elif sub == "export":
        screen.run_worker(
            screen._dataset_export(rest),
            name="dataset_export",
            group="dataset",
            exclusive=False,
            exit_on_error=False,
        )
    else:
        screen._post_line(
            "system",
            t("chat.dataset.unknownSub", sub=sub),
            severity="warning",
        )


# Map user-facing mode names → canonical ``ChatMode`` literal.
# Accepts every spelling :func:`_resolve_default_mode` accepts plus a
# bare ``""`` for "/mode" (no arg) which reports the current mode.
_MODE_ALIASES: dict[str, ChatMode] = {
    "interactive": "interactive",
    "ad_hoc": "interactive",  # legacy spelling
    "ad-hoc": "interactive",  # legacy spelling
    "adhoc": "interactive",  # legacy spelling
    "ad": "interactive",  # legacy shorthand
    "chat": "interactive",  # display name for interactive ("Чат" / "Chat")
    "production": "production",
    "prod": "production",
    "p": "production",
    "agent": "production",  # display name for production ("Создать агента")
    "agents": "production",
}


def _mode_display_label(mode: ChatMode) -> str:
    """Localized, user-facing name for a mode (the internal key is
    unchanged — only the display label is translated/renamed)."""
    if mode == "production":
        return t("chat.mode.agent")
    return t("chat.mode.chat")


@_register("mode")
def _cmd_mode(screen: ChatScreen, arg: str) -> None:
    """``/mode``                → print the current mode.
    ``/mode interactive``  → flip to Interactive (legacy ``ad-hoc`` still works).
    ``/mode production``   → flip to Production.

    All recognised aliases mirror :data:`_MODE_ALIASES`.  An unknown
    argument lands a ``warning``-severity system line listing valid
    forms — the user sees the available choices inline.
    """
    raw = (arg or "").strip().lower()
    if not raw:
        label = _mode_display_label(screen.mode)
        screen._post_line(
            "system",
            t("chat.mode.current", label=label, mode=screen.mode),
        )
        return
    target = _MODE_ALIASES.get(raw)
    if target is None:
        screen._post_line(
            "system",
            t("chat.mode.unknown", arg=arg),
            severity="warning",
        )
        return
    if target == screen.mode:
        # Already there — say so but don't log a no-op flip.
        screen._post_line(
            "system",
            t("chat.mode.already", label=_mode_display_label(target)),
        )
        return
    # Single write — `watch_mode` handles the log + reverse-syncs the
    # RadioSet, so the visible toggle catches up automatically.
    screen.mode = target


__all__ = ["ChatLine", "ChatScreen"]
