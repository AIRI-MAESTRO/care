"""Help screen + tutorial walkthrough data layer (TODO §9 P3).

The future ``HelpScreen`` (and the optional walkthrough overlay
on the WelcomeScreen) needs structured content to render:

* a sequence of :class:`TutorialStep` records walking the
  canonical user flow ("generate Agent A → save → generate B/C
  → return to A → re-run"), and
* a :class:`KeyBinding` registry the screen renders as a
  cheat-sheet panel.

This module owns the content + a renderer. The TUI screen is
gated on §1 P0, but the same data drives the future ``care
help`` CLI subcommand (deferred — sits in the §9 P2 CLI list)
+ a static "Quick reference" section in README.md / docs.

Both the tutorial + keybindings are **mutable registries** so
plugins / per-project overrides can append (matches the
:func:`care.runtime.register_provider_factory` pattern). The
shipped defaults cover the canonical flow + the documented
global key bindings; callers extend without forking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


KeyCategory = Literal[
    "global", "chat", "library", "generation", "execution", "evolution",
]
"""Which surface a binding applies to.

``"global"`` — always available (Ctrl+Q quit, Ctrl+P palette).
``"chat"`` — ChatScreen slash commands (``/mode``, ``/dataset``,
``/evolution``, …). Surfaces in the ``care help`` CLI and the
future HelpScreen.
``"library"`` / ``"generation"`` / ``"execution"`` /
``"evolution"`` — only when the respective screen has focus.
"""


@dataclass(frozen=True)
class KeyBinding:
    """One key + what it does on the cheat-sheet panel.

    Frozen so the registry can hand snapshots out across the
    UI without defensive copies.

    Fields:
        key: User-visible key combo (``"Ctrl+G"``, ``"Esc"``,
            ``"/"`` — same form the screen shows on the
            footer).
        action: One-line description ("Generate pipeline",
            "Cancel in-flight run", "Focus search").
        category: See :class:`KeyCategory`.
        screen: Free-form screen identifier (e.g.
            ``"DemoScreen"`` / ``"LibraryScreen"``). Empty for
            globals. Used by the future "Bindings for this
            screen" filter.
    """

    key: str
    action: str
    category: KeyCategory = "global"
    screen: str = ""


@dataclass(frozen=True)
class TutorialStep:
    """One step in the welcome-screen walkthrough.

    Frozen so the registry's snapshot is safe to pass through
    Textual messages. Each step renders as a card on the
    walkthrough overlay with title + body + (optional) hint
    keystroke.

    Fields:
        title: One-line section header.
        body: Multi-line markdown the screen renders.
        hint_key: Optional key combo the screen highlights
            ("Press Ctrl+G to continue", "Enter to open").
            Empty when the step is read-only (the welcome
            intro, the closing summary).
        screen: Which screen the step relates to — drives
            "jump to this screen" actions on the overlay.
            Empty for screen-agnostic steps.
    """

    title: str
    body: str
    hint_key: str = ""
    screen: str = ""


class HelpRegistry:
    """Mutable store for tutorial steps + key bindings.

    Mirrors the pattern :class:`care.runtime.TaskRegistry` uses
    — ordered append, listener subscription, snapshot reads.
    The default registry (see :func:`default_registry`) ships
    the canonical-flow tutorial + every documented global
    binding; callers extend with their own steps.
    """

    def __init__(self) -> None:
        self._steps: list[TutorialStep] = []
        self._bindings: list[KeyBinding] = []

    # ------------------------------------------------------------------
    # Tutorial
    # ------------------------------------------------------------------

    def add_step(self, step: TutorialStep) -> None:
        """Append a tutorial step. Order matters — the screen
        renders the cards in insertion order."""
        self._steps.append(step)

    def steps(self) -> tuple[TutorialStep, ...]:
        """Snapshot of every tutorial step in insertion order."""
        return tuple(self._steps)

    def step_titles(self) -> tuple[str, ...]:
        """Titles for the welcome screen's progress dots."""
        return tuple(s.title for s in self._steps)

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def add_binding(self, binding: KeyBinding) -> None:
        """Register a key binding. Duplicates (same key + same
        category) are silently allowed — the screen typically
        filters before rendering, and overwriting on register
        would surprise plugin authors."""
        self._bindings.append(binding)

    def bindings(self) -> tuple[KeyBinding, ...]:
        """Every registered binding in insertion order."""
        return tuple(self._bindings)

    def by_category(self, category: KeyCategory) -> tuple[KeyBinding, ...]:
        """Bindings filtered to a single category. The cheat-
        sheet panel calls this twice — once for ``"global"``
        bindings rendered always, once for the active screen's
        category."""
        return tuple(b for b in self._bindings if b.category == category)

    def by_screen(self, screen: str) -> tuple[KeyBinding, ...]:
        """Bindings filtered to a specific screen name."""
        return tuple(b for b in self._bindings if b.screen == screen)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def format_text(self) -> str:
        """CLI-friendly plain-text rendering — tutorial first,
        then bindings grouped by category."""
        lines: list[str] = []
        if self._steps:
            lines.append("# Tutorial")
            lines.append("")
            for i, step in enumerate(self._steps, 1):
                lines.append(f"## {i}. {step.title}")
                lines.append(step.body)
                if step.hint_key:
                    lines.append(f"  ⌨ {step.hint_key}")
                lines.append("")
        if self._bindings:
            lines.append("# Key bindings")
            for category in (
                "global", "chat", "library", "generation",
                "execution", "evolution",
            ):
                cat_bindings = self.by_category(category)  # type: ignore[arg-type]
                if not cat_bindings:
                    continue
                lines.append("")
                lines.append(f"## {category}")
                for b in cat_bindings:
                    suffix = f" ({b.screen})" if b.screen else ""
                    lines.append(f"  {b.key:<14} {b.action}{suffix}")
        return "\n".join(lines).rstrip()

    def format_markdown(self) -> str:
        """Markdown rendering — same content but with `**bold**`
        keystrokes for the future `care help` CLI's --markdown
        flag + a copy-paste-friendly README quick-reference."""
        lines: list[str] = []
        if self._steps:
            lines.append("## Walkthrough")
            lines.append("")
            for i, step in enumerate(self._steps, 1):
                lines.append(f"**{i}. {step.title}**")
                lines.append("")
                lines.append(step.body)
                if step.hint_key:
                    lines.append("")
                    lines.append(f"> Try: `{step.hint_key}`")
                lines.append("")
        if self._bindings:
            lines.append("## Keys")
            lines.append("")
            for category in (
                "global", "chat", "library", "generation",
                "execution", "evolution",
            ):
                cat_bindings = self.by_category(category)  # type: ignore[arg-type]
                if not cat_bindings:
                    continue
                lines.append(f"**{category}**")
                lines.append("")
                for b in cat_bindings:
                    suffix = f" *({b.screen})*" if b.screen else ""
                    lines.append(f"- `{b.key}` — {b.action}{suffix}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Built-in canonical-flow registry
# ---------------------------------------------------------------------------


_DEFAULT_STEPS: tuple[TutorialStep, ...] = (
    TutorialStep(
        title="Welcome to CARE",
        body=(
            "Collaborative Agent Reasoning Ecosystem.\n"
            "Generate agent chains with MAGE, run them through CARL, "
            "evolve them via GigaEvo Platform — all from one TUI."
        ),
    ),
    TutorialStep(
        title="Describe your task",
        body=(
            "On the QueryScreen, type what you want the agent to do "
            "and optionally add context files. The text and files become "
            "the seed for MAGE's chain generation."
        ),
        hint_key="Ctrl+G",
        screen="QueryScreen",
    ),
    TutorialStep(
        title="Watch the chain build",
        body=(
            "GenerationScreen streams MAGE's progress live: domain "
            "analysis, step planning, DAG building, step descriptions. "
            "Hit Esc to cancel mid-flight."
        ),
        hint_key="Esc",
        screen="GenerationScreen",
    ),
    TutorialStep(
        title="Save to your library",
        body=(
            "The SaveAgentModal pre-fills name + description from MAGE's "
            "suggestions. Tweak, save, and the agent appears in the "
            "LibraryScreen alongside everything else."
        ),
        hint_key="Ctrl+S",
        screen="SaveAgentModal",
    ),
    TutorialStep(
        title="Re-run from the library",
        body=(
            "LibraryScreen lists every saved agent with last-run time, "
            "fitness, favourite status. Hit Enter to inspect, R to run, "
            "E to edit, F to favourite."
        ),
        hint_key="Enter",
        screen="LibraryScreen",
    ),
    TutorialStep(
        title="Evolve when you're stuck",
        body=(
            "EvolutionScreen submits the chain to GigaEvo Platform's GA, "
            "watches the fitness curve, and lets you accept the best "
            "individual back to Memory as a new version."
        ),
        screen="EvolutionScreen",
    ),
    TutorialStep(
        title="Everything is configurable",
        body=(
            "`~/.config/care/config.toml` + `CARE_*` env vars + `./care.toml` "
            "per-project overrides. See `.env.example` for every knob, or "
            "run the SettingsScreen wizard from the welcome panel."
        ),
        hint_key="Ctrl+,",
    ),
    # Chat-surface steps — added when the dual-mode ChatScreen
    # became the primary entry-point. The single chat surface
    # subsumes Query / Generation / SaveAgentModal flows above
    # for everyday work; the older screen-based steps remain
    # for the headless / palette-driven paths.
    TutorialStep(
        title="Chat is the primary surface",
        body=(
            "Type a task in natural language; CARE generates a chain "
            "(MAGE), runs it (CARL), and prints the result inline. "
            "Slash commands cover everything else — type `/help` in "
            "chat for the full list."
        ),
        hint_key="/help",
        screen="ChatScreen",
    ),
    TutorialStep(
        title="Pick a mode: Ad-Hoc or Production",
        body=(
            "`/mode` shows the current mode; `/mode production` "
            "switches. Ad-Hoc runs on the spot and saves nothing — "
            "the agent may loop until done. Production saves the "
            "chain to Memory under a stable `chain_id`, seeds a "
            "baseline dataset entry for quality tracking, and (when "
            "Platform is wired) kicks off an evolution run."
        ),
        hint_key="/mode production",
        screen="ChatScreen",
    ),
    TutorialStep(
        title="Measure quality with /dataset",
        body=(
            "Every Production save seeds a dataset entry. Grow it "
            "with `/dataset add <chain_id> \"<task>\" --expected "
            "\"<out>\"`, score the lot with `/dataset run <chain_id>`, "
            "and export to JSONL for external eval frameworks via "
            "`/dataset export <chain_id> <path>`."
        ),
        hint_key="/dataset list <chain_id>",
        screen="ChatScreen",
    ),
    TutorialStep(
        title="Watch evolution with /evolution",
        body=(
            "`/evolution <run_id>` renders the live state of an "
            "evolution run inline (generation, population, best "
            "score). `/evolution watch <run_id>` streams events as "
            "they happen so you see each generation tick by."
        ),
        hint_key="/evolution watch <run_id>",
        screen="ChatScreen",
    ),
)


_DEFAULT_BINDINGS: tuple[KeyBinding, ...] = (
    # Globals.
    KeyBinding(key="Ctrl+P", action="Open command palette"),
    KeyBinding(key="Ctrl+Q", action="Quit CARE"),
    KeyBinding(key="Ctrl+S", action="Save current artifact"),
    KeyBinding(key="Ctrl+R", action="Re-run current chain"),
    KeyBinding(key="Ctrl+K", action="Browse capability catalog"),
    KeyBinding(key="Esc", action="Back / cancel in-flight work"),
    KeyBinding(key="?", action="Open this help screen"),
    # Library.
    KeyBinding(key="Enter", action="Open agent in InspectionScreen", category="library", screen="LibraryScreen"),
    KeyBinding(key="R", action="Run the selected agent", category="library", screen="LibraryScreen"),
    KeyBinding(key="E", action="Edit the selected agent", category="library", screen="LibraryScreen"),
    KeyBinding(key="F", action="Toggle favourite", category="library", screen="LibraryScreen"),
    KeyBinding(key="Del", action="Delete (with confirmation)", category="library", screen="LibraryScreen"),
    KeyBinding(key="Ctrl+F", action="Focus search", category="library", screen="LibraryScreen"),
    KeyBinding(key="/", action="Focus search (alt)", category="library", screen="LibraryScreen"),
    # Generation.
    KeyBinding(key="Ctrl+G", action="Generate pipeline", category="generation", screen="QueryScreen"),
    KeyBinding(key="Ctrl+L", action="Clear inputs", category="generation", screen="QueryScreen"),
    # Execution.
    KeyBinding(key="Space", action="Pause / resume run", category="execution", screen="ExecutionScreen"),
    # Evolution.
    KeyBinding(key="A", action="Accept the current best individual", category="evolution", screen="EvolutionScreen"),
    # Chat — slash commands. ``key`` carries the slash form so
    # the renderers can show "/mode" alongside "Ctrl+G"; the
    # category sorts them under their own section in the help
    # output rather than mixing with terminal-key globals.
    KeyBinding(key="/help", action="Show chat help (mode-aware)", category="chat", screen="ChatScreen"),
    KeyBinding(key="/tour", action="5-step guided walkthrough", category="chat", screen="ChatScreen"),
    KeyBinding(key="/mode", action="Show or switch chat mode (ad_hoc / production)", category="chat", screen="ChatScreen"),
    KeyBinding(key="/library", action="Open the saved-agents library", category="chat", screen="ChatScreen"),
    KeyBinding(key="/run <chain_id>", action="Open a saved chain for execution", category="chat", screen="ChatScreen"),
    KeyBinding(key="/dataset list <chain_id>", action="List dataset entries for a chain", category="chat", screen="ChatScreen"),
    KeyBinding(key="/dataset add <chain_id> \"<task>\" --expected \"<out>\"", action="Append a scored entry", category="chat", screen="ChatScreen"),
    KeyBinding(key="/dataset run <chain_id>", action="Replay every entry and score it", category="chat", screen="ChatScreen"),
    KeyBinding(key="/dataset export <chain_id> <path>", action="Write entries as JSONL", category="chat", screen="ChatScreen"),
    KeyBinding(key="/evolution <run_id>", action="Render evolution run state inline", category="chat", screen="ChatScreen"),
    KeyBinding(key="/evolution watch <run_id>", action="Stream evolution events live", category="chat", screen="ChatScreen"),
)


def default_registry() -> HelpRegistry:
    """Build a fresh registry pre-loaded with the canonical-flow
    tutorial + every documented binding.

    Callers wanting to extend (e.g. a plugin adding its own
    screen) start from this base and append. The returned
    registry is mutable; clones aren't shared between calls.
    """
    registry = HelpRegistry()
    for step in _DEFAULT_STEPS:
        registry.add_step(step)
    for binding in _DEFAULT_BINDINGS:
        registry.add_binding(binding)
    return registry


# ---------------------------------------------------------------------------
# Plugin-friendly hooks
# ---------------------------------------------------------------------------


HelpRegistryExtension = Callable[[HelpRegistry], None]
"""Callable that augments a help registry — used by plugins to
register their own tutorial step(s) + key bindings."""

_EXTENSIONS: list[HelpRegistryExtension] = []


def register_help_extension(extension: HelpRegistryExtension) -> None:
    """Register a registry mutator that runs after the
    defaults. Called by :func:`build_registry`. Mirrors the
    convention :func:`care.runtime.register_provider_factory`
    + :func:`care.runtime.register_telemetry_backend` use."""
    _EXTENSIONS.append(extension)


def unregister_help_extension(extension: HelpRegistryExtension) -> bool:
    """Drop a previously-registered extension. Returns whether
    it was found. Mostly for tests."""
    try:
        _EXTENSIONS.remove(extension)
    except ValueError:
        return False
    return True


def build_registry() -> HelpRegistry:
    """Build a populated registry: defaults + every registered
    extension applied in registration order. The future
    HelpScreen calls this on mount; the CLI subcommand calls it
    too so plugin help shows up in both surfaces."""
    registry = default_registry()
    for extension in list(_EXTENSIONS):
        try:
            extension(registry)
        except Exception:  # noqa: BLE001
            # Extensions are best-effort — a buggy plugin must
            # not break the help screen.
            continue
    return registry


__all__ = [
    "HelpRegistry",
    "HelpRegistryExtension",
    "KeyBinding",
    "KeyCategory",
    "TutorialStep",
    "build_registry",
    "default_registry",
    "register_help_extension",
    "unregister_help_extension",
]
