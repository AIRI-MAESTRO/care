"""Regenerate `docs/screens/README.md` from live registries
(TODO §8 P3 — auto-build the Screens-at-a-glance table).

Projects three source-of-truth registries into the canonical
markdown shape:

* `care.screens.__all__` — every shipped screen / modal class
  (filtered through ``_NON_SCREEN_EXPORTS`` so dismiss envelopes
  + factory helpers don't surface as user-visible rows).
* `ChatScreen._COMMAND_BLURBS` — slash → purpose mapping
  consumed by the `/help` modal too, so the doc and the in-app
  help stay implicitly synchronised.
* `_SCREEN_METADATA` — the small per-screen curation dict
  in this file. Records the section (canonical / supporting /
  modal), explicit trigger override (e.g. `(boot)` for
  WelcomeScreen, `(Ctrl+P)` for CommandPaletteModal), status
  (M0 / M1 / M2), and a one-line purpose. Screens absent from
  this dict surface in the "unclassified" trailing list with a
  reminder to file them.

Usage:

    # Compare-only (fails non-zero on drift — CI mode):
    uv run --extra dev python scripts/generate_screens_index.py --check

    # Regenerate the on-disk doc:
    uv run --extra dev python scripts/generate_screens_index.py --write

The wrapper test
``tests/test_screens_index_doc.py::TestAutoGenerator`` invokes
``--check`` mode so a doc that drifts from the registries breaks
the suite, not just the next contributor's PR review.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = PROJECT_ROOT / "docs" / "screens" / "README.md"


Section = Literal["canonical", "supporting", "modal"]


class _Row:
    __slots__ = ("name", "trigger", "status", "purpose", "section")

    def __init__(
        self,
        name: str,
        trigger: str,
        status: str,
        purpose: str,
        section: Section,
    ) -> None:
        self.name = name
        self.trigger = trigger
        self.status = status
        self.purpose = purpose
        self.section = section


# Per-screen metadata. Each entry pins the row's section, the
# trigger string the doc renders (slash command in backticks,
# `(boot)` / `(legacy)` / binding chord for non-slash entry
# points), the status (M0 = shipped, M1 = in-flight, M2 =
# filed), and the one-line purpose.
_SCREEN_METADATA: dict[str, tuple[Section, str, str, str]] = {
    # Four canonical screens (chat-centric primary map).
    "ChatScreen": (
        "canonical", "(boot)", "M0",
        "Natural-language input, mode toggle, slash palette, "
        "artifact pill, Production action toolbar",
    ),
    "ArtifactsScreen": (
        "canonical", "`/artifacts`", "M0",
        "Current-chat artifacts (chain / stage / tool / "
        "dataset / synth output); save, copy, drop, inspect",
    ),
    "LibraryScreen": (
        "canonical", "`/library`", "M0",
        "Saved chains — sort, filter, tag-pool, recency strip, "
        "mean cost, bulk import / export",
    ),
    "EvolutionScreen": (
        "canonical", "`/evolution`", "M0",
        "Run + watch a GA over a chain; Pareto front, fitness "
        "curve, scatter plot, cost meter, accept",
    ),
    # Supporting screens.
    "WelcomeScreen": (
        "supporting", "(boot)", "M0",
        "Boot splash; routes to ChatScreen (returning users) "
        "or SettingsScreen (first-run / missing creds)",
    ),
    "SettingsScreen": (
        "supporting", "`/settings`", "M0",
        "Edit MAGE / Memory / Platform creds + theme + "
        "advanced knobs",
    ),
    "EvolutionDashboard": (
        "supporting", "`/evolution`", "M0",
        "List of active + recent evolution runs; Enter opens "
        "EvolutionScreen, `c` compares two",
    ),
    "RunsScreen": (
        "supporting", "`/runs`", "M0",
        "Local run history (`~/.cache/care/runs/`); Enter "
        "opens ReplayScreen sidecar",
    ),
    "LogsScreen": (
        "supporting", "`/logs`", "M0",
        "Tail the rolling app log; `m` toggles a module filter",
    ),
    "CostDashboardScreen": (
        "supporting", "`/cost`", "M0",
        "Token + USD spend rollup by provider / chain / "
        "session",
    ),
    "SandboxTrustScreen": (
        "supporting", "`/sandbox`", "M0",
        "Audit + revoke trusted AgentSkills (SHA-pinned trust "
        "store)",
    ),
    "ProfileScreen": (
        "supporting", "`/profile`", "M0",
        "List credential profiles under "
        "`~/.config/care/profiles/`",
    ),
    "CatalogScreen": (
        "supporting", "(Ctrl+K)", "M0",
        "Browse installed capabilities (skills / MCP / tools / "
        "cards)",
    ),
    "MarketplaceScreen": (
        "supporting", "`/marketplace`", "M0",
        "Search shared agent_skill listings on Memory",
    ),
    "HelpScreen": (
        "supporting", "`/help`", "M0",
        "Tutorial + every binding (filtered by active screen)",
    ),
    "InspectionScreen": (
        "supporting", "(Library `Enter`)", "M0",
        "Saved-chain detail + run history + Integration pane",
    ),
    "EditAgentScreen": (
        "supporting", "(Library `e`)", "M0",
        "Inline edit + save-as-new-version + promote-to-stable",
    ),
    "ExecutionScreen": (
        "supporting", "(Library `r`)", "M0",
        "Live CARL run + token streaming",
    ),
    "ReplayScreen": (
        "supporting", "(Runs `Enter`)", "M0",
        "Step through a saved ReasoningResult",
    ),
    "QueryScreen": (
        "supporting", "(legacy)", "M0",
        "Pre-chat \"+ New agent\" task form — still "
        "reachable, no longer canonical",
    ),
    "GenerationScreen": (
        "supporting", "(legacy)", "M0",
        "Pre-chat live-MAGE-progress surface — superseded by "
        "ChatScreen's inline progress lines",
    ),
    "DemoScreen": (
        "supporting", "(boot fallback)", "M0",
        "First-run / config-error fallback so users see "
        "something",
    ),
    "TaskListDrawer": (
        "supporting", "(Ctrl+B)", "M0",
        "In-flight workers panel",
    ),
    # Modals.
    "CommandPaletteModal": (
        "modal", "`Ctrl+P` (any screen)", "M0",
        "Fuzzy palette over commands + saved entities",
    ),
    "ConfirmModal": (
        "modal", "Destructive actions", "M0",
        "OK / Cancel confirm for destructive actions (bulk "
        "delete, accept-winner)",
    ),
    "DiffModal": (
        "modal", "Library `D`, EvolutionScreen `D`", "M0",
        "Side-by-side compare two chains / individual vs. "
        "parent",
    ),
    "LineageModal": (
        "modal", "Library `l`", "M0",
        "Walk a chain's ancestry DAG",
    ),
    "ConflictModal": (
        "modal", "Save-with-name-collision", "M0",
        "Resolve a name collision on save",
    ),
    "SaveAgentModal": (
        "modal", "Post-generation", "M0",
        "Tag + name a freshly-generated chain before "
        "persistence",
    ),
    "TagEditorModal": (
        "modal", "Library `T`, Artifacts `s`", "M0",
        "Edit tags (bulk) + optional editable title (§3 P3 "
        "save-flow path)",
    ),
    "ImportModal": (
        "modal", "Library `i`", "M0",
        "Import a chain bundle (tar.gz)",
    ),
    "ExportModal": (
        "modal", "Library `x`", "M0",
        "Export saved-Memory entities into a tarball",
    ),
    "ExportChainModal": (
        "modal", "Evolution `x`", "M0",
        "Export a single chain payload to disk (JSON / Python)",
    ),
    "ResumeModal": (
        "modal", "`/resume`", "M0",
        "Rehydrate a Production-mode transcript",
    ),
    "RunContextModal": (
        "modal", "Library / Execution `r`", "M0",
        "Re-run form: task + context-file picker + tags",
    ),
    "SaveReport": (
        "modal", "After save-all batch", "M0",
        "Post-mortem table of save-all outcomes",
    ),
    "UseItNowModal": (
        "modal", "Post-save + accept-winner", "M0",
        "Copy-paste recipe (python / curl / cli) for the "
        "saved chain",
    ),
    "HumanInputModal": (
        "modal", "CARL human-input step", "M0",
        "Block CARL execution for a human-supplied answer",
    ),
    "EvolutionLaunchModal": (
        "modal", "Library `v` / `E`", "M0",
        "Budget / rubric / objectives picker before "
        "EvolutionScreen launches",
    ),
    "EvolutionCompareModal": (
        "modal", "Dashboard `c` after multi-select", "M0",
        "Side-by-side fitness curves for two evolution runs",
    ),
    "OnboardingScreen": (
        "supporting", "(boot, planned)", "M1",
        "uvx first-run wizard — §1 P0 still in flight",
    ),
}


# Names exported from `care.screens` that aren't user-visible
# screens — dataclasses, registries, factory helpers.
_NON_SCREEN_EXPORTS: frozenset[str] = frozenset({
    "ConflictModalResult",
    "DiffResult",
    "EditAgentEvent",
    "EvolutionIndividual",
    "EvolutionLaunchSpec",
    "EvolutionRunRow",
    "EvolutionRunState",
    "ExecutionState",
    "ExportChainResult",
    "ExportRequest",
    "GenerationProgress",
    "HumanInputResult",
    "ImportRequest",
    "InspectionAction",
    "InspectionPayload",
    "LaunchRequested",
    "LineageResult",
    "MarketplaceInstalled",
    "PaletteSelection",
    "QuerySubmission",
    "ResumeResult",
    "RunContextResult",
    "SaveAgentAction",
    "SaveAgentResult",
    "SaveReportResult",
    "SaveReportRow",
    "SettingsSnapshot",
    "TagEditorResult",
    "UseItNowResult",
    "CatalogPromoteRequest",
    "default_next_screen",
    "parse_evolution_run_row",
})


_HEADER = """# Screens at a glance

Single-page reference for every Textual screen + modal CARE ships.
Use this as the navigation index when implementing a new feature
("which screen owns X?") or when writing a doc page ("where does
this fit in the user journey?").

Per-screen reference pages (canonical bindings, compose tree, test
file, design constraints) live under `docs/screens/<screen>.md` once
each lands — filed as the §8 P1 follow-up. This README stays the
top-level "what exists, what's its slash command, what does it do".

> This file is **auto-generated** by
> `scripts/generate_screens_index.py`. Edit
> `_SCREEN_METADATA` in that script (or the table headings /
> intro paragraphs in `_HEADER` / `_FOOTER`) and re-run
> `python scripts/generate_screens_index.py --write` to update.
> A regression test (`tests/test_screens_index_doc.py
> ::TestAutoGenerator`) enforces lockstep so the doc can't
> drift from the live `care.screens.__all__` /
> `_COMMAND_BLURBS` registries.

## Four canonical screens

The chat-centric refactor (Phases 1–6) collapsed the original
`Query → Generation → Inspection` flow into four user-visible
screens. ChatScreen is the primary entry; the other three are
reached on demand via slash commands + the Production-mode action
toolbar.

"""


_SUPPORTING_INTRO = """
## Supporting screens (reached from the four above)

"""


_MODAL_INTRO = """
## Modal screens (overlays)

Modals layer on top of screens but never own the primary navigation
target. Most are pushed via row actions, save flows, or the dismiss
of a screen-level interaction.

"""


_FOOTER = """
## Status legend

- **M0** — Shipped + tested in the v0.1 release path. Owns its
  documented bindings + flows.
- **M1** — In flight or behind a tracked TODO blocker.
- **M2** — Filed but not started; check `TODO.md` for the priority.

## Where to look next

- Per-screen reference: `docs/screens/<screen>.md` (filed as §8 P1
  follow-up — pending).
- Slash command reference: every command in the table above is
  registered in `_COMMAND_HANDLERS` inside
  [`care/screens/chat.py`](../../care/screens/chat.py) with a
  one-line blurb in `_COMMAND_BLURBS`. The `/help` modal in-app
  reads the same registries so it stays in lockstep with this
  doc.
- Architecture overview: [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
  — the four-screen map in §1 is the canonical mental model for
  how the screens compose.
- Full work plan: [`TODO.md`](../../TODO.md) — every screen has
  shipping notes and follow-up tasks.
"""


def _user_facing_screens() -> list[str]:
    from care.screens import __all__ as exports
    return [
        name for name in exports
        if name not in _NON_SCREEN_EXPORTS
        and name.endswith(("Screen", "Modal", "Drawer"))
    ]


def _render_table(rows: list[_Row], headings: tuple[str, ...]) -> str:
    lines: list[str] = []
    header = "| " + " | ".join(headings) + " |"
    sep_cells: list[str] = []
    for idx, head in enumerate(headings):
        if idx == 0:
            sep_cells.append("-" * (len(head) + 2))
        elif head == "Status":
            sep_cells.append("------")
        else:
            sep_cells.append("-" * (len(head) + 2))
    separator = "| " + " | ".join(sep_cells) + " |"
    lines.append(header)
    lines.append(separator)
    for row in rows:
        lines.append(
            f"| `{row.name}` | {row.trigger} | {row.status} | "
            f"{row.purpose} |"
        )
    return "\n".join(lines) + "\n"


def render_markdown() -> str:
    """Project registries + metadata dict into the canonical
    markdown shape. Returns the full file body."""
    rows_by_section: dict[Section, list[_Row]] = {
        "canonical": [],
        "supporting": [],
        "modal": [],
    }
    unknown: list[str] = []
    seen: set[str] = set()
    for name in _user_facing_screens():
        seen.add(name)
        meta = _SCREEN_METADATA.get(name)
        if meta is None:
            unknown.append(name)
            continue
        section, trigger, status, purpose = meta
        rows_by_section[section].append(
            _Row(name, trigger, status, purpose, section),
        )
    # Stable order within each section: canonical keeps its
    # explicit order from the metadata dict; supporting +
    # modal sort alphabetically.
    canonical_order = [
        name for name, meta in _SCREEN_METADATA.items()
        if meta[0] == "canonical"
    ]
    rows_by_section["canonical"].sort(
        key=lambda r: canonical_order.index(r.name)
        if r.name in canonical_order else 1000,
    )
    rows_by_section["supporting"].sort(key=lambda r: r.name)
    rows_by_section["modal"].sort(key=lambda r: r.name)

    # Add metadata-only screens (like OnboardingScreen) that
    # don't ship in `care.screens.__all__` yet but the doc
    # should still mention.
    metadata_only = [
        name for name in _SCREEN_METADATA
        if name not in seen
    ]
    for name in metadata_only:
        section, trigger, status, purpose = _SCREEN_METADATA[name]
        rows_by_section[section].append(
            _Row(name, trigger, status, purpose, section),
        )
        # Re-sort the section after the late-add. Canonical
        # preserves the metadata-dict insertion order so the
        # four-screen map reads Chat → Artifacts → Library →
        # Evolution regardless of which entries surface via
        # `__all__` vs. metadata-only late-add.
        if section == "canonical":
            rows_by_section["canonical"].sort(
                key=lambda r: canonical_order.index(r.name)
                if r.name in canonical_order else 1000,
            )
        elif section == "supporting":
            rows_by_section["supporting"].sort(key=lambda r: r.name)
        elif section == "modal":
            rows_by_section["modal"].sort(key=lambda r: r.name)

    out: list[str] = [_HEADER]
    out.append(_render_table(
        rows_by_section["canonical"],
        ("Screen", "Slash", "Status", "Primary purpose"),
    ))
    out.append(_SUPPORTING_INTRO)
    out.append(_render_table(
        rows_by_section["supporting"],
        ("Screen", "Trigger", "Status", "Primary purpose"),
    ))
    out.append(_MODAL_INTRO)
    out.append(_render_table(
        rows_by_section["modal"],
        ("Modal", "Triggered from", "Status", "Primary purpose"),
    ))
    if unknown:
        out.append(
            "\n## Unclassified (FIX-ME)\n\n"
            "These screens are exported from `care.screens.__all__`\n"
            "but `_SCREEN_METADATA` in `scripts/generate_screens_index.py`\n"
            "doesn't list them. Add an entry to keep the index honest:\n\n"
        )
        for name in unknown:
            out.append(f"- `{name}`\n")
    out.append(_FOOTER)
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate docs/screens/README.md from "
        "live registries.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Compare against the on-disk doc; exit non-zero on "
            "drift (CI mode)."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate the doc file in place.",
    )
    args = parser.parse_args()

    generated = render_markdown()
    if args.write:
        DOC_PATH.write_text(generated, encoding="utf-8")
        print(f"wrote {DOC_PATH.relative_to(PROJECT_ROOT)} "
              f"({len(generated)} bytes)")
        return 0
    if args.check:
        try:
            on_disk = DOC_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"could not read {DOC_PATH}: {exc}", file=sys.stderr)
            return 2
        if on_disk == generated:
            print(f"{DOC_PATH.relative_to(PROJECT_ROOT)} matches "
                  "generator output.")
            return 0
        print(
            f"{DOC_PATH.relative_to(PROJECT_ROOT)} drifted from "
            "generator output. Re-run with --write to fix.",
            file=sys.stderr,
        )
        return 1
    # No flag: print to stdout.
    sys.stdout.write(generated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
