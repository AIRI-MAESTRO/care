"""Doc-lint for `docs/screens/README.md` (TODO §8 P2 — Single-page
Screens at a glance reference).

Drift guards: when a new screen / modal lands or a slash command
is renamed, the doc must stay in lockstep. Two checks:

1. Every screen class in `care.screens.__all__` ending in
   ``Screen`` / ``Modal`` / ``Drawer`` MUST appear by name in the
   doc (catches "shipped a screen but forgot to document it").
2. Every slash command in the doc's status table MUST be
   registered in `_COMMAND_HANDLERS` on `ChatScreen` (catches
   "renamed `/x` to `/y` and forgot the doc").

Skipped exclusions are explicit: a small allow-list covers classes
that aren't user-visible screens (data envelopes, registries) or
slashes that the doc deliberately tags as ``(boot)`` / ``(legacy)``
/ ``(planned)``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENS_DOC = PROJECT_ROOT / "docs" / "screens" / "README.md"


# Names exported from `care.screens` that aren't actually screen
# widgets — dismiss envelopes, registries, factory helpers, etc.
# These shouldn't appear in the user-visible screens index.
_NON_SCREEN_EXPORTS: frozenset[str] = frozenset({
    # Dismiss envelopes / result dataclasses.
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
    # Factory / projection helpers (not screens).
    "CatalogPromoteRequest",
    "default_next_screen",
    "parse_evolution_run_row",
})


def _doc_body() -> str:
    return SCREENS_DOC.read_text(encoding="utf-8")


def _command_handlers() -> dict[str, object]:
    """Pull the live `_COMMAND_HANDLERS` registry from ChatScreen."""
    from care.screens.chat import _COMMAND_HANDLERS
    return dict(_COMMAND_HANDLERS)


def _user_facing_screens() -> list[str]:
    from care.screens import __all__ as exports

    out: list[str] = []
    for name in exports:
        if name in _NON_SCREEN_EXPORTS:
            continue
        # Heuristic: user-visible classes end in one of these.
        if name.endswith(("Screen", "Modal", "Drawer")):
            out.append(name)
    return sorted(out)


class TestScreenIndexDoc:
    def test_doc_exists(self):
        assert SCREENS_DOC.is_file(), (
            f"Missing {SCREENS_DOC.relative_to(PROJECT_ROOT)} — "
            "Screens at a glance reference must exist (§8 P2)."
        )

    def test_doc_mentions_every_user_facing_screen(self):
        body = _doc_body()
        missing = [
            name for name in _user_facing_screens()
            if name not in body
        ]
        assert not missing, (
            f"`docs/screens/README.md` doesn't mention these "
            f"user-facing screens / modals: {missing}. Add a "
            "table row or extend `_NON_SCREEN_EXPORTS` in this "
            "test if the class isn't actually a navigation "
            "target."
        )

    def test_slash_commands_in_doc_are_registered(self):
        body = _doc_body()
        registered = set(_command_handlers().keys())
        # Pull every `/<word>` mention from the table cells. The
        # table escapes "(boot)" / "(legacy)" / "(planned)" for
        # non-slash columns, so we only match the actual
        # `/<command>` form.
        mentioned = set(re.findall(r"`/([a-z][a-z_-]*)`", body))
        unknown = mentioned - registered
        assert not unknown, (
            f"`docs/screens/README.md` mentions slash commands "
            f"that aren't registered in ChatScreen's "
            f"_COMMAND_HANDLERS: {sorted(unknown)}. Either "
            "register the handler or update the doc."
        )

    def test_doc_lists_four_canonical_screens(self):
        # The four-screen map is the canonical mental model;
        # any future restructure should keep these four
        # explicitly mentioned in a section.
        body = _doc_body()
        for name in (
            "ChatScreen",
            "ArtifactsScreen",
            "LibraryScreen",
            "EvolutionScreen",
        ):
            assert name in body, (
                f"Canonical four-screen map missing {name}"
            )

    def test_doc_has_status_legend(self):
        body = _doc_body()
        # The legend pins the M0 / M1 vocabulary; if someone
        # rotates the labels, every status cell needs to update.
        assert "## Status legend" in body
        assert "M0" in body
        assert "M1" in body


@pytest.fixture
def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


class TestReadmeCrossLink:
    def test_readme_links_screens_index(self, _readme):
        assert "docs/screens" in _readme or "screens at a glance" in _readme.lower(), (
            "README should cross-link to docs/screens/README.md "
            "so the index is discoverable from the repo landing "
            "page (§8 P2)."
        )


# ---------------------------------------------------------------------------
# Auto-generator lockstep (§8 P3 — iter 83)
# ---------------------------------------------------------------------------


class TestAutoGenerator:
    """Lockstep guard: the on-disk
    `docs/screens/README.md` must match what
    `scripts/generate_screens_index.py::render_markdown()`
    produces — otherwise the doc has drifted from the live
    `_SCREEN_METADATA` / registries and a contributor
    forgot to re-run the generator.
    """

    def test_doc_matches_generator_output(self):
        # Import the generator lazily so the test file stays
        # safe to import even when the scripts/ dir isn't on
        # sys.path.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_generate_screens_index",
            PROJECT_ROOT / "scripts" / "generate_screens_index.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        generated = module.render_markdown()
        on_disk = SCREENS_DOC.read_text(encoding="utf-8")
        if on_disk != generated:
            raise AssertionError(
                "docs/screens/README.md drifted from "
                "scripts/generate_screens_index.py output. "
                "Re-run:\n\n"
                "    uv run python scripts/generate_screens_index.py --write\n\n"
                "to regenerate."
            )

    def test_generator_includes_every_canonical_screen(self):
        """Sanity: the metadata dict actually carries entries
        for the four canonical screens. Catches a future PR
        that demotes one without updating the categorisation."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_generate_screens_index_meta",
            PROJECT_ROOT / "scripts" / "generate_screens_index.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        meta = module._SCREEN_METADATA
        for name in (
            "ChatScreen",
            "ArtifactsScreen",
            "LibraryScreen",
            "EvolutionScreen",
        ):
            assert name in meta, (
                f"_SCREEN_METADATA missing canonical screen "
                f"{name!r}"
            )
            assert meta[name][0] == "canonical", (
                f"{name} is not tagged `canonical` in "
                "_SCREEN_METADATA"
            )
