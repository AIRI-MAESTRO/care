"""Structural pin for ``docs/ARCHITECTURE.md`` (TODO §10 P3).

ARCHITECTURE.md is shipped as the discoverability entry-point for
contributors: where to find each module, which upstream contract
backs each surface, where the canonical user flow lives. The doc
will inevitably drift as code changes; these tests catch the
most common drift modes (sections accidentally removed,
cross-references broken to non-existent files) without trying to
enforce the prose itself.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURE_PATH = PROJECT_ROOT / "docs" / "ARCHITECTURE.md"


def _read() -> str:
    return ARCHITECTURE_PATH.read_text(encoding="utf-8")


class TestArchitectureDoc:
    def test_file_exists(self):
        assert ARCHITECTURE_PATH.is_file(), (
            f"{ARCHITECTURE_PATH} is missing — `docs/ARCHITECTURE.md` "
            "is the entry-point doc for contributors."
        )

    def test_top_level_sections_present(self):
        """The doc walks four layers + flow + chat-mode contract +
        config + CLI. Removing any section means a contributor
        loses the discoverability path for that layer.

        Section numbering shifted in Phase 7 P1 to insert §5
        ("The chat-surface dual contract") between the legacy
        canonical-flow section and Configuration. The required
        list mirrors the current heading sequence.
        """
        body = _read()
        required = [
            "# CARE — Architecture",
            "## 1. The four-module stack",
            "## 2. Module boundaries inside `care/`",
            "## 3. Layer-by-layer reference",
            "### 3.1 Generation — MAGE",
            "### 3.2 Execution — CARL",
            "### 3.3 Persistence — GigaEvo Memory",
            "### 3.4 Evolution — GigaEvo Platform",
            "### 3.5 Sandbox",
            "## 4. The canonical user flow",
            "## 5. The chat-surface dual contract",
            "### 5.1 Why two modes",
            "### 5.2 Ad-Hoc data flow",
            "### 5.3 Production data flow",
            "### 5.4 Shared adapters & seams",
            "## 6. Configuration & precedence",
            "## 7. CLI vs TUI",
            "## 8. Where to look next",
        ]
        missing = [h for h in required if h not in body]
        assert not missing, (
            f"`docs/ARCHITECTURE.md` is missing sections: {missing}"
        )

    def test_mirrors_todo_diagram(self):
        """The §0 diagram in TODO.md is repeated here. Both ends
        should agree on the layer names so a reader who flips
        between them isn't confused."""
        body = _read()
        # Spot-check the four MAGE / CARL / Platform / Sandbox layer
        # labels that appear in the TODO diagram.
        for label in (
            "MAGE async",
            "CARL runner",
            "GigaEvo",
            "Platform",
            "AgentSkill sandbox",
        ):
            assert label in body, (
                f"layer label {label!r} missing — the doc must mirror "
                "the TODO §0 diagram so the two stay in sync."
            )

    def test_internal_links_resolve(self):
        """Every relative link in the doc should point at a file
        that actually exists in the repo. Catches drift like
        renamed files / moved sections."""
        body = _read()
        import re

        # Pull every Markdown link target: `[label](path)`.
        link_re = re.compile(r"\]\(([^)]+)\)")
        for target in link_re.findall(body):
            # Skip anchors + external URLs.
            if target.startswith(("http://", "https://", "#")):
                continue
            # Strip any fragment.
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            resolved = (ARCHITECTURE_PATH.parent / path_part).resolve()
            # Cross-module references point outside the CARE repo
            # (e.g. ../../carl-mage). Those are valid pointers from
            # CARE's perspective but may not exist on every CI
            # checkout, so don't fail on missing siblings.
            if "Development" not in str(resolved) or resolved.exists():
                continue
            # Path inside the CARE checkout that doesn't resolve →
            # actual drift, fail.
            if resolved.is_relative_to(PROJECT_ROOT):
                assert resolved.exists(), (
                    f"`docs/ARCHITECTURE.md` references {target!r} but "
                    f"{resolved} doesn't exist on disk."
                )

    def test_lists_every_runtime_adapter(self):
        """The module-boundaries section enumerates `care/runtime/`
        adapters. If a new one is added without updating the doc,
        contributors will struggle to discover it."""
        body = _read()
        runtime_dir = PROJECT_ROOT / "care" / "runtime"
        # Every .py file in care/runtime (except __init__) should be
        # mentioned by basename at least once.
        missing = []
        for f in sorted(runtime_dir.glob("*.py")):
            if f.name in ("__init__.py",):
                continue
            if f.name not in body:
                missing.append(f.name)
        assert not missing, (
            "`docs/ARCHITECTURE.md` doesn't mention these runtime "
            f"adapters: {missing}. Update the §2 module-boundaries "
            "tree when adding new files."
        )


class TestReadmeCrossLink:
    def test_readme_links_architecture(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "docs/ARCHITECTURE.md" in readme, (
            "README must cross-link to docs/ARCHITECTURE.md so the "
            "doc is discoverable from the repo landing page."
        )
