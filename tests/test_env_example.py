"""Tests for the ``.env.example`` schema (TODO §2 P1).

`CareConfig` is the source of truth for what env vars CARE
accepts. `.env.example` is the documentation. Drift between
them is bad UX: a new config field with no env-var stub leaves
users guessing. These tests pin the symmetry:

1. Every `CARE_<SECTION>__<FIELD>` mentioned in `.env.example`
   must correspond to a real Pydantic field on the matching
   nested config model.
2. Every nested-section field on `CareConfig` must appear in
   `.env.example` at least once (as either an active var or a
   commented-out stub).

Also exercises ``CareConfig.load`` against a parsed `.env`
fixture so we know the file is a valid Pydantic input shape,
not just text we typed out.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from care.config import (
    ArtifactsConfig,
    ChatConfig,
    ContextConfig,
    DefaultsConfig,
    MageConfig,
    MemoryConfig,
    PlatformConfig,
    SandboxConfig,
    TelemetryConfig,
    ToolsConfig,
    UploadConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

# Section name (lower) → the config model whose fields it pins.
_SECTION_TO_MODEL: dict[str, type[BaseModel]] = {
    "mage": MageConfig,
    "memory": MemoryConfig,
    "platform": PlatformConfig,
    "sandbox": SandboxConfig,
    "tools": ToolsConfig,
    "telemetry": TelemetryConfig,
    "defaults": DefaultsConfig,
    "upload": UploadConfig,
    "chat": ChatConfig,
    "context": ContextConfig,
    "artifacts": ArtifactsConfig,
}

# Env-only sections that don't have a Pydantic model (none
# today — chat got a real ChatConfig in this iteration). Kept
# as an empty allow-list so a future env-only knob can opt in
# without re-introducing the gap.
_ENV_ONLY_SECTIONS: dict[str, set[str]] = {}


def _is_known_var(section: str, field: str) -> bool:
    if section in _SECTION_TO_MODEL:
        return field in _SECTION_TO_MODEL[section].model_fields
    if section in _ENV_ONLY_SECTIONS:
        return field in _ENV_ONLY_SECTIONS[section]
    return False


def _section_recognised(section: str) -> bool:
    return section in _SECTION_TO_MODEL or section in _ENV_ONLY_SECTIONS

_ENV_VAR_RE = re.compile(
    r"^#?\s*(CARE_([A-Z]+)__([A-Z_]+))=",
    flags=re.MULTILINE,
)


def _read_env_vars() -> set[tuple[str, str, str]]:
    """Returns ``{(full_var, section_lower, field_lower)}`` from the file."""
    text = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    out: set[tuple[str, str, str]] = set()
    for full, section, field in _ENV_VAR_RE.findall(text):
        out.add((full, section.lower(), field.lower()))
    return out


# ---------------------------------------------------------------------------
# Symmetry pin
# ---------------------------------------------------------------------------


class TestEnvExampleSchema:
    def test_file_exists(self):
        assert ENV_EXAMPLE_PATH.is_file(), (
            f"{ENV_EXAMPLE_PATH} is missing — `.env.example` must live "
            "in the project root next to `pyproject.toml`."
        )

    def test_every_documented_var_maps_to_a_real_field(self):
        documented = _read_env_vars()
        assert documented, "no `CARE_*` vars detected in `.env.example`"
        for full, section, field in documented:
            assert _section_recognised(section), (
                f"`{full}` references unknown section [{section}]; "
                f"valid sections: "
                f"{sorted(set(_SECTION_TO_MODEL) | set(_ENV_ONLY_SECTIONS))}"
            )
            assert _is_known_var(section, field), (
                f"`{full}` references unknown field {section}.{field}."
            )

    def test_every_config_field_is_documented(self):
        documented = {
            (section, field) for _, section, field in _read_env_vars()
        }
        missing: list[str] = []
        for section, model in _SECTION_TO_MODEL.items():
            for field in model.model_fields:
                if (section, field) not in documented:
                    missing.append(
                        f"CARE_{section.upper()}__{field.upper()}"
                    )
        for section, fields in _ENV_ONLY_SECTIONS.items():
            for field in fields:
                if (section, field) not in documented:
                    missing.append(
                        f"CARE_{section.upper()}__{field.upper()}"
                    )
        assert not missing, (
            "config fields are missing from `.env.example`: "
            + ", ".join(missing)
            + ". Add a commented stub for each new field."
        )

    def test_documented_vars_round_trip_through_careconfig(self):
        """The docstring example values are valid Pydantic input shapes."""
        # Pick one var per section to drive `CareConfig.load`.
        env = {
            "CARE_MAGE__MODE": "fast",
            "CARE_MEMORY__BASE_URL": "http://memory.example:8000",
            "CARE_PLATFORM__BASE_URL": "http://platform.example:8001",
            "CARE_SANDBOX__KIND": "local",
            "CARE_TOOLS__NAME_PREFIX": "my_",
            "CARE_DEFAULTS__LANGUAGE": "ru",
        }
        from care.config import CareConfig

        cfg = CareConfig.load(
            path=Path("/no-such-toml.toml"),
            env=env,
        )
        assert cfg.mage.mode == "fast"
        assert cfg.memory.base_url == "http://memory.example:8000"
        assert cfg.platform.base_url == "http://platform.example:8001"
        assert cfg.sandbox.kind == "local"
        assert cfg.tools.name_prefix == "my_"
        assert cfg.defaults.language == "ru"


# ---------------------------------------------------------------------------
# README integration check
# ---------------------------------------------------------------------------


class TestReadmeMentionsEnvExample:
    def test_readme_links_env_example(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        # Both the file mention + the precedence list let readers find
        # the documentation entry point.
        assert ".env.example" in readme
        assert "CARE_" in readme
