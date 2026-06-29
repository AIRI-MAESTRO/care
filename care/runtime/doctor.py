"""`care doctor` diagnostics layer (TODO §1 P1).

Aggregates the existing first-run probes + boot-time
environment audit into a single human-readable report. The
CLI `care doctor` subcommand consumes this; future
OnboardingScreen `?` buttons will share the same registry
so the wizard's per-step "test connection" UX matches the
shell tool.

Five sections in the report:

1. **Environment** — CARE_* env vars currently set
   (values redacted for any key containing `key` /
   `token` / `secret`).
2. **Config** — path to the active TOML + whether it
   exists.
3. **Filesystem** — output of
   :func:`care.runtime.user_paths.ensure_user_dirs` so the
   user sees which of the three XDG dirs are healthy.
4. **Extras** — which optional Python dependencies the
   current interpreter can import (`mmar_carl`, `openai`,
   `anthropic`, `docker`, `e2b`, `plotext`, `pypdf`,
   `rich_pixels`, `textual`).
5. **Probes** — output of :func:`run_all_probes` for the
   memory / mage / platform services configured in the
   loaded `CareConfig`.

Sections 1-4 are pure (no network); section 5 is async +
exercises the actual upstream services. The pure helpers
have their own tests; the async aggregate is exercised
end-to-end via the `care doctor` CLI test.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from care.config import DEFAULT_CONFIG_PATH


_REDACT_TOKENS: tuple[str, ...] = ("KEY", "TOKEN", "SECRET")


_EXTRAS: tuple[tuple[str, str], ...] = (
    # (extras-display-name, module-import-name).
    ("mmar_carl", "mmar_carl"),
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("docker", "docker"),
    ("e2b", "e2b"),
    ("plotext", "plotext"),
    ("pypdf", "pypdf"),
    ("rich_pixels", "rich_pixels"),
    ("textual", "textual"),
    # Document text-extraction for @-file refs (care/runtime/document_extract).
    ("python_docx", "docx"),
    ("python_pptx", "pptx"),
    ("openpyxl", "openpyxl"),
    ("odfpy", "odf"),
    ("striprtf", "striprtf"),
)


@dataclass(frozen=True)
class EnvVarRow:
    """One CARE_* env var the doctor surfaces."""

    name: str
    value: str
    redacted: bool


@dataclass(frozen=True)
class ExtraStatus:
    """One optional dependency's import status."""

    name: str
    installed: bool
    version: str = ""


@dataclass(frozen=True)
class DoctorReport:
    """Aggregate of every pure-side diagnostic.

    The async probe results land here too once
    :func:`care.first_run.run_all_probes` returns.
    """

    config_path: Path
    config_exists: bool
    env_vars: tuple[EnvVarRow, ...] = ()
    extras: tuple[ExtraStatus, ...] = ()
    user_path_lines: tuple[str, ...] = ()
    probes_text: str = ""

    def format_text(self) -> str:
        """Render the report as a single multi-line string
        the CLI prints to stdout."""
        lines: list[str] = []
        lines.append("== Config ==")
        lines.append(f"  path: {self.config_path}")
        lines.append(
            f"  exists: {'yes' if self.config_exists else 'no'}"
        )
        lines.append("")
        lines.append("== Environment ==")
        if not self.env_vars:
            lines.append("  (no CARE_* env vars set)")
        else:
            for row in self.env_vars:
                lines.append(f"  {row.name} = {row.value}")
        lines.append("")
        lines.append("== Filesystem ==")
        if not self.user_path_lines:
            lines.append("  (no user-path report)")
        else:
            for line in self.user_path_lines:
                lines.append(f"  {line}")
        lines.append("")
        lines.append("== Extras ==")
        for row in self.extras:
            badge = "✓" if row.installed else "·"
            label = (
                f"{row.name} {row.version}".rstrip()
                if row.installed else row.name
            )
            lines.append(f"  {badge} {label}")
        if self.probes_text:
            lines.append("")
            lines.append("== Probes ==")
            for line in self.probes_text.splitlines():
                lines.append(f"  {line}")
        return "\n".join(lines)


def collect_env_vars(
    env: dict[str, str] | None = None,
) -> tuple[EnvVarRow, ...]:
    """Snapshot the CARE_* env vars from ``env`` (default
    :data:`os.environ`).

    Values containing any of the :data:`_REDACT_TOKENS`
    substrings in the *name* are masked to
    ``"<redacted N chars>"`` so a user pasting the report
    publicly doesn't leak their secrets.
    """
    source = env if env is not None else dict(os.environ)
    rows: list[EnvVarRow] = []
    for name in sorted(source):
        if not name.startswith("CARE_"):
            continue
        value = source[name]
        if _should_redact(name):
            display = (
                f"<redacted {len(value)} chars>"
                if value
                else "<empty>"
            )
            rows.append(
                EnvVarRow(
                    name=name, value=display, redacted=True,
                )
            )
        else:
            rows.append(
                EnvVarRow(
                    name=name, value=value, redacted=False,
                )
            )
    return tuple(rows)


def _should_redact(name: str) -> bool:
    upper = name.upper()
    return any(token in upper for token in _REDACT_TOKENS)


def collect_extras() -> tuple[ExtraStatus, ...]:
    """Probe each optional dependency's import availability.

    Doesn't actually import the module (that would pay the
    full import cost); uses :func:`importlib.util.find_spec`
    so the doctor stays cheap. When the module IS importable
    the helper makes a best-effort to pull a ``__version__``
    attribute — wrapped in a try/except so a malformed
    package can't crash the doctor.
    """
    rows: list[ExtraStatus] = []
    for display_name, module_name in _EXTRAS:
        spec = None
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            rows.append(
                ExtraStatus(name=display_name, installed=False),
            )
            continue
        version = _safe_version(module_name)
        rows.append(
            ExtraStatus(
                name=display_name,
                installed=True,
                version=version,
            ),
        )
    return tuple(rows)


def _safe_version(module_name: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(module_name)
    except PackageNotFoundError:
        return ""
    except Exception:  # noqa: BLE001
        return ""


def collect_user_path_lines() -> tuple[str, ...]:
    """Run :func:`ensure_user_dirs` + project to lines.

    Returns the lines from :meth:`UserPathReport.format_text`
    as a tuple so the doctor report can be deep-frozen.
    """
    from care.runtime.user_paths import ensure_user_dirs

    report = ensure_user_dirs()
    return tuple(report.format_text().splitlines())


def compose_report(
    *,
    config_path: Path | None = None,
    env: dict[str, str] | None = None,
    probes_text: str = "",
) -> DoctorReport:
    """Build a :class:`DoctorReport` from the pure helpers.

    Async probes are run separately (the CLI handler does it)
    and the resulting text is passed in via ``probes_text``.
    """
    target = config_path or DEFAULT_CONFIG_PATH
    return DoctorReport(
        config_path=target,
        config_exists=target.exists(),
        env_vars=collect_env_vars(env),
        extras=collect_extras(),
        user_path_lines=collect_user_path_lines(),
        probes_text=probes_text,
    )


__all__ = [
    "DoctorReport",
    "EnvVarRow",
    "ExtraStatus",
    "collect_env_vars",
    "collect_extras",
    "collect_user_path_lines",
    "compose_report",
]
