"""uvx packaging smoke test (TODO §1 P0).

Confirms `care` boots end-to-end through the same install path
`uvx care` will use after the v0.1 PyPI publish. Catches the
class of bug where a top-level import resolves only because the
dev venv happens to have an optional extra installed — under
`uvx --from . care --help`, the wheel's *required* deps are the
only thing on `sys.path`, so any accidental dependency on
``carl`` / ``openai`` / ``anthropic`` / ``docker`` / ``e2b`` /
``pypdf`` (a required dep that some flows accidentally treat as
optional) shows up here.

Two layers:

* **Fast path** (`test_help_exits_zero_under_uv_run`): always
  runs in CI. Spawns ``uv run python -m care --help`` against the
  current venv. Confirms the CLI dispatch chain imports cleanly
  + `--help` exits 0. Cheap (~1 s).
* **Slow path** (`test_help_exits_zero_under_uvx`): gated behind
  the ``CARE_RUN_UVX_SMOKE`` env var so CI can opt in on tag
  builds / nightly. Spawns
  ``uvx --from <project_root> care --help`` against a freshly
  built isolated environment. Catches required-dep gaps that
  the fast path misses (the dev venv has everything; the uvx
  env has only what the wheel declares). Skipped + announced
  when uvx isn't on PATH.

Both layers assert the same anchor strings — "CARE" + the
"Running with no subcommand launches the TUI." preamble from
``care.cli.main``'s argparse description — so a future
description rewrite that drops either phrase will surface here
before it confuses a fresh user.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Anchor strings the `care --help` output is contractually required
# to contain. Kept short so a wording change that preserves the
# core meaning doesn't break the test — we want the bar to be
# "fresh users still see what this tool is", not "exact string
# match".
_HELP_ANCHORS: tuple[str, ...] = (
    "CARE",
    "Collaborative Agent Reasoning Ecosystem",
)


def _run_help(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run `care --help` via ``command`` and return the
    completed process. Captures stdout + stderr as text. Raises
    a clear failure if the subprocess times out (the default
    `TimeoutExpired` message is short on context).

    Always runs against `PROJECT_ROOT` so relative paths in
    ``--from .`` resolve to the repo root regardless of where
    pytest was invoked from.
    """
    try:
        return subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # `pytest.fail` raises `_pytest.outcomes.Failed` (NoReturn),
        # but the type checker doesn't always see it — the
        # explicit `raise AssertionError(...) from exc` keeps the
        # signature honest and preserves the original exception
        # chain for debugging from CI logs.
        pytest.fail(
            f"`{' '.join(command)}` exceeded {timeout}s timeout. "
            f"stdout so far: {exc.stdout!r}; stderr so far: {exc.stderr!r}"
        )
        raise AssertionError("unreachable") from exc


def _assert_help_output(proc: subprocess.CompletedProcess[str]) -> None:
    """Shared assertions for either smoke path. Surfaces stdout
    + stderr in the failure message so debugging from CI logs
    doesn't need a local re-run."""
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    for anchor in _HELP_ANCHORS:
        assert anchor in combined, (
            f"missing anchor {anchor!r} in --help output:\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    # The argparse `prog` line ships first; ensure it's the
    # canonical `care` invocation regardless of how the entry
    # point was launched. Catches accidental `python -m care`
    # leaking through as the displayed prog.
    assert "usage: care" in combined, (
        f"usage line not anchored on `care`:\n{combined[:400]}"
    )


class TestFastPath:
    """Runs every CI cycle — single subprocess, current venv."""

    def test_help_exits_zero_under_uv_run(self):
        # `uv run --no-sync` skips the lock check (the test
        # already runs under `uv run pytest` so the env is
        # ready) — keeps the smoke under ~1 s on a warm cache.
        # Falls back to the bare interpreter when `uv` isn't on
        # PATH (rare for CARE devs but possible on a stripped
        # CI image — the bare interpreter path still validates
        # the import surface).
        if shutil.which("uv"):
            command = [
                "uv", "run", "--no-sync", "python",
                "-c",
                "from care.cli import main; main(['--help'])",
            ]
        else:
            command = [
                sys.executable, "-c",
                "from care.cli import main; main(['--help'])",
            ]
        proc = _run_help(command, timeout=30.0)
        _assert_help_output(proc)


@pytest.mark.skipif(
    not shutil.which("uvx"),
    reason="uvx not on PATH — skipping the isolated-install smoke.",
)
@pytest.mark.skipif(
    os.environ.get("CARE_RUN_UVX_SMOKE", "").lower() not in {"1", "true", "yes"},
    reason=(
        "uvx isolated-install smoke is opt-in (set "
        "CARE_RUN_UVX_SMOKE=1 to run). Skipped by default so "
        "the regular `make test` stays fast — CI runs this on "
        "tag builds + nightly."
    ),
)
class TestUvxIsolatedInstall:
    """The full `uvx --from . care --help` round-trip — slow
    (~10-30 s wheel build + venv setup) so gated by env var.
    """

    def test_help_exits_zero_under_uvx(self):
        # Rely on the user's existing uv cache for upstream
        # deps (openai, pillow, pydantic-core, …) — `--no-cache`
        # forces a re-download per run, which routinely blows
        # past any sane timeout on a cold machine. The CARE
        # wheel itself + the path-source upstreams (`mmar-mage`,
        # `gigaevo-client`) are always re-resolved because
        # `--from <project_root>` invalidates them. That's what
        # we want to exercise — the wheel build + entry-point
        # dispatch path matches `uvx care` after PyPI publish.
        proc = _run_help(
            [
                "uvx",
                "--from", str(PROJECT_ROOT),
                "care", "--help",
            ],
            timeout=300.0,
        )
        _assert_help_output(proc)
