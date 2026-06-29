"""Audit `.github/workflows/*.yml` (TODO §7 P0 PyPI publish flow).

Guards two release-critical contracts:

* `publish.yml` triggers on a `v*` tag, builds with `uv build`, and
  publishes via PyPI Trusted Publishing (OIDC — no PyPI token in
  repo secrets). A misconfigured permission scope or wrong
  trigger would silently swallow the publish step.
* `ci.yml` runs ruff + pytest on every supported Python version
  declared in `pyproject.toml`'s classifiers, so a future
  classifier update doesn't drift away from what CI actually
  exercises.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = PROJECT_ROOT / ".github" / "workflows"


def _load_workflow(name: str) -> dict:
    path = WORKFLOW_DIR / name
    assert path.is_file(), f"missing workflow file: {path}"
    return yaml.safe_load(path.read_text())


def _supported_python_versions() -> set[str]:
    data = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text()
    )
    out: set[str] = set()
    for line in data["project"]["classifiers"]:
        prefix = "Programming Language :: Python :: 3."
        if line.startswith(prefix):
            tail = line[len(prefix):].strip()
            if tail.isdigit():
                out.add(f"3.{tail}")
    return out


class TestPublishWorkflow:
    def test_publish_file_exists(self) -> None:
        assert (WORKFLOW_DIR / "publish.yml").is_file()

    def test_publish_triggers_on_version_tag(self) -> None:
        wf = _load_workflow("publish.yml")
        # PyYAML may parse the `on:` key as Python's `True` since
        # YAML 1.1 treats `on` as a boolean alias.
        triggers = wf.get("on") or wf.get(True) or {}
        push = triggers.get("push") or {}
        tags = push.get("tags") or []
        assert "v*" in tags, (
            f"publish workflow must trigger on v* tag (got {tags!r})"
        )

    def test_publish_uses_oidc_trusted_publishing(self) -> None:
        wf = _load_workflow("publish.yml")
        jobs = wf["jobs"]
        assert "publish" in jobs, "missing `publish` job"
        publish_job = jobs["publish"]
        # OIDC requires `id-token: write` at the job level.
        perms = publish_job.get("permissions") or {}
        assert perms.get("id-token") == "write", (
            "publish job must request id-token: write for "
            "PyPI Trusted Publishing"
        )
        # The pypa publish action must run with no token: secret
        # (OIDC-only).
        steps = publish_job.get("steps") or []
        publish_steps = [
            s for s in steps
            if "pypa/gh-action-pypi-publish" in str(s.get("uses") or "")
        ]
        assert publish_steps, (
            "publish job must include pypa/gh-action-pypi-publish step"
        )
        for step in publish_steps:
            with_block = step.get("with") or {}
            assert "password" not in with_block, (
                "OIDC trusted publishing must NOT pass a "
                "password — that's the legacy token flow"
            )

    def test_publish_environment_named_pypi(self) -> None:
        wf = _load_workflow("publish.yml")
        publish_job = wf["jobs"]["publish"]
        env = publish_job.get("environment")
        # Environment may be a string or a mapping with `name:`.
        if isinstance(env, str):
            name = env
        elif isinstance(env, dict):
            name = env.get("name")
        else:
            name = None
        assert name == "pypi", (
            f"publish job must gate on the `pypi` environment "
            f"(got {name!r})"
        )

    def test_publish_builds_with_uv(self) -> None:
        wf = _load_workflow("publish.yml")
        build_job = wf["jobs"]["build"]
        steps = build_job.get("steps") or []
        run_lines = " ".join(
            str(s.get("run", "")) for s in steps
        )
        assert "uv build" in run_lines, (
            "build job must invoke `uv build`"
        )

    def test_publish_uploads_artifacts(self) -> None:
        wf = _load_workflow("publish.yml")
        build_job = wf["jobs"]["build"]
        steps = build_job.get("steps") or []
        uses = [str(s.get("uses") or "") for s in steps]
        assert any(
            u.startswith("actions/upload-artifact") for u in uses
        ), "build job must upload the dist/ artifacts"

        publish_job = wf["jobs"]["publish"]
        steps = publish_job.get("steps") or []
        uses = [str(s.get("uses") or "") for s in steps]
        assert any(
            u.startswith("actions/download-artifact") for u in uses
        ), "publish job must download the dist/ artifacts"


class TestCiWorkflow:
    def test_ci_file_exists(self) -> None:
        assert (WORKFLOW_DIR / "ci.yml").is_file()

    def test_ci_triggers_on_pr_and_main_push(self) -> None:
        wf = _load_workflow("ci.yml")
        triggers = wf.get("on") or wf.get(True) or {}
        assert "pull_request" in triggers, (
            "CI must run on every PR"
        )
        push = triggers.get("push") or {}
        assert "main" in (push.get("branches") or []), (
            "CI must also run on push to main"
        )

    def test_ci_matrix_matches_supported_python(self) -> None:
        wf = _load_workflow("ci.yml")
        jobs = wf["jobs"]
        assert "lint-and-test" in jobs, (
            "ci.yml must define a lint-and-test job"
        )
        job = jobs["lint-and-test"]
        matrix = (
            (job.get("strategy") or {}).get("matrix") or {}
        )
        # Matrix lists Python versions as strings to keep "3.10"
        # from being parsed as a float.
        py = {str(v) for v in (matrix.get("python") or [])}
        supported = _supported_python_versions()
        assert supported, (
            "pyproject.toml classifiers must declare at least one "
            "Python 3.x version"
        )
        assert py == supported, (
            f"CI matrix {sorted(py)} drifted from pyproject "
            f"classifiers {sorted(supported)} — keep them aligned"
        )

    def test_ci_runs_ruff(self) -> None:
        wf = _load_workflow("ci.yml")
        steps = wf["jobs"]["lint-and-test"].get("steps") or []
        run_lines = " ".join(
            str(s.get("run", "")) for s in steps
        )
        assert "ruff check" in run_lines, (
            "CI must run `ruff check`"
        )

    def test_ci_runs_pytest(self) -> None:
        wf = _load_workflow("ci.yml")
        steps = wf["jobs"]["lint-and-test"].get("steps") or []
        run_lines = " ".join(
            str(s.get("run", "")) for s in steps
        )
        assert "pytest" in run_lines, "CI must run pytest"
