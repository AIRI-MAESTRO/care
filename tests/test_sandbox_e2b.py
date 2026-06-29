"""Tests for ``care.sandbox.e2b.E2BSandboxBackend`` (TODO §6.1 P1).

The e2b SDK isn't installed in CARE's dev env so the missing-dep
branch gets exercised for real. Functional tests use a stub
``Sandbox`` injected via ``sandbox_factory=`` — same testability
pattern Docker uses.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from care.sandbox.backend import SandboxError, SandboxTimeoutError
from care.sandbox.e2b import (
    DEFAULT_TEMPLATE,
    SKILL_DIR,
    WORKSPACE_DIR,
    E2BSandboxBackend,
)


# ---------------------------------------------------------------------------
# Stub sandbox
# ---------------------------------------------------------------------------


@dataclass
class _FilesSpy:
    writes: list[dict[str, Any]] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)
    storage: dict[str, bytes] = field(default_factory=dict)
    raise_on_write: Exception | None = None
    raise_on_read: Exception | None = None

    def write(self, path: str, data: Any) -> None:
        self.writes.append({"path": path, "data": data})
        if self.raise_on_write is not None:
            raise self.raise_on_write
        if isinstance(data, bytes):
            self.storage[path] = data
        elif isinstance(data, str):
            self.storage[path] = data.encode("utf-8")
        else:
            self.storage[path] = bytes(data)

    def read(self, path: str) -> Any:
        self.reads.append(path)
        if self.raise_on_read is not None:
            raise self.raise_on_read
        if path not in self.storage:
            raise FileNotFoundError(path)
        return self.storage[path]


@dataclass
class _CommandsSpy:
    calls: list[dict[str, Any]] = field(default_factory=list)
    next_exit_code: int = 0
    next_stdout: Any = "ok"
    next_stderr: Any = ""
    raise_on_run: Exception | None = None

    def run(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        self.calls.append(
            {
                "cmd": cmd,
                "envs": dict(envs or {}),
                "cwd": cwd,
                "timeout": timeout,
            }
        )
        if self.raise_on_run is not None:
            raise self.raise_on_run

        class _R:
            def __init__(self, ec, out, err):
                self.exit_code = ec
                self.stdout = out
                self.stderr = err

        return _R(self.next_exit_code, self.next_stdout, self.next_stderr)


@dataclass
class _StubSandbox:
    sandbox_id: str = "vm-1234"
    template: str = ""
    api_key: str | None = None
    timeout: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)
    killed: bool = False
    files: _FilesSpy = field(default_factory=_FilesSpy)
    commands: _CommandsSpy = field(default_factory=_CommandsSpy)
    raise_on_kill: Exception | None = None

    def kill(self) -> None:
        if self.raise_on_kill is not None:
            raise self.raise_on_kill
        self.killed = True


@dataclass
class _FactorySpy:
    calls: list[dict[str, Any]] = field(default_factory=list)
    next_sandbox: _StubSandbox | None = None
    raise_on_call: Exception | None = None

    def __call__(self, **kwargs: Any) -> _StubSandbox:
        self.calls.append(kwargs)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        sandbox = self.next_sandbox or _StubSandbox()
        sandbox.template = kwargs.get("template", "")
        sandbox.api_key = kwargs.get("api_key")
        sandbox.timeout = kwargs.get("timeout", 0.0)
        sandbox.metadata = dict(kwargs.get("metadata") or {})
        return sandbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill(tmp_path: Path, sha: str = "e" * 64, files: dict[str, bytes] | None = None) -> Any:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    payload = files or {"SKILL.md": b"---\nname: x\n---\n"}
    for rel, data in payload.items():
        target = skill_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    class _Skill:
        pass

    s = _Skill()
    s.sha256 = sha
    s.path = skill_dir
    s.allowed_tools = ["Read"]
    return s


def _backend(
    factory: _FactorySpy | None = None,
    **kwargs: Any,
) -> tuple[E2BSandboxBackend, _FactorySpy]:
    f = factory or _FactorySpy()
    backend = E2BSandboxBackend(
        sandbox_factory=f,
        **kwargs,
    )
    return backend, f


# ---------------------------------------------------------------------------
# Construction + lazy import
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self):
        backend, _ = _backend()
        assert backend.name == "e2b"
        assert backend.unsafe is False
        assert backend._template == "base"
        assert DEFAULT_TEMPLATE == "base"

    def test_template_override(self):
        backend, _ = _backend(template="custom-image")
        assert backend._template == "custom-image"

    def test_missing_sdk_raises(self):
        try:
            import e2b  # noqa: F401
        except ImportError:
            backend = E2BSandboxBackend()
            skill = _make_skill_inline()
            with pytest.raises(SandboxError, match="e2b SDK is not installed"):
                backend._make_sandbox(skill)
        else:
            pytest.skip("e2b SDK is installed; missing-dep path skipped")


def _make_skill_inline() -> Any:
    class _Skill:
        sha256 = "0" * 64
        path = Path("/tmp")
        allowed_tools: list[str] = []

    return _Skill()


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_factory_called_with_documented_kwargs(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(
            factory=factory,
            template="my-image",
            api_key="k-secret",
            default_timeout=600.0,
            metadata={"env": "prod"},
        )
        skill = _make_skill(tmp_path, sha="d" * 64)
        handle = asyncio.run(backend.prepare(skill, workspace=None))

        assert len(factory.calls) == 1
        call = factory.calls[0]
        assert call["template"] == "my-image"
        assert call["api_key"] == "k-secret"
        assert call["timeout"] == 600.0
        # CARE-side labels always added.
        assert call["metadata"]["care.sandbox"] == "true"
        assert call["metadata"]["care.sandbox.skill_sha256"] == "d" * 64
        # Caller metadata still flows through.
        assert call["metadata"]["env"] == "prod"
        # Handle carries the sandbox.
        assert handle.backend_name == "e2b"
        assert handle.skill_sha256 == "d" * 64
        assert handle.workspace == Path(WORKSPACE_DIR)
        assert handle.extras["sandbox_id"] == "vm-1234"
        assert handle.extras["skill_mount"] == SKILL_DIR
        assert handle.extras["workspace_mount"] == WORKSPACE_DIR

    def test_factory_failure_wraps(self, tmp_path: Path):
        factory = _FactorySpy(raise_on_call=RuntimeError("api down"))
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        with pytest.raises(SandboxError, match="failed to provision e2b sandbox"):
            asyncio.run(backend.prepare(skill, workspace=None))

    def test_skill_path_must_exist(self, tmp_path: Path):
        backend, _ = _backend()

        class _BadSkill:
            sha256 = "c" * 64
            path = tmp_path / "no-skill"
            allowed_tools: list[str] = []

        with pytest.raises(SandboxError, match="skill path is not"):
            asyncio.run(backend.prepare(_BadSkill(), workspace=None))

    def test_staging_uploads_every_file(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(
            tmp_path,
            files={
                "SKILL.md": b"---\nname: x\n---\n",
                "scripts/run.py": b"print('hi')",
                "data/notes.txt": b"abc",
            },
        )
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        sandbox = handle.extras["sandbox"]
        paths = sorted(call["path"] for call in sandbox.files.writes)
        assert paths == [
            f"{SKILL_DIR}/SKILL.md",
            f"{SKILL_DIR}/data/notes.txt",
            f"{SKILL_DIR}/scripts/run.py",
        ]
        # Contents staged verbatim.
        assert sandbox.files.storage[f"{SKILL_DIR}/scripts/run.py"] == b"print('hi')"

    def test_staging_failure_wraps(self, tmp_path: Path):
        factory = _FactorySpy()
        stub = _StubSandbox()
        stub.files.raise_on_write = RuntimeError("disk full")
        factory.next_sandbox = stub
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        with pytest.raises(SandboxError, match="failed to stage"):
            asyncio.run(backend.prepare(skill, workspace=None))


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_dispatches_to_commands_run(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        sandbox = handle.extras["sandbox"]
        sandbox.commands.next_stdout = "hello"
        sandbox.commands.next_stderr = ""
        result = asyncio.run(
            backend.run(
                handle,
                ["python", "-c", "print('hi')"],
                env={"FOO": "bar"},
                network="open",
            )
        )
        assert result.exit_code == 0
        assert result.stdout == b"hello"
        assert result.network_enforced is False
        call = sandbox.commands.calls[0]
        # shell-quoted form.
        assert call["cmd"] == "python -c 'print('\"'\"'hi'\"'\"')'"
        assert call["envs"] == {"FOO": "bar"}
        assert call["cwd"] == WORKSPACE_DIR

    def test_run_str_outputs_become_bytes(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        sandbox = handle.extras["sandbox"]
        sandbox.commands.next_stdout = "string out"
        sandbox.commands.next_stderr = "string err"
        result = asyncio.run(backend.run(handle, ["echo"], network="open"))
        assert result.stdout == b"string out"
        assert result.stderr == b"string err"

    def test_run_propagates_exit_code(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        sandbox = handle.extras["sandbox"]
        sandbox.commands.next_exit_code = 5
        result = asyncio.run(backend.run(handle, ["false"], network="open"))
        assert result.exit_code == 5
        assert result.ok is False

    def test_empty_cmd_raises(self, tmp_path: Path):
        backend, _ = _backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        with pytest.raises(SandboxError, match="non-empty"):
            asyncio.run(backend.run(handle, [], network="open"))

    def test_network_none_raises(self, tmp_path: Path):
        backend, _ = _backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        with pytest.raises(SandboxError, match="doesn't support network='none'"):
            asyncio.run(backend.run(handle, ["echo"], network="none"))

    def test_network_skill_declared_raises(self, tmp_path: Path):
        backend, _ = _backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        with pytest.raises(SandboxError, match="skill_declared"):
            asyncio.run(backend.run(handle, ["echo"], network="skill_declared"))

    def test_timeout_error_becomes_sandbox_timeout(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        sandbox = handle.extras["sandbox"]
        sandbox.commands.raise_on_run = TimeoutError("timed out")
        with pytest.raises(SandboxTimeoutError) as excinfo:
            asyncio.run(backend.run(handle, ["sleep", "30"], timeout=5.0, network="open"))
        assert excinfo.value.result is not None
        assert excinfo.value.result.timed_out is True

    def test_generic_exception_wraps(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        sandbox = handle.extras["sandbox"]
        sandbox.commands.raise_on_run = RuntimeError("api error")
        with pytest.raises(SandboxError, match="sandbox.commands.run failed"):
            asyncio.run(backend.run(handle, ["echo"], network="open"))

    def test_run_after_cleanup_raises(self, tmp_path: Path):
        backend, _ = _backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        handle.extras["sandbox"] = None
        with pytest.raises(SandboxError, match="has no live sandbox"):
            asyncio.run(backend.run(handle, ["echo"], network="open"))


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_write_then_read(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        asyncio.run(backend.write_file(handle, "out/result.txt", b"data"))
        out = asyncio.run(backend.read_file(handle, "out/result.txt"))
        assert out == b"data"
        sandbox = handle.extras["sandbox"]
        # write went to the workspace-prefixed path.
        target_path = f"{WORKSPACE_DIR}/out/result.txt"
        write_paths = [call["path"] for call in sandbox.files.writes]
        # First N writes were the staging; the new one is the
        # workspace write.
        assert target_path in write_paths

    def test_read_missing_file_raises(self, tmp_path: Path):
        backend, _ = _backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        with pytest.raises(SandboxError, match="file not found"):
            asyncio.run(backend.read_file(handle, "no-such.txt"))

    def test_read_absolute_path_blocked(self, tmp_path: Path):
        backend, _ = _backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        with pytest.raises(SandboxError, match="absolute path not allowed"):
            asyncio.run(backend.read_file(handle, "/etc/passwd"))

    def test_read_traversal_blocked(self, tmp_path: Path):
        backend, _ = _backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        with pytest.raises(SandboxError, match="escapes workspace"):
            asyncio.run(backend.read_file(handle, "../escape"))

    def test_write_traversal_blocked(self, tmp_path: Path):
        backend, _ = _backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        with pytest.raises(SandboxError, match="escapes workspace"):
            asyncio.run(backend.write_file(handle, "../escape", b"x"))

    def test_empty_path_rejected(self, tmp_path: Path):
        backend, _ = _backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        with pytest.raises(SandboxError, match="must not be empty"):
            asyncio.run(backend.read_file(handle, ""))

    def test_write_error_wraps(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        sandbox = handle.extras["sandbox"]
        sandbox.files.raise_on_write = RuntimeError("disk full")
        with pytest.raises(SandboxError, match="could not write"):
            asyncio.run(backend.write_file(handle, "x.txt", b"x"))


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_kills_sandbox(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        sandbox = handle.extras["sandbox"]
        asyncio.run(backend.cleanup(handle))
        assert sandbox.killed is True
        assert handle.extras["cleaned"] is True

    def test_cleanup_idempotent(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        sandbox = handle.extras["sandbox"]
        asyncio.run(backend.cleanup(handle))
        sandbox.killed = False
        asyncio.run(backend.cleanup(handle))
        assert sandbox.killed is False

    def test_cleanup_tolerates_kill_failure(self, tmp_path: Path):
        factory = _FactorySpy()
        backend, _ = _backend(factory=factory)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        sandbox = handle.extras["sandbox"]
        sandbox.raise_on_kill = RuntimeError("already gone")
        # Must not raise.
        asyncio.run(backend.cleanup(handle))
        assert handle.extras["cleaned"] is True


# ---------------------------------------------------------------------------
# Real e2b integration (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("CARE_TEST_E2B"),
    reason="set CARE_TEST_E2B=1 + E2B_API_KEY to run real-cloud integration",
)
class TestRealE2BIntegration:
    def test_round_trip(self, tmp_path: Path):
        backend = E2BSandboxBackend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        try:
            result = asyncio.run(
                backend.run(handle, ["echo", "hi"], network="open")
            )
            assert result.exit_code == 0
            assert b"hi" in result.stdout
        finally:
            asyncio.run(backend.cleanup(handle))
