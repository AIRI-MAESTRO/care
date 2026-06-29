"""Tests for ``care.sandbox.firejail.FirejailSandboxBackend``
(TODO §6.1 P2).

Tests run on every CARE host — even non-Linux — by passing
``require_linux=False`` to the constructor and using a stub
``subprocess_runner`` that records argv without spawning real
processes. A separate opt-in real-binary test exercises the full
path when ``CARE_TEST_FIREJAIL=1`` is set on a Linux host with
firejail installed.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from care.sandbox.backend import SandboxError, SandboxTimeoutError
from care.sandbox.firejail import (
    DEFAULT_EXECUTABLE,
    FirejailSandboxBackend,
)


# ---------------------------------------------------------------------------
# Stub subprocess runner
# ---------------------------------------------------------------------------


@dataclass
class _RunnerSpy:
    """Records every runner invocation so tests can assert."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    next_exit_code: int = 0
    next_stdout: bytes = b""
    next_stderr: bytes = b""
    next_timed_out: bool = False

    async def __call__(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        stdin: bytes | None,
        timeout: float | None,
    ) -> tuple[int, bytes, bytes, bool]:
        self.calls.append(
            {
                "argv": list(argv),
                "cwd": cwd,
                "env": dict(env),
                "stdin": stdin,
                "timeout": timeout,
            }
        )
        return (
            self.next_exit_code,
            self.next_stdout,
            self.next_stderr,
            self.next_timed_out,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill(tmp_path: Path, sha: str = "f" * 64) -> Any:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text("---\nname: x\n---\n", encoding="utf-8")

    class _Skill:
        pass

    s = _Skill()
    s.sha256 = sha
    s.path = skill_dir
    s.allowed_tools = ["Read"]
    return s


def _mkws(tmp_path: Path) -> Path:
    """Pre-create the workspace dir prepare() expects to find."""
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _make_backend(
    executable: str = "firejail",
    *,
    runner: _RunnerSpy | None = None,
    default_cpu: float | None = None,
    default_mem: str | None = None,
) -> tuple[FirejailSandboxBackend, _RunnerSpy]:
    spy = runner or _RunnerSpy()
    backend = FirejailSandboxBackend(
        executable=executable,
        default_cpu=default_cpu,
        default_mem=default_mem,
        subprocess_runner=spy,
        require_linux=False,
    )
    return backend, spy


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_executable_is_firejail(self):
        backend, _ = _make_backend()
        assert backend.name == "firejail"
        assert backend.unsafe is False
        assert DEFAULT_EXECUTABLE == "firejail"

    def test_bwrap_executable_supported(self):
        backend, _ = _make_backend(executable="bwrap")
        assert backend.name == "bwrap"

    def test_absolute_path_executable_accepted(self):
        backend, _ = _make_backend(executable="/opt/local/bin/firejail")
        # `name` strips the dirname so audit logs stay tidy.
        assert backend.name == "firejail"

    def test_unsupported_executable_raises(self):
        with pytest.raises(SandboxError, match="unsupported sandbox executable"):
            FirejailSandboxBackend(
                executable="nsjail",
                subprocess_runner=_RunnerSpy(),
                require_linux=False,
            )

    def test_require_linux_blocks_non_linux(self, monkeypatch):
        import sys as _sys

        monkeypatch.setattr(_sys, "platform", "darwin")
        with pytest.raises(SandboxError, match="Linux-only"):
            FirejailSandboxBackend(
                executable="firejail",
                subprocess_runner=_RunnerSpy(),
                require_linux=True,
            )

    def test_require_linux_false_allows_construction_on_non_linux(self, monkeypatch):
        import sys as _sys

        monkeypatch.setattr(_sys, "platform", "darwin")
        # No raise — useful for cross-platform unit tests that only
        # exercise argv construction.
        FirejailSandboxBackend(
            executable="firejail",
            subprocess_runner=_RunnerSpy(),
            require_linux=False,
        )


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_prepare_creates_workspace_when_none(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        assert handle.workspace.exists()
        assert handle.workspace.is_dir()
        assert handle.extras["owns_workspace"] is True
        assert handle.extras["allowed_tools"] == ["Read"]
        # Cleanup tears it down.
        asyncio.run(backend.cleanup(handle))
        assert not handle.workspace.exists()

    def test_prepare_adopts_existing_workspace(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        ws = tmp_path / "ws"
        ws.mkdir()
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        assert handle.workspace == ws.resolve()
        assert handle.extras["owns_workspace"] is False

    def test_workspace_must_exist_when_supplied(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        # File exists but isn't a directory.
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(SandboxError, match="not a directory"):
            asyncio.run(backend.prepare(skill, workspace=f))

    def test_workspace_path_must_exist_at_all(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        with pytest.raises(SandboxError, match="workspace does not exist"):
            asyncio.run(backend.prepare(skill, workspace=tmp_path / "nope"))

    def test_skill_path_must_exist(self, tmp_path: Path):
        backend, _ = _make_backend()

        class _BadSkill:
            sha256 = "c" * 64
            path = tmp_path / "no-skill"
            allowed_tools: list[str] = []

        with pytest.raises(SandboxError, match="skill path is not a directory"):
            asyncio.run(backend.prepare(_BadSkill(), workspace=tmp_path))


# ---------------------------------------------------------------------------
# Firejail argv
# ---------------------------------------------------------------------------


class TestFirejailArgv:
    def test_default_argv_layout(self, tmp_path: Path):
        backend, spy = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        # Ensure workspace exists for prepare.
        Path(handle.workspace).mkdir(exist_ok=True)
        asyncio.run(backend.run(handle, ["python", "main.py"]))

        argv = spy.calls[0]["argv"]
        # Header
        assert argv[0] == "firejail"
        assert "--quiet" in argv
        assert "--noprofile" in argv
        assert "--private" in argv
        assert f"--chdir={handle.workspace}" in argv
        assert f"--whitelist={handle.workspace}" in argv
        assert f"--read-only={handle.extras['skill_path']}" in argv
        # Default network is `none`.
        assert "--net=none" in argv
        # `--` separator before the wrapped cmd.
        sep_idx = argv.index("--")
        assert argv[sep_idx + 1 :] == ["python", "main.py"]

    def test_open_network_omits_net_flag(self, tmp_path: Path):
        backend, spy = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(backend.run(handle, ["echo"], network="open"))
        argv = spy.calls[0]["argv"]
        # No --net=* anywhere (open inherits host).
        assert not any(a.startswith("--net=") for a in argv)

    def test_skill_declared_falls_back_to_net_none(self, tmp_path: Path):
        backend, spy = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(backend.run(handle, ["echo"], network="skill_declared"))
        argv = spy.calls[0]["argv"]
        assert "--net=none" in argv

    def test_rlimit_cpu_set_when_supplied(self, tmp_path: Path):
        backend, spy = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(backend.run(handle, ["echo"], cpu=8.0))
        argv = spy.calls[0]["argv"]
        assert "--rlimit-cpu=8" in argv

    def test_rlimit_as_set_when_mem_supplied(self, tmp_path: Path):
        backend, spy = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(backend.run(handle, ["echo"], mem="512m"))
        argv = spy.calls[0]["argv"]
        # 512m → 512 * 1024 * 1024.
        assert f"--rlimit-as={512 * 1024 * 1024}" in argv

    def test_default_cpu_mem_used_when_run_omits_them(self, tmp_path: Path):
        backend, spy = _make_backend(default_cpu=4.0, default_mem="1g")
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(backend.run(handle, ["echo"]))
        argv = spy.calls[0]["argv"]
        assert "--rlimit-cpu=4" in argv
        assert f"--rlimit-as={1024 * 1024 * 1024}" in argv

    def test_run_cpu_overrides_default(self, tmp_path: Path):
        backend, spy = _make_backend(default_cpu=4.0)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(backend.run(handle, ["echo"], cpu=16.0))
        argv = spy.calls[0]["argv"]
        assert "--rlimit-cpu=16" in argv
        assert "--rlimit-cpu=4" not in argv

    def test_unparseable_mem_dropped_silently(self, tmp_path: Path):
        backend, spy = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        # "1xb" is unparseable — backend just doesn't add the flag.
        asyncio.run(backend.run(handle, ["echo"], mem="1xb"))
        argv = spy.calls[0]["argv"]
        assert not any(a.startswith("--rlimit-as=") for a in argv)


# ---------------------------------------------------------------------------
# Bwrap argv
# ---------------------------------------------------------------------------


class TestBwrapArgv:
    def test_bwrap_default_argv_layout(self, tmp_path: Path):
        backend, spy = _make_backend(executable="bwrap")
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(backend.run(handle, ["python", "main.py"]))

        argv = spy.calls[0]["argv"]
        assert argv[0] == "bwrap"
        assert "--unshare-net" in argv
        # ro-bind + bind pairs.
        assert "--ro-bind" in argv
        assert "--bind" in argv
        assert "--chdir" in argv
        # `--` separator
        sep_idx = argv.index("--")
        assert argv[sep_idx + 1 :] == ["python", "main.py"]

    def test_bwrap_open_network(self, tmp_path: Path):
        backend, spy = _make_backend(executable="bwrap")
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(backend.run(handle, ["echo"], network="open"))
        argv = spy.calls[0]["argv"]
        assert "--unshare-net" not in argv


# ---------------------------------------------------------------------------
# run() result + timeout
# ---------------------------------------------------------------------------


class TestRunResult:
    def test_runner_output_propagates(self, tmp_path: Path):
        spy = _RunnerSpy(
            next_exit_code=0,
            next_stdout=b"hi",
            next_stderr=b"warn",
        )
        backend, _ = _make_backend(runner=spy)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        result = asyncio.run(backend.run(handle, ["echo"]))
        assert result.stdout == b"hi"
        assert result.stderr == b"warn"
        assert result.exit_code == 0
        assert result.ok is True

    def test_run_propagates_nonzero_exit(self, tmp_path: Path):
        spy = _RunnerSpy(next_exit_code=3)
        backend, _ = _make_backend(runner=spy)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        result = asyncio.run(backend.run(handle, ["false"]))
        assert result.exit_code == 3
        assert result.ok is False

    def test_run_empty_cmd_raises(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        with pytest.raises(SandboxError, match="non-empty"):
            asyncio.run(backend.run(handle, []))

    def test_timeout_raises_sandbox_timeout(self, tmp_path: Path):
        spy = _RunnerSpy(next_timed_out=True, next_exit_code=-9)
        backend, _ = _make_backend(runner=spy)
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        with pytest.raises(SandboxTimeoutError) as excinfo:
            asyncio.run(backend.run(handle, ["sleep", "10"], timeout=1.0))
        assert excinfo.value.result is not None
        assert excinfo.value.result.timed_out is True

    def test_network_open_flips_enforcement(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        result = asyncio.run(backend.run(handle, ["echo"], network="open"))
        assert result.network_enforced is False

    def test_env_layered_on_top_of_defaults(self, tmp_path: Path):
        backend, spy = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(
            backend.run(handle, ["echo"], env={"CUSTOM": "v", "HOME": "/override"})
        )
        env = spy.calls[0]["env"]
        # Defaults present.
        assert "PATH" in env
        # Overrides win.
        assert env["CUSTOM"] == "v"
        assert env["HOME"] == "/override"

    def test_stdin_passed_through_to_runner(self, tmp_path: Path):
        backend, spy = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(backend.run(handle, ["cat"], stdin=b"hello"))
        assert spy.calls[0]["stdin"] == b"hello"


# ---------------------------------------------------------------------------
# File IO + traversal guard
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_write_then_read(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        asyncio.run(backend.write_file(handle, "out/r.txt", b"abc"))
        assert asyncio.run(backend.read_file(handle, "out/r.txt")) == b"abc"

    def test_read_missing_file_raises(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        with pytest.raises(SandboxError, match="file not found"):
            asyncio.run(backend.read_file(handle, "missing.txt"))

    def test_traversal_blocked_read(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        with pytest.raises(SandboxError, match="escapes workspace"):
            asyncio.run(backend.read_file(handle, "../../etc/passwd"))

    def test_traversal_blocked_write(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=_mkws(tmp_path)))
        with pytest.raises(SandboxError, match="escapes workspace"):
            asyncio.run(backend.write_file(handle, "../escape", b"x"))


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_idempotent(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        ws = handle.workspace
        asyncio.run(backend.cleanup(handle))
        # Second call is a no-op.
        asyncio.run(backend.cleanup(handle))
        assert not ws.exists()
        assert handle.extras["cleaned"] is True

    def test_cleanup_skips_adopted_workspace(self, tmp_path: Path):
        backend, _ = _make_backend()
        skill = _make_skill(tmp_path)
        ws = tmp_path / "ws"
        ws.mkdir()
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        asyncio.run(backend.cleanup(handle))
        # User-supplied workspace stays.
        assert ws.exists()


# ---------------------------------------------------------------------------
# Real-binary integration (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("CARE_TEST_FIREJAIL"),
    reason="set CARE_TEST_FIREJAIL=1 to enable real-binary integration",
)
class TestRealFirejailIntegration:
    def test_round_trip(self, tmp_path: Path):
        backend = FirejailSandboxBackend()
        skill = _make_skill(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=None))
        try:
            result = asyncio.run(
                backend.run(handle, ["python3", "-c", "print('hi')"])
            )
            assert result.exit_code == 0
            assert b"hi" in result.stdout
        finally:
            asyncio.run(backend.cleanup(handle))
