"""Tests for ``care.sandbox.LocalSandboxBackend`` (TODO §6.1 P0).

Uses **real subprocesses** (``echo``, ``cat``, ``sleep``, ``python``)
so the asyncio plumbing is exercised end-to-end rather than mocked.
That's possible cheaply here because every command is a standard
POSIX utility available on macOS/Linux dev boxes + CI.

Coverage layers:
1. **Value types** — ``SandboxHandle`` / ``RunResult`` round-trip;
   ``RunResult.ok`` predicate.
2. **Protocol conformance** — ``LocalSandboxBackend`` satisfies the
   ``SandboxBackend`` runtime-checkable Protocol.
3. **Lifecycle** — ``prepare`` / ``run`` / ``read_file`` /
   ``write_file`` / ``cleanup`` against a real subprocess.
4. **Traversal guard** — ``read_file`` / ``write_file`` reject
   paths that escape the workspace.
5. **Timeout** — long-running command is killed and raises
   ``SandboxTimeoutError`` with the partial result on the cause.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from care.sandbox import (
    LocalSandboxBackend,
    NetworkPolicy,
    ResolvedSkillLike,
    RunResult,
    SandboxBackend,
    SandboxError,
    SandboxHandle,
    SandboxTimeoutError,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeSkill:
    """Duck-typed :class:`ResolvedSkillLike` for tests."""

    sha256: str = "a" * 64
    path: Path = Path("/tmp/fake-skill")
    allowed_tools: list[str] | None = None

    def __post_init__(self):
        if self.allowed_tools is None:
            self.allowed_tools = []


@pytest.fixture
def backend() -> LocalSandboxBackend:
    return LocalSandboxBackend()


@pytest.fixture
def fake_skill() -> _FakeSkill:
    return _FakeSkill()


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class TestSandboxHandle:
    def test_defaults(self):
        h = SandboxHandle(
            backend_name="local",
            workspace=Path("/tmp/ws"),
            skill_sha256="x" * 64,
        )
        assert h.network_enforced is True
        assert h.extras == {}

    def test_extras_independent_per_handle(self):
        """Default-factory should produce a fresh dict each time —
        a classic mutable-default bug guard."""
        a = SandboxHandle("local", Path("/a"), "x")
        b = SandboxHandle("local", Path("/b"), "y")
        a.extras["k"] = 1
        assert b.extras == {}


class TestRunResult:
    def test_ok_true_on_zero_exit_no_timeout(self):
        r = RunResult(exit_code=0, stdout=b"", stderr=b"", duration_seconds=0.1)
        assert r.ok is True

    @pytest.mark.parametrize(
        "exit_code,timed_out",
        [(1, False), (0, True), (-9, True), (137, False)],
    )
    def test_ok_false_otherwise(self, exit_code: int, timed_out: bool):
        r = RunResult(
            exit_code=exit_code,
            stdout=b"",
            stderr=b"",
            duration_seconds=0.1,
            timed_out=timed_out,
        )
        assert r.ok is False


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_local_backend_satisfies_sandbox_backend(self, backend):
        """``SandboxBackend`` is ``@runtime_checkable`` so structural
        conformance is enough."""
        assert isinstance(backend, SandboxBackend)

    def test_network_policy_literal_values(self):
        """Pin the canonical set so a typo upstream surfaces here."""
        values: set[NetworkPolicy] = {"none", "skill_declared", "open"}
        assert values == {"none", "skill_declared", "open"}

    def test_resolved_skill_like_accepts_fake_shape(self):
        skill: ResolvedSkillLike = _FakeSkill()
        assert skill.sha256 == "a" * 64


# ---------------------------------------------------------------------------
# Lifecycle (real subprocess)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_creates_workspace_when_none(backend, fake_skill):
    handle = await backend.prepare(fake_skill)
    try:
        assert handle.backend_name == "local"
        assert handle.skill_sha256 == fake_skill.sha256
        assert handle.network_enforced is False
        assert handle.workspace.exists()
        assert handle.workspace.is_dir()
        assert handle.extras["owns_workspace"] is True
    finally:
        await backend.cleanup(handle)
        # Created workspace must be gone after cleanup.
        assert not handle.workspace.exists()


@pytest.mark.asyncio
async def test_prepare_adopts_existing_workspace(backend, fake_skill, tmp_path):
    handle = await backend.prepare(fake_skill, workspace=tmp_path)
    try:
        assert handle.workspace == tmp_path.resolve()
        assert handle.extras["owns_workspace"] is False
    finally:
        await backend.cleanup(handle)
        # Caller-supplied workspace must survive cleanup.
        assert tmp_path.exists()


@pytest.mark.asyncio
async def test_prepare_rejects_nonexistent_workspace(backend, fake_skill, tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(SandboxError, match="does not exist"):
        await backend.prepare(fake_skill, workspace=missing)


@pytest.mark.asyncio
async def test_prepare_rejects_workspace_that_is_a_file(
    backend, fake_skill, tmp_path
):
    file_ws = tmp_path / "not-a-dir.txt"
    file_ws.write_text("x")
    with pytest.raises(SandboxError, match="not a directory"):
        await backend.prepare(fake_skill, workspace=file_ws)


@pytest.mark.asyncio
async def test_run_echo_captures_stdout(backend, fake_skill):
    handle = await backend.prepare(fake_skill)
    try:
        result = await backend.run(handle, ["echo", "hello world"])
        assert result.ok
        assert result.exit_code == 0
        assert result.stdout.strip() == b"hello world"
        assert result.duration_seconds >= 0
        assert result.network_enforced is False
    finally:
        await backend.cleanup(handle)


@pytest.mark.asyncio
async def test_run_nonzero_exit_propagates_as_result(backend, fake_skill):
    """A failing command should NOT raise — exit code is enough."""
    handle = await backend.prepare(fake_skill)
    try:
        result = await backend.run(
            handle, ["python", "-c", "import sys; sys.exit(7)"]
        )
        assert result.ok is False
        assert result.exit_code == 7
        assert result.timed_out is False
    finally:
        await backend.cleanup(handle)


@pytest.mark.asyncio
async def test_run_empty_cmd_raises(backend, fake_skill):
    handle = await backend.prepare(fake_skill)
    try:
        with pytest.raises(SandboxError, match="non-empty"):
            await backend.run(handle, [])
    finally:
        await backend.cleanup(handle)


@pytest.mark.asyncio
async def test_run_stdin_piped_through(backend, fake_skill):
    handle = await backend.prepare(fake_skill)
    try:
        result = await backend.run(handle, ["cat"], stdin=b"piped input\n")
        assert result.ok
        assert result.stdout == b"piped input\n"
    finally:
        await backend.cleanup(handle)


@pytest.mark.asyncio
async def test_run_uses_workspace_as_cwd(backend, fake_skill, tmp_path):
    handle = await backend.prepare(fake_skill, workspace=tmp_path)
    try:
        result = await backend.run(handle, ["pwd"])
        assert result.ok
        # macOS resolves /var/folders → /private/var/folders; both
        # are valid forms of the same path.
        actual = Path(result.stdout.decode().strip()).resolve()
        assert actual == tmp_path.resolve()
    finally:
        await backend.cleanup(handle)


@pytest.mark.asyncio
async def test_run_env_overrides_layered_on_defaults(backend, fake_skill):
    handle = await backend.prepare(fake_skill)
    try:
        result = await backend.run(
            handle,
            ["python", "-c", "import os; print(os.environ.get('CARE_TEST'))"],
            env={"CARE_TEST": "hello"},
        )
        assert result.ok
        assert result.stdout.strip() == b"hello"
    finally:
        await backend.cleanup(handle)


# ---------------------------------------------------------------------------
# Read / write file (traversal guards)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_and_read_round_trip(backend, fake_skill):
    handle = await backend.prepare(fake_skill)
    try:
        await backend.write_file(handle, "out/report.json", b'{"k":1}')
        assert (handle.workspace / "out" / "report.json").exists()
        data = await backend.read_file(handle, "out/report.json")
        assert data == b'{"k":1}'
    finally:
        await backend.cleanup(handle)


@pytest.mark.asyncio
async def test_read_file_missing_raises(backend, fake_skill):
    handle = await backend.prepare(fake_skill)
    try:
        with pytest.raises(SandboxError, match="not found"):
            await backend.read_file(handle, "does-not-exist.txt")
    finally:
        await backend.cleanup(handle)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape.txt",
        "../../etc/passwd",
        "/etc/passwd",
        "out/../../escape.txt",
    ],
)
async def test_traversal_guards_block_escape(backend, fake_skill, bad_path):
    handle = await backend.prepare(fake_skill)
    try:
        with pytest.raises(SandboxError, match="outside workspace"):
            await backend.read_file(handle, bad_path)
        with pytest.raises(SandboxError, match="outside workspace"):
            await backend.write_file(handle, bad_path, b"x")
    finally:
        await backend.cleanup(handle)


@pytest.mark.asyncio
async def test_empty_path_rejected(backend, fake_skill):
    handle = await backend.prepare(fake_skill)
    try:
        with pytest.raises(SandboxError, match="non-empty"):
            await backend.read_file(handle, "")
    finally:
        await backend.cleanup(handle)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_timeout_kills_process_and_raises(backend, fake_skill):
    handle = await backend.prepare(fake_skill)
    try:
        with pytest.raises(SandboxTimeoutError) as exc_info:
            await backend.run(
                handle,
                ["python", "-c", "import time; time.sleep(5)"],
                timeout=0.5,
            )
        # Partial result is attached directly to the exception so
        # the audit log can record duration/exit_code/timed_out.
        result = exc_info.value.result
        assert result is not None
        assert result.timed_out is True
        assert result.exit_code == -9
        assert result.duration_seconds >= 0.4  # roughly the timeout
    finally:
        await backend.cleanup(handle)


# ---------------------------------------------------------------------------
# Cleanup idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_is_idempotent(backend, fake_skill):
    handle = await backend.prepare(fake_skill)
    await backend.cleanup(handle)
    # Second call is a no-op, no exception.
    await backend.cleanup(handle)
    assert handle.extras["cleaned"] is True


@pytest.mark.asyncio
async def test_cleanup_does_not_delete_caller_supplied_workspace(
    backend, fake_skill, tmp_path
):
    sentinel = tmp_path / "marker.txt"
    sentinel.write_text("survive me")
    handle = await backend.prepare(fake_skill, workspace=tmp_path)
    await backend.cleanup(handle)
    assert sentinel.exists()
    assert sentinel.read_text() == "survive me"


# ---------------------------------------------------------------------------
# Concurrency smoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runs_can_interleave(backend, fake_skill):
    """Two prepare/run/cleanup lifecycles in parallel — verifies
    the backend isn't holding any process-global mutable state."""
    async def one_cycle(value: str) -> str:
        handle = await backend.prepare(fake_skill)
        try:
            result = await backend.run(handle, ["echo", value])
            return result.stdout.decode().strip()
        finally:
            await backend.cleanup(handle)

    outs = await asyncio.gather(*(one_cycle(v) for v in ("a", "b", "c")))
    assert sorted(outs) == ["a", "b", "c"]
