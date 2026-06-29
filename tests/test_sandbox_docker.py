"""Tests for ``care.sandbox.docker.DockerSandboxBackend``
(TODO §6.1 P0).

The backend talks to the Docker SDK; full integration testing
requires a daemon. To keep the suite hermetic + fast we ship a
:class:`_StubDockerClient` that records every call's kwargs and
returns a :class:`_StubContainer`. Tests assert the kwargs
match the spec (image / network / cpu / mem / pids / read_only
/ tmpfs / mounts / labels), the lifecycle is correct, file IO
respects the path-traversal guard, and missing-dep paths
surface friendly errors.

A separate optional test class exercises the real daemon when
the ``CARE_TEST_DOCKER`` env var is set — skipped by default so
CI doesn't need Docker.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from care.sandbox.backend import SandboxError, SandboxTimeoutError
from care.sandbox.docker import (
    CARE_LABEL,
    DEFAULT_IMAGE,
    SKILL_MOUNT,
    WORKSPACE_MOUNT,
    DockerSandboxBackend,
)


# ---------------------------------------------------------------------------
# Stub Docker client
# ---------------------------------------------------------------------------


@dataclass
class _StubContainer:
    """Mimics enough of `docker.models.containers.Container`."""

    id: str = "container-abc"
    started: bool = False
    stopped: bool = False
    removed: bool = False
    exec_calls: list[dict[str, Any]] = field(default_factory=list)
    next_output: Any = (b"hello", b"")
    next_exit_code: int = 0
    raise_on_exec: Exception | None = None

    def start(self) -> None:
        self.started = True

    def stop(self, *, timeout: int = 10) -> None:
        self.stopped = True

    def remove(self, *, force: bool = False) -> None:
        self.removed = True

    def exec_run(self, **kwargs: Any) -> Any:
        self.exec_calls.append(kwargs)
        if self.raise_on_exec is not None:
            raise self.raise_on_exec

        class _Result:
            def __init__(self, output, exit_code):
                self.output = output
                self.exit_code = exit_code

        return _Result(self.next_output, self.next_exit_code)


class _StubContainers:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.next_container = _StubContainer()
        self.raise_on_create: Exception | None = None

    def create(self, **kwargs: Any) -> _StubContainer:
        self.create_calls.append(kwargs)
        if self.raise_on_create is not None:
            raise self.raise_on_create
        return self.next_container


class _StubDockerClient:
    def __init__(self) -> None:
        self.containers = _StubContainers()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill(tmp_path: Path, sha: str = "a" * 64) -> Any:
    """Return a duck-typed `ResolvedSkillLike` rooted at tmp_path/skill."""
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
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


# ---------------------------------------------------------------------------
# Construction / lazy-import branch
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_image_is_python_312_slim(self):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        assert backend._image == "python:3.12-slim"
        assert DEFAULT_IMAGE == "python:3.12-slim"

    def test_custom_image(self):
        backend = DockerSandboxBackend(
            image="my-org/care-runner:1.0",
            client_factory=lambda: _StubDockerClient(),
        )
        assert backend._image == "my-org/care-runner:1.0"

    def test_protocol_advertises_safe(self):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        assert backend.name == "docker"
        assert backend.unsafe is False

    def test_lazy_docker_import_friendly_error(self):
        """No client_factory + no docker dep → friendly SandboxError.

        This dev env *does* have docker installed (via the
        `docker` extra in the prior iteration), so this test
        skips when the SDK is present and exercises the
        ImportError path only when it's not."""
        try:
            import docker  # noqa: F401
        except ImportError:
            backend = DockerSandboxBackend()
            with pytest.raises(SandboxError, match="docker SDK is not installed"):
                backend._get_client()
        else:
            pytest.skip("docker SDK is installed; missing-dep path skipped")


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_creates_container_with_spec_defaults(self, tmp_path: Path):
        client = _StubDockerClient()
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path, sha="b" * 64)
        ws = _mkws(tmp_path)

        handle = asyncio.run(backend.prepare(skill, workspace=ws))

        assert len(client.containers.create_calls) == 1
        call = client.containers.create_calls[0]
        # Spec defaults.
        assert call["image"] == "python:3.12-slim"
        assert call["command"] == ["sleep", "infinity"]
        assert call["network_mode"] == "none"
        assert call["read_only"] is True
        assert call["tmpfs"] == {"/tmp": "size=64m,mode=1777"}
        assert call["pids_limit"] == 256
        # cpu_quota = default_cpu (2.0) * 100_000.
        assert call["cpu_quota"] == 200_000
        assert call["cpu_period"] == 100_000
        assert call["mem_limit"] == "1g"
        assert call["working_dir"] == WORKSPACE_MOUNT
        # Mounts: workspace rw, skill ro.
        volumes = call["volumes"]
        assert volumes[str(ws.resolve())] == {
            "bind": WORKSPACE_MOUNT,
            "mode": "rw",
        }
        skill_mount = volumes[str(Path(skill.path).resolve())]
        assert skill_mount == {"bind": SKILL_MOUNT, "mode": "ro"}
        # Labels include the SHA pin.
        labels = call["labels"]
        assert labels[CARE_LABEL] == "true"
        assert labels["care.sandbox.skill_sha256"] == "b" * 64
        # Container started.
        assert client.containers.next_container.started is True
        # Handle carries the container + SHA.
        assert handle.backend_name == "docker"
        assert handle.skill_sha256 == "b" * 64
        assert handle.network_enforced is True
        assert handle.extras["container_id"] == "container-abc"
        assert handle.extras["container_name"].startswith("care-skill-bbbbbbbbbbbb-")
        assert handle.extras["allowed_tools"] == ["Read"]

    def test_container_name_includes_sha_prefix(self, tmp_path: Path):
        client = _StubDockerClient()
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path, sha="deadbeef" * 8)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        # First 12 chars of the SHA appear in the container name.
        assert "deadbeefdead" in handle.extras["container_name"]
        # Random suffix is hex.
        suffix = handle.extras["container_name"].rsplit("-", 1)[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_custom_resource_defaults(self, tmp_path: Path):
        client = _StubDockerClient()
        backend = DockerSandboxBackend(
            client_factory=lambda: client,
            default_cpu=4.0,
            default_mem="2g",
            default_pids=128,
        )
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        asyncio.run(backend.prepare(skill, workspace=ws))
        call = client.containers.create_calls[0]
        assert call["cpu_quota"] == 400_000
        assert call["mem_limit"] == "2g"
        assert call["pids_limit"] == 128

    def test_missing_workspace_raises(self, tmp_path: Path):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        skill = _make_skill(tmp_path)
        with pytest.raises(SandboxError, match="requires an explicit workspace"):
            asyncio.run(backend.prepare(skill, workspace=None))

    def test_workspace_must_exist(self, tmp_path: Path):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        skill = _make_skill(tmp_path)
        with pytest.raises(SandboxError, match="not an existing directory"):
            asyncio.run(backend.prepare(skill, workspace=tmp_path / "does-not-exist"))

    def test_skill_path_must_exist(self, tmp_path: Path):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())

        class _BadSkill:
            sha256 = "c" * 64
            path = tmp_path / "no-skill-here"
            allowed_tools: list[str] = []

        ws = _mkws(tmp_path)
        with pytest.raises(SandboxError, match="skill path is not"):
            asyncio.run(backend.prepare(_BadSkill(), workspace=ws))

    def test_create_failure_wrapped(self, tmp_path: Path):
        client = _StubDockerClient()
        client.containers.raise_on_create = RuntimeError("daemon down")
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        with pytest.raises(SandboxError, match="failed to create docker container"):
            asyncio.run(backend.prepare(skill, workspace=ws))


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_dispatches_exec_with_kwargs(self, tmp_path: Path):
        client = _StubDockerClient()
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))

        result = asyncio.run(
            backend.run(
                handle,
                ["python", "-c", "print('hi')"],
                env={"FOO": "bar"},
            )
        )

        assert result.exit_code == 0
        assert result.stdout == b"hello"
        assert result.stderr == b""
        assert result.ok is True

        container = client.containers.next_container
        assert len(container.exec_calls) == 1
        call = container.exec_calls[0]
        assert call["cmd"] == ["python", "-c", "print('hi')"]
        assert call["environment"] == {"FOO": "bar"}
        assert call["workdir"] == WORKSPACE_MOUNT
        assert call["demux"] is True

    def test_run_propagates_exit_code(self, tmp_path: Path):
        client = _StubDockerClient()
        client.containers.next_container.next_exit_code = 7
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        result = asyncio.run(backend.run(handle, ["false"]))
        assert result.exit_code == 7
        assert result.ok is False

    def test_run_splits_stdout_stderr(self, tmp_path: Path):
        client = _StubDockerClient()
        client.containers.next_container.next_output = (b"out", b"err")
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        result = asyncio.run(backend.run(handle, ["echo"]))
        assert result.stdout == b"out"
        assert result.stderr == b"err"

    def test_run_handles_non_tuple_output(self, tmp_path: Path):
        # Some stubs (and older docker-py) return a single bytes blob.
        client = _StubDockerClient()
        client.containers.next_container.next_output = b"all on stdout"
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        result = asyncio.run(backend.run(handle, ["echo"]))
        assert result.stdout == b"all on stdout"
        assert result.stderr == b""

    def test_run_empty_cmd_raises(self, tmp_path: Path):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        with pytest.raises(SandboxError, match="non-empty"):
            asyncio.run(backend.run(handle, []))

    def test_run_exec_failure_wrapped(self, tmp_path: Path):
        client = _StubDockerClient()
        client.containers.next_container.raise_on_exec = RuntimeError("exec broken")
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        with pytest.raises(SandboxError, match="exec_run failed"):
            asyncio.run(backend.run(handle, ["python"]))

    def test_run_network_open_flips_enforcement_flag(self, tmp_path: Path):
        client = _StubDockerClient()
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        result = asyncio.run(backend.run(handle, ["echo"], network="open"))
        assert result.network_enforced is False

    def test_run_after_cleanup_raises(self, tmp_path: Path):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        # Wipe the container reference to simulate post-cleanup state.
        handle.extras["container"] = None
        with pytest.raises(SandboxError, match="has no live container"):
            asyncio.run(backend.run(handle, ["echo"]))


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_exceeded_raises(self, tmp_path: Path, monkeypatch):
        client = _StubDockerClient()
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))

        # Make `time.monotonic` jump 100s between the start
        # snapshot and the result snapshot so the backend
        # decides the run timed out.
        import types

        import care.sandbox.docker as docker_mod

        counter = iter([1000.0, 1100.0])
        fake_time = types.SimpleNamespace(monotonic=lambda: next(counter))
        monkeypatch.setattr(docker_mod, "time", fake_time)
        with pytest.raises(SandboxTimeoutError) as excinfo:
            asyncio.run(backend.run(handle, ["echo"], timeout=10.0))
        assert excinfo.value.result is not None
        assert excinfo.value.result.timed_out is True
        assert excinfo.value.result.duration_seconds == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# File IO + traversal guard
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_write_then_read(self, tmp_path: Path):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))

        asyncio.run(backend.write_file(handle, "out/result.txt", b"payload"))
        out = asyncio.run(backend.read_file(handle, "out/result.txt"))
        assert out == b"payload"

    def test_read_missing_file_raises(self, tmp_path: Path):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        with pytest.raises(SandboxError, match="file not found"):
            asyncio.run(backend.read_file(handle, "missing.txt"))

    def test_read_path_traversal_blocked(self, tmp_path: Path):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        with pytest.raises(SandboxError, match="escapes workspace"):
            asyncio.run(backend.read_file(handle, "../../etc/passwd"))

    def test_write_path_traversal_blocked(self, tmp_path: Path):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        with pytest.raises(SandboxError, match="escapes workspace"):
            asyncio.run(
                backend.write_file(handle, "../escape.txt", b"x")
            )

    def test_empty_path_rejected(self, tmp_path: Path):
        backend = DockerSandboxBackend(client_factory=lambda: _StubDockerClient())
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        with pytest.raises(SandboxError, match="must not be empty"):
            asyncio.run(backend.read_file(handle, ""))


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_stops_and_removes(self, tmp_path: Path):
        client = _StubDockerClient()
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        container = client.containers.next_container
        asyncio.run(backend.cleanup(handle))
        assert container.stopped is True
        assert container.removed is True
        assert handle.extras["cleaned"] is True

    def test_cleanup_idempotent(self, tmp_path: Path):
        client = _StubDockerClient()
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))
        asyncio.run(backend.cleanup(handle))
        # Reset spy + cleanup again — must be a no-op.
        container = client.containers.next_container
        container.stopped = False
        container.removed = False
        asyncio.run(backend.cleanup(handle))
        assert container.stopped is False
        assert container.removed is False

    def test_cleanup_tolerates_already_stopped(self, tmp_path: Path):
        client = _StubDockerClient()
        backend = DockerSandboxBackend(client_factory=lambda: client)
        skill = _make_skill(tmp_path)
        ws = _mkws(tmp_path)
        handle = asyncio.run(backend.prepare(skill, workspace=ws))

        container = client.containers.next_container

        def _raise_on_stop(*args, **kwargs):
            raise RuntimeError("already stopped")

        container.stop = _raise_on_stop
        # Must not raise.
        asyncio.run(backend.cleanup(handle))
        # Cleanup still flips the flag.
        assert handle.extras["cleaned"] is True


# ---------------------------------------------------------------------------
# Real-daemon integration (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("CARE_TEST_DOCKER"),
    reason="set CARE_TEST_DOCKER=1 to enable real-daemon integration",
)
class TestRealDaemonIntegration:
    def test_round_trip(self, tmp_path: Path):
        backend = DockerSandboxBackend()
        with tempfile.TemporaryDirectory() as ws:
            skill = _make_skill(Path(ws))
            handle = asyncio.run(backend.prepare(skill, workspace=Path(ws)))
            try:
                result = asyncio.run(
                    backend.run(handle, ["python", "-c", "print('hi')"])
                )
                assert result.exit_code == 0
                assert result.stdout.strip() == b"hi"
            finally:
                asyncio.run(backend.cleanup(handle))
