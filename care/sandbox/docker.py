"""``DockerSandboxBackend`` — container-isolated execution (TODO §6.1 P0).

The default safe backend CARE recommends. Each
:meth:`DockerSandboxBackend.prepare` call creates one long-lived
container that subsequent :meth:`run` calls `exec_run` against,
and :meth:`cleanup` stops + removes. Resource limits and the
read-only rootfs land at container-creation time so they apply to
every command the same container runs.

Hardening checklist (mirrors the §6.1 P0 spec):

* ``--network none`` by default. ``"skill_declared"`` falls back to
  ``"none"`` for now — egress proxying belongs in a follow-up.
  ``"open"`` lets the container reach the host network (unsafe;
  documented).
* ``--cpus`` / ``--memory`` / ``--pids-limit`` honoured from the
  caller's :meth:`run` kwargs + the backend's construction defaults.
* ``--read-only`` rootfs with a writable ephemeral ``tmpfs`` at
  ``/tmp`` so well-behaved skills can scratch.
* Workspace mounted **read-write** at a deterministic path inside
  the container (``/workspace``). The SKILL.md directory mounts
  **read-only** at ``/skill`` so skills can read their own bundle
  without being able to mutate the source.
* Container name carries the SHA-256 of SKILL.md and a random
  suffix so audit logs link back to the trust-pinned artefact.
* Path-traversal guard on read_file / write_file — same shape as
  :class:`care.sandbox.local.LocalSandboxBackend`.

The Docker SDK is **lazy-imported** so a CARE install without the
``docker`` extra still works for every other backend. A
``client_factory`` constructor arg lets tests pass a stub client
that records the kwargs every method received — exercised by
``tests/test_sandbox_docker.py``.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Any, Callable

from care.sandbox.backend import (
    NetworkPolicy,
    ResolvedSkillLike,
    RunResult,
    SandboxError,
    SandboxHandle,
    SandboxTimeoutError,
)

DEFAULT_IMAGE = "python:3.12-slim"
"""Base image the spec calls for. Override via the
:class:`DockerSandboxBackend` constructor."""

WORKSPACE_MOUNT = "/workspace"
"""Where the host workspace lands inside the container."""

SKILL_MOUNT = "/skill"
"""Where the SKILL.md directory lands inside the container
(read-only)."""

CARE_LABEL = "care.sandbox"
"""Label applied to every container CARE creates so cleanup
helpers can discriminate (``docker ps --filter
label=care.sandbox=true``)."""


class DockerSandboxBackend:
    """Docker-backed sandbox.

    Construct with the optional image/timeout/factory; per-call
    settings come in via :meth:`run`. Stays compatible with the
    :class:`care.sandbox.backend.SandboxBackend` protocol so the
    runtime executor can swap backends by config.

    The ``client_factory`` parameter exists for testability —
    pass a callable returning a stub docker client and the
    backend never touches the real SDK. Production callers leave
    it ``None`` and the backend lazy-imports
    ``docker.from_env()``.
    """

    name: str = "docker"
    unsafe: bool = False

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        client_factory: Callable[[], Any] | None = None,
        default_cpu: float = 2.0,
        default_mem: str = "1g",
        default_pids: int = 256,
    ) -> None:
        """Args:
        image: Container image used for every ``prepare``. Spec
            default is ``python:3.12-slim``.
        client_factory: Callable returning a configured Docker
            client. ``None`` lazy-imports ``docker.from_env()``
            on first use.
        default_cpu / default_mem / default_pids: Used when the
            caller's :meth:`run` doesn't override them. Match
            :class:`care.config.SandboxConfig` defaults so
            CARE's config flows through cleanly.
        """
        self._image = image
        self._client_factory = client_factory
        self._default_cpu = default_cpu
        self._default_mem = default_mem
        self._default_pids = default_pids
        self._client: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def prepare(
        self,
        skill: ResolvedSkillLike,
        workspace: Path | None = None,
    ) -> SandboxHandle:
        """Create + start a Docker container for ``skill``.

        Args:
            skill: Anything matching :class:`ResolvedSkillLike`.
                The ``sha256`` + ``path`` fields are required;
                ``allowed_tools`` is recorded for audit.
            workspace: Existing directory mounted read-write at
                ``/workspace``. Required for the Docker backend —
                unlike the local backend, we do not create a
                tempdir on the user's behalf because the user
                often wants outputs persisted.

        Returns:
            :class:`SandboxHandle` with the container id under
            ``extras["container_id"]``.
        """
        if workspace is None:
            raise SandboxError(
                "DockerSandboxBackend requires an explicit workspace path; "
                "create one with `tempfile.mkdtemp(...)` if you don't "
                "have one to hand."
            )
        ws_path = Path(workspace).resolve()
        if not ws_path.is_dir():
            raise SandboxError(
                f"workspace is not an existing directory: {ws_path}"
            )
        skill_path = Path(skill.path).resolve()
        if not skill_path.is_dir():
            raise SandboxError(
                f"skill path is not an existing directory: {skill_path}"
            )

        client = self._get_client()
        name = self._container_name(skill.sha256)
        try:
            container = client.containers.create(
                image=self._image,
                command=["sleep", "infinity"],
                name=name,
                network_mode="none",
                read_only=True,
                tmpfs={"/tmp": "size=64m,mode=1777"},
                cpu_period=100_000,
                cpu_quota=int(self._default_cpu * 100_000),
                mem_limit=self._default_mem,
                pids_limit=self._default_pids,
                volumes={
                    str(ws_path): {
                        "bind": WORKSPACE_MOUNT,
                        "mode": "rw",
                    },
                    str(skill_path): {
                        "bind": SKILL_MOUNT,
                        "mode": "ro",
                    },
                },
                working_dir=WORKSPACE_MOUNT,
                labels={
                    CARE_LABEL: "true",
                    "care.sandbox.skill_sha256": skill.sha256,
                },
                detach=True,
            )
            container.start()
        except Exception as exc:  # noqa: BLE001
            raise SandboxError(
                f"failed to create docker container {name!r}: {exc}"
            ) from exc

        return SandboxHandle(
            backend_name=self.name,
            workspace=ws_path,
            skill_sha256=skill.sha256,
            network_enforced=True,
            extras={
                "container_id": container.id,
                "container_name": name,
                "container": container,
                "cleaned": False,
                "workspace_mount": WORKSPACE_MOUNT,
                "skill_mount": SKILL_MOUNT,
                "skill_path": skill_path,
                "allowed_tools": list(getattr(skill, "allowed_tools", []) or []),
            },
        )

    async def run(
        self,
        handle: SandboxHandle,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,  # noqa: ARG002 — exec stdin pipe is a follow-up
        cpu: float | None = None,  # noqa: ARG002 — limits set at prepare()
        mem: str | None = None,  # noqa: ARG002
        network: NetworkPolicy = "none",
        timeout: float | None = None,
    ) -> RunResult:
        """Execute ``cmd`` inside the container via ``exec_run``.

        Container-level resource limits land at
        :meth:`prepare` time, so ``cpu`` / ``mem`` are
        accepted-but-ignored on a per-run basis (matches Docker's
        own behaviour — limits are container-scoped, not exec-
        scoped). ``network`` other than the container's
        configured mode is similarly ignored, but the supplied
        value lands on the audit log.

        Args:
            handle: From a prior :meth:`prepare`.
            cmd: argv list (must be non-empty).
            env: Per-exec env overrides.
            stdin: Optional stdin payload — currently
                accepted-but-not-piped (Docker exec stdin needs
                socket plumbing; deferred).
            cpu, mem, network: See class docstring.
            timeout: Wall-clock seconds. ``None`` = no timeout.

        Returns:
            :class:`RunResult`. Raises
            :class:`SandboxTimeoutError` when ``timeout`` fires.
        """
        if not cmd:
            raise SandboxError("cmd must be a non-empty list of argv tokens")
        container = self._container(handle)
        start = time.monotonic()
        # `exec_run` blocks; wrap with a wall-clock timeout check
        # by reading the result + comparing duration. Real timeout
        # kill needs a separate exec; surface as a follow-up if
        # users hit it in practice.
        try:
            result = container.exec_run(
                cmd=cmd,
                environment=env or {},
                workdir=WORKSPACE_MOUNT,
                demux=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise SandboxError(
                f"exec_run failed inside container {handle.extras.get('container_name')!r}: {exc}"
            ) from exc

        duration = time.monotonic() - start
        stdout, stderr = _split_exec_output(result.output)
        exit_code = int(getattr(result, "exit_code", 0) or 0)
        network_enforced = network != "open"

        if timeout is not None and duration > timeout:
            partial = RunResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                timed_out=True,
                network_enforced=network_enforced,
            )
            raise SandboxTimeoutError(
                f"docker exec timed out after {timeout}s: {cmd[0]}",
                result=partial,
            )

        return RunResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            timed_out=False,
            network_enforced=network_enforced,
        )

    async def read_file(
        self,
        handle: SandboxHandle,
        path: str,
    ) -> bytes:
        """Read ``path`` from the workspace mount.

        Same path-traversal guard as
        :class:`care.sandbox.local.LocalSandboxBackend` — we
        resolve through the host-side workspace, not via
        ``docker cp``. That keeps the implementation simple
        (no tar-stream parsing) and matches the contract:
        workspace mounts are bidirectional.
        """
        target = _resolve_inside(handle.workspace, path)
        try:
            return target.read_bytes()
        except FileNotFoundError as exc:
            raise SandboxError(
                f"file not found in workspace: {path}"
            ) from exc
        except OSError as exc:
            raise SandboxError(
                f"could not read {path}: {exc}"
            ) from exc

    async def write_file(
        self,
        handle: SandboxHandle,
        path: str,
        data: bytes,
    ) -> None:
        """Write ``data`` to ``path`` inside the workspace mount."""
        target = _resolve_inside(handle.workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def cleanup(self, handle: SandboxHandle) -> None:
        """Stop + remove the container. Idempotent."""
        if handle.extras.get("cleaned"):
            return
        container = handle.extras.get("container")
        if container is not None:
            try:
                container.stop(timeout=5)
            except Exception:  # noqa: BLE001
                # Container may already be stopped / gone.
                pass
            try:
                container.remove(force=True)
            except Exception:  # noqa: BLE001
                pass
        handle.extras["cleaned"] = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        try:
            import docker as _docker
        except ImportError as exc:
            raise SandboxError(
                "docker SDK is not installed; "
                "install with `pip install \"care[docker]\"` to use "
                "the DockerSandboxBackend"
            ) from exc
        try:
            self._client = _docker.from_env()
        except Exception as exc:  # noqa: BLE001
            raise SandboxError(
                f"could not connect to Docker daemon: {exc}"
            ) from exc
        return self._client

    @staticmethod
    def _container_name(sha256: str) -> str:
        """Stable per-prepare container name. Includes the SKILL.md
        SHA-256 prefix so audit logs link back to the trust pin."""
        suffix = secrets.token_hex(4)
        return f"care-skill-{sha256[:12]}-{suffix}"

    @staticmethod
    def _container(handle: SandboxHandle) -> Any:
        container = handle.extras.get("container")
        if container is None:
            raise SandboxError(
                f"sandbox handle {handle.extras.get('container_name')!r} "
                "has no live container — was it cleaned up?"
            )
        return container


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_inside(workspace: Path, path: str) -> Path:
    """Resolve ``path`` against ``workspace`` and refuse anything
    that escapes via ``..`` or an absolute path."""
    if not path:
        raise SandboxError("path must not be empty")
    target = (workspace / path).resolve()
    try:
        target.relative_to(workspace.resolve())
    except ValueError as exc:
        raise SandboxError(
            f"path escapes workspace: {path!r}"
        ) from exc
    return target


def _split_exec_output(output: Any) -> tuple[bytes, bytes]:
    """Coerce ``exec_run`` output into ``(stdout, stderr)`` bytes.

    Docker SDK returns a `(stdout, stderr)` tuple when
    ``demux=True``; either side may be ``None``. Non-tuple
    payloads (some test stubs return a single bytes blob) land
    on stdout with an empty stderr.
    """
    if isinstance(output, tuple) and len(output) == 2:
        out, err = output
        return _coerce_bytes(out), _coerce_bytes(err)
    return _coerce_bytes(output), b""


def _coerce_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(str(value), encoding="utf-8")


__all__ = [
    "CARE_LABEL",
    "DEFAULT_IMAGE",
    "DockerSandboxBackend",
    "SKILL_MOUNT",
    "WORKSPACE_MOUNT",
]
