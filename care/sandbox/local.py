"""``LocalSandboxBackend`` — host-subprocess fallback (TODO §6.1 P0).

Wraps the current CARL behaviour: launch the skill via
``asyncio.create_subprocess_exec`` directly on the host. **Not a
real sandbox.** Marked ``unsafe=True`` so CARE's startup banner
can scream when a user accidentally selects this in production.

Use cases that justify keeping it around:

- Development on a machine without Docker.
- CI smoke tests where spinning up Docker is wasted overhead.
- The user explicitly wants their tools to see the host workspace
  (e.g. when iterating on a skill they wrote themselves).

The backend honours ``timeout`` strictly, captures stdout/stderr
into bytes, and prevents the most obvious path-traversal mistakes
on the ``read_file`` / ``write_file`` surface. ``cpu`` / ``mem``
arguments are accepted for API parity but ignored — the docstring
on those parameters in :class:`SandboxBackend` warns about this.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path

from care.sandbox.backend import (
    NetworkPolicy,
    ResolvedSkillLike,
    RunResult,
    SandboxError,
    SandboxHandle,
    SandboxTimeoutError,
)


class LocalSandboxBackend:
    """Host-subprocess sandbox backend (no isolation).

    Construct without arguments; per-call settings come in via
    :meth:`run`. Workspaces created by :meth:`prepare` are removed
    on :meth:`cleanup`; workspaces the caller supplied are left
    alone.
    """

    name: str = "local"
    unsafe: bool = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def prepare(
        self,
        skill: ResolvedSkillLike,
        workspace: Path | None = None,
    ) -> SandboxHandle:
        """Create (or adopt) a workspace and return a handle.

        Args:
            skill: Anything matching :class:`ResolvedSkillLike`.
            workspace: Pre-existing directory to use. When ``None``
                a temp directory is created and tracked for cleanup.

        Returns:
            A :class:`SandboxHandle` with ``backend_name="local"``
            and ``network_enforced=False`` (the host network is
            always reachable on this backend).
        """
        owns_workspace = workspace is None
        ws_path = (
            Path(tempfile.mkdtemp(prefix="care-local-sandbox-"))
            if owns_workspace
            else Path(workspace).resolve()
        )
        if not ws_path.exists():
            raise SandboxError(
                f"workspace does not exist and could not be created: {ws_path}"
            )
        if not ws_path.is_dir():
            raise SandboxError(f"workspace path is not a directory: {ws_path}")
        return SandboxHandle(
            backend_name=self.name,
            workspace=ws_path,
            skill_sha256=skill.sha256,
            network_enforced=False,
            extras={"owns_workspace": owns_workspace, "cleaned": False},
        )

    async def run(
        self,
        handle: SandboxHandle,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        cpu: float | None = None,  # noqa: ARG002 — accepted for parity
        mem: str | None = None,  # noqa: ARG002
        network: NetworkPolicy = "none",  # noqa: ARG002 — ignored on local
        timeout: float | None = None,
    ) -> RunResult:
        """Run ``cmd`` as a subprocess inside the workspace.

        ``cpu`` / ``mem`` / ``network`` are accepted but ignored.
        ``timeout`` is honoured strictly: the process is killed
        with ``SIGKILL`` after the timeout fires, partial
        stdout/stderr are still returned on the result, and a
        :class:`SandboxTimeoutError` is raised so callers can
        distinguish a timeout from a non-zero exit.
        """
        if not cmd:
            raise SandboxError("cmd must be a non-empty list of argv tokens")
        merged_env = self._build_env(env)

        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(handle.workspace),
            env=merged_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin),
                timeout=timeout,
            )
            duration = time.monotonic() - start
            return RunResult(
                exit_code=proc.returncode if proc.returncode is not None else -1,
                stdout=stdout or b"",
                stderr=stderr or b"",
                duration_seconds=duration,
                timed_out=False,
                network_enforced=False,
            )
        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            stdout_buf, stderr_buf = await self._kill_and_drain(proc)
            partial = RunResult(
                exit_code=-9,
                stdout=stdout_buf,
                stderr=stderr_buf,
                duration_seconds=duration,
                timed_out=True,
                network_enforced=False,
            )
            raise SandboxTimeoutError(
                f"command timed out after {timeout}s: {cmd[0]}",
                result=partial,
            )

    async def read_file(
        self,
        handle: SandboxHandle,
        path: str,
    ) -> bytes:
        """Read ``path`` (relative to the workspace).

        Raises :class:`SandboxError` if ``path`` resolves outside
        the workspace (path-traversal guard) or doesn't exist.
        """
        target = self._resolve_inside(handle, path)
        try:
            return target.read_bytes()
        except FileNotFoundError as exc:
            raise SandboxError(f"file not found in workspace: {path}") from exc
        except OSError as exc:
            raise SandboxError(f"could not read {path}: {exc}") from exc

    async def write_file(
        self,
        handle: SandboxHandle,
        path: str,
        data: bytes,
    ) -> None:
        """Write ``data`` to ``path`` (relative to the workspace).

        Creates parent directories as needed. Same traversal guard
        as :meth:`read_file`.
        """
        target = self._resolve_inside(handle, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def cleanup(self, handle: SandboxHandle) -> None:
        """Remove the workspace if we created it. Idempotent."""
        if handle.extras.get("cleaned"):
            return
        if handle.extras.get("owns_workspace"):
            shutil.rmtree(handle.workspace, ignore_errors=True)
        handle.extras["cleaned"] = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_env(env: dict[str, str] | None) -> dict[str, str]:
        """Build a minimal env, layering caller overrides on top.

        Default keys: ``PATH``, ``HOME``, ``LANG``, ``LC_ALL`` from
        the host. CARE explicitly does NOT inherit the full host env
        — secrets and project paths must be passed in explicitly.
        """
        base = {
            k: os.environ[k]
            for k in ("PATH", "HOME", "LANG", "LC_ALL")
            if k in os.environ
        }
        if env:
            base.update(env)
        return base

    @staticmethod
    def _resolve_inside(handle: SandboxHandle, path: str) -> Path:
        """Resolve ``path`` to an absolute path under the workspace
        and reject anything that escapes."""
        if not path:
            raise SandboxError("path must be non-empty")
        workspace = handle.workspace.resolve()
        candidate = (workspace / path).resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as exc:
            raise SandboxError(
                f"refusing to access path outside workspace: {path}"
            ) from exc
        return candidate

    @staticmethod
    async def _kill_and_drain(
        proc: asyncio.subprocess.Process,
    ) -> tuple[bytes, bytes]:
        """Best-effort: kill the process, then drain whatever
        output it had buffered."""
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=2.0
            )
        except (asyncio.TimeoutError, ProcessLookupError):
            stdout, stderr = b"", b""
        return stdout or b"", stderr or b""


__all__ = ["LocalSandboxBackend"]
