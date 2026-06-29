"""``FirejailSandboxBackend`` — Linux-only lightweight sandbox
(TODO §6.1 P2).

For users who want subprocess isolation **without Docker
overhead** — laptops without a daemon running, CI runners that
can't access the Docker socket, anyone allergic to multi-second
container startup latency. Wraps the host's ``firejail`` (or
``bwrap``) binary, which together cover most Linux distros.

This backend's posture sits between :class:`LocalSandboxBackend`
(no isolation, ``unsafe=True``) and :class:`DockerSandboxBackend`
(full container isolation, ``unsafe=False``). Firejail provides
real namespace + seccomp isolation but inherits more of the host
than a container does, so it's still ``unsafe=False`` for CARE's
purposes but documented as "lighter touch".

Tradeoffs vs Docker:

* **Faster startup** — no daemon round-trip, no image pull.
* **Linux-only** — refuses to construct on non-Linux platforms.
* **No ``read_only`` rootfs** — firejail's overlay-fs gives
  some protection but isn't strictly read-only; we add
  ``--private`` so the home dir is ephemeral.
* **Network policy mapped** to firejail's ``--net=none`` /
  ``--net=host`` / no-flag-for-default. ``skill_declared``
  falls back to ``--net=none`` until egress proxying lands.

The implementation is structured so a stub subprocess runner can
verify argv construction without firejail actually being installed
— same testability pattern :class:`DockerSandboxBackend` uses for
``docker``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Awaitable, Callable

from care.sandbox.backend import (
    NetworkPolicy,
    ResolvedSkillLike,
    RunResult,
    SandboxError,
    SandboxHandle,
    SandboxTimeoutError,
)

DEFAULT_EXECUTABLE = "firejail"
"""Default sandbox helper. ``bwrap`` is the documented fallback
for distros where firejail isn't packaged (e.g. recent Fedora);
pass ``executable="bwrap"`` to use it."""

# Per-network-policy argv fragments. Each entry contributes zero
# or more flags to the firejail / bwrap command line.
_FIREJAIL_NETWORK_FLAGS: dict[NetworkPolicy, tuple[str, ...]] = {
    "none": ("--net=none",),
    # Egress proxying lives in Platform §4.5c; until then we
    # close the network for skill_declared too — safer default
    # than silently leaving it open.
    "skill_declared": ("--net=none",),
    "open": (),  # No flag → inherit host network.
}

_BWRAP_NETWORK_FLAGS: dict[NetworkPolicy, tuple[str, ...]] = {
    "none": ("--unshare-net",),
    "skill_declared": ("--unshare-net",),
    "open": (),
}


SubprocessRunner = Callable[
    [list[str], Path, dict[str, str], bytes | None, float | None],
    Awaitable[tuple[int, bytes, bytes, bool]],
]
"""Injectable subprocess runner. Returns
``(exit_code, stdout, stderr, timed_out)``. Tests pass a stub;
production uses the built-in :func:`_default_runner` over
:func:`asyncio.create_subprocess_exec`."""


class FirejailSandboxBackend:
    """Subprocess sandbox via firejail / bwrap.

    Construct with the executable name (defaults to ``firejail``)
    plus optional resource defaults. Per-call settings come in
    via :meth:`run`.

    The :class:`SandboxBackend` protocol's ``name`` reflects
    whichever executable was picked at construction so audit
    logs can tell ``firejail`` and ``bwrap`` runs apart.
    """

    unsafe: bool = False

    def __init__(
        self,
        *,
        executable: str = DEFAULT_EXECUTABLE,
        default_cpu: float | None = None,
        default_mem: str | None = None,
        subprocess_runner: SubprocessRunner | None = None,
        require_linux: bool = True,
    ) -> None:
        """Args:
        executable: ``"firejail"`` (default) or ``"bwrap"``.
            Anything else raises — keeps the two supported argv
            shapes honest. Override with an absolute path
            (``/opt/local/bin/firejail``) when ``$PATH`` doesn't
            include it.
        default_cpu / default_mem: Forwarded into the argv
            (``--rlimit-cpu``, ``--rlimit-as``) when set. Match
            :class:`care.config.SandboxConfig` defaults.
        subprocess_runner: Injectable async runner — used by
            tests to record argv without spawning real
            processes.
        require_linux: When ``True`` (the default), construction
            on non-Linux hosts raises immediately so the user
            sees a clear error instead of a cryptic
            "firejail: command not found". Set ``False`` in
            cross-platform unit tests that exercise argv
            construction.
        """
        exe_basename = os.path.basename(executable)
        if exe_basename not in ("firejail", "bwrap"):
            raise SandboxError(
                f"unsupported sandbox executable {executable!r}; "
                "use 'firejail' or 'bwrap'"
            )
        if require_linux and not sys.platform.startswith("linux"):
            raise SandboxError(
                f"FirejailSandboxBackend is Linux-only; "
                f"current platform is {sys.platform!r}. "
                "Use DockerSandboxBackend on macOS / Windows."
            )
        self._executable = executable
        self._default_cpu = default_cpu
        self._default_mem = default_mem
        self._subprocess_runner = subprocess_runner or _default_runner

    @property
    def name(self) -> str:
        return os.path.basename(self._executable)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def prepare(
        self,
        skill: ResolvedSkillLike,
        workspace: Path | None = None,
    ) -> SandboxHandle:
        """Create (or adopt) a workspace and verify the executable
        is on ``$PATH``.

        Args:
            skill: Anything matching :class:`ResolvedSkillLike`.
            workspace: Pre-existing directory to use. When
                ``None``, a temp directory is created and
                tracked for cleanup.

        Returns:
            :class:`SandboxHandle` with the resolved executable
            path under ``extras["executable"]``.
        """
        # Allow constructor-supplied stub runners to skip the
        # which() check — tests don't have firejail installed.
        if self._subprocess_runner is _default_runner:
            resolved_exe = shutil.which(self._executable)
            if resolved_exe is None:
                raise SandboxError(
                    f"{self._executable!r} not found on PATH; "
                    f"install it (`apt install firejail` / `dnf install bubblewrap`) "
                    "or use a different sandbox backend."
                )
        else:
            resolved_exe = self._executable

        owns_workspace = workspace is None
        ws_path = (
            Path(tempfile.mkdtemp(prefix="care-firejail-sandbox-"))
            if owns_workspace
            else Path(workspace).resolve()
        )
        if not ws_path.exists():
            raise SandboxError(
                f"workspace does not exist: {ws_path}"
            )
        if not ws_path.is_dir():
            raise SandboxError(
                f"workspace path is not a directory: {ws_path}"
            )
        skill_path = Path(skill.path).resolve()
        if not skill_path.is_dir():
            raise SandboxError(
                f"skill path is not a directory: {skill_path}"
            )
        return SandboxHandle(
            backend_name=self.name,
            workspace=ws_path,
            skill_sha256=skill.sha256,
            network_enforced=True,
            extras={
                "executable": resolved_exe,
                "owns_workspace": owns_workspace,
                "cleaned": False,
                "skill_path": skill_path,
                "allowed_tools": list(
                    getattr(skill, "allowed_tools", []) or []
                ),
            },
        )

    async def run(
        self,
        handle: SandboxHandle,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        cpu: float | None = None,
        mem: str | None = None,
        network: NetworkPolicy = "none",
        timeout: float | None = None,
    ) -> RunResult:
        """Build the firejail argv + execute via the configured
        subprocess runner.

        Args:
            handle: From a prior :meth:`prepare`.
            cmd: argv list (must be non-empty).
            env: Per-run env overrides. Layered on top of a
                minimal default (PATH / HOME).
            stdin: Optional stdin payload piped to the wrapped
                process.
            cpu: ``--rlimit-cpu`` in seconds when set. Overrides
                ``default_cpu`` from construction.
            mem: ``--rlimit-as`` virtual-memory cap (Docker-style
                suffix, e.g. ``"512m"``).
            network: Maps to firejail's network flag.
            timeout: Wall-clock seconds before the process is
                killed.

        Returns:
            :class:`RunResult`. Raises
            :class:`SandboxTimeoutError` when ``timeout`` fires.
        """
        if not cmd:
            raise SandboxError("cmd must be a non-empty list of argv tokens")
        argv = self._build_argv(
            handle,
            cmd,
            cpu=cpu if cpu is not None else self._default_cpu,
            mem=mem if mem is not None else self._default_mem,
            network=network,
        )
        merged_env = _build_env(env)

        start = time.monotonic()
        exit_code, stdout, stderr, timed_out = await self._subprocess_runner(
            argv,
            handle.workspace,
            merged_env,
            stdin,
            timeout,
        )
        duration = time.monotonic() - start
        network_enforced = network != "open"
        if timed_out:
            partial = RunResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                timed_out=True,
                network_enforced=network_enforced,
            )
            raise SandboxTimeoutError(
                f"{self.name} command timed out after {timeout}s: {cmd[0]}",
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

        Same path-traversal guard as the other backends.
        """
        target = _resolve_inside(handle.workspace, path)
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
        """Write ``data`` to ``path`` (relative to the workspace)."""
        target = _resolve_inside(handle.workspace, path)
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

    def _build_argv(
        self,
        handle: SandboxHandle,
        cmd: list[str],
        *,
        cpu: float | None,
        mem: str | None,
        network: NetworkPolicy,
    ) -> list[str]:
        """Construct the wrapper argv.

        Public so tests can assert on it without invoking the
        full :meth:`run` loop — but in practice the dedicated
        :func:`build_argv_for_test` helper is what tests reach
        for (kept off the protocol surface).
        """
        executable = handle.extras.get("executable") or self._executable
        argv: list[str] = [executable]

        is_bwrap = self.name == "bwrap"
        if is_bwrap:
            argv.extend(_BWRAP_NETWORK_FLAGS.get(network, ()))
            # Read-only bind for the skill dir, read-write for workspace.
            argv.extend(
                [
                    "--ro-bind",
                    str(handle.extras["skill_path"]),
                    str(handle.extras["skill_path"]),
                    "--bind",
                    str(handle.workspace),
                    str(handle.workspace),
                    "--chdir",
                    str(handle.workspace),
                    "--proc",
                    "/proc",
                    "--dev",
                    "/dev",
                ]
            )
            if cpu is not None:
                # bwrap doesn't have rlimit flags; document the
                # limitation via the audit log later. Accept the
                # value silently to keep parity with firejail.
                pass
            if mem is not None:
                pass
            argv.append("--")
        else:
            argv.append("--quiet")
            argv.append("--noprofile")
            argv.append("--private")
            argv.append(f"--chdir={handle.workspace}")
            argv.append(f"--whitelist={handle.workspace}")
            argv.append(f"--read-only={handle.extras['skill_path']}")
            argv.extend(_FIREJAIL_NETWORK_FLAGS.get(network, ()))
            if cpu is not None:
                argv.append(f"--rlimit-cpu={int(cpu)}")
            if mem is not None:
                mem_bytes = _parse_mem(mem)
                if mem_bytes is not None:
                    argv.append(f"--rlimit-as={mem_bytes}")
            argv.append("--")

        argv.extend(cmd)
        return argv


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


def _build_env(env: dict[str, str] | None) -> dict[str, str]:
    """Minimal env, layering caller overrides on top."""
    base = {
        "PATH": os.environ.get(
            "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "HOME": "/tmp",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    if env:
        base.update(env)
    return base


def _parse_mem(value: str) -> int | None:
    """Parse a Docker-style memory string (``"512m"``, ``"1g"``)
    into bytes. Returns ``None`` on parse failure rather than
    raising — the worst case is we drop the rlimit, not error
    the run."""
    s = value.strip().lower()
    if not s or len(s) < 2:
        return None
    unit = s[-1]
    head = s[:-1]
    if not head.isdigit():
        return None
    multiplier = {"k": 1024, "m": 1024**2, "g": 1024**3}.get(unit)
    if multiplier is None:
        return None
    return int(head) * multiplier


async def _default_runner(
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    stdin: bytes | None,
    timeout: float | None,
) -> tuple[int, bytes, bytes, bool]:
    """Production subprocess runner using
    :func:`asyncio.create_subprocess_exec`."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin),
            timeout=timeout,
        )
        exit_code = proc.returncode if proc.returncode is not None else -1
        return exit_code, stdout or b"", stderr or b"", False
    except asyncio.TimeoutError:
        # Kill + drain partial output.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = await proc.communicate()
        except Exception:  # noqa: BLE001
            stdout, stderr = b"", b""
        return -9, stdout or b"", stderr or b"", True


__all__ = [
    "DEFAULT_EXECUTABLE",
    "FirejailSandboxBackend",
    "SubprocessRunner",
]
