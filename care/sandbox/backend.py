"""``SandboxBackend`` protocol + shared value types (TODO §6.1 P0).

Every concrete backend (``LocalSandboxBackend``,
``DockerSandboxBackend``, ``E2BSandboxBackend``, ...) implements
:class:`SandboxBackend`. CARE's execution code calls the protocol
without caring which backend is plugged in — the choice comes from
``CareConfig.sandbox.kind``.

Lifecycle::

    handle = await backend.prepare(skill, workspace)
    try:
        result = await backend.run(
            handle, ["python", "main.py"], env={}, stdin=None,
            cpu=2.0, mem="1g", network="none", timeout=30.0,
        )
        data = await backend.read_file(handle, "out/report.json")
    finally:
        await backend.cleanup(handle)

The skill argument is duck-typed via :class:`ResolvedSkillLike` so
CARE's tests + first-iteration backends don't need to import
``mmar_carl`` at module-load time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

NetworkPolicy = Literal["none", "skill_declared", "open"]
"""Sandbox network access mode.

- ``"none"`` (default): no outbound network at all.
- ``"skill_declared"``: only the domains the SKILL.md manifest's
  ``allowed-tools`` block whitelists (e.g. ``WebFetch(domain:foo)``).
- ``"open"``: full host network. Only safe for trusted skills.
"""


class ResolvedSkillLike(Protocol):
    """Duck-typed view of CARL's ``ResolvedSkill`` (and friends).

    Backends only need the bits they actually use: a stable
    identifier (``sha256``), the on-disk directory holding the skill
    payload (``path``), and the manifest's allowed-tools tokens (for
    ``skill_declared`` networking). Anything more is backend-specific
    and read off ``getattr`` with sensible defaults.
    """

    sha256: str
    path: Path
    allowed_tools: list[str]


@dataclass
class SandboxHandle:
    """Opaque per-prepare ticket the backend uses to track state.

    Stays a plain dataclass so backends can subclass for backend-
    specific extras (``DockerSandboxHandle.container_id``,
    ``E2BSandboxHandle.session_id``). CARE's execution code only
    touches the documented common fields.
    """

    backend_name: str
    workspace: Path
    skill_sha256: str
    network_enforced: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Outcome of one :meth:`SandboxBackend.run` call.

    Mirrors the shape of ``subprocess.CompletedProcess`` but adds
    fields useful for CARE's audit log + UI:

    - ``timed_out``: ``True`` when the run was killed because it
      exceeded the configured ``timeout``.
    - ``duration_seconds``: wall-clock seconds from start to finish.
    - ``network_enforced``: echoed from the handle so the audit log
      records whether network restrictions actually applied.
    """

    exit_code: int
    stdout: bytes
    stderr: bytes
    duration_seconds: float
    timed_out: bool = False
    network_enforced: bool = True

    @property
    def ok(self) -> bool:
        """Conventional "did this succeed" predicate: exit code 0
        and no timeout."""
        return self.exit_code == 0 and not self.timed_out


class SandboxError(RuntimeError):
    """Base for sandbox-level failures (e.g. workspace setup failed,
    network policy unsupported by backend)."""


class SandboxTimeoutError(SandboxError):
    """A ``run()`` call exceeded its ``timeout``. The backend killed
    the process and attached the partial :class:`RunResult` on the
    ``result`` attribute before raising so callers can still log
    duration / exit code / partial stdout."""

    def __init__(self, message: str, *, result: RunResult | None = None) -> None:
        super().__init__(message)
        self.result = result


@runtime_checkable
class SandboxBackend(Protocol):
    """Protocol every concrete sandbox backend implements.

    Backends should be async-first so CARE can interleave multiple
    skill runs without thread-pool contention. Backends that don't
    enforce a given knob (e.g. ``LocalSandboxBackend`` can't really
    cap CPU) document the limitation in their class docstring and
    set the corresponding ``RunResult.network_enforced`` /
    ``SandboxHandle.network_enforced`` field accurately.
    """

    name: str
    """Human-readable backend identifier (e.g. ``"local"``,
    ``"docker"``). Used in audit logs + the TUI banner."""

    unsafe: bool
    """``True`` when the backend cannot enforce isolation (only
    appropriate for development). CARE warns when an ``unsafe``
    backend is selected in a production-looking config."""

    async def prepare(
        self,
        skill: ResolvedSkillLike,
        workspace: Path,
    ) -> SandboxHandle:
        """Set up everything needed to run ``skill`` inside
        ``workspace``. Returns a handle the rest of the lifecycle
        methods key on."""
        ...

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
        """Execute ``cmd`` inside the sandbox.

        Args:
            handle: From a prior :meth:`prepare` call.
            cmd: argv list.
            env: Environment overrides on top of a minimal default.
                ``None`` means use only the backend's defaults.
            stdin: Optional stdin payload to pipe in.
            cpu, mem: Resource limits. Backends that can't enforce
                them log a warning instead of erroring.
            network: One of ``"none" | "skill_declared" | "open"``.
            timeout: Wall-clock seconds before the run is killed.
                ``None`` = no timeout (not recommended).

        Returns:
            A :class:`RunResult`. Raises :class:`SandboxTimeoutError`
            when ``timeout`` is exceeded; other failures inside the
            sandbox surface as a ``RunResult`` with non-zero
            ``exit_code`` rather than an exception.
        """
        ...

    async def read_file(
        self,
        handle: SandboxHandle,
        path: str,
    ) -> bytes:
        """Read ``path`` (relative to the sandbox workspace).

        Implementations MUST reject path-traversal attempts
        (``../``, absolute paths escaping the workspace) so a
        misbehaving skill can't peek at host files.
        """
        ...

    async def write_file(
        self,
        handle: SandboxHandle,
        path: str,
        data: bytes,
    ) -> None:
        """Write ``data`` to ``path`` inside the workspace.

        Same path-traversal restrictions as :meth:`read_file`.
        """
        ...

    async def cleanup(self, handle: SandboxHandle) -> None:
        """Tear down everything :meth:`prepare` allocated.

        Idempotent — calling twice on the same handle is a no-op on
        the second call. Backends that adopt a pre-existing
        workspace (rather than creating one) MUST NOT delete it on
        cleanup.
        """
        ...


__all__ = [
    "NetworkPolicy",
    "ResolvedSkillLike",
    "RunResult",
    "SandboxBackend",
    "SandboxError",
    "SandboxHandle",
    "SandboxTimeoutError",
]
