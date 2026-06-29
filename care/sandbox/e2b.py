"""``E2BSandboxBackend`` — cloud microVM sandbox (TODO §6.1 P1).

For users who can't (or don't want to) run Docker on their CARE
host — laptops without virtualization, locked-down CI runners,
contributors who'd rather not babysit a daemon. Wraps the e2b.dev
Python SDK (`e2b` package) which provisions an isolated microVM
per ``prepare()`` call.

Tradeoffs vs Docker:

* **No local daemon needed** — just an API key + an internet
  connection.
* **Hosted execution** — every run pays a network round-trip; not
  ideal for tight inner-loop iteration.
* **Cost scales with usage** — e2b charges per second of sandbox
  uptime. The :meth:`cleanup` method calls ``sandbox.kill()`` to
  release the VM as soon as the chain finishes.
* **Network policy mapping is approximate.** e2b sandboxes have
  internet by default; we currently surface ``network="open"``
  faithfully but ``"none"`` / ``"skill_declared"`` raise a
  :class:`SandboxError` until e2b adds outbound-network controls.

The Docker-backend's injectable-factory testability pattern
applies here too: a ``sandbox_factory`` callable returns
something duck-typed against ``e2b.Sandbox``, so the test suite
runs without an e2b account.
"""

from __future__ import annotations

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

DEFAULT_TEMPLATE = "base"
"""e2b sandbox template. Override via the constructor when
provisioning a custom image (e.g. ``"python-3.12"`` once it's
published)."""

WORKSPACE_DIR = "/home/user/workspace"
"""Where CARE writes the workspace inside the sandbox. e2b's
default home is ``/home/user``; we anchor under it so the user
account always has write access."""

SKILL_DIR = "/home/user/skill"
"""Where the SKILL.md directory is staged inside the sandbox.
e2b sandboxes don't support read-only mounts the way Docker
does, so this is a regular directory we ``files.write`` SKILL
contents into during ``prepare()``."""


SandboxFactory = Callable[..., Any]
"""Callable returning something duck-typed against ``e2b.Sandbox``.

Production builds use ``e2b.Sandbox`` lazy-imported by
:meth:`E2BSandboxBackend._make_sandbox`; tests pass a stub
factory so no real microVM is provisioned."""


class E2BSandboxBackend:
    """e2b.dev cloud microVM backend.

    Construct with the template name + API key (or set
    ``E2B_API_KEY`` in the environment); per-call settings come
    via :meth:`run`.

    The optional ``sandbox_factory`` constructor argument exists
    solely so tests can inject a stub Sandbox without ever
    pulling the real SDK. Production callers leave it ``None``
    and the backend lazy-imports ``e2b.Sandbox`` on first use.
    """

    name: str = "e2b"
    unsafe: bool = False

    def __init__(
        self,
        *,
        template: str = DEFAULT_TEMPLATE,
        api_key: str | None = None,
        sandbox_factory: SandboxFactory | None = None,
        default_timeout: float = 300.0,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Args:
        template: e2b template identifier. Defaults to
            ``"base"`` — override when the user provisioned a
            CARE-specific image.
        api_key: e2b API key. ``None`` lets the SDK fall back to
            the ``E2B_API_KEY`` environment variable.
        sandbox_factory: Callable that returns a Sandbox
            duck-typed object. ``None`` uses ``e2b.Sandbox``.
        default_timeout: Sandbox-side wall-clock TTL in seconds
            (after which e2b auto-kills the VM regardless of
            what CARE does). 300s matches e2b's own default.
        metadata: Free-form key/value labels forwarded to the
            sandbox at creation. CARE always adds
            ``{"care.sandbox": "true"}`` so its sandboxes are
            easy to identify in the e2b dashboard.
        """
        self._template = template
        self._api_key = api_key
        self._sandbox_factory = sandbox_factory
        self._default_timeout = default_timeout
        self._metadata = dict(metadata or {})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def prepare(
        self,
        skill: ResolvedSkillLike,
        workspace: Path | None = None,  # noqa: ARG002 — e2b is remote, no host workspace
    ) -> SandboxHandle:
        """Provision a microVM, stage the skill files, and
        return a handle.

        Args:
            skill: Anything matching :class:`ResolvedSkillLike`.
                The ``sha256`` becomes a sandbox metadata label;
                the ``path`` is walked + every file uploaded.
            workspace: Accepted for protocol parity but **ignored** —
                e2b sandboxes are remote, so the host workspace
                doesn't exist inside the VM. CARE writes outputs
                to the sandbox-side workspace dir; CARE's
                :meth:`read_file` / :meth:`write_file` round-trip
                through ``files.read`` / ``files.write``.

        Returns:
            :class:`SandboxHandle` with the sandbox object on
            ``extras["sandbox"]``.
        """
        sandbox = self._make_sandbox(skill)
        skill_path = Path(skill.path).resolve()
        if not skill_path.is_dir():
            raise SandboxError(
                f"skill path is not an existing directory: {skill_path}"
            )

        # Stage every regular file under the SKILL dir into the
        # microVM. e2b's `files.write` expects (path, contents)
        # so we walk + relay.
        for file_path in sorted(skill_path.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(skill_path).as_posix()
            target = f"{SKILL_DIR}/{rel}"
            try:
                data = file_path.read_bytes()
            except OSError as exc:
                raise SandboxError(
                    f"could not read SKILL file {file_path}: {exc}"
                ) from exc
            try:
                sandbox.files.write(target, data)
            except Exception as exc:  # noqa: BLE001
                raise SandboxError(
                    f"failed to stage {rel} into e2b sandbox: {exc}"
                ) from exc

        return SandboxHandle(
            backend_name=self.name,
            workspace=Path(WORKSPACE_DIR),
            skill_sha256=skill.sha256,
            network_enforced=False,
            extras={
                "sandbox": sandbox,
                "sandbox_id": getattr(sandbox, "sandbox_id", None),
                "cleaned": False,
                "workspace_mount": WORKSPACE_DIR,
                "skill_mount": SKILL_DIR,
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
        stdin: bytes | None = None,  # noqa: ARG002 — e2b commands.run doesn't pipe stdin
        cpu: float | None = None,  # noqa: ARG002 — template-level
        mem: str | None = None,  # noqa: ARG002 — template-level
        network: NetworkPolicy = "none",
        timeout: float | None = None,
    ) -> RunResult:
        """Execute ``cmd`` in the sandbox via
        ``sandbox.commands.run``.

        Args:
            handle: From a prior :meth:`prepare`.
            cmd: argv list (must be non-empty). e2b's
                ``commands.run`` accepts a shell string, so we
                join with ``shlex.join`` so the shell sees the
                arguments quoted correctly.
            env: Per-run env overrides.
            stdin: Accepted but not piped — e2b's commands API
                doesn't support stdin out of the box.
            cpu / mem: Accepted but ignored. e2b enforces these
                at the template level, not per-command.
            network: ``"open"`` is the only supported policy
                today; ``"none"`` / ``"skill_declared"`` raise
                :class:`SandboxError` until e2b adds outbound
                network controls.
            timeout: Wall-clock seconds before the command is
                killed. Forwarded to e2b verbatim.

        Returns:
            :class:`RunResult`. Raises
            :class:`SandboxTimeoutError` when ``timeout`` fires.
        """
        if not cmd:
            raise SandboxError("cmd must be a non-empty list of argv tokens")
        if network in ("none", "skill_declared"):
            raise SandboxError(
                f"E2BSandboxBackend doesn't support network={network!r} yet "
                "— pass network='open' or use a different backend"
            )
        sandbox = self._sandbox(handle)
        import shlex

        shell_cmd = shlex.join(cmd)
        start = time.monotonic()
        try:
            result = sandbox.commands.run(
                shell_cmd,
                envs=env or {},
                cwd=WORKSPACE_DIR,
                timeout=timeout,
            )
        except _TimeoutLike as exc:
            duration = time.monotonic() - start
            partial = RunResult(
                exit_code=-9,
                stdout=_coerce_bytes(getattr(exc, "stdout", b"")),
                stderr=_coerce_bytes(getattr(exc, "stderr", b"")),
                duration_seconds=duration,
                timed_out=True,
                network_enforced=False,
            )
            raise SandboxTimeoutError(
                f"e2b command timed out after {timeout}s: {cmd[0]}",
                result=partial,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            # If the caller (or stub) reraises a real
            # `TimeoutError`, surface it as timeout the same way.
            if isinstance(exc, TimeoutError):
                duration = time.monotonic() - start
                partial = RunResult(
                    exit_code=-9,
                    stdout=b"",
                    stderr=b"",
                    duration_seconds=duration,
                    timed_out=True,
                    network_enforced=False,
                )
                raise SandboxTimeoutError(
                    f"e2b command timed out after {timeout}s: {cmd[0]}",
                    result=partial,
                ) from exc
            raise SandboxError(
                f"sandbox.commands.run failed: {exc}"
            ) from exc

        duration = time.monotonic() - start
        return RunResult(
            exit_code=int(getattr(result, "exit_code", 0) or 0),
            stdout=_coerce_bytes(getattr(result, "stdout", b"")),
            stderr=_coerce_bytes(getattr(result, "stderr", b"")),
            duration_seconds=duration,
            timed_out=False,
            network_enforced=False,
        )

    async def read_file(
        self,
        handle: SandboxHandle,
        path: str,
    ) -> bytes:
        """Read ``path`` from the sandbox workspace via
        ``files.read``. Path-traversal guard applies."""
        target = _resolve_sandbox_path(WORKSPACE_DIR, path)
        sandbox = self._sandbox(handle)
        try:
            data = sandbox.files.read(target)
        except FileNotFoundError as exc:
            raise SandboxError(
                f"file not found in workspace: {path}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise SandboxError(
                f"could not read {path} from e2b sandbox: {exc}"
            ) from exc
        return _coerce_bytes(data)

    async def write_file(
        self,
        handle: SandboxHandle,
        path: str,
        data: bytes,
    ) -> None:
        """Write ``data`` to ``path`` in the sandbox workspace."""
        target = _resolve_sandbox_path(WORKSPACE_DIR, path)
        sandbox = self._sandbox(handle)
        try:
            sandbox.files.write(target, data)
        except Exception as exc:  # noqa: BLE001
            raise SandboxError(
                f"could not write {path} to e2b sandbox: {exc}"
            ) from exc

    async def cleanup(self, handle: SandboxHandle) -> None:
        """Kill the microVM. Idempotent."""
        if handle.extras.get("cleaned"):
            return
        sandbox = handle.extras.get("sandbox")
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception:  # noqa: BLE001
                # Sandbox may already be gone; nothing to do.
                pass
        handle.extras["cleaned"] = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_sandbox(self, skill: ResolvedSkillLike) -> Any:
        factory = self._sandbox_factory
        if factory is None:
            try:
                from e2b import Sandbox
            except ImportError as exc:
                raise SandboxError(
                    "e2b SDK is not installed; "
                    "install with `pip install \"care[e2b]\"` to use "
                    "the E2BSandboxBackend"
                ) from exc
            factory = Sandbox  # type: ignore[assignment]
        merged_metadata = dict(self._metadata)
        merged_metadata.setdefault("care.sandbox", "true")
        merged_metadata.setdefault("care.sandbox.skill_sha256", skill.sha256)
        try:
            return factory(
                template=self._template,
                api_key=self._api_key,
                timeout=self._default_timeout,
                metadata=merged_metadata,
            )
        except Exception as exc:  # noqa: BLE001
            raise SandboxError(
                f"failed to provision e2b sandbox: {exc}"
            ) from exc

    @staticmethod
    def _sandbox(handle: SandboxHandle) -> Any:
        sandbox = handle.extras.get("sandbox")
        if sandbox is None:
            raise SandboxError(
                "sandbox handle has no live sandbox — was it cleaned up?"
            )
        return sandbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TimeoutLike(Exception):
    """Sentinel base so we can pattern-match e2b's timeout
    exception class lazily.

    The real e2b SDK raises a custom timeout subclass; we don't
    want to hard-import it from a deep submodule. Tests pass any
    Exception they like as long as it isinstance-resembles a
    timeout — handled in the broad ``except Exception`` branch
    below.
    """


def _resolve_sandbox_path(workspace: str, path: str) -> str:
    """Validate ``path`` doesn't escape the sandbox workspace.

    e2b stores files at remote paths so we can't ``Path.resolve``
    against the host; instead reject any ``..`` or absolute
    component up front + then prepend the workspace.
    """
    if not path:
        raise SandboxError("path must not be empty")
    if path.startswith("/"):
        raise SandboxError(
            f"absolute path not allowed inside sandbox: {path!r}"
        )
    parts = [p for p in path.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise SandboxError(f"path escapes workspace: {path!r}")
    return f"{workspace}/{'/'.join(parts)}"


def _coerce_bytes(value: Any) -> bytes:
    """Map e2b's ``str`` returns into ``bytes`` for the shared
    :class:`RunResult` shape."""
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
    "DEFAULT_TEMPLATE",
    "E2BSandboxBackend",
    "SKILL_DIR",
    "SandboxFactory",
    "WORKSPACE_DIR",
]
