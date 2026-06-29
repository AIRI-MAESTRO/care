"""CARL ``SkillRuntime`` adapter over CARE's ``SandboxBackend``
(TODO §6.2 P0).

CARL's ``AgentSkillStepExecutor`` looks up its runtime by name from
a registry: when a chain hits an ``agent_skill`` step,
``mmar_carl.get_skill_runtime(name)`` returns a runtime that
satisfies the ``SkillRuntime`` protocol. CARE wants every skill
execution to flow through **its own** sandbox layer (so CARE's
trust prompts, network-policy translation, audit log, and chosen
backend kind all apply) — this module is the bridge.

`CareSkillRuntime` implements CARL's protocol but delegates every
operation to a CARE :class:`SandboxBackend`. Source CARL gets a
:class:`SkillRuntimeHandle`; CARE's backend gets the matching
:class:`SandboxHandle` stashed in the handle's ``backend`` dict.

Registration::

    from care.runtime import CareSkillRuntime, register_with_carl
    from care.sandbox import LocalSandboxBackend

    runtime = CareSkillRuntime(
        backend=LocalSandboxBackend(),
        sandbox_config=cfg.sandbox,
        trust_store=trust_store,
    )
    register_with_carl(runtime, name="local")  # overrides CARL default

CARL is imported **lazily** inside :func:`register_with_carl` so a
broken CARL install doesn't break CARE startup. Tests use the
bridge directly without touching CARL's registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from care.config import SandboxConfig
from care.sandbox import (
    NetworkPolicy,
    SandboxBackend,
    SandboxHandle,
    SkillTrustStore,
)
from care.sandbox.network_policy import resolve_network_policy


class TrustRefusedError(RuntimeError):
    """A skill whose SHA isn't in the trust store was about to run.

    Raised by :meth:`CareSkillRuntime.prepare` when the runtime is
    configured with ``strict_trust=True`` and the skill hasn't been
    approved. CARE's UI surfaces this as a "Trust this skill?" prompt
    that, on accept, registers the SHA with the store and retries.
    """


@dataclass
class _BridgeHandle:
    """What we stash in CARL's ``SkillRuntimeHandle.backend`` dict.

    Holds the CARE sandbox handle so :meth:`CareSkillRuntime.run`
    can hand it back to the underlying backend, plus the per-prepare
    network policy + resource limits the chain author chose at
    `prepare` time."""

    care_handle: SandboxHandle
    cpu_limit: float
    mem_limit: str
    network: NetworkPolicy


class CareSkillRuntime:
    """Adapter: implement CARL's ``SkillRuntime`` Protocol by
    delegating to a CARE :class:`SandboxBackend`.

    The class avoids hard imports of ``mmar_carl`` at module load —
    CARL types are only touched inside :meth:`prepare` (creating a
    ``SkillRuntimeHandle``) and :meth:`run` (creating a
    ``RuntimeRunResult``). Both imports are lazy so a broken CARL
    install doesn't break CARE startup; the bridge is constructable
    and testable without CARL present (tests can pass a fake-handle
    factory via :meth:`_handle_class` if they want).

    Args:
        backend: The CARE :class:`SandboxBackend` to delegate to.
        sandbox_config: A :class:`care.config.SandboxConfig` — its
            ``cpu_limit`` / ``mem_limit`` / ``network_policy`` are
            applied to every ``run`` call (CARL doesn't surface
            those knobs on ``run`` itself; they live on ``prepare``'s
            ``config`` dict instead).
        trust_store: Optional :class:`SkillTrustStore`. When given +
            ``strict_trust=True``, :meth:`prepare` refuses skills
            whose SHA isn't approved.
        strict_trust: Defaults to ``False`` (warn-but-allow); the
            UI flips this to ``True`` once the trust prompt flow
            ships in §6.3.
    """

    def __init__(
        self,
        *,
        backend: SandboxBackend,
        sandbox_config: SandboxConfig,
        trust_store: SkillTrustStore | None = None,
        strict_trust: bool = False,
    ) -> None:
        self._backend = backend
        self._cfg = sandbox_config
        self._trust = trust_store
        self._strict_trust = strict_trust

    # CARL reads `name` off the class to register in its registry.
    @property
    def name(self) -> str:
        """The CARL-side registry key. Mirrors the wrapped backend's
        name so callers can debug which sandbox actually ran a step."""
        return self._backend.name

    @property
    def backend(self) -> SandboxBackend:
        """The wrapped CARE backend. Exposed for tests + advanced
        callers (e.g. inspecting whether `unsafe=True`)."""
        return self._backend

    @property
    def trust_store(self) -> SkillTrustStore | None:
        return self._trust

    # ------------------------------------------------------------------
    # CARL SkillRuntime Protocol methods
    # ------------------------------------------------------------------

    async def prepare(
        self,
        skill: Any,
        workspace: Path | None,
        config: dict[str, Any] | None = None,
    ) -> Any:
        """Build a workspace, run the trust gate, return a CARL handle.

        CARL hands in ``skill`` (a duck-typed object with ``sha256`` /
        ``path`` / ``allowed_tools``), an optional ``workspace`` root
        (CARE creates a tempdir when ``None``), and a ``config`` dict
        with per-run overrides (``timeout`` / ``network_allowlist`` /
        per-skill resource bumps). The resolved CARE
        :class:`SandboxHandle` is stashed in the returned
        ``SkillRuntimeHandle.backend`` dict under ``"care_handle"``.
        """
        self._enforce_trust(skill)

        care_handle = await self._backend.prepare(skill, workspace)

        # Conventional in/out subdirs CARL skills write to.
        workspace_in = care_handle.workspace / "in"
        workspace_out = care_handle.workspace / "out"
        workspace_in.mkdir(parents=True, exist_ok=True)
        workspace_out.mkdir(parents=True, exist_ok=True)

        cpu, mem, network = self._resolve_per_run_limits(skill, config)
        bridge = _BridgeHandle(
            care_handle=care_handle,
            cpu_limit=cpu,
            mem_limit=mem,
            network=network,
        )
        handle_cls = self._handle_class()
        return handle_cls(
            workspace_root=care_handle.workspace,
            workspace_in=workspace_in,
            workspace_out=workspace_out,
            backend={
                "care_handle": care_handle,
                "bridge": bridge,
                "backend_name": self._backend.name,
                "network_enforced": care_handle.network_enforced,
            },
        )

    async def run(
        self,
        handle: Any,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        timeout: float | None = None,
        cwd: str | None = None,  # noqa: ARG002 — CARL passes; we honour via cwd in backend
    ) -> Any:
        """Forward ``cmd`` to CARE's backend and wrap the result in
        CARL's :class:`RuntimeRunResult`."""
        bridge = self._bridge_for(handle)
        care_result = await self._backend.run(
            bridge.care_handle,
            cmd,
            env=env,
            stdin=stdin,
            cpu=bridge.cpu_limit,
            mem=bridge.mem_limit,
            network=bridge.network,
            timeout=timeout,
        )
        result_cls = self._result_class()
        return result_cls(
            stdout=care_result.stdout,
            stderr=care_result.stderr,
            exit_code=care_result.exit_code,
            duration_s=care_result.duration_seconds,
        )

    async def read_file(self, handle: Any, path: str) -> bytes:
        return await self._backend.read_file(self._care_handle(handle), path)

    async def write_file(self, handle: Any, path: str, data: bytes) -> None:
        await self._backend.write_file(self._care_handle(handle), path, data)

    async def cleanup(self, handle: Any) -> None:
        await self._backend.cleanup(self._care_handle(handle))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enforce_trust(self, skill: Any) -> None:
        """Trust gate. ``strict_trust=False`` (default) is a warn-
        but-allow flow until the first-run prompt UI ships."""
        if self._trust is None:
            return
        sha = getattr(skill, "sha256", None) or (
            skill.get("sha256") if isinstance(skill, dict) else None
        )
        if sha is None:
            return
        if self._trust.is_trusted(sha):
            return
        if self._strict_trust:
            raise TrustRefusedError(
                f"Skill SHA {sha!r} is not in the trust store; refusing to run."
            )

    def _resolve_per_run_limits(
        self,
        skill: Any,
        config: dict[str, Any] | None,
    ) -> tuple[float, str, NetworkPolicy]:
        """Layer per-skill overrides on top of CareConfig defaults.

        ``config`` is CARL's ``runtime_config`` dict — known keys CARE
        respects: ``cpu_limit``, ``mem_limit``, ``network``,
        ``network_allowlist`` (the last one is read by future
        backends via :func:`resolve_network_policy`).
        """
        cfg = config or {}
        cpu = float(cfg.get("cpu_limit", self._cfg.cpu_limit))
        mem = str(cfg.get("mem_limit", self._cfg.mem_limit))
        raw_net = cfg.get("network", self._cfg.network_policy)
        # Validate via the existing policy resolver — keeps the
        # error message + literal set in one place.
        allowed_tools = getattr(skill, "allowed_tools", None)
        if allowed_tools is None and isinstance(skill, dict):
            allowed_tools = skill.get("allowed_tools")
        resolved = resolve_network_policy(
            raw_net,  # type: ignore[arg-type]
            allowed_tools=allowed_tools,
            override_domains=cfg.get("network_allowlist"),
        )
        return cpu, mem, resolved.policy

    @staticmethod
    def _bridge_for(handle: Any) -> _BridgeHandle:
        bridge = handle.backend.get("bridge") if hasattr(handle, "backend") else None
        if not isinstance(bridge, _BridgeHandle):
            raise RuntimeError(
                "CareSkillRuntime handle is missing its bridge — was it "
                "constructed by a different runtime?"
            )
        return bridge

    @classmethod
    def _care_handle(cls, handle: Any) -> SandboxHandle:
        return cls._bridge_for(handle).care_handle

    # Lazy import isolation: tests / smoke runs can monkey-patch these.

    @staticmethod
    def _handle_class() -> type:
        """Import CARL's ``SkillRuntimeHandle`` lazily."""
        from mmar_carl.skill_runtime import SkillRuntimeHandle

        return SkillRuntimeHandle

    @staticmethod
    def _result_class() -> type:
        """Import CARL's ``RuntimeRunResult`` lazily."""
        from mmar_carl.skill_runtime import RuntimeRunResult

        return RuntimeRunResult


# Module-level singletons that the shim class consults at instantiation
# time. CARL's registry takes a no-args class and calls ``cls()`` itself,
# so we need this indirection to feed in the configured runtime.
_REGISTERED_RUNTIMES: dict[str, CareSkillRuntime] = {}


def _build_shim_class(registry_name: str) -> type:
    """Create a no-args wrapper class CARL can register and call ``cls()`` on.

    The wrapper looks up the configured :class:`CareSkillRuntime` from
    :data:`_REGISTERED_RUNTIMES` on instantiation and forwards every
    protocol method to it. One subclass per registered name keeps
    different ``register_with_carl`` calls from clobbering each other.
    """

    class _CareRuntimeShim:
        def __init__(self) -> None:
            self._delegate = _REGISTERED_RUNTIMES[type(self).name]

        async def prepare(self, skill, workspace, config=None):
            return await self._delegate.prepare(skill, workspace, config)

        async def run(self, handle, cmd, *, env=None, stdin=None, timeout=None, cwd=None):
            return await self._delegate.run(
                handle,
                cmd,
                env=env,
                stdin=stdin,
                timeout=timeout,
                cwd=cwd,
            )

        async def read_file(self, handle, path):
            return await self._delegate.read_file(handle, path)

        async def write_file(self, handle, path, data):
            await self._delegate.write_file(handle, path, data)

        async def cleanup(self, handle):
            await self._delegate.cleanup(handle)

    # CARL reads the registry key from the class-level ``name`` attribute
    # (declared as a ``ClassVar`` on the SkillRuntime Protocol). Setting
    # it after class creation avoids the class-body-scoping issue with
    # closures over ``registry_name``.
    _CareRuntimeShim.name = registry_name  # type: ignore[attr-defined]
    _CareRuntimeShim.__qualname__ = f"_CareRuntimeShim_{registry_name}"
    _CareRuntimeShim.__name__ = f"_CareRuntimeShim_{registry_name}"
    return _CareRuntimeShim


def register_with_carl(
    runtime: CareSkillRuntime,
    *,
    name: str | None = None,
) -> str:
    """Wire ``runtime`` into CARL's ``SKILL_RUNTIME_REGISTRY``.

    CARL's registry takes a **class** (not an instance) and calls
    ``cls()`` later with no args, so this helper:

    1. Stores ``runtime`` in :data:`_REGISTERED_RUNTIMES` keyed by
       ``name``.
    2. Builds a tiny shim class whose ``__init__`` looks up the
       runtime from that dict and forwards every protocol method
       to it.
    3. Calls CARL's ``register_skill_runtime(name, shim_cls)``.

    Args:
        runtime: A constructed :class:`CareSkillRuntime`.
        name: Override the registry key. Defaults to
            ``runtime.name`` (the wrapped backend's name).

    Returns:
        The registry name actually used. CARE-side call-sites can
        compare against :func:`mmar_carl.list_skill_runtimes` for
        post-registration assertions.
    """
    from mmar_carl.skill_runtime import register_skill_runtime

    registry_name = name or runtime.name
    _REGISTERED_RUNTIMES[registry_name] = runtime
    shim_cls = _build_shim_class(registry_name)
    register_skill_runtime(registry_name, shim_cls)
    return registry_name


def unregister(name: str) -> bool:
    """Drop a CARE runtime registration. Returns whether anything
    was removed. Mainly for tests; CARL doesn't expose its own
    unregister, so the shim class stays in CARL's registry but
    its delegate lookup will start raising ``KeyError`` until a
    new :func:`register_with_carl` re-binds the name."""
    if name not in _REGISTERED_RUNTIMES:
        return False
    del _REGISTERED_RUNTIMES[name]
    return True


__all__ = [
    "CareSkillRuntime",
    "TrustRefusedError",
    "register_with_carl",
    "unregister",
]
