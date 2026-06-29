"""Tests for ``care.runtime.skill_runtime_adapter`` (TODO §6.2 P0).

Strategy: instead of importing the real `mmar_carl` package (which
the adapter explicitly defers to keep CARE startup independent of
CARL install state), the tests monkey-patch
:meth:`CareSkillRuntime._handle_class` and `._result_class` with
local lookalikes. That exercises every protocol method end-to-end
against the real :class:`LocalSandboxBackend` without needing CARL.

Coverage layers:
1. Trust gate: skill with untrusted SHA + strict mode raises;
   warn-but-allow lets the run proceed.
2. Per-run limits resolution: CareConfig defaults applied, per-skill
   overrides win, allowed_tools fed through the network-policy
   resolver.
3. End-to-end protocol delegation: prepare → run echo → read/write
   workspace files → cleanup, all against real subprocesses.
4. Registration helper: stores runtime in module-level dict and
   builds a no-args shim class that forwards every method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from care.config import CareConfig
from care.runtime import (
    CareSkillRuntime,
    TrustRefusedError,
    register_with_carl,
    unregister,
)
from care.runtime import skill_runtime_adapter as adapter_mod
from care.sandbox import LocalSandboxBackend, SkillTrustStore


# ---------------------------------------------------------------------------
# Fakes that stand in for CARL's SkillRuntimeHandle + RuntimeRunResult.
# Same field names + initializer keywords; the adapter is duck-typed.
# ---------------------------------------------------------------------------


@dataclass
class _FakeHandle:
    workspace_root: Path
    workspace_in: Path
    workspace_out: Path
    backend: dict = field(default_factory=dict)


@dataclass
class _FakeRunResult:
    stdout: bytes
    stderr: bytes
    exit_code: int
    duration_s: float


@dataclass
class _FakeSkill:
    sha256: str = "a" * 64
    path: Path = Path("/tmp/fake-skill")
    allowed_tools: list[str] = field(default_factory=list)


def _mkws(tmp_path: Path) -> Path:
    """Create + return a fresh workspace dir under ``tmp_path``."""
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return ws


@pytest.fixture(autouse=True)
def patch_carl_classes(monkeypatch):
    """Swap the lazy CARL imports for our local fakes so the tests
    run without needing ``mmar_carl`` installed."""
    monkeypatch.setattr(CareSkillRuntime, "_handle_class", staticmethod(lambda: _FakeHandle))
    monkeypatch.setattr(CareSkillRuntime, "_result_class", staticmethod(lambda: _FakeRunResult))


@pytest.fixture
def cfg(tmp_path):
    return CareConfig.load(path=tmp_path / "missing.toml", env={})


@pytest.fixture
def runtime(cfg):
    return CareSkillRuntime(
        backend=LocalSandboxBackend(),
        sandbox_config=cfg.sandbox,
    )


# ---------------------------------------------------------------------------
# Construction + introspection
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_name_mirrors_backend(self, runtime):
        assert runtime.name == "local"

    def test_backend_accessor_returns_wrapped_backend(self, runtime):
        assert isinstance(runtime.backend, LocalSandboxBackend)
        assert runtime.backend.unsafe is True

    def test_trust_store_none_by_default(self, runtime):
        assert runtime.trust_store is None


# ---------------------------------------------------------------------------
# Trust gate
# ---------------------------------------------------------------------------


class TestTrustGate:
    @pytest.mark.asyncio
    async def test_strict_raises_when_sha_untrusted(self, cfg, tmp_path):
        trust = SkillTrustStore.load(path=tmp_path / "trust.json")
        rt = CareSkillRuntime(
            backend=LocalSandboxBackend(),
            sandbox_config=cfg.sandbox,
            trust_store=trust,
            strict_trust=True,
        )
        with pytest.raises(TrustRefusedError, match="not in the trust store"):
            await rt.prepare(_FakeSkill(), workspace=None)

    @pytest.mark.asyncio
    async def test_strict_allows_when_sha_trusted(self, cfg, tmp_path):
        trust = SkillTrustStore.load(path=tmp_path / "trust.json")
        trust.trust(sha256="a" * 64, uri="local:///x", name="x")
        rt = CareSkillRuntime(
            backend=LocalSandboxBackend(),
            sandbox_config=cfg.sandbox,
            trust_store=trust,
            strict_trust=True,
        )
        handle = await rt.prepare(_FakeSkill(), workspace=_mkws(tmp_path))
        # Survived the gate; clean up.
        await rt.cleanup(handle)

    @pytest.mark.asyncio
    async def test_warn_but_allow_when_not_strict(self, cfg, tmp_path):
        """Default ``strict_trust=False`` lets untrusted skills run."""
        trust = SkillTrustStore.load(path=tmp_path / "trust.json")
        rt = CareSkillRuntime(
            backend=LocalSandboxBackend(),
            sandbox_config=cfg.sandbox,
            trust_store=trust,
        )
        handle = await rt.prepare(_FakeSkill(), workspace=None)
        await rt.cleanup(handle)

    @pytest.mark.asyncio
    async def test_no_trust_store_no_gate(self, runtime, tmp_path):
        handle = await runtime.prepare(_FakeSkill(), workspace=_mkws(tmp_path))
        await runtime.cleanup(handle)


# ---------------------------------------------------------------------------
# Per-run limits resolution
# ---------------------------------------------------------------------------


class TestLimitsResolution:
    @pytest.mark.asyncio
    async def test_defaults_from_care_config(self, runtime, tmp_path):
        handle = await runtime.prepare(_FakeSkill(), workspace=_mkws(tmp_path))
        bridge = handle.backend["bridge"]
        assert bridge.cpu_limit == 2.0  # from SandboxConfig default
        assert bridge.mem_limit == "1g"
        # default network policy in SandboxConfig is "skill_declared"
        assert bridge.network == "skill_declared"
        await runtime.cleanup(handle)

    @pytest.mark.asyncio
    async def test_per_run_overrides_win(self, runtime, tmp_path):
        handle = await runtime.prepare(
            _FakeSkill(),
            workspace=_mkws(tmp_path),
            config={"cpu_limit": 4.0, "mem_limit": "512m", "network": "none"},
        )
        bridge = handle.backend["bridge"]
        assert bridge.cpu_limit == 4.0
        assert bridge.mem_limit == "512m"
        assert bridge.network == "none"
        await runtime.cleanup(handle)

    @pytest.mark.asyncio
    async def test_unknown_network_policy_raises(self, runtime, tmp_path):
        with pytest.raises(ValueError, match="unknown CARE network policy"):
            await runtime.prepare(
                _FakeSkill(),
                workspace=_mkws(tmp_path),
                config={"network": "bogus"},
            )


# ---------------------------------------------------------------------------
# End-to-end: prepare → run → read/write → cleanup
# ---------------------------------------------------------------------------


class TestProtocolDelegation:
    @pytest.mark.asyncio
    async def test_workspace_layout_includes_in_out(self, runtime, tmp_path):
        handle = await runtime.prepare(_FakeSkill(), workspace=_mkws(tmp_path))
        assert handle.workspace_root == (tmp_path / "ws").resolve()
        assert handle.workspace_in.exists() and handle.workspace_in.is_dir()
        assert handle.workspace_out.exists() and handle.workspace_out.is_dir()
        await runtime.cleanup(handle)

    @pytest.mark.asyncio
    async def test_run_returns_carl_shaped_result(self, runtime, tmp_path):
        handle = await runtime.prepare(_FakeSkill(), workspace=_mkws(tmp_path))
        try:
            result = await runtime.run(handle, ["echo", "hi"])
            assert isinstance(result, _FakeRunResult)
            assert result.exit_code == 0
            assert result.stdout.strip() == b"hi"
            assert result.duration_s >= 0
        finally:
            await runtime.cleanup(handle)

    @pytest.mark.asyncio
    async def test_read_write_file_round_trip(self, runtime, tmp_path):
        handle = await runtime.prepare(_FakeSkill(), workspace=_mkws(tmp_path))
        try:
            await runtime.write_file(handle, "out/report.json", b'{"k":1}')
            data = await runtime.read_file(handle, "out/report.json")
            assert data == b'{"k":1}'
        finally:
            await runtime.cleanup(handle)

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent(self, runtime, tmp_path):
        handle = await runtime.prepare(_FakeSkill(), workspace=_mkws(tmp_path))
        await runtime.cleanup(handle)
        await runtime.cleanup(handle)  # second call: no exception

    @pytest.mark.asyncio
    async def test_bridge_lookup_rejects_foreign_handle(self, runtime):
        """A handle that wasn't built by CareSkillRuntime must fail
        loudly — protects against runtimes getting crossed at run time."""
        foreign = _FakeHandle(
            workspace_root=Path("/x"),
            workspace_in=Path("/x"),
            workspace_out=Path("/x"),
            backend={},  # no "bridge" key
        )
        with pytest.raises(RuntimeError, match="missing its bridge"):
            await runtime.run(foreign, ["echo"])


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


class TestRegisterWithCarl:
    """Patch CARL's ``register_skill_runtime`` so we don't touch the
    real registry — just verify the shim class CARL would call ``cls()``
    on does forward to our runtime."""

    def test_register_stores_runtime_in_module_dict(
        self, runtime, monkeypatch
    ):
        captured: dict = {}

        def fake_register(name, cls):
            captured["name"] = name
            captured["cls"] = cls

        # Patch the lazy import via a stub module so the helper picks
        # up our spy when it does `from mmar_carl.skill_runtime import
        # register_skill_runtime`.
        import sys
        import types

        fake_mod = types.ModuleType("mmar_carl")
        fake_sub = types.ModuleType("mmar_carl.skill_runtime")
        fake_sub.register_skill_runtime = fake_register  # type: ignore[attr-defined]
        fake_mod.skill_runtime = fake_sub  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mmar_carl", fake_mod)
        monkeypatch.setitem(sys.modules, "mmar_carl.skill_runtime", fake_sub)

        try:
            used_name = register_with_carl(runtime, name="local")
            assert used_name == "local"
            assert captured["name"] == "local"
            # The class CARL received is a no-args shim.
            shim_cls = captured["cls"]
            assert shim_cls.__name__ == "_CareRuntimeShim_local"
            assert shim_cls.name == "local"
            instance = shim_cls()
            # Forwarding works: protocol methods delegate to runtime.
            assert instance._delegate is runtime
        finally:
            unregister("local")

    def test_shim_class_raises_when_unregistered(self, runtime, monkeypatch):
        """If the underlying runtime is dropped from the registry,
        the shim class — still in CARL's registry — raises on its
        next instantiation. CARE's tests use this to clean up
        without monkey-patching CARL's internals."""
        captured = {}

        def fake_register(name, cls):
            captured["cls"] = cls

        import sys
        import types

        fake_mod = types.ModuleType("mmar_carl")
        fake_sub = types.ModuleType("mmar_carl.skill_runtime")
        fake_sub.register_skill_runtime = fake_register  # type: ignore[attr-defined]
        fake_mod.skill_runtime = fake_sub  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mmar_carl", fake_mod)
        monkeypatch.setitem(sys.modules, "mmar_carl.skill_runtime", fake_sub)

        register_with_carl(runtime, name="local-temp")
        unregister("local-temp")
        with pytest.raises(KeyError):
            captured["cls"]()  # delegate lookup fails

    def test_unregister_returns_bool(self):
        # No prior registration → False.
        assert unregister("does-not-exist") is False


# ---------------------------------------------------------------------------
# Module-state hygiene
# ---------------------------------------------------------------------------


def test_module_registered_dict_lives_at_known_path():
    """Pin the module-level singleton so tests + future debugging
    know where to look."""
    assert isinstance(adapter_mod._REGISTERED_RUNTIMES, dict)
