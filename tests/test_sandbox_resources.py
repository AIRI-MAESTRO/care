"""Tests for ``care.sandbox.resources`` (TODO §6.2 P1).

Pure-function coverage. Five areas:

1. ``parse_resources_block`` — accepts every documented key shape,
   rejects malformed values loudly.
2. ``resolve_resources`` defaults path — manifest absent → config
   values intact, ``source="config"``.
3. ``resolve_resources`` override path — manifest values applied,
   ``source`` tag reflects partial vs full override.
4. Policy gates — over-ceiling requests raise by default; opt-in
   with ``allow_manifest_upscale=True`` lets them through.
5. ``apply_to_sandbox_config`` — produces a new ``SandboxConfig``
   without mutating the input.
"""

from __future__ import annotations

import pytest

from care.config import SandboxConfig
from care.sandbox import (
    ResolvedResources,
    ResourceOverrideError,
    ResourcePolicy,
    apply_to_sandbox_config,
    parse_resources_block,
    resolve_resources,
)


@pytest.fixture
def defaults() -> SandboxConfig:
    """Plain default config: cpu=2.0, mem=1g, pids=256."""
    return SandboxConfig()


# ---------------------------------------------------------------------------
# parse_resources_block
# ---------------------------------------------------------------------------


class TestParseResourcesBlock:
    def test_none_returns_empty(self):
        assert parse_resources_block(None) == {}

    def test_empty_dict(self):
        assert parse_resources_block({}) == {}

    def test_canonical_keys(self):
        out = parse_resources_block(
            {"cpu": 4, "memory": "2g", "pids": 512, "timeout": 60.5}
        )
        assert out == {
            "cpu": 4.0,
            "memory": "2g",
            "pids": 512,
            "timeout": 60.5,
        }

    @pytest.mark.parametrize(
        "key,value", [("mem", "2g"), ("mem_limit", "2g")]
    )
    def test_memory_aliases(self, key, value):
        out = parse_resources_block({key: value})
        assert out["memory"] == "2g"

    def test_pids_limit_alias(self):
        out = parse_resources_block({"pids_limit": 128})
        assert out["pids"] == 128

    def test_unknown_keys_ignored(self):
        """Forward-compat: a future schema bump that adds a key
        shouldn't break older CARE installs."""
        out = parse_resources_block({"cpu": 1.0, "future_knob": "wtf"})
        assert out == {"cpu": 1.0}

    def test_non_dict_raises(self):
        with pytest.raises(ResourceOverrideError, match="must be a dict"):
            parse_resources_block(["cpu", "1.0"])

    @pytest.mark.parametrize("bad", [-1, 0, "not-a-number"])
    def test_cpu_must_be_positive_numeric(self, bad):
        with pytest.raises(ResourceOverrideError, match="cpu"):
            parse_resources_block({"cpu": bad})

    @pytest.mark.parametrize("bad", ["1024", "1tb", "abc", "1.5g"])
    def test_memory_must_be_docker_style(self, bad):
        with pytest.raises(ResourceOverrideError, match="memory"):
            parse_resources_block({"memory": bad})

    def test_memory_case_normalised(self):
        assert parse_resources_block({"memory": "2G"})["memory"] == "2g"

    @pytest.mark.parametrize("bad", [-1, 0, "abc"])
    def test_pids_must_be_positive_int(self, bad):
        with pytest.raises(ResourceOverrideError, match="pids"):
            parse_resources_block({"pids": bad})


# ---------------------------------------------------------------------------
# resolve_resources: defaults
# ---------------------------------------------------------------------------


class TestResolveDefaults:
    def test_no_manifest(self, defaults):
        r = resolve_resources(defaults=defaults)
        assert r.cpu_limit == defaults.cpu_limit
        assert r.mem_limit == defaults.mem_limit
        assert r.pids_limit == defaults.pids_limit
        assert r.timeout is None
        assert r.source == "config"

    def test_empty_manifest(self, defaults):
        r = resolve_resources(defaults=defaults, manifest_resources={})
        assert r.source == "config"


# ---------------------------------------------------------------------------
# resolve_resources: overrides
# ---------------------------------------------------------------------------


class TestResolveOverrides:
    def test_partial_override_sets_source_mixed(self, defaults):
        r = resolve_resources(
            defaults=defaults,
            manifest_resources={"cpu": 1.0},  # < default 2.0
        )
        assert r.cpu_limit == 1.0
        assert r.mem_limit == defaults.mem_limit
        assert r.pids_limit == defaults.pids_limit
        assert r.source == "mixed"

    def test_full_override_sets_source_manifest(self, defaults):
        r = resolve_resources(
            defaults=defaults,
            manifest_resources={
                "cpu": 1.0,
                "memory": "512m",
                "pids": 64,
                "timeout": 30,
            },
        )
        assert r.cpu_limit == 1.0
        assert r.mem_limit == "512m"
        assert r.pids_limit == 64
        assert r.timeout == 30.0
        assert r.source == "manifest"

    def test_accepts_preparsed_dict(self, defaults):
        """Caller can pass output of ``parse_resources_block`` directly
        — the resolver detects the canonical shape and skips re-validation."""
        parsed = parse_resources_block({"cpu": 1.5})
        r = resolve_resources(defaults=defaults, manifest_resources=parsed)
        assert r.cpu_limit == 1.5

    def test_accepts_raw_dict(self, defaults):
        """Raw dict with manifest-style keys is parsed inline."""
        r = resolve_resources(
            defaults=defaults,
            manifest_resources={"cpu": 1.5, "memory": "512m"},
        )
        assert r.cpu_limit == 1.5
        assert r.mem_limit == "512m"


# ---------------------------------------------------------------------------
# Policy gates
# ---------------------------------------------------------------------------


class TestPolicyGates:
    def test_default_policy_blocks_over_ceiling_cpu(self, defaults):
        with pytest.raises(ResourceOverrideError, match="exceeds operator ceiling"):
            resolve_resources(
                defaults=defaults,
                manifest_resources={"cpu": 99.0},  # >> default 2.0
            )

    def test_default_policy_blocks_over_ceiling_memory(self, defaults):
        with pytest.raises(ResourceOverrideError, match="exceeds operator ceiling"):
            resolve_resources(
                defaults=defaults,
                manifest_resources={"memory": "16g"},  # >> default 1g
            )

    def test_default_policy_blocks_over_ceiling_pids(self, defaults):
        with pytest.raises(ResourceOverrideError, match="exceeds operator ceiling"):
            resolve_resources(
                defaults=defaults,
                manifest_resources={"pids": 9999},
            )

    def test_opt_in_allows_upscale(self, defaults):
        r = resolve_resources(
            defaults=defaults,
            manifest_resources={
                "cpu": 8.0,
                "memory": "4g",
                "pids": 2048,
            },
            policy=ResourcePolicy(allow_manifest_upscale=True),
        )
        assert r.cpu_limit == 8.0
        assert r.mem_limit == "4g"
        assert r.pids_limit == 2048

    def test_memory_within_ceiling_still_works(self, defaults):
        """Manifest can lower a value even with strict policy."""
        r = resolve_resources(
            defaults=defaults,
            manifest_resources={"memory": "256m"},  # < default 1g
        )
        assert r.mem_limit == "256m"


# ---------------------------------------------------------------------------
# apply_to_sandbox_config
# ---------------------------------------------------------------------------


class TestApplyToSandboxConfig:
    def test_returns_new_config_with_resolved_fields(self, defaults):
        resolved = ResolvedResources(
            cpu_limit=1.0,
            mem_limit="512m",
            pids_limit=64,
            timeout=30.0,
            source="manifest",
        )
        out = apply_to_sandbox_config(defaults, resolved)
        assert out.cpu_limit == 1.0
        assert out.mem_limit == "512m"
        assert out.pids_limit == 64
        # Non-resource fields preserved.
        assert out.kind == defaults.kind
        assert out.network_policy == defaults.network_policy
        # Original config unmutated.
        assert defaults.cpu_limit == 2.0

    def test_timeout_not_persisted_on_sandbox_config(self, defaults):
        """SandboxConfig has no timeout field; timeout is per-run."""
        resolved = ResolvedResources(
            cpu_limit=2.0, mem_limit="1g", pids_limit=256, timeout=60.0
        )
        out = apply_to_sandbox_config(defaults, resolved)
        assert not hasattr(out, "timeout")


# ---------------------------------------------------------------------------
# ResolvedResources shape
# ---------------------------------------------------------------------------


class TestResolvedResourcesShape:
    def test_frozen(self):
        r = ResolvedResources(
            cpu_limit=2.0, mem_limit="1g", pids_limit=256
        )
        with pytest.raises(AttributeError):
            r.cpu_limit = 4.0  # type: ignore[misc]

    def test_default_source_config(self):
        r = ResolvedResources(
            cpu_limit=2.0, mem_limit="1g", pids_limit=256
        )
        assert r.source == "config"
        assert r.timeout is None
