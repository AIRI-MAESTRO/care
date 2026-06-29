"""Per-skill resource-limit overrides parsed from SKILL.md
manifests (TODO §6.2 P1).

Skills can self-declare resource needs in a ``metadata.resources``
block of their SKILL.md frontmatter::

    metadata:
      resources:
        cpu: 4.0          # cores; float
        memory: 2g        # Docker-style suffix string
        pids: 512         # integer process cap
        timeout: 120      # seconds; float

CARE merges those declared values on top of the ``SandboxConfig``
defaults (which set the ambient cpu/mem/pids/timeout for every
skill). The skill's declarations **lower** the limit when stricter
than the default (defensive default) but never exceed the
operator's ceiling — config wins on conflicts unless the operator
explicitly opted into manifest-trust via
``ResourcePolicy.allow_manifest_upscale=True``.

This module owns just the parse + merge logic. Wiring into
:class:`CareSkillRuntime` is a follow-up — exposing the parsed
:class:`ResolvedResources` lets the adapter consult it without
re-validating shapes per call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from care.config import SandboxConfig

_MEM_SUFFIX_RE = re.compile(r"^(\d+)([kmgKMG])$")
"""Match Docker-style memory strings: digits + single k/m/g suffix."""


class ResourceOverrideError(ValueError):
    """Raised when a manifest's ``metadata.resources`` block carries
    a value the loader can't parse / can't safely apply (e.g. a
    negative cpu, an unknown memory suffix, or — in default policy
    — a request larger than the operator ceiling)."""


@dataclass(frozen=True)
class ResourcePolicy:
    """How CARE treats per-skill manifest requests.

    Default is the **safe / paranoid** mode: manifests can lower
    their own cpu/mem/pids/timeout below the operator ceiling but
    cannot exceed it. Operators who trust their skill set flip
    ``allow_manifest_upscale=True`` (lets the manifest pin a
    higher ceiling for that one run — useful for batched data jobs
    that legitimately need more headroom).
    """

    allow_manifest_upscale: bool = False
    """When ``True``, a manifest can request *more* cpu / memory /
    pids than ``CareConfig.sandbox`` defaults. When ``False``
    (default), manifest requests get clamped to the config
    ceiling and a :class:`ResourceOverrideError` is raised when
    the manifest asks for something the operator hasn't
    permitted."""


@dataclass(frozen=True)
class ResolvedResources:
    """Concrete per-run limits after merging defaults + manifest.

    Mirrors the subset of ``SandboxConfig`` that backends actually
    consume during ``run()``. Frozen so it can hash into audit-log
    entries + flow through Textual messages without defensive
    copies.
    """

    cpu_limit: float
    mem_limit: str
    pids_limit: int
    timeout: float | None = None
    source: str = "config"
    """Where each field's final value came from in aggregate —
    one of ``"config"`` (nothing overridden), ``"manifest"`` (every
    field overridden), or ``"mixed"`` (partial override). Useful
    for the audit log + the TUI status banner."""


def parse_resources_block(
    raw: Any,
) -> dict[str, float | int | str]:
    """Validate + normalise a ``metadata.resources`` dict from a
    SKILL.md manifest.

    Accepts the documented keys (``cpu``, ``memory`` / ``mem`` /
    ``mem_limit``, ``pids`` / ``pids_limit``, ``timeout``). Unknown
    keys are ignored (a future schema bump can add fields without
    breaking older CARE installs). Missing / ``None`` ``raw`` →
    empty dict.

    Raises:
        ResourceOverrideError: When ``raw`` isn't a dict, or any
            recognised key carries a malformed value
            (negative cpu, bad memory suffix, etc.).
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ResourceOverrideError(
            f"metadata.resources must be a dict; got {type(raw).__name__}"
        )

    out: dict[str, float | int | str] = {}

    if "cpu" in raw:
        out["cpu"] = _parse_positive_float(raw["cpu"], field="cpu")

    if "memory" in raw or "mem" in raw or "mem_limit" in raw:
        value = raw.get("memory", raw.get("mem", raw.get("mem_limit")))
        out["memory"] = _parse_memory(value)

    if "pids" in raw or "pids_limit" in raw:
        value = raw.get("pids", raw.get("pids_limit"))
        out["pids"] = _parse_positive_int(value, field="pids")

    if "timeout" in raw:
        out["timeout"] = _parse_positive_float(raw["timeout"], field="timeout")

    return out


def resolve_resources(
    *,
    defaults: SandboxConfig,
    manifest_resources: Any = None,
    policy: ResourcePolicy | None = None,
) -> ResolvedResources:
    """Merge ``defaults`` with a parsed ``metadata.resources`` block.

    Args:
        defaults: The :class:`SandboxConfig` from ``CareConfig``.
        manifest_resources: Either the raw dict from a SKILL.md
            manifest or a pre-parsed dict from
            :func:`parse_resources_block`. Either works.
        policy: How to treat manifest values that exceed defaults.
            ``None`` uses :class:`ResourcePolicy()` (safe defaults).

    Returns:
        :class:`ResolvedResources`. The ``source`` field reports
        whether the manifest contributed anything (``config`` /
        ``manifest`` / ``mixed``) so audit + UI can render it.
    """
    policy = policy or ResourcePolicy()

    # Allow caller to pass either raw or pre-parsed.
    if manifest_resources is None:
        parsed: dict[str, float | int | str] = {}
    elif _looks_pre_parsed(manifest_resources):
        parsed = dict(manifest_resources)  # already validated
    else:
        parsed = parse_resources_block(manifest_resources)

    cpu, cpu_overridden = _merge_numeric(
        default=defaults.cpu_limit,
        requested=parsed.get("cpu"),
        field="cpu",
        allow_upscale=policy.allow_manifest_upscale,
    )
    pids, pids_overridden = _merge_numeric(
        default=defaults.pids_limit,
        requested=parsed.get("pids"),
        field="pids",
        allow_upscale=policy.allow_manifest_upscale,
    )

    mem_value = parsed.get("memory")
    mem_overridden = False
    mem_limit = defaults.mem_limit
    if mem_value is not None:
        mem_limit = _merge_memory(
            default=defaults.mem_limit,
            requested=str(mem_value),
            allow_upscale=policy.allow_manifest_upscale,
        )
        mem_overridden = mem_limit != defaults.mem_limit or (
            policy.allow_manifest_upscale and mem_limit == str(mem_value).lower()
        )

    timeout_value = parsed.get("timeout")
    timeout_overridden = timeout_value is not None
    timeout = float(timeout_value) if timeout_value is not None else None

    overrides = [
        cpu_overridden,
        mem_overridden,
        pids_overridden,
        timeout_overridden,
    ]
    if not any(overrides):
        source = "config"
    elif all(overrides):
        source = "manifest"
    else:
        source = "mixed"

    return ResolvedResources(
        cpu_limit=cpu,
        mem_limit=mem_limit,
        pids_limit=int(pids),
        timeout=timeout,
        source=source,
    )


def apply_to_sandbox_config(
    defaults: SandboxConfig,
    resolved: ResolvedResources,
) -> SandboxConfig:
    """Return a new :class:`SandboxConfig` with the resolved
    cpu/mem/pids applied. Useful when a backend wants a
    fully-merged config object without juggling two value sets.
    Timeout is not stored on ``SandboxConfig`` (it's per-run) —
    callers read it off the resolved object directly.
    """
    return defaults.model_copy(
        update={
            "cpu_limit": resolved.cpu_limit,
            "mem_limit": resolved.mem_limit,
            "pids_limit": resolved.pids_limit,
        }
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _looks_pre_parsed(value: Any) -> bool:
    """Heuristic: a dict whose keys are exactly the documented
    output of :func:`parse_resources_block` is treated as already
    validated. Any unknown key sends it through the parser."""
    if not isinstance(value, dict):
        return False
    allowed = {"cpu", "memory", "pids", "timeout"}
    return set(value.keys()).issubset(allowed)


def _parse_positive_float(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ResourceOverrideError(
            f"{field} must be numeric; got {value!r}"
        ) from exc
    if out <= 0:
        raise ResourceOverrideError(f"{field} must be > 0; got {out}")
    return out


def _parse_positive_int(value: Any, *, field: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ResourceOverrideError(
            f"{field} must be an integer; got {value!r}"
        ) from exc
    if out <= 0:
        raise ResourceOverrideError(f"{field} must be > 0; got {out}")
    return out


def _parse_memory(value: Any) -> str:
    """Same shape rules as ``SandboxConfig.mem_limit``: digits +
    one of k/m/g, case-insensitive."""
    if value is None:
        raise ResourceOverrideError("memory value is required when key present")
    s = str(value).strip().lower()
    if not _MEM_SUFFIX_RE.match(s):
        raise ResourceOverrideError(
            f"memory must look like '512m', '1g', or '4096k'; got {value!r}"
        )
    return s


def _merge_numeric(
    *,
    default: float | int,
    requested: float | int | str | None,
    field: str,
    allow_upscale: bool,
) -> tuple[float, bool]:
    """Merge one cpu/pids value. Returns ``(value, overridden)``."""
    if requested is None:
        return float(default), False
    asked = float(requested)
    if asked > float(default) and not allow_upscale:
        raise ResourceOverrideError(
            f"{field}={asked} exceeds operator ceiling {default}; enable "
            "ResourcePolicy(allow_manifest_upscale=True) to permit"
        )
    return asked, asked != float(default)


def _merge_memory(
    *,
    default: str,
    requested: str,
    allow_upscale: bool,
) -> str:
    """Merge memory limits. Manifest can request smaller without
    issue; larger requests are gated by ``allow_upscale``."""
    asked_bytes = _memory_to_bytes(requested)
    default_bytes = _memory_to_bytes(default)
    if asked_bytes > default_bytes and not allow_upscale:
        raise ResourceOverrideError(
            f"memory={requested} ({asked_bytes} bytes) exceeds operator "
            f"ceiling {default} ({default_bytes} bytes); enable "
            "ResourcePolicy(allow_manifest_upscale=True) to permit"
        )
    return requested.lower()


_MEM_UNITS = {"k": 1024, "m": 1024**2, "g": 1024**3}


def _memory_to_bytes(value: str) -> int:
    """Convert ``"1g"`` / ``"512m"`` / ``"4096k"`` → bytes for
    comparison. Caller has already validated the shape."""
    m = _MEM_SUFFIX_RE.match(value.strip().lower())
    if m is None:
        # parse_resources_block / SandboxConfig validator should
        # have caught this — defensive
        raise ResourceOverrideError(f"invalid memory string: {value!r}")
    digits, suffix = m.group(1), m.group(2).lower()
    return int(digits) * _MEM_UNITS[suffix]


# Kept exported for tests + future audit-log shapes.
__all__ = [
    "ResolvedResources",
    "ResourceOverrideError",
    "ResourcePolicy",
    "apply_to_sandbox_config",
    "parse_resources_block",
    "resolve_resources",
]


# `replace` is imported above for forward compat; future iterations may
# return tweaked copies of ResolvedResources.
_ = replace
