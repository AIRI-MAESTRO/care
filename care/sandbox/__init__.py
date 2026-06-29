"""CARE-side AgentSkill sandboxing.

This package defines the backend abstraction CARE uses when an
`agent_skill` step in a CARL chain needs to run untrusted code. CARL
itself ships per-runtime backends (`LocalSkillRuntime`,
`DockerSkillRuntime`, etc.), but CARE adds a thin host-side layer
to:

- enforce CARE's own resource defaults (read from
  `CareConfig.sandbox`),
- write an audit log on every ``run()`` call (TODO §6.2 P1),
- prompt the user before executing a skill whose SHA hasn't been
  trusted yet (TODO §6.3 P0).

The public surface lives in :mod:`care.sandbox.backend` (the
``SandboxBackend`` Protocol + value types) and per-backend modules
under this package.
"""

from __future__ import annotations

from care.sandbox.audit import (
    AUDIT_FORMAT_VERSION,
    DEFAULT_AUDIT_PATH,
    SandboxAuditEntry,
    SandboxAuditError,
    SandboxAuditLogger,
)
from care.sandbox.backend import (
    NetworkPolicy,
    ResolvedSkillLike,
    RunResult,
    SandboxBackend,
    SandboxError,
    SandboxHandle,
    SandboxTimeoutError,
)
from care.sandbox.docker import (
    CARE_LABEL,
    DEFAULT_IMAGE,
    SKILL_MOUNT,
    WORKSPACE_MOUNT,
    DockerSandboxBackend,
)
from care.sandbox.e2b import (
    DEFAULT_TEMPLATE,
    SKILL_DIR,
    WORKSPACE_DIR,
    E2BSandboxBackend,
)
from care.sandbox.firejail import (
    DEFAULT_EXECUTABLE,
    FirejailSandboxBackend,
)
from care.sandbox.local import LocalSandboxBackend
from care.sandbox.network_policy import (
    CARE_TO_CARL_POLICY,
    ResolvedNetworkPolicy,
    parse_webfetch_domains,
    resolve_network_policy,
    translate_to_carl_policy,
)
from care.sandbox.output_mediation import (
    DEFAULT_MAX_FILE_BYTES,
    FindingKind,
    MediationReport,
    OutputFinding,
    Severity,
    scan_output_dir,
)
from care.sandbox.resources import (
    ResolvedResources,
    ResourceOverrideError,
    ResourcePolicy,
    apply_to_sandbox_config,
    parse_resources_block,
    resolve_resources,
)
from care.sandbox.trust import (
    DEFAULT_TRUST_PATH,
    STORE_FORMAT_VERSION,
    SkillTrustStore,
    SkillTrustStoreError,
    TrustPolicy,
    TrustRecord,
)

__all__ = [
    "AUDIT_FORMAT_VERSION",
    "CARE_LABEL",
    "CARE_TO_CARL_POLICY",
    "DEFAULT_AUDIT_PATH",
    "DEFAULT_EXECUTABLE",
    "DEFAULT_IMAGE",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_TEMPLATE",
    "DEFAULT_TRUST_PATH",
    "DockerSandboxBackend",
    "E2BSandboxBackend",
    "FirejailSandboxBackend",
    "STORE_FORMAT_VERSION",
    "SKILL_DIR",
    "SKILL_MOUNT",
    "WORKSPACE_DIR",
    "WORKSPACE_MOUNT",
    "FindingKind",
    "LocalSandboxBackend",
    "MediationReport",
    "NetworkPolicy",
    "OutputFinding",
    "ResolvedNetworkPolicy",
    "ResolvedResources",
    "ResolvedSkillLike",
    "ResourceOverrideError",
    "ResourcePolicy",
    "RunResult",
    "SandboxAuditEntry",
    "SandboxAuditError",
    "SandboxAuditLogger",
    "SandboxBackend",
    "SandboxError",
    "SandboxHandle",
    "SandboxTimeoutError",
    "Severity",
    "SkillTrustStore",
    "SkillTrustStoreError",
    "TrustPolicy",
    "TrustRecord",
    "apply_to_sandbox_config",
    "parse_resources_block",
    "parse_webfetch_domains",
    "resolve_network_policy",
    "resolve_resources",
    "scan_output_dir",
    "translate_to_carl_policy",
]
