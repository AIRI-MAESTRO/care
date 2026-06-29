"""Network policy translation for AgentSkill sandboxing (TODO §6.2 P0).

CARE's user-facing names (``"none" | "skill_declared" | "open"``)
intentionally differ from CARL's internal runtime names
(``"none" | "allowlist" | "host"``) — CARE optimises for the user
("only the domains the skill itself declared it needs") while CARL
talks in implementation terms. This module owns the two-way
translation plus the `WebFetch(domain:*)` extraction the
``skill_declared`` mode needs.

The parse is duplicated from
:mod:`mmar_carl.skill_runtime.parse_network_allowlist_from_allowed_tools`
on purpose: CARE's startup must not import ``mmar_carl`` at module
load (a broken CARL install can't break CARE startup). Behaviour is
pinned in tests so it can't drift silently.

Public surface::

    parse_webfetch_domains(allowed_tools)         -> list[str]
    translate_to_carl_policy(policy)              -> str
    resolve_network_policy(policy, ...)           -> ResolvedNetworkPolicy
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from care.sandbox.backend import NetworkPolicy

_WEBFETCH_DOMAIN_RE = re.compile(
    r"WebFetch\s*\(\s*domain\s*:\s*([^\s)]+)\s*\)",
    re.IGNORECASE,
)
"""Match ``WebFetch(domain:host)`` tokens — same regex as CARL's
upstream parser, kept local so CARE's startup doesn't need to
import ``mmar_carl``."""

CARE_TO_CARL_POLICY: dict[NetworkPolicy, str] = {
    "none": "none",
    "skill_declared": "allowlist",
    "open": "host",
}
"""How CARE's user-facing policy names map to CARL's internal
runtime-policy names. Used when CARE delegates execution to a CARL
backend that expects the upstream literal."""


def parse_webfetch_domains(
    allowed_tools: list[str] | str | None,
) -> list[str]:
    """Extract `WebFetch(domain:host)` hosts from a skill's
    ``allowed-tools``.

    Bare ``WebFetch`` (no ``domain:`` constraint) is intentionally
    skipped — an unconstrained network policy is the chain author's
    choice (``policy="open"``), not something a manifest token
    silently widens.

    Args:
        allowed_tools: The raw SKILL.md frontmatter string, a
            pre-tokenised list, or ``None`` for skills with no
            restrictions.

    Returns:
        Sorted, de-duplicated list of host strings. Empty when no
        ``WebFetch(domain:*)`` tokens are present (also when input
        is ``None`` / empty).
    """
    if allowed_tools is None:
        return []
    text = (
        allowed_tools
        if isinstance(allowed_tools, str)
        else " ".join(str(t) for t in allowed_tools)
    )
    matches = _WEBFETCH_DOMAIN_RE.findall(text)
    return sorted({m.strip() for m in matches if m.strip()})


def translate_to_carl_policy(policy: NetworkPolicy) -> str:
    """Translate CARE's policy name to CARL's runtime policy name.

    Used when a CARE sandbox backend wraps a CARL ``SkillRuntime``
    and needs to forward the policy. Raises ``ValueError`` on an
    unknown literal — defends against typos in user-supplied
    config that slipped past the Pydantic Literal at the edge.
    """
    try:
        return CARE_TO_CARL_POLICY[policy]
    except KeyError as exc:
        raise ValueError(
            f"unknown CARE network policy {policy!r}; "
            f"expected one of {sorted(CARE_TO_CARL_POLICY)}"
        ) from exc


@dataclass(frozen=True)
class ResolvedNetworkPolicy:
    """Output of :func:`resolve_network_policy`.

    Frozen so backends can pass it around without worrying about
    mutation. Carries both names (CARE + CARL) so callers don't
    need to re-translate when forwarding to upstream runtimes.

    Fields:
        policy: CARE-side name (echo of the input).
        carl_policy: Equivalent CARL-side name.
        domains: Sorted, de-duplicated allowlist. Empty when
            ``policy != "skill_declared"`` — the field exists so
            backends can rely on it without None-checks.
        enforced: Whether the chosen sandbox backend can actually
            enforce the policy. ``LocalSandboxBackend`` always
            returns ``False`` here; ``DockerSandboxBackend`` returns
            ``True`` for ``none`` + ``skill_declared`` and ``False``
            for ``open``. Defaulted to ``True`` because the resolver
            itself is backend-agnostic; backends override via the
            ``enforced_override`` arg.
    """

    policy: NetworkPolicy
    carl_policy: str
    domains: tuple[str, ...]
    enforced: bool = True


def resolve_network_policy(
    policy: NetworkPolicy,
    *,
    allowed_tools: list[str] | str | None = None,
    override_domains: list[str] | tuple[str, ...] | None = None,
    enforced_override: bool | None = None,
) -> ResolvedNetworkPolicy:
    """Compute the concrete allowlist + CARL-name for ``policy``.

    Args:
        policy: CARE-side policy literal.
        allowed_tools: SKILL.md ``allowed-tools`` tokens. Only
            consulted when ``policy == "skill_declared"``.
        override_domains: Operator-supplied extra hosts merged into
            the allowlist (also only for ``"skill_declared"``).
            Use this when the user has trusted a skill but wants to
            grant it one extra domain not in the manifest.
        enforced_override: Backend-specific knob. ``None`` keeps
            the default ``True``; pass ``False`` from a backend
            that can't enforce isolation (e.g. ``LocalSandbox``).

    Returns:
        :class:`ResolvedNetworkPolicy`. For ``"none"`` /
        ``"open"`` the ``domains`` tuple is empty.

    Raises:
        ValueError: If ``policy`` isn't a known CARE policy name.
    """
    carl = translate_to_carl_policy(policy)
    if policy == "skill_declared":
        manifest_hosts = parse_webfetch_domains(allowed_tools)
        extra_hosts = [
            str(h).strip()
            for h in (override_domains or [])
            if str(h).strip()
        ]
        domains = tuple(sorted({*manifest_hosts, *extra_hosts}))
    else:
        domains = ()
    enforced = True if enforced_override is None else bool(enforced_override)
    return ResolvedNetworkPolicy(
        policy=policy,
        carl_policy=carl,
        domains=domains,
        enforced=enforced,
    )


__all__ = [
    "CARE_TO_CARL_POLICY",
    "ResolvedNetworkPolicy",
    "parse_webfetch_domains",
    "resolve_network_policy",
    "translate_to_carl_policy",
]
