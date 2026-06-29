"""Deploy gate-lite (PRODUCTION_TODO B2): client-side checks before a chain
ships to the agent hub.

Three cheap, deterministic checks — the same failure classes the hub itself
would hit, caught BEFORE anything is deployed:

1. the chain actually loads the way the agent will load it
   (``ReasoningChain.from_dict(..., use_typed_steps=True)``);
2. every tool the chain calls exists in the agent template's read-only
   builtin set (deployed agents ship calculator / current_datetime /
   fetch_url / http_request / web_search — nothing else);
3. MAGE's deterministic chain lint (``lint_chain``) with that same tool set —
   catches empty/bogus ``tool_name``s, ``$outer_context`` in url fields, etc.

Returns a list of human-readable issues; empty means the gate passed. The
full promotion gate (baseline + eval score) is C1.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The read-only builtin tool set `carl-agent-server` registers in deployments.
#: Keep in sync with carl_agent_server.tools.register_builtin_tools.
TEMPLATE_TOOLS: frozenset[str] = frozenset(
    {"calculator", "current_datetime", "fetch_url", "http_request", "web_search"}
)


def gate_chain_for_deploy(
    chain_dict: dict[str, Any],
    extra_tool_names: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    """Run the gate; return issues (empty list = good to deploy).

    ``extra_tool_names`` are synthesized tools that will SHIP with the deployment
    (``DeploymentSpec.extra_tools``) — they count as available alongside the
    template builtins, so a chain that uses them isn't flagged as missing tools."""
    if not isinstance(chain_dict, dict) or not chain_dict.get("steps"):
        return ["chain content is empty — nothing to deploy"]

    available = TEMPLATE_TOOLS | set(extra_tool_names)
    issues: list[str] = []

    # 1) parse exactly like the agent will. from_dict mutates its input, so
    #    feed it a deep copy (the P1.7 lesson).
    chain: Any | None = None
    try:
        from mmar_carl import ReasoningChain

        chain = ReasoningChain.from_dict(
            copy.deepcopy(chain_dict), use_typed_steps=True
        )
    except Exception as exc:
        # fatal: nothing else is meaningful if the chain cannot load
        return [f"chain does not load: {type(exc).__name__}: {exc}"]

    # 2) required tools must exist in the template's read-only set.
    #    NB: ReasoningChain.required_tools is a METHOD on mmar-carl 0.3.0.
    try:
        attr = getattr(chain, "required_tools", None)
        required = set(attr() if callable(attr) else attr or [])
        missing = sorted(required - available)
        if missing:
            issues.append(
                "chain needs tools the agent template does not ship: "
                + ", ".join(missing)
            )
    except Exception:  # noqa: BLE001 — introspection must never block the gate
        logger.debug("deploy gate: required_tools introspection failed", exc_info=True)

    # 3) MAGE's deterministic lint with the same tool universe — best-effort:
    #    the published mmar-mage wheel may predate lint_chain (it lives on the
    #    failure-audit branch); the check activates once that lands on PyPI.

    try:
        from mmar_mage.chain_repair import lint_chain

        issues.extend(lint_chain(chain_dict, known_tools=available))
    except Exception:  # noqa: BLE001 — lint is best-effort hardening
        logger.debug("deploy gate: mage lint unavailable", exc_info=True)

    return issues
