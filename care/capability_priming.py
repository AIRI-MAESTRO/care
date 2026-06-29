"""Capability priming for MAGE generation (TODO §4 P2).

Before MAGE runs, CARE wants to tell it about every capability the
user already has on disk so the planner can build a chain that uses
them — without asking MAGE to redo the discovery work CARE just
finished in :mod:`care.catalog`.

This module is the bridge:

* :class:`CapabilityPayload` — CARE-side frozen dataclass that
  represents the planner-context payload using plain dict/tuple
  primitives. CARE doesn't import ``mmar_mage`` at this layer
  (matches the convention established in :mod:`care.runtime.mage_poster`),
  so callers can build the payload, log it, ship it across
  processes, and serialise it without paying MAGE's import cost.
* :func:`build_capability_payload` — takes a
  :class:`care.catalog.CapabilityCatalog` (and any extra catalogs)
  and converts every entry into the shape MAGE's
  :class:`CapabilityContext` consumes.
* :meth:`CapabilityPayload.to_mage_context` — lazy-imports MAGE
  when the caller is actually about to invoke
  ``generator.generate(capabilities=...)`` and returns the real
  ``CapabilityContext`` instance.

The two-step split keeps unit tests cheap (we exercise the dict
shape directly) and keeps CARE's hard dependency surface free of
``mmar_mage``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from care.catalog import CapabilityCatalog, CapabilityCatalogEntry

_log = logging.getLogger("care.capability_priming")


@dataclass(frozen=True)
class CapabilityPayload:
    """Planner-context payload ready to hand to MAGE.

    Mirrors the field set of ``mmar_mage.CapabilityContext`` but
    uses plain Python primitives (``tuple[dict[str, Any], ...]``)
    so CARE doesn't need ``mmar_mage`` importable to construct or
    inspect it. Frozen so the same payload can be passed across
    code paths without defensive copies.

    Fields:
        tools: Tool entries — at minimum each carries ``name`` +
            ``source``; optional ``summary``, ``tags``, plus any
            free-form metadata the catalog captured.
        mcp_servers: MCP server entries with the original TOML
            section preserved under ``config`` for the planner.
        agent_skills: AgentSkill entries with the SKILL.md
            metadata + (optionally) the SHA-256 trust digest.
        environment_id: Tag forwarded to MAGE's legacy
            ``capability_registry`` lookup. Defaults to ``"default"``
            mirroring MAGE's own default.
    """

    tools: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    mcp_servers: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    agent_skills: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    environment_id: str = "default"

    @property
    def is_empty(self) -> bool:
        return not (self.tools or self.mcp_servers or self.agent_skills)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view — useful for logging and for callers
        who serialise the payload (e.g. into a run artifact)."""
        return {
            "tools": [dict(t) for t in self.tools],
            "mcp_servers": [dict(s) for s in self.mcp_servers],
            "agent_skills": [dict(s) for s in self.agent_skills],
            "environment_id": self.environment_id,
        }

    def to_mage_context(self) -> Any:
        """Lazily import MAGE and instantiate a ``CapabilityContext``.

        Raises:
            CapabilityPrimingError: when ``mmar_mage`` isn't
                importable — friendly message rather than a raw
                ``ImportError`` so the SettingsScreen can surface
                "MAGE not installed; install with `pip install
                mmar-mage`".

        Returns:
            A ready-to-pass ``mmar_mage.CapabilityContext`` instance.
        """
        try:
            from mmar_mage.agents.capability_lookup_agent import (
                AgentSkillEntry,
                CapabilityContext,
            )
        except ImportError as exc:
            raise CapabilityPrimingError(
                "mmar_mage is not installed; install it with "
                "`pip install mmar-mage` to use capability priming"
            ) from exc

        # Use `model_validate` for the dict payloads so MAGE can
        # apply its own coercion / extra-field handling.
        skills = [AgentSkillEntry.model_validate(s) for s in self.agent_skills]
        return CapabilityContext(
            tools=[dict(t) for t in self.tools],
            mcp_servers=[dict(s) for s in self.mcp_servers],
            agent_skills=skills,
            environment_id=self.environment_id,
        )


class CapabilityPrimingError(RuntimeError):
    """Raised when the priming workflow can't proceed — typically
    because ``mmar_mage`` isn't installed when the caller asked for
    a real :class:`CapabilityContext`."""


def build_capability_payload(
    catalog: CapabilityCatalog,
    *,
    environment_id: str = "default",
    compute_skill_sha: bool = True,
) -> CapabilityPayload:
    """Turn a :class:`CapabilityCatalog` into a MAGE-shaped payload.

    The catalog enumerates what CARE found; this function shapes
    those entries into the dicts the MAGE planner expects:

    * ``agent_skill`` entries → ``AgentSkillEntry``-shaped dicts.
      The SKILL.md is hashed on demand (``compute_skill_sha=True``)
      so the planner's pin matches what CARE's sandbox trust store
      uses. Pass ``False`` when scanning is happening on the hot
      path and the caller doesn't need pinning (e.g. a search-only
      pre-flight).
    * ``mcp_server`` entries → dicts with the original TOML body
      under ``config`` plus a top-level ``command`` / ``args``
      pair so MAGE's prompt rendering can use them directly.
    * ``tool`` entries → dicts with ``name`` + ``source`` + summary;
      tools loaded from disk are runtime-only callables, so we
      don't try to introspect signatures here (that happens in
      :mod:`care.tools` when the chain actually runs).
    * ``memory_card`` entries are **dropped** — they live in a
      different channel (MAGE reads them via Memory search, not
      via the planner-context input).

    Args:
        catalog: The discovery output from
            :func:`care.catalog.build_catalog`.
        environment_id: Forwarded to MAGE's legacy lookup. Most
            CARE users leave this at ``"default"``.
        compute_skill_sha: When ``True``, read each SKILL.md and
            stamp its SHA-256 on the entry. Skipped silently for
            paths that can't be read (file moved / permission
            denied) — the planner just gets an empty ``sha256``.

    Returns:
        A populated :class:`CapabilityPayload`.
    """
    tools = tuple(_tool_dict(e) for e in catalog.by_kind("tool"))
    mcp_servers = tuple(_mcp_dict(e) for e in catalog.by_kind("mcp_server"))
    agent_skills = tuple(
        _agent_skill_dict(e, compute_sha=compute_skill_sha)
        for e in catalog.by_kind("agent_skill")
    )
    return CapabilityPayload(
        tools=tools,
        mcp_servers=mcp_servers,
        agent_skills=agent_skills,
        environment_id=environment_id,
    )


# ---------------------------------------------------------------------------
# Per-kind converters
# ---------------------------------------------------------------------------


def _tool_dict(entry: CapabilityCatalogEntry) -> dict[str, Any]:
    """Catalog tool entry → MAGE ``tools`` list element."""
    return {
        "name": entry.name,
        "source": entry.source,
        "description": entry.summary,
        "tags": list(entry.tags),
    }


def _mcp_dict(entry: CapabilityCatalogEntry) -> dict[str, Any]:
    """Catalog MCP entry → MAGE ``mcp_servers`` list element."""
    meta = dict(entry.metadata or {})
    return {
        "name": entry.name,
        "description": entry.summary,
        "tags": list(entry.tags),
        "command": meta.get("command"),
        "args": meta.get("args") or [],
        "config": meta,
    }


def _agent_skill_dict(
    entry: CapabilityCatalogEntry,
    *,
    compute_sha: bool,
) -> dict[str, Any]:
    """Catalog skill entry → MAGE ``AgentSkillEntry``-shaped dict.

    The MAGE entry takes ``uri`` as the canonical pointer; CARE
    uses ``local://<abs-path>`` for on-disk skills (matches the
    convention in :mod:`care.skills`).
    """
    manifest = (entry.metadata or {}).get("manifest", {}) or {}
    allowed_tools = list(
        (entry.metadata or {}).get("allowed_tools") or []
    )
    skill_md_path = Path(entry.source)
    sha = _sha_skill_md(skill_md_path) if compute_sha else ""
    return {
        "name": entry.name,
        "description": entry.summary,
        "uri": f"local://{skill_md_path.resolve()}",
        "manifest_summary": str(manifest.get("description", entry.summary) or ""),
        "sha256": sha,
        "tags": list(entry.tags),
        "allowed_tools": allowed_tools,
        "source": "local",
        "relevance": 0.0,
        "why": "",
    }


def _sha_skill_md(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        # Skill discovered but file moved/locked since the scan —
        # leave SHA empty; the planner falls back to URI matching.
        return ""


def build_capabilities_for_generation(
    config: Any = None,
    *,
    query: str | None = None,
) -> Any | None:
    """Assemble the MAGE ``CapabilityContext`` to pass into generation.

    Without this, MAGE plans against whatever tool names it invents
    from its few-shot examples — which may or may not match the tools
    CARE actually registers at execution time. Advertising CARE's
    bundled standard tools (``web_search`` / ``fetch_url`` /
    ``calculator`` / ``current_datetime``), with their call signature in
    each ``description``, keeps the *generated* tool steps aligned with
    the *registered* callables.

    Args:
        config: A :class:`~care.config.CareConfig` (or anything with a
            ``tools`` attribute). ``None`` advertises the builtins
            unconditionally.
        query: The user's task text. Threaded through for relevance-based
            capability selection — e.g. searching Memory for saved tools
            that match the task (wired in B2). Optional; builtins +
            cached tools are advertised regardless of ``query``.

    Returns:
        A ready-to-pass ``mmar_mage.CapabilityContext``, or ``None``
        when there's nothing to advertise / MAGE isn't importable.
        Callers treat ``None`` as "generate without capabilities" — the
        prior behaviour — so this is always safe to call.
    """
    try:
        from care.builtin_tools import builtin_tool_specs
    except Exception:  # noqa: BLE001
        return None

    tools_cfg = getattr(config, "tools", None)
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _merge(new_specs: list[dict[str, Any]]) -> None:
        for spec in new_specs:
            name = spec.get("name")
            if name and name not in seen:
                seen.add(name)
                specs.append(spec)

    if tools_cfg is None or getattr(tools_cfg, "enable_builtins", True):
        _merge(builtin_tool_specs(tools_cfg))

    # Advertise previously-synthesised tools (disk cache) so the planner
    # reuses their exact names instead of re-inventing them.
    try:
        from care.tool_synthesis import cached_tool_specs

        _merge(cached_tool_specs(config))
    except Exception:  # noqa: BLE001
        pass

    # Recall tools saved to Memory relevant to this task, so a tool
    # synthesised in an earlier session/machine is reused by name.
    if query and getattr(tools_cfg, "recall_tools_from_memory", True):
        _merge(_recall_memory_tool_specs(config, query))

    if not specs:
        return None

    payload = CapabilityPayload(tools=tuple(specs))
    try:
        return payload.to_mage_context()
    except CapabilityPrimingError:
        # MAGE not installed — generation will run without priming.
        return None


def _recall_memory_tool_specs(config: Any, query: str) -> list[dict[str, Any]]:
    """Search Memory for saved synthesised tools relevant to ``query``.

    Returns MAGE tool specs for ``agent_skill`` entities tagged
    ``care:synthesized-tool`` (the runnable tools CARE persists) so the
    planner reuses them by name. Best-effort: any Memory error → ``[]``.
    Synchronous — Memory is local + fast, and a down service fails fast
    and is swallowed.
    """
    mem_cfg = getattr(config, "memory", None)
    if not getattr(mem_cfg, "base_url", None):
        return []
    try:
        from care.memory import CareMemory
        from care.tool_synthesis import SYNTH_TAG

        mem = CareMemory.from_config(config)
        hits = mem.search(query, entity_type="agent_skill", search_type="bm25", top_k=5)
    except Exception as exc:  # noqa: BLE001
        _log.info("memory tool recall failed: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for hit in hits or []:
        if not isinstance(hit, dict) or SYNTH_TAG not in (hit.get("tags") or []):
            continue  # only advertise our own runnable synthesised tools
        name = hit.get("name")
        if not name:
            continue
        content = hit.get("content") if isinstance(hit.get("content"), dict) else {}
        desc = content.get("description") or hit.get("snippet") or name
        out.append(
            {
                "name": name,
                "source": "memory",
                "description": f"{name}: {desc} (previously synthesised + saved — reuse it).",
                "tags": list(hit.get("tags") or []),
            }
        )
    return out


__all__ = [
    "CapabilityPayload",
    "CapabilityPrimingError",
    "build_capabilities_for_generation",
    "build_capability_payload",
]
