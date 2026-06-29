"""One-shot local Platform stack bootstrap for MAESTRO startup.

Runs automatically when :class:`CarePlatform` is constructed so users
do not need the manual checklist (``sync_platform_llm_models.py``,
``sync_runner_gigaevo_tools.sh``, …) before launching evolution.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from care.runtime.platform_llm_sync import (
    PlatformLlmSyncResult,
    try_sync_platform_llm_registry,
)
from care.runtime.evolution_chain_templates import verify_chain_template_source
from care.runtime.runner_tools_sync import (
    RunnerToolsSyncResult,
    sync_runner_gigaevo_tools,
)

_log = logging.getLogger("care.platform_bootstrap")

_BOOTSTRAP_DONE = False


@dataclass(frozen=True)
class PlatformBootstrapReport:
    skipped: bool = False
    skip_reason: str = ""
    llm: PlatformLlmSyncResult | None = None
    tools: RunnerToolsSyncResult | None = None
    messages: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        if self.skipped:
            return True
        return bool(self.messages)


def _is_local_platform_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", ""}


def bootstrap_local_platform_stack(cfg: Any) -> PlatformBootstrapReport:
    """Sync LLM registry + runner metric tools (best-effort, idempotent)."""
    if os.environ.get("CARE_PLATFORM__AUTO_BOOTSTRAP", "1") == "0":
        return PlatformBootstrapReport(
            skipped=True,
            skip_reason="CARE_PLATFORM__AUTO_BOOTSTRAP=0",
        )

    plat = getattr(cfg, "platform", None)
    base_url = str(getattr(plat, "base_url", "") or "").strip()
    if not base_url:
        return PlatformBootstrapReport(
            skipped=True,
            skip_reason="platform.base_url empty",
        )

    if not _is_local_platform_url(base_url):
        return PlatformBootstrapReport(
            skipped=True,
            skip_reason=f"remote platform ({base_url})",
        )

    messages: list[str] = []

    llm = try_sync_platform_llm_registry(cfg)
    if llm is not None:
        messages.append(llm.message)
        _log.info("platform bootstrap llm: %s", llm.message)
    else:
        messages.append("llm_models.yml sync skipped (validation error)")
        _log.warning("platform bootstrap: llm_models.yml sync failed")

    tools = sync_runner_gigaevo_tools()
    if tools is not None:
        messages.append(tools.message)
        _log.info("platform bootstrap tools: %s", tools.message)

    tpl_ok, tpl_msg = verify_chain_template_source()
    messages.append(tpl_msg)
    if tpl_ok:
        _log.info("platform bootstrap templates: %s", tpl_msg)
    else:
        _log.warning("platform bootstrap templates: %s", tpl_msg)

    return PlatformBootstrapReport(
        llm=llm,
        tools=tools,
        messages=tuple(messages),
    )


def bootstrap_local_platform_once(cfg: Any) -> PlatformBootstrapReport | None:
    """Run :func:`bootstrap_local_platform_stack` at most once per process."""
    global _BOOTSTRAP_DONE
    if _BOOTSTRAP_DONE:
        return None
    _BOOTSTRAP_DONE = True
    report = bootstrap_local_platform_stack(cfg)
    if not report.skipped:
        _log.info(
            "platform bootstrap complete: %s",
            "; ".join(report.messages) or "nothing to do",
        )
    return report


__all__ = [
    "PlatformBootstrapReport",
    "bootstrap_local_platform_once",
    "bootstrap_local_platform_stack",
]
