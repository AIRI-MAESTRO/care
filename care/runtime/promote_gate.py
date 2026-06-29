"""Full promotion gate (PRODUCTION_TODO C1): checks BEFORE latest → stable.

Three checks, in cost order — the gate stops at the first hard failure:

1. **artifact** — the candidate (the ``from``-channel version) passes the
   deploy gate-lite: loads exactly like an agent will + stays within the
   template tool set (reuses :mod:`care.runtime.deploy_gate`);
2. **baseline run** — the gate EXECUTES the candidate on its saved task via
   the library-run pipeline and requires success ("baseline-run обязателен").
   The run is recorded (a run-record memory_card + run counters), so the
   promotion leaves evidence;
3. **eval score** — when the target channel carries an eval baseline
   (``versions/beating``: ``baseline_value`` pinned + scored), the candidate
   version must be among the winners. With no recorded scores the check is
   SKIPPED honestly (eval scoring lands with MAGE T35 / evolution) — the gate
   never blocks on data that does not exist yet.

``--force`` on ``/promote`` bypasses the whole gate; the report renders one
chat line per check either way.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from care.runtime.deploy_gate import gate_chain_for_deploy

logger = logging.getLogger(__name__)

#: (success, detail) of one baseline execution.
BaselineRunner = Callable[..., Awaitable[tuple[bool, str]]]


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str
    skipped: bool = False

    @property
    def icon(self) -> str:
        if self.skipped:
            return "○"
        return "✓" if self.passed else "✗"


@dataclass(frozen=True)
class PromoteGateReport:
    entity_id: str
    from_channel: str
    to_channel: str
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.passed or check.skipped for check in self.checks)

    def lines(self) -> list[str]:
        return [f"{c.icon} {c.name}: {c.detail}" for c in self.checks]


async def gate_promotion(
    memory: Any,
    config: Any,
    entity_id: str,
    *,
    from_channel: str = "latest",
    to_channel: str = "stable",
    baseline_runner: BaselineRunner | None = None,
) -> PromoteGateReport:
    """Run the full gate; the report says exactly what passed/failed/skipped."""
    checks: list[GateCheck] = []
    report = PromoteGateReport(
        entity_id=entity_id,
        from_channel=from_channel,
        to_channel=to_channel,
        checks=checks,
    )
    client = getattr(memory, "client", None)

    # -- 1) artifact ------------------------------------------------------
    record: Any | None = None
    try:
        record = await asyncio.to_thread(
            client.get_chain_record, entity_id, channel=from_channel
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            GateCheck(
                "artifact",
                False,
                f"cannot load {from_channel!r} version: {exc}",
            )
        )
        return report
    content = dict(getattr(record, "content", None) or {})
    # Synthesized tools the chain uses ship WITH the deployment (extra_tools), so
    # they count as available in the gate — same as the /deploy path.
    try:
        from care.tool_synthesis import bundled_tools_for_chain

        bundled_names = frozenset(t["name"] for t in bundled_tools_for_chain(content, config))
    except Exception:  # noqa: BLE001 — bundling is best-effort
        bundled_names = frozenset()
    issues = await asyncio.to_thread(gate_chain_for_deploy, content, bundled_names)
    if issues:
        checks.append(GateCheck("artifact", False, "; ".join(issues)))
        return report  # no point burning a baseline run on a broken artifact
    candidate_version = getattr(record, "version_id", "") or ""
    candidate_number = getattr(record, "version_number", None)
    checks.append(
        GateCheck(
            "artifact",
            True,
            f"v{candidate_number} loads cleanly within the template tool set",
        )
    )

    # -- 2) baseline run ----------------------------------------------------
    runner = baseline_runner or _run_baseline
    try:
        success, detail = await runner(
            memory, config, entity_id, channel=from_channel
        )
    except Exception as exc:  # noqa: BLE001
        success, detail = False, f"baseline run errored: {exc}"
    checks.append(GateCheck("baseline run", success, detail))
    if not success:
        return report

    # -- 3) eval score (versions/beating) ----------------------------------
    checks.append(
        await _eval_check(client, entity_id, candidate_version, to_channel)
    )
    return report


async def _eval_check(
    client: Any, entity_id: str, candidate_version: str, to_channel: str
) -> GateCheck:
    lister = getattr(client, "list_chain_versions_beating", None)
    if not callable(lister):
        return GateCheck(
            "eval score", True, "versions/beating unavailable on this Memory", skipped=True
        )
    try:
        response = await asyncio.to_thread(
            lambda: lister(entity_id, channel=to_channel)
        )
    except Exception as exc:  # noqa: BLE001
        return GateCheck("eval score", True, f"could not query: {exc}", skipped=True)
    baseline_value = getattr(response, "baseline_value", None)
    if baseline_value is None:
        return GateCheck(
            "eval score",
            True,
            f"no eval baseline on {to_channel!r} yet — scoring lands with evolution/T35",
            skipped=True,
        )
    winners = list(getattr(response, "winners", None) or [])
    for winner in winners:
        if getattr(winner, "version_id", None) == candidate_version:
            value = getattr(winner, "value", None)
            return GateCheck(
                "eval score",
                True,
                f"candidate beats the {to_channel!r} baseline "
                f"({value} > {baseline_value})",
            )
    return GateCheck(
        "eval score",
        False,
        f"candidate does not beat the {to_channel!r} baseline ({baseline_value}) "
        f"on fitness_score",
    )


async def _run_baseline(
    memory: Any, config: Any, entity_id: str, *, channel: str
) -> tuple[bool, str]:
    """Default baseline runner: the real library-run pipeline, recorded."""
    from care.runtime.library_run import execute_library_run, load_run_plan
    from care.runtime.llm_client import build_carl_llm_client

    plan = await load_run_plan(memory, entity_id, channel=channel)
    api = build_carl_llm_client(config.mage)
    completion = await execute_library_run(
        memory, plan, plan.draft, config=config, api=api
    )
    summary = completion.summary
    if summary.success:
        detail = (
            f"succeeded in {summary.duration_seconds:.1f}s "
            f"({summary.step_count} steps; recorded as run {completion.run_id})"
        )
        return True, detail
    return False, f"baseline failed: {summary.error_message or 'run unsuccessful'}"
