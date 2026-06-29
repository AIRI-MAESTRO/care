"""MAGE intermediate-artifact projection (TODO §4 P1).

`MAGEResult.intermediate_artifacts` is MAGE's per-stage output
ledger — six known keys today (``domain_analysis``,
``step_plan``, ``dag``, ``critique``, ``verification``,
``refine``), each holding the agent's ``model_dump()`` so the
artifact round-trips through JSON.

CARE's future InspectionScreen wants to surface these as
**collapsible panes**: a stage header + a one-line summary +
the full body when the pane is expanded. Different stages need
different summary shapes (domain_analysis is a paragraph;
step_plan is a step count; dag is a node/edge count; …) — this
module owns the per-stage projection rules so the UI just
renders.

The projector is duck-typed against `mmar_mage`: takes a
`MAGEResult`-like (with `.intermediate_artifacts`), the
artifacts dict directly, or any plain dict mapping stage name →
payload. No upstream import; the renderer doesn't care.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Order MAGE produces these in. The view preserves it so the
# UI's "DomainAnalysis → StepPlan → DAG → …" pane stack stays
# in pipeline order even when the source dict happens to be
# in a different order.
_STAGE_ORDER: tuple[str, ...] = (
    "domain_analysis",
    "step_plan",
    "dag",
    "critique",
    "verification",
    "refine",
)


_STAGE_HEADERS: dict[str, str] = {
    "domain_analysis": "Domain analysis",
    "step_plan": "Step plan",
    "dag": "DAG structure",
    "critique": "Step critique",
    "verification": "Chain verification",
    "refine": "Refinement",
}


@dataclass(frozen=True)
class IntermediateArtifact:
    """One MAGE stage's intermediate output.

    Frozen so the future UI can stash the artifact on a
    collapsible-pane widget without defensive copies.

    Fields:
        stage: MAGE stage key (``"domain_analysis"`` / ``"step_plan"`` /
            ``"dag"`` / ``"critique"`` / ``"verification"`` /
            ``"refine"``). Unknown keys are preserved — future
            MAGE versions may add stages and CARE shouldn't
            silently drop them.
        header: Human-readable stage name for the pane header
            ("Step plan", "DAG structure", …). Falls back to a
            title-cased version of the stage key for unknown
            stages.
        summary: One-line scan-friendly summary used as the
            collapsed-pane label. Per-stage rules in
            :func:`_summarise_stage`.
        body: Multi-line expanded-pane content. Renders the
            structured artifact fields as `key: value` lines
            so the UI doesn't have to parse JSON.
        raw: The original artifact payload (typically the
            agent's `model_dump()` dict). Kept verbatim so
            screens that want a JSON viewer can use it.
    """

    stage: str
    header: str
    summary: str
    body: str
    raw: Any


@dataclass(frozen=True)
class IntermediateArtifactsView:
    """Ordered collection of :class:`IntermediateArtifact` panes.

    Frozen so the projection can flow through messages /
    persisted run records as-is. Lookup helpers
    (:meth:`by_stage`, :attr:`is_empty`) cover the cases the UI
    cares about; :meth:`format_text` renders the full block for
    CLI / footer use.
    """

    artifacts: tuple[IntermediateArtifact, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return len(self.artifacts) == 0

    def stages(self) -> tuple[str, ...]:
        """Stage keys in render order."""
        return tuple(a.stage for a in self.artifacts)

    def by_stage(self, stage: str) -> IntermediateArtifact | None:
        """Look up a single artifact by stage key. Returns
        ``None`` when the stage didn't run (fast mode skips
        most enhancements)."""
        for art in self.artifacts:
            if art.stage == stage:
                return art
        return None

    def format_text(self) -> str:
        """Multi-pane render for CLI / log lines. Each pane is
        a "# <header>" heading + the summary + the body."""
        if self.is_empty:
            return "no intermediate artifacts"
        chunks: list[str] = []
        for art in self.artifacts:
            chunk = [f"# {art.header}", art.summary]
            if art.body:
                chunk.append(art.body)
            chunks.append("\n".join(chunk))
        return "\n\n".join(chunks)


def project_intermediate_artifacts(source: Any) -> IntermediateArtifactsView:
    """Build an :class:`IntermediateArtifactsView` from a MAGE
    result-or-dict.

    Args:
        source: One of three accepted shapes:

            - A :class:`mmar_mage.MAGEResult`-like (or anything
              with ``.intermediate_artifacts``). The dict is
              pulled off and projected.
            - The intermediate-artifacts dict directly
              (``{"domain_analysis": {...}, ...}``).
            - ``None`` or an empty dict — returns an empty
              view.

    Returns:
        Populated :class:`IntermediateArtifactsView`. Stages
        from the source appear in MAGE's pipeline order
        (`_STAGE_ORDER`) followed by any unknown stages in
        insertion order. Empty / missing stages are dropped.
    """
    artifacts_dict = _extract_artifacts_dict(source)
    if not artifacts_dict:
        return IntermediateArtifactsView()

    ordered_keys: list[str] = []
    seen: set[str] = set()
    for key in _STAGE_ORDER:
        if key in artifacts_dict:
            ordered_keys.append(key)
            seen.add(key)
    for key in artifacts_dict:
        if key not in seen:
            ordered_keys.append(key)

    artifacts: list[IntermediateArtifact] = []
    for key in ordered_keys:
        payload = artifacts_dict.get(key)
        if payload is None or payload == {} or payload == []:
            continue
        artifacts.append(
            IntermediateArtifact(
                stage=key,
                header=_STAGE_HEADERS.get(key, key.replace("_", " ").title()),
                summary=_summarise_stage(key, payload),
                body=_render_body(key, payload),
                raw=payload,
            )
        )
    return IntermediateArtifactsView(artifacts=tuple(artifacts))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_artifacts_dict(source: Any) -> dict[str, Any]:
    """Coerce ``source`` into the artifacts dict. Returns ``{}``
    for any input we can't recognise — better than raising for
    callers passing optional fields."""
    if source is None:
        return {}
    if isinstance(source, dict):
        if "intermediate_artifacts" in source and isinstance(
            source["intermediate_artifacts"], dict
        ):
            return dict(source["intermediate_artifacts"])
        return dict(source)
    nested = getattr(source, "intermediate_artifacts", None)
    if isinstance(nested, dict):
        return dict(nested)
    return {}


def _summarise_stage(stage: str, payload: Any) -> str:
    """One-line collapsed-pane summary per known stage. Unknown
    stages fall back to a length-based generic summary."""
    if not isinstance(payload, dict):
        return _generic_summary(payload)
    if stage == "domain_analysis":
        return _summarise_domain_analysis(payload)
    if stage == "step_plan":
        return _summarise_step_plan(payload)
    if stage == "dag":
        return _summarise_dag(payload)
    if stage == "critique":
        return _summarise_critique(payload)
    if stage == "verification":
        return _summarise_verification(payload)
    if stage == "refine":
        return _summarise_refine(payload)
    return _generic_summary(payload)


def _summarise_domain_analysis(payload: dict[str, Any]) -> str:
    domain = payload.get("domain") or "unknown"
    task_type = payload.get("task_type")
    complexity = payload.get("complexity")
    suggested_steps = payload.get("suggested_step_count")
    bits: list[str] = [f"domain={domain}"]
    if task_type:
        bits.append(f"type={task_type}")
    if complexity:
        bits.append(f"complexity={complexity}")
    if suggested_steps:
        bits.append(f"suggested_steps={suggested_steps}")
    return ", ".join(bits)


def _summarise_step_plan(payload: dict[str, Any]) -> str:
    steps = payload.get("steps") or payload.get("planned_steps") or []
    if isinstance(steps, list):
        return f"{len(steps)} step{'s' if len(steps) != 1 else ''} planned"
    return _generic_summary(payload)


def _summarise_dag(payload: dict[str, Any]) -> str:
    nodes = payload.get("nodes") or payload.get("steps") or []
    edges = payload.get("edges") or payload.get("dependencies") or []
    n_nodes = len(nodes) if isinstance(nodes, list) else 0
    n_edges = len(edges) if isinstance(edges, list) else 0
    return f"{n_nodes} node{'s' if n_nodes != 1 else ''}, {n_edges} edge{'s' if n_edges != 1 else ''}"


def _summarise_critique(payload: dict[str, Any]) -> str:
    score = payload.get("overall_score") or payload.get("score")
    failing = payload.get("failing_step_numbers") or payload.get("failing") or []
    bits: list[str] = []
    if isinstance(score, (int, float)):
        bits.append(f"score={float(score):.2f}")
    if isinstance(failing, list):
        bits.append(
            f"{len(failing)} failing step{'s' if len(failing) != 1 else ''}"
        )
    return ", ".join(bits) or _generic_summary(payload)


def _summarise_verification(payload: dict[str, Any]) -> str:
    passed = payload.get("passed")
    issues = payload.get("issues") or []
    if isinstance(passed, bool):
        return (
            "passed"
            if passed
            else f"failed ({len(issues) if isinstance(issues, list) else 0} issue"
            + ("s" if not isinstance(issues, list) or len(issues) != 1 else "")
            + ")"
        )
    return _generic_summary(payload)


def _summarise_refine(payload: dict[str, Any]) -> str:
    iters = payload.get("iterations") or payload.get("refine_iterations")
    delta = payload.get("quality_delta") or payload.get("refine_quality_delta")
    bits: list[str] = []
    if isinstance(iters, int):
        bits.append(f"iterations={iters}")
    if isinstance(delta, (int, float)):
        bits.append(f"Δ={float(delta):+.2f}")
    return ", ".join(bits) or _generic_summary(payload)


def _generic_summary(payload: Any) -> str:
    if isinstance(payload, dict):
        return f"{len(payload)} field{'s' if len(payload) != 1 else ''}"
    if isinstance(payload, list):
        return f"{len(payload)} item{'s' if len(payload) != 1 else ''}"
    return type(payload).__name__


def _render_body(stage: str, payload: Any) -> str:  # noqa: ARG001 — stage reserved for future per-stage render rules
    """Expanded-pane body. Renders structured payloads as
    `key: value` lines so the UI doesn't need a JSON viewer to
    look at them. Lists of dicts get bullet-prefixed; deeply
    nested payloads truncate after a single level (the user
    can drop to a raw-JSON view for the rest).
    """
    if isinstance(payload, dict):
        return _render_dict(payload)
    if isinstance(payload, list):
        return _render_list(payload)
    return str(payload)


def _render_dict(payload: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = "  " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_render_dict(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}: ({len(value)} items)")
            for i, item in enumerate(_truncate(value, max_items=5)):
                lines.append(f"{prefix}  - {_render_inline(item)}")
                _ = i
            if len(value) > 5:
                lines.append(f"{prefix}  … {len(value) - 5} more")
        else:
            lines.append(f"{prefix}{key}: {_render_inline(value)}")
    return "\n".join(lines)


def _render_list(payload: list[Any]) -> str:
    if not payload:
        return "(empty)"
    rendered: list[str] = []
    for item in _truncate(payload, max_items=10):
        rendered.append(f"- {_render_inline(item)}")
    if len(payload) > 10:
        rendered.append(f"… {len(payload) - 10} more")
    return "\n".join(rendered)


def _render_inline(value: Any) -> str:
    """Render a scalar / small-dict inline. Long strings get
    truncated so the pane stays scannable."""
    if isinstance(value, str):
        if len(value) > 120:
            return value[:117] + "..."
        return value
    if isinstance(value, dict):
        head = ", ".join(
            f"{k}={_render_inline(v)}"
            for k, v in list(value.items())[:3]
        )
        suffix = "" if len(value) <= 3 else f", … +{len(value) - 3}"
        return "{" + head + suffix + "}"
    return repr(value)


def _truncate(items: Iterable[Any], *, max_items: int) -> list[Any]:
    out: list[Any] = []
    for i, item in enumerate(items):
        if i >= max_items:
            break
        out.append(item)
    return out


__all__ = [
    "IntermediateArtifact",
    "IntermediateArtifactsView",
    "project_intermediate_artifacts",
]
