"""MAGE-metadata summary projection (TODO §4 P0).

After a MAGE generation finishes, CARE's TUI wants to surface
**every interesting field** from :class:`MAGEMetadata` to the
user — not just the chain itself. The TODO spec calls out
eleven specific fields: domain, num_steps, stages completed,
memory hits used, web results, cold-start flag, critique score,
refine iterations, ToT branches, MCTS sims, feedback recall
count.

This module is the data-projection layer behind that surface.
It takes a duck-typed input (a :class:`mmar_mage.MAGEResult`, a
:class:`MAGEMetadata` directly, or a plain dict) and returns a
CARE-stable :class:`MetadataSummary` frozen dataclass that the
TUI / CLI render. CARE doesn't import ``mmar_mage`` here —
`getattr` does the heavy lifting so the projector tolerates
older MAGE installs (missing optional fields are filled with
sensible defaults).

The :meth:`MetadataSummary.format_text` method produces a
multi-line block suitable for an InspectionScreen footer or
``care generate`` CLI output, so the same projection drives both
surfaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetadataSummary:
    """Projected view of one MAGE generation's metadata.

    Every field corresponds to a slot the TODO §4 P0 spec
    calls out, plus a handful of useful extras the same
    `MAGEMetadata` model carries (model name, generation time,
    verification flag, quality deltas).

    Frozen so the summary can be passed across screens / log
    handlers / saved-run records without defensive copies.
    """

    # Core info (always present on `MAGEMetadata`).
    domain: str = "general"
    num_steps: int = 0
    mode: str = "deep"
    """Generation mode (``"fast"`` / ``"deep"``). Source: the
    enclosing `MAGEResult.mode` field when summarising a result;
    falls back to the metadata's own default otherwise."""
    model: str = ""
    generation_time_seconds: float = 0.0

    # Deep-pipeline stage trail.
    stages_completed: tuple[str, ...] = field(default_factory=tuple)
    """Names of MAGE deep-mode stages that ran. Empty in fast
    mode or when the field isn't populated. CARE renders this as
    a checked-list / stage-trail badge bar."""

    # Research hits.
    memory_hits_used: int = 0
    web_results_used: int = 0
    was_cold_start: bool = False

    # Quality enhancement counters.
    step_critique_score: float | None = None
    """0-1 score from the step-critique stage; ``None`` when
    critique didn't run."""
    verification_passed: bool | None = None
    refine_iterations: int | None = None
    refine_quality_delta: float | None = None
    tot_branches_explored: int | None = None
    """Tree-of-Thought branch count. ``None`` when ToT didn't
    run (most fast-mode runs)."""
    mcts_simulations_run: int | None = None
    mcts_best_reward: float | None = None
    feedback_recalled: int | None = None

    # CARE-library defaults (MAGE §3.2). Surfaced here so the
    # SaveAgentModal can pre-fill its inputs.
    suggested_display_name: str = ""
    suggested_description: str = ""
    suggested_tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation. Tuples become lists for
        round-trip through ``json.dumps``."""
        out = asdict(self)
        out["stages_completed"] = list(self.stages_completed)
        out["suggested_tags"] = list(self.suggested_tags)
        return out

    def format_text(self) -> str:
        """Multi-line human-readable summary.

        Rendered into the InspectionScreen footer + the ``care
        generate`` CLI's final output. Fields that didn't apply
        to this run (``None`` values) are skipped so the block
        stays scan-friendly.
        """
        lines = [
            f"domain: {self.domain}",
            f"mode: {self.mode}",
            f"steps: {self.num_steps}",
        ]
        if self.model:
            lines.append(f"model: {self.model}")
        if self.generation_time_seconds:
            lines.append(
                f"generation: {self.generation_time_seconds:.2f}s"
            )
        if self.stages_completed:
            lines.append(f"stages: {', '.join(self.stages_completed)}")
        if self.memory_hits_used or self.was_cold_start:
            cold = " (cold start)" if self.was_cold_start else ""
            lines.append(f"memory hits: {self.memory_hits_used}{cold}")
        if self.web_results_used:
            lines.append(f"web results: {self.web_results_used}")
        # Quality enhancement block — only when something fired.
        quality_bits: list[str] = []
        if self.step_critique_score is not None:
            quality_bits.append(
                f"critique={self.step_critique_score:.2f}"
            )
        if self.verification_passed is not None:
            quality_bits.append(
                "verify=passed" if self.verification_passed else "verify=failed"
            )
        if self.refine_iterations is not None:
            quality_bits.append(f"refine_iters={self.refine_iterations}")
        if self.refine_quality_delta is not None:
            quality_bits.append(
                f"refine_Δ={self.refine_quality_delta:+.2f}"
            )
        if self.tot_branches_explored is not None:
            quality_bits.append(f"tot={self.tot_branches_explored}")
        if self.mcts_simulations_run is not None:
            quality_bits.append(f"mcts_sims={self.mcts_simulations_run}")
        if self.mcts_best_reward is not None:
            quality_bits.append(
                f"mcts_reward={self.mcts_best_reward:.2f}"
            )
        if self.feedback_recalled is not None:
            quality_bits.append(f"feedback={self.feedback_recalled}")
        if quality_bits:
            lines.append("quality: " + ", ".join(quality_bits))
        return "\n".join(lines)


def summarise_mage_result(source: Any) -> MetadataSummary:
    """Project a MAGE generation's metadata into a
    :class:`MetadataSummary`.

    Args:
        source: One of three accepted shapes — duck-typed so
            CARE doesn't pull ``mmar_mage`` at this layer:

            - A :class:`mmar_mage.MAGEResult` (or anything
              with ``.metadata`` + ``.mode``). We pick
              ``mode`` off the result and the rest off
              ``.metadata``.
            - A :class:`mmar_mage.MAGEMetadata` directly (or
              any object exposing the documented attribute
              names).
            - A plain `dict` — keys map to field names on
              :class:`MetadataSummary`.

    Returns:
        A populated :class:`MetadataSummary`. Missing fields
        keep their dataclass defaults (matches the convention
        MAGE itself uses, where optional metadata fields are
        ``None`` when the corresponding stage didn't run).
    """
    # Unwrap MAGEResult: it has `.metadata` (the MAGEMetadata)
    # plus `.mode` we want to surface on the summary.
    mode = _get(source, "mode", None)
    metadata_obj: Any = source
    nested = _get(source, "metadata", None)
    if nested is not None and not _looks_like_metadata(source):
        metadata_obj = nested

    if mode is None:
        mode = _get(metadata_obj, "mode", "deep")

    stages = _get(metadata_obj, "deep_stages_completed", None) or ()
    suggested_tags = _get(metadata_obj, "suggested_tags", ()) or ()

    return MetadataSummary(
        domain=str(_get(metadata_obj, "domain", "general") or "general"),
        num_steps=int(_get(metadata_obj, "num_steps", 0) or 0),
        mode=str(mode or "deep"),
        model=str(_get(metadata_obj, "model", "") or ""),
        generation_time_seconds=float(
            _get(metadata_obj, "generation_time_seconds", 0.0) or 0.0
        ),
        stages_completed=tuple(stages),
        memory_hits_used=int(
            _get(metadata_obj, "memory_hits_used", 0) or 0
        ),
        web_results_used=int(
            _get(metadata_obj, "web_results_used", 0) or 0
        ),
        was_cold_start=bool(_get(metadata_obj, "was_cold_start", False)),
        step_critique_score=_opt_float(
            _get(metadata_obj, "step_critique_score", None)
        ),
        verification_passed=_opt_bool(
            _get(metadata_obj, "verification_passed", None)
        ),
        refine_iterations=_opt_int(
            _get(metadata_obj, "refine_iterations", None)
        ),
        refine_quality_delta=_opt_float(
            _get(metadata_obj, "refine_quality_delta", None)
        ),
        tot_branches_explored=_opt_int(
            _get(metadata_obj, "tot_branches_explored", None)
        ),
        mcts_simulations_run=_opt_int(
            _get(metadata_obj, "mcts_simulations_run", None)
        ),
        mcts_best_reward=_opt_float(
            _get(metadata_obj, "mcts_best_reward", None)
        ),
        feedback_recalled=_opt_int(
            _get(metadata_obj, "feedback_recalled", None)
        ),
        suggested_display_name=str(
            _get(metadata_obj, "suggested_display_name", "") or ""
        ),
        suggested_description=str(
            _get(metadata_obj, "suggested_description", "") or ""
        ),
        suggested_tags=tuple(suggested_tags),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_METADATA_FINGERPRINT = ("domain", "num_steps", "memory_hits_used")
"""Attribute names present on `MAGEMetadata` but NOT on
`MAGEResult` — used to disambiguate the two when both shapes
arrive through the same duck-typed path."""


def _looks_like_metadata(source: Any) -> bool:
    """True when ``source`` is a metadata object (not a result
    wrapping one). Lets the unwrap step avoid double-stepping
    through ``source.metadata.metadata`` on weird nested
    inputs."""
    # `MAGEResult.metadata` is the MAGEMetadata; pure metadata
    # objects don't have a `chain_json` field.
    if hasattr(source, "chain_json"):
        return False
    if isinstance(source, dict) and "chain_json" in source:
        return False
    return any(hasattr(source, name) or (isinstance(source, dict) and name in source) for name in _METADATA_FINGERPRINT)


def _get(source: Any, name: str, default: Any) -> Any:
    """Attribute-or-key getter so the projector works against
    dicts, Pydantic models, and raw `dataclass`-style objects."""
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


__all__ = [
    "MetadataSummary",
    "summarise_mage_result",
]
