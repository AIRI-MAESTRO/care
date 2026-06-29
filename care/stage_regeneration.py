"""Per-stage MAGE re-runs (TODO §4 P1).

The user opens an InspectionScreen on a saved chain, decides the
DAG is wrong but the plan is fine, and clicks **Regenerate DAG
only**. We don't want to re-drive the whole MAGE pipeline —
:meth:`MAGEGenerator.build_dag(plan)` already does the right
thing, plus parallel entry points for every other stage (MAGE
§3.7, shipped).

This module is the CARE-side dispatcher that takes:

* a `RegenerateStage` literal (which stage to re-run),
* the inputs that stage needs (drawn from a prior `MAGEResult` /
  `intermediate_artifacts`),
* an already-constructed `MAGEGenerator`,

…and calls the right method with the right kwargs. Returns a
:class:`StageArtifact` carrying the produced artifact + the stage
name so the caller can stash it back into a `MAGEResult.intermediate_artifacts`
slot without having to know which key matches.

CARE doesn't import `mmar_mage` at the top — the generator
argument is duck-typed against the per-stage methods. Tests pass
a stub that records calls without instantiating any real LLM
plumbing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal


RegenerateStage = Literal[
    "domain",
    "plan",
    "dag",
    "describe",
    "critique",
    "verify",
    "refine",
]
"""One of MAGE's seven per-stage entrypoints. Naming mirrors
:class:`MAGEGenerator`'s method names with the `analyze_`/`_steps`/
`_chain` boilerplate trimmed so the CARE-side enum stays tight."""


_STAGE_METHODS: dict[RegenerateStage, str] = {
    "domain": "analyze_domain",
    "plan": "plan_steps",
    "dag": "build_dag",
    "describe": "describe_steps",
    "critique": "critique_steps",
    "verify": "verify_chain",
    "refine": "refine",
}


# Mandatory kwargs per stage — keys MUST exist on the input dict.
# Extra keys are allowed (forwarded as-is); missing keys raise
# `StageRegenerationError`. This keeps callers honest about which
# stage they actually have the inputs for — much friendlier than
# the SDK's downstream TypeError when a kwarg is missing.
_REQUIRED_INPUTS: dict[RegenerateStage, tuple[str, ...]] = {
    "domain": ("query",),
    "plan": ("query", "domain_analysis"),
    "dag": ("plan",),
    "describe": ("query", "dag"),
    "critique": ("query", "steps", "domain_analysis"),
    "verify": ("query", "chain_dict", "domain_analysis"),
    "refine": ("query", "chain_dict", "domain_analysis"),
}


class StageRegenerationError(RuntimeError):
    """Raised when the dispatcher can't proceed — unknown stage,
    missing required inputs, generator missing the per-stage
    method (older `mmar_mage` install), or the underlying call
    raised."""


@dataclass(frozen=True)
class StageArtifact:
    """One stage's re-run output.

    Frozen so caller code can stash the result somewhere safe
    without defensive copies. ``stage`` echoes the dispatcher
    input; ``artifact`` is whatever MAGE returned — typed via
    ``Any`` because CARE doesn't import the MAGE schema types
    at this layer.
    """

    stage: RegenerateStage
    artifact: Any
    """Stage-specific MAGE return — `DomainAnalysis`,
    `StepPlan`, `DAGStructure`, `list[CARLStepSchema]`,
    `StepCritiqueResult`, etc. CARE callers cast based on
    ``stage``."""


async def regenerate_stage(
    generator: Any,
    stage: RegenerateStage,
    inputs: dict[str, Any],
) -> StageArtifact:
    """Re-run a single MAGE stage.

    Args:
        generator: Any object exposing the per-stage method
            names — typically `mmar_mage.MAGEGenerator`, but
            tests pass a duck-typed stub.
        stage: Which stage to re-run.
        inputs: Stage inputs. Required keys per stage:

            - ``"domain"`` → ``query``
            - ``"plan"`` → ``query`` + ``domain_analysis``
              (optional: ``memory_digest``, ``web_digest``,
              ``capability_context``, ``template_skeleton``,
              ``skill_digest``, ``allowed_step_types``)
            - ``"dag"`` → ``plan``
            - ``"describe"`` → ``query`` + ``dag``
              (optional: ``allowed_step_types``)
            - ``"critique"`` → ``query`` + ``steps`` +
              ``domain_analysis``
              (optional: ``threshold``)
            - ``"verify"`` → ``query`` + ``chain_dict`` +
              ``domain_analysis``
            - ``"refine"`` → ``query`` + ``chain_dict`` +
              ``domain_analysis``

            Any extra keys are forwarded to the stage method.
            Missing required keys raise
            :class:`StageRegenerationError` before the method
            is called.

    Returns:
        :class:`StageArtifact` carrying the stage name + the
        raw MAGE return.

    Raises:
        StageRegenerationError: Unknown stage, missing required
            input, missing per-stage method on ``generator``,
            or downstream exception during the call.
    """
    if stage not in _STAGE_METHODS:
        raise StageRegenerationError(
            f"unknown stage {stage!r}; supported: "
            f"{', '.join(sorted(_STAGE_METHODS))}"
        )

    missing = [k for k in _REQUIRED_INPUTS[stage] if k not in inputs]
    if missing:
        raise StageRegenerationError(
            f"stage {stage!r} requires inputs {missing!r}"
        )

    method_name = _STAGE_METHODS[stage]
    method = getattr(generator, method_name, None)
    if method is None or not callable(method):
        raise StageRegenerationError(
            f"generator has no {method_name!r} method; "
            "upgrade mmar_mage to a version that ships per-stage "
            "entrypoints (MAGE §3.7)"
        )

    required = _REQUIRED_INPUTS[stage]
    positional = [inputs[k] for k in required]
    extras = {k: v for k, v in inputs.items() if k not in required}

    try:
        result = method(*positional, **extras)
    except StageRegenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StageRegenerationError(
            f"stage {stage!r} failed: {exc}"
        ) from exc

    # Per-stage methods on MAGEGenerator are `async def`; some
    # test stubs (and the future synchronous variants) return a
    # plain value. Handle both.
    if asyncio.iscoroutine(result):
        try:
            result = await result
        except StageRegenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StageRegenerationError(
                f"stage {stage!r} failed: {exc}"
            ) from exc

    return StageArtifact(stage=stage, artifact=result)


def supported_stages() -> list[RegenerateStage]:
    """List of stage names this dispatcher knows how to invoke.

    Sorted for stable UI rendering (the future "Regenerate"
    dropdown enumerates these). Pure-Python no-deps so callers
    can import it without paying any cost."""
    return sorted(_STAGE_METHODS)


__all__ = [
    "RegenerateStage",
    "StageArtifact",
    "StageRegenerationError",
    "regenerate_stage",
    "supported_stages",
]
