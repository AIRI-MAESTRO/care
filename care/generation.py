"""MAGE generation wiring (TODO §4 P0).

The future ``GenerationScreen`` needs to drive
``MAGEGenerator(config=MAGEConfig.from_toml(...),
progress=...)`` from CARE's :class:`CareConfig` + a
:class:`care.runtime.MagePoster`. The data-layer glue is
bounded:

* Translate ``CareConfig.mage`` (CARE's narrow Pydantic
  surface) into a fully-populated ``mmar_mage.MAGEConfig``.
* Lazy-import ``mmar_mage.MAGEGenerator`` so a CARE install
  without the ``mage`` extra still imports this module
  cleanly. Missing dep raises a friendly
  :class:`GenerationError` rather than a raw ``ImportError``.
* Wrap the async ``generate(query, ...)`` call so screen
  code (or the future ``care generate`` CLI) doesn't have to
  juggle kwarg names. Forwards ``context_files`` / ``cancel``
  / ``capabilities`` verbatim.

The actual screen work — composing the worker, wiring
``MagePoster`` into the app, rendering the streamed DAG — is
gated on §1 P0 UI. This module is the data-layer floor the
screen builds on.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig

from care.config import CareConfig, MageConfig


class GenerationError(RuntimeError):
    """Raised when the generation wiring can't proceed —
    missing API key, malformed config, or downstream
    ``generate()`` raised."""


def build_mage_config(
    care_config: CareConfig | MageConfig,
    *,
    mode: Literal["fast", "deep"] | None = None,
    deployable: bool = False,
) -> Any:
    """Build a ``mmar_mage.MAGEConfig`` from CARE's config.

    Translates the fields ``CareConfig.mage`` carries (provider,
    api_key, base_url, model, research toggles, etc.) into the
    ``MAGEConfig`` shape MAGE's generator expects. Validation
    lives on the MAGE side — we forward and let MAGE's Pydantic
    model error out if a value is invalid (the resulting
    ``ValidationError`` wraps in :class:`GenerationError`).

    Args:
        care_config: Either a full :class:`CareConfig` or just
            the ``mage`` section. Library code grabbing this
            from outside the TUI typically hand-rolls a
            :class:`MageConfig`.
        mode: Override for ``CareConfig.mage.mode``. The
            future ``GenerationScreen``'s Fast/Deep checkbox
            uses this to override without mutating the user's
            saved config.

    Returns:
        A fully-populated ``MAGEConfig`` instance.

    Raises:
        GenerationError: The translated config fails
            ``MAGEConfig`` validation, or required fields are
            missing.
    """
    mage = (
        care_config.mage if isinstance(care_config, CareConfig) else care_config
    )

    if not mage.api_key:
        raise GenerationError(
            "CareConfig.mage.api_key must be set to run generation; "
            "set CARE_MAGE__API_KEY or fill the `[mage] api_key` "
            "field in your config.toml"
        )

    resolved_mode = mode or mage.mode

    # `provider="custom"` tells MAGE to use whatever endpoint
    # we hand it via base_url, without applying provider-
    # specific defaults. CARE's user surface is single-endpoint
    # now (base_url + api_key + model); the provider concept
    # only lives on the MAGE side for internal routing.
    kwargs: dict[str, Any] = {
        "mode": resolved_mode,
        "provider": "custom",
        "api_key": mage.api_key,
        "enable_memory_research": mage.enable_memory_research,
        "enable_web_research": mage.enable_web_research,
        "enable_capability_lookup": getattr(mage, "enable_capability_lookup", False),
        "enable_memory_skill_lookup": getattr(mage, "enable_memory_skill_lookup", False),
        "enable_skill_discovery": getattr(mage, "enable_skill_discovery", False),
        "memory_search_mode": getattr(mage, "memory_search_mode", "bm25"),
        "memory_relevance_threshold": getattr(mage, "memory_relevance_threshold", 0.0),
        # Topology + depth — CARE wants richer shapes than MAGE's
        # short-linear defaults (see MageConfig docstrings).
        "enable_topology_selection": getattr(mage, "enable_topology_selection", True),
        "topology_max_candidates": getattr(mage, "topology_max_candidates", 3),
        "simplicity_bias": getattr(mage, "simplicity_bias", 0.2),
        "simplicity_max_steps": getattr(mage, "simplicity_max_steps", 7),
    }
    if mage.base_url:
        kwargs["base_url"] = mage.base_url
    if mage.model:
        kwargs["model"] = mage.model
    if mage.web_search_api_key:
        kwargs["web_search_api_key"] = mage.web_search_api_key
    if getattr(mage, "web_search_provider", None):
        kwargs["web_search_provider"] = mage.web_search_provider

    # Point MAGE's MemoryManager at the SAME Memory service CARE uses, so
    # recall/save hit one store instead of MAGEConfig's localhost:8002
    # default ignoring CARE's configured URL.
    if isinstance(care_config, CareConfig) and getattr(care_config.memory, "base_url", None):
        kwargs["memory_base_url"] = care_config.memory.base_url

    # Deployable (production) chains run headless — there's no human to answer a
    # `human_input` step, and it would stall the agent. Forbid that step type so MAGE
    # builds the agent to consume its input ($outer_context) directly. Ad-hoc leaves it
    # allowed (the user is right there to answer mid-run prompts).
    if deployable:
        try:
            from mmar_mage.schemas import VALID_STEP_TYPES

            kwargs["allowed_step_types"] = sorted(set(VALID_STEP_TYPES) - {"human_input"})
        except Exception:  # noqa: BLE001 — never block generation on this nicety
            pass

    try:
        return MAGEConfig(**kwargs)
    except Exception as exc:  # noqa: BLE001 — pydantic ValidationError is the common one
        raise GenerationError(
            f"failed to build MAGEConfig: {exc}"
        ) from exc


def build_mage_generator(
    care_config: CareConfig | MageConfig,
    *,
    progress: Any = None,
    mode: Literal["fast", "deep"] | None = None,
    deployable: bool = False,
) -> Any:
    """Construct a ``mmar_mage.MAGEGenerator`` ready for
    :func:`run_generation`.

    Args:
        care_config: Same as :func:`build_mage_config`.
        progress: A duck-typed ``MAGEProgressCallback`` —
            typically the :class:`care.runtime.MagePoster`
            wired to the running app. ``None`` keeps MAGE's
            default (no-op) progress reporter.
        mode: Mode override forwarded to
            :func:`build_mage_config`.

    Returns:
        A configured ``MAGEGenerator``.
    """
    config = build_mage_config(care_config, mode=mode, deployable=deployable)

    try:
        return MAGEGenerator(config=config, progress=progress)
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(
            f"failed to construct MAGEGenerator: {exc}"
        ) from exc


def _prepend_user_context(query: str, user_context: str) -> str:
    """Prefix ``query`` with a standing user-context block (CARE.md + a
    recalled LTM digest) so generation is personalised. No-op when empty."""
    block = (user_context or "").strip()
    if not block:
        return query
    return f"{block}\n\n---\n\nTASK:\n{query}"


async def run_generation(
    generator: Any,
    query: str,
    *,
    context_files: list[dict[str, Any]] | None = None,
    cancel: asyncio.Event | None = None,
    capabilities: Any = None,
    user_context: str = "",
) -> Any:
    """Drive ``generator.generate(...)`` and return the result.

    Thin async wrapper so the future ``GenerationScreen``
    worker (and the future ``care generate`` CLI) don't have
    to remember MAGE's kwarg names. Forwards everything
    verbatim.

    Args:
        generator: From :func:`build_mage_generator` (or a
            duck-typed stub for tests).
        query: The user's text query.
        context_files: Optional list of context-file refs
            (``path`` / ``sha256`` / ``size_bytes`` dicts).
            CARE-side helpers in :mod:`care.runtime.executor`
            build this from the file picker.
        cancel: Optional ``asyncio.Event`` — wired to CARE's
            `Esc` key by the screen. Setting it aborts
            generation between stages.
        capabilities: Optional ``mmar_mage.CapabilityContext`` —
            typically built via
            :func:`care.build_capability_payload(...).to_mage_context()`.

    Returns:
        The :class:`mmar_mage.MAGEResult` MAGE returned.
        Callers project it via :func:`care.summarise_mage_result`
        and :func:`care.project_intermediate_artifacts`.

    Raises:
        GenerationError: ``generator.generate(...)`` raised, or
            the supplied generator doesn't expose
            ``.generate(...)``.
    """
    if not isinstance(query, str) or not query.strip():
        raise GenerationError("query must be a non-empty string")

    # Personalise PLANNING: prepend the standing user context (CARE.md + a
    # recalled LTM digest) so the planner builds chains aware of who the user
    # is + what they've asked before. Empty context → query unchanged.
    query = _prepend_user_context(query, user_context)

    generate = getattr(generator, "generate", None)
    if not callable(generate):
        raise GenerationError(
            "generator is missing `generate(...)` — pass a "
            "MAGEGenerator or a duck-typed equivalent"
        )

    try:
        coro: Any = generate(
            query,
            context_files=context_files,
            cancel=cancel,
            capabilities=capabilities,
        )
        return await coro
    except GenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(
            f"generate() raised: {exc}"
        ) from exc


async def run_edit(
    generator: Any,
    instruction: str,
    *,
    entity_id: str | None = None,
    chain: dict[str, Any] | None = None,
    channel: str = "latest",
    save: bool = False,
    cancel: asyncio.Event | None = None,
) -> Any:
    """Drive ``generator.edit(...)`` and return the ``MAGEEditResult``.

    Thin async wrapper mirroring :func:`run_generation` for the NL *edit* flow:
    ChatScreen's ``/revise`` worker and the library "Revise (AI)" action call
    this so they don't have to juggle MAGE's kwarg names.

    Args:
        generator: From :func:`build_mage_generator` (or a duck-typed stub).
        instruction: What to change, in natural language.
        entity_id: Id of the saved chain to edit. When omitted (and no
            ``chain`` is given) MAGE resolves the chain from ``instruction``
            via memory search (and may return a disambiguation result).
        chain: An in-memory chain dict to edit directly (skips resolution).
        channel: Version channel to load from / save to.
        save: Persist the edited chain as a new version when an id is known.
        cancel: Optional ``asyncio.Event`` — wired to CARE's `Esc` key.

    Returns:
        The :class:`mmar_mage.MAGEEditResult` MAGE returned (which may carry
        ``needs_disambiguation=True`` + ``candidates`` instead of an edit).

    Raises:
        GenerationError: ``generator.edit(...)`` raised, or the supplied
            generator doesn't expose ``.edit(...)``.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise GenerationError("instruction must be a non-empty string")

    edit = getattr(generator, "edit", None)
    if not callable(edit):
        raise GenerationError(
            "generator is missing `edit(...)` — pass a MAGEGenerator or a "
            "duck-typed equivalent"
        )

    try:
        return await edit(
            instruction,
            chain=chain,
            entity_id=entity_id,
            channel=channel,
            save=save,
            cancel=cancel,
        )
    except GenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"edit() raised: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


__all__ = [
    "GenerationError",
    "build_mage_config",
    "build_mage_generator",
    "run_generation",
    "run_edit",
]
