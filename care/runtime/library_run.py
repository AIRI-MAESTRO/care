"""Re-run-from-library data layer (TODO §3 P1).

The LibraryScreen's "Run" action on a saved agent loads the chain
from Memory, opens the `RunContextModal` seeded with the chain's
stored task description + context files, and (on confirm) runs
the chain through CARL's executor, persisting a `memory_card`
run record linked back to the source agent entity.

The Textual screen + modal are gated on TODO §1 P0 multi-screen
workflow, but the load + execute orchestration is bounded and
independent — this module ships it now.

What this module provides:

* :class:`LibraryRunPlan` — frozen bundle returned by
  :func:`load_run_plan` carrying the CARL chain object,
  identity metadata, and a pre-populated :class:`RunContextDraft`
  ready for the modal to bind to.
* :class:`LibraryRunError` — single error class the screen
  catches.
* :func:`load_run_plan` — async helper that fetches the chain
  via the SDK, builds the draft, and packages everything for
  the modal.
* :func:`execute_library_run` — async orchestrator that takes a
  plan + a finalised draft (post-modal-confirm), primes a CARL
  context, runs the chain, and records the run completion.
  Returns the typed :class:`care.runtime.RunCompletion`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from care.runtime.executor import execute_chain_async, prime_from_saved_chain
from care.runtime.run_context_draft import (
    RunContextDraft,
    apply_overrides,
    build_extra_kwargs,
    extract_run_context_draft,
    validate_run_context_draft,
)
from care.runtime.run_recorder import RunCompletion, record_run_completion


EntityKind = Literal["chain", "agent", "agent_skill"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LibraryRunError(RuntimeError):
    """Raised when the library-run flow fails — chain not in
    Memory, SDK unreachable, invalid pre-flight state. The
    LibraryScreen catches this and shows a friendly toast."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryRunPlan:
    """Everything the modal needs to drive a re-run.

    Frozen so it flows through Textual messages without defensive
    copies. The :attr:`draft` slot is the initial form state the
    modal mutates; the rest is read-only context the modal uses
    to render the header ("Re-run of <name> v3 …").
    """

    chain: Any
    entity_id: str
    entity_type: EntityKind = "chain"
    channel: str = "latest"
    display_name: str = ""
    draft: RunContextDraft = None  # type: ignore[assignment]

    @property
    def has_chain(self) -> bool:
        return self.chain is not None


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


async def load_run_plan(
    memory: Any,
    entity_id: str,
    *,
    channel: str = "latest",
    entity_type: EntityKind = "chain",
    source_name: str = "",
    timeout: float = 10.0,
) -> LibraryRunPlan:
    """Fetch a saved chain + build the initial RunContextDraft.

    The caller is typically the LibraryScreen's "Run" action
    handler. The returned plan is handed to the RunContextModal
    which mutates :attr:`LibraryRunPlan.draft` via the
    :mod:`care.runtime.run_context_draft` mutators (`set_task`,
    `add_file`, etc.) and finally calls :func:`execute_library_run`
    on confirm.

    Args:
        memory: A `CareMemory` facade (or any object exposing
            ``.client.get_chain(...)``). Tests pass stubs.
        entity_id: Memory entity id of the saved chain.
        channel: Memory channel to load (default ``"latest"`` —
            the LibraryScreen's home view; ``"stable"`` for
            "Run stable version" actions).
        entity_type: Currently only ``"chain"`` actually loads;
            ``"agent"`` / ``"agent_skill"`` reserved for the
            forthcoming agent runtime. Passed through into the
            plan so the eventual `record_run_completion` knows
            which typed router to ping.
        source_name: Human-readable label for the modal header
            and the draft's `source_name`. Falls back to the
            chain's CARE metadata `display_name`.
        timeout: Per-call deadline in seconds for the SDK fetch.

    Returns:
        :class:`LibraryRunPlan` with the materialised CARL chain
        plus a pre-populated :class:`RunContextDraft`.

    Raises:
        LibraryRunError: ``entity_id`` is empty, the SDK fetch
            timed out / failed, or the loaded chain projects
            into an empty draft (no CARE metadata + no entity
            id — re-run would have nothing to run on).
    """
    if not entity_id:
        raise LibraryRunError("entity_id is required to load a run plan")

    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    if client is None:
        raise LibraryRunError(
            "memory facade does not expose client.get_chain()"
        )

    start = time.monotonic()
    try:
        chain = await asyncio.wait_for(
            asyncio.to_thread(_fetch_chain_object, client, entity_id, channel),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        latency = (time.monotonic() - start) * 1000
        raise LibraryRunError(
            f"chain fetch timed out after {timeout:.1f}s ({latency:.0f}ms elapsed)"
        ) from exc
    except LibraryRunError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LibraryRunError(
            f"chain fetch failed: {type(exc).__name__}: {exc}"
        ) from exc

    if chain is None:
        raise LibraryRunError(
            f"chain {entity_id!r} not found on channel {channel!r}"
        )

    # Stamp the entity_id on the chain so `extract_run_context_draft`
    # picks it up (CARL chain objects don't always carry entity_id
    # natively — the SDK projects them from `_get_entity` content
    # which loses the id).
    if not getattr(chain, "entity_id", None):
        try:
            setattr(chain, "entity_id", entity_id)
        except (AttributeError, TypeError):
            # Frozen or otherwise immutable chain shape — fall back
            # to passing entity_id via the draft directly below.
            pass

    draft = extract_run_context_draft(chain, source_name=source_name)
    if not draft.source_entity_id:
        # When the chain object refused to accept entity_id, patch
        # the draft so downstream `record_run_completion` knows
        # which agent to bump.
        from dataclasses import replace

        draft = replace(draft, source_entity_id=entity_id)

    label = source_name or draft.source_name
    return LibraryRunPlan(
        chain=chain,
        entity_id=entity_id,
        entity_type=entity_type,
        channel=channel,
        display_name=label,
        draft=draft,
    )


async def execute_library_run(
    memory: Any,
    plan: LibraryRunPlan,
    draft: RunContextDraft,
    *,
    config: Any,
    api: Any,
    streamer: Any = None,
    tools_path: Optional[str] = None,
    run_id: Optional[str] = None,
    author: Optional[str] = None,
    record_completion: bool = True,
) -> RunCompletion:
    """End-to-end re-run: prime context → execute → record card.

    Composes the existing single-purpose helpers
    (:func:`care.runtime.prime_from_saved_chain` +
    :func:`care.runtime.execute_chain_async` +
    :func:`care.runtime.record_run_completion`) so the
    LibraryScreen's "Run" handler is a single ``await`` call.

    Args:
        memory: `CareMemory` facade for the completion-record
            persistence.
        plan: :class:`LibraryRunPlan` returned by
            :func:`load_run_plan`.
        draft: User-finalised :class:`RunContextDraft` (post-
            modal-confirm). Validation issues should be cleared
            BEFORE calling this; the function re-runs validation
            internally and raises :class:`LibraryRunError` on
            any unresolved error.
        config: Caller's :class:`CareConfig` — used to apply the
            draft's per-run model / provider overrides via
            :func:`apply_overrides`. The session-wide config is
            never mutated.
        api: LLM-API-like object to bind to the new context. The
            caller typically builds this via
            :func:`care.runtime.build_llm_client(applied_config.mage)`
            BEFORE calling so the overrides take effect; the
            applied config is returned via the run's metadata
            for transparency.
        streamer: Optional `CarlStreamer` to attach to the
            context — drives the LLMChunk / StepStarted / etc.
            Textual messages.
        tools_path: Forwarded to :func:`prime_from_saved_chain`.
        run_id: Override the synthesised run id (defaults to
            ``"run-<UTC timestamp>"``).
        author: Author tag for the recorded `memory_card`.
        record_completion: ``False`` skips the completion-record
            write — useful for dry-run / live-tracing modes.

    Returns:
        :class:`care.runtime.RunCompletion` with the persisted
        ``memory_card`` id + the typed :class:`RunSummary` of
        the just-finished run.

    Raises:
        LibraryRunError: validation issues remain on the draft,
            or the executor / record step raised an
            unrecoverable error.
    """
    if plan.chain is None:
        raise LibraryRunError("plan has no chain object loaded")

    issues = validate_run_context_draft(draft, check_files=True)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        msgs = "; ".join(i.message for i in errors)
        raise LibraryRunError(f"run draft has unresolved errors: {msgs}")

    extras = build_extra_kwargs(draft)
    # If the chain has document-reading skill steps (docx/pdf/…), rewrite them
    # to read the attached file from $memory.input.<key> (+ a task placeholder
    # so the model actually sees it) — the same bridge the chat path uses, so
    # a document chain run from the library behaves identically. The model
    # classifies read-vs-create (keyword heuristic is the fallback).
    reads = None
    try:
        from care.skill_file_inputs import classify_reads

        _cd = _chain_to_dict(plan.chain)
        if _cd is not None:
            reads = await classify_reads(api, _cd)
    except Exception:  # noqa: BLE001 — heuristic fallback
        reads = None
    run_chain = _apply_skill_file_bridge(plan.chain, draft, extras, reads=reads)
    applied_config = apply_overrides(config, draft)

    try:
        # ``config=`` is what registers CARE's builtin tools (web_search,
        # current_datetime, …) on the context — without it every chain that
        # planned a builtin dies with "Tool '<name>' not registered in
        # context" (the C1 promotion gate's baseline runs through here).
        context = prime_from_saved_chain(
            run_chain,
            api=api,
            config=applied_config,
            streamer=streamer,
            tools_path=tools_path,
            **extras,
        )
    except Exception as exc:  # noqa: BLE001
        raise LibraryRunError(
            f"failed to prime context for re-run: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        result = await execute_chain_async(run_chain, context)
    except Exception as exc:  # noqa: BLE001
        raise LibraryRunError(
            f"chain execution failed: {type(exc).__name__}: {exc}"
        ) from exc

    if not record_completion:
        # Caller doesn't want a card written (e.g. dry-run); still
        # return a `RunCompletion`-shaped value with the result
        # summary so the calling code stays single-branch.
        from care.runtime.run_recorder import (
            extract_final_output,
            summarise_reasoning_result,
        )

        summary = summarise_reasoning_result(result)
        return RunCompletion(
            memory_card_entity_id="",
            agent_entity_id=plan.entity_id,
            run_id=run_id or _utc_run_id(),
            summary=summary,
            agent_recorded=False,
            final_output=extract_final_output(result),
        )

    return await asyncio.to_thread(
        record_run_completion,
        memory,
        agent_entity_id=plan.entity_id,
        agent_name=plan.display_name or plan.entity_id,
        result=result,
        query=draft.task_description or None,
        run_id=run_id,
        agent_entity_type=plan.entity_type,
        author=author,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_to_dict(chain: Any) -> Optional[dict]:
    """Best-effort ``ReasoningChain`` → plain dict (for the skill bridge)."""
    if isinstance(chain, dict):
        return chain
    to_dict = getattr(chain, "to_dict", None)
    if callable(to_dict):
        try:
            out = to_dict(full=True)
        except TypeError:
            out = to_dict()
        except Exception:  # noqa: BLE001
            return None
        if isinstance(out, dict):
            return out
    return None


def _apply_skill_file_bridge(
    chain: Any, draft: RunContextDraft, extras: dict, *, reads: Any = None,
) -> Any:
    """Rewrite document-reading skill steps to consume the attached file.

    Mirrors the chat path's :func:`care.skill_file_inputs.apply_file_inputs`
    so a docx/pdf/… chain re-run from the library actually feeds the document
    to the skill (binds ``$memory.input.<key>`` + injects a ``{param}``
    placeholder into the skill task). Merges the file payload into
    ``extras['files']``. Best-effort: any failure returns the chain unchanged
    so a normal run is never blocked.
    """
    active = [cf for cf in draft.active_files if cf.path]
    if not active:
        return chain
    try:
        import copy

        from mmar_carl import ReasoningChain

        from care.runtime.file_loading import load_file
        from care.skill_file_inputs import (
            apply_file_inputs,
            requires_file_input,
        )

        chain_dict = _chain_to_dict(chain)
        if not chain_dict or not requires_file_input(chain_dict, reads=reads):
            return chain
        attachments = [(cf.path, load_file(cf.path).content) for cf in active]
        new_dict, skill_files = apply_file_inputs(
            chain_dict, attachments, reads=reads,
        )
        if not skill_files:
            return chain
        files = dict(extras.get("files") or {})
        files.update(skill_files)
        extras["files"] = files
        extras["load_files_from_metadata"] = False
        return ReasoningChain.from_dict(
            copy.deepcopy(new_dict), use_typed_steps=True,
        )
    except Exception:  # noqa: BLE001 — never block a normal run on the bridge
        return chain


def _fetch_chain_object(client: Any, entity_id: str, channel: str) -> Any:
    """Fetch a saved chain as a runnable ``ReasoningChain``.

    The SDK's ``get_chain`` parses via the legacy ``use_typed_steps=False``
    path, whose ``StepDescription`` union predates ``AgentSkillStepConfig`` —
    so any chain with an ``agent_skill`` step (docx / pdf / … skills) makes it
    raise a Pydantic ``ValidationError`` and the Run button silently dies on
    the load. Prefer the raw-dict accessor + a typed parse (the same
    ``from_dict(use_typed_steps=True)`` path that executes every MAGE step
    type), and only fall back to ``get_chain`` for SDKs that lack the raw
    accessor or for content the typed parse can't handle.
    """
    get_dict = getattr(client, "get_chain_dict", None)
    if callable(get_dict):
        try:
            raw = get_dict(entity_id, channel)
        except Exception:  # noqa: BLE001 — fall back to get_chain below
            raw = None
        if isinstance(raw, dict) and raw:
            try:
                import copy

                from mmar_carl import ReasoningChain

                return ReasoningChain.from_dict(
                    copy.deepcopy(raw), use_typed_steps=True,
                )
            except Exception:  # noqa: BLE001 — fall back to get_chain
                pass
    get_chain = getattr(client, "get_chain", None)
    if callable(get_chain):
        return get_chain(entity_id, channel)
    raise LibraryRunError(
        "memory facade does not expose client.get_chain()/get_chain_dict()"
    )


def _utc_run_id() -> str:
    """UTC-stamped human-readable run id; matches the format
    `record_run_completion` uses when run_id isn't supplied."""
    return f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"


__all__ = [
    "EntityKind",
    "LibraryRunError",
    "LibraryRunPlan",
    "execute_library_run",
    "load_run_plan",
]
