"""CARE-side chain execution entrypoint (TODO §5 P1).

Wraps CARL's ``ReasoningContext`` + ``ReasoningChain.execute_async``
into a single helper the future ExecutionScreen calls. Three modes:

1. **Fresh run** from a query + context files (used right after
   MAGE generation, before the chain is saved).
2. **Re-run from saved chain** — recovers `task_description` +
   `context_files` from the chain's CARE metadata block via
   CARL's ``ReasoningContext.from_chain_inputs`` (PREPARE.md §5.6).
3. **Replay with overrides** — caller supplies their own
   `query` / `files` to override what the saved chain stored.

Every public helper accepts an optional :class:`CarlStreamer` —
when given, its ``attach`` populates every ``on_*`` callback on
the new context so the screen sees lifecycle events without extra
plumbing. Tool registration via CARL's `register_tools_from_path`
(PREPARE.md §5.7) is the other plug-in seam: pass `tools_path` and
every `@carl_tool`-decorated callable in that glob is registered.

CARL is imported **lazily** inside the helpers, so a broken or
absent ``mmar_carl`` install can't break CARE module load. The
executor is a thin coordinator — it doesn't decide which LLM to
use or where to persist results (that's `CareConfig.mage` and
:func:`care.runtime.record_run_completion`'s job, respectively).
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

from care.config import CareConfig

_log = logging.getLogger("care.runtime.executor")


# Grounds every reasoning step in the tool results that precede it. Without
# this, an ``llm`` step happily ignores a web_search / current_datetime result
# sitting in its History and answers from stale training knowledge instead —
# e.g. "the tournament hasn't happened yet" even when the search result already
# states the final score. CARL renders this once at the top of every step
# prompt (ReasoningContext.system_prompt → PromptTemplate.format_chain_prompt),
# so it reaches the synthesis step that has to consume a tool's output.
_GROUNDING_SYSTEM_PROMPT = (
    "You are the reasoning core of a tool-using agent. The History section of a "
    "step may contain results returned by tools (web_search, current_datetime, "
    "fetch_url, http_request, calculator, …) — often a line beginning with "
    "'Answer:' or fetched page text. Treat those tool results as ground truth: "
    "they reflect the real, current world and OVERRIDE your own training "
    "knowledge. Base your answer strictly on them. If a tool result states the "
    "current date, a score, a price, a winner, a latest release, etc., report it "
    "as given — even if it contradicts what you think you know. Never reply that "
    "information is unavailable, or that an event 'hasn't happened yet', when a "
    "tool result in the History already provides it. Answer in the user's language."
)


class ExecutionError(RuntimeError):
    """Raised when the executor can't build a runnable context —
    e.g. CARL isn't installed, an LLM api factory raised, or a
    saved chain's metadata refers to files that no longer exist
    and ``strict_files=True``."""


def build_run_context(
    *,
    query: str,
    api: Any,
    config: CareConfig | None = None,
    files: dict[str, str] | None = None,
    streamer: Any = None,
    tools_path: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
    human_input_provider: Any = None,
    long_term_memory: Any = None,
    session_id: str | None = None,
    user_context: str = "",
) -> Any:
    """Construct a fresh :class:`ReasoningContext` for a one-shot run.

    Args:
        query: User's task description. Persisted on the chain's
            ``outer_context`` slot (CARL reads from there on every
            step).
        api: An LLM-API-like object that CARL's steps will call.
            Caller builds this from ``CareConfig.mage`` — we don't
            do it here because LLM-client construction varies
            across providers.
        config: Optional :class:`CareConfig`. Used for language /
            namespace defaults; safe to omit during early dev.
        files: Optional ``{name: text}`` dict pre-loaded into
            ``context.memory["input"]`` so a step can ``${input.file}``
            reference it.
        streamer: Optional :class:`CarlStreamer`. When given,
            :meth:`CarlStreamer.attach` populates the context's
            ``on_*`` callbacks.
        tools_path: Optional glob like ``"~/.config/care/tools/*.py"``.
            Forwarded to ``context.register_tools_from_path`` to
            discover ``@carl_tool``-decorated callables.
        extra_kwargs: Free-form kwargs forwarded to the
            ``ReasoningContext`` constructor (``language=``,
            ``system_prompt=``, etc.) — covers fields CARE doesn't
            mirror explicitly yet.

    Returns:
        A configured ``ReasoningContext``. Type is ``Any`` because
        the import is lazy and the caller doesn't need static
        access to the CARL type.
    """
    ctx_cls = _reasoning_context_cls()
    kwargs: dict[str, Any] = dict(extra_kwargs or {})
    if config is not None:
        kwargs.setdefault("language", config.defaults.language)
    # Always-inject the standing user context (CARE.md + LTM digest) into the
    # grounding system prompt so EVERY step — including the one that produces
    # the user-facing answer — is personalised, not just the planner.
    grounding = _GROUNDING_SYSTEM_PROMPT
    _uc = (user_context or "").strip()
    if _uc:
        grounding = f"{_uc}\n\n{grounding}"
    kwargs.setdefault("system_prompt", grounding)
    # Attach CARL's native long-term memory so the chain can recall durable
    # user context on demand (``$ltm.<key>`` refs / a memory step), scoped by
    # ``session_id``. Optional — omitted call-sites get no LTM.
    if long_term_memory is not None:
        kwargs.setdefault("long_term_memory", long_term_memory)
    if session_id:
        kwargs.setdefault("session_id", session_id)
    memory = {"input": dict(files or {})}
    try:
        ctx = ctx_cls(
            outer_context=query,
            api=api,
            memory=memory,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        raise ExecutionError(
            f"failed to construct ReasoningContext: {exc}"
        ) from exc

    _attach_streamer(ctx, streamer)
    _apply_default_tools(ctx, config)
    _register_tools(ctx, tools_path)
    _wire_human_input(ctx, human_input_provider)
    return ctx


def prime_from_saved_chain(
    chain: Any,
    *,
    api: Any,
    config: CareConfig | None = None,
    outer_context: str | None = None,
    files: dict[str, str] | None = None,
    load_files_from_metadata: bool = True,
    streamer: Any = None,
    tools_path: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build a context for re-running a saved chain.

    Delegates to CARL's :meth:`ReasoningContext.from_chain_inputs`
    (PREPARE.md §5.6) which recovers ``task_description`` +
    ``context_files`` from the chain's CARE metadata block.

    Args:
        chain: A ``ReasoningChain`` previously persisted by CARE
            (so it carries `set_care_metadata` data).
        api: LLM-API-like object (see :func:`build_run_context`).
        outer_context: Override the saved task. ``None`` means
            "use the chain's stored ``task_description``" — that's
            the standard re-run flow. The RunContextModal (§1.3
            P1) passes the user's edited query when they choose
            to override.
        files: Augment / override the metadata-loaded
            ``input`` files. Wins on key clash so the modal can
            patch a single attachment without redoing every file.
        load_files_from_metadata: ``False`` skips auto-loading of
            the saved file paths — useful when the user wants to
            supply a completely fresh file set.
        streamer / tools_path / extra_kwargs: Same as
            :func:`build_run_context`.
    """
    ctx_cls = _reasoning_context_cls()
    kwargs: dict[str, Any] = dict(extra_kwargs or {})
    kwargs.setdefault("system_prompt", _GROUNDING_SYSTEM_PROMPT)
    try:
        ctx = ctx_cls.from_chain_inputs(
            chain,
            api=api,
            outer_context=outer_context,
            files=files,
            load_files_from_metadata=load_files_from_metadata,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        raise ExecutionError(
            f"failed to prime context from saved chain: {exc}"
        ) from exc

    _attach_streamer(ctx, streamer)
    _apply_default_tools(ctx, config)
    _register_tools(ctx, tools_path)
    return ctx


async def execute_chain_async(chain: Any, context: Any) -> Any:
    """Run ``chain`` against ``context`` and return the
    :class:`ReasoningResult`.

    Thin wrapper that exists so call-sites have ONE import point
    for the execution surface (lets future iterations swap in
    cancellation / timeout handling without screen-side changes).
    """
    if not hasattr(chain, "execute_async"):
        raise ExecutionError(
            f"chain object missing ``execute_async``; got "
            f"{type(chain).__name__}"
        )
    _normalize_tool_input_literals(chain)
    try:
        return await chain.execute_async(context)
    except Exception as exc:  # noqa: BLE001
        # ExecutionError lets the screen distinguish "the chain
        # crashed" from "the executor itself broke". CARL's own
        # validation errors keep their original type via __cause__.
        raise ExecutionError(f"chain execution failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Internals — kept module-level so tests can monkey-patch them
# ---------------------------------------------------------------------------


def _normalize_tool_input_literals(chain: Any) -> None:
    """Quote plain-literal tool inputs so CARL's resolver passes them through.

    CARL's ``resolve_context_reference`` only returns a literal verbatim when
    it is wrapped in quotes; a bare, non-``$`` string falls through every
    branch to a metadata lookup and resolves to ``None``. MAGE emits search
    queries and other tool arguments as bare literals (e.g.
    ``{"query": "UEFA … winner score"}``), so without this they reach the tool
    as an empty value — the tool then reports "empty query" and the synthesis
    step has nothing to work with. We wrap such values in double quotes in
    place; ``$refs`` and already-quoted values are skipped, so the pass is
    idempotent and safe to run on every chain (including saved re-runs and
    augment-rewritten ``$outer_context`` steps).
    """
    for step in getattr(chain, "steps", None) or []:
        cfg = getattr(step, "step_config", None)
        mapping = getattr(cfg, "input_mapping", None)
        if not isinstance(mapping, dict):
            continue
        for key, val in list(mapping.items()):
            if not isinstance(val, str) or not val:
                continue
            if val.startswith("$"):
                continue  # dynamic reference — let CARL resolve it
            if len(val) >= 2 and val[0] in "\"'" and val[-1] == val[0]:
                continue  # already a quoted literal
            mapping[key] = f'"{val}"'


def _reasoning_context_cls() -> type:
    """Lazy import of :class:`mmar_carl.ReasoningContext`."""
    try:
        from mmar_carl.models.context import ReasoningContext
    except ImportError as exc:
        raise ExecutionError(
            "mmar_carl is not installed; install the runtime to execute chains."
        ) from exc
    return ReasoningContext


def _attach_streamer(context: Any, streamer: Any) -> None:
    """Wire the streamer's callbacks onto the context if one was
    supplied. Silent no-op when ``streamer`` is None so call-sites
    can stay uniform."""
    if streamer is None:
        return
    if hasattr(streamer, "attach"):
        streamer.attach(context)


def _wire_human_input(context: Any, provider: Any) -> None:
    """P6.4 — bridge a CARL ``human_input`` step to a CARE-side ``provider``.

    CARL calls ``context.on_human_input_requested(prompt, future)`` and blocks
    on ``future`` until someone resolves it. We install a handler that calls
    ``provider(prompt)`` (sync OR async), coerces the answer to ``str`` and
    resolves the future. Best-effort: a provider that errors resolves to ``""``
    so the chain never hangs; ``provider=None`` leaves CARL's own
    ``fallback_value`` path untouched."""
    if provider is None:
        return

    async def _handler(prompt: Any, future: Any) -> None:
        try:
            answer = provider(prompt)
            if inspect.isawaitable(answer):
                answer = await answer
            if not future.done():
                future.set_result("" if answer is None else str(answer))
        except Exception as exc:  # noqa: BLE001
            _log.info("human_input provider failed: %s", exc)
            if not future.done():
                future.set_result("")

    try:
        context.on_human_input_requested = _handler
    except Exception as exc:  # noqa: BLE001
        _log.info("human_input wiring skipped: %s", exc)


def _apply_default_tools(context: Any, config: CareConfig | None) -> None:
    """Register CARE's bundled standard tools + the user's
    ``@carl_tool`` directory onto ``context``.

    This is what makes a MAGE-generated ``tool`` step ("call
    ``web_search``") actually resolve at execution time — without it
    every chain that planned a tool died with ``Tool '<name>' not
    registered in context``.

    Order: builtins first, then the user's ``config.tools.path`` so a
    same-named user tool overrides a builtin (``register_tool``
    overwrites). Defensive at every step — tool wiring must never abort
    a run, so each failure is logged and swallowed.
    """
    if config is None:
        return
    tools_cfg = config.tools
    if getattr(tools_cfg, "enable_builtins", True):
        try:
            from care.builtin_tools import register_builtin_tools

            register_builtin_tools(context, tools_cfg, config.sandbox)
        except Exception as exc:  # noqa: BLE001
            _log.warning("builtin tool registration failed: %s", exc)
    # Cached synthesised tools (generated in an earlier run) — register so
    # a tool created once is reused instead of regenerated, and is present
    # before execution so the chain never re-hits "not registered".
    try:
        from care.tool_synthesis import register_cached_tools

        register_cached_tools(context, config)
    except Exception as exc:  # noqa: BLE001
        _log.warning("cached tool registration failed: %s", exc)
    # The chat run-path historically never passed ``tools_path``, so the
    # user's ``~/.config/care/tools`` directory was silently ignored.
    # Wire it here so every execution path honours it uniformly.
    try:
        from care.tools import load_tools_into_context

        load_tools_into_context(context, config)
    except Exception as exc:  # noqa: BLE001
        _log.warning("user tool loading failed: %s", exc)


def _register_tools(context: Any, tools_path: str | None) -> None:
    """Forward to ``context.register_tools_from_path`` when the
    caller supplied a glob. Tilde + env-var expansion is done here
    so the conventional ``~/.config/care/tools/*.py`` value works
    without callers pre-expanding."""
    if not tools_path:
        return
    expanded = str(Path(tools_path).expanduser())
    if not hasattr(context, "register_tools_from_path"):
        # CARL's preflight contract (PREPARE.md §5.7) ships the
        # method on ReasoningContext. If a stripped-down replacement
        # is plugged in, surface the limitation.
        raise ExecutionError(
            "context object does not expose register_tools_from_path; "
            "upgrade mmar_carl or drop the tools_path argument."
        )
    context.register_tools_from_path(expanded)


__all__ = [
    "ExecutionError",
    "build_run_context",
    "execute_chain_async",
    "prime_from_saved_chain",
]
