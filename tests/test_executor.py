"""Tests for ``care.runtime.executor`` (TODO §5 P1).

The executor lazily imports ``mmar_carl.models.context.ReasoningContext``
via :func:`_reasoning_context_cls`. Tests monkey-patch that single
seam with a local fake so the whole module is exercised without
needing CARL installed.

Coverage layers:
1. ``build_run_context`` — forwards query / files / language /
   extra_kwargs to the ReasoningContext constructor; attaches
   streamer + tools_path when supplied; wraps construction
   errors in ``ExecutionError``.
2. ``prime_from_saved_chain`` — forwards to
   ``ReasoningContext.from_chain_inputs`` with the right kwargs;
   wraps errors.
3. ``execute_chain_async`` — calls ``chain.execute_async(context)``
   and propagates the result; rejects objects without the method;
   wraps execution errors.
4. ``_register_tools`` — expands ``~`` correctly, no-ops on empty
   path, raises when the context lacks the method.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from care.config import CareConfig
from care.runtime import (
    ExecutionError,
    build_run_context,
    execute_chain_async,
    prime_from_saved_chain,
)
from care.runtime import executor as executor_mod


# ---------------------------------------------------------------------------
# Fakes that stand in for CARL's ReasoningContext + ReasoningChain.
# ---------------------------------------------------------------------------


class _FakeContext:
    """Captures every constructor / mutator call so tests can
    assert on them."""

    last_init: dict[str, Any] | None = None

    def __init__(self, **kwargs):
        _FakeContext.last_init = kwargs
        self.outer_context = kwargs.get("outer_context")
        self.api = kwargs.get("api")
        self.memory = kwargs.get("memory") or {}
        self.language = kwargs.get("language")
        self.system_prompt = kwargs.get("system_prompt")
        self.on_step_start = None
        self.on_step_complete = None
        self.on_chain_complete = None
        self.on_progress = None
        self.on_llm_chunk = None
        self.on_human_input_requested = None
        self.on_step_event = None
        self.tools_registered: list[str] = []

    @classmethod
    def from_chain_inputs(cls, chain, *, api, **kwargs):
        """Mimic CARL's classmethod — captures inputs so the test
        can assert."""
        ctx = cls(api=api, **kwargs)
        ctx.from_chain_called_with = {"chain": chain, "api": api, **kwargs}  # type: ignore[attr-defined]
        return ctx

    def register_tools_from_path(self, glob):
        self.tools_registered.append(glob)


class _FakeContextRaisingCtor:
    """Stand-in that raises on construction — tests the
    `ExecutionError` wrapping path."""

    def __init__(self, **kwargs):
        raise RuntimeError("simulated ctor failure")

    @classmethod
    def from_chain_inputs(cls, chain, **kwargs):
        raise RuntimeError("simulated from_chain_inputs failure")


class _FakeContextNoToolsMethod:
    """Doesn't expose ``register_tools_from_path``."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeChain:
    """Mimics CARL's chain.execute_async surface."""

    def __init__(self, result: Any = "ok", *, raises: Exception | None = None):
        self._result = result
        self._raises = raises
        self.called_with: Any = None

    async def execute_async(self, context):
        self.called_with = context
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.fixture
def patch_carl(monkeypatch):
    """Swap the lazy CARL import for our local fake."""
    monkeypatch.setattr(
        executor_mod, "_reasoning_context_cls", lambda: _FakeContext
    )
    yield


@pytest.fixture
def cfg(tmp_path):
    return CareConfig.load(path=tmp_path / "missing.toml", env={})


# ---------------------------------------------------------------------------
# build_run_context
# ---------------------------------------------------------------------------


class TestBuildRunContext:
    def test_minimal_inputs(self, patch_carl):
        ctx = build_run_context(query="weather report", api="api-obj")
        assert ctx.outer_context == "weather report"
        assert ctx.api == "api-obj"
        assert ctx.memory == {"input": {}}
        # No language unless config was supplied.
        assert ctx.language is None

    def test_files_pre_loaded_into_input_memory(self, patch_carl):
        ctx = build_run_context(
            query="q",
            api="api",
            files={"report.pdf": "binary contents...", "notes.md": "# notes"},
        )
        assert ctx.memory["input"]["report.pdf"] == "binary contents..."
        assert ctx.memory["input"]["notes.md"] == "# notes"

    def test_files_dict_copied_not_referenced(self, patch_carl):
        """Mutating the caller's dict after construction must NOT
        affect the context's memory (otherwise long-lived dicts
        could leak into the chain's memory state)."""
        files = {"a": "1"}
        ctx = build_run_context(query="q", api="api", files=files)
        files["a"] = "2"
        assert ctx.memory["input"]["a"] == "1"

    def test_language_from_care_config(self, patch_carl, cfg):
        ctx = build_run_context(query="q", api="api", config=cfg)
        assert ctx.language == cfg.defaults.language

    def test_grounding_system_prompt_default(self, patch_carl):
        # Without an explicit override, every run is grounded so the
        # synthesis step trusts tool results over its training knowledge.
        ctx = build_run_context(query="q", api="api")
        assert ctx.system_prompt == executor_mod._GROUNDING_SYSTEM_PROMPT
        assert "tool results" in ctx.system_prompt.lower()

    def test_extra_kwargs_forwarded(self, patch_carl):
        # An explicit system_prompt must win over the grounding default.
        ctx = build_run_context(
            query="q",
            api="api",
            extra_kwargs={"system_prompt": "you are helpful"},
        )
        assert ctx.system_prompt == "you are helpful"

    def test_explicit_extra_kwarg_overrides_config_language(
        self, patch_carl, cfg
    ):
        ctx = build_run_context(
            query="q",
            api="api",
            config=cfg,
            extra_kwargs={"language": "ru"},
        )
        assert ctx.language == "ru"

    def test_streamer_attach_called(self, patch_carl):
        attached: list = []

        class Streamer:
            def attach(self, ctx):
                attached.append(ctx)

        streamer = Streamer()
        ctx = build_run_context(query="q", api="api", streamer=streamer)
        assert attached == [ctx]

    def test_streamer_none_is_noop(self, patch_carl):
        # Just ensure no exception.
        build_run_context(query="q", api="api", streamer=None)

    def test_tools_path_expanded_and_registered(self, patch_carl):
        ctx = build_run_context(
            query="q", api="api", tools_path="~/.config/care/tools/*.py"
        )
        assert len(ctx.tools_registered) == 1
        # Tilde expanded to user home.
        registered = ctx.tools_registered[0]
        assert registered.startswith(os.path.expanduser("~"))
        assert registered.endswith("tools/*.py")

    def test_tools_path_none_is_noop(self, patch_carl):
        ctx = build_run_context(query="q", api="api", tools_path=None)
        assert ctx.tools_registered == []

    def test_construction_failure_wrapped_in_execution_error(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            executor_mod,
            "_reasoning_context_cls",
            lambda: _FakeContextRaisingCtor,
        )
        with pytest.raises(ExecutionError, match="failed to construct"):
            build_run_context(query="q", api="api")


# ---------------------------------------------------------------------------
# prime_from_saved_chain
# ---------------------------------------------------------------------------


class TestPrimeFromSavedChain:
    def test_forwards_to_from_chain_inputs(self, patch_carl):
        chain = object()
        ctx = prime_from_saved_chain(chain, api="api")
        assert ctx.from_chain_called_with["chain"] is chain
        assert ctx.from_chain_called_with["api"] == "api"
        # Defaults forwarded.
        assert ctx.from_chain_called_with["outer_context"] is None
        assert ctx.from_chain_called_with["files"] is None
        assert ctx.from_chain_called_with["load_files_from_metadata"] is True

    def test_overrides_forwarded(self, patch_carl):
        ctx = prime_from_saved_chain(
            object(),
            api="api",
            outer_context="override task",
            files={"x.txt": "fresh"},
            load_files_from_metadata=False,
            extra_kwargs={"system_prompt": "be careful"},
        )
        captured = ctx.from_chain_called_with
        assert captured["outer_context"] == "override task"
        assert captured["files"] == {"x.txt": "fresh"}
        assert captured["load_files_from_metadata"] is False
        assert captured["system_prompt"] == "be careful"

    def test_streamer_attach_called(self, patch_carl):
        called: list = []

        class S:
            def attach(self, ctx):
                called.append(ctx)

        s = S()
        ctx = prime_from_saved_chain(object(), api="api", streamer=s)
        assert called == [ctx]

    def test_failure_wrapped(self, monkeypatch):
        monkeypatch.setattr(
            executor_mod,
            "_reasoning_context_cls",
            lambda: _FakeContextRaisingCtor,
        )
        with pytest.raises(ExecutionError, match="failed to prime"):
            prime_from_saved_chain(object(), api="api")


# ---------------------------------------------------------------------------
# execute_chain_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_chain_async_forwards_context_and_returns_result(
    patch_carl,
):
    chain = _FakeChain(result={"success": True})
    context = build_run_context(query="q", api="api")
    result = await execute_chain_async(chain, context)
    assert result == {"success": True}
    assert chain.called_with is context


@pytest.mark.asyncio
async def test_execute_chain_async_rejects_chain_without_method():
    class Bad:
        pass

    with pytest.raises(ExecutionError, match="missing ``execute_async``"):
        await execute_chain_async(Bad(), context=object())


@pytest.mark.asyncio
async def test_execute_chain_async_wraps_chain_failure(patch_carl):
    boom = RuntimeError("step 1 timed out")
    chain = _FakeChain(raises=boom)
    context = build_run_context(query="q", api="api")
    with pytest.raises(ExecutionError, match="chain execution failed") as exc_info:
        await execute_chain_async(chain, context)
    # Original exception preserved on __cause__.
    assert exc_info.value.__cause__ is boom


# ---------------------------------------------------------------------------
# _register_tools edge cases
# ---------------------------------------------------------------------------


class TestRegisterTools:
    def test_no_path_noop(self, patch_carl):
        # No exception when tools_path is None.
        build_run_context(query="q", api="api")

    def test_context_missing_method_raises(self, monkeypatch):
        monkeypatch.setattr(
            executor_mod,
            "_reasoning_context_cls",
            lambda: _FakeContextNoToolsMethod,
        )
        with pytest.raises(ExecutionError, match="register_tools_from_path"):
            build_run_context(query="q", api="api", tools_path="/tmp/*.py")


# ---------------------------------------------------------------------------
# Lazy import surface
# ---------------------------------------------------------------------------


def test_reasoning_context_cls_raises_execution_error_when_carl_missing(
    monkeypatch,
):
    """If ``import mmar_carl`` fails, the helper must raise
    `ExecutionError` (not `ImportError`) so callers handle one
    error class."""
    import sys

    # Force the import to fail by injecting a `None` module for
    # ``mmar_carl.models.context``.
    monkeypatch.setitem(sys.modules, "mmar_carl.models.context", None)
    with pytest.raises(ExecutionError, match="mmar_carl is not installed"):
        executor_mod._reasoning_context_cls()


# ---------------------------------------------------------------------------
# _normalize_tool_input_literals — quote bare literals for CARL's resolver
# ---------------------------------------------------------------------------


def _chain_with_inputs(mapping):
    """Tiny stand-in exposing the
    ``chain.steps[].step_config.input_mapping`` shape the normalizer walks."""
    from types import SimpleNamespace

    return SimpleNamespace(
        steps=[SimpleNamespace(step_config=SimpleNamespace(input_mapping=mapping))]
    )


class TestNormalizeToolInputLiterals:
    def _norm(self, mapping):
        chain = _chain_with_inputs(mapping)
        executor_mod._normalize_tool_input_literals(chain)
        return chain.steps[0].step_config.input_mapping

    def test_bare_literal_is_quoted(self):
        # The exact failure: a bare query string CARL resolves to None.
        out = self._norm({"query": "UEFA Champions League 2026 winner"})
        assert out["query"] == '"UEFA Champions League 2026 winner"'

    def test_dynamic_refs_left_untouched(self):
        out = self._norm({"query": "$outer_context", "n": "$history[-1]"})
        assert out == {"query": "$outer_context", "n": "$history[-1]"}

    def test_already_quoted_left_untouched(self):
        out = self._norm({"a": '"hi"', "b": "'yo'"})
        assert out == {"a": '"hi"', "b": "'yo'"}

    def test_non_string_and_empty_left_untouched(self):
        out = self._norm({"n": 5, "blank": "", "flag": True})
        assert out == {"n": 5, "blank": "", "flag": True}

    def test_idempotent(self):
        once = self._norm({"query": "live scores today"})
        twice = self._norm(dict(once))
        assert once == twice == {"query": '"live scores today"'}

    def test_missing_or_non_dict_mapping_is_safe(self):
        from types import SimpleNamespace

        # input_mapping=None and a None step_config — neither blows up.
        chain = SimpleNamespace(
            steps=[
                SimpleNamespace(step_config=SimpleNamespace(input_mapping=None)),
                SimpleNamespace(step_config=None),
            ]
        )
        executor_mod._normalize_tool_input_literals(chain)  # no raise

    def test_no_steps_is_safe(self):
        from types import SimpleNamespace

        executor_mod._normalize_tool_input_literals(SimpleNamespace(steps=None))


@pytest.mark.asyncio
async def test_execute_chain_async_normalizes_literals_before_run():
    """The chain sees quoted literals by the time it executes — proving
    the normalizer runs inside the execution chokepoint, not just in tests."""
    from types import SimpleNamespace

    captured: dict[str, Any] = {}
    mapping = {"query": "who won the cup"}

    class _ChainWithSteps:
        def __init__(self):
            self.steps = [
                SimpleNamespace(
                    step_config=SimpleNamespace(input_mapping=mapping)
                )
            ]

        async def execute_async(self, context):
            captured["query"] = self.steps[0].step_config.input_mapping["query"]
            return "ok"

    await execute_chain_async(_ChainWithSteps(), context=object())
    assert captured["query"] == '"who won the cup"'


def _carl_supports_human_input() -> bool:
    """The ``on_human_input_requested`` protocol only exists on the
    agent-features CARL; the pinned ``mmar_carl 0.2.0`` lacks the field."""
    try:
        from mmar_carl import ReasoningContext

        ReasoningContext(
            outer_context="x", api=object(),
        ).on_human_input_requested = lambda p, f: None
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    not _carl_supports_human_input(),
    reason="installed CARL lacks on_human_input_requested "
    "(needs agent-features CARL; main pins mmar_carl 0.2.0)",
)
class TestHumanInput:
    """A ``human_input`` step PAUSES; a CARE-supplied provider RESUMES it with
    the answer, which then flows into the chain (history -> $history[-1])."""

    _STEP = {
        "goal": "ask the user",
        "steps": [{
            "number": 1, "title": "Ask name", "step_type": "human_input",
            "step_config": {
                "prompt": "What is your name?",
                "output_key": "name", "fallback_value": "(none)",
            },
        }],
    }

    @staticmethod
    def _chain():
        from mmar_carl import ReasoningChain

        return ReasoningChain.from_dict(TestHumanInput._STEP, use_typed_steps=True)

    @pytest.mark.asyncio
    async def test_provider_resumes_and_answer_flows(self):
        ctx = build_run_context(
            query="ask", api=object(), config=CareConfig(),
            human_input_provider=lambda prompt: "Alice",
        )
        result = await execute_chain_async(self._chain(), ctx)
        assert result.success is True
        assert result.step_results[0].result == "Alice"  # resumed with the answer
        # the answer entered the history a downstream $history[-1] step reads
        assert any("Alice" in str(h) for h in (ctx.history or []))

    @pytest.mark.asyncio
    async def test_async_provider_resolves(self):
        async def provider(prompt):
            return "Bob"

        ctx = build_run_context(
            query="ask", api=object(), config=CareConfig(),
            human_input_provider=provider,
        )
        result = await execute_chain_async(self._chain(), ctx)
        assert result.step_results[0].result == "Bob"

    @pytest.mark.asyncio
    async def test_no_provider_uses_fallback(self):
        ctx = build_run_context(query="ask", api=object(), config=CareConfig())
        result = await execute_chain_async(self._chain(), ctx)
        assert result.step_results[0].result == "(none)"  # non-interactive

    def test_build_run_context_wires_handler_only_when_given(self):
        wired = build_run_context(
            query="x", api=object(), config=CareConfig(),
            human_input_provider=lambda p: "y",
        )
        assert callable(wired.on_human_input_requested)
        plain = build_run_context(query="x", api=object(), config=CareConfig())
        assert getattr(plain, "on_human_input_requested", None) is None


class TestLtmAndUserContext:
    """LTM + standing user-context wiring on ``build_run_context``."""

    def test_user_context_prepended_to_system_prompt(self):
        ctx = build_run_context(
            query="x", api=object(), config=CareConfig(),
            user_context="## What I remember\n- role: PM",
        )
        # the always-inject digest rides in the grounding system prompt
        assert "role: PM" in ctx.system_prompt

    def test_long_term_memory_and_session_attached(self):
        from mmar_carl import InMemoryLTM

        ltm = InMemoryLTM()
        ctx = build_run_context(
            query="x", api=object(), config=CareConfig(),
            long_term_memory=ltm, session_id="s1",
        )
        assert ctx.long_term_memory is ltm
        assert ctx.session_id == "s1"

    def test_defaults_no_ltm_no_injection(self):
        ctx = build_run_context(query="x", api=object(), config=CareConfig())
        assert getattr(ctx, "long_term_memory", None) is None
        assert "What I remember" not in (getattr(ctx, "system_prompt", "") or "")
