"""Tests for ``care.stage_regeneration`` (TODO §4 P1).

Coverage:

1. **Stage dispatch** — every supported stage calls the right
   ``MAGEGenerator`` method with the documented positional +
   keyword arguments.
2. **Async / sync result handling** — async methods are
   awaited; plain returns pass through unchanged.
3. **Input validation** — missing required keys raise before
   the generator is touched.
4. **Missing-method path** — older MAGE installs without
   per-stage entrypoints get a friendly error.
5. **Error wrapping** — downstream exceptions become
   ``StageRegenerationError``; `StageRegenerationError` raised
   inside the method propagates unchanged.
6. **`supported_stages()`** — stable sorted list.
"""

from __future__ import annotations

import asyncio

import pytest

from care.stage_regeneration import (
    StageArtifact,
    StageRegenerationError,
    regenerate_stage,
    supported_stages,
)


class _GenSpy:
    """Records every per-stage method call."""

    def __init__(
        self,
        *,
        async_methods: bool = True,
        return_value: object = "ok",
        raise_on_call: Exception | None = None,
    ):
        self.calls: list[dict[str, object]] = []
        self._async = async_methods
        self._ret = return_value
        self._raise = raise_on_call

    def _record(self, name: str, args: tuple, kwargs: dict) -> object:
        self.calls.append({"method": name, "args": args, "kwargs": kwargs})
        if self._raise is not None:
            raise self._raise

        if self._async:
            async def _co():
                return self._ret

            return _co()
        return self._ret

    # MAGE per-stage methods.
    def analyze_domain(self, *args, **kwargs):
        return self._record("analyze_domain", args, kwargs)

    def plan_steps(self, *args, **kwargs):
        return self._record("plan_steps", args, kwargs)

    def build_dag(self, *args, **kwargs):
        return self._record("build_dag", args, kwargs)

    def describe_steps(self, *args, **kwargs):
        return self._record("describe_steps", args, kwargs)

    def critique_steps(self, *args, **kwargs):
        return self._record("critique_steps", args, kwargs)

    def verify_chain(self, *args, **kwargs):
        return self._record("verify_chain", args, kwargs)

    def refine(self, *args, **kwargs):
        return self._record("refine", args, kwargs)


# ---------------------------------------------------------------------------
# Stage dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_domain_calls_analyze_domain(self):
        gen = _GenSpy()
        result = asyncio.run(
            regenerate_stage(gen, "domain", {"query": "weather"})
        )
        assert result.stage == "domain"
        assert result.artifact == "ok"
        assert gen.calls == [
            {"method": "analyze_domain", "args": ("weather",), "kwargs": {}}
        ]

    def test_plan_calls_plan_steps_with_required_positional(self):
        gen = _GenSpy()
        asyncio.run(
            regenerate_stage(
                gen,
                "plan",
                {"query": "q", "domain_analysis": {"d": 1}},
            )
        )
        assert gen.calls[0] == {
            "method": "plan_steps",
            "args": ("q", {"d": 1}),
            "kwargs": {},
        }

    def test_plan_optional_inputs_forwarded_as_kwargs(self):
        gen = _GenSpy()
        asyncio.run(
            regenerate_stage(
                gen,
                "plan",
                {
                    "query": "q",
                    "domain_analysis": {"d": 1},
                    "memory_digest": "mem",
                    "capability_context": "caps",
                    "allowed_step_types": ["llm"],
                },
            )
        )
        call = gen.calls[0]
        assert call["args"] == ("q", {"d": 1})
        assert call["kwargs"] == {
            "memory_digest": "mem",
            "capability_context": "caps",
            "allowed_step_types": ["llm"],
        }

    def test_dag_calls_build_dag(self):
        gen = _GenSpy()
        asyncio.run(regenerate_stage(gen, "dag", {"plan": {"steps": []}}))
        assert gen.calls[0] == {
            "method": "build_dag",
            "args": ({"steps": []},),
            "kwargs": {},
        }

    def test_describe_calls_describe_steps(self):
        gen = _GenSpy()
        asyncio.run(
            regenerate_stage(
                gen,
                "describe",
                {"query": "q", "dag": {"nodes": []}, "allowed_step_types": ["llm"]},
            )
        )
        assert gen.calls[0] == {
            "method": "describe_steps",
            "args": ("q", {"nodes": []}),
            "kwargs": {"allowed_step_types": ["llm"]},
        }

    def test_critique_calls_critique_steps(self):
        gen = _GenSpy()
        asyncio.run(
            regenerate_stage(
                gen,
                "critique",
                {
                    "query": "q",
                    "steps": [{"number": 1}],
                    "domain_analysis": {"d": 1},
                    "threshold": 0.5,
                },
            )
        )
        assert gen.calls[0] == {
            "method": "critique_steps",
            "args": ("q", [{"number": 1}], {"d": 1}),
            "kwargs": {"threshold": 0.5},
        }

    def test_verify_calls_verify_chain(self):
        gen = _GenSpy()
        asyncio.run(
            regenerate_stage(
                gen,
                "verify",
                {
                    "query": "q",
                    "chain_dict": {"steps": []},
                    "domain_analysis": {"d": 1},
                },
            )
        )
        assert gen.calls[0]["method"] == "verify_chain"

    def test_refine_calls_refine(self):
        gen = _GenSpy()
        asyncio.run(
            regenerate_stage(
                gen,
                "refine",
                {
                    "query": "q",
                    "chain_dict": {"steps": []},
                    "domain_analysis": {"d": 1},
                },
            )
        )
        assert gen.calls[0]["method"] == "refine"


# ---------------------------------------------------------------------------
# Async + sync result handling
# ---------------------------------------------------------------------------


class TestAsyncSync:
    def test_async_method_is_awaited(self):
        gen = _GenSpy(async_methods=True, return_value="async-ret")
        result = asyncio.run(regenerate_stage(gen, "domain", {"query": "x"}))
        assert result.artifact == "async-ret"

    def test_sync_method_returns_directly(self):
        gen = _GenSpy(async_methods=False, return_value="sync-ret")
        result = asyncio.run(regenerate_stage(gen, "domain", {"query": "x"}))
        assert result.artifact == "sync-ret"

    def test_returns_stage_artifact_object(self):
        gen = _GenSpy()
        result = asyncio.run(regenerate_stage(gen, "domain", {"query": "x"}))
        assert isinstance(result, StageArtifact)
        with pytest.raises(Exception):
            # Frozen.
            result.stage = "plan"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_unknown_stage_raises(self):
        gen = _GenSpy()
        with pytest.raises(StageRegenerationError, match="unknown stage"):
            asyncio.run(regenerate_stage(gen, "bogus", {}))  # type: ignore[arg-type]

    def test_missing_required_input_raises(self):
        gen = _GenSpy()
        with pytest.raises(
            StageRegenerationError, match="requires inputs.*'domain_analysis'"
        ):
            # `plan` needs query + domain_analysis.
            asyncio.run(regenerate_stage(gen, "plan", {"query": "q"}))
        # Generator never touched when input validation fails.
        assert gen.calls == []

    def test_empty_inputs_dict_raises(self):
        gen = _GenSpy()
        with pytest.raises(StageRegenerationError, match="requires inputs"):
            asyncio.run(regenerate_stage(gen, "domain", {}))
        assert gen.calls == []


# ---------------------------------------------------------------------------
# Missing-method path
# ---------------------------------------------------------------------------


class TestMissingMethod:
    def test_generator_without_per_stage_method_raises(self):
        class _OldMage:
            # No `analyze_domain` method — simulates an older
            # mmar-mage install before §3.7 shipped.
            pass

        with pytest.raises(
            StageRegenerationError, match="upgrade mmar_mage"
        ):
            asyncio.run(
                regenerate_stage(_OldMage(), "domain", {"query": "x"})
            )

    def test_non_callable_attribute_raises(self):
        class _Mage:
            analyze_domain = "not callable"

        with pytest.raises(StageRegenerationError, match="upgrade mmar_mage"):
            asyncio.run(
                regenerate_stage(_Mage(), "domain", {"query": "x"})
            )


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


class TestErrorWrapping:
    def test_sync_exception_wraps(self):
        gen = _GenSpy(
            async_methods=False, raise_on_call=RuntimeError("LLM down")
        )
        with pytest.raises(
            StageRegenerationError, match="'domain' failed.*LLM down"
        ):
            asyncio.run(regenerate_stage(gen, "domain", {"query": "x"}))

    def test_async_exception_wraps(self):
        """When the coroutine itself raises, the dispatcher
        still surfaces the error as `StageRegenerationError`."""

        class _RaisingGen:
            def analyze_domain(self, query):
                async def _co():
                    raise RuntimeError("inside coroutine")

                return _co()

        with pytest.raises(
            StageRegenerationError, match="'domain' failed.*inside coroutine"
        ):
            asyncio.run(
                regenerate_stage(_RaisingGen(), "domain", {"query": "x"})
            )

    def test_stage_regeneration_error_propagates_unwrapped(self):
        """A `StageRegenerationError` raised by the method
        shouldn't get double-wrapped — callers handle one
        exception class with the original message intact."""

        class _CustomGen:
            def analyze_domain(self, query):
                async def _co():
                    raise StageRegenerationError("descriptive cause")

                return _co()

        with pytest.raises(
            StageRegenerationError, match="^descriptive cause$"
        ):
            asyncio.run(
                regenerate_stage(_CustomGen(), "domain", {"query": "x"})
            )


# ---------------------------------------------------------------------------
# supported_stages()
# ---------------------------------------------------------------------------


class TestSupportedStages:
    def test_returns_all_seven_stages(self):
        names = supported_stages()
        assert set(names) == {
            "critique",
            "dag",
            "describe",
            "domain",
            "plan",
            "refine",
            "verify",
        }

    def test_is_sorted(self):
        names = supported_stages()
        assert names == sorted(names)
