"""Tests for ``care.preflight.validate_chain`` (TODO §4 P2).

Two coverage layers:

1. **Real CARL parse.** The installed `mmar-carl 0.2.0` ships
   `ReasoningChain.from_dict(..., use_typed_steps=True)`, so we
   build a valid chain dict and assert the result carries a real
   parsed chain — no mocks. Then we hand it malformed input and
   assert the Pydantic `ValidationError` lands on `parse_errors`
   as ``"field.path: message"`` lines.
2. **Duck-typed preflight.** The installed CARL doesn't yet ship
   `chain.preflight`, so we exercise the "preflight unavailable"
   branch for real. We also feed `validate_chain` a stub chain
   that **does** have a `.preflight` method to verify the
   `missing_*` plumbing works without depending on the dev branch.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from care.preflight import PreflightResult, validate_chain


def _valid_chain_dict() -> dict[str, Any]:
    """Minimal typed-step chain dict the installed CARL parses cleanly."""
    return {
        "task_description": "demo",
        "steps": [
            {
                "number": 1,
                "title": "first",
                "step_type": "llm",
                "aim": "say hi",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_default_result_is_failure(self):
        r = PreflightResult(parsed=False, parse_errors=("oops",))
        assert not r.is_valid
        assert not r.ok
        assert "chain failed to parse" in r.format_text()

    def test_ok_with_preflight_text(self):
        r = PreflightResult(
            parsed=True,
            required_tools=("web_search",),
        )
        assert r.is_valid
        assert r.ok
        assert "preflight: ok" in r.format_text()
        assert "1 tool" in r.format_text()

    def test_ok_singular_plural(self):
        r = PreflightResult(parsed=True, required_mcp_servers=("a", "b"))
        text = r.format_text()
        assert "2 mcp servers" in text
        assert "0 tools" in text

    def test_text_when_preflight_skipped(self):
        r = PreflightResult(parsed=True, preflight_available=False)
        text = r.format_text()
        assert "preflight skipped" in text

    def test_text_when_missing_listed(self):
        r = PreflightResult(
            parsed=True,
            missing_tools=("x",),
            missing_mcp_servers=("y", "z"),
        )
        text = r.format_text()
        assert "missing dependencies" in text
        assert "tools: x" in text
        assert "mcp servers: y, z" in text
        assert not r.ok
        # `is_valid` (parsed) is still True even when deps are missing —
        # the chain itself is well-formed.
        assert r.is_valid

    def test_frozen(self):
        r = PreflightResult(parsed=True)
        with pytest.raises(Exception):
            r.parsed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Real CARL parse
# ---------------------------------------------------------------------------


class TestRealCarlParse:
    def test_dict_payload_parses(self):
        r = validate_chain(_valid_chain_dict())
        assert r.is_valid
        assert r.chain is not None
        # CARE's chain attribute is the actual parsed ReasoningChain.
        assert hasattr(r.chain, "steps")
        assert len(r.chain.steps) == 1

    def test_json_string_payload_parses(self):
        r = validate_chain(json.dumps(_valid_chain_dict()))
        assert r.is_valid

    def test_bytes_payload_parses(self):
        r = validate_chain(json.dumps(_valid_chain_dict()).encode("utf-8"))
        assert r.is_valid

    def test_invalid_json_string_recorded(self):
        r = validate_chain("not json at all")
        assert not r.is_valid
        assert len(r.parse_errors) == 1
        assert "invalid JSON" in r.parse_errors[0]

    def test_invalid_utf8_bytes_recorded(self):
        r = validate_chain(b"\xff\xfe\xfd not utf8")
        assert not r.is_valid
        assert "invalid utf-8" in r.parse_errors[0]

    def test_missing_required_step_field_reports_error(self):
        # Steps require `number` + `title` + `step_type`. Omit them.
        # CARL's typed-step parser raises eagerly so we may see a
        # KeyError instead of a multi-field Pydantic ValidationError.
        # The contract is "every failure surfaces at least one
        # descriptive error", not a specific exception shape.
        broken = {"task_description": "d", "steps": [{"step_type": "llm"}]}
        r = validate_chain(broken)
        assert not r.is_valid
        assert len(r.parse_errors) >= 1
        joined = " | ".join(r.parse_errors)
        assert "number" in joined or "title" in joined

    def test_pydantic_validation_error_uses_field_path_format(self):
        # Force a real Pydantic ValidationError by using untyped
        # steps — the `StepDescription` model surfaces every
        # missing field at once with `loc=("steps", 0, "number")`.
        broken = {"task_description": "d", "steps": [{"step_type": "llm"}]}
        from care.preflight import validate_chain as vc
        r = vc(broken, use_typed_steps=False)
        assert not r.is_valid
        joined = " | ".join(r.parse_errors)
        # Pydantic v2 errors include the field name in the loc path.
        assert "number" in joined
        assert "title" in joined

    def test_unsupported_payload_type(self):
        r = validate_chain(42)
        assert not r.is_valid
        assert "expected dict" in r.parse_errors[0]

    def test_already_parsed_chain_passes_through(self):
        # First parse it, then re-validate the parsed object.
        first = validate_chain(_valid_chain_dict())
        assert first.is_valid
        second = validate_chain(first.chain)
        assert second.is_valid
        # Same object — not re-parsed.
        assert second.chain is first.chain


# ---------------------------------------------------------------------------
# Preflight unavailable (real installed CARL)
# ---------------------------------------------------------------------------


class TestPreflightUnavailable:
    def test_installed_carl_marks_preflight_unavailable(self):
        # mmar-carl 0.2.0 doesn't ship `.preflight` yet — the parse
        # still succeeds and the report tells callers explicitly.
        r = validate_chain(_valid_chain_dict())
        assert r.is_valid
        # Either the installed CARL has preflight or it doesn't —
        # both cases should leave `is_valid=True` and `ok=True`
        # (no deps required from this trivial chain).
        assert r.ok
        if not r.preflight_available:
            text = r.format_text()
            assert "preflight skipped" in text


# ---------------------------------------------------------------------------
# Stub-driven preflight (verifies the `missing_*` plumbing)
# ---------------------------------------------------------------------------


class _StubReport:
    required_tools = ["web_search", "calc"]
    required_mcp_servers = ["weather"]
    required_skills: list[str] = []
    missing_tools = ["calc"]
    missing_mcp_servers: list[str] = []
    missing_skills: list[str] = []


class _StubChain:
    """Quacks like a parsed ReasoningChain: has `.steps` + `.preflight`."""

    steps: list[Any] = []

    def __init__(self, report: Any = None, *, raise_exc: Exception | None = None):
        self._report = report or _StubReport()
        self._raise = raise_exc

    def preflight(self, context: Any) -> Any:
        if self._raise:
            raise self._raise
        return self._report


class TestStubPreflight:
    def test_required_and_missing_round_trip(self):
        chain = _StubChain()
        r = validate_chain(chain, context=object())
        assert r.is_valid
        assert r.preflight_available
        assert r.required_tools == ("web_search", "calc")
        assert r.required_mcp_servers == ("weather",)
        assert r.missing_tools == ("calc",)
        assert not r.ok  # one tool missing

    def test_preflight_called_with_context_arg(self):
        captured: list[Any] = []

        class _CapturingChain(_StubChain):
            def preflight(self, context: Any) -> Any:  # type: ignore[override]
                captured.append(context)
                return _StubReport()

        sentinel = object()
        validate_chain(_CapturingChain(), context=sentinel)
        assert captured == [sentinel]

    def test_preflight_exception_does_not_lose_parse(self):
        chain = _StubChain(raise_exc=RuntimeError("boom"))
        r = validate_chain(chain)
        assert r.is_valid
        # The chain still parsed; the exception is surfaced as a
        # parse_errors entry so the caller can show it.
        assert any("preflight raised" in e for e in r.parse_errors)
        assert r.preflight_available

    def test_chain_without_preflight_marks_unavailable(self):
        class _Bare:
            steps: list[Any] = []

        r = validate_chain(_Bare())
        assert r.is_valid
        assert r.preflight_available is False

    def test_format_text_includes_missing_breakdown(self):
        r = validate_chain(_StubChain(), context=object())
        text = r.format_text()
        assert "missing dependencies" in text
        assert "calc" in text


# ---------------------------------------------------------------------------
# Integration with real chain + stub context (intersection works)
# ---------------------------------------------------------------------------


class TestIntegrationWithContext:
    def test_real_chain_with_context_runs_preflight_when_available(self):
        # Build a real chain. If the installed CARL ships preflight,
        # we exercise it; otherwise the path is the "unavailable"
        # branch we already covered above.
        r = validate_chain(_valid_chain_dict(), context=None)
        assert r.is_valid
        # No tool steps → `required_tools` must be empty when
        # preflight is available; absent when it isn't. Either way
        # the chain has no missing deps.
        assert r.missing_tools == ()
        assert r.missing_mcp_servers == ()
        assert r.missing_skills == ()
        assert r.ok
