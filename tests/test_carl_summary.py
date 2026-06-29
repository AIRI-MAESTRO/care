"""Tests for ``care.carl_summary.summarise_carl_result``.

The function exists to project whatever shape CARL hands back
(real ``ReasoningResult``, dict, ``SimpleNamespace`` stub, …)
into the single assistant-line string ChatScreen renders.
"""

from __future__ import annotations

from types import SimpleNamespace

from care.carl_summary import summarise_carl_result


class TestAttributeChain:
    def test_final_output_wins_over_others(self):
        result = SimpleNamespace(
            final_output="from final_output",
            output="from output",
            answer="from answer",
        )
        assert summarise_carl_result(result) == "from final_output"

    def test_final_answer_picked_when_final_output_missing(self):
        result = SimpleNamespace(
            final_answer="from final_answer", answer="from answer",
        )
        assert summarise_carl_result(result) == "from final_answer"

    def test_output_picked_when_finals_missing(self):
        assert (
            summarise_carl_result(SimpleNamespace(output="just output"))
            == "just output"
        )

    def test_answer_picked_when_outputs_missing(self):
        assert (
            summarise_carl_result(SimpleNamespace(answer="just answer"))
            == "just answer"
        )

    def test_summary_picked_last(self):
        assert (
            summarise_carl_result(SimpleNamespace(summary="just summary"))
            == "just summary"
        )

    def test_text_field_also_recognised(self):
        assert (
            summarise_carl_result(SimpleNamespace(text="raw text"))
            == "raw text"
        )

    def test_empty_string_skipped(self):
        result = SimpleNamespace(output="", answer="non-empty")
        assert summarise_carl_result(result) == "non-empty"

    def test_whitespace_only_skipped(self):
        result = SimpleNamespace(output="   ", answer="real text")
        assert summarise_carl_result(result) == "real text"


class TestDictChain:
    def test_dict_with_output(self):
        assert (
            summarise_carl_result({"output": "from dict"}) == "from dict"
        )

    def test_dict_priority_matches_attr_priority(self):
        assert (
            summarise_carl_result(
                {"final_output": "first", "answer": "second"},
            )
            == "first"
        )


class TestOutputsFallback:
    def test_last_outputs_value_used(self):
        result = SimpleNamespace(
            outputs={"step1": "intermediate", "step2": "final"},
        )
        assert summarise_carl_result(result) == "final"

    def test_outputs_with_wrapped_value(self):
        result = SimpleNamespace(
            outputs={
                "step1": "intermediate",
                "step2": SimpleNamespace(text="wrapped final"),
            },
        )
        assert summarise_carl_result(result) == "wrapped final"

    def test_dict_outputs_falls_through_to_wrapped(self):
        result = {
            "outputs": {
                "s1": {"output": "nested final"},
            },
        }
        assert summarise_carl_result(result) == "nested final"


class TestFallbacks:
    def test_none_returns_fallback(self):
        assert summarise_carl_result(None) == (
            "Chain executed (no textual output)."
        )

    def test_object_with_no_known_fields_strs(self):
        result = SimpleNamespace()  # no known fields, no outputs
        # str() of SimpleNamespace is "namespace()" — non-empty,
        # so the function returns it rather than the fallback.
        assert summarise_carl_result(result)  # truthy

    def test_object_with_str_yielding_empty_uses_fallback(self):
        class _Blank:
            def __str__(self):
                return "   "

        assert summarise_carl_result(_Blank()) == (
            "Chain executed (no textual output)."
        )

    def test_non_string_field_value_skipped(self):
        # `.output` is an int, not a string — should keep searching.
        result = SimpleNamespace(output=42, answer="string answer")
        assert summarise_carl_result(result) == "string answer"


class TestStepResultsFallback:
    """Current CARL (mmar_carl >= 0.2) returns a flat
    ``step_results`` list with the user's answer in the terminal
    step's ``result`` field — no ``outputs`` mapping. The
    summariser walks ``step_results`` from the end before
    falling through to ``str(result)``."""

    @staticmethod
    def _step(number, title, *, ok=True, result="", result_data=None):
        return SimpleNamespace(
            step_number=number,
            step_title=title,
            success=ok,
            result=result,
            result_data=result_data,
        )

    def test_terminal_step_result_wins(self):
        result = SimpleNamespace(
            success=True,
            step_results=[
                self._step(1, "extract", result='{"sections": []}'),
                self._step(
                    2, "summarise",
                    result="The team documentation covers six areas.",
                ),
            ],
        )
        assert (
            summarise_carl_result(result)
            == "The team documentation covers six areas."
        )

    def test_skips_failed_terminal_step(self):
        """A failing trailing step gets skipped so the user sees
        the last successful payload."""
        result = SimpleNamespace(
            success=False,
            step_results=[
                self._step(1, "ok", result="good answer"),
                self._step(2, "broken", ok=False, result=""),
            ],
        )
        assert summarise_carl_result(result) == "good answer"

    def test_structured_output_data_fallback(self):
        """STRUCTURED_OUTPUT steps with an empty `result` but
        populated `result_data` surface a JSON dump rather
        than the raw repr."""
        result = SimpleNamespace(
            success=True,
            step_results=[
                self._step(
                    1, "extract",
                    result="",
                    result_data={"sections": ["a", "b"]},
                ),
            ],
        )
        rendered = summarise_carl_result(result)
        assert "sections" in rendered
        assert '"a"' in rendered

    def test_dict_shape_also_supported(self):
        """Tests that pass a dict (no attribute access available)
        still surface the terminal step result."""
        result = {
            "success": True,
            "step_results": [
                {
                    "step_number": 1,
                    "step_title": "summarise",
                    "success": True,
                    "result": "summary text",
                    "result_data": None,
                },
            ],
        }
        assert summarise_carl_result(result) == "summary text"


class TestReExport:
    def test_re_exported_from_module(self):
        from care.carl_summary import __all__

        assert "summarise_carl_result" in __all__
