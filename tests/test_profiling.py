"""Tests for ``care.profiling`` (TODO §5 P1).

Coverage:

1. **Summary shape** — empty / frozen / `is_empty` / `step_count`.
2. **Dict input** — when CARL's `get_profiling_summary()` output
   (or a replayed dict) lands directly.
3. **`ReasoningResult` with helper** — when the source object
   exposes the method (newer CARL).
4. **`ReasoningResult` without helper (older CARL fallback)** —
   the projector walks `.step_results` + `.history`.
5. **`format_text`** — chain-level lines + per-step rows;
   skipped/failed badges.
6. **Byte formatting** — K/M/G suffixes via the rendered output.
7. **Edge cases** — non-dict step rows, stringy numbers,
   helper that raises (falls through to manual walk).
"""

from __future__ import annotations

import pytest

from care.profiling import (
    ProfilingSummary,
    StepProfile,
    project_profiling,
)


# ---------------------------------------------------------------------------
# Summary shape
# ---------------------------------------------------------------------------


class TestSummaryShape:
    def test_empty_defaults(self):
        s = ProfilingSummary()
        assert s.is_empty
        assert s.step_count == 0
        assert s.steps == ()
        assert s.total_execution_time_s == 0.0
        assert s.total_history_bytes == 0
        assert s.peak_memory_bytes == 0
        assert s.token_usage == {}
        assert s.format_text() == "no profiling data"

    def test_frozen(self):
        s = ProfilingSummary()
        with pytest.raises(Exception):
            s.peak_memory_bytes = 1  # type: ignore[misc]

    def test_step_profile_frozen(self):
        sp = StepProfile(
            step_number=1,
            step_title="t",
            step_type="llm",
            execution_time_s=0.1,
            history_bytes_added=10,
            memory_bytes_after=20,
            history_bytes_after=30,
            batch_index=0,
        )
        with pytest.raises(Exception):
            sp.step_number = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Dict input (replay path)
# ---------------------------------------------------------------------------


class TestDictInput:
    def test_projects_summary_dict(self):
        payload = {
            "steps": [
                {
                    "step_number": 1,
                    "step_title": "extract",
                    "step_type": "llm",
                    "execution_time_s": 0.12,
                    "history_bytes_added": 256,
                    "memory_bytes_after": 1024,
                    "history_bytes_after": 256,
                    "batch_index": 0,
                    "skipped": False,
                    "success": True,
                },
                {
                    "step_number": 2,
                    "step_title": "summarise",
                    "step_type": "llm",
                    "execution_time_s": 0.34,
                    "history_bytes_added": 128,
                    "memory_bytes_after": 2048,
                    "history_bytes_after": 384,
                    "batch_index": 1,
                    "skipped": False,
                    "success": True,
                },
            ],
            "total_execution_time_s": 0.46,
            "total_history_bytes": 384,
            "peak_memory_bytes": 2048,
            "token_usage": {"total_tokens": 320},
        }
        s = project_profiling(payload)
        assert s.step_count == 2
        assert s.total_execution_time_s == 0.46
        assert s.peak_memory_bytes == 2048
        assert s.token_usage == {"total_tokens": 320}
        # Per-step.
        assert s.steps[0].step_title == "extract"
        assert s.steps[1].step_type == "llm"
        assert s.steps[1].history_bytes_added == 128

    def test_empty_dict_yields_empty_summary(self):
        s = project_profiling({})
        assert s.is_empty

    def test_non_list_steps_dropped(self):
        s = project_profiling({"steps": "not a list", "peak_memory_bytes": 100})
        # Steps coerced to empty; other fields preserved.
        assert s.steps == ()
        assert s.peak_memory_bytes == 100

    def test_non_dict_step_rows_skipped(self):
        s = project_profiling(
            {
                "steps": [
                    {"step_number": 1, "step_title": "ok"},
                    "garbage",
                    {"step_number": 2, "step_title": "also-ok"},
                ]
            }
        )
        assert [r.step_number for r in s.steps] == [1, 2]

    def test_stringy_numbers_coerced(self):
        s = project_profiling(
            {
                "steps": [
                    {
                        "step_number": "3",
                        "execution_time_s": "1.5",
                        "history_bytes_added": "100",
                        "memory_bytes_after": "200",
                    }
                ]
            }
        )
        assert s.steps[0].step_number == 3
        assert s.steps[0].execution_time_s == 1.5
        assert s.steps[0].history_bytes_added == 100

    def test_unparseable_numbers_default(self):
        s = project_profiling(
            {
                "steps": [
                    {
                        "step_number": "nope",
                        "execution_time_s": "junk",
                    }
                ]
            }
        )
        assert s.steps[0].step_number == 0
        assert s.steps[0].execution_time_s == 0.0


# ---------------------------------------------------------------------------
# ReasoningResult-like with helper
# ---------------------------------------------------------------------------


class TestHelperPath:
    def test_helper_method_called_when_present(self):
        calls = []

        class _Result:
            def get_profiling_summary(self):
                calls.append("called")
                return {
                    "steps": [{"step_number": 1, "step_title": "x"}],
                    "total_execution_time_s": 0.5,
                }

        s = project_profiling(_Result())
        assert calls == ["called"]
        assert s.step_count == 1
        assert s.total_execution_time_s == 0.5

    def test_helper_raising_falls_back_to_manual_walk(self):
        class _Result:
            def get_profiling_summary(self):
                raise RuntimeError("helper bug")

            step_results = []
            history = []
            total_execution_time = 0.0
            token_usage = {}

        # Doesn't raise — falls through to the manual walk which
        # yields an empty summary.
        s = project_profiling(_Result())
        assert s.is_empty

    def test_helper_returning_non_dict_falls_through(self):
        class _Result:
            def get_profiling_summary(self):
                return "not a dict"

            step_results = []
            history = []
            total_execution_time = 0.0
            token_usage = {}

        s = project_profiling(_Result())
        assert s.is_empty


# ---------------------------------------------------------------------------
# Older-CARL manual fallback
# ---------------------------------------------------------------------------


class TestManualFallback:
    def test_walks_step_results_and_history(self):
        class _StepResult:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        result_obj = type(
            "_R",
            (),
            {
                "step_results": [
                    _StepResult(
                        step_number=1,
                        step_title="extract",
                        step_type="llm",
                        execution_time=0.1,
                        skipped=False,
                        success=True,
                        profiling={
                            "history_bytes_added": 50,
                            "memory_bytes_after": 100,
                            "history_bytes_after": 50,
                            "batch_index": 0,
                        },
                    ),
                    _StepResult(
                        step_number=2,
                        step_title="summarise",
                        step_type="llm",
                        execution_time=0.2,
                        skipped=False,
                        success=False,
                        profiling={
                            "history_bytes_added": 25,
                            "memory_bytes_after": 150,
                            "history_bytes_after": 75,
                            "batch_index": 1,
                        },
                    ),
                ],
                "history": ["abc", "defghij"],
                "total_execution_time": 0.3,
                "token_usage": {"total_tokens": 42},
            },
        )()
        s = project_profiling(result_obj)
        assert s.step_count == 2
        assert s.total_execution_time_s == 0.3
        # `total_history_bytes` = len("abc") + len("defghij") = 3+7
        assert s.total_history_bytes == 10
        # Peak across both steps.
        assert s.peak_memory_bytes == 150
        # Per-step.
        assert s.steps[0].history_bytes_added == 50
        assert s.steps[1].success is False
        assert s.token_usage == {"total_tokens": 42}

    def test_handles_enum_step_type(self):
        class _Enum:
            value = "tool"

        class _R:
            step_results = [
                type(
                    "_S",
                    (),
                    {
                        "step_number": 1,
                        "step_title": "t",
                        "step_type": _Enum(),
                        "execution_time": 0.0,
                        "profiling": {},
                        "skipped": False,
                        "success": True,
                    },
                )()
            ]
            history = []
            total_execution_time = 0.0
            token_usage = {}

        s = project_profiling(_R())
        assert s.steps[0].step_type == "tool"

    def test_missing_step_type_falls_back_to_unknown(self):
        class _R:
            step_results = [
                type(
                    "_S",
                    (),
                    {
                        "step_number": 1,
                        "step_title": "x",
                        "execution_time": 0.0,
                        "profiling": {},
                    },
                )()
            ]
            history = []
            total_execution_time = 0.0
            token_usage = {}

        s = project_profiling(_R())
        assert s.steps[0].step_type == "unknown"

    def test_none_history_treated_as_empty(self):
        class _R:
            step_results = []
            history = None
            total_execution_time = 0.0
            token_usage = {}

        s = project_profiling(_R())
        assert s.total_history_bytes == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_input_yields_empty(self):
        s = project_profiling(None)
        assert s.is_empty

    def test_unrecognised_input_yields_empty(self):
        # Anything without `.get_profiling_summary` AND without
        # `.step_results` falls through to the manual walk and
        # produces an empty summary.
        s = project_profiling(42)
        assert s.is_empty


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_per_step_rows_render(self):
        s = project_profiling(
            {
                "steps": [
                    {
                        "step_number": 1,
                        "step_title": "extract",
                        "step_type": "llm",
                        "execution_time_s": 0.1,
                        "history_bytes_added": 256,
                        "memory_bytes_after": 2048,
                        "batch_index": 0,
                    },
                    {
                        "step_number": 2,
                        "step_title": "summarise",
                        "step_type": "llm",
                        "execution_time_s": 0.2,
                        "history_bytes_added": 1024 * 5,
                        "memory_bytes_after": 1024 * 1024 * 2,
                        "batch_index": 1,
                    },
                ],
                "total_execution_time_s": 0.3,
                "peak_memory_bytes": 1024 * 1024 * 2,
                "total_history_bytes": 5376,
                "token_usage": {"total_tokens": 99},
            }
        )
        text = s.format_text()
        assert "steps: 2" in text
        assert "total time: 0.300s" in text
        assert "peak memory: 2.0M" in text
        assert "total history: 5.2K" in text
        assert "total tokens: 99" in text
        # Per-step rows present.
        assert "#1 extract (llm)" in text
        assert "batch 0" in text
        assert "#2 summarise (llm)" in text
        assert "batch 1" in text

    def test_skipped_step_badged(self):
        s = project_profiling(
            {
                "steps": [
                    {
                        "step_number": 1,
                        "step_title": "x",
                        "step_type": "llm",
                        "skipped": True,
                    }
                ]
            }
        )
        text = s.format_text()
        assert "[SKIPPED]" in text

    def test_failed_step_badged(self):
        s = project_profiling(
            {
                "steps": [
                    {
                        "step_number": 1,
                        "step_title": "x",
                        "step_type": "llm",
                        "success": False,
                    }
                ]
            }
        )
        text = s.format_text()
        assert "[FAILED]" in text

    def test_missing_batch_index_shows_dash(self):
        s = project_profiling(
            {
                "steps": [
                    {
                        "step_number": 1,
                        "step_title": "x",
                        "step_type": "llm",
                        "batch_index": None,
                    }
                ]
            }
        )
        text = s.format_text()
        assert "batch -" in text

    def test_byte_formatter_via_format_text_gigabytes(self):
        s = project_profiling(
            {
                "steps": [
                    {
                        "step_number": 1,
                        "step_title": "huge",
                        "step_type": "llm",
                        "memory_bytes_after": 5 * 1024 * 1024 * 1024,
                    }
                ],
                "peak_memory_bytes": 5 * 1024 * 1024 * 1024,
            }
        )
        text = s.format_text()
        assert "5.00G" in text


# ---------------------------------------------------------------------------
# Real CARL (best-effort)
# ---------------------------------------------------------------------------


def _carl_has_helper() -> bool:
    try:
        from mmar_carl.models.results import ReasoningResult

        return hasattr(ReasoningResult, "get_profiling_summary")
    except ImportError:
        return False


class TestRealCarl:
    @pytest.mark.skipif(
        not _carl_has_helper(),
        reason="installed mmar_carl doesn't ship get_profiling_summary yet",
    )
    def test_real_helper_runs(self):
        from mmar_carl.models.results import ReasoningResult

        # Minimal fields ReasoningResult needs (0.3.0 requires `success`).
        result = ReasoningResult(success=True, step_results=[], history=[])
        s = project_profiling(result)
        # No steps yet, but the projector returned a populated
        # summary object via the helper path.
        assert s.is_empty
