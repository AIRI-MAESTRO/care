"""Tests for ``care.replay`` (TODO §5 P2).

Six coverage layers:

1. **Input coercion** — None / dict / JSON string /
   ReasoningResult-like / RunRecord-like all funnel into the
   same `ReplaySession` shape.
2. **Step projection** — every documented step field round-
   trips; long results truncate; non-string results coerce.
3. **Navigation** — `next` / `previous` / `seek` cursor moves
   with clamping; `at_start` / `at_end` predicates.
4. **Empty session** — `is_empty=True`, current=`None`,
   navigation is safe.
5. **JSON load failure** — wraps in `ReplayError`.
6. **`format_text`** — header bits + step block + skipped /
   failed badges.
"""

from __future__ import annotations

import json

import pytest

from care.replay import (
    ReplayError,
    ReplaySession,
    ReplayStep,
    load_replay,
)


def _step_dict(**overrides) -> dict:
    base = {
        "step_number": 1,
        "step_title": "extract",
        "step_type": "llm",
        "result": "Forecast: 21C, partly cloudy.",
        "result_data": None,
        "success": True,
        "skipped": False,
        "error_message": None,
        "execution_time": 0.12,
        "updated_history": ["hello"],
        "token_usage": {"total": 50},
        "model": "claude-opus",
    }
    base.update(overrides)
    return base


def _result_dict(steps: list[dict] | None = None, **overrides) -> dict:
    base = {
        "step_results": steps if steps is not None else [_step_dict()],
        "total_execution_time": 0.34,
        "token_usage": {"total": 50},
        "final_answer": "21C",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


class TestInputCoercion:
    def test_none_yields_empty_session(self):
        s = load_replay(None)
        assert isinstance(s, ReplaySession)
        assert s.is_empty
        assert s.current() is None

    def test_empty_dict_without_step_results_is_empty(self):
        s = load_replay({"hello": "world"})
        assert s.is_empty

    def test_reasoning_result_dict(self):
        s = load_replay(_result_dict())
        assert not s.is_empty
        assert s.step_count == 1
        assert s.total_execution_time_s == 0.34
        assert s.final_answer == "21C"

    def test_run_record_dict_with_chain_metadata(self):
        record = {
            "chain_id": "ent-42",
            "chain_title": "Weather agent",
            "result": _result_dict(),
        }
        s = load_replay(record)
        assert s.chain_id == "ent-42"
        assert s.chain_title == "Weather agent"
        assert s.step_count == 1

    def test_json_string(self):
        s = load_replay(json.dumps(_result_dict()))
        assert s.step_count == 1
        assert s.current().step_title == "extract"

    def test_invalid_json_raises_replay_error(self):
        with pytest.raises(ReplayError, match="failed to parse"):
            load_replay("{ not json")

    def test_reasoning_result_object(self):
        class _SR:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class _R:
            step_results = [
                _SR(
                    step_number=1,
                    step_title="t",
                    step_type="llm",
                    result="payload",
                    success=True,
                    skipped=False,
                    execution_time=0.05,
                    updated_history=["a", "b"],
                    token_usage={},
                ),
            ]
            total_execution_time = 0.05
            token_usage = {}
            final_answer = "done"

        s = load_replay(_R())
        assert s.step_count == 1
        assert s.current().step_title == "t"
        assert s.final_answer == "done"

    def test_run_record_object(self):
        class _SR:
            step_number = 1
            step_title = "t"
            step_type = "llm"
            result = "hi"
            success = True
            skipped = False
            execution_time = 0.0
            updated_history = []
            token_usage = {}

        class _Result:
            step_results = [_SR()]
            total_execution_time = 0.0
            token_usage = {}
            final_answer = ""

        class _Record:
            chain_id = "ent-1"
            chain_title = "Test chain"
            result = _Result()

        s = load_replay(_Record())
        assert s.chain_id == "ent-1"
        assert s.chain_title == "Test chain"
        assert s.step_count == 1


# ---------------------------------------------------------------------------
# Step projection
# ---------------------------------------------------------------------------


class TestStepProjection:
    def test_all_fields_round_trip(self):
        s = load_replay(_result_dict([_step_dict(
            history_bytes_added=100,  # extra fields ignored
            step_type="tool",
            model="gpt-4",
        )]))
        step = s.current()
        assert step.step_number == 1
        assert step.step_title == "extract"
        assert step.step_type == "tool"
        assert step.success is True
        assert step.skipped is False
        assert step.execution_time_s == 0.12
        assert step.history_snapshot == ("hello",)
        assert step.token_usage == {"total": 50}
        assert step.model == "gpt-4"

    def test_long_result_truncates(self):
        long_text = "x" * 5000
        s = load_replay(_result_dict([_step_dict(result=long_text)]))
        step = s.current()
        assert step.result_truncated is True
        assert len(step.result_preview) == 4000  # _MAX_RESULT_PREVIEW_CHARS
        assert step.result_preview == long_text[:4000]

    def test_short_result_not_truncated(self):
        s = load_replay(_result_dict([_step_dict(result="hi")]))
        step = s.current()
        assert step.result_truncated is False
        assert step.result_preview == "hi"

    def test_non_string_result_coerced(self):
        s = load_replay(_result_dict([_step_dict(result=42)]))
        assert s.current().result_preview == "42"

    def test_skipped_step(self):
        s = load_replay(
            _result_dict([_step_dict(skipped=True, success=False)])
        )
        step = s.current()
        assert step.skipped is True
        assert step.success is False

    def test_error_step(self):
        s = load_replay(
            _result_dict(
                [
                    _step_dict(
                        success=False,
                        error_message="LLM 503",
                    )
                ]
            )
        )
        step = s.current()
        assert step.success is False
        assert step.error_message == "LLM 503"

    def test_step_type_enum_coerced(self):
        class _Enum:
            value = "tool"

        s = load_replay(_result_dict([_step_dict(step_type=_Enum())]))
        assert s.current().step_type == "tool"

    def test_malformed_step_dropped(self):
        # No step_number AND no step_title — can't be identified.
        s = load_replay(_result_dict([
            {"step_type": "llm"},
            _step_dict(step_number=2, step_title="ok"),
        ]))
        # Only the good step survives.
        assert s.step_count == 1
        assert s.current().step_title == "ok"

    def test_history_truncated_to_tuple(self):
        s = load_replay(
            _result_dict([_step_dict(updated_history=["a", "b", "c"])])
        )
        # Tuples are immutable so the session can be frozen.
        assert isinstance(s.current().history_snapshot, tuple)
        assert s.current().history_snapshot == ("a", "b", "c")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


class TestNavigation:
    def _three_step_session(self) -> ReplaySession:
        return load_replay(
            _result_dict(
                [
                    _step_dict(step_number=1, step_title="extract"),
                    _step_dict(step_number=2, step_title="summarise"),
                    _step_dict(step_number=3, step_title="format"),
                ]
            )
        )

    def test_starts_at_zero(self):
        s = self._three_step_session()
        assert s.cursor == 0
        assert s.current().step_title == "extract"
        assert s.at_start is True
        assert s.at_end is False

    def test_next_advances(self):
        s = self._three_step_session()
        assert s.next().step_title == "summarise"
        assert s.cursor == 1
        assert s.at_start is False
        assert s.at_end is False
        assert s.next().step_title == "format"
        assert s.cursor == 2
        assert s.at_end is True

    def test_next_clamps_at_end(self):
        s = self._three_step_session()
        s.seek(2)
        assert s.next().step_title == "format"
        # No-op when at the end.
        assert s.cursor == 2

    def test_previous_reverses(self):
        s = self._three_step_session()
        s.seek(2)
        assert s.previous().step_title == "summarise"
        assert s.previous().step_title == "extract"
        # Clamps at the start.
        assert s.previous().step_title == "extract"
        assert s.cursor == 0
        assert s.at_start is True

    def test_seek_negative_counts_from_end(self):
        s = self._three_step_session()
        assert s.seek(-1).step_title == "format"
        assert s.cursor == 2
        assert s.seek(-2).step_title == "summarise"

    def test_seek_out_of_bounds_clamps(self):
        s = self._three_step_session()
        # Past the end clamps to last index.
        assert s.seek(999).step_title == "format"
        # Past the start clamps to 0.
        assert s.seek(-999).step_title == "extract"

    def test_restart_returns_to_zero(self):
        s = self._three_step_session()
        s.seek(2)
        assert s.restart().step_title == "extract"
        assert s.cursor == 0

    def test_at_method_raises_out_of_range(self):
        s = self._three_step_session()
        with pytest.raises(IndexError):
            s.at(99)
        # Valid index works.
        assert s.at(1).step_title == "summarise"

    def test_step_titles(self):
        s = self._three_step_session()
        assert s.step_titles() == ("extract", "summarise", "format")


# ---------------------------------------------------------------------------
# Empty session navigation
# ---------------------------------------------------------------------------


class TestEmptySession:
    def test_navigation_safe(self):
        s = load_replay(None)
        assert s.is_empty
        assert s.current() is None
        assert s.next() is None
        assert s.previous() is None
        assert s.seek(0) is None
        assert s.restart() is None

    def test_format_text(self):
        s = load_replay(None)
        assert s.format_text() == "no steps to replay"


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_header_and_step_block(self):
        s = load_replay(
            {
                "chain_id": "ent-42",
                "chain_title": "Weather",
                "result": _result_dict(
                    [_step_dict(step_number=1, step_title="extract")]
                ),
            }
        )
        text = s.format_text()
        assert "chain: Weather" in text
        assert "id: ent-42" in text
        assert "step 1/1" in text
        assert "#1 extract (llm)" in text
        assert "time: 0.120s" in text
        assert "model: claude-opus" in text

    def test_truncation_annotation_in_text(self):
        s = load_replay(_result_dict([_step_dict(result="x" * 5000)]))
        text = s.format_text()
        assert "truncated at 4000 chars" in text

    def test_failed_badge(self):
        s = load_replay(
            _result_dict(
                [
                    _step_dict(
                        success=False,
                        error_message="boom",
                    )
                ]
            )
        )
        text = s.format_text()
        assert "[FAILED]" in text
        assert "error: boom" in text

    def test_skipped_badge(self):
        s = load_replay(_result_dict([_step_dict(skipped=True)]))
        text = s.format_text()
        assert "[SKIPPED]" in text

    def test_history_snapshot_count(self):
        s = load_replay(
            _result_dict([_step_dict(updated_history=["a", "b", "c"])])
        )
        text = s.format_text()
        assert "history snapshot: 3 entries" in text

    def test_singular_history_entry(self):
        s = load_replay(
            _result_dict([_step_dict(updated_history=["only"])])
        )
        text = s.format_text()
        assert "history snapshot: 1 entry" in text


# ---------------------------------------------------------------------------
# Step shape
# ---------------------------------------------------------------------------


class TestStepShape:
    def test_step_frozen(self):
        s = load_replay(_result_dict())
        step = s.current()
        assert isinstance(step, ReplayStep)
        with pytest.raises(Exception):
            step.step_title = "other"  # type: ignore[misc]
