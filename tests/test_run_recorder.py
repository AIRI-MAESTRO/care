"""Tests for ``care.runtime.run_recorder`` (TODO §3 P0).

Coverage layers:
1. ``summarise_reasoning_result`` duck-types correctly across all
   supported attribute / dict shapes.
2. ``RunSummary`` shape + ``status_label`` predicate.
3. ``record_run_completion`` end-to-end against a ``respx``-mocked
   Memory server: posts a memory_card with the right shape, fires
   the ``run-recorded`` ping, returns a typed ``RunCompletion``.
4. Failure paths: ``record_run`` failure surfaces as
   ``agent_recorded=False`` (card still saved).
5. Tag and run-id behaviour: extra_tags merge + dedupe; auto run_id
   format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import pytest
import respx
from gigaevo_client import GigaEvoClient

from care.memory import CareMemory
from care.runtime import (
    RunCompletion,
    RunSummary,
    record_run_completion,
    summarise_reasoning_result,
)

BASE = "http://test-memory:8000"


@pytest.fixture
def memory():
    return CareMemory(GigaEvoClient(base_url=BASE, api_key="sk-test", timeout=5.0))


# ---------------------------------------------------------------------------
# Duck-typed summarisation
# ---------------------------------------------------------------------------


@dataclass
class _FakeCarlResult:
    success: bool = True
    step_results: list = None  # type: ignore[assignment]
    duration_seconds: float = 0.0
    total_tokens: int | None = None
    error_message: str | None = None
    metrics: dict | None = None

    def __post_init__(self):
        if self.step_results is None:
            self.step_results = []
        if self.metrics is None:
            self.metrics = {}


class TestSummariseReasoningResult:
    def test_attribute_shape(self):
        result = _FakeCarlResult(
            success=True,
            step_results=[object(), object()],
            duration_seconds=3.5,
            total_tokens=42,
            metrics={"llm_calls": 4},
        )
        s = summarise_reasoning_result(result)
        assert s.success is True
        assert s.step_count == 2
        assert s.duration_seconds == 3.5
        assert s.total_tokens == 42
        assert s.metrics == {"llm_calls": 4}
        assert s.error_message is None

    def test_dict_input(self):
        s = summarise_reasoning_result(
            {
                "success": False,
                "steps": [1, 2, 3],
                "elapsed_seconds": 1.25,
                "tokens": 100,
                "error": "step 2 timed out",
            }
        )
        assert s.success is False
        assert s.step_count == 3
        assert s.duration_seconds == 1.25
        assert s.total_tokens == 100
        assert s.error_message == "step 2 timed out"

    def test_success_inferred_when_no_explicit_flag(self):
        """When ``success`` is missing, presence of an error_message
        is the deciding signal."""
        s = summarise_reasoning_result({"steps": [1]})
        assert s.success is True
        s = summarise_reasoning_result({"steps": [1], "error_message": "x"})
        assert s.success is False

    def test_empty_input_yields_failed_summary_when_no_steps(self):
        s = summarise_reasoning_result({})
        # No error → success=True per the fallback, but step_count=0.
        assert s.success is True
        assert s.step_count == 0
        assert s.duration_seconds == 0.0
        assert s.total_tokens is None


class TestRunSummary:
    def test_status_label(self):
        assert RunSummary(success=True).status_label == "success"
        assert RunSummary(success=False).status_label == "failed"

    def test_frozen(self):
        s = RunSummary(success=True)
        with pytest.raises(AttributeError):
            s.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# End-to-end record_run_completion
# ---------------------------------------------------------------------------


def _mock_routes(*, card_id: str = "card-1", record_run_status: int = 200):
    """Helper: standard happy-path respx mocks for both writes."""
    captured: dict = {}

    def _card_handler(request):
        captured["card_body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "entity_type": "memory_card",
                "entity_id": card_id,
                "version_id": "v-1",
                "channel": "latest",
            },
        )

    def _record_handler(request):
        captured["record_body"] = json.loads(request.content)
        if record_run_status == 200:
            return httpx.Response(
                200,
                json={
                    "entity_type": "chain",
                    "entity_id": "agent-42",
                    "version_id": "v-1",
                    "channel": "latest",
                    "etag": "e",
                    "meta": {"name": "weather-bot"},
                    "content": {},
                    "run_count": 1,
                    "last_run_at": "2026-05-19T12:00:00Z",
                },
            )
        return httpx.Response(record_run_status, json={"detail": "boom"})

    respx.post(f"{BASE}/v1/memory-cards").mock(side_effect=_card_handler)
    respx.post(
        re.compile(rf"{BASE}/v1/chains/[^/]+/run-recorded")
    ).mock(side_effect=_record_handler)
    return captured


class TestRecordRunCompletion:
    @respx.mock
    def test_happy_path_returns_completion(self, memory):
        captured = _mock_routes()
        finished = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
        completion = record_run_completion(
            memory,
            agent_entity_id="agent-42",
            agent_name="weather-bot",
            query="weather report for SF",
            result=_FakeCarlResult(
                success=True, step_results=[1, 2, 3], duration_seconds=2.5
            ),
            finished_at=finished,
        )
        assert isinstance(completion, RunCompletion)
        assert completion.memory_card_entity_id == "card-1"
        assert completion.agent_entity_id == "agent-42"
        assert completion.agent_recorded is True
        # Auto-generated run id pinned to finished_at.
        assert completion.run_id == "run-20260519T120000000000"

        # Card body shape.
        card_body = captured["card_body"]
        assert "success" in card_body["meta"]["name"]
        # Always-on tags merged into meta.tags.
        assert "agent_run" in card_body["meta"]["tags"]
        assert "agent:agent-42" in card_body["meta"]["tags"]
        assert "status:success" in card_body["meta"]["tags"]
        # Content carries the user query verbatim.
        assert card_body["content"]["task_description"] == "weather report for SF"
        # Usage block has the run id + agent id + metrics.
        usage = card_body["content"]["usage"]
        assert usage["run_id"] == "run-20260519T120000000000"
        assert usage["agent_entity_id"] == "agent-42"
        assert usage["agent_name"] == "weather-bot"
        assert usage["metrics"]["step_count"] == 3
        assert usage["metrics"]["duration_seconds"] == 2.5
        assert usage["metrics"]["exit_status"] == "success"

        # Run-recorded ping forwards the run_id.
        assert captured["record_body"] == {"run_id": "run-20260519T120000000000"}

    @respx.mock
    def test_failed_run_records_failure_status(self, memory):
        captured = _mock_routes()
        completion = record_run_completion(
            memory,
            agent_entity_id="agent-1",
            agent_name="x",
            result=_FakeCarlResult(
                success=False, step_results=[1], error_message="timeout step 1"
            ),
        )
        assert completion.summary.success is False
        assert completion.summary.error_message == "timeout step 1"
        card_body = captured["card_body"]
        assert "status:failed" in card_body["meta"]["tags"]
        assert "failed" in card_body["content"]["description"]
        # Truncated error still surfaces in the description.
        assert "timeout" in card_body["content"]["description"]

    @respx.mock
    def test_extra_tags_merged_and_deduped(self, memory):
        captured = _mock_routes()
        record_run_completion(
            memory,
            agent_entity_id="a-1",
            agent_name="x",
            result=_FakeCarlResult(),
            extra_tags=["finance", "agent_run", "demo"],  # "agent_run" already present
        )
        tags = captured["card_body"]["meta"]["tags"]
        assert tags.count("agent_run") == 1
        assert "finance" in tags
        assert "demo" in tags

    @respx.mock
    def test_caller_supplied_run_id_wins(self, memory):
        captured = _mock_routes()
        completion = record_run_completion(
            memory,
            agent_entity_id="a-1",
            agent_name="x",
            result=_FakeCarlResult(),
            run_id="custom-ulid-7",
        )
        assert completion.run_id == "custom-ulid-7"
        assert captured["record_body"] == {"run_id": "custom-ulid-7"}

    @respx.mock
    def test_record_run_failure_surfaces_as_agent_recorded_false(self, memory):
        """Counter-bump failure must NOT roll back the card save —
        the card already landed; we leave it as a soft signal."""
        captured = _mock_routes(record_run_status=503)
        completion = record_run_completion(
            memory,
            agent_entity_id="a-1",
            agent_name="x",
            result=_FakeCarlResult(),
        )
        assert completion.agent_recorded is False
        # Card still saved.
        assert completion.memory_card_entity_id == "card-1"
        assert "card_body" in captured

    @respx.mock
    def test_accepts_pre_built_run_summary(self, memory):
        captured = _mock_routes()
        summary = RunSummary(
            success=True,
            step_count=5,
            duration_seconds=10.0,
            total_tokens=2500,
            metrics={"llm_calls": 7},
        )
        completion = record_run_completion(
            memory,
            agent_entity_id="a-1",
            agent_name="x",
            result=summary,
        )
        assert completion.summary is summary
        usage = captured["card_body"]["content"]["usage"]
        assert usage["metrics"]["step_count"] == 5
        assert usage["metrics"]["total_tokens"] == 2500
        assert usage["metrics"]["llm_calls"] == 7

    @respx.mock
    def test_agent_entity_type_forwarded(self, memory):
        """When the source isn't a chain (e.g. CARE re-runs an
        agent_skill demo), the right typed route must be hit."""
        captured: dict = {}

        def _card(request):
            return httpx.Response(
                200,
                json={
                    "entity_type": "memory_card",
                    "entity_id": "card-x",
                    "version_id": "v",
                    "channel": "latest",
                },
            )

        def _record(request):
            captured["url"] = str(request.url)
            return httpx.Response(
                200,
                json={
                    "entity_type": "agent_skill",
                    "entity_id": "skill-9",
                    "version_id": "v-1",
                    "channel": "latest",
                    "etag": "e",
                    "meta": {"name": "pdf-extract"},
                    "content": {},
                    "run_count": 1,
                    "last_run_at": "2026-05-19T12:00:00Z",
                },
            )

        respx.post(f"{BASE}/v1/memory-cards").mock(side_effect=_card)
        respx.post(
            re.compile(rf"{BASE}/v1/agent-skills/[^/]+/run-recorded")
        ).mock(side_effect=_record)

        record_run_completion(
            memory,
            agent_entity_id="skill-9",
            agent_name="pdf-extract",
            result=_FakeCarlResult(),
            agent_entity_type="agent_skill",
        )
        assert "agent-skills" in captured["url"]
        assert "skill-9" in captured["url"]
