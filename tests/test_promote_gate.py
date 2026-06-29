"""C1 — the promotion gate: artifact → mandatory baseline run → eval-vs-baseline.

Fake Memory client + injected baseline runner — no network, no LLM."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from care.runtime.promote_gate import gate_promotion

# Inlined (not imported from tests.test_chat_deploy): the local mmar-mage
# editable override puts the flat-layout carl-mage root on sys.path, where its
# `tests/` shadows ours and breaks `tests.*` imports.
SAMPLE_CHAIN: dict[str, Any] = {
    "name": "Echo Researcher",
    "max_workers": 1,
    "timeout": 60.0,
    "steps": [
        {
            "step_type": "llm",
            "number": 1,
            "title": "Answer",
            "aim": "Answer the question",
            "reasoning_questions": "",
            "step_context_queries": [],
            "stage_action": "Answer",
            "example_reasoning": "",
            "dependencies": [],
            "retry_max": 1,
        }
    ],
}

BROKEN_CHAIN = dict(
    SAMPLE_CHAIN,
    steps=[
        {
            "step_type": "tool",
            "number": 1,
            "title": "T",
            "dependencies": [],
            "step_config": {"tool_name": "run_python", "input_mapping": {}},
        }
    ],
)


class FakeClient:
    def __init__(
        self,
        *,
        content: dict[str, Any] | None = None,
        baseline_value: float | None = None,
        winners: list[Any] | None = None,
        beating_error: Exception | None = None,
        record_error: Exception | None = None,
    ) -> None:
        self.content = content if content is not None else dict(SAMPLE_CHAIN)
        self.baseline_value = baseline_value
        self.winners = winners or []
        self.beating_error = beating_error
        self.record_error = record_error

    def get_chain_record(self, entity_id: str, *, channel: str = "latest") -> Any:
        if self.record_error:
            raise self.record_error
        return SimpleNamespace(
            entity_id=entity_id,
            version_id="vid-0003",
            version_number=3,
            meta={"display_name": "Weather"},
            content=self.content,
        )

    def list_chain_versions_beating(self, entity_id: str, *, channel: str = "stable", **_: Any) -> Any:
        if self.beating_error:
            raise self.beating_error
        return SimpleNamespace(baseline_value=self.baseline_value, winners=self.winners)


def _memory(client: FakeClient) -> SimpleNamespace:
    return SimpleNamespace(client=client)


def _runner(success: bool = True, detail: str = "succeeded in 1.0s"):
    calls: list[dict[str, Any]] = []

    async def run(memory: Any, config: Any, entity_id: str, *, channel: str) -> tuple[bool, str]:
        calls.append({"entity_id": entity_id, "channel": channel})
        return success, detail

    run.calls = calls  # type: ignore[attr-defined]
    return run


def _winner(version_id: str = "vid-0003", value: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(version_id=version_id, version_number=3, value=value, delta=0.1)


async def test_all_checks_pass():
    client = FakeClient(baseline_value=0.8, winners=[_winner()])
    runner = _runner()
    report = await gate_promotion(
        _memory(client), SimpleNamespace(), "chain-1", baseline_runner=runner
    )
    assert report.ok is True
    assert [c.name for c in report.checks] == ["artifact", "baseline run", "eval score"]
    assert all(c.passed for c in report.checks)
    assert runner.calls == [{"entity_id": "chain-1", "channel": "latest"}]
    assert "beats" in report.checks[2].detail


async def test_artifact_failure_stops_before_baseline():
    client = FakeClient(content=BROKEN_CHAIN)
    runner = _runner()
    report = await gate_promotion(
        _memory(client), SimpleNamespace(), "chain-1", baseline_runner=runner
    )
    assert report.ok is False
    assert len(report.checks) == 1  # baseline never ran on a broken artifact
    assert runner.calls == []
    assert "run_python" in report.checks[0].detail


async def test_baseline_failure_refuses():
    client = FakeClient(baseline_value=0.8, winners=[_winner()])
    report = await gate_promotion(
        _memory(client),
        SimpleNamespace(),
        "chain-1",
        baseline_runner=_runner(success=False, detail="baseline failed: boom"),
    )
    assert report.ok is False
    assert len(report.checks) == 2  # eval not reached
    assert report.checks[1].passed is False


async def test_eval_failure_refuses():
    client = FakeClient(baseline_value=0.8, winners=[_winner(version_id="vid-9999")])
    report = await gate_promotion(
        _memory(client), SimpleNamespace(), "chain-1", baseline_runner=_runner()
    )
    assert report.ok is False
    assert report.checks[2].passed is False
    assert "does not beat" in report.checks[2].detail


async def test_eval_skipped_without_baseline():
    client = FakeClient(baseline_value=None)
    report = await gate_promotion(
        _memory(client), SimpleNamespace(), "chain-1", baseline_runner=_runner()
    )
    assert report.ok is True  # skipped ≠ failed
    assert report.checks[2].skipped is True
    assert "no eval baseline" in report.checks[2].detail


async def test_eval_skipped_on_query_error():
    client = FakeClient(beating_error=RuntimeError("503"))
    report = await gate_promotion(
        _memory(client), SimpleNamespace(), "chain-1", baseline_runner=_runner()
    )
    assert report.ok is True
    assert report.checks[2].skipped is True


async def test_unloadable_record_fails_artifact():
    client = FakeClient(record_error=KeyError("nope"))
    report = await gate_promotion(
        _memory(client), SimpleNamespace(), "chain-1", baseline_runner=_runner()
    )
    assert report.ok is False
    assert report.checks[0].name == "artifact"
    assert report.checks[0].passed is False
