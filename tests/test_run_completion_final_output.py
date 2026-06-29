"""B4 gap-fill: RunCompletion.final_output — the chain's answer text surfaces
on the completion record (both the recorded and the no-record branches), so
headless callers and the C1 gate read the answer without re-parsing results."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from care.runtime.run_recorder import (
    extract_final_output,
    record_run_completion,
)


class _FakeMemory:
    def __init__(self) -> None:
        self.cards: list[dict[str, Any]] = []
        self.client = SimpleNamespace(_record_run=lambda *a, **k: None)

    def save_memory_card(self, content: Any, **kwargs: Any) -> str:
        self.cards.append({"content": content, **kwargs})
        return "card-1"


def _result(answer: str | None = "THE ANSWER") -> SimpleNamespace:
    return SimpleNamespace(
        success=True,
        step_results=[],
        token_usage={"total": 5},
        execution_time=0.1,
        get_final_output=lambda: answer,
    )


def test_record_run_completion_carries_final_output():
    completion = record_run_completion(
        _FakeMemory(),
        agent_entity_id="chain-1",
        agent_name="Echo",
        result=_result(),
        query="q",
    )
    assert completion.final_output == "THE ANSWER"
    assert completion.summary.success is True


def test_extract_final_output_duck_typing():
    assert extract_final_output(SimpleNamespace(get_final_output=lambda: "x")) == "x"
    assert extract_final_output(SimpleNamespace(final_output="y")) == "y"
    assert extract_final_output(object()) is None

    def _boom() -> str:
        raise RuntimeError("nope")

    assert extract_final_output(SimpleNamespace(get_final_output=_boom)) is None


async def test_no_record_branch_carries_final_output(monkeypatch):
    """execute_library_run(record_completion=False) keeps the answer too."""
    import care.runtime.library_run as library_run
    from care.runtime.run_context_draft import RunContextDraft

    plan = library_run.LibraryRunPlan(
        chain=object(),
        entity_id="chain-1",
        display_name="Echo",
        draft=RunContextDraft(source_entity_id="chain-1", task_description="t"),
    )
    monkeypatch.setattr(
        library_run, "prime_from_saved_chain", lambda *a, **k: SimpleNamespace()
    )

    async def fake_execute(chain: Any, context: Any) -> Any:
        return _result("NO-RECORD ANSWER")

    monkeypatch.setattr(library_run, "execute_chain_async", fake_execute)
    completion = await library_run.execute_library_run(
        SimpleNamespace(),  # memory unused on the no-record branch
        plan,
        plan.draft,
        config=SimpleNamespace(),
        api=object(),
        record_completion=False,
    )
    assert completion.final_output == "NO-RECORD ANSWER"
    assert completion.memory_card_entity_id == ""
    assert completion.agent_recorded is False
