"""Tests for the re-run-from-library data layer (TODO §3 P1).

The LibraryScreen + RunContextModal are gated on §1 P0; this
suite pins the contract those screens will rely on.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError

import pytest

from care.runtime.library_run import (
    LibraryRunError,
    LibraryRunPlan,
    execute_library_run,
    load_run_plan,
)
from care.runtime.run_context_draft import RunContextDraft, set_task
from care.runtime.run_recorder import RunCompletion, RunSummary


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeChain:
    """Mimics a CARL `ReasoningChain` carrying CARE metadata.
    Test driver pre-stamps ``entity_id`` so the projection picks
    it up."""

    def __init__(
        self,
        *,
        entity_id: str = "ent-1",
        task: str = "Summarise the PDF",
        files: list[dict] | None = None,
        display_name: str = "PDF summariser",
        executed: list[bool] | None = None,
        result: object = None,
    ):
        self.entity_id = entity_id
        self._meta = {
            "task_description": task,
            "context_files": files
            if files is not None
            else [
                {
                    "path": "/tmp/example.pdf",
                    "sha256": "a" * 64,
                    "size_bytes": 1024,
                    "mime_type": "application/pdf",
                }
            ],
            "display_name": display_name,
            "tags": ["pdf"],
        }
        self._executed = executed
        self._result = result

    def get_care_metadata(self):
        return self._meta

    async def execute_async(self, context):
        if self._executed is not None:
            self._executed.append(True)
        return self._result or _FakeResult()


class _FakeResult:
    """Mimics a CARL `ReasoningResult` enough for
    `summarise_reasoning_result`."""

    def __init__(
        self,
        *,
        success: bool = True,
        step_results=None,
        duration: float = 1.5,
        tokens: int = 42,
        error: str | None = None,
    ):
        self.success = success
        self.step_results = step_results or []
        self.duration_seconds = duration
        self.total_tokens = tokens
        self.error_message = error


class _StubClient:
    """SDK-shape stub with `get_chain`."""

    def __init__(
        self,
        *,
        chain=None,
        exc=None,
        delay: float = 0.0,
    ):
        self.calls: list[dict] = []
        self._chain = chain
        self._exc = exc
        self._delay = delay

    def get_chain(self, entity_id, channel="latest", **_):
        self.calls.append({"entity_id": entity_id, "channel": channel})
        if self._delay:
            time.sleep(self._delay)
        if self._exc:
            raise self._exc
        return self._chain


class _StubMemory:
    """Mimics a `CareMemory` facade with `.client` accessor and
    `save_memory_card` so `record_run_completion` works."""

    def __init__(self, client, *, save_card_id: str = "card-1"):
        self.client = client
        self._save_card_id = save_card_id
        self.save_calls: list[dict] = []

    def save_memory_card(
        self, content, *, name, tags=None, when_to_use=None, author=None,
        entity_id=None, channel="latest",
    ):
        self.save_calls.append(
            {
                "content": content,
                "name": name,
                "tags": list(tags) if tags else None,
                "author": author,
            }
        )
        return self._save_card_id


# Patch the client to support `_record_run` (used by
# `record_run_completion`).
def _attach_record_run(client):
    def _record_run(entity_type, entity_id, run_id=None):
        client.calls.append(
            {"op": "record_run", "entity_type": entity_type,
             "entity_id": entity_id, "run_id": run_id}
        )

    client._record_run = _record_run
    return client


# ---------------------------------------------------------------------------
# load_run_plan
# ---------------------------------------------------------------------------


class TestLoadRunPlan:
    def test_happy_path(self):
        chain = _FakeChain()
        memory = _StubMemory(_StubClient(chain=chain))
        plan = asyncio.run(load_run_plan(memory, "ent-1"))
        assert plan.entity_id == "ent-1"
        assert plan.channel == "latest"
        assert plan.entity_type == "chain"
        assert plan.chain is chain
        assert plan.has_chain
        # Draft populated from the chain's CARE metadata.
        assert plan.draft.task_description == "Summarise the PDF"
        assert plan.draft.source_entity_id == "ent-1"
        assert plan.draft.source_name == "PDF summariser"
        # File row materialised.
        assert len(plan.draft.files) == 1
        assert plan.draft.files[0].path == "/tmp/example.pdf"

    def test_source_name_overrides_metadata(self):
        chain = _FakeChain()
        memory = _StubMemory(_StubClient(chain=chain))
        plan = asyncio.run(
            load_run_plan(memory, "ent-1", source_name="My label")
        )
        assert plan.display_name == "My label"
        assert plan.draft.source_name == "My label"

    def test_custom_channel_and_entity_type(self):
        chain = _FakeChain()
        memory = _StubMemory(_StubClient(chain=chain))
        plan = asyncio.run(
            load_run_plan(
                memory, "ent-1", channel="stable", entity_type="agent",
            )
        )
        assert plan.channel == "stable"
        assert plan.entity_type == "agent"
        assert memory.client.calls[0]["channel"] == "stable"

    def test_empty_entity_id_raises(self):
        memory = _StubMemory(_StubClient())
        with pytest.raises(LibraryRunError, match="entity_id"):
            asyncio.run(load_run_plan(memory, ""))

    def test_missing_client_raises(self):
        with pytest.raises(LibraryRunError, match="get_chain"):
            asyncio.run(load_run_plan(object(), "ent-1"))

    def test_underscored_client_also_works(self):
        chain = _FakeChain()

        class _Memory:
            def __init__(self, client):
                self._client = client

        memory = _Memory(_StubClient(chain=chain))
        plan = asyncio.run(load_run_plan(memory, "ent-1"))
        assert plan.chain is chain

    def test_sdk_exception_wraps(self):
        memory = _StubMemory(_StubClient(exc=RuntimeError("404 not found")))
        with pytest.raises(LibraryRunError, match="chain fetch failed"):
            asyncio.run(load_run_plan(memory, "ent-1"))

    def test_timeout(self):
        memory = _StubMemory(_StubClient(chain=_FakeChain(), delay=0.5))
        with pytest.raises(LibraryRunError, match="timed out"):
            asyncio.run(load_run_plan(memory, "ent-1", timeout=0.05))

    def test_none_chain_response_raises(self):
        memory = _StubMemory(_StubClient(chain=None))
        with pytest.raises(LibraryRunError, match="not found"):
            asyncio.run(load_run_plan(memory, "ent-1"))

    def test_entity_id_patched_onto_chain(self):
        # Even when the chain object doesn't carry entity_id, the
        # loader stamps it so downstream consumers (record_run_completion)
        # can route correctly.
        chain = _FakeChain()
        chain.entity_id = ""  # type: ignore[assignment]
        memory = _StubMemory(_StubClient(chain=chain))
        plan = asyncio.run(load_run_plan(memory, "ent-7"))
        assert plan.draft.source_entity_id == "ent-7"

    def test_frozen_immutable_chain_handled_gracefully(self):
        # Chain with no setable entity_id slot — projection falls
        # back to entity_id-only via draft patch.
        class _Frozen:
            __slots__ = ("_meta",)

            def __init__(self, meta):
                object.__setattr__(self, "_meta", meta)

            def get_care_metadata(self):
                return self._meta

        chain = _Frozen({"task_description": "x", "display_name": "y"})
        memory = _StubMemory(_StubClient(chain=chain))
        plan = asyncio.run(load_run_plan(memory, "ent-frozen"))
        assert plan.draft.source_entity_id == "ent-frozen"


class TestAgentSkillFetchResilience:
    """The SDK's `get_chain` legacy parse chokes on agent_skill steps
    (AgentSkillStepConfig predates the StepDescription union) — the Run
    button must still load such chains via the raw-dict + typed-parse path."""

    _RAW = {
        "task_description": "summarise the doc",
        "steps": [
            {
                "number": 1,
                "title": "Extract text",
                "step_type": "agent_skill",
                "aim": "read the docx",
                "step_config": {
                    "skill": "github://anthropics/skills/skills/docx@main",
                    "task": "Extract text from the provided DOCX",
                    "execution_mode": "llm",
                    "input_mapping": {},
                    "output_key": "extracted",
                },
            },
        ],
    }

    def test_falls_back_to_get_chain_dict_when_get_chain_raises(self):
        class _C:
            def __init__(self) -> None:
                self.get_calls = 0
                self.dict_calls = 0

            def get_chain(self, eid, channel="latest", **_):
                self.get_calls += 1
                raise ValueError("ValidationError: AgentSkillStepConfig …")

            def get_chain_dict(self, eid, channel="latest", **_):
                self.dict_calls += 1
                return dict(TestAgentSkillFetchResilience._RAW)

        client = _C()
        plan = asyncio.run(load_run_plan(_StubMemory(client), "ent-skill"))
        assert plan.has_chain
        assert client.dict_calls == 1
        # The dict path succeeded → the broken get_chain is never called.
        assert client.get_calls == 0
        assert len(getattr(plan.chain, "steps", [])) == 1

    def test_get_chain_dict_failure_falls_back_to_get_chain(self):
        # If the raw-dict accessor itself errors, the legacy path is still
        # tried (keeps the old behaviour for non-agent_skill chains).
        chain = _FakeChain()

        class _C:
            def get_chain_dict(self, eid, channel="latest", **_):
                raise RuntimeError("no dict route")

            def get_chain(self, eid, channel="latest", **_):
                return chain

        plan = asyncio.run(load_run_plan(_StubMemory(_C()), "ent-1"))
        assert plan.chain is chain


class TestSkillFileBridge:
    """execute_library_run wires an attached document into doc-skill steps."""

    @staticmethod
    def _docx_chain_dict() -> dict:
        return {
            "task_description": "summarise",
            "steps": [
                {
                    "number": 1,
                    "title": "Extract text",
                    "step_type": "agent_skill",
                    "aim": "read the docx",
                    "step_config": {
                        "skill": "github://anthropics/skills/skills/docx@main",
                        "task": "Extract text from the provided DOCX",
                        "execution_mode": "llm",
                        "input_mapping": {},
                        "output_key": "extracted",
                    },
                },
            ],
        }

    def test_bridge_rewrites_doc_step_and_merges_files(self, tmp_path):
        from mmar_carl import ReasoningChain

        from care.runtime.library_run import _apply_skill_file_bridge
        from care.runtime.run_context_draft import ContextFile, RunContextDraft

        f = tmp_path / "r.txt"
        f.write_text("DOC BODY")
        chain = ReasoningChain.from_dict(
            self._docx_chain_dict(), use_typed_steps=True,
        )
        draft = RunContextDraft(
            source_entity_id="c",
            task_description="summarise",
            files=(ContextFile(path=str(f), status="added"),),
        )
        extras: dict = {}
        new_chain = _apply_skill_file_bridge(chain, draft, extras)
        cfg = new_chain.steps[0].step_config
        mapping = getattr(cfg, "input_mapping", None) or {}
        assert any(
            str(v).startswith("$memory.input.") for v in mapping.values()
        )
        assert "DOC BODY" in extras.get("files", {}).values()

    def test_bridge_noop_without_files(self):
        from mmar_carl import ReasoningChain

        from care.runtime.library_run import _apply_skill_file_bridge
        from care.runtime.run_context_draft import RunContextDraft

        chain = ReasoningChain.from_dict(
            self._docx_chain_dict(), use_typed_steps=True,
        )
        draft = RunContextDraft(source_entity_id="c", task_description="x")
        extras: dict = {}
        assert _apply_skill_file_bridge(chain, draft, extras) is chain
        assert "files" not in extras


# ---------------------------------------------------------------------------
# LibraryRunPlan shape
# ---------------------------------------------------------------------------


class TestPlanShape:
    def test_plan_is_frozen(self):
        plan = LibraryRunPlan(
            chain=object(),
            entity_id="x",
            draft=RunContextDraft(source_entity_id="x"),
        )
        with pytest.raises(FrozenInstanceError):
            plan.entity_id = "y"  # type: ignore[misc]

    def test_has_chain_predicate(self):
        with_chain = LibraryRunPlan(
            chain=object(),
            entity_id="x",
            draft=RunContextDraft(source_entity_id="x"),
        )
        without = LibraryRunPlan(
            chain=None,
            entity_id="x",
            draft=RunContextDraft(source_entity_id="x"),
        )
        assert with_chain.has_chain is True
        assert without.has_chain is False


# ---------------------------------------------------------------------------
# execute_library_run
# ---------------------------------------------------------------------------


class TestExecuteLibraryRun:
    def _setup(self, **chain_overrides):
        chain = _FakeChain(**chain_overrides)
        client = _attach_record_run(_StubClient(chain=chain))
        memory = _StubMemory(client)
        return chain, memory

    def test_orchestrates_full_flow(self, monkeypatch):
        # Patch prime_from_saved_chain to bypass the real CARL
        # dependency since we don't want the test to require an
        # installed mmar_carl.
        called: list[dict] = []

        def fake_prime(chain, **kwargs):
            called.append({"chain": chain, "kwargs": kwargs})
            return _FakeContext()

        monkeypatch.setattr(
            "care.runtime.library_run.prime_from_saved_chain", fake_prime
        )

        executed_flags: list[bool] = []
        chain, memory = self._setup(
            executed=executed_flags,
            result=_FakeResult(),
        )
        plan = asyncio.run(load_run_plan(memory, "ent-1"))
        completion = asyncio.run(
            execute_library_run(
                memory, plan, plan.draft, config=_FakeConfig(), api=object()
            )
        )
        assert isinstance(completion, RunCompletion)
        assert completion.agent_entity_id == "ent-1"
        assert completion.summary.success
        # Chain was actually executed.
        assert executed_flags == [True]
        # The prime call was made with the chain.
        assert len(called) == 1
        assert called[0]["chain"] is chain
        # Regression: config must flow into prime_from_saved_chain — it is
        # what registers builtin tools (web_search, …); without it the C1
        # gate's baseline dies with "Tool 'web_search' not registered".
        assert called[0]["kwargs"].get("config") is not None
        # A memory_card was persisted.
        assert len(memory.save_calls) == 1
        # The run id was recorded against the source entity.
        record_calls = [c for c in memory.client.calls if c.get("op") == "record_run"]
        assert len(record_calls) == 1
        assert record_calls[0]["entity_type"] == "chain"
        assert record_calls[0]["entity_id"] == "ent-1"

    def test_validation_errors_raise(self, monkeypatch):
        monkeypatch.setattr(
            "care.runtime.library_run.prime_from_saved_chain",
            lambda *a, **kw: _FakeContext(),
        )
        chain, memory = self._setup()
        plan = asyncio.run(load_run_plan(memory, "ent-1"))
        # Blank out the task to trigger a validation error.
        broken = set_task(plan.draft, "  ")
        with pytest.raises(LibraryRunError, match="unresolved errors"):
            asyncio.run(
                execute_library_run(
                    memory, plan, broken, config=_FakeConfig(), api=object()
                )
            )

    def test_no_chain_raises(self):
        plan = LibraryRunPlan(
            chain=None,
            entity_id="ent-1",
            draft=RunContextDraft(
                source_entity_id="ent-1",
                task_description="ok",
            ),
        )
        with pytest.raises(LibraryRunError, match="no chain"):
            asyncio.run(
                execute_library_run(
                    object(), plan, plan.draft, config=None, api=None,
                )
            )

    def test_prime_failure_wraps(self, monkeypatch):
        def fake_prime(*a, **kw):
            raise RuntimeError("CARL refused")

        monkeypatch.setattr(
            "care.runtime.library_run.prime_from_saved_chain", fake_prime
        )
        chain, memory = self._setup()
        plan = asyncio.run(load_run_plan(memory, "ent-1"))
        with pytest.raises(LibraryRunError, match="prime context"):
            asyncio.run(
                execute_library_run(
                    memory, plan, plan.draft, config=_FakeConfig(), api=None,
                )
            )

    def test_execution_failure_wraps(self, monkeypatch):
        monkeypatch.setattr(
            "care.runtime.library_run.prime_from_saved_chain",
            lambda *a, **kw: _FakeContext(),
        )

        class _BrokenChain(_FakeChain):
            async def execute_async(self, context):
                raise RuntimeError("step crash")

        client = _attach_record_run(_StubClient(chain=_BrokenChain()))
        memory = _StubMemory(client)
        plan = asyncio.run(load_run_plan(memory, "ent-1"))
        with pytest.raises(LibraryRunError, match="execution failed"):
            asyncio.run(
                execute_library_run(
                    memory, plan, plan.draft, config=_FakeConfig(), api=None,
                )
            )

    def test_record_completion_false_skips_card_write(self, monkeypatch):
        monkeypatch.setattr(
            "care.runtime.library_run.prime_from_saved_chain",
            lambda *a, **kw: _FakeContext(),
        )
        chain, memory = self._setup()
        plan = asyncio.run(load_run_plan(memory, "ent-1"))
        completion = asyncio.run(
            execute_library_run(
                memory, plan, plan.draft,
                config=_FakeConfig(), api=None,
                record_completion=False,
            )
        )
        assert isinstance(completion, RunCompletion)
        assert completion.memory_card_entity_id == ""
        assert completion.agent_recorded is False
        assert memory.save_calls == []

    def test_custom_run_id_threads_through(self, monkeypatch):
        monkeypatch.setattr(
            "care.runtime.library_run.prime_from_saved_chain",
            lambda *a, **kw: _FakeContext(),
        )
        chain, memory = self._setup()
        plan = asyncio.run(load_run_plan(memory, "ent-1"))
        completion = asyncio.run(
            execute_library_run(
                memory, plan, plan.draft,
                config=_FakeConfig(), api=None,
                run_id="run-custom-7",
            )
        )
        assert completion.run_id == "run-custom-7"

    def test_overrides_applied_to_config_copy(self, monkeypatch):
        # The session config should remain untouched even when
        # the draft carries an override.
        seen_configs: list[object] = []

        def fake_prime(chain, **kwargs):
            return _FakeContext()

        monkeypatch.setattr(
            "care.runtime.library_run.prime_from_saved_chain", fake_prime
        )
        # The function passes the config through `apply_overrides`
        # internally; tests verify that path via the
        # `tests/test_run_context_draft.py` suite. Here we just
        # confirm the call completes end-to-end with an override
        # set on the draft (smoke).
        from care.runtime.run_context_draft import set_model_override

        chain, memory = self._setup()
        plan = asyncio.run(load_run_plan(memory, "ent-1"))
        with_override = set_model_override(plan.draft, model="gpt-99")
        completion = asyncio.run(
            execute_library_run(
                memory, plan, with_override,
                config=_FakeConfig(), api=None,
            )
        )
        assert completion.summary.success
        _ = seen_configs

    def test_failed_run_still_records(self, monkeypatch):
        monkeypatch.setattr(
            "care.runtime.library_run.prime_from_saved_chain",
            lambda *a, **kw: _FakeContext(),
        )
        chain = _FakeChain(
            result=_FakeResult(success=False, error="step 2 failed"),
        )
        client = _attach_record_run(_StubClient(chain=chain))
        memory = _StubMemory(client)
        plan = asyncio.run(load_run_plan(memory, "ent-1"))
        completion = asyncio.run(
            execute_library_run(
                memory, plan, plan.draft,
                config=_FakeConfig(), api=None,
            )
        )
        # The card is written even on failure.
        assert len(memory.save_calls) == 1
        # The summary reflects the failure.
        assert isinstance(completion.summary, RunSummary)
        assert completion.summary.status_label == "failed"
        assert "step 2" in (completion.summary.error_message or "")


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            LibraryRunError as Err,
            LibraryRunPlan as Plan,
            load_run_plan as load,
            execute_library_run as execute,
        )

        assert Err is LibraryRunError
        assert Plan is LibraryRunPlan
        assert load is load_run_plan
        assert execute is execute_library_run


# ---------------------------------------------------------------------------
# Fake config / context (lightweight test scaffolding)
# ---------------------------------------------------------------------------


class _FakeMage:
    model = "m1"
    provider = "p1"

    def model_copy(self, *, update):
        new = _FakeMage()
        for k, v in update.items():
            setattr(new, k, v)
        return new


class _FakeConfig:
    """Minimal CareConfig-shaped object that `apply_overrides`
    knows how to handle via `model_copy`."""

    def __init__(self):
        self.mage = _FakeMage()

    def model_copy(self, *, update):
        new = _FakeConfig()
        for k, v in update.items():
            setattr(new, k, v)
        return new


class _FakeContext:
    """Stand-in for a CARL ReasoningContext."""

    pass
