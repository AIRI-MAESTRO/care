"""Tests for the agent side-by-side comparison data layer (TODO §1.3 P2).

The diff modal is gated on §1 P0; this suite pins the contract
the modal will bind to.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError

import pytest

from care.runtime.agent_diff import (
    AgentDiff,
    AgentDiffError,
    FieldDiff,
    MetadataDiff,
    StepDiff,
    diff_chains,
    fetch_agent_diff,
)


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------


def _step(
    *,
    number: int,
    title: str = "Step",
    step_type: str = "llm",
    **extras,
) -> dict:
    base = {"number": number, "title": title, "step_type": step_type}
    base.update(extras)
    return base


def _chain(
    *,
    steps: list | None = None,
    display_name: str = "Agent",
    description: str = "A test agent",
    tags: list[str] | None = None,
    task_description: str = "Do a thing",
) -> dict:
    return {
        "steps": steps if steps is not None else [],
        "content": {
            "metadata": {
                "care": {
                    "display_name": display_name,
                    "description": description,
                    "tags": tags if tags is not None else [],
                    "task_description": task_description,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# diff_chains — step diffs
# ---------------------------------------------------------------------------


class TestDiffChainsSteps:
    def test_identical_chains_all_unchanged(self):
        chain = _chain(
            steps=[
                _step(number=1, title="Plan"),
                _step(number=2, title="Execute"),
            ]
        )
        diff = diff_chains(chain, chain)
        assert len(diff.steps) == 2
        assert all(s.kind == "unchanged" for s in diff.steps)
        assert diff.has_changes is False
        assert diff.added_steps == 0
        assert diff.removed_steps == 0
        assert diff.modified_steps == 0
        assert diff.unchanged_steps == 2

    def test_added_step_on_right(self):
        left = _chain(steps=[_step(number=1, title="Plan")])
        right = _chain(
            steps=[
                _step(number=1, title="Plan"),
                _step(number=2, title="Execute"),
            ]
        )
        diff = diff_chains(left, right)
        assert diff.added_steps == 1
        added = next(s for s in diff.steps if s.kind == "added")
        assert added.number == 2
        assert added.title_right == "Execute"
        assert added.title_left is None
        # Field diffs surface every leaf of the added step.
        assert any(f.field == "title" and f.right_value == "Execute" for f in added.fields)

    def test_removed_step_on_left(self):
        left = _chain(
            steps=[
                _step(number=1, title="Plan"),
                _step(number=2, title="Execute"),
            ]
        )
        right = _chain(steps=[_step(number=1, title="Plan")])
        diff = diff_chains(left, right)
        assert diff.removed_steps == 1
        removed = next(s for s in diff.steps if s.kind == "removed")
        assert removed.number == 2
        assert removed.title_left == "Execute"
        assert removed.title_right is None

    def test_modified_step_records_field_diffs(self):
        left = _chain(
            steps=[_step(number=1, title="Plan", config={"prompt": "v1"})]
        )
        right = _chain(
            steps=[_step(number=1, title="Plan", config={"prompt": "v2"})]
        )
        diff = diff_chains(left, right)
        assert diff.modified_steps == 1
        modified = diff.steps[0]
        assert modified.kind == "modified"
        # Field diff includes the nested prompt change.
        prompt_diff = next(
            f for f in modified.fields if f.field == "config.prompt"
        )
        assert prompt_diff.left_value == "v1"
        assert prompt_diff.right_value == "v2"

    def test_title_change_only(self):
        left = _chain(steps=[_step(number=1, title="Old")])
        right = _chain(steps=[_step(number=1, title="New")])
        diff = diff_chains(left, right)
        assert diff.modified_steps == 1
        modified = diff.steps[0]
        title_diff = next(f for f in modified.fields if f.field == "title")
        assert title_diff.left_value == "Old"
        assert title_diff.right_value == "New"

    def test_added_field_present_only_on_right(self):
        left = _chain(steps=[_step(number=1, title="Step")])
        right = _chain(
            steps=[
                _step(number=1, title="Step", retry_max=3),
            ]
        )
        diff = diff_chains(left, right)
        modified = diff.steps[0]
        assert modified.kind == "modified"
        retry_diff = next(
            f for f in modified.fields if f.field == "retry_max"
        )
        assert retry_diff.left_present is False
        assert retry_diff.right_present is True
        assert retry_diff.right_value == 3

    def test_removed_field_present_only_on_left(self):
        left = _chain(steps=[_step(number=1, title="Step", retry_max=3)])
        right = _chain(steps=[_step(number=1, title="Step")])
        diff = diff_chains(left, right)
        modified = diff.steps[0]
        retry_diff = next(
            f for f in modified.fields if f.field == "retry_max"
        )
        assert retry_diff.left_present is True
        assert retry_diff.right_present is False
        assert retry_diff.left_value == 3

    def test_steps_sorted_by_number(self):
        # Out-of-order input → diff sorts by canonical step number.
        left = _chain(
            steps=[
                _step(number=3, title="Third"),
                _step(number=1, title="First"),
            ]
        )
        right = _chain(
            steps=[
                _step(number=2, title="Second"),
                _step(number=1, title="First"),
            ]
        )
        diff = diff_chains(left, right)
        assert [s.number for s in diff.steps] == [1, 2, 3]

    def test_transient_fields_ignored(self):
        # `sub_chain`, `agents`, `metrics`, `cache`, `base_step`
        # are runtime-only and the projection drops them.
        left = _chain(
            steps=[_step(number=1, title="x", metrics={"runs": 5})]
        )
        right = _chain(
            steps=[_step(number=1, title="x", metrics={"runs": 99})]
        )
        diff = diff_chains(left, right)
        # Should be unchanged (metrics dropped).
        assert diff.steps[0].kind == "unchanged"

    def test_step_type_carried_through(self):
        left = _chain(
            steps=[_step(number=1, title="t", step_type="llm")]
        )
        right = _chain(
            steps=[_step(number=1, title="t", step_type="tool")]
        )
        diff = diff_chains(left, right)
        modified = diff.steps[0]
        assert modified.step_type_left == "llm"
        assert modified.step_type_right == "tool"

    def test_label_uses_right_then_left(self):
        # Right wins on modified; falls back to left for removed.
        left = _chain(steps=[_step(number=1, title="LeftTitle")])
        right = _chain(steps=[_step(number=1, title="RightTitle")])
        diff = diff_chains(left, right)
        assert diff.steps[0].label == "RightTitle"

        removed_only = diff_chains(
            _chain(steps=[_step(number=1, title="GoneTitle")]),
            _chain(steps=[]),
        )
        assert removed_only.steps[0].label == "GoneTitle"

    def test_list_field_compared_by_value(self):
        left = _chain(
            steps=[_step(number=1, title="x", dependencies=[1, 2])]
        )
        right = _chain(
            steps=[_step(number=1, title="x", dependencies=[1, 2])]
        )
        diff = diff_chains(left, right)
        assert diff.steps[0].kind == "unchanged"
        right = _chain(
            steps=[_step(number=1, title="x", dependencies=[1, 3])]
        )
        diff = diff_chains(left, right)
        assert diff.steps[0].kind == "modified"


# ---------------------------------------------------------------------------
# diff_chains — metadata
# ---------------------------------------------------------------------------


class TestDiffChainsMetadata:
    def test_display_name_change(self):
        left = _chain(display_name="Alpha")
        right = _chain(display_name="Beta")
        diff = diff_chains(left, right)
        assert diff.metadata.has_changes
        name_diff = next(
            f for f in diff.metadata.fields if f.field == "display_name"
        )
        assert name_diff.left_value == "Alpha"
        assert name_diff.right_value == "Beta"

    def test_tag_added_and_removed(self):
        left = _chain(tags=["a", "b"])
        right = _chain(tags=["b", "c"])
        diff = diff_chains(left, right)
        assert diff.metadata.added_tags == ("c",)
        assert diff.metadata.removed_tags == ("a",)

    def test_identical_metadata_no_changes(self):
        chain = _chain(
            display_name="Same",
            description="Same",
            tags=["x"],
            task_description="task",
        )
        diff = diff_chains(chain, chain)
        assert diff.metadata.has_changes is False

    def test_task_description_change(self):
        left = _chain(task_description="Old task")
        right = _chain(task_description="New task")
        diff = diff_chains(left, right)
        task_diff = next(
            f for f in diff.metadata.fields if f.field == "task_description"
        )
        assert task_diff.left_value == "Old task"
        assert task_diff.right_value == "New task"


# ---------------------------------------------------------------------------
# Aggregate accessors
# ---------------------------------------------------------------------------


class TestAgentDiffAggregate:
    def test_format_summary_no_diff(self):
        diff = diff_chains(_chain(), _chain())
        assert diff.format_summary() == "no differences"

    def test_format_summary_counts_each_kind(self):
        left = _chain(
            steps=[
                _step(number=1, title="A"),
                _step(number=2, title="B"),
                _step(number=3, title="C"),
            ]
        )
        right = _chain(
            steps=[
                _step(number=1, title="A2"),  # modified
                _step(number=2, title="B"),  # unchanged
                _step(number=4, title="D"),  # added (3 removed)
            ]
        )
        diff = diff_chains(left, right)
        summary = diff.format_summary()
        assert "+1" in summary
        assert "-1" in summary
        assert "~1" in summary
        assert "of 4 steps" in summary

    def test_labels_pulled_from_metadata(self):
        left = _chain(display_name="Alpha")
        right = _chain(display_name="Beta")
        diff = diff_chains(left, right)
        assert diff.left_label == "Alpha"
        assert diff.right_label == "Beta"

    def test_labels_override(self):
        diff = diff_chains(
            _chain(display_name="Alpha"), _chain(display_name="Beta"),
            left_label="Override Left", right_label="Override Right",
        )
        assert diff.left_label == "Override Left"
        assert diff.right_label == "Override Right"

    def test_entity_ids_stamped(self):
        diff = diff_chains(
            _chain(), _chain(),
            left_entity_id="ent-1", right_entity_id="ent-2",
        )
        assert diff.left_entity_id == "ent-1"
        assert diff.right_entity_id == "ent-2"


# ---------------------------------------------------------------------------
# Frozen models
# ---------------------------------------------------------------------------


class TestFrozenModels:
    def test_field_diff_frozen(self):
        fd = FieldDiff(field="x")
        with pytest.raises(FrozenInstanceError):
            fd.field = "y"  # type: ignore[misc]

    def test_step_diff_frozen(self):
        sd = StepDiff(number=1, kind="unchanged")
        with pytest.raises(FrozenInstanceError):
            sd.kind = "modified"  # type: ignore[misc]

    def test_metadata_diff_frozen(self):
        md = MetadataDiff()
        with pytest.raises(FrozenInstanceError):
            md.added_tags = ("x",)  # type: ignore[misc]

    def test_agent_diff_frozen(self):
        ad = AgentDiff()
        with pytest.raises(FrozenInstanceError):
            ad.left_label = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


class TestCoercion:
    def test_to_dict_object_accepted(self):
        class _ChainObj:
            def to_dict(self):
                return _chain(
                    steps=[_step(number=1, title="From to_dict")],
                )

        diff = diff_chains(_ChainObj(), _chain(steps=[_step(number=1, title="From to_dict")]))
        assert diff.steps[0].kind == "unchanged"

    def test_model_dump_object_accepted(self):
        class _ChainObj:
            def model_dump(self, **_kw):
                return _chain(steps=[_step(number=1, title="From model_dump")])

        diff = diff_chains(_ChainObj(), _ChainObj())
        assert len(diff.steps) == 1
        assert diff.steps[0].kind == "unchanged"

    def test_flat_chain_dict_accepted(self):
        # Without the `content.metadata.care` wrapper.
        left = {"steps": [_step(number=1, title="x")], "display_name": "Flat"}
        right = {"steps": [_step(number=1, title="x")], "display_name": "Flat2"}
        diff = diff_chains(left, right)
        assert diff.left_label == "Flat"
        assert diff.right_label == "Flat2"

    def test_none_chain_treated_as_empty(self):
        diff = diff_chains(None, _chain(steps=[_step(number=1, title="x")]))
        # Right has 1 step → added.
        assert diff.added_steps == 1


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(self, *, chains=None, exc=None, delay=0.0):
        self.calls: list[dict] = []
        self._chains = chains or {}
        self._exc = exc
        self._delay = delay

    def get_chain_dict(self, entity_id, channel="latest", **_):
        self.calls.append({"entity_id": entity_id, "channel": channel})
        if self._delay:
            time.sleep(self._delay)
        if self._exc:
            raise self._exc
        return self._chains.get(entity_id)


class _StubMemory:
    def __init__(self, client):
        self.client = client


class TestFetchAgentDiff:
    def test_happy_path(self):
        left = _chain(steps=[_step(number=1, title="Plan")])
        right = _chain(
            steps=[
                _step(number=1, title="Plan"),
                _step(number=2, title="Execute"),
            ]
        )
        memory = _StubMemory(
            _StubClient(chains={"left": left, "right": right})
        )
        diff = asyncio.run(
            fetch_agent_diff(memory, "left", "right")
        )
        assert diff.left_entity_id == "left"
        assert diff.right_entity_id == "right"
        assert diff.added_steps == 1
        # Channel propagated.
        assert all(c["channel"] == "latest" for c in memory.client.calls)

    def test_concurrent_fetch(self):
        # Both calls run via gather — wall-clock < 2*delay.
        left = _chain()
        right = _chain()
        memory = _StubMemory(
            _StubClient(
                chains={"left": left, "right": right}, delay=0.05,
            )
        )
        start = time.monotonic()
        asyncio.run(fetch_agent_diff(memory, "left", "right"))
        elapsed = time.monotonic() - start
        # Two 0.05s fetches in parallel → under 0.09s.
        assert elapsed < 0.09, (
            f"expected concurrent fetch under 0.09s, got {elapsed:.3f}s"
        )

    def test_empty_left_raises(self):
        memory = _StubMemory(_StubClient())
        with pytest.raises(AgentDiffError, match="left_entity_id"):
            asyncio.run(fetch_agent_diff(memory, "", "right"))

    def test_empty_right_raises(self):
        memory = _StubMemory(_StubClient())
        with pytest.raises(AgentDiffError, match="right_entity_id"):
            asyncio.run(fetch_agent_diff(memory, "left", ""))

    def test_missing_client_raises(self):
        with pytest.raises(AgentDiffError, match="get_chain_dict"):
            asyncio.run(fetch_agent_diff(object(), "a", "b"))

    def test_left_not_found_raises(self):
        memory = _StubMemory(
            _StubClient(chains={"right": _chain()})
        )
        with pytest.raises(AgentDiffError, match=r"left.*not found"):
            asyncio.run(fetch_agent_diff(memory, "left", "right"))

    def test_right_not_found_raises(self):
        memory = _StubMemory(
            _StubClient(chains={"left": _chain()})
        )
        with pytest.raises(AgentDiffError, match=r"right.*not found"):
            asyncio.run(fetch_agent_diff(memory, "left", "right"))

    def test_sdk_exception_wraps(self):
        memory = _StubMemory(_StubClient(exc=RuntimeError("503")))
        with pytest.raises(AgentDiffError, match="fetch failed"):
            asyncio.run(fetch_agent_diff(memory, "a", "b"))

    def test_timeout_wraps(self):
        memory = _StubMemory(
            _StubClient(chains={"a": _chain(), "b": _chain()}, delay=0.5)
        )
        with pytest.raises(AgentDiffError, match="timed out"):
            asyncio.run(
                fetch_agent_diff(memory, "a", "b", timeout=0.05)
            )

    def test_labels_propagate(self):
        memory = _StubMemory(
            _StubClient(chains={"a": _chain(), "b": _chain()})
        )
        diff = asyncio.run(
            fetch_agent_diff(
                memory, "a", "b",
                left_label="L", right_label="R",
            )
        )
        assert diff.left_label == "L"
        assert diff.right_label == "R"

    def test_underscored_client_works(self):
        class _M:
            def __init__(self, client):
                self._client = client

        memory = _M(_StubClient(chains={"a": _chain(), "b": _chain()}))
        diff = asyncio.run(fetch_agent_diff(memory, "a", "b"))
        assert diff.added_steps == 0

    def test_custom_channel_forwarded(self):
        memory = _StubMemory(
            _StubClient(chains={"a": _chain(), "b": _chain()})
        )
        asyncio.run(fetch_agent_diff(memory, "a", "b", channel="stable"))
        assert all(c["channel"] == "stable" for c in memory.client.calls)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            AgentDiff as A,
            AgentDiffError as E,
            FieldDiff as F,
            StepDiff as S,
            MetadataDiff as M,
            diff_chains as dc,
            fetch_agent_diff as fad,
        )

        assert A is AgentDiff
        assert E is AgentDiffError
        assert F is FieldDiff
        assert S is StepDiff
        assert M is MetadataDiff
        assert dc is diff_chains
        assert fad is fetch_agent_diff
