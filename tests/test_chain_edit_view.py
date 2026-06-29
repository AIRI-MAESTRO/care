"""Tests for ``care.runtime.chain_edit_view`` — edit-plan transcript rendering."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from care.runtime.chain_edit_view import (
    format_revise_confirm_body,
    render_disambiguation_lines,
    render_edit_plan_lines,
    render_step_delta,
    revise_result_has_changes,
)


def _result(**kw: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "summary": "",
        "edits": [],
        "before_chain_dict": {},
        "chain_dict": {},
        "candidates": [],
        "needs_disambiguation": False,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _edit(op: str, target: int | None = None, rationale: str = "") -> SimpleNamespace:
    return SimpleNamespace(op=op, target_step_number=target, rationale=rationale)


def test_step_delta_reports_name_count_and_steps() -> None:
    before = {"name": "A", "steps": [{"number": 1, "title": "S1", "step_type": "llm"}]}
    after = {
        "name": "B",
        "steps": [
            {"number": 1, "title": "S1", "step_type": "llm"},
            {"number": 2, "title": "S2", "step_type": "structured_output"},
        ],
    }
    lines = render_step_delta(before, after)
    assert any("name:" in line and "'A'" in line and "'B'" in line for line in lines)
    assert any("steps: 1 → 2" in line for line in lines)
    assert any("2. S2 [structured_output]" in line for line in lines)


def test_step_delta_no_changes_when_identical() -> None:
    chain = {"name": "A", "steps": [{"number": 1, "title": "S1", "step_type": "llm"}]}
    lines = render_step_delta(chain, dict(chain))
    # no name/count lines, just the single after-step row
    assert not any(line.startswith("name:") for line in lines)
    assert not any(line.startswith("steps:") for line in lines)
    assert lines == ["1. S1 [llm]"]


def test_edit_plan_lines_summary_edits_and_delta() -> None:
    res = _result(
        summary="renamed + added step",
        edits=[
            _edit("set_chain_field", rationale="apply name"),
            _edit("insert_step", target=3, rationale="add validation"),
        ],
        before_chain_dict={"name": "A", "steps": []},
        chain_dict={"name": "B", "steps": []},
    )
    lines = render_edit_plan_lines(res)
    assert lines[0] == "renamed + added step"
    assert any("set_chain_field" in line and "apply name" in line for line in lines)
    assert any("insert_step step 3" in line and "add validation" in line for line in lines)
    assert any("name:" in line for line in lines)


def test_edit_plan_lines_no_change() -> None:
    lines = render_edit_plan_lines(_result(summary="", edits=[]))
    assert lines[0] == "No change."
    assert any("(no edits)" in line for line in lines)


def test_disambiguation_lines_lists_candidates() -> None:
    res = _result(
        needs_disambiguation=True,
        candidates=[
            SimpleNamespace(entity_id="a1", name="Alpha", score=0.82),
            SimpleNamespace(entity_id="b2", name="", score=0.5),
        ],
    )
    lines = render_disambiguation_lines(res)
    assert "Multiple chains match" in lines[0]
    assert any("a1" in line and "Alpha" in line and "0.82" in line for line in lines)
    assert any("(unnamed)" in line for line in lines)


def test_revise_result_has_changes_with_edits_or_dict_delta() -> None:
    assert not revise_result_has_changes(_result())
    assert revise_result_has_changes(_result(edits=[_edit("set_chain_field")]))
    assert revise_result_has_changes(
        _result(
            before_chain_dict={"steps": []},
            chain_dict={"steps": [{"number": 1}]},
        )
    )


def test_format_revise_confirm_body_includes_preview() -> None:
    res = _result(
        summary="renamed",
        edits=[_edit("set_chain_field", rationale="apply name")],
        before_chain_dict={"name": "A", "steps": []},
        chain_dict={"name": "B", "steps": []},
    )
    body = format_revise_confirm_body(res, intro="Save?")
    assert body.startswith("Save?")
    assert "renamed" in body
    assert "set_chain_field" in body
