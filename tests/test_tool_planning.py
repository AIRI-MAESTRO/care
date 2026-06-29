"""Tests for live-data tool routing (care.tool_planning)."""

from __future__ import annotations

import asyncio

from care import tool_planning
from care.config import CareConfig

_DATE_CHAIN = {
    "steps": [
        {"step_type": "llm", "number": 1, "title": "Determine current date",
         "aim": "figure out today's date and weekday"},
        {"step_type": "llm", "number": 2, "title": "Format response",
         "aim": "format", "dependencies": [1]},
    ]
}


class _FakeAPI:
    def __init__(self, reply):  # noqa: ANN001
        self.reply = reply
        self.prompts = []

    async def get_response_with_retries(self, prompt):  # noqa: ANN001
        self.prompts.append(prompt)
        return self.reply


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_json_variants():
    p = tool_planning._parse_json
    assert p('{"a": 1}') == {"a": 1}
    assert p('```json\n{"a": 2}\n```') == {"a": 2}
    assert p('sure: {"a": 3} done') == {"a": 3}
    assert p("no json here") is None


def test_rewrite_drops_llm_fields_and_maps_params():
    chain = {"steps": [{
        "step_type": "llm", "number": 1, "title": "t", "aim": "a",
        "reasoning_questions": "q?", "step_context_queries": ["x"],
    }]}
    assert tool_planning._rewrite_step_to_tool(chain, 1, "get_x", ["q"])
    s = chain["steps"][0]
    assert s["step_type"] == "tool"
    assert "reasoning_questions" not in s and "step_context_queries" not in s
    assert s["step_config"]["tool_name"] == "get_x"
    assert s["step_config"]["input_mapping"] == {"q": "$outer_context"}


def test_rewrite_zero_arg_builtin_has_empty_mapping():
    chain = {"steps": [{"step_type": "llm", "number": 1, "title": "t"}]}
    tool_planning._rewrite_step_to_tool(chain, 1, "current_datetime", ["foo"])
    assert chain["steps"][0]["step_config"]["input_mapping"] == {}


def test_rewrite_web_search_uses_query_outer_context():
    # The classifier often returns the search TEXT as "params"; for a
    # known builtin we ignore it and use the correct signature.
    chain = {"steps": [{"step_type": "llm", "number": 1, "title": "search"}]}
    tool_planning._rewrite_step_to_tool(
        chain, 1, "web_search", ["Imagine Dragons latest track release"]
    )
    sc = chain["steps"][0]["step_config"]
    assert sc["tool_name"] == "web_search"
    assert sc["input_mapping"] == {"query": "$outer_context"}


# ---------------------------------------------------------------------------
# augment_chain_for_live_data
# ---------------------------------------------------------------------------


def test_routes_date_step_to_current_datetime():
    api = _FakeAPI('{"needs_tool": true, "step_number": 1, '
                   '"tool_name": "current_datetime", "params": [], "reason": "live date"}')
    aug = asyncio.run(tool_planning.augment_chain_for_live_data(
        _DATE_CHAIN, task="what day is it", api=api, config=CareConfig()))
    assert aug["rewrote"] is True
    assert aug["tool_name"] == "current_datetime"
    s1 = aug["chain_dict"]["steps"][0]
    assert s1["step_type"] == "tool"
    assert s1["step_config"]["tool_name"] == "current_datetime"
    assert s1["step_config"]["input_mapping"] == {}
    # original dict untouched (deepcopy)
    assert _DATE_CHAIN["steps"][0]["step_type"] == "llm"


def test_routes_to_new_tool_with_params():
    api = _FakeAPI('{"needs_tool": true, "step_number": 1, '
                   '"tool_name": "get_weather", "params": ["city"]}')
    chain = {"steps": [{"step_type": "llm", "number": 1, "title": "weather", "aim": "get weather"}]}
    aug = asyncio.run(tool_planning.augment_chain_for_live_data(
        chain, task="weather in moscow", api=api, config=CareConfig()))
    assert aug["rewrote"] and aug["tool_name"] == "get_weather"
    sc = aug["chain_dict"]["steps"][0]["step_config"]
    assert sc["input_mapping"] == {"city": "$outer_context"}


def test_noop_when_chain_already_has_tool():
    api = _FakeAPI('{"needs_tool": true, "step_number": 1, "tool_name": "x"}')
    chain = {"steps": [{"step_type": "tool", "number": 1,
                        "step_config": {"tool_name": "web_search"}}]}
    aug = asyncio.run(tool_planning.augment_chain_for_live_data(
        chain, task="t", api=api, config=CareConfig()))
    assert aug == {"rewrote": False}
    assert api.prompts == []  # classifier not called when a tool already exists


def test_noop_when_classifier_declines():
    api = _FakeAPI('{"needs_tool": false}')
    aug = asyncio.run(tool_planning.augment_chain_for_live_data(
        _DATE_CHAIN, task="2+2?", api=api, config=CareConfig()))
    assert aug["rewrote"] is False


def test_gated_off():
    cfg = CareConfig()
    cfg.tools.route_live_data_to_tools = False
    api = _FakeAPI('{"needs_tool": true, "step_number": 1, "tool_name": "current_datetime"}')
    aug = asyncio.run(tool_planning.augment_chain_for_live_data(
        _DATE_CHAIN, task="t", api=api, config=cfg))
    assert aug == {"rewrote": False}
    assert api.prompts == []


# ---------------------------------------------------------------------------
# downgrade_unsupported_step_types — keep richer topologies CARL-runnable
# ---------------------------------------------------------------------------


def test_downgrade_rewrites_unsupported_types_to_llm(monkeypatch):
    # A map_reduce/debate-style chain whose exotic types the installed CARL
    # can't load → rewritten to llm, but the fan-out/fan-in SHAPE survives.
    # Pin the loadable set to base CARL so the test is deterministic whether
    # the installed mmar_carl is 0.2.0 (base 7) or the agent-features build.
    monkeypatch.setattr(
        tool_planning, "_carl_loadable_step_types",
        lambda: tool_planning._FALLBACK_LOADABLE_STEP_TYPES,
    )
    cd = {"steps": [
        {"step_number": 1, "step_type": "tool", "dependencies": [],
         "step_config": {"tool_name": "web_search", "input_mapping": {"query": "x"}}},
        {"step_number": 2, "step_type": "debate", "aim": "argue",
         "dependencies": [1], "step_config": {"debate_rounds": 3}},
        {"step_number": 3, "step_type": "evaluation", "aim": "judge",
         "dependencies": [2], "step_config": {"criteria": "good"}},
    ]}
    out = tool_planning.downgrade_unsupported_step_types(cd)
    assert [s["step_type"] for s in out["steps"]] == ["tool", "llm", "llm"]
    # exotic nested config dropped on the downgraded steps; the tool keeps its own
    assert "step_config" not in out["steps"][1]
    assert "step_config" not in out["steps"][2]
    assert out["steps"][0]["step_config"]["tool_name"] == "web_search"
    # topology SHAPE (dependencies) preserved verbatim
    assert [s.get("dependencies") for s in out["steps"]] == [[], [1], [2]]


def test_downgrade_leaves_loadable_types_untouched():
    cd = {"steps": [
        {"step_number": 1, "step_type": "llm", "dependencies": []},
        {"step_number": 2, "step_type": "transform", "dependencies": [1],
         "step_config": {"x": 1}},
    ]}
    tool_planning.downgrade_unsupported_step_types(cd)
    assert [s["step_type"] for s in cd["steps"]] == ["llm", "transform"]
    assert cd["steps"][1]["step_config"] == {"x": 1}  # config kept for loadable types


def test_downgrade_idempotent_and_safe_on_degenerate_input(monkeypatch):
    # base CARL (no `debate`) so the downgrade fires regardless of install
    monkeypatch.setattr(
        tool_planning, "_carl_loadable_step_types",
        lambda: tool_planning._FALLBACK_LOADABLE_STEP_TYPES,
    )
    assert tool_planning.downgrade_unsupported_step_types({}) == {}
    assert tool_planning.downgrade_unsupported_step_types({"steps": None}) == {"steps": None}
    cd = {"steps": [{"step_number": 1, "step_type": "debate", "dependencies": []}]}
    tool_planning.downgrade_unsupported_step_types(cd)
    tool_planning.downgrade_unsupported_step_types(cd)  # second pass is a no-op
    assert cd["steps"][0]["step_type"] == "llm"
