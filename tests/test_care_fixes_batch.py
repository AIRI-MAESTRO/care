"""Regression tests for the care-fixes batch (PR #20):

- ``tools.tag_filter`` env coercion — a bare ``CARE_TOOLS__TAG_FILTER`` string
  used to abort ``CareConfig.load()`` with a list_type ValidationError.
- chain-name sanitization — MAGE's suggested name leaked the generation
  preamble (``--- TASK: (Today is …)``) into the saved/displayed name.
- the reuse-cache parameterization gate — a bare ``json.dumps`` raised on a
  step_config already upgraded to Pydantic objects and was misread as
  "not parameterized", silently disabling chain reuse.
"""

from __future__ import annotations

from care.config import CareConfig, ToolsConfig
from care.runtime.save_agent_form import sanitize_chain_name
from care.screens.chat import ChatScreen


class TestTagFilterCoercion:
    def test_bare_string_becomes_single_item_list(self):
        assert ToolsConfig(tag_filter="foo").tag_filter == ["foo"]

    def test_comma_separated_string_splits(self):
        assert ToolsConfig(tag_filter="a, b ,c").tag_filter == ["a", "b", "c"]

    def test_blank_string_is_none(self):
        assert ToolsConfig(tag_filter="").tag_filter is None
        assert ToolsConfig(tag_filter="   ").tag_filter is None

    def test_list_and_none_passthrough(self):
        assert ToolsConfig(tag_filter=["x", "y"]).tag_filter == ["x", "y"]
        assert ToolsConfig(tag_filter=None).tag_filter is None

    def test_env_var_does_not_crash_load(self, monkeypatch):
        # The shipped bug: CARE_TOOLS__TAG_FILTER=foo aborted CareConfig.load().
        monkeypatch.setenv("CARE_TOOLS__TAG_FILTER", "foo")
        cfg = CareConfig.load()
        assert cfg.tools.tag_filter == ["foo"]


class TestSanitizeChainName:
    def test_strips_truncated_today_and_task_preamble(self):
        garbage = (
            "Finance — --- TASK: (Today is 2026-06-18T04:39:05+00:00 "
            "(Thursday, 18 "
        )
        assert sanitize_chain_name(garbage) == "Finance"

    def test_strips_full_preamble_keeps_real_task(self):
        name = (
            "Report — (Today is 2026-06-18. Treat this as the current "
            "date/time.) extract revenue"
        )
        assert sanitize_chain_name(name) == "Report — extract revenue"

    def test_clean_name_unchanged(self):
        assert (
            sanitize_chain_name("Quarterly Report Extractor")
            == "Quarterly Report Extractor"
        )

    def test_all_preamble_becomes_empty(self):
        assert (
            sanitize_chain_name(
                "--- TASK: (Today is X. Treat this as the current date/time.)"
            )
            == ""
        )

    def test_truncates_long_names(self):
        out = sanitize_chain_name("x" * 200, max_len=80)
        assert len(out) <= 81 and out.endswith("…")

    def test_empty_input(self):
        assert sanitize_chain_name("") == ""


class _TypedCfg:
    """Mimics a CARL ToolStepConfig left in the dict by ``from_dict`` — NOT
    JSON-serializable, and its ``str()`` carries the $outer_context marker."""

    def __str__(self) -> str:
        return "tool_name='web_search' input_mapping={'query': '$outer_context'}"


class TestReuseCacheGate:
    def test_survives_typed_step_config(self):
        # Before default=str this json.dumps raised → swallowed → False, so any
        # chain whose step_config got upgraded to objects was never cached.
        chain = {"steps": [{"step_type": "tool", "step_config": _TypedCfg()}]}
        assert ChatScreen._chain_is_parameterized(chain) is True

    def test_plain_parameterized_dict(self):
        assert (
            ChatScreen._chain_is_parameterized(
                {"steps": [{"tool_input_mapping": {"q": "$outer_context"}}]}
            )
            is True
        )

    def test_non_parameterized_and_non_dict(self):
        assert (
            ChatScreen._chain_is_parameterized(
                {"steps": [{"tool_input_mapping": {"q": "Moscow"}}]}
            )
            is False
        )
        assert ChatScreen._chain_is_parameterized("nope") is False
