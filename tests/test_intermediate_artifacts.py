"""Tests for ``care.intermediate_artifacts`` (TODO §4 P1).

Coverage layers:

1. **Input coercion** — `MAGEResult`-like, intermediate-artifacts
   dict directly, ``None``, weird types all funnel into a clean
   :class:`IntermediateArtifactsView`.
2. **Per-stage summarisation** — each known stage gets a
   tailored one-line summary (counts, scores, pass/fail).
3. **Ordering** — pipeline order preserved even when the input
   dict happens to be in a different key order; unknown stages
   appended after the known ones.
4. **Body rendering** — nested dicts indent; long strings
   truncate; lists with >5 items show "more" tail.
5. **`format_text`** — multi-pane render assembles every stage's
   header / summary / body.
6. **Empty input** — `is_empty=True`, "no intermediate
   artifacts" text.
"""

from __future__ import annotations

import pytest

from care.intermediate_artifacts import (
    IntermediateArtifact,
    IntermediateArtifactsView,
    project_intermediate_artifacts,
)


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


class TestInputCoercion:
    def test_none_yields_empty_view(self):
        view = project_intermediate_artifacts(None)
        assert isinstance(view, IntermediateArtifactsView)
        assert view.is_empty
        assert view.format_text() == "no intermediate artifacts"

    def test_empty_dict_yields_empty_view(self):
        view = project_intermediate_artifacts({})
        assert view.is_empty

    def test_artifacts_dict_directly(self):
        # No wrapping `intermediate_artifacts` key — caller passed
        # the dict that lives under that key already.
        view = project_intermediate_artifacts(
            {"domain_analysis": {"domain": "weather"}}
        )
        assert not view.is_empty
        assert view.stages() == ("domain_analysis",)

    def test_mage_result_dict_unwraps_intermediate_artifacts(self):
        # MAGEResult-shaped dict: has `intermediate_artifacts` key.
        view = project_intermediate_artifacts(
            {
                "chain_json": "{}",
                "intermediate_artifacts": {
                    "domain_analysis": {"domain": "weather"},
                },
            }
        )
        assert view.stages() == ("domain_analysis",)

    def test_attribute_access_via_object(self):
        class _Result:
            intermediate_artifacts = {
                "step_plan": {"steps": [1, 2, 3]},
            }

        view = project_intermediate_artifacts(_Result())
        assert view.stages() == ("step_plan",)

    def test_unrecognised_input_yields_empty(self):
        # A bare int can't be an artifacts source — return empty
        # rather than raising.
        view = project_intermediate_artifacts(42)
        assert view.is_empty


# ---------------------------------------------------------------------------
# Per-stage summaries
# ---------------------------------------------------------------------------


class TestPerStageSummaries:
    def test_domain_analysis(self):
        view = project_intermediate_artifacts(
            {
                "domain_analysis": {
                    "domain": "weather",
                    "task_type": "lookup",
                    "complexity": "low",
                    "suggested_step_count": 2,
                },
            }
        )
        art = view.by_stage("domain_analysis")
        assert art is not None
        assert "domain=weather" in art.summary
        assert "type=lookup" in art.summary
        assert "complexity=low" in art.summary
        assert "suggested_steps=2" in art.summary
        assert art.header == "Domain analysis"

    def test_step_plan_counts_steps(self):
        view = project_intermediate_artifacts(
            {"step_plan": {"steps": ["a", "b", "c", "d"]}}
        )
        art = view.by_stage("step_plan")
        assert art is not None
        assert "4 steps planned" in art.summary

    def test_step_plan_singular(self):
        view = project_intermediate_artifacts(
            {"step_plan": {"steps": ["only"]}}
        )
        assert "1 step planned" in view.by_stage("step_plan").summary

    def test_dag_node_edge_counts(self):
        view = project_intermediate_artifacts(
            {
                "dag": {
                    "nodes": [{"id": 1}, {"id": 2}, {"id": 3}],
                    "edges": [{"from": 1, "to": 2}],
                },
            }
        )
        art = view.by_stage("dag")
        assert art is not None
        assert "3 nodes" in art.summary
        assert "1 edge," in art.summary or "1 edge" in art.summary

    def test_critique_renders_score_and_failures(self):
        view = project_intermediate_artifacts(
            {
                "critique": {
                    "overall_score": 0.82,
                    "failing_step_numbers": [3, 7],
                },
            }
        )
        art = view.by_stage("critique")
        assert art is not None
        assert "score=0.82" in art.summary
        assert "2 failing steps" in art.summary

    def test_verification_passed(self):
        view = project_intermediate_artifacts(
            {"verification": {"passed": True}}
        )
        assert view.by_stage("verification").summary == "passed"

    def test_verification_failed_with_issues(self):
        view = project_intermediate_artifacts(
            {
                "verification": {
                    "passed": False,
                    "issues": ["A", "B"],
                },
            }
        )
        text = view.by_stage("verification").summary
        assert "failed" in text
        assert "2" in text

    def test_refine_iterations_and_delta(self):
        view = project_intermediate_artifacts(
            {"refine": {"iterations": 3, "quality_delta": 0.15}}
        )
        text = view.by_stage("refine").summary
        assert "iterations=3" in text
        assert "+0.15" in text

    def test_unknown_stage_generic_summary(self):
        view = project_intermediate_artifacts(
            {"custom_stage": {"a": 1, "b": 2}}
        )
        art = view.by_stage("custom_stage")
        assert art is not None
        assert "2 fields" in art.summary
        # Header falls back to title-cased.
        assert art.header == "Custom Stage"

    def test_unknown_stage_with_list_payload(self):
        view = project_intermediate_artifacts(
            {"custom": [1, 2, 3, 4]}
        )
        assert "4 items" in view.by_stage("custom").summary

    def test_unknown_stage_scalar_payload(self):
        view = project_intermediate_artifacts({"custom": "hello"})
        assert view.by_stage("custom").summary == "str"


# ---------------------------------------------------------------------------
# Ordering + dropping empty stages
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_known_stages_in_pipeline_order(self):
        # Source dict in reverse order — projector restores pipeline order.
        view = project_intermediate_artifacts(
            {
                "refine": {"iterations": 1},
                "verification": {"passed": True},
                "dag": {"nodes": [1]},
                "step_plan": {"steps": [1]},
                "domain_analysis": {"domain": "x"},
                "critique": {"overall_score": 0.5},
            }
        )
        assert view.stages() == (
            "domain_analysis",
            "step_plan",
            "dag",
            "critique",
            "verification",
            "refine",
        )

    def test_unknown_stages_appended_after_known(self):
        view = project_intermediate_artifacts(
            {
                "step_plan": {"steps": [1]},
                "extension": {"thing": 1},
                "domain_analysis": {"domain": "x"},
            }
        )
        stages = view.stages()
        # Known stages first, in pipeline order.
        assert stages[:2] == ("domain_analysis", "step_plan")
        # Unknown stage after.
        assert "extension" in stages[2:]

    def test_empty_stage_payloads_dropped(self):
        # Empty dict / list / None payloads don't produce an
        # artifact (nothing to render).
        view = project_intermediate_artifacts(
            {
                "domain_analysis": {"domain": "x"},
                "step_plan": {},  # dropped
                "dag": None,  # dropped
                "critique": [],  # dropped
            }
        )
        assert view.stages() == ("domain_analysis",)


# ---------------------------------------------------------------------------
# Body rendering
# ---------------------------------------------------------------------------


class TestBodyRendering:
    def test_dict_renders_key_value_lines(self):
        view = project_intermediate_artifacts(
            {
                "domain_analysis": {
                    "domain": "weather",
                    "complexity": "low",
                },
            }
        )
        body = view.by_stage("domain_analysis").body
        assert "domain: weather" in body
        assert "complexity: low" in body

    def test_nested_dict_indents(self):
        view = project_intermediate_artifacts(
            {
                "custom": {
                    "outer": "ok",
                    "nested": {"inner": "child"},
                },
            }
        )
        body = view.by_stage("custom").body
        assert "nested:" in body
        assert "  inner: child" in body

    def test_long_string_truncated(self):
        long = "x" * 200
        view = project_intermediate_artifacts(
            {"custom": {"text": long}}
        )
        body = view.by_stage("custom").body
        assert "..." in body
        assert long not in body

    def test_list_of_dicts_renders_bullets(self):
        view = project_intermediate_artifacts(
            {
                "step_plan": {
                    "steps": [
                        {"id": 1, "title": "a"},
                        {"id": 2, "title": "b"},
                    ],
                },
            }
        )
        body = view.by_stage("step_plan").body
        assert "(2 items)" in body
        assert "id=1" in body

    def test_long_list_truncated(self):
        view = project_intermediate_artifacts(
            {"step_plan": {"steps": list(range(20))}}
        )
        body = view.by_stage("step_plan").body
        # Default cap of 5 items shown inside nested dicts.
        assert "more" in body


# ---------------------------------------------------------------------------
# Aggregate view
# ---------------------------------------------------------------------------


class TestAggregateView:
    def test_format_text_includes_headers(self):
        view = project_intermediate_artifacts(
            {
                "domain_analysis": {"domain": "weather"},
                "step_plan": {"steps": [1, 2]},
            }
        )
        text = view.format_text()
        assert "# Domain analysis" in text
        assert "# Step plan" in text
        # Stages separated by blank lines.
        assert "\n\n" in text

    def test_by_stage_returns_none_for_unknown(self):
        view = project_intermediate_artifacts({"domain_analysis": {"d": 1}})
        assert view.by_stage("step_plan") is None

    def test_view_frozen(self):
        view = project_intermediate_artifacts({"domain_analysis": {"d": 1}})
        with pytest.raises(Exception):
            view.artifacts = ()  # type: ignore[misc]

    def test_artifact_frozen(self):
        view = project_intermediate_artifacts({"domain_analysis": {"d": 1}})
        art = view.artifacts[0]
        assert isinstance(art, IntermediateArtifact)
        with pytest.raises(Exception):
            art.summary = "other"  # type: ignore[misc]

    def test_raw_payload_preserved(self):
        payload = {"domain": "x", "complexity": "high"}
        view = project_intermediate_artifacts({"domain_analysis": payload})
        # `raw` is the original dict (or a copy that equals it).
        assert view.artifacts[0].raw == payload


# ---------------------------------------------------------------------------
# Real MAGE roundtrip (gated on mage extra)
# ---------------------------------------------------------------------------


def _mage_installed() -> bool:
    try:
        import mmar_mage  # noqa: F401
    except ImportError:
        return False
    return True


class TestMageRoundtrip:
    @pytest.mark.skipif(
        not _mage_installed(),
        reason="mmar_mage not installed",
    )
    def test_real_mage_result_unwraps(self):
        from mmar_mage.schemas import MAGEResult

        result = MAGEResult(
            chain_json="{}",
            chain_dict={"steps": []},
            intermediate_artifacts={
                "domain_analysis": {
                    "domain": "weather",
                    "task_type": "lookup",
                },
                "step_plan": {"steps": [1, 2, 3]},
                "dag": {"nodes": [1, 2, 3], "edges": [{"from": 1, "to": 2}]},
            },
        )
        view = project_intermediate_artifacts(result)
        assert view.stages() == ("domain_analysis", "step_plan", "dag")
        assert "weather" in view.by_stage("domain_analysis").summary
        assert "3 steps planned" in view.by_stage("step_plan").summary
        assert "3 nodes" in view.by_stage("dag").summary
