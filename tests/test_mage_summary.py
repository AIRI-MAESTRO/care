"""Tests for ``care.mage_summary.summarise_mage_result`` (TODO §4 P0).

Three coverage layers:

1. **Dict input** — the projector pulls every documented field
   from a plain dict so callers don't need MAGE installed.
2. **Object input** — attribute-style access (the path real
   `MAGEMetadata` / `MAGEResult` take).
3. **Real MAGE roundtrip** — when the ``mage`` extra is
   installed, build a real `MAGEMetadata` + `MAGEResult` and
   verify every field round-trips into the summary. Skipped
   when MAGE isn't available.

`MetadataSummary.format_text` gets its own tiny coverage block
since the future InspectionScreen footer renders it directly.
"""

from __future__ import annotations

from dataclasses import is_dataclass

import pytest

from care.mage_summary import MetadataSummary, summarise_mage_result


def _mage_installed() -> bool:
    try:
        import mmar_mage  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# MetadataSummary shape
# ---------------------------------------------------------------------------


class TestSummaryShape:
    def test_defaults(self):
        s = MetadataSummary()
        assert s.domain == "general"
        assert s.num_steps == 0
        assert s.mode == "deep"
        assert s.stages_completed == ()
        assert s.was_cold_start is False
        # Every optional field starts None.
        assert s.step_critique_score is None
        assert s.verification_passed is None
        assert s.refine_iterations is None
        assert s.refine_quality_delta is None
        assert s.tot_branches_explored is None
        assert s.mcts_simulations_run is None
        assert s.mcts_best_reward is None
        assert s.feedback_recalled is None

    def test_to_dict_converts_tuples_to_lists(self):
        s = MetadataSummary(
            stages_completed=("plan", "dag"),
            suggested_tags=("finance", "demo"),
        )
        d = s.to_dict()
        assert d["stages_completed"] == ["plan", "dag"]
        assert d["suggested_tags"] == ["finance", "demo"]

    def test_summary_is_dataclass(self):
        assert is_dataclass(MetadataSummary())

    def test_summary_is_frozen(self):
        s = MetadataSummary()
        with pytest.raises(Exception):
            s.domain = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Dict input
# ---------------------------------------------------------------------------


class TestDictInput:
    def test_projects_every_documented_field(self):
        payload = {
            "domain": "finance",
            "num_steps": 7,
            "mode": "deep",
            "model": "claude-opus",
            "generation_time_seconds": 12.5,
            "deep_stages_completed": ["plan", "dag", "describe"],
            "memory_hits_used": 4,
            "web_results_used": 2,
            "was_cold_start": False,
            "step_critique_score": 0.82,
            "verification_passed": True,
            "refine_iterations": 3,
            "refine_quality_delta": 0.15,
            "tot_branches_explored": 5,
            "mcts_simulations_run": 30,
            "mcts_best_reward": 0.91,
            "feedback_recalled": 2,
            "suggested_display_name": "Finance Advisor",
            "suggested_description": "Helps with finance",
            "suggested_tags": ["finance", "advisor"],
        }
        s = summarise_mage_result(payload)
        assert s.domain == "finance"
        assert s.num_steps == 7
        assert s.mode == "deep"
        assert s.model == "claude-opus"
        assert s.generation_time_seconds == 12.5
        assert s.stages_completed == ("plan", "dag", "describe")
        assert s.memory_hits_used == 4
        assert s.web_results_used == 2
        assert s.was_cold_start is False
        assert s.step_critique_score == 0.82
        assert s.verification_passed is True
        assert s.refine_iterations == 3
        assert s.refine_quality_delta == 0.15
        assert s.tot_branches_explored == 5
        assert s.mcts_simulations_run == 30
        assert s.mcts_best_reward == 0.91
        assert s.feedback_recalled == 2
        assert s.suggested_display_name == "Finance Advisor"
        assert s.suggested_tags == ("finance", "advisor")

    def test_empty_dict_yields_defaults(self):
        s = summarise_mage_result({})
        assert s == MetadataSummary()

    def test_partial_dict_keeps_defaults_for_missing(self):
        # Older MAGE installs may not populate every field; the
        # projector treats missing keys as their defaults.
        s = summarise_mage_result({"domain": "weather", "num_steps": 2})
        assert s.domain == "weather"
        assert s.num_steps == 2
        # Untouched optional fields stay None.
        assert s.step_critique_score is None
        assert s.tot_branches_explored is None

    def test_dict_with_nested_metadata_unwraps(self):
        # MAGEResult-shaped dict: has `metadata` + `mode` +
        # `chain_json`.
        payload = {
            "chain_json": "{}",
            "mode": "fast",
            "metadata": {
                "domain": "weather",
                "num_steps": 2,
                "memory_hits_used": 1,
            },
        }
        s = summarise_mage_result(payload)
        # `mode` came from the outer result.
        assert s.mode == "fast"
        # Nested metadata fields landed.
        assert s.domain == "weather"
        assert s.num_steps == 2
        assert s.memory_hits_used == 1

    def test_coerces_stringy_numbers(self):
        # Some sources stream metadata as JSON, where numbers
        # may arrive as strings. Optional coercion stays
        # forgiving — converts what it can, defaults what it
        # can't.
        s = summarise_mage_result(
            {
                "num_steps": "5",
                "memory_hits_used": "3",
                "refine_iterations": "2",
            }
        )
        assert s.num_steps == 5
        assert s.memory_hits_used == 3
        assert s.refine_iterations == 2

    def test_unparseable_optional_falls_back_to_none(self):
        # Bad data shouldn't crash — falls back to None.
        s = summarise_mage_result(
            {"step_critique_score": "not a number"}
        )
        assert s.step_critique_score is None


# ---------------------------------------------------------------------------
# Object input
# ---------------------------------------------------------------------------


class TestObjectInput:
    def test_attribute_access_works(self):
        class _Meta:
            domain = "medicine"
            num_steps = 3
            mode = "fast"
            memory_hits_used = 1
            web_results_used = 0
            was_cold_start = True
            deep_stages_completed = ["plan"]
            step_critique_score = 0.7
            tot_branches_explored = 4

        s = summarise_mage_result(_Meta())
        assert s.domain == "medicine"
        assert s.num_steps == 3
        assert s.mode == "fast"
        assert s.memory_hits_used == 1
        assert s.was_cold_start is True
        assert s.stages_completed == ("plan",)
        assert s.step_critique_score == 0.7
        assert s.tot_branches_explored == 4

    def test_result_like_object_unwraps_metadata(self):
        class _Meta:
            domain = "finance"
            num_steps = 4
            memory_hits_used = 2

        class _Result:
            chain_json = "{}"
            mode = "deep"
            metadata = _Meta()

        s = summarise_mage_result(_Result())
        assert s.domain == "finance"
        assert s.num_steps == 4
        assert s.memory_hits_used == 2
        assert s.mode == "deep"

    def test_none_values_become_defaults(self):
        class _Meta:
            domain = None
            num_steps = None
            stages_completed = None
            memory_hits_used = None

        s = summarise_mage_result(_Meta())
        assert s.domain == "general"
        assert s.num_steps == 0
        assert s.memory_hits_used == 0


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_minimal_layout(self):
        s = MetadataSummary(domain="weather", num_steps=2)
        text = s.format_text()
        # Core lines always present.
        assert "domain: weather" in text
        assert "mode: deep" in text
        assert "steps: 2" in text
        # No quality block when nothing fired.
        assert "quality:" not in text

    def test_quality_block_renders_only_set_fields(self):
        s = MetadataSummary(
            step_critique_score=0.82,
            verification_passed=True,
            refine_iterations=2,
            refine_quality_delta=0.15,
        )
        text = s.format_text()
        assert "critique=0.82" in text
        assert "verify=passed" in text
        assert "refine_iters=2" in text
        assert "refine_Δ=+0.15" in text
        # tot/mcts not set → not in output.
        assert "tot=" not in text
        assert "mcts" not in text

    def test_cold_start_annotation(self):
        s = MetadataSummary(memory_hits_used=0, was_cold_start=True)
        text = s.format_text()
        assert "memory hits: 0 (cold start)" in text

    def test_omits_zero_web_results(self):
        s = MetadataSummary(web_results_used=0)
        text = s.format_text()
        # Skip the line when there's nothing to report.
        assert "web results" not in text

    def test_stages_rendered(self):
        s = MetadataSummary(stages_completed=("plan", "dag", "describe"))
        text = s.format_text()
        assert "stages: plan, dag, describe" in text

    def test_verify_failed_text(self):
        s = MetadataSummary(verification_passed=False)
        text = s.format_text()
        assert "verify=failed" in text


# ---------------------------------------------------------------------------
# Real MAGE roundtrip (opt-in via mage extra)
# ---------------------------------------------------------------------------


class TestMageRoundtrip:
    @pytest.mark.skipif(
        not _mage_installed(),
        reason="mmar_mage not installed",
    )
    def test_real_mage_metadata_projects(self):
        from mmar_mage.schemas import MAGEMetadata

        meta = MAGEMetadata(
            domain="finance",
            num_steps=5,
            generation_time_seconds=8.0,
            model="o3-mini",
            deep_stages_completed=["domain", "plan", "dag"],
            memory_hits_used=2,
            web_results_used=1,
            was_cold_start=False,
            step_critique_score=0.88,
            verification_passed=True,
            refine_iterations=1,
            tot_branches_explored=4,
            mcts_simulations_run=20,
            mcts_best_reward=0.75,
            feedback_recalled=2,
            suggested_display_name="Finance Bot",
            suggested_description="Talks finance.",
            suggested_tags=["finance"],
        )
        s = summarise_mage_result(meta)
        assert s.domain == "finance"
        assert s.num_steps == 5
        assert s.model == "o3-mini"
        assert s.generation_time_seconds == 8.0
        assert s.stages_completed == ("domain", "plan", "dag")
        assert s.memory_hits_used == 2
        assert s.web_results_used == 1
        assert s.step_critique_score == 0.88
        assert s.verification_passed is True
        assert s.refine_iterations == 1
        assert s.tot_branches_explored == 4
        assert s.mcts_simulations_run == 20
        assert s.mcts_best_reward == 0.75
        assert s.feedback_recalled == 2
        assert s.suggested_display_name == "Finance Bot"
        assert s.suggested_tags == ("finance",)

    @pytest.mark.skipif(
        not _mage_installed(),
        reason="mmar_mage not installed",
    )
    def test_real_mage_result_unwraps(self):
        from mmar_mage.schemas import MAGEMetadata, MAGEResult

        meta = MAGEMetadata(
            domain="weather", num_steps=2, memory_hits_used=1
        )
        result = MAGEResult(
            chain_json="{}",
            chain_dict={"steps": []},
            mode="fast",
            metadata=meta,
        )
        s = summarise_mage_result(result)
        assert s.domain == "weather"
        assert s.num_steps == 2
        assert s.memory_hits_used == 1
        assert s.mode == "fast"


# ---------------------------------------------------------------------------
# to_dict round-trip
# ---------------------------------------------------------------------------


class TestToDict:
    def test_round_trip(self):
        s = MetadataSummary(
            domain="weather",
            num_steps=2,
            stages_completed=("plan",),
            step_critique_score=0.7,
            suggested_tags=("alpha",),
        )
        d = s.to_dict()
        # Tuples become lists for JSON friendliness.
        assert isinstance(d["stages_completed"], list)
        assert isinstance(d["suggested_tags"], list)
        # Optional `None` fields included so consumers know the
        # full schema.
        assert d["verification_passed"] is None
        assert d["tot_branches_explored"] is None
