"""Tests for gigavolve Redis live probes."""

from __future__ import annotations

import json
from unittest.mock import patch

from care.runtime.evolution_redis_probe import (
    extract_chain_config_from_program_code,
    probe_best_fitness,
    probe_fitness_history,
    probe_ga_generation,
    probe_program_chain_config,
    probe_programs_counts,
)


class TestProbeGaGeneration:
    def test_uses_program_lineage_not_engine_counter(self) -> None:
        program = {
            "lineage": {"generation": 3},
            "metrics": {"fitness": 0.2, "is_valid": 1.0},
        }

        def fake_keys(pattern: str) -> list[str]:
            assert pattern.endswith(":program:*")
            return ["abc:program:seed"]

        def fake_command(*args: str) -> str | None:
            if args[:2] == ("GET", "abc:program:seed"):
                return json.dumps(program)
            return ""

        with patch(
            "care.runtime.evolution_redis_probe._redis_keys",
            side_effect=fake_keys,
        ), patch(
            "care.runtime.evolution_redis_probe._redis_command",
            side_effect=fake_command,
        ):
            assert probe_ga_generation("exp_abc") == 2

    def test_seed_program_shows_generation_zero(self) -> None:
        program = {"lineage": {"generation": 1}, "metrics": {"is_valid": 1.0, "fitness": 0.0}}

        with patch(
            "care.runtime.evolution_redis_probe._redis_keys",
            return_value=["p:program:1"],
        ), patch(
            "care.runtime.evolution_redis_probe._redis_command",
            return_value=json.dumps(program),
        ):
            assert probe_ga_generation("exp_p") == 0


class TestProbeBestFitness:
    def test_reads_valid_program_metrics(self) -> None:
        programs = {
            "p:program:a": {"metrics": {"is_valid": 1.0, "fitness": 0.15}},
            "p:program:b": {"metrics": {"is_valid": 1.0, "fitness": 0.42}},
        }

        def fake_keys(pattern: str) -> list[str]:
            return list(programs.keys())

        def fake_command(*args: str) -> str | None:
            if args[0] == "GET":
                return json.dumps(programs[args[1]])
            return ""

        with patch(
            "care.runtime.evolution_redis_probe._redis_keys",
            side_effect=fake_keys,
        ), patch(
            "care.runtime.evolution_redis_probe._redis_command",
            side_effect=fake_command,
        ):
            assert probe_best_fitness("exp_p") == 0.42


class TestProbeProgramsCounts:
    def test_reads_latest_valid_invalid_counts(self) -> None:
        lists = {
            "p:metrics:history:program_metrics:programs_valid_count": [
                json.dumps({"s": 0, "v": 3, "k": "scalar"}),
                json.dumps({"s": 1, "v": 7, "k": "scalar"}),
            ],
            "p:metrics:history:program_metrics:programs_invalid_count": [
                json.dumps({"s": 1, "v": 2, "k": "scalar"}),
            ],
        }

        def fake_command(*args: str) -> str | None:
            # _last_scalar_int issues LRANGE key -1 -1.
            if args[0] == "LRANGE" and args[2] == "-1" and args[3] == "-1":
                return lists.get(args[1], [])[-1] if lists.get(args[1]) else ""
            return ""

        with patch(
            "care.runtime.evolution_redis_probe._redis_command",
            side_effect=fake_command,
        ):
            assert probe_programs_counts("exp_p") == (7, 2)

    def test_missing_counters_return_none(self) -> None:
        with patch(
            "care.runtime.evolution_redis_probe._redis_command",
            return_value="",
        ):
            assert probe_programs_counts("exp_p") == (None, None)

    def test_non_exp_id_returns_none(self) -> None:
        assert probe_programs_counts("legacy-id") == (None, None)


class TestProbeFitnessHistory:
    def test_builds_history_from_gen_mean_list(self) -> None:
        rows = [
            json.dumps({"s": 0, "v": 0.10, "k": "scalar"}),
            json.dumps({"s": 1, "v": 0.25, "k": "scalar"}),
            json.dumps({"s": 2, "v": -1000, "k": "scalar"}),  # dropped
        ]

        with patch(
            "care.runtime.evolution_redis_probe._redis_lrange",
            return_value=rows,
        ):
            history = probe_fitness_history("exp_p")
        assert history == [
            {"generation": 0, "best_fitness": 0.10},
            {"generation": 1, "best_fitness": 0.25},
        ]


class TestExtractChainConfigFromProgramCode:
    def test_reads_base_chain_config_literal(self) -> None:
        code = (
            'BASE_CHAIN_CONFIG: str = """'
            '{"name": "x", "steps": [{"number": 1, "title": "t"}]}'
            '"""'
        )
        chain = extract_chain_config_from_program_code(code)
        assert chain is not None
        assert len(chain["steps"]) == 1

    def test_extracts_from_call_program_pickle_blob(self) -> None:
        import base64

        payload = {
            "chain_config_json": json.dumps(
                {
                    "name": "evolved",
                    "steps": [{"number": 1}, {"number": 2}],
                },
            ),
        }
        blob = b"prefix chain_config_json\x00\x00" + payload["chain_config_json"].encode()
        program = {
            "code": "BASE_CHAIN_CONFIG: str = \"\"\"{broken\"\"\"",
            "stage_results": {
                "CallProgramFunction": {
                    "status": "completed",
                    "output": base64.b64encode(blob).decode(),
                },
            },
        }

        def fake_command(*args: str) -> str | None:
            if args[:2] == ("GET", "p:program:abc"):
                return json.dumps(program)
            return ""

        with patch(
            "care.runtime.evolution_redis_probe._redis_command",
            side_effect=fake_command,
        ):
            chain = probe_program_chain_config("exp_p", "abc")
        assert chain is not None
        assert len(chain["steps"]) == 2
