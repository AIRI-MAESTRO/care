"""Tests for Platform evolution chain adapter."""

from __future__ import annotations

import json
from pathlib import Path

from care.runtime.platform_chain_adapter import prepare_chain_for_platform_evolution
from care.runtime.platform_chain_gate import (
    gate_chain_for_platform_evolution,
    _looks_hardcoded_sample,
)

_CHAT_SUMMARIZER = {
    "version": "1.0",
    "task_description": "Summarize text from examples file",
    "steps": [
        {
            "number": 1,
            "title": "Read examples file",
            "step_type": "tool",
            "step_config": {
                "tool_name": "run_python",
                "input_mapping": {
                    "code": (
                        "with open('/home/volkova/care/examples/summarizer/eval.jsonl', 'r') "
                        "as f: print(f.read())"
                    ),
                },
            },
        },
        {
            "number": 2,
            "title": "Parse examples",
            "step_type": "structured_output",
            "dependencies": [1],
            "step_config": {
                "instruction": "Parse JSONL into input/expected pairs",
                "input_source": "$history[0]",
            },
        },
        {
            "number": 3,
            "title": "Generate summary",
            "step_type": "llm",
            "dependencies": [2],
            "aim": (
                "Produce a concise Russian summary of the target text "
                "'Компания выпустила новый смартфон с улучшенной камерой и батареей на два дня.', "
                "strictly following the style demonstrated in the provided examples"
            ),
            "stage_action": "Output a single-line Russian summary of the target text",
        },
    ],
}


class TestPlatformChainAdapter:
    def test_chat_summarizer_adapts_to_runnable_llm(self) -> None:
        prepared = prepare_chain_for_platform_evolution(_CHAT_SUMMARIZER)
        assert prepared.adapted
        assert gate_chain_for_platform_evolution(prepared.chain) == []
        steps = prepared.chain["steps"]
        assert len(steps) == 1
        assert steps[0]["step_type"] == "llm"
        blob = json.dumps(prepared.chain, ensure_ascii=False)
        assert "/home/volkova" not in blob
        assert "run_python" not in blob

    def test_platform_seed_unchanged(self) -> None:
        raw = Path("examples/summarizer/chain.json").read_text(encoding="utf-8")
        chain = json.loads(raw)
        prepared = prepare_chain_for_platform_evolution(chain)
        assert gate_chain_for_platform_evolution(prepared.chain) == []
        assert prepared.chain["steps"][0]["title"] == "Summarize paragraph"

    def test_bundled_synth_tool_kept_in_chain(self) -> None:
        chain = {
            "version": "1.0",
            "steps": [
                {
                    "number": 1,
                    "title": "Fetch rate",
                    "step_type": "tool",
                    "step_config": {
                        "tool_name": "get_exchange_rate",
                        "input_mapping": {"pair": "$sample.input"},
                    },
                },
                {
                    "number": 2,
                    "title": "Answer",
                    "step_type": "llm",
                    "dependencies": [1],
                    "aim": "Use the rate from context",
                    "stage_action": "Reply with one sentence using $history[-1]",
                },
            ],
        }
        prepared = prepare_chain_for_platform_evolution(
            chain,
            bundled_tool_names=frozenset({"get_exchange_rate"}),
        )
        assert len(prepared.chain["steps"]) == 2
        assert prepared.chain["steps"][0]["step_config"]["tool_name"] == "get_exchange_rate"
        assert gate_chain_for_platform_evolution(
            prepared.chain,
            bundled_tool_names=frozenset({"get_exchange_rate"}),
        ) == []

    def test_weather_mcp_becomes_llm(self) -> None:
        raw = Path("examples/weather/chain.json").read_text(encoding="utf-8")
        chain = json.loads(raw)
        prepared = prepare_chain_for_platform_evolution(chain)
        assert gate_chain_for_platform_evolution(prepared.chain) == []
        types = [s["step_type"] for s in prepared.chain["steps"]]
        assert types == ["llm", "llm"]
        assert "$inputs" not in json.dumps(prepared.chain)

    def test_collapse_uses_generic_aim_when_task_has_examples(self) -> None:
        chain = {
            "version": "1.0",
            "metadata": {
                "description": (
                    "мне нужна цепочка summarization\n"
                    '<file path="/home/volkova/care/examples/summarizer/eval.jsonl">\n'
                    '{"input": "Климатические изменения ускоряют таяние ледников. '
                    "Это повышает уровень моря и угрожает прибрежным городам.\", "
                    '"expected": "Таяние ледников из-за изменения климата..."}'
                ),
            },
            "steps": [
                {
                    "number": 1,
                    "step_type": "tool",
                    "step_config": {
                        "tool_name": "run_python",
                        "input_mapping": {"code": "open('/home/volkova/x.jsonl')"},
                    },
                },
                {
                    "number": 2,
                    "step_type": "llm",
                    "dependencies": [1],
                    "aim": "summarize",
                    "stage_action": "one line",
                },
            ],
        }
        prepared = prepare_chain_for_platform_evolution(chain)
        assert gate_chain_for_platform_evolution(prepared.chain) == []
        aim = prepared.chain["steps"][0]["aim"]
        assert not _looks_hardcoded_sample(aim)
        assert "outer_context" in prepared.chain["steps"][0]["stage_action"].lower() or "input" in aim.lower()
        chain = {
            "version": "1.0",
            "steps": [
                {
                    "number": 1,
                    "step_type": "tool",
                    "title": "read",
                    "step_config": {
                        "tool_name": "run_python",
                        "input_mapping": {
                            "code": "open('/home/volkova/care/x.jsonl')",
                        },
                    },
                },
                {
                    "number": 2,
                    "step_type": "llm",
                    "dependencies": [1],
                    "aim": "Read /home/volkova/care/examples/summarizer/eval.jsonl",
                    "stage_action": "Summarize input",
                },
            ],
        }
        prepared = prepare_chain_for_platform_evolution(chain)
        assert gate_chain_for_platform_evolution(prepared.chain) == []
        assert "/home/" not in json.dumps(prepared.chain)
        assert len(prepared.chain["steps"]) == 1

    def test_inputs_refs_rewritten(self) -> None:
        chain = {
            "version": "1.0",
            "steps": [
                {
                    "number": 1,
                    "title": "retrieve",
                    "step_type": "tool",
                    "step_config": {
                        "tool_name": "retrieve",
                        "input_mapping": {"query": "$inputs.city"},
                    },
                },
                {
                    "number": 2,
                    "title": "answer",
                    "step_type": "llm",
                    "dependencies": [1],
                    "aim": "Summarize weather",
                    "stage_action": "Use retrieved context",
                },
            ],
        }
        prepared = prepare_chain_for_platform_evolution(chain)
        mapping = prepared.chain["steps"][0]["step_config"]["input_mapping"]
        assert mapping["query"] == "$sample.city"
        assert gate_chain_for_platform_evolution(prepared.chain) == []
