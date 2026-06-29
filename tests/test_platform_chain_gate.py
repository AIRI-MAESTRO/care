"""Platform evolution chain preflight gate."""

from __future__ import annotations

from care.runtime.platform_chain_gate import gate_chain_for_platform_evolution

_GOOD = {
    "version": "1.0",
    "steps": [
        {
            "number": 1,
            "title": "Summarize",
            "step_type": "llm",
            "aim": "One-sentence summary",
            "stage_action": "Summarize the input paragraph from context.",
        }
    ],
}

_BAD_RUN_PYTHON = {
    "version": "1.0",
    "steps": [
        {
            "number": 1,
            "title": "Read file",
            "step_type": "tool",
            "step_config": {
                "tool_name": "run_python",
                "input_mapping": {
                    "code": "with open('/home/volkova/care/examples/summarizer/eval.jsonl') as f: print(f.read())",
                },
            },
        }
    ],
}

_BAD_HARDCODED = {
    "version": "1.0",
    "steps": [
        {
            "number": 1,
            "title": "Summarize",
            "step_type": "llm",
            "aim": (
                "Produce a concise Russian summary of the target text "
                "'Компания выпустила новый смартфон с улучшенной камерой и батареей на два дня.', "
                "strictly following the style demonstrated in the provided examples"
            ),
            "stage_action": "Output a single-line Russian summary of the target text",
        }
    ],
}


class TestPlatformChainGate:
    def test_good_chain_passes(self) -> None:
        assert gate_chain_for_platform_evolution(_GOOD) == []

    def test_run_python_host_path_blocked(self) -> None:
        issues = gate_chain_for_platform_evolution(_BAD_RUN_PYTHON)
        assert issues
        joined = " ".join(issues).lower()
        assert "run_python" in joined
        assert "unknown reference" in joined or "$-reference" in joined

    def test_hardcoded_sample_blocked(self) -> None:
        issues = gate_chain_for_platform_evolution(_BAD_HARDCODED)
        assert any("fixed sample" in i for i in issues)

    def test_structured_output_blocked(self) -> None:
        chain = {
            "version": "1.0",
            "steps": [
                {
                    "number": 2,
                    "title": "Parse",
                    "step_type": "structured_output",
                    "step_config": {"input_source": "$history[0]"},
                }
            ],
        }
        issues = gate_chain_for_platform_evolution(chain)
        assert any("structured_output" in i for i in issues)
