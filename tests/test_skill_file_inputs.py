"""Tests for care.skill_file_inputs — detecting + wiring file-consuming
document-skill steps (the chat "ran without my file" guard)."""

from __future__ import annotations

import pytest

from care.skill_file_inputs import (
    apply_file_inputs,
    classify_reads_intent,
    file_consuming_steps,
    reads_predicate,
    requires_file_input,
    skill_slug,
)
from care.skill_file_inputs import _parse_verdicts


def _doc_step(
    *,
    skill: str = "github://anthropics/skills/skills/docx@main",
    title: str = "Извлечь текст из DOCX",
    aim: str = "извлечь полный текст",
    task: str = "Extract all plain text from the provided DOCX file",
    input_mapping: dict | None = None,
    execution_mode: str = "llm",
) -> dict:
    cfg: dict = {
        "skill": skill,
        "task": task,
        "execution_mode": execution_mode,
        "output_key": "extracted_text",
    }
    if input_mapping is not None:
        cfg["input_mapping"] = input_mapping
    return {
        "number": 1,
        "step_type": "agent_skill",
        "title": title,
        "aim": aim,
        "step_config": cfg,
    }


def _chain(*steps: dict) -> dict:
    return {"steps": list(steps)}


# ---------------------------------------------------------------------------
# skill_slug
# ---------------------------------------------------------------------------


class TestSkillSlug:
    def test_github_uri(self):
        assert skill_slug("github://anthropics/skills/skills/docx@main") == "docx"

    def test_plain_slug(self):
        assert skill_slug("pdf-extractor") == "pdf-extractor"

    def test_empty(self):
        assert skill_slug("") == ""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestDetection:
    def test_read_docx_empty_mapping(self):
        # The shape MAGE actually emits: llm mode, empty input_mapping.
        assert requires_file_input(_chain(_doc_step(input_mapping={}))) is True

    def test_read_docx_outer_context_mapping(self):
        step = _doc_step(
            execution_mode="llm_agent",
            input_mapping={"file": "$outer_context"},
            task="Извлечь текст из DOCX переданного в $outer_context",
        )
        assert requires_file_input(_chain(step)) is True

    def test_pdf_and_xlsx_slugs_match(self):
        for slug in ("pdf", "xlsx", "pptx"):
            step = _doc_step(
                skill=f"github://anthropics/skills/skills/{slug}@main",
                title="Extract", aim="read the file",
                task="Extract content from the provided file",
            )
            assert requires_file_input(_chain(step)) is True

    def test_structured_skill_ref_detected(self):
        # ReasoningChain.to_dict() serialises `skill` as a dict, not a URI
        # string — detection must handle both (library/CLI round-trip path).
        step = {
            "number": 1, "step_type": "agent_skill",
            "title": "Extract", "aim": "read the document",
            "step_config": {
                "skill": {
                    "git_subdirectory": "skills/docx",
                    "git_url": "https://github.com/anthropics/skills",
                },
                "task": "Extract text from the provided file",
                "input_mapping": {},
            },
        }
        assert requires_file_input(_chain(step)) is True

    def test_create_doc_not_flagged(self):
        step = _doc_step(
            title="Create a Word report",
            aim="create a docx file",
            task="Create a new Word document with the summary",
        )
        assert requires_file_input(_chain(step)) is False

    def test_read_then_create_is_flagged(self):
        step = _doc_step(
            title="Read docx and create summary",
            aim="extract text then create a summary",
            task="Extract text from the provided file and create a summary",
        )
        assert requires_file_input(_chain(step)) is True

    def test_non_doc_skill_not_flagged(self):
        step = _doc_step(skill="github://x/skills/web-research@main")
        # Override slug to a non-doc skill via the skill field.
        step["step_config"]["skill"] = "github://x/skills/web-research@main"
        assert requires_file_input(_chain(step)) is False

    def test_non_agent_skill_step_ignored(self):
        llm = {"number": 1, "step_type": "llm", "step_config": {}}
        assert requires_file_input(_chain(llm)) is False

    def test_empty_chain(self):
        assert requires_file_input({}) is False
        assert requires_file_input({"steps": []}) is False
        assert file_consuming_steps(None) == []


# ---------------------------------------------------------------------------
# apply_file_inputs
# ---------------------------------------------------------------------------


class _StubApi:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[str] = []

    async def get_response_with_retries(self, prompt, retries):
        self.calls.append(prompt)
        return self.reply


class TestModelClassification:
    def test_parse_verdicts(self):
        assert _parse_verdicts('{"1": true, "2": false}') == {1: True, 2: False}
        assert _parse_verdicts('noise {"3": true} tail') == {3: True}
        assert _parse_verdicts("not json") == {}
        assert _parse_verdicts(None) == {}

    def test_model_overrides_heuristic(self):
        # A "create" step the heuristic would NOT flag — but the model says it
        # reads → it gets flagged.
        step = _doc_step(
            title="Create a Word report",
            aim="create a docx file",
            task="Create a new Word document",
        )
        chain = _chain(step)
        assert file_consuming_steps(chain) == []  # heuristic: not a read
        flagged = file_consuming_steps(chain, reads=reads_predicate({1: True}))
        assert len(flagged) == 1

    def test_model_negative_overrides_heuristic(self):
        # A "read" step the heuristic flags — but the model says it doesn't.
        chain = _chain(_doc_step())  # "extract" → heuristic reads
        assert file_consuming_steps(chain)  # heuristic: reads
        flagged = file_consuming_steps(chain, reads=reads_predicate({1: False}))
        assert flagged == []

    def test_unclassified_falls_back_to_heuristic(self):
        chain = _chain(_doc_step())  # extract → heuristic reads
        # verdicts empty → predicate returns None for step 1 → heuristic used
        flagged = file_consuming_steps(chain, reads=reads_predicate({}))
        assert len(flagged) == 1

    @pytest.mark.asyncio
    async def test_classify_reads_intent_calls_llm(self):
        from care.skill_file_inputs import doc_skill_steps

        steps = doc_skill_steps(_chain(_doc_step()))
        api = _StubApi('{"1": true}')
        verdicts = await classify_reads_intent(api, steps)
        assert verdicts == {1: True}
        assert api.calls  # the LLM was actually consulted

    @pytest.mark.asyncio
    async def test_classify_reads_intent_no_api_or_steps(self):
        assert await classify_reads_intent(None, [{"number": 1}]) == {}
        assert await classify_reads_intent(_StubApi("{}"), []) == {}

    @pytest.mark.asyncio
    async def test_classify_reads_intent_bad_reply_falls_back(self):
        api = _StubApi("the model rambled with no json")
        steps = [{"number": 1, "step_config": {}}]
        assert await classify_reads_intent(api, steps) == {}


class TestApplyFileInputs:
    def test_rewrites_empty_mapping_to_memory_input(self):
        chain = _chain(_doc_step(input_mapping={}))
        new, files = apply_file_inputs(chain, [("/abs/report.docx", "TXT")])
        assert files == {"report.docx": "TXT"}
        assert new["steps"][0]["step_config"]["input_mapping"] == {
            "file": "$memory.input.report.docx",
        }

    def test_reuses_existing_file_param(self):
        chain = _chain(
            _doc_step(input_mapping={"file": "$outer_context"}),
        )
        new, files = apply_file_inputs(chain, [("/x/Q3.docx", "T")])
        assert new["steps"][0]["step_config"]["input_mapping"]["file"] == (
            "$memory.input.Q3.docx"
        )

    def test_does_not_mutate_original(self):
        chain = _chain(_doc_step(input_mapping={}))
        original = chain["steps"][0]["step_config"]["input_mapping"]
        apply_file_inputs(chain, [("/a/b.docx", "x")])
        assert original == {}  # untouched

    def test_sanitises_key_with_spaces(self):
        chain = _chain(_doc_step(input_mapping={}))
        new, files = apply_file_inputs(chain, [("/p/My Report.docx", "x")])
        assert "My_Report.docx" in files
        assert new["steps"][0]["step_config"]["input_mapping"]["file"] == (
            "$memory.input.My_Report.docx"
        )

    def test_no_attachments_is_noop(self):
        chain = _chain(_doc_step(input_mapping={}))
        new, files = apply_file_inputs(chain, [])
        assert files == {}
        assert new is chain

    def test_no_file_steps_is_noop(self):
        chain = _chain({"number": 1, "step_type": "llm", "step_config": {}})
        new, files = apply_file_inputs(chain, [("/a/b.txt", "x")])
        assert files == {}

    def test_injects_task_placeholder(self):
        # The resolved input only reaches the LLM if the task references it as
        # {param} (CARL's _interpolate_task does task.format_map).
        chain = _chain(_doc_step(input_mapping={}, task="Extract text from the DOCX"))
        new, _ = apply_file_inputs(chain, [("/a/r.docx", "BODY")])
        task = new["steps"][0]["step_config"]["task"]
        assert task.rstrip().endswith("{file}")
        assert "r.docx" in task
        # Interpolates to the document text under the CARL contract.
        assert task.format_map({"file": "BODY"}).rstrip().endswith("BODY")

    def test_task_injection_escapes_existing_braces(self):
        # A stray brace in the original task must not break format_map nor
        # discard the substitution (KeyError → task returned as-is).
        chain = _chain(
            _doc_step(input_mapping={}, task="Read the file, return {x: 1}"),
        )
        new, _ = apply_file_inputs(chain, [("/a/r.docx", "BODY")])
        task = new["steps"][0]["step_config"]["task"]
        out = task.format_map({"file": "DOC"})
        assert "{x: 1}" in out
        assert out.rstrip().endswith("DOC")

    def test_task_injection_idempotent(self):
        chain = _chain(
            _doc_step(input_mapping={"file": "$x"}, task="Read {file}"),
        )
        new, _ = apply_file_inputs(chain, [("/a/r.docx", "BODY")])
        # Already references {file} → task left unchanged.
        assert new["steps"][0]["step_config"]["task"] == "Read {file}"

    def test_multiple_attachments_dedupe_keys(self):
        chain = _chain(_doc_step(input_mapping={}))
        new, files = apply_file_inputs(
            chain, [("/a/doc.docx", "A"), ("/b/doc.docx", "B")],
        )
        # Both kept under distinct keys; the step binds to the first.
        assert files["doc.docx"] == "A"
        assert files["doc.docx-2"] == "B"
        assert new["steps"][0]["step_config"]["input_mapping"]["file"] == (
            "$memory.input.doc.docx"
        )
