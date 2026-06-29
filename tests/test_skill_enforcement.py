"""Deterministic skill selection (:mod:`care.skill_enforcement`).

Detection (explicit `/skill` / «используй скилл» / file-keywords + produce verb)
and the chain rewrite that guarantees an `agent_skill` step. No LLM / network.
"""

from __future__ import annotations

from care.skill_enforcement import detect_requested_skill, ensure_skill_step


class TestDetect:
    def test_explicit_slash_skill(self):
        assert detect_requested_skill("сделай доклад /skill pptx") == "pptx"

    def test_explicit_russian_phrase(self):
        assert detect_requested_skill("используй скилл pptx, лекция") == "pptx"
        assert detect_requested_skill("используйте скилл docx") == "docx"

    def test_explicit_english_phrase_and_alias(self):
        assert detect_requested_skill("use the powerpoint skill please") == "pptx"
        assert detect_requested_skill("use skill xlsx") == "xlsx"

    def test_implicit_file_with_produce_verb(self):
        assert detect_requested_skill("Сделай PPTX: лекция по вероятности") == "pptx"
        assert detect_requested_skill("создай презентацию про ЦПТ") == "pptx"
        assert detect_requested_skill("make a word document about X") == "docx"
        assert detect_requested_skill("сгенерируй таблицу excel с данными") == "xlsx"

    def test_no_produce_verb_does_not_trigger(self):
        # a passing mention of a format must NOT force a skill
        assert detect_requested_skill("объясни, что такое pptx") is None
        assert detect_requested_skill("в чём разница pdf и docx?") is None

    def test_nothing_requested(self):
        assert detect_requested_skill("реши уравнение x^2=4") is None
        assert detect_requested_skill("") is None


class TestEnsureSkillStep:
    @staticmethod
    def _chain():
        return {
            "steps": [
                {"number": 1, "step_type": "llm", "title": "Draft", "dependencies": []},
                {
                    "number": 2, "step_type": "llm",
                    "title": "Format as PowerPoint", "aim": "format the deck",
                    "dependencies": [1], "llm_config": {"x": 1},
                },
            ],
        }

    def test_converts_final_step_to_agent_skill(self):
        out = ensure_skill_step(self._chain(), "pptx")
        last = out["steps"][-1]
        assert last["step_type"] == "agent_skill"
        cfg = last["step_config"]
        assert cfg["skill"] == "github://anthropics/skills/skills/pptx@main"
        assert cfg["execution_mode"] == "llm_agent"
        assert cfg["task"].startswith("format the deck")  # reused the step's aim
        assert "$history[-1]" in cfg["input_mapping"]["content"]
        assert "llm_config" not in last  # stale config dropped
        # earlier content step untouched
        assert out["steps"][0]["step_type"] == "llm"

    def test_sets_file_production_flags_on_new_step(self):
        # a file skill must be forced to actually write a file
        cfg = ensure_skill_step(self._chain(), "pptx")["steps"][-1]["step_config"]
        assert cfg["require_output_file"] is True
        assert cfg["persist_workspace"] is True
        assert cfg["execution_mode"] == "llm_agent"
        # the task is rewritten to demand a real file, not prose
        assert "ACTUAL" in cfg["task"] and "/workspace/out/" in cfg["task"]

    def test_sets_file_production_flags_on_existing_step(self):
        # planner already chose the skill (bare name) → normalise + force a file
        chain = {
            "steps": [{
                "number": 1, "step_type": "agent_skill",
                "step_config": {"skill": "pptx", "task": "build slides"},
            }],
        }
        cfg = ensure_skill_step(chain, "pptx")["steps"][0]["step_config"]
        assert cfg["skill"] == "github://anthropics/skills/skills/pptx@main"
        assert cfg["require_output_file"] is True
        assert cfg["persist_workspace"] is True
        assert "build slides" in cfg["task"]  # original intent preserved

    def test_noop_when_already_uses_skill(self):
        chain = {
            "steps": [{
                "number": 1, "step_type": "agent_skill",
                "step_config": {"skill": "github://anthropics/skills/skills/pptx@main"},
            }],
        }
        out = ensure_skill_step(chain, "pptx")
        assert out["steps"][0]["step_config"]["skill"].endswith("pptx@main")
        assert len(out["steps"]) == 1  # nothing appended/changed

    def test_fixes_bare_skill_ref_to_uri(self):
        # the planner chose agent_skill but named the skill "pptx" (bare) —
        # CARL can't resolve that; normalise it to the canonical registry URI.
        chain = {
            "steps": [{
                "number": 1, "step_type": "agent_skill",
                "title": "Создать PPTX-файл через навык pptx",
                "step_config": {"skill": "pptx", "task": "make the deck"},
            }],
        }
        out = ensure_skill_step(chain, "pptx")
        assert (
            out["steps"][0]["step_config"]["skill"]
            == "github://anthropics/skills/skills/pptx@main"
        )
        assert "make the deck" in out["steps"][0]["step_config"]["task"]  # kept

    def test_noop_for_unknown_skill(self):
        chain = self._chain()
        out = ensure_skill_step(chain, "nonesuch-skill")
        assert out["steps"][-1]["step_type"] == "llm"  # unchanged

    def test_noop_for_empty_or_missing(self):
        assert ensure_skill_step({"steps": []}, "pptx") == {"steps": []}
        assert ensure_skill_step({}, "pptx") == {}
        assert ensure_skill_step(self._chain(), None)["steps"][-1]["step_type"] == "llm"
