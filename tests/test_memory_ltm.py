"""LTM engine (:mod:`care.memory_ltm`) — attach, recall-digest, and the
conservative post-turn save-decision (dedup / supersede).

Deterministic: an ``InMemoryLTM`` store + a stub ``complete`` callable stand in
for CARL's file store and the LLM — no disk, no live OpenAI.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from mmar_carl import InMemoryLTM

from care import memory_ltm


def _cfg(**ctx) -> SimpleNamespace:
    base = dict(ltm_enabled=True, ltm_dir="~/.config/care/ltm", ltm_session_id="s1")
    base.update(ctx)
    return SimpleNamespace(context=SimpleNamespace(**base))


def _complete_returning(payload) -> object:
    """A stub ``complete`` that returns ``payload`` (dict→JSON, or str verbatim)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return lambda system, user: text


class TestBuildLtm:
    def test_disabled_returns_none(self):
        assert memory_ltm.build_ltm(_cfg(ltm_enabled=False)) is None

    def test_session_id_from_config(self):
        assert memory_ltm.ltm_session_id(_cfg(ltm_session_id="bob")) == "bob"
        assert memory_ltm.ltm_session_id(SimpleNamespace(context=None)) == "default"

    def test_enabled_builds_jsonfile_ltm(self, tmp_path):
        ltm = memory_ltm.build_ltm(_cfg(ltm_dir=str(tmp_path / "ltm")))
        assert ltm is not None
        ltm.store("role", "engineer", session_id="s1")
        assert ltm.retrieve("role", session_id="s1") == "engineer"


class TestRecallDigest:
    def test_empty_store_is_blank(self):
        assert memory_ltm.recall_digest(InMemoryLTM(), "s1") == ""
        assert memory_ltm.recall_digest(None, "s1") == ""

    def test_lists_stored_facts(self):
        ltm = InMemoryLTM()
        ltm.store("role", "data scientist", session_id="s1")
        ltm.store("preferred_language", "Russian", session_id="s1")
        digest = memory_ltm.recall_digest(ltm, "s1")
        assert "role: data scientist" in digest
        assert "preferred_language: Russian" in digest
        assert digest.startswith("## What I remember")

    def test_scoped_by_session(self):
        ltm = InMemoryLTM()
        ltm.store("role", "x", session_id="s1")
        assert memory_ltm.recall_digest(ltm, "s2") == ""  # different scope

    def test_capped_at_max_chars(self):
        ltm = InMemoryLTM()
        ltm.store("k", "v" * 5000, session_id="s1")
        digest = memory_ltm.recall_digest(ltm, "s1", max_chars=200)
        assert len(digest) <= 202  # cap + the " …" marker
        assert memory_ltm.recall_digest(ltm, "s1", max_chars=0) == ""


class TestDecideFacts:
    def test_parses_save_payload(self):
        out = memory_ltm.decide_facts(
            _complete_returning({"save": True, "facts": [{"key": "role", "value": "PM"}]}),
            query="I'm a product manager",
        )
        assert out["save"] is True
        assert out["facts"] == [{"key": "role", "value": "PM"}]

    def test_no_save(self):
        out = memory_ltm.decide_facts(
            _complete_returning({"save": False, "facts": []}), query="what's 2+2?",
        )
        assert out == {"save": False, "facts": []}

    def test_bad_json_is_no_save(self):
        out = memory_ltm.decide_facts(
            _complete_returning("sorry I can't do that"), query="x",
        )
        assert out == {"save": False, "facts": []}

    def test_fenced_json_extracted(self):
        out = memory_ltm.decide_facts(
            _complete_returning('```json\n{"save": true, "facts": [{"key":"a","value":"b"}]}\n```'),
            query="x",
        )
        assert out["save"] is True and out["facts"][0]["key"] == "a"

    def test_complete_raising_is_no_save(self):
        def _boom(system, user):
            raise RuntimeError("llm down")

        assert memory_ltm.decide_facts(_boom, query="x") == {"save": False, "facts": []}


class TestApplyFacts:
    def test_stores_new_fact(self):
        ltm = InMemoryLTM()
        saved = memory_ltm.apply_facts(ltm, "s1", [{"key": "role", "value": "engineer"}])
        assert saved == [{"key": "role", "value": "engineer", "superseded": False}]
        assert ltm.retrieve("role", session_id="s1") == "engineer"

    def test_dedup_identical_skipped(self):
        ltm = InMemoryLTM()
        ltm.store("role", "engineer", session_id="s1")
        saved = memory_ltm.apply_facts(ltm, "s1", [{"key": "role", "value": "engineer"}])
        assert saved == []  # already remembered, identically

    def test_supersede_changed_value(self):
        ltm = InMemoryLTM()
        ltm.store("role", "engineer", session_id="s1")
        saved = memory_ltm.apply_facts(ltm, "s1", [{"key": "role", "value": "manager"}])
        assert saved == [{"key": "role", "value": "manager", "superseded": True}]
        assert ltm.retrieve("role", session_id="s1") == "manager"

    def test_skips_blank_or_malformed(self):
        ltm = InMemoryLTM()
        saved = memory_ltm.apply_facts(
            ltm, "s1",
            [{"key": "", "value": "x"}, {"key": "k", "value": ""}, "not-a-dict", {"key": "ok", "value": "yes"}],
        )
        assert saved == [{"key": "ok", "value": "yes", "superseded": False}]


class TestSaveFromTurn:
    def test_end_to_end_save(self):
        ltm = InMemoryLTM()
        saved = memory_ltm.save_from_turn(
            ltm, "s1",
            query="By the way I always want answers in Russian.",
            answer="...",
            complete=_complete_returning(
                {"save": True, "facts": [{"key": "preferred_language", "value": "Russian"}]},
            ),
        )
        assert [s["key"] for s in saved] == ["preferred_language"]
        assert ltm.retrieve("preferred_language", session_id="s1") == "Russian"

    def test_no_save_writes_nothing(self):
        ltm = InMemoryLTM()
        saved = memory_ltm.save_from_turn(
            ltm, "s1", query="what time is it?",
            complete=_complete_returning({"save": False, "facts": []}),
        )
        assert saved == []
        assert ltm.keys(session_id="s1") == []

    def test_none_ltm_noops(self):
        assert memory_ltm.save_from_turn(
            None, "s1", query="x", complete=_complete_returning({"save": True, "facts": []}),
        ) == []

    def test_format_saved(self):
        assert memory_ltm.format_saved([]) == ""
        line = memory_ltm.format_saved(
            [{"key": "role", "value": "x"}, {"key": "lang", "value": "y"}],
        )
        assert line == "🧠 remembered: role, lang"


class TestRememberText:
    """Explicit `#…` / `/remember` — LLM-merge into memory with reconcile +
    a no-loss fallback."""

    def test_merges_and_stores(self):
        ltm = InMemoryLTM()
        saved = memory_ltm.remember_text(
            ltm, "s1",
            content="I always want answers in Russian",
            complete=_complete_returning(
                {"facts": [{"key": "preferred_language", "value": "Russian"}]},
            ),
        )
        assert [s["key"] for s in saved] == ["preferred_language"]
        assert ltm.retrieve("preferred_language", session_id="s1") == "Russian"

    def test_supersedes_existing_on_contradiction(self):
        ltm = InMemoryLTM()
        ltm.store("role", "engineer", session_id="s1")
        saved = memory_ltm.remember_text(
            ltm, "s1",
            content="actually I'm a manager now",
            # the LLM reuses the existing key to supersede
            complete=_complete_returning(
                {"facts": [{"key": "role", "value": "manager"}]},
            ),
        )
        assert saved == [{"key": "role", "value": "manager", "superseded": True}]
        assert ltm.retrieve("role", session_id="s1") == "manager"

    def test_fallback_stores_raw_when_llm_fails(self):
        ltm = InMemoryLTM()

        def _boom(system, user):
            raise RuntimeError("llm down")

        saved = memory_ltm.remember_text(
            ltm, "s1", content="My cat is named Pixel", complete=_boom,
        )
        # explicit note never silently lost — stored under a derived key
        assert len(saved) == 1
        assert saved[0]["key"].startswith("note_")
        assert "Pixel" in ltm.retrieve(saved[0]["key"], session_id="s1")

    def test_empty_content_noop(self):
        ltm = InMemoryLTM()
        assert memory_ltm.remember_text(
            ltm, "s1", content="   ", complete=_complete_returning({"facts": []}),
        ) == []
