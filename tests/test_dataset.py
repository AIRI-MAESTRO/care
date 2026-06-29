"""Tests for the reusable dataset-entry helpers."""

from __future__ import annotations

import json

from care.dataset import (
    DATASET_ENTRY_PREFIX,
    add_dataset_entry,
    collect_dataset_entries,
    entry_passes,
    export_entries_jsonl,
)


class _Mem:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.saved = []

    def list_entities(self, **kwargs):
        return self.rows

    def save_memory_card(self, content, *, name, tags, when_to_use=None):
        self.saved.append({"content": content, "name": name, "tags": tags})
        return "card-1"


class TestCollect:
    def test_filters_by_chain_tag(self):
        mem = _Mem(
            [
                {
                    "entity_id": "m1",
                    "tags": [f"{DATASET_ENTRY_PREFIX}c1"],
                    "content": {"task": "t", "expected": "foo", "status": "pass"},
                },
                {"entity_id": "m2", "tags": ["unrelated"], "content": {}},
            ]
        )
        out = collect_dataset_entries(mem, "c1")
        assert len(out) == 1
        assert out[0]["task"] == "t"
        assert out[0]["expected"] == "foo"
        assert out[0]["status"] == "pass"


class TestAdd:
    def test_saves_tagged_card(self):
        mem = _Mem()
        eid = add_dataset_entry(mem, "c1", "do x", "result-y", rubric="be terse")
        assert eid == "card-1"
        card = mem.saved[0]
        assert f"{DATASET_ENTRY_PREFIX}c1" in card["tags"]
        assert "scorer:rubric" in card["tags"]
        assert card["content"]["task"] == "do x"
        assert card["content"]["expected"] == "result-y"


class TestScorer:
    def test_substring_case_insensitive(self):
        assert entry_passes("The FOO bar", "foo") is True
        assert entry_passes("nope", "foo") is False

    def test_empty_expected_fails(self):
        assert entry_passes("anything", "") is False


class TestExport:
    def test_writes_jsonl(self, tmp_path):
        entries = [
            {"task": "t1", "expected": "e1", "status": "pass", "actual": "e1"},
            {"task": "t2", "expected": "e2", "status": "fail", "actual": "x"},
        ]
        out = tmp_path / "ds.jsonl"
        n = export_entries_jsonl(entries, out)
        assert n == 2
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["task"] == "t1" and first["expected"] == "e1"
