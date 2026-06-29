"""Memory-backed tool recall in capability priming (MAGE_TODO B2)."""

from __future__ import annotations

from care import capability_priming as cap
from care.config import CareConfig
from care.tool_synthesis import SYNTH_TAG


class _StubMem:
    def __init__(self, hits):  # noqa: ANN001
        self._hits = hits

    def search(self, query, *, entity_type, search_type, top_k):  # noqa: ANN001
        assert entity_type == "agent_skill"
        return self._hits


def _patch_mem(monkeypatch, hits):  # noqa: ANN001
    import care.memory

    class _StubCareMemory:
        @classmethod
        def from_config(cls, config):  # noqa: ANN001
            return _StubMem(hits)

    monkeypatch.setattr(care.memory, "CareMemory", _StubCareMemory)


def test_recall_includes_synth_tagged(monkeypatch):
    _patch_mem(monkeypatch, [
        {"name": "get_stock_price", "tags": [SYNTH_TAG],
         "content": {"description": "fetch a live stock price"}},
    ])
    specs = cap._recall_memory_tool_specs(CareConfig(), "stock price of AAPL")
    assert [s["name"] for s in specs] == ["get_stock_price"]
    assert specs[0]["source"] == "memory"
    assert "stock price" in specs[0]["description"]


def test_recall_skips_untagged(monkeypatch):
    _patch_mem(monkeypatch, [
        {"name": "foreign_skill", "tags": ["other"], "content": {"description": "x"}},
        {"name": "get_stock_price", "tags": [SYNTH_TAG], "content": {}},
    ])
    names = [s["name"] for s in cap._recall_memory_tool_specs(CareConfig(), "q")]
    assert names == ["get_stock_price"]  # only our synth-tagged tools


def test_recall_no_memory_when_base_url_empty():
    cfg = CareConfig()
    cfg.memory.base_url = ""
    assert cap._recall_memory_tool_specs(cfg, "q") == []


def test_recall_swallows_memory_errors(monkeypatch):
    import care.memory

    class _Boom:
        @classmethod
        def from_config(cls, config):  # noqa: ANN001
            raise RuntimeError("memory down")

    monkeypatch.setattr(care.memory, "CareMemory", _Boom)
    assert cap._recall_memory_tool_specs(CareConfig(), "q") == []


def test_build_capabilities_merges_memory_tool(monkeypatch):
    _patch_mem(monkeypatch, [
        {"name": "get_stock_price", "tags": [SYNTH_TAG],
         "content": {"description": "live stock price"}},
    ])
    caps = cap.build_capabilities_for_generation(CareConfig(), query="price of AAPL")
    assert caps is not None
    names = [t["name"] for t in caps.tools]
    assert "get_stock_price" in names  # recalled from Memory
    assert "web_search" in names       # builtins still advertised


def test_recall_skipped_when_flag_off(monkeypatch):
    _patch_mem(monkeypatch, [
        {"name": "get_stock_price", "tags": [SYNTH_TAG], "content": {}},
    ])
    cfg = CareConfig()
    cfg.tools.recall_tools_from_memory = False
    caps = cap.build_capabilities_for_generation(cfg, query="price of AAPL")
    names = [t["name"] for t in caps.tools] if caps else []
    assert "get_stock_price" not in names  # recall gated off
