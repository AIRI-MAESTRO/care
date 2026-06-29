"""Tests for on-demand tool synthesis (care.tool_synthesis).

Covers detection, the reuse→cache→synthesize→save report, disk caching,
startup registration, capability specs, and gating. The Docker sandbox
and Memory are stubbed; the cache writes to a tmp dir.
"""

from __future__ import annotations

import asyncio

from care import tool_synthesis
from care.config import CareConfig

_CHAIN = {
    "steps": [
        {
            "step_type": "tool",
            "title": "Get current weather",
            "aim": "fetch weather for the city",
            "step_config": {
                "tool_name": "get_current_weather",
                "input_mapping": {"city": "$outer_context"},
            },
        },
        {"step_type": "tool", "step_config": {"tool_name": "web_search"}},  # registered → skip
        {"step_type": "llm", "title": "Summarise"},
    ]
}


class _StubCtx:
    def __init__(self, existing=()):  # noqa: ANN001
        self._tool_registry = {n: (lambda **k: "") for n in existing}

    def register_tool(self, name, fn, **kw):  # noqa: ANN001
        self._tool_registry[name] = fn


class _FakeAPI:
    def __init__(self, code):  # noqa: ANN001
        self.code = code
        self.prompts = []

    async def get_response_with_retries(self, prompt):  # noqa: ANN001
        self.prompts.append(prompt)
        return self.code


def _cfg(tmp_path):
    cfg = CareConfig()
    cfg.tools.synthesized_tools_path = tmp_path / "synth"
    cfg.tools.save_synthesized_to_memory = False  # no network in tests
    return cfg


# ---------------------------------------------------------------------------
# Detection + helpers
# ---------------------------------------------------------------------------


def test_missing_tool_steps_detects_and_skips():
    missing = tool_synthesis._missing_tool_steps(_CHAIN, {"web_search"})
    assert [m[0] for m in missing] == ["get_current_weather"]
    _, params, desc = missing[0]
    assert params == ["city"] and "weather" in desc.lower()


def test_strip_helpers():
    assert tool_synthesis._strip_code_fences("```python\ndef f(): pass\n```") == "def f(): pass"
    assert tool_synthesis._strip_exit_prefix("[exit 0]\nhi") == "hi"
    assert tool_synthesis._strip_exit_prefix("[exit 1]\nboom").startswith("[exit 1]")


# ---------------------------------------------------------------------------
# Create path
# ---------------------------------------------------------------------------


def test_creates_caches_and_runs(tmp_path, monkeypatch):
    captured = {}

    async def fake_run(source, sandbox_cfg, *, timeout, max_chars):  # noqa: ANN001
        captured["source"] = source
        return "[exit 0]\n+14°C"

    monkeypatch.setattr(tool_synthesis, "run_python_source", fake_run)
    cfg = _cfg(tmp_path)
    ctx = _StubCtx({"web_search"})
    api = _FakeAPI("def get_current_weather(**kwargs):\n    return 'w'")

    report = asyncio.run(
        tool_synthesis.synthesize_missing_tools(_CHAIN, ctx, api=api, config=cfg)
    )
    assert report["created"] == ["get_current_weather"]
    assert report["reused"] == [] and report["failed"] == []
    assert "get_current_weather" in ctx._tool_registry
    # cache file written
    assert (cfg.tools.synthesized_tools_path / "get_current_weather.json").exists()
    # invoking routes through the sandbox + injects kwargs
    res = asyncio.run(ctx._tool_registry["get_current_weather"](city="Moscow"))
    assert res == "+14°C"
    assert "def get_current_weather" in captured["source"] and "Moscow" in captured["source"]


def test_skips_code_without_function(tmp_path, monkeypatch):
    async def fake_run(*a, **k):  # noqa: ANN002, ANN003
        return "[exit 0]\nx"

    monkeypatch.setattr(tool_synthesis, "run_python_source", fake_run)
    ctx = _StubCtx({"web_search"})
    api = _FakeAPI("print('no function')")  # never defines def get_current_weather
    report = asyncio.run(
        tool_synthesis.synthesize_missing_tools(_CHAIN, ctx, api=api, config=_cfg(tmp_path))
    )
    assert report["created"] == []
    assert report["failed"] and report["failed"][0][0] == "get_current_weather"


# ---------------------------------------------------------------------------
# Reuse from disk cache (no codegen)
# ---------------------------------------------------------------------------


def test_reuses_cached_tool(tmp_path, monkeypatch):
    async def fake_run(*a, **k):  # noqa: ANN002, ANN003
        return "[exit 0]\ncached-result"

    monkeypatch.setattr(tool_synthesis, "run_python_source", fake_run)
    cfg = _cfg(tmp_path)
    # Cache WITH stored sample args so the pre-reuse health-check verifies via
    # the (mocked) sandbox using those inputs — no LLM probe needed.
    tool_synthesis._save_cached_tool(
        cfg.tools, "get_current_weather",
        "def get_current_weather(**k):\n    return 'c'", ["city"], "weather",
        sample_args={"city": "London"},
    )
    ctx = _StubCtx({"web_search"})
    api = _FakeAPI("SHOULD NOT BE USED")

    report = asyncio.run(
        tool_synthesis.synthesize_missing_tools(_CHAIN, ctx, api=api, config=cfg)
    )
    assert report["reused"] == ["get_current_weather"]
    assert report["created"] == []
    assert api.prompts == []  # codegen never called — reused from cache (health-check OK)


def test_broken_cached_tool_is_rejected_and_resynthesised(tmp_path, monkeypatch):
    # A cached tool that now ERRORS its health-check must NOT be reused — CARE
    # re-synthesises a fresh implementation instead of running the broken one.
    cfg = _cfg(tmp_path)
    tool_synthesis._save_cached_tool(
        cfg.tools, "get_current_weather",
        "def get_current_weather(**k):\n    return 'error: boom'", ["city"], "weather",
        sample_args={"city": "London"},
    )

    async def fake_run(source, *a, **k):  # noqa: ANN001, ANN002, ANN003
        # The sandbox echoes what the tool produces: the cached (broken) source
        # yields an error string; the freshly-synthesised one yields a good value.
        return "[exit 0]\nerror: boom" if "error: boom" in source else "[exit 0]\nsunny 20C"

    monkeypatch.setattr(tool_synthesis, "run_python_source", fake_run)
    ctx = _StubCtx({"web_search"})
    api = _FakeAPI(
        "def get_current_weather(**k):\n    return 'sunny'\n#SELFTEST# {\"city\": \"London\"}"
    )

    report = asyncio.run(
        tool_synthesis.synthesize_missing_tools(_CHAIN, ctx, api=api, config=cfg)
    )
    assert report["reused"] == []                       # broken cache rejected
    assert report["created"] == ["get_current_weather"]  # re-synthesised fresh
    assert api.prompts                                   # codegen WAS invoked


# ---------------------------------------------------------------------------
# Startup registration + capability specs
# ---------------------------------------------------------------------------


def test_register_cached_tools_gates_on_health(tmp_path):
    import time as _t

    cfg = _cfg(tmp_path)
    # No health verdict yet → NOT eagerly registered; left for the
    # verify-on-use path (synthesize_missing_tools) to health-check first.
    tool_synthesis._save_cached_tool(
        cfg.tools, "get_weather", "def get_weather(**k):\n    return 'x'", ["city"], "w",
    )
    assert tool_synthesis.register_cached_tools(_StubCtx(), cfg) == []

    # A tool with a FRESH healthy verdict IS registered eagerly.
    tool_synthesis._save_cached_tool(
        cfg.tools, "get_rate", "def get_rate(**k):\n    return '1'", ["pair"], "fx",
        health={"ok": True, "checked_at": _t.time(), "detail": ""},
    )
    ctx = _StubCtx()
    names = tool_synthesis.register_cached_tools(ctx, cfg)
    assert names == ["get_rate"]
    assert "get_rate" in ctx._tool_registry and "get_weather" not in ctx._tool_registry


def test_register_cached_tools_no_verify_registers_all(tmp_path):
    # With verification disabled, the prior behaviour holds (register all).
    cfg = _cfg(tmp_path)
    cfg.tools.verify_cached_tools = False
    tool_synthesis._save_cached_tool(
        cfg.tools, "get_weather", "def get_weather(**k):\n    return 'x'", ["city"], "w",
    )
    ctx = _StubCtx()
    assert tool_synthesis.register_cached_tools(ctx, cfg) == ["get_weather"]
    assert "get_weather" in ctx._tool_registry


def test_cached_tool_specs(tmp_path):
    cfg = _cfg(tmp_path)
    tool_synthesis._save_cached_tool(
        cfg.tools, "get_weather", "def get_weather(**k): return 'x'", ["city"], "Weather lookup",
    )
    specs = tool_synthesis.cached_tool_specs(cfg)
    assert [s["name"] for s in specs] == ["get_weather"]
    assert specs[0]["source"] == "care:synthesized"
    assert "city" in specs[0]["description"]


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def test_gated_off_by_flag(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.tools.auto_synthesize_tools = False
    report = asyncio.run(
        tool_synthesis.synthesize_missing_tools(
            _CHAIN, _StubCtx(), api=_FakeAPI("def x(): pass"), config=cfg
        )
    )
    assert report == {"created": [], "reused": [], "failed": []}


def test_gated_off_without_docker(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.sandbox.kind = "local"
    report = asyncio.run(
        tool_synthesis.synthesize_missing_tools(
            _CHAIN, _StubCtx(), api=_FakeAPI("def x(): pass"), config=cfg
        )
    )
    assert report["created"] == [] and report["reused"] == []


def test_no_op_when_all_registered(tmp_path, monkeypatch):
    async def boom(*a, **k):  # noqa: ANN002, ANN003
        raise AssertionError("codegen should not run")

    monkeypatch.setattr(tool_synthesis, "run_python_source", boom)
    ctx = _StubCtx({"get_current_weather", "web_search"})
    report = asyncio.run(
        tool_synthesis.synthesize_missing_tools(
            _CHAIN, ctx, api=_FakeAPI("x"), config=_cfg(tmp_path)
        )
    )
    assert report == {"created": [], "reused": [], "failed": []}


# ---------------------------------------------------------------------------
# Web-grounded synthesis (_discover_api + research injection into codegen)
# ---------------------------------------------------------------------------


def _patch_web_search(monkeypatch, ws_impl):
    """Patch the factory `_discover_api` imports from care.builtin_tools."""
    import care.builtin_tools as bt

    def fake_make(provider, api_key, n):  # noqa: ANN001
        return ws_impl

    monkeypatch.setattr(bt, "_make_web_search", fake_make)


def test_discover_api_gated_off_by_flag(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.tools.web_search_api_key = "k"
    cfg.tools.ground_synthesis_with_web_search = False
    assert asyncio.run(
        tool_synthesis._discover_api("crypto price", ["coin"], cfg.tools)
    ) is None


def test_discover_api_none_without_key(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.tools.web_search_api_key = None  # no key → no search attempted
    assert asyncio.run(
        tool_synthesis._discover_api("crypto price", ["coin"], cfg.tools)
    ) is None


def test_discover_api_returns_digest_and_asks_for_keyless_json_api(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.tools.web_search_api_key = "k"
    seen: dict[str, str] = {}

    async def ws(query):  # noqa: ANN001
        seen["query"] = query
        return "Answer: use https://api.coingecko.com/api/v3/simple/price\n[1] CoinGecko docs"

    _patch_web_search(monkeypatch, ws)
    out = asyncio.run(
        tool_synthesis._discover_api("current price of a cryptocurrency", ["coin"], cfg.tools)
    )
    assert out and "coingecko" in out.lower()
    # the query steers toward a keyless JSON endpoint + includes the param hint
    assert "no API key" in seen["query"] and "JSON" in seen["query"] and "coin" in seen["query"]


def test_discover_api_graceful_on_search_error(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.tools.web_search_api_key = "k"

    async def ws(query):  # noqa: ANN001
        return "web_search error: 429 rate limited"

    _patch_web_search(monkeypatch, ws)
    assert asyncio.run(tool_synthesis._discover_api("x", [], cfg.tools)) is None


def test_generate_source_injects_research_into_prompt():
    api = _FakeAPI("def t(**kwargs):\n    return 'ok'")
    gen = tool_synthesis._resolve_llm(api)
    src, _sample = asyncio.run(
        tool_synthesis._generate_source(
            gen, "t", ["x"], "do x", research="Answer: call https://real.api/v1"
        )
    )
    assert src and "def t" in src
    # the research digest reached the codegen prompt verbatim
    assert "https://real.api/v1" in api.prompts[0]
    assert "WEB RESEARCH" in api.prompts[0]


def test_generate_source_without_research_has_no_research_block():
    api = _FakeAPI("def t(**kwargs):\n    return 'ok'")
    gen = tool_synthesis._resolve_llm(api)
    asyncio.run(tool_synthesis._generate_source(gen, "t", ["x"], "do x"))
    assert "WEB RESEARCH" not in api.prompts[0]


def test_split_selftest_extracts_and_strips():
    src = 'def t(**kwargs):\n    return "ok"\n#SELFTEST# {"city": "London"}'
    clean, sample = tool_synthesis._split_selftest(src)
    assert sample == {"city": "London"}
    assert "#SELFTEST#" not in clean and clean.endswith('return "ok"')


def test_split_selftest_absent_is_none():
    clean, sample = tool_synthesis._split_selftest('def t(**kwargs):\n    return "ok"')
    assert sample is None and "def t" in clean


def test_generate_source_parses_selftest_sample():
    api = _FakeAPI('def t(**kwargs):\n    return "ok"\n#SELFTEST# {"a": "1"}')
    gen = tool_synthesis._resolve_llm(api)
    src, sample = asyncio.run(tool_synthesis._generate_source(gen, "t", ["a"], "do"))
    assert sample == {"a": "1"} and "#SELFTEST#" not in src


def test_self_heal_retries_on_runtime_error(tmp_path, monkeypatch):
    # First codegen → a tool that self-tests to "error:"; the heal retry →
    # a working tool. The registered + cached source is the HEALED one.
    codes = iter([
        'def get_x(**kwargs):\n    return "error: bad endpoint"\n#SELFTEST# {"a": "1"}',
        'def get_x(**kwargs):\n    return "ok 42"\n#SELFTEST# {"a": "1"}',
    ])

    class _MultiAPI:
        prompts: list[str] = []

        async def get_response_with_retries(self, prompt):  # noqa: ANN001
            _MultiAPI.prompts.append(prompt)
            return next(codes)

    async def fake_run(src, *a, **k):  # noqa: ANN001, ANN002, ANN003
        return "error: bad endpoint" if "bad endpoint" in src else "ok 42"

    monkeypatch.setattr(tool_synthesis, "run_python_source", fake_run)
    chain = {"steps": [{
        "step_type": "tool", "title": "x", "aim": "get x",
        "step_config": {"tool_name": "get_x", "input_mapping": {"a": "x"}},
    }]}
    report = asyncio.run(tool_synthesis.synthesize_missing_tools(
        chain, _StubCtx(), api=_MultiAPI(), config=_cfg(tmp_path)))
    assert report["created"] == ["get_x"]
    # a heal round happened (2 codegen calls) and the cached source is the fix
    assert len(_MultiAPI.prompts) == 2
    import json as _j
    rec = _j.loads((tmp_path / "synth" / "get_x.json").read_text())
    assert "ok 42" in rec["source"] and "bad endpoint" not in rec["source"]


def test_cfg_field_reads_dict_and_object():
    from types import SimpleNamespace
    assert tool_synthesis._cfg_field({"tool_name": "x"}, "tool_name") == "x"
    assert tool_synthesis._cfg_field(SimpleNamespace(tool_name="y"), "tool_name") == "y"
    assert tool_synthesis._cfg_field(None, "tool_name") is None


def test_missing_tool_steps_reads_typed_step_config():
    # Regression: after ReasoningChain.from_dict, step_config is a typed
    # object (not a dict). Detection MUST still find the tool — otherwise
    # synthesis silently skips and the chain dies "Tool '<name>' not
    # registered in context" (the get_forex_rate bug).
    from types import SimpleNamespace
    cfg = SimpleNamespace(tool_name="get_forex_rate", input_mapping={"base": "x"})
    chain = {"steps": [{"step_type": "tool", "title": "fx", "step_config": cfg}]}
    missing = tool_synthesis._missing_tool_steps(chain, set())
    assert [m[0] for m in missing] == ["get_forex_rate"]
    assert missing[0][1] == ["base"]


def test_synthesize_emits_progress_to_notify(tmp_path, monkeypatch):
    async def fake_run(src, *a, **k):  # noqa: ANN001, ANN002, ANN003
        return "ok"

    monkeypatch.setattr(tool_synthesis, "run_python_source", fake_run)
    msgs: list[str] = []
    report = asyncio.run(tool_synthesis.synthesize_missing_tools(
        _CHAIN, _StubCtx({"web_search"}),
        api=_FakeAPI('def get_current_weather(**kwargs):\n    return "ok"'),
        config=_cfg(tmp_path), notify=msgs.append))
    assert report["created"] == ["get_current_weather"]
    assert any("building" in m for m in msgs)       # the "N tool(s) … building" line
    assert any("synthesising" in m for m in msgs)   # the per-tool synth line


def test_no_self_heal_when_tool_works(tmp_path, monkeypatch):
    # A tool that self-tests cleanly is NOT regenerated (single codegen call).
    class _OneAPI:
        n = 0

        async def get_response_with_retries(self, prompt):  # noqa: ANN001
            _OneAPI.n += 1
            return 'def get_y(**kwargs):\n    return "ok"\n#SELFTEST# {"a": "1"}'

    async def fake_run(src, *a, **k):  # noqa: ANN001, ANN002, ANN003
        return "ok"

    monkeypatch.setattr(tool_synthesis, "run_python_source", fake_run)
    chain = {"steps": [{
        "step_type": "tool", "title": "y", "aim": "get y",
        "step_config": {"tool_name": "get_y", "input_mapping": {"a": "x"}},
    }]}
    report = asyncio.run(tool_synthesis.synthesize_missing_tools(
        chain, _StubCtx(), api=_OneAPI(), config=_cfg(tmp_path)))
    assert report["created"] == ["get_y"]
    assert _OneAPI.n == 1  # no heal retry


# ---------------------------------------------------------------------------
# Bundling synthesized tools into a deployment
# ---------------------------------------------------------------------------


def test_bundled_tools_for_chain(tmp_path):
    cfg = _cfg(tmp_path)
    tool_synthesis._save_cached_tool(
        cfg.tools, "get_exchange_rate",
        "def get_exchange_rate(**k):\n    return '1 RUB = 0.0138 USD'", ["pair"], "fx",
    )
    chain = {"steps": [
        {"step_type": "tool", "step_config": {"tool_name": "get_exchange_rate"}},  # synthesized → bundle
        {"step_type": "tool", "step_config": {"tool_name": "web_search"}},          # builtin → not cached, skip
        {"step_type": "llm"},
    ]}
    bundled = tool_synthesis.bundled_tools_for_chain(chain, cfg)
    assert [b["name"] for b in bundled] == ["get_exchange_rate"]
    assert "def get_exchange_rate" in bundled[0]["source"]
    assert bundled[0]["params"] == ["pair"]


def test_bundled_tools_for_chain_empty_when_all_builtin(tmp_path):
    chain = {"steps": [{"step_type": "tool", "step_config": {"tool_name": "http_request"}}]}
    assert tool_synthesis.bundled_tools_for_chain(chain, _cfg(tmp_path)) == []


def test_bundled_tools_to_python_code_merges_sources(tmp_path):
    cfg = _cfg(tmp_path)
    tool_synthesis._save_cached_tool(
        cfg.tools, "get_rate",
        "def get_rate(pair: str) -> str:\n    return '1.0'", ["pair"], "fx",
    )
    bundled = tool_synthesis.bundled_tools_for_chain(
        {"steps": [{"step_type": "tool", "step_config": {"tool_name": "get_rate"}}]},
        cfg,
    )
    code = tool_synthesis.bundled_tools_to_python_code(bundled)
    assert "custom_tools.py" in code
    assert "def get_rate" in code
    assert tool_synthesis.bundled_tools_to_python_code([]) == ""
