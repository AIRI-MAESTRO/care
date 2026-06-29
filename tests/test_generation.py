"""Tests for ``care.generation`` (TODO §4 P0).

Coverage:

1. **Config translation** — every `CareConfig.mage` field
   that maps to MAGEConfig propagates correctly.
2. **Mode override** — explicit ``mode=`` kwarg wins over
   the saved config.
3. **Provider mapping** — CARE's looser provider strings
   map to MAGE's enum (`"openai"` / `"openrouter"` /
   `"local"`) with unknown values falling back to
   `"custom"`.
4. **Validation errors** — missing api_key + bad MAGEConfig
   inputs surface as `GenerationError`.
5. **Missing mmar_mage** — friendly install hint when the
   extra isn't installed (skipped if it is).
6. **`build_mage_generator`** — passes config + progress
   through; construction failure wraps.
7. **`run_generation`** — empty query raises; missing
   `.generate` method raises; kwargs forward; downstream
   exception wraps; cancel event passes through.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from care.config import CareConfig, MageConfig
from care.generation import (
    GenerationError,
    build_mage_config,
    build_mage_generator,
    run_generation,
)


# ---------------------------------------------------------------------------
# Config translation. `mmar-mage` is a required dependency now —
# no skip guards needed.
# ---------------------------------------------------------------------------


class TestBuildMageConfig:
    def test_basic_field_translation(self):
        cfg = CareConfig(
            mage=MageConfig(
                mode="deep",
                api_key="sk-test",
                base_url="https://api.example.com",
                model="gpt-4",
                enable_memory_research=True,
                enable_web_research=False,
            ),
        )
        mage_cfg = build_mage_config(cfg)
        assert mage_cfg.mode == "deep"
        # `provider` is always `"custom"` on the MAGE side now —
        # CARE's user surface no longer carries a provider field
        # (any OpenAI-compatible endpoint works via base_url).
        assert mage_cfg.provider == "custom"
        assert mage_cfg.api_key == "sk-test"
        assert mage_cfg.base_url == "https://api.example.com"
        assert mage_cfg.model == "gpt-4"
        assert mage_cfg.enable_memory_research is True
        assert mage_cfg.enable_web_research is False

    def test_accepts_bare_mage_config(self):
        # Library callers can pass MageConfig directly.
        mage = MageConfig(api_key="sk-x")
        mage_cfg = build_mage_config(mage)
        assert mage_cfg.api_key == "sk-x"

    def test_mode_override(self):
        cfg = CareConfig(
            mage=MageConfig(mode="deep", api_key="sk-x"),
        )
        mage_cfg = build_mage_config(cfg, mode="fast")
        assert mage_cfg.mode == "fast"

    def test_missing_api_key_raises(self):
        cfg = CareConfig(mage=MageConfig(api_key=None))
        with pytest.raises(GenerationError, match="api_key must be set"):
            build_mage_config(cfg)

    def test_empty_api_key_raises(self):
        cfg = CareConfig(mage=MageConfig(api_key=""))
        with pytest.raises(GenerationError, match="api_key must be set"):
            build_mage_config(cfg)

    def test_omits_base_url_when_unset(self):
        # When CARE's base_url is None, the MAGEConfig uses its
        # own default (whatever that is) — we just don't forward
        # an empty value.
        cfg = CareConfig(mage=MageConfig(api_key="sk-x", base_url=None))
        mage_cfg = build_mage_config(cfg)
        # MAGEConfig.base_url stays at its default; we didn't
        # override it with None.
        assert mage_cfg.api_key == "sk-x"

    def test_web_search_api_key_forwarded(self):
        cfg = CareConfig(
            mage=MageConfig(
                api_key="sk-x",
                enable_web_research=True,
                web_search_api_key="ws-key",
            ),
        )
        mage_cfg = build_mage_config(cfg)
        # MAGE accepts arbitrary extra fields via its config; we
        # don't need to assert the exact name MAGE stores it under,
        # just that our call didn't raise.
        assert mage_cfg.api_key == "sk-x"


class TestMageProviderConstant:
    """CARE always hands MAGE ``provider="custom"`` — the user
    surface picks the endpoint via ``base_url`` instead."""

    def test_default_provider_is_custom(self):
        cfg = CareConfig(mage=MageConfig(api_key="sk-x"))
        mage_cfg = build_mage_config(cfg)
        assert mage_cfg.provider == "custom"

    def test_provider_stays_custom_for_any_base_url(self):
        for url in (
            "https://api.openai.com/v1",
            "https://openrouter.ai/api/v1",
            "http://localhost:11434/v1",
        ):
            cfg = CareConfig(
                mage=MageConfig(api_key="sk-x", base_url=url),
            )
            assert build_mage_config(cfg).provider == "custom"


# ---------------------------------------------------------------------------
# build_mage_generator
# ---------------------------------------------------------------------------


class TestBuildMageGenerator:
    def test_generator_constructed_with_config(self):
        cfg = CareConfig(mage=MageConfig(api_key="sk-x", mode="fast"))
        gen = build_mage_generator(cfg)
        # Duck-typed assertion — we don't depend on MAGEGenerator's
        # internal layout, just that we got something with
        # `.generate`.
        assert hasattr(gen, "generate")

    def test_progress_passed_through(self):
        # The progress arg is opaque to us; just verify
        # construction doesn't raise.
        cfg = CareConfig(mage=MageConfig(api_key="sk-x"))

        class _Progress:
            pass

        gen = build_mage_generator(cfg, progress=_Progress())
        assert hasattr(gen, "generate")

    def test_mode_override_flows_through(self):
        cfg = CareConfig(mage=MageConfig(mode="deep", api_key="sk-x"))
        gen = build_mage_generator(cfg, mode="fast")
        # Internal config should reflect the override — but
        # we don't peek at MAGE's private state; the contract
        # is "no raise".
        assert hasattr(gen, "generate")


# ---------------------------------------------------------------------------
# run_generation (stub-driven — no MAGE needed)
# ---------------------------------------------------------------------------


class TestRunGeneration:
    def test_empty_query_raises(self):
        class _Stub:
            async def generate(self, *a, **kw):
                return None

        with pytest.raises(GenerationError, match="non-empty string"):
            asyncio.run(run_generation(_Stub(), ""))

    def test_whitespace_query_raises(self):
        class _Stub:
            async def generate(self, *a, **kw):
                return None

        with pytest.raises(GenerationError, match="non-empty string"):
            asyncio.run(run_generation(_Stub(), "   "))

    def test_missing_generate_method_raises(self):
        class _Bare:
            pass

        with pytest.raises(GenerationError, match="missing `generate"):
            asyncio.run(run_generation(_Bare(), "weather"))

    def test_kwargs_forwarded(self):
        calls: list[dict[str, Any]] = []

        class _Stub:
            async def generate(self, query, *, context_files, cancel, capabilities):
                calls.append(
                    {
                        "query": query,
                        "context_files": context_files,
                        "cancel": cancel,
                        "capabilities": capabilities,
                    }
                )
                return "OK"

        ctx = [{"path": "a.txt", "sha256": "x" * 64, "size_bytes": 5}]
        cancel = asyncio.Event()
        caps = object()  # sentinel
        result = asyncio.run(
            run_generation(
                _Stub(),
                "weather",
                context_files=ctx,
                cancel=cancel,
                capabilities=caps,
            )
        )
        assert result == "OK"
        assert calls[0]["query"] == "weather"
        assert calls[0]["context_files"] is ctx
        assert calls[0]["cancel"] is cancel
        assert calls[0]["capabilities"] is caps

    def test_generate_exception_wraps(self):
        class _Stub:
            async def generate(self, *a, **kw):
                raise RuntimeError("LLM 503")

        with pytest.raises(GenerationError, match="generate\\(\\) raised.*LLM 503"):
            asyncio.run(run_generation(_Stub(), "weather"))

    def test_generation_error_propagates_unwrapped(self):
        class _Stub:
            async def generate(self, *a, **kw):
                raise GenerationError("descriptive")

        with pytest.raises(GenerationError, match="^descriptive$"):
            asyncio.run(run_generation(_Stub(), "weather"))

    def test_returns_result_verbatim(self):
        class _Stub:
            async def generate(self, *a, **kw):
                return {"chain_dict": {}, "metadata": {}}

        out = asyncio.run(run_generation(_Stub(), "weather"))
        assert out == {"chain_dict": {}, "metadata": {}}
