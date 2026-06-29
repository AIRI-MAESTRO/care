"""Tests for ``care.runtime.llm_client``.

CARE's LLM surface is single-endpoint now: every supported
deployment is just an OpenAI-compatible HTTP endpoint, so the
factory always builds an ``openai.OpenAI`` client from
``MageConfig.base_url`` + ``MageConfig.api_key``. There is no
``provider`` field on the user surface and no factory
registration — those were removed when the "use only base_url"
simplification landed.

Coverage:

1. Happy path — given valid base_url + api_key, the factory
   returns a client wired to the right URL + key.
2. Missing ``base_url`` raises :class:`LLMClientError`.
3. Missing ``api_key`` raises :class:`LLMClientError`.
4. Explicit base_url is preserved verbatim (no normalisation).
"""

from __future__ import annotations

import pytest

from care.config import MageConfig
from care.runtime import LLMClientError, build_llm_client


class TestHappyPath:
    def test_returns_openai_client_with_url_and_key(self):
        cfg = MageConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
        )
        client = build_llm_client(cfg)
        # openai SDK coerces base_url into a URL object — stringify.
        assert str(client.base_url).startswith(
            "https://openrouter.ai/api/v1",
        )
        assert client.api_key == "sk-or-test"

    def test_custom_local_endpoint(self):
        cfg = MageConfig(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        client = build_llm_client(cfg)
        assert str(client.base_url).startswith(
            "http://localhost:11434/v1",
        )
        assert client.api_key == "ollama"


class TestRequiredFields:
    def test_missing_base_url_raises(self):
        cfg = MageConfig(api_key="sk-x")
        with pytest.raises(LLMClientError, match="base_url must be set"):
            build_llm_client(cfg)

    def test_blank_base_url_raises(self):
        cfg = MageConfig(base_url="   ", api_key="sk-x")
        with pytest.raises(LLMClientError, match="base_url must be set"):
            build_llm_client(cfg)

    def test_missing_api_key_raises(self):
        cfg = MageConfig(base_url="https://api.openai.com/v1")
        with pytest.raises(LLMClientError, match="api_key must be set"):
            build_llm_client(cfg)

    def test_blank_api_key_raises(self):
        cfg = MageConfig(
            base_url="https://api.openai.com/v1", api_key="   ",
        )
        with pytest.raises(LLMClientError, match="api_key must be set"):
            build_llm_client(cfg)


class TestCarlClient:
    """`build_carl_llm_client` returns a client CARL can drive
    (has `get_response_with_retries`). With a token counter
    attached, each `chat.completions.create` response folds its
    usage into the counter so the StatusBar reads real numbers
    instead of `in 0 / out 0`."""

    def test_returns_carl_shaped_client(self):
        from care.runtime.llm_client import build_carl_llm_client

        cfg = MageConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
            model="mistralai/mistral-medium-3-5",
        )
        client = build_carl_llm_client(cfg)
        # The CARL contract — see mmar_carl.models.LLMClientBase.
        assert hasattr(client, "get_response_with_retries")

    def test_missing_model_raises(self):
        from care.runtime.llm_client import build_carl_llm_client

        cfg = MageConfig(
            base_url="https://e.test/v1", api_key="k",
        )
        with pytest.raises(LLMClientError, match="model must be set"):
            build_carl_llm_client(cfg)

    @pytest.mark.asyncio
    async def test_response_usage_folded_into_counter(
        self, monkeypatch,
    ):
        """A `chat.completions.create` response with non-zero
        usage shows up on the bound counter after a
        `_make_request` call."""
        from types import SimpleNamespace

        from care.runtime.llm_client import build_carl_llm_client
        from care.runtime.status_bar import SessionTokenCounter

        cfg = MageConfig(
            base_url="https://e.test/v1",
            api_key="k",
            model="m",
        )
        counter = SessionTokenCounter()
        client = build_carl_llm_client(cfg, token_counter=counter)

        async def _fake_create(**_kw):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"),
                    ),
                ],
                usage=SimpleNamespace(
                    prompt_tokens=120,
                    completion_tokens=40,
                    total_tokens=160,
                ),
            )

        # Swap the real `chat.completions.create` for the fake
        # via attribute injection — the subclass reads
        # `self.client.chat.completions.create` so we stub
        # whatever the property returns.
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_fake_create),
            ),
        )
        # `client.client` is a property; patch the cached slot.
        client._client = fake_client

        text = await client._make_request("hi")
        assert text == "ok"
        snap = counter.snapshot()
        assert snap.prompt == 120
        assert snap.completion == 40
        assert snap.total == 160
        assert snap.calls == 1

    @pytest.mark.asyncio
    async def test_missing_usage_is_silent(self, monkeypatch):
        """A response without `usage` (some providers omit it on
        edge cases) doesn't crash and leaves the counter at 0."""
        from types import SimpleNamespace

        from care.runtime.llm_client import build_carl_llm_client
        from care.runtime.status_bar import SessionTokenCounter

        cfg = MageConfig(
            base_url="https://e.test/v1", api_key="k", model="m",
        )
        counter = SessionTokenCounter()
        client = build_carl_llm_client(cfg, token_counter=counter)

        async def _fake_create(**_kw):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="hi"),
                    ),
                ],
                usage=None,
            )

        client._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_fake_create),
            ),
        )
        text = await client._make_request("ping")
        assert text == "hi"
        assert counter.snapshot().total == 0


# ---------------------------------------------------------------------------
# instrument_mage_generator (§2 P1 — MAGE token counting fallback)
# ---------------------------------------------------------------------------


class TestInstrumentMageGenerator:
    """The wrap reaches into a MAGEGenerator's internal
    AsyncOpenAI client and folds `response.usage` into the
    counter — fallback for providers that leave
    `MAGEResult.metadata.usage` empty."""

    def _build_generator_stub(self):
        from types import SimpleNamespace

        completions = SimpleNamespace()
        # Will be overridden per-test to return a usage-bearing
        # response.
        completions.create = None
        chat = SimpleNamespace(completions=completions)
        async_openai_client = SimpleNamespace(chat=chat)
        llm = SimpleNamespace(_client=async_openai_client)
        return SimpleNamespace(_llm=llm)

    def test_no_counter_short_circuits(self):
        from care.runtime.llm_client import (
            instrument_mage_generator,
        )

        gen = self._build_generator_stub()
        assert instrument_mage_generator(gen, None) is False

    def test_no_generator_short_circuits(self):
        from care.runtime.llm_client import (
            instrument_mage_generator,
        )

        # Counter present but no generator.
        from care.runtime.status_bar import SessionTokenCounter

        assert instrument_mage_generator(
            None, SessionTokenCounter(),
        ) is False

    def test_unknown_shape_short_circuits(self):
        # A bare object with no `_llm` or `_client` attr should
        # return False, not raise.
        from types import SimpleNamespace

        from care.runtime.llm_client import (
            instrument_mage_generator,
        )
        from care.runtime.status_bar import SessionTokenCounter

        assert instrument_mage_generator(
            SimpleNamespace(), SessionTokenCounter(),
        ) is False

    def test_no_create_method_short_circuits(self):
        from care.runtime.llm_client import (
            instrument_mage_generator,
        )
        from care.runtime.status_bar import SessionTokenCounter

        gen = self._build_generator_stub()
        gen._llm._client.chat.completions.create = "not-callable"
        assert instrument_mage_generator(
            gen, SessionTokenCounter(),
        ) is False

    @pytest.mark.asyncio
    async def test_wrap_folds_usage_into_counter(self):
        from types import SimpleNamespace

        from care.runtime.llm_client import (
            instrument_mage_generator,
        )
        from care.runtime.status_bar import SessionTokenCounter

        async def _original_create(**_kw):
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=200,
                    completion_tokens=100,
                    total_tokens=300,
                ),
            )

        gen = self._build_generator_stub()
        gen._llm._client.chat.completions.create = _original_create

        counter = SessionTokenCounter()
        assert instrument_mage_generator(gen, counter) is True
        # Single call → single fold.
        response = await gen._llm._client.chat.completions.create()
        assert response.usage.prompt_tokens == 200
        snap = counter.snapshot()
        assert snap.prompt == 200
        assert snap.completion == 100
        assert snap.total == 300
        assert snap.calls == 1

    @pytest.mark.asyncio
    async def test_wrap_passes_through_args_and_kwargs(self):
        from types import SimpleNamespace

        from care.runtime.llm_client import (
            instrument_mage_generator,
        )
        from care.runtime.status_bar import SessionTokenCounter

        captured: dict = {}

        async def _original_create(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return SimpleNamespace(usage=None)

        gen = self._build_generator_stub()
        gen._llm._client.chat.completions.create = _original_create
        instrument_mage_generator(gen, SessionTokenCounter())

        await gen._llm._client.chat.completions.create(
            model="x", messages=[], temperature=0.2,
        )
        assert captured["kwargs"]["model"] == "x"
        assert captured["kwargs"]["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_wrap_idempotent_does_not_double_count(self):
        from types import SimpleNamespace

        from care.runtime.llm_client import (
            instrument_mage_generator,
        )
        from care.runtime.status_bar import SessionTokenCounter

        async def _original_create(**_kw):
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=50,
                    completion_tokens=25,
                    total_tokens=75,
                ),
            )

        gen = self._build_generator_stub()
        gen._llm._client.chat.completions.create = _original_create
        counter = SessionTokenCounter()
        # Wrap twice; second call must be a no-op (no double-
        # wrapping). One create call → one fold.
        assert instrument_mage_generator(gen, counter) is True
        assert instrument_mage_generator(gen, counter) is True
        await gen._llm._client.chat.completions.create()
        snap = counter.snapshot()
        assert snap.total == 75
        assert snap.calls == 1

    @pytest.mark.asyncio
    async def test_wrap_response_without_usage_safe(self):
        # A response with no `usage` attribute (some providers
        # omit it) shouldn't crash the wrap or counter.
        from types import SimpleNamespace

        from care.runtime.llm_client import (
            instrument_mage_generator,
        )
        from care.runtime.status_bar import SessionTokenCounter

        async def _original_create(**_kw):
            return SimpleNamespace()  # no usage

        gen = self._build_generator_stub()
        gen._llm._client.chat.completions.create = _original_create
        counter = SessionTokenCounter()
        assert instrument_mage_generator(gen, counter) is True
        response = await gen._llm._client.chat.completions.create()
        assert response is not None
        assert counter.snapshot().total == 0
