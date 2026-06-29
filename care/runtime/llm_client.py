"""LLM client wiring.

CARE drives MAGE and CARL through one OpenAI-compatible
endpoint: the user sets ``MageConfig.base_url`` + ``api_key``
+ ``model`` and both subsystems read their client through this
factory. There is no "provider" concept on the user surface —
any OpenAI-compatible URL works (OpenAI, OpenRouter, Groq,
DeepSeek, Together, a local Ollama / llama.cpp server, etc.).

The factory always returns an ``openai.OpenAI`` instance. The
duck-typed client shape stays the same as before
(``client.chat.completions.create(...)``), so CARL step
executors and MAGE generation code didn't have to change.
"""

from __future__ import annotations

from typing import Any

from care.config import MageConfig


class LLMClientError(RuntimeError):
    """Raised when the factory can't build a client — missing
    base_url / api_key, or the OpenAI SDK constructor blew up."""


def build_llm_client(config: MageConfig) -> Any:
    """Construct an OpenAI-compatible client from ``config``.

    Requires ``base_url`` and ``api_key`` to be set; the user
    points CARE at any OpenAI-compatible HTTP endpoint and CARE
    talks to it with the ``openai`` Python SDK.

    Raises:
        LLMClientError: When ``base_url`` / ``api_key`` is
            empty, the ``openai`` SDK isn't installed, or the
            client constructor raised.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMClientError(
            "openai SDK is not installed; install it to drive "
            "any LLM through CARE"
        ) from exc

    base_url = (config.base_url or "").strip()
    api_key = (config.api_key or "").strip()
    if not base_url:
        raise LLMClientError(
            "MageConfig.base_url must be set "
            "(e.g. https://openrouter.ai/api/v1)"
        )
    if not api_key:
        raise LLMClientError("MageConfig.api_key must be set")

    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception as exc:  # noqa: BLE001
        raise LLMClientError(
            f"failed to construct OpenAI client for {base_url!r}: {exc}"
        ) from exc


def build_carl_llm_client(
    config: MageConfig, *, token_counter: Any = None,
) -> Any:
    """Construct an LLM client CARL's step executors can drive.

    CARL's :class:`mmar_carl.models.LLMClientBase` requires
    ``get_response_with_retries(prompt, retries)`` on the API
    object handed to :class:`ReasoningContext`. The raw
    ``openai.OpenAI`` SDK doesn't expose that method, so passing
    the value returned by :func:`build_llm_client` to CARL
    crashes every step with
    ``AttributeError: 'OpenAI' object has no attribute 'get_response_with_retries'``.

    Returns an :class:`mmar_carl.llm.OpenAICompatibleClient`
    pre-configured with the same ``base_url`` / ``api_key`` /
    ``model`` triple, so chain execution actually reaches the
    LLM. MAGE keeps using :func:`build_llm_client` — only CARL
    needs the wrapper.

    When ``token_counter`` is provided, the returned client is a
    thin subclass that intercepts every ``chat.completions.create``
    response and folds ``response.usage`` into the counter so
    the StatusBar + per-iteration footer surface real-time
    token totals. CARL itself discards usage (its
    ``get_response_with_retries`` only returns text), so without
    this hook the chat would always render ``in 0 / out 0``.

    Raises:
        LLMClientError: When ``base_url`` / ``api_key`` / ``model``
            are missing, the ``mmar_carl`` package isn't
            installed, or the upstream constructor blew up.
    """
    base_url = (config.base_url or "").strip()
    api_key = (config.api_key or "").strip()
    model = (config.model or "").strip()
    if not base_url:
        raise LLMClientError("MageConfig.base_url must be set")
    if not api_key:
        raise LLMClientError("MageConfig.api_key must be set")
    if not model:
        raise LLMClientError(
            "MageConfig.model must be set to drive CARL "
            "(e.g. 'mistralai/mistral-medium-3-5')"
        )
    try:
        from mmar_carl.llm import OpenAIClientConfig, OpenAICompatibleClient
    except ImportError as exc:
        raise LLMClientError(
            "mmar_carl isn't installed; install `care[carl]` to "
            "execute chains"
        ) from exc

    cls = (
        _build_token_counting_client_cls(OpenAICompatibleClient)
        if token_counter is not None
        else OpenAICompatibleClient
    )
    try:
        client = cls(
            OpenAIClientConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMClientError(
            f"failed to construct CARL LLM client for {base_url!r}: {exc}"
        ) from exc
    if token_counter is not None:
        # The subclass reads `_token_counter` off the instance on
        # every request; attach after construction so the upstream
        # `__init__` doesn't need to know about our extra slot.
        client._token_counter = token_counter
    return client


def _build_token_counting_client_cls(base_cls: type) -> type:
    """Build (and cache) a subclass of ``base_cls`` whose
    ``_make_request`` captures token usage from the OpenAI
    response and folds it into ``self._token_counter`` (a
    :class:`care.runtime.status_bar.SessionTokenCounter`-shaped
    object). Cached so repeated calls within one process don't
    spawn fresh subclass identities for every chain run.
    """
    cached = _TOKEN_COUNTING_CLIENT_CACHE.get(base_cls)
    if cached is not None:
        return cached

    class TokenCountingOpenAIClient(base_cls):  # type: ignore[misc, valid-type]
        async def _make_request(self, prompt: str) -> str:  # type: ignore[override]
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
            }
            if self.config.max_tokens is not None:
                kwargs["max_tokens"] = self.config.max_tokens
            if self.config.extra_body:
                kwargs["extra_body"] = self.config.extra_body
            response = await self.client.chat.completions.create(**kwargs)
            _fold_response_usage_into_counter(
                response, getattr(self, "_token_counter", None),
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            return ""

    _TOKEN_COUNTING_CLIENT_CACHE[base_cls] = TokenCountingOpenAIClient
    return TokenCountingOpenAIClient


_TOKEN_COUNTING_CLIENT_CACHE: dict[type, type] = {}


def _fold_response_usage_into_counter(response: Any, counter: Any) -> None:
    """Pull ``prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens`` off an OpenAI ``ChatCompletion.usage``
    object and hand them to ``counter.add`` in the
    ``{"prompt", "completion", "total"}`` shape
    :class:`SessionTokenCounter` expects.

    Best-effort: missing usage, missing counter, or a counter
    that throws — all swallowed. The chain output is the
    important thing; a bookkeeping miss is acceptable.
    """
    if counter is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt = (
        getattr(usage, "prompt_tokens", None)
        or _dict_get(usage, "prompt_tokens")
        or 0
    )
    completion = (
        getattr(usage, "completion_tokens", None)
        or _dict_get(usage, "completion_tokens")
        or 0
    )
    total = (
        getattr(usage, "total_tokens", None)
        or _dict_get(usage, "total_tokens")
        or 0
    )
    try:
        counter.add(
            {
                "prompt": int(prompt or 0),
                "completion": int(completion or 0),
                "total": int(total or 0),
            },
        )
    except Exception:
        pass


def _dict_get(value: Any, key: str) -> Any:
    """Tiny helper — read a key off ``value`` only if it's a
    dict, otherwise return None. The OpenAI SDK exposes
    ``usage`` as a pydantic model with attribute access; some
    providers return a plain dict instead."""
    if isinstance(value, dict):
        return value.get(key)
    return None


def instrument_mage_generator(
    generator: Any, token_counter: Any,
) -> bool:
    """§2 P1 — wrap a MAGEGenerator's internal AsyncOpenAI
    client so every ``chat.completions.create`` call folds
    ``response.usage`` into ``token_counter``.

    MAGE owns its LLM client (constructed inside
    `MAGEGenerator.__init__` from `MAGEConfig`), so CARE can't
    inject a pre-built client. Some providers leave
    `MAGEResult.metadata.usage` (the canonical source for
    `MagePoster.handle_stage_completed`) empty, so the
    iteration-footer counter reads `0` even when MAGE
    consumed real tokens. The fallback this helper provides:
    after the generator is constructed, reach into
    ``generator._llm._client`` (the AsyncOpenAI instance) and
    monkey-patch ``chat.completions.create`` on the instance
    with a wrapper that captures usage in addition to passing
    through the response.

    Returns ``True`` when the wrap succeeded; ``False`` for any
    failure mode (unknown generator shape, AsyncOpenAI
    structure changed, etc.) — the original counter path
    (from `MAGEResult.metadata.usage`) still works in that
    case, just without the API-level fallback.

    Best-effort throughout: a single failure here NEVER blocks
    generation — the worst case is a token total that under-
    counts MAGE stages, which is what we ship today anyway.
    """
    if token_counter is None or generator is None:
        return False
    try:
        llm = generator._llm
        client = llm._client
        completions = client.chat.completions
    except Exception:
        return False
    original_create = getattr(completions, "create", None)
    if not callable(original_create):
        return False
    # Don't double-wrap a client that's already instrumented —
    # the closure carries a marker so re-instrumentation skips
    # cleanly.
    if getattr(original_create, "_care_token_wrapped", False):
        return True

    async def _wrapped_create(*args: Any, **kwargs: Any) -> Any:
        response = await original_create(*args, **kwargs)
        try:
            _fold_response_usage_into_counter(response, token_counter)
        except Exception:
            pass
        return response

    _wrapped_create._care_token_wrapped = True  # type: ignore[attr-defined]
    try:
        completions.create = _wrapped_create  # type: ignore[method-assign]
    except Exception:
        return False
    return True


__all__ = [
    "LLMClientError",
    "build_carl_llm_client",
    "build_llm_client",
    "instrument_mage_generator",
]
