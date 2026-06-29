"""Tests for ``care.generation.run_edit`` — the CARE→MAGE edit bridge."""

from __future__ import annotations

from typing import Any

import pytest

from care.generation import GenerationError, run_edit


class _FakeGen:
    """Duck-typed MAGEGenerator exposing ``edit``."""

    def __init__(self, *, result: Any = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    async def edit(
        self,
        instruction: str,
        *,
        chain: dict[str, Any] | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
        save: bool = False,
        cancel: Any = None,
    ) -> Any:
        self.calls.append(
            {
                "instruction": instruction,
                "entity_id": entity_id,
                "chain": chain,
                "channel": channel,
                "save": save,
            }
        )
        if self._exc is not None:
            raise self._exc
        return self._result


async def test_forwards_kwargs_and_returns_result() -> None:
    gen = _FakeGen(result="RESULT")
    out = await run_edit(gen, "do x", entity_id="abc", channel="stable", save=True)
    assert out == "RESULT"
    call = gen.calls[0]
    assert call["instruction"] == "do x"
    assert call["entity_id"] == "abc"
    assert call["channel"] == "stable"
    assert call["save"] is True


async def test_forwards_in_memory_chain() -> None:
    gen = _FakeGen(result="R")
    chain = {"name": "C", "steps": []}
    await run_edit(gen, "tweak", chain=chain)
    assert gen.calls[0]["chain"] == chain
    assert gen.calls[0]["entity_id"] is None


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
async def test_rejects_empty_or_non_string_instruction(bad: Any) -> None:
    gen = _FakeGen()
    with pytest.raises(GenerationError):
        await run_edit(gen, bad, entity_id="abc")  # type: ignore[arg-type]
    assert gen.calls == []


async def test_requires_edit_method() -> None:
    class NoEdit:
        pass

    with pytest.raises(GenerationError, match="missing `edit"):
        await run_edit(NoEdit(), "do x", entity_id="abc")


async def test_wraps_downstream_error() -> None:
    gen = _FakeGen(exc=RuntimeError("boom"))
    with pytest.raises(GenerationError, match="edit\\(\\) raised"):
        await run_edit(gen, "do x", entity_id="abc")
