"""Tests for ``care.marketplace.search_marketplace`` (TODO §8 P2).

The marketplace is a thin wrapper around the SDK's
``find_capability_matches`` plus CARE-side filtering / sorting.
We exercise:

1. **Stub-backed search** — the function forwards the right
   kwargs and shapes the result correctly.
2. **Empty-query short-circuit** — no backend call.
3. **CARE-side filters** — `tags` (all-of) and `min_score`
   threshold.
4. **Sort order** — listings come back sorted by score
   descending.
5. **Forward kwargs** — `namespace`, `deep`, `top_k` all reach
   the backend unchanged.
6. **Error wrapping** — backend exceptions become
   :class:`MarketplaceError` with the original chained via
   ``from``.
7. **Hit-shape tolerance** — works against both dict + attr-style
   hits (the SDK ships Pydantic models; tests use dicts).
"""

from __future__ import annotations

from typing import Any

import pytest

from care.marketplace import (
    MarketplaceError,
    MarketplaceListing,
    MarketplaceResult,
    search_marketplace,
)


class _StubMemory:
    """Records every call + returns whatever was queued."""

    def __init__(
        self,
        hits: list[Any] | None = None,
        raise_exc: Exception | None = None,
    ):
        self.calls: list[dict[str, Any]] = []
        self._hits = hits or []
        self._raise = raise_exc

    def find_capability_matches(
        self,
        rough_aim: str,
        *,
        top_k: int = 3,
        namespace: str | None = None,
        deep: bool = False,
    ) -> list[Any]:
        self.calls.append(
            {
                "rough_aim": rough_aim,
                "top_k": top_k,
                "namespace": namespace,
                "deep": deep,
            }
        )
        if self._raise is not None:
            raise self._raise
        return self._hits


def _hit(
    entity_id: str,
    name: str = "skill",
    score: float = 0.5,
    tags: tuple[str, ...] = (),
    description: str = "",
    matched_via: str = "skill_description",
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "name": name,
        "description": description,
        "score": score,
        "tags": list(tags),
        "matched_via": matched_via,
        "snippet": None,
    }


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_empty_result_predicates(self):
        r = MarketplaceResult()
        assert r.is_empty
        assert r.listings == ()
        assert r.by_tag("anything") == ()

    def test_by_tag_filter(self):
        listings = (
            MarketplaceListing(entity_id="a", name="A", tags=("pdf", "io")),
            MarketplaceListing(entity_id="b", name="B", tags=("net",)),
        )
        r = MarketplaceResult(listings=listings)
        assert [li.name for li in r.by_tag("pdf")] == ["A"]
        assert r.by_tag("absent") == ()

    def test_listing_is_frozen(self):
        li = MarketplaceListing(entity_id="x", name="X")
        with pytest.raises(Exception):
            li.score = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Empty-query short-circuit
# ---------------------------------------------------------------------------


class TestEmptyQuery:
    def test_empty_string_returns_empty_result(self):
        memory = _StubMemory(hits=[_hit("a")])
        result = search_marketplace(memory, "")
        assert result.is_empty
        # Backend was never touched.
        assert memory.calls == []

    def test_whitespace_only_returns_empty_result(self):
        memory = _StubMemory(hits=[_hit("a")])
        result = search_marketplace(memory, "   \n   ")
        assert result.is_empty
        assert memory.calls == []

    def test_none_query_returns_empty_result(self):
        memory = _StubMemory(hits=[_hit("a")])
        result = search_marketplace(memory, None)  # type: ignore[arg-type]
        assert result.is_empty
        assert memory.calls == []


# ---------------------------------------------------------------------------
# Basic backend forwarding
# ---------------------------------------------------------------------------


class TestBackendForwarding:
    def test_query_normalised(self):
        memory = _StubMemory(hits=[_hit("a")])
        search_marketplace(memory, "  extract pdf tables  ")
        # Whitespace stripped before forwarding.
        assert memory.calls[0]["rough_aim"] == "extract pdf tables"

    def test_default_top_k_is_ten(self):
        memory = _StubMemory(hits=[])
        search_marketplace(memory, "q")
        assert memory.calls[0]["top_k"] == 10

    def test_custom_top_k(self):
        memory = _StubMemory(hits=[])
        search_marketplace(memory, "q", top_k=3)
        assert memory.calls[0]["top_k"] == 3

    def test_namespace_forwarded(self):
        memory = _StubMemory(hits=[])
        search_marketplace(memory, "q", namespace="acmeco")
        assert memory.calls[0]["namespace"] == "acmeco"

    def test_deep_forwarded(self):
        memory = _StubMemory(hits=[])
        search_marketplace(memory, "q", deep=True)
        assert memory.calls[0]["deep"] is True


# ---------------------------------------------------------------------------
# Result projection
# ---------------------------------------------------------------------------


class TestResultProjection:
    def test_hit_dict_becomes_listing(self):
        memory = _StubMemory(
            hits=[
                _hit(
                    "ent-1",
                    name="PDF Extractor",
                    description="Extract tables from PDFs.",
                    score=0.92,
                    tags=("pdf", "extract"),
                ),
            ]
        )
        result = search_marketplace(memory, "pdf")
        assert len(result.listings) == 1
        li = result.listings[0]
        assert li.entity_id == "ent-1"
        assert li.name == "PDF Extractor"
        assert li.description == "Extract tables from PDFs."
        assert li.score == 0.92
        assert li.tags == ("pdf", "extract")
        assert li.matched_via == "skill_description"

    def test_attr_style_hit_supported(self):
        class _AttrHit:
            entity_id = "ent-2"
            name = "Attr"
            description = "Attr style hit"
            score = 0.7
            tags = ["net"]
            matched_via = "skill_full"
            snippet = "snippet"

        result = search_marketplace(_StubMemory(hits=[_AttrHit()]), "q")
        assert result.listings[0].entity_id == "ent-2"
        assert result.listings[0].score == 0.7
        assert result.listings[0].tags == ("net",)

    def test_missing_entity_id_raises(self):
        memory = _StubMemory(hits=[{"name": "missing-id"}])
        with pytest.raises(MarketplaceError, match="missing required"):
            search_marketplace(memory, "q")

    def test_query_preserved_on_result(self):
        memory = _StubMemory(hits=[_hit("a")])
        result = search_marketplace(memory, "  PDFs!  ")
        assert result.query == "PDFs!"


# ---------------------------------------------------------------------------
# Sort + filter
# ---------------------------------------------------------------------------


class TestSortAndFilter:
    def test_sorted_by_score_descending(self):
        memory = _StubMemory(
            hits=[
                _hit("low", score=0.2),
                _hit("hi", score=0.95),
                _hit("mid", score=0.5),
            ]
        )
        result = search_marketplace(memory, "q")
        assert [li.entity_id for li in result.listings] == ["hi", "mid", "low"]

    def test_min_score_filters(self):
        memory = _StubMemory(
            hits=[
                _hit("low", score=0.2),
                _hit("hi", score=0.95),
                _hit("mid", score=0.5),
            ]
        )
        result = search_marketplace(memory, "q", min_score=0.5)
        # `mid` is 0.5 exactly — included (`>=`).
        assert [li.entity_id for li in result.listings] == ["hi", "mid"]

    def test_tags_filter_all_of(self):
        memory = _StubMemory(
            hits=[
                _hit("a", tags=("pdf", "table")),
                _hit("b", tags=("pdf",)),
                _hit("c", tags=("table", "io")),
            ]
        )
        result = search_marketplace(memory, "q", tags=["pdf", "table"])
        # Only `a` carries both required tags.
        assert [li.entity_id for li in result.listings] == ["a"]

    def test_empty_tag_list_doesnt_filter(self):
        memory = _StubMemory(
            hits=[_hit("a", tags=("pdf",)), _hit("b", tags=())]
        )
        result = search_marketplace(memory, "q", tags=[])
        # Both pass through (empty-tags filter is a no-op).
        assert {li.entity_id for li in result.listings} == {"a", "b"}


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


class TestErrorWrapping:
    def test_backend_exception_wrapped(self):
        memory = _StubMemory(raise_exc=RuntimeError("backend down"))
        with pytest.raises(MarketplaceError, match="search failed.*backend down"):
            search_marketplace(memory, "q")

    def test_wrapped_exception_chains_original(self):
        original = RuntimeError("original")
        memory = _StubMemory(raise_exc=original)
        try:
            search_marketplace(memory, "q")
        except MarketplaceError as exc:
            assert exc.__cause__ is original
        else:
            pytest.fail("expected MarketplaceError")
