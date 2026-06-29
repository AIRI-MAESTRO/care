"""Capability marketplace browser (TODO §8 P2).

The marketplace is the cross-user discovery surface: a CARE user
types "I need to extract tables from a PDF" and the browser shows
every `agent_skill` shared across Memory that matches, ranked by
relevance, with provenance + tags + match-quality so the user can
pick what to install.

This module is the **data layer** the future `MarketplaceScreen`
will render. The screen itself is gated on the §1
``LibraryScreen`` work (which hasn't shipped yet); the function
here can already be called from the CLI / scripts / library
code, so it's useful standalone.

Memory's cross-namespace query support (Memory TODO §1.X) is the
upstream gating item — the SDK's
:func:`gigaevo_client.GigaEvoClient.find_capability_matches`
already accepts ``namespace=None`` for "default scope" today.
Once Memory adds explicit cross-namespace search, this function
forwards the kwarg through without changing its public surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MarketplaceError(RuntimeError):
    """Raised when the marketplace can't satisfy a query — bad
    backend, malformed result shape, network error wrapped."""


@dataclass(frozen=True)
class MarketplaceListing:
    """One AgentSkill on the marketplace.

    Frozen so the result tuple is safe to log / pass across
    screens without defensive copies.

    Fields:
        entity_id: Memory entity identifier — the user installs
            by calling
            ``client.get_agent_skill(entity_id, channel="latest")``
            and dropping the resolved files into the user's
            skill cache (CARL §5.9).
        name: Display name from the SKILL.md ``name`` field.
        description: One-line summary; may be empty when the
            SKILL.md author didn't supply one.
        score: 0-1 relevance score from the search backend.
            Higher is better.
        tags: SKILL.md ``tags`` — drive the screen's filter
            chips.
        matched_via: Which doc kind produced the match
            (``"skill_description"``, ``"skill_instructions"``,
            etc.) — lets the screen badge high-signal matches.
        snippet: Excerpt from the matching document; can be
            ``None`` when the backend didn't supply one.
    """

    entity_id: str
    name: str
    description: str = ""
    score: float = 0.0
    tags: tuple[str, ...] = field(default_factory=tuple)
    matched_via: str | None = None
    snippet: str | None = None


@dataclass(frozen=True)
class MarketplaceResult:
    """Aggregate of every listing the query matched."""

    listings: tuple[MarketplaceListing, ...] = field(default_factory=tuple)
    query: str = ""
    """The normalised query string the search used — useful for
    UI breadcrumbs."""

    @property
    def is_empty(self) -> bool:
        return len(self.listings) == 0

    def by_tag(self, tag: str) -> tuple[MarketplaceListing, ...]:
        """Filter listings to ones carrying ``tag``."""
        return tuple(li for li in self.listings if tag in li.tags)


def search_marketplace(
    memory: Any,
    query: str,
    *,
    top_k: int = 10,
    min_score: float = 0.0,
    tags: list[str] | None = None,
    namespace: str | None = None,
    deep: bool = False,
) -> MarketplaceResult:
    """Search the shared `agent_skill` catalog for capabilities
    matching ``query``.

    Args:
        memory: Anything exposing
            ``find_capability_matches(rough_aim, top_k, *,
            namespace=None, deep=False)`` — typically a
            :class:`care.CareMemory.client` or a
            :class:`gigaevo_client.GigaEvoClient`. Tests pass a
            stub.
        query: Free-text capability description. Empty / whitespace
            returns an empty result without calling the backend.
        top_k: Max listings to request from the backend. Note the
            server may apply its own cap.
        min_score: Drop listings whose backend score is strictly
            below this threshold. Default ``0.0`` keeps everything
            the backend returned (it already applies its own
            relevance gating).
        tags: When supplied, listings must carry **all** of these
            tags. CARE-side filter, applied after the backend
            returns its top-K, so a tag-tight query doesn't
            crowd out high-relevance hits.
        namespace: Forward to the SDK. ``None`` is the SDK's
            default-scope search; once Memory ships cross-namespace
            queries (TODO upstream), passing ``None`` will widen
            the search automatically.
        deep: Forward to the SDK's deep-search mode (matches
            against ``skill_instructions`` in addition to
            ``skill_description``). Useful for natural-language
            queries that paraphrase the skill body.

    Returns:
        :class:`MarketplaceResult` sorted by score descending. The
        list is empty when ``query`` is empty / whitespace, or
        when the backend returned no matches.

    Raises:
        MarketplaceError: When the backend raised an exception
            during the search. Wraps the original via ``from``
            so callers can chain-introspect.
    """
    normalised = (query or "").strip()
    if not normalised:
        return MarketplaceResult(query=normalised)

    try:
        raw_hits = memory.find_capability_matches(
            normalised,
            top_k=top_k,
            namespace=namespace,
            deep=deep,
        )
    except Exception as exc:  # noqa: BLE001
        raise MarketplaceError(
            f"marketplace search failed: {exc}"
        ) from exc

    listings = [_listing_from_hit(h) for h in (raw_hits or [])]

    if min_score > 0.0:
        listings = [li for li in listings if li.score >= min_score]

    if tags:
        tag_set = set(tags)
        listings = [
            li for li in listings if tag_set.issubset(set(li.tags))
        ]

    listings.sort(key=lambda li: li.score, reverse=True)
    return MarketplaceResult(
        listings=tuple(listings),
        query=normalised,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _listing_from_hit(hit: Any) -> MarketplaceListing:
    """Project a SDK ``CapabilityHit``-shaped object into a
    :class:`MarketplaceListing`.

    Duck-typed — the SDK's ``CapabilityHit`` is a Pydantic model;
    tests pass plain dicts / namespace-style objects with the
    same fields. Anything missing falls back to the defaults on
    :class:`MarketplaceListing`.
    """
    entity_id = _get(hit, "entity_id", "")
    if not entity_id:
        raise MarketplaceError(
            "marketplace hit missing required `entity_id`"
        )
    return MarketplaceListing(
        entity_id=str(entity_id),
        name=str(_get(hit, "name", entity_id)),
        description=str(_get(hit, "description", "") or ""),
        score=float(_get(hit, "score", 0.0) or 0.0),
        tags=tuple(_get(hit, "tags", ()) or ()),
        matched_via=_get(hit, "matched_via", None),
        snippet=_get(hit, "snippet", None),
    )


def _get(hit: Any, name: str, default: Any) -> Any:
    """Attribute-or-key getter so the function works against both
    Pydantic models and plain dicts."""
    if isinstance(hit, dict):
        return hit.get(name, default)
    return getattr(hit, name, default)


__all__ = [
    "MarketplaceError",
    "MarketplaceListing",
    "MarketplaceResult",
    "search_marketplace",
]
