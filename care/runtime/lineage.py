"""Lineage-view data layer (TODO §1.3 P1).

CARE's LibraryScreen exposes a "Show lineage" action on any saved
agent — opens a modal rendering the chain's evolution DAG with
fitness deltas, parent links, and a one-click "rerun from this
node" affordance. The Textual modal is gated on §1 P0 multi-screen
workflow, but the data layer is independent and well-bounded —
this module ships it now so the modal lands as a thin renderer.

What this module provides:

* :class:`LineageNode` — frozen per-version row enriched with a
  CARE-specific projection of ``evolution_meta`` (fitness +
  generation + objectives + parent ids + mutation kind), plus
  a derived ``is_root`` flag the modal uses to draw the tree
  spine.
* :class:`LineageGraph` — aggregate over BFS-ordered nodes with
  navigation helpers (:meth:`layers`, :meth:`children_of`,
  :meth:`best`, :meth:`find`) so the modal can render layer by
  layer without re-walking the parent links itself.
* :func:`fetch_chain_lineage` — async helper wrapping the SDK's
  sync ``get_chain_lineage`` in :func:`asyncio.to_thread` with
  a per-call deadline so the modal doesn't freeze on a hung
  Memory instance.
* :func:`build_lineage_graph` — pure projection from a
  :class:`gigaevo_client.models.LineageResponse` shape into the
  CARE-side view. Duck-typed so tests don't need the SDK
  installed.

Duck-typed boundaries: the fetch helper accepts any
``CareMemory``-like facade exposing a ``client`` with
``get_chain_lineage``. The pure projection accepts dicts OR the
real SDK models — same field names either way. Tests inject
plain dicts.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LineageError(RuntimeError):
    """Raised when lineage retrieval / projection fails:
    unreachable Memory, timeout, or a malformed response that
    can't be projected. The modal catches this and shows a
    friendly toast."""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineageNode:
    """One version in the lineage DAG.

    Mirrors :class:`gigaevo_client.models.LineageVersion` but
    extracts the CARE-rendered facts so the modal doesn't have
    to re-read ``evolution_meta`` on every refresh. Frozen so
    snapshots flow through Textual messages without defensive
    copies.

    Fields:
        version_id: Stable id from Memory; the modal links each
            node to a "Re-run from here" action via this id.
        version_number: 1-based monotonically increasing — drives
            the "v3" label.
        parents: List of ancestor ``version_id`` values. Empty
            on the root.
        depth: BFS depth from the starting version (0 for the
            head / queried version). Memory returns this directly.
        created_at: Wall-clock the version was saved.
        change_summary: Author-supplied short description.
        author: Optional author tag (Memory may omit).
        fitness: Primary-objective score extracted from
            ``evolution_meta["fitness_score"]`` (or the legacy
            ``"fitness"`` key). ``None`` for non-evolved nodes.
        generation: Generation number from the GA loop, if any.
        mutation_kind: ``"crossover"`` / ``"mutation"`` / etc.
            from ``evolution_meta``.
        objectives: List of objective names this version was
            scored on.
        evolution_meta: Raw JSONB so callers can mine extras.
    """

    version_id: str
    version_number: int
    parents: tuple[str, ...] = ()
    depth: int = 0
    created_at: Optional[datetime] = None
    change_summary: Optional[str] = None
    author: Optional[str] = None
    fitness: Optional[float] = None
    generation: Optional[int] = None
    mutation_kind: Optional[str] = None
    objectives: tuple[str, ...] = ()
    evolution_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        """``True`` when this node has no parents — the spine of
        the tree."""
        return len(self.parents) == 0

    @property
    def is_crossover(self) -> bool:
        """``True`` when this version has more than one parent
        — used to draw the merge marker in the tree."""
        return len(self.parents) > 1


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineageGraph:
    """Full ancestry DAG of a chain.

    BFS-ordered: the root version sits first; deeper layers
    follow. Frozen so the modal can hold a snapshot safely.

    Use :meth:`layers` to iterate by depth (canonical rendering
    order), :meth:`children_of` to draw edges, :meth:`best` to
    highlight the highest-fitness version, and :meth:`find` for
    O(N) lookup by version_id (N is small — Memory caps at 100).
    """

    entity_id: str
    root_version_id: str
    nodes: tuple[LineageNode, ...] = ()
    max_depth_reached: bool = False

    # --- Lookups --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def find(self, version_id: str) -> Optional[LineageNode]:
        """Locate a node by ``version_id`` or return ``None``."""
        for node in self.nodes:
            if node.version_id == version_id:
                return node
        return None

    def children_of(self, version_id: str) -> tuple[LineageNode, ...]:
        """Return every node that lists ``version_id`` in its
        ``parents``. The lineage walk is ancestor-first
        (parents → root), so children are inverse-direction. The
        modal uses this to render branching."""
        return tuple(n for n in self.nodes if version_id in n.parents)

    def layers(self) -> list[list[LineageNode]]:
        """Group nodes by ``depth`` ascending. Root layer first,
        deepest layer last. Empty list when the graph has no
        nodes (defensive against an empty Memory response)."""
        if not self.nodes:
            return []
        max_depth = max(n.depth for n in self.nodes)
        result: list[list[LineageNode]] = [[] for _ in range(max_depth + 1)]
        for node in self.nodes:
            result[node.depth].append(node)
        return result

    # --- Analytics ------------------------------------------------------

    @property
    def root(self) -> Optional[LineageNode]:
        """The version with depth=0 / no parents — modal renders
        it at the top of the tree. ``None`` when the graph is
        empty."""
        for node in self.nodes:
            if node.is_root:
                return node
        # Fall back to depth=0 if every node has parents (e.g.
        # the caller queried from a leaf and the response stops
        # short of the actual root because of `max_depth`).
        return self.nodes[0] if self.nodes else None

    def best(self) -> Optional[LineageNode]:
        """Highest-fitness node, or ``None`` when no node has a
        fitness score. Used by the modal to highlight the
        evolved winner — typically what the user wants to
        re-run."""
        scored = [n for n in self.nodes if n.fitness is not None]
        if not scored:
            return None
        return max(scored, key=lambda n: n.fitness or 0.0)

    def fitness_delta(self, version_id: str) -> Optional[float]:
        """Fitness improvement over the parent of ``version_id``.

        For a crossover (multiple parents) we use the best
        parent's score as the baseline — that's the "did this
        node actually win" check. Returns ``None`` when the
        node, the parent, or either fitness score is missing.
        """
        node = self.find(version_id)
        if node is None or node.fitness is None:
            return None
        parent_scores = [
            p.fitness
            for p in (self.find(pid) for pid in node.parents)
            if p is not None and p.fitness is not None
        ]
        if not parent_scores:
            return None
        return node.fitness - max(parent_scores)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def build_lineage_graph(response: Any) -> LineageGraph:
    """Project a Memory SDK ``LineageResponse`` (or a dict with
    the same fields) into a :class:`LineageGraph`.

    Tolerant projection — missing optional fields default,
    unknown extras pass through into ``LineageNode.evolution_meta``
    so future Memory additions don't break CARE's modal.

    Raises:
        LineageError: ``response`` is missing the required
            ``entity_id`` / ``root_version_id`` / ``versions``
            fields. The modal converts this into a friendly toast.
    """
    entity_id = _read(response, "entity_id")
    root_version_id = _read(response, "root_version_id")
    raw_versions = _read(response, "versions") or []
    max_depth_reached = bool(_read(response, "max_depth_reached") or False)

    if not entity_id:
        raise LineageError("lineage response missing entity_id")
    if not root_version_id:
        raise LineageError("lineage response missing root_version_id")

    nodes: list[LineageNode] = []
    for raw in raw_versions:
        evolution_meta = _read(raw, "evolution_meta") or {}
        if not isinstance(evolution_meta, dict):
            evolution_meta = {}
        node = LineageNode(
            version_id=str(_read(raw, "version_id") or ""),
            version_number=int(_read(raw, "version_number") or 0),
            parents=tuple(_read(raw, "parents") or ()),
            depth=int(_read(raw, "depth") or 0),
            created_at=_coerce_datetime(_read(raw, "created_at")),
            change_summary=_read(raw, "change_summary"),
            author=_read(raw, "author"),
            fitness=_extract_fitness(evolution_meta),
            generation=_extract_generation(evolution_meta),
            mutation_kind=_extract_mutation_kind(evolution_meta),
            objectives=_extract_objectives(evolution_meta),
            evolution_meta=dict(evolution_meta),
        )
        if not node.version_id:
            # Skip malformed rows rather than raising — Memory
            # may sneak in a placeholder during evolution mid-
            # write; the modal renders what's complete.
            continue
        nodes.append(node)

    # Sort by depth ascending for canonical BFS layout. Memory's
    # response is *typically* already BFS-ordered, but we don't
    # rely on it — a re-sort keeps the contract crisp.
    nodes.sort(key=lambda n: (n.depth, n.version_number))

    return LineageGraph(
        entity_id=str(entity_id),
        root_version_id=str(root_version_id),
        nodes=tuple(nodes),
        max_depth_reached=max_depth_reached,
    )


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------


async def fetch_chain_lineage(
    memory: Any,
    entity_id: str,
    *,
    channel: str = "latest",
    version_id: str | None = None,
    max_depth: int = 10,
    timeout: float = 10.0,
) -> LineageGraph:
    """Fetch a chain's lineage DAG via Memory and project to
    :class:`LineageGraph`.

    Wraps the sync SDK call in :func:`asyncio.to_thread` with a
    deadline so the modal doesn't freeze on a hung server. Any
    error — timeout, HTTP failure, malformed response — surfaces
    as :class:`LineageError` so the modal's toast handler stays
    single-branch.

    Args:
        memory: A ``CareMemory`` facade (or any object exposing
            ``.client.get_chain_lineage(...)``). Tests pass a
            stub.
        entity_id: Chain to walk.
        channel: Start from the version pinned to this channel.
            Ignored when ``version_id`` is supplied.
        version_id: Walk from a specific historical version.
        max_depth: BFS depth cap (1-100; Memory clamps).
        timeout: Per-call deadline in seconds.

    Returns:
        :class:`LineageGraph`.

    Raises:
        LineageError: Memory was unreachable, timed out, or
            returned a malformed response.
    """
    client = getattr(memory, "client", None) or getattr(
        memory, "_client", None
    )
    if client is None or not hasattr(client, "get_chain_lineage"):
        raise LineageError(
            "memory facade does not expose get_chain_lineage()"
        )

    start = time.monotonic()
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.get_chain_lineage,
                entity_id,
                channel=channel,
                version_id=version_id,
                max_depth=max_depth,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        latency = (time.monotonic() - start) * 1000
        raise LineageError(
            f"lineage fetch timed out after {timeout:.1f}s ({latency:.0f}ms elapsed)"
        ) from exc
    except LineageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LineageError(
            f"lineage fetch failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        return build_lineage_graph(response)
    except LineageError:
        raise
    except Exception as exc:  # noqa: BLE001
        # build_lineage_graph coerces version_number/depth via int() and
        # iterates parents — a malformed row would raise a raw traceback
        # outside the fetch guard. Surface the friendly LineageError instead.
        raise LineageError(
            f"lineage parse failed: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(obj: Any, name: str) -> Any:
    """Read ``name`` off a SDK model OR a dict — both shapes are
    supported so tests don't need the SDK installed."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _coerce_datetime(value: Any) -> Optional[datetime]:
    """Accept already-parsed ``datetime``, ISO-8601 string, or
    None. Anything else collapses to ``None`` so a flaky
    Memory response doesn't crash the modal."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _extract_fitness(meta: dict[str, Any]) -> Optional[float]:
    """Read the primary-objective fitness out of ``evolution_meta``.

    Mirrors the precedence Memory uses server-side:
    ``fitness_score`` (the §5 P1 standardised key) wins over the
    legacy ``fitness`` key. Non-numeric values collapse to
    ``None`` rather than crashing the projection.
    """
    for key in ("fitness_score", "fitness"):
        if key in meta:
            try:
                return float(meta[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_generation(meta: dict[str, Any]) -> Optional[int]:
    value = meta.get("generation")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_mutation_kind(meta: dict[str, Any]) -> Optional[str]:
    value = meta.get("mutation_kind")
    return str(value) if value else None


def _extract_objectives(meta: dict[str, Any]) -> tuple[str, ...]:
    value = meta.get("objectives")
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(v) for v in value)


__all__ = [
    "LineageError",
    "LineageGraph",
    "LineageNode",
    "build_lineage_graph",
    "fetch_chain_lineage",
]
