"""Tests for the lineage-view data layer (TODO §1.3 P1).

The Textual modal is gated on §1 P0 multi-screen workflow, but
the projection + async fetch ship now so the modal lands as a
thin renderer. Tests pin the contract the modal will rely on.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import pytest

from care.runtime.lineage import (
    LineageError,
    LineageGraph,
    LineageNode,
    build_lineage_graph,
    fetch_chain_lineage,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


def _version(
    *,
    version_id: str,
    version_number: int,
    parents=(),
    depth: int = 0,
    fitness: float | None = None,
    generation: int | None = None,
    mutation_kind: str | None = None,
    objectives=(),
    change_summary: str | None = None,
    author: str | None = None,
    created_at: str = "2026-05-19T12:00:00+00:00",
) -> dict:
    meta: dict = {}
    if fitness is not None:
        meta["fitness_score"] = fitness
    if generation is not None:
        meta["generation"] = generation
    if mutation_kind:
        meta["mutation_kind"] = mutation_kind
    if objectives:
        meta["objectives"] = list(objectives)
    return {
        "version_id": version_id,
        "version_number": version_number,
        "parents": list(parents),
        "depth": depth,
        "created_at": created_at,
        "change_summary": change_summary,
        "author": author,
        "evolution_meta": meta or None,
    }


def _response(versions: list[dict], *, entity_id="chain-1", root="v-1", max_depth_reached=False) -> dict:
    return {
        "entity_id": entity_id,
        "root_version_id": root,
        "versions": versions,
        "max_depth_reached": max_depth_reached,
    }


# ---------------------------------------------------------------------------
# build_lineage_graph
# ---------------------------------------------------------------------------


class TestBuildLineageGraph:
    def test_minimal_root_only(self):
        resp = _response(
            [_version(version_id="v-1", version_number=1)],
        )
        graph = build_lineage_graph(resp)
        assert graph.entity_id == "chain-1"
        assert graph.root_version_id == "v-1"
        assert len(graph) == 1
        node = graph.nodes[0]
        assert node.version_id == "v-1"
        assert node.version_number == 1
        assert node.is_root is True
        assert node.fitness is None
        assert node.is_crossover is False

    def test_extracts_evolution_meta_fields(self):
        resp = _response(
            [
                _version(
                    version_id="v-2",
                    version_number=2,
                    parents=["v-1"],
                    depth=1,
                    fitness=0.84,
                    generation=3,
                    mutation_kind="crossover",
                    objectives=["accuracy", "latency"],
                ),
            ]
        )
        graph = build_lineage_graph(resp)
        node = graph.nodes[0]
        assert node.fitness == 0.84
        assert node.generation == 3
        assert node.mutation_kind == "crossover"
        assert node.objectives == ("accuracy", "latency")
        assert node.parents == ("v-1",)

    def test_legacy_fitness_key_fallback(self):
        resp = _response(
            [
                {
                    "version_id": "v-1",
                    "version_number": 1,
                    "parents": [],
                    "depth": 0,
                    "created_at": "2026-05-19T12:00:00+00:00",
                    "evolution_meta": {"fitness": 0.72},
                }
            ]
        )
        graph = build_lineage_graph(resp)
        assert graph.nodes[0].fitness == 0.72

    def test_fitness_score_wins_over_legacy(self):
        resp = _response(
            [
                {
                    "version_id": "v-1",
                    "version_number": 1,
                    "parents": [],
                    "depth": 0,
                    "created_at": "2026-05-19T12:00:00+00:00",
                    "evolution_meta": {
                        "fitness_score": 0.9,
                        "fitness": 0.1,
                    },
                }
            ]
        )
        assert build_lineage_graph(resp).nodes[0].fitness == 0.9

    def test_non_numeric_fitness_collapses_to_none(self):
        resp = _response(
            [
                {
                    "version_id": "v-1",
                    "version_number": 1,
                    "parents": [],
                    "depth": 0,
                    "created_at": "2026-05-19T12:00:00+00:00",
                    "evolution_meta": {"fitness_score": "not-a-number"},
                }
            ]
        )
        assert build_lineage_graph(resp).nodes[0].fitness is None

    def test_iso_datetime_parsed(self):
        resp = _response(
            [_version(version_id="v-1", version_number=1, created_at="2026-05-19T12:34:56Z")]
        )
        node = build_lineage_graph(resp).nodes[0]
        assert isinstance(node.created_at, datetime)
        assert node.created_at.year == 2026

    def test_missing_entity_id_raises(self):
        with pytest.raises(LineageError, match="entity_id"):
            build_lineage_graph({"root_version_id": "v-1", "versions": []})

    def test_missing_root_version_id_raises(self):
        with pytest.raises(LineageError, match="root_version_id"):
            build_lineage_graph({"entity_id": "c-1", "versions": []})

    def test_empty_versions_list_succeeds(self):
        graph = build_lineage_graph(
            {
                "entity_id": "chain-1",
                "root_version_id": "v-1",
                "versions": [],
            }
        )
        assert len(graph) == 0
        assert graph.root is None

    def test_unknown_fields_pass_through_in_evolution_meta(self):
        resp = _response(
            [
                {
                    "version_id": "v-1",
                    "version_number": 1,
                    "parents": [],
                    "depth": 0,
                    "created_at": "2026-05-19T12:00:00+00:00",
                    "evolution_meta": {
                        "fitness_score": 0.5,
                        "experimental_metric": 42,
                    },
                }
            ]
        )
        node = build_lineage_graph(resp).nodes[0]
        # Raw meta preserved so the modal can mine extras.
        assert node.evolution_meta["experimental_metric"] == 42

    def test_malformed_row_without_version_id_skipped(self):
        resp = _response(
            [
                _version(version_id="v-1", version_number=1),
                {"version_id": "", "version_number": 0},  # placeholder
                _version(version_id="v-2", version_number=2, parents=["v-1"], depth=1),
            ]
        )
        graph = build_lineage_graph(resp)
        assert len(graph) == 2
        assert [n.version_id for n in graph.nodes] == ["v-1", "v-2"]

    def test_sorted_by_depth_then_version_number(self):
        # Memory may return out-of-order rows; we re-sort for
        # canonical BFS layout.
        resp = _response(
            [
                _version(version_id="v-2", version_number=2, parents=["v-1"], depth=1),
                _version(version_id="v-1", version_number=1, depth=0),
                _version(
                    version_id="v-3", version_number=3, parents=["v-1"], depth=1
                ),
            ]
        )
        graph = build_lineage_graph(resp)
        assert [n.version_id for n in graph.nodes] == ["v-1", "v-2", "v-3"]

    def test_crossover_detection(self):
        resp = _response(
            [
                _version(version_id="v-3", version_number=3, parents=["v-1", "v-2"], depth=2)
            ]
        )
        node = build_lineage_graph(resp).nodes[0]
        assert node.is_crossover is True
        assert node.is_root is False


# ---------------------------------------------------------------------------
# LineageGraph methods
# ---------------------------------------------------------------------------


class TestLineageGraph:
    def _sample(self) -> LineageGraph:
        resp = _response(
            [
                _version(version_id="v-1", version_number=1, depth=0, fitness=0.5),
                _version(
                    version_id="v-2",
                    version_number=2,
                    parents=["v-1"],
                    depth=1,
                    fitness=0.7,
                ),
                _version(
                    version_id="v-3",
                    version_number=3,
                    parents=["v-1"],
                    depth=1,
                    fitness=0.6,
                ),
                _version(
                    version_id="v-4",
                    version_number=4,
                    parents=["v-2", "v-3"],
                    depth=2,
                    fitness=0.83,
                ),
            ]
        )
        return build_lineage_graph(resp)

    def test_find_existing(self):
        graph = self._sample()
        node = graph.find("v-2")
        assert node is not None
        assert node.version_number == 2

    def test_find_unknown_returns_none(self):
        assert self._sample().find("v-999") is None

    def test_children_of(self):
        graph = self._sample()
        kids = graph.children_of("v-1")
        assert {k.version_id for k in kids} == {"v-2", "v-3"}
        # v-4 has both v-2 and v-3 as parents.
        assert {k.version_id for k in graph.children_of("v-2")} == {"v-4"}
        assert {k.version_id for k in graph.children_of("v-3")} == {"v-4"}

    def test_layers_grouped_by_depth(self):
        layers = self._sample().layers()
        assert len(layers) == 3
        assert [n.version_id for n in layers[0]] == ["v-1"]
        assert {n.version_id for n in layers[1]} == {"v-2", "v-3"}
        assert [n.version_id for n in layers[2]] == ["v-4"]

    def test_empty_graph_layers(self):
        empty = build_lineage_graph(
            {"entity_id": "c", "root_version_id": "v", "versions": []}
        )
        assert empty.layers() == []

    def test_root(self):
        assert self._sample().root.version_id == "v-1"

    def test_best_picks_highest_fitness(self):
        assert self._sample().best().version_id == "v-4"

    def test_best_none_when_no_fitness(self):
        graph = build_lineage_graph(
            _response([_version(version_id="v-1", version_number=1)])
        )
        assert graph.best() is None

    def test_fitness_delta_single_parent(self):
        # v-2 has fitness 0.7, parent v-1 has 0.5 → delta +0.2
        delta = self._sample().fitness_delta("v-2")
        assert delta == pytest.approx(0.2, abs=1e-9)

    def test_fitness_delta_crossover_uses_best_parent(self):
        # v-4 (fitness 0.83) has parents v-2 (0.7) and v-3 (0.6).
        # Best parent is 0.7 → delta = 0.13.
        delta = self._sample().fitness_delta("v-4")
        assert delta == pytest.approx(0.13, abs=1e-9)

    def test_fitness_delta_no_parent_fitness(self):
        # Root has no parent → None.
        assert self._sample().fitness_delta("v-1") is None

    def test_iteration_yields_nodes(self):
        graph = self._sample()
        assert list(graph) == list(graph.nodes)


# ---------------------------------------------------------------------------
# fetch_chain_lineage
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(self, *, response=None, exc=None, delay=0.0):
        self._response = response
        self._exc = exc
        self._delay = delay
        self.calls: list[dict] = []

    def get_chain_lineage(self, entity_id, *, channel="latest", version_id=None, max_depth=10):
        self.calls.append(
            {
                "entity_id": entity_id,
                "channel": channel,
                "version_id": version_id,
                "max_depth": max_depth,
            }
        )
        if self._delay:
            time.sleep(self._delay)
        if self._exc:
            raise self._exc
        return self._response


class _StubMemory:
    def __init__(self, client):
        self.client = client


class TestFetchChainLineage:
    def test_happy_path(self):
        resp = _response(
            [_version(version_id="v-1", version_number=1)]
        )
        memory = _StubMemory(_StubClient(response=resp))
        graph = asyncio.run(fetch_chain_lineage(memory, "chain-1"))
        assert graph.entity_id == "chain-1"
        call = memory.client.calls[0]
        assert call["entity_id"] == "chain-1"
        assert call["channel"] == "latest"
        assert call["max_depth"] == 10
        assert call["version_id"] is None

    def test_forwards_all_kwargs(self):
        resp = _response(
            [_version(version_id="v-1", version_number=1)]
        )
        memory = _StubMemory(_StubClient(response=resp))
        asyncio.run(
            fetch_chain_lineage(
                memory,
                "chain-1",
                channel="evolved",
                version_id="v-7",
                max_depth=25,
            )
        )
        call = memory.client.calls[0]
        assert call["channel"] == "evolved"
        assert call["version_id"] == "v-7"
        assert call["max_depth"] == 25

    def test_underscored_client_attr_also_works(self):
        resp = _response(
            [_version(version_id="v-1", version_number=1)]
        )

        class _MemoryUnderscored:
            def __init__(self, client):
                self._client = client

        client = _StubClient(response=resp)
        graph = asyncio.run(
            fetch_chain_lineage(_MemoryUnderscored(client), "chain-1")
        )
        assert graph.entity_id == "chain-1"

    def test_missing_client_raises(self):
        with pytest.raises(LineageError, match="get_chain_lineage"):
            asyncio.run(fetch_chain_lineage(object(), "chain-1"))

    def test_sdk_exception_wraps(self):
        memory = _StubMemory(_StubClient(exc=RuntimeError("503 Service Unavailable")))
        with pytest.raises(LineageError, match="lineage fetch failed"):
            asyncio.run(fetch_chain_lineage(memory, "chain-1"))

    def test_timeout(self):
        resp = _response([_version(version_id="v-1", version_number=1)])
        memory = _StubMemory(_StubClient(response=resp, delay=0.5))
        with pytest.raises(LineageError, match="timed out"):
            asyncio.run(
                fetch_chain_lineage(memory, "chain-1", timeout=0.05)
            )

    def test_malformed_response_raises_lineage_error(self):
        memory = _StubMemory(_StubClient(response={"versions": []}))
        with pytest.raises(LineageError, match="entity_id"):
            asyncio.run(fetch_chain_lineage(memory, "chain-1"))


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            LineageGraph as G,
            LineageNode as N,
            LineageError as E,
            build_lineage_graph as build,
            fetch_chain_lineage as fetch,
        )

        assert G is LineageGraph
        assert N is LineageNode
        assert E is LineageError
        assert build is build_lineage_graph
        assert fetch is fetch_chain_lineage


# ---------------------------------------------------------------------------
# SDK-shape integration (model objects, not dicts)
# ---------------------------------------------------------------------------


class _FakeLineageVersion:
    """Mimics ``gigaevo_client.models.LineageVersion`` — attribute
    access, not dict. Tests the duck-typed projection path
    against SDK-shaped objects."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeLineageResponse:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestSDKShape:
    def test_attribute_access_objects_work(self):
        resp = _FakeLineageResponse(
            entity_id="chain-1",
            root_version_id="v-1",
            versions=[
                _FakeLineageVersion(
                    version_id="v-1",
                    version_number=1,
                    parents=[],
                    depth=0,
                    evolution_meta={"fitness_score": 0.42},
                    change_summary="initial",
                    author="alice",
                    created_at=datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
                )
            ],
            max_depth_reached=False,
        )
        graph = build_lineage_graph(resp)
        assert graph.entity_id == "chain-1"
        node = graph.nodes[0]
        assert node.fitness == 0.42
        assert node.change_summary == "initial"
        assert node.author == "alice"
