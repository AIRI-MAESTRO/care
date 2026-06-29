"""Redis-probe fallback in CarePlatform._poll_experiment_events.

When the Platform's ``/results`` returns no live metrics (the common
local-stack case), the poll loop should fill the fitness curve + Programs
chart from gigavolve Redis so the EvolutionScreen isn't blank. This locks
in that the previously-dead ``probe_fitness_history`` + the new
``probe_programs_counts`` are actually wired.
"""

from __future__ import annotations

from unittest.mock import patch

import care.platform as platform
from care.platform import (
    CarePlatform,
    _extract_validation_rubric,
    _platform_fitness_history_looks_bogus,
    _resolve_display_generation,
)


class TestExtractValidationRubric:
    def test_recovers_stamped_rubric(self) -> None:
        desc = (
            "CARE-driven chain evolution. base_chain_id=abc.\n\n"
            "Validation rubric (user intent):\n"
            "Answers must be concise and factually correct."
        )
        assert (
            _extract_validation_rubric(desc)
            == "Answers must be concise and factually correct."
        )

    def test_none_when_no_rubric_block(self) -> None:
        assert _extract_validation_rubric("just a plain description") is None
        assert _extract_validation_rubric(None) is None


class _FakeClient:
    """Minimal PlatformClient stand-in for the poll loop."""

    def __init__(self, statuses: list[dict], results) -> None:
        self._statuses = statuses
        self._results = results
        self.results_calls = 0

    def get_status(self, experiment_id: str) -> dict:
        if self._statuses:
            return self._statuses.pop(0)
        return {"status": "completed"}

    def get_results(self, experiment_id: str):
        self.results_calls += 1
        return self._results


def _drain(plat: CarePlatform, experiment_id: str) -> list[dict]:
    events: list[dict] = []
    for event in plat._poll_experiment_events(experiment_id, interval=0.0):
        events.append(event)
        if len(events) > 50:  # safety net
            break
    return events


def test_probe_fills_fitness_and_programs_when_results_empty() -> None:
    client = _FakeClient(
        statuses=[{"status": "running"}, {"status": "completed"}],
        results={},  # no metrics → fallback must kick in
    )
    plat = CarePlatform(client)
    history = [
        {"generation": 0, "best_fitness": 0.1},
        {"generation": 1, "best_fitness": 0.25},
    ]
    with patch.object(platform, "_probe_live_generation", return_value=1), patch.object(
        platform, "_probe_live_best_fitness", return_value=None
    ), patch.object(
        platform, "_probe_live_fitness_history", return_value=history
    ), patch.object(
        platform, "_probe_live_programs_counts", return_value=(7, 2)
    ), patch(
        "time.sleep", return_value=None
    ):
        events = _drain(plat, "exp_abc")

    kinds = [e["event"] for e in events]
    assert "fitness_history_snapshot" in kinds
    assert "programs_snapshot" in kinds

    fh = next(e for e in events if e["event"] == "fitness_history_snapshot")
    assert fh["data"]["history"] == history
    assert fh["data"]["source"] == "redis_probe"

    ps = next(e for e in events if e["event"] == "programs_snapshot")
    assert ps["data"]["programs_valid"] == 7
    assert ps["data"]["programs_invalid"] == 2
    assert ps["data"]["source"] == "redis_probe"


class TestExperimentSseWithFallback:
    def test_uses_sse_when_available(self) -> None:
        frames = [
            {"event": "status", "data": {"status": "running"}},
            {"event": "completed", "data": {"status": "completed"}},
        ]

        class _Client:
            def stream_experiment_events(self, eid):
                assert eid == "exp_x"
                yield from frames

        plat = CarePlatform(_Client())
        assert list(plat.stream_events("exp_x")) == frames

    def test_falls_back_to_polling_when_sse_raises(self, monkeypatch) -> None:
        class _Client:
            def stream_experiment_events(self, eid):
                raise RuntimeError("404 not found")
                yield  # pragma: no cover — makes this a generator

        plat = CarePlatform(_Client())
        sentinel = [{"event": "status", "data": {"status": "polled"}}]
        monkeypatch.setattr(
            plat, "_poll_experiment_events", lambda eid, **kw: iter(sentinel)
        )
        assert list(plat.stream_events("exp_x")) == sentinel

    def test_falls_back_when_client_lacks_method(self, monkeypatch) -> None:
        class _Client:
            pass

        plat = CarePlatform(_Client())
        sentinel = [{"event": "x", "data": {}}]
        monkeypatch.setattr(
            plat, "_poll_experiment_events", lambda eid, **kw: iter(sentinel)
        )
        assert list(plat.stream_events("exp_x")) == sentinel

    def test_env_disable_forces_polling(self, monkeypatch) -> None:
        class _Client:
            def stream_experiment_events(self, eid):
                yield {"event": "sse", "data": {}}

        plat = CarePlatform(_Client())
        sentinel = [{"event": "polled", "data": {}}]
        monkeypatch.setattr(
            plat, "_poll_experiment_events", lambda eid, **kw: iter(sentinel)
        )
        monkeypatch.setenv("CARE_PLATFORM__EXPERIMENT_SSE", "0")
        assert list(plat.stream_events("exp_x")) == sentinel

    def test_empty_sse_stream_returns_without_polling(self, monkeypatch) -> None:
        """An SSE endpoint that closes immediately (no frames) is a clean
        end-of-stream, not an error — we must NOT then fall back to polling
        (which would re-poll an already-finished experiment)."""
        class _Client:
            def stream_experiment_events(self, eid):
                return
                yield  # pragma: no cover — makes this a generator

        plat = CarePlatform(_Client())
        polled = {"called": False}

        def _poll(eid, **kw):
            polled["called"] = True
            return iter([{"event": "polled", "data": {}}])

        monkeypatch.setattr(plat, "_poll_experiment_events", _poll)
        assert list(plat.stream_events("exp_x")) == []
        assert polled["called"] is False


class TestAcceptChainOverride:
    """Chain-experiment accept must promote the SELECTED individual's chain
    when one is supplied, falling back to the run's overall best."""

    def _plat_mem(self, best_chain):
        from types import SimpleNamespace

        class _Client:
            def get_results(self, eid):
                return {"best_chain_config": best_chain}

            def get_experiment(self, eid):
                return {
                    "name": "run",
                    "description": "base_chain_id=abcdef12-3456-7890-abcd-ef1234567890",
                }

        saved: dict = {}

        class _Mem:
            def save_chain(self, content, *, name, entity_id, channel):
                saved["content"] = content
                saved["entity_id"] = entity_id
                saved["channel"] = channel
                return SimpleNamespace(entity_id=entity_id, version=3)

        return CarePlatform(_Client()), _Mem(), saved

    def test_promotes_override_when_provided(self) -> None:
        best = {"steps": [{"id": "best"}]}
        override = {"steps": [{"id": "selected"}]}
        plat, mem, saved = self._plat_mem(best)
        plat.accept_individual("exp_x", "prog-7", memory=mem, chain_override=override)
        assert saved["content"] == override
        assert saved["channel"] == "stable"

    def test_falls_back_to_best_without_override(self) -> None:
        best = {"steps": [{"id": "best"}]}
        plat, mem, saved = self._plat_mem(best)
        plat.accept_individual("exp_x", "prog-7", memory=mem, chain_override=None)
        assert saved["content"] == best

    def test_falls_back_when_override_lacks_steps(self) -> None:
        best = {"steps": [{"id": "best"}]}
        plat, mem, saved = self._plat_mem(best)
        plat.accept_individual("exp_x", "prog-7", memory=mem, chain_override={"x": 1})
        assert saved["content"] == best


class TestRunningEvolutionCount:
    def test_counts_only_active_statuses(self) -> None:
        class _Client:
            def list_evolutions(self, **kwargs):
                return {
                    "items": [
                        {"status": "running"},
                        {"status": "initializing"},
                        {"status": "dispatching"},
                        {"status": "queued"},  # not "running now"
                        {"status": "completed"},
                        {"status": "failed"},
                    ]
                }

        plat = CarePlatform(_Client())
        assert plat.running_evolution_count() == 3

    def test_zero_on_error(self) -> None:
        class _Client:
            def list_evolutions(self, **kwargs):
                raise RuntimeError("platform down")

        plat = CarePlatform(_Client())
        assert plat.running_evolution_count() == 0

    def test_handles_alt_envelope_keys(self) -> None:
        class _Client:
            def list_evolutions(self, **kwargs):
                return {"evolutions": [{"status": "running"}]}

        plat = CarePlatform(_Client())
        assert plat.running_evolution_count() == 1


def test_poll_emits_cost_tick_delta_from_results_tokens() -> None:
    """When /results metrics carry cumulative ``total_tokens``, the poll loop
    emits a cost_tick with the *delta* so CARE's additive cost aggregator
    stays correct."""
    client = _FakeClient(
        statuses=[{"status": "running"}, {"status": "completed"}],
        results={"metrics": {"total_tokens": 1200}},
    )
    plat = CarePlatform(client)
    with patch.object(platform, "_probe_live_generation", return_value=0), patch.object(
        platform, "_probe_live_best_fitness", return_value=None
    ), patch.object(
        platform, "_probe_live_fitness_history", return_value=[]
    ), patch.object(
        platform, "_probe_live_programs_counts", return_value=(None, None)
    ), patch("time.sleep", return_value=None):
        events = _drain(plat, "exp_tok")

    ticks = [e for e in events if e["event"] == "cost_tick"]
    assert len(ticks) == 1
    assert ticks[0]["data"]["total_tokens"] == 1200


def test_final_results_emitted_on_terminal_poll() -> None:
    """A run that completes between polls must still surface its final
    ``/results`` (fitness curve / programs / tokens) — not just a bare
    ``completed`` event. The metric frames must precede the terminal one."""
    client = _FakeClient(
        statuses=[{"status": "completed"}],
        results={
            "metrics": {
                "fitness_history": [
                    {"generation": 0, "best_fitness": 0.2},
                    {"generation": 1, "best_fitness": 0.7},
                ],
                "programs_valid": 9,
                "programs_invalid": 1,
                "total_tokens": 5000,
            }
        },
    )
    plat = CarePlatform(client)
    with patch.object(platform, "_probe_live_generation", return_value=None), patch.object(
        platform, "_probe_live_best_fitness", return_value=None
    ), patch.object(
        platform, "_probe_live_fitness_history", return_value=[]
    ), patch.object(
        platform, "_probe_live_programs_counts", return_value=(None, None)
    ), patch("time.sleep", return_value=None):
        events = _drain(plat, "exp_done")

    kinds = [e["event"] for e in events]
    assert "fitness_history_snapshot" in kinds
    assert "cost_tick" in kinds
    assert "completed" in kinds
    # Final metrics are surfaced BEFORE the terminal event, not dropped.
    assert kinds.index("fitness_history_snapshot") < kinds.index("completed")
    assert kinds.index("cost_tick") < kinds.index("completed")
    fh = next(e for e in events if e["event"] == "fitness_history_snapshot")
    assert fh["data"]["source"] == "platform"


def test_terminal_poll_skips_redis_probe() -> None:
    """On a terminal status the runner container is gone, so the Redis probe
    (a docker-exec) must NOT run even when /results carries no metrics."""
    client = _FakeClient(statuses=[{"status": "completed"}], results={})
    plat = CarePlatform(client)
    calls = {"history": 0, "programs": 0}

    def _hist(_id):
        calls["history"] += 1
        return []

    def _progs(_id):
        calls["programs"] += 1
        return (None, None)

    with patch.object(platform, "_probe_live_generation", return_value=None), patch.object(
        platform, "_probe_live_best_fitness", return_value=None
    ), patch.object(
        platform, "_probe_live_fitness_history", side_effect=_hist
    ), patch.object(
        platform, "_probe_live_programs_counts", side_effect=_progs
    ), patch("time.sleep", return_value=None):
        events = _drain(plat, "exp_done")

    assert calls["history"] == 0
    assert calls["programs"] == 0
    assert any(e["event"] == "completed" for e in events)


def test_probe_skipped_when_results_supply_metrics() -> None:
    """When /results already carries history + program counts, the loop
    must NOT re-probe Redis (avoids a docker-exec every 2s)."""
    client = _FakeClient(
        statuses=[{"status": "running"}, {"status": "completed"}],
        results={
            "metrics": {
                "fitness_history": [{"generation": 0, "best_fitness": 0.3}],
                "programs_valid": 5,
                "programs_invalid": 1,
            }
        },
    )
    plat = CarePlatform(client)
    calls = {"history": 0, "programs": 0}

    def _hist(_id):
        calls["history"] += 1
        return []

    def _progs(_id):
        calls["programs"] += 1
        return (None, None)

    with patch.object(platform, "_probe_live_generation", return_value=0), patch.object(
        platform, "_probe_live_best_fitness", return_value=None
    ), patch.object(
        platform, "_probe_live_fitness_history", side_effect=_hist
    ), patch.object(
        platform, "_probe_live_programs_counts", side_effect=_progs
    ), patch(
        "time.sleep", return_value=None
    ):
        events = _drain(plat, "exp_abc")

    kinds = [e["event"] for e in events]
    # Platform-sourced history snapshot present, redis probe never run.
    assert "fitness_history_snapshot" in kinds
    fh = next(e for e in events if e["event"] == "fitness_history_snapshot")
    assert fh["data"]["source"] == "platform"
    assert calls["history"] == 0
    assert calls["programs"] == 0


class TestResolveDisplayGeneration:
    def test_prefers_probe_when_platform_exceeds_max(self) -> None:
        assert _resolve_display_generation(106, 5, max_iterations=8) == 5

    def test_keeps_platform_when_probe_lags(self) -> None:
        assert _resolve_display_generation(5, 0, max_iterations=8) == 5

    def test_caps_platform_without_probe(self) -> None:
        assert _resolve_display_generation(106, None, max_iterations=8) == 8

    def test_probe_can_advance_platform(self) -> None:
        assert _resolve_display_generation(2, 4, max_iterations=8) == 4


class TestBogusFitnessHistory:
    def test_detects_scheduler_ticks(self) -> None:
        history = [
            {"generation": 0, "best_fitness": 0.05},
            {"generation": 26, "best_fitness": 0.16},
            {"generation": 82, "best_fitness": 0.42},
        ]
        assert _platform_fitness_history_looks_bogus(
            history, max_iterations=8,
        )

    def test_ignores_sane_history(self) -> None:
        history = [
            {"generation": 0, "best_fitness": 0.05},
            {"generation": 3, "best_fitness": 0.16},
        ]
        assert not _platform_fitness_history_looks_bogus(
            history, max_iterations=8,
        )


def test_poll_uses_probe_generation_when_platform_bogus() -> None:
    class _Client(_FakeClient):
        def get_experiment(self, experiment_id: str) -> dict:
            return {"config": {"max_iterations": 8}}

    client = _Client(
        statuses=[{"status": "running"}, {"status": "completed"}],
        results={"metrics": {"generation": 106, "best_fitness": 0.427}},
    )
    plat = CarePlatform(client)
    with patch.object(platform, "_probe_live_generation", return_value=5), patch.object(
        platform, "_probe_live_best_fitness", return_value=0.427
    ), patch.object(
        platform, "_probe_live_fitness_history", return_value=[]
    ), patch.object(
        platform, "_probe_live_programs_counts", return_value=(None, None)
    ), patch("time.sleep", return_value=None):
        events = _drain(plat, "exp_bogus")

    status = next(e for e in events if e["event"] == "status")
    assert status["data"]["generation"] == 5
    best = next(e for e in events if e["event"] == "best_updated")
    assert best["data"]["generation"] == 5


def test_poll_replaces_bogus_platform_fitness_history() -> None:
    class _Client(_FakeClient):
        def get_experiment(self, experiment_id: str) -> dict:
            return {"config": {"max_iterations": 8}}

    bogus = [
        {"generation": 0, "best_fitness": 0.05},
        {"generation": 26, "best_fitness": 0.16},
        {"generation": 82, "best_fitness": 0.42},
    ]
    rebuilt = [
        {"generation": 0, "best_fitness": 0.05, "current_fitness": 0.05},
        {"generation": 3, "best_fitness": 0.42, "current_fitness": 0.42},
    ]
    client = _Client(
        statuses=[{"status": "running"}, {"status": "completed"}],
        results={"metrics": {"fitness_history": bogus, "generation": 106}},
    )
    plat = CarePlatform(client)
    with patch.object(platform, "_probe_live_generation", return_value=3), patch.object(
        platform, "_probe_live_best_fitness", return_value=None
    ), patch.object(
        platform, "_fitness_history_best_per_generation", return_value=rebuilt
    ), patch.object(
        platform, "_probe_live_programs_counts", return_value=(None, None)
    ), patch("time.sleep", return_value=None):
        events = _drain(plat, "exp_hist")

    fh = next(e for e in events if e["event"] == "fitness_history_snapshot")
    assert fh["data"]["history"] == rebuilt
    assert fh["data"]["source"] == "redis_probe"
