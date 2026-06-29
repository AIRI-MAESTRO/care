"""Tests for `care.runtime.local_run_history` (TODO §6 P1)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from care.runtime.local_run_history import (
    LocalRunEntry,
    build_run_entry,
    load_local_runs,
    record_local_run,
    runs_dir,
)


# ---------------------------------------------------------------------------
# Pure data class
# ---------------------------------------------------------------------------


class TestLocalRunEntry:
    def test_tokens_total_returns_none_when_both_missing(self) -> None:
        entry = LocalRunEntry(run_id="r")
        assert entry.tokens_total is None

    def test_tokens_total_sums_when_present(self) -> None:
        entry = LocalRunEntry(
            run_id="r", tokens_in=100, tokens_out=50,
        )
        assert entry.tokens_total == 150

    def test_tokens_total_handles_partial(self) -> None:
        entry = LocalRunEntry(run_id="r", tokens_in=200)
        assert entry.tokens_total == 200


# ---------------------------------------------------------------------------
# Record + load round-trip
# ---------------------------------------------------------------------------


class TestRecordAndLoad:
    def test_record_creates_day_file(self, tmp_path: Path) -> None:
        entry = LocalRunEntry(
            run_id="r-1",
            chain_id="chain-a",
            started_at=time.time(),
            status="success",
            duration_seconds=1.2,
        )
        path = record_local_run(entry, cache_dir=tmp_path)
        assert path.exists()
        assert path.parent == runs_dir(cache_dir=tmp_path)
        # File contains one JSON line.
        text = path.read_text().strip().splitlines()
        assert len(text) == 1
        parsed = json.loads(text[0])
        assert parsed["run_id"] == "r-1"
        assert parsed["chain_id"] == "chain-a"

    def test_round_trip_via_load_local_runs(self, tmp_path: Path) -> None:
        entry = LocalRunEntry(
            run_id="r-trip",
            chain_id="c",
            started_at=time.time(),
            duration_seconds=4.5,
            status="success",
            tokens_in=10, tokens_out=20,
            cost_usd=0.0021,
        )
        record_local_run(entry, cache_dir=tmp_path)
        rows = load_local_runs(cache_dir=tmp_path)
        assert len(rows) == 1
        loaded = rows[0]
        assert loaded.run_id == "r-trip"
        assert loaded.chain_id == "c"
        assert loaded.duration_seconds == 4.5
        assert loaded.tokens_in == 10
        assert loaded.tokens_out == 20
        assert loaded.cost_usd == 0.0021

    def test_load_returns_empty_when_dir_missing(
        self, tmp_path: Path,
    ) -> None:
        # No runs/ subdir yet → empty list.
        rows = load_local_runs(cache_dir=tmp_path)
        assert rows == []

    def test_load_skips_non_jsonl_files(
        self, tmp_path: Path,
    ) -> None:
        d = runs_dir(cache_dir=tmp_path)
        d.mkdir(parents=True)
        (d / "readme.txt").write_text("not a run file")
        (d / "broken-name.jsonl").write_text(
            '{"run_id": "r"}\n',
        )
        # Today's file with the date-shaped stem IS read.
        today = time.strftime("%Y-%m-%d", time.localtime())
        (d / f"{today}.jsonl").write_text(
            '{"run_id": "real", "started_at": 1000}\n',
        )
        rows = load_local_runs(cache_dir=tmp_path)
        assert [r.run_id for r in rows] == ["real"]

    def test_load_skips_malformed_lines(
        self, tmp_path: Path,
    ) -> None:
        d = runs_dir(cache_dir=tmp_path)
        d.mkdir(parents=True)
        today = time.strftime("%Y-%m-%d", time.localtime())
        (d / f"{today}.jsonl").write_text(
            '{"run_id": "good", "started_at": 100}\n'
            '{not-json}\n'
            '{"missing_id": "bad"}\n'
            '{"run_id": "good-2", "started_at": 200}\n',
        )
        rows = load_local_runs(cache_dir=tmp_path)
        assert [r.run_id for r in rows] == ["good-2", "good"]

    def test_load_normalises_legacy_ad_hoc_mode(
        self, tmp_path: Path,
    ) -> None:
        """Modes redesign P0 — a persisted row carrying the legacy
        ``ad_hoc`` mode label loads as ``interactive`` so historical
        rows group with current ones in the cost/usage rollups."""
        d = runs_dir(cache_dir=tmp_path)
        d.mkdir(parents=True)
        today = time.strftime("%Y-%m-%d", time.localtime())
        (d / f"{today}.jsonl").write_text(
            '{"run_id": "legacy", "started_at": 100, "mode": "ad_hoc"}\n'
            '{"run_id": "current", "started_at": 200, "mode": "interactive"}\n',
        )
        rows = load_local_runs(cache_dir=tmp_path)
        modes = {r.run_id: r.mode for r in rows}
        assert modes["legacy"] == "interactive"
        assert modes["current"] == "interactive"

    def test_load_sorts_newest_first(
        self, tmp_path: Path,
    ) -> None:
        for i, ts in enumerate([100.0, 300.0, 200.0]):
            record_local_run(
                LocalRunEntry(
                    run_id=f"r-{i}",
                    started_at=ts,
                    chain_id="c",
                ),
                cache_dir=tmp_path,
            )
        rows = load_local_runs(cache_dir=tmp_path)
        ids = [r.run_id for r in rows]
        # Sorted by started_at desc → r-1 (300), r-2 (200),
        # r-0 (100).
        assert ids == ["r-1", "r-2", "r-0"]

    def test_load_honors_limit(self, tmp_path: Path) -> None:
        for i in range(10):
            record_local_run(
                LocalRunEntry(
                    run_id=f"r-{i}",
                    started_at=float(i),
                ),
                cache_dir=tmp_path,
            )
        rows = load_local_runs(cache_dir=tmp_path, limit=3)
        assert len(rows) == 3
        # Newest three: started_at 9, 8, 7.
        assert [r.started_at for r in rows] == [9.0, 8.0, 7.0]

    def test_record_creates_runs_subdir_on_demand(
        self, tmp_path: Path,
    ) -> None:
        # Pre-existing tmp_path but no runs/ subdir.
        assert not (tmp_path / "runs").exists()
        record_local_run(
            LocalRunEntry(run_id="rx", started_at=1.0),
            cache_dir=tmp_path,
        )
        assert (tmp_path / "runs").is_dir()


# ---------------------------------------------------------------------------
# build_run_entry (§6 P1 shared projection)
# ---------------------------------------------------------------------------


class TestBuildRunEntry:
    def test_full_shape(self):
        from types import SimpleNamespace

        chain = SimpleNamespace(
            entity_id="c1", name="Forecaster",
        )
        result = SimpleNamespace(
            usage={"prompt": 100, "completion": 50},
        )
        entry = build_run_entry(
            run_id="r-1",
            chain=chain,
            task="hello world",
            result=result,
            started_at=1234.5,
            duration=2.0,
            status="success",
            mode="ad_hoc",
            provider="openai",
        )
        assert entry.run_id == "r-1"
        assert entry.chain_id == "c1"
        assert entry.chain_name == "Forecaster"
        assert entry.tokens_in == 100
        assert entry.tokens_out == 50
        assert entry.duration_seconds == 2.0
        assert entry.status == "success"
        assert entry.mode == "ad_hoc"
        assert entry.provider == "openai"
        assert entry.extra == {"task": "hello world"}
        assert entry.error == ""

    def test_missing_chain_fields_collapse_to_empty(self):
        from types import SimpleNamespace

        entry = build_run_entry(
            run_id="r",
            chain=SimpleNamespace(),
            started_at=0.0, duration=0.0, status="success",
        )
        assert entry.chain_id == ""
        assert entry.chain_name == ""

    def test_alternative_usage_keys(self):
        from types import SimpleNamespace

        chain = SimpleNamespace(entity_id="x")
        result = SimpleNamespace(
            usage={
                "prompt_tokens": 75,
                "completion_tokens": 30,
            },
        )
        entry = build_run_entry(
            run_id="r", chain=chain, result=result,
            started_at=0.0, duration=0.0, status="success",
        )
        assert entry.tokens_in == 75
        assert entry.tokens_out == 30

    def test_no_result_keeps_tokens_none(self):
        from types import SimpleNamespace

        entry = build_run_entry(
            run_id="r", chain=SimpleNamespace(entity_id="x"),
            result=None,
            started_at=0.0, duration=0.0,
            status="failure", error="boom",
        )
        assert entry.tokens_in is None
        assert entry.tokens_out is None
        assert entry.error == "boom"

    def test_task_truncates_to_200_chars(self):
        from types import SimpleNamespace

        long_task = "x" * 500
        entry = build_run_entry(
            run_id="r", chain=SimpleNamespace(entity_id="x"),
            task=long_task,
            started_at=0.0, duration=0.0, status="success",
        )
        assert entry.extra["task"] == "x" * 200

    def test_empty_task_omits_extra_key(self):
        from types import SimpleNamespace

        entry = build_run_entry(
            run_id="r", chain=SimpleNamespace(entity_id="x"),
            task="",
            started_at=0.0, duration=0.0, status="success",
        )
        assert "task" not in entry.extra

    def test_extra_arg_merges_with_task(self):
        from types import SimpleNamespace

        entry = build_run_entry(
            run_id="r", chain=SimpleNamespace(entity_id="x"),
            task="some task",
            started_at=0.0, duration=0.0, status="success",
            extra={"dataset": "ds-1", "custom": 42},
        )
        assert entry.extra == {
            "dataset": "ds-1",
            "custom": 42,
            "task": "some task",
        }

    def test_extra_none_preserves_old_behaviour(self):
        from types import SimpleNamespace

        entry = build_run_entry(
            run_id="r", chain=SimpleNamespace(entity_id="x"),
            task="t",
            started_at=0.0, duration=0.0, status="success",
            extra=None,
        )
        assert entry.extra == {"task": "t"}

    def test_write_replay_writes_sidecar_for_dict_result(
        self, tmp_path: Path,
    ) -> None:
        from types import SimpleNamespace

        from care.runtime.local_run_history import replays_dir

        chain = SimpleNamespace(entity_id="c")
        result = {
            "step_results": [{"id": "s1", "success": True}],
            "total_execution_time": 1.5,
            "final_answer": "done",
        }
        entry = build_run_entry(
            run_id="r-sidecar",
            chain=chain,
            result=result,
            started_at=0.0,
            duration=0.5,
            status="success",
            write_replay=True,
            cache_dir=tmp_path,
        )
        assert entry.replay_path
        path = replays_dir(cache_dir=tmp_path) / "r-sidecar.json"
        assert path.exists()
        # Reads back as JSON with the same shape.
        import json as _json
        body = _json.loads(path.read_text())
        assert body["final_answer"] == "done"
        assert entry.replay_path == str(path)

    def test_write_replay_handles_dataclass_with_to_dict(
        self, tmp_path: Path,
    ) -> None:
        from types import SimpleNamespace

        class _Stub:
            def to_dict(self):
                return {
                    "step_results": [],
                    "final_answer": "via to_dict",
                }

        entry = build_run_entry(
            run_id="r-to_dict",
            chain=SimpleNamespace(entity_id="c"),
            result=_Stub(),
            started_at=0.0, duration=0.0, status="success",
            write_replay=True,
            cache_dir=tmp_path,
        )
        assert entry.replay_path
        import json as _json
        body = _json.loads(
            Path(entry.replay_path).read_text(),
        )
        assert body["final_answer"] == "via to_dict"

    def test_write_replay_false_keeps_path_empty(
        self, tmp_path: Path,
    ) -> None:
        from types import SimpleNamespace

        entry = build_run_entry(
            run_id="r-no-sidecar",
            chain=SimpleNamespace(entity_id="c"),
            result={"step_results": []},
            started_at=0.0, duration=0.0, status="success",
            # write_replay defaults to False
            cache_dir=tmp_path,
        )
        assert entry.replay_path == ""

    def test_write_replay_with_none_result_keeps_path_empty(
        self, tmp_path: Path,
    ) -> None:
        from types import SimpleNamespace

        entry = build_run_entry(
            run_id="r-no-result",
            chain=SimpleNamespace(entity_id="c"),
            result=None,
            started_at=0.0, duration=0.0, status="failure",
            error="crashed",
            write_replay=True,
            cache_dir=tmp_path,
        )
        assert entry.replay_path == ""

    def test_replay_path_round_trips_through_jsonl(
        self, tmp_path: Path,
    ):
        record_local_run(
            LocalRunEntry(
                run_id="r-rt",
                started_at=1.0,
                replay_path="/tmp/some/replay.json",
            ),
            cache_dir=tmp_path,
        )
        rows = load_local_runs(cache_dir=tmp_path)
        assert rows[0].replay_path == "/tmp/some/replay.json"
