"""Tests for `care run --execute` recording hook (TODO §6 P1).

Drives `_record_cli_run` directly with stub payloads to
verify the CLI execute path writes `LocalRunEntry` rows
to `~/.cache/care/runs/`.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from care.cli import _record_cli_run
from care.runtime.local_run_history import load_local_runs


@pytest.fixture(autouse=True)
def _redirect_runs_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    from care.runtime import local_run_history as lrh
    from care.runtime import user_paths as up

    monkeypatch.setattr(up, "CARE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(lrh, "CARE_CACHE_DIR", tmp_path)
    yield


class TestRecordCliRun:
    def test_success_writes_row_with_cli_mode(
        self, tmp_path: Path,
    ):
        result = SimpleNamespace(
            usage={"prompt": 75, "completion": 25},
        )
        _record_cli_run(
            chain_dict={
                "steps": [],
                "metadata": {
                    "care": {"display_name": "WeatherChain"},
                },
            },
            chain_id="chain-A",
            task="get weather",
            result=result,
            started_at=time.time() - 1.0,
            duration=0.95,
            status="success",
        )
        rows = load_local_runs(cache_dir=tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row.chain_id == "chain-A"
        assert row.chain_name == "WeatherChain"
        assert row.status == "success"
        assert row.tokens_in == 75
        assert row.tokens_out == 25
        assert row.mode == "cli"
        assert row.run_id.startswith("cli-")

    def test_failure_writes_row_with_error(
        self, tmp_path: Path,
    ):
        _record_cli_run(
            chain_dict={"steps": []},
            chain_id="chain-bad",
            task="cause failure",
            result=None,
            started_at=time.time(),
            duration=0.05,
            status="failure",
            error="503 upstream",
        )
        rows = load_local_runs(cache_dir=tmp_path)
        assert rows[0].status == "failure"
        assert rows[0].error == "503 upstream"

    def test_missing_metadata_collapses_chain_name_to_empty(
        self, tmp_path: Path,
    ):
        _record_cli_run(
            chain_dict={"steps": []},  # no metadata
            chain_id="bare-id",
            task="",
            result=None,
            started_at=time.time(),
            duration=0.1,
            status="success",
        )
        rows = load_local_runs(cache_dir=tmp_path)
        assert rows[0].chain_id == "bare-id"
        assert rows[0].chain_name == ""

    def test_metadata_display_name_outside_care_namespace(
        self, tmp_path: Path,
    ):
        """Older chain payloads put `display_name` under
        `metadata.display_name` (not `metadata.care`). The
        helper reads both shapes."""
        _record_cli_run(
            chain_dict={
                "steps": [],
                "metadata": {"display_name": "OldShape"},
            },
            chain_id="old-id",
            task="",
            result=None,
            started_at=time.time(),
            duration=0.1,
            status="success",
        )
        rows = load_local_runs(cache_dir=tmp_path)
        assert rows[0].chain_name == "OldShape"

    def test_recorder_failure_does_not_raise(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from care.runtime import local_run_history as lrh

        def _explode(*_a: Any, **_kw: Any) -> None:
            raise OSError("permission denied")

        monkeypatch.setattr(
            lrh, "record_local_run", _explode,
        )
        # Should not raise.
        _record_cli_run(
            chain_dict={"steps": []},
            chain_id="x",
            task="",
            result=None,
            started_at=time.time(),
            duration=0.0,
            status="failure",
            error="x",
        )
        # And nothing landed on disk.
        rows = load_local_runs(cache_dir=tmp_path)
        assert rows == []
