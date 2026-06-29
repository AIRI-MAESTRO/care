"""Tests for chain template sync onto the live runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from care.runtime import evolution_chain_templates as ect
from care.runtime.evolution_chain_templates import (
    ChainTemplateSyncResult,
    _build_placeholders,
    maybe_sync_chain_templates,
    runner_helper_is_stale,
    schedule_chain_template_sync,
    sync_chain_templates_to_runner,
    sync_chain_templates_until_ready,
    sync_kwargs_from_experiment,
    verify_chain_template_source,
)


@pytest.fixture(autouse=True)
def _reset_chain_sync_state() -> None:
    ect._scheduled_syncs.clear()
    ect._last_sync_attempt.clear()


class TestVerifyChainTemplateSource:
    def test_reports_missing_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        missing = tmp_path / "nope"
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.default_template_dir",
            lambda: missing,
        )
        ok, msg = verify_chain_template_source()
        assert ok is False
        assert "missing" in msg


class TestSyncKwargsFromExperiment:
    def test_reads_validation_block(self) -> None:
        exp = {
            "config": {
                "parameters": {
                    "target_column": "answer",
                    "validation_criteria": {
                        "validation_type": "Continuous (0..1)",
                        "continuous_metric": "ROUGE-L",
                        "regexp_pattern": r"X:\s*(.+)",
                    },
                },
            },
        }
        kw = sync_kwargs_from_experiment(exp)
        assert kw["target_column"] == "answer"
        assert kw["continuous_metric"] == "ROUGE-L"
        assert kw["regexp_pattern"] == r"X:\s*(.+)"


class TestBuildPlaceholders:
    def test_task_name_is_full_uuid_not_truncated(self) -> None:
        exp = "exp_49a57e97-d6f9-4c20-9e84-8c824fed91fc"
        ph = _build_placeholders(
            experiment_id=exp,
            validation_type="Continuous (0..1)",
            continuous_metric="ROUGE-L",
            binary_method=None,
            target_column="expected",
            regexp_pattern="",
        )
        assert ph["task_name"] == "49a57e97-d6f9-4c20-9e84-8c824fed91fc"


class TestRunnerHelperIsStale:
    def test_missing_helper_is_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates._runner_path_exists",
            lambda *_a, **_k: False,
        )
        assert runner_helper_is_stale("exp_abcd") is True

    def test_chain_runner_helper_not_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates._runner_path_exists",
            lambda *_a, **_k: True,
        )
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.subprocess.run",
            lambda *_a, **_k: type(
                "P",
                (),
                {"returncode": 0, "stdout": "from chain_runner import run"},
            )(),
        )
        assert runner_helper_is_stale("exp_abcd") is False

    def test_unreadable_helper_counts_as_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates._runner_path_exists",
            lambda *_a, **_k: True,
        )
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.subprocess.run",
            lambda *_a, **_k: type(
                "P",
                (),
                {"returncode": 1, "stdout": ""},
            )(),
        )
        assert runner_helper_is_stale("exp_abcd") is True


class TestSyncChainTemplatesToRunner:
    def test_skips_non_experiment_ids(self) -> None:
        result = sync_chain_templates_to_runner("evo_legacy")
        assert result.ok is False
        assert result.copied == 0

    def test_copies_when_container_running(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tpl = tmp_path / "chain"
        tpl.mkdir()
        (tpl / "helper.py").write_text("from chain_runner import run\n", encoding="utf-8")
        (tpl / "validate.py").write_text(
            'VALIDATION_TYPE = "${validation_type}"\n',
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.default_template_dir",
            lambda: tpl,
        )
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates._docker_container_running",
            lambda *_a, **_k: True,
        )
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates._ensure_runner_dest_dir",
            lambda *_a, **_k: True,
        )
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.runner_helper_is_stale",
            lambda *_a, **_k: False,
        )

        calls: list[list[str]] = []

        def _fake_run(cmd, **_kw):
            calls.append(list(cmd))
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.subprocess.run",
            _fake_run,
        )

        result = sync_chain_templates_to_runner("exp_test")
        assert result.ok is True
        assert result.copied >= 2
        assert any("docker" in c and "cp" in c for c in calls)


class TestSyncUntilReady:
    def test_returns_when_already_current(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.runner_helper_is_stale",
            lambda *_a, **_k: False,
        )
        sync_mock = patch(
            "care.runtime.evolution_chain_templates.sync_chain_templates_to_runner",
        )
        with sync_mock as m:
            result = sync_chain_templates_until_ready("exp_ok", timeout=1)
        m.assert_not_called()
        assert result.ok is True

    def test_fixes_permissions_after_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates._fix_runner_file_permissions",
            lambda _c, _d, names, **_kw: calls.append(list(names)),
        )
        from care.runtime.evolution_chain_templates import _fix_runner_file_permissions

        _fix_runner_file_permissions("c", "/d", ["helper.py", "validate.py"])
        assert calls == [["helper.py", "validate.py"]]


class TestMaybeSyncChainTemplates:
    def test_schedules_when_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.runner_helper_is_stale",
            lambda *_a, **_k: True,
        )
        scheduled: list[str] = []
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.schedule_chain_template_sync",
            lambda exp_id, **_kw: scheduled.append(exp_id),
        )
        maybe_sync_chain_templates("exp_stale")
        assert scheduled == ["exp_stale"]

    def test_throttles_repeat_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.runner_helper_is_stale",
            lambda *_a, **_k: True,
        )
        count = 0

        def _schedule(exp_id, **_kw):
            nonlocal count
            count += 1

        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.schedule_chain_template_sync",
            _schedule,
        )
        maybe_sync_chain_templates("exp_x", min_interval=60)
        maybe_sync_chain_templates("exp_x", min_interval=60)
        assert count == 1


class TestScheduleChainTemplateSync:
    def test_dedupes_parallel_schedules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import threading

        started: list[str] = []
        hold = threading.Event()

        def _until_ready(exp_id, **_kw):
            started.append(exp_id)
            hold.wait(timeout=1)
            return ChainTemplateSyncResult(
                copied=1,
                experiment_id=exp_id,
                container="c",
                dest_dir="/d",
                ok=True,
                message="ok",
            )

        monkeypatch.setattr(
            "care.runtime.evolution_chain_templates.sync_chain_templates_until_ready",
            _until_ready,
        )
        schedule_chain_template_sync("exp_dup")
        schedule_chain_template_sync("exp_dup")
        import time

        time.sleep(0.05)
        assert started == ["exp_dup"]
        hold.set()
