"""Tests for automatic Platform bootstrap on MAESTRO startup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from care.runtime.evolution_chain_templates import _render_template
from care.runtime.platform_bootstrap import bootstrap_local_platform_stack
from care.runtime.runner_tools_sync import sync_runner_gigaevo_tools


class TestBootstrapLocalPlatformStack:
    def test_skips_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARE_PLATFORM__AUTO_BOOTSTRAP", "0")

        class _Cfg:
            platform = type("P", (), {"base_url": "http://localhost:8000"})()

        report = bootstrap_local_platform_stack(_Cfg())
        assert report.skipped is True

    def test_runs_llm_and_tools_for_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARE_PLATFORM__AUTO_BOOTSTRAP", raising=False)

        class _Cfg:
            platform = type("P", (), {"base_url": "http://localhost:8000"})()

        with patch(
            "care.runtime.platform_bootstrap.try_sync_platform_llm_registry",
            return_value=type(
                "R",
                (),
                {"message": "synced llm"},
            )(),
        ) as llm_mock, patch(
            "care.runtime.platform_bootstrap.sync_runner_gigaevo_tools",
            return_value=type(
                "T",
                (),
                {"message": "copied tools"},
            )(),
        ) as tools_mock, patch(
            "care.runtime.platform_bootstrap.verify_chain_template_source",
            return_value=(True, "chain templates ready"),
        ):
            report = bootstrap_local_platform_stack(_Cfg())

        llm_mock.assert_called_once()
        tools_mock.assert_called_once()
        assert "synced llm" in report.messages
        assert "copied tools" in report.messages
        assert "chain templates ready" in report.messages


class TestRunnerToolsSync:
    def test_skips_when_container_not_running(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        core = tmp_path / "gigaevo-core"
        (core / "tools").mkdir(parents=True)
        (core / "tools" / "comparison.py").write_text("# stub", encoding="utf-8")

        with patch(
            "care.runtime.runner_tools_sync._docker_container_running",
            return_value=False,
        ):
            result = sync_runner_gigaevo_tools(core_dir=core)

        assert result is not None
        assert result.copied == ()


class TestChainTemplateRender:
    def test_substitutes_validation_placeholders(self) -> None:
        raw = 'VALIDATION_TYPE = "${validation_type}"\nMETRIC = "${metric}"\nPATH = "${task_name}"\n'
        out = _render_template(
            raw,
            {
                "validation_type": "Continuous (0..1)",
                "metric": "ROUGE-L",
                "task_name": "49a57e97-d6f9-4c20-9e84-8c824fed91fc",
            },
        )
        assert 'VALIDATION_TYPE = "Continuous (0..1)"' in out
        assert 'METRIC = "ROUGE-L"' in out
        assert "49a57e97-d6f9-4c20-9e84-8c824fed91fc" in out
        assert "49a57e97-d6f9-4c20-9e84-8c824fed" not in out or "91fc" in out
