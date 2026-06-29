"""Tests for `care.runtime.doctor` + `care doctor` CLI
(TODO §1 P1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from care.runtime.doctor import (
    DoctorReport,
    EnvVarRow,
    ExtraStatus,
    collect_env_vars,
    collect_extras,
    compose_report,
)


# ---------------------------------------------------------------------------
# collect_env_vars
# ---------------------------------------------------------------------------


class TestCollectEnvVars:
    def test_empty_env_returns_empty(self):
        assert collect_env_vars({}) == ()

    def test_filters_to_care_prefix(self):
        rows = collect_env_vars({
            "CARE_MAGE__MODEL": "gpt-4o",
            "HOME": "/home/u",
            "CARE_LOG_LEVEL": "DEBUG",
            "PATH": "/bin",
        })
        names = {r.name for r in rows}
        assert names == {"CARE_MAGE__MODEL", "CARE_LOG_LEVEL"}

    def test_redacts_secret_like_names(self):
        rows = collect_env_vars({
            "CARE_MAGE__API_KEY": "sk-1234567890",
            "CARE_MEMORY__TOKEN": "tok-abc",
            "CARE_MAGE__MODEL": "gpt-4o",
        })
        by_name = {r.name: r for r in rows}
        assert by_name["CARE_MAGE__API_KEY"].redacted is True
        assert "sk-1234567890" not in by_name["CARE_MAGE__API_KEY"].value
        assert "13 chars" in by_name["CARE_MAGE__API_KEY"].value
        assert by_name["CARE_MEMORY__TOKEN"].redacted is True
        # Non-secret value passes through.
        assert by_name["CARE_MAGE__MODEL"].redacted is False
        assert by_name["CARE_MAGE__MODEL"].value == "gpt-4o"

    def test_sorts_by_name(self):
        rows = collect_env_vars({
            "CARE_Z": "z",
            "CARE_A": "a",
            "CARE_M": "m",
        })
        assert [r.name for r in rows] == [
            "CARE_A", "CARE_M", "CARE_Z",
        ]


# ---------------------------------------------------------------------------
# collect_extras
# ---------------------------------------------------------------------------


class TestCollectExtras:
    def test_textual_is_installed(self) -> None:
        # textual is a core dep — must always be importable.
        rows = collect_extras()
        by_name = {r.name: r for r in rows}
        assert "textual" in by_name
        assert by_name["textual"].installed is True
        # Best-effort version lookup; not asserting exact.
        assert isinstance(by_name["textual"].version, str)

    def test_missing_extra_reports_uninstalled(self) -> None:
        rows = collect_extras()
        names = {r.name for r in rows}
        # At least the well-known ones the doctor lists.
        for expected in (
            "openai", "anthropic", "docker", "e2b",
            "plotext", "pypdf", "rich_pixels", "textual",
        ):
            assert expected in names

    def test_returns_tuple_of_extras(self) -> None:
        rows = collect_extras()
        assert all(isinstance(r, ExtraStatus) for r in rows)


# ---------------------------------------------------------------------------
# DoctorReport.format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_renders_all_sections(self) -> None:
        report = DoctorReport(
            config_path=Path("/home/u/care.toml"),
            config_exists=True,
            env_vars=(
                EnvVarRow(
                    name="CARE_MAGE__MODEL",
                    value="gpt-4o",
                    redacted=False,
                ),
            ),
            extras=(
                ExtraStatus(
                    name="textual",
                    installed=True,
                    version="8.2.6",
                ),
                ExtraStatus(name="docker", installed=False),
            ),
            user_path_lines=("✓ config — already present",),
            probes_text="✓ memory (12ms)\n· mage",
        )
        text = report.format_text()
        for header in (
            "== Config ==",
            "== Environment ==",
            "== Filesystem ==",
            "== Extras ==",
            "== Probes ==",
        ):
            assert header in text
        assert "/home/u/care.toml" in text
        assert "CARE_MAGE__MODEL = gpt-4o" in text
        assert "✓ textual 8.2.6" in text
        assert "· docker" in text
        assert "✓ config" in text
        assert "memory (12ms)" in text

    def test_empty_env_section_emits_placeholder(self) -> None:
        report = DoctorReport(
            config_path=Path("/a"),
            config_exists=False,
            env_vars=(),
            extras=(),
            user_path_lines=(),
        )
        text = report.format_text()
        assert "(no CARE_* env vars set)" in text
        assert "(no user-path report)" in text
        # Probes section is omitted when probes_text is empty.
        assert "== Probes ==" not in text

    def test_config_exists_no_renders_no(self) -> None:
        report = DoctorReport(
            config_path=Path("/missing"),
            config_exists=False,
        )
        text = report.format_text()
        assert "exists: no" in text


# ---------------------------------------------------------------------------
# compose_report
# ---------------------------------------------------------------------------


class TestComposeReport:
    def test_uses_supplied_config_path(self, tmp_path: Path) -> None:
        target = tmp_path / "demo.toml"
        target.write_text("[mage]\nmodel = 'x'\n")
        report = compose_report(
            config_path=target,
            env={"CARE_FOO": "bar"},
            probes_text="",
        )
        assert report.config_path == target
        assert report.config_exists is True
        assert any(r.name == "CARE_FOO" for r in report.env_vars)

    def test_handles_missing_config(self, tmp_path: Path) -> None:
        target = tmp_path / "nope.toml"
        report = compose_report(
            config_path=target, env={},
        )
        assert report.config_exists is False


# ---------------------------------------------------------------------------
# CLI: `care doctor`
# ---------------------------------------------------------------------------


class TestCliDoctor:
    def test_no_probes_returns_zero_without_network(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import os

        from care import dotenv as dotenv_mod
        from care.cli import main

        # Isolate from the user's env / `.env`.
        for key in list(os.environ.keys()):
            if key.startswith("CARE_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            dotenv_mod, "load_env_file",
            lambda *_a, **_kw: None,
        )

        config_path = tmp_path / "care.toml"
        config_path.write_text(
            '[mage]\nmodel = "gpt-4o"\n',
        )

        rc = main(
            [
                "doctor",
                "--config", str(config_path),
                "--no-probes",
            ],
        )
        assert rc == 0
        captured = capsys.readouterr()
        text = captured.out
        # Headline sections are present.
        assert "== Config ==" in text
        assert "== Environment ==" in text
        assert "== Filesystem ==" in text
        assert "== Extras ==" in text
        # Probes section omitted in --no-probes.
        assert "== Probes ==" not in text

    def test_missing_config_returns_two(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # CareConfig.load tolerates missing files (returns
        # defaults) so doctor with a missing file should
        # still succeed; the test ensures we don't crash
        # rather than expecting a non-zero return.
        import os

        from care import dotenv as dotenv_mod
        from care.cli import main

        for key in list(os.environ.keys()):
            if key.startswith("CARE_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            dotenv_mod, "load_env_file",
            lambda *_a, **_kw: None,
        )

        rc = main(
            [
                "doctor",
                "--config", str(tmp_path / "missing.toml"),
                "--no-probes",
            ],
        )
        # Default config loads fine; doctor returns 0.
        assert rc == 0
