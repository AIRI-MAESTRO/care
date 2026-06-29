"""Tests for ``care.first_run`` (TODO §2 P0).

Six coverage layers:

1. **Probe shape** — `ProbeResult` frozen; `FirstRunReport`
   aggregates correctly.
2. **probe_memory** — happy path, skipped, failure, latency
   stamping.
3. **probe_mage** — skipped on missing api_key, happy path
   via stub factory, failure wraps.
4. **probe_platform** — same shape as memory.
5. **run_all_probes** — fans out concurrently + aggregates;
   `all_ok` / `any_failed` predicates.
6. **write_initial_config** — atomic write, parent dir
   creation, overwrite guard, round-trip via tomllib.
"""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

import pytest

from care.config import CareConfig, MageConfig, MemoryConfig, PlatformConfig
from care.first_run import (
    FirstRunConfigError,
    FirstRunReport,
    ProbeResult,
    probe_mage,
    probe_memory,
    probe_platform,
    run_all_probes,
    write_initial_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(
    *,
    memory_url: str = "http://memory:8000",
    memory_key: str = "k",
    platform_url: str = "http://platform:8001",
    platform_key: str = "k",
    mage_key: str | None = "sk-test",
    mage_base_url: str = "https://api.openai.com/v1",
) -> CareConfig:
    return CareConfig(
        mage=MageConfig(api_key=mage_key, base_url=mage_base_url),
        memory=MemoryConfig(base_url=memory_url, api_key=memory_key),
        platform=PlatformConfig(base_url=platform_url, api_key=platform_key),
    )


class _StubMemory:
    def __init__(self, *, raise_exc: Exception | None = None, detail: dict | None = None):
        self._raise = raise_exc
        self._detail = detail or {"status": "ok"}

    def health_check(self):
        if self._raise is not None:
            raise self._raise
        return self._detail


class _StubPlatform:
    def __init__(self, *, raise_exc: Exception | None = None, detail: dict | None = None):
        self._raise = raise_exc
        self._detail = detail or {"status": "ok", "version": "0.5"}

    def health_check(self):
        if self._raise is not None:
            raise self._raise
        return self._detail


# ---------------------------------------------------------------------------
# ProbeResult / FirstRunReport
# ---------------------------------------------------------------------------


class TestShape:
    def test_probe_result_frozen(self):
        r = ProbeResult(service="memory", status="ok")
        with pytest.raises(Exception):
            r.status = "failed"  # type: ignore[misc]

    def test_report_all_ok_true_when_all_ok(self):
        ok = lambda s: ProbeResult(service=s, status="ok")  # noqa: E731
        report = FirstRunReport(
            memory=ok("memory"), mage=ok("mage"), platform=ok("platform"),
        )
        assert report.all_ok is True
        assert report.any_failed is False

    def test_report_all_ok_false_when_any_skipped(self):
        report = FirstRunReport(
            memory=ProbeResult(service="memory", status="ok"),
            mage=ProbeResult(service="mage", status="skipped", error="no key"),
            platform=ProbeResult(service="platform", status="ok"),
        )
        assert report.all_ok is False
        assert report.any_failed is False

    def test_report_all_ok_true_when_platform_skipped(self):
        # Platform is optional — its skipped/failed status
        # does NOT pull `all_ok` to False as long as Memory +
        # MAGE are ok.
        report = FirstRunReport(
            memory=ProbeResult(service="memory", status="ok"),
            mage=ProbeResult(service="mage", status="ok"),
            platform=ProbeResult(
                service="platform", status="skipped", error="no key",
            ),
        )
        assert report.all_ok is True
        assert report.platform_ok is False

    def test_report_all_ok_true_when_platform_failed(self):
        # Same — Platform failure (e.g. evolution service down)
        # doesn't gate the Settings save flow.
        report = FirstRunReport(
            memory=ProbeResult(service="memory", status="ok"),
            mage=ProbeResult(service="mage", status="ok"),
            platform=ProbeResult(
                service="platform", status="failed", error="503",
            ),
        )
        assert report.all_ok is True
        assert report.platform_ok is False
        assert report.any_failed is True

    def test_report_any_failed(self):
        report = FirstRunReport(
            memory=ProbeResult(service="memory", status="failed", error="503"),
            mage=ProbeResult(service="mage", status="ok"),
            platform=ProbeResult(service="platform", status="ok"),
        )
        assert report.all_ok is False
        assert report.any_failed is True

    def test_format_text_includes_badges(self):
        report = FirstRunReport(
            memory=ProbeResult(service="memory", status="ok", latency_ms=12.3),
            mage=ProbeResult(service="mage", status="skipped", error="no key"),
            platform=ProbeResult(service="platform", status="failed", error="503"),
        )
        text = report.format_text()
        assert "✓ memory" in text
        assert "12ms" in text
        assert "· mage" in text
        assert "no key" in text
        assert "✗ platform" in text
        assert "503" in text


# ---------------------------------------------------------------------------
# probe_memory
# ---------------------------------------------------------------------------


class TestProbeMemory:
    def test_happy_path(self):
        memory = _StubMemory(detail={"status": "ok"})
        result = asyncio.run(
            probe_memory(_cfg(), memory_factory=lambda c: memory)
        )
        assert result.service == "memory"
        assert result.status == "ok"
        assert result.latency_ms is not None
        assert result.detail == {"status": "ok"}
        assert result.error == ""

    def test_skipped_when_base_url_empty(self):
        cfg = _cfg(memory_url="")
        result = asyncio.run(probe_memory(cfg))
        assert result.status == "skipped"
        assert "base_url" in result.error
        # No probe attempt → no latency.
        assert result.latency_ms is None

    def test_failure_wraps_exception(self):
        memory = _StubMemory(raise_exc=RuntimeError("DB down"))
        result = asyncio.run(
            probe_memory(_cfg(), memory_factory=lambda c: memory)
        )
        assert result.status == "failed"
        assert "RuntimeError" in result.error
        assert "DB down" in result.error
        assert result.latency_ms is not None

    def test_non_dict_detail_coerced(self):
        # Some implementations return a string from health_check
        # — the probe wraps it into {"raw": ...}.
        memory = _StubMemory(detail=None)
        # detail=None defaults to {"status": "ok"} per stub
        # construction. Override post-hoc to test the coercion.
        memory._detail = "ok"
        result = asyncio.run(
            probe_memory(_cfg(), memory_factory=lambda c: memory)
        )
        assert result.detail == {"raw": "ok"}


# ---------------------------------------------------------------------------
# probe_mage
# ---------------------------------------------------------------------------


class TestProbeMage:
    def test_skipped_when_api_key_missing(self):
        cfg = _cfg(mage_key=None)
        result = asyncio.run(probe_mage(cfg))
        assert result.status == "skipped"
        assert "api_key" in result.error

    def test_skipped_when_api_key_empty(self):
        cfg = _cfg(mage_key="")
        result = asyncio.run(probe_mage(cfg))
        assert result.status == "skipped"

    def test_happy_path_via_stub_factory(self):
        class _Client:
            base_url = "https://api.openai.com/v1"

        result = asyncio.run(
            probe_mage(_cfg(), client_factory=lambda mc: _Client())
        )
        assert result.status == "ok"
        assert "openai.com" in result.detail["base_url"]

    def test_failure_wraps(self):
        def _bad(mage_config):
            raise RuntimeError("LLM down")

        result = asyncio.run(probe_mage(_cfg(), client_factory=_bad))
        assert result.status == "failed"
        assert "LLM down" in result.error

    def test_shallow_probe_skips_round_trip(self):
        """deep=False must NOT call models.list (keeps boot fast)."""
        calls = {"n": 0}

        class _Models:
            def list(self_inner):
                calls["n"] += 1
                raise RuntimeError("should not be called")

        class _Client:
            base_url = "https://api.openai.com/v1"
            models = _Models()

        result = asyncio.run(
            probe_mage(_cfg(), client_factory=lambda mc: _Client())
        )
        assert result.status == "ok"
        assert calls["n"] == 0

    def test_deep_probe_round_trip_ok(self):
        class _Models:
            def list(self_inner):
                return ["m1"]

        class _Client:
            base_url = "https://api.openai.com/v1"
            models = _Models()

        result = asyncio.run(
            probe_mage(_cfg(), client_factory=lambda mc: _Client(), deep=True)
        )
        assert result.status == "ok"
        assert "ok" in result.detail.get("round_trip", "")

    def test_deep_probe_auth_error_fails(self):
        class _AuthErr(Exception):
            status_code = 403

        class _Models:
            def list(self_inner):
                raise _AuthErr("Token expired")

        class _Client:
            base_url = "https://api.openai.com/v1"
            models = _Models()

        result = asyncio.run(
            probe_mage(_cfg(), client_factory=lambda mc: _Client(), deep=True)
        )
        assert result.status == "failed"
        assert "expired or invalid" in result.error
        assert "403" in result.error

    def test_deep_probe_404_models_unsupported_is_ok(self):
        class _NotFound(Exception):
            status_code = 404

        class _Models:
            def list(self_inner):
                raise _NotFound("no /models")

        class _Client:
            base_url = "https://api.openai.com/v1"
            models = _Models()

        result = asyncio.run(
            probe_mage(_cfg(), client_factory=lambda mc: _Client(), deep=True)
        )
        assert result.status == "ok"
        assert "not exposed" in result.detail.get("round_trip", "")


# ---------------------------------------------------------------------------
# probe_platform
# ---------------------------------------------------------------------------


class TestProbePlatform:
    def test_happy_path(self):
        platform = _StubPlatform()
        result = asyncio.run(
            probe_platform(_cfg(), platform_factory=lambda c: platform)
        )
        assert result.status == "ok"
        assert result.detail.get("version") == "0.5"

    def test_skipped_when_base_url_empty(self):
        cfg = _cfg(platform_url="")
        result = asyncio.run(probe_platform(cfg))
        assert result.status == "skipped"

    def test_failure_wraps(self):
        platform = _StubPlatform(raise_exc=RuntimeError("unreachable"))
        result = asyncio.run(
            probe_platform(_cfg(), platform_factory=lambda c: platform)
        )
        assert result.status == "failed"
        assert "unreachable" in result.error


# ---------------------------------------------------------------------------
# run_all_probes
# ---------------------------------------------------------------------------


class TestRunAllProbes:
    def test_all_three_run(self):
        class _MClient:
            base_url = "https://x"

        report = asyncio.run(
            run_all_probes(
                _cfg(),
                memory_factory=lambda c: _StubMemory(),
                client_factory=lambda mc: _MClient(),
                platform_factory=lambda c: _StubPlatform(),
            )
        )
        assert isinstance(report, FirstRunReport)
        assert report.memory.status == "ok"
        assert report.mage.status == "ok"
        assert report.platform.status == "ok"
        assert report.all_ok is True

    def test_partial_failure_aggregated(self):
        class _MClient:
            base_url = "https://x"

        report = asyncio.run(
            run_all_probes(
                _cfg(),
                memory_factory=lambda c: _StubMemory(raise_exc=RuntimeError("boom")),
                client_factory=lambda mc: _MClient(),
                platform_factory=lambda c: _StubPlatform(),
            )
        )
        assert report.memory.status == "failed"
        assert report.mage.status == "ok"
        assert report.platform.status == "ok"
        assert report.all_ok is False
        assert report.any_failed is True


# ---------------------------------------------------------------------------
# write_initial_config
# ---------------------------------------------------------------------------


class TestWriteInitialConfig:
    def test_writes_toml(self, tmp_path: Path):
        cfg = CareConfig(
            mage=MageConfig(
                mode="fast",
                base_url="https://api.openai.com/v1",
                api_key="sk-test",
            ),
        )
        target = tmp_path / "config.toml"
        # §1 P1 default now offloads `api_key` to a keystore;
        # pass `store_secrets=False` to preserve the literal
        # round-trip the existing assertion expects.
        result = write_initial_config(
            target, cfg, store_secrets=False,
        )
        assert result == target.resolve()
        # File is valid TOML + round-trips.
        with target.open("rb") as fp:
            data = tomllib.load(fp)
        assert data["mage"]["mode"] == "fast"
        assert data["mage"]["api_key"] == "sk-test"
        assert data["mage"]["base_url"] == "https://api.openai.com/v1"

    def test_writes_every_section_with_defaults(self, tmp_path: Path):
        target = tmp_path / "config.toml"
        cfg = CareConfig()
        write_initial_config(target, cfg)
        with target.open("rb") as fp:
            data = tomllib.load(fp)
        # All 7 nested sections present.
        for section in (
            "mage", "memory", "platform", "sandbox", "tools", "telemetry", "defaults"
        ):
            assert section in data

    def test_creates_parent_dirs(self, tmp_path: Path):
        target = tmp_path / "nested" / "deeper" / "config.toml"
        cfg = CareConfig()
        write_initial_config(target, cfg)
        assert target.is_file()

    def test_refuses_to_overwrite_by_default(self, tmp_path: Path):
        target = tmp_path / "config.toml"
        target.write_text("existing")
        with pytest.raises(FirstRunConfigError, match="refusing to overwrite"):
            write_initial_config(target, CareConfig())

    def test_overwrite_true_replaces(self, tmp_path: Path):
        target = tmp_path / "config.toml"
        target.write_text("existing junk")
        write_initial_config(target, CareConfig(), overwrite=True)
        body = target.read_text(encoding="utf-8")
        assert "junk" not in body
        assert "[mage]" in body

    def test_atomic_no_tempfile_leftover(self, tmp_path: Path):
        target = tmp_path / "config.toml"
        write_initial_config(target, CareConfig())
        # After a clean write, no `.care-config-*.tmp` files
        # should be sitting around.
        leftover = [
            p for p in tmp_path.iterdir()
            if p.name.startswith(".care-config-")
        ]
        assert leftover == []

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        result = write_initial_config("~/care.toml", CareConfig())
        assert result == (tmp_path / "care.toml").resolve()

    def test_string_path_accepted(self, tmp_path: Path):
        target = tmp_path / "config.toml"
        result = write_initial_config(str(target), CareConfig())
        assert result == target.resolve()

    def test_round_trip_via_careconfig_load(self, tmp_path: Path):
        original = CareConfig(
            mage=MageConfig(
                mode="deep",
                api_key="sk-ant",
                base_url="https://api.example.com",
                enable_web_research=True,
            ),
        )
        target = tmp_path / "config.toml"
        write_initial_config(target, original)
        # Reload via CareConfig.load and compare.
        loaded = CareConfig.load(path=target, env={})
        assert loaded.mage.mode == "deep"
        assert loaded.mage.api_key == "sk-ant"
        assert loaded.mage.base_url == "https://api.example.com"
        assert loaded.mage.enable_web_research is True
