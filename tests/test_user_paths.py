"""Tests for `care.runtime.user_paths` (TODO §7 P0).

Covers:

* :func:`ensure_user_dirs` creates the three CARE dirs.
* Re-running is idempotent — second pass reports `existed=True`.
* Permission failures surface in the report rather than raising.
* `CareApp` runs the setup on construction + records the report.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from care.runtime.user_paths import (
    UserPathReport,
    UserPathResult,
    collect_user_paths,
    ensure_user_dirs,
)


class TestEnsureUserDirs:
    def test_creates_all_three_dirs(self, tmp_path: Path) -> None:
        report = ensure_user_dirs(
            config_dir=tmp_path / "config" / "care",
            cache_dir=tmp_path / "cache" / "care",
            state_dir=tmp_path / "state" / "care",
        )
        assert report.all_ok is True
        kinds = {r.kind for r in report.results}
        assert kinds == {"config", "cache", "state"}
        for r in report.results:
            assert r.ok is True
            assert r.existed is False
            assert r.path.is_dir()
            assert r.error == ""

    def test_idempotent_second_run_reports_existed(
        self, tmp_path: Path,
    ) -> None:
        config = tmp_path / "config" / "care"
        cache = tmp_path / "cache" / "care"
        state = tmp_path / "state" / "care"
        ensure_user_dirs(
            config_dir=config, cache_dir=cache, state_dir=state,
        )
        report = ensure_user_dirs(
            config_dir=config, cache_dir=cache, state_dir=state,
        )
        assert report.all_ok is True
        for r in report.results:
            assert r.existed is True

    def test_pre_existing_directory_is_left_alone(
        self, tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache" / "care"
        cache.mkdir(parents=True)
        (cache / "stuff.txt").write_text("hello")
        report = ensure_user_dirs(
            config_dir=tmp_path / "config" / "care",
            cache_dir=cache,
            state_dir=tmp_path / "state" / "care",
        )
        cache_result = report.by_kind("cache")
        assert cache_result is not None
        assert cache_result.ok is True
        assert cache_result.existed is True
        assert (cache / "stuff.txt").read_text() == "hello"

    @pytest.mark.skipif(
        os.geteuid() == 0,
        reason="root bypasses POSIX write permissions",
    )
    def test_unwritable_parent_lands_on_failure_with_friendly_error(
        self, tmp_path: Path,
    ) -> None:
        # Make a parent that the user can't write into.
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)  # r-x------
        try:
            report = ensure_user_dirs(
                config_dir=locked / "care",
                cache_dir=tmp_path / "cache" / "care",
                state_dir=tmp_path / "state" / "care",
            )
            assert report.all_ok is False
            failures = report.failures
            assert len(failures) == 1
            assert failures[0].kind == "config"
            assert "could not create" in failures[0].error
            # Sibling kinds succeed independently.
            cache = report.by_kind("cache")
            state = report.by_kind("state")
            assert cache is not None and cache.ok is True
            assert state is not None and state.ok is True
        finally:
            locked.chmod(0o700)

    def test_format_text_emits_one_line_per_dir(
        self, tmp_path: Path,
    ) -> None:
        report = ensure_user_dirs(
            config_dir=tmp_path / "config",
            cache_dir=tmp_path / "cache",
            state_dir=tmp_path / "state",
        )
        text = report.format_text()
        assert "✓ config" in text
        assert "✓ cache" in text
        assert "✓ state" in text
        # One newline-separated line per kind.
        assert text.count("\n") == 2

    def test_path_is_not_directory_surfaces_friendly_error(
        self, tmp_path: Path,
    ) -> None:
        target = tmp_path / "not_a_dir"
        target.write_text("I'm a file")
        report = ensure_user_dirs(
            config_dir=target,
            cache_dir=tmp_path / "cache",
            state_dir=tmp_path / "state",
        )
        # `mkdir(exist_ok=True)` raises FileExistsError when the
        # path exists as a regular file — ensure_user_dirs
        # converts that to an error report rather than blowing up.
        result = report.by_kind("config")
        assert result is not None
        assert result.ok is False
        assert "not a directory" in result.error or (
            "could not create" in result.error
        )


class TestUserPathReport:
    def test_all_ok_with_empty_results_is_true(self) -> None:
        # vacuously true — no failures means all_ok holds.
        assert UserPathReport().all_ok is True

    def test_failures_returns_only_failed_results(self) -> None:
        ok = UserPathResult(
            kind="config", path=Path("/a"), ok=True,
        )
        bad = UserPathResult(
            kind="cache", path=Path("/b"), ok=False, error="boom",
        )
        report = UserPathReport(results=(ok, bad))
        assert report.failures == (bad,)
        assert report.all_ok is False


class TestCollectUserPaths:
    def test_yields_in_canonical_order(self, tmp_path: Path) -> None:
        paths = list(collect_user_paths(
            config_dir=tmp_path / "c",
            cache_dir=tmp_path / "ca",
            state_dir=tmp_path / "s",
        ))
        assert paths == [tmp_path / "c", tmp_path / "ca", tmp_path / "s"]


class TestCareAppIntegration:
    def test_care_app_records_report_on_init(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Redirect every dir at the user_paths module level so
        # CareApp.__init__ creates them under tmp_path.
        from care.runtime import user_paths

        cfg = tmp_path / "config" / "care"
        cache = tmp_path / "cache" / "care"
        state = tmp_path / "state" / "care"
        monkeypatch.setattr(user_paths, "CARE_CONFIG_DIR", cfg)
        monkeypatch.setattr(user_paths, "CARE_CACHE_DIR", cache)
        monkeypatch.setattr(user_paths, "CARE_STATE_DIR", state)

        from care.app import CareApp

        app = CareApp(mode="returning")
        try:
            assert isinstance(app.user_path_report, UserPathReport)
            assert app.user_path_report.all_ok is True
            assert cfg.is_dir()
            assert cache.is_dir()
            assert state.is_dir()
        finally:
            # Don't actually run the Textual loop — we're only
            # exercising __init__ here.
            pass
