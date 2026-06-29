"""Tests for `care.runtime.log_discovery` (TODO §6 P2)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from care.runtime.log_discovery import (
    LOG_LEVELS,
    active_log_path,
    find_log_files,
    tail_log_lines,
)


# ---------------------------------------------------------------------------
# active_log_path
# ---------------------------------------------------------------------------


class TestActiveLogPath:
    def test_returns_none_without_env_or_handler(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("CARE_LOG_FILE", raising=False)
        root = logging.getLogger()
        # Strip any stray named handler from earlier tests.
        for h in list(root.handlers):
            if getattr(h, "name", "") == "care-app-file":
                root.removeHandler(h)
        assert active_log_path() is None

    def test_prefers_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "from-env.log"
        monkeypatch.setenv("CARE_LOG_FILE", str(target))
        assert active_log_path() == target

    def test_falls_back_to_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("CARE_LOG_FILE", raising=False)
        target = tmp_path / "from-handler.log"
        target.touch()
        handler = logging.FileHandler(target)
        handler.set_name("care-app-file")
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            resolved = active_log_path()
            assert resolved is not None
            assert resolved == target
        finally:
            root.removeHandler(handler)
            handler.close()


# ---------------------------------------------------------------------------
# find_log_files
# ---------------------------------------------------------------------------


class TestFindLogFiles:
    def test_empty_when_no_search_dirs(
        self, tmp_path: Path,
    ) -> None:
        # Pass a never-existing dir so the heuristic returns
        # empty even if the user happens to have ./logs/ on
        # disk.
        rows = find_log_files(
            search_dirs=[tmp_path / "missing"],
        )
        assert rows == []

    def test_finds_matching_files_newest_first(
        self, tmp_path: Path,
    ) -> None:
        d = tmp_path / "logs"
        d.mkdir()
        old = d / "care-app-2025-01-01.log"
        old.write_text("old\n")
        new = d / "care-app-2026-06-04.log"
        new.write_text("new\n")
        # Non-matching file should be skipped.
        (d / "other.log").write_text("ignored")
        import os
        os.utime(old, (1000.0, 1000.0))
        os.utime(new, (2000.0, 2000.0))
        rows = find_log_files(search_dirs=[d])
        assert rows[0] == new
        assert rows[1] == old


# ---------------------------------------------------------------------------
# tail_log_lines
# ---------------------------------------------------------------------------


class TestTailLogLines:
    def test_missing_file_returns_empty(
        self, tmp_path: Path,
    ) -> None:
        assert tail_log_lines(tmp_path / "missing.log") == []

    def test_returns_last_n_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "tail.log"
        path.write_text("\n".join(
            f"2026-06-04T10:00:00 [INFO] care.x: line {i}"
            for i in range(20)
        ))
        rows = tail_log_lines(path, max_lines=5)
        assert len(rows) == 5
        assert "line 15" in rows[0]
        assert "line 19" in rows[-1]

    def test_filters_by_level_floor(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "level.log"
        path.write_text("\n".join([
            "2026-06-04T10:00:00 [DEBUG] care.x: debug-line",
            "2026-06-04T10:00:01 [INFO] care.x: info-line",
            "2026-06-04T10:00:02 [WARNING] care.x: warn-line",
            "2026-06-04T10:00:03 [ERROR] care.x: error-line",
        ]))
        rows = tail_log_lines(path, level_floor="WARNING")
        assert any("warn-line" in r for r in rows)
        assert any("error-line" in r for r in rows)
        assert not any("debug-line" in r for r in rows)
        assert not any("info-line" in r for r in rows)

    def test_traceback_continuation_keeps_with_parent(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "trace.log"
        path.write_text("\n".join([
            "2026-06-04T10:00:00 [INFO] care.x: kept",
            "2026-06-04T10:00:01 [ERROR] care.x: header",
            "Traceback (most recent call last):",
            '  File "x.py", line 1, in <module>',
            "  RuntimeError: boom",
            "2026-06-04T10:00:02 [DEBUG] care.x: dropped",
        ]))
        rows = tail_log_lines(path, level_floor="ERROR")
        text = "\n".join(rows)
        # ERROR header + its 3 continuation lines kept.
        assert "ERROR] care.x: header" in text
        assert "Traceback" in text
        assert "RuntimeError" in text
        # INFO and DEBUG lines filtered out.
        assert "info-line" not in text
        assert "DEBUG" not in text

    def test_unknown_level_raises(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "x.log"
        path.write_text("[INFO] hi")
        with pytest.raises(ValueError, match="unknown level"):
            tail_log_lines(path, level_floor="LOUD")

    def test_module_substr_keeps_matching_loggers(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "mod.log"
        path.write_text("\n".join([
            "2026-06-04T10:00:00 [INFO] care.chat: hi-chat",
            "2026-06-04T10:00:01 [INFO] httpx.client: req",
            "2026-06-04T10:00:02 [INFO] care.chat.input: typed",
            "2026-06-04T10:00:03 [INFO] care.app: boot",
        ]))
        rows = tail_log_lines(path, module_substr="care.chat")
        text = "\n".join(rows)
        assert "hi-chat" in text
        assert "typed" in text
        assert "req" not in text
        assert "boot" not in text

    def test_module_substr_is_case_insensitive(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "case.log"
        path.write_text("\n".join([
            "2026-06-04T10:00:00 [INFO] CARE.Chat: mixed",
            "2026-06-04T10:00:01 [INFO] other: skip",
        ]))
        rows = tail_log_lines(path, module_substr="CHAT")
        assert any("mixed" in r for r in rows)
        assert not any("skip" in r for r in rows)

    def test_module_substr_empty_is_passthrough(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "pass.log"
        path.write_text("\n".join([
            "2026-06-04T10:00:00 [INFO] a: one",
            "2026-06-04T10:00:01 [INFO] b: two",
        ]))
        rows = tail_log_lines(path, module_substr="")
        assert len(rows) == 2

    def test_module_filter_combines_with_level_floor(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "combo.log"
        path.write_text("\n".join([
            "2026-06-04T10:00:00 [INFO] care.chat: info-chat",
            "2026-06-04T10:00:01 [WARNING] care.chat: warn-chat",
            "2026-06-04T10:00:02 [WARNING] other: warn-other",
        ]))
        rows = tail_log_lines(
            path,
            level_floor="WARNING",
            module_substr="care.chat",
        )
        text = "\n".join(rows)
        assert "warn-chat" in text
        assert "info-chat" not in text
        assert "warn-other" not in text

    def test_module_filter_keeps_continuation_under_match(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "trace.log"
        path.write_text("\n".join([
            "2026-06-04T10:00:00 [ERROR] care.chat: kept-header",
            "Traceback (most recent call last):",
            "  File \"x.py\", line 1",
            "  RuntimeError: boom",
            "2026-06-04T10:00:01 [INFO] other: dropped",
        ]))
        rows = tail_log_lines(path, module_substr="care.chat")
        text = "\n".join(rows)
        assert "kept-header" in text
        assert "Traceback" in text
        assert "RuntimeError" in text
        assert "dropped" not in text

    def test_log_levels_constant_order(self) -> None:
        assert LOG_LEVELS == (
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
        )
