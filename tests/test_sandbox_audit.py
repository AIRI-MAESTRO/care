"""Tests for ``care.sandbox.SandboxAuditLogger`` (TODO §6.2 P1).

Real file IO via ``tmp_path`` throughout — no monkey-patching of
`pathlib` / `open`. The integration test at the bottom runs a real
``echo`` through ``LocalSandboxBackend`` and verifies the audit
entry round-trips through disk.

Coverage layers:
1. ``SandboxAuditEntry`` round-trips through to_dict / from_dict.
2. ``build_entry`` populates every field correctly given a
   :class:`SandboxHandle` + :class:`RunResult` + fake clock.
3. ``log_run`` appends a JSON line (parent dir auto-created);
   second call appends without overwriting.
4. ``log_run`` returns ``False`` (not raises) when the path is a
   directory or otherwise unwritable.
5. ``tail`` + ``all_entries`` parse correctly; unknown version
   raises.
6. End-to-end: real subprocess via :class:`LocalSandboxBackend`,
   audit line written, parsed back to a matching entry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from care.sandbox import (
    AUDIT_FORMAT_VERSION,
    LocalSandboxBackend,
    RunResult,
    SandboxAuditEntry,
    SandboxAuditError,
    SandboxAuditLogger,
    SandboxHandle,
)

FROZEN_TS = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def frozen_clock():
    return lambda: FROZEN_TS


@pytest.fixture
def handle(tmp_path: Path) -> SandboxHandle:
    return SandboxHandle(
        backend_name="local",
        workspace=tmp_path,
        skill_sha256="a" * 64,
        network_enforced=False,
    )


@pytest.fixture
def run_result() -> RunResult:
    return RunResult(
        exit_code=0,
        stdout=b"hello\n",
        stderr=b"",
        duration_seconds=0.123,
        timed_out=False,
        network_enforced=False,
    )


# ---------------------------------------------------------------------------
# SandboxAuditEntry shape
# ---------------------------------------------------------------------------


class TestSandboxAuditEntry:
    def test_round_trip(self):
        entry = SandboxAuditEntry(
            timestamp=FROZEN_TS,
            backend_name="local",
            skill_sha256="a" * 64,
            cmd=("echo", "hi"),
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
            stdout_sha256="x" * 64,
            stderr_sha256="y" * 64,
            network_enforced=False,
            files_written=("out/report.json",),
            extras={"run_id": "r-1"},
        )
        restored = SandboxAuditEntry.from_dict(entry.to_dict())
        assert restored == entry

    def test_frozen(self):
        entry = SandboxAuditEntry(
            timestamp=FROZEN_TS,
            backend_name="local",
            skill_sha256="a",
            cmd=("x",),
            exit_code=0,
            duration_seconds=0.0,
            timed_out=False,
            stdout_sha256="0",
            stderr_sha256="0",
            network_enforced=False,
        )
        with pytest.raises(AttributeError):
            entry.exit_code = 1  # type: ignore[misc]

    def test_from_dict_rejects_unknown_version(self):
        with pytest.raises(SandboxAuditError, match="unknown audit-log version"):
            SandboxAuditEntry.from_dict(
                {
                    "version": 99,
                    "timestamp": FROZEN_TS.isoformat(),
                    "backend_name": "local",
                    "skill_sha256": "x",
                    "cmd": [],
                    "exit_code": 0,
                    "duration_seconds": 0,
                    "timed_out": False,
                    "stdout_sha256": "0",
                    "stderr_sha256": "0",
                    "network_enforced": False,
                }
            )


# ---------------------------------------------------------------------------
# build_entry (pure)
# ---------------------------------------------------------------------------


class TestBuildEntry:
    def test_populates_every_field(self, tmp_path, frozen_clock, handle, run_result):
        # Write one out/ file so files_written has something to find.
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "report.json").write_text("{}")

        logger = SandboxAuditLogger(path=tmp_path / "audit.log", clock=frozen_clock)
        entry = logger.build_entry(handle, ["echo", "hi"], run_result)

        assert entry.timestamp == FROZEN_TS
        assert entry.backend_name == "local"
        assert entry.skill_sha256 == "a" * 64
        assert entry.cmd == ("echo", "hi")
        assert entry.exit_code == 0
        assert entry.duration_seconds == 0.123
        assert entry.network_enforced is False
        assert entry.timed_out is False
        assert "out/report.json" in entry.files_written
        # Hashes are deterministic.
        assert entry.stdout_sha256 == (
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        )
        # SHA of empty bytes is the canonical empty-SHA hash.
        assert (
            entry.stderr_sha256
            == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_no_out_dir_yields_empty_files_written(
        self, tmp_path, frozen_clock, handle, run_result
    ):
        # No out/ subdir under the workspace.
        logger = SandboxAuditLogger(path=tmp_path / "audit.log", clock=frozen_clock)
        entry = logger.build_entry(handle, ["x"], run_result)
        assert entry.files_written == ()

    def test_extras_passthrough(self, tmp_path, frozen_clock, handle, run_result):
        logger = SandboxAuditLogger(path=tmp_path / "audit.log", clock=frozen_clock)
        entry = logger.build_entry(
            handle, ["x"], run_result, extras={"run_id": "r-7", "tag": "demo"}
        )
        assert entry.extras == {"run_id": "r-7", "tag": "demo"}


# ---------------------------------------------------------------------------
# log_run (write path)
# ---------------------------------------------------------------------------


class TestLogRun:
    def test_creates_parent_dir_and_writes_jsonline(
        self, tmp_path, frozen_clock, handle, run_result
    ):
        log_path = tmp_path / "deep" / "deeper" / "audit.log"
        logger = SandboxAuditLogger(path=log_path, clock=frozen_clock)

        assert logger.log_run(handle, ["echo", "hi"], run_result) is True
        assert log_path.exists()
        line = log_path.read_text().strip()
        # One valid JSON object on the line.
        parsed = json.loads(line)
        assert parsed["version"] == AUDIT_FORMAT_VERSION
        assert parsed["cmd"] == ["echo", "hi"]
        assert parsed["exit_code"] == 0

    def test_appends_not_overwrites(self, tmp_path, frozen_clock, handle, run_result):
        log_path = tmp_path / "audit.log"
        logger = SandboxAuditLogger(path=log_path, clock=frozen_clock)
        assert logger.log_run(handle, ["echo", "1"], run_result) is True
        assert logger.log_run(handle, ["echo", "2"], run_result) is True
        lines = [line for line in log_path.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        cmds = [json.loads(line)["cmd"] for line in lines]
        assert cmds == [["echo", "1"], ["echo", "2"]]

    def test_returns_false_on_write_failure(
        self, tmp_path, frozen_clock, handle, run_result, capsys
    ):
        """When the path resolves to a directory, the append raises
        ``IsADirectoryError`` (subclass of ``OSError``); the logger
        must swallow + return False so the skill run isn't affected."""
        log_path = tmp_path / "audit-as-dir"
        log_path.mkdir()  # path is a dir, not a file
        logger = SandboxAuditLogger(path=log_path, clock=frozen_clock)
        assert logger.log_run(handle, ["echo"], run_result) is False
        captured = capsys.readouterr()
        assert "failed to write" in captured.err


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


class TestRead:
    def test_tail_empty_when_log_missing(self, tmp_path):
        logger = SandboxAuditLogger(path=tmp_path / "missing.log")
        assert logger.tail() == []
        assert logger.all_entries() == []

    def test_tail_returns_last_n(self, tmp_path, frozen_clock, handle, run_result):
        log_path = tmp_path / "audit.log"
        logger = SandboxAuditLogger(path=log_path, clock=frozen_clock)
        for i in range(5):
            logger.log_run(handle, ["cmd", str(i)], run_result)
        last_two = logger.tail(n=2)
        assert [e.cmd for e in last_two] == [("cmd", "3"), ("cmd", "4")]

    def test_tail_zero_or_negative_returns_empty(
        self, tmp_path, frozen_clock, handle, run_result
    ):
        log_path = tmp_path / "audit.log"
        logger = SandboxAuditLogger(path=log_path, clock=frozen_clock)
        logger.log_run(handle, ["x"], run_result)
        assert logger.tail(n=0) == []
        assert logger.tail(n=-1) == []

    def test_all_entries_parses_every_line(
        self, tmp_path, frozen_clock, handle, run_result
    ):
        log_path = tmp_path / "audit.log"
        logger = SandboxAuditLogger(path=log_path, clock=frozen_clock)
        for i in range(3):
            logger.log_run(handle, ["cmd", str(i)], run_result)
        entries = logger.all_entries()
        assert len(entries) == 3
        assert all(isinstance(e, SandboxAuditEntry) for e in entries)

    def test_corrupt_line_raises(self, tmp_path):
        log_path = tmp_path / "audit.log"
        log_path.write_text("{not json\n")
        logger = SandboxAuditLogger(path=log_path)
        with pytest.raises(SandboxAuditError, match="could not parse"):
            logger.all_entries()


# ---------------------------------------------------------------------------
# End-to-end: real subprocess via LocalSandboxBackend
# ---------------------------------------------------------------------------


class _FakeSkill:
    sha256 = "f" * 64
    path = Path("/tmp/fake-skill")
    allowed_tools: list = []


@pytest.mark.asyncio
async def test_logs_real_subprocess_run(tmp_path, frozen_clock):
    """Run an actual ``echo`` through LocalSandboxBackend; verify
    the audit entry round-trips through disk and matches what the
    backend returned."""
    backend = LocalSandboxBackend()
    handle = await backend.prepare(_FakeSkill(), workspace=tmp_path)
    try:
        result = await backend.run(handle, ["echo", "audit-me"])
        assert result.ok
        # Create a fake "out/" file the skill "produced" so the
        # files_written field has content.
        (tmp_path / "out").mkdir(exist_ok=True)
        (tmp_path / "out" / "report.txt").write_text("ok")

        log_path = tmp_path / "audit.log"
        logger = SandboxAuditLogger(path=log_path, clock=frozen_clock)
        assert logger.log_run(handle, ["echo", "audit-me"], result) is True

        [entry] = logger.tail()
        assert entry.cmd == ("echo", "audit-me")
        assert entry.exit_code == 0
        assert entry.duration_seconds == result.duration_seconds
        assert entry.timed_out is False
        assert entry.network_enforced is False
        assert "out/report.txt" in entry.files_written
    finally:
        await backend.cleanup(handle)
