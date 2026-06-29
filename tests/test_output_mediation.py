"""Tests for ``care.sandbox.output_mediation`` (TODO §6.2 P2).

Real file IO in ``tmp_path`` for every scenario. Coverage:

1. Missing / empty / not-a-directory ``out/`` → informational
   findings (never block).
2. Executable bit → blocked.
3. Binary magic bytes (ELF, Mach-O, PE) → blocked.
4. Shebang scripts → warning.
5. Network tokens in text → warning.
6. Symlink escape → blocked.
7. Large file → warning (still scanned within cap).
8. ``MediationReport`` shape: `is_clean`, `has_blockers`,
   `findings_by_severity`.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from care.sandbox import (
    DEFAULT_MAX_FILE_BYTES,
    MediationReport,
    OutputFinding,
    scan_output_dir,
)


def _mkws(tmp_path: Path) -> Path:
    """Create the workspace + the conventional ``out/`` subdir."""
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Missing / empty / wrong-type out/
# ---------------------------------------------------------------------------


class TestEmptyAndMissing:
    def test_missing_out_dir_yields_info_only(self, tmp_path):
        # No `out/` at all.
        report = scan_output_dir(tmp_path)
        assert report.total_files == 0
        assert report.scanned_bytes == 0
        assert report.is_clean is True
        [finding] = report.findings
        assert finding.severity == "info"
        assert finding.kind == "empty_out"

    def test_out_exists_but_is_file_yields_warning(self, tmp_path):
        (tmp_path / "out").write_text("oops")
        report = scan_output_dir(tmp_path)
        [finding] = report.findings
        assert finding.severity == "warning"
        assert finding.kind == "empty_out"

    def test_empty_out_dir_yields_info(self, tmp_path):
        _mkws(tmp_path)
        report = scan_output_dir(tmp_path)
        assert report.total_files == 0
        assert report.is_clean is True
        assert any(f.kind == "empty_out" for f in report.findings)


# ---------------------------------------------------------------------------
# Executable bit
# ---------------------------------------------------------------------------


class TestExecutableBit:
    def test_executable_text_file_blocked(self, tmp_path):
        ws = _mkws(tmp_path)
        target = ws / "out" / "run.txt"
        target.write_text("hello")
        target.chmod(0o755)
        report = scan_output_dir(ws)
        blockers = [f for f in report.findings if f.severity == "blocked"]
        kinds = {f.kind for f in blockers}
        assert "executable_bit" in kinds

    def test_non_executable_text_file_clean(self, tmp_path):
        ws = _mkws(tmp_path)
        target = ws / "out" / "report.txt"
        target.write_text("Sales: $1,000\n")
        target.chmod(0o644)
        report = scan_output_dir(ws)
        # No blocked findings.
        assert report.has_blockers is False


# ---------------------------------------------------------------------------
# Binary magic bytes
# ---------------------------------------------------------------------------


class TestBinaryMagic:
    @pytest.mark.parametrize(
        "magic,label_fragment",
        [
            (b"\x7fELF\x01\x01\x01\x00" + b"\x00" * 8, "ELF"),
            (b"MZ\x90\x00\x03\x00\x00\x00", "PE"),
            (b"\xcf\xfa\xed\xfe" + b"\x00" * 12, "Mach-O"),
        ],
    )
    def test_known_magic_blocked(self, tmp_path, magic, label_fragment):
        ws = _mkws(tmp_path)
        binary = ws / "out" / "thing.bin"
        binary.write_bytes(magic + b"\x00" * 100)
        binary.chmod(0o644)  # no exec bit, isolate the magic-bytes path
        report = scan_output_dir(ws)
        blockers = report.findings_by_severity("blocked")
        assert any(
            f.kind == "binary_content" and label_fragment in f.detail
            for f in blockers
        )

    def test_arbitrary_bytes_not_blocked(self, tmp_path):
        ws = _mkws(tmp_path)
        (ws / "out" / "noise.bin").write_bytes(b"random binary data " * 50)
        report = scan_output_dir(ws)
        # No false-positive magic-byte match for arbitrary content.
        assert not any(f.kind == "binary_content" for f in report.findings)


# ---------------------------------------------------------------------------
# Shebang scripts
# ---------------------------------------------------------------------------


class TestShebang:
    def test_shebang_emits_warning(self, tmp_path):
        ws = _mkws(tmp_path)
        (ws / "out" / "helper.py").write_text(
            "#!/usr/bin/env python3\nprint('hi')\n"
        )
        report = scan_output_dir(ws)
        warnings = report.findings_by_severity("warning")
        shebang_findings = [f for f in warnings if f.kind == "shebang_script"]
        assert len(shebang_findings) == 1
        assert "python3" in shebang_findings[0].detail

    def test_text_without_shebang_no_warning(self, tmp_path):
        ws = _mkws(tmp_path)
        (ws / "out" / "report.md").write_text("# Sales report\n\nAll good.\n")
        report = scan_output_dir(ws)
        assert not any(f.kind == "shebang_script" for f in report.findings)


# ---------------------------------------------------------------------------
# Network tokens
# ---------------------------------------------------------------------------


class TestNetworkTokens:
    @pytest.mark.parametrize(
        "snippet",
        [
            "Visit https://example.com for details",
            "Run `curl https://api.example/data > /tmp/x`",
            "wget http://internal.example/payload.sh",
            "Connect via nc 10.0.0.1 8080",
        ],
    )
    def test_token_emits_warning(self, tmp_path, snippet):
        ws = _mkws(tmp_path)
        (ws / "out" / "doc.txt").write_text(snippet)
        report = scan_output_dir(ws)
        warnings = [f for f in report.findings if f.kind == "network_token"]
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"

    def test_no_tokens_no_finding(self, tmp_path):
        ws = _mkws(tmp_path)
        (ws / "out" / "data.csv").write_text("col1,col2\n1,2\n")
        report = scan_output_dir(ws)
        assert not any(f.kind == "network_token" for f in report.findings)

    def test_binary_files_skip_token_scan(self, tmp_path):
        """A binary blob with bytes that happen to contain "wget"
        shouldn't trigger the text scan."""
        ws = _mkws(tmp_path)
        # 30% non-printable → classified as binary, scan skipped.
        binary = b"\x00\x01\x02" * 200 + b"wget " * 50
        (ws / "out" / "blob.bin").write_bytes(binary)
        report = scan_output_dir(ws)
        # No token finding because the file is classified non-textual.
        assert not any(f.kind == "network_token" for f in report.findings)


# ---------------------------------------------------------------------------
# Symlink escape
# ---------------------------------------------------------------------------


class TestSymlinkEscape:
    def test_symlink_outside_workspace_blocked(self, tmp_path):
        ws = _mkws(tmp_path)
        # Create the target outside the workspace.
        outside = tmp_path.parent / "outside-target.txt"
        outside.write_text("secret")
        link = ws / "out" / "escape.link"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        report = scan_output_dir(ws)
        blockers = report.findings_by_severity("blocked")
        assert any(f.kind == "symlink_escape" for f in blockers)

    def test_symlink_inside_workspace_clean(self, tmp_path):
        ws = _mkws(tmp_path)
        inside = ws / "out" / "data.txt"
        inside.write_text("hi")
        link = ws / "out" / "alias.link"
        try:
            os.symlink(inside, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        report = scan_output_dir(ws)
        # Symlink resolves inside ws → no escape finding.
        assert not any(f.kind == "symlink_escape" for f in report.findings)


# ---------------------------------------------------------------------------
# Large file
# ---------------------------------------------------------------------------


class TestLargeFile:
    def test_oversize_file_warns_but_still_scans(self, tmp_path):
        ws = _mkws(tmp_path)
        small_cap = 1024  # 1 KiB
        # Magic bytes at start, then padding past the cap.
        (ws / "out" / "big.bin").write_bytes(
            b"\x7fELF\x01\x01\x01\x00" + b"\x00" * (small_cap * 2)
        )
        report = scan_output_dir(ws, max_file_bytes=small_cap)
        kinds = {f.kind for f in report.findings}
        # Both findings present: large_file warning AND binary_content block
        # (because the cap still included the magic bytes at offset 0).
        assert "large_file" in kinds
        assert "binary_content" in kinds

    def test_default_cap_value_constant(self):
        """Pin the documented default so a silent bump is visible."""
        assert DEFAULT_MAX_FILE_BYTES == 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------


class TestMediationReport:
    def test_is_clean_when_only_info(self, tmp_path):
        report = scan_output_dir(tmp_path)
        assert report.is_clean is True
        assert report.has_blockers is False

    def test_is_clean_false_when_warning(self, tmp_path):
        ws = _mkws(tmp_path)
        (ws / "out" / "doc.txt").write_text("see https://example.com")
        report = scan_output_dir(ws)
        assert report.is_clean is False
        assert report.has_blockers is False

    def test_has_blockers_when_blocked_present(self, tmp_path):
        ws = _mkws(tmp_path)
        (ws / "out" / "evil.bin").write_bytes(b"\x7fELF" + b"\x00" * 100)
        report = scan_output_dir(ws)
        assert report.has_blockers is True
        assert report.is_clean is False

    def test_findings_by_severity_filters(self, tmp_path):
        ws = _mkws(tmp_path)
        (ws / "out" / "doc.txt").write_text("https://example.com")
        (ws / "out" / "binary.bin").write_bytes(b"\x7fELF" + b"\x00" * 50)
        report = scan_output_dir(ws)
        blocked = report.findings_by_severity("blocked")
        warning = report.findings_by_severity("warning")
        assert all(f.severity == "blocked" for f in blocked)
        assert all(f.severity == "warning" for f in warning)
        assert len(blocked) >= 1
        assert len(warning) >= 1

    def test_report_is_frozen(self):
        r = MediationReport(findings=(), total_files=0, scanned_bytes=0)
        with pytest.raises((AttributeError, TypeError)):
            r.total_files = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------


class TestOutputFinding:
    def test_frozen(self):
        f = OutputFinding(severity="info", kind="empty_out", path="out")
        with pytest.raises(AttributeError):
            f.severity = "blocked"  # type: ignore[misc]

    def test_defaults(self):
        f = OutputFinding(severity="info", kind="empty_out", path="out")
        assert f.detail == ""
        assert f.bytes_inspected == 0


# ---------------------------------------------------------------------------
# Integration: multi-finding scan
# ---------------------------------------------------------------------------


class TestMixedTree:
    def test_mixed_tree_aggregates_correctly(self, tmp_path):
        """Plant one of each kind + a clean file; verify the
        report aggregates without losing any finding."""
        ws = _mkws(tmp_path)
        # Clean text file
        (ws / "out" / "ok.csv").write_text("col\n1\n")
        # Shebang script
        (ws / "out" / "script.sh").write_text("#!/bin/sh\necho ok\n")
        # Executable
        (ws / "out" / "binary").write_bytes(b"\x7fELF" + b"\x00" * 64)
        (ws / "out" / "binary").chmod(0o755)
        # Network token in text
        (ws / "out" / "report.md").write_text("see https://example.com")

        report = scan_output_dir(ws)
        kinds = {f.kind for f in report.findings}
        assert "shebang_script" in kinds
        assert "binary_content" in kinds
        assert "executable_bit" in kinds
        assert "network_token" in kinds
        assert report.total_files == 4
        assert report.has_blockers is True

    def test_nested_directories_scanned(self, tmp_path):
        ws = _mkws(tmp_path)
        nested = ws / "out" / "sub" / "deeper"
        nested.mkdir(parents=True)
        (nested / "doc.txt").write_text("plain text")
        report = scan_output_dir(ws)
        assert report.total_files == 1
        assert report.is_clean is True


# ---------------------------------------------------------------------------
# Symlink behaviour edge cases
# ---------------------------------------------------------------------------


def test_executable_perms_helper_works():
    """Sanity: confirm the stat mask we use matches what Path.chmod
    produces. Catches the unlikely case where a future Python's
    stat module changes the constants."""
    assert stat.S_IXUSR == 0o100
    assert stat.S_IXGRP == 0o010
    assert stat.S_IXOTH == 0o001
