"""Tests for care.runtime.hint_fit."""

from care.runtime.hint_fit import fit_line, fit_segments


def test_fit_segments_keeps_all_when_room():
    assert fit_segments(["a", "b", "c"], 20) == "a · b · c"


def test_fit_segments_drops_trailing():
    assert fit_segments(["Enter — send", "@file — attach", "Esc"], 32) == (
        "Enter — send · @file — attach"
    )


def test_fit_segments_truncates_single_segment():
    assert fit_segments(["very long hint text"], 10) == "very long…"


def test_fit_line_noop_when_fits():
    assert fit_line("short", 10) == "short"


def test_fit_line_truncates():
    assert fit_line("abcdefghij", 6) == "abcde…"
