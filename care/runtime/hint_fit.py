"""Fit hint lines to terminal width — drop or truncate segments."""

from __future__ import annotations

from typing import Sequence

DEFAULT_HINT_SEP = " · "
ELLIPSIS = "…"


def fit_segments(
    segments: Sequence[str],
    width: int,
    *,
    sep: str = DEFAULT_HINT_SEP,
) -> str:
    """Join *segments* with *sep*, dropping trailing ones until the line fits.

    When even a single segment exceeds *width*, truncate it with an ellipsis.
    Non-positive *width* returns the full join (caller's fallback).
    """
    if not segments:
        return ""
    if width <= 0:
        return sep.join(segments)
    for count in range(len(segments), 0, -1):
        line = sep.join(segments[:count])
        if len(line) <= width:
            return line
    head = segments[0]
    if len(head) <= width:
        return head
    if width <= 1:
        return ELLIPSIS
    return head[: width - 1] + ELLIPSIS


def fit_line(text: str, width: int) -> str:
    """Truncate a single line with ellipsis when needed."""
    if width <= 0 or len(text) <= width:
        return text
    if width <= 1:
        return ELLIPSIS
    return text[: width - 1] + ELLIPSIS


__all__ = ["DEFAULT_HINT_SEP", "ELLIPSIS", "fit_line", "fit_segments"]
