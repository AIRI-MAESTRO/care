"""System-clipboard helper.

Bridges Textual's OSC-52 clipboard channel with a platform-
specific fallback for terminals that don't honour the
sequence. The motivating case is macOS Terminal.app, which
disables OSC 52 by default — Textual's :meth:`App.copy_to_clipboard`
quietly no-ops there.

VS Code / Cursor integrated terminals honour OSC 52 but can
mis-decode non-ASCII UTF-8 (Cyrillic pastes as ``Ð¡Ðµ…`` mojibake).
When a native clipboard helper (``xclip``, ``wl-copy``, …) is
available we prefer it and skip OSC 52 on Linux and VS Code-like
terminals so the OS clipboard gets real UTF-8.

API:

* :func:`copy_text(app, text)` — write `text` to both the
  OSC-52 channel (so iTerm2 / Ghostty / WezTerm / etc. pick
  it up) and the OS-native channel (``pbcopy`` on macOS,
  ``xclip``/``xsel`` on Linux, ``clip`` on Windows) when
  available. Returns ``True`` on at least one successful write.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import Any

_log = logging.getLogger("care.clipboard")


def copy_text(app: Any, text: str) -> bool:
    """Copy `text` to the system clipboard via every available
    channel. Returns ``True`` when at least one channel
    accepted the write.

    On Linux and VS Code / Cursor terminals, a successful
    native write skips OSC 52 so Cyrillic and other non-ASCII
    text is not corrupted by a buggy terminal decoder.
    """
    if text is None:
        return False
    text = str(text)

    native_ok = _copy_via_native(text)
    if native_ok and _prefer_native_over_osc52():
        return True

    osc_ok = _copy_via_osc52(app, text)
    return native_ok or osc_ok


def copy_selection(app: Any, source: Any) -> int | None:
    """Copy the active text selection to the clipboard and surface a
    transient toast — the copy-on-drag-release gesture the ChatScreen
    pioneered, factored out so every screen can share it.

    ``source`` is the Textual screen exposing ``get_selected_text()``.
    Returns the number of characters copied, or ``None`` when there was
    nothing selected or every clipboard channel failed.

    Feedback rides through ``app.notify`` (a transient toast) rather
    than any per-screen transcript so the gesture stays uniform across
    screens that have no chat-line concept.
    """
    try:
        selection = source.get_selected_text() or ""
    except Exception:  # noqa: BLE001
        return None
    selection = selection.strip("\n")
    if not selection.strip():
        return None
    if not copy_text(app, selection):
        return None
    chars = len(selection)
    preview = selection if chars <= 40 else selection[:37] + "…"
    preview_one_line = " ".join(preview.split())
    try:
        app.notify(
            f"Copied {chars} char{'s' if chars != 1 else ''}: "
            f"{preview_one_line}",
            timeout=2.0,
        )
    except Exception:  # noqa: BLE001
        # `notify` is best-effort: scripts driving a screen outside
        # `App.run_test()` may have no notify channel; the clipboard
        # was still written.
        pass
    return chars


def _prefer_native_over_osc52() -> bool:
    """When native clipboard tools succeed, skip OSC 52 on paths
    where the terminal decoder is known to corrupt UTF-8."""
    if not sys.platform.startswith("linux"):
        return _is_vscode_like_terminal()
    return True


def _is_vscode_like_terminal() -> bool:
    """True for VS Code / Cursor integrated terminals."""
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program.lower() in {"vscode", "cursor"}:
        return True
    ipc = os.environ.get("VSCODE_IPC_HOOK_CLI", "")
    return "vscode" in ipc.lower() or "cursor" in ipc.lower()


def _copy_via_osc52(app: Any, text: str) -> bool:
    try:
        app.copy_to_clipboard(text)
        return True
    except Exception as exc:  # noqa: BLE001
        _log.debug("OSC 52 clipboard write failed: %s", exc)
        return False


def _copy_via_native(text: str) -> bool:
    cli = _native_clipboard_cli()
    if cli is not None and _pipe_to_cli(cli, text):
        return True
    if sys.platform.startswith("linux"):
        return _copy_via_tkinter(text)
    return False


def _pipe_to_cli(cli: list[str], text: str) -> bool:
    try:
        proc = subprocess.run(
            cli,
            input=text.encode("utf-8"),
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0:
            return True
        _log.debug("clipboard CLI %s exit=%s", cli, proc.returncode)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _log.debug("clipboard CLI %s failed: %s", cli, exc)
    return False


def _copy_via_tkinter(text: str) -> bool:
    """Best-effort X11/Wayland clipboard when no CLI helper is installed."""
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        import tkinter as tk
    except ImportError:
        return False
    try:
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return True
    except Exception as exc:  # noqa: BLE001
        _log.debug("tkinter clipboard write failed: %s", exc)
        return False


def _native_clipboard_cli() -> list[str] | None:
    """Pick the right `(argv,)` for piping text into the OS
    clipboard. Returns ``None`` when none is installed.
    """
    if sys.platform == "darwin":
        return ["pbcopy"]
    if sys.platform.startswith("win"):
        return ["clip"]
    # Linux / BSD — prefer wl-copy (Wayland), then xclip / xsel.
    for argv in (
        ["wl-copy"],
        ["xclip", "-selection", "clipboard", "-i"],
        ["xsel", "--clipboard", "--input"],
    ):
        if shutil.which(argv[0]):
            return argv
    return None


__all__ = ["copy_text"]
