"""Clipboard helper behaviour — native vs OSC-52 ordering."""

from __future__ import annotations

import sys

import care.runtime.clipboard as clipboard


class _App:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def copy_to_clipboard(self, text: str) -> None:
        self.calls.append(text)


def test_native_success_skips_osc52_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    app = _App()
    monkeypatch.setattr(clipboard, "_copy_via_native", lambda _text: True)
    monkeypatch.setattr(clipboard, "_copy_via_osc52", lambda _app, _text: True)

    assert clipboard.copy_text(app, "Сейчас") is True
    assert app.calls == []


def test_osc52_used_when_native_unavailable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    app = _App()
    monkeypatch.setattr(clipboard, "_copy_via_native", lambda _text: False)
    monkeypatch.setattr(clipboard, "_copy_via_osc52", lambda _app, text: app.copy_to_clipboard(text) or True)

    assert clipboard.copy_text(app, "Сейчас") is True
    assert app.calls == ["Сейчас"]


def test_vscode_like_terminal_detected(monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    assert clipboard._is_vscode_like_terminal() is True
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.setenv("VSCODE_IPC_HOOK_CLI", "/tmp/cursor-ipc.sock")
    assert clipboard._is_vscode_like_terminal() is True
