"""Live pilot for Phase 9 P3 — /imgpreview terminal-graphics support.

End-to-end:
1. With no protocol env vars set → `/imgpreview status` warns.
2. Set KITTY_WINDOW_ID → `/imgpreview status` reports Kitty.
3. Set TERM_PROGRAM=iTerm.app → `/imgpreview status` reports iTerm2.
4. Set TERM_PROGRAM=WezTerm → `/imgpreview status` reports WezTerm.
5. Write a fake PNG to disk; `/imgpreview <path>` builds the
   protocol-specific escape sequence and reports it.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App  # noqa: E402
from textual.widgets import Input  # noqa: E402

from care.screens.chat import ChatScreen  # noqa: E402


class _Host(App):
    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


async def _run_imgpreview_once(*, env_setup: dict[str, str | None]):
    """Spin a fresh ChatScreen with the given env, run
    `/imgpreview status`, and return the matched system-line
    text."""
    # Apply env
    for k, v in env_setup.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        inp = screen.query_one("#chat-input", Input)
        inp.value = "/imgpreview status"
        await inp.action_submit()
        await pilot.pause()
        msgs = [
            line.text for line in screen._lines
            if line.role == "system"
            and ("Inline-graphics" in line.text
                 or "No inline-graphics" in line.text)
        ]
    return msgs[-1] if msgs else ""


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        os.environ["CARE_CHAT__THEME_SIDECAR"] = str(Path(td) / "theme.txt")
        os.environ["CARE_CHAT__TUTORIAL_SIDECAR"] = str(Path(td) / "tut.json")
        os.environ["CARE_CHAT__SESSION_LOG_DIR"] = str(Path(td) / "sess")
        os.environ["CARE_CHAT__BRANCHES_DIR"] = str(Path(td) / "branches")

        # 1. No protocol
        text = await _run_imgpreview_once(env_setup={
            "KITTY_WINDOW_ID": None,
            "TERM_PROGRAM": None,
            "LC_TERMINAL": None,
        })
        print(f"[1] no-protocol: {text!r}")
        assert "No inline-graphics" in text
        print("[1] OK — no-protocol path warns\n")

        # 2. Kitty
        text = await _run_imgpreview_once(env_setup={
            "KITTY_WINDOW_ID": "42",
            "TERM_PROGRAM": None,
            "LC_TERMINAL": None,
        })
        print(f"[2] kitty: {text!r}")
        assert "kitty" in text
        print("[2] OK — Kitty detected\n")

        # 3. iTerm2
        text = await _run_imgpreview_once(env_setup={
            "KITTY_WINDOW_ID": None,
            "TERM_PROGRAM": "iTerm.app",
            "LC_TERMINAL": None,
        })
        print(f"[3] iterm2: {text!r}")
        assert "iterm2" in text
        print("[3] OK — iTerm2 detected\n")

        # 4. WezTerm
        text = await _run_imgpreview_once(env_setup={
            "KITTY_WINDOW_ID": None,
            "TERM_PROGRAM": "WezTerm",
            "LC_TERMINAL": None,
        })
        print(f"[4] wezterm: {text!r}")
        assert "wezterm" in text
        print("[4] OK — WezTerm detected\n")

        # 5. Path build with kitty
        os.environ["KITTY_WINDOW_ID"] = "1"
        os.environ.pop("TERM_PROGRAM", None)
        os.environ.pop("LC_TERMINAL", None)
        img = Path(td) / "tiny.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake-payload")

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            inp = screen.query_one("#chat-input", Input)
            inp.value = f"/imgpreview {img}"
            await inp.action_submit()
            await pilot.pause()
            reports = [
                line.text for line in screen._lines
                if line.role == "system"
                and "Built Kitty graphics" in line.text
            ]
        print(f"[5] image build report: {reports[-1] if reports else ''}")
        assert reports
        assert "tiny.png" in reports[-1]
        # Verify direct builder output looks right.
        bytes_ = img.read_bytes()
        seq = ChatScreen._build_kitty_image_sequence(bytes_)
        assert seq.startswith("\x1b_G")
        assert seq.endswith("\x1b\\")
        print(f"[5] direct kitty sequence length = {len(seq)} bytes")
        print("[5] OK — image → escape sequence built\n")

        print("=" * 60)
        print("ALL CHECKS PASSED — /imgpreview detection + sequence build")


if __name__ == "__main__":
    asyncio.run(main())
