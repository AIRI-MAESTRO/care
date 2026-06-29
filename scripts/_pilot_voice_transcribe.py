"""Live pilot for Phase 9 P3 — /voice transcribe sub-command.

End-to-end against a real Textual session with a monkeypatched
Whisper backend (so the test doesn't require Whisper actually
installed):

1. /voice transcribe with no path → usage warning.
2. /voice transcribe <missing-file> → not-found warning.
3. /voice transcribe <text-file.txt> → non-audio warning.
4. /voice transcribe <fake.wav> with stub backend →
   transcribed text drops into the input, system line
   reports word count.
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


def _stub_backend(_cls):
    return "whisper"


def _stub_invoke(_backend, _path):
    return "this is a transcribed three word body"


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        os.environ["CARE_CHAT__THEME_SIDECAR"] = str(Path(td) / "theme.txt")
        os.environ["CARE_CHAT__TUTORIAL_SIDECAR"] = str(Path(td) / "tut.json")
        os.environ["CARE_CHAT__SESSION_LOG_DIR"] = str(Path(td) / "sess")
        os.environ["CARE_CHAT__BRANCHES_DIR"] = str(Path(td) / "branches")

        # Stub the backend so we don't need real Whisper.
        original_detect = ChatScreen._detect_voice_backend
        original_invoke = ChatScreen._invoke_whisper
        ChatScreen._detect_voice_backend = classmethod(_stub_backend)
        ChatScreen._invoke_whisper = staticmethod(_stub_invoke)

        try:
            app = _Host()
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen_stack[-1]
                inp = screen.query_one("#chat-input", Input)

                # 1. Missing path
                inp.value = "/voice transcribe"
                await inp.action_submit()
                await pilot.pause()
                usage = [
                    line.text for line in screen._lines
                    if line.role == "system"
                    and "Usage: `/voice transcribe" in line.text
                ]
                assert usage
                print(f"[1] usage warning posted: {usage[-1][:60]!r}…")

                # 2. Missing file
                ghost = Path(td) / "absent.wav"
                inp.value = f"/voice transcribe {ghost}"
                await inp.action_submit()
                await pilot.pause()
                missing = [
                    line for line in screen._lines
                    if line.role == "system"
                    and "not found" in line.text
                ]
                assert missing
                print("[2] missing-file warning posted")

                # 3. Non-audio
                txt = Path(td) / "notes.txt"
                txt.write_text("hello", encoding="utf-8")
                inp.value = f"/voice transcribe {txt}"
                await inp.action_submit()
                await pilot.pause()
                non_audio = [
                    line for line in screen._lines
                    if line.role == "system"
                    and "doesn't look like an audio file" in line.text
                ]
                assert non_audio
                print("[3] non-audio warning posted")

                # 4. Real flow with stub
                audio = Path(td) / "voice.wav"
                audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEdata")
                inp.value = f"/voice transcribe {audio}"
                await inp.action_submit()
                await pilot.pause()
                print(f"[4] input after transcribe = {inp.value!r}")
                assert inp.value == "this is a transcribed three word body"
                confirmations = [
                    line.text for line in screen._lines
                    if line.role == "system"
                    and "Transcribed" in line.text
                    and "→ input" in line.text
                ]
                assert confirmations
                print(f"[4] confirmation: {confirmations[-1]}")
                assert "7 words" in confirmations[-1]
        finally:
            ChatScreen._detect_voice_backend = original_detect
            ChatScreen._invoke_whisper = original_invoke

        print("=" * 60)
        print("ALL CHECKS PASSED — /voice transcribe works end-to-end")


if __name__ == "__main__":
    asyncio.run(main())
