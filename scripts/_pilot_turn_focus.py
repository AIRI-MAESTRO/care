"""Live pilot for Phase 9 P1 — Ctrl+0 turn-focus mode.

End-to-end:
1. Build a transcript with 3 turns, each with a user + assistant.
2. Toggle focus mode ON — confirm earlier turns' widgets pick
   up the hidden class while the latest turn stays visible.
3. Post a new user line while focus mode is ON — confirm the
   previous turn auto-collapses.
4. Toggle focus mode OFF — confirm every widget restores.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App  # noqa: E402

from care.screens.chat import ChatScreen  # noqa: E402


class _Host(App):
    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        os.environ["CARE_CHAT__THEME_SIDECAR"] = str(Path(td) / "theme.txt")
        os.environ["CARE_CHAT__TUTORIAL_SIDECAR"] = str(Path(td) / "tut.json")
        os.environ["CARE_CHAT__SESSION_LOG_DIR"] = str(Path(td) / "sess")

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            for turn in range(1, 4):
                screen._post_line("user", f"turn {turn} prompt")
                screen._post_line("assistant", f"turn {turn} reply")
            await pilot.pause()
            print(f"[setup] current_turn = {screen._current_turn}")
            print(f"[setup] focus_mode = {screen._turn_focus_mode}")

            # Pre-focus: nothing hidden.
            hidden_before = list(
                screen.query(f".{ChatScreen._TURN_HIDDEN_CLASS}").results(),
            )
            print(f"[1] hidden widgets before focus = {len(hidden_before)}")
            assert hidden_before == []

            # Focus ON
            screen.action_focus_current_turn()
            await pilot.pause()
            hidden_after = list(
                screen.query(f".{ChatScreen._TURN_HIDDEN_CLASS}").results(),
            )
            turn3 = list(screen.query(".chat-line-turn-3").results())
            turn1 = list(screen.query(".chat-line-turn-1").results())
            turn2 = list(screen.query(".chat-line-turn-2").results())
            print(f"[2] focus ON: hidden = {len(hidden_after)}")
            print(f"[2] turn-3 widgets = {len(turn3)} (should all be visible)")
            print(f"[2] turn-1 widgets = {len(turn1)} (should be hidden)")
            print(f"[2] turn-2 widgets = {len(turn2)} (should be hidden)")
            for w in turn3:
                assert ChatScreen._TURN_HIDDEN_CLASS not in w.classes
            for w in turn1 + turn2:
                assert ChatScreen._TURN_HIDDEN_CLASS in w.classes
            print("[2] OK — focus ON hides earlier turns only\n")

            # New user line auto-collapses prior turn 3
            screen._post_line("user", "turn 4 prompt")
            await pilot.pause()
            turn3_after_new = list(
                screen.query(".chat-line-turn-3").results(),
            )
            turn4 = list(screen.query(".chat-line-turn-4").results())
            print(f"[3] new turn 4 mounted: turn-4 widgets = {len(turn4)}")
            print(
                f"[3] previous turn-3 widgets hidden = "
                f"{all(ChatScreen._TURN_HIDDEN_CLASS in w.classes for w in turn3_after_new)}",
            )
            for w in turn3_after_new:
                assert ChatScreen._TURN_HIDDEN_CLASS in w.classes
            for w in turn4:
                assert ChatScreen._TURN_HIDDEN_CLASS not in w.classes
            print("[3] OK — auto-collapse on new turn while focus ON\n")

            # Focus OFF restores
            screen.action_focus_current_turn()
            await pilot.pause()
            hidden_final = list(
                screen.query(f".{ChatScreen._TURN_HIDDEN_CLASS}").results(),
            )
            print(f"[4] focus OFF: hidden = {len(hidden_final)}")
            assert hidden_final == []
            print("[4] OK — focus OFF restores every turn\n")

        print("=" * 60)
        print("ALL CHECKS PASSED — Ctrl+0 turn-focus collapse works")


if __name__ == "__main__":
    asyncio.run(main())
