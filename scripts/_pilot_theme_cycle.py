"""Live pilot for Phase 9 P2 — Alt+T theme cycle.

Exercises the cycle action end-to-end:

1. Capture current theme; call `action_cycle_theme` and
   confirm it lands on the next alphabetical theme.
2. Cycle a second time and confirm it advances another step.
3. Force the active theme to the LAST one in the list and
   cycle — confirm wrap-around to the first.
4. Confirm each cycle persists to the sidecar so the choice
   survives a restart.
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
        sidecar = Path(td) / "theme.txt"
        os.environ["CARE_CHAT__THEME_SIDECAR"] = str(sidecar)
        os.environ["CARE_CHAT__TUTORIAL_SIDECAR"] = str(Path(td) / "tut.json")
        os.environ["CARE_CHAT__SESSION_LOG_DIR"] = str(Path(td) / "sessions")

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            themes = screen._available_theme_names()
            print(f"[setup] {len(themes)} themes registered; first 3: {themes[:3]}")
            initial = app.theme
            print(f"[1] initial app.theme = {initial!r}")
            idx_initial = themes.index(initial)
            expected_after_one = themes[(idx_initial + 1) % len(themes)]

            screen.action_cycle_theme()
            await pilot.pause()
            print(f"[2] after 1st cycle: {app.theme!r} (expected {expected_after_one!r})")
            assert app.theme == expected_after_one

            screen.action_cycle_theme()
            await pilot.pause()
            expected_after_two = themes[(idx_initial + 2) % len(themes)]
            print(f"[3] after 2nd cycle: {app.theme!r} (expected {expected_after_two!r})")
            assert app.theme == expected_after_two

            # Sidecar tracks the latest pick.
            saved = sidecar.read_text(encoding="utf-8").strip()
            print(f"[4] sidecar contents = {saved!r}")
            assert saved == app.theme

            # Wrap-around: jump to the last and cycle.
            app.theme = themes[-1]
            await pilot.pause()
            print(f"[5] forced to last theme: {app.theme!r}")
            screen.action_cycle_theme()
            await pilot.pause()
            print(f"[6] after cycle from last: {app.theme!r} (expected {themes[0]!r})")
            assert app.theme == themes[0]

        print("=" * 60)
        print("ALL STEPS PASSED — Alt+T cycle, wrap-around, sidecar persistence")


if __name__ == "__main__":
    asyncio.run(main())
