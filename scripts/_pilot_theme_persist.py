"""Live pilot for Phase 9 P2 — /theme sidecar persistence.

Exercises the round-trip end-to-end in a real Textual run:

1. First Host boot — sidecar starts empty; /theme nord lands.
   Confirm app.theme == nord, sidecar contains "nord".
2. Second Host boot — same sidecar; nothing typed.
   Confirm app.theme == nord on mount (no /theme invocation).
3. Third Host boot — sidecar pre-populated with an unknown
   theme name; nothing typed. Confirm boot doesn't crash and
   app.theme is one of the registered themes.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Ensure the project root is on sys.path so the script runs
# from outside the test harness.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App  # noqa: E402
from textual.widgets import Input  # noqa: E402

from care.screens.chat import ChatScreen  # noqa: E402


class _Host(App):
    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


async def _step1_apply_persists(sidecar: Path) -> None:
    print(f"[step 1] sidecar starts {'present' if sidecar.exists() else 'absent'}")
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        inp = screen.query_one("#chat-input", Input)
        inp.value = "/theme nord"
        await inp.action_submit()
        await pilot.pause()
        active = app.theme
    print(f"[step 1] app.theme after /theme nord = {active!r}")
    print(f"[step 1] sidecar present? {sidecar.exists()}")
    if sidecar.exists():
        body = sidecar.read_text(encoding="utf-8")
        print(f"[step 1] sidecar body = {body!r}")
    assert active == "nord", f"expected nord, got {active!r}"
    assert sidecar.read_text(encoding="utf-8") == "nord"
    print("[step 1] OK — apply + persist round-trip\n")


async def _step2_boot_restores(sidecar: Path) -> None:
    assert sidecar.exists(), "sidecar should still be present from step 1"
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        active = app.theme
    print(f"[step 2] fresh boot picked up app.theme = {active!r}")
    assert active == "nord", f"expected nord on boot, got {active!r}"
    print("[step 2] OK — boot reads sidecar and applies\n")


async def _step3_unknown_theme_falls_back(sidecar: Path) -> None:
    sidecar.write_text("totally-fake-theme-xyz", encoding="utf-8")
    print("[step 3] sidecar overwritten with bogus theme name")
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        active = app.theme
        registered = screen._available_theme_names()
    print(f"[step 3] app.theme = {active!r}")
    print(f"[step 3] registered themes: {registered[:5]}{'…' if len(registered) > 5 else ''}")
    assert active in registered, "boot should have fallen back to a real theme"
    assert active != "totally-fake-theme-xyz"
    print("[step 3] OK — unknown theme tolerated, boot survived\n")


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        sidecar = Path(td) / "theme_preference.txt"
        os.environ["CARE_CHAT__THEME_SIDECAR"] = str(sidecar)
        # Tutorial sidecar separation so /tour doesn't pop the
        # welcome line into something unexpected.
        os.environ["CARE_CHAT__TUTORIAL_SIDECAR"] = str(Path(td) / "tutorial.json")
        os.environ["CARE_CHAT__SESSION_LOG_DIR"] = str(Path(td) / "sessions")
        await _step1_apply_persists(sidecar)
        await _step2_boot_restores(sidecar)
        await _step3_unknown_theme_falls_back(sidecar)
    print("=" * 60)
    print("ALL STEPS PASSED — /theme sidecar persistence works end-to-end")


if __name__ == "__main__":
    asyncio.run(main())
