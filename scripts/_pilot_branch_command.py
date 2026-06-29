"""Live pilot for Phase 9 P3 — /branch transcript checkpoints.

End-to-end against a real Textual session:
1. Build transcript A, save as `experiment-a`.
2. Diverge transcript (add a fresh user line that isn't in A).
3. Save the divergent state as `experiment-b`.
4. `/branch list` shows both.
5. `/branch switch experiment-a` rehydrates A — the divergent
   line is gone.
6. `/branch delete experiment-b` removes the sidecar.
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


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        os.environ["CARE_CHAT__THEME_SIDECAR"] = str(Path(td) / "theme.txt")
        os.environ["CARE_CHAT__TUTORIAL_SIDECAR"] = str(
            Path(td) / "tut.json"
        )
        os.environ["CARE_CHAT__SESSION_LOG_DIR"] = str(Path(td) / "sess")
        branches_dir = Path(td) / "branches"
        os.environ["CARE_CHAT__BRANCHES_DIR"] = str(branches_dir)

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            inp = screen.query_one("#chat-input", Input)

            # 1. Build transcript A
            screen._post_line("user", "transcript A prompt")
            screen._post_line("assistant", "transcript A reply")
            await pilot.pause()
            inp.value = "/branch experiment-a"
            await inp.action_submit()
            await pilot.pause()
            sidecar_a = branches_dir / "experiment-a.json"
            print(f"[1] saved {sidecar_a.name}: exists = {sidecar_a.exists()}")
            assert sidecar_a.exists()

            # 2. Diverge
            screen._post_line("user", "diverged prompt")
            await pilot.pause()
            print(f"[2] diverged transcript size = {len(screen._lines)}")

            # 3. Save divergent
            inp.value = "/branch experiment-b"
            await inp.action_submit()
            await pilot.pause()
            sidecar_b = branches_dir / "experiment-b.json"
            print(f"[3] saved {sidecar_b.name}: exists = {sidecar_b.exists()}")
            assert sidecar_b.exists()

            # 4. List
            inp.value = "/branch list"
            await inp.action_submit()
            await pilot.pause()
            listings = [
                line.text for line in screen._lines
                if line.role == "system"
                and "Saved branches" in line.text
            ]
            assert listings
            print("[4] listing body:")
            for raw in listings[-1].splitlines():
                print(f"    {raw}")
            assert "experiment-a" in listings[-1]
            assert "experiment-b" in listings[-1]

            # 5. Switch back to A
            inp.value = "/branch switch experiment-a"
            await inp.action_submit()
            await pilot.pause()
            texts_after = [line.text for line in screen._lines]
            print(f"[5] post-switch transcript size = {len(screen._lines)}")
            assert any("transcript A prompt" in t for t in texts_after)
            assert any("transcript A reply" in t for t in texts_after)
            assert not any("diverged prompt" in t for t in texts_after)
            assert any("↳ branched from" in t for t in texts_after)
            print("[5] OK — switched, divergent prompt is gone\n")

            # 6. Delete experiment-b
            inp.value = "/branch delete experiment-b"
            await inp.action_submit()
            await pilot.pause()
            print(f"[6] sidecar_b after delete: exists = {sidecar_b.exists()}")
            assert not sidecar_b.exists()
            # sidecar_a still there.
            assert sidecar_a.exists()
            print("[6] OK — delete removed only the target\n")

        print("=" * 60)
        print("ALL CHECKS PASSED — /branch save / list / switch / delete")


if __name__ == "__main__":
    asyncio.run(main())
