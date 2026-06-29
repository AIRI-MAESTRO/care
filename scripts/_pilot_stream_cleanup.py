"""Live pilot for Phase 9 P3 — stream-preview cleanup on chain
completion.

End-to-end against a real Textual session:
1. Spawn two stream previews (simulating a multi-step
   iteration), confirm widget tracking accrues.
2. Post a normal tool line (not a preview), confirm it's
   distinct from preview widgets.
3. Run cleanup, confirm preview widgets are unmounted +
   removed from `_lines`, but the regular tool line
   survives.
4. Call cleanup again (idempotency), confirm it's a no-op.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App  # noqa: E402
from textual.css.query import NoMatches  # noqa: E402

from care.runtime.carl_streamer import LlmChunk, StepCompleted  # noqa: E402
from care.screens.chat import ChatScreen  # noqa: E402


class _Host(App):
    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        os.environ["CARE_CHAT__THEME_SIDECAR"] = str(Path(td) / "theme.txt")
        os.environ["CARE_CHAT__TUTORIAL_SIDECAR"] = str(Path(td) / "tut.json")
        os.environ["CARE_CHAT__SESSION_LOG_DIR"] = str(Path(td) / "sess")
        os.environ["CARE_CHAT__BRANCHES_DIR"] = str(Path(td) / "branches")

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]

            # 1. Two streams across a step boundary
            lines_initial = len(screen._lines)
            screen.on_llm_chunk(LlmChunk(chunk="step 1 streaming"))
            screen.on_step_completed(StepCompleted(result=None))
            screen.on_llm_chunk(LlmChunk(chunk="step 2 streaming"))
            await pilot.pause()
            tracked = list(screen._iteration_stream_widget_ids)
            print(f"[1] tracked preview widgets: {tracked}")
            assert len(tracked) == 2
            # Both widgets are present in the DOM.
            for wid in tracked:
                widget = screen.query_one(f"#{wid}")
                print(f"    found widget {wid}: {widget.__class__.__name__}")

            # 2. Post a normal tool line (NOT a stream preview)
            screen._post_line("tool", "▶ post-cleanup tool ping")
            await pilot.pause()
            normal_tool_widget_id = f"chat-line-{screen._line_counter}"
            print(f"[2] normal tool line id = {normal_tool_widget_id}")
            assert normal_tool_widget_id not in tracked

            # 3. Cleanup
            removed = screen._remove_iteration_stream_previews()
            await pilot.pause()
            print(f"[3] cleanup removed {removed} widgets")
            assert removed == 2
            for wid in tracked:
                try:
                    screen.query_one(f"#{wid}")
                    print(f"    WIDGET STILL THERE: {wid}")
                    raise SystemExit(1)
                except NoMatches:
                    print(f"    confirmed gone: {wid}")
            # The normal tool line is still mounted.
            screen.query_one(f"#{normal_tool_widget_id}")
            print(f"    survivor: {normal_tool_widget_id}")
            print(f"[3] _lines length now = {len(screen._lines)} "
                  f"(initial {lines_initial} + 1 tool ping)")
            assert len(screen._lines) == lines_initial + 1

            # 4. Idempotency
            second = screen._remove_iteration_stream_previews()
            print(f"[4] second cleanup call removed {second} (should be 0)")
            assert second == 0

        print("=" * 60)
        print("ALL CHECKS PASSED — stream preview cleanup works")


if __name__ == "__main__":
    asyncio.run(main())
