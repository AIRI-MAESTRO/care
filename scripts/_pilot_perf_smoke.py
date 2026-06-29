"""Live pilot for the chat perf/smoothness work (TODO §Perf P-9).

A headless regression guard for P-1…P-4. Drives a real ``ChatScreen`` under
Textual's pilot and asserts the structural invariants the optimizations rely
on:

1. **P-1 — bounded transcript.** Post far more lines than the render cap and
   confirm the number of *mounted* widgets stays ≤ ``_MAX_RENDERED_LINES``
   (the full ``_lines`` model stays complete).
2. **P-2 — cheap toggles.** Compact-mode toggle and the search-highlight pass
   each complete under a wall-clock budget even on a long transcript (they no
   longer do a per-line ``query_one``).
3. **P-4 — search still resolves.** With motion off the debounced search runs
   synchronously, so matches are present right after the call.

Run: ``uv run scripts/_pilot_perf_smoke.py``
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Deterministic + motion-off so we measure structure, not tweens (and the
# debounced search runs synchronously). Must be set before textual imports.
os.environ.setdefault("TEXTUAL_ANIMATIONS", "none")

from textual.app import App  # noqa: E402

from care.screens.chat import ChatScreen  # noqa: E402

LINES = 1000
COMPACT_BUDGET_S = 3.0
SEARCH_BUDGET_S = 2.0


class _Host(App):
    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        os.environ["CARE_CHAT__THEME_SIDECAR"] = str(Path(td) / "theme.txt")
        os.environ["CARE_CHAT__TUTORIAL_SIDECAR"] = str(Path(td) / "tut.json")
        os.environ["CARE_CHAT__SESSION_LOG_DIR"] = str(Path(td) / "sess")

        app = _Host()
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            cap = ChatScreen._MAX_RENDERED_LINES

            # 1) Post far more lines than the cap (mix roles so Markdown +
            #    Static widgets both participate). One sentinel needle so the
            #    search has something to find.
            t0 = time.perf_counter()
            for i in range(LINES):
                role = ("assistant", "tool", "user", "system")[i % 4]
                tag = " NEEDLE" if i % 250 == 0 else ""
                screen._post_line(role, f"perf line {i}{tag}")
            await pilot.pause()
            post_dt = time.perf_counter() - t0
            print(f"[1] posted {LINES} lines in {post_dt:.2f}s")

            # P-1 — mounted widget count is bounded; the model keeps them all.
            transcript = screen.query_one("#chat-transcript")
            mounted = len(transcript.children)
            model = len(screen._lines)
            print(f"[1] mounted widgets = {mounted} (cap {cap}); "
                  f"_lines model = {model}")
            assert mounted <= cap, (
                f"P-1 regression: {mounted} mounted widgets > cap {cap}"
            )
            assert model >= LINES, (
                f"model should retain every line, got {model} < {LINES}"
            )
            print("[1] OK — transcript widget tree is bounded\n")

            # 2) P-2 — compact-mode toggle over a long transcript is cheap.
            t0 = time.perf_counter()
            screen.action_toggle_compact_mode()
            await pilot.pause()
            compact_dt = time.perf_counter() - t0
            print(f"[2] compact-mode toggle took {compact_dt:.3f}s "
                  f"(budget {COMPACT_BUDGET_S}s)")
            assert compact_dt < COMPACT_BUDGET_S, (
                f"P-2 regression: compact toggle {compact_dt:.2f}s "
                f"≥ budget {COMPACT_BUDGET_S}s"
            )
            screen.action_toggle_compact_mode()  # restore
            await pilot.pause()
            print("[2] OK — compact toggle within budget\n")

            # 3) P-2/P-4 — search-highlight pass is cheap and resolves
            #    synchronously under reduced motion.
            t0 = time.perf_counter()
            screen._apply_search_query("NEEDLE")
            await pilot.pause()
            search_dt = time.perf_counter() - t0
            matches = len(screen._search_matches)
            print(f"[3] search found {matches} matches in {search_dt:.3f}s "
                  f"(budget {SEARCH_BUDGET_S}s)")
            assert matches > 0, "P-4 regression: synchronous search found nothing"
            assert search_dt < SEARCH_BUDGET_S, (
                f"P-2 regression: search {search_dt:.2f}s ≥ budget "
                f"{SEARCH_BUDGET_S}s"
            )
            print("[3] OK — search within budget\n")

        print("=" * 60)
        print("ALL CHECKS PASSED — transcript stays bounded + toggles cheap")


if __name__ == "__main__":
    asyncio.run(main())
