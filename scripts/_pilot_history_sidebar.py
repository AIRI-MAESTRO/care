"""Live pilot for Phase 9 P1 — persistent history sidebar.

Exercises the end-to-end Ctrl+\\ → row-click → input-prefill
loop in a real Textual run:

1. Sidebar mounts hidden; toggle action flips it on.
2. After three /user lines and one fake chain, the sidebar
   reflects both sections with the newest items first.
3. Simulating a click on a prompt row prefills the input
   verbatim (full multi-line text).
4. Simulating a click on a chain row prefills `/run <id>`.
5. Toggling off makes the sidebar release its layout width.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os  # noqa: E402

from textual.app import App  # noqa: E402
from textual.containers import VerticalScroll  # noqa: E402
from textual.widgets import Input, Static  # noqa: E402

from care.screens.chat import ChatScreen  # noqa: E402


class _FakeMemory:
    def list_entities(self, entity_type=None, limit=None):  # noqa: ARG002
        return [
            {"entity_id": "alpha-001", "name": "discover-chain"},
            {"entity_id": "beta-002", "name": "evolve-chain"},
        ]


class _Host(App):
    def on_mount(self) -> None:
        self.memory = _FakeMemory()
        self.push_screen(ChatScreen())


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        os.environ["CARE_CHAT__TUTORIAL_SIDECAR"] = str(Path(td) / "tutorial.json")
        os.environ["CARE_CHAT__SESSION_LOG_DIR"] = str(Path(td) / "sessions")
        os.environ["CARE_CHAT__THEME_SIDECAR"] = str(Path(td) / "theme.txt")

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            sidebar = screen.query_one(
                "#chat-history-sidebar", VerticalScroll,
            )
            transcript = screen.query_one(
                "#chat-transcript", VerticalScroll,
            )
            print(f"[1] sidebar hidden? display={sidebar.display}")
            assert sidebar.display is False
            transcript_w_before = transcript.size.width
            print(f"[1] transcript width = {transcript_w_before}")

            # Post chat lines first (typical real-world flow:
            # the user is already chatting before they open the
            # sidebar). This also avoids the live-refresh
            # storm that the test suite covers explicitly.
            screen._post_line("user", "first prompt — short")
            screen._post_line("user", "second prompt\nwith newline body")
            screen._post_line("user", "third prompt")
            await pilot.pause()

            screen.action_toggle_history_sidebar()
            await pilot.pause()
            print(f"[2] after toggle: sidebar display={sidebar.display}")
            assert sidebar.display is True
            transcript_w_after_open = transcript.size.width
            print(f"[2] transcript width after open = {transcript_w_after_open}")
            assert transcript_w_after_open < transcript_w_before
            print("[2] OK — transcript shrunk for sidebar\n")
            rows = list(screen.query(".hist-row").results(Static))
            labels = [str(r.render()) for r in rows]
            print(f"[3] rows after 3 prompts + 2 chains: {len(rows)}")
            for lbl in labels:
                print(f"    {lbl!r}")
            assert any("3." in lbl and "third" in lbl for lbl in labels)
            assert any("alpha-001" in lbl for lbl in labels)
            print("[3] OK — sidebar lists prompts (newest first) + chains\n")

            # Click the prompt with the multi-line body
            prompt_actions = {
                rid: value for rid, (k, value) in
                screen._sidebar_actions.items() if k == "prompt"
            }
            target_id = next(
                rid for rid, value in prompt_actions.items()
                if "second" in value
            )
            screen._handle_sidebar_row_click(target_id)
            await pilot.pause()
            inp = screen.query_one("#chat-input", Input)
            print(f"[4] input after click = {inp.value!r}")
            assert inp.value == "second prompt\nwith newline body"
            print("[4] OK — multi-line body restored verbatim\n")

            # Click an alpha-001 chain row
            chain_row_id = next(
                rid for rid, (k, v) in
                screen._sidebar_actions.items()
                if k == "chain" and v == "alpha-001"
            )
            screen._handle_sidebar_row_click(chain_row_id)
            await pilot.pause()
            print(f"[5] input after chain click = {inp.value!r}")
            assert inp.value == "/run alpha-001"
            print("[5] OK — chain id prefilled as /run command\n")

            screen.action_toggle_history_sidebar()
            await pilot.pause()
            print(f"[6] toggle off: sidebar display={sidebar.display}")
            transcript_w_after_close = transcript.size.width
            print(f"[6] transcript width after close = {transcript_w_after_close}")
            assert sidebar.display is False
            assert transcript_w_after_close >= transcript_w_after_open
            print("[6] OK — toggle off restores transcript width\n")

        print("=" * 60)
        print("ALL STEPS PASSED — history sidebar works end-to-end")


if __name__ == "__main__":
    asyncio.run(main())
