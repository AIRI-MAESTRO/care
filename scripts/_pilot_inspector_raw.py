"""Live pilot for Phase 9 P2 — prompt-inspector raw-call capture.

End-to-end:
1. Feed LlmChunk events into the ChatScreen as if CARL were
   streaming an answer mid-iteration. Confirm the buffer
   accumulates and survives a StepCompleted reset (so
   multi-step iterations roll up correctly).
2. Capture provenance and confirm the buffer flows into the
   ``raw_response`` field.
3. Post an assistant line with that provenance, run
   ``action_inspect_last``, and confirm the fenced
   raw-response block lands in the rendered system line.
4. Confirm `_RAW_CAPTURE_MAX_CHARS` actually bounds a runaway
   stream (50KB → trimmed body with the ellipsis marker).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App  # noqa: E402

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

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]

            # 1. Multi-step stream → accumulated buffer
            screen.on_llm_chunk(LlmChunk(chunk="Thinking about "))
            screen.on_llm_chunk(LlmChunk(chunk="the problem…\n"))
            screen.on_step_completed(StepCompleted(result=None))
            screen.on_llm_chunk(LlmChunk(chunk="Answer: 42"))
            await pilot.pause()
            buffer_text = "".join(screen._iteration_raw_response)
            print(f"[1] iteration buffer = {buffer_text!r}")
            assert "Thinking about the problem" in buffer_text
            assert "Answer: 42" in buffer_text
            print("[1] OK — buffer survives StepCompleted reset\n")

            # 2. Capture provenance
            payload = screen._capture_provenance(
                iteration=1,
                started_at=0.0,
                tokens_before=None,
                token_split_before=None,
            )
            print(f"[2] provenance keys = {sorted(payload.keys())}")
            assert "raw_response" in payload
            assert "Answer: 42" in payload["raw_response"]
            print("[2] OK — raw_response landed in provenance\n")

            # 3. Inspector renders the fenced block
            screen._post_line(
                "assistant",
                "Final: 42",
                provenance=payload,
            )
            screen.action_inspect_last()
            await pilot.pause()
            inspector_line = next(
                (
                    line for line in screen._lines
                    if line.role == "system"
                    and "Prompt inspector" in line.text
                ),
                None,
            )
            assert inspector_line is not None, "no inspector line posted"
            print("[3] inspector line body:")
            for raw in inspector_line.text.splitlines():
                print(f"    {raw}")
            assert "raw response" in inspector_line.text
            assert "```text" in inspector_line.text
            assert "Answer: 42" in inspector_line.text
            print("\n[3] OK — inspector renders fenced raw-response block\n")

            # 4. Runaway stream truncation
            screen._iteration_raw_response = []
            screen.on_llm_chunk(LlmChunk(chunk="x" * 50_000))
            await pilot.pause()
            payload2 = screen._capture_provenance(
                iteration=2,
                started_at=0.0,
                tokens_before=None,
                token_split_before=None,
            )
            print(
                f"[4] raw_response length after 50KB stream = "
                f"{len(payload2['raw_response'])}",
            )
            assert len(payload2["raw_response"]) < 50_000
            assert "truncated" in payload2["raw_response"]
            print("[4] OK — runaway stream bounded with ellipsis marker\n")

        print("=" * 60)
        print("ALL CHECKS PASSED — prompt-inspector raw capture works")


if __name__ == "__main__":
    asyncio.run(main())
