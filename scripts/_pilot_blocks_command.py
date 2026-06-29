"""Live pilot for Phase 9 P2 — /blocks per-block code-block actions.

End-to-end:
1. Post a multi-block assistant reply (python + bash) plus a
   system line containing a third block.
2. Run `/blocks` — confirm the listing carries 3 indexed rows
   with the right language tags.
3. Run `/blocks copy 2` — confirm the bash block lands in the
   clipboard stub.
4. Run `/blocks save 3 <path>` — confirm the third block is
   written to disk verbatim.
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

import care.runtime.clipboard as _clipboard  # noqa: E402
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
        out_path = Path(td) / "block3.txt"

        # Stub the clipboard so we can verify the copy payload
        # without depending on a real pasteboard.
        copied: list[str] = []
        original = _clipboard.copy_text

        def _fake_copy(_app, body):
            copied.append(body)
            return True

        _clipboard.copy_text = _fake_copy

        try:
            app = _Host()
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen_stack[-1]
                screen._post_line(
                    "assistant",
                    "Two snippets:\n"
                    "```python\nprint('alpha')\n```\n\n"
                    "```bash\necho beta\n```",
                )
                screen._post_line(
                    "system",
                    "Plus this:\n```sql\nSELECT 1;\n```",
                )
                await pilot.pause()
                inp = screen.query_one("#chat-input", Input)

                # 1. List
                inp.value = "/blocks"
                await inp.action_submit()
                await pilot.pause()
                listing_lines = [
                    line.text for line in screen._lines
                    if line.role == "system"
                    and "Code blocks in the transcript" in line.text
                ]
                assert listing_lines, "no listing line found"
                listing = listing_lines[-1]
                print("[1] listing rendered:")
                for raw in listing.splitlines():
                    print(f"    {raw}")
                assert "python" in listing
                assert "bash" in listing
                assert "sql" in listing
                assert "1." in listing
                assert "2." in listing
                assert "3." in listing
                print("[1] OK — 3 indexed blocks across roles\n")

                # 2. Copy
                inp.value = "/blocks copy 2"
                await inp.action_submit()
                await pilot.pause()
                print(f"[2] clipboard receives: {copied!r}")
                assert copied == ["echo beta"]
                print("[2] OK — /blocks copy 2 copied the bash block\n")

                # 3. Save
                inp.value = f"/blocks save 3 {out_path}"
                await inp.action_submit()
                await pilot.pause()
                print(f"[3] file written? {out_path.exists()}")
                assert out_path.exists()
                body = out_path.read_text(encoding="utf-8")
                print(f"[3] file body = {body!r}")
                assert body == "SELECT 1;"
                print("[3] OK — /blocks save 3 wrote the SQL block to disk\n")
        finally:
            _clipboard.copy_text = original

        print("=" * 60)
        print("ALL CHECKS PASSED — /blocks lists + copies + saves any block")


if __name__ == "__main__":
    asyncio.run(main())
