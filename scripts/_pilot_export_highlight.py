"""Live pilot for Phase 9 P2 — /export html syntax highlighting.

Runs an end-to-end /export html through the real chat
machinery against a transcript containing python + shell
fenced blocks, then verifies the output HTML carries:

1. The Pygments stylesheet (`.body pre code .k {…}` anchors)
2. Highlighted keyword spans (`<span class="k">def</span>`)
3. Language classes (`class="language-python"`) on each fenced
   block.
4. The file is a single standalone HTML — no external assets,
   no missing `<style>` for the highlight classes.
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
        out_path = Path(td) / "transcript.html"

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            screen._post_line("user", "show me a hello world")
            screen._post_line(
                "assistant",
                (
                    "Here's Python:\n\n"
                    "```python\n"
                    "def hello(name):\n"
                    "    return f'Hi {name}'\n"
                    "```\n\n"
                    "And shell:\n\n"
                    "```bash\n"
                    "echo 'hi'\n"
                    "```\n"
                ),
            )
            await pilot.pause()
            inp = screen.query_one("#chat-input", Input)
            inp.value = f"/export html {out_path}"
            await inp.action_submit()
            await pilot.pause()

        assert out_path.exists(), f"export didn't write to {out_path}"
        body = out_path.read_text(encoding="utf-8")
        print(f"[1] file written: {out_path} ({len(body)} chars)")

        # Pygments stylesheet present
        assert ".body pre code .k" in body, "missing Pygments CSS rule"
        print("[2] OK — Pygments stylesheet rules inlined")

        # Both languages tagged
        assert 'class="language-python"' in body
        assert 'class="language-bash"' in body
        print("[3] OK — both fenced blocks tagged with language class")

        # Highlighted keyword span
        assert 'class="k">def' in body
        print("[4] OK — Python keyword 'def' wrapped in highlight span")

        # Function name styled
        assert 'class="nf">hello' in body
        print("[5] OK — function-name 'hello' carries Pygments span class")

        # No external assets — single-file standalone export
        assert "<link rel=\"stylesheet\"" not in body
        assert "<script src=" not in body
        print("[6] OK — single-file standalone (no external assets)")

        print("=" * 60)
        print("ALL CHECKS PASSED — /export html ships highlighted code")


if __name__ == "__main__":
    asyncio.run(main())
