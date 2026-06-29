"""Live pilot for chat input enhancements:

* ``@`` opens a file suggestion popup
* ``/`` opens a command suggestion popup
* ↑ / ↓ navigate the popup when open, recall prompt history
  when the popup is closed
* Tab commits the highlighted row
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App  # noqa: E402
from textual.widgets import Input, Static  # noqa: E402

from care.screens.chat import ChatScreen  # noqa: E402


class _Host(App):
    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        os.environ["CARE_CHAT__THEME_SIDECAR"] = str(td_path / "theme.txt")
        os.environ["CARE_CHAT__TUTORIAL_SIDECAR"] = str(td_path / "tut.json")
        os.environ["CARE_CHAT__SESSION_LOG_DIR"] = str(td_path / "sess")
        os.environ["CARE_CHAT__BRANCHES_DIR"] = str(td_path / "branches")

        # Lay down a tiny file tree so the file index has
        # something to surface. A sibling at the parent level
        # exercises `@../` navigation.
        (td_path / "alpha.py").write_text("# alpha\n")
        (td_path / "beta.md").write_text("# beta\n")
        (td_path / "src").mkdir()
        (td_path / "src" / "gamma.ts").write_text("// gamma\n")
        (td_path / "sibling.txt").write_text("hi\n")
        inner = td_path / "project"
        inner.mkdir()
        (inner / "local.py").write_text("# local\n")

        os.chdir(inner)

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            inp = screen.query_one("#chat-input", Input)
            row = screen.query_one("#chat-autocomplete-row", Static)

            # 1. Slash popup
            inp.value = "/he"
            inp.cursor_position = 3
            await pilot.pause()
            assert screen._autocomplete_open, "slash popup should be open"
            assert screen._autocomplete_kind == "slash"
            rendered = str(row.render())
            print(f"[1] /he popup rows:\n{rendered}")
            assert "► /help" in rendered

            # 2. Tab completes the top slash match
            screen.action_slash_autocomplete()
            await pilot.pause()
            print(f"[2] after Tab: input value = {inp.value!r}")
            assert inp.value == "/help "
            assert not screen._autocomplete_open

            # 3. @ popup
            inp.value = ""
            inp.cursor_position = 0
            await pilot.pause()
            inp.value = "@loc"
            inp.cursor_position = 4
            await pilot.pause()
            assert screen._autocomplete_open, "@ popup should be open"
            assert screen._autocomplete_kind == "file"
            rendered = str(row.render())
            print(f"[3] @loc popup rows:\n{rendered}")
            assert "local.py" in rendered

            # 4. Tab completes the selected file path
            screen.action_slash_autocomplete()
            await pilot.pause()
            print(f"[4] after Tab: input value = {inp.value!r}")
            assert inp.value == "@local.py "

            # 4b. `@../` re-roots to the parent directory.
            inp.value = "@../sib"
            inp.cursor_position = len(inp.value)
            await pilot.pause()
            rendered = str(row.render())
            print(f"[4b] @../sib popup rows:\n{rendered}")
            assert "@../sibling.txt" in rendered
            screen.action_slash_autocomplete()
            await pilot.pause()
            print(f"[4b] after Tab: input value = {inp.value!r}")
            assert inp.value == "@../sibling.txt "

            # 5. Up/Down move selection while popup is open
            inp.value = "/"
            inp.cursor_position = 1
            await pilot.pause()
            assert screen._autocomplete_open
            first = screen._autocomplete_matches[0]
            screen.action_recall_next()
            await pilot.pause()
            second = screen._autocomplete_matches[
                screen._autocomplete_selected
            ]
            print(f"[5] after ↓: marker on {second!r} (was {first!r})")
            assert second != first
            rendered = str(row.render())
            assert f"► /{second}" in rendered

            # 6a. Ctrl+C copies any selected text through the
            # platform-aware clipboard helper.
            from unittest.mock import patch
            screen.get_selected_text = lambda: "highlighted body"
            with patch(
                "care.runtime.clipboard.copy_text",
                return_value=True,
            ) as fake:
                screen.action_copy_text()
                await pilot.pause()
            assert fake.called, "copy_text not invoked"
            args, _ = fake.call_args
            print(f"[6a] copy_text({args[1]!r}) — clipboard write OK")
            assert args[1] == "highlighted body"

            # Reset selection for the next checks.
            screen.get_selected_text = lambda: None

            # 6. History recall when popup is closed
            inp.value = ""
            await pilot.pause()
            inp.value = "/help"
            await inp.action_submit()
            await pilot.pause()
            inp.value = "summarise the docs"
            await inp.action_submit()
            await pilot.pause()
            assert not screen._autocomplete_open
            screen.action_recall_prev()
            print(f"[6] ↑ #1: {inp.value!r}")
            assert inp.value == "summarise the docs"
            screen.action_recall_prev()
            print(f"    ↑ #2: {inp.value!r}")
            assert inp.value == "/help"
            screen.action_recall_next()
            screen.action_recall_next()
            print(f"    ↓ ↓ : {inp.value!r}")
            assert inp.value == ""

        print("=" * 60)
        print("ALL CHECKS PASSED — chat input suggestions work")


if __name__ == "__main__":
    asyncio.run(main())
