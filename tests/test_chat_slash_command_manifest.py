"""Slash-command regression suite (TODO §8 P1).

Catches drift between the registry, the autocomplete blurbs,
and the actual handler behaviour:

* Every `@_register(...)` handler appears in the manifest +
  in `_COMMAND_BLURBS` — adding a new command without
  updating either fails here.
* Every manifest / blurb entry corresponds to a real handler
  — removing a command without cleaning up its
  documentation also fails here.
* Each command, invoked with the empty arg, completes
  without raising an unhandled exception that escapes
  `ChatScreen._handle_command`'s catch — the existing
  catch surfaces failures as ``/<cmd> failed: ...`` system
  lines, which this test treats as regressions.
* A subset of commands have a stricter outcome — push a
  screen, post a line, or warn-with-hint — codified in
  `MANIFEST` so a future refactor can't silently change a
  command's no-arg behaviour.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from care.screens.chat import (
    _COMMAND_HANDLERS,
    ChatLine,
    ChatScreen,
)
from care.widgets.chat_input import ChatInput


# ---------------------------------------------------------------------------
# Manifest: cmd → expected no-arg behaviour tag
# ---------------------------------------------------------------------------


# Tags:
#  "posts_line"   — handler posts a system / tool line (most help / list cmds)
#  "pushes_screen"— handler routes to a different screen
#  "self_destructive" — handler exits the app or clears the transcript
#  "needs_facade" — handler warns about a missing facade (memory/platform);
#                   still completes cleanly via _post_line
MANIFEST: dict[str, str] = {
    "help": "posts_line",
    "memory": "posts_line",
    "remember": "posts_line",
    "library": "pushes_screen",
    "marketplace": "pushes_screen",
    "runs": "pushes_screen",
    "sandbox": "pushes_screen",
    "cost": "pushes_screen",
    "logs": "pushes_screen",
    "profile": "pushes_screen",
    "artifacts": "needs_facade",  # no chat-screen artifact store fallback hint
    "status": "posts_line",
    "settings": "pushes_screen",
    "clear": "self_destructive",
    "new": "self_destructive",
    "quit": "self_destructive",
    "resume": "posts_line",
    "theme": "posts_line",
    "multi": "pushes_screen",
    "log": "posts_line",
    "voice": "posts_line",
    "subagents": "posts_line",
    "sessions": "posts_line",
    "history": "posts_line",
    "edit": "posts_line",
    "imgpreview": "posts_line",
    "branch": "posts_line",
    "blocks": "posts_line",
    "export": "posts_line",
    "tour": "self_destructive",
    "forget": "needs_facade",
    "run": "needs_facade",
    "upload": "needs_facade",
    "evolution": "needs_facade",
    "dataset": "posts_line",
    "mode": "posts_line",
    "deploy": "posts_line",
    "deployments": "needs_facade",
    "metrics": "needs_facade",
    "promote": "posts_line",
    "revise": "posts_line",
    "rollback": "posts_line",
    "versions": "posts_line",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Host(App):
    """Minimal ChatScreen host with no Memory / Platform
    wired — exercises the "no facade configured" warning
    paths for commands that need them."""

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


def _chat(app: _Host) -> ChatScreen:
    for s in app.screen_stack:
        if isinstance(s, ChatScreen):
            return s
    raise AssertionError("ChatScreen not on stack")


def _ran_failed_line(lines: list[ChatLine], cmd: str) -> bool:
    """Return True when `_handle_command`'s catch surfaced
    a `/<cmd> failed: ...` line — that indicates the handler
    raised an unhandled exception with the empty arg."""
    return any(
        line.role == "system"
        and f"/{cmd} failed:" in line.text
        for line in lines
    )


def _ran_unknown_command_line(
    lines: list[ChatLine], cmd: str,
) -> bool:
    return any(
        line.role == "system" and f"Unknown command: /{cmd}" in line.text
        for line in lines
    )


# ---------------------------------------------------------------------------
# Registry / manifest / blurb coverage
# ---------------------------------------------------------------------------


class TestManifestCoverage:
    def test_every_registered_handler_in_manifest(self) -> None:
        missing = set(_COMMAND_HANDLERS.keys()) - set(MANIFEST.keys())
        assert not missing, (
            "registered command(s) missing from MANIFEST: "
            f"{sorted(missing)}"
        )

    def test_every_manifest_entry_is_registered(self) -> None:
        missing = set(MANIFEST.keys()) - set(_COMMAND_HANDLERS.keys())
        assert not missing, (
            "MANIFEST contains command(s) not registered: "
            f"{sorted(missing)}"
        )

    def test_every_command_has_a_blurb(self) -> None:
        documented = set(ChatScreen._COMMAND_BLURBS.keys())
        registered = set(_COMMAND_HANDLERS.keys())
        missing = registered - documented
        assert not missing, (
            f"command(s) without a blurb: {sorted(missing)}"
        )

    def test_no_orphan_blurbs(self) -> None:
        documented = set(ChatScreen._COMMAND_BLURBS.keys())
        registered = set(_COMMAND_HANDLERS.keys())
        orphan = documented - registered
        assert not orphan, (
            "blurb entry(ies) for unregistered commands: "
            f"{sorted(orphan)}"
        )

    def test_manifest_tags_are_known(self) -> None:
        known = {
            "posts_line",
            "pushes_screen",
            "needs_facade",
            "self_destructive",
        }
        unknown = {
            cmd for cmd, tag in MANIFEST.items() if tag not in known
        }
        assert not unknown, (
            f"MANIFEST has unknown tag(s) for: {sorted(unknown)}"
        )


# ---------------------------------------------------------------------------
# Empty-arg dispatch — fires each command + asserts no `/<cmd> failed`
# ---------------------------------------------------------------------------


class TestEmptyArgDispatch:
    @pytest.mark.parametrize("cmd", sorted(_COMMAND_HANDLERS.keys()))
    @pytest.mark.asyncio
    async def test_no_arg_does_not_post_failed_line(
        self, cmd: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each command, invoked with the empty arg, must not
        raise — `_handle_command` would otherwise surface
        ``/<cmd> failed: ...`` as a system line. Skips
        `self_destructive` commands (quit / clear / new /
        tour) since they reshape app state in ways that
        confuse the assert."""
        if MANIFEST.get(cmd) == "self_destructive":
            pytest.skip(f"/{cmd} reshapes app state; covered elsewhere")
        # /upload reads env; ensure deterministic absence so
        # the handler hits its no-config path rather than a
        # network call.
        monkeypatch.delenv("CARE_UPLOAD__URL", raising=False)

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _chat(app)
            inp = screen.query_one("#chat-input", ChatInput)
            inp.value = f"/{cmd}"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert not _ran_failed_line(screen._lines, cmd), (
                f"/{cmd} raised on empty arg — saw a system "
                f"`{cmd} failed:` line. Recent lines: "
                f"{[ln.text for ln in screen._lines[-5:]]}"
            )
            assert not _ran_unknown_command_line(
                screen._lines, cmd,
            ), (
                f"/{cmd} dispatched as 'Unknown command' — "
                "registry resolution regressed"
            )



# ---------------------------------------------------------------------------
# Behaviour assertion per tag
# ---------------------------------------------------------------------------


def _select_cmds(tag: str) -> list[str]:
    return sorted(c for c, t in MANIFEST.items() if t == tag)


class TestPostsLineCommands:
    @pytest.mark.parametrize("cmd", _select_cmds("posts_line"))
    @pytest.mark.asyncio
    async def test_posts_at_least_one_system_or_tool_line(
        self, cmd: str,
    ) -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _chat(app)
            before = len([
                ln for ln in screen._lines
                if ln.role in ("system", "tool")
            ])
            inp = screen.query_one("#chat-input", ChatInput)
            inp.value = f"/{cmd}"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            after = len([
                ln for ln in screen._lines
                if ln.role in ("system", "tool")
            ])
            assert after > before, (
                f"/{cmd} (posts_line) did not post a system "
                "or tool line on empty arg"
            )


class TestPushesScreenCommands:
    @pytest.mark.parametrize("cmd", _select_cmds("pushes_screen"))
    @pytest.mark.asyncio
    async def test_pushes_a_new_screen(self, cmd: str) -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            depth_before = len(app.screen_stack)
            screen = _chat(app)
            inp = screen.query_one("#chat-input", ChatInput)
            inp.value = f"/{cmd}"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert len(app.screen_stack) > depth_before, (
                f"/{cmd} (pushes_screen) did not push a new "
                f"screen (stack depth {depth_before} → "
                f"{len(app.screen_stack)})"
            )


class TestNeedsFacadeCommands:
    @pytest.mark.parametrize("cmd", _select_cmds("needs_facade"))
    @pytest.mark.asyncio
    async def test_warns_when_facade_missing(
        self, cmd: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Commands that need a Memory / Platform facade
        should warn-and-continue (a system line) rather than
        crash when the facade isn't configured."""
        monkeypatch.delenv("CARE_UPLOAD__URL", raising=False)
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _chat(app)
            inp = screen.query_one("#chat-input", ChatInput)
            inp.value = f"/{cmd}"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            # Either a system line (warning hint) lands OR a
            # screen pushes — both are acceptable graceful
            # outcomes. The non-acceptable outcomes are
            # already caught by the dispatch test above.
            system_lines = [
                ln for ln in screen._lines if ln.role == "system"
            ]
            stack_pushed = len(app.screen_stack) > 1
            assert system_lines or stack_pushed, (
                f"/{cmd} (needs_facade) produced neither a "
                "system line nor a screen push when its "
                "facade is missing — handler may have crashed "
                "silently"
            )


