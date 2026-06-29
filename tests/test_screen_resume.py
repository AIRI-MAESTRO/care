"""Pilot tests for ResumeModal + WelcomeScreen wiring
(TODO §1.1 P0.37).

Exercises:
* `WelcomeScreen.on_mount` calls `RunStateStore().load()`.
* Non-None snapshot → push `ResumeModal`.
* `Resume` dismisses with `action="resume"` and posts
  `WelcomeScreen.ResumeRequested(state)`.
* `Discard` calls `store.clear()` and dismisses with
  `action="discard"`.
* `Escape` dismisses without clearing the store.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from care.runtime.run_state import RunState, RunStateStore
from care.screens.resume import ResumeModal, ResumeResult
from care.screens.welcome import WelcomeScreen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path, *, state: RunState | None = None) -> RunStateStore:
    store = RunStateStore(path=tmp_path / "run_state.json")
    if state is not None:
        store.save(state)
    return store


def _seed_state() -> RunState:
    return RunState(
        run_id="abcdef1234",
        kind="mage_generation",
        label="Generate weather report",
        payload={"query": "weather"},
    )


# ---------------------------------------------------------------------------
# ResumeModal in isolation
# ---------------------------------------------------------------------------


class _ModalHost(App):
    def __init__(self, *, state: RunState, store: RunStateStore) -> None:
        super().__init__()
        self._state = state
        self._modal_store = store
        self.dismissed: list[ResumeResult] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(result):
            self.dismissed.append(result)

        self.push_screen(
            ResumeModal(self._state, store=self._modal_store),
            _on_dismiss,
        )


class TestResumeModal:
    @pytest.mark.asyncio
    async def test_resume_dismisses_with_resume_action(self, tmp_path):
        state = _seed_state()
        store = _make_store(tmp_path, state=state)
        app = _ModalHost(state=state, store=store)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ResumeModal)
            modal.query_one("#resume-btn-resume", Button).press()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed[0].action == "resume"
            assert app.dismissed[0].state == state
            # Store left intact (host re-primes from the
            # carried state; clearing is a separate gesture).
            assert store.load() == state

    @pytest.mark.asyncio
    async def test_discard_clears_store(self, tmp_path):
        state = _seed_state()
        store = _make_store(tmp_path, state=state)
        app = _ModalHost(state=state, store=store)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ResumeModal)
            modal.query_one("#resume-btn-discard", Button).press()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed[0].action == "discard"
            assert app.dismissed[0].state is None
            # Store cleared.
            assert store.load() is None

    @pytest.mark.asyncio
    async def test_escape_cancels_without_clearing(self, tmp_path):
        state = _seed_state()
        store = _make_store(tmp_path, state=state)
        app = _ModalHost(state=state, store=store)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ResumeModal)
            modal.action_cancel()
            for _ in range(3):
                await pilot.pause()
            assert app.dismissed[0].action == "cancel"
            # Store left intact — next launch offers the same
            # prompt again.
            assert store.load() == state


# ---------------------------------------------------------------------------
# WelcomeScreen integration
# ---------------------------------------------------------------------------


class _WelcomeHost(App):
    def __init__(self, *, store: RunStateStore) -> None:
        super().__init__()
        self._welcome_store = store
        self.resumed: list[RunState] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        # Long splash so the auto-route doesn't fire before the
        # test's pause.
        self.push_screen(
            WelcomeScreen(
                splash_seconds=5.0,
                run_state_store=self._welcome_store,
            ),
        )

    def on_welcome_screen_resume_requested(
        self, event: WelcomeScreen.ResumeRequested,
    ) -> None:
        if event.state is not None:
            self.resumed.append(event.state)


class TestWelcomeIntegration:
    @pytest.mark.asyncio
    async def test_no_stored_state_no_modal(self, tmp_path):
        store = _make_store(tmp_path)
        app = _WelcomeHost(store=store)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # WelcomeScreen on top, no modal pushed.
            screen = app.screen_stack[-1]
            assert isinstance(screen, WelcomeScreen)
            assert screen.resume_state is None

    @pytest.mark.asyncio
    async def test_stored_state_pushes_modal(self, tmp_path):
        state = _seed_state()
        store = _make_store(tmp_path, state=state)
        app = _WelcomeHost(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            assert isinstance(app.screen_stack[-1], ResumeModal)

    @pytest.mark.asyncio
    async def test_resume_posts_resume_requested(self, tmp_path):
        state = _seed_state()
        store = _make_store(tmp_path, state=state)
        app = _WelcomeHost(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ResumeModal)
            modal.action_resume()
            for _ in range(4):
                await pilot.pause()
            assert len(app.resumed) == 1
            assert app.resumed[0].run_id == state.run_id

    @pytest.mark.asyncio
    async def test_discard_clears_and_no_resume_posted(self, tmp_path):
        state = _seed_state()
        store = _make_store(tmp_path, state=state)
        app = _WelcomeHost(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ResumeModal)
            modal.action_discard()
            for _ in range(4):
                await pilot.pause()
            assert app.resumed == []
            assert store.load() is None
            # WelcomeScreen recorded the action.
            welcome = next(
                s for s in app.screen_stack
                if isinstance(s, WelcomeScreen)
            )
            assert welcome.resume_action == "discard"


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import ResumeModal as M
        from care.screens import ResumeResult as R

        assert M is ResumeModal
        assert R is ResumeResult
