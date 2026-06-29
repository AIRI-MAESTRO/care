"""Tests for `ChatScreen._record_local_run` (TODO §6 P1).

Drives the helper directly with stub chain + result + config
shapes so we don't need to spin up the full CARL pipeline.
The helper is the only seam the ad-hoc CARL completion path
calls to populate the `/runs` screen; broken plumbing here
would leave that screen empty even though chains executed.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App, ComposeResult

from care.runtime.local_run_history import (
    load_local_runs,
    runs_dir,
)
from care.runtime.user_paths import CARE_CACHE_DIR
from care.screens.chat import ChatScreen


class _Host(App):
    """Minimal host so `ChatScreen.app.config.mage.provider`
    resolves through the same plumbing the real recorder
    walks. We stub the config rather than pulling
    `CareConfig.load()` so the test doesn't depend on the
    user's real `.env`."""

    def __init__(self, *, provider: str = "openai"):
        super().__init__()
        self.config = SimpleNamespace(
            mage=SimpleNamespace(provider=provider),
        )

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


def _chat(app: _Host) -> ChatScreen:
    for s in app.screen_stack:
        if isinstance(s, ChatScreen):
            return s
    raise AssertionError("ChatScreen not on stack")


@pytest.fixture(autouse=True)
def _redirect_runs_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Point the local-run-history writer at a tmp cache so
    we don't pollute the user's real `~/.cache/care/runs/`.

    The recorder reads `CARE_CACHE_DIR` lazily through
    `record_local_run`'s default cache_dir → `runs_dir(...)`,
    which falls through to the module-level constant on
    `care.runtime.user_paths`. Monkeypatching there is enough
    — the chat code path doesn't construct its own override."""
    from care.runtime import local_run_history as lrh
    from care.runtime import user_paths as up

    monkeypatch.setattr(up, "CARE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(lrh, "CARE_CACHE_DIR", tmp_path)
    yield


# ---------------------------------------------------------------------------
# Recording — success / failure paths
# ---------------------------------------------------------------------------


class TestRecordLocalRun:
    @pytest.mark.asyncio
    async def test_success_writes_row(
        self, tmp_path: Path,
    ):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = _chat(app)
            chain = SimpleNamespace(
                entity_id="chain-A",
                name="Forecaster",
            )
            result = SimpleNamespace(
                usage={"prompt": 100, "completion": 50},
                step_results=[],
                success=True,
            )
            chat._record_local_run(
                chain=chain,
                task="hi there",
                result=result,
                started_at=time.time() - 5.0,
                duration=4.7,
                status="success",
            )
            rows = load_local_runs(cache_dir=tmp_path)
            assert len(rows) == 1
            row = rows[0]
            assert row.chain_id == "chain-A"
            assert row.chain_name == "Forecaster"
            assert row.status == "success"
            assert row.tokens_in == 100
            assert row.tokens_out == 50
            assert row.duration_seconds == 4.7
            assert row.provider == "openai"
            # `mode` defaults to the screen's mode reactive
            # (`interactive` for a fresh ChatScreen).
            assert row.mode == "interactive"
            assert row.extra.get("task") == "hi there"

    @pytest.mark.asyncio
    async def test_failure_writes_row_with_error(
        self, tmp_path: Path,
    ):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = _chat(app)
            chain = SimpleNamespace(entity_id="x", name="X")
            chat._record_local_run(
                chain=chain,
                task="break it",
                result=None,
                started_at=time.time() - 1.0,
                duration=0.9,
                status="failure",
                error="LLMClientError: 503",
            )
            rows = load_local_runs(cache_dir=tmp_path)
            assert len(rows) == 1
            row = rows[0]
            assert row.status == "failure"
            assert row.error == "LLMClientError: 503"

    @pytest.mark.asyncio
    async def test_missing_chain_fields_collapse_to_empty(
        self, tmp_path: Path,
    ):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = _chat(app)
            # Chain with no entity_id / name (a freshly
            # generated chain that hasn't been saved yet).
            chain = SimpleNamespace()
            chat._record_local_run(
                chain=chain,
                task="fresh",
                result=SimpleNamespace(
                    usage={}, step_results=[], success=True,
                ),
                started_at=time.time(),
                duration=0.5,
                status="success",
            )
            rows = load_local_runs(cache_dir=tmp_path)
            assert len(rows) == 1
            row = rows[0]
            assert row.chain_id == ""
            assert row.chain_name == ""
            # Empty usage → tokens stay None.
            assert row.tokens_in is None
            assert row.tokens_out is None

    @pytest.mark.asyncio
    async def test_handles_alternative_usage_keys(
        self, tmp_path: Path,
    ):
        """Some result variants emit OpenAI-shaped
        `prompt_tokens` / `completion_tokens` instead of
        CARL's `prompt` / `completion`. The recorder reads
        both."""
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = _chat(app)
            chain = SimpleNamespace(entity_id="y")
            result = SimpleNamespace(
                usage={
                    "prompt_tokens": 200,
                    "completion_tokens": 75,
                },
                step_results=[],
                success=True,
            )
            chat._record_local_run(
                chain=chain,
                task="alt-shape",
                result=result,
                started_at=time.time(),
                duration=1.0,
                status="success",
            )
            rows = load_local_runs(cache_dir=tmp_path)
            assert rows[0].tokens_in == 200
            assert rows[0].tokens_out == 75

    @pytest.mark.asyncio
    async def test_recorder_failure_does_not_raise(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A broken recorder must not crash the chat flow.
        Patch `record_local_run` to throw + verify the helper
        swallows it."""
        from care.runtime import local_run_history as lrh

        def _explode(*_a: Any, **_kw: Any) -> None:
            raise OSError("permission denied")

        monkeypatch.setattr(
            lrh, "record_local_run", _explode,
        )
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = _chat(app)
            # Should not raise.
            chat._record_local_run(
                chain=SimpleNamespace(entity_id="z"),
                task="boom test",
                result=None,
                started_at=time.time(),
                duration=0.1,
                status="failure",
                error="x",
            )
            # And nothing got written.
            rows = load_local_runs(cache_dir=tmp_path)
            assert rows == []


class TestDatasetRunHook:
    @pytest.mark.asyncio
    async def test_dataset_id_lands_in_extra(
        self, tmp_path: Path,
    ):
        """When the dataset runner calls
        `_record_local_run(... dataset_id="chain-X")`, the
        row's `extra` dict carries
        `{"dataset": "chain-X"}` so `/runs` can group by
        source dataset."""
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = _chat(app)
            chain = SimpleNamespace(
                entity_id="chain-A", name="Forecaster",
            )
            chat._record_local_run(
                chain=chain,
                task="dataset run entry 1",
                result=SimpleNamespace(
                    usage={}, step_results=[], success=True,
                ),
                started_at=time.time(),
                duration=0.5,
                status="success",
                dataset_id="dataset-weather",
            )
            rows = load_local_runs(cache_dir=tmp_path)
            assert len(rows) == 1
            row = rows[0]
            assert row.extra.get("dataset") == "dataset-weather"
            assert row.extra.get("task") == (
                "dataset run entry 1"
            )

    @pytest.mark.asyncio
    async def test_dataset_id_empty_omits_extra_key(
        self, tmp_path: Path,
    ):
        """A bare `_record_local_run` call (without
        dataset_id) should NOT inject an empty
        `extra["dataset"]` slot."""
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = _chat(app)
            chat._record_local_run(
                chain=SimpleNamespace(entity_id="x"),
                task="bare run",
                result=None,
                started_at=time.time(),
                duration=0.1,
                status="success",
            )
            rows = load_local_runs(cache_dir=tmp_path)
            assert rows[0].extra.get("dataset") is None
            assert rows[0].extra.get("task") == "bare run"


class TestExtractFirstStepError:
    def test_returns_first_non_empty_error(self):
        result = SimpleNamespace(step_results=[
            SimpleNamespace(success=True, error_message=""),
            SimpleNamespace(
                success=False,
                error_message="step 2 boom",
            ),
            SimpleNamespace(
                success=False,
                error_message="step 3 also broke",
            ),
        ])
        out = ChatScreen._extract_first_step_error(result)
        assert out == "step 2 boom"

    def test_falls_back_to_generic_message(self):
        # All steps succeeded → fallback message.
        result = SimpleNamespace(step_results=[
            SimpleNamespace(success=True, error_message=""),
        ])
        out = ChatScreen._extract_first_step_error(result)
        assert "no per-step error" in out

    def test_truncates_long_error(self):
        result = SimpleNamespace(step_results=[
            SimpleNamespace(
                success=False,
                error_message="x" * 500,
            ),
        ])
        out = ChatScreen._extract_first_step_error(result)
        assert len(out) == 280


# Silence unused-import warning when no test in this file
# references CARE_CACHE_DIR directly (the autouse fixture
# uses it through the user_paths module).
_ = CARE_CACHE_DIR
_ = runs_dir
