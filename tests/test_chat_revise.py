"""Tests for the ``/revise`` chat command dispatch (NL chain editing).

The dispatch layer (`_cmd_revise`) is unit-tested here against a duck-typed
screen — it only touches ``screen._post_line`` and
``screen.run_worker(screen._run_edit(...))``. The full ``_run_edit`` worker
flow (preview → confirm → save) is exercised by the integration tests in this
file's ``TestRunEdit`` class.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from textual.app import App

from care.config import CareConfig, MageConfig
from care.screens.chat import ChatScreen, _COMMAND_HANDLERS


def _fake_screen() -> SimpleNamespace:
    posted: list[dict[str, Any]] = []
    spawned: list[dict[str, Any]] = []
    coros: list[Any] = []

    def _post_line(role: str, text: str, *, severity: str | None = None, **_: Any) -> None:
        posted.append({"role": role, "text": text, "severity": severity})

    def _run_worker(coro: Any, **kw: Any) -> None:
        # Capture + close the coroutine so it doesn't warn about being un-awaited.
        coros.append(coro)
        try:
            coro.close()
        except Exception:
            pass
        spawned.append(kw)

    def _run_edit(raw: str) -> Any:
        # Return a dummy coroutine object so run_worker has something to close.
        async def _c() -> None:
            return None

        screen.last_edit_arg = raw
        return _c()

    screen = SimpleNamespace(
        _post_line=_post_line,
        run_worker=_run_worker,
        _run_edit=_run_edit,
        posted=posted,
        spawned=spawned,
        last_edit_arg=None,
    )
    return screen


def test_revise_is_registered() -> None:
    assert "revise" in _COMMAND_HANDLERS


def test_revise_empty_arg_shows_usage() -> None:
    screen = _fake_screen()
    _COMMAND_HANDLERS["revise"](screen, "   ")
    assert screen.spawned == []  # no worker spawned
    assert screen.posted and screen.posted[0]["severity"] == "warning"
    assert "Usage:" in screen.posted[0]["text"]


def test_revise_spawns_worker_with_arg() -> None:
    screen = _fake_screen()
    _COMMAND_HANDLERS["revise"](screen, "abc123 add a validation step")
    assert len(screen.spawned) == 1
    kw = screen.spawned[0]
    assert kw["group"] == "generate"
    assert kw["exclusive"] is True
    assert kw["name"] == "chat_edit"
    assert screen.last_edit_arg == "abc123 add a validation step"


def test_revise_strips_arg() -> None:
    screen = _fake_screen()
    _COMMAND_HANDLERS["revise"](screen, "  add a step  ")
    assert screen.last_edit_arg == "add a step"


# ---------------------------------------------------------------------------
# Full _run_edit worker flow (preview → confirm → save) — mounted ChatScreen
# ---------------------------------------------------------------------------


def _edit(op: str, target: int | None = None, rationale: str = "r") -> SimpleNamespace:
    return SimpleNamespace(op=op, target_step_number=target, rationale=rationale)


def _result(
    *,
    edits: list[Any] | None = None,
    summary: str = "changed",
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    entity_id: str | None = "abc",
    needs_disambiguation: bool = False,
    candidates: list[Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        edits=edits or [],
        summary=summary,
        before_chain_dict=before or {},
        chain_dict=after or {},
        entity_id=entity_id,
        needs_disambiguation=needs_disambiguation,
        candidates=candidates or [],
    )


class _FakeGen:
    """Duck-typed MAGEGenerator whose ``edit`` returns a canned result."""

    def __init__(self, result: SimpleNamespace) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def edit(
        self,
        instruction: str,
        *,
        chain: dict[str, Any] | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
        save: bool = False,
        cancel: Any = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {"instruction": instruction, "entity_id": entity_id, "chain": chain, "save": save}
        )
        return self._result


class _FakeMemory:
    """CareMemory stand-in: ``client.get_chain_dict`` + ``save_chain``."""

    def __init__(self, *, known: dict[str, dict[str, Any]] | None = None) -> None:
        self._known = known or {}
        self.saved: list[dict[str, Any]] = []
        self.client = SimpleNamespace(get_chain_dict=self._get)

    def _get(self, entity_id: str, channel: str = "latest") -> dict[str, Any]:
        if entity_id in self._known:
            return dict(self._known[entity_id])
        raise KeyError(entity_id)  # unknown id ⇒ caller treats token as prose

    def save_chain(
        self,
        chain: Any,
        *,
        name: str,
        query: str | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
        parent_version_id: str | None = None,
        change_summary: str | None = None,
        **_: Any,
    ) -> str:
        self.saved.append({
            "entity_id": entity_id,
            "name": name,
            "channel": channel,
            "parent_version_id": parent_version_id,
            "change_summary": change_summary,
        })
        return "newid42"

    def get_entity(
        self,
        entity_id: str,
        *,
        entity_type: str = "chain",
        channel: str = "latest",
    ) -> dict[str, Any]:
        return {"entity_id": entity_id, "version_id": "vid-new", "version_number": 2}


async def _drive_revise(
    monkeypatch: Any,
    *,
    raw: str,
    result: SimpleNamespace,
    memory: _FakeMemory,
    confirm: bool = True,
    session_chain: dict[str, Any] | None = None,
    session_version_id: str | None = None,
    session_version_number: int | None = None,
) -> tuple[_FakeGen, list[tuple[str, str]]]:
    fake_gen = _FakeGen(result)
    monkeypatch.setattr(
        "care.generation.build_mage_generator", lambda *a, **k: fake_gen
    )

    class _MemHost(App):
        def on_mount(self) -> None:
            self.memory = memory  # type: ignore[attr-defined]
            self.push_screen(ChatScreen())

    app = _MemHost()
    app.config = CareConfig(  # type: ignore[attr-defined]
        mage=MageConfig(api_key="sk", base_url="https://e.test/v1", model="m"),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        assert isinstance(screen, ChatScreen)
        if session_chain is not None:
            screen._begin_chain_session(
                chain_dict=session_chain,
                display_name="Doc",
                task="task",
                chain_id="abc",
                version_id=session_version_id,
                version_number=session_version_number,
            )

        async def _modal(modal: Any, *_a: Any, **_k: Any) -> Any:
            from care.screens.confirm import ConfirmModal

            if isinstance(modal, ConfirmModal):
                return confirm
            return confirm

        app.push_screen_wait = _modal  # type: ignore[assignment, method-assign]
        await screen._run_edit(raw)
        await pilot.pause()
        lines = [(ln.role, ln.text) for ln in screen._lines]
    return fake_gen, lines


class TestRunEdit:
    async def test_by_id_applies_edit_in_session_without_save(
        self, monkeypatch: Any,
    ) -> None:
        chain = {
            "name": "Doc",
            "steps": [{"number": 1, "title": "S1", "step_type": "llm", "aim": "a", "dependencies": []}],
        }
        after = {"name": "Invoice QA", "steps": chain["steps"]}
        result = _result(
            edits=[_edit("set_chain_field", rationale="rename")],
            summary="renamed to Invoice QA",
            before=chain,
            after=after,
            entity_id="abc",
        )
        mem = _FakeMemory(known={"abc": chain})
        gen, lines = await _drive_revise(
            monkeypatch, raw="abc rename to Invoice QA", result=result, memory=mem, confirm=True
        )
        assert gen.calls[0]["entity_id"] == "abc"
        assert gen.calls[0]["instruction"] == "rename to Invoice QA"
        assert gen.calls[0]["chain"] == chain
        assert any("renamed to Invoice QA" in t for _, t in lines)
        assert mem.saved == []
        assert any(r == "assistant" for r, t in lines if "applied" in t.lower() or "Правка" in t)

    async def test_discard_does_not_save(self, monkeypatch: Any) -> None:
        chain = {"name": "Doc", "steps": [{"number": 1, "title": "S1", "step_type": "llm", "aim": "a", "dependencies": []}]}
        result = _result(
            edits=[_edit("set_chain_field")],
            before=chain,
            after={"name": "X", "steps": chain["steps"]},
            entity_id="abc",
        )
        mem = _FakeMemory(known={"abc": chain})
        _gen, lines = await _drive_revise(
            monkeypatch, raw="abc rename it", result=result, memory=mem, confirm=False
        )
        assert mem.saved == []
        assert any("discarded" in t.lower() for _, t in lines)

    async def test_session_selected_version_used_as_edit_base(
        self, monkeypatch: Any,
    ) -> None:
        chain_v1 = {
            "name": "Doc",
            "steps": [{"number": 1, "title": "S1", "step_type": "llm", "aim": "a", "dependencies": []}],
        }
        chain_v0 = {
            "name": "Doc old",
            "steps": [{"number": 1, "title": "S0", "step_type": "llm", "aim": "a", "dependencies": []}],
        }
        result = _result(
            edits=[_edit("set_chain_field")],
            before=chain_v0,
            after={"name": "Doc v2", "steps": chain_v0["steps"]},
            entity_id="abc",
        )
        mem = _FakeMemory(known={"abc": chain_v1})
        gen, lines = await _drive_revise(
            monkeypatch,
            raw="abc tweak step",
            result=result,
            memory=mem,
            confirm=True,
            session_chain=chain_v0,
            session_version_id="vid-0",
            session_version_number=1,
        )
        assert gen.calls[0]["chain"] == chain_v0
        assert mem.saved == []
        assert any("applied" in t.lower() or "Правка" in t for _, t in lines)

    async def test_disambiguation_lists_candidates(self, monkeypatch: Any) -> None:
        result = _result(
            edits=[],
            needs_disambiguation=True,
            candidates=[
                SimpleNamespace(entity_id="a", name="Alpha", score=0.8),
                SimpleNamespace(entity_id="b", name="Beta", score=0.7),
            ],
            entity_id=None,
        )
        mem = _FakeMemory(known={})  # first token won't resolve → prose/search mode
        gen, lines = await _drive_revise(
            monkeypatch, raw="change the pdf summarizer", result=result, memory=mem, confirm=True
        )
        assert mem.saved == []
        assert any("Multiple chains match" in t for _, t in lines)
        assert any("Alpha" in t for _, t in lines)
        # nothing loaded ⇒ MAGE resolves: edit() called with no id / no chain
        assert gen.calls[0]["entity_id"] is None
        assert gen.calls[0]["chain"] is None

    async def test_noop_when_no_edits(self, monkeypatch: Any) -> None:
        chain = {"name": "Doc", "steps": [{"number": 1, "title": "S1", "step_type": "llm", "aim": "a", "dependencies": []}]}
        result = _result(edits=[], before=chain, after=chain, entity_id="abc", summary="No change needed.")
        mem = _FakeMemory(known={"abc": chain})
        _gen, lines = await _drive_revise(
            monkeypatch, raw="abc do nothing", result=result, memory=mem, confirm=True
        )
        assert mem.saved == []
        assert any("No change needed" in t for _, t in lines)


class TestSeedInput:
    async def test_seed_input_sets_value_and_focus(self) -> None:
        from care.widgets.chat_input import ChatInput

        class _H(App):
            def on_mount(self) -> None:
                self.push_screen(ChatScreen())

        app = _H()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, ChatScreen)
            screen.seed_input("/revise abc ")
            await pilot.pause()
            inp = screen.query_one("#chat-input", ChatInput)
            assert inp.value == "/revise abc "
