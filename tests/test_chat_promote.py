"""C1 — the ``/promote`` chat command: gate wiring, refuse/force, channel promote."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import care.screens.chat as chat_module
from care.runtime.promote_gate import GateCheck, PromoteGateReport
from care.screens.chat import (
    _COMMAND_HANDLERS,
    ChatScreen,
    _parse_promote_args,
)


class TestParse:
    def test_defaults(self):
        assert _parse_promote_args("chain-1") == ("chain-1", "latest", "stable", False)

    def test_flags(self):
        ref, from_ch, to_ch, force = _parse_promote_args(
            "weather agent --from dev --to prod --force"
        )
        assert ref == "weather agent"
        assert (from_ch, to_ch, force) == ("dev", "prod", True)


def test_promote_registered_and_usage():
    assert "promote" in _COMMAND_HANDLERS
    posted: list[dict[str, Any]] = []
    screen = SimpleNamespace(
        _post_line=lambda role, text, severity=None, **_: posted.append(
            {"text": text, "severity": severity}
        ),
        run_worker=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no worker")),
    )
    _COMMAND_HANDLERS["promote"](screen, "")
    assert posted and "Usage:" in posted[0]["text"]


# ------------------------------------------------------------------ worker
def _report(ok: bool) -> PromoteGateReport:
    checks = [
        GateCheck("artifact", True, "v3 loads cleanly"),
        GateCheck("baseline run", ok, "succeeded" if ok else "baseline failed: boom"),
    ]
    return PromoteGateReport(
        entity_id="chain-1", from_channel="latest", to_channel="stable", checks=checks
    )


class FakeClient:
    def __init__(self) -> None:
        self.promoted: list[tuple[str, str, str]] = []
        self.fail_promote: Exception | None = None

    def get_chain_record(self, entity_id: str, *, channel: str = "latest") -> Any:
        return SimpleNamespace(
            entity_id=entity_id,
            version_id="vid-0003",
            version_number=3,
            meta={"display_name": "Weather Agent"},
            content={"name": "Weather"},
        )

    def promote(self, entity_id: str, from_channel: str, to_channel: str, entity_type: str = "chain") -> dict:
        if self.fail_promote:
            raise self.fail_promote
        self.promoted.append((entity_id, from_channel, to_channel))
        return {"to_channel": to_channel}


def _fake_self(client: Any) -> SimpleNamespace:
    posted: list[dict[str, Any]] = []
    fake = SimpleNamespace(
        app=SimpleNamespace(
            memory=SimpleNamespace(client=client), config=SimpleNamespace()
        ),
        _post_line=lambda role, text, severity=None, **_: posted.append(
            {"role": role, "text": text, "severity": severity}
        ),
        posted=posted,
    )
    fake._fetch_chain_record = ChatScreen._fetch_chain_record.__get__(fake)
    fake._find_chain_by_name = ChatScreen._find_chain_by_name.__get__(fake)
    return fake


async def test_promote_happy_path(monkeypatch):
    gate_calls: list[dict[str, Any]] = []

    async def fake_gate(memory, config, entity_id, *, from_channel, to_channel):
        gate_calls.append({"entity_id": entity_id, "from": from_channel, "to": to_channel})
        return _report(ok=True)

    monkeypatch.setattr(chat_module, "gate_promotion", fake_gate)
    client = FakeClient()
    self = _fake_self(client)
    await ChatScreen._run_promote(self, "chain-1")
    assert gate_calls == [{"entity_id": "chain-1", "from": "latest", "to": "stable"}]
    assert client.promoted == [("chain-1", "latest", "stable")]
    text = json.dumps(self.posted, ensure_ascii=False)
    assert "⬆ Promoted" in text
    assert "✓ artifact" in text  # the gate report rendered line-by-line
    assert "hot-reload automatically" in text


async def test_gate_failure_refuses_promotion(monkeypatch):
    async def fake_gate(*a: Any, **k: Any) -> PromoteGateReport:
        return _report(ok=False)

    monkeypatch.setattr(chat_module, "gate_promotion", fake_gate)
    client = FakeClient()
    self = _fake_self(client)
    await ChatScreen._run_promote(self, "chain-1")
    assert client.promoted == []  # refused
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "Promotion refused" in errors[0]["text"]
    assert "--force" in errors[0]["text"]


async def test_force_skips_the_gate(monkeypatch):
    async def exploding_gate(*a: Any, **k: Any) -> Any:  # pragma: no cover
        raise AssertionError("gate must not run with --force")

    monkeypatch.setattr(chat_module, "gate_promotion", exploding_gate)
    client = FakeClient()
    self = _fake_self(client)
    await ChatScreen._run_promote(self, "chain-1 --force")
    assert client.promoted == [("chain-1", "latest", "stable")]
    warnings = [p for p in self.posted if p["severity"] == "warning"]
    assert warnings and "skipping the promotion gate" in warnings[0]["text"]


async def test_promote_error_surfaced(monkeypatch):
    async def fake_gate(*a: Any, **k: Any) -> PromoteGateReport:
        return _report(ok=True)

    monkeypatch.setattr(chat_module, "gate_promotion", fake_gate)
    client = FakeClient()
    client.fail_promote = RuntimeError("channel locked")
    self = _fake_self(client)
    await ChatScreen._run_promote(self, "chain-1")
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "channel locked" in errors[0]["text"]


async def test_unresolvable_ref():
    class NoChainClient:
        def get_chain_record(self, *a: Any, **k: Any) -> Any:
            raise KeyError("nope")

    self = _fake_self(NoChainClient())
    await ChatScreen._run_promote(self, "ghost")
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "Could not resolve" in errors[0]["text"]
