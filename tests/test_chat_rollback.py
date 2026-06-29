"""B5 — ``/rollback``: pin the channel at an earlier version (the deployment
rollback lever; attached agents hot-reload via their watcher)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from care.screens.chat import (
    _COMMAND_HANDLERS,
    ChatScreen,
    _parse_rollback_args,
)


class TestParse:
    def test_defaults(self):
        assert _parse_rollback_args("chain-1") == ("chain-1", "stable", None)

    def test_flags(self):
        ref, channel, to = _parse_rollback_args(
            "weather agent --channel latest --to vid-0002"
        )
        assert ref == "weather agent"
        assert channel == "latest"
        assert to == "vid-0002"


def test_rollback_registered_and_usage():
    assert "rollback" in _COMMAND_HANDLERS
    posted: list[dict[str, Any]] = []
    screen = SimpleNamespace(
        _post_line=lambda role, text, severity=None, **_: posted.append(
            {"text": text, "severity": severity}
        ),
        run_worker=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no worker")),
    )
    _COMMAND_HANDLERS["rollback"](screen, "  ")
    assert posted and "Usage:" in posted[0]["text"]


# ----------------------------------------------------------------- worker
def _version(number: int) -> SimpleNamespace:
    return SimpleNamespace(version_id=f"vid-{number:04d}", version_number=number)


class FakeClient:
    def __init__(self, *, current: int = 3, versions: list[int] = (1, 2, 3)) -> None:
        self.current = current
        self.versions = list(versions)
        self.pinned: list[tuple[str, str, str]] = []
        self.fail_pin: Exception | None = None

    def get_chain_record(self, entity_id: str, *, channel: str = "latest") -> Any:
        return SimpleNamespace(
            entity_id=entity_id,
            version_id=f"vid-{self.current:04d}",
            version_number=self.current,
            channel=channel,
            meta={"display_name": "Weather Agent"},
            content={"name": "Weather"},
        )

    def list_versions(self, entity_id: str, entity_type: str = "chain", limit: int = 20) -> list[Any]:
        return [_version(n) for n in self.versions]

    def pin_channel(self, entity_id: str, channel: str, version_id: str, entity_type: str = "chain") -> dict:
        if self.fail_pin:
            raise self.fail_pin
        self.pinned.append((entity_id, channel, version_id))
        return {"channel": channel, "version_id": version_id}


def _fake_self(client: Any) -> SimpleNamespace:
    posted: list[dict[str, Any]] = []
    fake = SimpleNamespace(
        app=SimpleNamespace(memory=SimpleNamespace(client=client)),
        _post_line=lambda role, text, severity=None, **_: posted.append(
            {"role": role, "text": text, "severity": severity}
        ),
        posted=posted,
    )
    fake._fetch_chain_record = ChatScreen._fetch_chain_record.__get__(fake)
    fake._find_chain_by_name = ChatScreen._find_chain_by_name.__get__(fake)
    return fake


async def test_rollback_pins_previous_version():
    client = FakeClient(current=3, versions=[1, 2, 3])
    self = _fake_self(client)
    await ChatScreen._run_rollback(self, "chain-1")
    assert client.pinned == [("chain-1", "stable", "vid-0002")]
    text = json.dumps(self.posted, ensure_ascii=False)
    assert "⏪" in text and "v3 → v2" in text
    assert "hot-reload automatically" in text


async def test_rollback_to_explicit_version():
    client = FakeClient(current=3)
    self = _fake_self(client)
    await ChatScreen._run_rollback(self, "chain-1 --to vid-0001 --channel latest")
    assert client.pinned == [("chain-1", "latest", "vid-0001")]


async def test_rollback_on_first_version_warns():
    client = FakeClient(current=1, versions=[1])
    self = _fake_self(client)
    await ChatScreen._run_rollback(self, "chain-1")
    assert client.pinned == []
    warnings = [p for p in self.posted if p["severity"] == "warning"]
    assert warnings and "earliest version" in warnings[0]["text"]


async def test_rollback_pin_failure_surfaces():
    client = FakeClient()
    client.fail_pin = RuntimeError("pin denied")
    self = _fake_self(client)
    await ChatScreen._run_rollback(self, "chain-1")
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "pin denied" in errors[0]["text"]


async def test_rollback_unresolvable_ref():
    class NoChainClient:
        def get_chain_record(self, *a: Any, **k: Any) -> Any:
            raise KeyError("nope")

    self = _fake_self(NoChainClient())
    await ChatScreen._run_rollback(self, "ghost")
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "Could not resolve" in errors[0]["text"]
