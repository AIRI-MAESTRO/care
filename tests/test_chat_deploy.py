"""B2 — the ``/deploy`` chat command: parsing, gate-lite, hub hand-off.

Dispatch is unit-tested against a duck-typed screen (the `/revise` pattern);
the ``_run_deploy`` worker runs as an unbound coroutine on a SimpleNamespace
"self" with the hub functions monkeypatched at the chat-module seam — no
network, no processes, no Textual app.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import care.screens.chat as chat_module
from care.config import CareConfig
from care.runtime.agent_hub import HubDeployment, HubError, HubUnavailableError
from care.runtime.deploy_gate import TEMPLATE_TOOLS, gate_chain_for_deploy
from care.screens.chat import (
    _COMMAND_HANDLERS,
    ChatScreen,
    _parse_deploy_args,
    _slugify_agent_name,
)

SAMPLE_CHAIN: dict[str, Any] = {
    "name": "Echo Researcher",
    "max_workers": 1,
    "timeout": 60.0,
    "steps": [
        {
            "step_type": "llm",
            "number": 1,
            "title": "Answer",
            "aim": "Answer the question",
            "reasoning_questions": "",
            "step_context_queries": [],
            "stage_action": "Answer",
            "example_reasoning": "",
            "dependencies": [],
            "retry_max": 1,
        }
    ],
}


# --------------------------------------------------------------------- parse
class TestParseDeployArgs:
    def test_defaults(self):
        assert _parse_deploy_args("abc-123") == ("abc-123", "stable", None)

    def test_flags(self):
        ref, channel, name = _parse_deploy_args(
            "weather agent --channel latest --name wx"
        )
        assert ref == "weather agent"
        assert channel == "latest"
        assert name == "wx"

    def test_flags_anywhere(self):
        ref, channel, name = _parse_deploy_args("--name wx my chain --channel x")
        assert (ref, channel, name) == ("my chain", "x", "wx")


class TestSlugify:
    def test_display_name(self):
        assert _slugify_agent_name("Weather Agent 2.0!") == "weather-agent-2.0"

    def test_fallback(self):
        assert _slugify_agent_name("***") == "agent"


# ---------------------------------------------------------------------- gate
class TestDeployGate:
    def test_clean_chain_passes(self):
        assert gate_chain_for_deploy(dict(SAMPLE_CHAIN)) == []

    def test_unknown_tool_fails(self):
        chain = dict(SAMPLE_CHAIN)
        chain["steps"] = [
            {
                "step_type": "tool",
                "number": 1,
                "title": "T",
                "dependencies": [],
                "step_config": {"tool_name": "run_python", "input_mapping": {}},
            }
        ]
        issues = gate_chain_for_deploy(chain)
        assert issues and any("run_python" in issue for issue in issues)

    def test_template_tool_is_allowed(self):
        chain = dict(SAMPLE_CHAIN)
        chain["steps"] = [
            {
                "step_type": "tool",
                "number": 1,
                "title": "Time",
                "dependencies": [],
                "step_config": {"tool_name": "current_datetime", "input_mapping": {}},
            }
        ]
        assert gate_chain_for_deploy(chain) == []

    def test_empty_chain_fails(self):
        assert gate_chain_for_deploy({}) == ["chain content is empty — nothing to deploy"]

    def test_template_set_matches_agent_server(self):
        assert TEMPLATE_TOOLS == {
            "calculator",
            "current_datetime",
            "fetch_url",
            "http_request",
            "web_search",
        }


# ------------------------------------------------------------------ dispatch
def _fake_screen() -> SimpleNamespace:
    posted: list[dict[str, Any]] = []
    spawned: list[dict[str, Any]] = []

    def _post_line(role: str, text: str, *, severity: str | None = None, **_: Any) -> None:
        posted.append({"role": role, "text": text, "severity": severity})

    def _run_worker(coro: Any, **kw: Any) -> None:
        try:
            coro.close()
        except Exception:
            pass
        spawned.append(kw)

    def _run_deploy(raw: str) -> Any:
        async def _c() -> None:
            return None

        screen.last_deploy_arg = raw
        return _c()

    screen = SimpleNamespace(
        _post_line=_post_line,
        run_worker=_run_worker,
        _run_deploy=_run_deploy,
        posted=posted,
        spawned=spawned,
        last_deploy_arg=None,
    )
    return screen


def test_deploy_is_registered():
    assert "deploy" in _COMMAND_HANDLERS
    assert _COMMAND_HANDLERS["revise"].__name__ == "_cmd_revise"  # neighbour intact


def test_deploy_empty_shows_usage():
    screen = _fake_screen()
    _COMMAND_HANDLERS["deploy"](screen, "  ")
    assert screen.spawned == []
    assert screen.posted[0]["severity"] == "warning"
    assert "Usage:" in screen.posted[0]["text"]


def test_deploy_spawns_worker():
    screen = _fake_screen()
    _COMMAND_HANDLERS["deploy"](screen, "chain-1 --channel latest")
    assert screen.last_deploy_arg == "chain-1 --channel latest"
    assert screen.spawned and screen.spawned[0]["group"] == "generate"


# ------------------------------------------------------------- _run_deploy
class FakeHub:
    def __init__(self, *, deploy_error: Exception | None = None) -> None:
        self.deploy_error = deploy_error
        self.deployed: list[dict[str, Any]] = []

    def agent_url(self, name: str) -> str:
        return f"http://127.0.0.1:8080/agents/{name}"

    def docs_url(self, name: str) -> str:
        return f"{self.agent_url(name)}/docs"

    async def deploy(self, spec: dict[str, Any]) -> HubDeployment:
        if self.deploy_error is not None:
            raise self.deploy_error
        self.deployed.append(spec)
        return HubDeployment.from_payload(
            {
                "name": spec["name"],
                "url": f"/agents/{spec['name']}",
                "display_name": "Echo Researcher",
                "version": "v2 (abcd1234)",
                "ready": True,
                "ready_reason": "ok",
                "entity_id": spec.get("entity_id"),
                "channel": spec.get("channel"),
                "chain_file": None,
                "source": "memory",
                "deployed_at": "2026-06-10T12:00:00+00:00",
                "runs": 0,
            }
        )


class FakeClient:
    def __init__(self, record: Any | None) -> None:
        self.record = record

    def get_chain_record(self, entity_id: str, *, channel: str = "latest") -> Any:
        if self.record is None:
            raise KeyError(entity_id)
        return self.record


def _fake_self(record: Any | None) -> SimpleNamespace:
    posted: list[dict[str, Any]] = []

    def _post_line(role: str, text: str, *, severity: str | None = None, **_: Any) -> None:
        posted.append({"role": role, "text": text, "severity": severity})

    fake = SimpleNamespace(
        app=SimpleNamespace(
            memory=SimpleNamespace(client=FakeClient(record)),
            config=CareConfig(),
        ),
        _post_line=_post_line,
        posted=posted,
        _fetch_chain_record=None,
        _find_chain_by_name=None,
    )
    # bind the real helper methods onto the fake self
    fake._fetch_chain_record = ChatScreen._fetch_chain_record.__get__(fake)
    fake._find_chain_by_name = ChatScreen._find_chain_by_name.__get__(fake)
    return fake


def _record(content: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        entity_id="chain-1",
        version_id="vid-1",
        version_number=2,
        channel="stable",
        meta={"display_name": "Echo Researcher"},
        content=content,
    )


async def test_run_deploy_happy_path(monkeypatch):
    fake_hub = FakeHub()

    async def fake_ensure(hub_config, *, env=None, **_):
        return fake_hub

    monkeypatch.setattr(chat_module, "ensure_hub_running", fake_ensure)
    self = _fake_self(_record(dict(SAMPLE_CHAIN)))
    await ChatScreen._run_deploy(self, "chain-1")
    assert len(fake_hub.deployed) == 1
    spec = fake_hub.deployed[0]
    assert spec["name"] == "echo-researcher"
    assert spec["entity_id"] == "chain-1"
    assert spec["channel"] == "stable"
    # C4: a per-agent api key is generated and passed in the spec + posted
    assert spec["api_key"] and len(spec["api_key"]) >= 20
    text = json.dumps(self.posted, ensure_ascii=False)
    assert "Deploy gate passed" in text
    assert "/agents/echo-researcher/docs" in text
    assert "api key:" in text
    assert spec["api_key"] in text
    assert "🚀" in text


async def test_run_deploy_gate_blocks(monkeypatch):
    fake_hub = FakeHub()

    async def fake_ensure(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("hub must not be contacted when the gate fails")

    monkeypatch.setattr(chat_module, "ensure_hub_running", fake_ensure)
    bad = dict(SAMPLE_CHAIN)
    bad["steps"] = [
        {
            "step_type": "tool",
            "number": 1,
            "title": "T",
            "dependencies": [],
            "step_config": {"tool_name": "run_python", "input_mapping": {}},
        }
    ]
    self = _fake_self(_record(bad))
    await ChatScreen._run_deploy(self, "chain-1")
    assert fake_hub.deployed == []
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "Deploy gate failed" in errors[0]["text"]


async def test_run_deploy_unresolvable_ref(monkeypatch):
    self = _fake_self(None)  # id fetch raises; no list_chains on the fake client
    await ChatScreen._run_deploy(self, "ghost-chain")
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "Could not resolve" in errors[0]["text"]


async def test_run_deploy_hub_conflict(monkeypatch):
    fake_hub = FakeHub(deploy_error=HubError("hub 409: deployment exists"))

    async def fake_ensure(*a, **k):
        return fake_hub

    monkeypatch.setattr(chat_module, "ensure_hub_running", fake_ensure)
    self = _fake_self(_record(dict(SAMPLE_CHAIN)))
    await ChatScreen._run_deploy(self, "chain-1 --name taken")
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "Deploy rejected" in errors[0]["text"]


async def test_run_deploy_hub_down(monkeypatch):
    async def fake_ensure(*a, **k):
        raise HubUnavailableError("hub did not become healthy within 15s")

    monkeypatch.setattr(chat_module, "ensure_hub_running", fake_ensure)
    self = _fake_self(_record(dict(SAMPLE_CHAIN)))
    await ChatScreen._run_deploy(self, "chain-1")
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "Hub unavailable" in errors[0]["text"]


@pytest.mark.parametrize("flag_channel", ["latest", "stable"])
async def test_run_deploy_channel_passthrough(monkeypatch, flag_channel):
    fake_hub = FakeHub()

    async def fake_ensure(*a, **k):
        return fake_hub

    monkeypatch.setattr(chat_module, "ensure_hub_running", fake_ensure)
    self = _fake_self(_record(dict(SAMPLE_CHAIN)))
    await ChatScreen._run_deploy(self, f"chain-1 --channel {flag_channel}")
    assert fake_hub.deployed[0]["channel"] == flag_channel
