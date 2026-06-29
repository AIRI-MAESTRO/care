"""B3 — the ``/deployments`` chat command: list rows, undeploy/reload/docs
actions, down-hub hint. The HubClient seam is monkeypatched at the chat
module — no network."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import care.screens.chat as chat_module
from care.config import CareConfig
from care.runtime.agent_hub import HubDeployment, HubError
from care.screens.chat import _COMMAND_HANDLERS, ChatScreen, _format_uptime


def _deployment(name: str, *, ready: bool = True, runs: int = 2) -> HubDeployment:
    return HubDeployment.from_payload(
        {
            "name": name,
            "url": f"/agents/{name}",
            "display_name": name.title(),
            "version": "v2 (abcd1234)",
            "ready": ready,
            "ready_reason": "ok" if ready else "missing dependencies: tool:x",
            "entity_id": "chain-1",
            "channel": "stable",
            "chain_file": None,
            "source": "memory",
            "deployed_at": (
                datetime.now(timezone.utc) - timedelta(minutes=5)
            ).isoformat(),
            "runs": runs,
        }
    )


class FakeHubClient:
    """Stands in for chat_module.HubClient — records calls, canned replies."""

    instances: list["FakeHubClient"] = []
    healthy = True
    deployments: list[HubDeployment] = []
    reload_result: tuple[bool, HubDeployment] | None = None
    error: Exception | None = None

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.calls: list[tuple[str, Any]] = []
        FakeHubClient.instances.append(self)

    def agent_url(self, name: str) -> str:
        return f"{self.base_url}/agents/{name}"

    def docs_url(self, name: str) -> str:
        return f"{self.agent_url(name)}/docs"

    async def health(self) -> dict[str, Any] | None:
        return {"status": "ok"} if FakeHubClient.healthy else None

    async def list_deployments(self) -> list[HubDeployment]:
        if FakeHubClient.error:
            raise FakeHubClient.error
        self.calls.append(("list", None))
        return list(FakeHubClient.deployments)

    async def undeploy(self, name: str) -> None:
        if FakeHubClient.error:
            raise FakeHubClient.error
        self.calls.append(("undeploy", name))

    async def reload(self, name: str) -> tuple[bool, HubDeployment]:
        self.calls.append(("reload", name))
        assert FakeHubClient.reload_result is not None
        return FakeHubClient.reload_result


def _fake_self() -> SimpleNamespace:
    posted: list[dict[str, Any]] = []

    def _post_line(role: str, text: str, *, severity: str | None = None, **_: Any) -> None:
        posted.append({"role": role, "text": text, "severity": severity})

    return SimpleNamespace(
        app=SimpleNamespace(config=CareConfig()),
        _post_line=_post_line,
        posted=posted,
    )


def _reset(monkeypatch, **overrides: Any) -> None:
    FakeHubClient.instances = []
    FakeHubClient.healthy = overrides.get("healthy", True)
    FakeHubClient.deployments = overrides.get("deployments", [])
    FakeHubClient.reload_result = overrides.get("reload_result")
    FakeHubClient.error = overrides.get("error")
    monkeypatch.setattr(chat_module, "HubClient", FakeHubClient)


# ------------------------------------------------------------------ dispatch
def test_deployments_is_registered():
    assert "deployments" in _COMMAND_HANDLERS


def test_unknown_subcommand_shows_usage():
    posted: list[dict[str, Any]] = []
    screen = SimpleNamespace(
        _post_line=lambda role, text, severity=None, **_: posted.append(
            {"text": text, "severity": severity}
        ),
        run_worker=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no worker")),
    )
    _COMMAND_HANDLERS["deployments"]( screen, "explode now")
    assert posted and "Usage:" in posted[0]["text"]


def test_action_without_name_shows_usage():
    posted: list[dict[str, Any]] = []
    screen = SimpleNamespace(
        _post_line=lambda role, text, severity=None, **_: posted.append(
            {"text": text, "severity": severity}
        ),
        run_worker=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no worker")),
    )
    _COMMAND_HANDLERS["deployments"](screen, "undeploy")
    assert posted and "Usage:" in posted[0]["text"]


def test_bare_command_spawns_list_worker():
    spawned: list[dict[str, Any]] = []

    def _run_worker(coro: Any, **kw: Any) -> None:
        coro.close()
        spawned.append(kw)

    captured: dict[str, Any] = {}

    def _run_deployments(action: str, name: str | None) -> Any:
        async def _c() -> None:
            return None

        captured["action"] = action
        captured["name"] = name
        return _c()

    screen = SimpleNamespace(
        _post_line=lambda *a, **k: None,
        run_worker=_run_worker,
        _run_deployments=_run_deployments,
    )
    _COMMAND_HANDLERS["deployments"](screen, "")
    assert captured == {"action": "list", "name": None}
    assert spawned and spawned[0]["group"] == "generate"


# ------------------------------------------------------------------- worker
async def test_list_renders_rows(monkeypatch):
    _reset(
        monkeypatch,
        deployments=[_deployment("weather"), _deployment("news", ready=False)],
    )
    self = _fake_self()
    await ChatScreen._run_deployments(self, "list", None)
    text = json.dumps(self.posted, ensure_ascii=False)
    assert "● weather — Weather (v2 (abcd1234)) · ready · runs 2" in text
    assert "⚠ missing dependencies: tool:x" in text
    assert "/agents/weather/docs" in text
    assert "actions: /deployments undeploy" in text
    assert "· up " in text  # uptime rendered from deployed_at


async def test_list_empty(monkeypatch):
    _reset(monkeypatch, deployments=[])
    self = _fake_self()
    await ChatScreen._run_deployments(self, "list", None)
    assert any("No deployments yet" in p["text"] for p in self.posted)


async def test_hub_down_posts_hint_without_autostart(monkeypatch):
    _reset(monkeypatch, healthy=False)
    self = _fake_self()
    await ChatScreen._run_deployments(self, "list", None)
    assert any("is not running" in p["text"] for p in self.posted)
    # the only instance just probed health — no other calls
    assert FakeHubClient.instances[0].calls == []


async def test_undeploy_calls_and_posts(monkeypatch):
    _reset(monkeypatch)
    self = _fake_self()
    await ChatScreen._run_deployments(self, "undeploy", "weather")
    assert ("undeploy", "weather") in FakeHubClient.instances[0].calls
    assert any("Undeployed" in p["text"] for p in self.posted)


async def test_reload_success_and_canary(monkeypatch):
    _reset(monkeypatch, reload_result=(True, _deployment("weather")))
    self = _fake_self()
    await ChatScreen._run_deployments(self, "reload", "weather")
    assert any("Reloaded" in p["text"] for p in self.posted)

    _reset(monkeypatch, reload_result=(False, _deployment("weather")))
    self = _fake_self()
    await ChatScreen._run_deployments(self, "reload", "weather")
    warnings = [p for p in self.posted if p["severity"] == "warning"]
    assert warnings and "kept the previous version" in warnings[0]["text"]


async def test_docs_opens_and_posts_url(monkeypatch):
    _reset(monkeypatch)
    opened: list[str] = []
    monkeypatch.setattr(chat_module, "open_url", lambda url: (opened.append(url), True)[1])
    self = _fake_self()
    await ChatScreen._run_deployments(self, "docs", "weather")
    assert opened == ["http://127.0.0.1:8080/agents/weather/docs"]
    assert any("(opened in browser)" in p["text"] for p in self.posted)


async def test_hub_error_is_posted(monkeypatch):
    _reset(monkeypatch, error=HubError("hub 404: deployment 'ghost' not found"))
    self = _fake_self()
    await ChatScreen._run_deployments(self, "undeploy", "ghost")
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "404" in errors[0]["text"]


# ------------------------------------------------------------------- helper
def test_format_uptime():
    five_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    assert _format_uptime(five_min_ago).endswith("m")
    hours_ago = (datetime.now(timezone.utc) - timedelta(hours=3, minutes=2)).isoformat()
    assert _format_uptime(hours_ago).startswith("3h")
    assert _format_uptime("not-a-date") == ""
