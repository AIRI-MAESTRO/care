"""D4 CARE side — /metrics: per-agent usage + cost from the hub, no autostart."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from care.screens.chat import (
    _COMMAND_HANDLERS,
    ChatScreen,
    _format_metrics_row,
)


# --------------------------------------------------------- row formatter
def test_row_unavailable():
    assert _format_metrics_row("a", None) == "● a — metrics unavailable"


def test_row_unpriced():
    row = _format_metrics_row("weather", {"run_count": 3, "total_tokens": 120, "total_cost_usd": None})
    assert "runs 3" in row and "tokens 120" in row and "(unpriced)" in row


def test_row_priced_with_budget():
    row = _format_metrics_row(
        "weather",
        {"run_count": 2, "total_tokens": 1500, "total_cost_usd": 0.0030,
         "budget_usd": 0.5, "remaining_usd": 0.497, "over_budget": False},
    )
    assert "$0.0030" in row and "budget $0.50" in row and "left $0.4970" in row
    assert "OVER BUDGET" not in row


def test_row_over_budget_flag():
    row = _format_metrics_row(
        "weather",
        {"run_count": 9, "total_tokens": 9, "total_cost_usd": 0.51,
         "budget_usd": 0.5, "remaining_usd": 0.0, "over_budget": True},
    )
    assert "⚠ OVER BUDGET" in row


def test_metrics_registered():
    assert "metrics" in _COMMAND_HANDLERS


# --------------------------------------------------------------- worker
class FakeHub:
    def __init__(self, *, healthy: bool = True, deployments=None, metrics=None) -> None:
        self._healthy = healthy
        self._deployments = deployments or []
        self._metrics = metrics or {}
        self.base_url = "http://hub"

    async def health(self):
        return {"status": "ok"} if self._healthy else None

    async def list_deployments(self):
        return self._deployments

    async def agent_metrics(self, name: str):
        return self._metrics.get(name)


def _fake_self(hub: FakeHub) -> SimpleNamespace:
    posted: list[dict[str, Any]] = []
    return SimpleNamespace(
        app=SimpleNamespace(config=SimpleNamespace(hub=SimpleNamespace(base_url="http://hub", timeout=5))),
        _post_line=lambda role, text, severity=None, **_: posted.append(
            {"role": role, "text": text, "severity": severity}
        ),
        posted=posted,
        _hub=hub,
    )


_CURRENT: dict[str, Any] = {}


@pytest.fixture(autouse=True)
def _patch_hubclient(monkeypatch):
    """`HubClient(base_url, timeout=...)` in the worker returns the fake hub
    stashed in `_CURRENT` for the test."""
    import care.screens.chat as chat_module

    monkeypatch.setattr(chat_module, "HubClient", lambda *a, **k: _CURRENT["hub"])
    yield


async def test_metrics_lists_per_agent_cost():
    hub = FakeHub(
        deployments=[SimpleNamespace(name="weather"), SimpleNamespace(name="echo")],
        metrics={
            "weather": {"run_count": 2, "total_tokens": 1500, "total_cost_usd": 0.003,
                        "budget_usd": 0.5, "remaining_usd": 0.497, "over_budget": False},
            "echo": {"run_count": 1, "total_tokens": 10, "total_cost_usd": 0.001,
                     "budget_usd": None, "remaining_usd": None, "over_budget": False},
        },
    )
    _CURRENT["hub"] = hub
    self = _fake_self(hub)
    await ChatScreen._run_metrics(self)
    text = "\n".join(p["text"] for p in self.posted)
    assert "Agent metrics on http://hub" in text
    assert "weather" in text and "echo" in text
    # total spend line sums the two priced agents
    assert "total spend across priced agents: $0.0040" in text


async def test_metrics_hub_down_warns():
    hub = FakeHub(healthy=False)
    _CURRENT["hub"] = hub
    self = _fake_self(hub)
    await ChatScreen._run_metrics(self)
    warnings = [p for p in self.posted if p["severity"] == "warning"]
    assert warnings and "not running" in warnings[0]["text"]


async def test_metrics_no_deployments():
    hub = FakeHub(deployments=[])
    _CURRENT["hub"] = hub
    self = _fake_self(hub)
    await ChatScreen._run_metrics(self)
    assert any("No deployments yet" in p["text"] for p in self.posted)


async def test_metrics_unavailable_agent_row():
    hub = FakeHub(deployments=[SimpleNamespace(name="ghost")], metrics={})  # agent returns None
    _CURRENT["hub"] = hub
    self = _fake_self(hub)
    await ChatScreen._run_metrics(self)
    assert any("metrics unavailable" in p["text"] for p in self.posted)
