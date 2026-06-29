"""B1 — hub driver: control-API client, autostart + health-wait, env mapping.

All HTTP is respx-mocked; autostart uses a fake ``popen`` — no processes, no
network, no real environment reads.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import httpx
import pytest
import respx

from care.config import CareConfig, HubConfig
from care.runtime.agent_hub import (
    HubClient,
    HubDeployment,
    HubError,
    HubUnavailableError,
    ensure_hub_running,
    hub_env,
)

BASE = "http://127.0.0.1:8080"

_DEPLOYMENT = {
    "name": "weather",
    "url": "/agents/weather",
    "display_name": "Weather Agent",
    "version": "v3 (abc12345)",
    "ready": True,
    "ready_reason": "ok",
    "entity_id": "chain-1",
    "channel": "stable",
    "chain_file": None,
    "source": "memory",
    "deployed_at": "2026-06-10T12:00:00+00:00",
    "runs": 4,
}


class TestHubClient:
    @respx.mock
    async def test_health_up(self):
        respx.get(f"{BASE}/healthz").mock(
            return_value=httpx.Response(200, json={"status": "ok", "deployments": 2})
        )
        payload = await HubClient(BASE).health()
        assert payload == {"status": "ok", "deployments": 2}

    @respx.mock
    async def test_health_down_is_none(self):
        respx.get(f"{BASE}/healthz").mock(side_effect=httpx.ConnectError("refused"))
        assert await HubClient(BASE).health() is None

    @respx.mock
    async def test_list_deployments(self):
        respx.get(f"{BASE}/deployments").mock(
            return_value=httpx.Response(200, json=[_DEPLOYMENT])
        )
        items = await HubClient(BASE).list_deployments()
        assert items == [HubDeployment.from_payload(_DEPLOYMENT)]
        assert items[0].display_name == "Weather Agent"
        assert items[0].runs == 4

    @respx.mock
    async def test_non_json_2xx_body_raises_hub_error(self):
        # A 2xx with a non-JSON body must raise HubError (callers catch it),
        # not a raw JSONDecodeError out of _request.
        respx.get(f"{BASE}/deployments/weather").mock(
            return_value=httpx.Response(200, text="<html>not json</html>")
        )
        with pytest.raises(HubError):
            await HubClient(BASE).get_deployment("weather")

    @respx.mock
    async def test_reload_empty_body_does_not_crash(self):
        # Empty 2xx body → _request returns None; reload must not AttributeError.
        respx.post(f"{BASE}/deployments/weather/reload").mock(
            return_value=httpx.Response(200, content=b"")
        )
        reloaded, _dep = await HubClient(BASE).reload("weather")
        assert reloaded is False

    @respx.mock
    async def test_agent_metrics(self):
        report = {"run_count": 3, "total_tokens": 120, "total_cost_usd": 0.004}
        respx.get(f"{BASE}/agents/weather/metrics").mock(
            return_value=httpx.Response(200, json=report)
        )
        assert await HubClient(BASE).agent_metrics("weather") == report

    @respx.mock
    async def test_agent_metrics_none_on_failure(self):
        respx.get(f"{BASE}/agents/old/metrics").mock(
            return_value=httpx.Response(404, json={"detail": "not found"})
        )
        assert await HubClient(BASE).agent_metrics("old") is None

    @respx.mock
    async def test_deploy_posts_spec_and_parses(self):
        route = respx.post(f"{BASE}/deployments").mock(
            return_value=httpx.Response(201, json=_DEPLOYMENT)
        )
        spec = {"name": "weather", "entity_id": "chain-1", "channel": "stable"}
        deployment = await HubClient(BASE).deploy(spec)
        assert deployment.name == "weather"
        import json as _json

        assert _json.loads(route.calls.last.request.content) == spec

    @respx.mock
    async def test_deploy_conflict_surfaces_hub_detail(self):
        respx.post(f"{BASE}/deployments").mock(
            return_value=httpx.Response(
                409, json={"detail": "deployment 'weather' already exists"}
            )
        )
        with pytest.raises(HubError, match="409.*already exists"):
            await HubClient(BASE).deploy({"name": "weather", "entity_id": "e"})

    @respx.mock
    async def test_undeploy_and_reload(self):
        respx.delete(f"{BASE}/deployments/weather").mock(
            return_value=httpx.Response(200, json={"deleted": "weather"})
        )
        respx.post(f"{BASE}/deployments/weather/reload").mock(
            return_value=httpx.Response(
                200, json={"reloaded": True, "deployment": _DEPLOYMENT}
            )
        )
        client = HubClient(BASE)
        await client.undeploy("weather")
        reloaded, deployment = await client.reload("weather")
        assert reloaded is True
        assert deployment.version.startswith("v3")

    @respx.mock
    async def test_unreachable_raises_unavailable(self):
        respx.get(f"{BASE}/deployments").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(HubUnavailableError, match="unreachable"):
            await HubClient(BASE).list_deployments()

    def test_urls(self):
        client = HubClient(BASE + "/")
        assert client.agent_url("x") == f"{BASE}/agents/x"
        assert client.docs_url("x") == f"{BASE}/agents/x/docs"


class TestEnsureHubRunning:
    @respx.mock
    async def test_already_up_does_not_spawn(self):
        respx.get(f"{BASE}/healthz").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )

        def explode(*args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("popen must not be called when the hub is up")

        client = await ensure_hub_running(HubConfig(), popen=explode)
        assert isinstance(client, HubClient)

    @respx.mock
    async def test_autostart_spawns_and_waits(self, tmp_path):
        respx.get(f"{BASE}/healthz").mock(
            side_effect=[
                httpx.ConnectError("not yet"),
                httpx.ConnectError("still booting"),
                httpx.Response(200, json={"status": "ok"}),
            ]
        )
        spawned: dict = {}

        def fake_popen(command, **kwargs):
            spawned["command"] = command
            spawned["kwargs"] = kwargs
            return SimpleNamespace(pid=4242)

        config = HubConfig(state_file=str(tmp_path / "state.json"), start_timeout=5)
        client = await ensure_hub_running(
            config, env={"AGENT_LLM_API_KEY": "k"}, popen=fake_popen, poll_interval=0
        )
        assert isinstance(client, HubClient)
        command = spawned["command"]
        assert command[:2] == ["carl-agent-hub", "serve"]
        assert command[command.index("--port") + 1] == "8080"
        assert command[command.index("--state-file") + 1] == str(tmp_path / "state.json")
        assert spawned["kwargs"]["start_new_session"] is True
        assert spawned["kwargs"]["stderr"] == subprocess.STDOUT
        assert spawned["kwargs"]["env"] == {"AGENT_LLM_API_KEY": "k"}

    @respx.mock
    async def test_autostart_off_raises(self):
        respx.get(f"{BASE}/healthz").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(HubUnavailableError, match="autostart is off"):
            await ensure_hub_running(HubConfig(autostart=False))

    @respx.mock
    async def test_autostart_missing_binary_hints_install(self, tmp_path):
        # carl-agent-hub not on PATH -> FileNotFoundError; the error should
        # point at `pip install maestro-care[deploy]`.
        respx.get(f"{BASE}/healthz").mock(side_effect=httpx.ConnectError("down"))

        def missing(*a, **k):
            raise FileNotFoundError(2, "No such file or directory", "carl-agent-hub")

        config = HubConfig(state_file=str(tmp_path / "s.json"))
        with pytest.raises(HubUnavailableError, match=r"maestro-care\[deploy\]"):
            await ensure_hub_running(config, env={}, popen=missing, poll_interval=0)

    @respx.mock
    async def test_start_timeout_raises_with_log_hint(self, tmp_path):
        respx.get(f"{BASE}/healthz").mock(side_effect=httpx.ConnectError("never up"))
        config = HubConfig(
            state_file=str(tmp_path / "s.json"), start_timeout=0.05
        )
        with pytest.raises(HubUnavailableError, match="agent-hub.log"):
            await ensure_hub_running(
                config,
                env={},
                popen=lambda *a, **k: SimpleNamespace(pid=1),
                poll_interval=0,
            )


class TestHubEnv:
    def test_maps_care_settings_to_agent_vars(self):
        config = CareConfig()
        config.mage.api_key = "sk-llm"
        config.mage.model = "openai/gpt-4o"
        config.mage.base_url = "https://openrouter.ai/api/v1"
        config.memory.base_url = "http://mem:8002"
        env = hub_env(config, base={})
        assert env["AGENT_LLM_API_KEY"] == "sk-llm"
        assert env["AGENT_LLM_MODEL"] == "openai/gpt-4o"
        assert env["AGENT_LLM_BASE_URL"] == "https://openrouter.ai/api/v1"
        assert env["AGENT_MEMORY_URL"] == "http://mem:8002"
        assert "AGENT_MEMORY_API_KEY" not in env  # unset values are not exported

    def test_existing_env_wins(self):
        config = CareConfig()
        config.mage.api_key = "from-config"
        env = hub_env(config, base={"AGENT_LLM_API_KEY": "explicit"})
        assert env["AGENT_LLM_API_KEY"] == "explicit"


class TestHubConfig:
    def test_defaults(self):
        config = CareConfig()
        assert config.hub.base_url == "http://127.0.0.1:8080"
        assert config.hub.autostart is True
        assert config.hub.port == 8080
        assert config.hub.agent_server_cmd == ["carl-agent-hub", "serve"]
