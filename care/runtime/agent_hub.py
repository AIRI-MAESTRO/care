"""Hub driver (PRODUCTION_TODO B1): the carl-agent-server control-API client
plus hub autostart.

CARE deploys chains into the **agent hub** — one lightweight process hosting N
agents, each mounted at ``/agents/<name>`` with its own Swagger. This module
gives the chat surface:

* :class:`HubClient` — thin async httpx wrappers over the control API
  (``/healthz``, ``GET/POST/DELETE /deployments``,
  ``POST /deployments/{name}/reload``), parsing replies into
  :class:`HubDeployment`;
* :func:`ensure_hub_running` — probe ``/healthz``; when the hub is down and
  ``[hub].autostart`` is on, spawn ``agent_server_cmd`` detached (stdout to
  ``~/.care/agent-hub.log``) and poll until healthy or ``start_timeout``;
* :func:`hub_env` — the env block an autostarted hub inherits: CARE's
  LLM / Memory settings mapped onto the agent server's ``AGENT_*`` variables,
  so deployed agents run on the same model + Memory the TUI uses.

Everything is injection-friendly (``popen`` / ``base`` parameters) so tests
never spawn processes or read the real environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from care.config import CareConfig, HubConfig

logger = logging.getLogger(__name__)

HUB_LOG_FILE = "~/.care/agent-hub.log"


class HubError(RuntimeError):
    """A control-API call failed (4xx/5xx) — message carries the hub's detail."""


class HubUnavailableError(HubError):
    """The hub is down and could not be (auto)started."""


@dataclass(frozen=True)
class HubDeployment:
    """Parsed ``DeploymentInfo`` from the hub control API."""

    name: str
    url: str
    display_name: str
    version: str
    ready: bool
    ready_reason: str
    entity_id: str | None
    channel: str | None
    chain_file: str | None
    source: str
    deployed_at: str
    runs: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "HubDeployment":
        return cls(
            name=str(payload.get("name", "")),
            url=str(payload.get("url", "")),
            display_name=str(payload.get("display_name", "")),
            version=str(payload.get("version", "")),
            ready=bool(payload.get("ready", False)),
            ready_reason=str(payload.get("ready_reason", "")),
            entity_id=payload.get("entity_id"),
            channel=payload.get("channel"),
            chain_file=payload.get("chain_file"),
            source=str(payload.get("source", "")),
            deployed_at=str(payload.get("deployed_at", "")),
            runs=int(payload.get("runs", 0)),
        )


class HubClient:
    """Async client for the hub control API (one short-lived request each)."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------- urls
    def agent_url(self, name: str) -> str:
        return f"{self.base_url}/agents/{name}"

    def docs_url(self, name: str) -> str:
        return f"{self.agent_url(name)}/docs"

    # ------------------------------------------------------------- calls
    async def health(self) -> dict[str, Any] | None:
        """``/healthz`` payload, or ``None`` when the hub is unreachable."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/healthz")
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {"status": str(data)}
        except (httpx.HTTPError, ValueError):
            return None

    async def list_deployments(self) -> list[HubDeployment]:
        payload = await self._request("GET", "/deployments")
        return [HubDeployment.from_payload(item) for item in payload or []]

    async def get_deployment(self, name: str) -> HubDeployment:
        payload = await self._request("GET", f"/deployments/{name}")
        return HubDeployment.from_payload(payload or {})

    async def deploy(self, spec: dict[str, Any]) -> HubDeployment:
        """POST a deployment spec; raises :class:`HubError` with the hub's
        detail on conflicts (409) / unloadable chains (422)."""
        payload = await self._request("POST", "/deployments", json=spec)
        return HubDeployment.from_payload(payload or {})

    async def undeploy(self, name: str) -> None:
        await self._request("DELETE", f"/deployments/{name}")

    async def reload(self, name: str) -> tuple[bool, HubDeployment]:
        payload = await self._request("POST", f"/deployments/{name}/reload") or {}
        deployment = HubDeployment.from_payload(payload.get("deployment") or {})
        return bool(payload.get("reloaded")), deployment

    async def agent_metrics(self, name: str) -> dict[str, Any] | None:
        """The agent's own ``GET /metrics`` (usage + cost, D4). ``None`` if the
        agent doesn't expose it (older build) or the read fails — metrics are
        informational, never fatal."""
        try:
            payload = await self._request("GET", f"/agents/{name}/metrics")
        except (HubError, HubUnavailableError):
            return None
        return payload if isinstance(payload, dict) else None

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method, f"{self.base_url}{path}", **kwargs
                )
        except httpx.HTTPError as exc:
            raise HubUnavailableError(
                f"hub at {self.base_url} is unreachable: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise HubError(
                f"hub {response.status_code}: {_error_detail(response)}"
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:  # JSONDecodeError is a ValueError subclass
            raise HubError(
                f"hub returned a non-JSON body for {method} {path}: {exc}"
            ) from exc


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except ValueError:
        pass
    return response.text[:300]


def hub_env(
    config: CareConfig, *, base: dict[str, str] | None = None
) -> dict[str, str]:
    """Environment for an autostarted hub: CARE settings → ``AGENT_*`` vars.

    Deployed agents then run on the same LLM + Memory the TUI uses. Values
    already present in ``base`` (the caller's environment) are NOT
    overridden — an explicitly exported ``AGENT_LLM_API_KEY`` wins.
    """
    env = dict(os.environ if base is None else base)
    mapping = {
        "AGENT_LLM_API_KEY": config.mage.api_key,
        "AGENT_LLM_MODEL": config.mage.model,
        "AGENT_LLM_BASE_URL": config.mage.base_url,
        "AGENT_MEMORY_URL": config.memory.base_url,
        "AGENT_MEMORY_API_KEY": config.memory.api_key,
        "AGENT_WEB_SEARCH_API_KEY": config.mage.web_search_api_key,
    }
    for key, value in mapping.items():
        if value and key not in env:
            env[key] = str(value)
    return env


async def ensure_hub_running(
    hub_config: HubConfig,
    *,
    env: dict[str, str] | None = None,
    popen: Callable[..., Any] = subprocess.Popen,
    poll_interval: float = 0.3,
) -> HubClient:
    """Return a :class:`HubClient` for a LIVE hub, autostarting it if allowed.

    Probe ``/healthz``; if down and ``autostart`` is on, spawn
    ``agent_server_cmd --port … --state-file …`` as a detached process
    (output appended to ``~/.care/agent-hub.log``) and poll until healthy or
    ``start_timeout`` elapses. Raises :class:`HubUnavailableError` otherwise.
    """
    client = HubClient(hub_config.base_url, timeout=hub_config.timeout)
    if await client.health() is not None:
        return client
    if not hub_config.autostart:
        raise HubUnavailableError(
            f"hub at {hub_config.base_url} is down and [hub].autostart is off — "
            f"start it manually: {' '.join(hub_config.agent_server_cmd)}"
        )

    log_path = Path(HUB_LOG_FILE).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = list(hub_config.agent_server_cmd) + [
        "--port",
        str(hub_config.port),
        "--state-file",
        str(Path(hub_config.state_file).expanduser()),
    ]
    logger.info("hub: autostarting %s (log: %s)", " ".join(command), log_path)
    try:
        with open(log_path, "ab") as log_file:
            popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env if env is not None else os.environ.copy(),
            )
    except OSError as exc:
        hint = ""
        if isinstance(exc, FileNotFoundError):
            # The agent server is a separate package; deploying needs it on PATH.
            hint = (
                f" — `{hub_config.agent_server_cmd[0]}` not found; install the "
                f"agent server with `pip install maestro-care[deploy]` "
                f"(or `pip install carl-agent-server`)"
            )
        raise HubUnavailableError(
            f"could not start the hub ({' '.join(command)}): {exc}{hint}"
        ) from exc

    deadline = time.monotonic() + hub_config.start_timeout
    while time.monotonic() < deadline:
        if await client.health() is not None:
            logger.info("hub: up at %s", hub_config.base_url)
            return client
        await asyncio.sleep(poll_interval)
    raise HubUnavailableError(
        f"hub did not become healthy within {hub_config.start_timeout:.0f}s — "
        f"see {log_path}"
    )
