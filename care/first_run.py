"""First-run config + connectivity validation (TODO §2 P0).

CARE's TUI `SettingsScreen` runs the first time a user opens
the app without `~/.config/care/config.toml` present. The
screen collects MAGE / Memory / Platform credentials, hits each
service with a small probe, and writes the TOML if everything's
reachable.

This module is the data layer:

* :class:`ProbeResult` — frozen per-service probe outcome
  (ok / latency / error / version when available).
* :class:`FirstRunReport` — aggregates three probes + tells the
  screen whether all services are reachable.
* :func:`probe_memory`, :func:`probe_mage`, :func:`probe_platform`
  — async functions that hit each service and return a result.
  Friendly to call from a Textual worker.
* :func:`write_initial_config(path, config)` — atomic TOML
  writer so a crash mid-write doesn't truncate the file.

Duck-typed against CARE's facades — :class:`CareMemory` /
:class:`CarePlatform` / :func:`build_mage_generator` are the
production paths, but the probes accept anything that exposes
the relevant call. Tests inject stubs so no SDK / HTTP / real
service is touched.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from care.config import CareConfig


ProbeStatus = Literal["ok", "skipped", "failed"]
"""Per-service outcome:

* ``ok`` — service responded; ``latency_ms`` populated.
* ``skipped`` — config wasn't filled in (e.g. no API key), so
  the probe didn't even try. ``error`` carries the explanation.
* ``failed`` — probe attempt errored. ``error`` carries the
  message; ``latency_ms`` may still be set when the failure
  happened mid-request.
"""


@dataclass(frozen=True)
class ProbeResult:
    """One service's connectivity probe.

    Frozen so the report flows through Textual messages /
    persisted draft state without defensive copies.

    Fields:
        service: ``"memory"`` / ``"mage"`` / ``"platform"``.
        status: See :class:`ProbeStatus`.
        latency_ms: Round-trip wall-clock when measured.
            ``None`` for skipped probes.
        error: Failure / skip reason. Empty when ``status="ok"``.
        detail: Free-form details — Memory's health JSON,
            MAGE's resolved model name, Platform's reported
            version. Empty dict when nothing useful surfaced.
    """

    service: Literal["memory", "mage", "platform"]
    status: ProbeStatus
    latency_ms: float | None = None
    error: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FirstRunReport:
    """Aggregate of all three probes."""

    memory: ProbeResult
    mage: ProbeResult
    platform: ProbeResult

    @property
    def all_ok(self) -> bool:
        """``True`` when every *required* probe returned
        ``"ok"``.

        Memory + MAGE are required to use CARE productively
        (library reads, generation) so they gate this flag.
        Platform powers the evolution flow only — its
        ``"skipped"`` / ``"failed"`` status is surfaced via
        :attr:`platform_ok` / :meth:`format_text` but does
        NOT pull ``all_ok`` down to False on its own.
        """
        return self.memory.status == "ok" and self.mage.status == "ok"

    @property
    def platform_ok(self) -> bool:
        """``True`` when the Platform probe returned ``"ok"``.

        Read this when a flow specifically needs Platform
        (e.g. EvolutionScreen). The Settings "Save" gate uses
        :attr:`all_ok`, which treats Platform as optional."""
        return self.platform.status == "ok"

    @property
    def any_failed(self) -> bool:
        """``True`` when any probe explicitly failed (skipped
        doesn't count). Useful for "fix and retry" UI."""
        return any(
            r.status == "failed"
            for r in (self.memory, self.mage, self.platform)
        )

    def format_text(self) -> str:
        """Human-readable summary for the screen footer / CLI."""
        lines: list[str] = []
        for result in (self.memory, self.mage, self.platform):
            badge = {
                "ok": "✓",
                "skipped": "·",
                "failed": "✗",
            }.get(result.status, "?")
            line = f"{badge} {result.service}"
            if result.latency_ms is not None:
                line += f" ({result.latency_ms:.0f}ms)"
            if result.error:
                line += f" — {result.error}"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


async def probe_memory(
    config: CareConfig,
    *,
    memory_factory: Any = None,
) -> ProbeResult:
    """Hit Memory's health endpoint via :class:`CareMemory`.

    Args:
        config: Caller's full :class:`CareConfig`.
        memory_factory: Callable returning a CareMemory-like
            object (anything with ``health_check()``). ``None``
            uses :meth:`CareMemory.from_config`. Tests inject
            a stub.

    Returns:
        :class:`ProbeResult` with ``service="memory"``.
    """
    if not config.memory.base_url:
        return ProbeResult(
            service="memory",
            status="skipped",
            error="memory.base_url is empty",
        )
    start = time.monotonic()
    try:
        memory = _make_memory(config, memory_factory)
        # `health_check` is sync on the SDK side; run it in a
        # thread so the Textual event loop doesn't block on
        # network I/O.
        detail = await asyncio.to_thread(memory.health_check)
        latency = (time.monotonic() - start) * 1000
    except Exception as exc:  # noqa: BLE001
        latency = (time.monotonic() - start) * 1000
        return ProbeResult(
            service="memory",
            status="failed",
            latency_ms=latency,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ProbeResult(
        service="memory",
        status="ok",
        latency_ms=latency,
        detail=detail if isinstance(detail, dict) else {"raw": detail},
    )


def _auth_status_code(exc: Exception) -> int | None:
    """Best-effort HTTP status off an openai-SDK / httpx exception."""
    code = getattr(exc, "status_code", None)
    if code is None:
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


async def probe_mage(
    config: CareConfig,
    *,
    client_factory: Any = None,
    deep: bool = False,
) -> ProbeResult:
    """Validate MAGE-side LLM connectivity.

    The probe always builds an LLM client via
    :func:`care.runtime.build_llm_client` and inspects its
    ``model`` / ``base_url`` — enough to confirm the
    credentials parse and the SDK is installed.

    When ``deep=True`` (the ``care doctor`` path) it ALSO makes a
    lightweight authenticated ``/models`` round-trip so an expired
    or invalid token is caught — without it every ``care generate``
    would 403 while the probe stayed green. Boot / SettingsScreen
    callers leave ``deep=False`` to keep the dev-loop fast.

    Auth failures (401/403) report ``failed``; a backend that
    simply doesn't expose ``/models`` (404/405) still reports
    ``ok`` (credentials parsed, endpoint reachable) with a note.

    Args:
        config: Caller's :class:`CareConfig`.
        client_factory: Callable returning an LLM client given
            a ``MageConfig``. ``None`` uses
            :func:`care.runtime.build_llm_client`. Tests inject
            a stub.
        deep: When ``True``, perform the authenticated round-trip.

    Returns:
        :class:`ProbeResult` with ``service="mage"``.
    """
    if not config.mage.api_key:
        return ProbeResult(
            service="mage",
            status="skipped",
            error="mage.api_key is empty",
        )
    start = time.monotonic()
    try:
        factory = client_factory or _default_llm_factory
        client = await asyncio.to_thread(factory, config.mage)
        latency = (time.monotonic() - start) * 1000
    except Exception as exc:  # noqa: BLE001
        latency = (time.monotonic() - start) * 1000
        return ProbeResult(
            service="mage",
            status="failed",
            latency_ms=latency,
            error=f"{type(exc).__name__}: {exc}",
        )

    detail: dict[str, Any] = {
        "base_url": str(
            getattr(client, "base_url", "")
            or config.mage.base_url
            or "",
        ),
        "model": config.mage.model or "",
    }

    if deep:
        try:
            await asyncio.to_thread(lambda: client.models.list())
            detail["round_trip"] = "models.list ok"
        except Exception as exc:  # noqa: BLE001
            latency = (time.monotonic() - start) * 1000
            code = _auth_status_code(exc)
            if code in (401, 403):
                return ProbeResult(
                    service="mage",
                    status="failed",
                    latency_ms=latency,
                    error=(
                        f"authentication failed ({code}) — the MAGE API "
                        f"key/token is expired or invalid; run `care init` "
                        f"or refresh it. ({type(exc).__name__})"
                    ),
                )
            if code in (404, 405):
                # /models not exposed by this backend — creds parsed
                # and the endpoint answered, so call it ok with a note.
                detail["round_trip"] = (
                    f"/models not exposed ({code}); credentials accepted"
                )
            else:
                return ProbeResult(
                    service="mage",
                    status="failed",
                    latency_ms=latency,
                    error=f"{type(exc).__name__}: {exc}",
                )

    latency = (time.monotonic() - start) * 1000
    return ProbeResult(
        service="mage",
        status="ok",
        latency_ms=latency,
        detail=detail,
    )


_DEFAULT_PLATFORM_URL = "http://localhost:8000"


async def probe_platform(
    config: CareConfig,
    *,
    platform_factory: Any = None,
) -> ProbeResult:
    """Hit Platform's health endpoint via :class:`CarePlatform`.

    Platform is optional — when the user hasn't opted in to a
    specific deployment (no api_key AND base_url left at the
    default ``http://localhost:8000``) the probe skips without
    firing a network call so the SettingsScreen / `care
    validate` report doesn't show a noisy "connection refused"
    line on every boot for the common no-Platform setup.
    """
    if not config.platform.base_url:
        return ProbeResult(
            service="platform",
            status="skipped",
            error="platform.base_url is empty",
        )
    opted_in = bool(config.platform.api_key) or (
        config.platform.base_url != _DEFAULT_PLATFORM_URL
    )
    if not opted_in:
        return ProbeResult(
            service="platform",
            status="skipped",
            error=(
                "platform not configured (optional — set "
                "CARE_PLATFORM__API_KEY or point "
                "CARE_PLATFORM__BASE_URL at your deployment "
                "to enable the evolution flow)"
            ),
        )
    start = time.monotonic()
    try:
        platform = _make_platform(config, platform_factory)
        detail = await asyncio.to_thread(platform.health_check)
        latency = (time.monotonic() - start) * 1000
    except Exception as exc:  # noqa: BLE001
        latency = (time.monotonic() - start) * 1000
        return ProbeResult(
            service="platform",
            status="failed",
            latency_ms=latency,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ProbeResult(
        service="platform",
        status="ok",
        latency_ms=latency,
        detail=detail if isinstance(detail, dict) else {"raw": detail},
    )


async def run_all_probes(
    config: CareConfig,
    *,
    memory_factory: Any = None,
    client_factory: Any = None,
    platform_factory: Any = None,
    deep: bool = False,
) -> FirstRunReport:
    """Run all three probes concurrently and return the
    aggregated report.

    The future SettingsScreen calls this in its `validate`
    handler; the screen reads :attr:`FirstRunReport.all_ok` to
    decide whether to enable the "Save & continue" button.

    ``deep=True`` (the ``care doctor`` path) makes the MAGE probe
    perform an authenticated ``/models`` round-trip so expired
    tokens are caught; boot/validate leave it ``False`` for speed.
    """
    memory_task = probe_memory(config, memory_factory=memory_factory)
    mage_task = probe_mage(config, client_factory=client_factory, deep=deep)
    platform_task = probe_platform(config, platform_factory=platform_factory)
    memory, mage, platform = await asyncio.gather(
        memory_task, mage_task, platform_task
    )
    return FirstRunReport(memory=memory, mage=mage, platform=platform)


# ---------------------------------------------------------------------------
# Config writer
# ---------------------------------------------------------------------------


class FirstRunConfigError(RuntimeError):
    """Raised when the config writer can't persist the TOML —
    parent directory unwritable, refusing to overwrite, etc."""


def write_initial_config(
    path: Path | str,
    config: CareConfig,
    *,
    overwrite: bool = False,
    store_secrets: bool = True,
) -> Path:
    """Atomically serialise ``config`` to ``path`` as TOML.

    Args:
        path: Destination file. Parent directories are
            auto-created (matches XDG conventions).
            Tilde-expanded.
        config: The :class:`CareConfig` to persist.
        overwrite: When ``False`` (default), refuses to
            replace an existing file — the wizard is for
            first-run only. Pass ``True`` to overwrite (rare
            — most flows use the regular config editor).
        store_secrets: When ``True`` (default, §1 P1),
            offloads every non-empty `*_api_key` literal to
            the detected keystore + writes back the
            ``keystore://service/key`` URL instead. Pass
            ``False`` from tests / migration scripts that
            want the literal-on-disk shape.

    Returns:
        Resolved absolute path the file was written to.

    Raises:
        FirstRunConfigError: Path already exists +
            ``overwrite=False``, or the write failed.
    """
    target = Path(str(path)).expanduser()
    if target.exists() and not overwrite:
        raise FirstRunConfigError(
            f"refusing to overwrite existing config at {target} "
            "(pass overwrite=True to replace)"
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FirstRunConfigError(
            f"could not create parent directory {target.parent}: {exc}"
        ) from exc

    body = _render_toml(config, store_secrets=store_secrets)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=".care-config-",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fp:
            fp.write(body)
        os.replace(tmp_name, target)
    except OSError as exc:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise FirstRunConfigError(
            f"failed to write {target}: {exc}"
        ) from exc
    return target.resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(config: CareConfig, factory: Any) -> Any:
    if factory is not None:
        return factory(config)
    from care.memory import CareMemory

    return CareMemory.from_config(config)


def _make_platform(config: CareConfig, factory: Any) -> Any:
    if factory is not None:
        return factory(config)
    from care.platform import CarePlatform

    return CarePlatform.from_config(config)


def _default_llm_factory(mage_config: Any) -> Any:
    from care.runtime.llm_client import build_llm_client

    return build_llm_client(mage_config)


def _render_toml(
    config: CareConfig, *, store_secrets: bool = False,
) -> str:
    """Serialise ``CareConfig`` as TOML.

    Python's stdlib doesn't ship a TOML writer (only the
    reader). For the small, well-typed shape CARE uses, a
    hand-rolled writer keeps the dependency surface tight
    (the optional ``tomli-w`` package was proposed in §9 P2
    but deferred). Sorted sections + skipped `None` values
    produce diff-friendly output the user can re-edit.

    When ``store_secrets`` is True (§1 P1), every non-empty
    `*_api_key` literal is offloaded to the detected keystore
    via :func:`care.config._store_api_key_secrets`. The
    in-memory ``config`` is untouched; the rewritten dict
    payload is what gets serialised.
    """
    sections: list[tuple[str, dict[str, Any]]] = []
    # Order matches `.env.example` so users who toggle between
    # the two see the same layout.
    payload = config.model_dump(mode="python")
    if store_secrets:
        from care.config import _store_api_key_secrets

        _store_api_key_secrets(payload)
    for section in ("mage", "memory", "platform", "sandbox", "tools", "telemetry", "defaults"):
        if section in payload:
            sections.append((section, payload[section]))
    chunks: list[str] = []
    for section, values in sections:
        chunks.append(f"[{section}]")
        for key, value in values.items():
            if value is None:
                continue
            chunks.append(f"{key} = {_format_toml_value(value)}")
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        # Inline-table form for one-level nested dicts (we
        # don't currently use this, but kept for completeness).
        pairs = ", ".join(
            f"{k} = {_format_toml_value(v)}" for k, v in value.items()
        )
        return "{ " + pairs + " }"
    if value is None:
        return "\"\""
    # Strings — escape backslash + quote.
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


__all__ = [
    "FirstRunConfigError",
    "FirstRunReport",
    "ProbeResult",
    "ProbeStatus",
    "probe_mage",
    "probe_memory",
    "probe_platform",
    "run_all_probes",
    "write_initial_config",
]
