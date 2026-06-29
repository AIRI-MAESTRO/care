"""On-demand tool synthesis with reuse, caching, and Memory persistence.

A capable MAGE planner emits a ``tool`` step naming a tool by intent —
``get_current_weather``, ``weather_api`` — that nobody registered, and
CARL aborts with ``Tool 'X' not registered in context``. This module
makes that self-healing.

Pipeline, per missing tool (``synthesize_missing_tools``):

1. **Disk cache** — a previously-synthesised ``<name>.json`` under
   ``CareConfig.tools.synthesized_tools_path``. Reuse if present.
2. **Memory** — when Memory is configured, look the tool up by name
   (``agent_skill`` entity). Reuse + populate the local cache if found.
3. **Synthesize** — ask the LLM for a small stdlib-only implementation,
   register it as a sandboxed callable, write it to the disk cache, and
   (best-effort) save it to Memory for cross-session/-machine reuse.

Reuse also happens *before* generation: cached tools are registered onto
every context at startup (:func:`register_cached_tools`) and advertised
to MAGE (:func:`cached_tool_specs`), so a tool synthesised once is simply
present next time — the planner reuses its name instead of inventing a
new one.

Safety: generated code only ever runs inside the Docker sandbox
(:func:`care.builtin_tools.run_python_source`), never in CARE's process.
Every step is defensive — failures are reported, never raised.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from care.builtin_tools import run_python_source

_log = logging.getLogger("care.tool_synthesis")

#: Memory tag marking an entity as a CARE-synthesised runnable tool.
SYNTH_TAG = "care:synthesized-tool"

#: How many times a failing self-test may regenerate-and-retry the tool
#: before we give up and register the last attempt anyway.
_MAX_HEALS = 2

_SYNTH_PROMPT = """You are writing ONE Python tool function for an autonomous agent.

Define exactly one NON-async function with this signature:

    def {name}(**kwargs) -> str:

Goal of the tool: {description}
Expected keys in kwargs: {params}

Hard rules:
- Use ONLY the Python standard library (urllib.request, urllib.parse, json, datetime, math). NO third-party packages — `requests`, `httpx`, pip installs are unavailable.
- For live external data, call a KEYLESS public API. Examples:
    weather  -> https://wttr.in/<city>?format=j1   or  https://api.open-meteo.com/v1/forecast?...
    geocode  -> https://geocoding-api.open-meteo.com/v1/search?name=<city>
  Never use an endpoint that requires an API key.
- ALWAYS send a browser-like User-Agent — many keyless APIs return HTTP 403 to urllib's default agent. Use:
    req = urllib.request.Request(url, headers={{"User-Agent": "Mozilla/5.0", "Accept": "application/json"}})
    with urllib.request.urlopen(req, timeout=10) as r: data = json.load(r)
  If a request still fails with 403/404, try a DIFFERENT keyless provider rather than the same one.
- Read inputs defensively: city = kwargs.get("city") or kwargs.get("location") or "".
- Return a concise, human-readable STRING. On any error, return a short string starting with "error:".
- Do NOT print, do NOT call the function, do NOT add markdown fences.
- Keep it under ~45 lines.
- After the function, output ONE final line exactly of the form `#SELFTEST# <json>` where <json> is a realistic sample kwargs dict for a single call (e.g. `#SELFTEST# {{"city": "London"}}`). This line is used to test-run your function and is then discarded.

Output the Python code now:"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def synthesize_missing_tools(
    chain_dict: dict[str, Any],
    context: Any,
    *,
    api: Any,
    config: Any,
    notify: Callable[[str], None] | None = None,
) -> dict[str, list]:
    """Ensure every tool the chain references is registered on ``context``.

    Returns a report ``{"created": [names], "reused": [(name, source)],
    "failed": [(name, reason)]}`` — ``reused``/``created`` carry the tool
    name; ``failed`` carries ``(name, reason)``. Always safe to call.

    ``notify`` (optional) receives short progress strings for the build
    pipeline (search → synthesise → self-test → heal) so a UI can show the
    multi-step tool creation live instead of a silent gap before execution.
    """
    report: dict[str, list] = {"created": [], "reused": [], "failed": []}

    def _say(msg: str) -> None:
        if notify is None:
            return
        try:
            notify(msg)
        except Exception:  # noqa: BLE001 — UI feedback is best-effort
            pass

    tools_cfg = getattr(config, "tools", None)
    if tools_cfg is None or not getattr(tools_cfg, "auto_synthesize_tools", True):
        return report
    if not getattr(tools_cfg, "enable_code_exec", True):
        return report
    sandbox_cfg = getattr(config, "sandbox", None)
    if getattr(sandbox_cfg, "kind", "docker") != "docker":
        _log.info("tool synthesis skipped: sandbox kind != docker")
        return report
    if not hasattr(context, "register_tool"):
        return report

    generate = _resolve_llm(api)
    registered = set(getattr(context, "_tool_registry", {}) or {})
    missing = _missing_tool_steps(chain_dict, registered)
    if not missing:
        return report
    _say("🔧 " + str(len(missing)) + " tool(s) not registered — building: "
         + ", ".join(n for n, _, _ in missing))

    memory = _get_memory(config) if getattr(tools_cfg, "save_synthesized_to_memory", True) else None
    _ttl = int(getattr(tools_cfg, "cached_tool_health_ttl_s", 86400) or 0)

    for name, params, description in missing:
        # 1. disk cache --------------------------------------------------
        cached = _load_cached_tool(tools_cfg, name)
        if cached and cached.get("source"):
            # Health-check before reuse: a tool that worked once can rot (a
            # stale endpoint, or one hardcoded to a single input). If it now
            # errors, fall through and re-synthesise instead of running it.
            healthy: bool | None = True
            if getattr(tools_cfg, "verify_cached_tools", True):
                healthy = _fresh_verdict(cached, _ttl)
                if healthy is None:
                    _say(f"🧪 '{name}': verifying cached tool…")
                    healthy = await _verify_tool_record(
                        name, cached, sandbox_cfg, tools_cfg, generate=generate
                    )
            if healthy is not False and _register(
                context, name, cached["source"], sandbox_cfg, tools_cfg
            ):
                report["reused"].append(name)
                continue
            if healthy is False:
                _say(f"   ↳ cached '{name}' failed health-check — re-synthesising")
                _log.info("tool synthesis: cached %r failed health-check; re-synthesising", name)

        # 2. Memory ------------------------------------------------------
        if memory is not None:
            src = await _memory_find_tool(memory, name)
            if src:
                mem_ok: bool | None = True
                if getattr(tools_cfg, "verify_cached_tools", True):
                    _say(f"🧪 '{name}': verifying tool from Memory…")
                    mem_ok = await _verify_tool_record(
                        name,
                        {"source": src, "params": params, "description": description},
                        sandbox_cfg, tools_cfg, generate=generate,
                    )
                if mem_ok is not False and _register(context, name, src, sandbox_cfg, tools_cfg):
                    _save_cached_tool(
                        tools_cfg, name, src, params, description,
                        health=(
                            {"ok": True, "checked_at": time.time(), "detail": ""}
                            if mem_ok else None
                        ),
                    )
                    report["reused"].append(name)
                    continue

        # 3. synthesize — web-grounded so codegen targets a real endpoint
        #    instead of one the LLM guessed from stale training.
        if generate is None:
            report["failed"].append((name, "no LLM client available"))
            continue
        _say(f"🔎 '{name}': searching the web for a real API…")
        research = await _discover_api(description, params, tools_cfg)
        if research:
            _say(f"🛠 '{name}': synthesising (grounded on web research)…")
            _log.info(
                "tool synthesis: grounding %r on web research (%d chars)",
                name, len(research),
            )
        else:
            _say(f"🛠 '{name}': synthesising…")
        source, sample_args = await _generate_source(
            generate, name, params, description, research=research,
        )
        if not source:
            report["failed"].append((name, "code generation failed"))
            continue
        # 3b. self-heal LOOP — run the tool with the model's sample inputs;
        #     on a runtime "error:" (e.g. wrong endpoint path / 404) regenerate
        #     with the failure fed back and RE-TEST, up to ``_MAX_HEALS`` times.
        #     Re-testing is what catches a heal that's still broken (a single
        #     no-retest pass let a 404'ing forex tool through). Only runs when
        #     the model gave sample args to test with.
        probe: str | None = None
        ok: bool | None = None
        if getattr(tools_cfg, "self_test_synthesized_tools", True) and sample_args is not None:
            for _round in range(_MAX_HEALS + 1):
                _say(f"🧪 '{name}': self-testing…")
                probe = await _run_selftest(name, source, sample_args, sandbox_cfg, tools_cfg)
                ok = probe is None or not probe.strip().lower().startswith("error:")
                if ok or _round == _MAX_HEALS:
                    if not ok:
                        _say(f"   ↳ '{name}' still failing after {_MAX_HEALS} heal(s): {probe[:80]}")
                        _log.info("tool synthesis: %r unhealed after %d tries: %s", name, _MAX_HEALS, probe[:120])
                    break
                _say(f"   ↳ '{name}' self-test failed — healing ({_round + 1}/{_MAX_HEALS})…")
                _log.info("tool synthesis: %r self-test failed (%s); healing", name, probe[:120])
                healed, _ = await _generate_source(
                    generate, name, params, description,
                    research=research, feedback=probe[:300], attempts=1,
                )
                if not healed:
                    break
                source = healed
        if not _register(context, name, source, sandbox_cfg, tools_cfg):
            report["failed"].append((name, "registration failed"))
            continue
        _save_cached_tool(
            tools_cfg, name, source, params, description,
            sample_args=sample_args if isinstance(sample_args, dict) else None,
            health=(
                {"ok": bool(ok), "checked_at": time.time(), "detail": "" if ok else (probe or "")[:200]}
                if ok is not None
                else None
            ),
        )
        if memory is not None:
            await _memory_save_tool(memory, name, description, source, params)
        report["created"].append(name)

    return report


# ---------------------------------------------------------------------------
# Startup reuse: register + advertise cached tools
# ---------------------------------------------------------------------------


def register_cached_tools(context: Any, config: Any) -> list[str]:
    """Register every disk-cached synthesised tool onto ``context``.

    Called at context build so a tool synthesised in an earlier run is
    present from the start — the chain never hits "not registered" for it
    again. Returns the names registered."""
    tools_cfg = getattr(config, "tools", None)
    if tools_cfg is None or not getattr(tools_cfg, "auto_synthesize_tools", True):
        return []
    if not hasattr(context, "register_tool"):
        return []
    sandbox_cfg = getattr(config, "sandbox", None)
    verify = bool(getattr(tools_cfg, "verify_cached_tools", True))
    ttl = int(getattr(tools_cfg, "cached_tool_health_ttl_s", 86400) or 0)
    out: list[str] = []
    for name, record in _load_all_cached(tools_cfg).items():
        source = record.get("source")
        if not source:
            continue
        # With verification on, only eagerly register tools whose health is
        # FRESH + OK. Unknown / stale / failing ones are deliberately left
        # unregistered so the async synthesize_missing_tools path health-checks
        # them (and re-synthesises a broken one) before the chain runs.
        if verify and _fresh_verdict(record, ttl) is not True:
            continue
        if _register(context, name, source, sandbox_cfg, tools_cfg):
            out.append(name)
    if out:
        _log.info("registered %d cached synthesised tools: %s", len(out), ", ".join(out))
    return out


def cached_tool_specs(config: Any) -> list[dict[str, Any]]:
    """MAGE-shaped descriptors for cached synthesised tools so the
    planner reuses their exact names instead of inventing new ones."""
    tools_cfg = getattr(config, "tools", None)
    if tools_cfg is None or not getattr(tools_cfg, "auto_synthesize_tools", True):
        return []
    verify = bool(getattr(tools_cfg, "verify_cached_tools", True))
    ttl = int(getattr(tools_cfg, "cached_tool_health_ttl_s", 86400) or 0)
    specs: list[dict[str, Any]] = []
    for name, record in _load_all_cached(tools_cfg).items():
        # Don't tempt the planner with a tool we've already proven broken.
        if verify and _fresh_verdict(record, ttl) is False:
            continue
        desc = (record.get("description") or "").strip()
        params = record.get("params") or []
        sig = ", ".join(params) if params else "**kwargs"
        specs.append(
            {
                "name": name,
                "source": "care:synthesized",
                "description": (
                    f"{name}({sig}) -> str. {desc} "
                    "(previously synthesised, ready to reuse)."
                ).strip(),
                "tags": ["synthesized", "external"],
            }
        )
    return specs


def bundled_tools_for_chain(chain_dict: dict[str, Any], config: Any) -> list[dict[str, Any]]:
    """Synthesized-tool definitions for the tools a chain references — to SHIP
    with a deployment (``DeploymentSpec.extra_tools``) so the hub can run a tool
    it doesn't bundle in its builtin set.

    Returns ``[{name, source, params, description}]`` for each referenced
    ``tool`` step whose ``tool_name`` is found in the local synth cache. Builtins
    (never cached) and tools without a cached source are skipped — so the result
    is exactly the locally-synthesized tools that must travel to the hub."""
    tools_cfg = getattr(config, "tools", None)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in chain_dict.get("steps") or []:
        if not isinstance(step, dict) or str(step.get("step_type") or "").lower() != "tool":
            continue
        name = _cfg_field(step.get("step_config"), "tool_name") or step.get("tool_name")
        if not name or name in seen:
            continue
        seen.add(name)
        record = _load_cached_tool(tools_cfg, name)
        if record and record.get("source"):
            out.append(
                {
                    "name": name,
                    "source": record["source"],
                    "params": record.get("params") or [],
                    "description": record.get("description") or "",
                }
            )
    return out


def bundled_tools_to_python_code(bundled: list[dict[str, Any]]) -> str:
    """Merge synthesized tool sources into one ``custom_tools.py`` module.

    Platform's experiment builder writes this string to
    ``problems/exp_*/custom_tools.py``; ``helper._load_custom_tools``
    registers each top-level callable by name.
    """
    if not bundled:
        return ""
    parts: list[str] = [
        '"""CARE synthesized tools — shipped to Platform as custom_tools.py."""',
        "from __future__ import annotations",
        "",
    ]
    for tool in bundled:
        name = str(tool.get("name") or "").strip()
        source = str(tool.get("source") or "").strip()
        if not name or not source:
            continue
        desc = str(tool.get("description") or "").strip()
        if desc:
            parts.append(f"# Tool: {name} — {desc}")
        else:
            parts.append(f"# Tool: {name}")
        parts.append(source)
        parts.append("")
    body = "\n".join(parts).strip()
    return f"{body}\n" if body else ""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _cfg_field(cfg: Any, key: str) -> Any:
    """Read a field from a step's ``step_config`` whether it's still a plain
    dict (before :meth:`ReasoningChain.from_dict`) or has been turned into a
    typed ``*StepConfig`` object (``from_dict`` mutates the chain dict IN
    PLACE). Without this, calling synthesis *after* ``from_dict`` reads an
    object with ``.get`` missing → no tool detected → the chain dies with
    "Tool '<name>' not registered"."""
    if isinstance(cfg, dict):
        return cfg.get(key)
    return getattr(cfg, key, None)


def _missing_tool_steps(
    chain_dict: dict[str, Any],
    registered: set[str],
) -> list[tuple[str, list[str], str]]:
    """``(tool_name, param_names, description)`` for every ``tool`` step
    whose tool isn't already registered. De-duplicated by name. Robust to a
    ``step_config`` that is either a dict or a typed object (see
    :func:`_cfg_field`)."""
    out: list[tuple[str, list[str], str]] = []
    seen: set[str] = set()
    for step in chain_dict.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if str(step.get("step_type") or "").lower() != "tool":
            continue
        cfg = step.get("step_config")
        tool_name = _cfg_field(cfg, "tool_name") or step.get("tool_name")
        if not tool_name or tool_name in registered or tool_name in seen:
            continue
        seen.add(tool_name)
        mapping = _cfg_field(cfg, "input_mapping") or step.get("input_mapping") or {}
        params = list(mapping.keys()) if isinstance(mapping, dict) else []
        title = str(step.get("title") or "").strip()
        aim = str(step.get("aim") or _cfg_field(cfg, "aim") or step.get("stage_action") or "").strip()
        out.append((tool_name, params, ". ".join(p for p in (title, aim) if p)))
    return out


# ---------------------------------------------------------------------------
# Web-grounded discovery — find a real API before generating
# ---------------------------------------------------------------------------


async def _discover_api(
    description: str,
    params: list[str],
    tools_cfg: Any,
) -> str | None:
    """Web-search for a real keyless public API for this capability.

    Synthesis otherwise guesses an endpoint from the model's training data,
    which is stale/wrong for anything outside the prompt's hardcoded
    weather/geocode examples. Feeding real search results into the codegen
    prompt grounds the tool on a live endpoint. Returns a short research
    digest (provider answer + top hits) or ``None`` when grounding is off /
    ``web_search`` isn't configured / the search fails. Never raises.
    """
    if not getattr(tools_cfg, "ground_synthesis_with_web_search", True):
        return None
    api_key = getattr(tools_cfg, "web_search_api_key", None)
    if not api_key:
        return None
    provider = getattr(tools_cfg, "web_search_provider", "tavily") or "tavily"
    try:
        from care.builtin_tools import _make_web_search

        ws = _make_web_search(provider, api_key, 5)
        hint = ", ".join(p for p in params if p)
        query = (
            "free public REST API, no API key required, JSON response, for: "
            f"{description}. Give the exact endpoint URL and an example request"
            + (f" (inputs: {hint})" if hint else "")
        )
        out = await ws(query)
    except Exception as exc:  # noqa: BLE001
        _log.info("api discovery search failed for %r: %s", description, exc)
        return None
    if not out or out.lower().startswith(("web_search error", "web_search:")):
        return None
    return out[:1600]


# ---------------------------------------------------------------------------
# Codegen + sandboxed callable
# ---------------------------------------------------------------------------


def _split_selftest(source: str) -> tuple[str, dict[str, Any] | None]:
    """Pull a trailing ``#SELFTEST# {json}`` line out of generated source.

    Returns ``(clean_source, sample_kwargs)``. ``sample_kwargs`` is the
    realistic call the model suggested for the self-heal test, or ``None``
    when absent/unparseable."""
    import re

    sample: dict[str, Any] | None = None
    m = re.search(r"^[ \t]*#SELFTEST#[ \t]*(\{.*\})[ \t]*$", source, re.MULTILINE)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                sample = parsed
        except Exception:  # noqa: BLE001
            sample = None
        source = source[: m.start()] + source[m.end():]
    return source.strip(), sample


async def _generate_source(
    generate: Callable[[str], Any],
    name: str,
    params: list[str],
    description: str,
    *,
    attempts: int = 2,
    research: str | None = None,
    feedback: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """LLM-generate a tool implementation. Returns ``(source, sample_args)``.

    ``source`` is the function code (``#SELFTEST#`` sentinel stripped) or
    ``None``; ``sample_args`` is the realistic kwargs the model suggested for
    the self-heal test (or ``None``). ``research`` (a web digest of real
    public APIs from :func:`_discover_api`) and ``feedback`` (a prior live-test
    failure to fix) are injected into the prompt when present."""
    base = _SYNTH_PROMPT.format(
        name=name,
        params=", ".join(params) if params else "(infer from the goal)",
        description=description or name,
    )
    if research:
        base += (
            "\n\nWEB RESEARCH — real public APIs found online for THIS exact "
            "capability. PREFER a keyless JSON endpoint from here over your own "
            "recollection (your training data may be stale or the endpoint may "
            "have moved):\n"
            f"{research}\n"
        )
    if feedback:
        base += (
            "\n\nYOUR PREVIOUS ATTEMPT FAILED a live test run with:\n"
            f"{feedback}\n"
            "The endpoint URL/path or response parsing is likely wrong — fix it "
            "(use the exact URL from the research; verify the path + query keys).\n"
        )
    for i in range(attempts):
        prompt = base if i == 0 else base + "\n\nReturn the function source, then the #SELFTEST# line."
        try:
            raw = await generate(prompt)
        except Exception as exc:  # noqa: BLE001
            _log.warning("tool synthesis: codegen error for %r: %s", name, exc)
            continue
        source = _strip_code_fences(str(raw or ""))
        if f"def {name}" in source:
            return _split_selftest(source)
    _log.warning("tool synthesis: generated code never defined def %s", name)
    return None, None


def _make_synth_callable(
    name: str,
    source: str,
    sandbox_cfg: Any,
    tools_cfg: Any,
) -> Callable[..., Any]:
    """Wrap ``source`` as an async tool that runs it in the Docker sandbox,
    injecting the call's kwargs and returning the function's result."""
    timeout = int(getattr(tools_cfg, "code_exec_timeout", 60) or 60)
    max_chars = int(getattr(tools_cfg, "fetch_url_max_chars", 4000) or 4000)

    async def _synth_tool(**kwargs: Any) -> str:
        args_repr = repr(json.dumps(kwargs, ensure_ascii=False, default=str))
        trailer = (
            "\n\nif __name__ == '__main__':\n"
            "    import json as _j, sys as _sys\n"
            f"    _args = _j.loads({args_repr})\n"
            f"    _r = {name}(**_args)\n"
            "    _sys.stdout.write(_r if isinstance(_r, str) "
            "else _j.dumps(_r, ensure_ascii=False, default=str))\n"
        )
        out = await run_python_source(
            source + trailer, sandbox_cfg, timeout=timeout, max_chars=max_chars
        )
        return _strip_exit_prefix(out)

    return _synth_tool


async def _run_selftest(
    name: str,
    source: str,
    sample_args: dict[str, Any],
    sandbox_cfg: Any,
    tools_cfg: Any,
) -> str | None:
    """Run a freshly-synthesised tool once with the model's sample inputs.

    Returns the tool's output string, or ``None`` if the self-test harness
    itself failed (sandbox down, etc.) — so callers don't mistake an infra
    hiccup for a tool ``error:`` and waste a heal attempt."""
    try:
        fn = _make_synth_callable(name, source, sandbox_cfg, tools_cfg)
        return await fn(**sample_args)
    except Exception as exc:  # noqa: BLE001
        _log.info("tool synthesis: self-test harness error for %r: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Health-check — verify a cached/reused tool still works before using it
# ---------------------------------------------------------------------------


def _fresh_verdict(record: dict[str, Any], ttl_s: int) -> bool | None:
    """A cached tool's health verdict if still fresh, else ``None``.

    Returns the stored ``ok`` bool when the record carries a ``health``
    block checked within ``ttl_s`` seconds; ``None`` when there's no
    verdict or it's stale (caller should re-verify). ``ttl_s <= 0`` always
    re-verifies."""
    if ttl_s <= 0:
        return None
    h = record.get("health")
    if not isinstance(h, dict) or "ok" not in h:
        return None
    checked = h.get("checked_at")
    try:
        if checked is None or (time.time() - float(checked)) > ttl_s:
            return None
    except (TypeError, ValueError):
        return None
    return bool(h["ok"])


def _persist_health(tools_cfg: Any, name: str, ok: bool, detail: str) -> None:
    """Write a fresh health verdict back into a tool's cache file (best-effort)."""
    record = _load_cached_tool(tools_cfg, name)
    if not record:
        return
    record["health"] = {"ok": bool(ok), "checked_at": time.time(), "detail": detail[:200]}
    try:
        path = _cache_dir(tools_cfg) / f"{_safe_name(name)}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _log.warning("tool synthesis: failed to persist health for %r: %s", name, exc)


async def _generate_probe_args(
    generate: Callable[[str], Any], name: str, record: dict[str, Any]
) -> dict[str, Any] | None:
    """Ask the LLM for a realistic kwargs dict to health-test a tool that
    predates stored sample args. Returns ``None`` on any failure."""
    params = record.get("params") or []
    desc = (record.get("description") or "").strip()
    sig = ", ".join(params) if params else "**kwargs"
    prompt = (
        f"Tool `{name}({sig})` — {desc}\n"
        "Return ONLY a JSON object of realistic example keyword arguments to "
        'call it once for a health check (e.g. {"location": "London"}). '
        "JSON only, no prose, no code fences."
    )
    try:
        raw = await generate(prompt)
    except Exception:  # noqa: BLE001
        return None
    try:
        parsed = json.loads(_strip_code_fences(str(raw or "")).strip())
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


async def _verify_tool_record(
    name: str,
    record: dict[str, Any],
    sandbox_cfg: Any,
    tools_cfg: Any,
    *,
    generate: Callable[[str], Any] | None = None,
) -> bool | None:
    """Run a cached tool's self-test once and classify it.

    Returns ``True`` (works), ``False`` (returned a runtime ``error:``), or
    ``None`` (couldn't test — no sample inputs even after asking the LLM, or
    the sandbox harness itself hiccuped). A ``None`` is NEVER treated as
    unhealthy by callers — we only drop a tool that demonstrably errored.
    On a definite verdict the result is persisted to the tool's cache file."""
    source = record.get("source")
    if not source:
        return None
    sample = record.get("sample_args")
    if not isinstance(sample, dict) and generate is not None:
        sample = await _generate_probe_args(generate, name, record)
    if not isinstance(sample, dict):
        return None  # unverifiable — nothing to call it with
    probe = await _run_selftest(name, source, sample, sandbox_cfg, tools_cfg)
    if probe is None:
        return None  # sandbox/infra hiccup — not the tool's fault
    ok = not probe.strip().lower().startswith("error:")
    _persist_health(tools_cfg, name, ok, "" if ok else probe)
    return ok


def _register(
    context: Any,
    name: str,
    source: str,
    sandbox_cfg: Any,
    tools_cfg: Any,
) -> bool:
    """Register a sandboxed callable for ``source`` under ``name``."""
    try:
        context.register_tool(name, _make_synth_callable(name, source, sandbox_cfg, tools_cfg))
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("tool synthesis: register failed for %r: %s", name, exc)
        return False


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def _cache_dir(tools_cfg: Any) -> Path:
    raw = getattr(tools_cfg, "synthesized_tools_path", "~/.config/care/synthesized_tools")
    return Path(raw).expanduser()


def _safe_name(name: str) -> str:
    """Filesystem-safe slug for a tool name (keeps the cache flat)."""
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80] or "tool"


def _load_cached_tool(tools_cfg: Any, name: str) -> dict[str, Any] | None:
    path = _cache_dir(tools_cfg) / f"{_safe_name(name)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log.warning("tool synthesis: bad cache file %s: %s", path, exc)
        return None


def _load_all_cached(tools_cfg: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    directory = _cache_dir(tools_cfg)
    if not directory.exists():
        return out
    for path in directory.glob("*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        name = rec.get("name")
        if name and rec.get("source"):
            out[name] = rec
    return out


def _save_cached_tool(
    tools_cfg: Any,
    name: str,
    source: str,
    params: list[str],
    description: str,
    *,
    sample_args: dict[str, Any] | None = None,
    health: dict[str, Any] | None = None,
) -> None:
    try:
        directory = _cache_dir(tools_cfg)
        directory.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "name": name,
            "params": params,
            "description": description,
            "source": source,
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
        # Persist the self-test inputs + verdict so a later run can re-verify
        # the tool (health-check) without re-deriving sample args.
        if isinstance(sample_args, dict):
            record["sample_args"] = sample_args
        if isinstance(health, dict):
            record["health"] = health
        (directory / f"{_safe_name(name)}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("tool synthesis: failed to cache %r: %s", name, exc)


# ---------------------------------------------------------------------------
# Memory (best-effort, runs sync SDK calls off the event loop)
# ---------------------------------------------------------------------------


def _get_memory(config: Any) -> Any:
    mem_cfg = getattr(config, "memory", None)
    if not getattr(mem_cfg, "base_url", None):
        return None
    try:
        from care.memory import CareMemory

        return CareMemory.from_config(config)
    except Exception as exc:  # noqa: BLE001
        _log.info("tool synthesis: Memory unavailable: %s", exc)
        return None


async def _memory_find_tool(memory: Any, name: str) -> str | None:
    import asyncio

    def _do() -> str | None:
        hit = memory.find_entity_by_name(name=name, entity_type="agent_skill")
        if not hit:
            return None
        content = hit.get("content") or {}
        manifest = content.get("manifest") or {}
        if not manifest.get("care_synthesized"):
            return None  # not one of ours — don't blindly exec foreign skills
        return content.get("instructions") or manifest.get("care_source") or None

    try:
        return await asyncio.to_thread(_do)
    except Exception as exc:  # noqa: BLE001
        _log.info("tool synthesis: Memory lookup failed for %r: %s", name, exc)
        return None


async def _memory_save_tool(
    memory: Any,
    name: str,
    description: str,
    source: str,
    params: list[str],
) -> None:
    import asyncio

    def _do() -> None:
        manifest = {
            "name": name,
            "description": description,
            "care_synthesized": True,
            "params": params,
            "care_source": source,
        }
        memory.save_agent_skill(
            skill_uri=f"care-synth://{name}",
            manifest=manifest,
            sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            instructions=source,
            name=name,
            description=description,
            tags=[SYNTH_TAG],
        )

    try:
        await asyncio.to_thread(_do)
        _log.info("tool synthesis: saved %r to Memory", name)
    except Exception as exc:  # noqa: BLE001
        _log.info("tool synthesis: Memory save failed for %r: %s", name, exc)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _strip_exit_prefix(out: str) -> str:
    if out.startswith("[exit 0]\n"):
        return out[len("[exit 0]\n"):]
    return out


def _resolve_llm(api: Any) -> Callable[[str], Any] | None:
    for attr in ("get_response_with_retries", "get_response"):
        fn = getattr(api, attr, None)
        if callable(fn):
            async def _call(prompt: str, _fn: Any = fn) -> str:
                return await _fn(prompt)

            return _call
    return None


__all__ = [
    "SYNTH_TAG",
    "bundled_tools_for_chain",
    "bundled_tools_to_python_code",
    "cached_tool_specs",
    "register_cached_tools",
    "synthesize_missing_tools",
]
