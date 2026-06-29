"""Live Redis probes for gigaevo chain experiments (local Docker stacks).

Platform ``/results`` stays empty while ``tools.comparison`` is missing.
These helpers read gigavolve Redis directly so MAESTRO can show meaningful
GA generation + fitness during a run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any


def _redis_container() -> str:
    return os.environ.get(
        "CARE_PLATFORM__GIGAVOLVE_REDIS_CONTAINER",
        "gigaevo-platform-redis-gigavolve-1",
    )


def _redis_db() -> str:
    return os.environ.get("CARE_PLATFORM__GIGAVOLVE_REDIS_DB", "1")


def _problem_prefix(experiment_id: str) -> str | None:
    if not experiment_id.startswith("exp_"):
        return None
    return experiment_id.removeprefix("exp_")


def _redis_command(*args: str) -> str | None:
    if os.environ.get("CARE_PLATFORM__PROBE_REDIS_GENERATION", "1") == "0":
        return None
    try:
        proc = subprocess.run(
            ["docker", "exec", _redis_container(), "redis-cli", "-n", _redis_db(), *args],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout or ""


def _redis_lrange(key: str) -> list[str]:
    raw = _redis_command("LRANGE", key, "0", "-1")
    if not raw:
        return []
    return [line for line in raw.splitlines() if line.strip()]


def _last_scalar_int(key: str) -> int | None:
    """Return the ``v`` field of the *last* entry of a metric LIST as a
    non-negative int.

    Runner metric lists store ``{"s": <seq>, "t": <ts>, "v": <val>,
    "k": "scalar"}`` rows; the latest count lives at the tail. Returns
    ``None`` when the list is empty / unreadable / negative so callers
    can tell "no data yet" from a real ``0``.
    """
    raw = _redis_command("LRANGE", key, "-1", "-1")
    if not raw:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            value = row.get("v")
            if isinstance(value, (int, float)) and value >= 0:
                return int(value)
    return None


def _redis_keys(pattern: str) -> list[str]:
    raw = _redis_command("KEYS", pattern)
    if not raw:
        return []
    return [line for line in raw.splitlines() if line.strip()]


def _parse_scalar_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def probe_ga_generation(experiment_id: str) -> int | None:
    """Return 0-based GA generation from program lineages in Redis.

    Avoids ``evolution_engine:generation`` — that counter tracks the
    MultiIsland scheduler and grows much faster than user ``max_iterations``.
    """
    problem = _problem_prefix(experiment_id)
    if problem is None:
        return None

    max_lineage = 0
    for key in _redis_keys(f"{problem}:program:*"):
        raw = _redis_command("GET", key)
        if not raw:
            continue
        try:
            program = json.loads(raw)
        except json.JSONDecodeError:
            continue
        lineage = program.get("lineage") if isinstance(program, dict) else None
        if isinstance(lineage, dict):
            gen = lineage.get("generation")
            if isinstance(gen, (int, float)) and gen >= 0:
                max_lineage = max(max_lineage, int(gen))
        elif isinstance(program, dict):
            gen = program.get("generation")
            if isinstance(gen, (int, float)) and gen >= 0:
                max_lineage = max(max_lineage, int(gen))

    if max_lineage <= 0:
        rows = _parse_scalar_rows(
            _redis_lrange(f"{problem}:metrics:history:program_metrics:valid_gen_fitness_mean"),
        )
        steps = [
            int(row["s"])
            for row in rows
            if isinstance(row.get("s"), (int, float)) and row.get("v") is not None
        ]
        if steps:
            return max(steps)
        return None

    # gigaevo lineage starts at 1 for the seed program → display 0-based gen.
    return max(0, max_lineage - 1)


def probe_best_fitness(experiment_id: str) -> float | None:
    """Best valid program fitness currently stored in Redis."""
    problem = _problem_prefix(experiment_id)
    if problem is None:
        return None

    best: float | None = None
    for key in _redis_keys(f"{problem}:program:*"):
        raw = _redis_command("GET", key)
        if not raw:
            continue
        try:
            program = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(program, dict):
            continue
        metrics = program.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if metrics.get("is_valid") not in (1, 1.0, True):
            continue
        fit = metrics.get("fitness")
        if not isinstance(fit, (int, float)):
            continue
        if fit <= -999:
            continue
        best = fit if best is None else max(best, float(fit))

    if best is not None:
        return best

    rows = _parse_scalar_rows(
        _redis_lrange(f"{problem}:metrics:history:program_metrics:valid_frontier_fitness"),
    )
    values = [
        float(row["v"])
        for row in rows
        if isinstance(row.get("v"), (int, float)) and float(row["v"]) > -999
    ]
    return max(values) if values else None


def probe_fitness_history(experiment_id: str) -> list[dict[str, Any]]:
    """Build a minimal fitness_history list from Redis gen aggregates."""
    problem = _problem_prefix(experiment_id)
    if problem is None:
        return []

    rows = _parse_scalar_rows(
        _redis_lrange(f"{problem}:metrics:history:program_metrics:valid_gen_fitness_mean"),
    )
    history: list[dict[str, Any]] = []
    for row in rows:
        step = row.get("s")
        value = row.get("v")
        if not isinstance(step, (int, float)) or not isinstance(value, (int, float)):
            continue
        if float(value) <= -999:
            continue
        history.append(
            {
                "generation": int(step),
                "best_fitness": float(value),
            },
        )
    return history


def _lenient_json_loads(raw: str) -> Any:
    """Parse evolved-program chain JSON when strict ``json.loads`` fails.

    Mirrors the backslash-escape repair in gigavolve ``_safe_json_loads``
    so CARE can recover ``BASE_CHAIN_CONFIG`` blobs that LLM mutations
  wrote with invalid escapes."""
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    bs = chr(92)
    valid_after = set('"\\/bfnrtu')
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == bs and i + 1 < len(text) and text[i + 1] not in valid_after:
            out.append(bs)
            out.append(bs)
        else:
            out.append(text[i])
        i += 1
    fixed = "".join(out)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def _parse_chain_config_body(body: str) -> dict[str, Any] | None:
    """Return a chain dict when ``body`` parses to JSON with a ``steps`` list."""
    for candidate in (body.strip(),):
        parsed = None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = _lenient_json_loads(candidate)
        if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
            return parsed
    return None


def _extract_chain_from_pickle_blob(blob: bytes) -> dict[str, Any] | None:
    """Recover ``chain_config_json`` embedded in a gigavolve stage pickle.

    Unpickling needs the runner's ``gigaevo`` package, which CARE doesn't ship.
    The JSON literal is still present in the pickle bytes after
    ``CallProgramFunction`` ran ``_safe_json_loads`` at evaluation time."""
    if not blob:
        return None
    text = blob.decode("utf-8", errors="ignore")
    marker = "chain_config_json"
    idx = text.find(marker)
    if idx == -1:
        return None
    brace = text.find("{", idx)
    if brace == -1:
        return None
    depth = 0
    for offset, ch in enumerate(text[brace:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[brace : brace + offset + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    parsed = _lenient_json_loads(candidate)
                if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
                    return parsed
                return None
    return None


def _extract_chain_from_stage_results(program: dict[str, Any]) -> dict[str, Any] | None:
    """Read evaluated chain JSON from completed stage outputs."""
    import base64

    stages = program.get("stage_results")
    if not isinstance(stages, dict):
        return None
    for stage_name in ("CallProgramFunction", "CallValidatorFunction"):
        stage = stages.get(stage_name)
        if not isinstance(stage, dict):
            continue
        if stage.get("status") != "completed":
            continue
        output = stage.get("output")
        if not isinstance(output, str) or not output:
            continue
        try:
            blob = base64.b64decode(output)
        except (TypeError, ValueError):
            continue
        chain = _extract_chain_from_pickle_blob(blob)
        if chain is not None:
            return chain
    return None


def extract_chain_config_from_program_code(program_code: str) -> dict[str, Any] | None:
    """Pull a CARL chain dict out of an evolved gigavolve program module.

    Tries ``CHAIN_CONFIG_JSON`` then ``BASE_CHAIN_CONFIG`` literal assignments
    (same order as the Platform's ``results.py`` helper). Uses lenient JSON
    parsing so partially broken LLM escapes still render in the Versions tab."""
    if not program_code or not program_code.strip():
        return None
    for var in ("CHAIN_CONFIG_JSON", "BASE_CHAIN_CONFIG"):
        pattern = re.compile(
            rf"{re.escape(var)}\s*(?::\s*str\s*)?=\s*"
            rf"(?P<q>\"\"\"|'''|\"|')(?P<body>.*?)(?P=q)",
            re.DOTALL,
        )
        match = pattern.search(program_code)
        if not match:
            continue
        body = match.group("body")
        body = body.replace('\\"\\"\\"', '"""').replace("\\'\\'\\'", "'''")
        chain = _parse_chain_config_body(body)
        if chain is not None:
            return chain
    return None


def probe_program_chain_config(
    experiment_id: str,
    program_id: str,
) -> dict[str, Any] | None:
    """Read ``<problem>:program:<id>`` from gigavolve Redis and extract chain JSON."""
    problem = _problem_prefix(experiment_id)
    if problem is None or not program_id:
        return None
    raw = _redis_command("GET", f"{problem}:program:{program_id}")
    if not raw:
        return None
    try:
        program = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(program, dict):
        return None
    stages_chain = _extract_chain_from_stage_results(program)
    if stages_chain is not None:
        return stages_chain
    code = program.get("code")
    if not isinstance(code, str):
        return None
    return extract_chain_config_from_program_code(code)


def probe_programs_counts(experiment_id: str) -> tuple[int | None, int | None]:
    """Latest ``(valid, invalid)`` program counts from Redis metric lists.

    Mirrors the Platform's server-side read of
    ``<problem>:metrics:history:program_metrics:programs_{valid,invalid}_count``
    so CARE's Programs chart fills on a local stack even when the
    Platform's ``/results`` returns no live metrics. Returns
    ``(None, None)`` when neither counter has landed yet — the caller
    distinguishes that from a real ``0``.
    """
    problem = _problem_prefix(experiment_id)
    if problem is None:
        return (None, None)
    base = f"{problem}:metrics:history:program_metrics:"
    valid = _last_scalar_int(base + "programs_valid_count")
    invalid = _last_scalar_int(base + "programs_invalid_count")
    return (valid, invalid)


__all__ = [
    "extract_chain_config_from_program_code",
    "probe_best_fitness",
    "probe_fitness_history",
    "probe_ga_generation",
    "probe_program_chain_config",
    "probe_programs_counts",
]
