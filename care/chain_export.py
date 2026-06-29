"""Generated-chain export (TODO §9 P3).

Saved chains in Memory are the canonical persistence layer, but
users often want a file artefact too — to commit to source
control, hand off to a colleague, or `python` the chain directly
without going through CARL's loader. This module ships the two
export targets the TODO calls out:

* **JSON.** Dump the chain dict (``ReasoningChain.to_dict()``)
  to disk. Always available — no upstream deps needed.
* **Python.** Render the chain through MAGE's
  ``CodeGenerator.generate(chain_dict, query, config)`` so the
  output is a runnable ``ChainBuilder``-fluent script. Lazy-
  imports ``mmar_mage``; surfaces a friendly
  :class:`ChainExportError` when the optional dep isn't
  installed.

Format selection prefers an explicit ``format=`` kwarg; falls
back to the destination path's extension; raises rather than
guessing when neither is decisive (better than silently writing
JSON to a ``foo.py`` file).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ExportFormat = Literal["json", "python", "markdown"]
"""The file shapes :func:`export_chain` writes.

Use ``"json"`` for round-trip-safe storage that CARL's
``ReasoningChain.from_json`` reads back. Use ``"python"`` for a
human-runnable script (no round-trip guarantee — the script is a
fresh build path, not a serialisation). Use ``"markdown"`` for a
human-readable walkthrough (``## Step N`` + ``### Aim`` …) that ends
with a fenced ``python`` block of the CARL build script."""


@dataclass(frozen=True)
class ExportResult:
    """Outcome from :func:`export_chain`.

    Frozen so the result can be passed to status lines / logs /
    tests without defensive copies. Field set is intentionally
    narrow — anything richer than "what was written and how
    much" leaks implementation detail callers shouldn't rely on.
    """

    path: Path
    format: ExportFormat
    bytes_written: int


class ChainExportError(RuntimeError):
    """Raised when the export can't proceed — unknown format, a
    missing optional dep (``mmar_mage`` for ``"python"``), or
    invalid input that can't be normalised into a chain dict."""


def export_chain(
    chain: Any,
    path: Path | str,
    *,
    format: ExportFormat | None = None,
    query: str = "",
    title: str = "",
    mage_config: Any = None,
) -> ExportResult:
    """Write ``chain`` to ``path``.

    Args:
        chain: One of three accepted shapes:
            - A `dict` matching ``ReasoningChain.to_dict()``.
            - A JSON `str` (must decode to a dict).
            - An object exposing ``.to_dict()`` (typically a
              ``ReasoningChain``). Duck-typed via ``hasattr`` so
              the function doesn't pull `mmar_carl` for an
              isinstance check.
        path: Destination file. Tilde-expanded. Parent
            directories are NOT auto-created — callers must
            already have a writable directory in hand. (Exports
            usually land next to the user's chosen filename, not
            a system path the caller forgot to mkdir.)
        format: Explicit choice. ``None`` infers from ``path``'s
            extension (``.json`` → ``"json"``, ``.py`` →
            ``"python"``). Raises :class:`ChainExportError`
            when neither the kwarg nor the extension is
            decisive.
        query: Original user query — used in the ``"python"``
            output's docstring. Ignored for ``"json"``. Empty
            string when the caller doesn't have one.
        mage_config: ``mmar_mage.MAGEConfig`` (or any duck-typed
            substitute). Only consulted for ``"python"``
            output. ``None`` triggers a lazy default-construct
            (``MAGEConfig()``) — convenient for CLI users who
            just want the script and don't care about config
            knobs.

    Returns:
        :class:`ExportResult` describing what was written.

    Raises:
        ChainExportError: For unknown formats, malformed input,
            missing optional deps, or write failures.
    """
    dest = Path(str(path)).expanduser()
    fmt = _resolve_format(format, dest)
    chain_dict = _coerce_chain_dict(chain)

    if fmt == "json":
        body = json.dumps(chain_dict, indent=2, sort_keys=True)
    elif fmt == "markdown":
        body = _render_markdown(
            chain_dict, query=query, title=title, mage_config=mage_config,
        )
    else:
        body = _render_python(chain_dict, query=query, mage_config=mage_config)

    try:
        bytes_written = dest.write_text(body, encoding="utf-8")
    except OSError as exc:
        raise ChainExportError(f"write failed: {exc}") from exc

    return ExportResult(path=dest, format=fmt, bytes_written=bytes_written)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_EXTENSION_TO_FORMAT: dict[str, ExportFormat] = {
    ".json": "json",
    ".py": "python",
    ".md": "markdown",
    ".markdown": "markdown",
}


def _resolve_format(
    explicit: ExportFormat | None,
    path: Path,
) -> ExportFormat:
    if explicit is not None:
        if explicit not in ("json", "python", "markdown"):
            raise ChainExportError(
                f"unknown export format {explicit!r}; "
                "expected 'json', 'python' or 'markdown'"
            )
        return explicit
    suffix = path.suffix.lower()
    fmt = _EXTENSION_TO_FORMAT.get(suffix)
    if fmt is None:
        raise ChainExportError(
            f"cannot infer format from {path.name!r}; "
            f"pass format='json' or format='python' explicitly"
        )
    return fmt


def _coerce_chain_dict(chain: Any) -> dict[str, Any]:
    if isinstance(chain, dict):
        return chain
    if isinstance(chain, (bytes, bytearray)):
        try:
            chain = chain.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ChainExportError(
                f"chain bytes are not valid utf-8: {exc}"
            ) from exc
    if isinstance(chain, str):
        try:
            data = json.loads(chain)
        except json.JSONDecodeError as exc:
            raise ChainExportError(
                f"chain JSON failed to parse at line {exc.lineno}, col {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(data, dict):
            raise ChainExportError(
                f"chain JSON must decode to a dict; got {type(data).__name__}"
            )
        return data
    to_dict = getattr(chain, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if not isinstance(result, dict):
            raise ChainExportError(
                f"chain.to_dict() must return a dict; got {type(result).__name__}"
            )
        return result
    raise ChainExportError(
        f"expected dict / JSON str / object with .to_dict(); got {type(chain).__name__}"
    )


# Step ``type`` → human label. Mirrors the InspectionScreen's
# `_STEP_TYPE_LABELS` so the walkthrough reads the same as the UI.
_STEP_TYPE_LABELS: dict[str, str] = {
    "llm": "AI",
    "tool": "Tool",
    "mcp": "MCP",
    "mcp_resource": "MCP Resource",
    "memory": "Memory",
    "transform": "Transform",
    "conditional": "Conditional",
    "structured_output": "Structured Output",
    "agent_skill": "Agent Skill",
    "evaluation": "Evaluation",
    "agent_handoff": "Agent Handoff",
    "parallel_sampling": "Parallel Sampling",
    "tool_discovery": "Tool Discovery",
    "human_input": "Human Input",
    "supervisor": "Supervisor",
    "debate": "Debate",
}


def _first(step: dict[str, Any], *keys: str, default: str = "") -> str:
    """First non-empty stringified value among ``keys``."""
    for k in keys:
        v = step.get(k)
        if v not in (None, "", [], {}):
            return str(v)
    return default


def _step_title(step: dict[str, Any], index: int) -> str:
    return _first(
        step, "title", "name", "step_name", "label", default=f"Step {index}",
    )


def _step_type_label(step: dict[str, Any]) -> str:
    raw = (_first(step, "step_type", "type") or "").strip().lower()
    return _STEP_TYPE_LABELS.get(raw, raw.replace("_", " ").title() or "Step")


def _step_deps(step: dict[str, Any]) -> list[Any]:
    for k in ("dependencies", "deps", "depends_on"):
        v = step.get(k)
        if isinstance(v, (list, tuple)) and v:
            return list(v)
    return []


def _render_markdown(
    chain_dict: dict[str, Any],
    *,
    query: str,
    title: str,
    mage_config: Any,
) -> str:
    """Human-readable chain walkthrough + a fenced ``python`` CARL block.

    The narrative half uses ``## Step N. <title>`` headings with an
    ``### Aim`` (and type / dependencies / config) so a reader understands
    the chain without running anything; the file ends with a valid Python
    build script (MAGE's `CodeGenerator`, or a `ReasoningChain.from_dict`
    reconstruction when `mmar_mage` isn't installed)."""
    name = title or _first(chain_dict, "name", "display_name", default="Chain")
    steps = chain_dict.get("steps") or []

    out: list[str] = [f"# {name}", ""]
    description = _first(chain_dict, "description", "summary")
    if description:
        out += [description, ""]
    if query:
        out += [f"> **Task:** {query}", ""]
    meta: list[str] = []
    domain = _first(chain_dict, "domain")
    if domain:
        meta.append(f"**Domain:** {domain}")
    meta.append(f"**Steps:** {len(steps)}")
    out += ["  ·  ".join(meta), ""]

    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        number = step.get("number") or step.get("step_id") or i
        out.append(f"## Step {number}. {_step_title(step, i)}")
        out.append("")
        out.append(f"_Type: {_step_type_label(step)}_")
        out.append("")
        aim = _first(step, "aim", "description", "stage_action")
        out.append("### Aim")
        out.append(aim or "_(no aim recorded)_")
        out.append("")
        deps = _step_deps(step)
        if deps:
            out.append(f"**Depends on:** {', '.join(str(d) for d in deps)}")
            out.append("")
        for label, keys in (
            (
                "Tool",
                (
                    "step_config.tool_name", "config.tool_name",
                    "tool_name", "tool",
                ),
            ),
            ("Model", ("llm_config.model", "model")),
            ("Reasoning", ("reasoning_questions",)),
        ):
            val = _nested_first(step, keys)
            if val:
                out.append(f"**{label}:** {val}")
                out.append("")

    out += ["## Python (CARL)", ""]
    out += [
        "Runnable build script for this chain "
        "(`ReasoningChain` via the CARL builder):",
        "",
        "```python",
        _python_code_block(chain_dict, query=query, mage_config=mage_config),
        "```",
        "",
    ]
    return "\n".join(out).rstrip() + "\n"


def _nested_first(step: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Resolve dotted keys (``config.tool_name``) or flat keys, first hit."""
    for key in keys:
        if "." in key:
            head, tail = key.split(".", 1)
            sub = step.get(head)
            if isinstance(sub, dict) and sub.get(tail) not in (None, "", [], {}):
                return str(sub[tail])
        elif step.get(key) not in (None, "", [], {}):
            return str(step[key])
    return ""


def _python_code_block(
    chain_dict: dict[str, Any], *, query: str, mage_config: Any,
) -> str:
    """The Python source embedded in the markdown export. Prefers MAGE's
    `CodeGenerator`; falls back to a `ReasoningChain.from_dict`
    reconstruction (always valid, no `mmar_mage` needed)."""
    try:
        return _render_python(chain_dict, query=query, mage_config=mage_config)
    except ChainExportError:
        return _render_python_fallback(chain_dict, query=query)


def _render_python_fallback(chain_dict: dict[str, Any], *, query: str) -> str:
    """Dependency-light Python: embed the chain dict + reconstruct via
    ``ReasoningChain.from_dict``. Valid CARL code even without `mmar_mage`."""
    literal = json.dumps(chain_dict, indent=4, sort_keys=True, ensure_ascii=False)
    header = f'"""CARL chain export for: {query}"""' if query else \
        '"""CARL chain export."""'
    return (
        f"{header}\n\n"
        "from mmar_carl import ReasoningChain\n\n"
        f"CHAIN = {literal}\n\n"
        "# Reconstruct the executable chain from its serialised form.\n"
        "chain = ReasoningChain.from_dict(CHAIN)\n"
    )


# ---------------------------------------------------------------------------
# CARL-compat → MAGE-flat step flattening
#
# Saved chains live in CARL-compat form: per-type config is nested under
# ``step_config`` (e.g. a tool step's name is at ``step_config.tool_name`` and
# its inputs at ``step_config.input_mapping``). MAGE's ``CodeGenerator``, on the
# other hand, reads the FLAT MAGE names off the top of the step
# (``step["tool_name"]`` / ``step["tool_input_mapping"]``). Handing the nested
# form straight to the generator produced ``tool_name='unknown'`` /
# ``input_mapping={}``. We invert MAGE's own ``CARL_FIELD_MAP`` to lift the
# nested config back to flat names before generating code.
# ---------------------------------------------------------------------------

# Fallback inverse maps (nested CARL key → flat MAGE key), used when
# ``mmar_mage.carl_export.CARL_FIELD_MAP`` can't be imported. Tool is the
# common case the export bug reported; the rest mirror CARL_FIELD_MAP.
_FALLBACK_INVERSE_FIELD_MAP: dict[str, dict[str, str]] = {
    "tool": {
        "tool_name": "tool_name",
        "tool_description": "tool_description",
        "input_mapping": "tool_input_mapping",
        "output_key": "tool_output_key",
        "timeout": "tool_timeout",
    },
    "memory": {
        "operation": "memory_operation",
        "memory_key": "memory_key",
        "value_source": "memory_value_source",
        "namespace": "memory_namespace",
    },
    "transform": {
        "transform_type": "transform_type",
        "input_key": "transform_input_key",
        "output_format": "transform_output_format",
        "expression": "transform_expression",
    },
    "conditional": {
        "branches": "condition_branches",
        "default_step": "condition_default_step",
        "condition_context_key": "condition_context_key",
    },
}


def _inverse_field_maps() -> dict[str, dict[str, str]]:
    """``{step_type: {nested_carl_key: flat_mage_key}}`` — the inverse of
    MAGE's ``CARL_FIELD_MAP``. Falls back to a built-in subset when MAGE
    isn't importable."""
    try:
        from mmar_mage.carl_export import CARL_FIELD_MAP
    except Exception:  # noqa: BLE001 — degrade to the built-in subset
        return _FALLBACK_INVERSE_FIELD_MAP
    return {
        step_type: {nested: flat for flat, nested in mapping.items()}
        for step_type, mapping in CARL_FIELD_MAP.items()
    }


def _flatten_step(step: dict[str, Any], inverse: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Lift a CARL-form step's nested ``step_config`` / ``config`` back to
    the flat MAGE field names ``CodeGenerator`` expects. Idempotent on a
    step that's already flat (no nested config → returned unchanged)."""
    if not isinstance(step, dict):
        return step
    cfg = step.get("step_config")
    if not isinstance(cfg, dict):
        cfg = step.get("config")
    if not isinstance(cfg, dict):
        return step
    step_type = str(step.get("step_type") or step.get("type") or "llm")
    mapping = inverse.get(step_type, {})
    flat = dict(step)
    for nested_key, value in cfg.items():
        # MCP nests a ``server`` sub-object; lift its members under the
        # ``mcp_server_*`` flat names the generator reads.
        if step_type in ("mcp", "mcp_resource") and nested_key == "server" and isinstance(value, dict):
            for sk, sv in value.items():
                flat.setdefault(f"mcp_server_{sk}", sv)
            continue
        flat_key = mapping.get(nested_key)
        if flat_key is None:
            # No declared inverse — keep the value reachable under a
            # type-prefixed flat name (never clobber an existing key).
            flat_key = nested_key if step_type == "llm" else f"{step_type}_{nested_key}"
        flat.setdefault(flat_key, value)
    flat.pop("step_config", None)
    flat.pop("config", None)
    return flat


def _flatten_chain(chain_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``chain_dict`` with every step flattened from
    CARL-compat (nested ``step_config``) into MAGE-flat field names."""
    steps = chain_dict.get("steps")
    if not isinstance(steps, list):
        return chain_dict
    inverse = _inverse_field_maps()
    out = dict(chain_dict)
    out["steps"] = [_flatten_step(s, inverse) for s in steps]
    return out


def _render_python(
    chain_dict: dict[str, Any],
    *,
    query: str,
    mage_config: Any,
) -> str:
    """Lazy-import MAGE's CodeGenerator and render the script."""
    try:
        from mmar_mage.code_generator import CodeGenerator
    except ImportError as exc:
        raise ChainExportError(
            "mmar_mage is not installed; "
            "install with `pip install \"care[mage]\"` to enable .py export"
        ) from exc

    if mage_config is None:
        try:
            from mmar_mage import MAGEConfig
        except ImportError as exc:  # pragma: no cover — same path as above
            raise ChainExportError(
                "mmar_mage is not installed; install with `pip install \"care[mage]\"`"
            ) from exc
        try:
            mage_config = MAGEConfig()
        except Exception as exc:  # noqa: BLE001
            raise ChainExportError(
                f"failed to construct default MAGEConfig: {exc}"
            ) from exc

    try:
        return CodeGenerator.generate(
            _flatten_chain(chain_dict), query, mage_config,
        )
    except Exception as exc:  # noqa: BLE001
        raise ChainExportError(
            f"CodeGenerator.generate failed: {exc}"
        ) from exc


__all__ = [
    "ChainExportError",
    "ExportFormat",
    "ExportResult",
    "export_chain",
]
