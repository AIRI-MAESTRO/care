"""Pre-flight chain validation (TODO §4 P2).

Two layers of validation, run together in :func:`validate_chain`:

1. **Parse validation.** Hand the raw payload (JSON string, dict,
   or already-typed :class:`mmar_carl.ReasoningChain`) to CARL's
   `ReasoningChain.from_dict(..., use_typed_steps=True)` /
   `from_json(...)` and capture every Pydantic `ValidationError`
   as a structured list. CARE's TUI / CLI surface this list
   inline so the user fixes the chain shape **before** committing
   to a run.
2. **Capability gap.** When the parse succeeds and the user
   supplied a populated :class:`ReasoningContext` (typically the
   one CARE just primed in :mod:`care.runtime.executor`), call
   `chain.preflight(context)` (CARL §5.7) to compare the
   chain's declared `required_tools` / `required_mcp_servers` /
   `required_skills` against the context's actual registry. Any
   gap lands on `missing_*` and the screen offers a "register
   before run" action.

CARL's `preflight` lives on the dev branch but the installed
`mmar-carl 0.2.0` may not ship it yet. The module degrades
gracefully: if the parsed chain doesn't have `.preflight`, the
report still ships parse status + an `extras.preflight_skipped`
hint so the UI can show "preflight not available on this CARL
version" instead of a hard failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PreflightResult:
    """CARE-side combined parse + preflight report.

    Frozen so the same instance can be passed across screens / log
    handlers without defensive copies. Fields are CARE-stable —
    they don't depend on CARL's `PreflightReport` shape so a CARL
    version bump that changes that surface doesn't ripple here.

    Fields:
        parsed: ``True`` when the payload validated against
            :class:`ReasoningChain`. ``False`` means look at
            `parse_errors` for what went wrong.
        parse_errors: One entry per Pydantic validation error,
            already formatted "field.path: message" so callers
            can render them as a bullet list. Empty when
            `parsed=True`.
        required_tools/mcp_servers/skills: What the chain
            references. Empty tuple when the chain didn't parse
            or CARL's preflight isn't available.
        missing_tools/mcp_servers/skills: Subset of the above
            that the supplied context can't satisfy.
        chain: The parsed chain object, or ``None`` if parsing
            failed. Caller can act on this directly (e.g. pass
            to `execute_chain_async`) without re-parsing.
        preflight_available: ``False`` when CARL's `.preflight`
            method wasn't found on the parsed chain (older
            installs). UI can hint "upgrade CARL for tool gap
            detection".
    """

    parsed: bool
    parse_errors: tuple[str, ...] = field(default_factory=tuple)
    required_tools: tuple[str, ...] = field(default_factory=tuple)
    required_mcp_servers: tuple[str, ...] = field(default_factory=tuple)
    required_skills: tuple[str, ...] = field(default_factory=tuple)
    missing_tools: tuple[str, ...] = field(default_factory=tuple)
    missing_mcp_servers: tuple[str, ...] = field(default_factory=tuple)
    missing_skills: tuple[str, ...] = field(default_factory=tuple)
    chain: Any = None
    preflight_available: bool = True

    @property
    def ok(self) -> bool:
        """``True`` when parse succeeded and nothing is missing —
        the user can run the chain right now."""
        return self.parsed and not (
            self.missing_tools or self.missing_mcp_servers or self.missing_skills
        )

    @property
    def is_valid(self) -> bool:
        """``True`` when the chain *parsed* (regardless of missing
        capabilities). Use this when offering "edit" vs "register
        deps" — invalid chains can't even reach the deps step."""
        return self.parsed

    def format_text(self) -> str:
        """One-line ok / multi-line error breakdown suitable for
        stdout / TUI footer rendering."""
        if not self.parsed:
            lines = ["chain failed to parse"]
            lines.extend(f"  {e}" for e in self.parse_errors)
            return "\n".join(lines)
        if self.ok:
            n_t = len(self.required_tools)
            n_m = len(self.required_mcp_servers)
            n_s = len(self.required_skills)
            suffix = (
                f"({n_t} tool{'s' if n_t != 1 else ''}, "
                f"{n_m} mcp server{'s' if n_m != 1 else ''}, "
                f"{n_s} skill{'s' if n_s != 1 else ''})"
            )
            if not self.preflight_available:
                return f"chain parsed; preflight skipped {suffix}"
            return f"preflight: ok {suffix}"
        lines = ["preflight: missing dependencies"]
        if self.missing_tools:
            lines.append(f"  tools: {', '.join(self.missing_tools)}")
        if self.missing_mcp_servers:
            lines.append(f"  mcp servers: {', '.join(self.missing_mcp_servers)}")
        if self.missing_skills:
            lines.append(f"  skills: {', '.join(self.missing_skills)}")
        return "\n".join(lines)


def validate_chain(
    payload: Any,
    *,
    context: Any = None,
    use_typed_steps: bool = True,
) -> PreflightResult:
    """Parse + preflight a CARL chain.

    Args:
        payload: One of three shapes:
            - A JSON string (will be `json.loads` + `from_dict`).
            - A `dict` matching CARL's chain serialisation
              (passed to `ReasoningChain.from_dict`).
            - An already-parsed `ReasoningChain` (no re-parse).
        context: Optional CARL :class:`ReasoningContext`. When
            supplied, `chain.preflight(context)` runs to populate
            the `missing_*` fields. When `None`, the report still
            includes `required_*` (so the UI can list what the
            chain needs).
        use_typed_steps: Forwarded to `ReasoningChain.from_dict`.
            CARE uses typed steps by default because they're the
            shape MAGE emits + the shape that survives round-trip
            through `chain.preflight`.

    Returns:
        A populated :class:`PreflightResult` — never raises.
        Parse errors land on `parse_errors`; missing CARL
        `preflight` method lands on `preflight_available=False`.
    """
    chain, parse_errors = _parse(payload, use_typed_steps=use_typed_steps)
    if chain is None:
        return PreflightResult(parsed=False, parse_errors=tuple(parse_errors))

    preflight_fn = getattr(chain, "preflight", None)
    if not callable(preflight_fn):
        # Older CARL installs (or callers passing a fake chain) —
        # surface what we have without inventing structure.
        return PreflightResult(
            parsed=True,
            chain=chain,
            preflight_available=False,
        )

    try:
        report = preflight_fn(context)
    except Exception as exc:  # noqa: BLE001
        # CARL's preflight is best-effort; if it throws we still
        # want the parse signal back to the user.
        return PreflightResult(
            parsed=True,
            parse_errors=(f"preflight raised: {exc}",),
            chain=chain,
            preflight_available=True,
        )

    return PreflightResult(
        parsed=True,
        chain=chain,
        required_tools=tuple(getattr(report, "required_tools", []) or []),
        required_mcp_servers=tuple(
            getattr(report, "required_mcp_servers", []) or []
        ),
        required_skills=tuple(getattr(report, "required_skills", []) or []),
        missing_tools=tuple(getattr(report, "missing_tools", []) or []),
        missing_mcp_servers=tuple(
            getattr(report, "missing_mcp_servers", []) or []
        ),
        missing_skills=tuple(getattr(report, "missing_skills", []) or []),
        preflight_available=True,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse(
    payload: Any,
    *,
    use_typed_steps: bool,
) -> tuple[Any, list[str]]:
    """Coerce ``payload`` into a parsed chain. Returns
    ``(chain, errors)`` — ``chain`` is ``None`` when parsing fails
    and ``errors`` carries the formatted explanations."""
    # Already a chain — duck-typed: anything exposing `.steps` is
    # treated as parsed. Avoids importing `ReasoningChain` for an
    # `isinstance` check that would force a hard dep.
    if hasattr(payload, "steps") and not isinstance(payload, (str, dict, bytes)):
        return payload, []

    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            return None, [f"payload: invalid utf-8 ({exc})"]

    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            return None, [
                f"payload: invalid JSON at line {exc.lineno}, col {exc.colno}: {exc.msg}"
            ]
    elif isinstance(payload, dict):
        data = payload
    else:
        return None, [
            f"payload: expected dict / JSON string / ReasoningChain, got {type(payload).__name__}"
        ]

    try:
        from mmar_carl import ReasoningChain
    except ImportError as exc:
        return None, [f"mmar_carl: {exc}"]

    try:
        chain = ReasoningChain.from_dict(data, use_typed_steps=use_typed_steps)
    except Exception as exc:  # noqa: BLE001
        return None, _format_validation_errors(exc)

    return chain, []


def _format_validation_errors(exc: Exception) -> list[str]:
    """Convert any parse exception (Pydantic ValidationError, raw
    KeyError, etc.) into the ``"field.path: message"`` form CARE
    renders in its UI bullet list.

    Pydantic v2 `ValidationError` exposes `.errors()`; everything
    else gets a single best-effort line so callers always have
    something to display.
    """
    errors_method = getattr(exc, "errors", None)
    if callable(errors_method):
        try:
            raw: Any = errors_method()
        except Exception:  # noqa: BLE001
            raw = []
        formatted: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                loc = ".".join(str(p) for p in item.get("loc", ())) or "<root>"
                msg = item.get("msg", "validation error")
                formatted.append(f"{loc}: {msg}")
        if formatted:
            return formatted
    return [f"{type(exc).__name__}: {exc}"]


__all__ = [
    "PreflightResult",
    "validate_chain",
]
