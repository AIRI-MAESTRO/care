"""Load persistent user/project context from CARE.md files (P1.1).

CARE.md is the chain-building analog of Claude Code's CLAUDE.md: a global
``~/.config/care/CARE.md`` plus a per-project ``./CARE.md`` whose contents are
injected into MAGE generation as a standing "user / project context" block
(preferences, recurring domain, constraints, house style). P1.2 wires the
returned string into the generation prompt; this module is just the loader.

Pure + best-effort: missing or empty files yield ``""`` (a silent no-op), and
any read error is swallowed + logged so a malformed file never blocks a run.
The merged block is capped (``ContextConfig.max_chars``) so a large CARE.md
can't dominate the prompt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger("care.context_md")

_DEFAULT_GLOBAL = Path("~/.config/care/CARE.md")
_DEFAULT_PROJECT_FILENAME = "CARE.md"
_DEFAULT_MAX_CHARS = 8000

# P5.6 — auto-learned facts live under this heading in CARE.md so they ride
# into generation via :func:`load_user_context` like any other CARE.md content.
_LEARNED_HEADING = "## Auto-learned facts"
_CARE_MD_SCAFFOLD = (
    "# CARE.md\n\n"
    "<!-- Persistent context CARE injects into every generation: your role,\n"
    "     preferences, recurring constraints. Edit freely. CARE keeps the\n"
    "     auto-learned facts below up to date from your sessions. -->\n\n"
    f"{_LEARNED_HEADING}\n"
)


def _read_text(path: Path) -> str:
    """Read + strip a file; ``""`` when absent/unreadable (never raises)."""
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # noqa: BLE001 — context is best-effort
        _log.info("CARE.md unreadable (%s): %s", path, exc)
    return ""


def load_user_context(
    config: Any = None,
    *,
    project_dir: Path | str | None = None,
    profile: dict[str, Any] | None = None,
) -> str:
    """Return the merged context block, or ``""`` when there is none.

    Reads the global CARE.md first, then the per-project file (project LAST so
    it augments / overrides the global), each under a labelled ``##`` header.
    When ``profile`` (the learned ``user_profile`` dict, P1.3/P1.4) is given,
    its rendered block is appended last. Honors ``CareConfig.context``
    (enabled / paths / max_chars) when a ``config`` is given.

    Args:
        config: A :class:`~care.config.CareConfig` (or anything exposing a
            ``.context`` section). ``None`` → defaults.
        project_dir: Directory to look for the per-project file in. ``None`` →
            current working directory.
        profile: Optional learned-preferences dict to append as a block.
    """
    ctx_cfg = getattr(config, "context", None)
    if ctx_cfg is not None and not getattr(ctx_cfg, "enabled", True):
        return ""

    global_path = Path(getattr(ctx_cfg, "global_path", _DEFAULT_GLOBAL)).expanduser()
    project_filename = getattr(ctx_cfg, "project_filename", _DEFAULT_PROJECT_FILENAME)
    max_chars = int(getattr(ctx_cfg, "max_chars", _DEFAULT_MAX_CHARS))

    blocks: list[str] = []
    global_text = _read_text(global_path)
    if global_text:
        blocks.append(f"## Global user context ({global_path})\n{global_text}")

    pdir = Path(project_dir) if project_dir is not None else Path.cwd()
    project_text = _read_text(pdir / project_filename)
    if project_text:
        blocks.append(f"## Project context ({project_filename})\n{project_text}")

    profile_block = format_profile_block(profile)
    if profile_block:
        blocks.append(profile_block)

    if not blocks:
        return ""

    merged = "\n\n".join(blocks)
    if len(merged) > max_chars:
        merged = merged[:max_chars].rstrip() + "\n…[CARE.md truncated]"
    return merged


def format_profile_block(profile: dict[str, Any] | None) -> str:
    """Render the learned ``user_profile`` dict (P1.3) as a markdown block for
    injection into generation context. Empty/absent profile → ``""``."""
    if not isinstance(profile, dict) or not profile:
        return ""
    # NOTE: deliberately does NOT list learned tool names — feeding
    # previously-synthesised tool names ("frequently used tools: …") back into
    # generation made the planner re-invent them (the P1.4 pollution loop).
    # Only stable, non-tool preferences go here.
    lines = ["## Learned user preferences (from past runs)"]
    recent = profile.get("recent_domains") or []
    if isinstance(recent, list) and recent:
        doms = ", ".join(str(d) for d in recent[:5])
        lines.append(
            f"- Recent domains: {doms} — assume baseline familiarity here; "
            "skip the basics and go deeper."
        )
    last_mode = profile.get("last_mode")
    if last_mode:
        depth = (
            "thorough, multi-step answers"
            if str(last_mode).strip().lower() == "deep"
            else "concise, to-the-point answers"
        )
        lines.append(f"- Preferred depth: {last_mode} — favour {depth}.")
    run_count = profile.get("run_count")
    if isinstance(run_count, int) and run_count > 0:
        lines.append(
            f"- Experienced user ({run_count} prior task"
            f"{'s' if run_count != 1 else ''}) — be direct, don't over-explain."
        )
    return "\n".join(lines) if len(lines) > 1 else ""


def default_global_care_md() -> Path:
    """The default global CARE.md path (``~/.config/care/CARE.md``), expanded."""
    return _DEFAULT_GLOBAL.expanduser()


def ensure_care_md(path: Path | str) -> Path:
    """P5.6 — create a scaffold CARE.md at ``path`` if it's absent (parents
    included). Idempotent; returns the expanded path. Best-effort on the
    write — a failure is logged, not raised (context is never load-bearing)."""
    p = Path(path).expanduser()
    try:
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(_CARE_MD_SCAFFOLD, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _log.info("CARE.md scaffold write failed (%s): %s", p, exc)
    return p


def merge_learned_fact(path: Path | str, key: str, value: str) -> bool:
    """P5.6 — add or update a ``- <key>: <value>`` bullet under the
    auto-learned facts section of the CARE.md at ``path`` (scaffolding the
    file/section when needed).

    Dedup + supersede: a byte-identical bullet is a no-op; a bullet whose
    ``key`` already exists is REPLACED (so a changed preference doesn't pile
    up duplicates). Returns ``True`` iff the file changed. Best-effort — any
    error is logged and returns ``False`` (never blocks a turn).
    """
    key = (key or "").strip()
    value = (value or "").strip()
    if not key or not value:
        return False
    p = ensure_care_md(path)
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _log.info("CARE.md unreadable for fact-merge (%s): %s", p, exc)
        return False

    bullet = f"- {key}: {value}"
    key_prefix = f"- {key}:".lower()
    lines = text.splitlines()

    # Locate (or append) the auto-learned section.
    sec = next(
        (i for i, ln in enumerate(lines) if ln.strip() == _LEARNED_HEADING),
        None,
    )
    if sec is None:
        lines += ["", _LEARNED_HEADING, bullet]
        _write(p, lines)
        return True

    # Scan the section (until the next heading) for an exact dup or same key.
    for i in range(sec + 1, len(lines)):
        if lines[i].startswith("## "):
            break  # next section — fact not present here
        stripped = lines[i].strip()
        if stripped == bullet:
            return False  # exact duplicate → no-op
        if stripped.lower().startswith(key_prefix):
            lines[i] = bullet  # supersede the stale value
            _write(p, lines)
            return True

    # New key — insert right after the heading (skip a blank line if present).
    at = sec + 1
    if at < len(lines) and not lines[at].strip():
        at += 1
    lines.insert(at, bullet)
    _write(p, lines)
    return True


def _write(path: Path, lines: list[str]) -> None:
    """Best-effort write of ``lines`` back to ``path`` (single trailing NL)."""
    try:
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _log.info("CARE.md write failed (%s): %s", path, exc)
