"""P6.5 — artifact sink: collect the files a chain/skill produced and land
them in a real, stable directory the user can open.

CARL AgentSkill steps (LLM_AGENT / SCRIPT modes) write files to a sandbox
``/workspace/out`` and surface them on each step's ``result_data`` as
``output_files: [{"name", "path", "size"}, ...]`` (typed via
``StepExecutionResult.as_skill_output()``). Those paths live in a temp /
sandbox dir that gets cleaned up — useless to the user once the run ends.
This module copies them OUT into a stable artifacts directory and returns
the saved paths, so the TUI can show ``📄 saved: <path>`` lines.

Cross-platform by construction (Phase 6 hard requirement): every path is a
:class:`pathlib.Path`, the default root is ``Path.home() / ".care" /
"artifacts"`` (valid on Windows / macOS / Linux), each file is placed by its
``Path(...).name`` leaf (so a CARL ``name`` carrying sub-dirs or a foreign
separator can never escape the target), and nothing is hardcoded to POSIX.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

__all__ = [
    "collect_output_files",
    "default_artifacts_root",
    "missing_required_output",
    "resolve_artifacts_root",
    "save_run_artifacts",
]


def default_artifacts_root() -> Path:
    """The OS-agnostic default artifacts root: ``~/.care/artifacts``.

    Resolved against :meth:`pathlib.Path.home`, so it follows the running
    user's home directory on any platform.
    """
    return Path.home() / ".care" / "artifacts"


def resolve_artifacts_root(care_config: Any = None) -> Path:
    """Artifacts root for this run.

    ``care_config.artifacts.dir`` (env ``CARE_ARTIFACTS__DIR``) wins when
    set; otherwise :func:`default_artifacts_root`. ``~`` is expanded so a
    configured ``~/Documents/care`` resolves per-user / per-OS.
    """
    artifacts_cfg = getattr(care_config, "artifacts", None)
    configured = getattr(artifacts_cfg, "dir", None) if artifacts_cfg else None
    if configured:
        return Path(configured).expanduser()
    return default_artifacts_root()


def collect_output_files(result: Any) -> list[dict[str, Any]]:
    """Gather every ``output_files`` entry from a CARL ``ReasoningResult``.

    Reads the typed ``step.as_skill_output()`` view first (AgentSkill steps),
    then falls back to a raw ``result_data['output_files']`` list so a
    non-skill step that still writes files (e.g. a tool / script step) is
    caught too. Entries are deduped by source ``path``. Duck-typed — any
    object exposing ``step_results`` works, so tests need no real CARL.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in getattr(result, "step_results", None) or []:
        files: list[Any] = []
        as_skill = getattr(step, "as_skill_output", None)
        if callable(as_skill):
            try:
                skill = as_skill()
            except Exception:  # noqa: BLE001 — typed view is best-effort
                skill = None
            if skill is not None:
                files = list(getattr(skill, "output_files", None) or [])
        if not files:
            data = getattr(step, "result_data", None)
            if isinstance(data, dict) and isinstance(data.get("output_files"), list):
                files = data["output_files"]
        for entry in files:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if not path or path in seen:
                continue
            seen.add(path)
            out.append(entry)
    return out


def missing_required_output(result: Any) -> bool:
    """``True`` when a step required a file but produced none.

    CARL's LLM_AGENT loop sets ``result_data['no_output_file'] = True`` on a
    file-producing skill step (``require_output_file``) that finished without
    writing anything — i.e. the model "described" the artifact in prose instead
    of creating it. CARE uses this to warn honestly rather than relay the
    model's "file created" claim. Duck-typed; never raises.
    """
    for step in getattr(result, "step_results", None) or []:
        data = getattr(step, "result_data", None)
        if isinstance(data, dict) and data.get("no_output_file") is True:
            return True
    return False


def save_run_artifacts(
    result: Any,
    *,
    care_config: Any = None,
    dest: Path | None = None,
    slug: str = "",
    run_name: str = "run",
) -> list[Path]:
    """Copy every file a chain/skill produced into a stable dir; return the
    saved paths (empty when the run produced none).

    Destination precedence: explicit ``dest`` (the per-request override) →
    ``<artifacts_root>/<run_name>[-<slug>]``. The directory is created if
    needed. Each source is copied by its sanitized basename; on a name clash
    a ``-N`` suffix is appended so nothing is silently overwritten.

    Best-effort throughout — an unwritable directory or a single failed copy
    is logged and skipped, never raised, so artifact handling can't sink an
    otherwise-successful turn.
    """
    entries = collect_output_files(result)
    if not entries:
        return []

    if dest is not None:
        target = Path(dest).expanduser()
    else:
        root = resolve_artifacts_root(care_config)
        clean = _slugify(slug)
        target = root / (f"{run_name}-{clean}" if clean else run_name)

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log.info("artifacts dir unavailable (%s): %s", target, exc)
        return []

    saved: list[Path] = []
    for entry in entries:
        src = Path(str(entry.get("path", "")))
        # `name` may carry sub-dirs (CARL's rglob rel path) or a foreign
        # separator — keep only the leaf so we never escape ``target``.
        leaf = Path(str(entry.get("name") or src.name)).name or src.name
        out_path = _unique_path(target / leaf)
        try:
            shutil.copy2(src, out_path)
        except OSError as exc:
            _log.info("could not copy artifact %s: %s", src, exc)
            continue
        saved.append(out_path)
    return saved


# --------------------------------------------------------------------------- #
#  Internals                                                                   #
# --------------------------------------------------------------------------- #


def _slugify(text: str, *, max_len: int = 40) -> str:
    """Lowercase, alnum-and-dash slug for a per-run subdir name. Collapses
    runs of separators, trims dashes, caps length. ``""`` when nothing
    usable (the caller then uses the bare ``run_name``)."""
    keep: list[str] = []
    for ch in (text or "").strip().lower():
        if ch.isalnum():
            keep.append(ch)
        elif keep and keep[-1] != "-":
            keep.append("-")
    return "".join(keep).strip("-")[:max_len].strip("-")


def _unique_path(path: Path) -> Path:
    """``path`` if free, else the first ``<stem>-<N><suffix>`` that isn't
    taken — so two files sharing a basename both survive."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 1000):
        candidate = path.with_name(f"{stem}-{i}{suffix}")
        if not candidate.exists():
            return candidate
    return path
