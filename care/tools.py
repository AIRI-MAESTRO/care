"""Tool-registry loader (TODO §5 P1).

CARE keeps per-user ``@carl_tool``-decorated callables in a single
directory (default ``~/.config/care/tools/``). This module is the
glue that turns a configured directory into a populated CARL
:class:`ReasoningContext` at chain start-up:

* CARL already ships
  :meth:`ReasoningContext.register_tools_from_path` — it imports
  every ``*.py`` under a glob, walks the module for callables
  carrying ``__carl_tool__ = True`` (set by the ``@carl_tool``
  decorator), and registers them. CARE doesn't re-implement that
  loader; it just calls into it with the user's configured
  directory, optional tag whitelist, and namespace prefix.
* The catalog (`care.catalog._scan_tools_dir`) enumerates the
  same directory for the CLI / TUI but never imports anything.
  The catalog is **discovery**; this module is **activation**.

The TUI's run hook calls :func:`load_tools_into_context` after
``ReasoningContext.from_chain_inputs(...)`` and before chain
execution, so every chain run sees the user's tools without
needing per-chain wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from care.config import CareConfig, ToolsConfig


@dataclass(frozen=True)
class LoadedTools:
    """What :func:`load_tools_into_context` registered.

    Frozen so reports / status lines can pass it around without
    defensive copies. ``directory`` is the absolute path the loader
    scanned — useful for the TUI banner ("Loaded 4 tools from
    /home/x/.config/care/tools/") and for logging that survives
    config moves.
    """

    names: tuple[str, ...]
    directory: Path
    skipped: bool = False
    """``True`` when the configured directory didn't exist — first-run
    users have no tools yet, so the loader is a no-op rather than
    raising. Differentiates "loaded 0 tools" from "directory
    missing" for the TUI banner."""


def load_tools_into_context(
    context: Any,
    config: CareConfig | ToolsConfig,
) -> LoadedTools:
    """Load the user's ``@carl_tool`` files into ``context``.

    Args:
        context: A CARL :class:`ReasoningContext`. Duck-typed —
            anything with
            ``register_tools_from_path(glob, *, tag_filter, name_prefix)``
            works (the test suite uses a stub).
        config: Either a full :class:`CareConfig` or just its
            ``tools`` section. CARE's TUI usually passes the full
            config; library callers grabbing the loader from
            ``care.tools`` will often hand-roll a ``ToolsConfig``.

    Returns:
        A :class:`LoadedTools` describing what was registered.
        ``skipped=True`` when the tools directory doesn't exist
        (no error — first-run is normal).

    Notes:
        The loader **doesn't** raise on broken plugin files —
        CARL's :meth:`register_tools_from_path` logs at ``DEBUG``
        and continues, so one corrupt tool file can't block the
        whole startup. Surface the loader's return list to confirm
        what actually made it in.
    """
    tools_cfg = config.tools if isinstance(config, CareConfig) else config
    directory = Path(tools_cfg.path).expanduser().resolve()

    if not directory.exists():
        return LoadedTools(
            names=(),
            directory=directory,
            skipped=True,
        )

    glob_pattern = str(directory / "*.py")
    registered = context.register_tools_from_path(
        glob_pattern,
        tag_filter=tools_cfg.tag_filter,
        name_prefix=tools_cfg.name_prefix,
    )
    return LoadedTools(
        names=tuple(registered),
        directory=directory,
    )


__all__ = [
    "LoadedTools",
    "load_tools_into_context",
]
