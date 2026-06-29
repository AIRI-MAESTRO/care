"""CARE command-line interface (TODO §9 P2 + §8 P1).

Same entry-point as the TUI: running ``care`` with no arguments
launches the Textual app (existing behaviour); running ``care
<subcommand>`` dispatches to one of the headless CLI handlers.

Subcommands shipped in this iteration — each wraps a primitive
that already has its own data layer + tests, so the CLI is a
thin presentation surface:

* ``care catalog`` — render :func:`care.build_catalog` output
  as a JSON document or a grouped text listing.
* ``care validate <file>`` — run :func:`care.validate_chain` on
  a chain JSON file and exit non-zero if it doesn't parse.
* ``care import <pattern>...`` — batch-save chains via
  :func:`care.import_chains`. Defaults to ``--dry-run`` so the
  user has to opt into the destructive run (the inverse of the
  module-level default — matches how ``rsync --dry-run`` /
  ``terraform plan`` work).

* ``care run <chain-id>`` — fetch + preflight a saved chain;
  ``--execute`` runs it end-to-end via CARL (``--task`` /
  ``--input k=v`` overrides, ``--json`` output incl.
  ``final_answer``, exit codes 0 success / 1 failed run / 2
  resolution errors, ``--save-result`` persists a memory card).
  Headless by design (PRODUCTION_TODO B4) — the C1 promotion
  gate drives baseline runs through it.

Remaining MAGE/Platform-driven subcommands (``generate``,
``evolve``) stay deferred: they need broader integration with the
runtime executor that already has its own backlog.

The CLI is **stream-injectable**: every subcommand handler takes
``stdout`` / ``stderr`` arguments, so tests can capture output
via ``io.StringIO`` without monkey-patching ``sys``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, TextIO

from care.bulk_import import import_chains
from care.catalog import build_catalog
from care.config import CareConfig
from care.preflight import validate_chain


def main(argv: list[str] | None = None) -> int:
    """Entry point shared with ``pyproject.toml`` scripts.

    Args:
        argv: Arguments after the program name. ``None`` uses
            ``sys.argv[1:]``. Empty list / no subcommand launches
            the TUI (current behaviour) so users who type plain
            ``care`` still get the app.

    Returns:
        Process exit code. ``0`` on success, non-zero on
        per-subcommand failure (see each handler).
    """
    args_in = sys.argv[1:] if argv is None else list(argv)
    # Pull project-local `.env` overrides into `os.environ`
    # BEFORE the config loader and the file logger configure
    # themselves — both read the env, so the order matters.
    # Real shell env wins over the file.
    from care.dotenv import load_env_file
    from care.logging_setup import configure_from_env

    load_env_file()
    configure_from_env()

    if not args_in or args_in[0].startswith("-") and args_in[0] not in (
        "-h",
        "--help",
        "-V",
        "--version",
    ):
        # No subcommand → TUI.
        return _run_tui()

    parser = _build_parser()
    try:
        ns = parser.parse_args(args_in)
    except SystemExit as exc:
        # argparse's own --help / error paths return via SystemExit;
        # tests want a plain int.
        return int(exc.code or 0)

    handler = getattr(ns, "_handler", None)
    if handler is None:
        # Top-level `care --help` lands here; argparse already
        # printed help.
        return 0
    return handler(ns, sys.stdout, sys.stderr)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="care",
        description=(
            "CARE — Collaborative Agent Reasoning Ecosystem. "
            "Running with no subcommand launches the TUI."
        ),
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Print CARE version and exit.",
    )
    sub = parser.add_subparsers(dest="subcommand")

    cat = sub.add_parser(
        "catalog",
        help="List installed capabilities (skills / MCP servers / tools).",
    )
    cat.add_argument(
        "--skills",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Directory containing `<name>/SKILL.md` folders. May be "
            "repeated. Defaults to none (the catalog ships only "
            "MCP + tool entries unless you supply this)."
        ),
    )
    cat.add_argument(
        "--mcp-config",
        type=Path,
        default=None,
        help="Path to an mcp_servers.toml file.",
    )
    cat.add_argument(
        "--tools",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of @carl_tool Python files to enumerate.",
    )
    cat.add_argument(
        "--kind",
        choices=("agent_skill", "mcp_server", "tool", "memory_card"),
        default=None,
        help="Filter the listing to a single entry kind.",
    )
    cat.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the grouped text listing.",
    )
    cat.set_defaults(_handler=_cmd_catalog)

    val = sub.add_parser(
        "validate",
        help="Parse + preflight a chain JSON file.",
    )
    val.add_argument(
        "file",
        type=Path,
        help="Path to a chain JSON file (bare-chain or wrapper form).",
    )
    val.add_argument(
        "--json",
        action="store_true",
        help="Emit the PreflightResult as JSON instead of text.",
    )
    val.set_defaults(_handler=_cmd_validate)

    imp = sub.add_parser(
        "import",
        help="Validate chain JSON files (preview saves with --apply).",
    )
    imp.add_argument(
        "patterns",
        nargs="+",
        metavar="PATTERN",
        help=(
            "File paths or glob patterns. Use `**/*.json` for "
            "recursive directory walks."
        ),
    )
    imp.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually save chains to Memory. Without this flag the "
            "command runs as a dry-run (preview only)."
        ),
    )
    imp.add_argument(
        "--channel",
        default="latest",
        help="Default channel for entries that don't specify one.",
    )
    imp.set_defaults(_handler=_cmd_import)

    mem = sub.add_parser(
        "memory",
        help="Browse the configured Memory instance.",
    )
    mem_sub = mem.add_subparsers(dest="memory_subcommand")
    mem_ls = mem_sub.add_parser(
        "ls",
        help="List entities saved in Memory.",
    )
    mem_ls.add_argument(
        "--entity-type",
        choices=("chain", "agent", "agent_skill", "memory_card"),
        default="chain",
        help="Entity type to list (default: chain).",
    )
    mem_ls.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum rows to print (default: 20).",
    )
    mem_ls.add_argument(
        "--channel",
        default="latest",
        help="Version channel to read (default: latest).",
    )
    mem_ls.add_argument(
        "--namespace",
        default=None,
        help="Restrict to a single CARE namespace.",
    )
    mem_ls.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Filter by tag (repeatable, AND semantics).",
    )
    mem_ls.add_argument(
        "--q",
        default=None,
        help="Case-insensitive substring filter on name/description.",
    )
    mem_ls.add_argument(
        "--favourites-only",
        action="store_true",
        help="Restrict to favourited entities.",
    )
    mem_ls.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the text table.",
    )
    mem_ls.set_defaults(_handler=_cmd_memory_ls)

    mem_history = mem_sub.add_parser(
        "history",
        help="List recorded runs for a saved chain.",
    )
    mem_history.add_argument(
        "chain_id",
        help="Memory entity_id of the chain to query.",
    )
    mem_history.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum rows to print (default: 20).",
    )
    mem_history.add_argument(
        "--channel",
        default="latest",
        help="Version channel to read (default: latest).",
    )
    mem_history.add_argument(
        "--namespace",
        default=None,
        help="Restrict to a single CARE namespace.",
    )
    mem_history.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the text listing.",
    )
    mem_history.set_defaults(_handler=_cmd_memory_history)

    mem_show = mem_sub.add_parser(
        "show",
        help="Print a single entity's metadata + content.",
    )
    mem_show.add_argument(
        "entity_id",
        help="Memory entity_id to fetch.",
    )
    mem_show.add_argument(
        "--entity-type",
        choices=("chain", "agent", "agent_skill", "memory_card", "step"),
        default="chain",
        help="Entity type (default: chain).",
    )
    mem_show.add_argument(
        "--channel",
        default="latest",
        help="Version channel to read (default: latest).",
    )
    mem_show.add_argument(
        "--content-only",
        action="store_true",
        help="Print only the `content` body, skipping metadata.",
    )
    mem_show.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw entity payload as JSON.",
    )
    mem_show.set_defaults(_handler=_cmd_memory_show)

    evo = sub.add_parser(
        "evolve",
        help="Submit an evolution run for a saved chain.",
    )
    evo.add_argument(
        "chain_id",
        help="Memory entity_id of the chain to evolve (the seed).",
    )
    evo.add_argument(
        "--mode",
        choices=("full_chain", "per_step"),
        default="full_chain",
        help="Evolution mode (default: full_chain).",
    )
    evo.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Max generations (default: 5).",
    )
    evo.add_argument(
        "--population",
        type=int,
        default=8,
        help="Population size per generation (default: 8).",
    )
    evo.add_argument(
        "--validation-criteria",
        default="",
        help="Free-form prompt for the validation judge.",
    )
    evo.add_argument(
        "--validation-type",
        choices=("Continuous (0..1)", "Binary (0/1)"),
        default="Continuous (0..1)",
        help="Platform validation mode (default: Continuous).",
    )
    evo.add_argument(
        "--metric",
        choices=("ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "BLEU"),
        default="ROUGE-L",
        help="Continuous validation metric (default: ROUGE-L).",
    )
    evo.add_argument(
        "--binary-method",
        choices=("equality", "substring", "regexp"),
        default="equality",
        help="Binary validation method when --validation-type is Binary.",
    )
    evo.add_argument(
        "--target-column",
        default="expected",
        help="Dataset column to score against (default: expected).",
    )
    evo.add_argument(
        "--objective",
        action="append",
        default=None,
        metavar="NAME",
        help="Multi-objective fitness term (repeatable).",
    )
    evo.add_argument(
        "--test-data-path",
        default=None,
        metavar="PATH",
        help="Path to evaluation test data.",
    )
    evo.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Validation fitness threshold for early stop.",
    )
    evo.add_argument(
        "--wait",
        action="store_true",
        help=(
            "Block on the SSE stream until the evolution completes; "
            "print per-generation progress lines as they arrive."
        ),
    )
    evo.add_argument(
        "--accept",
        action="store_true",
        help=(
            "After --wait completes, promote the best individual via "
            "platform.accept_individual(...). Requires --wait."
        ),
    )
    evo.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary instead of human-readable lines.",
    )
    evo.set_defaults(_handler=_cmd_evolve)

    hlp = sub.add_parser(
        "help",
        help="Render the tutorial walkthrough + key cheat-sheet.",
    )
    hlp.add_argument(
        "--markdown",
        action="store_true",
        help="Emit Markdown (README quick-reference) instead of plain text.",
    )
    hlp.add_argument(
        "--category",
        choices=("global", "library", "generation", "execution", "evolution"),
        default=None,
        help=(
            "Restrict the bindings listing to a single category. "
            "Tutorial steps still render in full."
        ),
    )
    hlp.add_argument(
        "--screen",
        default=None,
        metavar="NAME",
        help=(
            "Restrict the bindings listing to a single screen "
            "(e.g. `LibraryScreen`)."
        ),
    )
    hlp.add_argument(
        "--commands",
        action="store_true",
        help=(
            "Print the CLI-subcommand ↔ TUI-verb parity table (which "
            "TUI affordances have a headless twin) instead of the tutorial."
        ),
    )
    hlp.set_defaults(_handler=_cmd_help)

    run = sub.add_parser(
        "run",
        help="Fetch a saved chain from Memory and preflight it.",
    )
    run.add_argument(
        "chain_id",
        help="Memory entity_id of the chain to fetch.",
    )
    run.add_argument(
        "--channel",
        default="latest",
        help="Version channel to read (default: latest).",
    )
    run.add_argument(
        "--export",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Also write the fetched chain to a file via "
            "care.export_chain. Format inferred from extension "
            "(.json / .py) or pass --export-format."
        ),
    )
    run.add_argument(
        "--export-format",
        choices=("json", "python"),
        default=None,
        help="Override the export format (overrides extension).",
    )
    run.add_argument(
        "--json",
        action="store_true",
        help="Emit the PreflightResult as JSON instead of text.",
    )
    run.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute the chain via CARL after preflight. Builds an "
            "LLM client from `CareConfig.mage.*` and runs the chain "
            "end-to-end. Needs `mmar_carl` installed + an LLM API key."
        ),
    )
    run.add_argument(
        "--task",
        default=None,
        metavar="TEXT",
        help=(
            "Override `outer_context` for the execution. Without "
            "this flag CARL primes the context from the chain's "
            "stored `CareChainMetadata.task_description`."
        ),
    )
    run.add_argument(
        "--input",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Add a `KEY=VALUE` pair to `context.memory['input']`. "
            "Repeatable. Use to inject runtime inputs (e.g. "
            "`--input city=Paris`) without storing them on the chain."
        ),
    )
    run.add_argument(
        "--file",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Attach a file's contents to `context.memory['input']` under "
            "its basename (e.g. `--file ./report.pdf` is readable as "
            "`${input.report.pdf}`). Repeatable. Use to supply the input "
            "files a saved chain expects."
        ),
    )
    run.add_argument(
        "--classify-files",
        choices=("heuristic", "model"),
        default="heuristic",
        help=(
            "How to decide whether a skill step READS an input file (and "
            "should consume `--file`): `heuristic` = keyword match (fast, "
            "deterministic — the default, best for scripts); `model` = ask "
            "the LLM to classify (matches the TUI; needs an LLM key)."
        ),
    )
    run.add_argument(
        "--save-result",
        default=None,
        metavar="NAME",
        help=(
            "After --execute, persist a run digest as a memory_card "
            "named NAME (tagged `run-record` + `chain:<id>`). Requires "
            "--execute."
        ),
    )
    run.add_argument(
        "--log",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write a structured debug log of the run (invocation, "
            "fetch, preflight, export, execution, persistence) to "
            "PATH. The file is opened in append mode so repeated "
            "runs build up a timeline."
        ),
    )
    run.add_argument(
        "--log-level",
        choices=("debug", "info"),
        default="info",
        help=(
            "Verbosity of --log output. `debug` includes full "
            "tracebacks and the fetched chain payload; `info` "
            "(default) keeps it to key milestones."
        ),
    )
    run.set_defaults(_handler=_cmd_run)

    gen = sub.add_parser(
        "generate",
        help="Generate a CARL chain from a free-form task via MAGE.",
    )
    gen.add_argument(
        "query",
        help="Free-form task description (e.g. 'weather report for SF').",
    )
    gen.add_argument(
        "--mode",
        choices=("fast", "deep"),
        default=None,
        help=(
            "MAGE generation mode. Defaults to MAGEConfig's own "
            "default when omitted."
        ),
    )
    gen.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write the generated chain to a file via "
            "care.export_chain. Format inferred from extension "
            "(.json / .py) or pass --output-format."
        ),
    )
    gen.add_argument(
        "--output-format",
        choices=("json", "python"),
        default=None,
        help="Override the export format (overrides extension).",
    )
    gen.add_argument(
        "--save",
        default=None,
        metavar="NAME",
        help=(
            "Save the generated chain to Memory under NAME (creates "
            "a new entity). Without this flag the chain is only "
            "printed / written to disk."
        ),
    )
    gen.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the generated chain as JSON instead of a summary "
            "line."
        ),
    )
    gen.set_defaults(_handler=_cmd_generate)

    rep = sub.add_parser(
        "replay",
        help="Step through a saved ReasoningResult / RunRecord JSON.",
    )
    rep.add_argument(
        "source",
        type=Path,
        help=(
            "Path to a JSON file carrying a ReasoningResult or "
            "RunRecord shape. Use `-` to read from stdin."
        ),
    )
    rep.add_argument(
        "--step",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Render the step at index N (0-indexed). Default: "
            "render the whole session (one block per step)."
        ),
    )
    rep.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit a structured payload with chain metadata + "
            "every step instead of the text walkthrough."
        ),
    )
    rep.set_defaults(_handler=_cmd_replay)

    lin = sub.add_parser(
        "lineage",
        help="Walk a chain's ancestry DAG.",
    )
    lin.add_argument(
        "chain_id",
        help="Memory entity_id of the chain to query.",
    )
    lin.add_argument(
        "--channel",
        default="latest",
        help="Start from this channel's head (default: latest).",
    )
    lin.add_argument(
        "--version-id",
        default=None,
        help="Walk from a specific historical version_id.",
    )
    lin.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="BFS-depth cap, 1-100 (default: 10).",
    )
    lin.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the text listing.",
    )
    lin.set_defaults(_handler=_cmd_lineage)

    srch = sub.add_parser(
        "search",
        help="BM25 / vector / hybrid search across saved entities.",
    )
    srch.add_argument(
        "query",
        help="Free-form search string.",
    )
    srch.add_argument(
        "--entity-type",
        choices=("chain", "agent", "agent_skill", "memory_card", "step"),
        default="chain",
        help="Entity type to search (default: chain).",
    )
    srch.add_argument(
        "--search-type",
        choices=("bm25", "vector", "hybrid"),
        default="bm25",
        help="Search backend (default: bm25).",
    )
    srch.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum hits to return (default: 10).",
    )
    srch.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the text listing.",
    )
    srch.set_defaults(_handler=_cmd_search)

    diff = sub.add_parser(
        "diff",
        help="Compare two saved chains side-by-side.",
    )
    diff.add_argument(
        "left",
        help="Memory entity_id of the left-hand chain.",
    )
    diff.add_argument(
        "right",
        help="Memory entity_id of the right-hand chain.",
    )
    diff.add_argument(
        "--channel",
        default="latest",
        help="Memory channel to read both chains from (default: latest).",
    )
    diff.add_argument(
        "--left-label",
        default="",
        help="Display label for the left side (default: entity_id).",
    )
    diff.add_argument(
        "--right-label",
        default="",
        help="Display label for the right side (default: entity_id).",
    )
    diff.add_argument(
        "--json",
        action="store_true",
        help="Emit the diff payload as JSON.",
    )
    diff.set_defaults(_handler=_cmd_diff)

    fav = sub.add_parser(
        "favourite",
        help="Toggle the favourite flag on a saved entity.",
    )
    fav.add_argument(
        "entity_id",
        help="Memory entity_id to star / unstar.",
    )
    fav.add_argument(
        "--entity-type",
        choices=("chain", "agent", "agent_skill", "memory_card"),
        default="chain",
        help="Entity type (default: chain).",
    )
    fav.add_argument(
        "--off",
        action="store_true",
        help="Unstar (set favourite=false) instead of starring.",
    )
    fav.add_argument(
        "--json",
        action="store_true",
        help="Emit the updated entity as JSON.",
    )
    fav.set_defaults(_handler=_cmd_favourite)

    # ---- version & channel management (TUI /versions, /rollback, /promote)
    vers = sub.add_parser(
        "versions",
        help="List an entity's version history + which channels point where.",
    )
    vers.add_argument("entity_id", help="Memory entity_id.")
    vers.add_argument(
        "--entity-type",
        choices=("chain", "agent", "agent_skill", "memory_card", "step"),
        default="chain",
        help="Entity type (default: chain).",
    )
    vers.add_argument(
        "--limit", type=int, default=20, help="Max versions to list."
    )
    vers.add_argument("--json", action="store_true", help="Emit JSON.")
    vers.set_defaults(_handler=_cmd_versions)

    roll = sub.add_parser(
        "rollback",
        help="Repoint a channel at a specific version (pin, not revert).",
    )
    roll.add_argument("entity_id", help="Memory entity_id.")
    roll.add_argument(
        "--to",
        dest="to_version",
        required=True,
        help="Version id to pin the channel to (see `care versions`).",
    )
    roll.add_argument(
        "--channel",
        default="stable",
        help="Channel to repoint (default: stable).",
    )
    roll.add_argument(
        "--entity-type",
        choices=("chain", "agent", "agent_skill", "memory_card", "step"),
        default="chain",
        help="Entity type (default: chain).",
    )
    roll.set_defaults(_handler=_cmd_rollback)

    prom = sub.add_parser(
        "promote",
        help="Copy one channel pointer to another (default latest -> stable).",
    )
    prom.add_argument("entity_id", help="Memory entity_id.")
    prom.add_argument(
        "--from", dest="from_channel", default="latest", help="Source channel."
    )
    prom.add_argument(
        "--to", dest="to_channel", default="stable", help="Target channel."
    )
    prom.add_argument(
        "--entity-type",
        choices=("chain", "agent", "agent_skill", "memory_card", "step"),
        default="chain",
        help="Entity type (default: chain).",
    )
    prom.set_defaults(_handler=_cmd_promote)

    forget = sub.add_parser(
        "forget",
        help="Soft-delete a saved entity (recoverable via Memory trash).",
    )
    forget.add_argument("entity_id", help="Memory entity_id to delete.")
    forget.add_argument(
        "--entity-type",
        choices=("chain", "agent", "agent_skill", "memory_card", "step"),
        default="chain",
        help="Entity type (default: chain).",
    )
    forget.add_argument(
        "--force",
        action="store_true",
        help="Actually delete (without --force this only previews).",
    )
    forget.set_defaults(_handler=_cmd_forget)

    rev = sub.add_parser(
        "revise",
        help="AI-edit a saved chain into a new version (TUI /revise twin).",
    )
    rev.add_argument("chain_id", help="Memory entity_id of the chain to edit.")
    rev.add_argument(
        "change",
        nargs="+",
        help="What to change, in natural language (quote it).",
    )
    rev.add_argument(
        "--channel", default="latest", help="Version channel (default: latest)."
    )
    rev.add_argument(
        "--mode", default=None, help="MAGE mode override (fast|deep)."
    )
    rev.add_argument(
        "--yes",
        action="store_true",
        help="Save the edited chain as a new version (default: preview only).",
    )
    rev.add_argument(
        "--json", action="store_true", help="Emit the edited chain as JSON."
    )
    rev.set_defaults(_handler=_cmd_revise)

    exp = sub.add_parser(
        "export",
        help="Export saved chains (+ AgentSkills) to a portable bundle tarball.",
    )
    exp.add_argument("output", help="Output tarball path (e.g. bundle.tar.gz).")
    exp.add_argument(
        "entity_ids", nargs="+", help="Chain entity_ids to export."
    )
    exp.add_argument(
        "--skill",
        action="append",
        default=[],
        dest="skills",
        metavar="SKILL_ID",
        help="AgentSkill entity_id to bundle (repeatable).",
    )
    exp.add_argument(
        "--channel", default="latest", help="Memory channel to read from."
    )
    exp.set_defaults(_handler=_cmd_export)

    # ---- agent hub (TUI /deploy, /deployments, /metrics) ----------------
    dep = sub.add_parser(
        "deploy",
        help="Deploy a saved chain to the agent hub as an HTTP agent.",
    )
    dep.add_argument("chain_id", help="Memory entity_id of the chain.")
    dep.add_argument(
        "--name", default=None, help="Agent name (default: derived from id)."
    )
    dep.add_argument(
        "--channel", default="stable", help="Channel to deploy (default: stable)."
    )
    dep.set_defaults(_handler=_cmd_deploy)

    deps = sub.add_parser(
        "deployments",
        help="List agents currently deployed on the hub.",
    )
    deps.add_argument("--json", action="store_true", help="Emit JSON.")
    deps.set_defaults(_handler=_cmd_deployments)

    met = sub.add_parser(
        "metrics",
        help="Show usage/cost metrics for deployed agents.",
    )
    met.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Agent name (omit to summarise all deployments).",
    )
    met.add_argument("--json", action="store_true", help="Emit JSON.")
    met.set_defaults(_handler=_cmd_metrics)

    # ---- eval dataset (TUI /dataset list|add|run|export) ----------------
    ds = sub.add_parser(
        "dataset",
        help="Manage a chain's eval dataset (list/add/run/export).",
    )
    ds_sub = ds.add_subparsers(dest="dataset_action", required=True)

    ds_list = ds_sub.add_parser("list", help="List a chain's dataset entries.")
    ds_list.add_argument("chain_id")
    ds_list.add_argument("--json", action="store_true")

    ds_add = ds_sub.add_parser("add", help="Add a dataset test case.")
    ds_add.add_argument("chain_id")
    ds_add.add_argument("task", help="The input task/prompt for this case.")
    ds_add.add_argument(
        "--expected", required=True, help="Expected output (substring match)."
    )
    ds_add.add_argument(
        "--rubric",
        default="",
        help="Optional LLM-judge rubric (TUI run only; CLI scores by substring).",
    )

    ds_run = ds_sub.add_parser(
        "run", help="Replay every entry through the chain + substring-score."
    )
    ds_run.add_argument("chain_id")
    ds_run.add_argument("--json", action="store_true")

    ds_exp = ds_sub.add_parser(
        "export", help="Export entries as JSONL for external eval frameworks."
    )
    ds_exp.add_argument("chain_id")
    ds_exp.add_argument("output", help="Output .jsonl path.")

    ds.set_defaults(_handler=_cmd_dataset)

    # ---- long-term memory (TUI /remember, /memory) ----------------------
    rem = sub.add_parser(
        "remember",
        help="Save an explicit note to long-term memory (TUI /remember twin).",
    )
    rem.add_argument("note", nargs="+", help="The note text (quote it).")
    rem.set_defaults(_handler=_cmd_remember)

    notes = sub.add_parser(
        "notes",
        help="Show long-term memory notes (TUI /memory twin).",
    )
    notes.add_argument(
        "--max-chars", type=int, default=2000, help="Digest size cap."
    )
    notes.set_defaults(_handler=_cmd_notes)

    mkt = sub.add_parser(
        "marketplace",
        help="Search the shared agent_skill marketplace.",
    )
    mkt.add_argument(
        "query",
        help="Free-text capability description.",
    )
    mkt.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum listings to fetch (default: 10).",
    )
    mkt.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop listings below this relevance score.",
    )
    mkt.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Require listing carry tag (repeatable, AND).",
    )
    mkt.add_argument(
        "--namespace",
        default=None,
        help="Memory namespace to scope the search.",
    )
    mkt.add_argument(
        "--deep",
        action="store_true",
        help="Match against skill_instructions in addition to "
             "skill_description (slower, more recall).",
    )
    mkt.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the text listing.",
    )
    mkt.set_defaults(_handler=_cmd_marketplace)

    init = sub.add_parser(
        "init",
        help=(
            "Quick-start: write a minimal `.env` with MAGE creds "
            "so a fresh checkout can `care` straight away."
        ),
    )
    init.add_argument(
        "--env-path",
        default="./.env",
        help="Where to write the .env (default: ./.env).",
    )
    init.add_argument(
        "--api-key",
        default=None,
        help="MAGE API key. Prompted interactively when omitted.",
    )
    init.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL "
            "(default prompt: https://openrouter.ai/api/v1)."
        ),
    )
    init.add_argument(
        "--model",
        default=None,
        help=(
            "Model id the endpoint understands "
            "(default prompt: anthropic/claude-3.5-sonnet)."
        ),
    )
    init.add_argument(
        "--mode",
        choices=("interactive", "production", "ad_hoc"),
        default=None,
        help="Chat default mode (default prompt: interactive). "
        "Legacy 'ad_hoc' is accepted and normalised to 'interactive'.",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target file if it already exists.",
    )
    init.add_argument(
        "--non-interactive",
        action="store_true",
        help=(
            "Don't prompt — every unset value falls back to its "
            "documented default. Required for unattended runs."
        ),
    )
    init.set_defaults(_handler=_cmd_init)

    doctor = sub.add_parser(
        "doctor",
        help=(
            "Diagnostic report — env vars set / config path / "
            "extras installed / network probes against Memory, "
            "MAGE, and Platform."
        ),
    )
    doctor.add_argument(
        "--config",
        default=None,
        help=(
            "Override the config path (default: "
            "`~/.config/care/config.toml`)."
        ),
    )
    doctor.add_argument(
        "--no-probes",
        action="store_true",
        help=(
            "Skip the network probes (env / config / "
            "extras only). Useful in CI / offline runs."
        ),
    )
    doctor.set_defaults(_handler=_cmd_doctor)

    migrate = sub.add_parser(
        "migrate-secrets",
        help=(
            "Migrate literal `*_api_key` values in "
            "`~/.config/care/config.toml` into the system "
            "keystore + rewrite the TOML with "
            "`keystore://…` URLs (§1 P1)."
        ),
    )
    migrate.add_argument(
        "--config",
        default=None,
        help=(
            "Override the config path (default: "
            "`~/.config/care/config.toml`)."
        ),
    )
    migrate.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print what would migrate without touching the "
            "keystore or rewriting the TOML."
        ),
    )
    migrate.set_defaults(_handler=_cmd_migrate_secrets)

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_catalog(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Render the capability catalog."""
    catalog = build_catalog(
        skills_paths=args.skills or None,
        mcp_config_path=args.mcp_config,
        tools_path=args.tools,
    )
    entries = (
        catalog.by_kind(args.kind) if args.kind else catalog.entries
    )

    if args.json:
        payload = {
            "entries": [
                {
                    "kind": e.kind,
                    "name": e.name,
                    "source": e.source,
                    "summary": e.summary,
                    "tags": list(e.tags),
                }
                for e in entries
            ],
            "errors": list(catalog.errors),
        }
        json.dump(payload, stdout, indent=2)
        stdout.write("\n")
    else:
        if not entries:
            stdout.write("catalog: no entries\n")
        else:
            current_kind: str | None = None
            for e in entries:
                if e.kind != current_kind:
                    if current_kind is not None:
                        stdout.write("\n")
                    stdout.write(f"# {e.kind}\n")
                    current_kind = e.kind
                summary = f" — {e.summary}" if e.summary else ""
                stdout.write(f"- {e.name}{summary}\n")
        if catalog.errors:
            stderr.write("\ncatalog warnings:\n")
            for err in catalog.errors:
                stderr.write(f"  {err}\n")

    return 0


def _cmd_validate(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Parse + preflight one chain JSON file. Exit non-zero
    when the chain doesn't parse."""
    try:
        raw = args.file.read_text(encoding="utf-8")
    except OSError as exc:
        stderr.write(f"care validate: read failed: {exc}\n")
        return 2

    result = validate_chain(raw)

    if args.json:
        payload = {
            "parsed": result.parsed,
            "ok": result.ok,
            "preflight_available": result.preflight_available,
            "parse_errors": list(result.parse_errors),
            "required_tools": list(result.required_tools),
            "required_mcp_servers": list(result.required_mcp_servers),
            "required_skills": list(result.required_skills),
            "missing_tools": list(result.missing_tools),
            "missing_mcp_servers": list(result.missing_mcp_servers),
            "missing_skills": list(result.missing_skills),
        }
        json.dump(payload, stdout, indent=2)
        stdout.write("\n")
    else:
        stdout.write(result.format_text() + "\n")

    if not result.parsed:
        return 1
    return 0


def _cmd_import(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Bulk-import chain files. Dry-run by default; ``--apply``
    saves each valid chain to the configured Memory."""
    if args.apply:
        try:
            memory = _build_memory()
        except CliMemoryError as exc:
            stderr.write(f"care import: {exc}\n")
            return 2
        report = import_chains(args.patterns, memory, dry_run=False)
    else:
        report = import_chains(args.patterns, dry_run=True)
    stdout.write(report.format_text() + "\n")
    return 0 if report.all_ok else 1


def _cmd_memory_ls(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """List entities from the configured Memory instance.

    Reads ``CareConfig.load()`` to materialise the connection
    settings, builds a :class:`CareMemory` facade, and forwards
    the listing filters to :meth:`CareMemory.list_entities`.
    """
    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care memory ls: {exc}\n")
        return 2

    try:
        rows = memory.list_entities(
            entity_type=args.entity_type,
            limit=args.limit,
            channel=args.channel,
            namespace=args.namespace,
            tags=args.tag,
            q=args.q,
            favourites_only=args.favourites_only or None,
        )
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care memory ls: lookup failed: {exc}\n")
        return 2

    if args.json:
        json.dump({"entities": rows}, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    if not rows:
        stdout.write("memory: no entities\n")
        return 0

    for row in rows:
        stdout.write(_format_memory_row(row) + "\n")
    return 0


def _format_memory_row(row: dict) -> str:
    """Format one Memory listing row as a single text line.

    Layout: ``<entity_id-prefix>  <display_name|name>  runs=N  fav  [tags]``.
    Keeps the line scannable in a terminal without forcing a
    multi-column table (which would need width-detection).
    """
    entity_id = str(row.get("entity_id") or "")
    short_id = entity_id[:12]
    display = (
        row.get("display_name")
        or (row.get("meta") or {}).get("name")
        or "(unnamed)"
    )
    runs = row.get("run_count") or 0
    fav = "★" if row.get("favourite") else " "
    meta = row.get("meta") or {}
    tags = meta.get("tags") if isinstance(meta, dict) else None
    tag_suffix = ""
    if tags:
        tag_suffix = "  [" + ", ".join(str(t) for t in tags) + "]"
    return f"{short_id}  {fav}  {display}  runs={runs}{tag_suffix}"


def _cmd_generate(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Generate a CARL chain from a free-form task via MAGE.

    Builds (or accepts an injected) :class:`MAGEGenerator`,
    awaits ``generator.generate(query)``, then renders the
    result. Optional outputs:

    * ``--save NAME``: persist via :meth:`CareMemory.save_chain`.
    * ``--output PATH``: write the chain dict to disk via
      :func:`care.export_chain`.
    * ``--json``: emit the chain dict to stdout as JSON.

    Otherwise prints a one-line summary
    (``generated chain: <steps> steps, mode=<mode>``).
    """
    from care.chain_export import ChainExportError, export_chain

    try:
        generator = _build_mage_generator(mode=args.mode)
    except CliMageError as exc:
        stderr.write(f"care generate: {exc}\n")
        return 2

    try:
        result = asyncio.run(generator.generate(args.query))
    except Exception as exc:  # noqa: BLE001
        hint = _friendly_llm_error(exc)
        if hint:
            stderr.write(f"care generate: {hint}\n")
        else:
            stderr.write(f"care generate: generation failed: {exc}\n")
        return 2

    chain_dict = _result_chain_dict(result)
    if not chain_dict:
        stderr.write(
            "care generate: MAGEResult.chain_dict is empty — nothing to "
            "save / export.\n",
        )
        return 2

    saved_entity_id: str | None = None
    if args.save:
        try:
            memory = _build_memory()
        except CliMemoryError as exc:
            stderr.write(f"care generate: {exc}\n")
            return 2
        try:
            saved_entity_id = memory.save_chain(
                chain_dict,
                name=args.save,
                query=args.query,
            )
        except Exception as exc:  # noqa: BLE001
            stderr.write(f"care generate: save failed: {exc}\n")
            return 2

    if args.output is not None:
        try:
            export = export_chain(
                chain_dict,
                args.output,
                format=args.output_format,
                query=args.query,
            )
        except ChainExportError as exc:
            stderr.write(f"care generate: export failed: {exc}\n")
            return 2
    else:
        export = None

    steps = chain_dict.get("steps") or []
    mode = getattr(result, "mode", None) or "unknown"

    if args.json:
        json.dump(chain_dict, stdout, indent=2, default=str)
        stdout.write("\n")
    else:
        stdout.write(
            f"generated chain: {len(steps)} steps, mode={mode}\n",
        )

    if saved_entity_id:
        stdout.write(f"saved: {saved_entity_id}\n")
    if export is not None:
        stdout.write(
            f"exported: {export.path} "
            f"({export.format}, {export.bytes_written} bytes)\n",
        )
    return 0


def _result_chain_dict(result) -> dict:
    """Pull the chain dict off a MAGE-result-like object.

    MAGE's ``MAGEResult`` exposes ``chain_dict`` directly;
    stubs / future variants may use ``chain.to_dict()`` or carry
    the dict at the top level. Probe in order so the CLI is
    resilient to result-shape evolution.
    """
    candidate = getattr(result, "chain_dict", None)
    if isinstance(candidate, dict):
        return candidate
    chain = getattr(result, "chain", None)
    if hasattr(chain, "to_dict"):
        try:
            data = chain.to_dict()
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, dict):
            return data
    if isinstance(chain, dict):
        return chain
    return {}


def _friendly_llm_error(exc: Exception) -> str | None:
    """Return a single friendly hint when ``exc`` is an LLM auth failure
    (expired/invalid key), else ``None``.

    MAGE retries 403s internally and logs three raw ``LLM call attempt
    N/3 failed: 403`` lines; this collapses that into one actionable
    message for the CLI error path."""
    code = getattr(exc, "status_code", None)
    if code is None:
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", None)
    text = f"{type(exc).__name__}: {exc}".lower()
    is_auth = code in (401, 403) or any(
        token in text
        for token in ("token expired", "expired", "unauthorized", "401", "403")
    )
    if not is_auth:
        return None
    return (
        "your LLM API key/token looks expired or invalid — run `care init` "
        "to set a fresh key, or refresh it in ~/.config/care/config.toml, "
        "then `care doctor` to confirm."
    )


class CliMageError(RuntimeError):
    """Raised when the CLI can't materialise a MAGEGenerator —
    missing `mmar_mage` install, malformed `MAGEConfig`, missing
    LLM credentials. Carries a single-line stderr message."""


def _build_mage_generator(*, mode: str | None = None):
    """Construct a :class:`MAGEGenerator` from `CareConfig.mage`.

    Tests inject a stub via :data:`_BUILD_MAGE_OVERRIDE`. The
    override receives the resolved ``mode`` so test fixtures can
    assert on it without re-parsing argv.
    """
    if _BUILD_MAGE_OVERRIDE is not None:
        return _BUILD_MAGE_OVERRIDE(mode)
    try:
        from mmar_mage import MAGEConfig, MAGEGenerator
    except Exception as exc:  # noqa: BLE001
        raise CliMageError(
            f"mmar_mage isn't installed: {exc}. "
            "Install with `pip install 'care[mage]'`.",
        ) from exc
    try:
        config = CareConfig.load()
    except Exception as exc:  # noqa: BLE001
        raise CliMageError(f"failed to load CareConfig: {exc}") from exc
    try:
        mage_config_kwargs: dict = {}
        if mode is not None:
            mage_config_kwargs["mode"] = mode
        # Forward the user's `care.toml` MAGE knobs (provider,
        # model, api_key, base_url) when present so the CLI uses
        # the same LLM credentials as the TUI.
        care_mage = getattr(config, "mage", None)
        if care_mage is not None:
            for field in ("provider", "model", "api_key", "base_url"):
                value = getattr(care_mage, field, None)
                if value:
                    mage_config_kwargs[field] = value
        mage_config = MAGEConfig(**mage_config_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise CliMageError(
            f"failed to construct MAGEConfig: {exc}",
        ) from exc
    try:
        return MAGEGenerator(config=mage_config)
    except Exception as exc:  # noqa: BLE001
        raise CliMageError(
            f"failed to construct MAGEGenerator: {exc}",
        ) from exc


# Test-injectable hook. Production callers must not set this.
_BUILD_MAGE_OVERRIDE = None


def _cmd_run(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Fetch a chain from Memory, preflight it, and optionally
    execute it via CARL.

    The fetch + validate path always runs. ``--export`` also
    writes the chain to a file via :func:`care.export_chain`.
    ``--execute`` builds a CARL executor (real LLM client +
    `ReasoningContext`) and runs the chain end-to-end.
    """
    from care.chain_export import ChainExportError, export_chain
    from care.preflight import validate_chain

    log = _configure_run_logger(
        getattr(args, "log", None),
        getattr(args, "log_level", "info"),
    )
    log.info(
        "run invoked: chain_id=%s channel=%s execute=%s export=%s save_result=%s",
        args.chain_id,
        args.channel,
        bool(args.execute),
        args.export,
        args.save_result,
    )

    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        log.error("memory backend unavailable: %s", exc)
        stderr.write(f"care run: {exc}\n")
        return 2

    log.info("fetching chain %s (channel=%s)", args.chain_id, args.channel)
    try:
        chain_dict = memory.get_chain(args.chain_id, channel=args.channel)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "fetch failed for %s: %s", args.chain_id, exc, exc_info=True,
        )
        stderr.write(
            f"care run: failed to fetch chain {args.chain_id!r}: {exc}\n",
        )
        return 2

    raw = json.dumps(chain_dict)
    log.debug("fetched chain payload (%d bytes): %s", len(raw), raw)
    result = validate_chain(raw)
    log.info(
        "preflight: parsed=%s ok=%s missing_tools=%d missing_mcp=%d missing_skills=%d",
        result.parsed,
        result.ok,
        len(result.missing_tools),
        len(result.missing_mcp_servers),
        len(result.missing_skills),
    )
    if result.parse_errors:
        log.info("preflight parse_errors: %s", list(result.parse_errors))

    if args.json:
        payload = {
            "entity_id": args.chain_id,
            "channel": args.channel,
            "parsed": result.parsed,
            "ok": result.ok,
            "preflight_available": result.preflight_available,
            "parse_errors": list(result.parse_errors),
            "required_tools": list(result.required_tools),
            "required_mcp_servers": list(result.required_mcp_servers),
            "required_skills": list(result.required_skills),
            "missing_tools": list(result.missing_tools),
            "missing_mcp_servers": list(result.missing_mcp_servers),
            "missing_skills": list(result.missing_skills),
        }
        json.dump(payload, stdout, indent=2)
        stdout.write("\n")
    else:
        stdout.write(f"chain: {args.chain_id} (channel={args.channel})\n")
        stdout.write(result.format_text() + "\n")

    if args.export is not None:
        log.info("exporting chain to %s (format=%s)", args.export, args.export_format)
        try:
            export = export_chain(
                chain_dict,
                args.export,
                format=args.export_format,
            )
        except ChainExportError as exc:
            log.error("export failed: %s", exc, exc_info=True)
            stderr.write(f"care run: export failed: {exc}\n")
            return 2
        log.info(
            "exported: path=%s format=%s bytes=%d",
            export.path, export.format, export.bytes_written,
        )
        stdout.write(
            f"exported: {export.path} "
            f"({export.format}, {export.bytes_written} bytes)\n",
        )

    if not result.parsed:
        # Even with --execute we refuse to run an unparseable
        # chain — surface the preflight failure as exit 1 and
        # skip execution.
        log.info("chain failed to parse; skipping execution.")
        return 1

    if args.save_result and not args.execute:
        log.error("--save-result requires --execute; aborting.")
        stderr.write(
            "care run: --save-result requires --execute (nothing to "
            "persist without an actual run).\n"
        )
        return 2

    if args.execute:
        log.info("building CARL executor")
        try:
            executor = _build_carl_executor()
        except CliCarlError as exc:
            log.error("executor unavailable: %s", exc, exc_info=True)
            stderr.write(f"care run: {exc}\n")
            return 2
        try:
            inputs = _parse_input_pairs(args.input)
        except ValueError as exc:
            log.error("input parse error: %s", exc)
            stderr.write(f"care run: {exc}\n")
            return 2
        try:
            file_inputs = _read_file_inputs(args.file)
        except ValueError as exc:
            log.error("file input error: %s", exc)
            stderr.write(f"care run: {exc}\n")
            return 2
        # Explicit `--input KEY=VALUE` text wins over a `--file` whose
        # basename collides with it; file contents only fill keys the
        # user didn't set directly.
        for _k, _v in file_inputs.items():
            inputs.setdefault(_k, _v)
        _warn_missing_context_files(chain_dict, inputs, stderr)
        # Document-skill bridge: if the chain has a doc-reading skill step,
        # rewrite it to read the --file attachment ($memory.input.<key> + a
        # task placeholder) — the same bridge the chat path uses. The
        # read-vs-create decision uses the keyword heuristic by default, or
        # the LLM when `--classify-files model` is passed.
        chain_dict, inputs = _apply_cli_skill_bridge(
            chain_dict, args.file, inputs,
            classify=getattr(args, "classify_files", "heuristic"),
        )
        log.info(
            "executing chain: task=%r inputs=%s",
            args.task, list(inputs.keys()),
        )
        import time as _time

        _started_at = _time.time()
        try:
            run_result = asyncio.run(
                executor(chain_dict, task=args.task, inputs=inputs),
            )
        except Exception as exc:  # noqa: BLE001
            log.error("execution failed: %s", exc, exc_info=True)
            stderr.write(f"care run: execution failed: {exc}\n")
            _record_cli_run(
                chain_dict=chain_dict,
                chain_id=args.chain_id,
                task=args.task or "",
                result=None,
                started_at=_started_at,
                duration=_time.time() - _started_at,
                status="failure",
                error=str(exc),
            )
            return 2
        log.info("execution succeeded=%s", _run_succeeded(run_result))
        log.debug("run_result=%r", run_result)
        _record_cli_run(
            chain_dict=chain_dict,
            chain_id=args.chain_id,
            task=args.task or "",
            result=run_result,
            started_at=_started_at,
            duration=_time.time() - _started_at,
            status=(
                "success" if _run_succeeded(run_result)
                else "failure"
            ),
            error=(
                "" if _run_succeeded(run_result)
                else "chain reported failure"
            ),
        )
        _render_run_result(run_result, stdout=stdout, as_json=args.json)

        if args.save_result and _run_succeeded(run_result):
            try:
                card_id = _persist_run_result(
                    memory=memory,
                    chain_id=args.chain_id,
                    chain_dict=chain_dict,
                    result=run_result,
                    name=args.save_result,
                    task=args.task,
                    inputs=inputs,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("save-result failed: %s", exc, exc_info=True)
                stderr.write(
                    f"care run: save-result failed: {exc}\n",
                )
                return 2
            log.info("saved-result card_id=%s", card_id)
            stdout.write(f"saved-result: {card_id}\n")

        if not _run_succeeded(run_result):
            log.info("execution did not succeed; exiting with status 1.")
            return 1
    log.info("run finished successfully.")
    return 0


def _configure_run_logger(
    log_path: Path | None, level_name: str,
):
    """Return a logger for `care run`. When ``log_path`` is set,
    attach a FileHandler at the requested level so the call is
    recorded as a structured timeline; otherwise return a logger
    that drops every record (verbose code paths stay cheap).
    """
    import logging

    logger = logging.getLogger("care.cli.run")
    # Reset existing handlers so repeated invocations don't
    # cumulatively attach new file handles to the same logger.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    logger.propagate = False

    if log_path is None:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)
        return logger

    level = logging.DEBUG if level_name == "debug" else logging.INFO
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    except OSError:
        # If we can't write the log, fall back to a no-op
        # logger rather than aborting the run.
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)
        return logger
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ),
    )
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def _persist_run_result(
    *,
    memory,
    chain_id: str,
    chain_dict: dict,
    result,
    name: str,
    task: str | None,
    inputs: dict[str, str],
) -> str:
    """Save the execution result as a memory_card.

    Delegates to :func:`care.runtime.record_run_completion`
    so saved-result cards land with the canonical tags
    (``agent_run`` + ``agent:<chain_id>`` + ``status:...``) the
    InspectionScreen's Run History tab already queries. The
    user-supplied ``name`` becomes a custom tag
    (``label:<name>``) so the same name can be reused across
    runs without colliding on Memory's display_name.

    Returns the entity_id of the saved memory_card.
    """
    from care.runtime import record_run_completion

    agent_name = _chain_display_name(chain_dict) or chain_id
    completion = record_run_completion(
        memory,
        agent_entity_id=chain_id,
        agent_name=agent_name,
        result=result,
        query=task,
        extra_tags=[f"label:{name}"] if name else None,
    )
    return completion.memory_card_entity_id


def _chain_display_name(chain_dict: dict) -> str | None:
    """Best-effort extraction of the chain's display name from
    the standard ``metadata.care.display_name`` location."""
    meta = chain_dict.get("metadata") if isinstance(chain_dict, dict) else None
    if not isinstance(meta, dict):
        return None
    care = meta.get("care")
    if not isinstance(care, dict):
        return None
    name = care.get("display_name") or care.get("name")
    if isinstance(name, str) and name:
        return name
    return None


def _parse_input_pairs(
    pairs: list[str] | None,
) -> dict[str, str]:
    """Parse `--input KEY=VALUE` pairs into a dict.

    Empty / None → empty dict. Raises ``ValueError`` (with a
    user-friendly message) on missing ``=`` so the handler can
    surface it as a single stderr line + exit 2.
    """
    out: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(
                f"--input {pair!r} must be in KEY=VALUE form",
            )
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(
                f"--input {pair!r} has an empty key",
            )
        out[key] = value
    return out


def _read_file_inputs(paths: list[str] | None) -> dict[str, str]:
    """Read `--file PATH` arguments into ``{basename: contents}``.

    Keyed by the file's basename so a step's ``${input.<basename>}``
    reference resolves — the same key CARL's ``from_chain_inputs``
    uses when auto-loading a chain's stored context files. Raises
    ``ValueError`` (surfaced as a stderr line + exit 2) when a path
    isn't a readable file.
    """
    from care.runtime.file_loading import load_file

    out: dict[str, str] = {}
    for raw in paths or []:
        p = Path(raw).expanduser()
        if not p.is_file():
            raise ValueError(f"--file {raw!r}: not a file")
        # Canonical loader: binary-safe (no UnicodeDecodeError crash),
        # size-capped, office/pdf extracted, images → data URI.
        loaded = load_file(p)
        if loaded.error and not loaded.memory_value:
            raise ValueError(f"--file {raw!r}: {loaded.error}")
        out[p.name] = loaded.memory_value
    return out


# Test seam: stub the classifier LLM client without real creds.
_BUILD_CLASSIFIER_API_OVERRIDE = None


def _build_classifier_api() -> Any:
    """Build an LLM client for `--classify-files model`. ``None`` on any
    failure (missing key / import) → caller falls back to the heuristic."""
    if _BUILD_CLASSIFIER_API_OVERRIDE is not None:
        return _BUILD_CLASSIFIER_API_OVERRIDE()
    try:
        from care.runtime.llm_client import build_carl_llm_client

        return build_carl_llm_client(CareConfig.load().mage)
    except Exception:  # noqa: BLE001
        return None


def _apply_cli_skill_bridge(
    chain_dict: Any,
    file_args: list[str] | None,
    inputs: dict[str, str],
    *,
    classify: str = "heuristic",
) -> tuple[Any, dict[str, str]]:
    """Wire a ``--file`` attachment into a document-reading skill step.

    If the chain has a docx/pdf/… read step, rewrite it (via
    :func:`care.skill_file_inputs.apply_file_inputs`) to read the file from
    ``$memory.input.<key>`` with a ``{param}`` placeholder in the skill task,
    and merge the file payload into ``inputs``. ``classify="model"`` asks the
    LLM to decide read-vs-create; otherwise the keyword heuristic is used.
    Returns the (possibly rewritten) ``chain_dict`` + ``inputs``. Best-effort —
    any failure leaves both unchanged so a normal run is never blocked.
    """
    if not file_args:
        return chain_dict, inputs
    try:
        from care.runtime.file_loading import load_file
        from care.skill_file_inputs import (
            apply_file_inputs,
            requires_file_input,
        )

        reads = None
        if classify == "model":
            api = _build_classifier_api()
            if api is not None:
                try:
                    from care.skill_file_inputs import classify_reads

                    reads = asyncio.run(classify_reads(api, chain_dict))
                except Exception:  # noqa: BLE001 — heuristic fallback
                    reads = None

        if not requires_file_input(chain_dict, reads=reads):
            return chain_dict, inputs
        attachments = [
            (raw, load_file(raw).content)
            for raw in file_args
            if Path(raw).expanduser().is_file()
        ]
        if not attachments:
            return chain_dict, inputs
        new_dict, skill_files = apply_file_inputs(
            chain_dict, attachments, reads=reads,
        )
        merged = dict(inputs)
        merged.update(skill_files)
        return new_dict, merged
    except Exception:  # noqa: BLE001 — never block a run on the bridge
        return chain_dict, inputs


def _warn_missing_context_files(
    chain_dict: Any, inputs: dict[str, str], stderr: Any,
) -> None:
    """Warn when a saved chain expects context files that aren't on
    disk and weren't supplied via ``--file`` / ``--input``.

    Without this, ``from_chain_inputs`` silently primes the run with a
    ``[missing context file: …]`` placeholder and the chain executes
    against empty inputs — the headless twin of the TUI's
    "required files missing" banner. Best-effort + non-fatal: a chain
    with no metadata, or one whose files all resolve, prints nothing.
    """
    try:
        from care.runtime.run_context_draft import (
            extract_run_context_draft,
            missing_active_files,
        )

        draft = extract_run_context_draft(chain_dict)
    except Exception:  # noqa: BLE001
        return
    missing = [
        cf
        for cf in missing_active_files(draft)
        if Path(cf.path).name not in inputs
    ]
    if not missing:
        return
    names = ", ".join(Path(cf.path).name for cf in missing)
    stderr.write(
        "care run: warning — this chain expects context file(s) not found "
        f"on disk and not provided via --file/--input: {names}. The run "
        "will proceed with empty/placeholder inputs; pass `--file <path>` "
        "to supply them.\n"
    )


def _record_cli_run(
    *,
    chain_dict,
    chain_id: str,
    task: str,
    result,
    started_at: float,
    duration: float,
    status: str,
    error: str = "",
) -> None:
    """§6 P1 — Record one `care run --execute` invocation as
    a `LocalRunEntry` under `~/.cache/care/runs/`.

    Mirrors the ChatScreen recorder; failures are swallowed
    at WARNING so a broken cache can't kill the CLI run.
    `chain_dict` is the dict CARL receives; we project a
    duck-typed object with `entity_id` (from CLI arg) +
    `name` (from the chain metadata) so the shared
    `build_run_entry` projection works unchanged.
    """
    import logging as _logging
    import time as _time
    from types import SimpleNamespace

    try:
        from care.runtime.local_run_history import (
            build_run_entry,
            record_local_run,
        )
    except Exception:
        return
    log = _logging.getLogger("care.cli.run")
    run_id = (
        "cli-"
        + _time.strftime(
            "%Y%m%dT%H%M%SZ", _time.gmtime(started_at),
        )
        + f"-{int((started_at * 1000) % 1000):03d}"
    )
    # Extract a display name from the chain payload — the
    # CARL chain object isn't constructed here (the
    # `executor` closure does that internally) so we read
    # the dict's metadata directly.
    chain_name = ""
    if isinstance(chain_dict, dict):
        meta = chain_dict.get("metadata") or {}
        if isinstance(meta, dict):
            care_meta = meta.get("care") or {}
            if isinstance(care_meta, dict):
                chain_name = str(
                    care_meta.get("display_name") or "",
                )
            chain_name = chain_name or str(
                meta.get("display_name") or "",
            )
    fake_chain = SimpleNamespace(
        entity_id=chain_id, name=chain_name,
    )
    provider = ""
    try:
        from care.config import CareConfig

        provider = str(
            getattr(
                getattr(CareConfig.load(), "mage", None),
                "provider", "",
            ) or ""
        )
    except Exception:
        provider = ""
    try:
        record_local_run(
            build_run_entry(
                run_id=run_id,
                chain=fake_chain,
                task=task,
                result=result,
                started_at=started_at,
                duration=duration,
                status=status,
                error=error,
                mode="cli",
                provider=provider,
                write_replay=True,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "local run history record failed: %s", exc,
            exc_info=False,
        )


def _render_run_result(result, *, stdout: TextIO, as_json: bool) -> None:
    """Print the executed chain's outcome in either text or JSON.

    Duck-typed against `mmar_carl.ReasoningResult`: probes
    `.success`, `.step_results`, `.final_answer`. Stubs in
    tests carry the same shape so the renderer doesn't reach
    through `isinstance`.
    """
    success = bool(getattr(result, "success", False))
    steps = getattr(result, "step_results", None) or []
    final_answer = getattr(result, "final_answer", None)
    if as_json:
        payload = {
            "executed": True,
            "success": success,
            "steps_completed": len(steps),
            "final_answer": (
                str(final_answer) if final_answer is not None else None
            ),
        }
        json.dump(payload, stdout, indent=2, default=str)
        stdout.write("\n")
        return
    status = "ok" if success else "failed"
    stdout.write(
        f"executed: status={status}, steps={len(steps)}\n",
    )
    if final_answer:
        snippet = str(final_answer)
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        stdout.write(f"final: {snippet}\n")


def _run_succeeded(result) -> bool:
    return bool(getattr(result, "success", False))


class CliCarlError(RuntimeError):
    """Raised when the CLI can't materialise a CARL executor —
    missing `mmar_carl`, missing LLM credentials, malformed
    config. Carries a single-line stderr message."""


def _build_carl_executor():
    """Construct an async ``(chain_dict) -> ReasoningResult``
    executor backed by CARL.

    Production path lazy-imports `mmar_carl`, reads
    `CareConfig.mage.*` for LLM credentials, builds an
    `OpenAICompatibleClient`, and returns a closure that
    bootstraps a `ReasoningContext.from_chain_inputs` per call.

    Tests inject a stub via
    :data:`_BUILD_CARL_EXECUTOR_OVERRIDE` that bypasses every
    `mmar_carl` import + LLM credential check.
    """
    if _BUILD_CARL_EXECUTOR_OVERRIDE is not None:
        return _BUILD_CARL_EXECUTOR_OVERRIDE()
    try:
        from mmar_carl import (
            OpenAIClientConfig,
            OpenAICompatibleClient,
            ReasoningChain,
            ReasoningContext,
        )
    except Exception as exc:  # noqa: BLE001
        raise CliCarlError(
            f"mmar_carl isn't installed: {exc}. "
            "Install with `pip install 'care[carl]'`.",
        ) from exc
    try:
        config = CareConfig.load()
    except Exception as exc:  # noqa: BLE001
        raise CliCarlError(f"failed to load CareConfig: {exc}") from exc
    mage = getattr(config, "mage", None)
    api_key = getattr(mage, "api_key", None)
    if not api_key:
        raise CliCarlError(
            "mage.api_key isn't set — `care run --execute` needs an LLM "
            "API key. Set `CARE_MAGE__API_KEY` or `care.toml [mage] api_key`.",
        )
    model = getattr(mage, "model", None) or "openai/gpt-4o"
    base_url = (
        getattr(mage, "base_url", None) or "https://openrouter.ai/api/v1"
    )
    try:
        client_config = OpenAIClientConfig(
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        api_client = OpenAICompatibleClient(client_config)
    except Exception as exc:  # noqa: BLE001
        raise CliCarlError(
            f"failed to construct CARL LLM client: {exc}",
        ) from exc

    async def _execute(chain_dict, *, task=None, inputs=None):
        chain = ReasoningChain.from_dict(chain_dict, use_typed_steps=True)
        kwargs: dict[str, Any] = {"api": api_client}
        if task is not None:
            kwargs["outer_context"] = task
        if inputs:
            kwargs["files"] = dict(inputs)
        ctx = ReasoningContext.from_chain_inputs(chain, **kwargs)
        return await chain.execute_async(ctx)

    return _execute


# Test-injectable hook. Production callers must not set this.
_BUILD_CARL_EXECUTOR_OVERRIDE = None


def _cmd_marketplace(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Search the shared agent_skill marketplace.

    Wraps :func:`care.search_marketplace` — the same data
    layer the TUI MarketplaceScreen consumes — so terminal
    users can browse without launching the TUI. Forwards the
    facade's `client` so the SDK's `find_capability_matches`
    is what gets called.
    """
    from care.marketplace import MarketplaceError, search_marketplace

    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care marketplace: {exc}\n")
        return 2

    client = getattr(memory, "client", None) or memory
    try:
        result = search_marketplace(
            client,
            args.query,
            top_k=args.top_k,
            min_score=args.min_score,
            tags=args.tag,
            namespace=args.namespace,
            deep=args.deep,
        )
    except MarketplaceError as exc:
        stderr.write(f"care marketplace: {exc}\n")
        return 2
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care marketplace: lookup failed: {exc}\n")
        return 2

    if args.json:
        payload = {
            "query": result.query,
            "namespace": args.namespace,
            "deep": args.deep,
            "listings": [
                {
                    "entity_id": li.entity_id,
                    "name": li.name,
                    "description": li.description,
                    "score": li.score,
                    "tags": list(li.tags),
                    "matched_via": li.matched_via,
                    "snippet": li.snippet,
                }
                for li in result.listings
            ],
        }
        json.dump(payload, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    if result.is_empty:
        stdout.write(
            f"marketplace: no listings for {args.query!r}\n",
        )
        return 0

    stdout.write(
        f"marketplace: {len(result.listings)} listing(s) "
        f"for {args.query!r}\n",
    )
    for li in result.listings:
        stdout.write(_format_marketplace_listing(li) + "\n")
    return 0


def _format_marketplace_listing(listing) -> str:
    """One-line summary of a single MarketplaceListing."""
    badge = "★" if listing.matched_via == "skill_description" else " "
    score = f"{listing.score:.3f}"
    tags = ", ".join(listing.tags) if listing.tags else ""
    suffix = f"  [{tags}]" if tags else ""
    return (
        f"  {badge}  {score}  "
        f"{listing.entity_id[:18]}  {listing.name}{suffix}"
    )


def _cmd_favourite(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Toggle the favourite flag on a library entity.

    Wraps :meth:`CareMemory.mark_favourite` so the CLI mirrors
    the TUI LibraryScreen's ``F`` key binding from the terminal.
    """
    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care favourite: {exc}\n")
        return 2

    value = not args.off
    try:
        response = memory.mark_favourite(
            args.entity_id,
            entity_type=args.entity_type,
            value=value,
        )
    except Exception as exc:  # noqa: BLE001
        stderr.write(
            f"care favourite: failed to {'star' if value else 'unstar'} "
            f"{args.entity_type} {args.entity_id!r}: {exc}\n",
        )
        return 2

    if args.json:
        json.dump(response, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    badge = "★" if value else " "
    name = (
        response.get("display_name")
        or (response.get("meta") or {}).get("name")
        or args.entity_id
    )
    verb = "starred" if value else "unstarred"
    stdout.write(
        f"{badge} {verb}: {args.entity_id} ({args.entity_type}) — {name}\n",
    )
    return 0


def _channels_by_version(memory: Any, entity_id: str) -> dict[str, list[str]]:
    """Map version_id → [channel names] for the entity, best-effort.

    Reads the entity's ``channels`` dict (Memory stores channel → version
    pointers there) so `care versions` can annotate which version each
    channel points at. Returns ``{}`` on any lookup failure."""
    try:
        entity = memory.get_entity(entity_id)
    except Exception:  # noqa: BLE001
        return {}
    channels = (entity or {}).get("channels") if isinstance(entity, dict) else None
    out: dict[str, list[str]] = {}
    if isinstance(channels, dict):
        for channel, version_id in channels.items():
            if version_id:
                out.setdefault(str(version_id), []).append(str(channel))
    return out


def _cmd_versions(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """List an entity's version history (CLI twin of TUI ``/versions``)."""
    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care versions: {exc}\n")
        return 2
    try:
        versions = memory.list_versions(
            args.entity_id, entity_type=args.entity_type, limit=args.limit
        )
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care versions: lookup failed: {exc}\n")
        return 2

    channel_of = _channels_by_version(memory, args.entity_id)

    if args.json:
        payload = [
            v.model_dump() if hasattr(v, "model_dump") else v for v in versions
        ]
        json.dump({"versions": payload}, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    if not versions:
        stdout.write(f"versions: none for {args.entity_id}\n")
        return 0

    for version in versions:
        number = getattr(version, "version_number", "?")
        vid = str(getattr(version, "version_id", "") or "")
        parts = [f"v{number}", vid[:12]]
        created = getattr(version, "created_at", None)
        if created is not None:
            parts.append(str(created)[:10])
        summary = getattr(version, "change_summary", None)
        if summary:
            parts.append(str(summary)[:50])
        line = "● " + " · ".join(parts)
        chans = channel_of.get(vid)
        if chans:
            line += "  ← " + ", ".join(chans)
        stdout.write(line + "\n")
    return 0


def _cmd_rollback(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Repoint a channel at a specific version (CLI twin of ``/rollback``)."""
    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care rollback: {exc}\n")
        return 2
    try:
        memory.pin_channel(
            args.entity_id,
            args.channel,
            args.to_version,
            entity_type=args.entity_type,
        )
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care rollback: failed: {exc}\n")
        return 2
    stdout.write(
        f"rolled back: {args.entity_id} channel {args.channel!r} → version "
        f"{args.to_version}\n"
    )
    return 0


def _cmd_promote(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Copy one channel pointer to another (CLI twin of ``/promote``).

    This is the direct channel move; the TUI ``/promote`` layers an
    interactive baseline/eval gate on top that doesn't fit a headless run.
    """
    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care promote: {exc}\n")
        return 2
    try:
        memory.promote(
            args.entity_id,
            from_channel=args.from_channel,
            to_channel=args.to_channel,
            entity_type=args.entity_type,
        )
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care promote: failed: {exc}\n")
        return 2
    stdout.write(
        f"promoted: {args.entity_id} {args.from_channel!r} → "
        f"{args.to_channel!r}\n"
    )
    return 0


def _cmd_forget(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Soft-delete a saved entity (CLI twin of TUI ``/forget``).

    Two-step UX: without ``--force`` it only previews so a fat-finger
    can't nuke an entity."""
    if not args.force:
        stdout.write(
            f"Would delete {args.entity_type} {args.entity_id}. "
            f"Re-run with --force to actually delete "
            f"(soft-delete — recoverable via Memory trash).\n"
        )
        return 0
    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care forget: {exc}\n")
        return 2
    try:
        ok = memory.delete_entity(args.entity_id, entity_type=args.entity_type)
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care forget: failed: {exc}\n")
        return 2
    if not ok:
        stderr.write(
            f"care forget: {args.entity_type} {args.entity_id} not found "
            f"(or already deleted).\n"
        )
        return 1
    stdout.write(f"forgot: {args.entity_type} {args.entity_id} (soft-deleted)\n")
    return 0


def _cmd_revise(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """AI-edit a saved chain into a new version (CLI twin of ``/revise``).

    MAGE plans a minimal targeted edit; without ``--yes`` the CLI just
    previews the plan, mirroring the TUI's confirm-before-save flow.
    """
    from care.generation import GenerationError, run_edit
    from care.runtime.chain_edit_view import (
        render_disambiguation_lines,
        render_edit_plan_lines,
        revise_result_has_changes,
    )

    instruction = " ".join(args.change).strip()
    if not instruction:
        stderr.write("care revise: change instruction is empty\n")
        return 2

    try:
        generator = _build_mage_generator(mode=args.mode)
    except CliMageError as exc:
        stderr.write(f"care revise: {exc}\n")
        return 2

    try:
        result = asyncio.run(
            run_edit(
                generator,
                instruction,
                entity_id=args.chain_id,
                channel=args.channel,
                save=False,
            )
        )
    except (GenerationError, Exception) as exc:  # noqa: BLE001
        hint = _friendly_llm_error(exc)
        if hint:
            stderr.write(f"care revise: {hint}\n")
        else:
            stderr.write(f"care revise: edit failed: {exc}\n")
        return 2

    if getattr(result, "needs_disambiguation", False):
        stderr.write("care revise: the chain reference was ambiguous:\n")
        for line in render_disambiguation_lines(result):
            stderr.write(f"  {line}\n")
        stderr.write("Re-run with an explicit chain id.\n")
        return 1

    if not revise_result_has_changes(result):
        stdout.write("care revise: MAGE proposed no changes.\n")
        return 0

    chain_dict = dict(getattr(result, "chain_dict", None) or {})

    if args.json:
        json.dump(chain_dict, stdout, indent=2, default=str)
        stdout.write("\n")
    else:
        stdout.write("Planned edit:\n")
        for line in render_edit_plan_lines(result):
            stdout.write(f"  {line}\n")

    if not args.yes:
        stdout.write(
            "\nPreview only — re-run with --yes to save this as a new "
            "version.\n"
        )
        return 0

    if not chain_dict:
        stderr.write("care revise: edited chain is empty — nothing to save.\n")
        return 2

    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care revise: {exc}\n")
        return 2

    # Reuse the existing chain's name for the new version.
    name = args.chain_id
    try:
        entity = memory.get_entity(args.chain_id)
        if isinstance(entity, dict):
            name = (
                entity.get("display_name")
                or (entity.get("meta") or {}).get("name")
                or name
            )
    except Exception:  # noqa: BLE001
        pass

    try:
        new_id = memory.save_chain(
            chain_dict,
            name=name,
            entity_id=args.chain_id,
            channel=args.channel,
            change_summary=f"revise: {instruction}"[:200],
        )
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care revise: save failed: {exc}\n")
        return 2

    stdout.write(f"\nsaved new version: {new_id}\n")
    return 0


class CliHubError(RuntimeError):
    """Raised when the CLI can't build a HubClient (hub base_url unset /
    agent_hub import failed). Carries a single-line stderr message."""


# Test-injectable hook (returns a HubClient-like object).
_BUILD_HUB_OVERRIDE = None


def _build_hub_client():
    """Construct a :class:`HubClient` from ``CareConfig.hub.base_url``."""
    if _BUILD_HUB_OVERRIDE is not None:
        return _BUILD_HUB_OVERRIDE()
    try:
        from care.runtime.agent_hub import HubClient
    except Exception as exc:  # noqa: BLE001
        raise CliHubError(f"failed to import HubClient: {exc}") from exc
    try:
        config = CareConfig.load()
    except Exception as exc:  # noqa: BLE001
        raise CliHubError(f"failed to load CareConfig: {exc}") from exc
    base = getattr(getattr(config, "hub", None), "base_url", None)
    if not base:
        raise CliHubError("hub base_url is not configured ([hub].base_url)")
    return HubClient(base)


def _cli_agent_slug(text: str) -> str:
    """Display name / id → url-safe agent name (mirror of the TUI slug)."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", (text or "").lower()).strip("-._")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:63] or "agent"


def _cmd_deploy(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Deploy a saved chain to the agent hub (CLI twin of ``/deploy``).

    Deploys by ``entity_id`` — the hub fetches the chain on the given
    channel and validates loadability itself (a 422 surfaces here).
    Requires the hub to be running (the TUI autostarts it; headless does
    not spin up a background process)."""
    import secrets

    from care.runtime.agent_hub import HubError, HubUnavailableError

    try:
        hub = _build_hub_client()
    except CliHubError as exc:
        stderr.write(f"care deploy: {exc}\n")
        return 2

    spec = {
        "name": args.name or _cli_agent_slug(args.chain_id),
        "entity_id": args.chain_id,
        "channel": args.channel,
        "api_key": secrets.token_urlsafe(24),
    }
    try:
        deployment = asyncio.run(hub.deploy(spec))
    except HubUnavailableError as exc:
        stderr.write(
            f"care deploy: hub is not running ({exc}). Start the agent hub "
            f"first, then retry.\n"
        )
        return 2
    except HubError as exc:
        stderr.write(f"care deploy: rejected: {exc}\n")
        return 2
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care deploy: failed: {exc}\n")
        return 2

    url = getattr(hub, "agent_url", lambda n: "")(deployment.name)
    stdout.write(f"deployed: {deployment.name}")
    if url:
        stdout.write(f"  →  {url}  (docs: {url}/docs)")
    stdout.write("\n")
    return 0


def _cmd_deployments(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """List agents deployed on the hub (CLI twin of ``/deployments``)."""
    from care.runtime.agent_hub import HubError, HubUnavailableError

    try:
        hub = _build_hub_client()
    except CliHubError as exc:
        stderr.write(f"care deployments: {exc}\n")
        return 2
    try:
        deployments = asyncio.run(hub.list_deployments())
    except HubUnavailableError as exc:
        stderr.write(f"care deployments: hub is not running ({exc}).\n")
        return 2
    except HubError as exc:
        stderr.write(f"care deployments: {exc}\n")
        return 2

    if args.json:
        payload = [d.__dict__ for d in deployments]
        json.dump({"deployments": payload}, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    if not deployments:
        stdout.write("deployments: none\n")
        return 0
    for dep in deployments:
        badge = "●" if dep.ready else "○"
        line = f"{badge} {dep.name}  {dep.url}  v{dep.version}  runs={dep.runs}"
        if not dep.ready and dep.ready_reason:
            line += f"  ({dep.ready_reason})"
        stdout.write(line + "\n")
    return 0


def _cmd_metrics(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Show usage/cost metrics for deployed agents (CLI twin of ``/metrics``).

    With a name, fetches that agent's ``/metrics``; without one, summarises
    run counts across all deployments."""
    from care.runtime.agent_hub import HubError, HubUnavailableError

    try:
        hub = _build_hub_client()
    except CliHubError as exc:
        stderr.write(f"care metrics: {exc}\n")
        return 2

    try:
        if args.name:
            metrics = asyncio.run(hub.agent_metrics(args.name))
            if metrics is None:
                stdout.write(
                    f"metrics: agent {args.name!r} exposes none "
                    f"(older build or not deployed)\n"
                )
                return 0
            if args.json:
                json.dump(metrics, stdout, indent=2, default=str)
                stdout.write("\n")
            else:
                for key, value in metrics.items():
                    stdout.write(f"  {key}: {value}\n")
            return 0

        deployments = asyncio.run(hub.list_deployments())
    except HubUnavailableError as exc:
        stderr.write(f"care metrics: hub is not running ({exc}).\n")
        return 2
    except HubError as exc:
        stderr.write(f"care metrics: {exc}\n")
        return 2

    if args.json:
        payload = {d.name: {"runs": d.runs, "ready": d.ready} for d in deployments}
        json.dump(payload, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0
    if not deployments:
        stdout.write("metrics: no deployments\n")
        return 0
    total = sum(d.runs for d in deployments)
    for dep in deployments:
        stdout.write(f"  {dep.name}: runs={dep.runs}\n")
    stdout.write(f"  total runs: {total}\n")
    return 0


def _cmd_remember(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Save an explicit note to long-term memory (CLI twin of ``/remember``).

    Stores the note verbatim (never silently lost). The TUI's LLM-merge
    dedup is TUI-only; the CLI relies on the same raw-note fallback."""
    from care import memory_ltm

    note = " ".join(args.note).strip()
    if not note:
        stderr.write("care remember: note is empty\n")
        return 2
    try:
        config = CareConfig.load()
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care remember: failed to load config: {exc}\n")
        return 2
    ltm = memory_ltm.build_ltm(config)
    if ltm is None:
        stderr.write(
            "care remember: long-term memory is disabled — enable it with "
            "CARE_CONTEXT__LTM_ENABLED=true (needs the `carl` extra).\n"
        )
        return 2
    session_id = memory_ltm.ltm_session_id(config)
    existing = memory_ltm.recall_digest(ltm, session_id, max_chars=2000)
    saved = memory_ltm.remember_text(
        ltm,
        session_id,
        content=note,
        complete=lambda _system, _user: "",  # no LLM → raw-note fallback
        existing_digest=existing,
    )
    if not saved:
        stderr.write("care remember: nothing was saved\n")
        return 1
    stdout.write(memory_ltm.format_saved(saved) + "\n")
    return 0


def _cmd_notes(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Show long-term memory notes (CLI twin of TUI ``/memory``)."""
    from care import memory_ltm

    try:
        config = CareConfig.load()
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care notes: failed to load config: {exc}\n")
        return 2
    ltm = memory_ltm.build_ltm(config)
    if ltm is None:
        stdout.write(
            "notes: long-term memory is disabled "
            "(set CARE_CONTEXT__LTM_ENABLED=true).\n"
        )
        return 0
    session_id = memory_ltm.ltm_session_id(config)
    digest = memory_ltm.recall_digest(
        ltm, session_id, max_chars=args.max_chars
    )
    stdout.write((digest or "notes: long-term memory is empty.") + "\n")
    return 0


def _cmd_dataset(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Manage a chain's eval dataset (CLI twin of TUI ``/dataset``).

    ``list`` / ``add`` / ``export`` are pure memory-card operations;
    ``run`` replays each entry through the chain (CARL executor) and
    scores by case-insensitive substring (the TUI's deterministic
    default; its LLM-judge rubric path stays in the TUI)."""
    from care.dataset import (
        add_dataset_entry,
        collect_dataset_entries,
        entry_passes,
        export_entries_jsonl,
    )

    action = args.dataset_action
    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care dataset: {exc}\n")
        return 2

    if action == "add":
        try:
            eid = add_dataset_entry(
                memory,
                args.chain_id,
                args.task,
                args.expected,
                rubric=args.rubric,
            )
        except Exception as exc:  # noqa: BLE001
            stderr.write(f"care dataset add: failed: {exc}\n")
            return 2
        stdout.write(f"added dataset entry: {eid}\n")
        return 0

    # list / run / export all read the entries first.
    try:
        entries = collect_dataset_entries(memory, args.chain_id)
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care dataset: lookup failed: {exc}\n")
        return 2

    if action == "list":
        if getattr(args, "json", False):
            json.dump({"entries": entries}, stdout, indent=2, default=str)
            stdout.write("\n")
            return 0
        if not entries:
            stdout.write(f"dataset: no entries for {args.chain_id}\n")
            return 0
        for entry in entries:
            task = str(entry.get("task") or "")[:50]
            status = entry.get("status") or "pending"
            stdout.write(f"  [{status}] {task}\n")
        stdout.write(f"  ({len(entries)} entries)\n")
        return 0

    if action == "export":
        try:
            count = export_entries_jsonl(entries, args.output)
        except OSError as exc:
            stderr.write(f"care dataset export: {exc}\n")
            return 2
        stdout.write(f"exported {count} entries → {args.output}\n")
        return 0

    if action == "run":
        if not entries:
            stdout.write(f"dataset: no entries for {args.chain_id}\n")
            return 0
        try:
            chain_dict = memory.get_chain(args.chain_id)
        except Exception as exc:  # noqa: BLE001
            stderr.write(f"care dataset run: fetch chain failed: {exc}\n")
            return 2
        try:
            executor = _build_carl_executor()
        except CliCarlError as exc:
            stderr.write(f"care dataset run: {exc}\n")
            return 2

        passed = 0
        scored = 0
        for entry in entries:
            task = entry.get("task") or ""
            expected = entry.get("expected") or ""
            if not task:
                continue
            scored += 1
            try:
                result = asyncio.run(executor(chain_dict, task=task, inputs={}))
            except Exception as exc:  # noqa: BLE001
                stdout.write(f"  ✗ (error) {str(task)[:40]} — {exc}\n")
                continue
            actual = str(getattr(result, "final_answer", "") or "")
            ok = entry_passes(actual, expected)
            passed += 1 if ok else 0
            badge = "✓" if ok else "✗"
            stdout.write(f"  {badge} {str(task)[:50]}\n")
        stdout.write(f"\nscore: {passed}/{scored} passed (substring)\n")
        return 0 if passed == scored else 1

    stderr.write(f"care dataset: unknown action {action!r}\n")
    return 2


def _cmd_export(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Export saved chains + AgentSkills to a bundle tarball (CLI twin of
    the Library's ``export bundle`` action)."""
    from care.runtime.library_bundle import (
        LibraryBundleError,
        export_library_bundle,
    )

    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care export: {exc}\n")
        return 2
    try:
        result = asyncio.run(
            export_library_bundle(
                memory,
                args.entity_ids,
                args.output,
                skill_entity_ids=args.skills,
                channel=args.channel,
            )
        )
    except LibraryBundleError as exc:
        stderr.write(f"care export: {exc}\n")
        return 2
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care export: failed: {exc}\n")
        return 2

    if result.error:
        stderr.write(f"care export: {result.error}\n")
        return 1

    stdout.write(
        f"exported {result.chain_count} chain(s) + {result.skill_count} "
        f"skill(s) → {result.path} ({result.bytes_written} bytes)\n"
    )
    skipped = list(result.skipped_chains) + list(result.skipped_skills)
    if skipped:
        stdout.write(f"skipped (fetch failed): {', '.join(skipped)}\n")
    return 0


def _cmd_diff(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Compare two saved chains side-by-side.

    Wraps :func:`care.runtime.fetch_agent_diff` so the CLI
    shares the same data layer the TUI's DiffModal consumes.

    Text output:
        diff: <left> ↔ <right>  (<format_summary()>)
        metadata: <field changes summary>
        + step N: <added title>
        - step N: <removed title>
        ~ step N: <modified title>
            <field>: <left> → <right>
        · step N: <unchanged title>

    ``--json`` emits the structured payload (per-step diff
    rows, metadata diff fields + tag deltas, summary counts).
    """
    from care.runtime import AgentDiffError, fetch_agent_diff

    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care diff: {exc}\n")
        return 2

    try:
        diff = asyncio.run(
            fetch_agent_diff(
                memory,
                args.left,
                args.right,
                left_label=args.left_label,
                right_label=args.right_label,
                channel=args.channel,
            ),
        )
    except AgentDiffError as exc:
        stderr.write(f"care diff: {exc}\n")
        return 2
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care diff: lookup failed: {exc}\n")
        return 2

    if args.json:
        payload = _diff_to_dict(diff)
        json.dump(payload, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    left = args.left_label or args.left
    right = args.right_label or args.right
    stdout.write(
        f"diff: {left} ↔ {right}  ({diff.format_summary()})\n",
    )
    meta_text = _format_metadata_diff(diff.metadata)
    if meta_text:
        stdout.write(f"metadata: {meta_text}\n")
    if not diff.steps:
        stdout.write("steps: (no step rows)\n")
        return 0
    for step in diff.steps:
        stdout.write(_format_step_diff_row(step) + "\n")
        for field_diff in step.fields:
            stdout.write(_format_field_diff_row(field_diff) + "\n")
    return 0


_DIFF_BADGES: dict[str, str] = {
    "added": "+",
    "removed": "-",
    "modified": "~",
    "unchanged": "·",
}


def _format_step_diff_row(step) -> str:
    """One-line summary of a single :class:`StepDiff`."""
    badge = _DIFF_BADGES.get(step.kind, "?")
    title = step.label
    return f"{badge} step {step.number}: {title}"


def _format_field_diff_row(field_diff) -> str:
    """One-line summary of a single :class:`FieldDiff`."""
    return (
        f"    {field_diff.field}: "
        f"{_truncate(field_diff.left_value)} → "
        f"{_truncate(field_diff.right_value)}"
    )


def _format_metadata_diff(meta) -> str:
    if not meta.has_changes:
        return ""
    parts: list[str] = []
    for f in meta.fields:
        parts.append(
            f"{f.field}: {_truncate(f.left_value)} → "
            f"{_truncate(f.right_value)}"
        )
    if meta.added_tags:
        parts.append(f"+tags: {', '.join(meta.added_tags)}")
    if meta.removed_tags:
        parts.append(f"-tags: {', '.join(meta.removed_tags)}")
    return "  ·  ".join(parts)


def _diff_to_dict(diff) -> dict:
    """JSON-friendly projection of an :class:`AgentDiff`."""
    meta = diff.metadata
    return {
        "left_entity_id": diff.left_entity_id,
        "right_entity_id": diff.right_entity_id,
        "left_label": diff.left_label,
        "right_label": diff.right_label,
        "summary": diff.format_summary(),
        "counts": {
            "added": diff.added_steps,
            "removed": diff.removed_steps,
            "modified": diff.modified_steps,
            "unchanged": diff.unchanged_steps,
            "total": len(diff.steps),
        },
        "metadata": {
            "fields": [_field_diff_dict(f) for f in meta.fields],
            "added_tags": list(meta.added_tags),
            "removed_tags": list(meta.removed_tags),
        },
        "steps": [
            {
                "number": s.number,
                "kind": s.kind,
                "label": s.label,
                "fields": [_field_diff_dict(f) for f in s.fields],
            }
            for s in diff.steps
        ],
    }


def _field_diff_dict(field_diff) -> dict:
    return {
        "field": field_diff.field,
        "left": field_diff.left_value,
        "right": field_diff.right_value,
        "left_present": field_diff.left_present,
        "right_present": field_diff.right_present,
    }


def _truncate(value, *, n: int = 40) -> str:
    """Render any value as a short single-line string."""
    s = "" if value is None else str(value)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def _cmd_search(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """BM25 / vector / hybrid search across saved entities.

    Wraps :meth:`CareMemory.search` so the CLI shares the same
    SDK surface the TUI's LibraryScreen search bar consumes.

    Output modes:

    * ``--json`` — emit the raw hit list as pretty-printed JSON.
    * Default — one line per hit: ``<score>  <entity_id>  <name>``.
      Header reports the query + hit count.
    """
    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care search: {exc}\n")
        return 2

    try:
        hits = memory.search(
            args.query,
            entity_type=args.entity_type,
            search_type=args.search_type,
            top_k=args.top_k,
        )
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care search: lookup failed: {exc}\n")
        return 2

    rows = list(hits or [])

    if args.json:
        json.dump(
            {
                "query": args.query,
                "entity_type": args.entity_type,
                "search_type": args.search_type,
                "hits": rows,
            },
            stdout,
            indent=2,
            default=str,
        )
        stdout.write("\n")
        return 0

    if not rows:
        stdout.write(
            f"search: no hits for {args.query!r} "
            f"({args.entity_type}, {args.search_type})\n",
        )
        return 0

    stdout.write(
        f"search: {len(rows)} hit(s) for {args.query!r} "
        f"({args.entity_type}, {args.search_type})\n",
    )
    for hit in rows:
        stdout.write(_format_search_hit(hit) + "\n")
    return 0


def _format_search_hit(hit: dict) -> str:
    """One-line summary of a single search hit."""
    score = hit.get("score")
    score_str = f"{float(score):.3f}" if isinstance(score, (int, float)) else "—"
    entity_id = str(hit.get("entity_id") or "?")
    name = (
        hit.get("display_name")
        or hit.get("name")
        or (hit.get("meta") or {}).get("name")
        or "(unnamed)"
    )
    return f"  {score_str}  {entity_id[:18]}  {name}"


def _cmd_lineage(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Walk a chain's ancestry DAG.

    Wraps :meth:`CareMemory.get_chain_lineage` so the CLI shares
    the same SDK surface the TUI's LineageModal consumes. Two
    output modes:

    * ``--json`` — emit the full :class:`LineageResponse` payload
      (entity_id, root_version_id, versions[], max_depth_reached)
      as pretty-printed JSON. Best for piping into `jq`.
    * Default — text listing: one summary header
      (`lineage: <chain_id> (root <version>, N version(s),
      max-depth cap reached)`) followed by one line per
      version sorted by BFS depth.
    """
    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care lineage: {exc}\n")
        return 2

    try:
        response = memory.get_chain_lineage(
            args.chain_id,
            channel=args.channel,
            version_id=args.version_id,
            max_depth=args.max_depth,
        )
    except Exception as exc:  # noqa: BLE001
        stderr.write(
            f"care lineage: failed to fetch lineage for "
            f"{args.chain_id!r}: {exc}\n",
        )
        return 2

    versions = list(getattr(response, "versions", None) or [])
    root_version = getattr(response, "root_version_id", "") or ""
    max_depth_reached = bool(getattr(response, "max_depth_reached", False))

    if args.json:
        payload = {
            "entity_id": getattr(response, "entity_id", args.chain_id),
            "root_version_id": root_version,
            "max_depth_reached": max_depth_reached,
            "versions": [_lineage_version_dict(v) for v in versions],
        }
        json.dump(payload, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    if not versions:
        stdout.write(
            f"lineage: no versions returned for {args.chain_id}\n",
        )
        return 0

    header_bits = [f"lineage: {args.chain_id}"]
    if root_version:
        header_bits.append(f"root {root_version}")
    header_bits.append(
        f"{len(versions)} version(s)",
    )
    if max_depth_reached:
        header_bits.append("max-depth cap reached")
    stdout.write(" · ".join(header_bits) + "\n")
    for v in sorted(
        versions,
        key=lambda x: (
            getattr(x, "depth", 0),
            -getattr(x, "version_number", 0),
        ),
    ):
        stdout.write(_format_lineage_version(v) + "\n")
    return 0


def _lineage_version_dict(version) -> dict:
    """Project a :class:`LineageVersion` (or dict) into a JSON-
    friendly dict for the ``--json`` payload."""
    if isinstance(version, dict):
        # Already projected — copy so the caller can mutate safely.
        return dict(version)
    return {
        "version_id": getattr(version, "version_id", ""),
        "version_number": getattr(version, "version_number", 0),
        "parents": list(getattr(version, "parents", None) or []),
        "evolution_meta": getattr(version, "evolution_meta", None),
        "change_summary": getattr(version, "change_summary", None),
        "author": getattr(version, "author", None),
        "created_at": getattr(version, "created_at", None),
        "depth": getattr(version, "depth", 0),
    }


def _format_lineage_version(version) -> str:
    """One-line summary of a single :class:`LineageVersion`."""
    if isinstance(version, dict):
        get = version.get
    else:
        get = lambda k, default=None: getattr(version, k, default)  # noqa: E731
    vid = get("version_id", "?")
    number = get("version_number", 0)
    parents = get("parents") or []
    depth = get("depth", 0)
    created = get("created_at")
    bits = [f"v{number} ({vid})"]
    bits.append(f"depth {depth}")
    if created is not None:
        try:
            stamp = created.strftime("%Y-%m-%d %H:%M")
        except AttributeError:
            stamp = str(created)
        bits.append(stamp)
    if parents:
        bits.append("parents=" + ", ".join(str(p) for p in parents))
    else:
        bits.append("root")
    return "  " + " · ".join(bits)


def _cmd_replay(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Step through a saved ReasoningResult / RunRecord JSON.

    Reads the source file (or stdin via ``-``), passes the text
    to :func:`care.load_replay`, and renders the result. Three
    modes:

    * ``--step N`` — render exactly one step's detail block
      (seeking via :meth:`ReplaySession.seek` so out-of-bounds
      clamps cleanly).
    * ``--json`` — emit a structured payload with chain
      metadata + every documented :class:`ReplayStep` field.
    * Default — walk every step, printing the seek-then-format
      output for each so the user gets a complete textual log.
    """
    from care.replay import ReplayError, load_replay

    try:
        raw = _read_source_text(args.source, stdin=sys.stdin)
    except OSError as exc:
        stderr.write(f"care replay: read failed: {exc}\n")
        return 2

    try:
        session = load_replay(raw)
    except ReplayError as exc:
        stderr.write(f"care replay: {exc}\n")
        return 2

    if args.json:
        payload = {
            "chain_id": session.chain_id,
            "chain_title": session.chain_title,
            "step_count": session.step_count,
            "total_execution_time_s": session.total_execution_time_s,
            "final_answer": session.final_answer,
            "token_usage": dict(session.token_usage),
            "steps": [
                {
                    "step_number": s.step_number,
                    "step_title": s.step_title,
                    "step_type": s.step_type,
                    "success": s.success,
                    "skipped": s.skipped,
                    "execution_time_s": s.execution_time_s,
                    "error_message": s.error_message,
                    "result_preview": s.result_preview,
                    "result_truncated": s.result_truncated,
                    "model": s.model,
                }
                for s in session.steps
            ],
        }
        json.dump(payload, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    if session.is_empty:
        stdout.write("replay: no steps in source\n")
        return 0

    if args.step is not None:
        session.seek(args.step)
        stdout.write(session.format_text() + "\n")
        return 0

    # Default: walk every step, printing format_text for each.
    for index in range(session.step_count):
        session.seek(index)
        if index > 0:
            stdout.write("\n")
        stdout.write(session.format_text() + "\n")
    return 0


def _read_source_text(path: Path, *, stdin: TextIO) -> str:
    """Read the raw text from a path or stdin. ``Path('-')``
    reads from stdin."""
    if str(path) == "-":
        return stdin.read()
    return path.read_text(encoding="utf-8")


# TUI slash-command twin for each CLI subcommand (curated — the CLI side
# is auto-discovered from the parser so it can't drift; only these
# annotations need upkeep when a twin is added).
_CLI_TUI_TWIN: dict[str, str] = {
    "init": "(first-run wizard)",
    "doctor": "/status",
    "migrate-secrets": "(setup)",
    "catalog": "/catalog",
    "validate": "(chain preflight)",
    "import": "Library import bundle",
    "export": "Library export bundle (X)",
    "generate": "free-form chat prompt",
    "run": "/run",
    "revise": "/revise",
    "replay": "/replay",
    "versions": "/versions",
    "rollback": "/rollback",
    "promote": "/promote",
    "forget": "/forget",
    "memory": "/library",
    "search": "/search",
    "diff": "/diff",
    "lineage": "/lineage",
    "favourite": "Library F (star)",
    "marketplace": "/marketplace",
    "evolve": "/evolve",
    "deploy": "/deploy",
    "deployments": "/deployments",
    "metrics": "/metrics",
    "dataset": "/dataset",
    "remember": "/remember",
    "notes": "/memory (LTM)",
    "help": "/help",
}

# Notable TUI verbs that don't yet have a headless twin (kept short — it's
# a discoverability pointer, not an exhaustive registry dump).
_TUI_ONLY_VERBS: tuple[str, ...] = (
    "/upload (attach a file to a turn — interactive only)",
    "/tour, /settings, /theme (interactive UI)",
)


def _render_command_parity() -> str:
    """Render the CLI ↔ TUI parity table for ``care help --commands``.

    CLI subcommand names are read live from the argparse parser (so the
    list can't drift), each annotated with its TUI twin; then the handful
    of TUI-only verbs are listed so users know what's still TUI-only."""
    parser = _build_parser()
    sub_action = next(
        (
            a
            for a in parser._actions
            if isinstance(a, argparse._SubParsersAction)
        ),
        None,
    )
    names = sorted(sub_action.choices.keys()) if sub_action is not None else []
    lines = ["CARE CLI ↔ TUI command parity", ""]
    lines.append("Headless subcommands (each with its TUI twin):")
    for name in names:
        twin = _CLI_TUI_TWIN.get(name, "(no direct TUI twin)")
        lines.append(f"  care {name:<16}  ↔  {twin}")
    lines.append("")
    lines.append("TUI-only (no headless twin yet):")
    for verb in _TUI_ONLY_VERBS:
        lines.append(f"  {verb}")
    return "\n".join(lines)


def _cmd_help(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Render the help registry via the shipped formatters.

    Wraps :func:`care.build_registry` so plugin-registered
    tutorial steps + bindings show up automatically. ``--markdown``
    switches to :meth:`HelpRegistry.format_markdown` for
    README-style output; the default is the plain-text format
    :class:`HelpScreen` would render. ``--category`` / ``--screen``
    narrow the bindings listing without dropping the tutorial.
    """
    if getattr(args, "commands", False):
        stdout.write(_render_command_parity())
        stdout.write("\n")
        return 0

    from care.help import HelpRegistry, build_registry

    registry = build_registry()

    if args.category or args.screen:
        # Re-project a filtered registry so the formatter renders
        # the requested subset only. Tutorial steps stay intact —
        # they aren't gated by category/screen.
        filtered = HelpRegistry()
        for step in registry.steps():
            filtered.add_step(step)
        for binding in registry.bindings():
            if args.category and binding.category != args.category:
                continue
            if args.screen and binding.screen != args.screen:
                continue
            filtered.add_binding(binding)
        registry = filtered

    if args.markdown:
        stdout.write(registry.format_markdown())
    else:
        stdout.write(registry.format_text() + "\n")
    return 0


def _cmd_memory_show(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Print a single entity's metadata + content.

    Wraps :meth:`CareMemory.get_entity` so the same listing
    surface that `care memory ls` walks can be drilled into for
    one row. Three rendering modes:

    * ``--json`` — emit the full SDK payload (``entity_id`` /
      ``version_id`` / ``channel`` / ``meta`` / ``content``) as
      pretty-printed JSON. Best for piping into `jq`.
    * ``--content-only`` — emit just the ``content`` body
      (still pretty-printed JSON). Handy for follow-on
      `care validate <file>` / `care export` pipelines.
    * Default — a human-friendly text block: id / channel /
      version / display name / tags header, followed by the
      content body pretty-printed.
    """
    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care memory show: {exc}\n")
        return 2

    try:
        payload = memory.get_entity(
            args.entity_id,
            entity_type=args.entity_type,
            channel=args.channel,
        )
    except Exception as exc:  # noqa: BLE001
        stderr.write(
            f"care memory show: failed to fetch "
            f"{args.entity_type} {args.entity_id!r}: {exc}\n",
        )
        return 2

    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, dict):
        content = {}

    if args.content_only:
        json.dump(content, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    if args.json:
        json.dump(payload, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    meta = payload.get("meta") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        meta = {}
    name = meta.get("name") or payload.get("display_name") or "(unnamed)"
    tags = meta.get("tags")
    tags_text = ", ".join(str(t) for t in tags) if tags else "—"
    stdout.write(f"entity: {args.entity_id} ({args.entity_type})\n")
    stdout.write(f"channel: {payload.get('channel') or args.channel}\n")
    stdout.write(f"version: {payload.get('version_id') or '?'}\n")
    stdout.write(f"name: {name}\n")
    stdout.write(f"tags: {tags_text}\n")
    stdout.write("content:\n")
    stdout.write(
        json.dumps(content, indent=2, default=str, ensure_ascii=False),
    )
    stdout.write("\n")
    return 0


def _cmd_memory_history(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """List recorded runs for a saved chain.

    Wraps :func:`care.runtime.fetch_run_history` (the same data
    layer the TUI's InspectionScreen Run History tab uses), then
    renders the entries via :meth:`RunHistoryEntry.format_one_line`
    or a JSON payload carrying every documented field. A summary
    header (built via :func:`summarize_run_history`) shows
    success / failure totals + duration + tokens.
    """
    from care.runtime import (
        RunHistoryError,
        fetch_run_history,
        summarize_run_history,
    )

    try:
        memory = _build_memory()
    except CliMemoryError as exc:
        stderr.write(f"care memory history: {exc}\n")
        return 2

    try:
        entries = asyncio.run(
            fetch_run_history(
                memory,
                args.chain_id,
                limit=args.limit,
                namespace=args.namespace,
                channel=args.channel,
            ),
        )
    except RunHistoryError as exc:
        stderr.write(f"care memory history: {exc}\n")
        return 2
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care memory history: lookup failed: {exc}\n")
        return 2

    summary = summarize_run_history(entries)

    if args.json:
        payload = {
            "chain_id": args.chain_id,
            "channel": args.channel,
            "summary": {
                "total_runs": summary.total_runs,
                "success_count": summary.success_count,
                "failure_count": summary.failure_count,
                "total_tokens": summary.total_tokens,
                "total_duration_seconds": summary.total_duration_seconds,
                "success_rate": summary.success_rate,
                "avg_duration_seconds": summary.avg_duration_seconds,
                "last_success_at": (
                    summary.last_success_at.isoformat()
                    if summary.last_success_at else None
                ),
                "last_failure_at": (
                    summary.last_failure_at.isoformat()
                    if summary.last_failure_at else None
                ),
            },
            "entries": [
                {
                    "card_id": e.card_id,
                    "agent_entity_id": e.agent_entity_id,
                    "run_id": e.run_id,
                    "finished_at": (
                        e.finished_at.isoformat() if e.finished_at else None
                    ),
                    "status": e.status,
                    "duration_seconds": e.duration_seconds,
                    "step_count": e.step_count,
                    "total_tokens": e.total_tokens,
                    "error_message": e.error_message,
                    "task_description": e.task_description,
                    "description": e.description,
                    "tags": list(e.tags),
                    "metrics": e.metrics,
                }
                for e in entries
            ],
        }
        json.dump(payload, stdout, indent=2, default=str)
        stdout.write("\n")
        return 0

    if summary.total_runs == 0:
        stdout.write(f"history: no runs recorded for {args.chain_id}\n")
        return 0

    rate = (
        f"{summary.success_rate * 100:.0f}%"
        if summary.success_rate is not None else "—"
    )
    avg = (
        f"{summary.avg_duration_seconds:.1f}s"
        if summary.avg_duration_seconds is not None else "—"
    )
    stdout.write(
        f"history: {summary.total_runs} run(s) "
        f"({summary.success_count} ok, {summary.failure_count} failed, "
        f"success rate {rate}, avg {avg})\n",
    )
    for entry in entries:
        stdout.write(entry.format_one_line() + "\n")
    return 0


def _cmd_evolve(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Submit an evolution run for a saved chain.

    Wraps :func:`care.build_evolution_request` to project the
    user-facing flags into the canonical request body, calls
    :meth:`CarePlatform.start_evolution` to submit, and (when
    ``--wait`` is set) drains
    :meth:`CarePlatform.stream_events` through an
    :class:`EvolutionProgressTracker` to render per-generation
    progress. ``--accept`` (requires ``--wait``) promotes the
    best-overall individual via ``accept_individual`` once the
    stream finishes.
    """
    from care.evolution_session import (
        EvolutionConfig,
        EvolutionPlan,
        EvolutionPlanError,
        EvolutionProgressTracker,
        build_evolution_request,
    )

    if args.accept and not args.wait:
        stderr.write(
            "care evolve: --accept requires --wait (cannot promote a "
            "winner without observing the run to completion).\n"
        )
        return 2

    try:
        platform = _build_platform()
    except CliPlatformError as exc:
        stderr.write(f"care evolve: {exc}\n")
        return 2

    config = EvolutionConfig(
        evolution_mode=args.mode,
        max_iterations=args.iterations,
        population_size=args.population,
        validation_criteria=args.validation_criteria or "",
        test_data_path=args.test_data_path,
        validation_threshold=args.threshold,
        validation_type=args.validation_type,
        continuous_metric=args.metric,
        binary_method=args.binary_method,
        target_column=args.target_column,
        objectives=tuple(args.objective or ()),
    )
    plan = EvolutionPlan(
        config=config,
        base_chain_entity_id=args.chain_id,
    )
    try:
        body = build_evolution_request(plan)
    except EvolutionPlanError as exc:
        stderr.write(f"care evolve: invalid plan: {exc}\n")
        return 2

    spec = dict(body)
    spec["base_chain_id"] = spec.pop("seed_chain_id", args.chain_id)

    # Fetch the chain content from Memory so we can submit via the
    # chain-experiment route — that's the path the live Platform
    # scheduler actually drains. Without the chain inlined, the
    # legacy /api/v1/evolutions endpoint accepts the record but no
    # dispatcher picks it up, so the run sits forever in `queued`.
    base_chain_content: dict | None = None
    try:
        memory = _build_memory()
        base_chain_content = memory.get_chain(args.chain_id)
    except CliMemoryError as exc:
        stderr.write(
            f"care evolve: warning — Memory unavailable, falling back to "
            f"legacy evolutions endpoint (no dispatcher will pick this up): "
            f"{exc}\n",
        )
    except Exception as exc:  # noqa: BLE001
        stderr.write(
            f"care evolve: warning — couldn't fetch chain content "
            f"({exc}); falling back to legacy evolutions endpoint.\n",
        )
    if base_chain_content is not None:
        spec["base_chain_content"] = base_chain_content
    try:
        ref = platform.start_evolution(**spec)
    except Exception as exc:  # noqa: BLE001
        stderr.write(f"care evolve: submit failed: {exc}\n")
        return 2

    evolution_id = ref.evolution_id
    status = ref.status or "queued"
    tracker = EvolutionProgressTracker()
    accepted_id: str | None = None

    if args.wait:
        try:
            for event in platform.stream_events(evolution_id):
                if not isinstance(event, dict):
                    continue
                kind = str(event.get("event") or event.get("type") or "")
                payload = event.get("data") or event
                if not isinstance(payload, dict):
                    payload = {}
                tracker.record_event({"event_type": kind, **payload})
                if kind in {"completed", "failed", "cancelled"}:
                    status = kind
                if not args.json:
                    line = _format_evolution_event(kind, payload, tracker)
                    if line:
                        stdout.write(line + "\n")
                if kind in {"completed", "failed", "cancelled"}:
                    break
        except Exception as exc:  # noqa: BLE001
            stderr.write(f"care evolve: stream failed: {exc}\n")
            return 2

        if args.accept and status == "completed":
            is_exp = evolution_id.startswith("exp_")
            best = tracker.best_overall
            winner = (
                best.best_individual_id
                if best is not None else None
            )
            # Chain experiments promote the persisted ``best_chain_config``
            # (the individual id is ignored) and need a CareMemory facade to
            # write the winner back — so accept even when the stream carried
            # no per-individual winner id. Legacy ``evo_*`` keeps the
            # server-side 2-arg path (no memory needed).
            if winner or is_exp:
                try:
                    if is_exp:
                        memory = _build_memory()
                        response = platform.accept_individual(
                            evolution_id, winner or "", memory=memory,
                        )
                        accepted_id = winner or (
                            response.get("chain_id")
                            if isinstance(response, dict) else None
                        ) or evolution_id
                    else:
                        platform.accept_individual(evolution_id, winner)
                        accepted_id = winner
                except CliMemoryError as exc:
                    stderr.write(
                        f"care evolve: accept needs a memory connection: {exc}\n",
                    )
                    return 2
                except Exception as exc:  # noqa: BLE001
                    stderr.write(
                        f"care evolve: accept failed: {exc}\n",
                    )
                    return 2

    if args.json:
        best = tracker.best_overall
        summary = {
            "evolution_id": evolution_id,
            "status": status,
            "generation": (best.generation if best else 0),
            "best_fitness": (best.best_fitness if best else None),
            "best_individual_id": (
                best.best_individual_id if best else None
            ),
            "accepted_individual_id": accepted_id,
        }
        json.dump(summary, stdout, indent=2, default=str)
        stdout.write("\n")
    else:
        stdout.write(
            f"evolution {evolution_id}: status={status}\n",
        )
        if accepted_id:
            stdout.write(f"accepted: {accepted_id}\n")
        best = tracker.best_overall
        if best is not None:
            stdout.write(
                f"best gen={best.generation} "
                f"fitness={best.best_fitness:.3f} "
                f"id={best.best_individual_id}\n"
            )
    return 0


def _format_evolution_event(
    kind: str, payload: dict, tracker,
) -> str:
    """Render a single SSE event as one terminal line.

    Returns an empty string for events the user doesn't care
    about (e.g. heartbeats) so the caller's `if line:` filter
    keeps the output tight.
    """
    gen = payload.get("generation")
    if kind == "generation_started" and isinstance(gen, int):
        return f"[gen {gen}] started"
    if kind == "individual_evaluated":
        ind = payload.get("individual_id") or "?"
        fit = payload.get("fitness")
        fit_str = f"{fit:.3f}" if isinstance(fit, (int, float)) else "—"
        return f"[gen {gen}] evaluated {ind} fitness={fit_str}"
    if kind == "best_updated":
        ind = (
            payload.get("best_individual_id")
            or payload.get("individual_id") or "?"
        )
        fit = payload.get("fitness") or payload.get("best_fitness")
        fit_str = f"{fit:.3f}" if isinstance(fit, (int, float)) else "—"
        return f"[gen {gen}] best now {ind} fitness={fit_str}"
    if kind == "completed":
        return "[done] evolution completed"
    if kind == "failed":
        err = payload.get("error") or ""
        return f"[fail] {err}".rstrip()
    if kind == "cancelled":
        return "[cancel] evolution cancelled"
    return ""


class CliPlatformError(RuntimeError):
    """Raised when the CLI can't materialise a Platform facade —
    typically a missing config file or unreachable server. Carries
    a single-line message ready for stderr."""


def _build_platform():
    """Construct a :class:`CarePlatform` from the on-disk
    :class:`CareConfig`. Mirrors :func:`_build_memory` — wraps
    import + config-load + construction errors in a single
    :class:`CliPlatformError`. Tests inject a stub via
    ``_BUILD_PLATFORM_OVERRIDE``.
    """
    if _BUILD_PLATFORM_OVERRIDE is not None:
        return _BUILD_PLATFORM_OVERRIDE()
    try:
        from care.platform import CarePlatform
    except Exception as exc:  # noqa: BLE001
        raise CliPlatformError(
            f"failed to import CarePlatform: {exc}",
        ) from exc
    try:
        config = CareConfig.load()
    except Exception as exc:  # noqa: BLE001
        raise CliPlatformError(
            f"failed to load CareConfig: {exc}",
        ) from exc
    try:
        return CarePlatform.from_config(config)
    except Exception as exc:  # noqa: BLE001
        raise CliPlatformError(
            f"failed to construct CarePlatform: {exc}",
        ) from exc


# Test-injectable hook. Production callers must not set this.
_BUILD_PLATFORM_OVERRIDE = None


class CliMemoryError(RuntimeError):
    """Raised when the CLI can't materialise a Memory facade —
    typically a missing config file or unreachable server. Carries
    a single-line message ready for stderr."""


# ---------------------------------------------------------------------------
# `care init` — quick-start (Phase 6 P2)
# ---------------------------------------------------------------------------

_INIT_DEFAULTS: dict[str, str] = {
    "base_url": "https://openrouter.ai/api/v1",
    "model": "anthropic/claude-3.5-sonnet",
    "mode": "interactive",
}


def _render_init_env(
    *,
    base_url: str,
    api_key: str,
    model: str,
    mode: str,
) -> str:
    """Render the minimal ``.env`` body :func:`_cmd_init` writes.

    Only the keys ``care init`` collects are emitted — the full
    template lives in ``.env.example`` for users who want to
    customise sandbox / telemetry / etc. settings.
    """
    return (
        "# Generated by `care init`. Edit freely.\n"
        "# Full reference: .env.example\n"
        "\n"
        "# MAGE generator — OpenAI-compatible HTTP endpoint.\n"
        f"CARE_MAGE__BASE_URL={base_url}\n"
        f"CARE_MAGE__API_KEY={api_key}\n"
        f"CARE_MAGE__MODEL={model}\n"
        "\n"
        "# ChatScreen boot mode (ad_hoc | production).\n"
        f"CARE_CHAT__DEFAULT_MODE={mode}\n"
    )


def _prompt_with_default(
    *,
    stdin: TextIO,
    stdout: TextIO,
    label: str,
    default: str,
    secret: bool = False,
) -> str:
    """Render ``"<label> [<default>]: "`` to stdout, read one
    line from stdin, return either the line or the default.

    ``secret=True`` redacts the default in the prompt so an
    existing key isn't echoed back to the screen. Doesn't try
    to disable terminal echo — that's a non-portable concern
    we defer to a future P3."""
    hint = "***" if secret and default else default
    suffix = f" [{hint}]" if hint else ""
    stdout.write(f"{label}{suffix}: ")
    stdout.flush()
    try:
        raw = stdin.readline()
    except Exception:  # noqa: BLE001
        return default
    value = (raw or "").rstrip("\r\n")
    return value if value else default


def _resolve_init_value(
    *,
    flag_value: str | None,
    field: str,
    label: str,
    non_interactive: bool,
    stdin: TextIO,
    stdout: TextIO,
    default_override: str | None = None,
    secret: bool = False,
) -> str:
    """Pick the value for one init field. Flag wins; otherwise
    prompt (or fall back to default in non-interactive mode)."""
    if flag_value is not None:
        return flag_value
    default = (
        default_override
        if default_override is not None
        else _INIT_DEFAULTS.get(field, "")
    )
    if non_interactive:
        return default
    return _prompt_with_default(
        stdin=stdin,
        stdout=stdout,
        label=label,
        default=default,
        secret=secret,
    )


def _cmd_init(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    stdin: TextIO | None = None,
) -> int:
    """Write a minimal ``.env`` so a fresh checkout can boot.

    Idempotency: refuses to overwrite an existing file unless
    ``--force`` is set. Honours the same precedence as the rest
    of the CLI — explicit flags beat interactive prompts beat
    defaults.

    Designed to be CI-driveable: ``--non-interactive`` skips
    every prompt so headless runs (and tests) don't block on
    stdin. Returns ``0`` on success, ``1`` on refused
    overwrite, ``2`` on write failure.
    """
    stdin_io: TextIO = stdin if stdin is not None else sys.stdin

    target = Path(args.env_path).expanduser()
    if target.exists() and not args.force:
        stderr.write(
            f"care init: {target} already exists. "
            "Re-run with --force to overwrite.\n",
        )
        return 1

    base_url = _resolve_init_value(
        flag_value=args.base_url,
        field="base_url",
        label="MAGE base URL",
        non_interactive=args.non_interactive,
        stdin=stdin_io,
        stdout=stdout,
    )
    api_key = _resolve_init_value(
        flag_value=args.api_key,
        field="api_key",
        label="MAGE API key",
        non_interactive=args.non_interactive,
        stdin=stdin_io,
        stdout=stdout,
        default_override="",
        secret=True,
    )
    model = _resolve_init_value(
        flag_value=args.model,
        field="model",
        label="Model id",
        non_interactive=args.non_interactive,
        stdin=stdin_io,
        stdout=stdout,
    )
    mode = _resolve_init_value(
        flag_value=args.mode,
        field="mode",
        label="Default chat mode (interactive / production)",
        non_interactive=args.non_interactive,
        stdin=stdin_io,
        stdout=stdout,
    )
    # Normalise the legacy ``ad_hoc`` spellings onto the canonical id so
    # only ``interactive`` / ``production`` are ever written to the .env.
    mode = {"ad_hoc": "interactive", "ad-hoc": "interactive",
            "adhoc": "interactive"}.get(str(mode).strip().lower(), mode)
    if mode not in ("interactive", "production"):
        stderr.write(
            f"care init: invalid --mode {mode!r}; "
            "expected interactive or production.\n",
        )
        return 1

    content = _render_init_env(
        base_url=base_url, api_key=api_key, model=model, mode=mode,
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        stderr.write(f"care init: failed to write {target}: {exc}\n")
        return 2

    stdout.write(f"✓ Wrote {target}\n")
    if not api_key:
        stdout.write(
            "⚠ MAGE API key left blank — generation will fail "
            "until CARE_MAGE__API_KEY is set.\n",
        )
    stdout.write(
        "Next: launch the TUI with `care` (no arguments). "
        "Type a task to drive a generation.\n",
    )
    return 0


def _build_memory():
    """Construct a :class:`CareMemory` from the on-disk
    :class:`CareConfig`. Wraps the import + config-load errors in
    a single :class:`CliMemoryError` so the calling handler
    doesn't need to discriminate.

    Returns a duck-typed memory facade — call sites use
    ``list_entities`` etc. through the protocol surface, never
    through `isinstance`. Tests inject a stub via the
    ``_BUILD_MEMORY_OVERRIDE`` hook.
    """
    if _BUILD_MEMORY_OVERRIDE is not None:
        return _BUILD_MEMORY_OVERRIDE()
    try:
        from care.memory import CareMemory
    except Exception as exc:  # noqa: BLE001
        raise CliMemoryError(f"failed to import CareMemory: {exc}") from exc
    try:
        config = CareConfig.load()
    except Exception as exc:  # noqa: BLE001
        raise CliMemoryError(f"failed to load CareConfig: {exc}") from exc
    try:
        return CareMemory.from_config(config)
    except Exception as exc:  # noqa: BLE001
        raise CliMemoryError(
            f"failed to construct CareMemory: {exc}",
        ) from exc


# Test-injectable hook. Production callers must not set this.
_BUILD_MEMORY_OVERRIDE = None


# ---------------------------------------------------------------------------
# TUI fallback
# ---------------------------------------------------------------------------


def _run_tui() -> int:
    """Defer to :mod:`care.app`. Lazy import so the CLI doesn't
    pay the Textual import cost on every ``care --help``.

    Catches ``KeyboardInterrupt`` so a Ctrl+C pressed before
    Textual has fully taken over the TTY (or during the final
    shutdown phase after ``app.exit()``) exits the process
    with a clean status code + short message instead of dumping
    a multi-frame traceback to the user's terminal. Inside the
    Textual event loop, Ctrl+C is already routed to
    ``CareApp.action_global_quit`` via a priority binding.
    """
    try:
        from care.app import run as _run
    except Exception as exc:  # noqa: BLE001
        # Textual import / setup issues shouldn't crash the CLI
        # outright — surface them and let the user fix their env.
        sys.stderr.write(f"care: TUI failed to start: {exc}\n")
        return 2
    try:
        _run()
    except KeyboardInterrupt:
        # Ctrl+C during early bootstrap (or during Textual's
        # finalisation pass after we already triggered exit())
        # would otherwise leave a `KeyboardInterrupt` traceback
        # on the user's terminal. Swallow it and exit cleanly.
        sys.stderr.write("care: interrupted.\n")
        return 130
    return 0


def _cmd_doctor(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """§1 P1 — Diagnostic report: env / config path / user-data
    dirs / extras / probes.

    Returns 0 when the overall picture is healthy
    (memory + mage probes both `ok` OR explicitly skipped),
    1 when any required probe failed. The network section
    can be skipped with `--no-probes` for offline / CI use.
    """
    import asyncio

    from care.config import CareConfig, DEFAULT_CONFIG_PATH
    from care.runtime.doctor import compose_report

    config_path = (
        Path(args.config).expanduser() if args.config
        else DEFAULT_CONFIG_PATH
    )
    try:
        config = CareConfig.load(path=config_path)
    except Exception as exc:  # noqa: BLE001
        stderr.write(
            f"care doctor: could not load config from "
            f"{config_path}: {exc}\n"
        )
        return 2

    probes_text = ""
    failed_required = False
    if not args.no_probes:
        from care.first_run import run_all_probes

        try:
            # `care doctor` is an explicit health check — run the deep
            # MAGE probe (authenticated /models round-trip) so an expired
            # token surfaces as failed instead of a false green.
            report = asyncio.run(run_all_probes(config, deep=True))
        except Exception as exc:  # noqa: BLE001
            stderr.write(
                f"care doctor: probes failed to run: {exc}\n"
            )
            probes_text = f"(probes errored: {exc})"
        else:
            probes_text = report.format_text()
            # Memory + MAGE are required; Platform is optional
            # (the `skipped` default is fine).
            failed_required = (
                report.memory.status == "failed"
                or report.mage.status == "failed"
            )

    full = compose_report(
        config_path=config_path,
        probes_text=probes_text,
    )
    stdout.write(full.format_text())
    stdout.write("\n")

    # Flag obviously-misconfigured fields (e.g. an API key pasted into a
    # *_base_url / *_model slot) that the probes alone won't catch.
    config_warnings = config.audit_fields()
    if config_warnings:
        stdout.write("\nConfig warnings:\n")
        for warning in config_warnings:
            stdout.write(f"  ⚠ {warning}\n")

    return 1 if failed_required else 0


def _cmd_migrate_secrets(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """§1 P1 — Migrate literal `*_api_key` values into the
    system keystore + rewrite `~/.config/care/config.toml`
    so the literals become `keystore://service/key` URLs.

    Idempotent: re-running on a fully-migrated config does
    nothing. ``--dry-run`` prints the report without touching
    the keystore or the file (uses a `MemoryKeystore` as a
    sink for the would-be writes).

    Returns 0 when at least one secret migrated (or nothing
    needed to migrate); 1 when a hard failure (e.g. keystore
    detection raised globally) blocked the migration.
    """
    from care.config import (
        CareConfig,
        DEFAULT_CONFIG_PATH,
        migrate_literal_secrets,
    )

    path = (
        Path(args.config).expanduser() if args.config
        else DEFAULT_CONFIG_PATH
    )
    if not path.exists():
        stderr.write(
            f"care: no config at {path}; nothing to migrate.\n"
        )
        return 0

    config = CareConfig.load(path=path)

    keystore = None
    if args.dry_run:
        from care.runtime.keystore import MemoryKeystore

        keystore = MemoryKeystore()

    report = migrate_literal_secrets(
        config,
        path=path if not args.dry_run else None,
        keystore=keystore,
    )

    stdout.write(report.format_text())
    stdout.write("\n")
    if args.dry_run and report.did_migrate:
        stdout.write(
            "(dry-run: keystore was a temporary "
            "MemoryKeystore; nothing was persisted)\n"
        )
    # Hard failure = every slot landed in skipped with a
    # keystore-detect-failed reason. Soft skips (already a
    # URL, empty value) still return 0.
    detected_fail = any(
        "detect failed" in reason
        for _, reason in report.skipped
    )
    if detected_fail and not report.migrated:
        return 1
    return 0


# Smoke entry-point reachable from `python -m care.cli`.
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "main",
]
