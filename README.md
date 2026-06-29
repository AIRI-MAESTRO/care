# MAESTRO CARE — Collaborative Agent Reasoning Ecosystem

[![PyPI](https://img.shields.io/pypi/v/maestro-care.svg?logo=pypi&logoColor=white)](https://pypi.org/project/maestro-care/)

A Textual TUI + CLI for generating, running, and evolving agent
chains. CARE is the user-facing consumer on top of a four-module stack —
the **agent chain generator** for chain generation, the **agent chain
format** for execution, **GigaEvo Memory** for persistence, and
**GigaEvo Platform** for evolution.

## Run

```bash
uv sync
uv run maestro init       # one-shot: write a minimal .env with generator creds
uv run maestro            # launch the TUI
uv run maestro --help     # list CLI subcommands
```

`maestro init` walks the minimum agent chain generator credentials (base URL, API key,
model) and writes a `./.env` so a fresh checkout can boot. Use
`--non-interactive` with explicit flags for CI / scripted setup:
`maestro init --non-interactive --api-key sk-... --base-url ... --model ...`.

The TUI opens directly into the **chat surface** — a Claude-Code-style
transcript with a prompt at the bottom and a mode toggle above it.
Type a task in natural language to generate + run an agent chain;
type a slash command (`/help`, `/mode`, `/library`, `/dataset`,
`/evolution`, `/tour`, …) for the non-chat affordances. First-time
users see a one-line offer to type `/tour` for a 5-step walkthrough.

### Chat modes

The toggle above the prompt picks one of two modes; `/mode` switches
the same setting via the keyboard. Default is **Ad-Hoc** —
configurable per-deployment with `CARE_CHAT__DEFAULT_MODE`.

| Mode           | What happens on every prompt                                                                                                                                                                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Ad-Hoc**     | The agent chain generator produces a chain, the agent chain format runs it on the spot, the answer prints inline. The agent may loop (ReAct) until it decides the task is done. **Nothing is saved.**                                                                                                                    |
| **Production** | The agent chain generator produces a _reproducible_ chain, CARE saves it to Memory under a stable `chain_id`, runs one baseline to seed a dataset entry, and (when Platform is wired) kicks off an evolution run. Use `/dataset list <chain_id>` / `/dataset run <chain_id>` / `/evolution watch <run_id>` to follow up. |

Production requires `CARE_MEMORY__BASE_URL` (and `CARE_MEMORY__API_KEY`
when the deployment enforces auth). Without Memory configured,
selecting Production auto-falls back to Ad-Hoc with a warning line —
the toggle's tooltip explains why.

First boot (no `~/.config/care/config.toml` on disk) lands on the
**Settings** screen so you can configure Memory / Platform base URLs +
the agent chain generator LLM credentials before doing anything else. Returning users
land on the chat surface; `/library` opens the saved-agents table
(last-run time, favourite status, fitness scores, row-action keys
below).

## Canonical user flow

Generate Agent A → save it → generate B and C → return to A from the
library → re-run from the same task + context files → optionally
evolve A and accept the best individual back into the stable channel.

## Screens

CARE ships **18+ screens** covering the full lifecycle:

| Screen                | Purpose                                                |
| --------------------- | ------------------------------------------------------ |
| `WelcomeScreen`       | Boot splash + recents sidebar                          |
| `LibraryScreen`       | Saved-agent table, row actions, search                 |
| `QueryScreen`         | + New agent — task description + context files         |
| `GenerationScreen`    | Live agent chain generator progress                    |
| `InspectionScreen`    | Saved-chain detail + run history                       |
| `EditAgentScreen`     | Inline edit + Save / Promote-to-stable                 |
| `ExecutionScreen`     | Live agent chain run + token streaming                 |
| `EvolutionScreen`     | GA + Pareto picker + accept-winner                     |
| `ReplayScreen`        | Step through a saved ReasoningResult                   |
| `CatalogScreen`       | Browse installed capabilities (skills/MCP/tools)       |
| `MarketplaceScreen`   | Search shared agent_skill listings on Memory           |
| `HelpScreen`          | Tutorial + key cheat-sheet                             |
| `SettingsScreen`      | Edit agent chain generator / Memory / Platform / theme |
| `TaskListDrawer`      | In-flight tasks panel                                  |
| `CommandPaletteModal` | Fuzzy palette over commands + saved entities           |
| `DiffModal`           | Side-by-side compare two chains                        |
| `LineageModal`        | Walk a chain's ancestry DAG                            |
| `ConflictModal`       | Resolve a name collision on save                       |

## Global keys

- `Ctrl+P` — Command palette (search commands + chains + skills)
- `Ctrl+B` — Task list drawer
- `Ctrl+K` — Capability catalog
- `Ctrl+S` — Save current artifact
- `Ctrl+R` — Re-run current chain
- `?` — Help (tutorial + every binding)
- `Esc` — Back / cancel
- `Ctrl+Q` — Quit

Each screen layers its own bindings — see `?` (Help) for the full set
filtered by the active screen.

A single-page reference table covering every screen + modal lives at
[`docs/screens/README.md`](docs/screens/README.md). It maps each
surface to its slash command + status (M0/M1) + primary purpose so a
new contributor can locate the right module from one click.

## Demo

A short asciicast walks the three CLI surfaces against a
hermetic seed directory. Reproduce or re-record with:

```bash
scripts/record_demo.sh
```

See [`examples/asciicast/recording_script.md`](examples/asciicast/recording_script.md)
for the per-act keystroke list. The output lands at
`docs/asciicasts/care-tour.cast` and can be embedded in this
README once recorded (`asciinema upload` for a hosted player or
a local relative link for offline reading).

## Architecture

CARE is the consumer at the top of a four-module stack — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the layer-by-layer
walk-through (generation / execution / persistence / evolution) with
cross-links to the upstream modules and CARE's internal module
boundaries.

## Configuration

CARE reads its configuration from three sources, in increasing precedence:

1. Defaults baked into `care.config.CareConfig`.
2. `~/.config/care/config.toml` (user-global TOML).
3. `./care.toml` in the current working directory (per-project overrides).
4. `CARE_*` environment variables.

Nested fields use double-underscores in env-var form, mirroring how the
TOML sections nest. For example, `[mage] mode = "fast"` in TOML is
`CARE_MAGE__MODE=fast` in the environment.

See [`.env.example`](.env.example) for the complete list of supported
variables with descriptions and defaults. Copy it to `.env` (or export the
vars in your shell) and override only what you need.

### Quick reference

| Section     | Env-var prefix      | Purpose                                                                         |
| ----------- | ------------------- | ------------------------------------------------------------------------------- |
| `mage`      | `CARE_MAGE__*`      | Agent chain generator (provider, API key, …)                                    |
| `memory`    | `CARE_MEMORY__*`    | GigaEvo Memory connection                                                       |
| `platform`  | `CARE_PLATFORM__*`  | GigaEvo Platform connection                                                     |
| `sandbox`   | `CARE_SANDBOX__*`   | AgentSkill sandbox backend + limits                                             |
| `tools`     | `CARE_TOOLS__*`     | `@carl_tool` registry + bundled builtins (web_search, …) + on-the-fly synthesis |
| `telemetry` | `CARE_TELEMETRY__*` | Opt-in event-stream sink (Langfuse, …)                                          |
| `defaults`  | `CARE_DEFAULTS__*`  | UI defaults (language, history size)                                            |

## CLI

`maestro` (no subcommand) launches the TUI. The headless subcommands share
the `CareConfig` and data layers the TUI uses — every screen's primary
affordance has a terminal twin.

**Setup**

- `maestro init [--non-interactive] [--api-key X] [--base-url Y] [--model Z] [--mode ad_hoc|production] [--force]` —
  one-shot quick-start: writes a minimal `.env` with the four agent
  chain generator + chat keys a fresh checkout needs. Refuses to overwrite an existing
  `.env` without `--force`.

**Setup (cont.)**

- `maestro doctor [--config PATH] [--no-probes]` — diagnostic report:
  which `CARE_*` env vars are set, the resolved config path, installed
  extras, and live probes against Memory, MAGE, and Platform (the MAGE
  probe makes an authenticated round-trip so an expired key shows red).
- `maestro migrate-secrets` — move literal `*_api_key` values in
  `~/.config/care/config.toml` into the system keystore and rewrite the
  TOML with `keystore://…` URLs.

**Discovery & validation**

- `maestro catalog [--json] [--kind ...]` — list installed AgentSkills,
  MCP servers, tools, capability memory cards.
- `maestro validate <chain.json>` — parse + preflight an agent chain.
- `maestro import <pattern>... [--apply]` — batch-validate (dry-run
  default) or import chain JSON files (`--apply` saves them to Memory).

**Generate / run / replay**

- `maestro generate "<task>" [--mode fast|deep] [--save NAME] [--output PATH]` —
  one-shot agent chain generation.
- `maestro run <chain_id> [--execute] [--task TEXT] [--input KEY=VAL] [--save-result NAME]` —
  fetch a saved chain, preflight, and optionally execute the chain.
- `maestro replay <run.json>` — step through a saved
  `ReasoningResult` / `RunRecord` JSON.

**Memory browse**

- `maestro memory ls [--entity-type ...] [--tag ...] [--q ...]` —
  list saved entities.
- `maestro memory show <entity_id> [--content-only]` — drill down on
  a single entity.
- `maestro memory history <chain_id>` — list recorded runs for a
  chain.
- `maestro search "<query>" [--search-type bm25|vector|hybrid]` —
  BM25 / vector / hybrid search across saved entities.
- `maestro diff <left_id> <right_id>` — side-by-side chain compare.
- `maestro lineage <chain_id>` — walk the ancestry DAG.
- `maestro favourite <entity_id> [--off]` — star / unstar a library
  entity.

**Edit, versions & lifecycle**

- `maestro revise <chain_id> "<change>" [--yes]` — AI-edit a chain into a
  new version (previews the plan; `--yes` saves).
- `maestro versions <id>` — version history + channel pointers.
- `maestro rollback <id> --to <version-id> [--channel]` — repoint a
  channel (reversible).
- `maestro promote <id> [--from latest --to stable]` — move a channel
  pointer.
- `maestro forget <id> [--force]` — soft-delete (preview without
  `--force`).
- `maestro export <out.tar.gz> <chain_id>... [--skill ID]` — pack a
  portable bundle (inverse of `import`).

**Capabilities & evolution**

- `maestro marketplace "<query>"` — search shared agent_skill listings.
- `maestro evolve <chain_id> [--wait] [--accept]` — submit + watch +
  accept an evolution run.

**Eval datasets**

- `maestro dataset list|add|run|export <chain_id> …` — manage a chain's
  eval cases; `run` substring-scores, `export` writes JSONL for external
  frameworks.

**Agent hub**

- `maestro deploy <chain_id> [--name --channel]` — deploy a chain as an
  HTTP agent (needs a running hub).
- `maestro deployments` / `maestro metrics [name]` — list deployments /
  show usage metrics.

**Long-term memory**

- `maestro remember "<note>"` — save an explicit note to long-term
  memory.
- `maestro notes` — show stored long-term-memory notes.

**UX**

- `maestro help [--markdown] [--commands]` — render the tutorial +
  cheat-sheet; `--commands` prints the CLI↔TUI parity table.

Run `maestro <subcommand> --help` for the full flag set on each.
