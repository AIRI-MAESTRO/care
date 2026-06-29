# CARE asciicast recording script

Step-by-step keystrokes the demo recording follows. Run
`scripts/record_demo.sh` to start an `asciinema rec` session
with the seed directory already in place; copy each command
below into the recording at the indicated pace.

## Setup (pre-recording)

```bash
# From the repo root:
scripts/record_demo.sh
```

The wrapper sets `CARE_MAGE__API_KEY=demo-key` (placeholder so
the validation probes don't refuse) and adds
`examples/asciicast/seed/` to the working directory's
`$PATH`-like config search hints. The actual recording then
walks the four-act demo below.

---

## Act 1: Catalog the bundled capabilities (~20s)

Show CARE discovering every kind of capability source from
the seed directory.

```bash
uv run care catalog --json | head -20    # JSON preview
uv run care catalog \
    --skills examples/asciicast/seed/skills \
    --mcp-config examples/asciicast/seed/mcp_servers.toml \
    --tools examples/asciicast/seed/tools
```

Expected highlights:
- One `# agent_skill` entry: `pdf-helper`.
- One `# mcp_server` entry: `weather`.
- One `# tool` entry: `demo_tool`.

## Act 2: Validate a chain (~10s)

Show structured validation surfacing a clean parse + the
preflight summary.

```bash
uv run care validate examples/asciicast/seed/chain.json
```

Expected output:

```text
chain parsed; preflight skipped (0 tools, 0 mcp servers, 0 skills)
```

## Act 3: Dry-run bulk import (~10s)

Show the safe-by-default `care import` dry run reporting one
validated chain.

```bash
uv run care import examples/asciicast/seed/chain.json
```

Expected output:

```text
bulk import: 0 imported, 1 validated, 0 failed
```

## Act 4: TUI snapshot (~30s — optional)

If recording the Textual UI as well: launch `uv run care`,
wait for the welcome screen, press `Ctrl+Q` to exit. The
recording closes here.

---

## Stopping + naming

Stop the recording with `Ctrl+D` inside the `asciinema rec`
session. The wrapper writes the output to
`docs/asciicasts/care-tour.cast`. Upload via
`asciinema upload <file>` if publishing, or keep local for
the README link.

## Why this layout

The four acts only use shipped CLI subcommands. Every
expected line is pinned by
`tests/test_asciicast_harness.py`, so the recording stays
honest — if a future refactor changes the output, the test
fails and the recording script needs an update.
