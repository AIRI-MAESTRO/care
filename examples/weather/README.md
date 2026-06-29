# Weather report — end-to-end CARE example

A minimal chain that goes from a user query (_"what's the weather in `<city>`?"_)
all the way to a friendly summary, exercising the two integration points CARE
users hit first:

1. **An MCP server** — `weather` from `mcp_servers.toml`. CARE's
   `CapabilityCatalog` discovers it; `MAGE` (or any chain author) references
   it via an `mcp` step.
2. **An LLM summariser step** — pulls the raw MCP payload out of memory and
   renders it as plain English.

## What's in this directory

| File                | Purpose                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `chain.json`        | The CARL chain — one `mcp` step + one `llm` step.                 |
| `mcp_servers.toml`  | MCP server registration. Points at `uvx mcp-weather` (placeholder). |
| `README.md`         | This walkthrough.                                                  |

## Try it

The chain is a static artifact — no MAGE generation needed. The whole flow:

```bash
# 1. Validate that the chain parses + preflights cleanly.
uv run care validate examples/weather/chain.json

# 2. See the MCP server in the catalog.
uv run care catalog --mcp-config examples/weather/mcp_servers.toml

# 3. (Optional) Dry-run a bulk import of the chain into Memory.
uv run care import examples/weather/chain.json
```

Steps 1–3 work offline, against `mmar-carl 0.2.0` only — no LLM key, no
Docker, no MCP server actually running.

To **execute** the chain, point CARE's runtime at a real LLM provider and
install/run the MCP weather server you want to back the `weather` entry in
`mcp_servers.toml`:

```bash
# Install the MCP server (one of):
uvx mcp-weather --help          # whichever uvx package your env has
# or: pip install mcp-weather

# Set the LLM provider keys in your shell / `~/.config/care/config.toml`:
export CARE_MAGE__PROVIDER=openai
export CARE_MAGE__API_KEY=sk-...

# Launch the TUI and import the chain via the LibraryScreen (P0 work in
# progress); for now the CLI dry-run is the supported path.
uv run care
```

## Chain shape

The chain has two steps wired top-to-bottom — `summarise_forecast` depends
on `fetch_forecast`'s output:

```text
[1] fetch_forecast (mcp)   tool=get_forecast, server=weather
       │  args: {city: $inputs.city, units: metric}
       │  → memory["forecast_raw"]
       ▼
[2] summarise_forecast (llm)   2-3 sentences from forecast_raw
```

The `arguments` + `argument_mapping` shape on the MCP step shows the two
binding modes side by side: `arguments.city = "$city"` is a literal
placeholder; `argument_mapping.city = "$inputs.city"` resolves the user's
input at runtime. Pick whichever your chain author convention prefers —
both reach the MCP server.

## What CARE primitives this exercises

| Step              | Primitive                       | Module                       |
| ----------------- | ------------------------------- | ---------------------------- |
| `care validate`   | `care.validate_chain`           | `care/preflight.py`          |
| `care catalog`    | `care.build_catalog`            | `care/catalog.py`            |
| `care import`     | `care.import_chains`            | `care/bulk_import.py`        |
| MCP discovery     | `_scan_mcp_config`              | `care/catalog.py` (private)  |
| Chain parse       | `mmar_carl.ReasoningChain.from_dict` | `mmar-carl` (lazy import) |

See [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) for the full
layer-by-layer reference.
