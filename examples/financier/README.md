# Financier helper — end-to-end CARE example

A chain that reads a quarterly financial statement out of a PDF, normalises
the headline figures into a typed object, and (after the run completes)
persists the result as a `memory_card` so future quarters can be compared
across time.

This example demonstrates three CARE primitives the weather example didn't
touch:

1. **AgentSkill discovery** — `skills/pdf-extractor/SKILL.md` is a real
   SKILL.md manifest that CARE's `CapabilityCatalog` finds (`care catalog
   --skills examples/financier/skills`).
2. **`structured_output` step** — schema-constrained JSON extraction with
   a real JSON Schema (`QuarterlyFinancials`) declared inline.
3. **Memory-card persistence** — at the end of the run, the caller writes
   a `memory_card` via `CareMemory.save_memory_card(...)` so the
   structured result becomes a searchable artefact.

## What's in this directory

| File / Dir                              | Purpose                                                            |
| --------------------------------------- | ------------------------------------------------------------------ |
| `chain.json`                            | The CARL chain — `tool` extract → `structured_output` normalise. |
| `skills/pdf-extractor/SKILL.md`         | AgentSkill manifest discovered by `care catalog`.                  |
| `README.md`                             | This walkthrough.                                                  |

## Try it

```bash
# 1. Validate the chain parses + preflights cleanly.
uv run care validate examples/financier/chain.json

# 2. See the bundled AgentSkill in the catalog.
uv run care catalog --skills examples/financier/skills

# 3. (Optional) Dry-run import the chain.
uv run care import examples/financier/chain.json
```

All three commands work offline against `mmar-carl 0.2.0` — no LLM key,
no PDF runtime, no Memory server.

## Chain shape

```text
[1] extract_pdf_text (tool)         tool_name=pdf_extractor
       │  input_mapping: {pdf_path: $inputs.pdf_path}
       │  → result.extracted
       ▼
[2] normalise_financials (structured_output)
       schema=QuarterlyFinancials {period, currency, revenue,
                                    net_income, total_expenses,
                                    notable_alerts?}
       strict_json=true
       → JSON matching the schema
```

The `tool` step is a placeholder for the future `agent_skill` step type —
CARL ships AGENT_SKILL on its dev branch (see
[PREPARE.md §5.1](../../PREPARE.md)), but the installed `mmar-carl 0.2.0`
doesn't expose it yet. To use the SKILL.md directly today, register a
Python tool named `pdf_extractor` via `care.load_tools_into_context(...)`
that calls the same SKILL.md body; when CARL's `agent_skill` step lands
in CARE's runtime, the chain swaps the `tool` step for `agent_skill`
without changing the schema or the downstream step.

## Persisting the run as a `memory_card`

After the chain finishes, the caller persists the structured result so
future runs can search prior quarters:

```python
from care import CareMemory
from gigaevo_client import GigaEvoClient

client = GigaEvoClient(base_url="...", api_key="...")
memory = CareMemory(client)

memory.save_memory_card(
    {
        "category": "financials",
        "task_description": "Q3 2025 statement, AcmeCo",
        "description": "Revenue +12% YoY; expenses +8%; no risk alerts.",
        "tags": ["financials", "acmeco", "Q3-2025"],
        # ... the QuarterlyFinancials JSON the structured_output step
        # produced lives in `content.metadata` or under a custom key.
    },
    name="AcmeCo Q3 2025 financials",
    tags=["financials", "acmeco"],
)
```

The same `memory_card` shape is what
`care.runtime.record_run_completion(...)` writes for any chain run
(see [TODO §5 P0](../../TODO.md)) — financial summaries are just one
more category in that channel.

## What CARE primitives this exercises

| Step                  | Primitive                          | Module                       |
| --------------------- | ---------------------------------- | ---------------------------- |
| `care validate`       | `care.validate_chain`              | `care/preflight.py`          |
| `care catalog --skills` | `care.build_catalog` → `_scan_skill_dir` | `care/catalog.py`     |
| `care import`         | `care.import_chains`               | `care/bulk_import.py`        |
| `pdf-extractor` discovery | `_parse_skill_md` frontmatter parser | `care/catalog.py` (private) |
| Memory-card persist   | `CareMemory.save_memory_card`      | `care/memory.py`             |
| Skill promotion       | `care.promote_skill_to_memory`     | `care/skills.py`             |

See [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) for the full
layer-by-layer reference.
