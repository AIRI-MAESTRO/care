---
name: pdf-extractor
description: Extract text + tables from a PDF financial statement.
tags:
  - pdf
  - finance
  - extraction
allowed-tools:
  - Read
  - Bash
---

# PDF extractor

A placeholder SKILL.md for the CARE `examples/financier/` bundle.

The real implementation would invoke `pdftotext` (or the
`anthropics/skills/pdf` SKILL distribution) on an input PDF and
return raw text + parsed tables. For example purposes the manifest
here is the surface CARE's catalog + the eventual `agent_skill`
step actually consume — the body content is documentation only.

## Inputs

- `pdf_path` (string) — absolute path to the source PDF.

## Outputs

- `text` (string) — extracted plain text.
- `tables` (list of objects) — `{name, columns, rows}` for each
  detected table.

## Notes

Once CARL ships the `agent_skill` step type (it's already on the
dev branch — see [PREPARE.md §5.1](../../../PREPARE.md)), the
`financier` chain can reference this skill via its
SHA-pinned `uri`. Until then the chain falls back to a `tool`
step that calls a Python implementation registered with
`care.load_tools_into_context`.
