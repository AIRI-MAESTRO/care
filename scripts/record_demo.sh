#!/usr/bin/env bash
# CARE asciicast recording wrapper (TODO §10 P3).
#
# Sets up a hermetic environment + drops you into an
# `asciinema rec` session ready to follow the recording script
# in `examples/asciicast/recording_script.md`.
#
# Usage:
#   scripts/record_demo.sh           # records to docs/asciicasts/care-tour.cast
#   scripts/record_demo.sh path.cast # custom output path
#
# Requirements:
#   - asciinema on PATH (`pip install asciinema` or `brew install asciinema`)
#   - `uv` on PATH
#   - run from the CARE repo root

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Output location.
OUT="${1:-docs/asciicasts/care-tour.cast}"
mkdir -p "$(dirname "$OUT")"

# Refuse to overwrite an existing recording without confirmation.
if [[ -f "$OUT" ]]; then
    read -r -p "$OUT already exists. Overwrite? [y/N] " REPLY
    case "$REPLY" in
        y|Y|yes|YES) rm -f "$OUT" ;;
        *) echo "aborted; no changes made"; exit 1 ;;
    esac
fi

# Validate prerequisites.
if ! command -v asciinema >/dev/null 2>&1; then
    echo "error: asciinema not on PATH"
    echo "install via: pip install asciinema (or brew install asciinema)"
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not on PATH"
    echo "install via: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
if [[ ! -d examples/asciicast/seed ]]; then
    echo "error: examples/asciicast/seed not found — running from wrong dir?"
    exit 1
fi

# Demo-friendly env. The placeholder MAGE key keeps the first-run
# probes from short-circuiting on missing credentials.
export CARE_MAGE__API_KEY="${CARE_MAGE__API_KEY:-demo-key}"
export CARE_MAGE__PROVIDER="${CARE_MAGE__PROVIDER:-openai}"

echo "▶ Recording to $OUT"
echo "  Follow examples/asciicast/recording_script.md"
echo "  Stop with Ctrl+D when done."
echo

# `--idle-time-limit` collapses long pauses; `--cols` / `--rows`
# pin a readable size for the README embed.
exec asciinema rec \
    --idle-time-limit 2 \
    --cols 120 \
    --rows 36 \
    --title "CARE — Collaborative Agent Reasoning Ecosystem" \
    "$OUT"
