#!/usr/bin/env bash
# Overlay current Platform chain templates onto a live runner experiment dir.
#
# Fixes stale helper.py (mmar_carl / broken LLM wrapper) and validate.py
# (Cyrillic ROUGE, summarization prompts) without rebuilding master-api.
#
# Usage:
#   ./scripts/sync_experiment_chain_templates.sh exp_<uuid>
#   ./scripts/sync_experiment_chain_templates.sh exp_<uuid> /path/to/gigaevo-platform/.../chain

set -euo pipefail

EXP_ID="${1:?experiment id (exp_...)}"
TEMPLATE_DIR="${2:-$(cd "$(dirname "$0")/../.." && pwd)/gigaevo-platform/master_api/src/folder_constructor/validate_templates/chain}"
CONTAINER="${CARE_PLATFORM__RUNNER_CONTAINER:-gigaevo-platform-runner-api-1-1}"
CLONE="${CARE_PLATFORM__GIGAVOLVE_CLONE_ROOT:-/tmp/gigavolve/gigaevo-core-1}"
DEST="${CLONE}/problems/${EXP_ID}"

if [[ ! -d "$TEMPLATE_DIR" ]]; then
  echo "ERROR: template dir not found: $TEMPLATE_DIR" >&2
  exit 1
fi

for f in helper.py validate.py chain_client.py chain_runner.py chain_types.py chain_validation.py context.py; do
  src="$TEMPLATE_DIR/$f"
  [[ -f "$src" ]] || continue
  docker cp "$src" "${CONTAINER}:${DEST}/${f}"
  docker exec "${CONTAINER}" bash -lc "chmod a+r '${DEST}/${f}' && chown gigaevouser:gigaevouser '${DEST}/${f}'"
  echo "copied $f"
done

echo "Done. Re-run validation or start a fresh evolution experiment."
