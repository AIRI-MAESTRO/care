#!/usr/bin/env bash
# Copy tools.comparison + tools.redis2pd into the live runner gigaevo-core clone.
#
# Runner stores gigaevo-core under GIGAVOLVE__CLONE_PATH (default
# /tmp/gigavolve/gigaevo-core-1). Platform's gigavolve_service calls:
#   python -m tools.comparison   # live metrics plot during run
#   python -m tools.redis2pd     # evolution_report.json for master-api
#
# Those modules live in the gigaevo-core repo (tools/), but the runtime
# clone is often synced from a branch that omits them — MAESTRO then
# shows empty metrics even when Redis has fitness data.
#
# Usage:
#   ./scripts/sync_runner_gigaevo_tools.sh
#   ./scripts/sync_runner_gigaevo_tools.sh /path/to/gigaevo-core my-runner-container

set -euo pipefail

CORE_DIR="${1:-$(cd "$(dirname "$0")/../.." && pwd)/gigaevo-core}"
CONTAINER="${2:-gigaevo-platform-runner-api-1-1}"
CLONE_TOOLS="/tmp/gigavolve/gigaevo-core-1/tools"

if [[ ! -f "$CORE_DIR/tools/comparison.py" ]]; then
  echo "ERROR: $CORE_DIR/tools/comparison.py not found" >&2
  exit 1
fi

for mod in comparison redis2pd; do
  src="$CORE_DIR/tools/${mod}.py"
  if [[ ! -f "$src" ]]; then
    echo "WARN: missing $src — skipping" >&2
    continue
  fi
  docker cp "$src" "${CONTAINER}:${CLONE_TOOLS}/${mod}.py"
  echo "copied ${mod}.py → ${CONTAINER}:${CLONE_TOOLS}/"
done

echo "Done. Metrics export should work on the next runner poll cycle."
