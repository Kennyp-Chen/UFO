#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-GPU9}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/data/hejunfu}"
REMOTE_RUN="${REMOTE_RUN:-runs/tech_piplus_2h0w_8gpu_seed4728}"
LOCAL_RUN="${LOCAL_RUN:-$PROJECT_DIR/runs/remote_tech_piplus_2h0w_8gpu_seed4728}"

exec "$PROJECT_DIR/scripts/pull_remote_checkpoint.sh" \
  --host "$REMOTE_HOST" \
  --remote-project "$REMOTE_PROJECT" \
  --remote-run "$REMOTE_RUN" \
  --local-run "$LOCAL_RUN" \
  --no-motion \
  "$@"
