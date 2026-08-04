#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-piplus-soccer}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/data/hejunfu/projects/BFM_piplus}"
REMOTE_RUN="${REMOTE_RUN:-runs/piplus_bfm_fb_6x4090_seed4728}"
LOCAL_RUN="${LOCAL_RUN:-runs/remote_piplus_bfm_fb_6x4090_seed4728}"
LOCAL_MOTION_DIR="${LOCAL_MOTION_DIR:-cache/remote_piplus/data/piplus}"
SYNC_MOTION=1
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

usage() {
  cat <<'EOF'
Pull the latest inference-only checkpoint from an active remote UFO run.

Usage: scripts/pull_remote_checkpoint.sh [options]

Options:
  --host HOST          SSH host alias
  --remote-project DIR Remote UFO project directory
  --remote-run DIR     Run path relative to the remote project
  --local-run DIR      Local model folder used by inference
  --motion-dir DIR     Local directory for the full-motion inference dataset
  --no-motion          Do not sync the PiPlus full-motion dataset
  -h, --help           Show this help

The optimizer state and replay buffers are intentionally excluded. Existing
local checkpoints remain usable until a fully validated download is ready.
EOF
}

while (($#)); do
  case "$1" in
    --host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-project)
      REMOTE_PROJECT="$2"
      shift 2
      ;;
    --remote-run)
      REMOTE_RUN="$2"
      shift 2
      ;;
    --local-run)
      LOCAL_RUN="$2"
      shift 2
      ;;
    --motion-dir)
      LOCAL_MOTION_DIR="$2"
      shift 2
      ;;
    --no-motion)
      SYNC_MOTION=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for command in ssh rsync sha256sum; do
  command -v "$command" >/dev/null || {
    echo "Required command not found: $command" >&2
    exit 1
  }
done

if [[ "$REMOTE_RUN" = /* || "$REMOTE_RUN" = *".."* ]]; then
  echo "--remote-run must be a path below --remote-project" >&2
  exit 2
fi
if [[ -e "$LOCAL_RUN/checkpoint" && ! -L "$LOCAL_RUN/checkpoint" ]]; then
  echo "$LOCAL_RUN/checkpoint already exists and is not managed by this script." >&2
  echo "Choose another --local-run to avoid overwriting it." >&2
  exit 1
fi
if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

remote_manifest() {
  ssh -o BatchMode=yes "$REMOTE_HOST" bash -s -- "$REMOTE_PROJECT" "$REMOTE_RUN" <<'REMOTE_SCRIPT'
set -euo pipefail
run_dir="$1/$2"
required=(
  "$run_dir/config.json"
  "$run_dir/checkpoint/train_status.json"
  "$run_dir/checkpoint/model/config.json"
  "$run_dir/checkpoint/model/init_kwargs.json"
  "$run_dir/checkpoint/model/model.safetensors"
)
for path in "${required[@]}"; do
  test -s "$path" || {
    echo "Missing or empty remote checkpoint file: $path" >&2
    exit 1
  }
done
sha256sum "$run_dir/checkpoint/train_status.json"
stat -c '%s %y %n' "${required[@]}"
REMOTE_SCRIPT
}

mkdir -p "$LOCAL_RUN/.checkpoints"
staging="$LOCAL_RUN/.checkpoints/.syncing"

attempt=1
use_existing=0
while ((attempt <= MAX_ATTEMPTS)); do
  echo "Reading remote checkpoint marker (attempt $attempt/$MAX_ATTEMPTS)..."
  before="$(remote_manifest)"
  checkpoint_id="$(printf '%s' "$before" | sha256sum | cut -c1-16)"
  final_checkpoint="$LOCAL_RUN/.checkpoints/$checkpoint_id"

  if [[ -s "$final_checkpoint/model/model.safetensors" && -s "$final_checkpoint/train_status.json" ]]; then
    after="$(remote_manifest)"
    if [[ "$before" == "$after" ]]; then
      use_existing=1
      break
    fi
  fi

  mkdir -p "$staging/checkpoint/model" "$staging/run-files"

  rsync -a --partial --info=progress2 \
    "$REMOTE_HOST:$REMOTE_PROJECT/$REMOTE_RUN/checkpoint/model/" \
    "$staging/checkpoint/model/"
  rsync -a --partial \
    "$REMOTE_HOST:$REMOTE_PROJECT/$REMOTE_RUN/checkpoint/train_status.json" \
    "$staging/checkpoint/train_status.json"
  rsync -a --partial \
    "$REMOTE_HOST:$REMOTE_PROJECT/$REMOTE_RUN/config.json" \
    "$staging/run-files/config.json"

  after="$(remote_manifest)"
  if [[ "$before" == "$after" ]]; then
    break
  fi

  echo "The remote checkpoint changed during transfer; retrying against the new save." >&2
  attempt=$((attempt + 1))
done

if [[ "$before" != "$after" ]]; then
  echo "Could not obtain a stable checkpoint after $MAX_ATTEMPTS attempts." >&2
  echo "The previous local checkpoint, if any, was left unchanged." >&2
  exit 75
fi

checkpoint_id="$(printf '%s' "$after" | sha256sum | cut -c1-16)"
final_checkpoint="$LOCAL_RUN/.checkpoints/$checkpoint_id"
if ((use_existing)); then
  rm -rf "$staging"
else
  mv "$staging/run-files/config.json" "$LOCAL_RUN/config.json"
  mv "$staging/checkpoint" "$final_checkpoint"
  rmdir "$staging/run-files" "$staging"
fi

if [[ ! -s "$LOCAL_RUN/config.json" ]]; then
  rsync -a --partial \
    "$REMOTE_HOST:$REMOTE_PROJECT/$REMOTE_RUN/config.json" \
    "$LOCAL_RUN/config.json"
fi

link_tmp="$LOCAL_RUN/.checkpoint.$$.tmp"
ln -s ".checkpoints/$checkpoint_id" "$link_tmp"
mv -Tf "$link_tmp" "$LOCAL_RUN/checkpoint"

if ((SYNC_MOTION)); then
  mkdir -p "$LOCAL_MOTION_DIR"
  rsync -a --partial --info=progress2 \
    "$REMOTE_HOST:$REMOTE_PROJECT/data/piplus/piplus_lse_lafan.pkl" \
    "$LOCAL_MOTION_DIR/piplus_lse_lafan.pkl"
fi

global_time="$(sed -n 's/^[[:space:]]*"global_time":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$LOCAL_RUN/checkpoint/train_status.json")"
model_size="$(du -h "$LOCAL_RUN/checkpoint/model/model.safetensors" | awk '{print $1}')"
echo
echo "Checkpoint ready: $LOCAL_RUN (global_time=${global_time:-unknown}, model=$model_size)"
if ((SYNC_MOTION)); then
  echo "Motion data ready: $LOCAL_MOTION_DIR/piplus_lse_lafan.pkl"
fi
