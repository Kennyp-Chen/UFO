#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Generate a side-by-side expert/policy tracking video.

Usage:
  scripts/generate_validation_video.sh \
    --weights PATH \
    --reference PATH \
    --robot-config PATH \
    [options]

Required:
  --weights PATH       Run directory, checkpoint directory, model directory,
                       or checkpoint/model/model.safetensors.
  --reference PATH     Reference motion dataset in UFO PKL format.
  --robot-config PATH  Robot YAML compatible with both weights and reference.

Options:
  --motion-id ID       Motion index in the reference PKL (default: 0).
  --output PATH        Copy the generated MP4 to this path.
  --device DEVICE      Torch device (default: cuda:0).
  --fps FPS            Simulation frequency (default: 50).
  --render-every N     Render every N simulation steps (default: 2, output 25 FPS).
  --render-size PX     Height of each side of the video (default: 480).
  --max-steps N        Optional rollout limit; omit it for the complete motion.
  -h, --help           Show this help.

CUDA_VISIBLE_DEVICES can be set before this command to select a physical GPU.
The video shows the reference motion on the left and the policy rollout on the right.
EOF
}

weights=""
reference=""
robot_config=""
motion_id=0
output=""
device="cuda:0"
fps=50
render_every=2
render_size=480
max_steps=""

while (($#)); do
  case "$1" in
    --weights)
      weights="${2:?--weights requires a path}"
      shift 2
      ;;
    --reference)
      reference="${2:?--reference requires a path}"
      shift 2
      ;;
    --robot-config)
      robot_config="${2:?--robot-config requires a path}"
      shift 2
      ;;
    --motion-id)
      motion_id="${2:?--motion-id requires an integer}"
      shift 2
      ;;
    --output)
      output="${2:?--output requires a path}"
      shift 2
      ;;
    --device)
      device="${2:?--device requires a value}"
      shift 2
      ;;
    --fps)
      fps="${2:?--fps requires an integer}"
      shift 2
      ;;
    --render-every)
      render_every="${2:?--render-every requires an integer}"
      shift 2
      ;;
    --render-size)
      render_size="${2:?--render-size requires an integer}"
      shift 2
      ;;
    --max-steps)
      max_steps="${2:?--max-steps requires an integer}"
      shift 2
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

for required_name in weights reference robot_config; do
  if [[ -z "${!required_name}" ]]; then
    echo "--${required_name//_/-} is required" >&2
    usage >&2
    exit 2
  fi
done

for value_name in motion_id fps render_every render_size; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "--${value_name//_/-} must be a non-negative integer" >&2
    exit 2
  fi
done
if ((fps == 0 || render_every == 0 || render_size == 0)); then
  echo "--fps, --render-every, and --render-size must be positive" >&2
  exit 2
fi
if [[ -n "$max_steps" ]] && { ! [[ "$max_steps" =~ ^[0-9]+$ ]] || ((max_steps == 0)); }; then
  echo "--max-steps must be a positive integer" >&2
  exit 2
fi

weights="$(realpath -m -s "$weights")"
if [[ -f "$weights" ]]; then
  if [[ "$(basename "$weights")" != "model.safetensors" ]]; then
    echo "Weight file must be named model.safetensors: $weights" >&2
    exit 2
  fi
  model_dir="$(dirname "$weights")"
elif [[ -d "$weights/checkpoint/model" ]]; then
  model_dir="$weights/checkpoint/model"
elif [[ -d "$weights/model" && "$(basename "$weights")" == "checkpoint" ]]; then
  model_dir="$weights/model"
elif [[ -d "$weights" && "$(basename "$weights")" == "model" ]]; then
  model_dir="$weights"
else
  echo "Could not find checkpoint/model below --weights: $weights" >&2
  exit 2
fi

checkpoint_dir="$(dirname "$model_dir")"
model_folder="$(dirname "$checkpoint_dir")"
for path in \
  "$model_dir/model.safetensors" \
  "$model_dir/config.json" \
  "$model_dir/init_kwargs.json" \
  "$model_folder/config.json"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing or empty checkpoint file: $path" >&2
    exit 2
  fi
done

reference="$(realpath -e "$reference")"
robot_config="$(realpath -e "$robot_config")"

command=(
  uv run --no-sync python -m humanoidverse.tracking_inference
  --model-folder "$model_folder"
  --data-path "$reference"
  --robot-config "$robot_config"
  --device "$device"
  --headless
  --disable-dr
  --disable-obs-noise
  --save-mp4
  --export-onnx false
  --motion-list "$motion_id"
  --fps "$fps"
  --render-every "$render_every"
  --render-size "$render_size"
  --log-every-steps 500
)
if [[ -n "$max_steps" ]]; then
  command+=(--max-steps "$max_steps")
fi

echo "Weights:  $model_dir/model.safetensors"
echo "Reference: $reference (motion_id=$motion_id)"
echo "Robot:     $robot_config"
echo "Timing:    simulation=${fps}Hz, video=$(awk -v f="$fps" -v n="$render_every" 'BEGIN { print f / n }') FPS"

cd "$PROJECT_DIR"
"${command[@]}"

generated="$model_folder/tracking_inference/tracking_${motion_id}.mp4"
if [[ ! -s "$generated" ]]; then
  echo "Expected video was not generated: $generated" >&2
  exit 1
fi

if [[ -n "$output" ]]; then
  output="$(realpath -m -s "$output")"
  if [[ "$output" != "$(realpath -m -s "$generated")" ]]; then
    mkdir -p "$(dirname "$output")"
    cp -f "$generated" "$output"
  fi
  generated="$output"
fi

echo "Video ready: $generated"
