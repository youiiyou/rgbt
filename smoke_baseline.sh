#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 VCM|BUPT [additional smoke arguments]" >&2
  exit 2
fi

dataset="${1^^}"
shift
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda_env="${CONDA_ENV:-rgbt}"
device="${DEVICE:-cuda}"
cuda_device="${CUDA_DEVICE:-0}"
command=(
  conda run --no-capture-output -n "$conda_env"
  python "$repo_dir/scripts/smoke_baseline.py"
  "$dataset"
  --root "${DATA_ROOT:-/data/ydl/datasets}"
  --device "$device"
)

cd "$repo_dir"
if [[ "$device" == "cuda" || "$device" == "auto" ]]; then
  CUDA_VISIBLE_DEVICES="$cuda_device" exec "${command[@]}" "$@"
else
  exec "${command[@]}" "$@"
fi
