#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /path/to/configs.yaml [/path/to/checkpoint.pth]" >&2
  exit 2
fi

config_file="$(realpath "$1")"
checkpoint="${2:-}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda_env="${CONDA_ENV:-rgbt}"
device="${DEVICE:-cuda}"
cuda_device="${CUDA_DEVICE:-0}"

cd "$repo_dir"
for gallery_mode in rgb ir mixed; do
  command=(
    conda run --no-capture-output -n "$conda_env"
    python "$repo_dir/test.py"
    --config_file "$config_file"
    --gallery_mode "$gallery_mode"
    --device "$device"
  )
  if [[ -n "$checkpoint" ]]; then
    command+=(--checkpoint "$checkpoint")
  fi
  if [[ "$device" == "cuda" || "$device" == "auto" ]]; then
    CUDA_VISIBLE_DEVICES="$cuda_device" "${command[@]}"
  else
    "${command[@]}"
  fi
done

exec conda run --no-capture-output -n "$conda_env" \
  python "$repo_dir/scripts/summarize_results.py" "$(dirname "$config_file")"
