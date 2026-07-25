#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 VCM|BUPT [additional train.py arguments]" >&2
  exit 2
fi

dataset="${1^^}"
shift
if [[ "$dataset" != "VCM" && "$dataset" != "BUPT" ]]; then
  echo "Dataset must be VCM or BUPT, got: $dataset" >&2
  exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda_env="${CONDA_ENV:-rgbt}"
data_root="${DATA_ROOT:-/data/ydl/datasets}"
output_root="${OUTPUT_ROOT:-/data/ydl/experiments/nuanceid}"
device="${DEVICE:-cuda}"
cuda_device="${CUDA_DEVICE:-0}"
dataset_lower="${dataset,,}"

annotation_file="$data_root/$dataset_lower/${dataset}.json"
command=(
  conda run --no-capture-output -n "$conda_env"
  python "$repo_dir/train.py"
  --name "${dataset_lower}_shared_text_6f_sdm_id"
  --dataset_name "$dataset"
  --root_dir "$data_root"
  --annotation_file "$annotation_file"
  --caption_source json
  --output_dir "$output_root"
  --device "$device"
  --loss_names sdm+id
  --batch_size 8
  --test_batch_size 8
  --num_epoch 30
  --num_workers 2
  --num_frames 6
  --train_caption_mode single
  --gallery_mode mixed
  --sampler random
  --lr 5e-6
  --warmup_epochs 1
  --seed 1
)

cd "$repo_dir"
if [[ "$device" == "cuda" || "$device" == "auto" ]]; then
  CUDA_VISIBLE_DEVICES="$cuda_device" exec "${command[@]}" "$@"
else
  exec "${command[@]}" "$@"
fi
