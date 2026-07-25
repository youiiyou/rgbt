#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec conda run -n rgbt python \
  "${repo_dir}/scripts/build_caption_json.py" \
  vcm \
  --root /data/ydl/datasets/vcm \
  "$@"
