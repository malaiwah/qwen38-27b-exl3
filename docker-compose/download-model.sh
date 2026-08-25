#!/bin/bash
# Download model weights for Qwen3.8-27B EXL3 K5K6-hydrated.
#
# Usage:
#   ./download-model.sh              # download our EXL3 model
#   ./download-model.sh --all        # download EXL3 + official FP8 + BF16 reference
set -euo pipefail

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface/hub}"

download() {
  local repo="$1"
  local dir="$HF_CACHE/models--${repo//\//--}"
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "Already downloaded: $repo ($(du -sh "$dir" | cut -f1))"
    return 0
  fi
  echo "Downloading: $repo"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$dir')
print('Done: $repo')
"
}

# Our EXL3 K5K6-hydrated model (21 GB)
download "malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated"

if [ "${1:-}" = "--all" ]; then
  # Official FP8 model for comparison (29 GB)
  download "Qwen/Qwen3.8-27B-FP8"
  # BF16 reference for KLD teacher (54 GB)
  download "Qwen/Qwen3.8-27B"
fi

echo ""
echo "Models downloaded to: $HF_CACHE"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and adjust paths"
echo "  2. Ensure patch files are in ./patches/"
echo "  3. Run: docker compose --profile <profile> up -d"
echo ""
echo "Profiles: balanced | fidelity | throughput | rtx6000 | rtx6000-bf16-kv | official-fp8 | bf16-reference"
