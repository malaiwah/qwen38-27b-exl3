#!/bin/bash
# Run the measured sparse EXL3/B12X route probe inside a maintenance transaction.
set -euo pipefail

[[ "${FRONTIER_TRANSACTION_ACTIVE:-}" == "1" ]] || {
  echo "frontier_route_probe_callback: maintenance transaction is not active" >&2
  exit 2
}
for name in FRONTIER_CAMPAIGN_CONTAINER FRONTIER_CAMPAIGN_IMAGE \
  FRONTIER_CAMPAIGN_CACHE_DIR FRONTIER_CAMPAIGN_WORK_DIR; do
  [[ -n "${!name:-}" ]] || {
    echo "frontier_route_probe_callback: missing ${name}" >&2
    exit 2
  }
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="${SCRIPT_DIR}/frontier_route_probe.py"
SOURCE_ROOT="/home/mbelleau/final-frontier-g0/source-patches"
FP4_HELPER="${SOURCE_ROOT}/exl3_fp4_conversion.py"
TRITON_HELPER="${SOURCE_ROOT}/triton_fp4_quant.py"
FP6_HELPER="${SOURCE_ROOT}/exl3_fp6_conversion.py"
B12X_PACKAGE="${SOURCE_ROOT}/b12x-base-1.2.1/b12x"
OUTPUT="${FRONTIER_CAMPAIGN_WORK_DIR}/runtime-routes.json"
LOG="${FRONTIER_CAMPAIGN_WORK_DIR}/runtime-routes.jsonl"

for path in "$PROBE" "$FP4_HELPER" "$TRITON_HELPER" "$FP6_HELPER"; do
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || {
    echo "frontier_route_probe_callback: missing immutable file ${path}" >&2
    exit 2
  }
done
[[ -d "$B12X_PACKAGE" && ! -L "$B12X_PACKAGE" ]] || {
  echo "frontier_route_probe_callback: missing immutable B12X package" >&2
  exit 2
}
[[ ! -e "$OUTPUT" && ! -e "$LOG" ]] || {
  echo "frontier_route_probe_callback: output already exists" >&2
  exit 2
}

podman run --rm \
  --name "${FRONTIER_CAMPAIGN_CONTAINER}" \
  --network none --ipc=host --device nvidia.com/gpu=all \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e XDG_CACHE_HOME=/cache/jit \
  -e CUDA_CACHE_PATH=/cache/jit \
  -e TRITON_CACHE_DIR=/cache/jit/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  -e B12X_COMPILE_CACHE_DIR=/cache/jit/b12x/compile \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/b12x-cute \
  -v "${FRONTIER_CAMPAIGN_CACHE_DIR}:/cache:rw" \
  -v "${FRONTIER_CAMPAIGN_WORK_DIR}:/work:rw" \
  -v "${PROBE}:/opt/frontier/frontier_route_probe.py:ro" \
  -v "${FP4_HELPER}:/opt/fp4/exl3_fp4_conversion.py:ro" \
  -v "${TRITON_HELPER}:/opt/fp4/triton_fp4_quant.py:ro" \
  -v "${FP6_HELPER}:/opt/fp6/exl3_fp6_conversion.py:ro" \
  -v "${B12X_PACKAGE}:/opt/venv/lib/python3.12/site-packages/b12x:ro" \
  --entrypoint /opt/venv/bin/python \
  "${FRONTIER_CAMPAIGN_IMAGE}" \
  /opt/frontier/frontier_route_probe.py \
  --out /work/runtime-routes.json \
  --log /work/runtime-routes.jsonl \
  --runtime-sha b19029d2309b26c4942425e52b74a0e6dd5d141e \
  --decode-rows 4 --prefill-rows 256 --reps 5
