#!/bin/bash
# Capture matched BF16-flow and quant-flow fixtures with the actual EXL3 converter.
set -euo pipefail

[[ "${FRONTIER_TRANSACTION_ACTIVE:-}" == "1" ]] || {
  echo "frontier_trellis_v3_capture_callback: transaction is not active" >&2
  exit 2
}
for name in FRONTIER_CAMPAIGN_CONTAINER FRONTIER_CAMPAIGN_IMAGE \
  FRONTIER_CAMPAIGN_CACHE_DIR FRONTIER_CAMPAIGN_WORK_DIR; do
  [[ -n "${!name:-}" ]] || {
    echo "frontier_trellis_v3_capture_callback: missing ${name}" >&2
    exit 2
  }
done

SOURCE=/home/mbelleau/final-frontier-g0/converter-source
BF16_REPO=/home/mbelleau/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B
EXT_CACHE_SOURCE=/home/mbelleau/final-frontier-g0/qkv-cache-a4
TOOL=/home/mbelleau/final-frontier-g02/tools/trellis_v3_capture.py
COMMON=/home/mbelleau/final-frontier-g02/tools/frontier_common.py
PLAN_ROOT=/home/mbelleau/final-frontier-g02/v3-plans
for path in "$SOURCE" "$BF16_REPO" "$EXT_CACHE_SOURCE"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "missing immutable v3 directory ${path}" >&2
    exit 2
  }
done
for path in "$TOOL" "$COMMON" "$PLAN_ROOT/capture-bf16-plan.json" "$PLAN_ROOT/capture-quant-plan.json"; do
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || {
    echo "missing immutable v3 input ${path}" >&2
    exit 2
  }
done
for output in \
  "$FRONTIER_CAMPAIGN_WORK_DIR/bf16-flow.json" \
  "$FRONTIER_CAMPAIGN_WORK_DIR/bf16-flow.safetensors" \
  "$FRONTIER_CAMPAIGN_WORK_DIR/quant-flow.json" \
  "$FRONTIER_CAMPAIGN_WORK_DIR/quant-flow.safetensors"; do
  [[ ! -e "$output" ]] || {
    echo "v3 output already exists: ${output}" >&2
    exit 2
  }
done
cp -a "$EXT_CACHE_SOURCE/." "$FRONTIER_CAMPAIGN_CACHE_DIR/"

run_capture() {
  local label=$1
  local plan=$2
  podman run --rm --replace \
    --name "$FRONTIER_CAMPAIGN_CONTAINER" \
    --network none --ipc=host --device nvidia.com/gpu=all \
    --tmpfs /usr/local/cuda-13.2/lib64:rw,size=16m \
    -e TRELLIS_V3_EXLLAMAV3_SOURCE=/src \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$SOURCE:/src:ro" \
    -v "$TOOL:/opt/frontier/trellis_v3_capture.py:ro" \
    -v "$COMMON:/opt/frontier/frontier_common.py:ro" \
    -v "$plan:/opt/frontier/plan.json:ro" \
    -v "$BF16_REPO:/models/bf16-repo:ro" \
    -v "$FRONTIER_CAMPAIGN_CACHE_DIR:/cache:rw" \
    -v "$FRONTIER_CAMPAIGN_WORK_DIR:/work:rw" \
    -v /home/mbelleau/final-frontier-g0/source-patches/b12x-base-1.2.1/b12x:/opt/venv/lib/python3.12/site-packages/b12x:ro \
    --entrypoint /bin/bash \
    "$FRONTIER_CAMPAIGN_IMAGE" \
    -lc 'set -euo pipefail
      ln -sf /usr/local/cuda-13.2/targets/x86_64-linux/lib/* /usr/local/cuda-13.2/lib64/
      export PYTHONPATH=/opt/frontier:/src
      export EXL3_BITS_FIXED='"'"'{"^.*self_attn\\..*$":6,"^.*linear_attn\\..*$":6}'"'"'
      export EXL3_BITS_OVERRIDE='"'"'{"^.*mlp\\.down_proj$":6,"^.*mlp\\.(gate|up)_proj$":5}'"'"'
      exec /opt/venv/bin/python /opt/frontier/trellis_v3_capture.py \
        --v3-plan /opt/frontier/plan.json --v3-out /work/'"$label"'.json \
        -i /models/bf16-repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
        -w /work/'"$label"'-converter -o /work/'"$label"'-out \
        -b 4 -hb 6 -mb 4 -vb 16 -cr 2 -cc 128 -cpi 0 -cb mcg -d 0' \
    >"$FRONTIER_CAMPAIGN_WORK_DIR/${label}.log" 2>&1
}

run_capture bf16-flow "$PLAN_ROOT/capture-bf16-plan.json"
run_capture quant-flow "$PLAN_ROOT/capture-quant-plan.json"
