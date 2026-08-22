#!/bin/bash
set -uo pipefail
COMBINED_SCRIPT=/home/mbelleau/qwen38-27b-exl3/patches/trellis-embed/combined_kld.py
BASE=/home/mbelleau/.cache/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf
BASE_IN_CTR=/root/.cache/huggingface/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf
SUITE_VIEW=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-evidence/capture-suite-view
REF=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/reference/hidden-bf16
HEAD=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/lm-head/weight.safetensors
LAUNCHER=/home/mbelleau/worktrees/r30-capture-smoke/r30-launcher
CAP=/home/mbelleau/worktrees/r30-capture-smoke/combined-capture
PREENC_EMBED=/tmp/combined-embed-preenc.pt
PREENC_HEAD=/tmp/combined-head-preenc.pt
IMAGE="docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
VLLM=/opt/venv/lib/python3.12/site-packages/vllm
BITS=${1:-8}

# Common podman args (shared by all steps)
COMMON_ARGS=(
  --device nvidia.com/gpu=all --ipc=host --network none
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
  --env VLLM_ALLOW_INSECURE_SERIALIZATION=1
  --env TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions
  --env PYTHONPATH=/opt/r30-ext
  --env LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
  --volume /home/mbelleau/.cache/huggingface/hub:/root/.cache/huggingface:ro
  --env HF_HOME=/root/.cache/huggingface
  --volume "$BASE:$BASE:ro"
  --volume /home/mbelleau/.cache/jit:/cache/jit:rw
  --volume /home/mbelleau/vllm-exl3-multiprecision.py:$VLLM/model_executor/layers/quantization/exl3.py:ro
  --volume /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp4_conversion.py:/opt/fp4/exl3_fp4_conversion.py:ro
  --volume /home/mbelleau/qwen38-27b-exl3/patches/triton_fp4_quant.py:/opt/fp4/triton_fp4_quant.py:ro
  --volume /home/mbelleau/worktrees/r30-torch-ext/exllamav3_ext/exllamav3_ext.so:/opt/r30-ext/exllamav3_ext.so:ro
  --volume "$(dirname "$COMBINED_SCRIPT"):$(dirname "$COMBINED_SCRIPT")":ro
  --volume "$HEAD:$HEAD:ro"
  --volume /tmp:/tmp:rw
)

# VLLM-specific args (for capture step only)
VLLM_ARGS=(
  --env VLLM_EXL3_MULTIPRECISION=0
  --env VLLM_USE_V2_MODEL_RUNNER=1
  --env VLLM_EXL3_PRE_RECONSTRUCT=0
  --env VLLM_EXL3_B12X_ANY_BITS=1
  --env VLLM_EXL3_B12X_MIN_M=128
  --env VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE=0
  --volume "$LAUNCHER:$LAUNCHER:ro"
  --volume "$SUITE_VIEW:$SUITE_VIEW:ro"
  --volume "$CAP:$CAP:rw"
  --volume /home/mbelleau/scheduler_patch.py:$VLLM/v1/core/sched/scheduler.py:ro
  --volume /home/mbelleau/qwen_gdn_linear_attn_patch.py:$VLLM/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro
  --volume /home/mbelleau/spec_decode_utils_patch.py:$VLLM/v1/worker/gpu/spec_decode/utils.py:ro
  --volume /home/mbelleau/autoregressive_speculator_patch.py:$VLLM/v1/worker/gpu/spec_decode/autoregressive/speculator.py:ro
  --volume /home/mbelleau/qwen3_5_mtp_patch.py:$VLLM/model_executor/models/qwen3_5_mtp.py:ro
  --volume /home/mbelleau/vllm-exl3-linear-ba.py:$VLLM/model_executor/layers/linear.py:ro
  --volume /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp6_conversion.py:/opt/fp6/exl3_fp6_conversion.py:ro
)

wait_gpu_free() {
  for i in $(seq 1 15); do
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_PROCS" -eq 0 ]; then
      echo "GPU free (after ${i}s)"
      return 0
    fi
    echo "  GPU has $GPU_PROCS proc(s), waiting..."
    sleep 2
  done
  echo "WARNING: GPU not free after 30s"
  return 1
}

rm -rf "$CAP" 2>/dev/null
mkdir -p "$CAP"

# Mask service to prevent auto-restart
systemctl --user mask qwen38-27b.service
systemctl --user stop qwen38-27b.service
podman stop -t 0 qwen38-27b 2>/dev/null
podman rm -f qwen38-27b 2>/dev/null
sleep 3
echo "service stopped + masked"

# --- Step 1: Pre-encode both embedding and lm_head ---
echo "=== Step 1: Pre-encoding embedding + lm_head to K${BITS} ==="
podman run --rm --name "combined-preenc-k${BITS}-$$" \
  "${COMMON_ARGS[@]}" \
  --entrypoint python3 "$IMAGE" \
  "$COMBINED_SCRIPT" pre-encode-all \
  --model "$BASE_IN_CTR" --head "$HEAD" --bits "$BITS" \
  --embed-out "$PREENC_EMBED" --head-out "$PREENC_HEAD"
PREENC_EXIT=$?
echo "pre-encode exit: $PREENC_EXIT"

if [ $PREENC_EXIT -ne 0 ]; then
  systemctl --user unmask qwen38-27b.service
  systemctl --user start qwen38-27b.service
  echo "service restored (pre-encode failed)"
  exit 1
fi

# --- Step 2: Capture hidden states with quantized embedding ---
echo "=== Step 2: Capturing with K${BITS} embedding ==="
wait_gpu_free
podman run --rm --name "combined-cap-k${BITS}-$$" \
  "${COMMON_ARGS[@]}" \
  "${VLLM_ARGS[@]}" \
  --entrypoint python3 "$IMAGE" \
  "$COMBINED_SCRIPT" capture \
  --model "$BASE_IN_CTR" --pre-encoded-embed "$PREENC_EMBED" \
  --suite "$SUITE_VIEW" --out "$CAP" \
  --quantization auto --kv-cache-dtype bfloat16 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.90 \
  --max-batched-tokens 2048
CAP_EXIT=$?
if [ $CAP_EXIT -ne 0 ] && [ $CAP_EXIT -ne 139 ]; then
  systemctl --user unmask qwen38-27b.service
  systemctl --user start qwen38-27b.service
  echo "service restored (capture failed)"
  exit 1
fi

# --- Step 3: Compute KLD with quantized lm_head ---
echo "=== Step 3: Computing combined K${BITS} KLD ==="
wait_gpu_free
podman run --rm --name "combined-kld-k${BITS}-$$" \
  "${COMMON_ARGS[@]}" \
  --volume "$REF:$REF:ro" \
  --volume "$CAP:$CAP:ro" \
  --volume "$SUITE_VIEW:$SUITE_VIEW:ro" \
  --entrypoint python3 "$IMAGE" \
  "$COMBINED_SCRIPT" kld \
  --pre-encoded-head "$PREENC_HEAD" --head "$HEAD" \
  --reference "$REF" --captured "$CAP" --suite "$SUITE_VIEW" \
  --output "/tmp/combined-k${BITS}-report.json"
KLD_EXIT=$?
echo "kld exit: $KLD_EXIT"

# Restore service
systemctl --user unmask qwen38-27b.service
systemctl --user start qwen38-27b.service
echo "service restored (unmasked + started)"

if [ -f "/tmp/combined-k${BITS}-report.json" ]; then
  echo "=== Combined K${BITS} (embed + lm_head) KLD ==="
  python3 -c "
import json
r = json.load(open('/tmp/combined-k${BITS}-report.json'))
for k in ('token_mean_kld','context_macro_mean_kld','p95_kld','p99_kld','p999_kld','max_kld','top1_agreement','scored_positions','contexts'):
    if k in r: print(f'  {k}: {r[k]}')
"
fi

exit $KLD_EXIT
