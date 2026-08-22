#!/bin/bash
set -uo pipefail
LAUNCHER=/home/mbelleau/worktrees/r30-capture-smoke/r30-launcher
FIDELITY=$LAUNCHER/tools/fidelity.py
TRELLIS_SCRIPT=/home/mbelleau/qwen38-27b-exl3/patches/trellis-embed/trellis_embed_kld.py
BASE=/home/mbelleau/.cache/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf
SUITE_VIEW=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-evidence/capture-suite-view
CAP=/home/mbelleau/worktrees/r30-capture-smoke/balanced-k6-capture
BASE_IN_CTR=/root/.cache/huggingface/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf
REF=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/reference/hidden-bf16
HEAD=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/lm-head/weight.safetensors
PREENC=/tmp/trellis-embed-preencoded-balanced.pt
IMAGE="docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
VLLM=/opt/venv/lib/python3.12/site-packages/vllm
BITS=${1:-6}

# Balanced profile env vars (from run-qwen38-27b.sh)
BALANCED_ENV=(
  --env VLLM_EXL3_FP4_LAYERS=,
  --env VLLM_EXL3_FP6_LAYERS=mlp.gate_up_proj
  --env VLLM_EXL3_PREFILL_RECONSTRUCT_M=1
  --env VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB=512
  --env VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE=0
  --env VLLM_EXL3_FOLD_FP32_BUDGET_MB=48
  --env VLLM_EXL3_B12X_ANY_BITS=1
  --env VLLM_EXL3_B12X_MIN_M=128
  --env VLLM_EXL3_SKIP_TRELLIS_PREP=0
  --env VLLM_EXL3_MULTIPRECISION=0
  --env VLLM_USE_V2_MODEL_RUNNER=1
  --env VLLM_ALLOW_INSECURE_SERIALIZATION=1
  --env TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions
  --env PYTHONPATH=/opt/r30-ext
  --env LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
  --env HF_HOME=/root/.cache/huggingface
)

COMMON_VOL=(
  --device nvidia.com/gpu=all --ipc=host --network none
  --volume /home/mbelleau/.cache/huggingface/hub:/root/.cache/huggingface:ro
  --volume "$BASE:$BASE:ro"
  --volume /home/mbelleau/.cache/jit:/cache/jit:rw
  --volume /home/mbelleau/vllm-exl3-multiprecision.py:$VLLM/model_executor/layers/quantization/exl3.py:ro
  --volume /home/mbelleau/scheduler_patch.py:$VLLM/v1/core/sched/scheduler.py:ro
  --volume /home/mbelleau/qwen_gdn_linear_attn_patch.py:$VLLM/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro
  --volume /home/mbelleau/spec_decode_utils_patch.py:$VLLM/v1/worker/gpu/spec_decode/utils.py:ro
  --volume /home/mbelleau/autoregressive_speculator_patch.py:$VLLM/v1/worker/gpu/spec_decode/autoregressive/speculator.py:ro
  --volume /home/mbelleau/qwen3_5_mtp_patch.py:$VLLM/model_executor/models/qwen3_5_mtp.py:ro
  --volume /home/mbelleau/vllm-exl3-linear-ba.py:$VLLM/model_executor/layers/linear.py:ro
  --volume /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp4_conversion.py:/opt/fp4/exl3_fp4_conversion.py:ro
  --volume /home/mbelleau/qwen38-27b-exl3/patches/triton_fp4_quant.py:/opt/fp4/triton_fp4_quant.py:ro
  --volume /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp6_conversion.py:/opt/fp6/exl3_fp6_conversion.py:ro
  --volume /home/mbelleau/worktrees/r30-torch-ext/exllamav3_ext/exllamav3_ext.so:/opt/r30-ext/exllamav3_ext.so:ro
  --volume "$(dirname "$TRELLIS_SCRIPT"):$(dirname "$TRELLIS_SCRIPT")":ro
  --volume /tmp:/tmp:rw
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

# --- Step 1: Pre-encode embedding to K{BITS} ---
echo "=== Step 1: Pre-encoding embedding to K${BITS} ==="
podman run --rm --name "balanced-preenc-k${BITS}-$$" \
  "${BALANCED_ENV[@]}" "${COMMON_VOL[@]}" \
  --entrypoint python3 "$IMAGE" \
  "$TRELLIS_SCRIPT" pre-encode \
  --model "$BASE_IN_CTR" --bits "$BITS" --output "$PREENC"
PREENC_EXIT=$?
echo "pre-encode exit: $PREENC_EXIT"

if [ $PREENC_EXIT -ne 0 ]; then
  systemctl --user unmask qwen38-27b.service
  systemctl --user start qwen38-27b.service
  echo "service restored (pre-encode failed)"
  exit 1
fi

# --- Step 2: Capture with balanced profile + K{BITS} embedding ---
echo "=== Step 2: Capturing with balanced profile + K${BITS} embedding ==="
wait_gpu_free
podman run --rm --name "balanced-cap-k${BITS}-$$" \
  "${BALANCED_ENV[@]}" "${COMMON_VOL[@]}" \
  --volume "$LAUNCHER:$LAUNCHER:ro" \
  --volume "$SUITE_VIEW:$SUITE_VIEW:ro" \
  --volume "$CAP:$CAP:rw" \
  --entrypoint python3 "$IMAGE" \
  "$TRELLIS_SCRIPT" capture \
  --model "$BASE_IN_CTR" --pre-encoded "$PREENC" \
  --suite "$SUITE_VIEW" --out "$CAP" \
  --quantization auto --kv-cache-dtype fp8_e4m3 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.945 \
  --filter all --max-batched-tokens 2048 --hash-shards
CAP_EXIT=$?
echo "capture exit: $CAP_EXIT"

# Exit 139 is vLLM segfault on shutdown — captures are still on disk
if [ $CAP_EXIT -ne 0 ] && [ $CAP_EXIT -ne 139 ]; then
  systemctl --user unmask qwen38-27b.service
  systemctl --user start qwen38-27b.service
  echo "service restored (capture failed)"
  exit 1
fi

# --- Step 3: Replay KLD ---
echo "=== Step 3: Replaying KLD ==="
wait_gpu_free
REPORT=$CAP/balanced-k${BITS}-replay-report.json
podman run --rm --name "balanced-replay-k${BITS}-$$" \
  --device nvidia.com/gpu=all --ipc=host --network none \
  --volume "$LAUNCHER:$LAUNCHER:ro" \
  --volume "$REF:$REF:ro" \
  --volume "$CAP:$CAP:rw" \
  --volume "$HEAD:$HEAD:ro" \
  --volume "$SUITE_VIEW:$SUITE_VIEW:ro" \
  --volume /home/mbelleau/.cache/jit:/cache/jit:rw \
  --entrypoint python3 "$IMAGE" "$FIDELITY" replay \
  --reference "$REF" --candidate "$CAP" \
  --head "$HEAD" --suite "$SUITE_VIEW" \
  --out "$REPORT" --filter all
REPLAY_EXIT=$?
echo "replay exit: $REPLAY_EXIT"

# Restore service
systemctl --user unmask qwen38-27b.service
systemctl --user start qwen38-27b.service
echo "service restored (unmasked + started)"

if [ -f "$REPORT" ]; then
  echo "=== Balanced + K${BITS} Embedding KLD ==="
  python3 -c "
import json
r = json.load(open('$REPORT'))
for k in ('mean_kld','token_mean_kld','context_macro_mean_kld','p95_kld','p99_kld','p999_kld','max_kld','top1_agreement'):
    if k in r: print(f'  {k}: {r[k]}')
"
fi

exit $REPLAY_EXIT
