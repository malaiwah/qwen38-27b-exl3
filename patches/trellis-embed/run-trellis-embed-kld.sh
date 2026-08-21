#!/bin/bash
set -uo pipefail
LAUNCHER=/home/mbelleau/worktrees/r30-capture-smoke/r30-launcher
FIDELITY=$LAUNCHER/tools/fidelity.py
TRELLIS_SCRIPT=/home/mbelleau/qwen38-27b-exl3/patches/trellis-embed/trellis_embed_kld.py
BASE=/home/mbelleau/.cache/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf
SUITE_VIEW=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-evidence/capture-suite-view
CAP=/home/mbelleau/worktrees/r30-capture-smoke/trellis-k6-capture
REF=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/reference/hidden-bf16
HEAD=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/lm-head/weight.safetensors
IMAGE="docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
VLLM=/opt/venv/lib/python3.12/site-packages/vllm
BITS=${1:-6}

rm -rf "$CAP" 2>/dev/null
mkdir -p "$CAP"

systemctl --user stop qwen38-27b.service
sleep 3
echo "service stopped"

# Capture with trellis K{BITS} embedding round-trip
podman run --rm --name "trellis-k${BITS}-$$" \
  --device nvidia.com/gpu=all --ipc=host --network none \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_EXL3_MULTIPRECISION=0 \
  --env VLLM_USE_V2_MODEL_RUNNER=1 \
  --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  --env TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  --env PYTHONPATH=/opt/r30-ext \
  --env VLLM_EXL3_PRE_RECONSTRUCT=0 \
  --env VLLM_EXL3_B12X_ANY_BITS=1 \
  --env VLLM_EXL3_B12X_MIN_M=128 \
  --env VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE=0 \
  --env LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  --volume "$LAUNCHER:$LAUNCHER:ro" \
  --volume "$BASE:$BASE:ro" \
  --volume "$SUITE_VIEW:$SUITE_VIEW:ro" \
  --volume "$CAP:$CAP:rw" \
  --volume /home/mbelleau/.cache/jit:/cache/jit:rw \
  --volume /home/mbelleau/vllm-exl3-multiprecision.py:$VLLM/model_executor/layers/quantization/exl3.py:ro \
  --volume /home/mbelleau/scheduler_patch.py:$VLLM/v1/core/sched/scheduler.py:ro \
  --volume /home/mbelleau/qwen_gdn_linear_attn_patch.py:$VLLM/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro \
  --volume /home/mbelleau/spec_decode_utils_patch.py:$VLLM/v1/worker/gpu/spec_decode/utils.py:ro \
  --volume /home/mbelleau/autoregressive_speculator_patch.py:$VLLM/v1/worker/gpu/spec_decode/autoregressive/speculator.py:ro \
  --volume /home/mbelleau/qwen3_5_mtp_patch.py:$VLLM/model_executor/models/qwen3_5_mtp.py:ro \
  --volume /home/mbelleau/vllm-exl3-linear-ba.py:$VLLM/model_executor/layers/linear.py:ro \
  --volume /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp4_conversion.py:/opt/fp4/exl3_fp4_conversion.py:ro \
  --volume /home/mbelleau/qwen38-27b-exl3/patches/triton_fp4_quant.py:/opt/fp4/triton_fp4_quant.py:ro \
  --volume /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp6_conversion.py:/opt/fp6/exl3_fp6_conversion.py:ro \
  --volume /home/mbelleau/worktrees/r30-torch-ext/exllamav3_ext/exllamav3_ext.so:/opt/r30-ext/exllamav3_ext.so:ro \
  --entrypoint python3 "$IMAGE" "$TRELLIS_SCRIPT" capture \
  --model "$BASE" --bits "$BITS" --suite "$SUITE_VIEW" --out "$CAP" \
  --quantization auto --kv-cache-dtype bfloat16 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.90 \
  --filter all --max-batched-tokens 2048 --hash-shards
CAPTURE_EXIT=$?
echo "capture exit: $CAPTURE_EXIT"

if [ $CAPTURE_EXIT -ne 0 ]; then
  systemctl --user start qwen38-27b.service
  echo "service restored (capture failed)"
  exit 1
fi

# Replay
REPORT=$CAP/trellis-k${BITS}-replay-report.json
podman run --rm --name "trellis-k${BITS}-replay-$$" \
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

systemctl --user start qwen38-27b.service
echo "service restored"

if [ -f "$REPORT" ]; then
  echo "=== Trellis K${BITS} Embedding KLD ==="
  python3 -c "
import json
r = json.load(open('$REPORT'))
for k in ('mean_kld','token_mean_kld','context_macro_mean_kld','p95_kld','p99_kld','p999_kld','max_kld','top1_agreement'):
    if k in r: print(f'  {k}: {r[k]}')
"
fi

exit $REPLAY_EXIT
