#!/bin/bash
set -uo pipefail
LAUNCHER=/home/mbelleau/worktrees/r30-capture-smoke/r30-launcher
CAND=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-candidate
EVID=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-evidence
CAP=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-capture
SUITE_VIEW=$EVID/capture-suite-view
FIDELITY=$LAUNCHER/tools/fidelity.py
IMAGE="docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
VLLM=/opt/venv/lib/python3.12/site-packages/vllm

mkdir -p "$CAP"

systemctl --user stop qwen38-27b.service 2>/dev/null
sleep 3
echo "service stopped"

podman run --rm --name "wave5-fp8kv-$$" \
  --device nvidia.com/gpu=all --ipc=host --network none \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_EXL3_MULTIPRECISION=0 \
  --env VLLM_EXL3_EMBED_ONLINE_BITS=6 \
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
  --volume "$CAND:$CAND:ro" \
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
  --entrypoint python3 "$IMAGE" "$FIDELITY" capture \
  --model "$CAND" --suite "$SUITE_VIEW" --out "$CAP" \
  --quantization auto --kv-cache-dtype fp8_e4m3 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.90 \
  --filter all --max-batched-tokens 2048 --hash-shards
CAPTURE_EXIT=$?
echo "capture exit: $CAPTURE_EXIT"

systemctl --user start qwen38-27b.service
echo "service restored"

exit $CAPTURE_EXIT
