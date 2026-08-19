#!/bin/bash
# Qwen3.8-27B K5K6 hydrated (serialized EXL3 K6 attention, vision-enabled) dedicated service for AIBoss.
# The base image, model revision, and PR #314 graph patch are qualification-pinned.
set -euo pipefail

IMAGE="${IMAGE:-docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b}"
NAME="${NAME:-qwen38-27b}"
PORT="${PORT:-8000}"

# Model source: local HF hub cache (no NFS dependency).
# The hydrated build serializes attention at K6 on disk — no online encode pass,
# 21.61 GB download, 178 s cold start, best fidelity in the family (v5 KLD 0.002760).
HF_CACHE_HOST="${HF_CACHE_HOST:-/home/mbelleau/.cache/huggingface/hub}"
MODEL_REPO="${MODEL_REPO:-malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated}"
MODEL_REVISION="${MODEL_REVISION:-ab3a91a13813df8096cb4c1d560ed3669035d0cf}"
REPO_CACHE_DIR="${HF_CACHE_HOST}/models--${MODEL_REPO//\//--}"
MODEL_SNAPSHOT_DIR="${REPO_CACHE_DIR}/snapshots/${MODEL_REVISION}"
[ -d "${MODEL_SNAPSHOT_DIR}" ] || {
  echo "Model snapshot is missing: ${MODEL_SNAPSHOT_DIR}" >&2
  exit 4
}
MODEL_IN_CTR="/root/.cache/huggingface/models--${MODEL_REPO//\//--}/snapshots/${MODEL_REVISION}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"

# PR #314 is mounted over the pinned GG r34 base image. Refuse to start if the
# installed overlay is absent or differs from the exact qualified source.
EXL3_PATCH_HOST="${EXL3_PATCH_HOST:-/home/mbelleau/vllm-exl3-multiprecision.py}"
EXL3_PATCH_SHA256="f212bf43bbd69d3a6aaadbc01f8eea986d95514047e91838cdc311e099266573"
EXL3_PATCH_CTR="/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py"
[ -f "${EXL3_PATCH_HOST}" ] || {
  echo "EXL3 graph patch is missing: ${EXL3_PATCH_HOST}" >&2
  exit 4
}
printf '%s  %s\n' "${EXL3_PATCH_SHA256}" "${EXL3_PATCH_HOST}" | sha256sum -c -

QUANTIZATION_CONFIG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*mtp\\..*","lm_head"]}'

# Qualified production profile for K5K6 hydrated on RTX 5090 (32 GB):
#   - gpu-memory-utilization 0.955: measured per-card value from physical 5090
#     qualification (docs/34, receipts/qualification-5090-context.json).
#   - max-model-len 180000: the hydrated build's demonstrated context with MTP-3
#     on a 5090 (docs/29). The sibling online K5K6 reaches ~185,600; the context
#     edition reaches 262,144 at K5 attention.
#   - max-num-batched-tokens 8192: vLLM warns that 4096 is suboptimal with
#     MTP-3 (draft token slots need headroom); 8192 accommodates the 3 draft
#     tokens per sequence across 8 concurrent seqs.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-238400}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# Graph-captured prefill has produced Xid 31 on short sm_120 prefills.
# FULL_DECODE_ONLY keeps prefill out of graphs. The pinned PR primes every
# finite dense decode row before capture; the opt-in remains explicit.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
VLLM_EXL3_GRAPH_DECODE="${VLLM_EXL3_GRAPH_DECODE:-1}"

# FlashInfer full attention previously faulted on short bf16-Q/fp8-KV prefills.
ATTN_BACKEND="${ATTN_BACKEND:-TRITON_ATTN}"

# Transformers 5.15 initializes Qwen2Tokenizer with max_length=2048. Without
# disabling processor-side truncation, images needing more than 2048 vision
# tokens fail before vLLM can enforce the real model context limit.
if [ -z "${MM_PROCESSOR_KWARGS:-}" ]; then
  MM_PROCESSOR_KWARGS='{"truncation":false}'
fi

# MTP-3: 3 speculative draft tokens. The hydrated build's quantized MTP draft
# head (attn K4, MLP K5/K6) delivers 113.8 tok/s single-stream (2x FP8) with
# 58.2% acceptance and 2.745 mean acceptance length (docs/22). On the live
# container, measured 77.8% acceptance at 3.33 mean length.
MTP="${MTP:-6}"
SPECULATIVE_TOKENS="${SPECULATIVE_TOKENS:-6}"
if [ "${MTP}" != "0" ]; then
  SPEC_CONFIG="$(printf '{"method":"mtp","num_speculative_tokens":%s}' "${SPECULATIVE_TOKENS}")"
else
  SPEC_CONFIG=""
fi

# One model owns this single GPU. Refuse to hide a conflicting workload.
for other in qwen36-27b vllm-sanity sglang-gemma4-31b llama-gemma4-31b qwen36-voip-vllm-bench; do
  if podman inspect "${other}" >/dev/null 2>&1; then
    echo "Refusing to start: conflicting container '${other}' exists" >&2
    exit 3
  fi
done

printf 'Starting Qwen3.8-27B K5K6-hydrated: revision=%s context=%s sequences=%s MTP=%s batched=%s graph=%s\n' \
  "${MODEL_REVISION:0:12}" "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}" "${MTP}" \
  "${MAX_NUM_BATCHED_TOKENS}" "${VLLM_EXL3_GRAPH_DECODE}"
podman rm -f "${NAME}" >/dev/null 2>&1 || true

# Under systemd, conmon supplies readiness and podman remains in the foreground.
if [ "${FOREGROUND:-0}" = "1" ]; then
  RUN_ARGS=(--sdnotify=conmon --label PODMAN_SYSTEMD_UNIT=qwen38-27b.service)
else
  RUN_ARGS=(-d)
fi

podman run "${RUN_ARGS[@]}" --replace \
  --name "${NAME}" \
  --tmpfs /usr/local/cuda-13.2/lib64:rw,size=16m \
  --device nvidia.com/gpu=all --ipc=host --network host \
  --health-cmd "curl -sf http://localhost:${PORT}/health || exit 1" \
  --health-interval 30s --health-timeout 10s --health-retries 3 \
  --health-start-period 45m \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e OMP_NUM_THREADS=8 -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_120a -e FLASHINFER_CUDA_ARCH_LIST=12.0f \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e XDG_CACHE_HOME=/cache/jit -e CUDA_CACHE_PATH=/cache/jit \
  -e TRITON_CACHE_DIR=/cache/jit/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e QUANTIZATION_CONFIG="${QUANTIZATION_CONFIG}" \
  -e MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS}" \
  -e SPEC_CONFIG="${SPEC_CONFIG}" \
  -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
  -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/jit/exl3-online \
  -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite \
  \
  -e VLLM_EXL3_MULTIPRECISION=1 \
  -e VLLM_EXL3_FP4_TRITON_DECODE=0 \
  -e VLLM_EXL3_GRAPH_DECODE="${VLLM_EXL3_GRAPH_DECODE}" \
  -e VLLM_EXL3_EMBED_ONLINE_BITS=6 \
  -e B12X_PACKED_B_MIN_N=1024 \
  -e VLLM_EXL3_FP4_PER_ROW_GS=0 \
  -e VLLM_EXL3_FP4_DRAFT_HEAD=0 \
  -e VLLM_EXL3_FP4_BANDED_SELFTEST=0 \
  -e VLLM_EXL3_FP8DG_PREFILL_M="${VLLM_EXL3_FP8DG_PREFILL_M:-0}" \
  -e VLLM_EXL3_FP8DG_SELFTEST="${VLLM_EXL3_FP8DG_SELFTEST:-0}" \
  -e VLLM_EXL3_FP8DG_CACHE="${VLLM_EXL3_FP8DG_CACHE:-0}" \
  -e VLLM_EXL3_FP4_LAYERS="${VLLM_EXL3_FP4_LAYERS:-mlp.gate_up_proj,mlp.down_proj,linear_attn.}" \
  -e VLLM_EXL3_FP6_LAYERS="${VLLM_EXL3_FP6_LAYERS:-}" \
    -e VLLM_EXL3_B12X_N_RANGE="${VLLM_EXL3_B12X_N_RANGE:-5120-36864}" \
  -e VLLM_EXL3_EXT_PATH=/opt/exllamav3 \
  -e VLLM_EXL3_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3 \
  -e VLLM_EXL3_ENCODER_REVISION=704aefd743b390af4bd0fb429d1906f9b964c7d8 \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${HF_CACHE_HOST}":/root/.cache/huggingface:ro \
  -v "${EXL3_PATCH_HOST}":"${EXL3_PATCH_CTR}":ro \
  -v /home/mbelleau/scheduler_patch.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro \
  -v /home/mbelleau/qwen3_5_mtp_patch.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py:ro \
  -v /home/mbelleau/.cache/jit:/cache/jit \
  -v /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp4_conversion.py:/opt/fp4/exl3_fp4_conversion.py:ro \
  -v /home/mbelleau/qwen38-27b-exl3/patches/triton_fp4_quant.py:/opt/fp4/triton_fp4_quant.py:ro \
  -v /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp6_conversion.py:/opt/fp6/exl3_fp6_conversion.py:ro \
  -v /home/mbelleau/vllm-exl3-linear-ba.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/linear.py:ro \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "set -euo pipefail; cd /; \
       ln -sf /usr/local/cuda-13.2/targets/x86_64-linux/lib/* /usr/local/cuda-13.2/lib64/ 2>/dev/null || true; rm -f /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__pycache__/exl3*.pyc /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/__pycache__/linear*.pyc; \
       SPEC_ARGS=(); if [ -n \"\${SPEC_CONFIG:-}\" ]; then SPEC_ARGS=(--speculative-config \"\${SPEC_CONFIG}\"); fi; \
       exec vllm serve '${MODEL_IN_CTR}' \
         --served-model-name '${SERVED_MODEL_NAME}' --trust-remote-code \
         --host 0.0.0.0 --port '${PORT}' \
         --quantization exl3 \
         --quantization-config \"\${QUANTIZATION_CONFIG}\" \
         --attention-backend '${ATTN_BACKEND}' \
         \
         --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
        --kv-cache-dtype fp8_e4m3 \
         --max-model-len '${MAX_MODEL_LEN}' \
        --max-num-seqs '${MAX_NUM_SEQS}' \
        --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}' \
        --compilation-config '{\"mode\":\"NONE\",\"cudagraph_mode\":\"'${CUDAGRAPH_MODE}'\"}' \
        --mm-processor-kwargs \"\${MM_PROCESSOR_KWARGS}\" \
        --mm-processor-cache-type shm \
        --default-chat-template-kwargs '{\"preserve_thinking\": true}' \
        --enable-chunked-prefill \
         --reasoning-parser qwen3 \
         --enable-auto-tool-choice --tool-call-parser qwen3_coder \
         \"\${SPEC_ARGS[@]}\""

[ "${FOREGROUND:-0}" = "1" ] && exit 0

printf 'Container launched; waiting for health on port %s\n' "${PORT}"
for _ in {1..360}; do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    printf 'Qwen3.8-27B is healthy on port %s\n' "${PORT}"
    podman logs "${NAME}" 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Available KV cache' | tail -2
    exit 0
  fi
  state="$(podman inspect -f '{{.State.Status}}' "${NAME}" 2>/dev/null || echo missing)"
  if [ "${state}" != "running" ]; then
    echo "Container died with state=${state}" >&2
    podman logs --tail 60 "${NAME}" >&2
    exit 2
  fi
  sleep 5
done

echo "Container is still running but did not become healthy within 30 minutes" >&2
exit 1
