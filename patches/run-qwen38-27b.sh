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
EXL3_PATCH_SHA256="dcede1b494984b3ec29fae5187e8aa692557e4658a1601c7dc0fc337737cbaa8"
EXL3_PATCH_CTR="/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py"
[ -f "${EXL3_PATCH_HOST}" ] || {
  echo "EXL3 graph patch is missing: ${EXL3_PATCH_HOST}" >&2
  exit 4
}
printf '%s  %s\n' "${EXL3_PATCH_SHA256}" "${EXL3_PATCH_HOST}" | sha256sum -c -

QUANTIZATION_CONFIG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*mtp\\..*","lm_head"]}'

# Qualified production profile for K5K6 hydrated on RTX 5090 (32 GB).
#
# max-num-batched-tokens 3072 (was 8192) — MEASURED 2026-08-19, this is a
# correctness/robustness setting, not a tuning knob:
#   At mnbt 8192 the engine took an unrecoverable CUDA OOM (EngineDeadError)
#   inside the GDN linear-attention prefill (fla chunk_o.py: o =
#   torch.empty_like(v)) on any prompt >~4k tokens: 4001 tok OK, 6000 tok
#   killed the engine. vLLM's profiled "peak activation" (2.84 GiB @8192)
#   under-predicts the real GDN-hybrid peak, leaving ~40-100 MiB of true
#   headroom, and a large prefill raises the allocator high-water mark so a
#   LATER large prefill dies even after an identical one succeeded.
#   Lowering the chunk budget bounds peak activation (2.32 GiB) AND shrinks
#   the CUDA-graph pool (0.89 -> 0.52 GiB), which together freed 0.93 GiB of
#   KV cache — so this buys full native context as a side effect.
#   3072 (not 2048) keeps the 2051-token PP bench inside ONE chunk; at 2048 it
#   spills into a second engine step and PP drops 6460 -> 4590 (the per-step
#   fixed overhead is ~130-142 ms, paid per chunk).
#   Evidence: receipts/robustness-context-2026-08-19.md (mixed stress gate:
#   24k prefill + vision + decode interleaved = ROBUST).
# max-model-len 262144 — full native context, unlocked by the KV freed above
#   (9.82 GiB KV vs 8.89 GiB at mnbt 8192). Was 238,400.
# PROFILE selects one of the two measured serving profiles. Full numbers and
# the proof that no single profile reaches all six north-star criteria are in
# receipts/frontier-2026-08-19.md; both are n=3-boot harness results.
#
#   throughput (default)  all-FP4.      PP 7665.6+/-20.4  TG fox 184.8+/-1.0
#                         essay 93.3    KLD 0.063759  p99 0.7010  ctx 250,000
#                         -> 4/6 criteria. The only profile with PP >= 7000.
#   fidelity              all-trellis.  PP 2987.7+/-4.4   TG fox 228.3+/-0.4
#                         essay 104.1   KLD 0.003405  p99 0.03489 ctx 238,400
#   balanced              gate_up FP6.  PP 3923.0         TG fox 206.0
#                         essay 96.3    KLD 0.005672  p99 0.05991 ctx 199,104
#                         -> also 5/6, but fails ctx instead of PP: 1.7x the
#                         prefill of `fidelity` and the best TG-essay measured
#                         anywhere (96.3, MTP acceptance 0.324), at KLD still
#                         2.1x inside the 0.012 budget.  Pick this when 199k
#                         context is enough.
#                         -> 5/6 criteria; fails only PP. KLD is within 26% of
#                         the checkpoint's own published 0.002700, and it beats
#                         the committed baseline 16.6x on KLD mean and 18.3x on
#                         p99 while giving +30% TG-fox. Prefill-heavy workloads
#                         still pay ~4.1x, which is the whole tradeoff.
#
# Explicit env always wins over the profile defaults (`:=` below).
PROFILE="${PROFILE:-throughput}"
case "${PROFILE}" in
  throughput)
    : "${VLLM_EXL3_FP4_LAYERS:=mlp.gate_up_proj,mlp.down_proj,linear_attn.,self_attn.}"
    : "${VLLM_EXL3_PREFILL_RECONSTRUCT_M:=1}"
    : "${VLLM_EXL3_SKIP_TRELLIS_PREP:=0}"
    # 2026-08-19: 250,000 now fails boot deterministically -- "9.26 GiB KV
    # needed ... 9.25 GiB available; estimated maximum model length 249696".
    # ~10 MiB of load/profile-time memory moved in the post-gates600 window
    # (GDN spec-row gate un-slicing is the prime suspect; it fixes a LIVE
    # correctness bug under MTP+concurrency, so context yields, not the fix).
    # Criterion is >= 238,400; 249,600 keeps margin below the 249,696 estimate.
    : "${MAX_MODEL_LEN:=249600}"
    ;;
  fidelity)
    # A single comma means "no FP4 layers" (empty would fall back to defaults).
    : "${VLLM_EXL3_FP4_LAYERS:=,}"
    # Reconstruct+fold then cuBLAS hgemm for the K5 mlp.gate_proj/up_proj that
    # B12X will not take.  Two bounds make it fit where it previously OOM'd:
    # MAX_MB excludes the lm_head (5120 x 248320 x 2 = 2.37 GiB in one
    # allocation), and CACHE=0 stops the FP16 weight cache from being populated
    # during vLLM's profiling forward, which otherwise leaves zero KV.
    # Worth +14.1% PP for +0.9% KLD (0.003407 -> 0.003437, CIs overlapping).
    : "${VLLM_EXL3_PREFILL_RECONSTRUCT_M:=1}"
    : "${VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB:=512}"
    : "${VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE:=0}"
    # The fold is now on the per-chunk hot path, so its FP32 chunk size is a
    # throughput knob.  Measured curve (PP, 2051-tok bench): 24->1955, 32->1960,
    # 48->1964, 64->1865, 96->1858, 192->1790, 384->1711 - a sharp cliff between
    # 48 and 64.  48 MB is the peak, +5.7% over the 96 MB default.  The fold
    # operates on independent 128x128 blocks so every budget is bit-identical
    # (verified: max_abs_diff 0.000e+00), making this a free win.
    : "${VLLM_EXL3_FOLD_FP32_BUDGET_MB:=48}"
    # Route K6 by row count: B12X for prefill, fused exl3_gemm for decode.
    # B12X owns prefill (PP 1504 -> 1967) but the fused kernel decodes better and
    # drafts better -- TG-essay 93.1+/-0.1 vs 90.0+/-0.2 and MTP acceptance
    # 0.304 vs 0.281, at unchanged PP (1965.4+/-2.1 vs 1966.8+/-8.3).  Costs
    # TG-fox 207.7 vs 211.1, which still clears its 190 threshold by 9.3%.
    # Set VLLM_EXL3_B12X_MIN_M=0 to prefer B12X at every row count instead.
    : "${VLLM_EXL3_B12X_MIN_M:=128}"
    # K5/K4 payloads through B12X as well.  Was broken until the b12x warm was
    # keyed per (device, bits) - lazily-initialised bit-width plans were taking
    # buffers from a CUDA graph's private pool.  Verified: 0 selftest mismatches,
    # KLD 0.003405 (vs 0.003437 without), n=3 PP 2987.7 (+52.0%), TG-fox 228.3,
    # TG-essay 104.1.  receipts/b12x-k5-cured-2026-08-19.md
    : "${VLLM_EXL3_B12X_ANY_BITS:=1}"
    # B12X W4A16 consumes the same packed trellis payload as the fused
    # exl3_gemm kernel but costs 0.30 ms CPU per call instead of 4.72 ms, and a
    # full 512-context KLD run proves it is fidelity-neutral (0.003407 vs
    # 0.003412, inside CI).  Worth +51% PP.  It needs slightly more headroom
    # because its load-time prep costs ~0.89 GiB of persistent memory.
    : "${VLLM_EXL3_SKIP_TRELLIS_PREP:=0}"
    : "${GPU_MEMORY_UTILIZATION:=0.945}"
    : "${MAX_MODEL_LEN:=238400}"
    ;;
  balanced)
    # Same trellis kernel stack as `fidelity`, but mlp.gate_up_proj is converted
    # to MXFP6 so the largest matrices become GEMM-resident.  Costs ~2.4 GiB more
    # than their trellis form, which is why the context ceiling is 199,104 rather
    # than 238,400 - that is the whole trade.
    : "${VLLM_EXL3_FP4_LAYERS:=,}"
    : "${VLLM_EXL3_FP6_LAYERS:=mlp.gate_up_proj}"
    : "${VLLM_EXL3_PREFILL_RECONSTRUCT_M:=1}"
    : "${VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB:=512}"
    : "${VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE:=0}"
    : "${VLLM_EXL3_FOLD_FP32_BUDGET_MB:=48}"
    : "${VLLM_EXL3_B12X_MIN_M:=128}"
    # Same cure as fidelity: K5/K4 through B12X. Measured +20.7% PP here
    # (3250.6 -> 3923.0), fox 206.0 [acc 1.000], vision OK.
    : "${VLLM_EXL3_B12X_ANY_BITS:=1}"
    : "${VLLM_EXL3_SKIP_TRELLIS_PREP:=0}"
    : "${GPU_MEMORY_UTILIZATION:=0.945}"
    : "${MAX_MODEL_LEN:=199104}"
    ;;
  *)
    echo "Unknown PROFILE='${PROFILE}' (expected: throughput | fidelity | balanced)" >&2
    exit 4
    ;;
esac
export VLLM_EXL3_FP4_LAYERS VLLM_EXL3_PREFILL_RECONSTRUCT_M
export VLLM_EXL3_SKIP_TRELLIS_PREP MAX_MODEL_LEN

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-250000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# NO_PATCH_MOUNTS=1 skips every patch bind-mount - for verifying a BAKED image
# (docker/Containerfile) that already contains them. With the stock base image
# this flag serves the UNPATCHED fork and will fail the gates; that is the point.
NO_PATCH_MOUNTS="${NO_PATCH_MOUNTS:-0}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3072}"
# fp8_e4m3 is the measured default; turboquant_k8v4 etc. are the fork-shipped
# sub-8-bit formats (own Triton backend, supports_spec_as_decode=False - measure
# MTP behaviour before trusting TG numbers on them).
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"

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
# A caller-provided SPEC_CONFIG (e.g. method dspark with an external draft)
# wins over the MTP default; MTP=0 alone still disables speculation entirely.
if [ -n "${SPEC_CONFIG:-}" ]; then
  :
elif [ "${MTP}" != "0" ]; then
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
# nsys needs perf-counter access for CUDA tracing inside the container.
if [ "${NSYS_PROFILE:-0}" = "1" ]; then
  RUN_ARGS+=(--cap-add=SYS_ADMIN)
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
  -e LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}" \
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
  -e VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}" \
  -e VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}" \
  -e DO_NOT_TRACK="${DO_NOT_TRACK:-1}" \
  -e VLLM_EXL3_SKIP_TRELLIS_PREP="${VLLM_EXL3_SKIP_TRELLIS_PREP:-0}" \
  -e VLLM_EXL3_PREFILL_RECONSTRUCT_M="${VLLM_EXL3_PREFILL_RECONSTRUCT_M:-1}" \
  -e PROFILER_CONFIG="${PROFILER_CONFIG:-}" \
  -e NSYS_PROFILE="${NSYS_PROFILE:-0}" -e NSYS_TAG="${NSYS_TAG:-run}" \
  -e VLLM_NVTX_SCOPES_FOR_PROFILING="${VLLM_NVTX_SCOPES_FOR_PROFILING:-1}" \
  -e VLLM_EXL3_FP4_LAYERS="${VLLM_EXL3_FP4_LAYERS:-mlp.gate_up_proj,mlp.down_proj,linear_attn.,self_attn.}" \
  -e VLLM_EXL3_FP6_LAYERS="${VLLM_EXL3_FP6_LAYERS:-}" \
  -e VLLM_EXL3_FP4_LAYER_RANGE="${VLLM_EXL3_FP4_LAYER_RANGE:-}" \
  -e VLLM_EXL3_FP6_LAYER_RANGE="${VLLM_EXL3_FP6_LAYER_RANGE:-}" \
  -e VLLM_EXL3_B12X_ANY_BITS="${VLLM_EXL3_B12X_ANY_BITS:-0}" \
  -e VLLM_EXL3_B12X_SELFTEST="${VLLM_EXL3_B12X_SELFTEST:-0}" \
  -e VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB="${VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB:-4096}" \
  -e VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE="${VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE:-1}" \
  -e VLLM_EXL3_B12X_MIN_M="${VLLM_EXL3_B12X_MIN_M:-0}" \
  -e VLLM_EXL3_FOLD_FP32_BUDGET_MB="${VLLM_EXL3_FOLD_FP32_BUDGET_MB:-96}" \
  -e VLLM_EXL3_B12X_N_RANGE="${VLLM_EXL3_B12X_N_RANGE:-5120-36864}" \
  -e VLLM_EXL3_EXT_PATH=/opt/exllamav3 \
  -e VLLM_EXL3_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3 \
  -e VLLM_EXL3_ENCODER_REVISION=704aefd743b390af4bd0fb429d1906f9b964c7d8 \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${HF_CACHE_HOST}":/root/.cache/huggingface:ro \
  $([ "${NO_PATCH_MOUNTS:-0}" = "1" ] || echo "-v ${EXL3_PATCH_HOST}:${EXL3_PATCH_CTR}:ro") \
  $([ "${NO_PATCH_MOUNTS:-0}" = "1" ] || echo "-v /home/mbelleau/scheduler_patch.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro") \
  $([ "${NO_PATCH_MOUNTS:-0}" = "1" ] || echo "-v /home/mbelleau/qwen_gdn_linear_attn_patch.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro") \
  $([ "${NO_PATCH_MOUNTS:-0}" = "1" ] || echo "-v /home/mbelleau/spec_decode_utils_patch.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/utils.py:ro") \
  $([ "${NO_PATCH_MOUNTS:-0}" = "1" ] || echo "-v /home/mbelleau/autoregressive_speculator_patch.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:ro") \
  $([ "${NO_PATCH_MOUNTS:-0}" = "1" ] || echo "-v /home/mbelleau/qwen3_5_mtp_patch.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py:ro") \
  -v /home/mbelleau/.cache/jit:/cache/jit \
  $([ "${NO_PATCH_MOUNTS:-0}" = "1" ] || echo "-v /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp4_conversion.py:/opt/fp4/exl3_fp4_conversion.py:ro") \
  $([ "${NO_PATCH_MOUNTS:-0}" = "1" ] || echo "-v /home/mbelleau/qwen38-27b-exl3/patches/triton_fp4_quant.py:/opt/fp4/triton_fp4_quant.py:ro") \
  $([ "${NO_PATCH_MOUNTS:-0}" = "1" ] || echo "-v /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp6_conversion.py:/opt/fp6/exl3_fp6_conversion.py:ro") \
  $([ "${NO_PATCH_MOUNTS:-0}" = "1" ] || echo "-v /home/mbelleau/vllm-exl3-linear-ba.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/linear.py:ro") \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "set -euo pipefail; cd /; \
       ln -sf /usr/local/cuda-13.2/targets/x86_64-linux/lib/* /usr/local/cuda-13.2/lib64/ 2>/dev/null || true; rm -f /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__pycache__/exl3*.pyc /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/__pycache__/linear*.pyc; \
       SPEC_ARGS=(); if [ -n \"\${SPEC_CONFIG:-}\" ]; then SPEC_ARGS=(--speculative-config \"\${SPEC_CONFIG}\"); fi; \
       LMO_ARGS=(); if [ \"\${LANGUAGE_MODEL_ONLY:-0}\" = \"1\" ]; then LMO_ARGS=(--language-model-only); fi; \
       PROF_ARGS=(); if [ -n \"\${PROFILER_CONFIG:-}\" ]; then PROF_ARGS=(--profiler-config \"\${PROFILER_CONFIG}\"); fi; \
       NSYS=(); if [ \"\${NSYS_PROFILE:-0}\" = \"1\" ]; then mkdir -p /cache/jit/nsys; \
         NSYS=(nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
               --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
               -o /cache/jit/nsys/\${NSYS_TAG:-run}); fi; \
       exec \"\${NSYS[@]}\" vllm serve '${MODEL_IN_CTR}' \
         --served-model-name '${SERVED_MODEL_NAME}' --trust-remote-code \
         --host 0.0.0.0 --port '${PORT}' \
         --quantization exl3 \
         \"\${LMO_ARGS[@]}\" \
         --quantization-config \"\${QUANTIZATION_CONFIG}\" \
         --attention-backend '${ATTN_BACKEND}' \
         \
         --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
        --kv-cache-dtype '${KV_CACHE_DTYPE}' \
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
         \"\${SPEC_ARGS[@]}\" \"\${PROF_ARGS[@]}\""

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
