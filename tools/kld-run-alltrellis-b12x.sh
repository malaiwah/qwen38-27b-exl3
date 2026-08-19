#!/bin/bash
# Per-layer KLD attribution: which FP4 group (gate_up / down / GDN in+out)
# contributes what to the flagship KLD=0.056732 (all with int6 online embeds).
# Resume-safe: skips captures that already have 513 files; the engine's
# shutdown-time abort (exit after all contexts are written) is tolerated.
set -uo pipefail

IMAGE="docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
MODEL_IN_CTR="/root/.cache/huggingface/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf"
QUANTIZATION_CONFIG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*mtp\\..*","lm_head"]}'

podman_kld() {
  local name="$1" patterns="$2" body="$3"
  podman run --rm --name "$name" \
    --tmpfs /usr/local/cuda-13.2/lib64:rw,size=16m \
    --device nvidia.com/gpu=all --ipc=host --network host \
    -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=8 \
    -e CUTE_DSL_ARCH=sm_120a -e FLASHINFER_CUDA_ARCH_LIST=12.0f \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    -e QCFG="$QUANTIZATION_CONFIG" \
    -e XDG_CACHE_HOME=/cache/jit -e CUDA_CACHE_PATH=/cache/jit \
    -e TRITON_CACHE_DIR=/cache/jit/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor \
    -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
    -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
    -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/jit/exl3-online \
    -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite \
    -e VLLM_EXL3_MULTIPRECISION=1 \
    -e VLLM_EXL3_FP4_TRITON_DECODE=0 \
    -e VLLM_EXL3_EMBED_ONLINE_BITS=6 \
    -e B12X_PACKED_B_MIN_N=1024 \
    -e VLLM_EXL3_FP4_PER_ROW_GS=0 \
    -e VLLM_EXL3_FP4_DRAFT_HEAD=0 \
    -e VLLM_EXL3_FP4_BANDED_SELFTEST=0 \
    -e VLLM_EXL3_FP8DG_PREFILL_M=0 \
    -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=0 \
    -e VLLM_EXL3_SKIP_TRELLIS_PREP=0 \
    -e VLLM_EXL3_B12X_N_RANGE="5120-36864" \
    -e VLLM_EXL3_FP4_LAYERS="$patterns" \
    -e VLLM_EXL3_EXT_PATH=/opt/exllamav3 \
    -e VLLM_EXL3_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3 \
    -e VLLM_EXL3_ENCODER_REVISION=704aefd743b390af4bd0fb429d1906f9b964c7d8 \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONUNBUFFERED=1 \
    -v /home/mbelleau/.cache/huggingface/hub:/root/.cache/huggingface:ro \
    -v /home/mbelleau/vllm-exl3-multiprecision.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py:ro \
    -v /home/mbelleau/scheduler_patch.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro \
    -v /home/mbelleau/qwen3_5_mtp_patch.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py:ro \
    -v /home/mbelleau/.cache/jit:/cache/jit \
    -v /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp4_conversion.py:/opt/fp4/exl3_fp4_conversion.py:ro \
    -v /home/mbelleau/qwen38-27b-exl3/patches/triton_fp4_quant.py:/opt/fp4/triton_fp4_quant.py:ro \
    -v /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp6_conversion.py:/opt/fp6/exl3_fp6_conversion.py:ro \
    -v /home/mbelleau/vllm-exl3-linear-ba.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/linear.py:ro \
    -v /tmp/kld-data:/kld-data \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -lc "set -uo pipefail; \
      ln -sf /usr/local/cuda-13.2/targets/x86_64-linux/lib/* /usr/local/cuda-13.2/lib64/ 2>/dev/null || true; \
      rm -f /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__pycache__/exl3*.pyc \
            /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/__pycache__/linear*.pyc; \
      $body"
}

run_one() {
  local tag="$1" patterns="$2"
  local capdir="/tmp/kld-data/captures/shard-0000/hidden-$tag"
  if [ "$(ls "$capdir" 2>/dev/null | wc -l)" -ge 513 ]; then
    echo "=== [$tag] capture exists, skipping ==="
  else
    echo "=== [$tag] capture (FP4 layers: '$patterns') ==="
    podman_kld "kld-cap-$tag" "$patterns" \
      "python /kld-data/fidelity.py capture \
        --model '$MODEL_IN_CTR' \
        --suite /kld-data/suite/shard-0000 \
        --out /kld-data/captures/shard-0000/hidden-$tag \
        --quantization exl3 \
        --quantization-config \"\$QCFG\" \
        --gpu-memory-utilization 0.85" || true
    if [ "$(ls "$capdir" 2>/dev/null | wc -l)" -lt 513 ]; then
      echo "=== [$tag] CAPTURE INCOMPLETE ($(ls "$capdir" 2>/dev/null | wc -l)/513) ==="
      return 1
    fi
  fi
  if [ -s "/tmp/kld-data/reports/report-$tag.json" ]; then
    echo "=== [$tag] report exists, skipping replay ==="
  else
    echo "=== [$tag] replay ==="
    podman_kld "kld-rep-$tag" "$patterns" \
      "python /kld-data/fidelity.py replay \
        --reference /kld-data/reference/hidden-bf16 \
        --candidate /kld-data/captures/shard-0000/hidden-$tag \
        --head /kld-data/lm-head/weight.safetensors \
        --suite /kld-data/suite/shard-0000 \
        --out /kld-data/reports/report-$tag.json" || true
  fi
  echo "=== [$tag] done ==="
}

systemctl --user stop qwen38-27b.service 2>/dev/null || true
sleep 3
podman rm -f qwen38-27b 2>/dev/null || true

run_one alltrellis-b12x ","

systemctl --user start qwen38-27b.service

echo "=== KLD attribution results ==="
for t in alltrellis-int6emb; do
  kld=$(jq -r '.mean_kld // .kld_mean // .context_macro_mean_kld' "/tmp/kld-data/reports/report-$t.json" 2>/dev/null || echo FAIL)
  echo "$t: KLD=$kld"
done
echo "refs: flagship(all-three)=0.056732  all-fp6=0.010699"
