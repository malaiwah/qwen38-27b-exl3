#!/bin/bash
# Recapture exact PROFILE=fidelity hidden states through the clean packed-INT6 runtime.
set -euo pipefail

[[ "${FRONTIER_TRANSACTION_ACTIVE:-}" == "1" ]] || {
  echo "frontier_fidelity_kld_callback: transaction is not active" >&2
  exit 2
}
for name in FRONTIER_CAMPAIGN_CONTAINER FRONTIER_CAMPAIGN_IMAGE \
  FRONTIER_CAMPAIGN_CACHE_DIR FRONTIER_CAMPAIGN_WORK_DIR; do
  [[ -n "${!name:-}" ]] || {
    echo "frontier_fidelity_kld_callback: missing ${name}" >&2
    exit 2
  }
done

KLD_ROOT=/tmp/kld-data
FIDELITY="$KLD_ROOT/fidelity.py"
MODEL_CACHE=/home/mbelleau/.cache/huggingface/hub
EXL3_SOURCE=/home/mbelleau/final-frontier-g02/source-patches/exl3.py
B12X=/home/mbelleau/final-frontier-g0/source-patches/b12x-base-1.2.1/b12x
FP4=/home/mbelleau/final-frontier-g0/source-patches/exl3_fp4_conversion.py
TRITON_FP4=/home/mbelleau/final-frontier-g0/source-patches/triton_fp4_quant.py
FP6=/home/mbelleau/final-frontier-g0/source-patches/exl3_fp6_conversion.py
for path in "$FIDELITY" "$EXL3_SOURCE" "$FP4" "$TRITON_FP4" "$FP6"; do
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || {
    echo "missing immutable KLD input ${path}" >&2
    exit 2
  }
done
for path in "$KLD_ROOT/suite/shard-0000" "$KLD_ROOT/reference/hidden-bf16" "$KLD_ROOT/lm-head" "$B12X"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "missing immutable KLD directory ${path}" >&2
    exit 2
  }
done
for output in "$FRONTIER_CAMPAIGN_WORK_DIR/capture" "$FRONTIER_CAMPAIGN_WORK_DIR/report.json" "$FRONTIER_CAMPAIGN_WORK_DIR/paired.json"; do
  [[ ! -e "$output" ]] || {
    echo "KLD output already exists: ${output}" >&2
    exit 2
  }
done

QCFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'

run_kld() {
  local name=$1
  shift
  podman run --rm --replace \
    --name "$name" \
    --network none --ipc=host --device nvidia.com/gpu=all \
    --tmpfs /usr/local/cuda-13.2/lib64:rw,size=16m \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e OMP_NUM_THREADS=8 -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
    -e CUTE_DSL_ARCH=sm_120a -e FLASHINFER_CUDA_ARCH_LIST=12.0f \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
    -e XDG_CACHE_HOME=/cache/jit -e CUDA_CACHE_PATH=/cache/jit \
    -e TRITON_CACHE_DIR=/cache/jit/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor \
    -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
    -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/jit/exl3-online -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite \
    -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 -e VLLM_EXL3_MULTIPRECISION=1 \
    -e VLLM_EXL3_EMBED_ONLINE_BITS=6 -e VLLM_EXL3_GRAPH_DECODE=1 \
    -e VLLM_EXL3_FP4_TRITON_DECODE=0 -e VLLM_EXL3_FP4_PER_ROW_GS=0 \
    -e VLLM_EXL3_FP4_DRAFT_HEAD=0 -e VLLM_EXL3_FP4_BANDED_SELFTEST=0 \
    -e VLLM_EXL3_FP8DG_PREFILL_M=0 -e VLLM_EXL3_FP8DG_SELFTEST=0 -e VLLM_EXL3_FP8DG_CACHE=0 \
    -e VLLM_EXL3_SKIP_TRELLIS_PREP=0 -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=1 \
    -e VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB=512 -e VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE=0 \
    -e VLLM_EXL3_B12X_MIN_M=128 -e VLLM_EXL3_B12X_ANY_BITS=1 \
    -e VLLM_EXL3_FOLD_FP32_BUDGET_MB=48 -e VLLM_EXL3_B12X_N_RANGE=5120-36864 \
    -e VLLM_EXL3_FP4_LAYERS=, -e VLLM_EXL3_FP6_LAYERS= \
    -e B12X_PACKED_B_MIN_N=1024 -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$MODEL_CACHE:/root/.cache/huggingface:ro" \
    -v "$KLD_ROOT:/kld-data:ro" \
    -v "$FRONTIER_CAMPAIGN_CACHE_DIR:/cache:rw" \
    -v "$FRONTIER_CAMPAIGN_WORK_DIR:/work:rw" \
    -v "$EXL3_SOURCE:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py:ro" \
    -v "$B12X:/opt/venv/lib/python3.12/site-packages/b12x:ro" \
    -v "$FP4:/opt/fp4/exl3_fp4_conversion.py:ro" \
    -v "$TRITON_FP4:/opt/fp4/triton_fp4_quant.py:ro" \
    -v "$FP6:/opt/fp6/exl3_fp6_conversion.py:ro" \
    --entrypoint /bin/bash \
    "$FRONTIER_CAMPAIGN_IMAGE" \
    -lc 'set -euo pipefail; ln -sf /usr/local/cuda-13.2/targets/x86_64-linux/lib/* /usr/local/cuda-13.2/lib64/; exec "$@"' bash "$@"
}

set +e
run_kld frontier-g02-kld-capture \
  /opt/venv/bin/python /kld-data/fidelity.py capture \
  --model /root/.cache/huggingface/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf \
  --suite /kld-data/suite/shard-0000 \
  --out /work/capture \
  --quantization exl3 \
  --quantization-config "$QCFG" \
  --gpu-memory-utilization 0.85 \
  >"$FRONTIER_CAMPAIGN_WORK_DIR/capture.log" 2>&1
CAPTURE_RC=$?
set -e
shopt -s nullglob
CAPTURE_ENTRIES=("$FRONTIER_CAMPAIGN_WORK_DIR/capture"/*)
CAPTURE_FILES=${#CAPTURE_ENTRIES[@]}
[[ "$CAPTURE_FILES" -eq 513 ]] || {
  echo "capture incomplete: rc=${CAPTURE_RC}, files=${CAPTURE_FILES}/513" >&2
  exit 2
}

run_kld frontier-g02-kld-replay \
  /opt/venv/bin/python /kld-data/fidelity.py replay \
  --reference /kld-data/reference/hidden-bf16 \
  --candidate /work/capture \
  --head /kld-data/lm-head/weight.safetensors \
  --suite /kld-data/suite/shard-0000 \
  --out /work/report.json \
  >"$FRONTIER_CAMPAIGN_WORK_DIR/replay.log" 2>&1

run_kld frontier-g02-kld-paired \
  /opt/venv/bin/python /kld-data/fidelity.py paired \
  --a /work/report.json \
  --b /kld-data/reports/report-alltrellis-anybits.json \
  --a-label clean-int6-g02 \
  --b-label historical-int6 \
  --out /work/paired.json \
  >"$FRONTIER_CAMPAIGN_WORK_DIR/paired.log" 2>&1

python3 - "$FRONTIER_CAMPAIGN_WORK_DIR/report.json" "$FRONTIER_CAMPAIGN_WORK_DIR/paired.json" <<'PY'
import json, math, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
paired = json.load(open(sys.argv[2], encoding="utf-8"))
if report.get("contexts") != 512 or report.get("scored_positions") != 1048064:
    raise SystemExit("KLD report completeness changed")
if report.get("suite_token_sha256") != "caef8a4628d6c07c162100895096f890cdf9cafc8e4c48b3d66035d737ee7cf7":
    raise SystemExit("KLD suite identity changed")
if report.get("head_sha256") != "25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff":
    raise SystemExit("KLD head identity changed")
mean = report.get("token_mean_kld")
if not isinstance(mean, (int, float)) or not math.isfinite(mean):
    raise SystemExit("KLD mean is not finite")
if not isinstance(paired, dict) or not paired:
    raise SystemExit("paired report is empty")
PY
