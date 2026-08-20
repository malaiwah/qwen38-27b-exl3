#!/bin/bash
# Serve one frozen execution-overlay candidate and record absolute Gate A runtime evidence.
set -euo pipefail

[[ "${FRONTIER_TRANSACTION_ACTIVE:-}" == "1" ]] || { echo "candidate transaction is not active" >&2; exit 2; }
CANDIDATE_ID=${1:-}
case "$CANDIDATE_ID" in
  candidate-01)
    FP4_LAYERS=,
    FP6_LAYERS=mlp.gate_up_proj
    FP4_RANGE=
    MAX_MODEL_LEN=199104
    ;;
  candidate-02)
    FP4_LAYERS=mlp.gate_up_proj
    FP6_LAYERS=
    FP4_RANGE=
    MAX_MODEL_LEN=238400
    ;;
  candidate-03)
    FP4_LAYERS=mlp.gate_up_proj
    FP6_LAYERS=
    FP4_RANGE=0-31
    MAX_MODEL_LEN=238400
    ;;
  *) echo "unknown candidate ID: ${CANDIDATE_ID}" >&2; exit 2 ;;
esac
for name in FRONTIER_CAMPAIGN_CONTAINER FRONTIER_CAMPAIGN_IMAGE FRONTIER_CAMPAIGN_MODEL_ROOT FRONTIER_CAMPAIGN_CACHE_DIR FRONTIER_CAMPAIGN_WORK_DIR; do
  [[ -n "${!name:-}" ]] || { echo "missing ${name}" >&2; exit 2; }
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_GATE="${SCRIPT_DIR}/verify-profile.sh"
BASELINE="${SCRIPT_DIR}/baseline-gate-a.json"
SOURCE_ROOT=/home/mbelleau/final-frontier-g0/source-patches
FP4_HELPER="${SOURCE_ROOT}/exl3_fp4_conversion.py"
TRITON_HELPER="${SOURCE_ROOT}/triton_fp4_quant.py"
FP6_HELPER="${SOURCE_ROOT}/exl3_fp6_conversion.py"
B12X_PACKAGE="${SOURCE_ROOT}/b12x-base-1.2.1/b12x"
PROFILE_JSON="${FRONTIER_CAMPAIGN_WORK_DIR}/profile.json"
INSPECT_JSON="${FRONTIER_CAMPAIGN_WORK_DIR}/container-inspect.json"
LOG_FILE="${FRONTIER_CAMPAIGN_WORK_DIR}/runtime.log"
ASSESSMENT="${FRONTIER_CAMPAIGN_WORK_DIR}/candidate-assessment.json"
for path in "$PROFILE_GATE" "$BASELINE" "$FP4_HELPER" "$TRITON_HELPER" "$FP6_HELPER"; do
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || { echo "missing immutable candidate input: ${path}" >&2; exit 2; }
done
[[ -d "$B12X_PACKAGE" && ! -L "$B12X_PACKAGE" ]] || { echo "missing B12X package" >&2; exit 2; }
[[ ! -e "$PROFILE_JSON" && ! -e "$INSPECT_JSON" && ! -e "$LOG_FILE" && ! -e "$ASSESSMENT" ]] || { echo "candidate output exists" >&2; exit 2; }

cleanup_log() {
  podman logs "${FRONTIER_CAMPAIGN_CONTAINER}" >"${LOG_FILE}.tmp" 2>&1 || true
  [[ ! -s "${LOG_FILE}.tmp" ]] || mv -f "${LOG_FILE}.tmp" "$LOG_FILE"
  rm -f "${LOG_FILE}.tmp"
}
trap cleanup_log EXIT
QUANTIZATION_CONFIG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
MM_PROCESSOR_KWARGS='{"truncation":false,"max_pixels":8388608}'
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":6}'

podman run -d --replace \
  --name "${FRONTIER_CAMPAIGN_CONTAINER}" --device nvidia.com/gpu=all --ipc=host --network host \
  --health-cmd 'curl -sf http://localhost:8000/health || exit 1' --health-interval 30s --health-timeout 10s --health-retries 3 --health-start-period 45m \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e OMP_NUM_THREADS=8 -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e CUTE_DSL_ARCH=sm_120a -e FLASHINFER_CUDA_ARCH_LIST=12.0f \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e XDG_CACHE_HOME=/cache/jit -e CUDA_CACHE_PATH=/cache/jit -e TRITON_CACHE_DIR=/cache/jit/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/jit/exl3-online -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
  -e VLLM_EXL3_MULTIPRECISION=1 -e VLLM_EXL3_EMBED_ONLINE_BITS=8 -e VLLM_EXL3_FP4_TRITON_DECODE=0 -e VLLM_EXL3_GRAPH_DECODE=1 \
  -e VLLM_EXL3_FP4_PER_ROW_GS=0 -e VLLM_EXL3_FP4_DRAFT_HEAD=0 -e VLLM_EXL3_FP4_BANDED_SELFTEST=0 \
  -e VLLM_EXL3_FP8DG_PREFILL_M=0 -e VLLM_EXL3_FP8DG_SELFTEST=0 -e VLLM_EXL3_FP8DG_CACHE=0 \
  -e VLLM_EXL3_SKIP_TRELLIS_PREP=0 -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=1 \
  -e VLLM_EXL3_FP4_LAYERS="$FP4_LAYERS" -e VLLM_EXL3_FP6_LAYERS="$FP6_LAYERS" -e VLLM_EXL3_FP4_LAYER_RANGE="$FP4_RANGE" \
  -e VLLM_EXL3_B12X_ANY_BITS=1 -e VLLM_EXL3_B12X_SELFTEST=0 -e VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB=512 -e VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE=0 \
  -e VLLM_EXL3_B12X_MIN_M=128 -e VLLM_EXL3_FOLD_FP32_BUDGET_MB=48 -e VLLM_EXL3_B12X_N_RANGE=5120-36864 -e B12X_PACKED_B_MIN_N=1024 \
  -e QUANTIZATION_CONFIG="$QUANTIZATION_CONFIG" -e MM_PROCESSOR_KWARGS="$MM_PROCESSOR_KWARGS" -e SPEC_CONFIG="$SPEC_CONFIG" \
  -e HF_HOME=/cache/hf -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${FRONTIER_CAMPAIGN_MODEL_ROOT}:/models/qwen38:ro" -v "${FRONTIER_CAMPAIGN_CACHE_DIR}:/cache:rw" -v "${FRONTIER_CAMPAIGN_WORK_DIR}:/work:rw" \
  -v "${FP4_HELPER}:/opt/fp4/exl3_fp4_conversion.py:ro" -v "${TRITON_HELPER}:/opt/fp4/triton_fp4_quant.py:ro" -v "${FP6_HELPER}:/opt/fp6/exl3_fp6_conversion.py:ro" \
  -v "${B12X_PACKAGE}:/opt/venv/lib/python3.12/site-packages/b12x:ro" \
  --entrypoint /opt/venv/bin/vllm "${FRONTIER_CAMPAIGN_IMAGE}" \
  serve /models/qwen38 --served-model-name Qwen3.8-27B --trust-remote-code --host 0.0.0.0 --port 8000 --quantization exl3 \
  --quantization-config "$QUANTIZATION_CONFIG" --attention-backend TRITON_ATTN --gpu-memory-utilization 0.95 --kv-cache-dtype fp8_e4m3 \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 4 --max-num-batched-tokens 3072 \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}' --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --mm-processor-cache-type shm \
  --default-chat-template-kwargs '{"preserve_thinking":true}' --enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config "$SPEC_CONFIG"

healthy=0
for _ in $(seq 1 420); do
  if curl -fsS --max-time 10 http://127.0.0.1:8000/health >/dev/null 2>&1; then healthy=1; break; fi
  state="$(podman inspect -f '{{.State.Status}}' "${FRONTIER_CAMPAIGN_CONTAINER}" 2>/dev/null || printf missing)"
  [[ "$state" == running ]] || { podman logs --tail 200 "${FRONTIER_CAMPAIGN_CONTAINER}" >&2 || true; exit 2; }
  sleep 5
done
[[ "$healthy" -eq 1 ]] || exit 2
podman inspect "${FRONTIER_CAMPAIGN_CONTAINER}" >"${INSPECT_JSON}.tmp"
mv "${INSPECT_JSON}.tmp" "$INSPECT_JSON"
set +e
BENCH_BASE_URL=http://127.0.0.1:8000 "$PROFILE_GATE" --baseline "$BASELINE" --json-out "$PROFILE_JSON" --no-boot
GATE_RC=$?
set -e
[[ "$GATE_RC" -eq 0 || "$GATE_RC" -eq 1 ]] || exit "$GATE_RC"
cleanup_log
trap - EXIT

python3 - "$CANDIDATE_ID" "$FP4_LAYERS" "$FP6_LAYERS" "$FP4_RANGE" "$MAX_MODEL_LEN" "$PROFILE_JSON" "$INSPECT_JSON" "$LOG_FILE" "$ASSESSMENT" <<'PY'
import hashlib, json, os, pathlib, sys
candidate, fp4, fp6, layer_range, max_len = sys.argv[1:6]
profile_path, inspect_path, log_path, out = map(pathlib.Path, sys.argv[6:10])

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()
profile=load(profile_path)
inspect=load(inspect_path)
log_text=log_path.read_text(encoding="utf-8",errors="replace")
fatal=[token for token in ("EngineDeadError","CUDA out of memory","silent fallback","falling back to","Traceback (most recent call last)") if token.lower() in log_text.lower()]
failed=[name for name,row in profile.get("metrics",{}).items() if row.get("verdict")!="PASS"]
failed.extend("fatal:"+token for token in fatal)
runtime_pass=not failed and profile.get("pass") is True
value={
  "schema":"qwen38-frontier-candidate-assessment/1","candidate_id":candidate,
  "tuple":{"fp4_layers":fp4,"fp6_layers":fp6,"fp4_layer_range":layer_range or None,"max_model_len":int(max_len),"payload_revision":"ab3a91a13813df8096cb4c1d560ed3669035d0cf","runtime_commit":"b19029d2309b26c4942425e52b74a0e6dd5d141e"},
  "runtime_gate_pass":runtime_pass,"failed_runtime_gates":failed,
  "kld":{"status":"required" if runtime_pass else "not_measured_due_deterministic_hard_gate","mean":None,"tails":None},
  "gate_a_pass":False,"disposition":"requires_kld" if runtime_pass else "no_go",
  "profile":profile,
  "artifacts":{"profile":{"path":"profile.json","sha256":sha(profile_path)},"inspect":{"path":"container-inspect.json","sha256":sha(inspect_path)},"runtime_log":{"path":"runtime.log","sha256":sha(log_path)}},
}
payload=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
tmp=out.with_name(out.name+".tmp")
with tmp.open("xb") as f: f.write(payload); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
