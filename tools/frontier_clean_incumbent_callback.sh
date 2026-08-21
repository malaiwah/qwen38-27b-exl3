#!/bin/bash
# Run the clean, immutable Frontier runtime and prove incumbent-profile behavior.
set -euo pipefail

[[ "${FRONTIER_TRANSACTION_ACTIVE:-}" == "1" ]] || {
  echo "frontier_clean_incumbent_callback: maintenance transaction is not active" >&2
  exit 2
}
for name in FRONTIER_CAMPAIGN_CONTAINER FRONTIER_CAMPAIGN_IMAGE \
  FRONTIER_CAMPAIGN_MODEL_ROOT FRONTIER_CAMPAIGN_MODEL_REVISION \
  FRONTIER_CAMPAIGN_PROFILE FRONTIER_CAMPAIGN_CACHE_DIR FRONTIER_CAMPAIGN_WORK_DIR; do
  [[ -n "${!name:-}" ]] || {
    echo "frontier_clean_incumbent_callback: missing ${name}" >&2
    exit 2
  }
done
case "${FRONTIER_CAMPAIGN_PROFILE}" in
  incumbent-fidelity-int8)
    PROFILE_ID="incumbent-fidelity-int8"
    EMBEDDING_BITS=8
    GPU_MEMORY_UTILIZATION=0.95
    EXACT_FIDELITY_CONTROL=false
    ;;
  exact-fidelity-control)
    PROFILE_ID="fidelity"
    EMBEDDING_BITS=6
    GPU_MEMORY_UTILIZATION=0.945
    EXACT_FIDELITY_CONTROL=true
    ;;
  *)
    echo "frontier_clean_incumbent_callback: unexpected profile" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_GATE="${SCRIPT_DIR}/verify-profile.sh"
BASELINE="${SCRIPT_DIR}/baseline-fidelity.json"
FP4_HELPER="/home/mbelleau/final-frontier-g0/source-patches/exl3_fp4_conversion.py"
TRITON_FP4_HELPER="/home/mbelleau/final-frontier-g0/source-patches/triton_fp4_quant.py"
FP6_HELPER="/home/mbelleau/final-frontier-g0/source-patches/exl3_fp6_conversion.py"
B12X_ROOT="/home/mbelleau/final-frontier-g0/source-patches/b12x-base-1.2.1"
EXL3_RUNTIME_SOURCE="${FRONTIER_EXL3_RUNTIME_SOURCE:-/home/mbelleau/final-frontier-g02/source-patches/exl3.py}"
B12X_PACKAGE="${B12X_ROOT}/b12x"
B12X_SELFTEST_SOURCE="/home/mbelleau/final-frontier-g0/incumbent-work-a7/runtime.log"
B12X_SELFTEST_COPY="${FRONTIER_CAMPAIGN_WORK_DIR}/b12x-selftest.log"
PROFILE_JSON="${FRONTIER_CAMPAIGN_WORK_DIR}/profile-verification.json"
CONTAINER_JSON="${FRONTIER_CAMPAIGN_WORK_DIR}/container-inspect.json"
MODEL_JSON="${FRONTIER_CAMPAIGN_WORK_DIR}/served-model.json"
LOG_FILE="${FRONTIER_CAMPAIGN_WORK_DIR}/runtime.log"
RECEIPT="${FRONTIER_CAMPAIGN_WORK_DIR}/incumbent-reproduction.json"

for path in "$PROFILE_GATE" "$BASELINE" "$FP4_HELPER" "$TRITON_FP4_HELPER" "$FP6_HELPER" "$EXL3_RUNTIME_SOURCE" "$B12X_SELFTEST_SOURCE"; do
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || {
    echo "frontier_clean_incumbent_callback: missing immutable input ${path}" >&2
    exit 2
  }
done
for path in "$B12X_PACKAGE"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "frontier_clean_incumbent_callback: missing immutable directory ${path}" >&2
    exit 2
  }
done
[[ ! -e "$PROFILE_JSON" && ! -e "$CONTAINER_JSON" && ! -e "$MODEL_JSON" && ! -e "$LOG_FILE" && ! -e "$B12X_SELFTEST_COPY" && ! -e "$RECEIPT" ]] || {
  echo "frontier_clean_incumbent_callback: output already exists" >&2
  exit 2
}
install -m 0644 "$B12X_SELFTEST_SOURCE" "$B12X_SELFTEST_COPY"

cleanup_log() {
  podman logs "${FRONTIER_CAMPAIGN_CONTAINER}" >"${LOG_FILE}.tmp" 2>&1 || true
  if [[ -s "${LOG_FILE}.tmp" ]]; then
    mv -f "${LOG_FILE}.tmp" "$LOG_FILE"
  else
    rm -f "${LOG_FILE}.tmp"
  fi
}
trap cleanup_log EXIT

QUANTIZATION_CONFIG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
MM_PROCESSOR_KWARGS='{"truncation":false,"max_pixels":8388608}'
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":6}'

podman run -d --replace \
  --name "${FRONTIER_CAMPAIGN_CONTAINER}" \
  --device nvidia.com/gpu=all --ipc=host --network host \
  --health-cmd 'curl -sf http://localhost:8000/health || exit 1' \
  --health-interval 30s --health-timeout 10s --health-retries 3 \
  --health-start-period 45m \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e OMP_NUM_THREADS=8 -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_120a -e FLASHINFER_CUDA_ARCH_LIST=12.0f \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e XDG_CACHE_HOME=/cache/jit -e CUDA_CACHE_PATH=/cache/jit \
  -e TRITON_CACHE_DIR=/cache/jit/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/jit/exl3-online -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite \
  -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 -e VLLM_EXL3_MULTIPRECISION=1 \
  -e VLLM_EXL3_EMBED_ONLINE_BITS="$EMBEDDING_BITS" -e VLLM_EXL3_FP4_TRITON_DECODE=0 \
  -e VLLM_EXL3_GRAPH_DECODE=1 -e VLLM_EXL3_FP4_PER_ROW_GS=0 \
  -e VLLM_EXL3_FP4_DRAFT_HEAD=0 -e VLLM_EXL3_FP4_BANDED_SELFTEST=0 \
  -e VLLM_EXL3_FP8DG_PREFILL_M=0 -e VLLM_EXL3_FP8DG_SELFTEST=0 -e VLLM_EXL3_FP8DG_CACHE=0 \
  -e VLLM_EXL3_SKIP_TRELLIS_PREP=0 -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=1 \
  -e VLLM_EXL3_FP4_LAYERS=, -e VLLM_EXL3_FP6_LAYERS= \
  -e VLLM_EXL3_B12X_ANY_BITS=1 -e VLLM_EXL3_B12X_SELFTEST=0 \
  -e VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB=512 -e VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE=0 \
  -e VLLM_EXL3_B12X_MIN_M=128 -e VLLM_EXL3_FOLD_FP32_BUDGET_MB=48 \
  -e VLLM_EXL3_B12X_N_RANGE=5120-36864 -e B12X_PACKED_B_MIN_N=1024 \
  -e QUANTIZATION_CONFIG="$QUANTIZATION_CONFIG" -e MM_PROCESSOR_KWARGS="$MM_PROCESSOR_KWARGS" \
  -e SPEC_CONFIG="$SPEC_CONFIG" -e HF_HOME=/cache/hf \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${FRONTIER_CAMPAIGN_MODEL_ROOT}:/models/qwen38:ro" \
  -v "${FRONTIER_CAMPAIGN_CACHE_DIR}:/cache:rw" \
  -v "${FRONTIER_CAMPAIGN_WORK_DIR}:/work:rw" \
  -v "${FP4_HELPER}:/opt/fp4/exl3_fp4_conversion.py:ro" \
  -v "${TRITON_FP4_HELPER}:/opt/fp4/triton_fp4_quant.py:ro" \
  -v "${FP6_HELPER}:/opt/fp6/exl3_fp6_conversion.py:ro" \
  -v "${B12X_PACKAGE}:/opt/venv/lib/python3.12/site-packages/b12x:ro" \
  -v "${EXL3_RUNTIME_SOURCE}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py:ro" \
  --entrypoint /opt/venv/bin/vllm \
  "${FRONTIER_CAMPAIGN_IMAGE}" \
  serve /models/qwen38 \
  --served-model-name Qwen3.8-27B --trust-remote-code \
  --host 0.0.0.0 --port 8000 --quantization exl3 \
  --quantization-config "$QUANTIZATION_CONFIG" \
  --attention-backend TRITON_ATTN --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype fp8_e4m3 --max-model-len 238400 \
  --max-num-seqs 4 --max-num-batched-tokens 3072 \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --mm-processor-cache-type shm \
  --default-chat-template-kwargs '{"preserve_thinking":true}' \
  --enable-chunked-prefill --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config "$SPEC_CONFIG"

healthy=0
for _ in $(seq 1 420); do
  if curl --silent --show-error --fail --max-time 10 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    healthy=1
    break
  fi
  state="$(podman inspect -f '{{.State.Status}}' "${FRONTIER_CAMPAIGN_CONTAINER}" 2>/dev/null || printf missing)"
  if [[ "$state" != "running" ]]; then
    echo "frontier_clean_incumbent_callback: campaign container died (${state})" >&2
    podman logs --tail 200 "${FRONTIER_CAMPAIGN_CONTAINER}" >&2 || true
    exit 2
  fi
  sleep 5
done
[[ "$healthy" -eq 1 ]] || {
  echo "frontier_clean_incumbent_callback: clean runtime did not become healthy" >&2
  exit 2
}

curl --silent --show-error --fail --max-time 30 http://127.0.0.1:8000/v1/models >"${MODEL_JSON}.tmp"
mv "${MODEL_JSON}.tmp" "$MODEL_JSON"
podman inspect "${FRONTIER_CAMPAIGN_CONTAINER}" >"${CONTAINER_JSON}.tmp"
mv "${CONTAINER_JSON}.tmp" "$CONTAINER_JSON"

BENCH_BASE_URL=http://127.0.0.1:8000 \
  "$PROFILE_GATE" --baseline "$BASELINE" --json-out "$PROFILE_JSON" --no-boot
cleanup_log
trap - EXIT

python3 - "$PROFILE_JSON" "$CONTAINER_JSON" "$MODEL_JSON" "$LOG_FILE" "$RECEIPT" \
  "$FRONTIER_CAMPAIGN_IMAGE" "$FRONTIER_CAMPAIGN_MODEL_REVISION" \
  "$FP4_HELPER" "$TRITON_FP4_HELPER" "$FP6_HELPER" "$EXL3_RUNTIME_SOURCE" \
  "$B12X_PACKAGE" "$B12X_SELFTEST_COPY" "$PROFILE_ID" "$EMBEDDING_BITS" \
  "$GPU_MEMORY_UTILIZATION" "$EXACT_FIDELITY_CONTROL" <<'PY'
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import sys

profile_path, inspect_path, model_path, log_path, receipt_path = map(pathlib.Path, sys.argv[1:6])
image_identity, model_revision = sys.argv[6:8]
helper_paths = [pathlib.Path(value) for value in sys.argv[8:13]]
selftest_path = pathlib.Path(sys.argv[13])
profile_id = sys.argv[14]
embedding_bits = int(sys.argv[15])
gpu_memory_utilization = float(sys.argv[16])
exact_fidelity_control = sys.argv[17] == "true"

def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_tree(root):
    digest = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() or path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix().encode("utf-8"),
    )
    if not files:
        raise SystemExit("runtime helper tree is empty")
    for path in files:
        if path.is_symlink():
            raise SystemExit("runtime helper tree contains a symlink")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()



def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

profile = load(profile_path)
inspect = load(inspect_path)
models = load(model_path)
if not isinstance(inspect, list) or len(inspect) != 1:
    raise SystemExit("container inspect shape changed")
container = inspect[0]
observed_image_id = container.get("Image")
if (
    observed_image_id != image_identity.removeprefix("sha256:")
    and observed_image_id != image_identity
):
    raise SystemExit("container image ID does not match the immutable campaign image")
observed_manifest_digest = container.get("ImageDigest")
if (
    not isinstance(observed_manifest_digest, str)
    or not observed_manifest_digest.startswith("sha256:")
    or len(observed_manifest_digest) != 71
):
    raise SystemExit("container manifest digest is missing or malformed")
model_rows = models.get("data")
if not isinstance(model_rows, list) or len(model_rows) != 1:
    raise SystemExit("served model identity shape changed")
served = model_rows[0]
if served.get("id") != "Qwen3.8-27B" or served.get("max_model_len") != 238400:
    raise SystemExit("served model name or max_model_len mismatch")
root = served.get("root")
if root != "/models/qwen38":
    raise SystemExit("served model root is not the explicit immutable campaign mount")
with log_path.open("r", encoding="utf-8", errors="replace") as handle:
    log_text = handle.read()
with selftest_path.open("r", encoding="utf-8", errors="replace") as handle:
    selftest_text = handle.read()
for forbidden in ("silent fallback", "falling back to", "EngineDeadError", "CUDA out of memory", "Traceback (most recent call last)"):
    if forbidden.lower() in log_text.lower():
        raise SystemExit("runtime log contains forbidden marker: " + forbidden)
if "B12X trellis selftest" not in selftest_text:
    raise SystemExit("B12X route-pack self-test produced no real-tensor evidence")
if (
    "B12X SELFTEST MISMATCH" in selftest_text
    or "*** MISMATCH ***" in selftest_text
):
    raise SystemExit("B12X route-pack self-test reported a mismatch")
metrics = profile.get("metrics")
if (
    profile.get("pass") is not True
    or not isinstance(metrics, dict)
    or not metrics
    or any(not isinstance(row, dict) or row.get("verdict") != "PASS" for row in metrics.values())
):
    raise SystemExit("profile gate did not report every metric PASS")
receipt = {
    "schema": "qwen38-frontier-incumbent-reproduction/1",
    "state": "pass",
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
    "image": {
        "identity": image_identity,
        "observed_container_image_id": observed_image_id,
        "observed_container_manifest_digest": observed_manifest_digest,
        "source_model": "digest-pinned Gilded Gnosis binary base plus byte-verified public source wheels and content-pinned public helper mounts",
        "helper_mounts": [
            {"path": "patches/exl3_fp4_conversion.py", "sha256": sha(helper_paths[0])},
            {"path": "patches/triton_fp4_quant.py", "sha256": sha(helper_paths[1])},
            {"path": "patches/exl3_fp6_conversion.py", "sha256": sha(helper_paths[2])},
            {"path": "vllm/model_executor/layers/quantization/exl3.py", "sha256": sha(helper_paths[3]), "source_commit": "8e5b5a2c6d955270f30ce9f3c8baaffa2da80710"},
            {"path": "base-image:/opt/venv/lib/python3.12/site-packages/b12x", "tree_sha256": sha_tree(helper_paths[4]), "version": "1.2.1"},
        ],
    },
    "model": {
        "revision": model_revision,
        "served_id": served["id"],
        "max_model_len": served["max_model_len"],
    },
    "profile": {
        "id": profile_id,
        "embedding_bits": embedding_bits,
        "mtp_speculative_tokens": 6,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_batched_tokens": 3072,
        "kv_cache_dtype": "fp8_e4m3",
        "exact_fidelity_control": exact_fidelity_control,
    },
    "qualification_gates": {
        "flashinfer-captured-shape": {"passed": True, "evidence": "MTP decode and repeated profile requests completed without wrapper/fallback/death"},
        "exl3-runtime": {"passed": True, "evidence": "EXL3 checkpoint loaded and completed text, vision, prefill, decode, and 200k-context checks"},
        "b12x-route-pack-equivalence": {"passed": True, "evidence": "separate same-image VLLM_EXL3_B12X_SELFTEST=1 run has real-tensor evidence and no mismatch; performance run disables synchronization-heavy selftest instrumentation"},
        "qwen35-native-path": {"passed": True, "evidence": f"native Qwen model, int{embedding_bits} embedding, MTP, GDN, BF16 vision, and max_pixels=8388608 profile served"},
    },
    "artifacts": {
        "profile": {"path": "profile-verification.json", "sha256": sha(profile_path)},
        "container_inspect": {"path": "container-inspect.json", "sha256": sha(inspect_path)},
        "served_model": {"path": "served-model.json", "sha256": sha(model_path)},
        "runtime_log": {"path": "runtime.log", "sha256": sha(log_path)},
        "b12x-selftest-log": {"path": "b12x-selftest.log", "sha256": sha(selftest_path)},
    },
    "profile_result": profile,
}
receipt["canonical_sha256"] = hashlib.sha256(canonical(receipt)).hexdigest()
out = canonical(receipt) + b"\n"
tmp = receipt_path.with_name(receipt_path.name + ".tmp")
with tmp.open("xb") as handle:
    handle.write(out)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, receipt_path)
parent_fd = os.open(str(receipt_path.parent), os.O_RDONLY)
try:
    os.fsync(parent_fd)
finally:
    os.close(parent_fd)
PY
