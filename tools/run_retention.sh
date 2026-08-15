#!/bin/bash
# Run the same deterministic task-retention smoke suite on BF16, all four EXL3
# artifacts, and official FP8. Every report is atomic and content-identified.
set -euo pipefail

W=/var/tmp/work
R=/home/mbelleau/research
PY=/home/mbelleau/qwen38-27b/.venv/bin/python
HARNESS=$R/tools/task_retention.py
GGRUN=$R/tools/ggrun.sh
ROOTFS=/var/tmp/gg-rootfs
ROOT=$ROOTFS/opt/venv/lib/python3.12/site-packages/vllm
PORT=8230
IMAGE='voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b'
ROOTFS_PROVENANCE=${ROOTFS_PROVENANCE:-$W/gg-rootfs-provenance.json}
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
SERVER_PID=

mapfile -t ROOTFS_FACTS < <(
  "$PY" "$R/tools/pull_rootfs.py" verify "$ROOTFS_PROVENANCE" "$IMAGE" "$ROOTFS"
)
[[ ${#ROOTFS_FACTS[@]} == 2 ]] || {
  echo "verified extracted-rootfs provenance is required" >&2
  exit 1
}
ROOTFS_MANIFEST=${ROOTFS_FACTS[0]}
ROOTFS_MANIFEST_SHA=${ROOTFS_FACTS[1]}
ROOTFS_PROVENANCE_SHA=$(sha256sum "$ROOTFS_PROVENANCE" | cut -d' ' -f1)
GGRUN_SHA=$(sha256sum "$GGRUN" | cut -d' ' -f1)

# Install the exact public patch artifacts into the extracted image before any run.
EXL3_SHA=2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee
QWEN_SHA=04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7
MTP_SHA=0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab
printf '%s  %s\n' \
  "$EXL3_SHA" "$R/tools/vllm-exl3-prefill-dispatch.py" \
  "$QWEN_SHA" "$R/tools/vllm-qwen3_5-embed-quant-config.py" \
  "$MTP_SHA" "$R/tools/vllm-qwen3_5_mtp-embed-quant-config.py" |
  sha256sum --quiet -c -
cp "$R/tools/vllm-exl3-prefill-dispatch.py" \
  "$ROOT/model_executor/layers/quantization/exl3.py"
cp "$R/tools/vllm-qwen3_5-embed-quant-config.py" \
  "$ROOT/model_executor/models/qwen3_5.py"
cp "$R/tools/vllm-qwen3_5_mtp-embed-quant-config.py" \
  "$ROOT/model_executor/models/qwen3_5_mtp.py"
find "$ROOT" -path '*/__pycache__/*.pyc' \
  \( -name 'exl3*.pyc' -o -name 'qwen3_5*.pyc' \) -delete
cmp "$R/tools/vllm-exl3-prefill-dispatch.py" \
  "$ROOT/model_executor/layers/quantization/exl3.py"
cmp "$R/tools/vllm-qwen3_5-embed-quant-config.py" \
  "$ROOT/model_executor/models/qwen3_5.py"
cmp "$R/tools/vllm-qwen3_5_mtp-embed-quant-config.py" \
  "$ROOT/model_executor/models/qwen3_5_mtp.py"

stop_server() {
  if [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
}
trap stop_server EXIT INT TERM

start_server() {
  local tag=$1 guest_model=$2 mode=$3 env_spec=$4
  local log=$W/ret-v2-$tag-server.log script=$W/ret-v2-server.sh
  local quant_line=
  if [[ $mode == exl3 ]]; then
    quant_line="--quantization exl3 --quantization-config '$CFG'"
  fi
  cat > "$script" <<EOF
#!/bin/bash
set -e
export VLLM_WORKER_MULTIPROC_METHOD=spawn $env_spec
exec vllm serve '$guest_model' --served-model-name m --max-model-len 8192 \\
  --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --max-num-seqs 4 \\
  --max-num-batched-tokens 2048 --mm-processor-kwargs '{"truncation":false}' \\
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' $quant_line \\
  --port $PORT --host 127.0.0.1
EOF
  chmod +x "$script"
  rm -f "$log"
  setsid "$GGRUN" bash /work/ret-v2-server.sh > "$log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 360); do
    if grep -qE "Uvicorn running on https?://127\\.0\\.0\\.1:$PORT" "$log" &&
       kill -0 "$SERVER_PID" 2>/dev/null &&
       curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" |
         "$PY" -c 'import json,sys; p=json.load(sys.stdin); assert any(x.get("id") == "m" for x in p["data"])'
    then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$tag server exited during startup" >&2
      tail -40 "$log" >&2
      return 1
    fi
    if grep -qE 'EngineCore failed|ValidationError|unrecognized arguments|ValueError|RuntimeError|Traceback' "$log"; then
      echo "$tag server failed during startup" >&2
      grep -E 'EngineCore failed|ValidationError|unrecognized arguments|ValueError|RuntimeError|Traceback' "$log" | tail -8 >&2
      return 1
    fi
    sleep 2
  done
  echo "$tag server startup timeout" >&2
  tail -40 "$log" >&2
  return 1
}

run_model() {
  local tag=$1 guest_model=$2 host_model=$3 revision=$4 evidence=$5 mode=$6 env_spec=$7
  local out=$W/kld3/tasks-v2-$tag.json bench_log=$W/ret-v2-$tag-bench.log
  local runtime
  runtime=$(printf '{"image":"%s","rootfs_provenance_sha256":"%s","rootfs_file_manifest_sha256":"%s","ggrun_sha256":"%s","identity_verified":true,"exl3_module_sha256":"%s","qwen3_5_module_sha256":"%s","qwen3_5_mtp_module_sha256":"%s","environment":"%s","max_model_len":8192,"max_num_seqs":4,"kv_cache_dtype":"fp8","cuda_graphs":"FULL_DECODE_ONLY"}' \
    "$IMAGE" "$ROOTFS_PROVENANCE_SHA" "$ROOTFS_MANIFEST_SHA" "$GGRUN_SHA" "$EXL3_SHA" "$QWEN_SHA" "$MTP_SHA" "$env_spec")
  echo "=== $tag"
  start_server "$tag" "$guest_model" "$mode" "$env_spec"
  local cmd=("$PY" "$HARNESS" --url "http://127.0.0.1:$PORT/v1" --label "$tag"
             --model-path "$host_model" --runtime-json "$runtime" --per-kind 10
             --concurrency 4 --out "$out")
  [[ -n $revision ]] && cmd+=(--model-revision "$revision")
  [[ -n $evidence ]] && cmd+=(--release-evidence "$evidence")
  if [[ $tag == bf16 ]]; then
    cmd+=(--require-all-pass)
  else
    cmd+=(--reference "$W/kld3/tasks-v2-bf16.json"
          --require-all-pass --require-no-regressions)
  fi
  "${cmd[@]}" | tee "$bench_log"
  stop_server
  sleep 5
}

run_model bf16 /models/Qwen3.8-27B /var/tmp/models/Qwen3.8-27B \
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 '' auto ''

# Refuse to compare candidates against a task set the BF16 model did not solve.
"$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); rates=[v["pass"]["rate"] for v in r["per_kind"].values()]; overall=r["overall"]["pass"]["rate"]; print("BF16", overall, rates); raise SystemExit(0 if overall >= .85 and min(rates) >= .7 else "BF16 task suite is not discriminative")' \
  "$W/kld3/tasks-v2-bf16.json"

run_model ctx /work/qwen38-ctx /var/tmp/work/qwen38-ctx '' \
  /var/tmp/work/qwen38-ctx/release-evidence.json exl3 \
  'VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_EMBED_BITS=8'
run_model hydrated /work/qwen38-hyd /var/tmp/work/qwen38-hyd '' \
  /var/tmp/work/qwen38-hyd/release-evidence.json exl3 \
  'VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128'
run_model k5k6 /work/Qwen3.8-27B-K5K6 /var/tmp/work/Qwen3.8-27B-K5K6 '' \
  /var/tmp/work/card2/release-evidence.json exl3 \
  'VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_ONLINE_TRELLIS_BITS=6 VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online'
run_model fp8 /models/Qwen3.8-27B-FP8 /var/tmp/models/Qwen3.8-27B-FP8 \
  017b9c7af6b5689d5dd426a76e0bc077eb5ca20a '' auto ''
run_model k4 /work/Qwen3.8-27B-K4 /var/tmp/work/Qwen3.8-27B-K4 '' \
  /var/tmp/work/card/release-evidence.json exl3 \
  'VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_ONLINE_TRELLIS_BITS=6 VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online'

for tag in bf16 ctx hydrated k5k6 fp8 k4; do
  cp "$W/kld3/tasks-v2-$tag.json" "$R/receipts/"
done

echo RETENTION_V2_DONE
