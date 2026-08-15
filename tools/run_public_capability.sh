#!/bin/bash
# Run the paired public MMLU-Pro capability suite over the remaining candidates.
#
# BF16 is the reference and must already exist as a complete report: every
# candidate is scored item-paired against it, so a candidate run without it
# would measure absolute accuracy and nothing else.
#
#   tools/run_public_capability.sh k5k6 hyd ctx fp8
#
# One server at a time, started through the pinned rootfs, verified on
# /v1/models before any request, and stopped before the next model loads.
set -euo pipefail

R=/home/mbelleau/research
W=${GG_WORK:-/var/tmp/work}
MODELS=${GG_MODELS:-/var/tmp/models}
PY=${PYTHON:-/home/mbelleau/qwen38-27b/.venv/bin/python}
GGRUN=$R/tools/ggrun.sh
OUT=$W/public-eval
SUITE=$OUT/suite-mmlupro-70.json
PLAN=$OUT/plan-mmlupro-70-v2.json
REFERENCE=$OUT/report-bf16.json
RUNTIME=$OUT/runtime-bf16.json
PORT=${PORT:-8242}
SERVER_PID=
QCFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'

[ $# -ge 1 ] || { echo "usage: run_public_capability.sh <candidate...>" >&2; exit 2; }
for f in "$SUITE" "$PLAN" "$REFERENCE" "$RUNTIME"; do
  [ -f "$f" ] || { echo "missing required input: $f" >&2; exit 1; }
done

guest_model_of() {
  case "$1" in
    k5k6) echo /work/Qwen3.8-27B-K5K6 ;;
    hyd)  echo /work/qwen38-hyd ;;
    ctx)  echo /work/qwen38-ctx ;;
    fp8)  echo /models/Qwen3.8-27B-FP8 ;;
    k4)   echo /models/Qwen3.8-27B-K4-public ;;
    *) echo "unknown candidate: $1" >&2; exit 1 ;;
  esac
}
host_model_of() {
  case "$1" in
    k5k6) echo "$W/Qwen3.8-27B-K5K6" ;;
    hyd)  echo "$W/qwen38-hyd" ;;
    ctx)  echo "$W/qwen38-ctx" ;;
    fp8)  echo "$MODELS/Qwen3.8-27B-FP8" ;;
    k4)   echo "$MODELS/Qwen3.8-27B-K4-public" ;;
  esac
}
# Every EXL3 candidate needs graph decode explicitly enabled: the backend refuses
# CUDA-graph capture without the autotune priming pass.
env_of() {
  case "$1" in
    fp8)  echo '' ;;
    hyd)  echo 'VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128' ;;
    ctx)  echo 'VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_EMBED_BITS=8' ;;
    k5k6|k4) echo 'VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_ONLINE_TRELLIS_BITS=6 VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online' ;;
  esac
}
quant_of() {
  case "$1" in
    fp8) echo '' ;;
    *)   echo "--quantization exl3 --quantization-config '$QCFG'" ;;
  esac
}
evidence_of() {
  case "$1" in
    k5k6) echo "$R/receipts/release-evidence-k5k6.json" ;;
    hyd)  echo "$R/receipts/release-evidence-hydrated.json" ;;
    ctx)  echo "$R/receipts/release-evidence-context.json" ;;
    k4)   echo "$R/receipts/release-evidence-k4.json" ;;
    fp8)  echo '' ;;
  esac
}

stop_server() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
}
trap stop_server EXIT INT TERM

for cand in "$@"; do
  report=$OUT/report-$cand.json
  if [ -f "$report" ]; then echo "== $cand already reported"; continue; fi
  log=$OUT/server-$cand.log
  script=$W/public-cap-server.sh
  cat > "$script" <<EOF
#!/bin/bash
set -e
exec vllm serve '$(guest_model_of "$cand")' --served-model-name m \\
  --max-model-len 8192 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \\
  --max-num-seqs 4 --max-num-batched-tokens 4096 \\
  --mm-processor-kwargs '{"truncation":false}' \\
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' $(quant_of "$cand") \\
  --port $PORT --host 127.0.0.1
EOF
  chmod +x "$script"
  rm -f "$log"
  echo "== $cand: starting server"
  GG_ENV="$(env_of "$cand")" setsid "$GGRUN" bash /work/public-cap-server.sh > "$log" 2>&1 &
  SERVER_PID=$!
  ready=0
  for _ in $(seq 1 900); do
    if grep -q "Application startup complete" "$log" 2>/dev/null &&
       curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" |
         "$PY" -c 'import json,sys; d=json.load(sys.stdin); assert any(x.get("id")=="m" for x in d["data"])'
    then ready=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "$cand: server exited during startup" >&2; tail -20 "$log" >&2; exit 1; }
    sleep 2
  done
  [ "$ready" = 1 ] || { echo "$cand: server startup timeout" >&2; tail -20 "$log" >&2; exit 1; }

  runtime=$("$PY" - "$RUNTIME" "$(env_of "$cand")" "$cand" <<'PYEOF'
import json, sys
runtime = json.load(open(sys.argv[1]))
runtime.update({"gpu_memory_utilization": 0.85, "env": sys.argv[2], "candidate": sys.argv[3],
                "quantization": "none" if sys.argv[3] == "fp8" else "exl3"})
print(json.dumps(runtime))
PYEOF
)
  cmd=("$PY" "$R/tools/public_capability.py" run --suite "$SUITE" --plan "$PLAN"
       --url "http://127.0.0.1:$PORT/v1" --model-name m --label "$cand"
       --model-path "$(host_model_of "$cand")" --runtime-json "$runtime"
       --reference "$REFERENCE" --out "$report" --concurrency 4 --max-tokens 5120)
  ev=$(evidence_of "$cand")
  [ -n "$ev" ] && cmd+=(--release-evidence "$ev")
  echo "== $cand: scoring"
  "${cmd[@]}"
  stop_server
  sleep 5
done
echo PUBLIC_CAPABILITY_DONE
