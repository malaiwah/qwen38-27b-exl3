#!/bin/bash
# Start the native-context profile, size prompts with the server tokenizer, and prove use.
set -euo pipefail

W=/var/tmp/work
R=/home/mbelleau/research
PY=/home/mbelleau/qwen38-27b/.venv/bin/python
ROOT=/var/tmp/gg-rootfs/opt/venv/lib/python3.12/site-packages/vllm
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
LOG=$W/native-v2-serve.log
SERVER_PID=

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
}
trap stop_server EXIT INT TERM

cp "$R/tools/vllm-exl3-prefill-dispatch.py" \
  "$ROOT/model_executor/layers/quantization/exl3.py"
cp "$R/tools/vllm-qwen3_5-embed-quant-config.py" \
  "$ROOT/model_executor/models/qwen3_5.py"
cp "$R/tools/vllm-qwen3_5_mtp-embed-quant-config.py" \
  "$ROOT/model_executor/models/qwen3_5_mtp.py"
find "$ROOT" -path '*/__pycache__/exl3*.pyc' -delete

rm -f "$LOG"
setsid "$W/ggrun.sh" bash -lc "export VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_EMBED_BITS=8
exec vllm serve /work/qwen38-ctx --served-model-name m --quantization exl3 \
  --quantization-config '$CFG' --max-model-len 262144 --gpu-memory-utilization 0.3184 \
  --kv-cache-dtype fp8 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --mm-processor-kwargs '{\"truncation\":false}' \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
  --port 8200 --host 127.0.0.1" > "$LOG" 2>&1 &
SERVER_PID=$!

started=false
for _ in $(seq 1 400); do
  if grep -q 'Application startup complete' "$LOG"; then
    started=true
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null ||
     grep -qE 'EngineCore failed|ValidationError|unrecognized arguments|ValueError|RuntimeError|Traceback' "$LOG"; then
    echo "native server failed during startup" >&2
    tail -40 "$LOG" >&2
    exit 1
  fi
  sleep 2
done
if [[ $started != true ]]; then
  echo "native server startup timeout" >&2
  tail -40 "$LOG" >&2
  exit 1
fi

grep -E 'GPU KV cache size:|Actual usage is' "$LOG" | tail -2
for DEPTH in 0.1 0.5 0.9; do
  "$PY" "$R/tools/longctx.py" --url http://127.0.0.1:8200/v1 \
    --corpus /var/tmp/work/kld3/corpus/literary --tokens 261800 --depth "$DEPTH" \
    --require-pass --out "$W/kld3/native-v2-needle-$DEPTH.json"
done
"$PY" "$R/tools/vision_eval.py" --url http://127.0.0.1:8200/v1 \
  --label ctx-embed8-native-v2 --cases 10 --out "$W/kld3/vision-ctx-embed8-native-v2.json"
"$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); o=r["overall"]; assert o["n"] == 30 and o["correct"] >= 24, o' \
  "$W/kld3/vision-ctx-embed8-native-v2.json"
echo NATIVE_V2_PROOF_DONE
