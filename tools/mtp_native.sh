#!/bin/bash
# Probe whether MTP and native context coexist when concurrency is honestly capped at one.
# This is a startup diagnostic, not a generation or quality proof.
set -euo pipefail
W=${W:-/var/tmp/work}
GGRUN=${GGRUN:-$W/ggrun.sh}
MODEL=${MODEL:-/work/qwen38-ctx}
PORT=${PORT:-8210}
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
SERVER_PID=
FAILED=0

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

for DEPTH in 1 3; do
  ROWS=$((DEPTH + 1))
  LOG=$W/mtpnative-seq1-$DEPTH.log
  SPEC_CONFIG=$(printf '{"method":"mtp","num_speculative_tokens":%d}' "$DEPTH")
  COMPILE_CONFIG=$(printf '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[%d]}' "$ROWS")
  rm -f "$LOG"
  GG_ENV='VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_EMBED_BITS=8' \
    setsid "$GGRUN" vllm serve "$MODEL" --served-model-name m --quantization exl3 \
    --quantization-config "$CFG" --max-model-len 262144 --gpu-memory-utilization 0.3205 \
    --kv-cache-dtype fp8 --max-num-seqs 1 --max-num-batched-tokens 2048 \
    --speculative-config "$SPEC_CONFIG" --mm-processor-kwargs '{"truncation":false}' \
    --compilation-config "$COMPILE_CONFIG" --port "$PORT" --host 127.0.0.1 \
    >"$LOG" 2>&1 &
  SERVER_PID=$!
  OUTCOME=timeout
  for _ in $(seq 1 420); do
    if grep -q 'Application startup complete' "$LOG"; then
      OUTCOME=fits
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null ||
       grep -qE 'EngineCore failed|ValidationError|ValueError|RuntimeError|Traceback' "$LOG"; then
      OUTCOME=failed
      break
    fi
    sleep 5
  done
  if [[ $OUTCOME == fits ]]; then
    echo "mtp=$DEPTH FITS 262144 startup"
  else
    echo "mtp=$DEPTH $OUTCOME" >&2
    FAILED=1
  fi
  grep -oE 'Sharing target model embedding weights|GPU KV cache size: [0-9,]+ tokens|Actual usage is .{0,160}|\([0-9.]+ GiB KV cache is needed[^)]*\)' "$LOG" | tail -4 || true
  stop_server
  sleep 18
done
echo MTPNATIVEDONE
exit "$FAILED"
