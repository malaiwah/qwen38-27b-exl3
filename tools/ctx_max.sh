#!/bin/bash
# What is the largest context this build actually serves on a 32 GB card, with MTP-3 and
# vision enabled, and how much does turning MTP off buy?
#
# Native 262,144 does not fit: measured KV requirement is 9.13 GiB with MTP-3 and 8.18 GiB
# without, against 7.71 GiB available. So the honest deliverable is the maximum that does fit,
# found by descending rather than claimed by modelling.
set -u
W=/var/tmp/work
MODEL=${MODEL:-/work/qwen38-ctx}
UTIL=${UTIL:-0.3184}
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'

try() {  # depth len
  local DEPTH=$1 LEN=$2
  local LOG=$W/ctxmax-$DEPTH-$LEN.log
  rm -f $LOG
  local SPEC=""
  [ "$DEPTH" != "0" ] && SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$DEPTH}'"
  $W/ggrun.sh bash -lc "export VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128
exec vllm serve $MODEL --served-model-name m --quantization exl3 \
  --quantization-config '$CFG' --max-model-len $LEN --gpu-memory-utilization $UTIL \
  --kv-cache-dtype fp8 --max-num-seqs 4 --max-num-batched-tokens 2048 $SPEC \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
  --port 8130 --host 127.0.0.1" > $LOG 2>&1 &
  for i in $(seq 1 400); do
    grep -q 'Application startup complete' $LOG && { echo "depth=$DEPTH len=$LEN FITS"; \
      grep -oE 'GPU KV cache size: [0-9,]+ tokens' $LOG | tail -1; \
      grep -oE 'Actual usage is .{0,150}' $LOG | tail -1; pkill -f 'port 8130'; sleep 18; return 0; }
    grep -qE 'ValueError|RuntimeError|Traceback' $LOG && { echo "depth=$DEPTH len=$LEN no"; \
      pkill -f 'port 8130'; sleep 18; return 1; }
    sleep 5
  done
  echo "depth=$DEPTH len=$LEN TIMEOUT"; pkill -f 'port 8130'; sleep 18; return 1
}

for DEPTH in 3 0; do
  for LEN in 245760 229376 212992 196608; do
    if try $DEPTH $LEN; then break; fi
  done
done
echo CTXMAXDONE
