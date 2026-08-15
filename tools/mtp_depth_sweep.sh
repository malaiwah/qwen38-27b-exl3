#!/bin/bash
# How much KV does each MTP draft depth cost, and what is the deepest draft that still fits
# native 262,144 context on a 32 GB card?
#
# The external RTX 5090 test showed 9.13 GiB of KV needed at 262,144 with MTP-3 against
# 8.18 GiB without it - so speculative decoding costs about 0.95 GiB of KV at native context.
# Nobody has measured how that scales with draft depth, which is the difference between
# "native context OR 2x decode" and "native context AND most of the speedup".
#
# Budget: a 5090 reports 31.39 GiB usable and vLLM at utilisation 0.97 budgets 30.44 GiB.
# This box has 95.6 GiB, so 0.3184 reproduces the same absolute budget.
set -u
W=/var/tmp/work
MODEL=${MODEL:-/work/qwen38-ctx}
LEN=${LEN:-262144}
UTIL=${UTIL:-0.3184}
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'

for DEPTH in 0 1 2 3; do
  TAG=mtp$DEPTH
  LOG=$W/ctx-fit-$TAG.log
  rm -f $LOG
  if [ "$DEPTH" = "0" ]; then SPEC=""; else
    SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$DEPTH}'"; fi
  $W/ggrun.sh bash -lc "export VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 \
VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online
exec vllm serve $MODEL --served-model-name m --quantization exl3 \
  --quantization-config '$CFG' --max-model-len $LEN --gpu-memory-utilization $UTIL \
  --kv-cache-dtype fp8 --max-num-seqs 4 --max-num-batched-tokens 2048 $SPEC \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
  --port 8120 --host 127.0.0.1" > $LOG 2>&1 &
  for i in $(seq 1 400); do
    grep -q 'Application startup complete' $LOG && { echo "== depth=$DEPTH FITS $LEN"; break; }
    grep -qE 'ValueError|RuntimeError|Traceback' $LOG && { echo "== depth=$DEPTH DOES NOT FIT"; break; }
    sleep 5
  done
  grep -oE 'GPU KV cache size: [0-9,]+ tokens' $LOG | tail -1
  grep -oE 'Actual usage is .{0,170}' $LOG | tail -1
  grep -oE '\([0-9.]+ GiB KV cache is needed, which is larger than the available[^)]*\)' $LOG | head -1
  grep -oE 'ValueError: .{0,150}' $LOG | head -1
  pkill -f 'port 8120'; sleep 20
done
echo MTPSWEEPDONE
