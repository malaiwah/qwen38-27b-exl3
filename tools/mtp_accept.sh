#!/bin/bash
# Does an int4 draft embedding still draft well? Acceptance is the metric that matters: the
# draft's errors are verified away, so the only cost of a coarser draft table is a lower
# acceptance rate, which shows up directly as throughput.
set -u
W=/var/tmp/work
BENCH=/home/mbelleau/qwen38-27b/.venv/bin/python
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
run() {  # tag mtpbits
  local TAG=$1 MB=$2
  local LOG=$W/accept-$TAG.log; rm -f $LOG
  $W/ggrun.sh bash -lc "export VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 \
VLLM_EXL3_EMBED_BITS=8 VLLM_EXL3_MTP_EMBED_BITS=$MB
exec vllm serve /work/qwen38-ctx --served-model-name m --quantization exl3 \
  --quantization-config '$CFG' --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 --max-num-seqs 4 \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
  --port 8220 --host 127.0.0.1" > $LOG 2>&1 &
  for i in $(seq 1 300); do
    grep -q 'Application startup complete' $LOG && break
    grep -qE 'ValueError|RuntimeError|Traceback' $LOG && { echo "$TAG BOOT FAILED"; pkill -f 'port 8220'; sleep 15; return 1; }
    sleep 5
  done
  cd $W && $BENCH bench.py --url http://127.0.0.1:8220/v1/completions --label $TAG \
    --tokens 256 --concurrency 1 4 2>&1 | tail -1 | /usr/bin/python3 -c "
import json,sys;d=json.load(sys.stdin)
print(d['label'],'TG',[r['aggregate_tok_s'] for r in d['runs']])"
  curl -sS http://127.0.0.1:8220/metrics 2>/dev/null | grep -E "^vllm:spec_decode_num_(draft|accepted)_tokens_total" | \
    /usr/bin/python3 -c "
import sys
vals={}
for line in sys.stdin:
    k,v=line.rsplit(' ',1)
    vals['draft' if 'draft' in k else 'accepted']=float(v)
if vals.get('draft'):
    print(f\"  drafted={vals['draft']:.0f} accepted={vals.get('accepted',0):.0f} \"
          f\"acceptance={100*vals.get('accepted',0)/vals['draft']:.1f}%\")"
  pkill -f 'port 8220'; sleep 18
}
run draft-int8 8
run draft-int4 4
echo ACCEPTDONE
