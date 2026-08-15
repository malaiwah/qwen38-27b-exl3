#!/bin/bash
# Does taking prefill away from B12X's native K6 kernel pay end to end?
#
# The serialized EXL3 path sends every K6/MCG shard to B12X at all row counts. On this build
# that is 208 attention projections + 64 down_proj + the head - the majority of the model -
# running a decode kernel during prefill. Microbenchmarks say reconstruct+FP8 is 1.76-2.07x
# faster at m=2048 and B12X is ~5x faster at m<=8, so the patch routes by row count inside the
# existing opaque op.
#
# Three servers, identical flags otherwise:
#   base    the module as published (B12X at every row count)
#   route   B12X for decode, reconstruct+hgemm for prefill
#   fp8     B12X for decode, reconstruct+FP8 for prefill
set -u
W=/var/tmp/work
R=/var/tmp/gg-rootfs/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
BENCH=/home/mbelleau/qwen38-27b/.venv/bin/python

run() {  # tag module extra-env
  local TAG=$1 MOD=$2 ENVS=$3
  cp "$MOD" $R/exl3.py
  find $R/__pycache__ -name 'exl3.*.pyc' -delete 2>/dev/null
  local LOG=$W/tp-$TAG.log
  rm -f $LOG
  $W/ggrun.sh bash -lc "export VLLM_EXL3_GRAPH_DECODE=1 $ENVS
exec vllm serve /work/qwen38-ctx --served-model-name m --quantization exl3 \
  --quantization-config '$CFG' --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 --max-num-seqs 8 \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
  --port 8140 --host 127.0.0.1" > $LOG 2>&1 &
  for i in $(seq 1 300); do
    grep -q 'Application startup complete' $LOG && break
    grep -qE 'ValueError|RuntimeError|Traceback' $LOG && { echo "$TAG BOOT FAILED"; \
      grep -oE '(ValueError|RuntimeError): .{0,140}' $LOG | head -1; pkill -f 'port 8140'; sleep 15; return 1; }
    sleep 5
  done
  cd $W
  for i in 1 2 3; do
    $BENCH bench.py --url http://127.0.0.1:8140/v1/completions --label $TAG-run$i \
      --tokens 256 --concurrency 1 4 8 --prefill-tokens 2048 6144 2>&1 | tail -1 \
      | /usr/bin/python3 -c "
import json,sys
d=json.load(sys.stdin)
print(d['label'],'TG',[r['aggregate_tok_s'] for r in d['runs']],
      'PP',[(p['prompt_tokens'],p['prefill_tok_s']) for p in d['prefill']])"
  done
  pkill -f 'port 8140'; sleep 18
}

cp $R/exl3.py $W/exl3.py.before-b12x-routing
run base  $W/exl3.py.before-b12x-routing 'VLLM_EXL3_PREFILL_RECONSTRUCT_M=128'
run route $W/gh/exl3-prefill.py           'VLLM_EXL3_PREFILL_RECONSTRUCT_M=128'
run fp8   $W/gh/exl3-prefill.py           'VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_PREFILL_FP8=1'
echo TPDONE
