#!/bin/bash
# Prefill attribution matrix. Written as a file rather than inline so the nested
# quoting of --quantization-config cannot break the run (it did, twice).
set -u
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
W=/var/tmp/work

serve() {           # name port "inner exports" [extra serve args...]
  local name=$1 port=$2 envs=$3; shift 3
  local extra="$*"
  $W/ggrun.sh bash -lc "export VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online $envs
exec vllm serve /work/Qwen3.8-27B-K5K6 --served-model-name m --quantization exl3 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 8 \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' $extra \
  --port $port --host 127.0.0.1" > $W/pp-$name.log 2>&1 &
  for i in $(seq 1 240); do
    grep -q 'Application startup complete' $W/pp-$name.log && break
    grep -qE 'ValueError|Traceback' $W/pp-$name.log && break
    sleep 5
  done
  printf '%-14s ' "$name"
  grep -oE 'Actual usage is [0-9.]+ GiB for weight' $W/pp-$name.log | tail -1 | tr -d '\n'
  /home/mbelleau/qwen38-27b/.venv/bin/python $W/bench.py \
    --url http://127.0.0.1:$port/v1/completions --label $name --tokens 128 \
    --concurrency 1 --prefill-tokens 2048 6144 2>&1 | tail -1 > $W/pp-$name.json
  /usr/bin/python3 -c "
import json;d=json.load(open('$W/pp-$name.json'))
print('  TG',[r['aggregate_tok_s'] for r in d['runs']],'PP',[(p['prompt_tokens'],p['prefill_tok_s']) for p in d['prefill']])" \
    2>/dev/null || { echo '  BENCH FAILED'; grep -oE 'ValueError: .{0,120}' $W/pp-$name.log | head -1; }
  pkill -f "port $port"; sleep 20
}

# D: overlay off + MLP dispatch off -> everything trellis-decoded at prefill (old behaviour, no attention overlay)
serve mlp-trellis 8080 'VLLM_EXL3_PREFILL_RECONSTRUCT_M=0'
# B: overlay on + 4x larger prefill chunk -> amortise reconstruct over more rows
serve chunk8192 8081 'VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_ONLINE_TRELLIS_BITS=6' \
  --quantization-config "$CFG" --max-num-batched-tokens 8192
echo MATRIXDONE
