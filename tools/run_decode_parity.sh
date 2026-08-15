#!/bin/bash
# Matched eager-vs-graph decode parity, run sequentially (the two servers cannot
# co-reside on one 96 GB GPU at 0.85 utilisation), plus a cuBLAS-vs-ext.hgemm
# microbenchmark at prefill shapes.
set -u
W=/var/tmp/work
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'

collect() {   # name port exports extra
  local name=$1 port=$2 envs=$3 extra=$4
  $W/ggrun.sh bash -lc "export VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online VLLM_EXL3_ONLINE_TRELLIS_BITS=6 $envs
exec vllm serve /work/Qwen3.8-27B-K5K6 --served-model-name m --quantization exl3 \
  --quantization-config '$CFG' --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --max-num-seqs 8 --kv-cache-dtype fp8 --seed 314159 $extra \
  --port $port --host 127.0.0.1" > $W/par-$name.log 2>&1 &
  for i in $(seq 1 240); do
    grep -q 'Application startup complete' $W/par-$name.log && break
    grep -qE 'ValueError|Traceback' $W/par-$name.log && { echo "$name BOOT FAILED"; grep -oE 'ValueError: .{0,120}' $W/par-$name.log | head -1; return 1; }
    sleep 5
  done
  /home/mbelleau/qwen38-27b/.venv/bin/python $W/kld3/decode_parity.py collect \
    --url http://127.0.0.1:$port/v1 --label $name --tokens 32 --seed 314159 --repeat \
    --out $W/kld3/parity-$name.json
  pkill -f "port $port"; sleep 25
}

collect eager 8090 'VLLM_EXL3_PREFILL_RECONSTRUCT_M=128' '--enforce-eager' || exit 1
collect graph 8091 'VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_GRAPH_DECODE=1' \
  "--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'" || exit 1

/home/mbelleau/qwen38-27b/.venv/bin/python $W/kld3/decode_parity.py compare \
  --a $W/kld3/parity-eager.json --b $W/kld3/parity-graph.json \
  --out $W/kld3/decode-parity-eager-vs-graph.json

echo "== cuBLAS vs ext.hgemm at prefill shapes"
$W/ggrun.sh bash -lc "export TORCH_EXTENSIONS_DIR=/work/torch-ext; python /work/kld3/gemm_cmp.py"
echo PARITYDONE
