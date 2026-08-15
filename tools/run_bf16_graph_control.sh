#!/bin/bash
# Control: is eager-vs-graph decode drift a property of the EXL3 patch, or of this
# vLLM build in general? Same harness, same prompts, on the unquantised BF16
# checkpoint - which touches none of the EXL3 code path.
set -u
W=/var/tmp/work

boot() {  # name port extra
  local name=$1 port=$2 extra=$3
  $W/ggrun.sh bash -lc "exec vllm serve /models/Qwen3.8-27B --served-model-name m \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 8 \
  --kv-cache-dtype fp8 --seed 314159 $extra --port $port --host 127.0.0.1" \
  > $W/par-$name.log 2>&1 &
  for i in $(seq 1 300); do
    grep -q 'Application startup complete' $W/par-$name.log && return 0
    grep -qE 'ValueError|Traceback' $W/par-$name.log && { echo "$name BOOT FAILED"; grep -oE '(ValueError|RuntimeError): .{0,120}' $W/par-$name.log | head -1; return 1; }
    sleep 5
  done
  echo "$name TIMEOUT"; return 1
}

boot bf16eager 8094 '--enforce-eager' || exit 1
/home/mbelleau/qwen38-27b/.venv/bin/python $W/kld3/decode_parity.py collect \
  --url http://127.0.0.1:8094/v1 --label bf16-eager --tokens 32 --seed 314159 --repeat \
  --out $W/kld3/parity-bf16-eager.json
pkill -f 'port 8094'; sleep 25

boot bf16graph 8095 "--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'" || exit 1
/home/mbelleau/qwen38-27b/.venv/bin/python $W/kld3/decode_parity.py collect \
  --url http://127.0.0.1:8095/v1 --label bf16-graph --tokens 32 --seed 314159 --repeat \
  --out $W/kld3/parity-bf16-graph.json
pkill -f 'port 8095'; sleep 20

/home/mbelleau/qwen38-27b/.venv/bin/python $W/kld3/decode_parity.py compare \
  --a $W/kld3/parity-bf16-eager.json --b $W/kld3/parity-bf16-graph.json \
  --out $W/kld3/decode-parity-bf16-eager-vs-graph.json
echo CONTROLDONE
