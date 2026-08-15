#!/bin/bash
# Does an int8 embedding table unlock native context, and what does it cost?
#
#   1. boot with the overlay and read the engine's own weight accounting
#   2. try native 262,144 on a 5090-sized budget, with MTP-3 and vision, which is the
#      configuration the whole context goal is about
#   3. capture and replay so the fidelity cost is measured, not assumed
set -uo pipefail
W=/var/tmp/work
R=/var/tmp/gg-rootfs/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization
cp $W/gh/exl3-prefill.py $R/exl3.py
find $R/__pycache__ -name 'exl3.*.pyc' -delete 2>/dev/null
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'

try() {  # tag len depth
  local TAG=$1 LEN=$2 DEPTH=$3
  local LOG=$W/embed-$TAG.log; rm -f $LOG
  local SPEC=""
  [ "$DEPTH" != "0" ] && SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$DEPTH}'"
  $W/ggrun.sh bash -lc "export VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 \
VLLM_EXL3_EMBED_BITS=8
exec vllm serve /work/qwen38-ctx --served-model-name m --quantization exl3 \
  --quantization-config '$CFG' --max-model-len $LEN --gpu-memory-utilization 0.3184 \
  --kv-cache-dtype fp8 --max-num-seqs 4 --max-num-batched-tokens 2048 $SPEC \
  --mm-processor-kwargs '{\"truncation\":false}' \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
  --port 8180 --host 127.0.0.1" > $LOG 2>&1 &
  for i in $(seq 1 400); do
    grep -q 'Application startup complete' $LOG && { echo "$TAG len=$LEN mtp=$DEPTH FITS"; break; }
    grep -qE 'ValueError|RuntimeError|Traceback' $LOG && { echo "$TAG len=$LEN mtp=$DEPTH no"; break; }
    sleep 5
  done
  grep -oE 'int8 embedding overlay: .{0,80}' $LOG | tail -1
  grep -oE 'GPU KV cache size: [0-9,]+ tokens' $LOG | tail -1
  grep -oE 'Actual usage is .{0,150}' $LOG | tail -1
  grep -oE '\([0-9.]+ GiB KV cache is needed[^)]*\)' $LOG | head -1
  grep -oE '(ValueError|RuntimeError|AttributeError|TypeError): .{0,130}' $LOG | head -1
  pkill -f 'port 8180'; sleep 18
}

try native-mtp3 262144 3
try native-nomtp 262144 0

echo "=== fidelity with the int8 embedding overlay"
rm -rf /var/tmp/work/kld3/hidden-ctx-embed8
$W/ggrun.sh bash -lc "export VLLM_ALLOW_INSECURE_SERIALIZATION=1 TORCH_EXTENSIONS_DIR=/work/torch-ext \
VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_EMBED_BITS=8
python /work/kld2/fidelity.py capture --model /work/qwen38-ctx --suite /work/kld3/suite \
  --out /work/kld3/hidden-ctx-embed8 --quantization exl3 --quantization-config '$CFG' \
  --gpu-memory-utilization 0.85 || exit 1
python /work/kld2/fidelity.py replay --reference /work/kld3/hidden-bf16 \
  --candidate /work/kld3/hidden-ctx-embed8 --head /work/kld2/lm_head.safetensors \
  --suite /work/kld3/suite --filter analysis --out /work/kld3/report-ctx-embed8.json || exit 1
python /work/kld2/fidelity.py paired --a /work/kld3/report-ctx-embed8.json \
  --b /work/kld3/report-ctx.json --a-label ctx-int8-embed --b-label ctx-bf16-embed \
  --out /work/kld3/paired-ctx-embed8.json || exit 1
python - <<'PY'
import json
D='/work/kld3'
for t,p in (('int8 embeddings','report-ctx-embed8.json'),('BF16 embeddings','report-ctx.json')):
    r=json.load(open(f'{D}/{p}')); b=r['context_bootstrap']
    print(f\"{t:18s} mean={r['token_mean_kld']:.6f} CI=[{b['ci95_low']:.5f},{b['ci95_high']:.5f}] top1={r['top1_agreement']*100:.2f}%\")
q=json.load(open(f'{D}/paired-ctx-embed8.json'))
print(f\"int8 embedding cost diff={q['difference_a_minus_b']:+.8f} CI=[{q['bootstrap_difference']['ci95_low']:+.8f},{q['bootstrap_difference']['ci95_high']:+.8f}] a_wins={q['a_wins']}/{q['contexts']}\")
PY" || echo "EMBED FIDELITY FAILED"
echo EMBEDTESTDONE
