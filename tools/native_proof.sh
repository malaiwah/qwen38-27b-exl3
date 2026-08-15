#!/bin/bash
# Prove the native-context configuration by using it: needle retrieval near 262,144 and a
# multimodal check on the same server.
set -u
W=/var/tmp/work
BENCH=/home/mbelleau/qwen38-27b/.venv/bin/python
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
LOG=$W/native-serve.log; rm -f $LOG
$W/ggrun.sh bash -lc "export VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_EMBED_BITS=8
exec vllm serve /work/qwen38-ctx --served-model-name m --quantization exl3 \
  --quantization-config '$CFG' --max-model-len 262144 --gpu-memory-utilization 0.3184 \
  --kv-cache-dtype fp8 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --mm-processor-kwargs '{\"truncation\":false}' \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
  --port 8200 --host 127.0.0.1" > $LOG 2>&1 &
for i in $(seq 1 400); do
  grep -q 'Application startup complete' $LOG && break
  grep -qE 'ValueError|RuntimeError|Traceback' $LOG && { echo "SERVE FAILED"; pkill -f 'port 8200'; exit 1; }
  sleep 5
done
grep -oE 'GPU KV cache size: [0-9,]+ tokens' $LOG | tail -1
for DEPTH in 0.1 0.5 0.9; do
  $BENCH $W/kld3/longctx.py --url http://127.0.0.1:8200/v1 \
    --corpus /var/tmp/work/kld3/corpus/literary --tokens 258000 --depth $DEPTH \
    --out $W/kld3/native-needle-$DEPTH.json || echo "needle @$DEPTH failed"
done
$BENCH $W/kld3/vision_eval.py --url http://127.0.0.1:8200/v1 --label ctx-embed8 --cases 10 \
  --out $W/kld3/vision-ctx-embed8.json 2>&1 | tail -2
pkill -f 'port 8200'; sleep 15
echo NATIVEPROOFDONE
