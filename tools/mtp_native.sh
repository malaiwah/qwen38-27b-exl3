#!/bin/bash
# Can MTP and native context coexist if the draft head's duplicate embedding table goes to
# int4 while the main table stays int8? The draft's output is verified before acceptance, so
# it can absorb an error the input path cannot.
set -u
W=/var/tmp/work
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
for DEPTH in 3 2 1; do
  LOG=$W/mtpnative-$DEPTH.log; rm -f $LOG
  $W/ggrun.sh bash -lc "export VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 \
VLLM_EXL3_EMBED_BITS=8 VLLM_EXL3_MTP_EMBED_BITS=4
exec vllm serve /work/qwen38-ctx --served-model-name m --quantization exl3 \
  --quantization-config '$CFG' --max-model-len 262144 --gpu-memory-utilization 0.3184 \
  --kv-cache-dtype fp8 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$DEPTH}' \
  --mm-processor-kwargs '{\"truncation\":false}' \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
  --port 8210 --host 127.0.0.1" > $LOG 2>&1 &
  for i in $(seq 1 420); do
    grep -q 'Application startup complete' $LOG && { echo "mtp=$DEPTH FITS 262144"; break; }
    grep -qE 'ValueError|RuntimeError|Traceback' $LOG && { echo "mtp=$DEPTH no"; break; }
    sleep 5
  done
  grep -oE 'int(4|8) embedding overlay: .{0,70}' $LOG | tail -2
  grep -oE 'GPU KV cache size: [0-9,]+ tokens' $LOG | tail -1
  grep -oE 'Actual usage is .{0,120}' $LOG | tail -1
  grep -oE '\([0-9.]+ GiB KV cache is needed[^)]*\)' $LOG | head -1
  pkill -f 'port 8210'; sleep 18
done
echo MTPNATIVEDONE
