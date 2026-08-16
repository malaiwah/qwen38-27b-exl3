#!/bin/bash
# Scratch-arena A/B on the physical RTX 5090: baseline overlay vs arena overlay.
# Qualified profile only (receipts/qualification-5090-context.json
# .qualified_profile): 0.955 util + expandable_segments, MTP-3, capture [4].
set -euo pipefail

WORK=~/scratcharena
TOOLS=~/qual5090/tools
CORPUS=~/qual5090/corpus/literary
PORT=8231
URL="http://127.0.0.1:$PORT"
IMAGE="docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
SNAP=/models/ctx-repo/snapshots/c45c273b0d6ef2859cb2d85b36dd52253c80d878
mkdir -p "$WORK/out" "$WORK/logs"

serve() {  # $1 = arm name, $2 = exl3 overlay path
  local arm=$1 overlay=$2
  podman rm -f "arena-$arm" >/dev/null 2>&1 || true
  podman run --rm --name "arena-$arm" \
    --device nvidia.com/gpu=all --ipc=host --network host \
    -v /mnt/vault/llm/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context:/models/ctx-repo:ro \
    -v ~/qual5090/cache/jit:/cache/jit:rw \
    -v ~/qual5090/cache/exl3-online:/cache/exl3-online:rw \
    -v "$overlay":/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py:ro \
    -v "$TOOLS"/vllm-qwen3_5-embed-quant-config.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5.py:ro \
    -v "$TOOLS"/vllm-qwen3_5_mtp-embed-quant-config.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py:ro \
    -e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1 \
    -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    --entrypoint /opt/venv/bin/vllm "$IMAGE" \
    serve "$SNAP" --served-model-name m --quantization exl3 \
    --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}' \
    --max-model-len 262144 --gpu-memory-utilization 0.955 \
    --kv-cache-dtype fp8 --max-num-seqs 1 --max-num-batched-tokens 2048 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --mm-processor-kwargs '{"truncation":false,"max_pixels":8388608}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}' \
    --host 127.0.0.1 --port $PORT \
    > "$WORK/logs/server-$arm.log" 2>&1 &
  echo $! > "$WORK/logs/server-$arm.pid"
  for i in $(seq 1 240); do
    sleep 5
    if curl -sf "$URL/v1/models" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$(cat "$WORK/logs/server-$arm.pid")" 2>/dev/null; then
      echo "server-$arm exited early"; tail -30 "$WORK/logs/server-$arm.log"; return 1
    fi
  done
  echo "server-$arm never became ready"; return 1
}

facts() {  # $1 = arm
  grep -E "GPU KV cache size|Maximum concurrency|Free memory on device|reconstruct scratch arena|Graph capturing finished|Initializing a V1 LLM|peak.*activation|Available KV cache" \
    "$WORK/logs/server-$1.log" | sed 's/^.*] //' | tee "$WORK/out/facts-$1.txt"
}

run_arm() {  # $1 = arm, $2 = overlay
  local arm=$1 overlay=$2
  echo "=== ARM $arm  overlay=$overlay"
  sha256sum "$overlay"
  serve "$arm" "$overlay"
  facts "$arm"
  python3 "$TOOLS"/vision_eval.py --url "$URL" --label "arena-ab-$arm" \
    --out "$WORK/out/vision-$arm.json"
  python3 "$WORK/greedy_probe.py" --url "$URL/v1/completions" \
    --corpus "$CORPUS" --label "$arm" --out "$WORK/out/greedy-$arm.json"
  python3 "$TOOLS"/bench.py --url "$URL/v1/completions" --model m \
    --tokens 256 --concurrency 1 1 1 --prefill-tokens 2048 \
    --label "arena-ab-$arm" > "$WORK/out/bench-$arm.json"
  if [ "$arm" = "arena" ]; then
    python3 "$TOOLS"/longctx.py --url "$URL" --corpus "$CORPUS" \
      --tokens 258941 --depth 0.5 --out "$WORK/out/needle-$arm.json" || true
  fi
  podman stop -t 20 "arena-$arm" >/dev/null 2>&1 || true
  wait "$(cat "$WORK/logs/server-$arm.pid")" 2>/dev/null || true
  sleep 5
}

case "${1:-all}" in
  baseline) run_arm baseline "$TOOLS/vllm-exl3-prefill-dispatch.py" ;;
  arena)    run_arm arena    "$TOOLS/vllm-exl3-scratch-arena.py" ;;
  all)      run_arm baseline "$TOOLS/vllm-exl3-prefill-dispatch.py"
            run_arm arena    "$TOOLS/vllm-exl3-scratch-arena.py" ;;
esac
echo DONE
