#!/bin/bash
# Phase 2: (a) baseline restart control for cross-restart greedy determinism,
# (b) vision suite with the correct /v1 URL on both arms, (c) needle on arena.
set -euo pipefail
cd ~/scratcharena
source /dev/stdin <<'EOF'
EOF
WORK=~/scratcharena
TOOLS=~/qual5090/tools
CORPUS=~/qual5090/corpus/literary
PORT=8231
URL="http://127.0.0.1:$PORT"
IMAGE="docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
SNAP=/models/ctx-repo/snapshots/c45c273b0d6ef2859cb2d85b36dd52253c80d878

serve() {
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
    curl -sf "$URL/v1/models" >/dev/null 2>&1 && return 0
    kill -0 "$(cat "$WORK/logs/server-$arm.pid")" 2>/dev/null || {
      echo "server-$arm exited"; tail -20 "$WORK/logs/server-$arm.log"; return 1; }
  done
  return 1
}

stop_arm() {
  podman stop -t 20 "arena-$1" >/dev/null 2>&1 || true
  wait "$(cat "$WORK/logs/server-$1.pid")" 2>/dev/null || true
  sleep 3
}

# (a) baseline restart control
serve baseline2 "$TOOLS/vllm-exl3-prefill-dispatch.py"
grep -E "GPU KV cache size|Maximum concurrency|Available KV" "$WORK/logs/server-baseline2.log" | sed 's/^.*] //' > "$WORK/out/facts-baseline2.txt"
python3 "$WORK/greedy_probe.py" --url "$URL/v1/completions" --corpus "$CORPUS" \
  --label baseline2 --out "$WORK/out/greedy-baseline2.json"
# within-instance repeat: determinism inside one server
python3 "$WORK/greedy_probe.py" --url "$URL/v1/completions" --corpus "$CORPUS" \
  --label baseline2-repeat --out "$WORK/out/greedy-baseline2-repeat.json"
python3 "$TOOLS"/vision_eval.py --url "$URL/v1" --label arena-ab-baseline2 \
  --out "$WORK/out/vision2-baseline.json"
stop_arm baseline2

# (b)+(c) arena arm: vision + needle with correct /v1
serve arena2 "$TOOLS/vllm-exl3-scratch-arena.py"
grep -E "GPU KV cache size|Maximum concurrency|Available KV|reconstruct scratch arena" "$WORK/logs/server-arena2.log" | sed 's/^.*] //' > "$WORK/out/facts-arena2.txt"
python3 "$WORK/greedy_probe.py" --url "$URL/v1/completions" --corpus "$CORPUS" \
  --label arena2 --out "$WORK/out/greedy-arena2.json"
python3 "$TOOLS"/vision_eval.py --url "$URL/v1" --label arena-ab-arena2 \
  --out "$WORK/out/vision2-arena.json"
python3 "$TOOLS"/longctx.py --url "$URL/v1" --corpus "$CORPUS" \
  --tokens 258941 --depth 0.5 --out "$WORK/out/needle-arena.json" || \
python3 "$TOOLS"/longctx.py --url "$URL" --corpus "$CORPUS" \
  --tokens 258941 --depth 0.5 --out "$WORK/out/needle-arena.json"
stop_arm arena2
echo PHASE2_DONE
