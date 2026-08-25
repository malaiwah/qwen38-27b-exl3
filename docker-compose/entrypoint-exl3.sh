#!/bin/bash
set -e
MODEL="${MODEL_PATH:-/root/.cache/huggingface/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated}"
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen3.8-27B}" \
  --port "${PORT:-8000}" \
  --quantization "${QUANT_METHOD:-exl3}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"preserve_thinking":true}' \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --attention-backend TRITON_ATTN \
  $([ -n "${SPEC_CONFIG}" ] && echo "--speculative-config ${SPEC_CONFIG}")
