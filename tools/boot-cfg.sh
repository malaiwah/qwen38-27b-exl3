#!/bin/bash
# Boot a manual instance with env overrides and ASSERT the effective config.
# Prevents the systemd unit from racing us for the port/container name.
set -uo pipefail
UTIL="${1}"; MML="${2}"; MNBT="${3}"
systemctl --user stop qwen38-27b.service 2>/dev/null
sleep 1; podman rm -f qwen38-27b >/dev/null 2>&1; sleep 1
GPU_MEMORY_UTILIZATION="$UTIL" MAX_MODEL_LEN="$MML" MAX_NUM_BATCHED_TOKENS="$MNBT" \
  /home/mbelleau/run-qwen38-27b.sh >/dev/null 2>&1
booted=no
for i in $(seq 1 50); do
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && booted=yes && break
  s=$(podman inspect -f '{{.State.Status}}' qwen38-27b 2>/dev/null || echo gone)
  [ "$s" != running ] && break
  sleep 10
done
if [ "$booted" != yes ]; then
  echo "  BOOT FAILED (util=$UTIL mml=$MML mnbt=$MNBT)"
  podman logs qwen38-27b 2>&1 | grep -oE "(ValueError|RuntimeError|max seq len|smaller KV cache|Try increasing).{0,120}" | sort -u | head -3 | sed 's/^/    /'
  exit 1
fi
# assert effective config from the engine's own arg dump
eff=$(podman logs qwen38-27b 2>&1 | grep -oE "'(max_model_len|gpu_memory_utilization|max_num_batched_tokens)': [0-9.]+" | sort -u | tr '\n' ' ')
ctx=$(curl -s http://localhost:8000/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0].get("max_model_len"))')
kv=$(podman logs qwen38-27b 2>&1 | grep -oE "Available KV cache memory: [0-9.]+ GiB" | tail -1)
act=$(podman logs qwen38-27b 2>&1 | grep -oE "Actual usage is .{0,120}" | tail -1)
echo "  EFFECTIVE: $eff"
echo "  ctx=$ctx | $kv"
echo "  $act"
[ "$ctx" = "$MML" ] || echo "  !! ASSERT FAIL: ctx=$ctx but requested MML=$MML"
