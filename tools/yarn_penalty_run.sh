#!/bin/bash
# Static-YaRN short-context penalty: boot each arm, run the identical probe set,
# tear down.  One variable between the arms: the rope configuration.
#
# Arm NATIVE: shipped config.json rope (rope_type "default"), --max-model-len 262144.
# Arm YARN:   Qwen's official 1M recipe (rope_type "yarn", factor 4.0,
#             original_max_position_embeddings 262144), --max-model-len 1000000,
#             VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
#
# Everything else -- weights, engine, quantization, KV dtype, cache modes,
# cudagraph mode, memory utilisation, sequence and token budgets, sampling, seed,
# prompts -- is pinned identical.  Timings recorded anywhere in this run are
# proot-inflated and are NOT performance results.
set -euo pipefail

RES=/home/mbelleau/research
W=/var/tmp/work/yarn
MODEL=/models/Qwen3.8-27B-EXL3-K5K6-hydrated
PORT=8231
BASE="http://127.0.0.1:$PORT"
QCFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
YARN_OVERRIDE='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
COMMON_ENV='VLLM_USE_V2_MODEL_RUNNER=1 VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128'

mkdir -p "$W"

serve_argv() {  # $1 = NATIVE|YARN ; prints one argv element per line
  printf '%s\n' serve "$MODEL" \
    --served-model-name qwen38 \
    --quantization exl3 \
    --quantization-config "$QCFG" \
    --kv-cache-dtype fp8 \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser qwen3 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --max-logprobs 20 \
    --seed 314159 \
    --host 127.0.0.1 --port "$PORT"
  case $1 in
    YARN)     printf '%s\n' --hf-overrides "$YARN_OVERRIDE" --max-model-len 1000000 ;;
    NATIVE1M) printf '%s\n' --max-model-len 1000000 ;;
    *)        printf '%s\n' --max-model-len 262144 ;;
  esac
}

# Control arm NATIVE1M isolates the window from the rope: it is arm NATIVE's rope
# configuration served at the YaRN arm's 1,000,000-token window, so the KV pool,
# block tables and long-max-model-len path all match the YaRN arm while the rope
# tables match the NATIVE arm.  If NATIVE-vs-NATIVE1M is null, the whole
# NATIVE-vs-YARN divergence is attributable to rope and nothing else.
arm_env() {
  case $1 in
    YARN|NATIVE1M) printf '%s VLLM_ALLOW_LONG_MAX_MODEL_LEN=1' "$COMMON_ENV" ;;
    *)             printf '%s' "$COMMON_ENV" ;;
  esac
}

record_argv() {  # $1 = arm
  local arm=$1
  serve_argv "$arm" > "$W/argv-$arm.txt"
  ARM="$arm" GGENV="$(arm_env "$arm")" ARGVFILE="$W/argv-$arm.txt" \
    python3 -c 'import json, os
argv = [l.rstrip("\n") for l in open(os.environ["ARGVFILE"])]
print(json.dumps({"arm": os.environ["ARM"],
                  "runner": "/home/mbelleau/research/tools/ggrun.sh (proot rootfs, no container runtime)",
                  "gg_env": os.environ["GGENV"],
                  "executable": "vllm",
                  "argv": argv}, indent=1))' > "$W/argv-$arm.json"
}

boot() {  # $1 = arm
  local arm=$1
  local log="$W/serve-$arm.log"
  echo "=== booting arm $arm ==="
  mapfile -t ARGV < <(serve_argv "$arm")
  rm -f "$log"
  GG_ENV="$(arm_env "$arm")" setsid "$RES/tools/ggrun.sh" vllm "${ARGV[@]}" > "$log" 2>&1 &
  SERVER_PGID=$!
  local waited=0
  while (( waited < 1800 )); do
    if curl -sf -m 5 "$BASE/health" >/dev/null 2>&1; then
      echo "health ok after ${waited}s"
      curl -sf -m 10 "$BASE/v1/models" > "$W/models-$arm.json"
      grep -c "Application startup complete" "$log" || true
      return 0
    fi
    if ! kill -0 "$SERVER_PGID" 2>/dev/null; then
      echo "SERVER DIED, tail of log:"; tail -40 "$log"; return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
  echo "TIMEOUT waiting for health"; tail -40 "$log"; return 1
}

teardown() {
  echo "=== teardown ==="
  pkill -f "gg-rootfs" 2>/dev/null || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 8
  for _ in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [[ ${used:-9999} -lt 2000 ]] && { echo "gpu free (${used} MiB)"; return 0; }
    pkill -9 -f "gg-rootfs" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 5
  done
  echo "WARNING: gpu memory still held: ${used} MiB"
}

RUN=${1:-all}

if [[ $RUN == all || $RUN == native ]]; then
  record_argv NATIVE
  teardown
  boot NATIVE
  python3 "$RES/tools/yarn_probe_run.py" generate --base "$BASE" \
    --probes "$W/probes.json" --out "$W/gen-native.json" 2>&1 | tail -5
  python3 "$RES/tools/yarn_probe_run.py" ladder --base "$BASE" \
    --probes "$W/probes.json" --forced "$W/gen-native.json" \
    --out "$W/raw-NATIVE-p1.json" --arm NATIVE --pass-label p1 2>&1 | tail -3
  python3 "$RES/tools/yarn_probe_run.py" ladder --base "$BASE" \
    --probes "$W/probes.json" --forced "$W/gen-native.json" \
    --out "$W/raw-NATIVE-p2.json" --arm NATIVE --pass-label p2-replicate 2>&1 | tail -3
  teardown
fi

if [[ $RUN == all || $RUN == yarn ]]; then
  record_argv YARN
  teardown
  boot YARN
  python3 "$RES/tools/yarn_probe_run.py" ladder --base "$BASE" \
    --probes "$W/probes.json" --forced "$W/gen-native.json" \
    --out "$W/raw-YARN-p1.json" --arm YARN --pass-label p1 2>&1 | tail -3
  python3 "$RES/tools/yarn_probe_run.py" generate --base "$BASE" \
    --probes "$W/probes.json" --out "$W/gen-yarn.json" 2>&1 | tail -3
  teardown
fi

if [[ $RUN == all || $RUN == control ]]; then
  record_argv NATIVE1M
  teardown
  boot NATIVE1M
  python3 "$RES/tools/yarn_probe_run.py" ladder --base "$BASE" \
    --probes "$W/probes.json" --forced "$W/gen-native.json" \
    --out "$W/raw-NATIVE1M-p1.json" --arm NATIVE1M --pass-label window-control 2>&1 | tail -3
  teardown
fi

# Cross-boot replicate: arm NATIVE again, from a cold engine and a cold prefix
# cache, at the SAME 262,144 window.  Separates "rebooting the server" from
# "changing the window", which the NATIVE1M control alone cannot do.
if [[ $RUN == all || $RUN == reboot ]]; then
  teardown
  boot NATIVE
  python3 "$RES/tools/yarn_probe_run.py" ladder --base "$BASE" \
    --probes "$W/probes.json" --forced "$W/gen-native.json" \
    --out "$W/raw-NATIVE-p3.json" --arm NATIVE --pass-label p3-cross-boot 2>&1 | tail -3
  teardown
fi
# Cross-boot free-running control: the free-running greedy comparison against the
# YaRN arm is only meaningful next to the divergence a plain reboot of the SAME
# arm produces.  Same launch line, same window, fresh engine, greedy generation.
if [[ $RUN == all || $RUN == regen ]]; then
  teardown
  boot NATIVE
  python3 "$RES/tools/yarn_probe_run.py" generate --base "$BASE" \
    --probes "$W/probes.json" --out "$W/gen-native-boot4.json" 2>&1 | tail -3
  teardown
fi


echo "ALL DONE"
