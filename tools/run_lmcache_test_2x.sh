#!/bin/bash
# LMCache L1 (CPU DRAM) ladder on the 2x RTX PRO 6000 Blackwell host (151.185.34.111).
#
# This is the 2x PORT of tools/run_lmcache_test.sh. That script is the record of the
# 5090/AIBoss run (receipts/lmcache-reuse-test.json) and is deliberately left untouched.
# The probe, the frozen prompt set and every pre-registered threshold are reused VERBATIM:
#
#   probe          tools/apc_poison_probe.py           sha256 a96168f9…  (unchanged)
#   prompts        receipts/lmcache-frozen-prompts.json  prompts_sha256 9978404b…
#   thresholds     THRESHOLDS dict inside the probe     (unchanged, asserted per arm)
#   fail rule      corruption detector fires OR any of A1/A2/A3 wrong; A4 diagnostic-only
#   baseline       L1cold 7/38 and L2warm 7/38 corrupted on the 5090
#
# WHAT THIS RUN ADDS OVER THE 5090 RUN
#   1. LMCache L1 is the point of the exercise: --l1-size-gb is LMCache's CPU pinned-DRAM
#      pool ("CPU pinned-DRAM L1 memory manager", lmcache/v1/distributed/memory_manager/
#      l1_memory_manager.py:2) i.e. the local_cpu / max_local_cpu_size tier. The 5090 run
#      used 12 GiB because that card's host had little spare DRAM. Here it is sized from
#      the 2x host's 314 GB (see LMC_L1_GB below).
#   2. Our fork's scheduler gate patch (local-inference-lab/vllm#403, patch file
#      patches/gg-vllm-hybrid-divergent-hit-gate.patch) is applied as a READ-ONLY OVERLAY
#      MOUNT over the vendored scheduler.py — never baked into the image — exactly the way
#      receipts/tb21-image-sr1.json layer A mounts tools/vllm-exl3-scratch-arena.py.
#      Both the pre-mount and post-mount sha256 and a sentinel symbol are asserted, and
#      the sentinel is checked in the LOADED BYTECODE, not only in the file, because the
#      image ships a stale __pycache__ entry for this module.
#   3. An UNPATCHED baseline arm pair (U1cold/U2warm) runs the same ladder on the same
#      host with the overlay absent. Without it a clean patched result on a different
#      model edition could not be distinguished from a trigger that never fired here.
#   4. L3restart is run TWICE, as receipts/lmcache-fix.json.unproven_until_gpu_ladder
#      demands: once over a FRESH poison-free L2 (written by the patched arms) and once
#      over the RETAINED poisoned L2 (written by the unpatched arms).
#
# NAMED DEVIATIONS FROM THE 5090 PRE-REGISTRATION (platform change, stated not hidden)
#   * model      : hydrated edition /models/qwen38-hyd, not the K5K6-context edition
#   * window     : 262,144 (the shipping 2x profile, docs/46 §18), not 196,608
#   * batching   : --max-num-seqs 64 --max-num-batched-tokens 8192 (shipping), not 1/2048.
#                  The probe still issues at concurrency 1, so one request is in flight.
#   * parallelism: --tensor-parallel-size 2. LMCacheMPConnector therefore registers two
#                  GPU clients; the MP corruption path needs more than one rank, which is
#                  why this must run on the 2x box.
#   * runtime    : docker 29.5.1 (containerd image store), not podman
#   * chat surface: --reasoning-parser/--tool-call-parser are omitted; the probe drives
#                  /v1/completions with raw token ids, so they cannot affect any arm.
#   Every arm below is identical to every other arm except for the one variable named in
#   its ARM_DESC.
#
# PHASES
#   stage            identity: image, overlay shas + sentinel, probe, prompts, wrapper,
#                    lmcache version, ForceP2P override, port collisions, DRAM budget
#   control          arm L0        : overlay mounted, LMCACHE_MODE=off, 38 requests
#   unpatched        arms U1cold/U2warm : NO overlay, LMCache live, wiped L2, 38+38
#   patched          arms L1cold/L2warm : overlay mounted, LMCache live, wiped L2, 38+38,
#                    then the reuse economics pass, then HOLDS the server up (staying
#                    attached to its log) until $W/gate-done appears so the five-check
#                    fidelity gate can be run against it from the driver VM, then flushes
#                    L1->L2, retains the clean L2, stops and scrapes
#   restart-fresh    arm L3fresh   : new server over the RETAINED CLEAN L2, 38 requests
#   restart-poison   arm L3poison  : new server over the RETAINED POISONED L2, 38 requests
#   receipt          assemble out/assembled.json
set -euo pipefail

W=${W:-/home/ubuntu/lmcachetest}
OUT=$W/out
LOGS=$W/logs
TOOLS=$W/tools
OVL=$W/overlay
L2=$W/l2                  # the directory every LMCache arm mounts at /l2
L2_POISON=$W/l2-poisoned  # retained copy of the UNPATCHED arms' stores
L2_CLEAN=$W/l2-clean      # retained copy of the PATCHED arms' stores

PORT=${PORT:-8000}
BIND=${BIND:-0.0.0.0}
MODELS=${MODELS:-/home/ubuntu/models}
MODEL=${MODEL:-/models/qwen38-hyd}
SERVED=${SERVED:-qwen38}

IMAGE=${IMAGE:-vllm:gg-r34-tb21-sr1}
IMAGE_ID=${IMAGE_ID:-sha256:237a50250eb8bcaf4f9a88d688a26753db4d2a73d902a4ef48e4ad23d958df3e}
VENV=/opt/venv/lib/python3.12/site-packages/vllm
SCHED_IN_IMAGE=$VENV/v1/core/sched/scheduler.py
WRAPPER=/usr/local/bin/glm52-lmcache-wrapper.sh

# Overlay identity. base = the sr1 image's own post-#52530 scheduler
# (receipts/tb21-image-sr1.json patch_layers[C].post_patch_sha256), gated = base with
# patches/gg-vllm-hybrid-divergent-hit-gate.patch applied at -p1 --fuzz=0 (offset +37,
# exactly as receipts/lmcache-fix.json.fixes.gg_vllm_scheduler_patch predicted).
SCHED_BASE_SHA=e98b7227e0b66eea137ed6ccbc9ee99bd43485892c11c648284c30cde206ed05
SCHED_GATED_SHA=b38981d89744a566e30a64a945d6633ef30f007c7158681d428ffa759b3ea779
GATE_PATCH_SHA=5f9ad10bedff87c4f37ce61bfe9be5cc9ed84dce99d9d3685c54b0341529839f
SENTINEL=supports_divergent_local_hybrid_hits

PROBE_SHA=a96168f9b0c32387fe0cd2b180cb23ff67061b6dbe8f46c1423041c904e7aeb6
PROMPTS_SHA=9978404b0ce2d032ebf762e17335d7b5a37dbc424f57f61db66ecf3f946f39df

# ------------------------------------------------------------------ LMCache L1: CPU DRAM
# Host DRAM 314 GB total. Budget: 21 GB of weights in page cache, ~40 GB resident for the
# engine core plus two TP worker processes, ~10 GB OS/agents. 160 GiB (172 GB) of pinned
# L1 therefore leaves ~70 GB unallocated headroom, and lazy allocation (--l1-use-lazy is
# the default) means only --l1-init-size-gb is touched at startup and the pool grows on
# demand, so an over-estimate cannot OOM the box at boot. This is the "ample DRAM on the
# vLLM hosts" configuration an API host would actually deploy.
LMC_L1_GB=${LMC_L1_GB:-160}
LMC_L1_INIT_GB=${LMC_L1_INIT_GB:-8}
# L2 disk: 886 GB free on /; 256 GiB is the wrapper's own default and is far larger than
# this workload's whole store (38 requests x <=15.5k tokens).
LMC_L2_GB=${LMC_L2_GB:-256}
LMC_L2_WORKERS=${LMC_L2_WORKERS:-8}
LMC_CPU_WORKERS=${LMC_CPU_WORKERS:-8}
# One GPU affinity worker per TP rank, so rank transfers are not serialised.
LMC_GPU_WORKERS=${LMC_GPU_WORKERS:-2}
# Chunk 1600 is pre-registered: it is the mamba block the frozen prompts are built around
# (prompts.mamba_block_tokens == 1600). Changing it would void the comparison.
LMC_CHUNK=${LMC_CHUNK:-1600}
# nv-hostengine (nvidia-dcgm) owns 127.0.0.1:5555 on this host and must not be disturbed,
# so the three LMCache ports are pinned explicitly to the 5090 run's numbers instead of
# being derived from PORT.
LMC_ZMQ_PORT=${LMC_ZMQ_PORT:-5796}
LMC_HTTP_PORT=${LMC_HTTP_PORT:-8330}
LMC_PROM_PORT=${LMC_PROM_PORT:-9331}

# ------------------------------------------------------------------ serving profile
# The shipping 2x TP2 profile (~/bin/serve-tp2.sh, docs/46 §18), minus the chat-surface
# parsers the probe cannot reach.
QUANT_CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
SPEC='{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,4,3],[5,64,1]]}'
MM_CFG='{"truncation":false}'
COMPILE='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
MAXLEN=${MAXLEN:-262144}
GPU_UTIL=${GPU_UTIL:-0.92}
SEQS=${SEQS:-64}
BATCHED=${BATCHED:-8192}
TP=${TP:-2}
READY_TIMEOUT=${READY_TIMEOUT:-2400}
# how long the patched phase holds the LMCache server up for the fidelity gate, in 5s ticks
GATE_HOLD_TIMEOUT=${GATE_HOLD_TIMEOUT:-720}

CONTAINER=
SAMPLER_PID=
declare -a docker_argv=() serve_argv=()

mkdir -p "$OUT" "$LOGS" "$TOOLS" "$OVL" "$L2"

die() { echo "FATAL: $*" >&2; exit 2; }
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

cleanup() {
  local rc=$?
  [[ -n $SAMPLER_PID ]] && kill "$SAMPLER_PID" 2>/dev/null
  if [[ -n $CONTAINER ]]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi
  return $rc
}
trap cleanup EXIT INT TERM

verify_image() {
  local got
  got=$(docker image inspect -f '{{.Id}}' "$IMAGE") || die "image $IMAGE missing"
  [[ $got == "$IMAGE_ID" ]] || die "image $IMAGE id $got != $IMAGE_ID"
  say "image identity verified: $IMAGE ($got)"
}

# The overlay assertion. Fail-closed in the same shape as tb21-image-sr1's build-time
# gates: patch file sha, pre-mount file sha, post-mount file sha, sentinel discrimination
# (0 hits in base, >0 in overlay) and the sentinel present in the LOADED BYTECODE.
verify_overlay() {
  echo "$GATE_PATCH_SHA  $TOOLS/gg-vllm-hybrid-divergent-hit-gate.patch" | sha256sum -c - ||
    die "gate patch file drifted"
  echo "$SCHED_BASE_SHA  $OVL/scheduler.base.py" | sha256sum -c - ||
    die "extracted base scheduler is not the sr1 scheduler"
  echo "$SCHED_GATED_SHA  $OVL/scheduler.patched.py" | sha256sum -c - ||
    die "gated overlay drifted"
  local base_hits
  base_hits=$(grep -c "$SENTINEL" "$OVL/scheduler.base.py" || true)
  [[ $base_hits == 0 ]] || die "sentinel $SENTINEL already present in the base file ($base_hits hits): it does not discriminate"
  grep -q "$SENTINEL" "$OVL/scheduler.patched.py" || die "sentinel $SENTINEL absent from the overlay"
  # in-image: mounted bytes, and the gate present in the executed bytecode
  docker run --rm --network none \
    -v "$OVL/scheduler.patched.py:$SCHED_IN_IMAGE:ro" \
    --entrypoint /bin/bash "$IMAGE" -lc "
      set -eu
      echo '$SCHED_GATED_SHA  $SCHED_IN_IMAGE' | sha256sum -c -
      /opt/venv/bin/python - <<'PY'
import vllm.v1.core.sched.scheduler as s

def consts(code, seen=None):
    seen = seen if seen is not None else set()
    for c in code.co_consts:
        if isinstance(c, str):
            seen.add(c)
        elif hasattr(c, 'co_consts'):
            consts(c, seen)
    return seen

live = consts(s.Scheduler.schedule.__code__)
assert '$SENTINEL' in live, 'gate absent from the LOADED BYTECODE (stale .pyc?)'
print('OVERLAY_BYTECODE_GATE: present')
PY
    " >"$OUT/overlay-identity.txt" 2>&1 || {
    tail -20 "$OUT/overlay-identity.txt" >&2
    die "overlay verification failed inside the image"
  }
  grep -q 'OVERLAY_BYTECODE_GATE: present' "$OUT/overlay-identity.txt" ||
    die "overlay bytecode gate assertion did not run"
  say "overlay verified: $SCHED_GATED_SHA, sentinel in loaded bytecode"
}

start_sampler() {
  nvidia-smi --query-gpu=index,timestamp,memory.total,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -lms 1000 >"$LOGS/gpu-$1.csv" 2>/dev/null &
  SAMPLER_PID=$!
}

startup_facts() {
  local log=$1 tag=$2
  {
    grep -oE 'Setting attention block size to [0-9]+ tokens[^.]*.' "$log" || true
    grep -oE 'Available KV cache memory: [0-9.]+ GiB' "$log" || true
    grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$log" || true
    grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$log" || true
    grep -oE "'enable_prefix_caching': [A-Za-z]+" "$log" | head -1 || true
    grep -oE "'mamba_cache_mode': '[a-z]+'" "$log" | head -1 || true
    grep -oE "'max_num_seqs': [0-9]+" "$log" | head -1 || true
    grep -oE 'num_spec_tokens=[0-9]+' "$log" | head -1 || true
    grep -oE 'Using [A-Z_]+ attention backend[^.]*' "$log" | head -1 || true
    grep -oE 'Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds' "$log" || true
    grep -oE 'Prefix caching in Mamba cache .[a-z]+. mode is[^.]*.' "$log" || true
    grep -oE 'LMCache ready: mode=[^ ]+ L1=[^ ]+ chunk=[^ ]+ L2=[^ ]+' "$log" || true
    grep -oE 'LMCache disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' "$log" || true
    grep -iE 'kv_connector|kv_transfer|KVConnector|LMCacheMP' "$log" | head -8 || true
  } | sed 's/^ *//' | sort -u >"$OUT/startup-facts-$tag.txt"
  cat "$OUT/startup-facts-$tag.txt" >&2
}

wait_ready() {
  local log=$1 i seen=0
  for ((i = 0; i < READY_TIMEOUT; i++)); do
    if curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      say "server ready after ${i}s"
      return 0
    fi
    if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
      seen=1
      [[ $(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null) == false ]] && {
        say "container $CONTAINER exited before readiness"
        return 1
      }
    elif ((seen == 1)); then
      say "container $CONTAINER disappeared before readiness"
      return 1
    elif ((i > 120)); then
      say "container $CONTAINER was never created"
      return 1
    fi
    sleep 1
  done
  say "server not ready within ${READY_TIMEOUT}s"
  return 1
}

gpu_used_total() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    awk '{s += $1} END {print s + 0}'
}

stop_server() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  CONTAINER=
  [[ -n $SAMPLER_PID ]] && { kill "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=; }
  local i
  for ((i = 0; i < 180; i++)); do
    (( $(gpu_used_total) < 1500 )) && return 0
    sleep 1
  done
  say "warning: GPU memory still held after teardown ($(gpu_used_total) MiB across 2 GPUs)"
}

require_free_gpu() {
  local used
  used=$(gpu_used_total)
  ((used < 3000)) || die "GPUs busy (${used} MiB used across both); another owner still holds them"
}

# ------------------------------------------------------------------ arm plumbing
declare -a ARM_ENV=()
ARM_DESC=
ARM_OVERLAY=
ARM_L2DIR=

lmcache_env() {
  ARM_ENV=(
    -e LMCACHE_MODE=disk
    -e LMCACHE_CHUNK_SIZE="$LMC_CHUNK"
    -e LMCACHE_L1_GB="$LMC_L1_GB"
    -e LMCACHE_L1_INIT_GB="$LMC_L1_INIT_GB"
    -e LMCACHE_L2_GB="$LMC_L2_GB"
    -e LMCACHE_L2_WORKERS="$LMC_L2_WORKERS"
    -e LMCACHE_MAX_CPU_WORKERS="$LMC_CPU_WORKERS"
    -e LMCACHE_MAX_GPU_WORKERS="$LMC_GPU_WORKERS"
    -e LMCACHE_L2_PATH=/l2/data
    -e LMCACHE_PORT="$LMC_ZMQ_PORT"
    -e LMCACHE_HTTP_PORT="$LMC_HTTP_PORT"
    -e LMCACHE_PROMETHEUS_PORT="$LMC_PROM_PORT"
    -e LMCACHE_LOG="/lmclogs/lmcache-mp-$1.log"
    -e LMCACHE_START_TIMEOUT=600
  )
}

configure_arm() {
  ARM_ENV=()
  ARM_OVERLAY=yes
  ARM_L2DIR=$L2
  local l1="L1 CPU pinned DRAM ${LMC_L1_GB} GiB (init ${LMC_L1_INIT_GB} GiB, lazy), L2 fs_native ${LMC_L2_GB} GiB, chunk ${LMC_CHUNK}, ${LMC_GPU_WORKERS} GPU workers, LRU 0.90/0.10, prefetch retain"
  case $1 in
  L0)
    ARM_DESC="control: sr1 image + gate overlay, shipping 2x TP2 profile (prefix caching + align, MTP-3, fp8 KV), LMCACHE_MODE=off — no LMCache; exactly one variable from the LMCache arms"
    ARM_ENV=(-e LMCACHE_MODE=off)
    ;;
  U1cold | U2warm)
    ARM_DESC="UNPATCHED baseline: NO gate overlay, LMCache kv_both live ($l1). Establishes whether this host + hydrated edition reproduces the 5090's 7/38 divergent-hit corruption at all; without it a clean patched arm is uninterpretable."
    lmcache_env unpatched
    ARM_OVERLAY=no
    ;;
  L1cold | L2warm)
    ARM_DESC="GATE PATCHED (local-inference-lab/vllm#403 overlay mounted read-only), LMCache kv_both live ($l1). The prediction under test: 0/38."
    lmcache_env patched
    ;;
  L3fresh)
    ARM_DESC="GATE PATCHED, NEW server over the RETAINED FRESH (poison-free) L2 written by the patched cold/warm passes: vLLM's own prefix cache and LMCache's L1 start empty, so any reuse is an LMCache L2 retrieval. Expected clean."
    lmcache_env restart-fresh
    ;;
  L3poison)
    ARM_DESC="GATE PATCHED, NEW server over the RETAINED POISONED L2 written by the UNPATCHED arms: receipts/lmcache-fix.json predicts this stays dirty until the directory is wiped, because the stores themselves were computed under a corrupted boundary state."
    lmcache_env restart-poison
    ARM_L2DIR=$L2_POISON
    ;;
  *) die "unknown arm '$1'" ;;
  esac
}

compose() {
  local arm=$1
  CONTAINER=lmcachetest-$arm
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  local -a overlay_argv=()
  if [[ $ARM_OVERLAY == yes ]]; then
    overlay_argv=(-v "$OVL/scheduler.patched.py:$SCHED_IN_IMAGE:ro")
  fi
  docker_argv=(
    docker run --rm --name "$CONTAINER"
    --gpus all --ipc host --network host
    -v "$MODELS:/models:ro"
    -v "$ARM_L2DIR:/l2:rw"
    -v "$LOGS:/lmclogs:rw"
    "${overlay_argv[@]}"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    -e VLLM_USE_V2_MODEL_RUNNER=1
    -e VLLM_EXL3_GRAPH_DECODE=1
    -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128
    -e PORT="$PORT"
    "${ARM_ENV[@]}"
    --entrypoint "$WRAPPER"
    "$IMAGE"
  )
  serve_argv=(
    /opt/venv/bin/vllm serve "$MODEL"
    --served-model-name "$SERVED"
    --tensor-parallel-size "$TP"
    --quantization exl3 --quantization-config "$QUANT_CFG"
    --max-model-len "$MAXLEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-num-seqs "$SEQS"
    --kv-cache-dtype fp8
    --max-num-batched-tokens "$BATCHED"
    --enable-prefix-caching --mamba-cache-mode align
    --speculative-config "$SPEC"
    --mm-processor-kwargs "$MM_CFG"
    --compilation-config "$COMPILE"
    --host "$BIND" --port "$PORT"
  )
  python3 - "$OUT/command-$arm.json" "$arm" "$IMAGE" "$ARM_DESC" "$ARM_OVERLAY" "$ARM_L2DIR" \
    "${docker_argv[@]}" -- "${serve_argv[@]}" <<'PY'
import json, pathlib, sys
out, arm, image, desc, overlay, l2dir, *rest = sys.argv[1:]
split = rest.index("--")
pathlib.Path(out).write_text(json.dumps({
    "schema": "qwen38-lmcache-2x-command/1", "arm": arm, "image": image,
    "description": desc, "gate_overlay_mounted": overlay == "yes",
    "l2_host_dir": l2dir,
    "docker_argv": rest[:split], "wrapper_argv": rest[split + 1:],
    "note": "the wrapper appends --kv-transfer-config with LMCacheMPConnector kv_both "
            "(mq_timeout 60, heartbeat 5) when LMCACHE_MODE=disk, and execs the model "
            "server unchanged when LMCACHE_MODE=off"},
    indent=2) + "\n")
PY
}

launch() {
  local arm=$1 log=$2
  say "launching arm $arm (overlay=$ARM_OVERLAY l2=$ARM_L2DIR)"
  "${docker_argv[@]}" "${serve_argv[@]}" >"$log" 2>&1 &
  wait_ready "$log"
}

snap_metrics() {
  local tag=$1
  curl -sf --max-time 15 "http://127.0.0.1:$PORT/metrics" >"$OUT/vllm-metrics-$tag.txt" 2>/dev/null || true
  curl -sf --max-time 15 "http://127.0.0.1:$LMC_PROM_PORT/metrics" >"$OUT/lmcache-metrics-$tag.txt" 2>/dev/null || true
  curl -sf --max-time 15 "http://127.0.0.1:$LMC_HTTP_PORT/status" >"$OUT/lmcache-status-$tag.json" 2>/dev/null || true
  free -b | awk 'NR<=2' >"$OUT/dram-$tag.txt"
}

l2_snapshot() {
  local tag=$1 dir=${2:-$L2}
  {
    echo "dir: $dir/data"
    echo "files: $(find "$dir/data" -type f 2>/dev/null | wc -l)"
    echo "bytes: $(du -sb "$dir/data" 2>/dev/null | cut -f1)"
  } | tee "$OUT/l2-snapshot-$tag.txt" >&2
}

run_probe() {
  local arm=$1
  python3 "$TOOLS/apc_poison_probe.py" run \
    --base "http://127.0.0.1:$PORT" --model "$SERVED" --prompts "$W/prompts.json" \
    --arm "$arm" --description "$ARM_DESC" --expect-prompts-sha256 "$PROMPTS_SHA" \
    --concurrency 1 \
    --out "$OUT/arm-$arm.json" 2>&1 | tee "$LOGS/probe-$arm.out"
}

# ------------------------------------------------------------------ phases
phase_stage() {
  verify_image
  echo "$PROBE_SHA  $TOOLS/apc_poison_probe.py" | sha256sum -c - || die "probe drifted"
  [[ -f $W/prompts.json ]] || die "prompts.json missing at $W (copy receipts/lmcache-frozen-prompts.json)"
  python3 - "$W/prompts.json" "$PROMPTS_SHA" <<'PY'
import hashlib, json, sys
prompts = json.load(open(sys.argv[1]))
assert prompts["prompts_sha256"] == sys.argv[2], prompts["prompts_sha256"]
body = {k: v for k, v in prompts.items() if k != "prompts_sha256"}
recomputed = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
assert recomputed == sys.argv[2], f"prompts body does not hash to its own field: {recomputed}"
assert len(prompts["correctness_requests"]) == 38, len(prompts["correctness_requests"])
assert prompts["mamba_block_tokens"] == 1600
print(json.dumps({"prompts_sha256_verified": True, "body_rehash_matches": True,
                  "correctness_requests": 38, "needles": len(prompts["needles"]),
                  "head_tokens": prompts["head_tokens"]}))
PY
  # extract the sr1 scheduler and build the gated overlay, fail-closed
  if [[ ! -f $OVL/scheduler.base.py ]]; then
    say "extracting the image's own scheduler.py"
    docker create --name lmc-extract "$IMAGE" >/dev/null
    docker cp "lmc-extract:$SCHED_IN_IMAGE" "$OVL/scheduler.base.py"
    docker rm -f lmc-extract >/dev/null
  fi
  if [[ ! -f $OVL/scheduler.patched.py ]]; then
    say "applying the gate patch (-p1 --fuzz=0)"
    rm -rf "$W/patchwork" && mkdir -p "$W/patchwork/vllm/v1/core/sched"
    cp "$OVL/scheduler.base.py" "$W/patchwork/vllm/v1/core/sched/scheduler.py"
    (cd "$W/patchwork" && patch -p1 --fuzz=0 -N <"$TOOLS/gg-vllm-hybrid-divergent-hit-gate.patch")
    cp "$W/patchwork/vllm/v1/core/sched/scheduler.py" "$OVL/scheduler.patched.py"
  fi
  verify_overlay
  # identity of everything the arms depend on, measured inside the image, CPU only
  docker run --rm --network none --entrypoint /bin/bash "$IMAGE" -lc "
    set -eu
    sha256sum $SCHED_IN_IMAGE $WRAPPER
    /opt/venv/bin/pip show lmcache | sed -n '1,2p'
    /opt/venv/bin/python - <<'PY'
import importlib.metadata as md, json
d = md.distribution('lmcache')
direct = d.read_text('direct_url.json')
print(json.dumps({'lmcache_version': d.version,
                  'direct_url': json.loads(direct) if direct else None}))
PY
    sed -n '1,3p' /opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/memory_manager/l1_memory_manager.py
  " >"$OUT/stage-identity.txt" 2>&1 || die "stage identity checks failed (see $OUT/stage-identity.txt)"
  grep -q "CPU pinned-DRAM L1 memory manager" "$OUT/stage-identity.txt" ||
    die "could not confirm LMCache L1 is the CPU DRAM tier"
  say "ForceP2P override (must stay; worth 7-22% of decode):"
  grep RegistryDwords: /proc/driver/nvidia/params | tee "$OUT/stage-forcep2p.txt" >&2
  grep -q 'ForceP2P=0x11' "$OUT/stage-forcep2p.txt" || die "ForceP2P override missing from the driver"
  say "listening ports that could collide:"
  ss -ltn "( sport = :$PORT or sport = :$LMC_HTTP_PORT or sport = :$LMC_PROM_PORT or sport = :$LMC_ZMQ_PORT )" |
    tee "$OUT/stage-ports.txt" >&2
  grep -qE ":($PORT|$LMC_HTTP_PORT|$LMC_PROM_PORT|$LMC_ZMQ_PORT)\b" "$OUT/stage-ports.txt" &&
    die "one of the harness ports is already bound (see $OUT/stage-ports.txt)"
  python3 - "$OUT/stage.json" "$W" "$IMAGE" "$IMAGE_ID" "$LMC_L1_GB" "$LMC_L1_INIT_GB" \
    "$LMC_L2_GB" "$LMC_CHUNK" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, work, image, image_id, l1, l1init, l2, chunk = sys.argv[1:]
work = pathlib.Path(work)
def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
prompts = json.loads((work / "prompts.json").read_text())
payload = {
    "schema": "qwen38-lmcache-2x-stage/1",
    "host": run("hostname"),
    "gpus": run("nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version,compute_mode",
                "--format=csv,noheader"),
    "docker_version": run("docker", "--version"),
    "dram": run("bash", "-lc", "free -g | sed -n 2p"),
    "disk": run("bash", "-lc", "df -h / | tail -1"),
    "image": {"tag": image, "id": image_id},
    "lmcache_l1_cpu_dram": {
        "l1_size_gb": float(l1), "l1_init_size_gb": float(l1init),
        "l1_use_lazy": "default True",
        "tier": "LMCache L1 == CPU pinned DRAM (local_cpu / max_local_cpu_size equivalent)",
        "l2_size_gb": float(l2), "chunk_size": int(chunk),
    },
    "harness_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((work / "tools").glob("*")) if p.is_file()},
    "overlay_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((work / "overlay").glob("*.py"))},
    "prompts_sha256": prompts["prompts_sha256"],
    "identity_capture": "out/stage-identity.txt",
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"stage": out, "prompts_sha256": payload["prompts_sha256"]}))
PY
  say "stage complete"
}

phase_control() {
  local arm=L0 log=$LOGS/server-L0.log
  require_free_gpu
  configure_arm $arm
  compose $arm
  start_sampler $arm
  launch $arm "$log" || { startup_facts "$log" $arm; tail -60 "$log" >&2; stop_server; die "arm L0 failed to start"; }
  startup_facts "$log" $arm
  snap_metrics "L0-pre"
  run_probe $arm
  snap_metrics "L0-post"
  stop_server
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm $arm --out "$OUT/windows-$arm.json"
  say "arm L0 complete"
}

# Unpatched baseline: does this host reproduce the divergent-hit corruption at all?
phase_unpatched() {
  local log=$LOGS/server-U1U2.log
  require_free_gpu
  say "wiping L2 for the UNPATCHED cold pass"
  rm -rf "$L2/data"; mkdir -p "$L2/data"
  configure_arm U1cold
  compose U1cold
  start_sampler U1cold
  launch U1cold "$log" || { startup_facts "$log" U1cold; tail -80 "$log" >&2; stop_server; die "unpatched LMCache arm failed to start"; }
  startup_facts "$log" U1cold
  l2_snapshot "U1cold-pre"
  snap_metrics "U1cold-pre"
  run_probe U1cold
  snap_metrics "U1cold-post"
  l2_snapshot "U1cold-post"
  say "warm pass on the SAME live unpatched server"
  configure_arm U2warm
  snap_metrics "U2warm-pre"
  run_probe U2warm
  snap_metrics "U2warm-post"
  sleep 45   # let asynchronous L1->L2 flushes land
  l2_snapshot "U2warm-post"
  stop_server
  say "retaining the UNPATCHED (poisoned) L2 at $L2_POISON"
  rm -rf "$L2_POISON"; mkdir -p "$L2_POISON"
  cp -a "$L2/data" "$L2_POISON/data"
  l2_snapshot "poisoned-retained" "$L2_POISON"
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm U1U2 --out "$OUT/windows-U1U2.json"
  say "unpatched baseline complete"
}

phase_patched() {
  local log=$LOGS/server-L1L2.log
  require_free_gpu
  say "wiping L2 for the PATCHED cold pass"
  rm -rf "$L2/data"; mkdir -p "$L2/data"
  configure_arm L1cold
  compose L1cold
  start_sampler L1cold
  launch L1cold "$log" || { startup_facts "$log" L1cold; tail -80 "$log" >&2; stop_server; die "patched LMCache arm failed to start"; }
  startup_facts "$log" L1cold
  l2_snapshot "L1cold-pre"
  snap_metrics "L1cold-pre"
  run_probe L1cold
  snap_metrics "L1cold-post"
  l2_snapshot "L1cold-post"
  say "warm pass on the SAME live patched server"
  configure_arm L2warm
  snap_metrics "L2warm-pre"
  run_probe L2warm
  snap_metrics "L2warm-post"
  # what an API host cares about: hit rate and TTFT, cold vs warm, on disjoint documents
  # (32k and 128k) that no correctness arm has touched. Named snapshots, deltas only.
  snap_metrics "reuse-L1-pre"
  python3 "$TOOLS/apc_poison_probe.py" reuse \
    --base "http://127.0.0.1:$PORT" --model "$SERVED" --prompts "$W/prompts.json" \
    --arm L1reuse \
    --description "LMCache L1 CPU-DRAM ${LMC_L1_GB} GiB, gate patched: cold then warm then warm-exact on disjoint 32k/128k documents" \
    --out "$OUT/reuse-L1.json" 2>&1 | tee "$LOGS/reuse-L1.out"
  snap_metrics "reuse-L1-post"
  # Hold the LMCache-enabled server up so the five-check fidelity gate can be run against
  # it from the driver VM (tools/tb21_gate.py --base-url http://10.0.0.4:8000). The phase
  # stays attached to the container's log the whole time; the gate signals completion by
  # creating $W/gate-done. The container is never orphaned.
  rm -f "$W/gate-done"
  say "HOLDING the LMCache-enabled server on $BIND:$PORT for the fidelity gate."
  say "run the gate now, then: touch $W/gate-done"
  local i
  for ((i = 0; i < GATE_HOLD_TIMEOUT; i++)); do
    [[ -f $W/gate-done ]] && break
    sleep 5
  done
  if [[ -f $W/gate-done ]]; then
    say "gate signalled complete after $((i * 5))s"
  else
    say "warning: gate never signalled within $((GATE_HOLD_TIMEOUT * 5))s; closing out anyway"
  fi
  snap_metrics "gate-post"
  say "flushing L1->L2 before the restart arms need the stores"
  sleep 45
  l2_snapshot "L2warm-post"
  snap_metrics "L1L2-final"
  stop_server
  say "retaining the PATCHED (poison-free) L2 at $L2_CLEAN"
  rm -rf "$L2_CLEAN"; mkdir -p "$L2_CLEAN"
  cp -a "$L2/data" "$L2_CLEAN/data"
  l2_snapshot "clean-retained" "$L2_CLEAN"
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm L1L2 --out "$OUT/windows-L1L2.json"
  say "patched arms complete (cold, warm, reuse, gate hold)"
}

phase_restart() {
  local arm=$1 src=$2 log
  log=$LOGS/server-$arm.log
  require_free_gpu
  [[ -d $src/data ]] || die "retained L2 missing at $src/data"
  if [[ $arm == L3fresh ]]; then
    say "restoring the retained CLEAN L2 into the live L2 directory"
    rm -rf "$L2/data"; mkdir -p "$L2"
    cp -a "$src/data" "$L2/data"
  fi
  configure_arm $arm
  l2_snapshot "$arm-pre" "$ARM_L2DIR"
  compose $arm
  start_sampler $arm
  launch $arm "$log" || { startup_facts "$log" $arm; tail -80 "$log" >&2; stop_server; die "$arm failed to start"; }
  startup_facts "$log" $arm
  snap_metrics "$arm-pre"
  run_probe $arm
  snap_metrics "$arm-post"
  l2_snapshot "$arm-post" "$ARM_L2DIR"
  stop_server
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm $arm --out "$OUT/windows-$arm.json"
  say "$arm complete"
}

phase_receipt() {
  python3 - "$OUT" "$W" <<'PY'
import importlib.util, json, pathlib, re, sys
out = pathlib.Path(sys.argv[1]); work = pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location(
    "apc_poison_probe", work / "tools" / "apc_poison_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
def load(name, default=None):
    p = out / name
    return json.loads(p.read_text()) if p.exists() else default
TAGS = ("L0", "U1cold", "U2warm", "L1cold", "L2warm", "L3fresh", "L3poison")
arms, windows, commands, facts, snaps = {}, {}, {}, {}, {}
for tag in TAGS:
    data = load(f"arm-{tag}.json")
    if data:
        arms[tag] = data
    c = load(f"command-{tag}.json")
    if c:
        commands[tag] = c
    f = out / f"startup-facts-{tag}.txt"
    if f.exists():
        facts[tag] = [ln for ln in f.read_text().splitlines() if ln.strip()]
for tag in ("L0", "U1U2", "L1L2", "L3fresh", "L3poison"):
    w = load(f"windows-{tag}.json")
    if w:
        windows[tag] = w
for p in sorted(out.glob("l2-snapshot-*.txt")):
    snaps[p.stem.replace("l2-snapshot-", "")] = p.read_text().strip().splitlines()
lm_metrics = {}
for p in sorted(out.glob("lmcache-metrics-*.txt")):
    tag = p.stem.replace("lmcache-metrics-", "")
    keep = {}
    for line in p.read_text().splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        if re.search(r"retriev|hit|store|lookup|evict|l1|l2|disk|memory", name, re.I):
            try:
                keep[name] = float(value)
            except ValueError:
                pass
    if keep:
        lm_metrics[tag] = keep
vllm_metrics = {}
for p in sorted(out.glob("vllm-metrics-*.txt")):
    tag = p.stem.replace("vllm-metrics-", "")
    keep = {}
    for line in p.read_text().splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        if re.search(r"prefix_cache|gpu_cache_usage|time_to_first_token", name):
            try:
                keep[name] = float(value)
            except ValueError:
                pass
    if keep:
        vllm_metrics[tag] = keep
dram = {}
for p in sorted(out.glob("dram-*.txt")):
    dram[p.stem.replace("dram-", "")] = p.read_text().strip().splitlines()
divergence = {}
if "L0" in arms:
    ref = arms["L0"]["rows"]
    for tag in TAGS[1:]:
        if tag in arms:
            divergence[f"{tag}_vs_L0"] = probe._divergence(ref, arms[tag]["rows"])
payload = {
    "arms": {tag: {"description": d["arm_description"], "summary": d["summary"],
                   "rows": d["rows"]} for tag, d in arms.items()},
    "log_windows": windows,
    "commands": commands,
    "startup_facts": facts,
    "l2_snapshots": snaps,
    "lmcache_metrics_filtered": lm_metrics,
    "vllm_metrics_filtered": vllm_metrics,
    "dram_snapshots": dram,
    "reuse_L1": load("reuse-L1.json"),
    "divergence_vs_control": divergence,
    "stage": load("stage.json"),
    "overlay_identity": (out / "overlay-identity.txt").read_text().splitlines()
                        if (out / "overlay-identity.txt").exists() else None,
}
(out / "assembled.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({
    "arms": {t: {"failed": a["summary"]["failed_requests"],
                 "corrupted": a["summary"]["corrupted_requests"],
                 "requests": a["summary"]["requests"],
                 "hit_rate": a["summary"]["cumulative_prefix_cache_hit_rate"]}
             for t, a in arms.items()}}, indent=2))
PY
}

case ${1:-} in
stage) phase_stage ;;
control) phase_control ;;
unpatched) phase_unpatched ;;
patched) phase_patched ;;
restart-fresh) phase_restart L3fresh "$L2_CLEAN" ;;
restart-poison) phase_restart L3poison "$L2_POISON" ;;
receipt) phase_receipt ;;
*) die "usage: $0 {stage|control|unpatched|patched|restart-fresh|restart-poison|receipt}" ;;
esac
