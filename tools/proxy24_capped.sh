#!/bin/bash
# 24 GiB-class PROXY qualification of the published context edition, on the physical
# RTX 5090, by capping what the engine may take. This is not a qualification of a
# 24 GB board and may never be reported as one: no 24 GB board exists in this project.
#
# Two emulation arms, because they test different things:
#
#   budget   --gpu-memory-utilization lowered so ceil(cuda_total * util) lands at or
#            under 24.0 GiB. Emulates a 24 GiB board's *engine budget* only. The rest
#            of the 32 GiB card stays free, so any allocation that lives outside the
#            budget -- the vision tower's transient, above all -- is unrealistically
#            comfortable. This is the arm the brief asked for.
#
#   board    the same budget a physical 24 GiB board would compute at the QUALIFIED
#            0.955 utilisation, plus a ballast process that holds cudaMemGetInfo free
#            down to what a 24 GiB board reports. Emulates budget AND card, so the
#            outside-budget slack is the real one. This is the arm that can actually
#            fail, and therefore the informative one.
#
# Mechanism, read out of the engine's own source in this image
# (vllm/v1/worker/utils.py request_memory, vllm/v1/worker/gpu_worker.py):
#   requested = ceil(init_snapshot.total_memory * gpu_memory_utilization)
#   startup refuses if init_snapshot.free_memory < requested
#   available_kv = requested - non_kv_cache_memory - cudagraph_estimate   [no free clamp]
# so utilisation sets the budget and the ballast sets the card. Both are needed.
#
# Gate set and harness are the physical qualification's, unmodified:
# receipts/qualification-5090-context.json, gates 1-7.
set -uo pipefail

W=/home/mbelleau/proxy24
QW=/home/mbelleau/qual5090          # corpus, image fixture, warm JIT and sampler live here
OUT=$W/out
LOGS=$W/logs
TAG=${TAG:?TAG required}
GATES=${GATES:?GATES required}
PORT=${PORT:-8300}
UTIL=${UTIL:?UTIL required}
MAXLEN=${MAXLEN:?MAXLEN required}
MTP=${MTP:?MTP required, 0 disables speculative decoding}
MAXPIX=${MAXPIX:-8388608}
TOKENS=${TOKENS:-0}                 # long-needle request length for g2/g6
TEXT_TOKENS=${TEXT_TOKENS:-0}       # text half of the combined g3 request
MINPROMPT=${MINPROMPT:-0}           # minimum accepted g3 prompt_tokens
BALLAST_FREE_GIB=${BALLAST_FREE_GIB:-0}   # 0 disables the ballast (budget arm)
ARM=${ARM:?ARM required, budget or board}
ALLOC_CONF=${ALLOC_CONF:-expandable_segments:True}
PREFLIGHT_MAX_USED_MIB=${PREFLIGHT_MAX_USED_MIB:-1500}

CTN=proxy24-$TAG
BCTN=proxy24-ballast
UUID=GPU-506a575d-01d7-b12e-9a0a-c1ab5f38ae0a
IMAGE='docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b'
REPO_DIR=/mnt/vault/llm/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context
REV=$(cat "$REPO_DIR/refs/main")
MODEL=/models/ctx-repo/snapshots/$REV
VLLM=/opt/venv/lib/python3.12/site-packages/vllm
PATCH=$QW/tools
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
COMPILE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'
MM=$(printf '{"truncation":false,"max_pixels":%d}' "$MAXPIX")
SRV=$LOGS/server-$TAG.log
BLOG=$LOGS/ballast-$TAG.log
PHASE=$W/phase-$TAG
SAMPLES=$LOGS/gpu-samples-$TAG.jsonl
STATUS=$OUT/gates-$TAG.jsonl
CMDFILE=$OUT/command-$TAG.json
SAMPLER_PID=
SERVER_SH=
api=http://127.0.0.1:$PORT/v1

mkdir -p "$OUT" "$LOGS"
: >"$STATUS"
: >"$SAMPLES"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOGS/driver-$TAG.log"; }
phase() { echo "$1" >"$PHASE"; log "PHASE $1"; }
since() { python3 -c 'import sys;print(round(float(sys.argv[2])-float(sys.argv[1]),3))' "$1" "$(date +%s.%N)"; }

point_sample() {
  python3 - "$UUID" "$1" "$OUT/point-samples-$TAG.jsonl" <<'PY'
import json, subprocess, sys, time
uuid, name, out = sys.argv[1:4]
q = "memory.total,memory.used,memory.free,utilization.gpu,power.draw"
raw = subprocess.run(["nvidia-smi", f"--id={uuid}", f"--query-gpu={q}",
                      "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, check=True).stdout.strip()
t, u, f_, g, p = [x.strip() for x in raw.split(",")]
row = {"name": name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "memory_total_mib": int(t), "memory_used_mib": int(u),
       "memory_free_mib": int(f_), "utilization_gpu_pct": int(g),
       "power_draw_w": float(p)}
with open(out, "a") as fh:
    fh.write(json.dumps(row) + "\n")
print(json.dumps(row))
PY
}

record() {  # record <gate> <rc> <wall> <artifact...>
  python3 - "$STATUS" "$@" <<'PY'
import json, sys, time
status, gate, rc, wall = sys.argv[1:5]
row = {"gate": gate, "rc": int(rc), "wall_sec": float(wall),
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "artifacts": sys.argv[5:]}
with open(status, "a") as fh:
    fh.write(json.dumps(row) + "\n")
print("RECORD " + json.dumps(row))
PY
}

cleanup() {
  [[ -n ${SAMPLER_PID:-} ]] && kill "$SAMPLER_PID" 2>/dev/null
  podman rm -f "$CTN" >/dev/null 2>&1
  podman rm -f "$BCTN" >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

# ---- patch identity, verified on this host against the qualification receipt ----
printf '%s  %s\n' \
  2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee \
    "$PATCH/vllm-exl3-prefill-dispatch.py" \
  04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7 \
    "$PATCH/vllm-qwen3_5-embed-quant-config.py" \
  0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab \
    "$PATCH/vllm-qwen3_5_mtp-embed-quant-config.py" |
  sha256sum -c - >"$OUT/patch-verify-$TAG.txt" 2>&1
grep -qv ': OK$' "$OUT/patch-verify-$TAG.txt" && { cat "$OUT/patch-verify-$TAG.txt"; log "patch digest mismatch"; exit 2; }

# ---- preflight ----
# Ownership is a hub fact, not a memory reading. PerfSweep5090's sweep tears its server
# down between rows, so the card reads idle for 30-60 s at every row boundary while the
# window is still owned by someone else; on 2026-08-16T02:59Z this cost that agent its
# `eagerbase` row. Compute mode Exclusive_Process does not help: it only bars a second
# context while one is live, so it admits anything that starts in a teardown gap.
# Therefore: a handover token, written by hand only after the outgoing owner names this
# agent over hub, is the gate. The memory reading is kept as a secondary sanity check.
HANDOVER=$W/HANDOVER
if [[ ! -s $HANDOVER ]]; then
  log "PREFLIGHT REFUSED: no handover token at $HANDOVER. The card is someone else's"
  log "until the outgoing owner names this agent over hub; write the token then."
  exit 4
fi
log "handover token: $(tr '\n' ' ' <"$HANDOVER")"
USED=$(nvidia-smi "--id=$UUID" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( USED > PREFLIGHT_MAX_USED_MIB )); then
  log "PREFLIGHT REFUSED: $USED MiB already in use on the card (limit $PREFLIGHT_MAX_USED_MIB)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$LOGS/driver-$TAG.log"
  exit 4
fi

log "attempt TAG=$TAG arm=$ARM gates='$GATES' util=$UTIL len=$MAXLEN mtp=$MTP ballast_free=$BALLAST_FREE_GIB rev=$REV"
podman rm -f "$CTN" "$BCTN" >/dev/null 2>&1

phase before_load
point_sample before_load
python3 "$QW/gpu_sampler.py" "$UUID" "$PHASE" "$SAMPLES" 1.0 &
SAMPLER_PID=$!

# ---- ballast, board arm only ----
if [[ $BALLAST_FREE_GIB != 0 ]]; then
  phase ballast
  podman run -d --name "$BCTN" --device nvidia.com/gpu=all --ipc=host --network host \
    -v "$W/tools/proxy24_ballast.py:/tools/proxy24_ballast.py:ro" \
    -v "$OUT:/out:rw" \
    --entrypoint /opt/venv/bin/python3 "$IMAGE" \
    /tools/proxy24_ballast.py --target-free-gib "$BALLAST_FREE_GIB" \
      --report "/out/ballast-$TAG.json" --hold-sec 7200 >"$BLOG" 2>&1
  for _ in $(seq 1 60); do
    [[ -s "$OUT/ballast-$TAG.json" ]] && break
    podman inspect -f '{{.State.Running}}' "$BCTN" 2>/dev/null | grep -q true || break
    sleep 2
  done
  if [[ ! -s "$OUT/ballast-$TAG.json" ]]; then
    podman logs "$BCTN" >>"$BLOG" 2>&1
    cat "$BLOG"
    log "BALLAST FAILED TAG=$TAG"
    exit 5
  fi
  cat "$OUT/ballast-$TAG.json"
  point_sample after_ballast
fi

SERVE_ARGS=(serve "$MODEL"
  --served-model-name m --quantization exl3
  --quantization-config "$CFG"
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL"
  --kv-cache-dtype fp8 --max-num-seqs 1 --max-num-batched-tokens 2048)
if (( MTP > 0 )); then
  SERVE_ARGS+=(--speculative-config "$(printf '{"method":"mtp","num_speculative_tokens":%d}' "$MTP")")
fi
SERVE_ARGS+=(--mm-processor-kwargs "$MM"
  --compilation-config "$COMPILE"
  --host 127.0.0.1 --port "$PORT")

RUN_ARGS=(run --rm --name "$CTN"
  --device nvidia.com/gpu=all --ipc=host --network host
  -v "$REPO_DIR:/models/ctx-repo:ro"
  -v "$QW/cache/jit:/cache/jit:rw"
  -v "$QW/cache/exl3-online:/cache/exl3-online:rw"
  -v "$PATCH/vllm-exl3-prefill-dispatch.py:$VLLM/model_executor/layers/quantization/exl3.py:ro"
  -v "$PATCH/vllm-qwen3_5-embed-quant-config.py:$VLLM/model_executor/models/qwen3_5.py:ro"
  -v "$PATCH/vllm-qwen3_5_mtp-embed-quant-config.py:$VLLM/model_executor/models/qwen3_5_mtp.py:ro"
  -e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1
  -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e "PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF"
  --entrypoint /opt/venv/bin/vllm
  "$IMAGE")

python3 - "$CMDFILE" "$TAG" "$ARM" "$UTIL" "$MAXPIX" "$MTP" "$MAXLEN" "$BALLAST_FREE_GIB" \
  "$ALLOC_CONF" "$REV" "${RUN_ARGS[@]}" -- "${SERVE_ARGS[@]}" <<'PY'
import json, pathlib, sys
out, tag, arm, util, maxpix, mtp, maxlen, ballast, alloc, rev = sys.argv[1:11]
rest = sys.argv[11:]
split = rest.index("--")
payload = {"tag": tag, "arm": arm, "gpu_memory_utilization": float(util),
           "max_pixels": int(maxpix), "mtp_depth": int(mtp),
           "max_model_len": int(maxlen),
           "prefix_caching_flag_passed": False,
           "ballast_target_free_gib": float(ballast) or None,
           "pytorch_cuda_alloc_conf": alloc,
           "model_revision": rev,
           "podman_argv": ["podman"] + rest[:split],
           "vllm_argv": ["/opt/venv/bin/vllm"] + rest[split + 1:]}
pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n")
PY

START_T=$(date +%s.%N)
phase startup
podman "${RUN_ARGS[@]}" "${SERVE_ARGS[@]}" >"$SRV" 2>&1 &
SERVER_SH=$!

READY=0
for _ in $(seq 1 480); do
  if curl -fsS --max-time 10 "$api/models" 2>/dev/null |
       python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if any(x.get("id")=="m" for x in d["data"]) else 1)'
  then READY=1; break; fi
  kill -0 "$SERVER_SH" 2>/dev/null || break
  grep -qE 'EngineCore failed|ValidationError|unrecognized arguments|No available memory for the cache blocks|Traceback \(most recent call last\)' "$SRV" 2>/dev/null && break
  sleep 5
done
STARTUP_WALL=$(since "$START_T")

FACTS_RE='Free memory on device \([0-9.]+/[0-9.]+ GiB\) on startup\. Desired GPU memory utilization is \([0-9.]+, [0-9.]+ GiB\)\. Actual usage is .{0,420}|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds|Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB|init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s \(compilation: [0-9.]+ s\)|Setting attention block size to [0-9]+ tokens.{0,70}|Padding mamba page size by [0-9.]+%.{0,70}|Add [0-9]+ padding layers, may waste at most [0-9.]+% KV cache memory|Using fp8 data type to store kv cache|[0-9.]+ GiB KV cache is needed[^)]{0,60}|num_spec_tokens=[0-9]+|enable_prefix_caching=[A-Za-z]+|mamba_cache_mode=.{0,20}|mamba_block_size=[0-9]+|Initializing a V1 LLM engine \(v[0-9A-Za-z.+_-]+\)'

if [[ $READY != 1 ]]; then
  phase startup_failed
  point_sample startup_failed
  grep -oE "$FACTS_RE|ValueError: .{0,300}|To serve at least one request .{0,300}" "$SRV" |
    tail -30 >"$OUT/startup-fail-$TAG.txt"
  cat "$OUT/startup-fail-$TAG.txt"
  record gate1_startup_native_allocation 1 "$STARTUP_WALL" "logs/server-$TAG.log" "out/startup-fail-$TAG.txt"
  log "STARTUP FAILED TAG=$TAG"
  exit 3
fi

phase startup_done
point_sample after_startup
grep -oE "$FACTS_RE" "$SRV" | sort -u >"$OUT/startup-facts-$TAG.txt"
cat "$OUT/startup-facts-$TAG.txt"
grep -F 'Initializing a V1 LLM engine' "$SRV" | head -1 >"$OUT/engine-config-$TAG.txt"

for g in $GATES; do
case $g in
g1)
  record gate1_startup_native_allocation 0 "$STARTUP_WALL" \
    "out/startup-facts-$TAG.txt" "out/command-$TAG.json" "logs/server-$TAG.log"
  ;;
g2)
  phase g2_long_needle
  T=$(date +%s.%N)
  python3 "$QW/tools/longctx.py" --url "$api" --corpus "$QW/corpus/literary" \
    --tokens "$TOKENS" --depth 0.5 --max-token-error 64 --require-pass \
    --out "$OUT/g2-needle-$TAG.json" \
    >"$LOGS/g2-$TAG.out" 2>"$LOGS/g2-$TAG.err"
  RC=$?
  record gate2_long_needle_exact $RC "$(since "$T")" "out/g2-needle-$TAG.json" \
    "logs/g2-$TAG.out" "logs/g2-$TAG.err"
  phase g2_released
  point_sample after_g2_release
  ;;
g3)
  phase gate3_combined
  T=$(date +%s.%N)
  python3 "$QW/tools/longmm.py" --url "$api" --corpus "$QW/corpus/literary" \
    --image "$QW/receipts/native-mtp-8mp-fixture.png" --require-pass \
    --text-tokens "$TEXT_TOKENS" --minimum-prompt-tokens "$MINPROMPT" \
    --out "$OUT/gate3-longmm-$TAG.json" \
    >"$LOGS/gate3-$TAG.out" 2>"$LOGS/gate3-$TAG.err"
  RC=$?
  record gate3_combined_long_text_and_7mp_image $RC "$(since "$T")" \
    "out/gate3-longmm-$TAG.json" "logs/gate3-$TAG.out" "logs/gate3-$TAG.err"
  phase gate3_released
  point_sample after_gate3_release
  ;;
g4)
  phase gate4_vision
  T=$(date +%s.%N)
  python3 "$QW/tools/vision_eval.py" --url "$api" --label "proxy24-$TAG" --cases 10 \
    --out "$OUT/gate4-vision-$TAG.json" \
    >"$LOGS/gate4-$TAG.out" 2>"$LOGS/gate4-$TAG.err"
  RC=$?
  if [[ $RC == 0 ]]; then
    python3 -c 'import json,sys; o=json.load(open(sys.argv[1]))["overall"]; sys.exit(0 if o["n"]==30 and o["correct"]>=24 else 1)' \
      "$OUT/gate4-vision-$TAG.json" || RC=64
  fi
  record gate4_image_suite_24_of_30 $RC "$(since "$T")" \
    "out/gate4-vision-$TAG.json" "logs/gate4-$TAG.out" "logs/gate4-$TAG.err"
  ;;
g5)
  phase g5_decode
  curl -fsS --max-time 20 "http://127.0.0.1:$PORT/metrics" >"$OUT/metrics-before-g5-$TAG.txt" 2>&1
  T=$(date +%s.%N)
  python3 "$QW/tools/bench.py" --url "http://127.0.0.1:$PORT/v1/completions" \
    --tokens 256 --concurrency 1 1 1 --prefill-tokens --label "proxy24-g5-$TAG" \
    >"$LOGS/g5-$TAG.out" 2>"$LOGS/g5-$TAG.err"
  RC=$?
  WALL=$(since "$T")
  curl -fsS --max-time 20 "http://127.0.0.1:$PORT/metrics" >"$OUT/metrics-after-g5-$TAG.txt" 2>&1
  record gate5_three_warmed_decode_runs $RC "$WALL" "logs/g5-$TAG.out" "logs/g5-$TAG.err" \
    "out/metrics-before-g5-$TAG.txt" "out/metrics-after-g5-$TAG.txt"
  ;;
g6)
  phase gate6_second_long
  T=$(date +%s.%N)
  python3 "$QW/tools/longctx.py" --url "$api" --corpus "$QW/corpus/literary" \
    --tokens "$TOKENS" --depth 0.25 --seed 20260816 --max-token-error 64 --require-pass \
    --out "$OUT/gate6-needle-$TAG.json" \
    >"$LOGS/gate6-$TAG.out" 2>"$LOGS/gate6-$TAG.err"
  RC=$?
  record gate6_second_long_request_after_release $RC "$(since "$T")" \
    "out/gate6-needle-$TAG.json" "logs/gate6-$TAG.out" "logs/gate6-$TAG.err"
  ;;
*) log "unknown gate $g"; exit 2 ;;
esac
done

phase shutdown
podman stop -t 90 "$CTN" >>"$LOGS/driver-$TAG.log" 2>&1
wait "$SERVER_SH" 2>/dev/null
podman rm -f "$BCTN" >/dev/null 2>&1
sleep 10
phase after_release
point_sample after_release
sleep 3
kill "$SAMPLER_PID" 2>/dev/null
SAMPLER_PID=
log "PROXY24_DONE TAG=$TAG"
