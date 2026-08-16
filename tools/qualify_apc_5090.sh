#!/bin/bash
# Serving qualification of the PROMOTED release unit, with prefix caching ON, on the
# physical RTX 5090 (AIBoss).
#
#   localhost/vllm:gg-r34-patched-apc, manifest sha256:16a936b877b90fc0...b0b79218
#
# This is a new qualification, not a re-measurement. Every published measurement before it
# ran with prefix caching OFF: it is default-off for this hybrid model
# (vllm/engine/arg_utils.py:2532-2534 excludes hybrid models), so
# receipts/qualification-5090-context.json's seven gates all passed with
# enable_prefix_caching=False in the banner. Turning it on changes scheduling, changes KV
# accounting, and puts Scheduler._mamba_block_aligned_split -- the function upstream
# PR #51113 fixes and the whole reason this image exists -- on every request's hot path.
#
# The gate set is receipts/qualification-5090-context.json's, unmodified, plus two the old
# set did not need:
#
#   g1  startup allocates a native-length request without exceeding utilisation 0.97
#   g2  the ~261,794-token text needle is retrieved exactly
#   g3  a combined 236,824-token plus seven-megapixel request returns code and colours
#   g4  the 30-case deterministic image suite stays at or above 24/30
#   g5  three warmed 256-token concurrency-1 runs report decode, acceptance, dispersion
#   g6  a second native-length request succeeds after the first is released
#   g7  receipt identity complete                       (evaluated by the receipt builder)
#   g8  the engine banner reports prefix caching actually ON and mamba_cache_mode align
#   g9  KV capacity at 262,144 has not regressed against the 265,122-token baseline
#
# g8 exists because without it the run proves nothing: a flag that argparse accepts and the
# engine then discards would produce seven green gates that say only "the old profile still
# works". g9 exists because prefix caching adds block bookkeeping to a profile that already
# fits 262,144 tokens into a hair over 9 GiB of KV; if it costs capacity, that cost is a
# published fact, not a rounding error to be absorbed.
#
# Differences from qual.sh, all deliberate:
#   - the image is the promoted superset tag, pinned by manifest digest and verified before
#     any launch, instead of the public r34 digest
#   - NO source bind mounts. The modules are in the image; `stage` proves it by resolving
#     each one through the import machinery inside the image and hashing the resolved
#     origin, with no GPU attached. qual.sh mounted three files read-only; this cannot.
#   - --enable-prefix-caching --mamba-cache-mode align on every launch
#
# One invocation is one freshly started vLLM process serving one subset of gates, so the
# window is interruptible at a gate boundary and the user's live qwen38-27b service can be
# restored at any point.
#
#   qualify_apc_5090.sh stage                    image identity + in-image modules, no GPU
#   TAG=E  GATES='g1 g2 g5 g6' qualify_apc_5090.sh gates
#   TAG=B3 GATES='g1 g3'       qualify_apc_5090.sh gates
#   TAG=F  GATES='g1 g4'       qualify_apc_5090.sh gates
#
# Every gate's raw stdout/stderr is captured. Nothing is hidden.
set -uo pipefail

W=${W:-/home/mbelleau/qualapc}
QW=${QW:-/home/mbelleau/qual5090}   # corpus, image fixture, harness, warm caches
OUT=$W/out
LOGS=$W/logs

IMAGE=${IMAGE:-localhost/vllm:gg-r34-patched-apc}
IMAGE_MANIFEST=${IMAGE_MANIFEST:-sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218}
PARENT_TAG=${PARENT_TAG:-localhost/vllm:gg-r34-patched}
PARENT_MANIFEST=sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27

# The four modules that must be inside the image, at these bytes, resolved by import.
VLLM=/opt/venv/lib/python3.12/site-packages/vllm
EXL3_SHA=2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee
QWEN_SHA=04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7
MTP_SHA=0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab
SCHED_SHA=b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf

# The qualified profile, from receipts/qualification-5090-context.json identity.qualified_profile.
UTIL=${UTIL:-0.955}
ALLOC_CONF=${ALLOC_CONF:-expandable_segments:True}
MAXLEN=${MAXLEN:-262144}
MAXPIX=${MAXPIX:-8388608}
MTP=${MTP:-3}
TOKENS=${TOKENS:-261800}
PORT=${PORT:-8251}
# Baselines this run is gated against, from the same receipt. Not from apc-poison-repro.json,
# whose 223,677 tokens were measured on the reporter's 196,608 / util 0.92 profile and are
# not comparable with the qualified one.
KV_TOKENS_BASELINE=${KV_TOKENS_BASELINE:-265122}
CONCURRENCY_BASELINE=${CONCURRENCY_BASELINE:-1.01}

UUID=GPU-506a575d-01d7-b12e-9a0a-c1ab5f38ae0a
REPO_DIR=/mnt/vault/llm/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
COMPILE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'
PREFLIGHT_MAX_USED_MIB=${PREFLIGHT_MAX_USED_MIB:-1500}

mkdir -p "$OUT" "$LOGS"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOGS/driver.log"; }
die() { log "FATAL: $*"; exit 2; }
since() { python3 -c 'import sys;print(round(float(sys.argv[2])-float(sys.argv[1]),3))' "$1" "$(date +%s.%N)"; }

# --------------------------------------------------------------------------- stage
# Image identity and in-image module identity, CPU only, no GPU device attached. Fails
# closed: a mismatch here means every gate below would be measuring something else.
stage_verify() {
  local got
  got=$(podman image inspect -f '{{index .RepoDigests 0}}' "$IMAGE" | sed 's/.*@//') ||
    die "image $IMAGE is not present on this host"
  [[ $got == "$IMAGE_MANIFEST" ]] ||
    die "image $IMAGE manifest $got != promoted digest $IMAGE_MANIFEST"
  log "image identity verified: $IMAGE ($got)"

  podman run --rm --network none --entrypoint /bin/bash "$IMAGE" -lc "
    set -eu
    printf '%s  %s\n' \
      $EXL3_SHA  $VLLM/model_executor/layers/quantization/exl3.py \
      $QWEN_SHA  $VLLM/model_executor/models/qwen3_5.py \
      $MTP_SHA   $VLLM/model_executor/models/qwen3_5_mtp.py \
      $SCHED_SHA $VLLM/v1/core/sched/scheduler.py | sha256sum -c -
    /opt/venv/bin/python - <<'PY'
import hashlib, importlib.util, json
want = {'vllm.model_executor.layers.quantization.exl3': '$EXL3_SHA',
        'vllm.model_executor.models.qwen3_5': '$QWEN_SHA',
        'vllm.model_executor.models.qwen3_5_mtp': '$MTP_SHA',
        'vllm.v1.core.sched.scheduler': '$SCHED_SHA'}
origins = {}
for name, digest in want.items():
    origin = importlib.util.find_spec(name).origin
    got = hashlib.sha256(open(origin, 'rb').read()).hexdigest()
    assert got == digest, (name, origin, got, digest)
    origins[name] = {'origin': origin, 'sha256': got}
print(json.dumps({'import_resolution_matches_published_bytes': True,
                  'modules': origins}, indent=2))
PY
" >"$OUT/module-verify.txt" 2>&1 || { cat "$OUT/module-verify.txt"; die "in-image module verification failed"; }
  cat "$OUT/module-verify.txt"

  # The patched scheduler must differ from the parent's, or the promotion is a no-op.
  local parent_digest parent_sched
  parent_digest=$(podman image inspect -f '{{index .RepoDigests 0}}' "$PARENT_TAG" | sed 's/.*@//') ||
    die "parent image $PARENT_TAG is not present on this host"
  [[ $parent_digest == "$PARENT_MANIFEST" ]] ||
    die "parent $PARENT_TAG manifest $parent_digest != $PARENT_MANIFEST"
  parent_sched=$(podman run --rm --network none --entrypoint /bin/bash \
    "$PARENT_TAG" -lc "sha256sum $VLLM/v1/core/sched/scheduler.py | cut -d' ' -f1")
  [[ $parent_sched == "$SCHED_SHA" ]] &&
    die "parent image already carries the patched scheduler; the promotion would be vacuous"
  log "parent $PARENT_TAG ($parent_digest) scheduler is $parent_sched (unpatched); promoted image is $SCHED_SHA"

  # Both flags must exist in this build, or a card would print a flag that does not work.
  podman run --rm --network none --entrypoint /opt/venv/bin/vllm "$IMAGE" serve --help=all 2>&1 |
    grep -E -- '^ *--(no-)?enable-prefix-caching|^ *--mamba-cache-mode' >"$OUT/flags.txt" ||
    die "this build does not expose --enable-prefix-caching / --mamba-cache-mode"
  cat "$OUT/flags.txt"

  python3 - "$OUT/stage.json" "$IMAGE" "$IMAGE_MANIFEST" "$parent_sched" <<'PY'
import json, pathlib, subprocess, sys, time
out, image, manifest, parent_sched = sys.argv[1:5]
cfg = subprocess.run(["podman", "image", "inspect", image, "--format", "{{.Id}}"],
                     capture_output=True, text=True, check=True).stdout.strip()
labels = json.loads(subprocess.run(
    ["podman", "image", "inspect", image, "--format", "{{json .Labels}}"],
    capture_output=True, text=True, check=True).stdout)
pathlib.Path(out).write_text(json.dumps({
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "image": image, "manifest_digest": manifest, "config_id": cfg,
    "parent_scheduler_sha256": parent_sched,
    "labels": labels,
    "source_bind_mounts_used_by_this_harness": [],
}, indent=2) + "\n")
print(json.dumps({"stage": "ok", "config_id": cfg}))
PY
}

# --------------------------------------------------------------------------- gates
stage_gates() {
  local TAG=${TAG:?TAG required} GATES=${GATES:?GATES required}
  local CTN=qualapc-$TAG
  local SRV=$LOGS/server-$TAG.log
  local PHASE=$W/phase-$TAG
  local SAMPLES=$LOGS/gpu-samples-$TAG.jsonl
  local STATUS=$OUT/gates-$TAG.jsonl
  local CMDFILE=$OUT/command-$TAG.json
  local api=http://127.0.0.1:$PORT/v1
  local SAMPLER_PID= SERVER_SH=
  : >"$STATUS"; : >"$SAMPLES"

  phase() { echo "$1" >"$PHASE"; log "PHASE $1"; }
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
  }
  trap cleanup EXIT INT TERM

  # ---- ownership. A handover token written only after the outgoing owner names this agent
  # over hub. The memory reading is a secondary sanity check, never the gate: a chain that
  # tears its server down between rows leaves the card reading idle while still owned.
  [[ -s $W/HANDOVER ]] || die "no $W/HANDOVER token; the card is someone else's until its holder names this agent"
  log "handover token: $(tr '\n' ' ' <"$W/HANDOVER")"
  local used
  used=$(nvidia-smi "--id=$UUID" --query-gpu=memory.used --format=csv,noheader,nounits)
  (( used <= PREFLIGHT_MAX_USED_MIB )) || die "GPU busy (${used} MiB used)"

  [[ -f $OUT/stage.json ]] || die "run the stage subcommand first: image identity is not verified"
  local REV; REV=$(cat "$REPO_DIR/refs/main")
  local MODEL=/models/ctx-repo/snapshots/$REV
  local SPEC; SPEC=$(printf '{"method":"mtp","num_speculative_tokens":%d}' "$MTP")
  local MM; MM=$(printf '{"truncation":false,"max_pixels":%d}' "$MAXPIX")

  log "attempt TAG=$TAG gates='$GATES' util=$UTIL len=$MAXLEN mtp=$MTP rev=$REV image=$IMAGE"
  podman rm -f "$CTN" >/dev/null 2>&1
  phase before_load
  point_sample before_load
  python3 "$QW/gpu_sampler.py" "$UUID" "$PHASE" "$SAMPLES" 1.0 &
  SAMPLER_PID=$!

  local SERVE_ARGS=(serve "$MODEL"
    --served-model-name m --quantization exl3
    --quantization-config "$CFG"
    --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL"
    --kv-cache-dtype fp8 --max-num-seqs 1 --max-num-batched-tokens 2048
    --speculative-config "$SPEC"
    --mm-processor-kwargs "$MM"
    --compilation-config "$COMPILE"
    --enable-prefix-caching --mamba-cache-mode align
    --host 127.0.0.1 --port "$PORT")
  # No -v over site-packages anywhere in this argv. That is the point of the promotion.
  local RUN_ARGS=(run --rm --name "$CTN"
    --device nvidia.com/gpu=all --ipc=host --network host
    -v "$REPO_DIR:/models/ctx-repo:ro"
    -v "$QW/cache/jit:/cache/jit:rw"
    -v "$QW/cache/exl3-online:/cache/exl3-online:rw"
    -e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1
    -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e "PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF"
    --entrypoint /opt/venv/bin/vllm
    "$IMAGE")

  python3 - "$CMDFILE" "$TAG" "$UTIL" "$MAXPIX" "$MTP" "$MAXLEN" "$ALLOC_CONF" "$REV" \
    "$IMAGE" "$IMAGE_MANIFEST" "${RUN_ARGS[@]}" -- "${SERVE_ARGS[@]}" <<'PY'
import json, pathlib, sys
out, tag, util, maxpix, mtp, maxlen, alloc, rev, image, manifest = sys.argv[1:11]
rest = sys.argv[11:]
split = rest.index("--")
podman_argv = ["podman"] + rest[:split]
mounts = [podman_argv[i + 1] for i, a in enumerate(podman_argv) if a == "-v"]
payload = {"tag": tag, "gpu_memory_utilization": float(util),
           "max_pixels": int(maxpix), "mtp_depth": int(mtp),
           "max_model_len": int(maxlen),
           "prefix_caching_flag_passed": True,
           "mamba_cache_mode_flag_passed": "align",
           "pytorch_cuda_alloc_conf": alloc,
           "model_revision": rev,
           "image": image, "image_manifest_digest": manifest,
           "source_bind_mounts_over_site_packages":
               [m for m in mounts if "site-packages" in m],
           "podman_argv": podman_argv,
           "vllm_argv": ["/opt/venv/bin/vllm"] + rest[split + 1:]}
assert not payload["source_bind_mounts_over_site_packages"], payload
pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n")
PY
  [[ $? == 0 ]] || die "command record refused: a source bind mount is present"

  local START_T; START_T=$(date +%s.%N)
  phase startup
  podman "${RUN_ARGS[@]}" "${SERVE_ARGS[@]}" >"$SRV" 2>&1 &
  SERVER_SH=$!

  local READY=0
  for _ in $(seq 1 480); do
    if curl -fsS --max-time 10 "$api/models" 2>/dev/null |
         python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if any(x.get("id")=="m" for x in d["data"]) else 1)' 2>/dev/null
    then READY=1; break; fi
    kill -0 "$SERVER_SH" 2>/dev/null || break
    grep -qE 'EngineCore failed|ValidationError|unrecognized arguments|No available memory for the cache blocks|Traceback' "$SRV" && break
    sleep 5
  done
  local STARTUP_WALL; STARTUP_WALL=$(since "$START_T")

  # Everything g1, g8 and g9 are decided on. The prefix-caching and mamba lines are the
  # additions: what the engine CHOSE, not what the flag asked for.
  local FACTS_RE='Free memory on device \([0-9.]+/[0-9.]+ GiB\) on startup\. Desired GPU memory utilization is \([0-9.]+, [0-9.]+ GiB\)\. Actual usage is .{0,420}|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds|Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB|init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s \(compilation: [0-9.]+ s\)|Setting attention block size to [0-9]+ tokens.{0,70}|Padding mamba page size by [0-9.]+%.{0,70}|Add [0-9]+ padding layers, may waste at most [0-9.]+% KV cache memory|Using fp8 data type to store kv cache|[0-9.]+ GiB KV cache is needed[^)]{0,60}|num_spec_tokens=[0-9]+|enable_prefix_caching=[A-Za-z]+|.enable_prefix_caching.: [A-Za-z]+|mamba_cache_mode=.{0,20}|.mamba_cache_mode.: .[a-z]+.|mamba_block_size=[0-9]+|Prefix caching in Mamba cache .{0,60}|prefix_caching_hash_algo=.{0,20}|Initializing a V1 LLM engine \(v[0-9A-Za-z.+_-]+\)'

  if [[ $READY != 1 ]]; then
    phase startup_failed
    point_sample startup_failed
    grep -oE "$FACTS_RE|ValueError: .{0,240}|To serve at least one request .{0,240}" "$SRV" |
      tail -30 >"$OUT/startup-fail-$TAG.txt"
    cat "$OUT/startup-fail-$TAG.txt"
    record gate1_startup_native_allocation 1 "$STARTUP_WALL" \
      "logs/server-$TAG.log" "out/startup-fail-$TAG.txt"
    log "STARTUP FAILED TAG=$TAG"
    exit 3
  fi

  phase startup_done
  point_sample after_startup
  grep -oE "$FACTS_RE" "$SRV" | sed 's/^ *//' | sort -u >"$OUT/startup-facts-$TAG.txt"
  cat "$OUT/startup-facts-$TAG.txt"
  grep -F 'Initializing a V1 LLM engine' "$SRV" | head -1 >"$OUT/engine-config-$TAG.txt"

  local g T RC WALL
  for g in $GATES; do
  case $g in
  g1)
    record gate1_startup_native_allocation 0 "$STARTUP_WALL" \
      "out/startup-facts-$TAG.txt" "out/command-$TAG.json" "logs/server-$TAG.log"
    # g8 and g9 are decided from the same banner, so they are recorded with g1 rather than
    # given a launch of their own. Both are hard failures: a green g2-g6 on a server that
    # silently disabled prefix caching would be a false qualification.
    python3 - "$OUT/startup-facts-$TAG.txt" "$OUT/engine-config-$TAG.txt" \
      "$OUT/banner-$TAG.json" "$KV_TOKENS_BASELINE" "$CONCURRENCY_BASELINE" "$MAXLEN" <<'PY'
import json, pathlib, re, sys
facts_path, cfg_path, out, kv_base, conc_base, maxlen = sys.argv[1:7]
facts = pathlib.Path(facts_path).read_text()
cfg = pathlib.Path(cfg_path).read_text()
blob = facts + "\n" + cfg


def first(pattern, default=None):
    m = re.search(pattern, blob)
    return m.group(1) if m else default


apc = first(r"['\"]?enable_prefix_caching['\"]?[=:] ?([A-Za-z]+)")
mode = first(r"['\"]?mamba_cache_mode['\"]?[=:] ?['\"]?([a-z]+)")
kv_tokens = first(r"GPU KV cache size: ([0-9,]+) tokens")
conc_pair = re.search(r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x", blob)
payload = {
    "enable_prefix_caching_banner": apc,
    "mamba_cache_mode_banner": mode,
    "mamba_block_size_banner": first(r"mamba_block_size=([0-9]+)"),
    "attention_block_size_tokens": first(r"Setting attention block size to ([0-9]+) tokens"),
    "kv_padding_layers": first(r"Add ([0-9]+) padding layers"),
    "kv_padding_waste_pct": first(r"may waste at most ([0-9.]+)% KV cache memory"),
    "available_kv_cache_gib": first(r"Available KV cache memory: ([0-9.]+) GiB"),
    "gpu_kv_cache_tokens": int(kv_tokens.replace(",", "")) if kv_tokens else None,
    "maximum_concurrency_window": conc_pair.group(1) if conc_pair else None,
    "maximum_concurrency": float(conc_pair.group(2)) if conc_pair else None,
    "prefix_caching_hash_algo": first(r"prefix_caching_hash_algo=([A-Za-z0-9_']+)"),
    "mamba_prefix_caching_note": first(r"(Prefix caching in Mamba cache .{0,60})"),
    "kv_tokens_baseline": int(kv_base),
    "concurrency_baseline": float(conc_base),
    "max_model_len": int(maxlen),
}
payload["gate8_banner_proves_prefix_caching_on"] = (
    payload["enable_prefix_caching_banner"] == "True"
    and payload["mamba_cache_mode_banner"] == "align")
payload["gate9_kv_capacity_not_regressed"] = (
    payload["gpu_kv_cache_tokens"] is not None
    and payload["gpu_kv_cache_tokens"] >= int(maxlen)
    and payload["maximum_concurrency"] is not None
    and payload["maximum_concurrency"] >= 1.0)
if payload["gpu_kv_cache_tokens"] is not None:
    delta = payload["gpu_kv_cache_tokens"] - int(kv_base)
    payload["kv_tokens_delta_vs_baseline"] = delta
    payload["kv_tokens_pct_vs_baseline"] = round(
        100.0 * payload["gpu_kv_cache_tokens"] / int(kv_base), 3)
    payload["prefix_caching_kv_cost_tokens"] = -delta if delta < 0 else 0
pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
sys.exit(0 if payload["gate8_banner_proves_prefix_caching_on"]
         and payload["gate9_kv_capacity_not_regressed"] else 65)
PY
    RC=$?
    record gate8_banner_prefix_caching_actually_on $(( RC == 0 ? 0 : RC )) 0 \
      "out/banner-$TAG.json" "out/startup-facts-$TAG.txt" "out/engine-config-$TAG.txt"
    record gate9_kv_capacity_not_regressed $(( RC == 0 ? 0 : RC )) 0 \
      "out/banner-$TAG.json" "out/startup-facts-$TAG.txt"
    if (( RC != 0 )); then
      log "GATE 8/9 FAILED TAG=$TAG: the banner does not prove prefix caching is on, or KV regressed"
      log "stopping this process; a green g2-g6 underneath a disabled cache would be a false qualification"
      phase gate89_failed
      break
    fi
    ;;
  g2)
    phase g2_long_needle
    T=$(date +%s.%N)
    python3 "$QW/tools/longctx.py" --url "$api" --corpus "$QW/corpus/literary" \
      --tokens "$TOKENS" --depth 0.5 --require-pass --out "$OUT/g2-needle-$TAG.json" \
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
    python3 "$QW/tools/vision_eval.py" --url "$api" --label "qualapc-$TAG" --cases 10 \
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
      --tokens 256 --concurrency 1 1 1 --prefill-tokens --label "qualapc-g5-$TAG" \
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
      --tokens "$TOKENS" --depth 0.25 --seed 20260816 --require-pass \
      --out "$OUT/gate6-needle-$TAG.json" \
      >"$LOGS/gate6-$TAG.out" 2>"$LOGS/gate6-$TAG.err"
    RC=$?
    record gate6_second_long_request_after_release $RC "$(since "$T")" \
      "out/gate6-needle-$TAG.json" "logs/gate6-$TAG.out" "logs/gate6-$TAG.err"
    # With prefix caching on, g6 shares a long literary prefix with g2 in the same process,
    # so the cache counters across the pair are the only in-qualification evidence that the
    # cache was not merely enabled but actually used. Not a gate; a recorded fact.
    curl -fsS --max-time 20 "http://127.0.0.1:$PORT/metrics" >"$OUT/metrics-after-g6-$TAG.txt" 2>&1
    ;;
  *) die "unknown gate $g" ;;
  esac
  done

  phase shutdown
  podman stop -t 90 "$CTN" >>"$LOGS/driver.log" 2>&1
  wait "$SERVER_SH" 2>/dev/null
  sleep 10
  phase after_release
  point_sample after_release
  sleep 3
  kill "$SAMPLER_PID" 2>/dev/null
  SAMPLER_PID=
  log "QUALAPC_DONE TAG=$TAG"
}

case ${1:-} in
  stage) stage_verify ;;
  gates) stage_gates ;;
  *) die "usage: $0 {stage|gates}   (gates needs TAG and GATES in the environment)" ;;
esac
