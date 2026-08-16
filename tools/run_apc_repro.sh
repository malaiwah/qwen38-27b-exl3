#!/bin/bash
# Prefix-cache / mamba-align poisoning reproduction on the physical RTX 5090 (AIBoss).
#
# One invocation is one phase, so the window is interruptible at a phase boundary and the
# user's live qwen38-27b service can be restored at any point:
#
#   run_apc_repro.sh stage        verify both image identities + baked modules, freeze prompts
#   run_apc_repro.sh arm A        unpatched release image, --enable-prefix-caching + align
#   run_apc_repro.sh arm B        unpatched release image, prefix caching off  (our condition)
#   run_apc_repro.sh arm C        patched superset image (PR #51113), prefix caching + align
#   run_apc_repro.sh arm D        arm C plus the PR #51812 GDN gate module, mounted
#   run_apc_repro.sh reuse        arm E: what prefix reuse actually buys (perf, not correctness)
#   run_apc_repro.sh receipt      assemble receipts/apc-poison-repro.json
#   run_apc_repro.sh restore      restore the user's live qwen38-27b service and prove it healthy
#
# Every arm is a freshly started server on the same card and changes exactly one thing
# against the arm before it. Arm D is reported on its own row and is never folded into the
# arm C verdict, so the two absent upstream fixes are never conflated.
set -euo pipefail

W=${W:-/home/mbelleau/apcrepro}
OUT=$W/out
LOGS=$W/logs
TOOLS=$W/tools
PORT=${PORT:-8241}
HF=${HF:-/mnt/vault/llm/huggingface}
CTX_REPO=$HF/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context
CTX_REV=c45c273b0d6ef2859cb2d85b36dd52253c80d878
JIT=${JIT:-/home/mbelleau/qual5090/cache/jit}
ONLINE_CACHE=${ONLINE_CACHE:-/home/mbelleau/qual5090/cache/exl3-online}
VENV=/opt/venv/lib/python3.12/site-packages/vllm

RELEASE_IMAGE=localhost/vllm:gg-r34-patched
RELEASE_MANIFEST=sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27
APC_IMAGE=localhost/vllm:gg-r34-patched-apc
APC_MANIFEST=sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218

EXL3_SHA=2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee
QWEN_SHA=04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7
MTP_SHA=0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab
SCHED_SHA=b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf
GDN_SHA=7cd3f5fe763b621048af4817951a841d99c8b700d9a56ded27ccaca5a56ccbe0
GDN_DEST=$VENV/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py

# The reporter's own serving configuration, taken verbatim where it does not conflict with
# what this edition needs to load at all. Every deviation is recorded in out/deviations.json.
QUANT_CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
MAXLEN=196608
GPU_UTIL=0.92
SEQS=1
BATCHED=2048
SPEC='{"method":"mtp","num_speculative_tokens":3}'
MM_CFG='{"truncation":false,"max_pixels":8388608}'
COMPILE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'
READY_TIMEOUT=${READY_TIMEOUT:-2400}

CONTAINER=
SAMPLER_PID=
declare -a podman_argv=() serve_argv=()

mkdir -p "$OUT" "$LOGS" "$TOOLS"

die() { echo "FATAL: $*" >&2; exit 2; }
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

cleanup() {
  local rc=$?
  [[ -n $SAMPLER_PID ]] && kill "$SAMPLER_PID" 2>/dev/null
  if [[ -n $CONTAINER ]]; then
    podman stop -t 20 "$CONTAINER" >/dev/null 2>&1 || true
    podman rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi
  return $rc
}
trap cleanup EXIT INT TERM

verify_image() {
  local image=$1 want=$2 got
  got=$(podman image inspect -f '{{index .RepoDigests 0}}' "$image" | sed 's/.*@//') ||
    die "image $image missing"
  [[ $got == "$want" ]] || die "image $image manifest $got != $want"
  say "image identity verified: $image ($got)"
}

# The patch modules must be the published bytes *inside the image*, resolved by the import
# machinery, with no bind mount, before any measurement. CPU-only container, no GPU device.
verify_modules_in_image() {
  local image=$1 tag=$2 extra=${3:-}
  podman run --rm --network none --entrypoint /bin/bash "$image" -lc "
    set -eu
    { printf '%s  %s\n' \
        $EXL3_SHA $VENV/model_executor/layers/quantization/exl3.py \
        $QWEN_SHA $VENV/model_executor/models/qwen3_5.py \
        $MTP_SHA $VENV/model_executor/models/qwen3_5_mtp.py
      $extra
    } | sha256sum -c -
    /opt/venv/bin/python - <<'PY'
import hashlib, importlib.util, json
want = {'vllm.model_executor.layers.quantization.exl3': '$EXL3_SHA',
        'vllm.model_executor.models.qwen3_5': '$QWEN_SHA',
        'vllm.model_executor.models.qwen3_5_mtp': '$MTP_SHA'}
origins = {}
for name, digest in want.items():
    origin = importlib.util.find_spec(name).origin
    got = hashlib.sha256(open(origin, 'rb').read()).hexdigest()
    assert got == digest, (name, origin, got, digest)
    origins[name] = origin
sched = '$VENV/v1/core/sched/scheduler.py'
print(json.dumps({'import_resolution_matches_published_bytes': True,
                  'origins': origins,
                  'scheduler_sha256': hashlib.sha256(open(sched, 'rb').read()).hexdigest()}))
PY
" | tee "$OUT/module-verify-$tag.txt"
}

start_sampler() {
  nvidia-smi --query-gpu=timestamp,memory.total,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -lms 500 >"$LOGS/gpu-$1.csv" 2>/dev/null &
  SAMPLER_PID=$!
}

startup_facts() {
  local log=$1 tag=$2
  {
    grep -oE 'Setting attention block size to [0-9]+ tokens[^.]*.' "$log" || true
    grep -oE 'mamba page size[^.]*.' "$log" || true
    grep -oE 'Available KV cache memory: [0-9.]+ GiB' "$log" || true
    grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$log" || true
    grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$log" || true
    grep -oE 'enable_prefix_caching=[A-Za-z]+' "$log" | head -1 || true
    grep -oE 'mamba_cache_mode=[A-Za-z.\047]+' "$log" | head -1 || true
    grep -oE 'mamba_block_size=[0-9]+' "$log" | head -1 || true
    grep -oE 'num_spec_tokens=[0-9]+' "$log" | head -1 || true
    grep -oE 'cudagraph_capture_sizes=\[[^]]*\]' "$log" | head -1 || true
    grep -oE 'Using [A-Z_]+ attention backend[^.]*' "$log" | head -1 || true
    grep -oE 'Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds' "$log" || true
    grep -oE 'init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s[^)]*.' "$log" || true
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
    if podman container exists "$CONTAINER" 2>/dev/null; then
      seen=1
    elif ((seen == 1)); then
      say "container $CONTAINER exited before readiness"
      return 1
    elif ((i > 90)); then
      say "container $CONTAINER was never created"
      return 1
    fi
    sleep 1
  done
  say "server not ready within ${READY_TIMEOUT}s"
  return 1
}

stop_server() {
  podman stop -t 30 "$CONTAINER" >/dev/null 2>&1 || true
  podman rm -f "$CONTAINER" >/dev/null 2>&1 || true
  CONTAINER=
  [[ -n $SAMPLER_PID ]] && { kill "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=; }
  local i
  for ((i = 0; i < 120; i++)); do
    [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 700 ]] && return 0
    sleep 1
  done
  say "warning: GPU memory still held after teardown"
}

require_free_gpu() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  ((used < 1500)) || die "GPU busy (${used} MiB used); another owner still holds the card"
}

# ------------------------------------------------------------------ arm definitions
declare -a ARM_ENV=() ARM_ARGS=() ARM_MOUNTS=()
ARM_IMAGE=; ARM_DESC=

configure_arm() {
  ARM_ENV=(); ARM_ARGS=(); ARM_MOUNTS=()
  case $1 in
  A)
    ARM_IMAGE=$RELEASE_IMAGE
    ARM_DESC='release image gg-r34-patched (PR #51113 ABSENT), --enable-prefix-caching --mamba-cache-mode align: the reporter'"'"'s condition'
    ARM_ARGS=(--enable-prefix-caching --mamba-cache-mode align)
    ;;
  B)
    ARM_IMAGE=$RELEASE_IMAGE
    ARM_DESC='release image gg-r34-patched, prefix caching off: the condition every published measurement ran in'
    ARM_ARGS=(--no-enable-prefix-caching)
    ;;
  C)
    ARM_IMAGE=$APC_IMAGE
    ARM_DESC='superset image gg-r34-patched-apc (PR #51113 PRESENT), --enable-prefix-caching --mamba-cache-mode align: the fix under test'
    ARM_ARGS=(--enable-prefix-caching --mamba-cache-mode align)
    ;;
  D)
    ARM_IMAGE=$APC_IMAGE
    ARM_DESC='arm C plus the PR #51812 GDN speculative gate module bind-mounted read-only; reported separately, never merged into the arm C verdict'
    ARM_ARGS=(--enable-prefix-caching --mamba-cache-mode align)
    ARM_MOUNTS=(-v "$TOOLS/vllm-qwen-gdn-spec-gates.py:$GDN_DEST:ro")
    ;;
  E)
    ARM_IMAGE=$APC_IMAGE
    ARM_DESC='superset image gg-r34-patched-apc, --enable-prefix-caching --mamba-cache-mode align, prefix-reuse economics only (performance, not correctness)'
    ARM_ARGS=(--enable-prefix-caching --mamba-cache-mode align)
    ;;
  *) die "unknown arm '$1'" ;;
  esac
}

compose() {
  local arm=$1
  CONTAINER=apcrepro-$arm
  podman_argv=(
    podman run --rm --name "$CONTAINER"
    --device nvidia.com/gpu=all --ipc=host --network host
    -v "$CTX_REPO:/models/ctx-repo:ro"
    -v "$JIT:/cache/jit:rw"
    -v "$ONLINE_CACHE:/cache/exl3-online:rw"
    "${ARM_MOUNTS[@]}"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1
    -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128
    "${ARM_ENV[@]}"
    --entrypoint /opt/venv/bin/vllm
    "$ARM_IMAGE"
  )
  serve_argv=(
    serve "/models/ctx-repo/snapshots/$CTX_REV"
    --served-model-name m
    --quantization exl3 --quantization-config "$QUANT_CFG"
    --max-model-len "$MAXLEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-num-seqs "$SEQS"
    --kv-cache-dtype fp8
    --max-num-batched-tokens "$BATCHED"
    "${ARM_ARGS[@]}"
    --speculative-config "$SPEC"
    --mm-processor-kwargs "$MM_CFG"
    --compilation-config "$COMPILE"
    --host 127.0.0.1 --port "$PORT"
  )
  python3 - "$OUT/command-$arm.json" "$arm" "$ARM_IMAGE" "$ARM_DESC" \
    "${podman_argv[@]}" -- "${serve_argv[@]}" <<'PY'
import json, pathlib, sys
out, arm, image, desc, *rest = sys.argv[1:]
split = rest.index("--")
pathlib.Path(out).write_text(json.dumps({
    "schema": "qwen38-apc-repro-command/1", "arm": arm, "image": image,
    "description": desc, "podman_argv": rest[:split], "vllm_argv": rest[split + 1:]},
    indent=2) + "\n")
PY
}

launch() {
  local arm=$1 log=$2
  say "launching arm $arm"
  "${podman_argv[@]}" "${serve_argv[@]}" >"$log" 2>&1 &
  wait_ready "$log"
}

# ------------------------------------------------------------------ phases
phase_stage() {
  verify_image "$RELEASE_IMAGE" "$RELEASE_MANIFEST"
  verify_image "$APC_IMAGE" "$APC_MANIFEST"
  verify_modules_in_image "$RELEASE_IMAGE" release
  verify_modules_in_image "$APC_IMAGE" apc \
    "printf '%s  %s\n' $SCHED_SHA $VENV/v1/core/sched/scheduler.py"
  # The release image must NOT carry the scheduler fix: that asymmetry is the experiment.
  grep -q "$SCHED_SHA" "$OUT/module-verify-release.txt" &&
    die "release image already carries the #51113 scheduler; arms A and C are not distinct"
  grep -q "$SCHED_SHA" "$OUT/module-verify-apc.txt" ||
    die "superset image does not carry the #51113 scheduler"
  [[ -f $TOOLS/vllm-qwen-gdn-spec-gates.py ]] ||
    die "missing $TOOLS/vllm-qwen-gdn-spec-gates.py (needed for arm D)"
  echo "$GDN_SHA  $TOOLS/vllm-qwen-gdn-spec-gates.py" | sha256sum -c -
  [[ -d $CTX_REPO/snapshots/$CTX_REV ]] || die "context snapshot missing at $CTX_REPO"

  if [[ ! -f $W/prompts.json ]]; then
    say "freezing the prompt set (CPU only, no GPU device)"
    podman run --rm --network none \
      -v "$CTX_REPO:/models/ctx-repo:ro" -v "$W:/work:rw" \
      --entrypoint /opt/venv/bin/python "$RELEASE_IMAGE" \
      /work/tools/apc_poison_probe.py build \
      --model "/models/ctx-repo/snapshots/$CTX_REV" --out /work/prompts.json |
      tee "$OUT/prompt-build.txt"
  fi

  python3 - "$OUT/stage.json" "$W" "$RELEASE_IMAGE" "$RELEASE_MANIFEST" \
    "$APC_IMAGE" "$APC_MANIFEST" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, work, rel, rel_d, apc, apc_d = sys.argv[1:]
work = pathlib.Path(work)
def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
prompts = json.loads((work / "prompts.json").read_text())
payload = {
    "schema": "qwen38-apc-repro-stage/1",
    "host": run("hostname"),
    "gpu": run("nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,compute_mode",
               "--format=csv,noheader"),
    "podman_version": run("podman", "--version"),
    "images": {
        "release": {"tag": rel, "manifest_digest": rel_d,
                    "config_id": run("podman", "image", "inspect", "-f", "{{.Id}}", rel)},
        "apc_superset": {"tag": apc, "manifest_digest": apc_d,
                         "config_id": run("podman", "image", "inspect", "-f", "{{.Id}}", apc)},
    },
    "harness_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((work / "tools").glob("*")) if p.is_file()},
    "prompts_sha256": prompts["prompts_sha256"],
    "prompt_set": {
        "seed": prompts["seed"],
        "mamba_block_tokens": prompts["mamba_block_tokens"],
        "head_tokens": prompts["head_tokens"],
        "tail_tokens": prompts["tail_tokens"],
        "needles_planted": len(prompts["needles"]),
        "correctness_requests": len(prompts["correctness_requests"]),
        "prompt_tokens": [r["prompt_tokens"] for r in prompts["correctness_requests"]],
        "prompt_tokens_mod_mamba_block": [r["prompt_tokens_mod_mamba_block"]
                                          for r in prompts["correctness_requests"]],
        "no_prompt_is_block_aligned": all(r["prompt_tokens_mod_mamba_block"]
                                          for r in prompts["correctness_requests"]),
        "off_block_chunk_ends_per_request": [len(r["chunk_ends_off_block"])
                                             for r in prompts["correctness_requests"]],
        "reuse_cases": [{"tag": c["length_tag"], "prompt_tokens": c["prompt_tokens"]}
                        for c in prompts["reuse_cases"]],
    },
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"stage": out, "prompts_sha256": payload["prompts_sha256"],
                  "no_prompt_is_block_aligned":
                      payload["prompt_set"]["no_prompt_is_block_aligned"]}))
PY
  say "stage complete"
}

phase_arm() {
  local arm=$1
  local log=$LOGS/server-$arm.log
  local sha
  sha=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["prompts_sha256"])' "$W/prompts.json")
  require_free_gpu
  configure_arm "$arm"
  compose "$arm"
  start_sampler "$arm"
  launch "$arm" "$log" || { startup_facts "$log" "$arm"; tail -40 "$log" >&2; stop_server; die "arm $arm failed to start"; }
  startup_facts "$log" "$arm"
  python3 "$TOOLS/apc_poison_probe.py" run \
    --base "http://127.0.0.1:$PORT" --model m --prompts "$W/prompts.json" \
    --arm "$arm" --description "$ARM_DESC" --expect-prompts-sha256 "$sha" \
    --out "$OUT/arm-$arm.json" 2>&1 | tee "$LOGS/probe-$arm.out"
  stop_server
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm "$arm" \
    --out "$OUT/windows-$arm.json"
  say "arm $arm complete"
}

phase_reuse() {
  local arm=E
  local log=$LOGS/server-$arm.log
  require_free_gpu
  configure_arm "$arm"
  compose "$arm"
  start_sampler "$arm"
  launch "$arm" "$log" || { startup_facts "$log" "$arm"; tail -40 "$log" >&2; stop_server; die "arm E failed to start"; }
  startup_facts "$log" "$arm"
  python3 "$TOOLS/apc_poison_probe.py" reuse \
    --base "http://127.0.0.1:$PORT" --model m --prompts "$W/prompts.json" \
    --arm "$arm" --description "$ARM_DESC" --out "$OUT/reuse-E.json" 2>&1 |
    tee "$LOGS/probe-$arm.out"
  stop_server
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm "$arm" \
    --out "$OUT/windows-$arm.json"
  say "arm E complete"
}

phase_verdict() {
  local -a args=()
  local arm
  for arm in A B C D; do
    [[ -f $OUT/arm-$arm.json ]] && args+=(--arm-file "$arm=$OUT/arm-$arm.json")
  done
  python3 "$TOOLS/apc_poison_probe.py" verdict "${args[@]}" --out "$OUT/verdict.json"
}

phase_receipt() {
  phase_verdict
  python3 - "$OUT" "$W" <<'PY'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1]); work = pathlib.Path(sys.argv[2])
def load(name, default=None):
    p = out / name
    return json.loads(p.read_text()) if p.exists() else default
stage = load("stage.json")
verdict = load("verdict.json")
arms, windows, commands, facts = {}, {}, {}, {}
for tag in "ABCDE":
    data = load(f"arm-{tag}.json")
    if data:
        arms[tag] = data
    w = load(f"windows-{tag}.json")
    if w:
        windows[tag] = w
    c = load(f"command-{tag}.json")
    if c:
        commands[tag] = c
    f = out / f"startup-facts-{tag}.txt"
    if f.exists():
        facts[tag] = [line for line in f.read_text().splitlines() if line.strip()]
reuse = load("reuse-E.json")
payload = {
    "arms": {tag: {"description": d["arm_description"], "summary": d["summary"],
                   "rows": d["rows"]} for tag, d in arms.items()},
    "log_windows": windows,
    "commands": commands,
    "startup_facts": facts,
    "reuse": reuse,
    "verdict": verdict,
    "stage": stage,
}
(out / "assembled.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"arms": sorted(arms), "reuse": bool(reuse),
                  "verdict": verdict and verdict["reproduction_verdict"]}))
PY
}

phase_restore() {
  say "restoring the user's live service"
  systemctl --user start qwen38-27b.service
  local i status health reachable
  for ((i = 0; i < 1500; i++)); do
    health=$(podman inspect -f '{{.State.Health.Status}}' qwen38-27b 2>/dev/null || echo none)
    [[ $health == healthy ]] && break
    sleep 2
  done
  status=$(systemctl --user is-active qwen38-27b.service || true)
  curl -sf --max-time 10 http://127.0.0.1:8000/health >/dev/null 2>&1 &&
    reachable=true || reachable=false
  python3 - "$OUT/restore.json" "$status" "$health" "$reachable" <<'PY'
import json, pathlib, subprocess, sys
out, status, health, reachable = sys.argv[1:]
def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
pathlib.Path(out).write_text(json.dumps({
    "schema": "qwen38-apc-repro-restore/1", "unit_active": status,
    "container_health": health, "health_endpoint_ok": reachable == "true",
    "image": run("podman", "inspect", "qwen38-27b", "--format", "{{.ImageName}}"),
    "image_digest": run("podman", "inspect", "qwen38-27b", "--format", "{{.ImageDigest}}"),
    "binds": run("podman", "inspect", "qwen38-27b", "--format", "{{json .HostConfig.Binds}}"),
    "state": run("podman", "inspect", "qwen38-27b", "--format", "{{.State.Status}}"),
    "started_at": run("podman", "inspect", "qwen38-27b", "--format", "{{.State.StartedAt}}"),
    "compute_apps": run("nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader"),
}, indent=2) + "\n")
print(json.dumps({"unit": status, "health": health, "endpoint": reachable}))
PY
  [[ $status == active && $health == healthy && $reachable == true ]] ||
    die "live service not healthy: unit=$status health=$health endpoint=$reachable"
  say "live service healthy"
}

case ${1:-} in
stage) phase_stage ;;
arm) phase_arm "${2:?arm letter required}" ;;
reuse) phase_reuse ;;
verdict) phase_verdict ;;
receipt) phase_receipt ;;
restore) phase_restore ;;
*) die "usage: $0 {stage|arm <A|B|C|D>|reuse|verdict|receipt|restore}" ;;
esac
