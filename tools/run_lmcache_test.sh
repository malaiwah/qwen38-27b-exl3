#!/bin/bash
# LMCache kv_both disk-reuse test-or-retire on the physical RTX 5090 (AIBoss).
#
# The one real user corruption report (cosmicnag, context edition discussion #1) has
# LMCache 0.5.2 kv_both disk mode at chunk 1600 (fs_native L2) as its sole surviving
# suspect after receipts/apc-poison-repro.json cleared prefix caching + align without
# LMCache on two independent physical 5090s. This driver puts LMCache back in the loop
# through the image's own glm52-lmcache-wrapper.sh — the exact wrapper the reporter ran —
# on the promoted four-module image (PR #51113 PRESENT), and drives the frozen poisoning
# probe (tools/apc_poison_probe.py, prompts_sha256 9978404b…, thresholds unchanged)
# through three passes:
#
#   run_lmcache_test.sh stage       identity: image, modules, lmcache version, wrapper, prompts
#   run_lmcache_test.sh control     arm L0: same wrapper, LMCACHE_MODE=off (no-LMCache control)
#   run_lmcache_test.sh coldwarm    arm L1cold: fresh server, EMPTY L2 -> 38 requests (stores)
#                                   arm L2warm: SAME live server -> 38 requests again (warm)
#   run_lmcache_test.sh restart     arm L3restart: NEW server, RETAINED L2 -> 38 requests
#                                   (reuse-after-restart: vLLM's own cache is empty by
#                                   construction, so any prefix reuse is LMCache's disk L2 —
#                                   the one path vLLM's own prefix cache never exercises)
#   run_lmcache_test.sh receipt     assemble out/assembled.json (incl. divergence vs L0)
#   run_lmcache_test.sh restore     restore the user's live qwen38-27b service and prove it
#
# Serving profile: the reporter's own stack shape, verbatim where the model loads at all —
# --max-model-len 196608 --gpu-memory-utilization 0.92 --max-num-seqs 1 --kv-cache-dtype fp8
# --max-num-batched-tokens 2048 --enable-prefix-caching --mamba-cache-mode align, MTP-3,
# FULL_DECODE_ONLY[4] — i.e. the published arm C profile, so arm C (0/38 failing) is a prior
# no-LMCache control on the identical image + profile. PYTORCH_CUDA_ALLOC_CONF is pinned to
# expandable_segments:False for ALL arms including L0, because the wrapper force-disables it
# whenever LMCache is live (vLLM rejects the pairing: LMCache registers KV by address); the
# reporter's failing runs predate his adoption of the expandable allocator, so this is his
# shape, and it keeps the control one variable away from the LMCache arms.
set -euo pipefail

W=${W:-/home/mbelleau/lmcachetest}
OUT=$W/out
LOGS=$W/logs
TOOLS=$W/tools
L2=$W/l2
PORT=${PORT:-8241}
HF=${HF:-/mnt/vault/llm/huggingface}
CTX_REPO=$HF/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context
CTX_REV=c45c273b0d6ef2859cb2d85b36dd52253c80d878
JIT=${JIT:-/home/mbelleau/qual5090/cache/jit}
ONLINE_CACHE=${ONLINE_CACHE:-/home/mbelleau/qual5090/cache/exl3-online}
VENV=/opt/venv/lib/python3.12/site-packages/vllm

APC_IMAGE=localhost/vllm:gg-r34-patched-apc
APC_MANIFEST=sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218

EXL3_SHA=2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee
QWEN_SHA=04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7
MTP_SHA=0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab
SCHED_SHA=b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf
PROBE_SHA=f415414ddf9dfc762eb872e6f6dca33ed1c040a22d3de976cd2e3ca4a65a5b1c
PROMPTS_SHA=9978404b0ce2d032ebf762e17335d7b5a37dbc424f57f61db66ecf3f946f39df
WRAPPER=/usr/local/bin/glm52-lmcache-wrapper.sh

# The reporter's serving configuration, verbatim where this edition loads at all.
QUANT_CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
MAXLEN=196608
GPU_UTIL=0.92
SEQS=1
BATCHED=2048
SPEC='{"method":"mtp","num_speculative_tokens":3}'
MM_CFG='{"truncation":false,"max_pixels":8388608}'
COMPILE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'
READY_TIMEOUT=${READY_TIMEOUT:-2400}

# Derived by the wrapper from PORT=8241: lmcache zmq 5796, http 8330, prometheus 9331.
LMC_HTTP_PORT=8330
LMC_PROM_PORT=9331

CONTAINER=
SAMPLER_PID=
declare -a podman_argv=() serve_argv=()

mkdir -p "$OUT" "$LOGS" "$TOOLS" "$L2"

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

start_sampler() {
  nvidia-smi --query-gpu=timestamp,memory.total,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -lms 500 >"$LOGS/gpu-$1.csv" 2>/dev/null &
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

# ------------------------------------------------------------------ arm plumbing
declare -a ARM_ENV=()
ARM_DESC=

configure_arm() {
  ARM_ENV=()
  case $1 in
  L0)
    ARM_DESC='control: promoted image, reporter profile (prefix caching + align, MTP-3), glm52-lmcache-wrapper.sh with LMCACHE_MODE=off — no LMCache, one variable from the LMCache arms'
    ARM_ENV=(-e LMCACHE_MODE=off)
    ;;
  L1cold | L2warm)
    ARM_DESC='LMCache kv_both disk (fs_native L2, chunk 1600, L1 12 GiB init 2 GiB, L2 32 GiB, 4 workers, LRU 0.90/0.10, prefetch retain): the reporter'"'"'s LMCache configuration on the promoted image'
    ARM_ENV=(
      -e LMCACHE_MODE=disk -e LMCACHE_CHUNK_SIZE=1600
      -e LMCACHE_L1_GB=12 -e LMCACHE_L1_INIT_GB=2
      -e LMCACHE_L2_GB=32 -e LMCACHE_L2_WORKERS=4
      -e LMCACHE_L2_PATH=/l2/data -e LMCACHE_LOG=/l2/lmcache-mp-cold.log
    )
    ;;
  L3restart)
    ARM_DESC='LMCache kv_both disk, RETAINED L2 after full server restart: vLLM'"'"'s own prefix cache is empty by construction, so any prefix reuse is LMCache disk retrieval — the reuse-after-restart path, the prime suspect mechanism'
    ARM_ENV=(
      -e LMCACHE_MODE=disk -e LMCACHE_CHUNK_SIZE=1600
      -e LMCACHE_L1_GB=12 -e LMCACHE_L1_INIT_GB=2
      -e LMCACHE_L2_GB=32 -e LMCACHE_L2_WORKERS=4
      -e LMCACHE_L2_PATH=/l2/data -e LMCACHE_LOG=/l2/lmcache-mp-restart.log
    )
    ;;
  *) die "unknown arm '$1'" ;;
  esac
}

compose() {
  local arm=$1
  CONTAINER=lmcachetest-$arm
  podman_argv=(
    podman run --rm --name "$CONTAINER"
    --device nvidia.com/gpu=all --ipc=host --network host
    -v "$CTX_REPO:/models/ctx-repo:ro"
    -v "$JIT:/cache/jit:rw"
    -v "$ONLINE_CACHE:/cache/exl3-online:rw"
    -v "$L2:/l2:rw"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    -e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1
    -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128
    -e DCP=1 -e PORT="$PORT"
    "${ARM_ENV[@]}"
    --entrypoint "$WRAPPER"
    "$APC_IMAGE"
  )
  serve_argv=(
    /opt/venv/bin/vllm serve "/models/ctx-repo/snapshots/$CTX_REV"
    --served-model-name m
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
    --host 127.0.0.1 --port "$PORT"
  )
  python3 - "$OUT/command-$arm.json" "$arm" "$APC_IMAGE" "$ARM_DESC" \
    "${podman_argv[@]}" -- "${serve_argv[@]}" <<'PY'
import json, pathlib, sys
out, arm, image, desc, *rest = sys.argv[1:]
split = rest.index("--")
pathlib.Path(out).write_text(json.dumps({
    "schema": "qwen38-lmcache-test-command/1", "arm": arm, "image": image,
    "description": desc, "podman_argv": rest[:split], "wrapper_argv": rest[split + 1:],
    "note": "the wrapper appends --kv-transfer-config with LMCacheMPConnector kv_both "
            "(mq_timeout 60, heartbeat 5) when LMCACHE_MODE=disk, and execs the model "
            "server unchanged when LMCACHE_MODE=off"},
    indent=2) + "\n")
PY
}

launch() {
  local arm=$1 log=$2
  say "launching arm $arm"
  "${podman_argv[@]}" "${serve_argv[@]}" >"$log" 2>&1 &
  wait_ready "$log"
}

snap_metrics() {
  local tag=$1
  curl -sf --max-time 10 "http://127.0.0.1:$PORT/metrics" >"$OUT/vllm-metrics-$tag.txt" 2>/dev/null || true
  curl -sf --max-time 10 "http://127.0.0.1:$LMC_PROM_PORT/metrics" >"$OUT/lmcache-metrics-$tag.txt" 2>/dev/null || true
  curl -sf --max-time 10 "http://127.0.0.1:$LMC_HTTP_PORT/status" >"$OUT/lmcache-status-$tag.json" 2>/dev/null || true
}

l2_snapshot() {
  local tag=$1
  {
    echo "files: $(find "$L2/data" -type f 2>/dev/null | wc -l)"
    echo "bytes: $(du -sb "$L2/data" 2>/dev/null | cut -f1)"
  } | tee "$OUT/l2-snapshot-$tag.txt" >&2
}

run_probe() {
  local arm=$1
  python3 "$TOOLS/apc_poison_probe.py" run \
    --base "http://127.0.0.1:$PORT" --model m --prompts "$W/prompts.json" \
    --arm "$arm" --description "$ARM_DESC" --expect-prompts-sha256 "$PROMPTS_SHA" \
    --concurrency 1 \
    --out "$OUT/arm-$arm.json" 2>&1 | tee "$LOGS/probe-$arm.out"
}

# ------------------------------------------------------------------ phases
phase_stage() {
  verify_image "$APC_IMAGE" "$APC_MANIFEST"
  echo "$PROBE_SHA  $TOOLS/apc_poison_probe.py" | sha256sum -c -
  [[ -d $CTX_REPO/snapshots/$CTX_REV ]] || die "context snapshot missing at $CTX_REPO"
  [[ -f $W/prompts.json ]] || die "prompts.json missing; copy the frozen set from /home/mbelleau/apcrepro"
  python3 - "$W/prompts.json" "$PROMPTS_SHA" <<'PY'
import json, sys
prompts = json.load(open(sys.argv[1]))
assert prompts["prompts_sha256"] == sys.argv[2], prompts["prompts_sha256"]
print(json.dumps({"prompts_sha256_verified": True}))
PY
  # identity of everything the arms depend on, measured inside the image, CPU only
  podman run --rm --network none --entrypoint /bin/bash "$APC_IMAGE" -lc "
    set -eu
    printf '%s  %s\n' \
      $EXL3_SHA $VENV/model_executor/layers/quantization/exl3.py \
      $QWEN_SHA $VENV/model_executor/models/qwen3_5.py \
      $MTP_SHA $VENV/model_executor/models/qwen3_5_mtp.py \
      $SCHED_SHA $VENV/v1/core/sched/scheduler.py | sha256sum -c -
    sha256sum $WRAPPER
    /opt/venv/bin/pip show lmcache | sed -n '1,2p'
    /opt/venv/bin/pip download lmcache 2>/dev/null | head -0 || true
    grep -o 'lmcache-0.5.2[^#]*#sha256=[0-9a-f]*' /opt/venv/*.txt 2>/dev/null || true
    /opt/venv/bin/python - <<'PY'
import importlib.metadata as md, json
d = md.distribution('lmcache')
direct = d.read_text('direct_url.json')
print(json.dumps({'lmcache_version': d.version,
                  'direct_url': json.loads(direct) if direct else None}))
PY
    /opt/venv/bin/lmcache server --help 2>&1 | sed -n '1,400p'
  " >"$OUT/stage-identity.txt" 2>&1 || die "stage identity checks failed (see $OUT/stage-identity.txt)"
  grep -q "$SCHED_SHA" "$OUT/stage-identity.txt" || die "promoted image lost the #51113 scheduler"
  say "listening ports that could collide:"
  ss -ltn "( sport = :$PORT or sport = :$LMC_HTTP_PORT or sport = :$LMC_PROM_PORT or sport = :5796 )" | tee "$OUT/stage-ports.txt" >&2
  python3 - "$OUT/stage.json" "$W" "$APC_IMAGE" "$APC_MANIFEST" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, work, apc, apc_d = sys.argv[1:]
work = pathlib.Path(work)
def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
prompts = json.loads((work / "prompts.json").read_text())
payload = {
    "schema": "qwen38-lmcache-test-stage/1",
    "host": run("hostname"),
    "gpu": run("nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,compute_mode",
               "--format=csv,noheader"),
    "podman_version": run("podman", "--version"),
    "image": {"tag": apc, "manifest_digest": apc_d,
              "config_id": run("podman", "image", "inspect", "-f", "{{.Id}}", apc)},
    "harness_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((work / "tools").glob("*")) if p.is_file()},
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
  launch $arm "$log" || { startup_facts "$log" $arm; tail -40 "$log" >&2; stop_server; die "arm L0 failed to start"; }
  startup_facts "$log" $arm
  snap_metrics "L0-pre"
  run_probe $arm
  snap_metrics "L0-post"
  stop_server
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm $arm \
    --out "$OUT/windows-$arm.json"
  say "arm L0 complete"
}

phase_coldwarm() {
  local log=$LOGS/server-L1L2.log
  require_free_gpu
  say "wiping L2 disk cache for the cold pass"
  rm -rf "$L2/data"; mkdir -p "$L2/data"
  configure_arm L1cold
  compose L1cold
  start_sampler L1cold
  launch L1cold "$log" || { startup_facts "$log" L1cold; tail -60 "$log" >&2; stop_server; die "LMCache arm failed to start"; }
  startup_facts "$log" L1cold
  l2_snapshot "L1cold-pre"
  snap_metrics "L1cold-pre"
  run_probe L1cold
  snap_metrics "L1cold-post"
  l2_snapshot "L1cold-post"
  say "warm pass on the SAME live server"
  configure_arm L2warm
  snap_metrics "L2warm-pre"
  run_probe L2warm
  snap_metrics "L2warm-post"
  # give asynchronous L1->L2 flushes time to land before the restart pass needs them
  sleep 45
  l2_snapshot "L2warm-post"
  cp -f "$L2/lmcache-mp-cold.log" "$LOGS/" 2>/dev/null || true
  stop_server
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm L1L2 \
    --out "$OUT/windows-L1L2.json"
  say "cold+warm passes complete"
}

phase_restart() {
  local arm=L3restart log=$LOGS/server-L3restart.log
  require_free_gpu
  [[ -d $L2/data ]] || die "L2 directory missing; run coldwarm first"
  l2_snapshot "L3restart-pre"
  configure_arm $arm
  compose $arm
  start_sampler $arm
  launch $arm "$log" || { startup_facts "$log" $arm; tail -60 "$log" >&2; stop_server; die "restart arm failed to start"; }
  startup_facts "$log" $arm
  snap_metrics "L3restart-pre"
  run_probe $arm
  snap_metrics "L3restart-post"
  l2_snapshot "L3restart-post"
  cp -f "$L2/lmcache-mp-restart.log" "$LOGS/" 2>/dev/null || true
  stop_server
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm $arm \
    --out "$OUT/windows-$arm.json"
  say "restart pass complete"
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
arms, windows, commands, facts, snaps = {}, {}, {}, {}, {}
for tag in ("L0", "L1cold", "L2warm", "L3restart"):
    data = load(f"arm-{tag}.json")
    if data:
        arms[tag] = data
    c = load(f"command-{tag}.json")
    if c:
        commands[tag] = c
    f = out / f"startup-facts-{tag}.txt"
    if f.exists():
        facts[tag] = [ln for ln in f.read_text().splitlines() if ln.strip()]
for tag in ("L0", "L1L2", "L3restart"):
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
        if re.search(r"retriev|hit|store|lookup|evict|l2|disk", name, re.I):
            try:
                keep[name] = float(value)
            except ValueError:
                pass
    lm_metrics[tag] = keep
divergence = {}
if "L0" in arms:
    ref = arms["L0"]["rows"]
    for tag in ("L1cold", "L2warm", "L3restart"):
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
    "divergence_vs_control": divergence,
    "stage": load("stage.json"),
}
(out / "assembled.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({
    "arms": {t: {"failed": a["summary"]["failed_requests"],
                 "corrupted": a["summary"]["corrupted_requests"],
                 "hit_rate": a["summary"]["cumulative_prefix_cache_hit_rate"]}
             for t, a in arms.items()}}))
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
  local gen
  gen=$(curl -sf --max-time 120 http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.8-27B","prompt":"2+2=","max_tokens":8,"temperature":0}' || echo '{}')
  python3 - "$OUT/restore.json" "$status" "$health" "$reachable" "$gen" <<'PY'
import json, pathlib, subprocess, sys
out, status, health, reachable, gen = sys.argv[1:]
def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
try:
    gen = json.loads(gen)
except json.JSONDecodeError:
    gen = {"raw": gen}
pathlib.Path(out).write_text(json.dumps({
    "schema": "qwen38-lmcache-test-restore/1", "unit_active": status,
    "container_health": health, "health_endpoint_ok": reachable == "true",
    "image": run("podman", "inspect", "qwen38-27b", "--format", "{{.ImageName}}"),
    "image_digest": run("podman", "inspect", "qwen38-27b", "--format", "{{.ImageDigest}}"),
    "binds": run("podman", "inspect", "qwen38-27b", "--format", "{{json .HostConfig.Binds}}"),
    "state": run("podman", "inspect", "qwen38-27b", "--format", "{{.State.Status}}"),
    "started_at": run("podman", "inspect", "qwen38-27b", "--format", "{{.State.StartedAt}}"),
    "generation_proof": {
        "served_model": gen.get("model"),
        "completion_tokens": gen.get("usage", {}).get("completion_tokens"),
        "finish_reason": (gen.get("choices") or [{}])[0].get("finish_reason"),
        "system_fingerprint": gen.get("system_fingerprint"),
    },
    "gpu_memory_used_mib": run("nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"),
    "compute_apps": run("nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader"),
}, indent=2) + "\n")
print(json.dumps({"unit": status, "health": health, "endpoint": reachable,
                  "completion_tokens": gen.get("usage", {}).get("completion_tokens")}))
PY
  [[ $status == active && $health == healthy && $reachable == true ]] ||
    die "live service not healthy: unit=$status health=$health endpoint=$reachable"
  say "live service healthy"
}

case ${1:-} in
stage) phase_stage ;;
control) phase_control ;;
coldwarm) phase_coldwarm ;;
restart) phase_restart ;;
receipt) phase_receipt ;;
restore) phase_restore ;;
*) die "usage: $0 {stage|control|coldwarm|restart|receipt|restore}" ;;
esac
