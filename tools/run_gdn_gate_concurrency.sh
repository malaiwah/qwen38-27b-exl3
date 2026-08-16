#!/bin/bash
# Does upstream vLLM PR #51812 (Qwen GDN speculative gate alignment) change anything at
# concurrency on the promoted four-module image? Physical RTX 5090 (AIBoss).
#
# The measurement is ordered cheapest-decisive-first, because the expensive comparison
# cannot resolve the question on its own: upstream's own effect is 0.002755 mean absolute
# chosen-logprob error, and this build's run-to-run floor for the same instrument is
# 0.0823 (receipts/apc-poison-repro.json arm Bn), i.e. ~30x larger. So:
#
#   run_gdn_gate_concurrency.sh stage        image + module identity, instrument checks,
#                                           freeze prompts as token ids  (CPU only)
#   run_gdn_gate_concurrency.sh reach R1     our published 8-stream recipe, instrumented:
#                                           does the defective path execute at all?
#   run_gdn_gate_concurrency.sh reach R2     adversarial mixed short-prompt traffic,
#                                           instrumented: can it be made to execute?
#   run_gdn_gate_concurrency.sh arm A        unpatched, 8 streams                (only if
#   run_gdn_gate_concurrency.sh arm An       unpatched repeated, the null arm     R1 or R2
#   run_gdn_gate_concurrency.sh arm B        module mounted read-only             fires)
#   run_gdn_gate_concurrency.sh receipt      assemble receipts/gdn-gate-concurrency.json
#   run_gdn_gate_concurrency.sh restore      restore the user's live service
#
# The instrument goes in vllm/v1/attention/backends/gdn_attn.py, which PR #51812 does not
# touch. Batch composition is therefore identical patched and unpatched, so ONE
# instrumented server answers the reachability question for both arms at once.
set -euo pipefail

W=${W:-/home/mbelleau/gdngate}
OUT=$W/out
LOGS=$W/logs
TOOLS=$W/tools
REACH=$W/reach
PORT=${PORT:-8251}
HF=${HF:-/mnt/vault/llm/huggingface}
CTX_REPO=$HF/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context
CTX_REV=c45c273b0d6ef2859cb2d85b36dd52253c80d878
JIT=${JIT:-/home/mbelleau/qual5090/cache/jit}
ONLINE_CACHE=${ONLINE_CACHE:-/home/mbelleau/qual5090/cache/exl3-online}
VENV=/opt/venv/lib/python3.12/site-packages/vllm

IMAGE=localhost/vllm:gg-r34-patched-apc
MANIFEST=sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218

# the four baked modules of the promoted unit
EXL3_SHA=2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee
QWEN_SHA=04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7
MTP_SHA=0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab
SCHED_SHA=b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf
# the module under test, and the file it replaces
GDN_SHA=7cd3f5fe763b621048af4817951a841d99c8b700d9a56ded27ccaca5a56ccbe0
GDN_VENDORED_SHA=663dacd324b6b8224a4cb312b3e9c0bad4322c515e982a85f13c3450ffdb7d61
GDN_DEST=$VENV/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
# the reachability instrument, and the file it replaces
INSTR_SHA=e42c9694a7f538ecd3931f9cdb177108c1c3617fd2ece6af2176e06cdd5afc43
INSTR_VENDORED_SHA=5cb3f14fbc3461256e985ea80f6329d75ebd721e67cb28155527389a3e726d45
INSTR_DEST=$VENV/v1/attention/backends/gdn_attn.py

# PerfSweep5090's explicit C8 profile from receipts/perf-sweep-5090.json, so the
# throughput numbers here are comparable to the ones already published. This is NOT the
# rank-1 single-sequence qualification profile.
QUANT_CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
MAXLEN=262144
GPU_UTIL=0.97
SEQS=8
BATCHED=2048
SPEC='{"method":"mtp","num_speculative_tokens":3}'
MM_CFG='{"truncation":false,"max_pixels":8388608}'
COMPILE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[32]}'
READY_TIMEOUT=${READY_TIMEOUT:-2400}
ADV_DURATION=${ADV_DURATION:-360}

CONTAINER=
SAMPLER_PID=
declare -a podman_argv=() serve_argv=()

mkdir -p "$OUT" "$LOGS" "$TOOLS" "$REACH"

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
  local got
  got=$(podman image inspect -f '{{index .RepoDigests 0}}' "$IMAGE" | sed 's/.*@//') ||
    die "image $IMAGE missing"
  [[ $got == "$MANIFEST" ]] || die "image $IMAGE manifest $got != $MANIFEST"
  say "image identity verified: $IMAGE ($got)"
}

# The four promoted modules must be the published bytes inside the image, and the two files
# we bind over must be the vendored bytes we think they are. CPU-only container, no GPU.
verify_modules_in_image() {
  podman run --rm --network none --entrypoint /bin/bash "$IMAGE" -lc "
    set -eu
    printf '%s  %s\n' \
      $EXL3_SHA $VENV/model_executor/layers/quantization/exl3.py \
      $QWEN_SHA $VENV/model_executor/models/qwen3_5.py \
      $MTP_SHA $VENV/model_executor/models/qwen3_5_mtp.py \
      $SCHED_SHA $VENV/v1/core/sched/scheduler.py \
      $GDN_VENDORED_SHA $GDN_DEST \
      $INSTR_VENDORED_SHA $INSTR_DEST | sha256sum -c -
  " | tee "$OUT/module-verify-image.txt"
}

# The instrument must differ from the vendored gdn_attn.py by insertions only, and the
# insertions must be removable to recover the vendored bytes exactly. Otherwise the
# instrument could be changing what it measures.
verify_instrument_additive() {
  # -i is required: without it podman does not attach stdin and the heredoc is silently
  # discarded, which would make this check pass by producing nothing at all.
  podman run --rm -i --network none \
    -v "$TOOLS/vllm-gdn-reach-instrument.py:/tmp/instrument.py:ro" \
    --entrypoint /opt/venv/bin/python "$IMAGE" - <<PY >"$OUT/instrument-additive.json"
import difflib, hashlib, json, py_compile
vendored = open("$INSTR_DEST").read().splitlines(keepends=True)
instrument = open("/tmp/instrument.py").read().splitlines(keepends=True)
sm = difflib.SequenceMatcher(None, vendored, instrument, autojunk=False)
ops = sm.get_opcodes()
changed = [o for o in ops if o[0] != "equal"]
kept_lines = []
for tag, _, _, j1, j2 in ops:
    if tag == "equal":
        kept_lines.extend(instrument[j1:j2])
kept = "".join(kept_lines)
rebuilt = hashlib.sha256(kept.encode()).hexdigest()
py_compile.compile("/tmp/instrument.py", doraise=True)
payload = {
    "vendored_sha256": hashlib.sha256("".join(vendored).encode()).hexdigest(),
    "instrument_sha256": hashlib.sha256("".join(instrument).encode()).hexdigest(),
    "opcodes_other_than_equal": [[o[0], o[1], o[2], o[3], o[4]] for o in changed],
    "purely_additive": all(o[0] == "insert" for o in changed),
    "lines_inserted": sum(o[4] - o[3] for o in changed),
    "vendored_bytes_recovered_by_deleting_insertions": rebuilt,
    "reversible": rebuilt == hashlib.sha256("".join(vendored).encode()).hexdigest(),
    "py_compile_ok": True,
}
assert payload["purely_additive"] and payload["reversible"], payload
print(json.dumps(payload, indent=2))
PY
  # fail closed: an empty or non-conforming report is a failed check, not a pass
  python3 - "$OUT/instrument-additive.json" "$INSTR_SHA" <<'PY' || die "instrument additivity check did not produce a conforming report"
import json, sys
report = json.loads(open(sys.argv[1]).read())
assert report["purely_additive"], report
assert report["reversible"], report
assert report["py_compile_ok"], report
assert report["instrument_sha256"] == sys.argv[2], report
assert all(op[0] == "insert" for op in report["opcodes_other_than_equal"]), report
print(json.dumps({"instrument_additivity": "verified",
                  "lines_inserted": report["lines_inserted"],
                  "vendored_sha256": report["vendored_sha256"]}))
PY
  cat "$OUT/instrument-additive.json" >&2
}

start_sampler() {
  nvidia-smi --query-gpu=timestamp,memory.total,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -lms 500 >"$LOGS/gpu-$1.csv" 2>/dev/null &
  SAMPLER_PID=$!
}

startup_facts() {
  local log=$1 tag=$2
  {
    grep -oE 'Available KV cache memory: [0-9.]+ GiB' "$log" || true
    grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$log" || true
    grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$log" || true
    grep -oE 'Using [A-Z_]+ attention backend[^.]*' "$log" | head -1 || true
    grep -oE "'max_num_seqs': [0-9]+" "$log" | head -1 || true
    grep -oE "'enable_prefix_caching': [A-Za-z]+" "$log" | head -1 || true
    grep -oE "'mamba_cache_mode': '[a-z]+'" "$log" | head -1 || true
    grep -oE 'num_spec_tokens=[0-9]+' "$log" | head -1 || true
    grep -oE 'cudagraph_capture_sizes=\[[^]]*\]' "$log" | head -1 || true
    grep -oE 'cudagraph_mode=[A-Za-z_.]+' "$log" | head -1 || true
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

# Acceptance requirement: the digests are checked in the process that is actually serving,
# not only in a throwaway CPU container, so a mount that silently failed cannot pass.
verify_in_running_container() {
  local tag=$1
  podman exec "$CONTAINER" /bin/bash -lc "
    sha256sum $GDN_DEST $INSTR_DEST $VENV/v1/core/sched/scheduler.py
    /opt/venv/bin/python -c '
import hashlib, importlib.util, json
names = {\"vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn\": None,
         \"vllm.v1.attention.backends.gdn_attn\": None}
for name in names:
    origin = importlib.util.find_spec(name).origin
    names[name] = {\"origin\": origin,
                   \"sha256\": hashlib.sha256(open(origin, \"rb\").read()).hexdigest()}
print(json.dumps(names, indent=2))'
  " | tee "$OUT/in-container-verify-$tag.txt"
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
ARM_DESC=

configure_arm() {
  ARM_ENV=(); ARM_ARGS=(--no-enable-prefix-caching); ARM_MOUNTS=()
  case $1 in
  # ---- reachability arms: instrument mounted, gate module NOT mounted. PR #51812 does
  # not touch gdn_attn.py, so batch composition is the same either way and one server
  # answers reachability for both A and B.
  R1)
    ARM_DESC='our published 8-stream concurrency recipe, gdn_attn.py instrumented: does the defective gather composition ever occur?'
    ;;
  R2)
    ARM_DESC='adversarial mixed traffic (eight speculative streams plus injected short prompts and full-prefix-cache-hit repeats), prefix caching and align mode ON to open the scheduler spec-padding path, gdn_attn.py instrumented'
    ARM_ARGS=(--enable-prefix-caching --mamba-cache-mode align)
    ;;
  # ---- comparison arms: no instrument, so timing is untouched.
  A)
    ARM_DESC='unpatched promoted image at eight streams: the reference'
    ;;
  An)
    ARM_DESC='arm A repeated on a second fresh server with byte-identical flags: the run-to-run null arm, which is what any A-versus-B difference must beat'
    ;;
  B)
    ARM_DESC='arm A plus the PR #51812 GDN speculative gate module bind-mounted read-only over the vendored file'
    ARM_MOUNTS=(-v "$TOOLS/vllm-qwen-gdn-spec-gates.py:$GDN_DEST:ro")
    ;;
  *) die "unknown arm '$1'" ;;
  esac
  case $1 in
  R1 | R2)
    ARM_MOUNTS+=(-v "$TOOLS/vllm-gdn-reach-instrument.py:$INSTR_DEST:ro")
    ARM_ENV=(-e "VLLM_GDN_REACH_OUT=/reach/reach")
    ;;
  esac
}

compose() {
  local arm=$1
  CONTAINER=gdngate-$arm
  podman_argv=(
    podman run --rm --name "$CONTAINER"
    --device nvidia.com/gpu=all --ipc=host --network host
    -v "$CTX_REPO:/models/ctx-repo:ro"
    -v "$JIT:/cache/jit:rw"
    -v "$ONLINE_CACHE:/cache/exl3-online:rw"
    -v "$REACH/$arm:/reach:rw"
    "${ARM_MOUNTS[@]}"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1
    -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128
    "${ARM_ENV[@]}"
    --entrypoint /opt/venv/bin/vllm
    "$IMAGE"
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
  python3 - "$OUT/command-$arm.json" "$arm" "$IMAGE" "$MANIFEST" "$ARM_DESC" \
    "${podman_argv[@]}" -- "${serve_argv[@]}" <<'PY'
import json, pathlib, sys
out, arm, image, manifest, desc, *rest = sys.argv[1:]
split = rest.index("--")
pathlib.Path(out).write_text(json.dumps({
    "schema": "qwen38-gdn-gate-command/1", "arm": arm, "image": image,
    "image_manifest_digest": manifest, "description": desc,
    "podman_argv": rest[:split], "vllm_argv": rest[split + 1:]}, indent=2) + "\n")
PY
}

launch() {
  local arm=$1 log=$2
  mkdir -p "$REACH/$arm"
  say "launching arm $arm"
  "${podman_argv[@]}" "${serve_argv[@]}" >"$log" 2>&1 &
  wait_ready "$log"
}

prompts_sha() {
  python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["prompts_sha256"])' "$W/prompts.json"
}

# ------------------------------------------------------------------ phases
phase_stage() {
  verify_image
  verify_modules_in_image
  verify_instrument_additive
  echo "$GDN_SHA  $TOOLS/vllm-qwen-gdn-spec-gates.py" | sha256sum -c -
  echo "$INSTR_SHA  $TOOLS/vllm-gdn-reach-instrument.py" | sha256sum -c -
  [[ -d $CTX_REPO/snapshots/$CTX_REV ]] || die "context snapshot missing at $CTX_REPO"

  if [[ ! -f $W/prompts.json ]]; then
    say "freezing the prompt set (CPU only, no GPU device)"
    podman run --rm --network none \
      -v "$CTX_REPO:/models/ctx-repo:ro" -v "$W:/work:rw" \
      --entrypoint /opt/venv/bin/python "$IMAGE" \
      /work/tools/apc_poison_probe.py build \
      --model "/models/ctx-repo/snapshots/$CTX_REV" --out /work/prompts.json |
      tee "$OUT/prompt-build.txt"
  fi

  python3 - "$OUT/stage.json" "$W" "$IMAGE" "$MANIFEST" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, work, image, manifest = sys.argv[1:]
work = pathlib.Path(work)
def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
prompts = json.loads((work / "prompts.json").read_text())
payload = {
    "schema": "qwen38-gdn-gate-stage/1",
    "host": run("hostname"),
    "gpu": run("nvidia-smi",
               "--query-gpu=name,uuid,memory.total,driver_version,compute_mode",
               "--format=csv,noheader"),
    "compute_capability": run("nvidia-smi", "--query-gpu=compute_cap",
                              "--format=csv,noheader"),
    "podman_version": run("podman", "--version"),
    "image": {"tag": image, "manifest_digest": manifest,
              "config_id": run("podman", "image", "inspect", "-f", "{{.Id}}", image)},
    "harness_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((work / "tools").glob("*")) if p.is_file()},
    "prompts_sha256": prompts["prompts_sha256"],
    "prompt_tokens": [r["prompt_tokens"] for r in prompts["correctness_requests"]],
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"stage": out, "prompts_sha256": payload["prompts_sha256"],
                  "compute_capability": payload["compute_capability"]}))
PY
  say "stage complete"
}

phase_reach() {
  local arm=$1
  local log=$LOGS/server-$arm.log
  require_free_gpu
  configure_arm "$arm"
  compose "$arm"
  start_sampler "$arm"
  launch "$arm" "$log" || { startup_facts "$log" "$arm"; tail -40 "$log" >&2; stop_server; die "arm $arm failed to start"; }
  startup_facts "$log" "$arm"
  verify_in_running_container "$arm"
  case $arm in
  R1)
    # the same 8-wide wave schedule the comparison arms use, so reachability is measured
    # under exactly the traffic the comparison would have run
    python3 "$TOOLS/apc_poison_probe.py" run \
      --base "http://127.0.0.1:$PORT" --model m --prompts "$W/prompts.json" \
      --arm "$arm" --description "$ARM_DESC" --expect-prompts-sha256 "$(prompts_sha)" \
      --concurrency "$SEQS" --out "$OUT/arm-$arm.json" 2>&1 | tee "$LOGS/probe-$arm.out"
    ;;
  R2)
    python3 "$TOOLS/gdn_gate_concurrency.py" adversarial \
      --base "http://127.0.0.1:$PORT" --model m --prompts "$W/prompts.json" \
      --arm "$arm" --description "$ARM_DESC" --expect-prompts-sha256 "$(prompts_sha)" \
      --streams "$SEQS" --injectors 3 --duration-s "$ADV_DURATION" \
      --chunk-tokens "$BATCHED" --out "$OUT/arm-$arm.json" 2>&1 | tee "$LOGS/probe-$arm.out"
    ;;
  esac
  # SIGTERM lets the workers' atexit hook flush; the periodic dump covers a hard kill.
  stop_server
  python3 "$TOOLS/gdn_gate_concurrency.py" reach --dir "$REACH/$arm" \
    --arm "$arm" --description "$ARM_DESC" --instrument-sha256 "$INSTR_SHA" \
    --out "$OUT/reach-$arm.json"
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm "$arm" \
    --out "$OUT/windows-$arm.json"
  say "reachability arm $arm complete"
}

phase_arm() {
  local arm=$1
  local log=$LOGS/server-$arm.log
  require_free_gpu
  configure_arm "$arm"
  compose "$arm"
  start_sampler "$arm"
  launch "$arm" "$log" || { startup_facts "$log" "$arm"; tail -40 "$log" >&2; stop_server; die "arm $arm failed to start"; }
  startup_facts "$log" "$arm"
  verify_in_running_container "$arm"
  python3 "$TOOLS/apc_poison_probe.py" run \
    --base "http://127.0.0.1:$PORT" --model m --prompts "$W/prompts.json" \
    --arm "$arm" --description "$ARM_DESC" --expect-prompts-sha256 "$(prompts_sha)" \
    --concurrency "$SEQS" --out "$OUT/arm-$arm.json" 2>&1 | tee "$LOGS/probe-$arm.out"
  stop_server
  python3 "$TOOLS/apc_poison_probe.py" scrape --log "$log" --arm "$arm" \
    --out "$OUT/windows-$arm.json"
  say "arm $arm complete"
}

phase_verdict() {
  local -a args=()
  local arm
  for arm in A An B R1; do
    [[ -f $OUT/arm-$arm.json ]] && args+=(--arm-file "$arm=$OUT/arm-$arm.json")
  done
  ((${#args[@]} >= 2)) || { say "fewer than two comparison arms; no divergence verdict"; return 0; }
  python3 "$TOOLS/apc_poison_probe.py" verdict "${args[@]}" --out "$OUT/verdict.json"
}

phase_receipt() {
  phase_verdict
  python3 "$TOOLS/gdn_gate_receipt.py" --work "$W" --out "$W/gdn-gate-concurrency.json"
  say "receipt at $W/gdn-gate-concurrency.json"
}

phase_restore() {
  say "restoring the user's live service is the owner's phase; nothing started here persists"
  podman ps --format '{{.Names}} {{.Image}} {{.Status}}' >&2
}

case ${1:-} in
stage) phase_stage ;;
reach) shift; phase_reach "${1:?arm}" ;;
arm) shift; phase_arm "${1:?arm}" ;;
verdict) phase_verdict ;;
receipt) phase_receipt ;;
restore) phase_restore ;;
*) die "usage: $0 {stage|reach R1|reach R2|arm A|arm An|arm B|verdict|receipt|restore}" ;;
esac
